"""ModelScope provider (api-inference.modelscope.cn) — free-tier
OpenAI-compatible inference API.

ModelScope (Alibaba's model hub) serves a free API-Inference tier for
registered accounts: a per-model daily request quota at no charge, keyed
by an SDK access token (``ms-…`` UUID from modelscope.cn). This is the
"free API key" provider class alongside the cookie-jar web providers.

Auth model
----------
    MODELSCOPE_API_KEY env
        ->  ``modelscope`` jar (modelscope_cookies.json, ``api_key`` field)
        ->  MODELSCOPE_COOKIES env JSON fallback

``available()`` is just key presence; the refresh rung verifies liveness
with an authenticated GET /v1/models (cheap, no quota burn).

Endpoints (both standard OpenAI shapes)
---------------------------------------
    GET  /v1/models           public — no auth needed; entries carry only
                              {id, owned_by, created} (no context info,
                              so context/max-output use fallbacks here)
    POST /v1/chat/completions
        {model, messages, stream: true, max_tokens?, temperature?}
        -> SSE ``data: {choices:[{delta:{content|reasoning_content}}]}``,
           [DONE]

Differences vs Groq/Cerebras
----------------------------
    - model ids are hub-namespaced (``Qwen/Qwen3.8-27B``,
      ``deepseek-ai/DeepSeek-V4-Pro``, …) and passed through verbatim.
    - reasoning streams as ``delta.reasoning_content`` (DeepSeek-style) on
      thinking-capable models (Qwen3.x hybrid, DeepSeek-V4) — mapped to
      ``thinking`` chunks here; ``delta.reasoning`` is honoured too. No
      ``reasoning_format`` request parameter exists.
    - image-edit entries (``*Image-Edit*``) have no chat endpoint and are
      skipped from the catalog; there is no image-generation endpoint.
    - 401 bodies nest under ``error``:
      ``{"error":{"message":"Authentication failed, …","request_id":…}}``.

Free-tier 429s map to ProviderRateLimitError via the shared classifier,
letting the router park the model and fall through.
"""

import base64
import json
import logging
import os
import re
import threading
import time
from typing import Any, Dict, Generator, List, Optional

from .base import (
    Provider,
    ProviderAuthError,
    ProviderError,
    ProviderUnavailableError,
    classify_http_error,
    http_get,
    http_post_stream,
    parse_sse_data,
)
from .jar import env_cookies, load_jar

logger = logging.getLogger('dsk.providers.modelscope')

MODELSCOPE_API_BASE = (os.getenv('I4F_MODELSCOPE_API_BASE', '') or
                       'https://api-inference.modelscope.cn/v1').rstrip('/')
MODELS_URL = f'{MODELSCOPE_API_BASE}/models'
CHAT_URL = f'{MODELSCOPE_API_BASE}/chat/completions'

_USER_AGENT = ('Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 '
               '(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')

_MODELS_TTL = 300.0
MODELSCOPE_CONTEXT_FALLBACK = 32768
MODELSCOPE_MAX_OUTPUT_FALLBACK = 8192

# Thinking-capable families on ModelScope (Qwen3.x hybrid, DeepSeek-V4
# hybrid, …-thinking): they stream delta.reasoning_content.
_RE_THINKING = re.compile(r'(?:qwen3|deepseek|.*thinking)', re.IGNORECASE)
# Vision-capable families (InternVL, ERNIE-4.5-VL, …-VL-/…-vision ids).
_RE_VISION = re.compile(r'(?:\bvl|internvl|vision)', re.IGNORECASE)


def _api_key() -> str:
    raw = (os.getenv('MODELSCOPE_API_KEY', '') or '').strip()
    if raw:
        return raw
    jar = load_jar('modelscope') or env_cookies('MODELSCOPE')
    return (jar.get('api_key') or jar.get('key') or jar.get('token')
            or '').strip()


def _headers(extra: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    hdr = {
        'User-Agent': _USER_AGENT,
        'Accept': '*/*',
        'Authorization': f'Bearer {_api_key()}',
    }
    if extra:
        hdr.update(extra)
    return hdr


def _classify(status: int, text: str,
              headers: Optional[Any] = None) -> ProviderError:
    if status in (401, 403):
        # {"error":{"message":"Authentication failed, please make sure "
        #           "that a valid ModelScope token is supplied.",
        #           "request_id": ...}}
        return ProviderAuthError(
            f'ModelScope token rejected (HTTP {status}): '
            f'{(text or "")[:200]}')
    return classify_http_error(status, text, headers)


def _parse_models(data: Any) -> List[Dict[str, Any]]:
    """Defensive parse of the /models payload (list or {"data": [...]}).

    ModelScope entries carry only {id, owned_by, created} — no context or
    output limits — so those use the module fallbacks. Image-edit entries
    (``*Image-Edit*``) have no chat endpoint and are skipped.
    """
    entries: Any = data.get('data') if isinstance(data, dict) else data
    if not isinstance(entries, list):
        return []
    out: List[Dict[str, Any]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        mid = str(entry.get('id') or '').strip()
        if not mid or 'image' in mid.lower():
            continue  # image-edit tools — no chat endpoint
        try:
            context = int(entry.get('context_window')
                          or entry.get('context_length') or 0)
        except (TypeError, ValueError):
            context = 0
        try:
            max_out = int(entry.get('max_completion_tokens')
                          or entry.get('max_tokens') or 0)
        except (TypeError, ValueError):
            max_out = 0
        out.append({
            'id': mid,
            'owned_by': str(entry.get('owned_by') or 'modelscope'),
            'context': context or MODELSCOPE_CONTEXT_FALLBACK,
            'max_out': min(max_out, context) if max_out else
            MODELSCOPE_MAX_OUTPUT_FALLBACK,
        })
    return out


class ModelScopeProvider(Provider):
    """api-inference.modelscope.cn free tier behind the unified contract.

    Dormant until an access token is provisioned (MODELSCOPE_API_KEY env,
    the ``modelscope`` jar, or the refresher signup rung which automates
    modelscope.cn sign-up + token creation).
    """

    name = 'modelscope'

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._catalog_ts = 0.0
        self._models: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------ auth
    def available(self, auth_key: Optional[str] = None) -> bool:
        return bool(_api_key())

    # ---------------------------------------------------------------- models
    def _refresh_catalog(self, no_proxy: bool = False) -> None:
        now = time.time()
        with self._lock:
            if self._models and now - self._catalog_ts < _MODELS_TTL:
                return
        try:
            # /models is public on ModelScope — fetched WITHOUT the bearer
            # so a stale/revoked token cannot 401 the catalog away.
            resp = http_get(MODELS_URL, headers={
                'User-Agent': _USER_AGENT,
                'Accept': 'application/json'}, timeout=30,
                no_proxy=no_proxy)
            if resp.status_code == 200:
                parsed = _parse_models(resp.json())
                if parsed:
                    with self._lock:
                        self._models = parsed
                        self._catalog_ts = now
                    return
            logger.warning('modelscope models HTTP %s — keeping %s cached '
                           'models', resp.status_code, len(self._models))
        except Exception as e:  # noqa: BLE001 — best-effort refresh
            logger.warning('modelscope models refresh failed: %s', e)

    def _find_model(self, key: str) -> Optional[Dict[str, Any]]:
        for m in self._models:
            if m['id'] == key:
                return m
        lowered = key.lower()
        for m in self._models:
            if m['id'].lower() == lowered or \
                    m['id'].lower().endswith('/' + lowered):
                return m
        return None

    def list_models(self, auth_key: Optional[str] = None) -> List[Dict[str, Any]]:
        if not self.available():
            raise ProviderAuthError(
                'no ModelScope token configured (MODELSCOPE_API_KEY env or '
                'modelscope_cookies.json): free tokens come from '
                'modelscope.cn (access tokens page)')
        self._refresh_catalog()
        out: List[Dict[str, Any]] = []
        for m in self._models:
            out.append({
                'id': m['id'],
                'upstream_model': m['id'],
                'thinking_enabled': bool(_RE_THINKING.search(m['id'])),
                'search_enabled': False,
                'vision': bool(_RE_VISION.search(m['id'])),
                'image_gen': False,
                'context_length': m['context'],
                'max_output_tokens': m['max_out'],
                'extra': {'owned_by': m['owned_by']},
            })
        return out

    # ---------------------------------------------------------------- stream
    def stream(self, prompt: str, *, model: str, thinking_enabled: bool = False,
               search_enabled: bool = False, temperature: Optional[float] = None,
               max_tokens: Optional[int] = None,
               images: Optional[List[Dict[str, Any]]] = None,
               image_generation: bool = False,
               no_proxy: bool = False,
               auth_key: Optional[str] = None) -> Generator[Dict[str, Any], None, None]:
        if not self.available():
            raise ProviderAuthError('no ModelScope token configured')
        if image_generation:
            raise ProviderError(
                'modelscope has no image generation endpoint')
        self._refresh_catalog(no_proxy=no_proxy)
        entry = self._find_model(model)
        if not entry:
            raise ProviderError(f'modelscope model not in catalog: {model}')

        content: Any = prompt
        if images:
            if not _RE_VISION.search(entry['id']):
                raise ProviderError(
                    f'modelscope model {entry["id"]!r} is not '
                    'vision-capable')
            parts: List[Dict[str, Any]] = [{'type': 'text', 'text': prompt}]
            for img in images:
                mime = img.get('mime') or 'image/png'
                data = img.get('data') or b''
                if not data:
                    continue
                uri = f'data:{mime};base64,' + base64.b64encode(data).decode()
                parts.append({'type': 'image_url', 'image_url': {'url': uri}})
            if len(parts) > 1:
                content = parts

        payload: Dict[str, Any] = {
            'model': entry['id'],
            'messages': [{'role': 'user', 'content': content}],
            'stream': True,
        }
        if temperature is not None:
            payload['temperature'] = temperature
        if max_tokens:
            payload['max_tokens'] = max_tokens

        resp = http_post_stream(
            CHAT_URL,
            headers=_headers({'Content-Type': 'application/json',
                              'Accept': 'text/event-stream'}),
            json_body=payload,
            timeout=300,
            no_proxy=no_proxy,
        )
        if resp.status_code != 200:
            try:
                error_text = resp.text or ''
            except Exception:  # pragma: no cover
                error_text = f'HTTP {resp.status_code}'
            raise _classify(resp.status_code, error_text, resp.headers)
        return self._iter_chunks(resp)

    def _iter_chunks(self, resp: Any) -> Generator[Dict[str, Any], None, None]:
        saw_content = False
        for line in resp.iter_lines():
            data = parse_sse_data(line)
            if not data:
                continue
            error = data.get('error')
            if error:
                message = (error.get('message') if isinstance(error, dict)
                           else str(error)) or json.dumps(error)[:300]
                raise ProviderError(f'modelscope stream error: {message}')
            choices = data.get('choices') or []
            if not choices:
                continue
            delta = (choices[0] or {}).get('delta') or {}
            reasoning = delta.get('reasoning_content') or delta.get('reasoning')
            if reasoning:
                saw_content = True
                yield {'content': str(reasoning), 'type': 'thinking',
                       'finish_reason': None}
            if delta.get('content'):
                saw_content = True
                yield {'content': str(delta['content']), 'type': 'text',
                       'finish_reason': None}
            if choices[0].get('finish_reason'):
                yield {'content': '', 'type': 'text', 'finish_reason': 'stop'}
                return
        if saw_content:
            yield {'content': '', 'type': 'text', 'finish_reason': 'stop'}
        else:
            raise ProviderUnavailableError(
                'modelscope stream produced no output')
