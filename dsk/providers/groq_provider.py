"""Groq provider (api.groq.com) — free-tier OpenAI-compatible inference API.

Groq runs an OpenAI-compatible inference API on dedicated LPU hardware. The
free tier (console.groq.com account → API key, ``gsk_…``) is rate-limited
per model (RPM/RPD/TPD) but costs nothing — this is the "free API key"
provider class alongside the cookie-jar web providers.

Auth model
----------
    GROQ_API_KEY env  →  ``groq`` jar (groq_cookies.json, ``api_key`` field)
                      →  GROQ_COOKIES env JSON fallback

``available()`` is just key presence; the refresh rung verifies liveness
with an authenticated GET /openai/v1/models (cheap, no quota burn).

Endpoints (both standard OpenAI shapes)
---------------------------------------
    GET  /openai/v1/models          {"data": [{id, owned_by, active,
                                     context_window, max_completion_tokens}]}
    POST /openai/v1/chat/completions
        {model, messages, stream: true, max_completion_tokens?, temperature?,
         reasoning_format?: "parsed"}
        → SSE ``data: {choices:[{delta:{content|reasoning}}]}``, [DONE]

``reasoning_format: "parsed"`` makes Groq split thinking into
``delta.reasoning`` (emitted as ``thinking`` chunks here) instead of inline
<think/> tags. It is only valid on reasoning models, so it is attached
based on a model-id heuristic (qwen3 / r1 / gpt-oss / thinking) — a 400
from a non-reasoning model would poison the router's fallback chain.

Free-tier 429s carry Retry-After and map to ProviderRateLimitError via the
shared classifier, letting the router park the model and fall through.
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

logger = logging.getLogger('dsk.providers.groq')

GROQ_API_BASE = (os.getenv('I4F_GROQ_API_BASE', '') or
                 'https://api.groq.com/openai/v1').rstrip('/')
MODELS_URL = f'{GROQ_API_BASE}/models'
CHAT_URL = f'{GROQ_API_BASE}/chat/completions'

_USER_AGENT = ('Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 '
               '(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')

_MODELS_TTL = 300.0
GROQ_CONTEXT_FALLBACK = 131072
GROQ_MAX_OUTPUT_FALLBACK = 8192

# Reasoning-capable model ids on Groq (qwen3 thinking, R1 distills, gpt-oss
# with reasoning_effort, kimivl-thinking, …): only these accept
# reasoning_format / stream delta.reasoning.
_RE_THINKING = re.compile(
    r'(?:^|/)(?:qwen3|r1|gpt-oss|.*thinking|deepseek)', re.IGNORECASE)
# Vision-capable families (llama-4 scout/maverick, gemma-3, VL models,
# maverick): accept image_url content parts.
_RE_VISION = re.compile(
    r'(?:scout|maverick|vision|\bvl[\-_.]|gemma-3)', re.IGNORECASE)


def _api_key() -> str:
    raw = (os.getenv('GROQ_API_KEY', '') or '').strip()
    if raw:
        return raw
    jar = load_jar('groq') or env_cookies('GROQ')
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
        # {"error":{"message":"Invalid API Key",
        #           "type":"invalid_request_error","code":"invalid_api_key"}}
        return ProviderAuthError(
            f'groq API key rejected (HTTP {status}): {(text or "")[:200]}')
    return classify_http_error(status, text, headers)


def _parse_models(data: Any) -> List[Dict[str, Any]]:
    """Defensive parse of the /models payload (list or {"data": [...]})."""
    entries: Any = data.get('data') if isinstance(data, dict) else data
    if not isinstance(entries, list):
        return []
    out: List[Dict[str, Any]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        mid = str(entry.get('id') or '').strip()
        if not mid or entry.get('active') is False:
            continue
        try:
            context = int(entry.get('context_window') or 0)
        except (TypeError, ValueError):
            context = 0
        try:
            max_out = int(entry.get('max_completion_tokens') or 0)
        except (TypeError, ValueError):
            max_out = 0
        out.append({
            'id': mid,
            'owned_by': str(entry.get('owned_by') or 'groq'),
            'context': context or GROQ_CONTEXT_FALLBACK,
            'max_out': min(max_out, context) if max_out else
            GROQ_MAX_OUTPUT_FALLBACK,
        })
    return out


class GroqProvider(Provider):
    """api.groq.com free tier behind the unified provider contract.

    Dormant until an API key is provisioned (GROQ_API_KEY env, the ``groq``
    jar, or the refresher signup rung which automates console.groq.com).
    """

    name = 'groq'

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
            resp = http_get(MODELS_URL, headers=_headers({
                'Accept': 'application/json'}), timeout=30,
                no_proxy=no_proxy)
            if resp.status_code == 200:
                parsed = _parse_models(resp.json())
                if parsed:
                    with self._lock:
                        self._models = parsed
                        self._catalog_ts = now
                    return
            logger.warning('groq models HTTP %s — keeping %s cached models',
                           resp.status_code, len(self._models))
        except Exception as e:  # noqa: BLE001 — best-effort refresh
            logger.warning('groq models refresh failed: %s', e)

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
                'no groq API key configured (GROQ_API_KEY env or '
                'groq_cookies.json): free keys come from console.groq.com')
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
            raise ProviderAuthError('no groq API key configured')
        if image_generation:
            raise ProviderError('groq has no image generation endpoint')
        self._refresh_catalog(no_proxy=no_proxy)
        entry = self._find_model(model)
        if not entry:
            raise ProviderError(f'groq model not in catalog: {model}')

        content: Any = prompt
        if images:
            if not _RE_VISION.search(entry['id']):
                raise ProviderError(
                    f'groq model {entry["id"]!r} is not vision-capable')
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
            payload['max_completion_tokens'] = max_tokens
        if thinking_enabled or _RE_THINKING.search(entry['id']):
            # reasoning models only — a non-reasoning model would 400
            payload['reasoning_format'] = 'parsed'

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
            if data.get('error'):
                error = data['error']
                message = (error.get('message') if isinstance(error, dict)
                           else str(error)) or json.dumps(error)[:300]
                raise ProviderError(f'groq stream error: {message}')
            choices = data.get('choices') or []
            if not choices:
                continue
            delta = (choices[0] or {}).get('delta') or {}
            reasoning = delta.get('reasoning')
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
                'groq stream produced no output')
