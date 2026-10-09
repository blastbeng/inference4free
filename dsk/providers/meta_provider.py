"""Meta Muse Spark provider (api.meta.ai/v1) — free OpenAI-compatible API.

Meta's developer platform (developers.meta.ai) exposes an OpenAI-compatible
inference API for the Muse Spark family (Muse Spark 1.1, 1.3, the open-weight
Muse Glimmer distills). Public preview: FREE for US developers — this is the
same "free API key" provider class as groq/cerebras/cohere.

Auth model
----------
    META_API_KEY env  →  ``meta`` jar (meta_cookies.json, ``api_key`` field)
                      →  META_COOKIES env JSON fallback

``available()`` is just key presence; the refresh rung verifies liveness with
an authenticated GET /v1/models (cheap, no token burn).

Endpoints (standard OpenAI shapes, verified live: both 401 OpenAI-style
``{"error":{"code":"invalid_api_key","type":"authentication_error"}}``
without a key)
---------------------------------------------------------------------------
    GET  /v1/models                {"data": [{id, owned_by, ...}]}
    POST /v1/chat/completions      {model, messages, stream: true, ...}
        → SSE ``data: {choices:[{delta:{content|reasoning_content}}]}``,
          ``data: [DONE]``

Geo note: the preview is US-developers-only — signup and inference run
through the proxy ladder (I4F_SIGNUP_PROXY / pool), never direct, or the
dashboard is geo-walled.

Catalog: the live /v1/models catalog wins when a key is present; the static
fallback below carries the ids documented on developers.meta.ai and is only
used when the catalog endpoint is unreachable (marked best-effort — ids may
drift in preview).
"""

import base64
import json
import logging
import os
import re
import threading
import time
from typing import Any, Dict, Generator, List, Optional, Tuple

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

logger = logging.getLogger('dsk.providers.meta')

META_API_BASE = (os.getenv('I4F_META_API_BASE', '') or
                 'https://api.meta.ai/v1').rstrip('/')
MODELS_URL = f'{META_API_BASE}/models'
CHAT_URL = f'{META_API_BASE}/chat/completions'

_USER_AGENT = ('Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 '
               '(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')

_MODELS_TTL = 300.0
META_CONTEXT_FALLBACK = 128000
META_MAX_OUTPUT_FALLBACK = 4096

# Static fallback catalog (documented on developers.meta.ai; live catalog
# wins whenever the key can reach /v1/models).
_STATIC_MODELS: List[Dict[str, Any]] = [
    {'id': 'muse-spark-1.1', 'owned_by': 'meta'},
    {'id': 'muse-spark-1.3', 'owned_by': 'meta'},
    {'id': 'muse-glimmer', 'owned_by': 'meta'},
]

_RE_THINKING = re.compile(r'(?:reasoning|thinking|[-_/]r1\b)', re.IGNORECASE)
# Muse Glimmer distills ship a "vision" text+image variant in the docs
_RE_VISION = re.compile(r'(?:vision|\bvl\b|image)', re.IGNORECASE)


def _api_key() -> str:
    raw = (os.getenv('META_API_KEY', '') or '').strip()
    if raw:
        return raw
    jar = load_jar('meta') or env_cookies('META')
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
        # {"error":{"code":"invalid_api_key","type":"authentication_error"}}
        return ProviderAuthError(
            f'meta API key rejected (HTTP {status}): {(text or "")[:200]}')
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
            context = int(entry.get('context_window')
                          or entry.get('context_length') or 0)
        except (TypeError, ValueError):
            context = 0
        try:
            max_out = int(entry.get('max_completion_tokens')
                          or entry.get('max_output_tokens') or 0)
        except (TypeError, ValueError):
            max_out = 0
        out.append({
            'id': mid,
            'owned_by': str(entry.get('owned_by') or 'meta') or 'meta',
            'context': context or META_CONTEXT_FALLBACK,
            'max_out': min(max_out, context) if max_out else
            META_MAX_OUTPUT_FALLBACK,
        })
    return out


def verify_key(key: str) -> Tuple[bool, str]:
    """Shared liveness probe (refresher rung + tests): GET /v1/models."""
    try:
        resp = http_get(MODELS_URL, headers={
            'Authorization': f'Bearer {key}', 'Accept': 'application/json',
        }, timeout=30)
    except Exception as e:  # noqa: BLE001
        return False, f'verify failed: {type(e).__name__}: {e}'
    if resp.status_code in (401, 403):
        return False, ('API key rejected - create a new key at '
                       'developers.meta.ai (dashboard) and set META_API_KEY')
    if resp.status_code == 429:
        return True, 'rate limited but key accepted (HTTP 429)'
    if resp.status_code != 200:
        return True, (f'key reachable, liveness inconclusive (HTTP '
                      f'{resp.status_code}) - validated at request time')
    try:
        models = _parse_models(resp.json())
    except Exception:  # noqa: BLE001
        models = []
    return True, f'API key valid ({len(models)} models visible)'


class MetaProvider(Provider):
    """api.meta.ai/v1 free preview behind the unified provider contract.

    Dormant until an API key is provisioned (META_API_KEY env, the ``meta``
    jar, or the refresher signup rung which automates the developers.meta.ai
    dashboard).
    """

    name = 'meta'

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
            else:
                logger.warning('meta models HTTP %s — using static fallback',
                               resp.status_code)
        except Exception as e:  # noqa: BLE001 — best-effort refresh
            logger.warning('meta models refresh failed: %s', e)
        with self._lock:
            self._models = [dict(m) for m in _STATIC_MODELS]
            self._catalog_ts = now

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
                'no Meta API key configured (META_API_KEY env or '
                'meta_cookies.json): the Muse Spark preview is free for US '
                'developers; create a key at developers.meta.ai')
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
            raise ProviderAuthError('no Meta API key configured')
        if image_generation:
            raise ProviderError('meta has no image generation endpoint')
        self._refresh_catalog(no_proxy=no_proxy)
        entry = self._find_model(model)
        if not entry:
            raise ProviderError(f'meta model not in catalog: {model}')

        content: Any = prompt
        if images:
            if not _RE_VISION.search(entry['id']):
                raise ProviderError(
                    f'meta model {entry["id"]!r} is not vision-capable')
            parts: List[Dict[str, Any]] = [{'type': 'text', 'text': prompt}]
            for img in images:
                mime = img.get('mime') or 'image/png'
                data = img.get('data') or b''
                if not data:
                    continue
                uri = f'data:{mime};base64,' + base64.b64encode(data).decode()
                parts.append({'type': 'image_url',
                              'image_url': {'url': uri}})
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
                raise ProviderUnavailableError(f'meta stream error: {message}')
            choices = data.get('choices') or []
            if not choices:
                continue
            delta = (choices[0] or {}).get('delta') or {}
            reasoning = delta.get('reasoning') or delta.get('reasoning_content')
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
        if saw_content:
            yield {'content': '', 'type': 'text', 'finish_reason': 'stop'}
        else:
            raise ProviderUnavailableError(
                'meta stream produced no output')
