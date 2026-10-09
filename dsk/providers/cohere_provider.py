"""Cohere provider (api.cohere.com) — free trial-key OpenAI-compatible API.

Cohere's trial keys (dashboard.cohere.com/api-keys) are free and need
no credit card: they unlock the full chat catalog (command-a /
command-r family, aya) with trial rate limits (small RPM and a monthly
call budget). This is the "free API key" provider class alongside the
cookie-jar web providers.

Auth model
----------
    COHERE_API_KEY env
        ->  ``cohere`` jar (cohere_cookies.json, ``api_key`` field)
        ->  COHERE_COOKIES env JSON fallback

``available()`` is just key presence; the refresh rung verifies
liveness with an authenticated native GET /v1/models (catalog only,
no quota burn).

Endpoints (mixed surface, both documented)
------------------------------------------
    GET  api.cohere.com/v1/models   NATIVE catalog (Bearer) — richer
                                    than the OpenAI shim: entries
                                    carry {name, endpoints,
                                    context_length, supports_vision,
                                    features}; paginated via
                                    next_page_token/page_token.
    POST api.cohere.ai/compatibility/v1/chat/completions
        OpenAI-compatible SSE (Bearer):
        {model, messages, stream: true, max_tokens?, temperature?}
        -> ``data: {choices:[{delta:{content|reasoning}}]}``, [DONE]
    (image input via standard OpenAI image_url data URIs for the
    aya-vision / vision-capable entries)

Catalog filter
--------------
    - native entries whose ``endpoints`` lack ``chat`` are not chat
      models (embed, rerank) — skipped; belt-and-braces id filter
      (embed|rerank) covers the bare OpenAI-shape fallback.
    - ``vision``: supports_vision flag or aya-vision-style id.
    - ``thinking``: id regex (command-a-reasoning style); Cohere
      surfaces reasoning via the compat layer when present.
    - context from context_length (128k on command-r+); there is no
      per-model max_completion_tokens — module fallback applies.

Error shapes (FLAT — message sits at the top level, not under
``error``; the same body serves native and compatibility hosts)
-------------------------------------------------------------------
    401 {"id":"...","message":"no api key supplied"}
    401 {"id":"...","message":"Incorrect API key provided: ****6789.
         You can find your API key at
         https://dashboard.cohere.com/api-keys."}
Trial-key 429s map to ProviderRateLimitError via the shared
classifier, letting the router park the model and fall through.
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

logger = logging.getLogger('dsk.providers.cohere')

COHERE_API_BASE = (os.getenv('I4F_COHERE_API_BASE', '') or
                   'https://api.cohere.com/v1').rstrip('/')
COHERE_COMPAT_BASE = (os.getenv('I4F_COHERE_COMPAT_BASE', '') or
                      'https://api.cohere.ai/compatibility/v1').rstrip('/')
MODELS_URL = f'{COHERE_API_BASE}/models'
CHAT_URL = f'{COHERE_COMPAT_BASE}/chat/completions'

_USER_AGENT = ('Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 '
               '(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')

_MODELS_TTL = 300.0
COHERE_CONTEXT_FALLBACK = 131072
COHERE_MAX_OUTPUT_FALLBACK = 8192
_MAX_CATALOG_PAGES = 3

# id filter for the bare OpenAI-shape fallback (embed/rerank models)
_RE_NON_CHAT = re.compile(r'(?:embed|rerank)', re.IGNORECASE)
# reasoning ids: command-a-reasoning-... (regex fallback, capability
# flags are read from features when present)
_RE_THINKING = re.compile(r'(?:reasoning|thinking)', re.IGNORECASE)


def _api_key() -> str:
    raw = (os.getenv('COHERE_API_KEY', '') or '').strip()
    if raw:
        return raw
    jar = load_jar('cohere') or env_cookies('COHERE')
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
        # {"id":"...","message":"no api key supplied"} /
        # {"message":"Incorrect API key provided: ..."}
        return ProviderAuthError(
            f'Cohere API key rejected (HTTP {status}): {(text or "")[:200]}')
    return classify_http_error(status, text, headers)


def _parse_models(data: Any) -> List[Dict[str, Any]]:
    """Defensive parse of the native catalog (and the bare compat one).

    Native: ``{"models":[{name, endpoints, context_length,
    supports_vision, features}], "next_page_token"}``. OpenAI shim
    fallback: ``{"data":[{id, owned_by}]}``. Native wins when both are
    present.
    """
    entries: Any = None
    if isinstance(data, dict):
        entries = data.get('models')
        if not isinstance(entries, list):
            entries = data.get('data')
    elif isinstance(data, list):
        entries = data
    if not isinstance(entries, list):
        return []
    out: List[Dict[str, Any]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        mid = str(entry.get('name') or entry.get('id') or '').strip()
        if not mid:
            continue
        endpoints = entry.get('endpoints')
        if isinstance(endpoints, list) and \
                'chat' not in [str(e) for e in endpoints]:
            continue  # embed/rerank — not chat models
        if _RE_NON_CHAT.search(mid):
            continue
        features = entry.get('features')
        features = [str(f).lower() for f in features] \
            if isinstance(features, list) else []
        try:
            context = int(entry.get('context_length')
                          or entry.get('context_window') or 0)
        except (TypeError, ValueError):
            context = 0
        try:
            max_out = int(entry.get('max_output_tokens')
                          or entry.get('max_completion_tokens') or 0)
        except (TypeError, ValueError):
            max_out = 0
        vision = (bool(entry.get('supports_vision')) or 'vision' in mid.lower())
        thinking = (bool(_RE_THINKING.search(mid))
                    or any('thinking' in f or 'reasoning' in f
                           for f in features))
        out.append({
            'id': mid,
            'owned_by': str(entry.get('owned_by') or 'cohere') or 'cohere',
            'vision': vision,
            'thinking': thinking,
            'context': context or COHERE_CONTEXT_FALLBACK,
            'max_out': min(max_out, context) if max_out else
            COHERE_MAX_OUTPUT_FALLBACK,
        })
    return out


def _fetch_catalog_pages(key: str, no_proxy: bool = False
                         ) -> Tuple[Optional[List[Any]], Optional[Any]]:
    """Follow the ListModels pagination (next_page_token/page_token).

    Returns (entries, None) on success or (None, response) when a page
    fails, so the caller can classify the failing response.
    """
    entries: List[Any] = []
    token = ''
    for _ in range(_MAX_CATALOG_PAGES):
        url = f'{MODELS_URL}?page_size=100'
        if token:
            url += f'&page_token={token}'
        resp = http_get(url, headers=_headers({
            'Accept': 'application/json'}), timeout=30, no_proxy=no_proxy)
        if resp.status_code != 200:
            return None, resp
        data = resp.json()
        page = data.get('models') if isinstance(data, dict) else data
        if isinstance(page, list):
            entries.extend(page)
        token = str(data.get('next_page_token')
                    or data.get('nextPageToken') or '') \
            if isinstance(data, dict) else ''
        if not token:
            break
    return entries, None


class CohereProvider(Provider):
    """api.cohere.com trial tier behind the unified provider contract.

    Dormant until an API key is provisioned (COHERE_API_KEY env, the
    ``cohere`` jar, or the refresher signup rung which automates the
    dashboard.cohere.com registration + trial key creation).
    """

    name = 'cohere'

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
            entries, failed = _fetch_catalog_pages(_api_key(),
                                                   no_proxy=no_proxy)
            if entries is not None:
                parsed = _parse_models({'models': entries})
                if parsed:
                    with self._lock:
                        self._models = parsed
                        self._catalog_ts = now
                    return
            status = failed.status_code if failed is not None else '?'
            logger.warning('cohere models HTTP %s — keeping %s cached '
                           'models', status, len(self._models))
        except Exception as e:  # noqa: BLE001 — best-effort refresh
            logger.warning('cohere models refresh failed: %s', e)

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
                'no Cohere API key configured (COHERE_API_KEY env or '
                'cohere_cookies.json): free trial keys come from '
                'dashboard.cohere.com/api-keys')
        self._refresh_catalog()
        out: List[Dict[str, Any]] = []
        for m in self._models:
            out.append({
                'id': m['id'],
                'upstream_model': m['id'],
                'thinking_enabled': m['thinking'],
                'search_enabled': False,
                'vision': m['vision'],
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
            raise ProviderAuthError('no Cohere API key configured')
        if image_generation:
            raise ProviderError('cohere has no image generation endpoint')
        self._refresh_catalog(no_proxy=no_proxy)
        entry = self._find_model(model)
        if not entry:
            raise ProviderError(f'cohere model not in catalog: {model}')

        content: Any = prompt
        if images:
            if not entry.get('vision'):
                raise ProviderError(
                    f'cohere model {entry["id"]!r} is not vision-capable')
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
            # Cohere uses flat {"message": ...} bodies; in-stream errors
            # may arrive nested (error) or flat (message)
            error = data.get('error')
            if isinstance(error, dict) and error.get('message'):
                raise ProviderError(
                    f'cohere stream error: {error["message"]}')
            if error:
                raise ProviderError(
                    f'cohere stream error: {json.dumps(error)[:300]}')
            if data.get('message') and not data.get('choices'):
                raise ProviderError(
                    f'cohere stream error: {data["message"]}')
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
                return
        if saw_content:
            yield {'content': '', 'type': 'text', 'finish_reason': 'stop'}
        else:
            raise ProviderUnavailableError(
                'cohere stream produced no output')
