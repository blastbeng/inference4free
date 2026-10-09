"""LLM7.io provider (api.llm7.io) — free-tier OpenAI-compatible API.

LLM7.io serves a genuinely free LLM gateway: accounts get a free daily
token quota usable on the non-usage-based catalog (``usage_based_only``
is False — the "turbo"/"pro" entries that cost nothing against the free
pool), keyed by an API token from dash.llm7.io/#/api-keys. This is the
"free API key" provider class alongside the cookie-jar web providers.

Auth model
----------
    LLM7_API_KEY env
        ->  ``llm7`` jar (llm7_cookies.json, ``api_key`` field)
        ->  LLM7_COOKIES env JSON fallback

``available()`` is just key presence; the refresh rung verifies liveness
with an authenticated GET /v1/balance (quota record, no quota burn).

Endpoints (standard OpenAI shapes, public OpenAPI at /openapi.json)
-------------------------------------------------------------------
    GET  /v1/models           PUBLIC — no bearer needed; the full
                              catalog is returned and filtered here down
                              to the free-pool entries. Entries carry:
                              {id, model_type, tier, pricing, modalities,
                              context_window:{tokens}, capabilities:
                              {vision, reasoning, tools, stream, ...},
                              usage_based_only, availability}
    POST /v1/chat/completions
        {model, messages, stream: true, max_tokens?, temperature?}
        -> SSE ``data: {choices:[{delta:{content|reasoning}}]}``, [DONE]
    GET  /v1/balance          AUTH — per-key quota record (liveness)

Catalog filter
--------------
    - ``usage_based_only`` truthy entries need paid credit — skipped.
    - non-``chat`` model types (embeddings/rerank, if any) — skipped.
    - ``stream`` False entries cannot satisfy this streaming contract —
      skipped.
    - ``vision`` from capabilities.vision (or image in
      modalities.input); ``thinking`` from capabilities.reasoning
      (id-regex fallback); context from context_window.tokens; there is
      no per-model max_completion_tokens — module fallback applies.

Error shapes (nested under ``error``)
-------------------------------------
    401 {"error":{"message":"Missing API key.","type":
         "authentication_error","code":"missing_api_key"}}
    401 {"error":{"message":"Your API key is invalid, expired, or
         revoked. Generate a new key at https://dash.llm7.io/#/api-keys",
         "type":"authentication_error","code":"invalid_api_key"}}
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

logger = logging.getLogger('dsk.providers.llm7')

LLM7_API_BASE = (os.getenv('I4F_LLM7_API_BASE', '') or
                 'https://api.llm7.io/v1').rstrip('/')
MODELS_URL = f'{LLM7_API_BASE}/models'
CHAT_URL = f'{LLM7_API_BASE}/chat/completions'
BALANCE_URL = f'{LLM7_API_BASE}/balance'

_USER_AGENT = ('Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 '
               '(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')

_MODELS_TTL = 300.0
LLM7_CONTEXT_FALLBACK = 131072
LLM7_MAX_OUTPUT_FALLBACK = 8192

# Fallback thinking flag for entries without capabilities.reasoning.
_RE_THINKING = re.compile(r'(?:reasoning|thinking|[-_/]r1\b)', re.IGNORECASE)


def _api_key() -> str:
    raw = (os.getenv('LLM7_API_KEY', '') or '').strip()
    if raw:
        return raw
    jar = load_jar('llm7') or env_cookies('LLM7')
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
        # {"error":{"message":"Missing API key."| "Your API key is
        #  invalid, expired, or revoked. ...","code":
        #  "missing_api_key"|"invalid_api_key"}}
        return ProviderAuthError(
            f'LLM7 API key rejected (HTTP {status}): {(text or "")[:200]}')
    return classify_http_error(status, text, headers)


def _parse_models(data: Any) -> List[Dict[str, Any]]:
    """Defensive parse of the public /models payload.

    Only free-pool entries survive: ``usage_based_only`` truthy entries
    need paid credit, non-chat model types have no chat endpoint and
    ``stream`` False entries cannot satisfy the streaming contract.
    """
    entries: Any = data.get('data') if isinstance(data, dict) else data
    if not isinstance(entries, list):
        return []
    out: List[Dict[str, Any]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        mid = str(entry.get('id') or '').strip()
        if not mid or entry.get('usage_based_only'):
            continue  # needs paid credit — not free-pool
        if str(entry.get('model_type') or 'chat').lower() != 'chat':
            continue
        if entry.get('stream') is False:
            continue  # this contract streams
        caps = entry.get('capabilities')
        caps = caps if isinstance(caps, dict) else {}
        modalities = entry.get('modalities')
        modalities = modalities if isinstance(modalities, dict) else {}
        inputs = [str(x).lower() for x in (modalities.get('input') or [])]
        cw = entry.get('context_window')
        cw = cw if isinstance(cw, dict) else {}
        try:
            context = int(cw.get('tokens') or entry.get('context_length') or 0)
        except (TypeError, ValueError):
            context = 0
        try:
            max_out = int(entry.get('max_completion_tokens')
                          or entry.get('max_tokens') or 0)
        except (TypeError, ValueError):
            max_out = 0
        vision = bool(caps.get('vision')) or 'image' in inputs
        thinking = (bool(caps.get('reasoning')) or bool(entry.get('reasoning'))
                    or bool(_RE_THINKING.search(mid)))
        out.append({
            'id': mid,
            'owned_by': str(entry.get('owned_by') or 'llm7') or 'llm7',
            'vision': vision,
            'thinking': thinking,
            'context': context or LLM7_CONTEXT_FALLBACK,
            'max_out': min(max_out, context) if max_out else
            LLM7_MAX_OUTPUT_FALLBACK,
        })
    return out


class Llm7Provider(Provider):
    """api.llm7.io free pool behind the unified provider contract.

    Dormant until an API key is provisioned (LLM7_API_KEY env, the
    ``llm7`` jar, or the refresher signup rung which automates
    dash.llm7.io sign-up + key creation).
    """

    name = 'llm7'

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
            # /models is public on LLM7 — fetched WITHOUT the bearer
            # (a stale/revoked key cannot 401 the catalog away)
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
            logger.warning('llm7 models HTTP %s — keeping %s cached '
                           'models', resp.status_code, len(self._models))
        except Exception as e:  # noqa: BLE001 — best-effort refresh
            logger.warning('llm7 models refresh failed: %s', e)

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
                'no LLM7 API key configured (LLM7_API_KEY env or '
                'llm7_cookies.json): free keys come from dash.llm7.io')
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
            raise ProviderAuthError('no LLM7 API key configured')
        if image_generation:
            raise ProviderError('llm7 has no image generation endpoint')
        self._refresh_catalog(no_proxy=no_proxy)
        entry = self._find_model(model)
        if not entry:
            raise ProviderError(f'llm7 model not in catalog: {model}')

        content: Any = prompt
        if images:
            if not entry.get('vision'):
                raise ProviderError(
                    f'llm7 model {entry["id"]!r} is not vision-capable')
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
                raise ProviderError(f'llm7 stream error: {message}')
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
                'llm7 stream produced no output')
