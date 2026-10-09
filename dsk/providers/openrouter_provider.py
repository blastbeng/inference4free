"""OpenRouter provider (openrouter.ai) — free-tier OpenAI-compatible API.

OpenRouter is an LLM gateway aggregating hundreds of upstream models. The
``:free`` variant of every model (``vendor/name:free``) is served from a
free pool keyed by an API key (``sk-or-v1-…`` from
openrouter.ai/settings/keys): daily request/token limits per model, no
charge. This is the "free API key" provider class alongside the
cookie-jar web providers.

Auth model
----------
    OPENROUTER_API_KEY env
        ->  ``openrouter`` jar (openrouter_cookies.json, ``api_key`` field)
        ->  OPENROUTER_COOKIES env JSON fallback

Endpoints (standard OpenAI shapes)
----------------------------------
    GET  /api/v1/models       PUBLIC — no bearer needed; the full 400+
                              catalog is returned and filtered here down
                              to the ``:free`` ids. Entries carry rich
                              metadata: context_length,
                              architecture.input_modalities,
                              top_provider.max_completion_tokens,
                              supported_parameters, pricing …
    POST /api/v1/chat/completions
        {model, messages, stream: true, max_tokens?, temperature?}
        -> SSE ``data: {choices:[{delta:{content|reasoning}}]}``, [DONE]

Differences vs Groq/Cerebras/ModelScope/Mistral API
---------------------------------------------------
    - the catalog is PUBLIC and UNAUTHENTICATED: fetched WITHOUT the
      bearer (like ModelScope) so a stale key cannot 401 the catalog
      away — but it still only advertises the :free subset this
      provider exposes.
    - per-model flags come from metadata, not id regexes: ``vision``
      from ``architecture.input_modalities`` containing ``image``,
      ``thinking`` from ``supported_parameters`` listing
      ``reasoning``/``include_reasoning`` (id-regex fallback).
    - ``max_output_tokens`` lives at ``top_provider.max_completion_tokens``.
    - reasoning streams as ``delta.reasoning`` (honours
      ``delta.reasoning_content`` too); free models do not inline
      ``<think`` tags.
    - error bodies nest under ``error``:
      ``{"error":{"message":"No cookie auth credentials found","code":401}}``
      on both chat and the authenticated liveness endpoint.

Liveness: GET /api/v1/key with the bearer (the catalog is public and
proves nothing about the key). A 200 returns the per-key usage/limit
record; 401 means rejected; a 429 still proves the key is accepted
(auth runs before rate limiting). Free-tier 429s map to
ProviderRateLimitError via the shared classifier, letting the router
park the model and fall through.
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

logger = logging.getLogger('dsk.providers.openrouter')

OPENROUTER_API_BASE = (os.getenv('I4F_OPENROUTER_API_BASE', '') or
                       'https://openrouter.ai/api/v1').rstrip('/')
MODELS_URL = f'{OPENROUTER_API_BASE}/models'
CHAT_URL = f'{OPENROUTER_API_BASE}/chat/completions'
KEY_URL = f'{OPENROUTER_API_BASE}/key'

_USER_AGENT = ('Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 '
               '(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')

_MODELS_TTL = 300.0
OPENROUTER_CONTEXT_FALLBACK = 131072
OPENROUTER_MAX_OUTPUT_FALLBACK = 8192

# Fallback thinking flag for entries without supported_parameters hints.
_RE_THINKING = re.compile(r'(?:reasoning|thinking|[-_/]r1\b)', re.IGNORECASE)


def _api_key() -> str:
    raw = (os.getenv('OPENROUTER_API_KEY', '') or '').strip()
    if raw:
        return raw
    jar = load_jar('openrouter') or env_cookies('OPENROUTER')
    return (jar.get('api_key') or jar.get('key') or jar.get('token')
            or '').strip()


def _headers(extra: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    hdr = {
        'User-Agent': _USER_AGENT,
        'Accept': '*/*',
        'Authorization': f'Bearer {_api_key()}',
        # attribution headers OpenRouter asks for (app ranking/analytics)
        'HTTP-Referer': 'https://github.com/inference4free',
        'X-Title': 'inference4free',
    }
    if extra:
        hdr.update(extra)
    return hdr


def _classify(status: int, text: str,
              headers: Optional[Any] = None) -> ProviderError:
    if status in (401, 403):
        # {"error":{"message":"No cookie auth credentials found","code":401}}
        return ProviderAuthError(
            f'OpenRouter API key rejected (HTTP {status}): '
            f'{(text or "")[:200]}')
    return classify_http_error(status, text, headers)


def _parse_models(data: Any) -> List[Dict[str, Any]]:
    """Defensive parse of the public /models payload.

    Only ``:free`` ids are exposed (the free pool this provider serves).
    Flags come from the entry metadata: ``vision`` from
    architecture.input_modalities, ``thinking`` from
    supported_parameters (regex fallback), context from
    context_length, max output from top_provider.max_completion_tokens.
    """
    entries: Any = data.get('data') if isinstance(data, dict) else data
    if not isinstance(entries, list):
        return []
    out: List[Dict[str, Any]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        mid = str(entry.get('id') or '').strip()
        if not mid or not mid.endswith(':free'):
            continue
        vendor = mid.split('/', 1)[0] if '/' in mid else 'openrouter'
        try:
            context = int(entry.get('context_length') or 0)
        except (TypeError, ValueError):
            context = 0
        top = entry.get('top_provider')
        top = top if isinstance(top, dict) else {}
        try:
            max_out = int(top.get('max_completion_tokens') or 0)
        except (TypeError, ValueError):
            max_out = 0
        arch = entry.get('architecture')
        arch = arch if isinstance(arch, dict) else {}
        inputs = [str(x).lower() for x in (arch.get('input_modalities')
                                           or [])]
        vision = 'image' in inputs
        sp = [str(x).lower() for x in (entry.get('supported_parameters')
                                       or [])]
        thinking = ('reasoning' in sp or 'include_reasoning' in sp
                    or bool(_RE_THINKING.search(mid)))
        out.append({
            'id': mid,
            'owned_by': vendor,
            'vision': vision,
            'thinking': thinking,
            'context': context or OPENROUTER_CONTEXT_FALLBACK,
            'max_out': min(max_out, context) if max_out else
            OPENROUTER_MAX_OUTPUT_FALLBACK,
        })
    return out


class OpenRouterProvider(Provider):
    """openrouter.ai :free pool behind the unified provider contract.

    Dormant until an API key is provisioned (OPENROUTER_API_KEY env, the
    ``openrouter`` jar, or the refresher signup rung which automates
    openrouter.ai sign-in + key creation).
    """

    name = 'openrouter'

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
            # /models is public on OpenRouter — fetched WITHOUT the bearer
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
            logger.warning('openrouter models HTTP %s — keeping %s cached '
                           'models', resp.status_code, len(self._models))
        except Exception as e:  # noqa: BLE001 — best-effort refresh
            logger.warning('openrouter models refresh failed: %s', e)

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
                'no OpenRouter API key configured (OPENROUTER_API_KEY env '
                'or openrouter_cookies.json): free keys come from '
                'openrouter.ai/settings/keys')
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
            raise ProviderAuthError('no OpenRouter API key configured')
        if image_generation:
            raise ProviderError(
                'openrouter has no image generation endpoint')
        self._refresh_catalog(no_proxy=no_proxy)
        entry = self._find_model(model)
        if not entry:
            raise ProviderError(f'openrouter model not in catalog: {model}')

        content: Any = prompt
        if images:
            if not entry.get('vision'):
                raise ProviderError(
                    f'openrouter model {entry["id"]!r} is not '
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
                raise ProviderError(f'openrouter stream error: {message}')
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
                'openrouter stream produced no output')
