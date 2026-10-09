"""Blackbox AI provider (api.blackbox.ai) — free-tier OpenAI-compatible API.

Blackbox runs an OpenAI-compatible inference API. The OLD reverse-engineered
web endpoint (www.blackbox.ai/api/chat with the ``validated`` uuid) is DEAD
(404 dpl_ HTML since the site redeploy) — the official documented surface is
``api.blackbox.ai/chat/completions`` with a Bearer key from
docs.blackbox.ai / the dashboard. This is the "free API key" provider class
alongside groq/cerebras/cohere.

Auth model
----------
    BLACKBOX_API_KEY env  →  ``blackbox`` jar (blackbox_cookies.json,
                              ``api_key`` field)
                          →  BLACKBOX_COOKIES env JSON fallback

``available()`` is just key presence; the refresh rung verifies liveness
with a 1-token POST /chat/completions (free tier — negligible burn; there is
no unauthenticated models endpoint).

Endpoints (documented)
----------------------
    POST /chat/completions
        {model, messages, stream: true, max_tokens?}
        → SSE ``data: {choices:[{delta}]}``, ``data: [DONE]``
    NOTE: no /v1 prefix — the path is root-level.

Catalog (verified via the historical RE proxy payload; static — Blackbox
proxies partner models and drifts without notice):
    meta-llama/Llama-3.3-70B-Instruct-Turbo
    deepseek-chat / deepseek-reasoner
    deepseek-ai/deepseek-llm-67b-chat
"""

import base64
import json
import logging
import os
import re
from typing import Any, Dict, Generator, List, Optional

from .base import (
    Provider,
    ProviderAuthError,
    ProviderError,
    ProviderUnavailableError,
    classify_http_error,
    http_post_stream,
    parse_sse_data,
)
from .jar import env_cookies, load_jar

logger = logging.getLogger('dsk.providers.blackbox')

BLACKBOX_API_BASE = (os.getenv('I4F_BLACKBOX_API_BASE', '') or
                     'https://api.blackbox.ai').rstrip('/')
CHAT_URL = f'{BLACKBOX_API_BASE}/chat/completions'

_USER_AGENT = ('Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 '
               '(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')

BLACKBOX_CONTEXT_FALLBACK = 32768
BLACKBOX_MAX_OUTPUT_FALLBACK = 4096

# Static catalog (partner models behind Blackbox; deepseek-reasoner streams
# reasoning deltas — the only thinking-capable entry).
_STATIC_MODELS: List[Dict[str, Any]] = [
    {'id': 'deepseek-chat', 'owned_by': 'deepseek',
     'context': 65536, 'max_out': 4096},
    {'id': 'deepseek-reasoner', 'owned_by': 'deepseek',
     'context': 65536, 'max_out': 8192},
    {'id': 'meta-llama/Llama-3.3-70B-Instruct-Turbo', 'owned_by': 'meta',
     'context': 131072, 'max_out': 4096},
    {'id': 'deepseek-ai/deepseek-llm-67b-chat', 'owned_by': 'deepseek',
     'context': 16384, 'max_out': 4096},
]

_RE_THINKING = re.compile(r'(?:reasoner|reasoning|thinking|[-_/]r1\b)',
                          re.IGNORECASE)
_RE_VISION = re.compile(r'(?:vision|\bvl\b|vl-)', re.IGNORECASE)


def _api_key() -> str:
    raw = (os.getenv('BLACKBOX_API_KEY', '') or '').strip()
    if raw:
        return raw
    jar = load_jar('blackbox') or env_cookies('BLACKBOX')
    return (jar.get('api_key') or jar.get('key') or jar.get('token')
            or '').strip()


def _headers(extra: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    hdr = {
        'User-Agent': _USER_AGENT,
        'Accept': '*/*',
        'Authorization': f'Bearer {_api_key()}',
        'Content-Type': 'application/json',
    }
    if extra:
        hdr.update(extra)
    return hdr


def _classify(status: int, text: str,
              headers: Optional[Any] = None) -> ProviderError:
    if status in (401, 403):
        return ProviderAuthError(
            f'blackbox API key rejected (HTTP {status}): {(text or "")[:200]}')
    return classify_http_error(status, text, headers)


def verify_key(key: str) -> 'tuple':
    """Shared liveness probe (refresher rung + tests): 1-token completion.

    401/403 → rejected; anything else proves the key routed (429 accepted,
    400 with a bad model still proves AUTH passed — auth runs first).
    """
    try:
        resp = http_post_stream(
            CHAT_URL,
            headers={'Authorization': f'Bearer {key}',
                     'Content-Type': 'application/json',
                     'Accept': 'application/json'},
            json_body={'model': 'deepseek-chat',
                       'messages': [{'role': 'user', 'content': 'ping'}],
                       'stream': False, 'max_tokens': 1},
            timeout=60,
        )
    except Exception as e:  # noqa: BLE001
        return False, f'verify failed: {type(e).__name__}: {e}'
    if resp.status_code in (401, 403):
        return False, ('API key rejected - create a new key at '
                       'docs.blackbox.ai and set BLACKBOX_API_KEY')
    if resp.status_code == 429:
        return True, 'rate limited but key accepted (HTTP 429)'
    if resp.status_code != 200:
        return True, (f'key reachable, liveness inconclusive (HTTP '
                      f'{resp.status_code}) - validated at request time')
    return True, 'API key valid (1-token completion OK)'


class BlackboxProvider(Provider):
    """api.blackbox.ai free tier behind the unified provider contract.

    Dormant until an API key is provisioned (BLACKBOX_API_KEY env, the
    ``blackbox`` jar, or the refresher signup rung which automates the
    Blackbox dashboard).
    """

    name = 'blackbox'

    def available(self, auth_key: Optional[str] = None) -> bool:
        return bool(_api_key())

    def list_models(self, auth_key: Optional[str] = None) -> List[Dict[str, Any]]:
        if not self.available():
            raise ProviderAuthError(
                'no Blackbox API key configured (BLACKBOX_API_KEY env or '
                'blackbox_cookies.json): the inference API is free; create a '
                'key per docs.blackbox.ai')
        out: List[Dict[str, Any]] = []
        for m in _STATIC_MODELS:
            out.append({
                'id': m['id'],
                'upstream_model': m['id'],
                'thinking_enabled': bool(_RE_THINKING.search(m['id'])),
                'search_enabled': False,
                'vision': bool(_RE_VISION.search(m['id'])),
                'image_gen': False,
                'context_length': m.get('context') or BLACKBOX_CONTEXT_FALLBACK,
                'max_output_tokens': m.get('max_out') or
                BLACKBOX_MAX_OUTPUT_FALLBACK,
                'extra': {'owned_by': m['owned_by']},
            })
        return out

    def stream(self, prompt: str, *, model: str, thinking_enabled: bool = False,
               search_enabled: bool = False, temperature: Optional[float] = None,
               max_tokens: Optional[int] = None,
               images: Optional[List[Dict[str, Any]]] = None,
               image_generation: bool = False,
               no_proxy: bool = False,
               auth_key: Optional[str] = None) -> Generator[Dict[str, Any], None, None]:
        if not self.available():
            raise ProviderAuthError('no Blackbox API key configured')
        if image_generation:
            raise ProviderError('blackbox has no image generation endpoint')
        if images:
            raise ProviderError(
                'blackbox: no vision-capable model in catalog')
        model_id = model
        for m in _STATIC_MODELS:
            if m['id'].lower() == model.lower():
                model_id = m['id']
                break

        payload: Dict[str, Any] = {
            'model': model_id,
            'messages': [{'role': 'user', 'content': prompt}],
            'stream': True,
        }
        if temperature is not None:
            payload['temperature'] = temperature
        if max_tokens:
            payload['max_tokens'] = max_tokens

        resp = http_post_stream(
            CHAT_URL,
            headers=_headers({'Accept': 'text/event-stream'}),
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
                raise ProviderUnavailableError(
                    f'blackbox stream error: {message}')
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
                'blackbox stream produced no output')
