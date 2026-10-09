"""Adapta provider — login-based multi-model workspace, session-cookie class.

adapta.org is a pt-BR product site (Framer) marketing "Adapta One", a
login-based multi-model workspace. No public API, no anonymous chat, and no
documented OpenAI-compatible surface were found (Oct 2026 research): the
chat endpoint must be reverse-engineered from an authenticated browser
session, same pluggable-endpoint approach as innerai.

Auth model
----------
    ADAPTA_COOKIES env (JSON dict)  →  ``adapta`` jar (adapta_cookies.json)
    I4F_ADAPTA_CHAT_URL env         →  RE'd chat endpoint (required)

``available()`` = cookies AND chat URL. With cookies but no endpoint the
refresher keeps the session warm and /setup reports the gap honestly.
"""

import json
import logging
import os
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

logger = logging.getLogger('dsk.providers.adapta')

ADAPTA_BASE = (os.getenv('I4F_ADAPTA_BASE', '') or
               'https://adapta.org').rstrip('/')
CHAT_URL = (os.getenv('I4F_ADAPTA_CHAT_URL', '') or '').strip()

_USER_AGENT = ('Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 '
               '(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')

ADAPTA_CONTEXT_FALLBACK = 32768
ADAPTA_MAX_OUTPUT_FALLBACK = 4096

# No verified catalog: no public API found. Deliberately EMPTY — a fabricated
# lineup would route real traffic into a guessed endpoint.
_STATIC_MODELS: List[Dict[str, Any]] = []


def _cookies() -> Dict[str, str]:
    jar = load_jar('adapta') or env_cookies('ADAPTA')
    if not jar:
        return {}
    return {str(k): str(v) for k, v in jar.items()
            if k not in ('api_key', 'email')}


def _chat_url() -> str:
    # read live (not import-time) so runtime env wiring works
    return ((os.getenv('I4F_ADAPTA_CHAT_URL', '') or CHAT_URL)
            or '').strip()


def available_session() -> bool:
    return bool(_cookies())


class AdaptaProvider(Provider):
    """adapta.org session provider — dormant until cookies + RE'd endpoint.

    The signup rung performs the browser login and stores cookies; the chat
    endpoint is then reverse-engineered from the authenticated session and
    plugged via I4F_ADAPTA_CHAT_URL.
    """

    name = 'adapta'

    def available(self, auth_key: Optional[str] = None) -> bool:
        return available_session() and bool(_chat_url())

    def list_models(self, auth_key: Optional[str] = None) -> List[Dict[str, Any]]:
        if not available_session():
            raise ProviderAuthError(
                'no adapta session cookies configured (ADAPTA_COOKIES env or '
                'adapta_cookies.json): adapta.org is login-only — browser '
                'login required')
        if not _chat_url():
            raise ProviderUnavailableError(
                'adapta chat endpoint not yet reverse-engineered — set '
                'I4F_ADAPTA_CHAT_URL from the authenticated browser '
                'network tab')
        if not _STATIC_MODELS:
            raise ProviderUnavailableError(
                'adapta catalog unknown — populate after the endpoint RE')
        out: List[Dict[str, Any]] = []
        for m in _STATIC_MODELS:
            out.append({
                'id': m['id'],
                'upstream_model': m['id'],
                'thinking_enabled': bool(m.get('thinking')),
                'search_enabled': False,
                'vision': bool(m.get('vision')),
                'image_gen': False,
                'context_length': ADAPTA_CONTEXT_FALLBACK,
                'max_output_tokens': ADAPTA_MAX_OUTPUT_FALLBACK,
                'extra': {'owned_by': 'adapta'},
            })
        return out

    def stream(self, prompt: str, *, model: str, thinking_enabled: bool = False,
               search_enabled: bool = False, temperature: Optional[float] = None,
               max_tokens: Optional[int] = None,
               images: Optional[List[Dict[str, Any]]] = None,
               image_generation: bool = False,
               no_proxy: bool = False,
               auth_key: Optional[str] = None) -> Generator[Dict[str, Any], None, None]:
        if not available_session():
            raise ProviderAuthError('no adapta session cookies configured')
        if not _chat_url():
            raise ProviderUnavailableError(
                'adapta chat endpoint not yet reverse-engineered — set '
                'I4F_ADAPTA_CHAT_URL from the authenticated browser '
                'network tab')
        if not _STATIC_MODELS:
            raise ProviderError(
                f'adapta model not in catalog (catalog empty): {model}')

        payload: Dict[str, Any] = {
            'model': model,
            'messages': [{'role': 'user', 'content': prompt}],
            'stream': True,
        }
        resp = http_post_stream(
            _chat_url(),
            headers={
                'User-Agent': _USER_AGENT,
                'Content-Type': 'application/json',
                'Accept': 'text/event-stream',
                'Origin': ADAPTA_BASE,
                'Referer': f'{ADAPTA_BASE}/',
            },
            json_body=payload,
            cookies=_cookies(),
            timeout=300,
            no_proxy=no_proxy,
        )
        if resp.status_code != 200:
            try:
                error_text = resp.text or ''
            except Exception:  # pragma: no cover
                error_text = f'HTTP {resp.status_code}'
            raise classify_http_error(resp.status_code, error_text,
                                      resp.headers)
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
                    f'adapta stream error: {message}')
            choices = data.get('choices') or []
            if not choices:
                continue
            delta = (choices[0] or {}).get('delta') or {}
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
                'adapta stream produced no output')
