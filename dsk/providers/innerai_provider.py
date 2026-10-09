"""Inner.ai provider — login-only AI workspace, session-cookie class.

inner.ai hosts a multi-model chat workspace behind a login wall. Live
probes (Oct 2026) show the host RESETS connections from every egress path
(direct AND pooled proxies — TLS-level filtering), so its chat endpoint has
NOT been reverse-engineered yet: the RE pass must happen through the shared
browser (DrissionPage renders fine where raw TLS is reset).

This provider therefore ships the full session-cookie contract with the
endpoint PLUGGABLE: it stays dormant until BOTH the browser-run cookies and
the confirmed chat URL exist.

Auth model
----------
    INNERAI_COOKIES env (JSON dict)  →  ``innerai`` jar (innerai_cookies.json)
    I4F_INNERAI_CHAT_URL env         →  RE'd chat endpoint (required)

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

logger = logging.getLogger('dsk.providers.innerai')

INNERAI_BASE = (os.getenv('I4F_INNERAI_BASE', '') or
                'https://inner.ai').rstrip('/')
CHAT_URL = (os.getenv('I4F_INNERAI_CHAT_URL', '') or '').strip()

_USER_AGENT = ('Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 '
               '(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')

INNERAI_CONTEXT_FALLBACK = 32768
INNERAI_MAX_OUTPUT_FALLBACK = 4096

# No verified catalog: the endpoint is unknown. Deliberately EMPTY — a
# fabricated lineup would route real traffic into a guessed endpoint.
_STATIC_MODELS: List[Dict[str, Any]] = []


def _cookies() -> Dict[str, str]:
    jar = load_jar('innerai') or env_cookies('INNERAI')
    if not jar:
        return {}
    return {str(k): str(v) for k, v in jar.items()
            if k not in ('api_key', 'email')}


def _chat_url() -> str:
    # read live (not import-time) so runtime env wiring works
    return ((os.getenv('I4F_INNERAI_CHAT_URL', '') or CHAT_URL)
            or '').strip()


def available_session() -> bool:
    return bool(_cookies())


class InnerAiProvider(Provider):
    """inner.ai session provider — dormant until cookies + RE'd endpoint.

    The signup rung performs the browser login and stores cookies; the chat
    endpoint is then reverse-engineered from the authenticated session
    (network tab) and plugged via I4F_INNERAI_CHAT_URL.
    """

    name = 'innerai'

    def available(self, auth_key: Optional[str] = None) -> bool:
        return available_session() and bool(_chat_url())

    def list_models(self, auth_key: Optional[str] = None) -> List[Dict[str, Any]]:
        if not available_session():
            raise ProviderAuthError(
                'no inner.ai session cookies configured (INNERAI_COOKIES env '
                'or innerai_cookies.json): browser login required — raw TLS '
                'from this egress is connection-reset, so cookies must come '
                'from the shared browser')
        if not _chat_url():
            raise ProviderUnavailableError(
                'inner.ai chat endpoint not yet reverse-engineered — set '
                'I4F_INNERAI_CHAT_URL from the authenticated browser '
                'network tab')
        if not _STATIC_MODELS:
            raise ProviderUnavailableError(
                'inner.ai catalog unknown — populate after the endpoint RE')
        out: List[Dict[str, Any]] = []
        for m in _STATIC_MODELS:
            out.append({
                'id': m['id'],
                'upstream_model': m['id'],
                'thinking_enabled': bool(m.get('thinking')),
                'search_enabled': False,
                'vision': bool(m.get('vision')),
                'image_gen': False,
                'context_length': INNERAI_CONTEXT_FALLBACK,
                'max_output_tokens': INNERAI_MAX_OUTPUT_FALLBACK,
                'extra': {'owned_by': 'inner.ai'},
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
            raise ProviderAuthError('no inner.ai session cookies configured')
        if not _chat_url():
            raise ProviderUnavailableError(
                'inner.ai chat endpoint not yet reverse-engineered — set '
                'I4F_INNERAI_CHAT_URL from the authenticated browser '
                'network tab')
        if not _STATIC_MODELS:
            raise ProviderError(
                f'inner.ai model not in catalog (catalog empty): {model}')

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
                'Origin': INNERAI_BASE,
                'Referer': f'{INNERAI_BASE}/',
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
                    f'inner.ai stream error: {message}')
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
                'inner.ai stream produced no output')
