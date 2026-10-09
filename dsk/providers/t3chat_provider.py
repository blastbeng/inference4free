"""t3.chat provider — free "LLM chat" web app behind a browser session.

t3.chat (Vercel-hosted, Zero-Data-Retention marketing) offers a multi-model
picker free of charge. Datacenter IPs hit the Vercel Security Checkpoint
(429 + HTML challenge — verified live), so access requires REAL browser
session cookies harvested by the refresher signup rung (browser passes the
checkpoint, cookies persist in the ``t3chat`` jar).

Auth model
----------
    T3CHAT_COOKIES env (JSON dict)  →  ``t3chat`` jar (t3chat_cookies.json)
        session cookies from an authenticated browser login.

``available()`` is cookie presence; the refresh rung GETs the root with the
cookies — 200 = alive, 429 = checkpoint (cookies stale / egress blocked).

Endpoints (reverse-engineered from the Next.js app; AI-SDK "UI message"
protocol)
---------------------------------------------------------------------------
    POST /api/chat
        {id: <uuidv7-ish>, model: <id>, messages: [{id, role, parts:
         [{type: "text", text}]}], ...}
        → AI-SDK data stream: SSE ``data: {"type":"text-delta","delta":".."} ``
          frames (v5) or the legacy ``0:"text"`` / ``2:[...]`` custom lines;
          a Vercel checkpoint arrives as 429 + HTML.

Catalog: scraped from the app's own model picker (ids embedded in the page
bundle) with a static fallback of the long-running lineup — a dormant
provider ships the fallback only when scraping is impossible.
"""

import json
import logging
import os
import re
import time
import uuid
from typing import Any, Dict, Generator, List, Optional, Tuple

from .base import (
    Provider,
    ProviderAuthError,
    ProviderError,
    ProviderUnavailableError,
    classify_http_error,
    http_get,
    http_post_stream,
)
from .jar import env_cookies, load_jar

logger = logging.getLogger('dsk.providers.t3chat')

T3CHAT_BASE = (os.getenv('I4F_T3CHAT_BASE', '') or
               'https://www.t3.chat').rstrip('/')
CHAT_URL = f'{T3CHAT_BASE}/api/chat'

_USER_AGENT = ('Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 '
               '(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')

_MODELS_TTL = 900.0
T3_CONTEXT_FALLBACK = 32768
T3_MAX_OUTPUT_FALLBACK = 4096

# Static fallback lineup (long-running entries of the t3.chat picker).
_STATIC_MODELS: List[Dict[str, Any]] = [
    {'id': 'gpt-4o', 'owned_by': 'openai', 'vision': True},
    {'id': 'gpt-4.1', 'owned_by': 'openai', 'vision': True},
    {'id': 'o4-mini', 'owned_by': 'openai', 'thinking': True},
    {'id': 'claude-sonnet-4', 'owned_by': 'anthropic', 'vision': True},
    {'id': 'gemini-2.5-pro', 'owned_by': 'google', 'thinking': True,
     'vision': True},
    {'id': 'gemini-2.5-flash', 'owned_by': 'google', 'thinking': True},
    {'id': 'deepseek-r1', 'owned_by': 'deepseek', 'thinking': True},
    {'id': 'deepseek-v3', 'owned_by': 'deepseek'},
    {'id': 'llama-4-maverick', 'owned_by': 'meta', 'vision': True},
    {'id': 'qwen-3-235b', 'owned_by': 'alibaba', 'thinking': True},
    {'id': 'grok-3', 'owned_by': 'xai'},
]

_RE_THINKING = re.compile(r'(?:reasoner|reasoning|thinking|[-_/]r1\b'
                          r'|o[134](-mini|-preview)?\b)', re.IGNORECASE)


def _cookies() -> Dict[str, str]:
    jar = load_jar('t3chat') or env_cookies('T3CHAT')
    if not jar:
        return {}
    out: Dict[str, str] = {}
    for k, v in jar.items():
        if k in ('api_key', 'email'):
            continue
        out[str(k)] = str(v)
    return out


def _is_checkpoint(status: int, text: str) -> bool:
    """Vercel Security Checkpoint: 429 (or 403) + challenge HTML."""
    if status not in (429, 403):
        return False
    return not (text or '').lstrip()[:1] == '{'


def _parse_stream_line(raw: Any) -> Optional[Tuple[str, str]]:
    """One AI-SDK data-stream line → (kind, text) or None.

    Handles both stream protocols:
      SSE   ``data: {"type":"text-delta","delta":"..."}`` (v5 typed frames)
      legacy ``0:"string"`` / ``0:"...",\"...\"`` (concat-escaped parts) and
      ``2:"[json]"`` custom data (skipped).
    """
    try:
        line = raw.decode('utf-8', 'ignore') \
            if isinstance(raw, (bytes, bytearray)) else str(raw or '')
    except Exception:  # pragma: no cover
        return None
    text = line.strip()
    if not text:
        return None
    if text.startswith('data:'):
        payload = text[5:].strip()
        if not payload or payload == '[DONE]':
            return None
        try:
            obj = json.loads(payload)
        except ValueError:
            return None
        if isinstance(obj, dict):
            # v5 UI message stream: only text-delta frames carry content
            if obj.get('type') in ('text-delta', 'text_delta'):
                delta = obj.get('delta') or obj.get('textDelta') or ''
                return ('text', str(delta)) if delta else None
            if obj.get('type') in ('reasoning-delta', 'reasoning_delta',
                                   'reasoning'):
                delta = obj.get('delta') or obj.get('reasoning') or ''
                return ('thinking', str(delta)) if delta else None
            if obj.get('error') or obj.get('type') == 'error':
                msg = obj.get('error') or obj.get('message') or 'stream error'
                return ('error', str(msg))
        return None
    # legacy custom-data format: <digit>:"..."
    m = re.match(r'^(\d):(.*)$', text)
    if not m:
        return None
    kind_code, body = m.group(1), m.group(2)
    if kind_code == '0':
        try:
            parts = json.loads(f'[{body}]') if body else []
        except ValueError:
            parts = []
        joined = ''.join(str(p) for p in parts if isinstance(p, str))
        return ('text', joined) if joined else None
    return None  # 1/2/3/9: annotations, tool calls, finish metadata


def _scrape_models(html: str) -> List[Dict[str, Any]]:
    """Best-effort model-id scrape from the app page (picker bundle).

    Model ids appear in the serialized options as short id-ish strings
    (gpt-4o, claude-sonnet-4, ...). Defensive: anything that fails simply
    yields an empty list.
    """
    ids = set(re.findall(
        r'"((?:gpt|o[134]|claude|gemini|deepseek|llama|qwen|grok|mistral'
        r'|phi|command)[a-z0-9.\-_]{1,40})"', html or '', re.IGNORECASE))
    known = {m['id'] for m in _STATIC_MODELS}
    out: List[Dict[str, Any]] = []
    for mid in sorted(ids):
        if re.fullmatch(r'[a-z0-9][a-z0-9.\-_]{2,49}', mid.lower()) and \
                not mid.lower().endswith(('.png', '.js', '.css', '.woff2')):
            out.append({'id': mid, 'owned_by': 't3.chat'})
    # keep static entries the scrape missed so the catalog never shrinks
    scraped = {m['id'] for m in out}
    for m in _STATIC_MODELS:
        if m['id'] not in scraped:
            out.append(dict(m))
    known = None  # noqa: F841 — documented intent: merge, not replace
    return out


def verify_cookies() -> Tuple[bool, str]:
    """Shared liveness probe (refresher rung + tests): GET / with cookies.

    200 → session alive; 429/403 challenge → cookies stale or egress
    blocklisted (Vercel Security Checkpoint).
    """
    cookies = _cookies()
    if not cookies:
        return False, ('no t3.chat cookies - browser login at www.t3.chat '
                       'then save them to t3chat_cookies.json')
    try:
        resp = http_get(f'{T3CHAT_BASE}/', headers={
            'User-Agent': _USER_AGENT, 'Accept': 'text/html'}, timeout=30,
            cookies=cookies)
    except Exception as e:  # noqa: BLE001
        return False, f'verify failed: {type(e).__name__}: {e}'
    text = ''
    try:
        text = resp.text or ''
    except Exception:  # pragma: no cover
        pass
    if _is_checkpoint(resp.status_code, text):
        return False, ('Vercel Security Checkpoint (HTTP '
                       f'{resp.status_code}) - cookies stale or the egress '
                       'IP is blocklisted; redo the browser login')
    if resp.status_code != 200:
        return False, (f'session rejected (HTTP {resp.status_code}) - '
                       'redo the browser login')
    return True, 'session cookies valid'


class T3ChatProvider(Provider):
    """t3.chat web app behind the unified provider contract.

    Dormant until session cookies are provisioned (T3CHAT_COOKIES env, the
    ``t3chat`` jar, or the refresher signup rung which logs in via the
    shared browser — the checkpoint blocks every headless-less path).
    """

    name = 't3chat'

    def __init__(self) -> None:
        self._catalog_ts = 0.0
        self._models: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------ auth
    def available(self, auth_key: Optional[str] = None) -> bool:
        return bool(_cookies())

    # ---------------------------------------------------------------- models
    def _refresh_catalog(self, no_proxy: bool = False) -> None:
        now = time.time()
        if self._models and now - self._catalog_ts < _MODELS_TTL:
            return
        try:
            resp = http_get(f'{T3CHAT_BASE}/', headers={
                'User-Agent': _USER_AGENT, 'Accept': 'text/html'},
                cookies=_cookies(), timeout=30, no_proxy=no_proxy)
            if resp.status_code == 200:
                scraped = _scrape_models(resp.text or '')
                if scraped:
                    self._models = scraped
                    self._catalog_ts = now
                    return
        except Exception as e:  # noqa: BLE001 — best-effort refresh
            logger.warning('t3chat models scrape failed: %s', e)
        self._models = [dict(m) for m in _STATIC_MODELS]
        self._catalog_ts = now

    def _find_model(self, key: str) -> Optional[Dict[str, Any]]:
        for m in self._models:
            if m['id'].lower() == key.lower():
                return m
        return None

    def list_models(self, auth_key: Optional[str] = None) -> List[Dict[str, Any]]:
        if not self.available():
            raise ProviderAuthError(
                'no t3.chat session cookies configured (T3CHAT_COOKIES env '
                'or t3chat_cookies.json): the Vercel checkpoint blocks '
                'datacenter IPs, so cookies must come from a browser login')
        self._refresh_catalog()
        out: List[Dict[str, Any]] = []
        for m in self._models:
            out.append({
                'id': m['id'],
                'upstream_model': m['id'],
                'thinking_enabled': bool(m.get('thinking')
                                         or _RE_THINKING.search(m['id'])),
                'search_enabled': False,
                'vision': bool(m.get('vision')),
                'image_gen': False,
                'context_length': T3_CONTEXT_FALLBACK,
                'max_output_tokens': T3_MAX_OUTPUT_FALLBACK,
                'extra': {'owned_by': m.get('owned_by') or 't3.chat'},
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
            raise ProviderAuthError('no t3.chat session cookies configured')
        if image_generation:
            raise ProviderError('t3chat has no image generation endpoint')
        if images:
            raise ProviderError('t3chat: image input not supported yet')
        self._refresh_catalog(no_proxy=no_proxy)
        if not self._find_model(model):
            raise ProviderError(f't3chat model not in catalog: {model}')

        payload: Dict[str, Any] = {
            'id': str(uuid.uuid4()),
            'model': model,
            'messages': [{'id': str(uuid.uuid4()), 'role': 'user',
                          'parts': [{'type': 'text', 'text': prompt}]}],
        }
        resp = http_post_stream(
            CHAT_URL,
            headers={
                'User-Agent': _USER_AGENT,
                'Content-Type': 'application/json',
                'Accept': 'text/event-stream',
                'Origin': T3CHAT_BASE,
                'Referer': f'{T3CHAT_BASE}/',
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
            if _is_checkpoint(resp.status_code, error_text):
                raise ProviderUnavailableError(
                    f't3chat Vercel Security Checkpoint (HTTP '
                    f'{resp.status_code}) - session cookies are not '
                    'passing; redo the browser login')
            raise classify_http_error(resp.status_code, error_text,
                                      resp.headers)
        return self._iter_chunks(resp)

    def _iter_chunks(self, resp: Any) -> Generator[Dict[str, Any], None, None]:
        saw_content = False
        for raw in resp.iter_lines():
            parsed = _parse_stream_line(raw)
            if not parsed:
                continue
            kind, text = parsed
            if kind == 'error':
                raise ProviderUnavailableError(f't3chat stream error: {text}')
            if text:
                saw_content = True
                yield {'content': text, 'type': kind, 'finish_reason': None}
        if saw_content:
            yield {'content': '', 'type': 'text', 'finish_reason': 'stop'}
        else:
            raise ProviderUnavailableError(
                't3chat stream produced no output')
