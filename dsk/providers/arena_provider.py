"""Arena.ai (formerly LMArena / lmarena.ai) provider — reverse-engineered.

Migration note (verified live 2026-10-08): lmarena.ai now redirects to
``arena.ai``. POSTs against the old host die with an empty Cloudflare 405
while the same paths on arena.ai answer normally — always hit ARENA_BASE_URL.

Dormant-by-design credential provider
-------------------------------------
Arena killed anonymous chat: the web app sits behind a login wall and the
stream API answers ``401 {"message":"User not found"}`` without a session.
Provisioning an account is operator-bound work:

  - every public disposable-mail pool we tried is domain-blocklisted
    ("This email domain is not permitted": tempmail.lol, mail.tm);
  - gmail/plus/dot variants all normalize to pre-registered bot accounts;
  - the signup API path requires a reCAPTCHA v3 Enterprise token.

So this module exposes the full reverse-engineered chat pipeline but stays
``available() == False`` until an authenticated cookie set lands in the
``arena`` jar (``arena_cookies.json`` or ``ARENA_COOKIES`` env JSON). The
model catalog, in contrast, is public and always live.

Endpoints
---------
    GET  /nextjs-api/model-catalog        anonymous, 144KB, rotating
        [{"arena": "text", "models": [{"id": uuid, "publicName", …,
          "capabilities": {"inputCapabilities": {"image": bool}}}]}, …]
        NOTE: this endpoint 403s when the request carries an Origin header.
    POST /nextjs-api/stream/create-evaluation   first turn of a chat
        {"id": uuid7, "mode": "direct", "modelAId": <catalog uuid>,
         "userMessageId": uuid7, "modelAMessageId": uuid7,
         "userMessage": {"content": str, "experimental_attachments": [],
                         "metadata": {}},
         "modality": "chat", "recaptchaV3Token": null}
        (the web client sends recaptchaV3Token: null when the widget fails,
        so it is optional in practice; the browser fetch sets no
        Content-Type at all — both application/json and text/plain pass)
    POST /nextjs-api/stream/post-to-evaluation/<session>   follow-up turns

Stream format (Vercel AI SDK UI message stream, one JSON line per code):
    0:"text"     answer delta
    g:"text"     reasoning delta (when a model streams thinking)
    3:"error"    terminal error — surfaced as ProviderError
    d:{finishReason, usage}   final message metadata
    e:/f:/2:/9:… step/data/tool frames — ignored.
"""

import json
import logging
import os
import re
import threading
import time
import uuid
from typing import Any, Dict, Generator, List, Optional

from .base import (
    Provider,
    ProviderAuthError,
    ProviderError,
    ProviderRateLimitError,
    ProviderUnavailableError,
    classify_http_error,
    http_get,
    http_post_stream,
)
from .jar import env_cookies, load_jar

logger = logging.getLogger('dsk.providers.arena')

ARENA_BASE_URL = (os.getenv('I4F_ARENA_BASE_URL', '') or
                  'https://arena.ai').rstrip('/')
CATALOG_URL = f'{ARENA_BASE_URL}/nextjs-api/model-catalog'
CREATE_EVAL_URL = f'{ARENA_BASE_URL}/nextjs-api/stream/create-evaluation'

_USER_AGENT = ('Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 '
               '(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')

_MODELS_TTL = 300.0

# uuidv7 shape used across the payloads (time-ordered ids the web client
# generates with crypto.randomUUID()-equivalent logic).
_RE_UUID = re.compile(
    r'^[0-9a-f]{8}-[0-9a-f]{4}-7[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$')


def uuid7() -> str:
    """RFC 9562 UUIDv7: 48-bit ms timestamp + random, version/variant bits."""
    ts_ms = int(time.time() * 1000)
    b = ts_ms.to_bytes(6, 'big') + os.urandom(10)      # 16 bytes total
    b = (b[:6]                                         # unix_ts_ms
         + bytes([b[6] & 0x0F | 0x70])                 # version 7 + rand_a
         + b[7:8]                                      # rand_b (kept)
         + bytes([b[8] & 0x3F | 0x80])                 # variant 10 + rand_b
         + b[9:])                                      # rand_b (kept)
    h = b.hex()
    return f'{h[:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:]}'


def _session_cookies() -> Dict[str, str]:
    raw = (os.getenv('ARENA_COOKIES', '') or '').strip()
    if raw:
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict) and parsed:
                return {str(k): str(v) for k, v in parsed.items()
                        if k and v is not None}
        except ValueError:
            pass
    return load_jar('arena') or env_cookies('ARENA')


def _headers(extra: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    # No Origin on the catalog GET (403 "Cross-origin catalog requests are
    # not permitted"); POSTs always carry it. No x-arena-request-caller —
    # the client-side fetch never sends it (that header is SSR-only).
    hdr = {
        'User-Agent': _USER_AGENT,
        'Accept': '*/*',
        'Referer': f'{ARENA_BASE_URL}/text/direct',
    }
    if extra:
        hdr.update(extra)
    return hdr


def _classify(status: int, text: str, headers: Optional[Any] = None) -> ProviderError:
    if status in (401, 403):
        # 401 {"message":"User not found"} = no/ expired session cookie;
        # 403 = Cloudflare WAF or login wall — both are auth-refresher work.
        return ProviderAuthError(
            f'arena session rejected (HTTP {status}): {(text or "")[:200]}')
    if status == 429:
        return ProviderRateLimitError(
            f'arena rate limited (HTTP 429): {(text or "")[:200]}')
    return classify_http_error(status, text, headers)


def _parse_catalog(data: Any) -> List[Dict[str, Any]]:
    """Text-arena models from the catalog: only userSelectable entries.

    The JSON mixes dicts and plain strings, so every element is type-checked
    before .get() (live crash: AttributeError on 'str' entries).
    """
    models: List[Dict[str, Any]] = []
    if not isinstance(data, list):
        return models
    for block in data:
        if not (isinstance(block, dict) and block.get('arena') == 'text'):
            continue
        for entry in block.get('models') or []:
            if not isinstance(entry, dict) or not entry.get('userSelectable'):
                continue
            mid = str(entry.get('id') or '').strip()
            if not mid:
                continue
            name = (str(entry.get('publicName') or entry.get('displayName')
                        or entry.get('name') or '').strip() or mid)
            caps = entry.get('capabilities') or {}
            if not isinstance(caps, dict):
                caps = {}
            in_caps = caps.get('inputCapabilities') or {}
            if not isinstance(in_caps, dict):
                in_caps = {}
            out_caps = caps.get('outputCapabilities') or {}
            if not isinstance(out_caps, dict):
                out_caps = {}
            models.append({
                'public_name': name,
                'upstream': mid,
                'organization': entry.get('organization') or None,
                'vision': bool(in_caps.get('image')),
                'image_out': bool(out_caps.get('image')),
            })
    return models


class ArenaProvider(Provider):
    """arena.ai direct-chat models behind the unified provider contract.

    Dormant until an authenticated session cookie set is provisioned (see
    module docstring). The public catalog still loads so ops tooling can
    verify upstream reachability while the account is missing.
    """

    name = 'arena'

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._catalog_ts = 0.0
        self._models: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------ auth
    def available(self, auth_key: Optional[str] = None) -> bool:
        return bool(_session_cookies())

    # ---------------------------------------------------------------- models
    def _refresh_catalog(self, no_proxy: bool = False) -> None:
        now = time.time()
        with self._lock:
            if self._models and now - self._catalog_ts < _MODELS_TTL:
                return
        try:
            resp = http_get(CATALOG_URL, headers=_headers(), timeout=30,
                            no_proxy=no_proxy)
            if resp.status_code == 200:
                parsed = _parse_catalog(resp.json())
                if parsed:
                    with self._lock:
                        self._models = parsed
                        self._catalog_ts = now
                    return
            logger.warning('arena catalog HTTP %s — keeping %s cached models',
                           resp.status_code, len(self._models))
        except Exception as e:  # noqa: BLE001 — best-effort refresh
            logger.warning('arena catalog refresh failed: %s', e)

    def _find_model(self, key: str) -> Optional[Dict[str, Any]]:
        for m in self._models:
            if key in (m['upstream'], m['public_name']):
                return m
        lowered = key.lower()
        for m in self._models:
            if m['public_name'].lower() == lowered:
                return m
        return None

    def list_models(self, auth_key: Optional[str] = None) -> List[Dict[str, Any]]:
        if not self.available():
            raise ProviderAuthError(
                'no arena.ai session configured (arena_cookies.json / '
                'ARENA_COOKIES): signup requires an allowlisted email domain')
        self._refresh_catalog()
        out: List[Dict[str, Any]] = []
        seen: set = set()
        for m in self._models:
            # publicName may repeat (same public name from two providers) —
            # keep the first, higher-ranked entry (catalog order = rank).
            if m['public_name'] in seen:
                continue
            seen.add(m['public_name'])
            out.append({
                'id': m['public_name'],
                'upstream_model': m['upstream'],
                'thinking_enabled': False,
                'search_enabled': False,
                'vision': m['vision'],
                'image_gen': False,
                'context_length': 131072,
                'max_output_tokens': 8192,
                'extra': {'organization': m['organization'],
                          'image_out': m['image_out']},
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
        cookies = _session_cookies()
        if not cookies:
            raise ProviderAuthError('no arena.ai session configured')
        if images:
            # experimental_attachments exists but its exact shape is
            # unverified — refuse rather than send a broken payload.
            raise ProviderError('arena image input is not supported yet')
        self._refresh_catalog(no_proxy=no_proxy)
        entry = self._find_model(model)
        if not entry:
            raise ProviderError(f'arena model not in catalog: {model}')

        payload = {
            'id': uuid7(),
            'mode': 'direct',
            'modelAId': entry['upstream'],
            'userMessageId': uuid7(),
            'modelAMessageId': uuid7(),
            'userMessage': {
                'content': prompt,
                'experimental_attachments': [],
                'metadata': {},
            },
            'modality': 'chat',
            'recaptchaV3Token': None,
        }
        resp = http_post_stream(
            CREATE_EVAL_URL,
            headers=_headers({'Origin': ARENA_BASE_URL}),
            json_body=payload,
            cookies=cookies,
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
        """Parse the AI SDK UI message stream (one `code:json` line each)."""
        yielded = 0
        for raw in resp.iter_lines():
            if isinstance(raw, bytes):
                line = raw.decode('utf-8', 'replace').strip()
            else:
                line = str(raw).strip()
            if not line or ':' not in line:
                continue
            code, _, rest = line.partition(':')
            if code == '0':          # text delta
                try:
                    text = json.loads(rest)
                except ValueError:
                    continue
                if text:
                    yielded += 1
                    yield {'content': str(text), 'type': 'text',
                           'finish_reason': None}
            elif code == 'g':        # reasoning delta
                try:
                    text = json.loads(rest)
                except ValueError:
                    continue
                if text:
                    yielded += 1
                    yield {'content': str(text), 'type': 'thinking',
                           'finish_reason': None}
            elif code == '3':        # terminal error frame
                try:
                    text = json.loads(rest)
                except ValueError:
                    text = rest
                raise ProviderError(f'arena stream error: {str(text)[:300]}')
            elif code == 'd':        # finish_message {finishReason, usage}
                finish = None
                try:
                    data = json.loads(rest)
                    if isinstance(data, dict):
                        reason = str(data.get('finishReason') or '').lower()
                        if reason and reason not in ('unknown', 'stop'):
                            finish = reason if reason != 'other' else 'stop'
                        elif reason == 'stop':
                            finish = 'stop'
                except ValueError:
                    pass
                yield {'content': '', 'type': 'text',
                       'finish_reason': finish or 'stop'}
                return
        # No `d:` frame but a clean EOF: honor what was streamed, else the
        # router treats the target as stalled and falls back.
        if yielded:
            yield {'content': '', 'type': 'text', 'finish_reason': 'stop'}
        else:
            raise ProviderUnavailableError('arena stream produced no output')
