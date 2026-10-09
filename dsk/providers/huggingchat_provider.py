"""HuggingChat provider (huggingface.co/chat) — reverse-engineered.

Verified live 2026-10-08 against the SvelteKit app shipped at
huggingface.co/chat (PUBLIC_VERSION 0.20.0, "Omni" router branding).

Auth model
----------
The chat backend answers ``401 {"error":"You have to be logged in."}`` for
every conversation/message POST without an authenticated session (the
anonymous visit gets pushed into an HF OAuth2/PKCE flow, client_id
8f1a1d63-…). The model catalog, in contrast, is public:

    GET /chat/api/v2/models        anonymous — {"json": [model, …]}

So this is a cookie-jar provider: an operator browser session (or a
bot-created account via the refresher signup rung) lands its cookies in
``huggingchat_cookies.json`` / ``HUGGINGCHAT_COOKIES`` env JSON and the
provider goes live. The HF email-verification message carries a magic link,
which the signup rung resolves through ``mailgen.fetch_magic_link``.

Endpoints
---------
    GET  /chat/api/v2/models            anonymous catalog (129 entries)
        {"json": [{"id", "displayName", "multimodal", "supportsReasoning",
                   "supportsTools", "unlisted", "isRouter", …}]}
    GET  /chat/api/v2/user              {"json": null} anonymous,
                                        {"json": {...}} authenticated
    POST /chat/conversation             create a conversation
        {"model": "<hf-org/name>", "preprompt": "", "mlAssistant": false}
        → 200 {"conversationId": "<24-hex>"}
    POST /chat/conversation/<id>        send a message, NDJSON answer
        multipart/form-data with one field:
          data = {"inputs": str, "id": <message uuid>, "is_retry": false,
                  "is_continue": false, "timezone": "UTC"}

Stream format (newline-delimited JSON, one event object per line)
-----------------------------------------------------------------
    {"type":"stream","token":"…"}            answer delta
    {"type":"reasoning", …}                  reasoning delta (thinking models)
    {"type":"finalAnswer","text":"…","len":N}  full answer text (canonical)
    {"type":"status","status":"error","message":"…"}   terminal failure
    {"type":"status","status":"finished"|"started"|"keepAlive"}
    {"type":"title"|"tool"|"file"|"plan"|"budget"|"turnState"|…}  ignored

The client treats ``finalAnswer.text`` as canonical: it replaces whatever
the ``stream`` deltas accumulated (the deltas may be truncated chunk
previews). This provider streams the deltas live and, on ``finalAnswer``,
emits only the missing suffix so OpenAI consumers see exactly the
canonical text once, with nothing duplicated.
"""

import json
import logging
import os
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
    http_post_raw,
)
from .jar import env_cookies, load_jar

logger = logging.getLogger('dsk.providers.huggingchat')

HUGGINGCHAT_BASE_URL = (os.getenv('I4F_HUGGINGCHAT_BASE_URL', '') or
                        'https://huggingface.co/chat').rstrip('/')
MODELS_URL = f'{HUGGINGCHAT_BASE_URL}/api/v2/models'
USER_URL = f'{HUGGINGCHAT_BASE_URL}/api/v2/user'
CREATE_URL = f'{HUGGINGCHAT_BASE_URL}/conversation'

_USER_AGENT = ('Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 '
               '(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')

_MODELS_TTL = 300.0
HUGGINGCHAT_CONTEXT_LENGTH = int(
    os.getenv('I4F_HUGGINGCHAT_CONTEXT_LENGTH', '131072'))
HUGGINGCHAT_MAX_OUTPUT = int(
    os.getenv('I4F_HUGGINGCHAT_MAX_OUTPUT', '8192'))


def _session_cookies() -> Dict[str, str]:
    raw = (os.getenv('HUGGINGCHAT_COOKIES', '') or '').strip()
    if raw:
        try:
            parsed = json.loads(raw)
        except ValueError:
            parsed = None
        if isinstance(parsed, dict) and parsed:
            return {str(k): str(v) for k, v in parsed.items()
                    if k and v is not None}
    return load_jar('huggingchat') or env_cookies('HUGGINGCHAT')


def _cookie_header(cookies: Dict[str, str]) -> str:
    return '; '.join(f'{k}={v}' for k, v in cookies.items())


def _headers(extra: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    hdr = {
        'User-Agent': _USER_AGENT,
        'Accept': '*/*',
        'Referer': f'{HUGGINGCHAT_BASE_URL}/',
        'Origin': HUGGINGCHAT_BASE_URL,
    }
    if extra:
        for k, v in extra.items():
            if v:  # empty value = drop the header (anonymous GETs)
                hdr[k] = v
            else:
                hdr.pop(k, None)
    return hdr


def _classify(status: int, text: str,
              headers: Optional[Any] = None) -> ProviderError:
    if status in (401, 403):
        # 401 {"error":"You have to be logged in."} = missing/expired HF
        # session; 403 = bot wall (CloudFront challenge) — refresher work.
        return ProviderAuthError(
            f'huggingchat session rejected (HTTP {status}): '
            f'{(text or "")[:200]}')
    if status == 429:
        return ProviderRateLimitError(
            f'huggingchat rate limited (HTTP 429): {(text or "")[:200]}')
    return classify_http_error(status, text, headers)


def _multipart(field: str, value: str, boundary: str) -> bytes:
    """Minimal multipart/form-data body (the web client sends exactly one
    ``data`` field for text-only turns)."""
    return (
        f'--{boundary}\r\n'
        f'Content-Disposition: form-data; name="{field}"\r\n'
        f'\r\n'
        f'{value}\r\n'
        f'--{boundary}--\r\n'
    ).encode('utf-8')


def _parse_catalog(data: Any) -> List[Dict[str, Any]]:
    """Chat-usable models from the v2 catalog.

    ``isRouter`` marks the "Omni" meta-router (the web app's own smart
    router — this app ships one, so it is redundant) and ``unlisted``
    entries are legacy/pinned models the UI no longer offers.
    """
    models: List[Dict[str, Any]] = []
    if isinstance(data, dict):
        data = data.get('json')
    if not isinstance(data, list):
        return models
    for entry in data:
        if not isinstance(entry, dict) or entry.get('isRouter') \
                or entry.get('unlisted'):
            continue
        mid = str(entry.get('id') or '').strip()
        if not mid:
            continue
        models.append({
            'upstream': mid,
            'display': (str(entry.get('displayName') or mid).strip() or mid),
            'vision': bool(entry.get('multimodal')),
            'thinking': bool(entry.get('supportsReasoning')),
            'tools': bool(entry.get('supportsTools')),
        })
    return models


class HuggingChatProvider(Provider):
    """huggingface.co/chat behind the unified provider contract.

    Dormant until an HF session lands in the ``huggingchat`` jar (see the
    module docstring). The public catalog still loads so ops tooling can
    verify upstream reachability while the account is missing.
    """

    name = 'huggingchat'

    # cold site crawl — documented minutes-long first discovery; run
    # sequentially, not in the parallel HTTP wave (see Provider.
    # discovery_slow in base.py).
    discovery_slow = True

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
            resp = http_get(MODELS_URL, headers=_headers({
                # anonymous GET: the browser sends no Origin on navigations
                'Origin': '',
                'Referer': f'{HUGGINGCHAT_BASE_URL}/',
            }), timeout=30, no_proxy=no_proxy)
            if resp.status_code == 200:
                parsed = _parse_catalog(resp.json())
                if parsed:
                    with self._lock:
                        self._models = parsed
                        self._catalog_ts = now
                    return
            logger.warning('huggingchat catalog HTTP %s — keeping %s cached '
                           'models', resp.status_code, len(self._models))
        except Exception as e:  # noqa: BLE001 — best-effort refresh
            logger.warning('huggingchat catalog refresh failed: %s', e)

    def _find_model(self, key: str) -> Optional[Dict[str, Any]]:
        for m in self._models:
            if key in (m['upstream'], m['display']):
                return m
        lowered = key.lower()
        for m in self._models:
            if m['display'].lower() == lowered:
                return m
        return None

    def list_models(self, auth_key: Optional[str] = None) -> List[Dict[str, Any]]:
        if not self.available():
            raise ProviderAuthError(
                'no HuggingChat session configured (huggingchat_cookies.json '
                '/ HUGGINGCHAT_COOKIES): chat needs an HF login, the model '
                'catalog alone is public')
        self._refresh_catalog()
        out: List[Dict[str, Any]] = []
        for m in self._models:
            out.append({
                'id': m['upstream'],
                'upstream_model': m['upstream'],
                'thinking_enabled': m['thinking'],
                'search_enabled': False,
                'vision': m['vision'],
                'image_gen': False,
                'context_length': HUGGINGCHAT_CONTEXT_LENGTH,
                'max_output_tokens': HUGGINGCHAT_MAX_OUTPUT,
                'extra': {'display': m['display'], 'tools': m['tools']},
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
            raise ProviderAuthError('no HuggingChat session configured')
        if images:
            # File upload goes through /chat/api/v2/conversations/<id>/files
            # with presigned blobs — not implemented yet.
            raise ProviderError('huggingchat image input is not supported yet')
        self._refresh_catalog(no_proxy=no_proxy)
        entry = self._find_model(model)
        if not entry:
            raise ProviderError(f'huggingchat model not in catalog: {model}')

        # 1) create the conversation
        create_body = json.dumps({
            'model': entry['upstream'],
            'preprompt': '',
            'mlAssistant': False,
        }).encode('utf-8')
        resp = http_post_raw(
            CREATE_URL, create_body,
            headers=_headers({
                'Content-Type': 'application/json',
                'Accept': 'application/json',
                'Cookie': _cookie_header(cookies),
            }),
            timeout=60, no_proxy=no_proxy)
        if resp.status_code != 200:
            try:
                error_text = resp.text or ''
            except Exception:  # pragma: no cover
                error_text = f'HTTP {resp.status_code}'
            raise _classify(resp.status_code, error_text, resp.headers)
        try:
            conversation_id = str((resp.json() or {}).get('conversationId')
                                   or '').strip()
        except ValueError:
            conversation_id = ''
        if not conversation_id:
            raise ProviderUnavailableError(
                'huggingchat conversation creation returned no id')

        # 2) post the message (multipart "data" field) and parse the NDJSON
        message_id = str(uuid.uuid4())
        data_json = json.dumps({
            'inputs': prompt,
            'id': message_id,
            'is_retry': False,
            'is_continue': False,
            'timezone': 'UTC',
        })
        boundary = f'i4f-{uuid.uuid4().hex}'
        body = _multipart('data', data_json, boundary)
        resp = http_post_raw(
            f'{CREATE_URL}/{conversation_id}', body,
            headers=_headers({
                'Content-Type': f'multipart/form-data; boundary={boundary}',
                'Accept': '*/*',
                'Cookie': _cookie_header(cookies),
            }),
            timeout=300, no_proxy=no_proxy)
        if resp.status_code != 200:
            try:
                error_text = resp.text or ''
            except Exception:  # pragma: no cover
                error_text = f'HTTP {resp.status_code}'
            raise _classify(resp.status_code, error_text, resp.headers)
        return self._iter_chunks(resp.content or b'')

    def _iter_chunks(self, data: bytes) -> Generator[Dict[str, Any], None, None]:
        """Parse the NDJSON event stream (one JSON object per line)."""
        streamed = ''
        finished = False
        for raw in data.splitlines():
            if finished:
                break
            if isinstance(raw, bytes):
                line = raw.decode('utf-8', 'replace').strip()
            else:
                line = str(raw).strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except ValueError:
                continue  # partial keepalive noise is tolerated upstream too
            if not isinstance(event, dict):
                continue
            etype = str(event.get('type') or '')
            if etype == 'stream':
                token = event.get('token')
                if token:
                    streamed += str(token)
                    yield {'content': str(token), 'type': 'text',
                           'finish_reason': None}
            elif etype == 'reasoning':
                text = (event.get('token') or event.get('reasoning')
                        or event.get('text') or event.get('content') or '')
                if text:
                    yield {'content': str(text), 'type': 'thinking',
                           'finish_reason': None}
            elif etype == 'finalAnswer':
                # Canonical full answer: emit only the suffix the stream
                # deltas did not already cover (the client does the same
                # diff against its accumulated content).
                text = str(event.get('text') or '')
                if text and text.startswith(streamed):
                    missing = text[len(streamed):]
                    if missing:
                        yield {'content': missing, 'type': 'text',
                               'finish_reason': None}
                streamed = text or streamed
            elif etype == 'status':
                status = str(event.get('status') or '')
                if status == 'error':
                    message = (str(event.get('message') or 'unknown error')
                               [:300])
                    raise ProviderError(f'huggingchat stream error: {message}')
                if status == 'finished':
                    finished = True
                # started / keepAlive: nothing to surface
            # title / tool / file / plan / budget / turnState / …: ignored
        if streamed or finished:
            yield {'content': '', 'type': 'text', 'finish_reason': 'stop'}
        else:
            raise ProviderUnavailableError(
                'huggingchat stream produced no output')
