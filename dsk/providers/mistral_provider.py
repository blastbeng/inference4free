"""Mistral provider via chat.mistral.ai (Le Chat) — reverse-engineered.

Ported from the mistral-proxy APK reverse-engineering: the mobile app talks to
the web app's private tRPC + chat endpoints with either an anonymous
``stableAnonymousIdentifier`` (5 msgs/day, rotated to reset the quota) or an
Ory Kratos session token — no official API, no paid keys.

How it works
------------
1. Bootstrap: GET ``https://auth.mistral.ai/self-service/registration/api``
   warms the Kratos/Cloudflare cookies (non-fatal).
2. Single call: POST ``/api/chat`` with ``mode: 'create'`` +
   ``productType: 'work'`` — the request both starts the conversation and
   streams the answer (no separate newChat call; ``agentId`` must be absent —
   ``null`` → HTTP 400). 2026-10: chat.mistral.ai merged Le Chat into the
   "Work" UI and the endpoint now validates that schema — the old body
   (top-level ``model`` + ``platform``) answers HTTP 400 with an empty body,
   and the model is selected by ``modelConfig: {model_alias, reasoning_effort}``.
   The body is NOT SSE —
   it is newline-delimited ``<type_num>:<json>`` frames (15=data patches,
   16=metadata, 6=error, 8=end). Assistant text arrives as JSON-patch ops on
   ``/contentChunks`` (replace = full snapshot, append = delta, including
   ``/contentChunks/N/text`` string deltas).
3. An Ory Kratos session token (``MISTRAL_SESSION_TOKEN``) is required:
   anonymous access now returns an account upsell instead of model output.
4. Quota (error code 6200): rotate the stable UUID and retry once.
"""

import codecs
import json
import logging
import os
import re
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Generator, List, Optional

from .base import (
    Provider,
    ProviderAuthError,
    ProviderError,
    ProviderRateLimitError,
    classify_http_error,
    http_get,
    http_post_stream,
)
from .jar import env_cookies, load_jar, save_jar

logger = logging.getLogger('dsk.providers.mistral')

# Mistral's anonymous tier no longer produces model output: the endpoint
# answers HTTP 200 with a chat message that is actually the account upsell
# ("## An account is now required to use Vibe  *Sign in...").  Detect it at
# parse time and surface it as an auth failure so the router falls back and
# selfheal classifies the provider as needing credentials.
_AUTH_WALL_RE = re.compile(r'account is now required', re.IGNORECASE)

MISTRAL_BASE_URL = 'https://chat.mistral.ai'
MISTRAL_AUTH_URL = 'https://auth.mistral.ai'
MISTRAL_CHAT_URL = f'{MISTRAL_BASE_URL}/api/chat'
MISTRAL_CHAT_MODE = os.getenv('MISTRAL_CHAT_MODE', 'create')
MISTRAL_PRODUCT_TYPE = os.getenv('MISTRAL_PRODUCT_TYPE', 'work')

# Model aliases harvested from the app's own JS bundles (schema:
# ``modelConfig = {model_alias, temperature?: 0..1, reasoning_effort?}``),
# the first verified live against /api/chat. ``efforts`` lists the
# ``reasoning_effort`` values the alias accepts; an alias without them must
# be sent WITHOUT the key.
MISTRAL_MODELS: Dict[str, Dict[str, Any]] = {
    'glm-5-latest-short': {'title': 'GLM-5 latest (Le Chat)',
                           'efforts': ['none', 'high']},
    'mistral-large-2411': {'title': 'Mistral Large 2411', 'efforts': []},
    'mistral-small-2603': {'title': 'Mistral Small 2603', 'efforts': []},
    'mistral-small-latest': {'title': 'Mistral Small latest', 'efforts': []},
    'ministral-8b-latest': {'title': 'Ministral 8B', 'efforts': []},
    'open-mistral-nemo': {'title': 'Open Mistral Nemo', 'efforts': []},
}
# Public ids kept alive from the pre-Work schema.
MISTRAL_LEGACY_IDS = {'mistral-large': 'mistral-large-2411',
                      'mistral-small': 'mistral-small-latest'}
# Default alias for the provider's ``auto`` route and for any unknown id.
MISTRAL_MODEL = os.getenv('MISTRAL_DEFAULT_MODEL', 'glm-5-latest-short')
# Feature flags the live app sends on every Work chat request.
MISTRAL_FEATURES = [f.strip() for f in os.getenv(
    'MISTRAL_FEATURES',
    'beta-code-interpreter,beta-imagegen,beta-trampoline,beta-websearch,'
    'agentic-harness').split(',') if f.strip()]
MISTRAL_TASK_CALLBACKS = ['ask_user_question', 'ask_user_confirmation',
                          'ask_enable_skill', 'enable_connector',
                          'ask_retry_or_continue_rate_limit',
                          'collect_workflow_input',
                          'delegate_workflow_execution']

# 2026-09: the Android app UA (le-chat-mobile/2.8.0) makes the server replace
# model output with an "This mode is no longer available — update your app"
# patch; a desktop-browser UA with platform:'web' streams normally.
MISTRAL_APP_UA = os.getenv('MISTRAL_UA', (
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
    '(KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36'
))

MISTRAL_CONTEXT_LENGTH = int(os.getenv('I4F_MISTRAL_CONTEXT_LENGTH', '131072'))
MISTRAL_MAX_OUTPUT = int(os.getenv('I4F_MISTRAL_MAX_OUTPUT', '8192'))

OTP_CODE_RE = re.compile(r'6200')


def _session_token() -> str:
    raw = (os.getenv('MISTRAL_SESSION_TOKEN', '') or '').strip()
    if raw:
        return raw
    jar = load_jar('mistral') or env_cookies('MISTRAL')
    return (jar.get('session_token') or '').strip()


def _anon_id() -> str:
    jar = load_jar('mistral') or env_cookies('MISTRAL')
    stable = (jar.get('stable_anon_id') or '').strip()
    if not stable:
        stable = str(uuid.uuid4())
        save_jar('mistral', {'stable_anon_id': stable})
    return stable


def _model_table() -> Dict[str, Dict[str, Any]]:
    """Env-overridable alias table: ``MISTRAL_MODELS=a,b,c`` keeps the known
    metadata for aliases it recognises and appends the rest verbatim."""
    raw = (os.getenv('MISTRAL_MODELS', '') or '').strip()
    if not raw:
        return dict(MISTRAL_MODELS)
    table: Dict[str, Dict[str, Any]] = {}
    for alias in [a.strip() for a in raw.split(',') if a.strip()]:
        table[alias] = dict(MISTRAL_MODELS.get(alias,
                                               {'title': alias, 'efforts': []}))
    return table


def _alias_for(model: Optional[str]) -> str:
    """Map a public/route id onto an upstream ``modelConfig.model_alias``."""
    table = _model_table()
    key = (model or '').strip().lower()
    if key.startswith('mistral/'):
        key = key.split('/', 1)[1]
    if key in table:
        return key
    legacy = MISTRAL_LEGACY_IDS.get(key)
    if legacy:
        return legacy
    for alias in table:
        if key and (key in alias or alias in key):
            return alias
    return MISTRAL_MODEL if MISTRAL_MODEL in table else next(iter(table))


def _effort_for(alias: str, thinking_enabled: bool) -> Optional[str]:
    """``reasoning_effort`` is optional in the schema and only some aliases
    accept it — returning None omits the key entirely."""
    efforts = _model_table().get(alias, {}).get('efforts') or []
    if not efforts:
        return None
    if thinking_enabled and 'high' in efforts:
        return 'high'
    return efforts[0]


def _tz_label(now: datetime) -> str:
    """``clientPromptData.userTimezone`` as the app sends it: 'T+00:00 (UTC)'."""
    local = now.astimezone()
    total = int((local.utcoffset() or timedelta(0)).total_seconds())
    sign = '+' if total >= 0 else '-'
    hours, minutes = divmod(abs(total) // 60, 60)
    return f'T{sign}{hours:02d}:{minutes:02d} ({local.tzname()})'


def _headers(auth: bool = False, accept: str = 'application/json') -> Dict[str, str]:
    headers = {
        'User-Agent': MISTRAL_APP_UA,
        'Accept': accept,
        'Content-Type': 'application/json',
        'Origin': MISTRAL_BASE_URL,
        'Referer': f'{MISTRAL_BASE_URL}/',
    }
    token = _session_token()
    if token:
        headers['Authorization'] = f'Bearer {token}'
        # chat.mistral.ai classifies Bearer-only requests as anonymous quota;
        # the Ory session cookie must ride along (council finding).
        jar = load_jar('mistral') or env_cookies('MISTRAL') or {}
        name = (jar.get('session_cookie_name') or '').strip() or 'ory_kratos_session'
        headers['Cookie'] = f'{name}={token}'
    return headers


class MistralProvider(Provider):
    name = 'mistral'

    def __init__(self) -> None:
        self._bootstrapped = False

    # ------------------------------------------------------------- bootstrap
    def _bootstrap(self, no_proxy: bool = False) -> None:
        """Warm Kratos cookies once per process (non-fatal on failure)."""
        if self._bootstrapped:
            return
        try:
            http_get(f'{MISTRAL_AUTH_URL}/self-service/registration/api',
                     headers={'Accept': 'application/json',
                              'User-Agent': MISTRAL_APP_UA},
                     no_proxy=no_proxy)
        except Exception as e:  # noqa: BLE001 - warm-up must never fail requests
            logger.debug('mistral bootstrap failed (non-fatal): %s', e)
        self._bootstrapped = True

    # ---------------------------------------------------------------- provider
    def available(self, auth_key: Optional[str] = None) -> bool:
        # Anonymous access is account-gated upstream (returns an upsell
        # instead of model output) — a Le Chat session token is required.
        return bool(_session_token())

    def list_models(self, auth_key: Optional[str] = None) -> List[Dict[str, Any]]:
        """The Work schema has no model-discovery endpoint; the alias table
        (harvested from the app's bundles, env-overridable) is the list."""
        entries: List[Dict[str, Any]] = []
        for alias, meta in _model_table().items():
            efforts = meta.get('efforts') or []
            entries.append({
                'id': alias,
                'upstream_model': alias,
                'thinking_enabled': bool(efforts),
                'search_enabled': 'beta-websearch' in MISTRAL_FEATURES,
                'vision': False,
                'image_gen': 'beta-imagegen' in MISTRAL_FEATURES,
                'context_length': MISTRAL_CONTEXT_LENGTH,
                'max_output_tokens': MISTRAL_MAX_OUTPUT,
                'extra': {'title': meta.get('title') or alias},
            })
        for legacy_id, alias in MISTRAL_LEGACY_IDS.items():
            if legacy_id in _model_table():
                continue
            entries.append({
                'id': legacy_id,
                'upstream_model': alias,
                'thinking_enabled': False,
                'search_enabled': 'beta-websearch' in MISTRAL_FEATURES,
                'vision': False,
                'image_gen': False,
                'context_length': MISTRAL_CONTEXT_LENGTH,
                'max_output_tokens': MISTRAL_MAX_OUTPUT,
                'extra': {'title': f'{legacy_id} (legacy id)'},
            })
        return entries

    def stream(self, prompt: str, *, model: str, thinking_enabled: bool = False,
               search_enabled: bool = False, temperature: Optional[float] = None,
               max_tokens: Optional[int] = None,
               images: Optional[List[Dict[str, Any]]] = None,
               image_generation: bool = False,
               no_proxy: bool = False,
               auth_key: Optional[str] = None) -> Generator[Dict[str, Any], None, None]:
        if images:
            raise ProviderError('mistral file attachments are not supported yet')
        alias = _alias_for(model)
        effort = _effort_for(alias, thinking_enabled)
        self._bootstrap(no_proxy=no_proxy)
        anonymous = not _session_token()
        retry_id: Optional[str] = None
        last_error: Optional[ProviderError] = None
        max_attempts = 2 if anonymous else 1
        attempt = 0
        # NOTE: _stream_once is a GENERATOR — the account wall and the quota
        # error raise lazily on the consumer's first next(), i.e. OUTSIDE any
        # "try: return self._stream_once(...)" wrapper. The retries below only
        # apply because the generator is drained INSIDE the try.
        while attempt < max_attempts:
            attempt += 1
            emitted = False
            try:
                for piece in self._stream_once(prompt, no_proxy=no_proxy,
                                               anon_id=retry_id,
                                               alias=alias, effort=effort,
                                               search_enabled=search_enabled):
                    emitted = True
                    yield piece
                return
            except ProviderRateLimitError as e:
                last_error = e
                if emitted:
                    raise  # mid-stream: retrying would duplicate output
                if attempt < max_attempts:
                    # Fresh anonymous identity resets the 5-msg/day quota.
                    # Generate the UUID HERE and pass it down: re-reading
                    # _anon_id() from the jar after the write would race with
                    # the refresher daemon's jar updates.
                    retry_id = str(uuid.uuid4())
                    save_jar('mistral', {'stable_anon_id': retry_id})
                    continue
                raise
            except ProviderAuthError as e:
                if emitted:
                    raise  # mid-stream failure: surfaced as-is
                # The account wall ALSO trips through pooled datacenter
                # egresses even with a valid session — retry once direct
                # before declaring the credential dead.
                if not no_proxy and 'account upsell' in str(e):
                    no_proxy = True
                    max_attempts = attempt + 1  # grant exactly one direct retry
                    continue
                raise
        raise last_error or ProviderError('mistral stream failed')

    def _stream_once(self, prompt: str,
                     no_proxy: bool = False,
                     anon_id: Optional[str] = None,
                     alias: Optional[str] = None,
                     effort: Optional[str] = None,
                     search_enabled: bool = False
                     ) -> Generator[Dict[str, Any], None, None]:
        model_alias = alias or _alias_for(None)
        now = datetime.now(timezone.utc)
        features = list(MISTRAL_FEATURES)
        if search_enabled and 'beta-websearch' not in features:
            features.append('beta-websearch')
        if effort is None:
            effort = _effort_for(model_alias, False)
        model_config: Dict[str, Any] = {'model_alias': model_alias}
        if effort:
            model_config['reasoning_effort'] = effort
        body = {
            'content': [{'type': 'text', 'text': prompt}],
            'transcriptionsMetadata': [],
            'incognito': False,
            'files': [],
            'features': features,
            'integrations': [],
            'libraries': [],
            'modelConfig': model_config,
            'reviewComments': [],
            # 'create' starts the conversation and streams the answer in one
            # call; no 'agentId' key (a null value → HTTP 400).
            'mode': MISTRAL_CHAT_MODE,
            # 2026-10: chat.mistral.ai merged Le Chat into the Work UI and
            # /api/chat now validates that schema — the legacy body (top-level
            # 'model' + 'platform') is rejected with HTTP 400, empty body.
            'productType': MISTRAL_PRODUCT_TYPE,
            'clientPromptData': {'currentDate': now.strftime('%Y-%m-%d'),
                                 'userTimezone': _tz_label(now)},
            'disabledFeatures': [],
            'stableAnonymousIdentifier': anon_id or _anon_id(),
            'supportedTaskCallbacks': MISTRAL_TASK_CALLBACKS,
        }
        response = http_post_stream(MISTRAL_CHAT_URL,
                                    headers=_headers(accept='text/event-stream'),
                                    json_body=body, no_proxy=no_proxy)
        if response.status_code not in (200, 201):
            try:
                error_text = response.text or ''
            except Exception:  # pragma: no cover
                error_text = f'HTTP {response.status_code}'
            if response.status_code == 429:
                raise ProviderRateLimitError(f'mistral rate limited: {error_text[:200]}')
            raise classify_http_error(response.status_code, error_text,
                                      response.headers)
        return self._iter_chunks(response)

    def _iter_chunks(self, response) -> Generator[Dict[str, Any], None, None]:
        """Parse the ``<type_num>:<json>`` line format with an incremental
        utf-8 decoder (multi-byte characters straddle chunk boundaries).

        The account upsell can arrive split across many tiny append deltas
        ("## ", "An ", "account ", …), so text is HELD until the first 300
        chars have been matched against the login-wall regex — holding (not
        just checking) is required, otherwise the deltas emitted before the
        phrase completes leak to the router and pollute the fallback answer.
        """
        decoder = codecs.getincrementaldecoder('utf-8')('replace')
        buffer = ''
        prefix = ''                       # accumulated text head
        held: List[Dict[str, Any]] = []   # pieces buffered during the window
        for chunk in response.iter_content(chunk_size=None):
            buffer += decoder.decode(chunk or b'')
            while '\n' in buffer:
                line, buffer = buffer.split('\n', 1)
                for piece in self._parse_line(line.strip()):
                    if len(prefix) < 300:
                        head = piece.get('content') \
                            if piece.get('type') == 'text' else None
                        if head:
                            prefix += head
                            if _AUTH_WALL_RE.search(prefix):
                                raise ProviderAuthError(
                                    'mistral anonymous access disabled '
                                    f'(account upsell): {prefix[:120]}')
                            held.append(piece)
                            continue
                    while held:               # window closed: drain in order
                        yield held.pop(0)
                    yield piece
        buffer += decoder.decode(b'', final=True)
        if buffer.strip():
            for piece in self._parse_line(buffer.strip()):
                while held:
                    yield held.pop(0)
                yield piece
        while held:
            yield held.pop(0)
        yield {'content': '', 'type': 'text', 'finish_reason': 'stop'}

    def _parse_line(self, line: str) -> List[Dict[str, Any]]:
        if not line or ':' not in line:
            return []
        colon = line.index(':')
        try:
            line_type = int(line[:colon])
        except ValueError:
            return []
        json_str = line[colon + 1:]
        if not json_str or json_str == 'null':
            return []
        try:
            data = json.loads(json_str)
        except ValueError:
            return []
        j = data.get('json', data) if isinstance(data, dict) else data
        if not isinstance(j, dict):
            return []

        if line_type == 6:  # error frame
            code = j.get('internalCode', 0)
            retry = j.get('retryAfterSeconds', 0)
            if code == 6200 or retry:
                raise ProviderRateLimitError(
                    f"mistral anonymous quota reached: {j.get('message', '')}",
                    retry_after=float(retry) if retry else None)
            raise ProviderError(f"mistral stream error {code}: "
                                f"{j.get('message', str(j))[:300]}")
        if line_type != 15:  # only data frames carry text
            return []

        msg_type = j.get('type')
        if msg_type != 'message':
            return []
        out: List[Dict[str, Any]] = []
        for patch in j.get('patches', []):
            op = patch.get('op')
            path = patch.get('path', '')
            value = patch.get('value')
            if path == '/' or '/contentChunks' not in path:
                continue
            if op == 'replace' and isinstance(value, list):
                # Full snapshot of the chunk list — emit only the tail (the
                # web client replaces, we diff to avoid re-emitting text).
                texts = [c.get('text', '') for c in value
                         if isinstance(c, dict) and c.get('type') == 'text']
                if texts:
                    if _AUTH_WALL_RE.search(texts[-1][:300]):
                        raise ProviderAuthError(
                            'mistral anonymous access disabled (account '
                            f'upsell): {texts[-1][:120]}')
                    out.append({'content': texts[-1], 'type': 'text',
                                'finish_reason': None})
            elif op == 'append' and isinstance(value, str) and value:
                # Delta append — e.g. path ``/contentChunks/0/text``.
                if _AUTH_WALL_RE.search(value[:300]):
                    raise ProviderAuthError(
                        'mistral anonymous access disabled (account '
                        f'upsell): {value[:120]}')
                out.append({'content': value, 'type': 'text',
                            'finish_reason': None})
        return out
