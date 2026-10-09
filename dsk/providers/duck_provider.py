"""Duck.ai provider — anonymous free access to DuckDuckGo AI Chat.

DuckDuckGo's AI chat (duck.ai) hands out anonymous, free, no-account access to
frontier models. This provider speaks the ``duckchat/v1`` protocol directly —
no official API, no paid keys, no browser.

How it works
------------
1. Warm (once per session TTL): GET ``/`` scrapes the live ``x-fe-version``
   token (``serp_...``) from the homepage HTML; ``/country.json``,
   ``/duckchat/v1/auth/token`` and the ``/?q=...&duckai=1`` chat page seed the
   browser-shaped cookie jar (``5``, ``ah``, ``dcs``, ``dcm``,
   ``isRecentChatOn`` + whatever the responses set).
2. VQD: GET ``/duckchat/v1/status`` with ``x-vqd-accept: 1`` answers the
   ``x-vqd-4`` session token and/or an ``x-vqd-hash-1`` anti-abuse challenge —
   a base64 bundle of obfuscated JS that must be executed against a
   browser-fidelity DOM and returns ``client_hashes`` + ``meta``. The solved
   bundle (base64 of the JSON result) is sent back as the ``x-vqd-hash-1``
   REQUEST header. Solving runs ``dsk/providers/duck_solver.js`` through node
   (stdlib-only ``vm`` sandbox, 5s timeout, no network access).
3. Chat: POST ``/duckchat/v1/chat`` (SSE) with the solved VQD headers,
   ``x-fe-signals`` (base64 interaction-event timeline) and ``x-fe-version``.
   ``reasoningEffort`` is REQUIRED on every request (omitting it →
   400 ERR_BAD_REQUEST). The response streams ``data: {...}`` lines whose
   ``message``/``content`` fields carry the answer deltas, and its
   ``x-vqd-hash-1`` response header carries the NEXT challenge (cached and
   solved lazily for the following request).
4. Failure taxonomy: HTTP 418 with ``ERR_BN_LIMIT`` = IP banned/rate-limited
   (no retry; proxy exit is demoted), any other 418 ``ERR_CHALLENGE`` /
   401 / 403 = challenge expired → one retry with a fresh VQD; 429 = rate
   limit (Retry-After honoured); 5xx = upstream unavailable.

Credentials: NONE (anonymous). The "renewal" surface is therefore the VQD
challenge (per-request), the ``x-fe-version`` token (per warm) and the egress
IP (429/ERR_BN_LIMIT rotate the pool exit — the refresher lists ``duck`` among
the egress-rotate providers).
"""

import codecs
import json
import logging
import os
import re
import shutil
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

from .base import (
    Provider,
    ProviderError,
    ProviderRateLimitError,
    ProviderUnavailableError,
    classify_http_error,
    http_get,
    http_post_stream,
)
from .jar import load_jar, save_jar

logger = logging.getLogger('dsk.providers.duck')

DUCK_BASE_URL = 'https://duck.ai'
DUCK_STATUS_URL = f'{DUCK_BASE_URL}/duckchat/v1/status'
DUCK_CHAT_URL = f'{DUCK_BASE_URL}/duckchat/v1/chat'
# Token-free (no VQD/challenge needed) live model catalog.
DUCK_MODELS_URL = f'{DUCK_BASE_URL}/duckchat/v1/models'
DUCK_COUNTRY_URL = f'{DUCK_BASE_URL}/country.json'
DUCK_AUTH_TOKEN_URL = f'{DUCK_BASE_URL}/duckchat/v1/auth/token'

# The real served x-fe-version token: serp_YYYYMMDD_HHMMSS_ET-<20..40 hex>.
# Bounded {20,40} keeps the pattern ReDoS-safe.
_FE_VERSION_RE = re.compile(r'serp_\d{8}_\d{6}_[A-Z]{2}-[0-9a-f]{20,40}')
_DEFAULT_FE_VERSION = ('serp_20260424_180649_ET-'
                       '0bdc33b2a02ebf8f235def65d887787f694720a1')

_USER_AGENT = (
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) '
    'Chrome/149.0.0.0 Safari/537.36'
)

# Browser-shaped baseline cookies; the warm requests refresh them.
_SEED_COOKIES: Dict[str, str] = {
    '5': '1', 'ah': 'wt-wt', 'dcs': '1', 'dcm': '3', 'isRecentChatOn': '1',
}

DUCK_CONTEXT_LENGTH = int(os.getenv('I4F_DUCK_CONTEXT_LENGTH', '16384'))
DUCK_MAX_OUTPUT = int(os.getenv('I4F_DUCK_MAX_OUTPUT', '4096'))
# Solver subprocess budget: the challenge vm runs with a 5s internal timeout.
_SOLVER_TIMEOUT = int(os.getenv('I4F_DUCK_SOLVER_TIMEOUT', '30'))

# Offline fallback catalog (captured live 2026-10-08) used only when the
# token-free /duckchat/v1/models endpoint is unreachable; list_models stays
# dynamic-first and self-heals against the live endpoint at runtime.
_FALLBACK_MODELS: List[Dict[str, Any]] = [
    {'id': 'gpt-6-luna', 'name': 'GPT-6 Luna', 'efforts': ['none', 'low']},
    {'id': 'gpt-5.4-mini', 'name': 'GPT-5.4 mini', 'efforts': ['none', 'low', 'medium']},
    {'id': 'claude-haiku-4-5', 'name': 'Claude Haiku 4.5', 'efforts': ['none', 'low', 'medium']},
    {'id': 'mistral-small-2603', 'name': 'Mistral Small 4', 'efforts': ['none']},
    {'id': 'tinfoil/gpt-oss-120b', 'name': 'gpt-oss 120B', 'efforts': ['none', 'low']},
    {'id': 'tinfoil/gemma4-31b', 'name': 'Gemma 4 31B', 'efforts': ['none']},
]
DEFAULT_DUCK_MODEL = 'gpt-5.4-mini'

# Retired/renamed ids → current wire ids (mirrors the live frontend aliases).
_MODEL_ALIASES: Dict[str, str] = {
    'gpt-4o-mini': 'gpt-5.4-mini',
    'gpt-5-mini': 'gpt-5.4-mini',
    'o3-mini': 'gpt-5.4-mini',
    'gpt-5.4-nano': 'gpt-5.4-mini',
    'llama-4-scout': 'gpt-5.4-mini',
    'claude-3-5-haiku-20241022': 'claude-haiku-4-5',
    'mistral-small-2501': 'mistral-small-2603',
    'gpt-oss-120b': 'tinfoil/gpt-oss-120b',
    'gemma4-31b': 'tinfoil/gemma4-31b',
}

# Models whose live A/B-verified request always carries reasoningEffort:'low'
# regardless of the advertised tiers.
_FORCED_LOW_EFFORT = {'claude-haiku-4-5', 'tinfoil/gpt-oss-120b'}

# Preference order for the automatic default model, matched (regex,
# case-insensitive) against the LIVE catalog: cheap/fast general ids first.
# Model ids drift over time — the default is always resolved against the
# current catalog, never a fixed id, so a retired default cannot strand
# unknown-model requests. The static DEFAULT_DUCK_MODEL is a last resort
# for when even the catalog endpoint is unreachable.
_DEFAULT_PREFERENCES = (r'gpt.*mini', r'gpt-oss', r'gpt', r'claude',
                        r'mistral', r'gemma', r'.')

_SOLVER_PATH = Path(__file__).with_name('duck_solver.js')

_WARM_TTL = float(os.getenv('I4F_DUCK_WARM_TTL', '1800'))     # 30 min
_MODELS_TTL = float(os.getenv('I4F_DUCK_MODELS_TTL', '600'))  # 10 min


def _effort_for(model: str, thinking_enabled: bool,
                live_efforts: Optional[List[str]] = None) -> str:
    """Pick the (REQUIRED) reasoningEffort value for a chat request."""
    if model in _FORCED_LOW_EFFORT:
        return 'low'
    if not thinking_enabled:
        return 'none'
    efforts = [e for e in (live_efforts or []) if e != 'none']
    if efforts:
        return efforts[0]  # lowest supported non-none effort
    return 'low'


class _SolverUnavailable(ProviderUnavailableError):
    """node is missing or the solver subprocess failed structurally."""


def _node_bin() -> str:
    raw = (os.getenv('I4F_DUCK_NODE', '') or '').strip()
    if raw:
        return raw
    node = shutil.which('node')
    if not node:
        raise _SolverUnavailable(
            'duck.ai requires the node runtime for the VQD challenge solver '
            '(duck_solver.js) — add nodejs to the image or set I4F_DUCK_NODE')
    return node


def _run_solver(req: Dict[str, Any], timeout: Optional[int] = None) -> Any:
    """Run one JSON command through the solver script; return its data field."""
    try:
        proc = subprocess.run(
            [_node_bin(), str(_SOLVER_PATH)],
            input=json.dumps(req).encode('utf-8'),
            capture_output=True, timeout=timeout or _SOLVER_TIMEOUT,
        )
    except FileNotFoundError as e:
        raise _SolverUnavailable(f'duck.ai solver runtime missing: {e}') from e
    except subprocess.TimeoutExpired as e:
        raise _SolverUnavailable('duck.ai challenge solver timed out') from e
    try:
        out = json.loads(proc.stdout or b'{}')
    except ValueError as e:
        raise _SolverUnavailable(
            f'duck.ai solver returned invalid output: '
            f'{proc.stderr.decode("utf-8", "ignore")[:120]}') from e
    if not out.get('ok'):
        raise _SolverUnavailable(
            f'duck.ai challenge solve failed: {out.get("error", "?")}')
    return out.get('data')


# HTML fragments quoted inside the challenge JS must be pre-parsed so the
# sandbox's innerHTML shim can serve a browser-normalized serialization and
# the descendant count (querySelectorAll('*').length fidelity).
_HTML_STRING_RE = re.compile(r"(['\"])(<[^'\"]{1,400}?)\1")


def _build_html_lookup(js: str) -> Dict[str, Dict[str, Any]]:
    """Parse every quoted HTML snippet in the challenge into {html, count}.

    ``count`` backs the shim's querySelectorAll('*').length — DESCENDANTS of
    the element whose innerHTML is the snippet. Uses lxml (HTML5-ish parser);
    challenges that carry no HTML literals simply yield an empty lookup.
    """
    lookup: Dict[str, Dict[str, Any]] = {}
    try:
        import lxml.html  # noqa: PLC0415 — optional heavy import at call time
    except ImportError:  # pragma: no cover — lxml is in requirements.txt
        return lookup
    seen = set()
    for match in _HTML_STRING_RE.finditer(js):
        html = match.group(2)
        if html in seen:
            continue
        seen.add(html)
        try:
            frag = lxml.html.fragment_fromstring(html, create_parent=True)
        except Exception:  # noqa: BLE001 — malformed snippet: skip it
            continue
        try:
            count = max(0, sum(1 for el in frag.iter()
                               if isinstance(getattr(el, 'tag', None), str)) - 1)
            serialized = lxml.html.tostring(frag, encoding='unicode')
            serialized = serialized[len('<div>'):-len('</div>')]
        except Exception:  # noqa: BLE001
            continue
        lookup[html] = {'html': serialized, 'count': count}
    return lookup


def _solve_challenge(challenge: str) -> str:
    """Solve an x-vqd-hash-1 challenge bundle; returns the request-header value."""
    try:
        js = __import__('base64').b64decode(challenge).decode('utf-8', 'ignore')
    except Exception as e:  # noqa: BLE001
        raise _SolverUnavailable(f'duck.ai challenge is not base64: {e}') from e
    lookup = _build_html_lookup(js)
    data = _run_solver({'cmd': 'solve', 'challenge': challenge,
                        'ua': _USER_AGENT, 'lookup': lookup})
    if not isinstance(data, str) or not data:
        raise _SolverUnavailable('duck.ai challenge solve returned no header value')
    return data


class DuckProvider(Provider):
    """Anonymous duck.ai (DuckDuckGo AI Chat) via the duckchat/v1 protocol."""

    name = 'duck'

    def __init__(self) -> None:
        # RLock: cookie bookkeeping nests inside _warm's locked section —
        # a plain Lock deadlocks there (found live: _warm hung forever on
        # the first Set-Cookie).
        self._lock = threading.RLock()
        self._warmed_at = 0.0
        self._fe_version = _DEFAULT_FE_VERSION
        self._cookies: Dict[str, str] = dict(_SEED_COOKIES)
        self._pending_challenge: Optional[str] = None
        self._durable_public_key: Optional[Dict[str, Any]] = None
        self._live_models: Tuple[float, Dict[str, Any]] = (0.0, {})

    # ------------------------------------------------------------------ auth
    def available(self, auth_key: Optional[str] = None) -> bool:
        """Anonymous provider: always available (the router health-probes it)."""
        return True

    # ---------------------------------------------------------------- models
    def _fetch_live_catalog(self, no_proxy: bool = False) -> Dict[str, Any]:
        """Token-free live model catalog (no VQD needed) with a TTL cache."""
        now = time.monotonic()
        ts, cached = self._live_models
        if cached and now - ts < _MODELS_TTL:
            return cached
        try:
            resp = http_get(DUCK_MODELS_URL, headers=self._headers(),
                            cookies=self._cookies, timeout=20, no_proxy=no_proxy)
            if resp.status_code == 200:
                data = resp.json()
                models = {}
                for m in (data or {}).get('models') or []:
                    if not isinstance(m, dict):
                        continue
                    tiers = m.get('accessTier') or []
                    if 'free' not in tiers:
                        continue  # plus/pro models only answer behind a sub
                    efforts = [e for e in (m.get('supportedReasoningEffort') or [])
                               if isinstance(e, str)]
                    models[str(m.get('id') or '')] = {
                        'name': str(m.get('name') or m.get('id') or ''),
                        'efforts': efforts or ['none'],
                    }
                models.pop('', None)
                if models:
                    self._live_models = (now, models)
                    return models
        except Exception as e:  # noqa: BLE001 — catalog is best-effort
            logger.debug('duck live model catalog unavailable: %s', e)
        return {m['id']: {'name': m['name'], 'efforts': list(m['efforts'])}
                for m in _FALLBACK_MODELS}

    def list_models(self, auth_key: Optional[str] = None) -> List[Dict[str, Any]]:
        catalog = self._fetch_live_catalog()
        models = []
        for model_id, meta in catalog.items():
            efforts = meta.get('efforts') or ['none']
            models.append({
                'id': model_id,
                'upstream_model': model_id,
                'thinking_enabled': any(e != 'none' for e in efforts),
                'search_enabled': False,
                'context_length': DUCK_CONTEXT_LENGTH,
                'max_output_tokens': DUCK_MAX_OUTPUT,
                'extra': {'efforts': efforts, 'title': meta.get('name') or model_id},
            })
        return models

    # ---------------------------------------------------------------- session
    def _headers(self, extra: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        headers = {
            'User-Agent': _USER_AGENT,
            'Accept-Language': 'en-US,en;q=0.9',
            'Origin': DUCK_BASE_URL,
            'Referer': f'{DUCK_BASE_URL}/',
            'Sec-Ch-Ua': '"Chromium";v="149", "Not-A.Brand";v="24", '
                         '"Google Chrome";v="149"',
            'Sec-Ch-Ua-Mobile': '?0',
            'Sec-Ch-Ua-Platform': '"Linux"',
            'Sec-Fetch-Dest': 'empty',
            'Sec-Fetch-Mode': 'cors',
            'Sec-Fetch-Site': 'same-origin',
            'Priority': 'u=1, i',
        }
        if extra:
            headers.update(extra)
        return headers

    def _absorb_cookies(self, resp) -> None:
        """Fold a response's Set-Cookie jar into the session cookies."""
        try:
            jar = getattr(resp, 'cookies', None) or {}
            items = jar.items() if hasattr(jar, 'items') else []
            for name, value in items:
                if name and value:
                    self._cookies_set(name, value)
        except Exception:  # noqa: BLE001 — cookie bookkeeping is best-effort
            pass

    def _cookies_set(self, name: str, value: str) -> None:
        with self._lock:
            self._cookies[name] = value

    def _warm(self, no_proxy: bool = False) -> None:
        """Browser-shaped warm-up: fe-version scrape + cookie seeding."""
        now = time.monotonic()
        if now - self._warmed_at < _WARM_TTL:
            return
        with self._lock:
            if now - self._warmed_at < _WARM_TTL:
                return
            for seed_name, seed_value in _SEED_COOKIES.items():
                self._cookies.setdefault(seed_name, seed_value)
            html_headers = self._headers({
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Sec-Fetch-Dest': 'document',
                'Sec-Fetch-Mode': 'navigate',
                'Sec-Fetch-Site': 'none',
                'Upgrade-Insecure-Requests': '1',
                'Cache-Control': 'no-cache',
                'Pragma': 'no-cache',
            })
            try:
                resp = http_get(f'{DUCK_BASE_URL}/', headers=html_headers,
                                cookies=self._cookies, timeout=30, no_proxy=no_proxy)
                self._absorb_cookies(resp)
                match = _FE_VERSION_RE.search(resp.text or '')
                if match:
                    self._fe_version = match.group(0)
            except Exception as e:  # noqa: BLE001 — warm-up is best-effort
                logger.debug('duck warm homepage failed: %s', e)
            for url in (DUCK_COUNTRY_URL, DUCK_AUTH_TOKEN_URL,
                        f'{DUCK_BASE_URL}/?q=DuckDuckGo+AI+Chat&ia=chat&duckai=1'):
                try:
                    resp = http_get(url, headers=html_headers,
                                    cookies=self._cookies, timeout=20,
                                    no_proxy=no_proxy)
                    self._absorb_cookies(resp)
                except Exception as e:  # noqa: BLE001
                    logger.debug('duck warm %s failed: %s', url, e)
            self._warmed_at = time.monotonic()
            logger.info('duck session warmed (fe-version %s, %d cookies)',
                        self._fe_version[:24], len(self._cookies))

    def _durable_key(self) -> Dict[str, Any]:
        """RSA-OAEP JWK for durableStream.publicKey (generated once, persisted)."""
        if self._durable_public_key:
            return self._durable_public_key
        jar = load_jar('duck') or {}
        key = jar.get('durable_public_key')
        if isinstance(key, dict) and key.get('n') and key.get('e'):
            self._durable_public_key = key
            return key
        key = _run_solver({'cmd': 'keygen'})
        if isinstance(key, dict) and key.get('n') and key.get('e'):
            self._durable_public_key = key
            try:
                save_jar('duck', {'durable_public_key': key})
            except Exception:  # noqa: BLE001 — persistence is best-effort
                pass
            return key
        raise _SolverUnavailable('duck.ai durable key generation failed')

    def _acquire_vqd(self, no_proxy: bool = False) -> Tuple[str, str]:
        """Return (vqd4, solved_hash) for one chat request.

        Reuses the challenge advertised on the previous chat response
        (``x-vqd-hash-1``) before spending a fresh /status call; never sends
        an unsolved challenge upstream (it burns 418 + IP rate-limit budget).
        """
        if self._pending_challenge:
            challenge, self._pending_challenge = self._pending_challenge, None
            try:
                return '', _solve_challenge(challenge)
            except ProviderError as e:
                logger.debug('duck pending challenge solve failed: %s', e)
        resp = http_get(DUCK_STATUS_URL, headers=self._headers({
            'Accept': '*/*', 'Cache-Control': 'no-store', 'x-vqd-accept': '1',
        }), cookies=self._cookies, timeout=30, no_proxy=no_proxy)
        self._absorb_cookies(resp)
        if resp.status_code == 429:
            retry = resp.headers.get('Retry-After')
            raise ProviderRateLimitError(
                'duck.ai status rate limited', float(retry) if retry else None)
        if resp.status_code != 200:
            raise ProviderUnavailableError(
                f'duck.ai status HTTP {resp.status_code}')
        vqd4 = resp.headers.get('x-vqd-4') or ''
        challenge = resp.headers.get('x-vqd-hash-1') or ''
        if not challenge and not vqd4:
            raise ProviderUnavailableError('duck.ai status issued no VQD token')
        if not challenge:
            return vqd4, ''
        return vqd4, _solve_challenge(challenge)

    # ----------------------------------------------------------------- stream
    def stream(self, prompt: str, *, model: str, thinking_enabled: bool = False,
               search_enabled: bool = False, temperature: Optional[float] = None,
               max_tokens: Optional[int] = None,
               images: Optional[List[Dict[str, Any]]] = None,
               image_generation: bool = False,
               no_proxy: bool = False,
               auth_key: Optional[str] = None) -> Generator[Dict[str, Any], None, None]:
        if images:
            raise ProviderError('duck.ai file/image attachments are not supported')
        self._warm(no_proxy=no_proxy)
        upstream_model = self._resolve_model(model)
        catalog = self._fetch_live_catalog(no_proxy=no_proxy)
        effort = _effort_for(upstream_model, thinking_enabled,
                             (catalog.get(upstream_model) or {}).get('efforts'))
        last_error: Optional[ProviderError] = None
        for _attempt in range(2):
            emitted = False
            try:
                vqd4, solved = self._acquire_vqd(no_proxy=no_proxy)
                response = self._post_chat(prompt, upstream_model, effort,
                                           vqd4, solved, no_proxy=no_proxy)
                if response.status_code == 200:
                    self._pending_challenge = (
                        response.headers.get('x-vqd-hash-1') or None)
                    self._absorb_cookies(response)
                    for piece in self._iter_chunks(response):
                        emitted = True
                        yield piece
                    return
                error_text = response.text or ''
                raise self._classify_chat_error(response.status_code, error_text,
                                                response.headers)
            except ProviderRateLimitError as e:
                last_error = e
                if emitted:
                    raise
                # IP-level refusal: demote the pooled exit so the next attempt
                # (this provider's retry or the router's fallback) uses a
                # fresh egress.
                self._demote_exit()
                continue
            except ProviderUnavailableError as e:
                last_error = e
                if emitted:
                    raise
                continue  # one fresh-VQD retry, then give up
            except GeneratorExit:
                raise
            except ProviderError as e:
                last_error = e
                raise
        raise last_error or ProviderError('duck.ai stream failed')

    def _post_chat(self, prompt: str, model: str, effort: str,
                   vqd4: str, solved: str, no_proxy: bool = False):
        payload = self._build_payload(prompt, model, effort)
        headers = self._headers({
            'Accept': 'text/event-stream',
            'Content-Type': 'application/json',
            'x-ddg-journey-id': uuid.uuid4().hex,
            'x-fe-signals': _run_solver({'cmd': 'signals'}),
            'x-fe-version': self._fe_version,
        })
        if vqd4:
            headers['x-vqd-4'] = vqd4
        if solved:
            headers['x-vqd-hash-1'] = solved
        return http_post_stream(DUCK_CHAT_URL, headers=headers, json_body=payload,
                                cookies=self._cookies, timeout=300,
                                no_proxy=no_proxy)

    def _build_payload(self, prompt: str, model: str, effort: str) -> Dict[str, Any]:
        return {
            'model': model,
            'metadata': {'toolChoice': {'NewsSearch': False, 'VideosSearch': False,
                                        'LocalSearch': False,
                                        'WeatherForecast': False}},
            'messages': [{'role': 'user', 'content': prompt}],
            'canUseTools': True,
            # reasoningEffort is REQUIRED — omitting it answers 400.
            'reasoningEffort': effort,
            'canUseApproxLocation': None,
            'canDelegateImageGeneration': None,
            'durableStream': {
                'messageId': str(uuid.uuid4()),
                'conversationId': str(uuid.uuid4()),
                'publicKey': self._durable_key(),
            },
        }

    def _default_model(self) -> str:
        """Default model picked from the LIVE catalog (self-healing).

        Preference-ordered cheap/fast ids first; ``DEFAULT_DUCK_MODEL`` only
        applies when the catalog endpoint is unreachable AND its static
        fallback is empty (never in practice).
        """
        live = self._fetch_live_catalog()
        for pattern in _DEFAULT_PREFERENCES:
            for mid in live:
                if re.search(pattern, mid, re.IGNORECASE):
                    return mid
        return DEFAULT_DUCK_MODEL

    def _resolve_model(self, model: Optional[str]) -> str:
        """Map public/route ids onto a live duckchat wire id (self-healing)."""
        clean = (model or '').strip()
        if clean.startswith('duck/'):
            clean = clean[len('duck/'):]
        clean = _MODEL_ALIASES.get(clean, clean)
        live = self._fetch_live_catalog()
        if live:
            if clean in live:
                return clean
            default = self._default_model()
            if clean:
                logger.warning('duck: model %r absent from the live duckchat '
                               'catalog — routing as %r', clean, default)
            return default
        return clean or DEFAULT_DUCK_MODEL

    @staticmethod
    def _classify_chat_error(status: int, text: str,
                             resp_headers=None) -> ProviderError:
        body = text or ''
        err_type = ''
        try:
            data = json.loads(body)
            if isinstance(data, dict):
                err_type = str(data.get('type') or '')
        except ValueError:
            pass
        if status == 418:
            if err_type == 'ERR_BN_LIMIT':
                # IP/session banned — retrying with a fresh VQD cannot help.
                return ProviderRateLimitError(
                    f'duck.ai rate limit / ban (ERR_BN_LIMIT): {body[:120]}')
            return ProviderUnavailableError(
                f'duck.ai challenge rejected (ERR_CHALLENGE): {body[:120]}')
        if status == 429:
            retry = None
            try:
                retry = float(resp_headers.get('Retry-After')) if resp_headers else None
            except (TypeError, ValueError):
                retry = None
            return ProviderRateLimitError(
                f'duck.ai rate limited: {body[:120]}', retry)
        if status == 400:
            # Schema drift (ERR_BAD_REQUEST) needs a code fix, not retries.
            return ProviderError(f'duck.ai bad request: {body[:200]}')
        if status in (401, 403):
            return ProviderUnavailableError(
                f'duck.ai refused the session ({status}): {body[:120]}')
        return classify_http_error(status, body, resp_headers)

    @staticmethod
    def _demote_exit() -> None:
        try:
            from dsk import proxies as _proxies
            _proxies.mark_failure(None)
        except Exception:  # noqa: BLE001 — pool demotion is best-effort
            pass

    def _iter_chunks(self, response) -> Generator[Dict[str, Any], None, None]:
        """Parse the duckchat SSE ``data: {...}`` line format.

        Each event carries the delta in ``message`` or ``content``; reasoning
        summaries arrive as ``role: "reasoning"`` events with ``summaryText``;
        ``[DONE]`` ends the stream. The response's own ``x-vqd-hash-1`` header
        (next challenge) was captured by the caller before iteration.
        """
        decoder = codecs.getincrementaldecoder('utf-8')('replace')
        buffer = ''
        last_reasoning = ''

        def _emit(piece: Optional[Dict[str, Any]]) -> Generator[Dict[str, Any], None, None]:
            nonlocal last_reasoning
            if not piece:
                return
            if piece.get('type') == 'thinking' and piece.get('cumulative'):
                # summaryText is the full summary so far, not a delta: forward
                # only the new suffix (a reset — non-prefix text — is forwarded
                # whole and becomes the new baseline).
                full = piece['content']
                if full.startswith(last_reasoning):
                    delta = full[len(last_reasoning):]
                else:
                    delta = full
                last_reasoning = full
                if delta:
                    yield {'content': delta, 'type': 'thinking', 'finish_reason': None}
                return
            yield piece

        for chunk in response.iter_content(chunk_size=None):
            buffer += decoder.decode(chunk or b'')
            while '\n' in buffer:
                line, buffer = buffer.split('\n', 1)
                yield from _emit(self._parse_line(line))
        buffer += decoder.decode(b'', final=True)
        for line in buffer.split('\n'):
            yield from _emit(self._parse_line(line))
        yield {'content': '', 'type': 'text', 'finish_reason': 'stop'}

    @staticmethod
    def _parse_line(line: Optional[str]) -> Optional[Dict[str, Any]]:
        if not line:
            return None
        text = line.strip()
        if not text or text == '[DONE]':
            return None
        if not text.startswith('data: '):
            return None
        try:
            data = json.loads(text[6:])
        except ValueError:
            return None
        if not isinstance(data, dict):
            return None
        if data.get('role') == 'reasoning':
            # Reasoning-summary events: ``summaryText`` carries the summary so
            # far — a string or a list of summary segments (usually empty: the
            # raw CoT arrives encrypted in ``encryptedText``). Emitted as the
            # standard thinking piece; ``_iter_chunks`` strips the
            # already-forwarded prefix, so a cumulative summary never repeats.
            reasoning = data.get('summaryText')
            if isinstance(reasoning, list):
                reasoning = ''.join(s for s in reasoning if isinstance(s, str))
            if isinstance(reasoning, str) and reasoning:
                return {'content': reasoning, 'type': 'thinking',
                        'finish_reason': None, 'cumulative': True}
            return None
        content = data.get('content') or data.get('message') or ''
        if isinstance(content, str) and content:
            return {'content': content, 'type': 'text', 'finish_reason': None}
        return None
