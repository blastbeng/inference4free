"""Credential renewal bot: keeps the three web sessions alive autonomously.

Nothing here ever asks a human to re-copy a cookie — the ladder of renewal
strategies is, from cheapest to most invasive:

  1. HTTP cookie refresh (safe, automatic)
       gemini    load the cookie jar, GET gemini.google.com/app, capture the
                 rotated ``__Secure-1PSIDTS`` from the Set-Cookie header and
                 write it back to ``data/gemini_cookies.json``
       chatgpt   GET chatgpt.com/api/auth/session with the jar, persist the
                 fresh ``accessToken``-issuing session cookies
       deepseek  nothing to rotate over HTTP (the userToken only changes on
                 login) — verified with a live probe instead

  2. Headless-browser re-login (default ON: I4F_REFRESHER_LOGIN=true, set
     false to disable). Uses the same DrissionPage/Chromium stack as
     the Cloudflare bypass; exports the fresh cookies/token automatically.
       DEEPSEEK_LOGIN_EMAIL / DEEPSEEK_LOGIN_PASSWORD
       CHATGPT_LOGIN_EMAIL  / CHATGPT_LOGIN_PASSWORD
       GEMINI_LOGIN_EMAIL   / GEMINI_LOGIN_PASSWORD   (Google anti-bot: best
                                                        effort only)
     Email verification codes (OTP) during login are fetched from an IMAP
     mailbox (see I4F_MAIL_* below), so the loop stays unmanned.

  3. Account auto-signup (default ON for ALL providers:
     I4F_REFRESHER_AUTOSIGNUP). Creates a fresh free account when even the
     login session is dead — and BOOTSTRAPS providers that have no
     credentials at all (the refresher daemon signs every missing provider
     up on its first cycle, so a fresh install comes up unattended). The
     e-mail address is AUTO-GENERATED (dsk/mailgen.py): a catch-all IMAP
     domain (I4F_MAIL_DOMAIN) when available, else a mail.tm throwaway
     account — the verification code is read from that mailbox
     automatically. Disable the auto-generation with I4F_MAIL_AUTOGEN=false.
     Created accounts are persisted to data/accounts.json so later renewal
     cycles can re-login with them. Google/OpenAI may still throw captcha
     or phone walls at automation — those rungs are best effort and their
     failures surface in history.jsonl like any other ladder miss.

Renewals are triggered two ways: the self-healing daemon calls ``renew``
whenever a provider probe classifies as ``auth``, and the refresher daemon
proactively rotates cookies every I4F_REFRESHER_TTL seconds. Every action is
logged to data/refresher/history.jsonl; all ladders respect per-provider
cooldowns and daily attempt caps, and I4F_REFRESHER=false disables everything.

Bot-managed credential files take precedence over env vars (documented in
README): data/deepseek_token, data/gemini_cookies.json, data/chatgpt_cookies.json.
Delete the file to hand control back to the environment.

Mail config (for OTP during browser flows):
    I4F_MAIL_AUTOGEN       auto-create throwaway mailboxes (default true)
    I4F_MAIL_DOMAIN        catch-all domain for autogen (optional; without
                           it mail.tm public temp-mail is used)
    I4F_MAIL_IMAP_HOST / _PORT (993) / _USER / _PASS
    I4F_MAIL_OTP_SENDER    substring matched against the sender (default deepseek)
    I4F_MAIL_OTP_REGEX     code regex (default \\\\b(\\\\d{6})\\\\b)
    I4F_MAIL_OTP_MAX_AGE   ignore older mail, minutes (default 30)

CLI:
    python -m dsk.refresher status
    python -m dsk.refresher refresh gemini
    python -m dsk.refresher login deepseek
    python -m dsk.refresher signup [deepseek|chatgpt|gemini]
    python -m dsk.refresher bootstrap   (create credentials for every provider that has none)
    python -m dsk.refresher mailgen   (create a throwaway mailbox as a test)
"""

import base64
import hashlib
import imaplib
import json
import logging
import os
import random
import re
import signal
import threading
import time
from contextlib import contextmanager
from datetime import date, datetime, timezone
from email import message_from_bytes
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from . import mailgen
from .providers.base import provider_enabled

logger = logging.getLogger('inference4free.refresher')

_BASE = Path(__file__).resolve().parent


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name, '').strip().lower()
    if not raw:
        return default
    return raw in ('1', 'true', 'yes', 'on')


def _data_dir() -> Path:
    base = (os.getenv('COOKIES_DIR') or os.getenv('I4F_SELFHEAL_DIR')
            or str(_BASE.parent / 'data'))
    path = Path(base)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _ttl() -> float:
    return max(300.0, float(os.getenv('I4F_REFRESHER_TTL', '21600') or 21600))


def _cooldown() -> float:
    return max(60.0, float(os.getenv('I4F_REFRESHER_COOLDOWN', '1800') or 1800))


def _max_renews() -> int:
    return max(1, int(os.getenv('I4F_REFRESHER_MAX_RENEWS', '6') or 6))


def _jar_path(name: str) -> Path:
    files = {'gemini': 'gemini_cookies.json', 'chatgpt': 'chatgpt_cookies.json',
             'deepseek': 'cookies.json', 'claude': 'claude_cookies.json',
             'grok': 'grok_cookies.json', 'mistral': 'mistral_cookies.json',
             'qwen': 'qwen_cookies.json', 'kimi': 'kimi_cookies.json',
             'copilot': 'copilot_cookies.json',
             'perplexity': 'perplexity_cookies.json', 'glm': 'glm_cookies.json',
             'duck': 'duck_cookies.json',
             'pollinations': 'pollinations_cookies.json',
             'arena': 'arena_cookies.json',
             'huggingchat': 'huggingchat_cookies.json',
             'groq': 'groq_cookies.json',
             'cerebras': 'cerebras_cookies.json',
             'modelscope': 'modelscope_cookies.json',
             'mistral_api': 'mistral_api_cookies.json',
             'openrouter': 'openrouter_cookies.json',
             'llm7': 'llm7_cookies.json',
             'google_ai_studio': 'google_ai_studio_cookies.json',
             'cohere': 'cohere_cookies.json',
             'cloudflare': 'cloudflare_cookies.json'}
    return _data_dir() / files[name]


class _State:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.started = False
        self.renewing: Dict[str, bool] = {}
        self.results: Dict[str, Dict[str, Any]] = {}
        self.counts: Dict[str, Tuple[str, int]] = {}  # provider -> (day, n)
        self.counts_seeded = False
        # inline request-path remediation bookkeeping
        self.inline_counts: Dict[str, Tuple[str, int]] = {}  # provider -> (hour, n)
        self.inline_last: Dict[str, Tuple[str, float]] = {}  # provider -> (ok|failed, ts)
        self.inline_threads: Dict[str, threading.Thread] = {}
        # circuit breaker per (provider, rung): after N consecutive failures a
        # rung cools down so e.g. an expensive browser login stops burning the
        # daily budget on a hopelessly blocked egress (cheap rungs keep trying)
        self.rung_fails: Dict[Tuple[str, str], int] = {}
        self.rung_blocked_until: Dict[Tuple[str, str], float] = {}


_STATE = _State()

# Global signup serialization (council finding): two concurrent signups for
# different providers still share the egress IP and mail backends — racing
# them invites IP/email-domain bans. One signup at a time, process-wide
# (the CLI is a separate OS context, hence a file lock, not a threading one).
_SIGNUP_LOCK_PATH = _data_dir() / 'signup.lock'


def _rotate_history(path) -> None:
    """Keep the JSONL log bounded: over ~1 MB keep only the newest 2000 lines."""
    try:
        if path.exists() and path.stat().st_size > 1_000_000:
            lines = path.read_text(encoding='utf-8').splitlines()
            path.write_text('\n'.join(lines[-2000:]) + '\n', encoding='utf-8')
    except OSError:
        pass


def _log_history(provider: str, event: str, detail: Any = '') -> None:
    entry = {'ts': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
             'provider': provider, 'event': event, 'detail': str(detail)[:1000]}
    with _STATE.lock:
        _STATE.results[provider] = entry
    try:
        path = _data_dir() / 'refresher' / 'history.jsonl'
        path.parent.mkdir(parents=True, exist_ok=True)
        _rotate_history(path)
        with path.open('a', encoding='utf-8') as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + '\n')
    except OSError:
        pass


# --------------------------------------------------------------- cookie jars
@contextmanager
def _file_lock(path: Path):
    """Exclusive advisory lock guarding cross-process jar/account writes.

    The refresher daemon thread, the request-path ``renew_inline`` ladder and
    CLI ``python -m dsk.refresher`` invocations are separate OS contexts that
    all read-modify-write the same JSON files — without the lock they clobber
    each other's updates (e.g. a fresh session token lost to a racing writer).
    """
    import fcntl
    fh = open(str(path) + '.lock', 'w')
    try:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        yield fh
    finally:
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        finally:
            fh.close()


def _norm_email(email: str) -> str:
    """Normalize gmail/googlemail addresses: lowercase + strip dots from the
    local part. emailnator hands out dot-variants of the SAME inbox, so
    'a.b@gmail.com' and 'ab@gmail.com' are one mailbox — comparing them
    naively orphans the signup mailbox and grows accounts.json unboundedly.
    """
    e = (email or '').strip().lower()
    if '@' in e:
        local, _, domain = e.partition('@')
        if domain in ('gmail.com', 'googlemail.com'):
            return local.replace('.', '') + '@gmail.com'
    return e


def _load_jar(name: str) -> Dict[str, str]:
    """Cookies as a flat {name: value} dict (jar file first, env fallback)."""
    jar: Dict[str, str] = {}
    path = _jar_path(name)
    try:
        if path.is_file():
            data = json.loads(path.read_text(encoding='utf-8'))
            if isinstance(data, dict) and isinstance(data.get('cookies'), dict):
                data = data['cookies']  # deepseek bypass format
            if isinstance(data, dict):
                jar = {str(k): str(v) for k, v in data.items()}
            elif isinstance(data, list):
                jar = {str(e.get('name')): str(e.get('value'))
                       for e in data if isinstance(e, dict) and e.get('name')}
    except (OSError, ValueError):
        jar = {}
    if name == 'gemini':
        jar.setdefault('__Secure-1PSID',
                       (os.getenv('GEMINI_1PSID', '') or
                        os.getenv('GEMINI_COOKIES_1PSID', '')).strip())
        jar.setdefault('__Secure-1PSIDTS',
                       (os.getenv('GEMINI_1PSIDTS', '') or
                        os.getenv('GEMINI_COOKIES_1PSIDTS', '')).strip())
    elif name == 'chatgpt':
        # Merge env credentials so the refresh rung, _has_creds and the
        # live verifiers all read ONE source (jar + env). The env bearer
        # is surfaced under the same key the provider reads from the jar.
        jar.setdefault('accessToken', (os.getenv('CHATGPT_ACCESS_TOKEN', '')
                                      or os.getenv('CHATGPT_SESSION_TOKEN', ''))
                       .strip())
        raw = (os.getenv('CHATGPT_SESSION_COOKIES', '') or '').strip()
        if raw:
            try:
                data = json.loads(raw)
                entries = data if isinstance(data, list) else \
                    list(data.items()) if isinstance(data, dict) else []
                for e in entries:
                    if isinstance(e, dict) and e.get('name'):
                        jar.setdefault(str(e.get('name')),
                                       str(e.get('value')))
                    elif isinstance(e, (list, tuple)) and len(e) == 2:
                        jar.setdefault(str(e[0]), str(e[1]))
            except (ValueError, AttributeError, TypeError):
                pass  # env JSON malformed — the provider reports it
    else:
        # Token providers keep their primary credential in one env var; merge
        # it so _has_creds and the live verifiers below agree on one source.
        primary = {'claude': ('sessionKey', 'CLAUDE_SESSION_KEY'),
                   'grok': ('sso', 'GROK_SSO'),
                   'kimi': ('token', 'KIMI_TOKEN'),
                   'mistral': ('session_token', 'MISTRAL_SESSION_TOKEN'),
                   'qwen': ('token', 'QWEN_TOKEN'),
                   'groq': ('api_key', 'GROQ_API_KEY'),
                   'cerebras': ('api_key', 'CEREBRAS_API_KEY'),
                   'modelscope': ('api_key', 'MODELSCOPE_API_KEY'),
                   'mistral_api': ('api_key', 'MISTRAL_API_KEY'),
                   'openrouter': ('api_key', 'OPENROUTER_API_KEY'),
                   'llm7': ('api_key', 'LLM7_API_KEY'),
                   'google_ai_studio': ('api_key', 'GOOGLE_AI_STUDIO_API_KEY'),
                   'cohere': ('api_key', 'COHERE_API_KEY'),
                   'cloudflare': ('api_key', 'CLOUDFLARE_API_TOKEN')}.get(name)
        if primary:
            jar.setdefault(primary[0], (os.getenv(primary[1], '') or '').strip())
    return {k: v for k, v in jar.items() if k and v and k != 'cookies'}


def _chatgpt_restore_token(jar: Dict[str, str], pre_token: str) -> None:
    """An incomplete login must not strand a working bearer.

    The login-in-progress page of chatgpt.com also sets a FRESH
    ``accessToken`` — a login-stage bearer that authenticates the model
    list but 403s on conversation endpoints. When the login did not
    complete, restore the previous token (or drop the login-stage one if
    there was none) so the next rung starts from known state."""
    new_token = jar.get('accessToken', '')
    if pre_token and new_token != pre_token:
        _save_jar('chatgpt', {'accessToken': pre_token})
    elif not pre_token and new_token:
        _remove_jar_key('chatgpt', 'accessToken')


def _remove_jar_key(name: str, key: str) -> None:
    """Drop one key from the jar file (same lock/atomicity as _save_jar)."""
    path = _jar_path(name)
    with _file_lock(path):
        try:
            data = json.loads(path.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            return
        if isinstance(data, dict) and data.pop(key, None) is not None:
            tmp = path.with_suffix('.new')
            tmp.write_text(json.dumps(data, indent=2), encoding='utf-8')
            os.replace(tmp, path)


def _save_jar(name: str, updates: Dict[str, str]) -> None:
    """Merge cookie updates into the jar file (atomic, bot-managed).

    The read-modify-write runs under an advisory file lock: the refresher
    daemon thread and CLI renewal processes both write these jars and would
    otherwise clobber each other's updates (a fresh session token lost to a
    racing writer). The tmp file is created 0o600 — write_text honours the
    default umask, leaving a world-readable window before the chmod.
    """
    path = _jar_path(name)
    with _file_lock(path):
        merged: Any
        try:
            existing = json.loads(path.read_text(encoding='utf-8')) if path.is_file() else None
        except (OSError, ValueError):
            existing = None
        if isinstance(existing, dict) and isinstance(existing.get('cookies'), dict):
            base, fmt = dict(existing['cookies']), 'bypass'
        elif isinstance(existing, list):
            base = {str(e.get('name')): str(e.get('value'))
                    for e in existing if isinstance(e, dict) and e.get('name')}
            fmt = 'list'
        else:
            base, fmt = dict(existing or {}), 'dict'
        base.update({k: v for k, v in updates.items() if k and v})
        if fmt == 'bypass':
            merged = {'cookies': base,
                      'user_agent': (existing or {}).get('user_agent', '')}
        elif fmt == 'list':
            merged = [{'name': k, 'value': v} for k, v in base.items()]
        else:
            merged = base
        tmp = path.with_suffix(path.suffix + '.new')
        fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, 'w', encoding='utf-8') as fh:
            fh.write(json.dumps(merged, indent=2, ensure_ascii=False))
        os.replace(tmp, path)
        try:
            path.chmod(0o600)
        except OSError:
            pass


def _save_deepseek_token(token: str) -> Path:
    path = _data_dir() / 'deepseek_token'
    tmp = path.with_suffix('.new')
    # create the tmp file with 0o600 from the start — write_text honours the
    # default umask, leaving a world-readable window before the chmod
    fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, 'w', encoding='utf-8') as fh:
        fh.write(token.strip())
    os.replace(tmp, path)
    try:
        path.chmod(0o600)
    except OSError:
        pass
    return path


# -------------------------------------------------- credential bootstrap
def _has_creds(name: str) -> bool:
    """True when the provider has any usable credential (file or env).

    Used to decide whether the daemon must bootstrap (auto-signup) a
    provider. For deepseek only the userToken counts — bypass cookies
    alone cannot serve requests."""
    if name == 'deepseek':
        if os.getenv('DEEPSEEK_AUTH_TOKEN', '').strip():
            return True
        try:
            f = _data_dir() / 'deepseek_token'
            return bool(f.is_file() and f.read_text(encoding='utf-8').strip())
        except OSError:
            return False
    if name == 'chatgpt':
        if (os.getenv('CHATGPT_ACCESS_TOKEN', '')
                or os.getenv('CHATGPT_SESSION_TOKEN', '')).strip():
            return True
        if os.getenv('CHATGPT_SESSION_COOKIES', '').strip():
            return True
        # a jar full of CloudFront/oai-did cookies is NOT a session: the
        # provider needs a session cookie (2026: auth-session-minimized)
        # or a usable bearer
        jar = _load_jar('chatgpt')
        return bool(jar.get('__Secure-next-auth.session-token')
                   or jar.get('auth-session-minimized')
                   or jar.get('oai-sc') or jar.get('accessToken'))
    if name in ('claude', 'grok', 'qwen', 'kimi'):
        env_key = {'claude': 'CLAUDE_SESSION_KEY', 'grok': 'GROK_SSO',
                   'qwen': 'QWEN_TOKEN', 'kimi': 'KIMI_TOKEN'}[name]
        if os.getenv(env_key, '').strip():
            return True
        jar = _load_jar(name)
        if name == 'claude':
            return bool(jar.get('sessionKey'))
        if name == 'grok':
            return bool(jar.get('sso') or jar.get('sso-rw'))
        if name == 'kimi':
            return bool(jar.get('token') or jar.get('jwt'))
        return bool(jar.get('token'))
    if name in ('copilot', 'perplexity', 'glm', 'duck', 'pollinations'):
        return True  # anonymous reverse-engineered modes always available
    if name == 'arena':
        # Dormant credential provider: the stream API answers 401 "User not
        # found" without an authenticated session. Public disposable pools
        # are domain-blocklisted and gmail aliases normalize to bot-squatted
        # accounts, so the only viable path is an operator-provided cookie
        # dump (or signup once I4F_MAIL_DOMAIN is configured). Any jar entry
        # counts as provisioned — the 401 gate is enforced at request time.
        if (os.getenv('ARENA_COOKIES', '') or '').strip():
            return True
        return bool(_load_jar('arena'))
    if name == 'huggingchat':
        # Chat POSTs answer 401 "You have to be logged in." without an HF
        # session; the model catalog is public. Any non-empty jar counts as
        # provisioned — liveness is verified by the refresh rung (v2/user).
        if (os.getenv('HUGGINGCHAT_COOKIES', '') or '').strip():
            return True
        return bool(_load_jar('huggingchat'))
    if name == 'groq':
        # Free-tier API key (gsk_...) from console.groq.com. The env
        # var merges into the jar under api_key (see _load_jar); any
        # of the provider's accepted key names counts as provisioned
        # - liveness is checked by the refresh rung (Bearer /models).
        if os.getenv('GROQ_API_KEY', '').strip():
            return True
        jar = _load_jar('groq')
        return bool(jar.get('api_key') or jar.get('key')
                    or jar.get('token'))
    if name == 'cerebras':
        # Free-tier API key (csk-...) from cloud.cerebras.ai. The
        # env var merges into the jar under api_key (see _load_jar);
        # any accepted key name counts as provisioned - liveness is
        # checked by the refresh rung (Bearer /models).
        if os.getenv('CEREBRAS_API_KEY', '').strip():
            return True
        jar = _load_jar('cerebras')
        return bool(jar.get('api_key') or jar.get('key')
                    or jar.get('token'))
    if name == 'modelscope':
        # Free-tier SDK token (ms-<uuid>) from modelscope.cn. The
        # env var merges into the jar under api_key (see _load_jar);
        # any accepted token name counts as provisioned - liveness
        # is checked by the refresh rung (Bearer /models).
        if os.getenv('MODELSCOPE_API_KEY', '').strip():
            return True
        jar = _load_jar('modelscope')
        return bool(jar.get('api_key') or jar.get('key')
                    or jar.get('token'))
    if name == 'mistral_api':
        # Free-tier API key from console.mistral.ai. The env var merges
        # into the jar under api_key (see _load_jar); any accepted key
        # name counts as provisioned - liveness is checked by the
        # refresh rung (Bearer /models).
        if os.getenv('MISTRAL_API_KEY', '').strip():
            return True
        jar = _load_jar('mistral_api')
        return bool(jar.get('api_key') or jar.get('key')
                    or jar.get('token'))
    if name == 'openrouter':
        # Free-pool API key (sk-or-v1-...) from openrouter.ai. The
        # env var merges into the jar under api_key (see _load_jar);
        # any accepted key name counts as provisioned - liveness is
        # checked by the refresh rung (Bearer /key, NOT the public
        # /models catalog).
        if os.getenv('OPENROUTER_API_KEY', '').strip():
            return True
        jar = _load_jar('openrouter')
        return bool(jar.get('api_key') or jar.get('key')
                    or jar.get('token'))
    if name == 'llm7':
        # Free-pool API key from dash.llm7.io. The env var merges
        # into the jar under api_key (see _load_jar); any accepted
        # key name counts as provisioned - liveness is checked by
        # the refresh rung (Bearer /v1/balance, NOT the public
        # /models catalog).
        if os.getenv('LLM7_API_KEY', '').strip():
            return True
        jar = _load_jar('llm7')
        return bool(jar.get('api_key') or jar.get('key')
                    or jar.get('token'))
    if name == 'google_ai_studio':
        # Free AI Studio API key (AIzaSy...) from
        # aistudio.google.com/apikey. The env var merges into the jar
        # under api_key (see _load_jar); GEMINI_API_KEY is accepted as
        # the official SDK alias. Liveness is checked by the refresh
        # rung (native ListModels, NOT anonymous).
        if (os.getenv('GOOGLE_AI_STUDIO_API_KEY', '').strip()
                or os.getenv('GEMINI_API_KEY', '').strip()):
            return True
        jar = _load_jar('google_ai_studio')
        return bool(jar.get('api_key') or jar.get('key')
                    or jar.get('token'))
    if name == 'cohere':
        # Free trial key from dashboard.cohere.com/api-keys. The env
        # var merges into the jar under api_key (see _load_jar); any
        # accepted key name counts as provisioned - liveness is
        # checked by the refresh rung (Bearer /v1/models).
        if os.getenv('COHERE_API_KEY', '').strip():
            return True
        jar = _load_jar('cohere')
        return bool(jar.get('api_key') or jar.get('key')
                    or jar.get('token'))
    if name == 'cloudflare':
        # Workers AI needs BOTH an API token and the 32-hex account
        # id. The env vars merge into the jar under api_key/
        # account_id (see _load_jar); liveness is checked by the
        # refresh rung (authenticated models/search - no neuron burn).
        if (os.getenv('CLOUDFLARE_API_TOKEN', '').strip()
                and os.getenv('CLOUDFLARE_ACCOUNT_ID', '').strip()):
            return True
        jar = _load_jar('cloudflare')
        return bool((jar.get('api_key') or jar.get('key')
                     or jar.get('token'))
                    and (jar.get('account_id') or jar.get('account')
                         or jar.get('accountid')))
    if name == 'mistral':
        jar = _load_jar('mistral') or {}
        env_token = os.getenv('MISTRAL_SESSION_TOKEN', '').strip()
        token = env_token or (jar.get('session_token') or '')
        if not token:
            return False
        # The Ory session dies server-side while the token string sits in
        # the jar: presence proves nothing. A cheap authenticated GET is
        # the real check — an expired session answers 401 and Mistral then
        # serves every chat as ANONYMOUS quota (5 msgs/day, IP-bound).
        #
        # 2026-10-09 measured: the check MUST be whoami on auth.mistral.ai,
        # sent as ``Cookie: <jar name>=<token>``. Two decoys look plausible
        # and are dead ends:
        #   - chat.mistral.ai/api/v1/usage answers 401 "Invalid API Key" to
        #     ANY session token (it is the API-key console endpoint) and
        #     403 "Just a moment..." (Cloudflare challenge) to the fixed
        #     cookie name -> _has_creds was permanently False, so the ladder
        #     re-logged the account in on EVERY cycle.
        #   - the token as a bare header, as Bearer, or under the fixed
        #     ``ory_kratos_session`` name all answer 401; only the jar's
        #     dynamic ``ory_session_<rand>`` name as a Cookie validates.
        cookie_name = (jar.get('session_cookie_name') or '').strip() \
            or 'ory_kratos_session'
        try:
            from curl_cffi import requests as _rq
            r = _rq.get('https://auth.mistral.ai/sessions/whoami',
                        headers={'Cookie': f'{cookie_name}={token}'},
                        impersonate='chrome120', timeout=20)
            if r.status_code == 200:
                return True
        except Exception:  # noqa: BLE001 — network error: cannot verify
            # A hiccup must not read as "no credentials": that would send
            # the ladder to signup and burn a fresh identity.
            return True
        if env_token and not (jar.get('session_cookie_name') or '').strip():
            # Hand-imported token, cookie name unknown: whoami cannot replay
            # it. The operator set it deliberately; the provider validates
            # it at request time.
            return True
        return False
    # gemini: a real session means a __Secure-1PSID cookie (env already
    # merged into the jar by _load_jar). Other google.com cookies alone
    # are not a usable session.
    return bool(_load_jar('gemini').get('__Secure-1PSID'))


_ACCOUNTS_FILE = 'accounts.json'


def _load_accounts() -> Dict[str, Dict[str, str]]:
    """Accounts the bot created itself (provider -> {email, password,...})."""
    try:
        data = json.loads((_data_dir() / _ACCOUNTS_FILE)
                          .read_text(encoding='utf-8'))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _save_account(name: str, email: str, password: str,
                  backend: str = '', extra: Optional[Dict[str, Any]] = None
                  ) -> None:
    """Persist a bot-created account so later renewals can re-login.

    When the caller doesn't supply a new ``mail_session`` (e.g. a signup
    rung that reuses a pre-existing email/password but has no fresh
    session from ``mailgen.create_email``), the previously stored
    ``mail_session`` / ``backend`` are PRESERVED — dropping it would
    orphan the signup mailbox and break the activation-link polling on
    the next cycle.
    """
    path = _data_dir() / _ACCOUNTS_FILE
    with _file_lock(path):
        accs = _load_accounts()
        prev = dict(accs.get(name) or {})
        entry = {'email': email, 'password': password, 'backend': backend,
                 'ts': time.strftime('%Y-%m-%dT%H:%M:%S%z')}
        if extra:
            entry.update(extra)
        # gmail dot-variants are the SAME emailnator inbox — compare normalized
        # so the signup mailbox is never orphaned and accounts.json stays one
        # entry per real mailbox
        same_email = (_norm_email(prev.get('email') or '')
                      == _norm_email(email or ''))
        if same_email and not entry.get('mail_session') and prev.get('mail_session'):
            entry['mail_session'] = prev['mail_session']
        if same_email and not entry.get('backend') and prev.get('backend'):
            entry['backend'] = prev['backend']
        accs[name] = entry
        tmp = path.with_suffix('.new')
        fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, 'w', encoding='utf-8') as fh:
            fh.write(json.dumps(accs, indent=2, ensure_ascii=False))
        os.replace(tmp, path)
        try:
            path.chmod(0o600)
        except OSError:
            pass


def _proxies_kwargs(url: str) -> Dict[str, Any]:
    try:
        from . import proxies as _px
        return _px.proxies_kwargs(url=url)
    except Exception:
        return {}


_UA = ('Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 '
       '(KHTML, like Gecko) Chrome/126.0 Safari/537.36')


# ------------------------------------------------------- HTTP refresh layer
def _http_get(url: str, cookies: Dict[str, str], timeout: int = 30,
              headers: Optional[Dict[str, str]] = None):
    import requests
    all_headers = {'User-Agent': _UA}
    if headers:
        all_headers.update(headers)
    return requests.get(url, cookies=cookies,
                        headers=all_headers, timeout=timeout,
                        allow_redirects=True, **_proxies_kwargs(url))


def refresh_gemini() -> Tuple[bool, str]:
    jar = _load_jar('gemini')
    if not jar.get('__Secure-1PSID'):
        return False, 'no gemini credentials (jar/env)'
    resp = _http_get('https://gemini.google.com/app', jar)
    if resp.status_code in (401, 403) or 'accounts.google.com' in str(resp.url):
        return False, 'session rejected (cookies expired) - re-login needed'
    if resp.status_code != 200:
        return False, f'HTTP {resp.status_code}'
    # The /app shell is served even to logged-out visitors — only the
    # presence of the XSRF token (SNlM0e) proves the cookies authenticate.
    # Without this check an expired session reads as "valid" forever and
    # the ladder never escalates past this rung.
    if '"SNlM0e"' not in resp.text:
        return False, ('no SNlM0e token on /app (cookies expired) '
                       '- re-login needed')
    rotated = {k: v for k, v in dict(resp.cookies).items()
               if k in ('__Secure-1PSID', '__Secure-1PSIDTS') and v}
    if rotated:
        _save_jar('gemini', rotated)
        return True, f"rotated: {','.join(rotated)}"
    return True, 'session valid (no rotation offered)'


def refresh_chatgpt() -> Tuple[bool, str]:
    jar = _load_jar('chatgpt')
    # 2026-10: the web app sets auth-session-minimized/oai-sc; the legacy
    # __Secure-next-auth.session-token remains valid when pasted manually.
    session_cookie = (jar.get('__Secure-next-auth.session-token')
                      or jar.get('auth-session-minimized')
                      or jar.get('oai-sc') or '').strip()
    token = (jar.get('accessToken') or '').strip()
    if not session_cookie and not token:
        return False, 'no chatgpt credentials (jar/env)'
    if not session_cookie:
        # A pasted bearer cannot be rotated over HTTP — the session endpoint
        # only issues tokens FOR session cookies. Validate it live instead:
        # reporting "ok" on an unchecked bearer made the ladder declare the
        # provider renewed while every generation 403'd downstream.
        check_cookies = {k: v for k, v in jar.items() if k != 'accessToken'}
        try:
            resp = _http_get('https://chatgpt.com/backend-api/me',
                             check_cookies,
                             headers={'Authorization': f'Bearer {token}'})
        except Exception as e:  # noqa: BLE001 — network issue is not auth
            return False, f'bearer check unreachable: {type(e).__name__}'
        if resp.status_code == 200:
            return True, 'bearer valid (backend-api/me 200)'
        return False, (f'bearer rejected (HTTP {resp.status_code}) and no '
                       'session cookie to rotate — re-login required')
    resp = _http_get('https://chatgpt.com/api/auth/session', jar)
    if resp.status_code in (401, 403):
        return False, 'session cookies rejected - re-login needed'
    if resp.status_code != 200:
        return False, f'HTTP {resp.status_code}'
    data = {}
    try:
        data = resp.json() or {}
    except ValueError:
        pass
    fresh_token = str(data.get('accessToken') or '').strip()
    if not fresh_token:
        # 200 without a token means the session cookies are expired —
        # the endpoint still answers, so this must NOT read as "valid"
        # (it made the ladder believe credentials were fine).
        return False, ('session reachable but no accessToken '
                       '(cookies expired) - re-login needed')
    new_cookies = {k: v for k, v in dict(resp.cookies).items() if v}
    # the endpoint's JSON also carries the fresh bearer — persist it too,
    # so the provider has a valid token even if the session cookies later
    # go stale before the next renewal rung runs.
    new_cookies['accessToken'] = fresh_token
    _save_jar('chatgpt', new_cookies)
    return True, 'session valid, token issued'


def refresh_deepseek() -> Tuple[bool, str]:
    """userToken cannot be rotated over HTTP; just verify it live."""
    try:
        from . import selfheal
        status, detail = selfheal._probe_once('deepseek')
        return (status == 'ok'), f'{status}: {detail}'
    except Exception as e:  # noqa: BLE001
        return False, f'probe failed: {e}'


def _manual_only(provider: str, hint: str):
    """Refresher/sign-up stub for token-based RE providers that have no
    automated account creation: their credential is a manually exported
    cookie/token, so the bot only reports how to set it."""
    def _f() -> Tuple[bool, str]:
        return False, f'{provider}: no automated signup — {hint}'
    return _f


def _anonymous(provider: str):
    """Refresher stub for providers that need no credential at all."""
    def _f() -> Tuple[bool, str]:
        return True, f'{provider}: anonymous access — nothing to refresh'
    return _f


def _egress_rotate_providers() -> set:
    """Anonymous providers whose refusals are a property of the exit IP.

    Anonymous Copilot is geo-blocked from EU egresses (edge 460), so the
    only lever is the exit IP and the early-return rotation rung is the
    whole renewal. Perplexity USED to sit here too ("fraud_authwall_upsell
    for datacenter IPs") — measured 2026-10 across direct AND several
    rotated pool egresses, that wall is session-shaped, not egress-shaped:
    no fresh IP ever lifted it, and the rotation early-return masked the
    real fix. Perplexity now has a credential path (the NextAuth magic-link
    signup rung), so it belongs to the credential ladder. Override with
    I4F_EGRESS_ROTATE.
    """
    raw = os.getenv('I4F_EGRESS_ROTATE', 'copilot,duck')
    return {p.strip().lower() for p in raw.split(',') if p.strip()}


def rotate_egress(name: str) -> Tuple[bool, str]:
    """Drop the provider's sticky proxy assignment and draw a new exit.

    Returns True when the provider now leaves through a *different* route, so
    the router's retry hits the upstream from a new IP. ``direct_ok=False``
    keeps the draw on the proxy pool: a geo-blocked provider must not be
    handed back the same datacenter direct egress that was just refused.
    """
    try:
        from . import proxies
    except Exception as e:  # noqa: BLE001
        return False, f'proxy pool unavailable: {e}'
    try:
        old = proxies.current(name)
        if old:
            proxies.mark_failure(old)
        pool = proxies.ensure_pool()
        new = proxies.get_proxy(name, direct_ok=False)
    except Exception as e:  # noqa: BLE001
        return False, f'egress rotation failed: {type(e).__name__}: {e}'
    if not new:
        return False, (f'{name}: no proxy available to rotate to '
                       f'(pool size {pool}) — anonymous egress is fixed')
    if new == old:
        return False, f'{name}: egress unchanged ({new})'
    return True, f'{name}: egress rotated {old or "direct"} -> {new}'


def refresh_claude() -> Tuple[bool, str]:
    """claude.ai sessionKey cannot be rotated over HTTP; verify it live."""
    if not _has_creds('claude'):
        return False, 'no claude credentials — set CLAUDE_SESSION_KEY or claude_cookies.json'
    try:
        resp = _http_get('https://claude.ai/api/organizations',
                         {'sessionKey': _load_jar('claude').get('sessionKey', '')})
    except Exception as e:  # noqa: BLE001
        return False, f'verify failed: {type(e).__name__}: {e}'
    if resp.status_code == 200:
        return True, 'session valid'
    if resp.status_code in (401, 403):
        return False, 'sessionKey rejected — re-export from claude.ai'
    return False, f'HTTP {resp.status_code}'


def refresh_grok() -> Tuple[bool, str]:
    if not _has_creds('grok'):
        return False, 'no grok credentials — set GROK_SSO or grok_cookies.json'
    jar = _load_jar('grok')
    cookies = {k: v for k, v in jar.items() if k in ('sso', 'sso-rw')}
    try:
        resp = _http_get('https://grok.com/rest/rate-limits', cookies)
    except Exception as e:  # noqa: BLE001
        return False, f'verify failed: {type(e).__name__}: {e}'
    if resp.status_code == 200:
        return True, 'session valid'
    if resp.status_code in (401, 403):
        return False, 'sso cookie rejected — re-export from grok.com'
    if resp.status_code == 429:
        return True, 'session valid (rate limited)'
    return False, f'HTTP {resp.status_code}'


def _qwen_headers() -> Dict[str, str]:
    """WAF-safe request headers for chat.qwen.ai (static bx-ua fingerprint
    from the provider module; falls back to a plain UA on import failure)."""
    try:
        from .providers.qwen_provider import _headers
        return _headers()
    except Exception:  # noqa: BLE001
        return {'User-Agent': _UA}


def _qwen_request(method: str, url: str, **kw) -> Tuple[Any, Optional[str]]:
    """``requests`` with the pooled egress, retried once DIRECT on failure.

    Pooled exits intermittently break TLS to chat.qwen.ai (self-signed
    cert / SOCKS refused). The qwen auth endpoints are WAF-free, so a
    direct connection is a safe second rung. Returns (response, note)
    where note explains the fallback (or None); the direct attempt's
    exception propagates if it also fails.
    """
    import requests
    timeout = kw.pop('timeout', 30)
    try:
        return (requests.request(method, url, timeout=timeout,
                                 **kw, **_proxies_kwargs(url)), None)
    except Exception as pooled_exc:  # noqa: BLE001
        note = f'direct-retry after {type(pooled_exc).__name__}'
    return requests.request(method, url, timeout=timeout, **kw), note


def _qwen_signin(email: str, password: str) -> Tuple[bool, str]:
    """HTTP re-login on chat.qwen.ai (OpenWebUI-style /api/v1/auths/signin).

    The signin path is NOT WAF-challenged (unlike /signup, which sits behind
    an Aliyun slide-captcha), so token renewal needs no browser: exchange the
    stored email/password for a fresh JWT and persist it in the jar.
    """
    import requests
    try:
        resp, note = _qwen_request(
            'POST', 'https://chat.qwen.ai/api/v1/auths/signin',
            json={'email': email, 'password': password},
            headers=_qwen_headers())
    except Exception as e:  # noqa: BLE001
        return False, f'signin failed: {type(e).__name__}: {e}'
    if resp.status_code != 200:
        detail = ''
        try:
            detail = str((resp.json() or {}).get('detail') or '')[:80]
        except Exception:  # noqa: BLE001
            pass
        return False, (f'signin rejected (HTTP {resp.status_code})'
                       + (f': {detail}' if detail else ''))
    try:
        body = resp.json() or {}
    except ValueError:
        return False, 'signin returned a non-JSON body'
    token = str(body.get('token') or '').strip()
    if not token:
        return False, 'signin ok but no token in response'
    _save_jar('qwen', {'token': token, 'email': email})
    return True, 're-signed in; fresh qwen token saved'


def refresh_qwen() -> Tuple[bool, str]:
    """Verify the stored token; re-login, or converge a pending activation."""
    jar = _load_jar('qwen')
    token = (jar.get('token') or '').strip()
    if token:
        try:
            resp, note = _qwen_request(
                'GET', 'https://chat.qwen.ai/api/v1/auths',
                headers={**_qwen_headers(),
                         'Authorization': f'Bearer {token}'})
            if note:
                _log_history('qwen', 'stage', f'verify fell back to direct ({note})')
        except Exception as e:  # noqa: BLE001
            return False, f'verify failed: {type(e).__name__}: {e}'
        if resp.status_code == 200:
            return True, 'token valid'
        if resp.status_code not in (401, 403):
            return False, f'HTTP {resp.status_code}'
    # token missing/rejected -> HTTP re-login from stored credentials
    # (chat.qwen.ai /signin is not WAF-gated, so no browser is needed)
    email, password = _creds('qwen')
    if not email or not password:
        return False, ('no qwen credentials — set QWEN_TOKEN, or '
                       'QWEN_LOGIN_EMAIL/QWEN_LOGIN_PASSWORD (or let the '
                       'qwen signup rung create an account) to enable '
                       'automatic re-login')
    ok, detail = _qwen_signin(email, password)
    if not ok:
        # 'pending activation' means the account exists but was never
        # confirmed — poll the signup mailbox and open the activation link
        # so renewal converges autonomously across cycles.
        if _qwen_pending(detail):
            _log_history('qwen', 'stage',
                         f'signin pending activation; activating ({detail})')
            return _qwen_activate(email, password)
        return False, f'token rejected; {detail}'
    return True, detail


def _qwen_pending(detail: str) -> bool:
    """True when a signin failure means 'account exists but unactivated'."""
    d = (detail or '').lower()
    return any(n in d for n in ('pending', 'unverified', 'not verified',
                                'verify', 'activation', 'activat'))


def _qwen_mail_session(email: str) -> Optional[Dict[str, Any]]:
    """Rebuild a mailgen session for the qwen account's signup mailbox."""
    stored = _load_accounts().get('qwen') or {}
    if stored.get('email') and stored.get('email') != email:
        return None
    ms = stored.get('mail_session')
    if isinstance(ms, dict) and ms.get('backend') and ms.get('address'):
        return dict(ms)
    if (stored.get('backend') or '').strip() and email:
        return {'backend': stored['backend'], 'address': email}
    return None


def _qwen_activate(email: str, password: str) -> Tuple[bool, str]:
    """Finish a pending chat.qwen.ai activation autonomously.

    New accounts submit to 'pending activation': the confirmation mail lands
    in the bot's signup mailbox. Poll it for the activation URL, open the
    link (plain GET first, then a real headed browser as fallback), and HTTP
    re-login. Called from the login and signup rungs, so the daemon converges
    on activation across renewal cycles with no operator action.
    """
    session = _qwen_mail_session(email)
    if not session:
        return False, 'no mailbox backend recorded for this qwen account'
    link = mailgen.fetch_otp(session, max_wait_s=120, sender_needle='qwen',
                             code_re=re.compile(
                                 r'(https://chat\.qwen\.ai/[^\s"\'<>]*'
                                 r'activate[^\s"\'<>]*)'))
    if not link:
        return False, 'activation link not in mailbox yet'
    _log_history('qwen', 'stage', 'activation link found; opening')
    try:
        _http_get(link, cookies={}, timeout=30)
    except Exception:  # noqa: BLE001 — the browser visit is authoritative
        pass
    page = None
    try:
        page = _browser(proxy=_pool_proxy(), headed=True)
        page.get(link)
        time.sleep(6)
    except Exception:  # noqa: BLE001 — the plain GET above may have sufficed
        pass
    finally:
        if page is not None:
            _close_page(page)
    ok, detail = _qwen_signin(email, password)
    if ok:
        _log_history('qwen', 'stage', 'activation completed; re-login ok')
    return ok, detail


def refresh_kimi() -> Tuple[bool, str]:
    """Lightweight liveness check; the authoritative probe runs at request time."""
    if not _has_creds('kimi'):
        return False, 'no kimi credentials — set KIMI_TOKEN or kimi_cookies.json'
    jar = _load_jar('kimi')
    token = jar.get('token') or jar.get('jwt') or ''
    try:
        resp = _http_get('https://www.kimi.com/', {'token': token})
    except Exception as e:  # noqa: BLE001
        return False, f'verify failed: {type(e).__name__}: {e}'
    if resp.status_code in (401, 403):
        return False, 'token rejected — re-export from kimi.com'
    return True, f'session reachable (HTTP {resp.status_code})'


def refresh_huggingchat() -> Tuple[bool, str]:
    """Verify the HF session against /chat/api/v2/user.

    The endpoint answers {"json": null} for anonymous visitors and the
    profile object for a live session — a precise, cheap liveness probe.
    """
    if not _has_creds('huggingchat'):
        return False, ('no huggingchat credentials — export cookies from '
                       'huggingface.co/chat to huggingchat_cookies.json or '
                       'set HUGGINGCHAT_COOKIES')
    jar = _load_jar('huggingchat')
    try:
        resp = _http_get('https://huggingface.co/chat/api/v2/user', jar,
                         headers={'Accept': 'application/json'})
    except Exception as e:  # noqa: BLE001
        return False, f'verify failed: {type(e).__name__}: {e}'
    if resp.status_code in (401, 403):
        return False, 'session rejected — re-export cookies from huggingface.co'
    if resp.status_code != 200:
        return True, f'session reachable, user check inconclusive (HTTP {resp.status_code})'
    try:
        user = (resp.json() or {}).get('json')
    except ValueError:
        user = None
    if user:
        name = user.get('name') or user.get('username') or 'user'
        return True, f'session live ({name})'
    return False, 'cookies present but session is anonymous — re-export from a logged-in huggingface.co'


def refresh_mistral() -> Tuple[bool, str]:
    """Best-effort token check: Ory Kratos whoami, lenient when unverifiable.

    The authoritative validation runs at request time (the provider raises
    on a rejected token), so an unavailable whoami endpoint is not an error.
    """
    if not _has_creds('mistral'):
        return False, ('no mistral credentials — set MISTRAL_SESSION_TOKEN '
                       'or mistral_cookies.json')
    jar = _load_jar('mistral') or {}
    token = (os.getenv('MISTRAL_SESSION_TOKEN', '') or '').strip()
    env_token = bool(token)
    cookie_name = jar.get('session_cookie_name') or 'ory_kratos_session'
    if not token:
        token = jar.get('session_token') or ''
    try:
        # NOTE: _http_get's second positional is COOKIES, not headers —
        # ``{cookie_name: token}`` is what sends ``Cookie: <name>=<token>``,
        # the only shape whoami accepts (measured 2026-10-09: the token as a
        # header, as Bearer, or under the fixed ory_kratos_session name all
        # answer 401).
        resp = _http_get('https://auth.mistral.ai/sessions/whoami',
                         {cookie_name: token})
    except Exception as e:  # noqa: BLE001
        return False, f'verify failed: {type(e).__name__}: {e}'
    if resp.status_code == 200:
        # chat.mistral.ai gates model output on a VERIFIED address: a session
        # over an unverified identity still hits the account-upsell wall.
        # Converge verification autonomously here (mailbox OTP) so the ladder
        # never needs a human.
        try:
            idn = (resp.json().get('identity') or {})
            unverified = [a for a in (idn.get('verifiable_addresses') or [])
                          if isinstance(a, dict) and not a.get('verified')]
        except Exception:  # noqa: BLE001
            unverified = []
        if not unverified:
            return True, 'session valid'
        acc = _load_accounts().get('mistral') or {}
        v_email = (unverified[0].get('value') or acc.get('email') or '')
        mail_session = acc.get('mail_session') \
            if isinstance(acc.get('mail_session'), dict) else None
        if not mail_session and acc.get('backend'):
            mail_session = {'backend': acc['backend'], 'address': v_email}
        ok, detail = _mistral_verify_email(v_email, mail_session)
        if ok:
            _log_history('mistral', 'verify', detail)
            return True, f'session valid (address verified: {detail})'
        return False, f'account unverified; verification failed: {detail}'
    if resp.status_code in (401, 403):
        # Only a hand-imported ENV token may pass unverified: whoami cannot
        # replay the dynamic ``ory_session_<rand>`` cookie name for it. A
        # jar-sourced token being rejected is a REAL expiry — return False so
        # the ladder escalates to re-login/signup instead of masking it.
        if env_token and not jar.get('session_cookie_name'):
            return True, ('whoami cannot replay the session cookie name — '
                          'env token unverified (validated at request time)')
        return False, 'session token rejected — re-export from chat.mistral.ai'
    return True, (f'whoami unavailable (HTTP {resp.status_code}) — '
                  'token unverified (validated at request time)')


def refresh_perplexity() -> Tuple[bool, str]:
    """Validate the signed-in perplexity jar against the NextAuth session.

    The upstream hard-walls anonymous sessions (``fraud_authwall_upsell`` on
    every answer, measured across direct AND rotated egress 2026-10), so the
    jar must hold a live NextAuth session: GET ``/api/auth/session`` with the
    jar cookies and require a user object. Unreachable endpoints read as
    "unverified" (True — validated at request time) so a network hiccup never
    churns identities; only a clear no-user/40x answer escalates to signup.
    """
    jar = _load_jar('perplexity')
    if not jar:
        return False, 'no perplexity jar — signup needed'
    cookie = '; '.join(f'{k}={v}' for k, v in jar.items()
                       if k != 'email' and v)
    if not cookie:
        return False, 'perplexity jar has no cookies — signup needed'
    from curl_cffi import requests as cffi
    try:
        resp = cffi.get('https://www.perplexity.ai/api/auth/session',
                        headers={'Cookie': cookie,
                                 'Accept': 'application/json',
                                 'User-Agent': (
                                     'Mozilla/5.0 (X11; Linux x86_64) '
                                     'AppleWebKit/537.36 (KHTML, like Gecko) '
                                     'Chrome/120.0.0.0 Safari/537.36')},
                        impersonate='chrome120', timeout=20,
                        **_proxies_kwargs('https://www.perplexity.ai'))
    except Exception as e:  # noqa: BLE001 — network hiccup, unverified
        return True, (f'session check unreachable ({type(e).__name__}) — '
                      'jar unverified, validated at request time')
    if resp.status_code == 200:
        try:
            user = (resp.json() or {}).get('user') or {}
        except ValueError:
            user = {}
        if user.get('email') or user.get('id'):
            return True, (f'session valid '
                          f'({user.get("email") or user.get("id")})')
        return False, 'session endpoint returned no user — expired, re-signup'
    return False, f'session check HTTP {resp.status_code} — re-signup'



def refresh_groq() -> Tuple[bool, str]:
    """Validate the Groq API key against GET /openai/v1/models.

    A live free-tier key answers 200 with the model catalog; an expired
    or revoked key answers 401 invalid_api_key. A 429 still proves the
    key is accepted (auth runs before rate limiting), so it reads as
    alive; any other status is inconclusive - the authoritative check
    runs at request time.
    """
    if not _has_creds('groq'):
        return False, ('no groq credentials - set GROQ_API_KEY or create '
                       'a key in console.groq.com/keys (groq_cookies.json)')
    key = (_load_jar('groq').get('api_key')
           or os.getenv('GROQ_API_KEY', '')).strip()
    try:
        resp = _http_get('https://api.groq.com/openai/v1/models', {},
                         headers={'Authorization': f'Bearer {key}',
                                  'Accept': 'application/json'})
    except Exception as e:  # noqa: BLE001
        return False, f'verify failed: {type(e).__name__}: {e}'
    if resp.status_code in (401, 403):
        return False, ('API key rejected - create a new one in '
                       'console.groq.com/keys')
    if resp.status_code == 429:
        return True, 'rate limited but key accepted (HTTP 429)'
    if resp.status_code != 200:
        return True, (f'key reachable, liveness inconclusive (HTTP '
                      f'{resp.status_code}) - validated at request time')
    try:
        n = len(((resp.json() or {}).get('data')) or [])
    except ValueError:
        n = 0
    return True, f'API key valid ({n} models visible)'


def refresh_cerebras() -> Tuple[bool, str]:
    """Validate the Cerebras API key against GET /v1/models.

    A live free-tier key answers 200 with the model catalog; an expired
    or revoked key answers 401 wrong_api_key (flat body, not nested
    under error like Groq's). A 429 still proves the key is accepted
    (auth runs before rate limiting), so it reads as alive; any other
    status is inconclusive - the authoritative check runs at request
    time.
    """
    if not _has_creds('cerebras'):
        return False, ('no cerebras credentials - set CEREBRAS_API_KEY or '
                       'create a key in cloud.cerebras.ai '
                       '(cerebras_cookies.json)')
    key = (_load_jar('cerebras').get('api_key')
           or os.getenv('CEREBRAS_API_KEY', '')).strip()
    try:
        resp = _http_get('https://api.cerebras.ai/v1/models', {},
                         headers={'Authorization': f'Bearer {key}',
                                  'Accept': 'application/json'})
    except Exception as e:  # noqa: BLE001
        return False, f'verify failed: {type(e).__name__}: {e}'
    if resp.status_code in (401, 403):
        return False, ('API key rejected - create a new one in '
                       'cloud.cerebras.ai')
    if resp.status_code == 429:
        return True, 'rate limited but key accepted (HTTP 429)'
    if resp.status_code != 200:
        return True, (f'key reachable, liveness inconclusive (HTTP '
                      f'{resp.status_code}) - validated at request time')
    try:
        n = len(((resp.json() or {}).get('data')) or [])
    except ValueError:
        n = 0
    return True, f'API key valid ({n} models visible)'


def refresh_modelscope() -> Tuple[bool, str]:
    """Validate the ModelScope token against GET /v1/models.

    A live free-tier token answers 200 with the model catalog; an expired
    or revoked token answers 401 with "Authentication failed" nested
    under error. A 429 still proves the token is accepted (auth runs
    before rate limiting), so it reads as alive; any other status is
    inconclusive - the authoritative check runs at request time.
    """
    if not _has_creds('modelscope'):
        return False, ('no modelscope credentials - set MODELSCOPE_API_KEY '
                       'or create a token in modelscope.cn '
                       '(modelscope_cookies.json)')
    key = (_load_jar('modelscope').get('api_key')
           or os.getenv('MODELSCOPE_API_KEY', '')).strip()
    try:
        resp = _http_get('https://api-inference.modelscope.cn/v1/models', {},
                         headers={'Authorization': f'Bearer {key}',
                                  'Accept': 'application/json'})
    except Exception as e:  # noqa: BLE001
        return False, f'verify failed: {type(e).__name__}: {e}'
    if resp.status_code in (401, 403):
        return False, ('API token rejected - create a new one in '
                       'modelscope.cn (access tokens page)')
    if resp.status_code == 429:
        return True, 'rate limited but token accepted (HTTP 429)'
    if resp.status_code != 200:
        return True, (f'token reachable, liveness inconclusive (HTTP '
                      f'{resp.status_code}) - validated at request time')
    try:
        n = len(((resp.json() or {}).get('data')) or [])
    except ValueError:
        n = 0
    return True, f'API token valid ({n} models visible)'


def refresh_mistral_api() -> Tuple[bool, str]:
    """Validate the Mistral API key against GET /v1/models.

    A live free-tier key answers 200 with the model catalog (a bare
    JSON array); an invalid key answers 401 with a flat FastAPI body
    {"detail": "Invalid API Key"}. A 429 still proves the key is
    accepted (auth runs before rate limiting), so it reads as alive;
    any other status is inconclusive - the authoritative check runs
    at request time.
    """
    if not _has_creds('mistral_api'):
        return False, ('no mistral API credentials - set MISTRAL_API_KEY '
                       'or create a free key in console.mistral.ai '
                       '(mistral_api_cookies.json)')
    key = (_load_jar('mistral_api').get('api_key')
           or os.getenv('MISTRAL_API_KEY', '')).strip()
    try:
        resp = _http_get('https://api.mistral.ai/v1/models', {},
                         headers={'Authorization': f'Bearer {key}',
                                  'Accept': 'application/json'})
    except Exception as e:  # noqa: BLE001
        return False, f'verify failed: {type(e).__name__}: {e}'
    if resp.status_code in (401, 403):
        return False, ('API key rejected - create a new free key in '
                       'console.mistral.ai (api-keys page)')
    if resp.status_code == 429:
        return True, 'rate limited but key accepted (HTTP 429)'
    if resp.status_code != 200:
        return True, (f'key reachable, liveness inconclusive (HTTP '
                      f'{resp.status_code}) - validated at request time')
    try:
        n = len(resp.json() or [])
    except ValueError:
        n = 0
    return True, f'API key valid ({n} models visible)'


def refresh_openrouter() -> Tuple[bool, str]:
    """Validate the OpenRouter key against GET /api/v1/key.

    The /models catalog is PUBLIC and proves nothing about the key -
    liveness runs against the authenticated /key endpoint: 200
    returns the per-key usage/limit record, 401
    {"error":{"message":"No cookie auth credentials found",
    "code":401}} means rejected. A 429 still proves the key is
    accepted (auth runs before rate limiting), so it reads as
    alive; any other status is inconclusive - the authoritative
    check runs at request time.
    """
    if not _has_creds('openrouter'):
        return False, ('no openrouter credentials - set '
                       'OPENROUTER_API_KEY or create a free key at '
                       'openrouter.ai/settings/keys '
                       '(openrouter_cookies.json)')
    key = (_load_jar('openrouter').get('api_key')
           or os.getenv('OPENROUTER_API_KEY', '')).strip()
    try:
        resp = _http_get('https://openrouter.ai/api/v1/key', {},
                         headers={'Authorization': f'Bearer {key}',
                                  'Accept': 'application/json'})
    except Exception as e:  # noqa: BLE001
        return False, f'verify failed: {type(e).__name__}: {e}'
    if resp.status_code in (401, 403):
        return False, ('API key rejected - create a new free key at '
                       'openrouter.ai/settings/keys')
    if resp.status_code == 429:
        return True, 'rate limited but key accepted (HTTP 429)'
    if resp.status_code != 200:
        return True, (f'key reachable, liveness inconclusive (HTTP '
                      f'{resp.status_code}) - validated at request time')
    return True, 'API key valid (usage/limit record returned)'


def refresh_llm7() -> Tuple[bool, str]:
    """Validate the LLM7 key against GET /v1/balance.

    The /models catalog is PUBLIC and proves nothing about the key -
    liveness runs against the authenticated quota endpoint: 200
    returns the per-key balance record (no quota burn), 401
    {"error":{"code":"invalid_api_key"}} means rejected. A 429 still
    proves the key is accepted (auth runs before rate limiting), so
    it reads as alive; any other status is inconclusive - the
    authoritative check runs at request time.
    """
    if not _has_creds('llm7'):
        return False, ('no llm7 credentials - get a free key at '
                       'dash.llm7.io/#/api-keys and set LLM7_API_KEY '
                       'or llm7_cookies.json')
    key = (_load_jar('llm7').get('api_key')
           or os.getenv('LLM7_API_KEY', '')).strip()
    try:
        resp = _http_get('https://api.llm7.io/v1/balance', {},
                         headers={'Authorization': f'Bearer {key}',
                                  'Accept': 'application/json'})
    except Exception as e:  # noqa: BLE001
        return False, f'verify failed: {type(e).__name__}: {e}'
    if resp.status_code in (401, 403):
        return False, ('API key rejected - create a new free key at '
                       'dash.llm7.io/#/api-keys')
    if resp.status_code == 429:
        return True, 'rate limited but key accepted (HTTP 429)'
    if resp.status_code != 200:
        return True, (f'key reachable, liveness inconclusive (HTTP '
                      f'{resp.status_code}) - validated at request time')
    return True, 'API key valid (balance returned)'


def refresh_google_ai_studio() -> Tuple[bool, str]:
    """Validate the AI Studio key against native ListModels.

    GET v1beta/models with x-goog-api-key returns the model catalog
    (first page, default 50 entries) only for a valid consumer key:
    200 -> valid; 403 {"status":"PERMISSION_DENIED"} (no identity) and
    400 "API key not valid." -> rejected - create a new free key at
    aistudio.google.com/apikey. A 429 still proves the key is
    accepted (quota runs after auth), so it reads as alive.
    """
    if not _has_creds('google_ai_studio'):
        return False, ('no google_ai_studio credentials - get a free '
                       'key at aistudio.google.com/apikey and set '
                       'GOOGLE_AI_STUDIO_API_KEY or '
                       'google_ai_studio_cookies.json')
    key = (_load_jar('google_ai_studio').get('api_key')
           or (os.getenv('GOOGLE_AI_STUDIO_API_KEY', '') or
               os.getenv('GEMINI_API_KEY', ''))).strip()
    try:
        resp = _http_get('https://generativelanguage.googleapis.com'
                         '/v1beta/models', {},
                         headers={'x-goog-api-key': key,
                                  'Accept': 'application/json'})
    except Exception as e:  # noqa: BLE001
        return False, f'verify failed: {type(e).__name__}: {e}'
    if resp.status_code in (401, 403):
        return False, ('API key rejected - create a new free key at '
                       'aistudio.google.com/apikey')
    if resp.status_code == 400:
        text = ''
        try:
            text = resp.text or ''
        except Exception:  # noqa: BLE001
            pass
        if re.search(r'API key not valid|Please pass a valid API key|'
                     r'API_KEY_INVALID', text, re.IGNORECASE):
            return False, ('API key rejected - create a new free key '
                           'at aistudio.google.com/apikey')
        return True, ('key reachable, liveness inconclusive (HTTP 400) '
                      '- validated at request time')
    if resp.status_code == 429:
        return True, 'rate limited but key accepted (HTTP 429)'
    if resp.status_code != 200:
        return True, (f'key reachable, liveness inconclusive (HTTP '
                      f'{resp.status_code}) - validated at request time')
    try:
        models = ((resp.json() or {}).get('models') or [])
    except Exception:  # noqa: BLE001
        models = []
    return True, f'API key valid ({len(models)} models visible)'


def refresh_cohere() -> Tuple[bool, str]:
    """Validate the Cohere trial key against GET /v1/models.

    The native catalog needs the bearer (401 {"message":"no api key
    supplied"} without one); 200 returns the first page of the model
    catalog without burning any trial quota. 401 "Incorrect API key
    provided" -> rejected - create a new trial key at
    dashboard.cohere.com/api-keys. A 429 still proves the key is
    accepted (auth runs before rate limiting), so it reads as alive.
    """
    if not _has_creds('cohere'):
        return False, ('no cohere credentials - create a free trial '
                       'key at dashboard.cohere.com/api-keys and set '
                       'COHERE_API_KEY or cohere_cookies.json')
    key = (_load_jar('cohere').get('api_key')
           or os.getenv('COHERE_API_KEY', '')).strip()
    try:
        resp = _http_get('https://api.cohere.com/v1/models', {},
                         headers={'Authorization': f'Bearer {key}',
                                  'Accept': 'application/json'})
    except Exception as e:  # noqa: BLE001
        return False, f'verify failed: {type(e).__name__}: {e}'
    if resp.status_code in (401, 403):
        return False, ('API key rejected - create a new trial key at '
                       'dashboard.cohere.com/api-keys')
    if resp.status_code == 429:
        return True, 'rate limited but key accepted (HTTP 429)'
    if resp.status_code != 200:
        return True, (f'key reachable, liveness inconclusive (HTTP '
                      f'{resp.status_code}) - validated at request time')
    try:
        models = ((resp.json() or {}).get('models') or [])
    except Exception:  # noqa: BLE001
        models = []
    return True, f'API key valid ({len(models)} models visible)'


def refresh_cloudflare() -> Tuple[bool, str]:
    """Verify the Workers AI token + account id with an authenticated
    catalog probe (models/search?per_page=1 - no neuron burn).

    The v4 envelope reports auth trouble as 401/403 code 10000, 400
    code 9106 (missing auth headers) or 404 code 7003 (unroutable
    account id - routing happens before auth); all three read as
    rejected. A 429 still proves the token routed and was accepted.
    """
    if not _has_creds('cloudflare'):
        return False, ('no cloudflare credentials - create a Workers AI '
                       'token at dash.cloudflare.com/profile/api-tokens '
                       'and set CLOUDFLARE_API_TOKEN + '
                       'CLOUDFLARE_ACCOUNT_ID or cloudflare_cookies.json')
    jar = _load_jar('cloudflare')
    key = (os.getenv('CLOUDFLARE_API_TOKEN', '')
           or jar.get('api_key') or jar.get('key')
           or jar.get('token') or '').strip()
    acc = (os.getenv('CLOUDFLARE_ACCOUNT_ID', '')
           or jar.get('account_id') or jar.get('account')
           or jar.get('accountid') or '').strip()
    url = ('https://api.cloudflare.com/client/v4/accounts/' + acc +
           '/ai/models/search?per_page=1&page=1')
    try:
        resp = _http_get(url, {},
                         headers={'Authorization': f'Bearer {key}',
                                  'Accept': 'application/json'})
    except Exception as e:  # noqa: BLE001
        return False, f'verify failed: {type(e).__name__}: {e}'
    rejected = ('create a Workers AI token at dash.cloudflare.com/profile/'
                'api-tokens (Account > Workers AI > Edit) and set '
                'CLOUDFLARE_API_TOKEN + CLOUDFLARE_ACCOUNT_ID')
    if resp.status_code in (401, 403):
        return False, ('Workers AI credentials rejected (HTTP '
                       f'{resp.status_code}) - {rejected}')
    if resp.status_code == 429:
        return True, 'rate limited but token accepted (HTTP 429)'
    if resp.status_code != 200:
        return False, (f'Workers AI credentials rejected (HTTP '
                       f'{resp.status_code}): '
                       f'{(resp.text or "")[:200]} - {rejected}')
    try:
        data = resp.json() or {}
    except Exception:  # noqa: BLE001
        data = {}
    if isinstance(data, dict) and data.get('success') is False:
        errs = '; '.join(str((e or {}).get('message') or e)
                         if isinstance(e, dict) else str(e)
                         for e in (data.get('errors') or [])[:2])
        return False, (f'Workers AI credentials rejected: {errs} - '
                       f'{rejected}')
    total = 0
    info = data.get('result_info') if isinstance(data, dict) else None
    if isinstance(info, dict):
        try:
            total = int(info.get('total_count') or 0)
        except (TypeError, ValueError):
            total = 0
    return True, f'API key valid ({total} models visible)'



# ---------------------------------------------------------------- Meta
def refresh_meta() -> Tuple[bool, str]:
    """Validate the Muse Spark preview key against GET /v1/models.

    The OpenAI-compatible catalog needs the bearer (401 OpenAI-style
    {"error":{"code":"invalid_api_key"}} without one); 200 returns the
    model list without burning tokens. The preview is US-developers-only,
    so a geo-block reads as 403 — the detail says so.
    """
    if not _has_creds('meta'):
        return False, ('no meta credentials - create a free preview key at '
                       'developers.meta.ai and set META_API_KEY or '
                       'meta_cookies.json')
    key = (_load_jar('meta').get('api_key')
           or os.getenv('META_API_KEY', '')).strip()
    try:
        resp = _http_get('https://api.meta.ai/v1/models', {},
                         headers={'Authorization': f'Bearer {key}',
                                  'Accept': 'application/json'})
    except Exception as e:  # noqa: BLE001
        return False, f'verify failed: {type(e).__name__}: {e}'
    if resp.status_code in (401, 403):
        return False, ('API key rejected (403 may be geo: the preview is '
                       'US-only) - create a new key at developers.meta.ai')
    if resp.status_code == 429:
        return True, 'rate limited but key accepted (HTTP 429)'
    if resp.status_code != 200:
        return True, (f'key reachable, liveness inconclusive (HTTP '
                      f'{resp.status_code}) - validated at request time')
    try:
        models = ((resp.json() or {}).get('data') or [])
    except Exception:  # noqa: BLE001
        models = []
    return True, f'API key valid ({len(models)} models visible)'


def refresh_blackbox() -> Tuple[bool, str]:
    """Validate the Blackbox key with a 1-token completion.

    There is no unauthenticated models endpoint, so liveness costs one
    free-tier token. 401/403 -> rejected; 429 still proves the key routed
    (auth runs before rate limiting).
    """
    if not _has_creds('blackbox'):
        return False, ('no blackbox credentials - create a key per '
                       'docs.blackbox.ai and set BLACKBOX_API_KEY or '
                       'blackbox_cookies.json')
    key = (_load_jar('blackbox').get('api_key')
           or os.getenv('BLACKBOX_API_KEY', '')).strip()
    import requests as _rq
    try:
        resp = _rq.post('https://api.blackbox.ai/chat/completions',
                        headers={'Authorization': f'Bearer {key}',
                                 'Content-Type': 'application/json',
                                 'User-Agent': _UA},
                        json={'model': 'deepseek-chat',
                              'messages': [{'role': 'user',
                                            'content': 'ping'}],
                              'stream': False, 'max_tokens': 1},
                        timeout=60,
                        **_proxies_kwargs('https://api.blackbox.ai'))
    except Exception as e:  # noqa: BLE001
        return False, f'verify failed: {type(e).__name__}: {e}'
    if resp.status_code in (401, 403):
        return False, ('API key rejected - create a new key per '
                       'docs.blackbox.ai')
    if resp.status_code == 429:
        return True, 'rate limited but key accepted (HTTP 429)'
    if resp.status_code != 200:
        return True, (f'key reachable, liveness inconclusive (HTTP '
                      f'{resp.status_code}) - validated at request time')
    return True, 'API key valid (1-token completion OK)'


def refresh_t3chat() -> Tuple[bool, str]:
    """Verify the t3.chat session against the app root.

    200 = cookies alive; 429/403 = the Vercel Security Checkpoint (cookies
    stale or the egress IP is blocklisted — redo the browser login).
    """
    if not _has_creds('t3chat'):
        return False, ('no t3chat cookies - browser login at www.t3.chat '
                       'then save them to t3chat_cookies.json or set '
                       'T3CHAT_COOKIES')
    jar = _load_jar('t3chat')
    cookies = {k: v for k, v in jar.items()
               if k not in ('api_key', 'email')}
    try:
        resp = _http_get('https://www.t3.chat/', cookies,
                         headers={'Accept': 'text/html'})
    except Exception as e:  # noqa: BLE001
        return False, f'verify failed: {type(e).__name__}: {e}'
    if resp.status_code in (429, 403):
        return False, (f'Vercel Security Checkpoint (HTTP '
                       f'{resp.status_code}) - cookies stale or egress '
                       'blocklisted; redo the browser login')
    if resp.status_code != 200:
        return False, (f'session rejected (HTTP {resp.status_code}) - '
                       'redo the browser login')
    return True, 'session cookies valid'


def refresh_innerai() -> Tuple[bool, str]:
    """Presence check + honest endpoint gap report (host resets raw TLS).

    inner.ai is login-only and connection-resets every non-browser egress
    path, so the cookies come from the shared browser and the chat
    endpoint must be RE'd from the authenticated session before the
    provider goes live (I4F_INNERAI_CHAT_URL).
    """
    if not _has_creds('innerai'):
        return False, ('no inner.ai cookies - browser login at inner.ai '
                       '(raw TLS is connection-reset from this egress, the '
                       'shared browser is the only path) then save them to '
                       'innerai_cookies.json or set INNERAI_COOKIES')
    if not (os.getenv('I4F_INNERAI_CHAT_URL', '') or '').strip():
        return True, ('cookies present; chat endpoint still not '
                      'reverse-engineered (set I4F_INNERAI_CHAT_URL) - '
                      'provider stays dormant')
    jar = _load_jar('innerai')
    cookies = {k: v for k, v in jar.items()
               if k not in ('api_key', 'email')}
    try:
        resp = _http_get('https://inner.ai/', cookies, timeout=20)
    except Exception as e:  # noqa: BLE001
        return True, (f'cookies present; host unreachable from this egress '
                      f'({type(e).__name__}) - session unverified')
    if resp.status_code == 200:
        return True, 'session cookies valid'
    return True, (f'session reachable, inconclusive '
                  f'(HTTP {resp.status_code})')


def refresh_adapta() -> Tuple[bool, str]:
    """Presence check + honest endpoint gap report (no public API found).

    adapta.org is a login-only workspace; cookies come from the shared
    browser and the chat endpoint must be RE'd from the authenticated
    session before the provider goes live (I4F_ADAPTA_CHAT_URL).
    """
    if not _has_creds('adapta'):
        return False, ('no adapta cookies - browser login at adapta.org '
                       'then save them to adapta_cookies.json or set '
                       'ADAPTA_COOKIES')
    if not (os.getenv('I4F_ADAPTA_CHAT_URL', '') or '').strip():
        return True, ('cookies present; chat endpoint still not '
                      'reverse-engineered (set I4F_ADAPTA_CHAT_URL) - '
                      'provider stays dormant')
    jar = _load_jar('adapta')
    cookies = {k: v for k, v in jar.items()
               if k not in ('api_key', 'email')}
    try:
        resp = _http_get('https://adapta.org/', cookies, timeout=20)
    except Exception as e:  # noqa: BLE001
        return True, (f'cookies present; host unreachable from this egress '
                      f'({type(e).__name__}) - session unverified')
    if resp.status_code == 200:
        return True, 'session cookies valid'
    return True, (f'session reachable, inconclusive '
                  f'(HTTP {resp.status_code})')


REFRESH = {'gemini': refresh_gemini, 'chatgpt': refresh_chatgpt,
           'deepseek': refresh_deepseek, 'claude': refresh_claude,
           'grok': refresh_grok, 'qwen': refresh_qwen, 'kimi': refresh_kimi,
           'mistral': refresh_mistral, 'copilot': _anonymous('copilot'),
           'perplexity': refresh_perplexity, 'glm': _anonymous('glm'),
           'duck': _anonymous('duck'),
           'pollinations': _anonymous('pollinations'),
           'huggingchat': refresh_huggingchat,
           'groq': refresh_groq,
           'cerebras': refresh_cerebras,
           'modelscope': refresh_modelscope,
           'mistral_api': refresh_mistral_api,
           'openrouter': refresh_openrouter,
           'llm7': refresh_llm7,
           'google_ai_studio': refresh_google_ai_studio,
           'cohere': refresh_cohere,
           'cloudflare': refresh_cloudflare,
          'meta': refresh_meta,
          'blackbox': refresh_blackbox,
          't3chat': refresh_t3chat,
          'innerai': refresh_innerai,
          'adapta': refresh_adapta,
           'arena': _manual_only(
               'arena',
               'session cookie required (browser login → arena_cookies.json '
               'or ARENA_COOKIES env); disposable-mail domains are '
               'blocklisted — signup needs I4F_MAIL_DOMAIN')}


# ------------------------------------------------------------------ IMAP OTP
def imap_otp(max_wait_s: int = 120, to_needle: Optional[str] = None) -> Optional[str]:
    """Poll the configured IMAP mailbox for a fresh verification code.

    ``to_needle`` restricts matches to mails addressed to that recipient —
    used by the catch-all autogen backend so unrelated codes are ignored.
    """
    host = os.getenv('I4F_MAIL_IMAP_HOST', '').strip()
    if not host:
        return None
    user = os.getenv('I4F_MAIL_IMAP_USER', '').strip()
    password = os.getenv('I4F_MAIL_IMAP_PASS', '').strip()
    port = int(os.getenv('I4F_MAIL_IMAP_PORT', '993') or 993)
    sender_needle = os.getenv('I4F_MAIL_OTP_SENDER', 'deepseek').strip().lower()
    code_re = re.compile(os.getenv('I4F_MAIL_OTP_REGEX', r'\b(\d{6})\b'))
    max_age_min = float(os.getenv('I4F_MAIL_OTP_MAX_AGE', '30') or 30)
    deadline = time.time() + max_wait_s
    while time.time() < deadline:
        try:
            box = imaplib.IMAP4_SSL(host, port, timeout=15)
            box.login(user, password)
            box.select('INBOX')
            since = date.today().strftime('%d-%b-%Y')
            typ, data = box.search(None, f'(SINCE "{since}")')
            for num in reversed((data[0] or b'').split()):
                typ, msg_data = box.fetch(num, '(RFC822)')
                if not msg_data or not msg_data[0]:
                    continue
                msg = message_from_bytes(msg_data[0][1])
                if to_needle:
                    recipients = ' '.join(str(msg.get(h, ''))
                                          for h in ('To', 'Delivered-To',
                                                    'X-Original-To')).lower()
                    if to_needle.lower() not in recipients:
                        continue
                sender = str(msg.get('From', '')).lower()
                if sender_needle and sender_needle not in sender:
                    continue
                age = _mail_age_min(msg)
                if age is not None and age > max_age_min:
                    continue
                body = _mail_body(msg)
                match = code_re.search(body)
                if match:
                    try:
                        box.logout()
                    except Exception:  # noqa: BLE001
                        pass
                    return match.group(1) or match.group(0)
            try:
                box.logout()
            except Exception:  # noqa: BLE001
                pass
        except Exception:  # noqa: BLE001
            pass
        time.sleep(6)
    return None


def _mail_age_min(msg) -> Optional[float]:
    try:
        from email.utils import parsedate_to_datetime
        dt = parsedate_to_datetime(str(msg.get('Date', '')))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - dt).total_seconds() / 60.0
    except Exception:
        return None


def _mail_body(msg) -> str:
    parts: List[str] = [str(msg.get('Subject', ''))]
    for part in msg.walk():
        ctype = part.get_content_type()
        if ctype not in ('text/plain', 'text/html'):
            continue
        payload = part.get_payload(decode=True)
        if not payload:
            continue
        charset = part.get_content_charset() or 'utf-8'
        try:
            parts.append(payload.decode(charset, errors='ignore'))
        except (LookupError, UnicodeDecodeError):
            parts.append(payload.decode('utf-8', errors='ignore'))
    return '\n'.join(parts)


# ------------------------------------------------------------ browser layer
def _signup_proxy() -> Optional[str]:
    """Egress for signup browsers.

    DeepSeek (CloudFront) blocks some datacenter/host IPs outright, so
    signups prefer an explicit ``I4F_SIGNUP_PROXY``; otherwise the ladder
    falls through to the dynamic pool and finally direct. Tor is never
    used. Returns None = direct connection.
    """
    explicit = os.getenv('I4F_SIGNUP_PROXY', '').strip()
    if explicit:
        return explicit
    return None


def _x_display_alive(number: int) -> bool:
    """True when an X server listens on display ``:number`` (delegated)."""
    from dsk.browser import _x_display_alive as _impl
    return _impl(number)


def _ensure_display() -> bool:
    """Best-effort X server for non-headless runs (delegated).

    The shared-browser module owns the display so every consumer of the
    ONE shared Chromium shares a single Xvfb."""
    from dsk.browser import ensure_display
    return ensure_display()


def _reap_dead_children() -> None:
    """Reap zombie children (delegated to the shared-browser module)."""
    from dsk.browser import reap_dead_children
    reap_dead_children()


def _kill_stale_browsers(user_data_path: Optional[str] = None,
                         port: Optional[int] = None) -> int:
    """Kill chromium processes wedged on a profile/port (delegated)."""
    from dsk.browser import kill_stale_browsers
    return kill_stale_browsers(user_data_path, port)


def _close_page(page) -> None:
    """Quit a DrissionPage browser and guarantee its Chrome dies with it.

    ``quit()`` raises on a wedged tab and leaves the Chrome it spawned alive;
    the process then owns the profile/auto-port dir forever (observed: 16
    leaked chromes during one signup run, each holding ~200 MB). After a
    failed quit the whole tree is killed by its debug port.
    """
    port = None
    for attr in ('_port', 'port'):
        try:
            port = int(getattr(page, attr, None))
            if port:
                break
        except (TypeError, ValueError):
            port = None
    try:
        page.quit()
    except Exception:  # noqa: BLE001
        pass
    if port:
        _kill_stale_browsers(None, port)


def _profile_port(profile: str) -> int:
    """Deterministic debug port for a persistent profile (delegated)."""
    from dsk.browser import profile_port
    return profile_port(profile)


def _clear_profile_lock(profile: str) -> None:
    """Remove orphaned Chrome singleton locks (delegated)."""
    from dsk.browser import clear_profile_lock
    clear_profile_lock(profile)


def _browser(proxy: Optional[str] = None, headed: bool = False,
             user_data_path: Optional[str] = None,
             local_port: Optional[int] = None):
    """Acquire a TAB of the shared Chromium (see dsk/browser.py).

    Every browser consumer is a tab of ONE Chromium process per
    (profile, proxy) key — the RAM rule (no more one-chromium-per-consumer).
    Rungs without a persistent profile pass ``fresh=True``: the manager
    wipes cookies+cache of the shared profile via CDP, reproducing the old
    ephemeral auto-port profile semantics inside one process.
    ``user_data_path`` keys a dedicated instance with a persistent profile
    (Cloudflare/Google score returning browsers far higher, and
    logins/cookies survive between attempts).
    """
    from dsk import browser as _shared
    return _shared.acquire(
        proxy=proxy, headed=headed, profile=user_data_path,
        port=local_port, fresh=(user_data_path is None))


def _ele_value(ele) -> str:
    for get in (lambda: ele.value, lambda: ele.attr('value')):
        try:
            return str(get() or '')
        except Exception:  # noqa: BLE001
            continue
    return ''


def _css_from_dp(sel: str) -> str:
    """Best-effort translation of a DrissionPage selector to plain CSS.

    ``_fill_react`` runs ``document.querySelector``, which speaks CSS only:
    DrissionPage's prefixed forms (``css:``, ``tag:``, ``@attr:value``) must
    be translated or the JS silently misses the field.
    """
    if sel.startswith('css:'):
        return sel[4:]
    if sel.startswith('tag:'):
        return sel[4:]
    if sel.startswith('@'):
        attr, _, val = sel[1:].partition(':')
        return f'[{attr}="{val}"]'
    return ''  # xpath / index selectors have no CSS equivalent


def _fill_first(page, selectors: List[str], value: str,
                timeout: float = 5.0) -> bool:
    """Fill the first matching field and VERIFY the value stuck.

    A consent overlay or a re-rendering SPA can swallow the typed
    value while ``input()`` still reports success (observed on
    dash.llm7.io: fill=True, field value empty -> the verification
    email was never sent). Read the value back and retry, so a
    silent no-op fill surfaces as a False instead of a phantom.
    """
    for sel in selectors:
        try:
            ele = page.ele(sel, timeout=timeout)
            if not ele:
                continue
            for _ in range(2):
                try:
                    ele.clear()
                    ele.input(value)
                except Exception:  # noqa: BLE001
                    continue
                if _ele_value(ele).strip() == value.strip():
                    return True
            # last resort: focus the field and type again
            try:
                ele.click()
                ele.clear()
                ele.input(value)
                if _ele_value(ele).strip() == value.strip():
                    return True
            except Exception:  # noqa: BLE001
                pass
            # React-controlled fields ignore both typing paths (observed on
            # dash.llm7.io: visible, enabled, yet typing never lands):
            # write through the native value setter so onChange fires and
            # the component state really updates.
            css = _css_from_dp(sel)
            if css:
                try:
                    if _fill_react(page, css, value) and \
                            _ele_value(ele).strip() == value.strip():
                        return True
                except Exception:  # noqa: BLE001
                    pass
            return False
        except Exception:  # noqa: BLE001
            continue
    return False


def _dismiss_consents(page, rounds: int = 3) -> None:
    """Best-effort dismissal of cookie/privacy consent overlays.

    EU CMP walls (llm7's 'privacy choices', OneTrust/Cookiebot banners)
    sit on top of the signup form and swallow the first fill: the
    rung types into a covered field, the value never lands and the
    verification email is never sent. Decline buttons come first
    (least tracking); accept buttons are the fallback that still
    clears the overlay.
    """
    texts = ['No thanks', 'No, thanks', 'Reject all', 'Rifiuta tutto',
             'Rifiuta', 'Save preferences', 'Save Preferences',
             'Accept all', 'Accept All', 'Accetta tutto', 'Accetta',
             'Accept', 'Consenti', 'Decline', 'Got it', '×']
    for _ in range(rounds):
        if not _click_any(page, texts):
            break
        time.sleep(2)


def _has_field(page, selectors: List[str], timeout: float = 5.0) -> bool:
    """Existence-only probe for a form field (no typing).

    The signup rungs probe the form BEFORE spending a disposable
    mailbox on an OAuth-only wall. Probing through _fill_first typed
    a dummy address into the live form; SPA forms re-render on
    validation and the later real fill then found no field (observed
    on dash.llm7.io). Checking without mutating keeps the cheap wall
    detection and leaves the form pristine for the real fill.
    """
    for sel in selectors:
        try:
            if page.ele(sel, timeout=timeout):
                return True
        except Exception:  # noqa: BLE001
            continue
    return False


def _fill_react(page, selector: str, value: str) -> bool:
    """Set a React-controlled input through the native value setter.

    React tracks input state in a synthetic store; a plain CDP ``input``
    (DrissionPage ``.input``) changes the DOM value without firing React's
    ``onChange``, so the component's state stays empty and its submit button
    is a no-op. Writing through ``HTMLInputElement.prototype``'s setter and
    dispatching a bubbling ``input`` event is the documented way to drive a
    controlled field. Returns True when the field ends up holding ``value``.
    """
    script = '''
const el = document.querySelector(arguments[0]);
if (!el) return false;
const proto = el.tagName === 'TEXTAREA'
  ? window.HTMLTextAreaElement.prototype : window.HTMLInputElement.prototype;
const setter = Object.getOwnPropertyDescriptor(proto, 'value').set;
setter.call(el, arguments[1]);
el.dispatchEvent(new Event('input', {bubbles: true}));
el.dispatchEvent(new Event('change', {bubbles: true}));
return el.value === arguments[1];
'''
    try:
        return bool(page.run_js(script, selector, value))
    except Exception:  # noqa: BLE001
        return False


def _body_head(page) -> str:
    """First 300 chars of the rendered body (lowercased), '' on failure."""
    try:
        return page.ele('tag:body').text[:300].lower()
    except Exception:  # noqa: BLE001 — detached/blank page
        return ''


_NET_LOG_JS = r"""
window.__i4f_log = window.__i4f_log || [];
if (!window.__i4f_hooked) {
  window.__i4f_hooked = true;
  const of = window.fetch;
  window.fetch = function(...a){
    return of.apply(this, a).then(r => {
      try { const c = r.clone();
        c.text().then(t => window.__i4f_log.push(
          [String((a[0]&&a[0].url)||a[0]), r.status, String(t).slice(0,500)])); }
      catch(e){}
      return r;
    });
  };
  const oo = XMLHttpRequest.prototype.open;
  const os = XMLHttpRequest.prototype.send;
  XMLHttpRequest.prototype.open = function(m,u){ this.__u=u; return oo.apply(this,arguments); };
  XMLHttpRequest.prototype.send = function(b){
    this.addEventListener('load', () => {
      try { window.__i4f_log.push(
        [String(this.__u), this.status, String(this.responseText).slice(0,500)]); }
      catch(e){}
    });
    return os.apply(this, arguments);
  };
}
"""


def _net_log_install(page) -> None:
    """Record fetch/XHR responses on the page (signup API verdicts)."""
    try:
        page.run_js(_NET_LOG_JS)
    except Exception:  # noqa: BLE001 — diagnostics only
        pass


def _net_log_read(page, needle: str = '', limit: int = 6) -> str:
    """Last few recorded request/response pairs, filtered by URL ``needle``."""
    try:
        log = page.run_js('return window.__i4f_log || [];') or []
    except Exception:  # noqa: BLE001 — diagnostics only
        return ''
    out = []
    for entry in log:
        try:
            url, status, body = entry[0], entry[1], entry[2]
        except Exception:  # noqa: BLE001
            continue
        if needle and needle.lower() not in str(url).lower():
            continue
        out.append(f'{status} {url} -> {str(body)[:200]}')
    return ' || '.join(out[-limit:])


def _click_any(page, targets: List[str]) -> bool:
    for t in targets:
        try:
            ele = page.ele(f'text:{t}', timeout=4)
            if ele:
                ele.click()
                return True
        except Exception:  # noqa: BLE001
            continue
    return False


def _js_click_text(page, texts: List[str]) -> bool:
    """Click the first visible element whose text matches, through JS.

    Native DrissionPage clicks have killed the container's Chrome (shared
    memory exhaustion); a JS click runs the page's own handlers without the
    CDP input path, so it is preferred for menu navigation in renewal flows.
    """
    # DrissionPage's run_js rejects a Python list argument (TypeError:
    # "type ... is not supported: <class 'list'>"), so the wanted strings are
    # serialised into the script as a JSON array literal instead of passed as
    # arguments[0] — passing the list made every call raise, and the broad
    # except turned it into a silent False for every signup menu click.
    payload = json.dumps([str(t) for t in texts])
    script = '''
const wanted = %s.map(t => String(t).trim().toLowerCase());
const INTERACTIVE = new Set(['A','BUTTON','INPUT']);
const ROLES = new Set(['button','menuitem','tab','link','checkbox']);
// Rank by interactivity: a real <button> fires the component's onClick,
// a matching <span> inside it does not — clicking the span was enough to
// report success while the form never advanced. Walk a text match up to
// its nearest interactive ancestor and click that instead.
function targetable(e) {
  return INTERACTIVE.has(e.tagName) || ROLES.has(e.getAttribute('role') || '');
}
function interactiveAncestor(e) {
  let n = e;
  for (let i = 0; n && i < 6; i++, n = n.parentElement) {
    if (targetable(n)) return n;
  }
  return null;
}
function visible(e) {
  const r = e.getBoundingClientRect();
  if (r.width <= 0 || r.height <= 0) return false;
  const s = getComputedStyle(e);
  return s.visibility !== 'hidden' && s.display !== 'none';
}
function score(e) {
  const t = (e.innerText || e.value || '').trim().toLowerCase();
  let s = 0;
  if (wanted.some(w => t === w)) s += 4;
  else if (wanted.some(w => t.includes(w))) s += 2;
  else return -1;
  if (targetable(e)) s += 3;
  if (e.type === 'submit') s += 1;
  return s;
}
const nodes = [...document.querySelectorAll(
  'a,button,div[role=button],div[role=menuitem],li,span,input')]
  .map(e => ({e: interactiveAncestor(e) || e, s: score(interactiveAncestor(e) || e)}))
  .filter(x => x.s >= 0 && visible(x.e))
  .sort((a, b) => b.s - a.s);
for (const x of nodes) { x.e.click(); return true; }
return false;
''' % payload
    try:
        return bool(page.run_js(script))
    except Exception:  # noqa: BLE001
        return False


def _select_first(page, selectors: List[str], value: str) -> bool:
    for sel in selectors:
        try:
            ele = page.ele(sel, timeout=3)
            if ele:
                ele.select.by_text(value)
                return True
        except Exception:  # noqa: BLE001
            continue
    return False


# Google's 2026 signup form has no <select> at all: Month and Gender are
# Material-Web comboboxes whose option lists live in shadow-root overlays, so
# select-by-text can never match them and the form silently refuses to
# advance (the ladder then reports the *next* step's field as missing).
_DEEP_QUERY_JS = '''
function deepAll(sel, root) {
  const out = [];
  const stack = [root];
  while (stack.length) {
    const r = stack.pop();
    let nodes = [];
    try { nodes = Array.from(r.querySelectorAll('*')); } catch (e) { continue; }
    for (const n of nodes) {
      try { if (n.matches && n.matches(sel)) out.push(n); } catch (e) {}
      if (n.shadowRoot) stack.push(n.shadowRoot);
    }
  }
  return out;
}
'''


def _combobox_pick(page, label: str, value: str) -> bool:
    """Choose an option in a combobox found by its accessible label.

    Opens the control, types the value through the native setter (so the
    framework registers it), then clicks the matching option anywhere in the
    shadow tree; falls back to arrow-key + Enter selection.
    """
    open_js = (_DEEP_QUERY_JS + '''
const want = %s.toLowerCase();
const all = deepAll('*', document);
const target = all.find(e =>
  (e.getAttribute('aria-label') || '').trim().toLowerCase() === want);
if (!target) return 'notfound';
try { target.scrollIntoView({block: 'center'}); } catch (e) {}
try { target.focus(); } catch (e) {}
try { target.click(); } catch (e) {}
const input = (target.matches && target.matches('input')) ? target
  : (target.querySelector && target.querySelector('input'));
if (input) {
  const d = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value');
  if (d && d.set) d.set.call(input, %s); else input.value = %s;
  ['input', 'change'].forEach(t =>
    input.dispatchEvent(new Event(t, {bubbles: true})));
}
return 'opened:' + target.tagName + (input ? '+input' : '');
''' % (json.dumps(label), json.dumps(value), json.dumps(value)))
    pick_js = (_DEEP_QUERY_JS + '''
const want = %s.trim().toLowerCase();
const opts = deepAll('[role=option], option, li, material-option, [role=listbox] > *',
                     document);
const text = o => (o.textContent || o.innerText || '').trim().toLowerCase();
const hit = opts.find(o => text(o) === want) || opts.find(o => text(o).includes(want));
if (!hit) return 'nooption:' + opts.length;
try { hit.click(); } catch (e) {}
return 'picked:' + hit.tagName;
''' % json.dumps(value))
    try:
        opened = page.run_js(open_js)
    except Exception as e:  # noqa: BLE001
        return False
    if not str(opened).startswith('opened'):
        return False
    time.sleep(1.2)
    try:
        picked = str(page.run_js(pick_js) or '')
    except Exception:  # noqa: BLE001
        picked = ''
    if picked.startswith('picked'):
        return True
    # Material comboboxes also accept type-and-take-the-highlighted-match.
    for key in ('ArrowDown', 'Enter'):
        try:
            page.run_js('''
const el = document.activeElement;
if (!el) return false;
const opts = {bubbles: true, cancelable: true, key: %s, code: %s, keyCode: 0};
['keydown','keypress','keyup'].forEach(t =>
  el.dispatchEvent(new KeyboardEvent(t, opts)));
return true;''' % (json.dumps(key), json.dumps(key)))
        except Exception:  # noqa: BLE001
            pass
        time.sleep(0.8)
        try:
            if str(page.run_js(pick_js) or '').startswith('picked'):
                return True
        except Exception:  # noqa: BLE001
            pass
    return False


def _wait_token(page, timeout_s: int = 150) -> Optional[str]:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            raw = page.run_js('return window.localStorage.getItem("userToken");')
            if raw:
                try:
                    return str(json.loads(raw).get('value') or '').strip() or str(raw)
                except (ValueError, AttributeError):
                    return str(raw)
        except Exception:  # noqa: BLE001
            pass
        time.sleep(3)
    return None


def _export_cookies(page, name: str, domains: Tuple[str, ...]) -> int:
    """Export browser cookies into the provider jar.

    ``page.cookies()`` hides httpOnly cookies (DrissionPage strips them), and
    the cookies that matter most — ``__Secure-next-auth.session-token`` for
    chatgpt, ``__Secure-1PSID`` for Google — ARE httpOnly. CDP
    ``Network.getAllCookies`` returns the full store including httpOnly ones;
    the JS-visible fallback is kept for engines that refuse the CDP call.
    """
    cookies: List[Any] = []
    try:
        raw = page.run_cdp('Network.getAllCookies')
        cookies = (raw or {}).get('cookies') or []
    except Exception:  # noqa: BLE001 — CDP refused; degrade to visible cookies
        try:
            cookies = page.cookies(all_domains=True) or []
        except TypeError:
            cookies = page.cookies() or []
    updates = {str(c.get('name')): str(c.get('value')) for c in cookies
               if isinstance(c, dict) and c.get('name') and c.get('value')
               and any(d in str(c.get('domain', '')) for d in domains)}
    if updates:
        _save_jar(name, updates)
    return len(updates)


def _creds(name: str) -> Tuple[str, str]:
    """Login credentials: env first, then accounts the bot created itself
    (data/accounts.json, written by the signup rungs)."""
    prefix = {'deepseek': 'DEEPSEEK', 'chatgpt': 'CHATGPT', 'gemini': 'GEMINI',
              'claude': 'CLAUDE', 'grok': 'GROK', 'mistral': 'MISTRAL',
              'qwen': 'QWEN', 'kimi': 'KIMI', 'copilot': 'COPILOT',
              'perplexity': 'PERPLEXITY', 'glm': 'GLM',
              'huggingchat': 'HUGGINGCHAT'}.get(name, name.upper())
    email = os.getenv(f'{prefix}_LOGIN_EMAIL', '').strip()
    password = os.getenv(f'{prefix}_LOGIN_PASSWORD', '').strip()
    if email and password:
        return email, password
    stored = _load_accounts().get(name) or {}
    return (stored.get('email') or email, stored.get('password') or password)


def _mail_session_for(name: str, email: str) -> Optional[Dict[str, Any]]:
    """Mailgen session for a stored account (generic _qwen_mail_session).

    Lets the login rungs of any provider feed an emailed OTP to
    ``mailgen.fetch_otp`` when the account was bot-created. Returns None
    when the login email is not the bot's mailbox (e.g. a manually
    stored account) — callers fall back to the configured IMAP mailbox."""
    stored = _load_accounts().get(name) or {}
    if stored.get('email') \
            and _norm_email(stored.get('email')) != _norm_email(email):
        return None
    ms = stored.get('mail_session')
    if isinstance(ms, dict) and ms.get('backend') and ms.get('address'):
        return dict(ms)
    if (stored.get('backend') or '').strip() and email:
        return {'backend': stored['backend'], 'address': email}
    return None


_DEEPSEEK_EMAIL_SELECTORS = ['@placeholder:email', '@placeholder:Email',
                             'css:input[type=text]', 'css:input[name=email]']
# type=email first: on claude.ai/login (and other SPA login pages) the real
# email box is input[type=email] while invisible 1x1 radio inputs share
# name=email — a name-first order filled the hidden radio and the email
# never landed, surfacing as a false "email field not found (bot wall?)".
_CHATGPT_EMAIL_SELECTORS = ['css:input[type=email]', '@placeholder:Enter your email',
                            '@placeholder:Email address', 'css:input[name=email]']
_GEMINI_EMAIL_SELECTORS = ['css:input[type=email]', '@placeholder:Email or phone']
_PASSWORD_SELECTORS = ['css:input[type=password]']


def browser_login(name: str) -> Tuple[bool, str]:
    """Headless re-login; exports fresh cookies/token into the data dir."""
    email, password = _creds(name)
    if not email or not password:
        return False, f'{name}: no login credentials configured'
    if name == 'qwen':
        # chat.qwen.ai /signin is not WAF-gated -> plain HTTP re-login,
        # no browser needed (and /signup sits behind an Aliyun slider,
        # so the API path is strictly more reliable here)
        ok, detail = _qwen_signin(email, password)
        if ok:
            return ok, detail
        if _qwen_pending(detail):
            # account exists but is unactivated: poll the signup mailbox for
            # the activation link, open it, then re-login — no operator needed
            ok_a, detail_a = _qwen_activate(email, password)
            if ok_a:
                return _qwen_signin(email, password)
            return False, f'{detail}; {detail_a}'
        return ok, detail
    if name == 'mistral':
        # Ory Kratos re-login is a pure-HTTP two-step flow — no browser
        # needed and no bot wall.
        token, detail = _mistral_kratos_login(email, password)
        if token:
            return True, f're-logged in via HTTP Kratos ({detail})'
        return False, f'kratos re-login: {detail}'
    proxy = _deepseek_egress() if name == 'deepseek' else None
    profile = ''
    if name in ('chatgpt', 'gemini'):
        # Persistent profile for the bot-walled hosts: Cloudflare/Google
        # score returning browsers far higher, and a one-time operator
        # login must survive across renewal attempts. The same profile is
        # reused by the chatgpt relay, so a login here upgrades it.
        profile = (os.getenv(f'I4F_BROWSER_PROFILE_{name.upper()}', '').strip()
                   or str(_data_dir() / 'browser' / name))
    try:
        # headed: CloudFront's WAF hard-403s headless clients but serves the
        # SPA (JS challenge -> aws-waf-token) to a windowed real Chrome
        page = _browser(proxy=proxy, headed=True,
                        user_data_path=profile or None)
    except Exception as e:  # noqa: BLE001
        return False, f'browser unavailable: {e}'
    try:
        if name == 'deepseek':
            # Root SPA first: /sign_in document GETs are CloudFront-403'd,
            # client-side routing is not.
            page.get('https://chat.deepseek.com/')
            time.sleep(6)
            _net_log_install(page)
            root_head = ((page.title or '') + ' ' + _body_head(page)).lower()
            if ('could not be satisfied' in root_head
                    or '403 error' in root_head):
                return False, (
                    f'CloudFront 403 via {proxy or "direct"} — DeepSeek '
                    f'blocks datacenter/host IPs; set I4F_SIGNUP_PROXY to '
                    f'a RESIDENTIAL proxy to unblock signup')
            if not _click_any(page, ['Log in', 'Login', '登录']):
                page.get('https://chat.deepseek.com/sign_in')
            time.sleep(4)
            if not _fill_first(page, _DEEPSEEK_EMAIL_SELECTORS, email):
                return False, 'email field not found'
            _fill_first(page, _PASSWORD_SELECTORS, password)
            _click_any(page, ['Log In', 'Log in', 'Sign In', '登录'])
            # some flows demand an emailed code
            if _fill_first(page, ['@placeholder:code', '@placeholder:Code',
                                  'css:input[name=code]'], ' ', timeout=2):
                code = imap_otp()
                if not code:
                    return False, 'login needs an email code but OTP not found'
                _fill_first(page, ['@placeholder:code', '@placeholder:Code',
                                   'css:input[name=code]'], code)
                _click_any(page, ['Log In', 'Verify', '确认', '验证'])
            token = _wait_token(page)
            _export_cookies(page, 'deepseek', ('deepseek.com',))
            if token:
                _save_deepseek_token(token)
                return True, 'userToken captured from browser session'
            return False, 'login finished but no userToken in localStorage'
        if name == 'chatgpt':
            page.get('https://chatgpt.com/auth/login')
            time.sleep(5)
            _click_any(page, ['Log in', 'Log In', 'Sign up'])
            time.sleep(4)
            if not _fill_first(page, _CHATGPT_EMAIL_SELECTORS, email):
                return False, 'email field not found (bot wall?)'
            _click_any(page, ['Continue', 'Next'])
            time.sleep(3)
            _fill_first(page, _PASSWORD_SELECTORS, password)
            _click_any(page, ['Continue', 'Log in'])
            # OpenAI demands an emailed verification code for sign-ins from
            # unrecognized devices — feed it from the account's mailbox
            # (or the configured IMAP) exactly like the deepseek flow.
            # strong selectors only for the WAIT (a stray text input on the
            # landed page must not fake a code step); the generic one is a
            # last resort when actually filling the code
            strong_code_sels = ['css:input[name=code]',
                                'css:input[inputmode=numeric]',
                                'css:input[autocomplete=one-time-code]',
                                '@placeholder:code', '@placeholder:Code']
            code_sels = strong_code_sels + ['css:input[type=text]']
            if _fill_first(page, strong_code_sels, ' ', timeout=20):
                ms = _mail_session_for('chatgpt', email)
                code = (mailgen.fetch_otp(ms, max_wait_s=180,
                                          sender_needle='openai')
                        if ms else imap_otp())
                if not code:
                    return False, (
                        'login needs an emailed code for '
                        f'{email} but no OTP is reachable — the account '
                        'mailbox is not the bot\'s (configure '
                        'I4F_MAIL_IMAP_* or CHATGPT_LOGIN_EMAIL/PASSWORD to '
                        'a bot mailbox) or log in once in the persistent '
                        'profile browser')
                _fill_first(page, code_sels, code)
                _click_any(page, ['Continue', 'Verify'])
                time.sleep(12)
            time.sleep(12)
            pre_token = _load_jar('chatgpt').get('accessToken', '')
            n = _export_cookies(page, 'chatgpt', ('chatgpt.com', 'openai.com'))
            # A login that exported only CloudFront/oai-did cookies is NOT
            # a login — require the actual session cookie.
            jar = _load_jar('chatgpt')
            if jar.get('__Secure-next-auth.session-token'):
                return True, f'{n} cookies exported (session token captured)'
            _chatgpt_restore_token(jar, pre_token)
            return False, (f'{n} cookies exported but no session token '
                           '(login incomplete or bot wall)')
        if name == 'gemini':
            # gemini (Google) — best effort, heavy anti-bot
            page.get('https://accounts.google.com/ServiceLogin')
            time.sleep(4)
            if not _fill_first(page, _GEMINI_EMAIL_SELECTORS, email):
                return False, 'google email field not found'
            _click_any(page, ['Next', 'Weiter'])
            time.sleep(4)
            _fill_first(page, _PASSWORD_SELECTORS, password)
            _click_any(page, ['Next', 'Weiter'])
            time.sleep(12)
            page.get('https://gemini.google.com/app')
            time.sleep(6)
            n = _export_cookies(page, 'gemini', ('google.com',))
            # non-1PSID google cookies are not a session — require the real
            # credential or the ladder will celebrate a failed login
            if _load_jar('gemini').get('__Secure-1PSID'):
                return True, f'{n} cookies exported (1PSID captured)'
            return False, (f'{n} cookies exported but no __Secure-1PSID '
                           '(2FA/anti-bot may block)')
        # claude / grok / kimi: their credentials are HTTP-only tokens
        # (sessionKey / sso / JWT) that no login form re-issues — nothing
        # to rotate in a browser here.
        return False, f'{name}: browser re-login not applicable'
    except Exception as e:  # noqa: BLE001
        return False, f'browser flow failed: {type(e).__name__}: {e}'
    finally:
        _close_page(page)


def _cool(proxy: Optional[str]) -> None:
    """Cooldown a blocked egress (mark_failure); safe on None/direct."""
    if not proxy:
        return
    try:
        from . import proxies as _proxies
        _proxies.mark_failure(proxy)
    except Exception:  # noqa: BLE001 — rotation is best-effort
        pass


def _ds_egress_ok(proxy: Optional[str]) -> bool:
    """Cheap CloudFront reachability probe for a signup/login egress.

    chat.deepseek.com hard-403s document GETs by IP reputation; a *browser*
    session (headed Chrome) solves the AWS WAF JS challenge when the IP is
    merely challenged rather than blocked. ``proxy=None`` probes the direct
    egress, so the ladder can prefer local traffic when the host IP passes.
    """
    try:
        from .providers.base import http_get
        r = http_get('https://chat.deepseek.com/',
                     proxies=({'http': proxy, 'https': proxy} if proxy
                              else None),
                     timeout=12)
    except Exception:  # noqa: BLE001 — treat as unusable
        return False
    if r.status_code == 200:
        return True
    # 202 + goku = AWS WAF JS challenge: the BROWSER solves it, so the exit
    # is usable. Only the hard CloudFront 403 ("could not be satisfied")
    # means the IP is blocked and the browser would fail too.
    body = (r.text[:500] or '').lower()
    return not ('could not be satisfied' in body or '403 error' in body)


def _pool_egresses(limit: int = 3, samples: int = 6) -> List[str]:
    """Up to ``limit`` DISTINCT pool exits that pass the reachability probe.

    Samples the sticky pool, cools down blocked exits, and returns only
    egresses CloudFront currently lets through."""
    try:
        from . import proxies as _proxies
        _proxies.ensure_pool()
    except Exception:  # noqa: BLE001 — direct remains the fallback
        pass
    out: List[str] = []
    seen: set = set()
    for _ in range(samples):
        if len(out) >= limit:
            break
        p = _pool_proxy()
        if not p:
            break
        if p in seen:
            _cool(p)  # sticky assignment: rotate the exit away, re-sample
            p = _pool_proxy()
            if not p or p in seen:
                break
        seen.add(p)
        if _ds_egress_ok(p):
            out.append(p)
        else:
            _cool(p)  # blocked exit: cooldown + force a fresh one
    return out


def _deepseek_egress() -> Optional[str]:
    """Best single egress for a DeepSeek browser session (login/renewal).

    Direct traffic wins whenever the host IP passes the reachability probe
    (a headed browser solves the AWS WAF challenge); free-proxy pool exits
    are only consulted when direct is hard-blocked.
    """
    explicit = _signup_proxy()
    if explicit and _ds_egress_ok(explicit):
        return explicit
    if _ds_egress_ok(None):
        return None
    egresses = _pool_egresses(limit=1, samples=4)
    return egresses[0] if egresses else None


def _pool_proxy() -> Optional[str]:
    """A random egress from the dynamic free-proxy pool (when enabled).

    ``direct_ok=False`` — the signup ladder already has its own direct
    rung, so the pool must never hand back the no-proxy sentinel.
    """
    try:
        from . import proxies as _proxies
        return _proxies.get_proxy('deepseek-signup', direct_ok=False)
    except Exception:  # pragma: no cover
        return None


def signup_deepseek() -> Tuple[bool, str]:
    """Create a fresh DeepSeek account — fully autonomous when possible.

    Credentials ladder:
      1. DEEPSEEK_LOGIN_EMAIL / DEEPSEEK_LOGIN_PASSWORD if configured;
      2. otherwise an auto-generated throwaway mailbox (dsk/mailgen.py):
         catch-all IMAP domain when I4F_MAIL_DOMAIN is set, else a mail.tm
         temp account. The verification code is read from that mailbox, so
         no human and no pre-existing account are needed.
    """
    email, password = _creds('deepseek')
    session = None
    generated = False
    if not email or not password:
        if not mailgen.autogen_enabled():
            return False, 'no DEEPSEEK_LOGIN_EMAIL/PASSWORD and mail autogen off'
        session, err = mailgen.create_email()
        if not session:
            return False, f'autogen mailbox unavailable: {err}'
        email = session['address']
        # mailbox password: the signup form needs one; tempmail.lol sessions
        # don't carry one (the inbox is token-addressed), so mint a form
        # password independent of the mailbox credentials.
        password = session.get('password') or mailgen.gen_password()
        generated = True
    # egress ladder: explicit I4F_SIGNUP_PROXY first, then up to 3 distinct
    # dynamic-pool exits that PASS the root-page reachability probe, then
    # direct (duplicates dropped). Tor is never used. Blocked exits are
    # cooled down so the next rung samples a fresh, hopefully-working IP
    # instead of the same blocked one.
    ladder: List[Optional[str]] = []
    seen: set = set()
    # direct first when the host IP passes the (cheap) reachability probe —
    # the free-proxy pool is unreliable and each dead rung costs a browser
    # startup; explicit I4F_SIGNUP_PROXY and pool exits follow, direct last
    # as the always-present fallback.
    ordered: List[Optional[str]] = []
    if _ds_egress_ok(None):
        ordered.append(None)
    if _signup_proxy():
        ordered.append(_signup_proxy())
    ordered += _pool_egresses()
    ordered.append(None)
    for p in ordered:
        if p is None or p not in seen:
            ladder.append(p)
            if p is not None:
                seen.add(p)
    for proxy in ladder or [None]:
        page = None
        try:
            # headed: the CloudFront WAF blocks headless clients outright
            # (403 "request blocked") while a windowed Chrome passes the JS
            # challenge and reaches the SPA — the direct rung is the reliable
            # one, so run it like a real browser.
            page = _browser(proxy=proxy, headed=True)
            # CloudFront 403s document GETs of /sign_up by IP reputation,
            # but the root SPA loads and routes to /sign_up CLIENT-SIDE
            # (no document request -> no WAF block). Root first, click
            # through; fall back to the direct document GET only if the
            # SPA entry point is missing.
            page.get('https://chat.deepseek.com/')
            _dismiss_consents(page)
            _net_log_install(page)  # record the send-code API verdict
            time.sleep(6)
            root_head = ((page.title or '') + ' ' + _body_head(page)).lower()
            if ('could not be satisfied' in root_head
                    or '403 error' in root_head):
                last_error = (
                    f'CloudFront 403 via {proxy or "direct"} — DeepSeek '
                    f'blocks datacenter/host IPs; set I4F_SIGNUP_PROXY to '
                    f'a RESIDENTIAL proxy to unblock signup')
                _log_history('deepseek', 'signup-blocked', last_error)
                _cool(proxy)  # blocked exit: cooldown + force a fresh one
                continue
            if not _click_any(page, ['Sign up', 'Sign Up', '注册']):
                page.get('https://chat.deepseek.com/sign_up')
            time.sleep(4)
            body_head = _body_head(page)
            if ('could not be satisfied' in body_head
                    or '403 error' in body_head):
                last_error = (
                    f'CloudFront 403 via {proxy or "direct"} — DeepSeek '
                    f'blocks datacenter/host IPs; set I4F_SIGNUP_PROXY to '
                    f'a RESIDENTIAL proxy to unblock signup')
                _log_history('deepseek', 'signup-blocked', last_error)
                _cool(proxy)  # blocked exit: cooldown + force a fresh one
                continue
            if not _fill_first(page, _DEEPSEEK_EMAIL_SELECTORS, email):
                last_error = 'email field not found'
                continue
            _fill_first(page, _PASSWORD_SELECTORS, password)
            clicked_send = _click_any(page, ['Send Code', 'Send code',
                                            '获取验证码'])
            # Capture the form's reaction to the send: a visible error means
            # the request was refused (rate limit, captcha, domain rejected
            # with a UI message); a silent accept followed by no OTP means
            # the mail was delivered nowhere (domain dropped server-side).
            time.sleep(4)
            send_feedback = _body_head(page).lower()
            send_note = ''
            for needle in ('too many', 'rate limit', 'captcha', 'verify',
                           'invalid', 'exist', 'error', 'failed'):
                if needle in send_feedback:
                    send_note = f' (page feedback: {needle})'
                    break
            send_api = _net_log_read(page, needle='code')
            _log_history('deepseek', 'stage',
                         f'send-code seen ({email}) clicked={clicked_send} '
                         + (send_api or 'no code API call recorded')
                         + ' || ALL: ' + (_net_log_read(page) or 'none'))
            if 'exist' in send_feedback and generated and session:
                # shared gmail dot/plus variants may already be registered —
                # mint a fresh mailbox and resend on this rung
                fresh, _ = mailgen.create_email()
                if fresh:
                    session = fresh
                    email = fresh['address']
                    password = (fresh.get('password')
                                or mailgen.gen_password())
                    _fill_first(page, _DEEPSEEK_EMAIL_SELECTORS, email)
                    _click_any(page, ['Send Code', 'Send code', '获取验证码'])
                    time.sleep(4)
                    send_note += ' (mailbox regenerated)'
            if generated:
                code = mailgen.fetch_otp(session, max_wait_s=180)
                if not code:
                    # fall back to the plain IMAP poller (no recipient filter)
                    code = imap_otp(max_wait_s=30)
            else:
                code = imap_otp(max_wait_s=180)
            if not code:
                return False, ('signup code email not found in mailbox — '
                               'no OTP arrived (gmail + disposable backends '
                               'tried); configure I4F_MAIL_DOMAIN + '
                               'I4F_MAIL_IMAP_HOST with a catch-all inbox '
                               'for guaranteed delivery'
                               + send_note)
            if not _fill_first(page, ['@placeholder:code', '@placeholder:Code',
                                      'css:input[name=code]'], code):
                return False, 'code field not found'
            _click_any(page, ['Sign Up', 'Sign up', '注册'])
            # Harvest: the fresh session's userToken lands in localStorage
            # once the SPA logs in — quit() without capturing it would throw
            # the whole signup away. Token + credentials both persisted so
            # every later renewal re-logs in instead of re-signing-up.
            time.sleep(8)
            token = _wait_token(page, timeout_s=90)
            _export_cookies(page, 'deepseek', ('deepseek.com',))
            if token:
                _save_deepseek_token(token)
                if generated:
                    _save_account('deepseek', email, password,
                                  (session or {}).get('backend', ''))
                _log_history('deepseek', 'signup-token',
                             f'captured via {proxy or "direct"}')
                return True, (f'account created and userToken captured '
                              f'({(session or {}).get("backend", "manual")}: '
                              f'{email})')
            return False, ('signup submitted but no userToken in localStorage '
                           '(verification may still be pending)')
        except Exception as e:  # noqa: BLE001
            last_error = f'signup flow failed: {type(e).__name__}: {e}'
            _log_history('deepseek', 'egress-failed',
                         f'{proxy or "direct"}: {last_error}')
        finally:
            if page is not None:
                _close_page(page)
    return False, last_error or 'all signup egresses failed'


def signup_chatgpt() -> Tuple[bool, str]:
    """Create a fresh ChatGPT account — fully autonomous (best effort).

    Uses an auto-generated throwaway mailbox (dsk/mailgen.py) for the
    verification code. OpenAI may still show an Arkose captcha or demand
    phone verification for some IPs; those cases end the attempt with a
    clear detail string and the ladder records a normal miss. Created
    account credentials are persisted so later renewals can re-login."""
    if not mailgen.autogen_enabled():
        return False, 'mail autogen disabled (I4F_MAIL_AUTOGEN=false)'
    session, err = mailgen.create_email()
    if not session:
        return False, f'autogen mailbox unavailable: {err}'
    email = session['address']
    # token-addressed backends (emailnator, tempmail.lol) carry no mailbox
    # password: mint a form password independent of the mailbox creds.
    password = session.get('password') or mailgen.gen_password()
    try:
        page = _browser(headed=True)
    except Exception as e:  # noqa: BLE001
        return False, f'browser unavailable: {e}'
    try:
        page.get('https://chatgpt.com/auth/login')
        time.sleep(6)
        _dismiss_consents(page)
        _click_any(page, ['Reject non-essential', 'Accept all'])
        time.sleep(1)
        # hook installed after navigation: it lives on the page's window and
        # a page load wipes it
        _net_log_install(page)
        # 2026-10 probe: chatgpt.com/auth/login is a combined "Log in or
        # sign up" page — an unknown email signing in CREATES the account.
        # There is no "Sign up" link at all, so the old entry click always
        # failed with "sign-up entry not found (bot wall?)".
        if not _fill_first(page, _CHATGPT_EMAIL_SELECTORS, email):
            return False, 'email field not found (bot wall?)'
        # Confirm the code-send request fires; a no-op Continue click (empty
        # React state, hidden Arkose challenge) otherwise surfaces as the
        # misleading 'verification email not found' below.
        _click_and_fire(page, ['Continue', 'Next'], 'chatgpt.com')
        # The code email is sent by the Continue click; fetch it first, then
        # enter it — the code page renders as soon as the send lands.
        code = mailgen.fetch_otp(session, max_wait_s=240,
                                 sender_needle='openai')
        if not code:
            return False, 'verification email not found (captcha/phone wall may have blocked signup)'
        code_ele = None
        code_deadline = time.time() + 60
        while time.time() < code_deadline:
            code_ele = page.ele('css:input[inputmode=numeric]', timeout=2) \
                or page.ele('css:input[name=code]', timeout=2) \
                or page.ele('css:input[autocomplete=one-time-code]', timeout=2)
            if code_ele:
                break
            time.sleep(3)
        if not code_ele:
            return False, 'verification code field not found'
        try:
            code_ele.clear()
            code_ele.input(code)
        except Exception:  # noqa: BLE001
            _fill_first(page, ['css:input[name=code]',
                               'css:input[inputmode=numeric]',
                               'css:input[autocomplete=one-time-code]',
                               '@placeholder:code', '@placeholder:Code',
                               'css:input[type=text]'], code)
        _click_any(page, ['Continue', 'Verify'])
        time.sleep(10)
        # 2026 flow: password is optional (code-only accounts). Fill it only
        # when the page asks — the old hard requirement failed the signup.
        if page.ele('css:input[type=password]', timeout=4):
            _fill_first(page, _PASSWORD_SELECTORS, password)
            _click_any(page, ['Continue', 'Next'])
        time.sleep(8)
        _save_account('chatgpt', email, password,
                      session.get('backend', ''))
        pre_token = _load_jar('chatgpt').get('accessToken', '')
        n = _export_cookies(page, 'chatgpt', ('chatgpt.com', 'openai.com'))
        via = f'account created ({session["backend"]}: {email})'
        # require an actual session cookie — CF cookies alone are not a
        # logged-in account (the old n>0 check celebrated failed signups).
        # 2026-10: the web app sets auth-session-minimized/oai-sc instead of
        # the legacy __Secure-next-auth.session-token.
        jar = _load_jar('chatgpt')
        if jar.get('__Secure-next-auth.session-token') \
                or jar.get('auth-session-minimized') or jar.get('oai-sc'):
            return True, f'{via}, {n} cookies exported (session token captured)'
        _chatgpt_restore_token(jar, pre_token)
        return False, f'{via} but no session token captured'
    except Exception as e:  # noqa: BLE001
        return False, f'chatgpt signup failed: {type(e).__name__}: {e}'
    finally:
        _close_page(page)


def signup_gemini() -> Tuple[bool, str]:
    """Create a fresh Google account for Gemini — best effort.

    Google's anti-bot (captcha, phone verification, unusual-traffic
    checks) blocks most automated attempts; every failure surfaces as a
    ladder miss. Uses an auto-generated mailbox for the verification
    code; the account is persisted for later re-login attempts."""
    if not mailgen.autogen_enabled():
        return False, 'mail autogen disabled (I4F_MAIL_AUTOGEN=false)'
    # googlemail.com — an alias of the emailnator gmail pool — is outright
    # rejected by Google's signup, and a gmail address is treated as a
    # username claim ("That username is taken"): the existing-email branch
    # needs a NON-gmail disposable (tempmail.lol domains pass Google's
    # blocklist, verified live 2026-10).
    try:
        page = _browser(headed=True)
    except Exception as e:  # noqa: BLE001
        return False, f'browser unavailable: {e}'
    try:

        def _submit_address(mail_page, address: str) -> str:
            """Drive the Google signup form through name/birthday/address
            and request the verification email. Returns '' once the code
            send has been requested, else a failure reason (no retries
            here — the caller decides whether a fresh mailbox round makes
            sense)."""
            # 2026-10: the /signup/v2/createaccount deep link redirects to
            # the sign-in identifier page, which has no firstName field —
            # the rung used to report that as a bot wall. /SignUp lands on
            # the real form (/lifecycle/steps/signup/name); the sign-in
            # page's "Create account" → "For my personal use" menu is the
            # fallback entry.
            mail_page.get('https://accounts.google.com/SignUp')
            time.sleep(6)
            if not mail_page.ele('css:input#firstName', timeout=5):
                _js_click_text(mail_page, ['Create account'])
                time.sleep(3)
                _js_click_text(mail_page, ['For my personal use'])
                time.sleep(5)
            if not _fill_first(mail_page, ['css:input#firstName',
                                           'css:input[name=firstName]'],
                               'Alex'):
                return 'google first-name field not found (bot wall?)'
            _fill_first(mail_page, ['css:input#lastName',
                                    'css:input[name=lastName]'], 'Free')
            _click_any(mail_page, ['Next', 'Weiter'])
            time.sleep(4)
            _fill_first(mail_page, ['css:input#day',
                                    'css:input[name=day]'], '12')
            if not _select_first(mail_page, ['css:select#month'], 'June'):
                # 2026 form: no <select> — Material combobox in a shadow root
                for label in ('Month', 'Choose your birth month',
                              'Birth month'):
                    if _combobox_pick(mail_page, label, 'June'):
                        break
            _fill_first(mail_page, ['css:input#year',
                                    'css:input[name=year]'], '1994')
            if not _select_first(mail_page, ['css:select#gender'],
                                 'Rather not say'):
                for label in ("What's your gender?", 'Gender',
                              'Choose your gender'):
                    if _combobox_pick(mail_page, label, 'Rather not say'):
                        break
            _click_any(mail_page, ['Next', 'Weiter'])
            time.sleep(4)
            # 2026-10 probe: the birthday Next lands on
            # lifecycle/steps/signup/collectemailphone — one text field
            # #emailPhone; the old #userName field and the 'Use your
            # existing email' button no longer exist. A NON-gmail address
            # is verified with an emailed code ("Verify your email
            # address"); a gmail address is treated as a username claim
            # ("That username is taken"), which is why the mailbox here is
            # non-gmail.
            if not _fill_first(mail_page, ['css:input#emailPhone',
                                           'css:input[name=emailPhone]',
                                           '@placeholder:Email address',
                                           'css:input[type=email]'], address):
                return 'email-phone field not found (bot wall?)'
            _net_log_install(mail_page)
            _click_and_fire(mail_page, ['Next', 'Weiter'],
                            'accounts.google.com')
            time.sleep(4)
            body = str(mail_page.run_js(
                'return document.body.innerText.slice(0, 3000);') or '')
            if 'cannot create an account with this domain' in body.lower():
                return ('google rejected mailbox domain '
                        f'({address.rsplit("@", 1)[-1]})')
            return ''

        # Two mailbox rounds, ONE browser: Google silently drops some
        # disposable providers' mail (measured 2026-10: the tempmail.lol
        # address was accepted by the form but the inbox stayed empty), so
        # when the first backend delivers nothing the form is re-run with
        # a fresh mailbox from the NEXT backend (mail.tm/mail.gw). The
        # retry re-navigates the form — it does not respawn Chromium.
        email = ''
        password = ''
        session = None
        code = None
        last_err = ''
        for round_no in range(2):
            session, err = mailgen.create_email(
                no_gmail=True,
                exclude_backends=('_tempmail_create',) if round_no else ())
            if not session:
                last_err = f'autogen mailbox unavailable: {err}'
                continue
            email = session['address']
            # token-addressed backends (emailnator, tempmail.lol) carry no
            # mailbox password: mint a form password independent of the
            # mailbox creds.
            password = session.get('password') or mailgen.gen_password()
            ferr = _submit_address(page, email)
            if ferr:
                last_err = ferr
                if 'bot wall' in ferr or 'not found' in ferr:
                    break  # form unreachable — a new mailbox won't help
                continue  # e.g. domain rejected — next backend may pass
            code = mailgen.fetch_otp(session, max_wait_s=180
                                     if round_no == 0 else 240,
                                     sender_needle='google')
            if code:
                break
            last_err = ('google verification email not found '
                        f'({session.get("backend")}; '
                        f'inbox: {mailgen._inbox_digest(session)})')
        if not code:
            return False, (last_err
                           or 'google verification email not found')
        if not _fill_first(page, ['css:input#code', 'css:input[name=code]',
                                  'css:input[inputmode=numeric]',
                                  '@placeholder:Enter code',
                                  'css:input[type=text]'], code):
            return False, 'google code field not found'
        _click_and_fire(page, ['Next', 'Weiter'], 'accounts.google.com')
        time.sleep(4)
        body = str(page.run_js(
            'return document.body.innerText.slice(0, 3000);') or '')
        if 'wrong code' in body.lower() or 'invalid code' in body.lower():
            return False, 'google rejected the verification code'
        if not _fill_first(page, ['css:input[name=Passwd]',
                                  'css:input[type=password]'], password):
            return False, (f'google password field not found '
                           f'(after code: {body[:90]!r})')
        _fill_first(page, ['css:input[name=PasswdAgain]'], password)
        _click_and_fire(page, ['Next', 'Weiter'], 'accounts.google.com')
        time.sleep(6)
        _click_any(page, ["Yes, I'm in", 'Skip', 'Not now', 'Confirm'])
        time.sleep(3)
        _click_any(page, ['I agree'])
        time.sleep(4)
        _save_account('gemini', email, password, session.get('backend', ''))
        via = f'account created ({session["backend"]}: {email})'
        # was a google.com session established at all before the gemini
        # hop? A signed-out landing there means the session died earlier
        # (verification wall / consent dismissed into sign-in).
        post_consent = str(_body_head(page) or '').replace('\n', ' ')[:90]
        # 2026-10 measured: right after consent Google can interject
        # "before creating an account, Google needs to verify some info
        # about you" — the account IS created (and saved below), but the
        # session is withheld until the check passes. Try the email branch
        # with the SAME mailbox (a second code, same fetch_otp): if Google
        # offers "verify your email" this completes the check without any
        # extra browser; a phone-only wall falls through to the poll below
        # and the cookie harvest stays as bounded as before.
        if 'verify some info' in post_consent.lower():
            _click_any(page, ['Verify your email', 'Get a code by email',
                              'Verify by email'])
            time.sleep(4)
            code2 = mailgen.fetch_otp(session, max_wait_s=180,
                                      sender_needle='google')
            if code2 and _fill_first(
                    page, ['css:input#code', 'css:input[name=code]',
                           'css:input[inputmode=numeric]',
                           '@placeholder:Enter code',
                           'css:input[type=text]'], code2):
                _click_and_fire(page, ['Next', 'Weiter'],
                                'accounts.google.com')
                time.sleep(6)
                post_consent = str(_body_head(page) or '').replace(
                    '\n', ' ')[:90]
            else:
                # the branch didn't complete — record what the interstitial
                # actually offers (2026-10: measured as Google's phone gate
                # on flagged signups; if an email option exists under other
                # wording the next attempt's detail will name it)
                try:
                    opts = page.run_js(
                        "return Array.from(document.querySelectorAll("
                        "'button,[role=button],a,li'))"
                        ".map(e => (e.innerText || '').trim())"
                        ".filter(t => t && t.length < 60)"
                        ".slice(0, 25);") or []
                except Exception:  # noqa: BLE001
                    opts = []
                seen = list(dict.fromkeys(str(t) for t in opts))
                if seen:
                    post_consent = (post_consent + ' | options: '
                                    + ' / '.join(seen[:6]))[:220]

        def _has_psid() -> bool:
            try:
                cookies = page.cookies(all_domains=True) or []
            except TypeError:
                cookies = page.cookies() or []
            except Exception:  # noqa: BLE001
                return False
            return any(c.get('name') == '__Secure-1PSID' for c in cookies)

        # Google interleaves post-signup interstitials before gemini.google.com
        # sets the session cookie (recovery-email prompt, Gemini Apps ToS,
        # welcome cards) — dismiss whatever appears and poll for the cookie
        # instead of betting everything on one fixed sleep.
        deadline = time.time() + 120
        while time.time() < deadline:
            page.get('https://gemini.google.com/app')
            _dismiss_consents(page)
            for _ in range(8):
                time.sleep(5)
                if _has_psid():
                    break
                _click_any(page, ['I agree', 'Accept all', 'Got it',
                                  'Continue', 'Not now', 'Skip',
                                  "Yes, I'm in", 'Yes, continue'])
            if _has_psid():
                break
        n = _export_cookies(page, 'gemini', ('google.com',))
        # require the real session credential (n>0 counted any google.com
        # cookie and celebrated failed signups)
        if _load_jar('gemini').get('__Secure-1PSID'):
            return True, f'{via}, {n} cookies exported (1PSID captured)'
        head = str(_body_head(page) or '').replace('\n', ' ')[:120]
        return False, (f'{via} but no __Secure-1PSID captured '
                       f'(post-consent: {post_consent!r}; '
                       f'page: {head or "blank"})')
    except Exception as e:  # noqa: BLE001
        return False, f'gemini signup failed: {type(e).__name__}: {e}'
    finally:
        _close_page(page)


# ---------------------------------------------------------------- token signups
# claude.ai / grok.com / kimi.com / chat.mistral.ai all expose free email+password
# signup forms; the resulting session lands as an HTTP-only cookie or local JWT
# that the refresher exports automatically — no human, no cookie export.


def _claude_session_key(page) -> str:
    """Return the claude.ai sessionKey cookie value, '' when absent."""
    cookies = []
    try:
        cookies = page.cookies(all_domains=True) or []
    except TypeError:
        cookies = page.cookies() or []
    except Exception:  # noqa: BLE001
        return ''
    for c in cookies:
        if 'claude.ai' in str(c.get('domain', '')) and c.get('name') == 'sessionKey':
            return str(c.get('value') or '')
    return ''


def _click_and_fire(page, texts: List[str], needle: str,
                    attempts: int = 3) -> bool:
    """Click a button and confirm it fired a network request.

    SPA submit buttons silently no-op when the framework's state is empty
    (React controlled inputs, disabled buttons), so the magic link / code
    email is never sent and the mail poll below times out with a
    misleading 'email not found'. The fetch/XHR log makes the difference
    observable: retry the click until a new request matching ``needle``
    appears, and log the recorded requests when none ever does.
    """
    def _count() -> int:
        try:
            log = page.run_js('return window.__i4f_log || [];') or []
        except Exception:  # noqa: BLE001
            return -1
        return len([e for e in log
                    if needle.lower() in str(e[0]).lower()])

    base = _count()
    for attempt in range(attempts):
        _js_click_text(page, texts)
        deadline = time.time() + 8
        while time.time() < deadline:
            time.sleep(2)
            now = _count()
            if now > base:
                return True
        base = now
        logger.debug('send click %r (%s) attempt %d fired no request',
                     texts[0], needle, attempt + 1)
    logger.info('send click %r (%s) never fired a request: %s',
                texts[0], needle, _net_log_read(page, needle, 8))
    return False


def signup_claude() -> Tuple[bool, str]:
    """Create a fresh claude.ai account and harvest the sessionKey cookie.

    2026-10: claude.ai signup is a magic-link flow, not password+OTP. The
    login SPA takes an email; "Continue with email" sends a one-time
    ``https://claude.ai/magic-link#token`` URL, and opening that link in the
    SAME browser session logs the new account in and sets the sessionKey
    cookie. There is no password field and no numeric code, so the old
    password+OTP rung could never complete. claude.ai/signup is a dead end.
    """
    if not mailgen.autogen_enabled():
        return False, 'mail autogen disabled (I4F_MAIL_AUTOGEN=false)'
    session, err = mailgen.create_email()
    if not session:
        return False, f'autogen mailbox unavailable: {err}'
    email = session['address']
    password = session.get('password') or mailgen.gen_password()
    page = None
    try:
        page = _browser(headed=True)
        # Step 1: fill the email box. The SPA renders it late and
        # intermittently, so poll across re-navigations; drive it through the
        # React native setter so the component registers the value — a plain
        # .input changes the DOM but leaves React state empty, so the
        # "Continue with email" button fires with an empty email and the
        # magic link is never sent.
        email_filled = False
        deadline = time.time() + 90
        while time.time() < deadline and not email_filled:
            page.get('https://claude.ai/login')
            time.sleep(5)
            _dismiss_consents(page)
            _click_any(page, ['Reject all cookies', 'Accept all cookies'])
            time.sleep(1)
            # hook re-installed after every navigation: it lives on the
            # page's window and a page load wipes it
            _net_log_install(page)
            sub_deadline = time.time() + 20
            while time.time() < sub_deadline:
                if _fill_react(page, 'input[type=email]', email):
                    email_filled = True
                    break
                time.sleep(2)
        if not email_filled:
            return False, 'email field not found (bot wall?)'
        # Step 2: press "Continue with email" (JS click — native clicks crash
        # the container Chrome) and record the send time for the inbox cut.
        # Confirm the send request actually fired: the button no-ops when
        # React state is empty, and polling mail for a link that was never
        # sent reads as a bot wall that is really a dead click.
        _click_and_fire(page, ['Continue with email', 'Continue'], 'claude.ai')
        sent_ts = time.time()
        # Step 3: the magic link is emailed; poll the shared inbox for it.
        # The send response advertises a numeric fallback code
        # (fallback_code_configuration, 6 digits) printed next to the link —
        # kept as a second way in when the link is not clickable in-session.
        magic, mail_body = mailgen.fetch_magic_link(
            session, url_needle='claude.ai/magic-link',
            sender_needle='', body_needle='claude.ai',
            # 240s expired seconds before emailnator's gmail forward
            # actually delivered the Anthropic mail (observed ~4-5 min
            # latency); 480s covers it without changing poll cadence.
            max_wait_s=480, after_ts=sent_ts, with_body=True)
        if not magic:
            # say WHY it likely failed: the mailbox backend (Anthropic
            # silently drops known disposable domains — the email never
            # arrives), and what the inbox actually received, so a real
            # bot wall is distinguishable from a filtered mailbox.
            domain = email.rsplit('@', 1)[-1]
            return False, ('claude magic-link email not found '
                           f'({session.get("backend")}: @{domain}; '
                           f'inbox: {mailgen._inbox_digest(session)}; '
                           'cand: '
                           f'{mailgen.debug_magic_candidates(session, "claude.ai/magic-link")})')
        # Step 4: open the magic link in the SAME session (the token is bound
        # to this browser's pendingLogin/device cookies) and poll for the
        # sessionKey the SPA sets once the exchange lands. When the link
        # exchange stalls on a challenge, the emailed 6-digit code entered
        # on the same page is the fallback.
        page.get(magic)
        session_key = ''
        code_m = re.search(r'\b(\d{6})\b', mail_body or '')
        code_filled = False
        retries = 0
        states: List[str] = []
        # The SPA exchange runs claude's browser check and a flagged
        # session lands on "couldn't verify your browser" (measured
        # 2026-10: the check failed once within 150s and the 6-digit
        # input never rendered on that error page). Give it a full 5
        # minutes of "try again" rounds — the page's own challenge widget
        # sometimes passes on a later round — and every second failure
        # re-open the magic link itself, which re-fires the exchange from
        # scratch (equivalent to "start over" while keeping the emailed
        # token). Poll for BOTH outcomes: the sessionKey, or the 6-digit
        # verify input the moment it renders.
        sub_deadline = time.time() + 300
        while time.time() < sub_deadline:
            session_key = _claude_session_key(page)
            if session_key:
                break
            head = _body_head(page)
            if head and (not states or states[-1] != head[:60]):
                states.append(head[:60])
            if 'verify' in head and 'try again' in head:
                retries += 1
                if retries % 2 == 0:
                    page.get(magic)
                    time.sleep(5)
                else:
                    _click_any(page, ['Try again', 'try again'])
            if code_m and not code_filled and _fill_first(
                    page, ['css:input[inputmode=numeric]',
                           'css:input[name=code]', '@placeholder:code',
                           'css:input[autocomplete=one-time-code]',
                           'css:input[type=text]'], code_m.group(1)):
                code_filled = True
                _click_any(page, ['Continue', 'Verify'])
            time.sleep(4)
        if session_key:
            _save_jar('claude', {'sessionKey': session_key})
            _save_account('claude', email, password, session.get('backend', ''))
            return True, (f'account created, sessionKey harvested '
                          f'({session.get("backend")}: {email})')
        mail_code = bool(re.search(r'\b\d{6}\b', mail_body or ''))
        try:
            final_url = str(page.url)
        except Exception:  # noqa: BLE001
            final_url = ''
        return False, ('signup finished but no sessionKey cookie captured '
                       f'(magic: {str(magic)[:90]}; final: {final_url} | '
                       f'{_body_head(page)[:150] or "blank"}; '
                       f'mail 6-digit code: {mail_code} filled: {code_filled} '
                       f'retries: {retries}; '
                       f'states: {" -> ".join(states[-3:]) or "-"} )')
    except Exception as e:  # noqa: BLE001
        return False, f'claude signup failed: {type(e).__name__}: {e}'
    finally:
        if page is not None:
            _close_page(page)


def signup_grok() -> Tuple[bool, str]:
    """Create a grok.com account and export sso.

    2026-10 probe: accounts.x.ai/sign-up is the SpaceXAI *API* portal —
    OAuth-only buttons (X / email / Apple / Google / GitHub), no plain email
    field, and grok.com web needs the X OAuth ``sso`` cookie. The rung
    therefore checks the wall before spending a disposable mailbox, and
    tells the operator to paste GROK_SSO / grok_cookies.json manually.
    """
    if not mailgen.autogen_enabled():
        return False, 'mail autogen disabled (I4F_MAIL_AUTOGEN=false)'
    page = None
    try:
        page = _browser(headed=True)
        page.get('https://accounts.x.ai/sign-up')
        time.sleep(6)
        _dismiss_consents(page)
        if not _has_field(page, _CHATGPT_EMAIL_SELECTORS):
            _click_any(page, ['Sign up', 'Create account', 'Sign in'])
            time.sleep(4)
            if not _has_field(page, _CHATGPT_EMAIL_SELECTORS):
                return False, ('grok signup is OAuth-only (X account required) — '
                               'set GROK_SSO or grok_cookies.json manually')
        session, err = mailgen.create_email()
        if not session:
            return False, f'autogen mailbox unavailable: {err}'
        email = session['address']
        password = session.get('password') or mailgen.gen_password()
        if not _fill_first(page, _CHATGPT_EMAIL_SELECTORS, email):
            return False, 'email field not found'
        _click_any(page, ['Continue', 'Next'])
        time.sleep(3)
        _fill_first(page, _PASSWORD_SELECTORS, password)
        _click_any(page, ['Continue', 'Sign up'])
        time.sleep(10)
        code = mailgen.fetch_otp(session, max_wait_s=240, sender_needle='x.ai')
        if not code:
            code = mailgen.fetch_otp(session, max_wait_s=60, sender_needle='')
        if not code:
            return False, 'grok verification email not found'
        if not _fill_first(page, ['css:input[name=code]', '@placeholder:code',
                                  'css:input[inputmode=numeric]',
                                  'css:input[type=text]'], code):
            return False, 'code field not found'
        _click_any(page, ['Verify', 'Continue'])
        time.sleep(12)
        page.get('https://grok.com/')
        time.sleep(6)
        jar_cookies = {}
        try:
            for c in (page.cookies(all_domains=True) or []):
                if 'grok.com' in str(c.get('domain', '')):
                    jar_cookies[c['name']] = c['value']
        except TypeError:
            for c in (page.cookies() or []):
                if 'grok.com' in str(c.get('domain', '')):
                    jar_cookies[c['name']] = c['value']
        sso = jar_cookies.get('sso') or jar_cookies.get('sso-rw') or ''
        if sso:
            _save_jar('grok', {'sso': sso, 'sso-rw': sso})
            _save_account('grok', email, password, session.get('backend', ''))
            return True, (f'account created, sso exported '
                          f'({session.get("backend")}: {email})')
        return False, 'signup finished but no sso cookie captured'
    except Exception as e:  # noqa: BLE001
        return False, f'grok signup failed: {type(e).__name__}: {e}'
    finally:
        if page is not None:
            _close_page(page)


def signup_kimi() -> Tuple[bool, str]:
    """Create a kimi.com account and save the JWT.

    2026-10 probe: www.kimi.com/login offers WeChat QR, a phone number
    (+86 only) and enterprise SSO — no email/password signup at all, so
    this rung can only work for an operator-supplied account. Probe the
    wall before spending a disposable mailbox.
    """
    if not mailgen.autogen_enabled():
        return False, 'mail autogen disabled (I4F_MAIL_AUTOGEN=false)'
    page = None
    try:
        page = _browser(headed=True)
        page.get('https://www.kimi.com/')
        time.sleep(6)
        _dismiss_consents(page)
        if not _click_any(page, ['Sign up', 'Sign Up', '注册', 'Log in', '登录']):
            return False, 'kimi auth entry not found'
        time.sleep(4)
        # prefer email/password over phone (no phone wall for email)
        _click_any(page, ['Email', '邮箱', 'Password login', '密码登录'])
        time.sleep(2)
        if not _has_field(page, _CHATGPT_EMAIL_SELECTORS):
            return False, ('kimi signup is phone(+86)/WeChat/SSO-only — '
                           'set KIMI_TOKEN or kimi_cookies.json manually')
        session, err = mailgen.create_email()
        if not session:
            return False, f'autogen mailbox unavailable: {err}'
        email = session['address']
        password = session.get('password') or mailgen.gen_password()
        if not _fill_first(page, _CHATGPT_EMAIL_SELECTORS, email):
            return False, 'email field not found'
        _fill_first(page, _PASSWORD_SELECTORS, password)
        _click_any(page, ['Sign up', 'Sign Up', '注册', 'Continue'])
        time.sleep(10)
        code = mailgen.fetch_otp(session, max_wait_s=240, sender_needle='kimi')
        if not code:
            code = mailgen.fetch_otp(session, max_wait_s=60, sender_needle='')
        if not code:
            return False, 'kimi verification email not found'
        if not _fill_first(page, ['css:input[name=code]', '@placeholder:code',
                                  'css:input[inputmode=numeric]',
                                  'css:input[type=text]'], code):
            return False, 'code field not found'
        _click_any(page, ['Verify', 'Continue', '确认'])
        time.sleep(12)
        token = ''
        try:
            token = str(page.run_js(
                'let hit="";'
                'for(let i=0;i<localStorage.length;i++){'
                'const k=localStorage.key(i);const v=localStorage.getItem(k);'
                'if(v&&v.length>40&&/eyJ[A-Za-z0-9_-]/.test(v)){hit=v;break;}}'
                'return hit;') or '')
        except Exception:  # noqa: BLE001
            pass
        if token:
            try:
                token = str(json.loads(token).get('value') or token)
            except (ValueError, AttributeError):
                pass
            _save_jar('kimi', {'token': token, 'email': email})
            _save_account('kimi', email, password, session.get('backend', ''))
            return True, (f'account created, JWT saved '
                          f'({session.get("backend")}: {email})')
        return False, 'signup finished but no JWT found in localStorage'
    except Exception as e:  # noqa: BLE001
        return False, f'kimi signup failed: {type(e).__name__}: {e}'
    finally:
        if page is not None:
            _close_page(page)


def signup_huggingchat() -> Tuple[bool, str]:
    """Create a huggingface.co account and save the chat session cookies.

    Flow (browser): /join form (email + password + username) → HF emails a
    verification link → open the link in the same session → the logged-in
    cookie set (hf-chat + HF session) is dumped into the huggingchat jar.
    """
    if not mailgen.autogen_enabled():
        return False, 'mail autogen disabled (I4F_MAIL_AUTOGEN=false)'
    page = None
    try:
        page = _browser(headed=True)
        page.get('https://huggingface.co/join')
        time.sleep(6)
        _dismiss_consents(page)
        if not _has_field(page, _CHATGPT_EMAIL_SELECTORS):
            return False, 'huggingface join form not found (email field)'
        session, err = mailgen.create_email()
        if not session:
            return False, f'autogen mailbox unavailable: {err}'
        email = session['address']
        password = session.get('password') or mailgen.gen_password()
        username = (email.split('@')[0] or 'user') + str(int(time.time()) % 10000)
        if not _fill_first(page, _CHATGPT_EMAIL_SELECTORS, email):
            return False, 'email field not found'
        _fill_first(page, _PASSWORD_SELECTORS, password)
        if not _fill_first(page, ['css:input[name=username]',
                                  '@placeholder:username',
                                  'css:input[id=username]'], username):
            logger.info('huggingchat: username field not found — continuing')
        _click_any(page, ['Sign Up', 'Sign up', 'Create account',
                          'Create Account', 'Continue'])
        time.sleep(10)
        # HF verifies by emailed magic link (fall back to an OTP code)
        link = mailgen.fetch_magic_link(session, url_needle='huggingface',
                                        max_wait_s=240)
        code = None
        if not link:
            code = mailgen.fetch_otp(session, max_wait_s=60,
                                     sender_needle='huggingface')
        if not link and not code:
            return False, 'huggingface verification email not found'
        if link:
            page.get(link)
            time.sleep(8)
        else:
            if not _fill_first(page, ['css:input[name=code]',
                                      'css:input[inputmode=numeric]',
                                      'css:input[type=text]'], code):
                return False, 'verification code field not found'
            _click_any(page, ['Verify', 'Continue', 'Submit'])
            time.sleep(8)
        page.get('https://huggingface.co/chat/')
        time.sleep(6)
        # CDP dump (httpOnly included — page.cookies() hides those, and the
        # HF session cookie IS httpOnly)
        saved = _export_cookies(page, 'huggingchat',
                                ('.huggingface.co', 'huggingface.co'))
        if not saved:
            return False, ('signup finished but no HF session cookies found '
                           '(captcha or email-domain blocklist likely)')
        _save_account('huggingchat', email, password,
                      session.get('backend', ''))
        return True, (f'account created, cookies saved '
                      f'({session.get("backend")}: {email})')
    except Exception as e:  # noqa: BLE001
        return False, f'huggingchat signup failed: {type(e).__name__}: {e}'
    finally:
        if page is not None:
            _close_page(page)


def signup_groq() -> Tuple[bool, str]:
    """Create a console.groq.com account and harvest a free-tier API key.

    Flow (browser): console.groq.com/login (email) -> emailed OTP code ->
    logged in -> /keys -> "Create API Key" -> the key is displayed ONCE
    in a dialog -> scrape the gsk_ value and store it in the groq jar
    under api_key (the exact key the provider reads).
    """
    if not mailgen.autogen_enabled():
        return False, 'mail autogen disabled (I4F_MAIL_AUTOGEN=false)'
    page = None
    try:
        page = _browser(headed=True)
        page.get('https://console.groq.com/login')
        time.sleep(8)
        _dismiss_consents(page)
        # probe the form before spending a disposable mailbox
        if not _has_field(page, _CHATGPT_EMAIL_SELECTORS):
            return False, ('groq login form not found (email field) - '
                           'create a key in console.groq.com/keys and set '
                           'GROQ_API_KEY manually')
        session, err = mailgen.create_email()
        if not session:
            return False, f'autogen mailbox unavailable: {err}'
        email = session['address']
        if not _fill_first(page, _CHATGPT_EMAIL_SELECTORS, email):
            return False, 'email field not found'
        _click_any(page, ['Continue', 'Sign in', 'Log in', 'Sign up'])
        time.sleep(8)
        code = mailgen.fetch_otp(session, max_wait_s=240,
                                 sender_needle='groq')
        if not code:
            code = mailgen.fetch_otp(session, max_wait_s=60,
                                     sender_needle='')
        if not code:
            return False, 'groq verification email not found'
        if not _fill_first(page, ['css:input[name=code]', '@placeholder:code',
                                  'css:input[inputmode=numeric]',
                                  'css:input[type=tel]',
                                  'css:input[type=text]'], code):
            return False, 'code field not found'
        _click_any(page, ['Verify', 'Continue', 'Submit'])
        time.sleep(10)
        page.get('https://console.groq.com/keys')
        time.sleep(6)
        if not _click_any(page, ['Create API Key', 'New API Key',
                                 'Create key', 'API Keys']):
            return False, 'create-key button not found on /keys'
        time.sleep(4)
        key = ''
        try:
            key = str(page.run_js(
                'let hit="";'
                'for (const el of document.querySelectorAll('
                '"input,code,pre,div")) {'
                'const t = el.value || el.textContent || "";'
                'const m = t.match(/gsk_[A-Za-z0-9]{20,}/);'
                'if (m) { hit = m[0]; break; } }'
                'return hit;') or '')
        except Exception:  # noqa: BLE001
            pass
        if not key:
            return False, ('signup finished but no gsk_ key found (captcha '
                           'or verification wall likely) - set GROQ_API_KEY '
                           'manually')
        _save_jar('groq', {'api_key': key, 'email': email})
        _save_account('groq', email, '', session.get('backend', ''))
        return True, (f'account created, API key saved '
                      f'({session.get("backend")}: {email})')
    except Exception as e:  # noqa: BLE001
        return False, f'groq signup failed: {type(e).__name__}: {e}'
    finally:
        if page is not None:
            _close_page(page)


def signup_cerebras() -> Tuple[bool, str]:
    """Create a cloud.cerebras.ai account and harvest a free-tier API key.

    Flow (browser): cloud.cerebras.ai sign-up (email) -> emailed OTP
    code -> logged in -> /platform/apikeys -> "Create API Key" -> the
    csk- value is displayed once -> scrape it and store it in the
    cerebras jar under api_key (the exact key the provider reads).
    """
    if not mailgen.autogen_enabled():
        return False, 'mail autogen disabled (I4F_MAIL_AUTOGEN=false)'
    page = None
    try:
        page = _browser(headed=True)
        page.get('https://cloud.cerebras.ai')
        time.sleep(8)
        _dismiss_consents(page)
        # probe the form before spending a disposable mailbox
        if not _has_field(page, _CHATGPT_EMAIL_SELECTORS):
            return False, ('cerebras login form not found (email field) - '
                           'create a key in cloud.cerebras.ai/platform/'
                           'apikeys and set CEREBRAS_API_KEY manually')
        session, err = mailgen.create_email()
        if not session:
            return False, f'autogen mailbox unavailable: {err}'
        email = session['address']
        if not _fill_first(page, _CHATGPT_EMAIL_SELECTORS, email):
            return False, 'email field not found'
        _click_any(page, ['Continue', 'Sign in', 'Log in', 'Sign up'])
        time.sleep(8)
        code = mailgen.fetch_otp(session, max_wait_s=240,
                                 sender_needle='cerebras')
        if not code:
            code = mailgen.fetch_otp(session, max_wait_s=60,
                                     sender_needle='')
        if not code:
            return False, 'cerebras verification email not found'
        if not _fill_first(page, ['css:input[name=code]', '@placeholder:code',
                                  'css:input[inputmode=numeric]',
                                  'css:input[type=tel]',
                                  'css:input[type=text]'], code):
            return False, 'code field not found'
        _click_any(page, ['Verify', 'Continue', 'Submit'])
        time.sleep(10)
        page.get('https://cloud.cerebras.ai/platform/apikeys')
        time.sleep(6)
        if not _click_any(page, ['Create API Key', 'New API Key',
                                 'Create key', 'API Keys']):
            return False, 'create-key button not found on /platform/apikeys'
        time.sleep(4)
        key = ''
        try:
            key = str(page.run_js(
                'let hit="";'
                'for (const el of document.querySelectorAll('
                '"input,code,pre,div")) {'
                'const t = el.value || el.textContent || "";'
                'const m = t.match(/csk-[A-Za-z0-9]{20,}/);'
                'if (m) { hit = m[0]; break; } }'
                'return hit;') or '')
        except Exception:  # noqa: BLE001
            pass
        if not key:
            return False, ('signup finished but no csk- key found (captcha '
                           'or verification wall likely) - set '
                           'CEREBRAS_API_KEY manually')
        _save_jar('cerebras', {'api_key': key, 'email': email})
        _save_account('cerebras', email, '', session.get('backend', ''))
        return True, (f'account created, API key saved '
                      f'({session.get("backend")}: {email})')
    except Exception as e:  # noqa: BLE001
        return False, f'cerebras signup failed: {type(e).__name__}: {e}'
    finally:
        if page is not None:
            _close_page(page)


def signup_modelscope() -> Tuple[bool, str]:
    """Create a modelscope.cn account and harvest a free SDK access token.

    Flow (browser): modelscope.cn sign-up (email) -> emailed OTP code ->
    logged in -> /my/myaccesstoken -> create an access token -> the
    ms-<uuid> value is displayed once -> scrape it and store it in the
    modelscope jar under api_key (the exact token the provider reads).
    The UI is Chinese-leaning; both English and Chinese button labels
    are probed. A phone-verification wall is probed honestly.
    """
    if not mailgen.autogen_enabled():
        return False, 'mail autogen disabled (I4F_MAIL_AUTOGEN=false)'
    page = None
    try:
        page = _browser(headed=True)
        page.get('https://modelscope.cn/signUp')
        time.sleep(8)
        _dismiss_consents(page)
        # probe the form before spending a disposable mailbox
        if not _has_field(page, _CHATGPT_EMAIL_SELECTORS):
            return False, ('modelscope signup form not found (email field, '
                           'phone wall likely) - create a token in '
                           'modelscope.cn/my/myaccesstoken and set '
                           'MODELSCOPE_API_KEY manually')
        session, err = mailgen.create_email()
        if not session:
            return False, f'autogen mailbox unavailable: {err}'
        email = session['address']
        if not _fill_first(page, _CHATGPT_EMAIL_SELECTORS, email):
            return False, 'email field not found'
        _click_any(page, ['Continue', 'Sign up', 'Sign in', 'Register',
                          '注册', '登录'])
        time.sleep(8)
        code = mailgen.fetch_otp(session, max_wait_s=240,
                                 sender_needle='modelscope')
        if not code:
            code = mailgen.fetch_otp(session, max_wait_s=60,
                                     sender_needle='')
        if not code:
            return False, 'modelscope verification email not found'
        if not _fill_first(page, ['css:input[name=code]', '@placeholder:code',
                                  'css:input[inputmode=numeric]',
                                  'css:input[type=tel]',
                                  'css:input[type=text]'], code):
            return False, 'code field not found'
        _click_any(page, ['Verify', 'Continue', 'Submit', '验证'])
        time.sleep(10)
        page.get('https://modelscope.cn/my/myaccesstoken')
        time.sleep(6)
        if not _click_any(page, ['Create Token', 'New Access Token',
                                 'Create access token', '创建令牌',
                                 '新建访问令牌', '获取SDK令牌', '访问令牌']):
            return False, ('create-token button not found on '
                           '/my/myaccesstoken')
        time.sleep(4)
        key = ''
        try:
            key = str(page.run_js(
                'let hit="";'
                'for (const el of document.querySelectorAll('
                '"input,code,pre,div,span")) {'
                'const t = el.value || el.textContent || "";'
                'const m = t.match(/ms-[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-'
                '[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}/);'
                'if (m) { hit = m[0]; break; } }'
                'return hit;') or '')
        except Exception:  # noqa: BLE001
            pass
        if not key:
            return False, ('signup finished but no ms- token found (captcha '
                           'or verification wall likely) - set '
                           'MODELSCOPE_API_KEY manually')
        _save_jar('modelscope', {'api_key': key, 'email': email})
        _save_account('modelscope', email, '', session.get('backend', ''))
        return True, (f'account created, access token saved '
                      f'({session.get("backend")}: {email})')
    except Exception as e:  # noqa: BLE001
        return False, f'modelscope signup failed: {type(e).__name__}: {e}'
    finally:
        if page is not None:
            _close_page(page)


def signup_mistral_api() -> Tuple[bool, str]:
    """Create a console.mistral.ai account and harvest a free API key.

    Flow (browser): console.mistral.ai sign-up (email) -> emailed OTP
    code -> logged in -> /api-keys -> create key -> the opaque token
    is displayed once -> scrape it and store it in the mistral_api
    jar under api_key (the exact token the provider reads). A
    phone-verification wall is probed honestly.
    """
    if not mailgen.autogen_enabled():
        return False, 'mail autogen disabled (I4F_MAIL_AUTOGEN=false)'
    page = None
    try:
        page = _browser(headed=True)
        page.get('https://console.mistral.ai/sign-up')
        time.sleep(8)
        _dismiss_consents(page)
        # probe the form before spending a disposable mailbox
        if not _has_field(page, _CHATGPT_EMAIL_SELECTORS):
            return False, ('mistral signup form not found (email '
                           'field, phone/SSO wall likely) - create a '
                           'free key in console.mistral.ai/api-keys '
                           'and set MISTRAL_API_KEY manually')
        session, err = mailgen.create_email()
        if not session:
            return False, f'autogen mailbox unavailable: {err}'
        email = session['address']
        if not _fill_first(page, _CHATGPT_EMAIL_SELECTORS, email):
            return False, 'email field not found'
        _click_any(page, ['Continue', 'Sign up', 'Sign in', 'Register',
                          '注册', '登录'])
        code = mailgen.fetch_otp(session, max_wait_s=240,
                                 sender_needle='mistral')
        if not code:
            code = mailgen.fetch_otp(session, max_wait_s=60,
                                     sender_needle='')
        if not code:
            return False, 'mistral verification email not found'
        if not _fill_first(page, ['css:input[name=code]',
                                  '@placeholder:code',
                                  'css:input[inputmode=numeric]',
                                  'css:input[type=tel]',
                                  'css:input[type=text]'], code):
            return False, 'code field not found'
        _click_any(page, ['Verify', 'Continue', 'Submit', '验证'])
        time.sleep(10)
        page.get('https://console.mistral.ai/api-keys')
        time.sleep(6)
        if not _click_any(page, ['Create new key', 'Create key',
                                 'New key', 'Create API key']):
            return False, ('create-key button not found on '
                           'console.mistral.ai/api-keys')
        time.sleep(4)
        key = ''
        try:
            key = str(page.run_js(
                'let hit="";'
                'for (const el of document.querySelectorAll('
                '"input,textarea,code,pre,div,span")) {'
                'const t = el.value || el.textContent || "";'
                'const m = t.match(/[A-Za-z0-9]{32,}/);'
                'if (m && !/example|placeholder/i.test(t)) '
                '{ hit = m[0]; break; } }'
                'return hit;') or '')
        except Exception:  # noqa: BLE001
            pass
        if not key:
            return False, ('signup finished but no API key found '
                           '(captcha or verification wall likely) - '
                           'create a key in console.mistral.ai/api-keys '
                           'and set MISTRAL_API_KEY manually')
        _save_jar('mistral_api', {'api_key': key, 'email': email})
        _save_account('mistral_api', email, '', session.get('backend', ''))
        return True, (f'account created, API key saved '
                      f'({session.get("backend")}: {email})')
    except Exception as e:  # noqa: BLE001
        return False, f'mistral_api signup failed: {type(e).__name__}: {e}'
    finally:
        if page is not None:
            _close_page(page)


def signup_openrouter() -> Tuple[bool, str]:
    """Create an openrouter.ai account and harvest a free :free key.

    Flow (browser): openrouter.ai/settings/keys -> sign-in wall probed
    for an email form (OAuth-only walls are reported honestly) ->
    email login (magic link or OTP) -> /settings/keys -> "Create
    key" -> the sk-or-v1-... token is displayed once -> scrape it
    and store it in the openrouter jar under api_key (the exact key
    the provider reads).
    """
    if not mailgen.autogen_enabled():
        return False, 'mail autogen disabled (I4F_MAIL_AUTOGEN=false)'
    page = None
    try:
        page = _browser(headed=True)
        page.get('https://openrouter.ai/settings/keys')
        time.sleep(8)
        _dismiss_consents(page)
        # probe the sign-in form before spending a disposable mailbox
        if not _has_field(page, _CHATGPT_EMAIL_SELECTORS):
            return False, ('openrouter sign-in form not found '
                           '(OAuth-only wall likely) - create a free '
                           'key at openrouter.ai/settings/keys and set '
                           'OPENROUTER_API_KEY manually')
        session, err = mailgen.create_email()
        if not session:
            return False, f'autogen mailbox unavailable: {err}'
        email = session['address']
        if not _fill_first(page, _CHATGPT_EMAIL_SELECTORS, email):
            return False, 'email field not found'
        _click_any(page, ['Continue', 'Sign in', 'Log in', 'Sign up',
                          '注册', '登录'])
        # OpenRouter verifies by emailed magic link or OTP code
        link = mailgen.fetch_magic_link(session,
                                        url_needle='openrouter',
                                        max_wait_s=240)
        code = None
        if not link:
            code = mailgen.fetch_otp(session, max_wait_s=60,
                                     sender_needle='openrouter')
        if not link and not code:
            return False, 'openrouter verification email not found'
        if link:
            page.get(link)
            time.sleep(8)
        else:
            if not _fill_first(page, ['css:input[name=code]',
                                      '@placeholder:code',
                                      'css:input[inputmode=numeric]',
                                      'css:input[type=tel]',
                                      'css:input[type=text]'], code):
                return False, 'code field not found'
            _click_any(page, ['Verify', 'Continue', 'Submit', '验证'])
        time.sleep(10)
        page.get('https://openrouter.ai/settings/keys')
        time.sleep(6)
        if not _click_any(page, ['Create key', 'Create API key',
                                 'New key', 'Create new key']):
            return False, ('create-key button not found on '
                           'openrouter.ai/settings/keys')
        time.sleep(4)
        key = ''
        try:
            key = str(page.run_js(
                'let hit="";'
                'for (const el of document.querySelectorAll('
                '"input,textarea,code,pre,div,span")) {'
                'const t = el.value || el.textContent || "";'
                'const m = t.match(/sk-or-v1-[A-Za-z0-9]{20,}/);'
                'if (m) { hit = m[0]; break; } }'
                'return hit;') or '')
        except Exception:  # noqa: BLE001
            pass
        if not key:
            return False, ('signup finished but no sk-or-v1- key '
                           'found (captcha or verification wall '
                           'likely) - create a key at '
                           'openrouter.ai/settings/keys and set '
                           'OPENROUTER_API_KEY manually')
        _save_jar('openrouter', {'api_key': key, 'email': email})
        _save_account('openrouter', email, '', session.get('backend', ''))
        return True, (f'account created, API key saved '
                      f'({session.get("backend")}: {email})')
    except Exception as e:  # noqa: BLE001
        return False, f'openrouter signup failed: {type(e).__name__}: {e}'
    finally:
        if page is not None:
            _close_page(page)


def signup_llm7() -> Tuple[bool, str]:
    """Create an llm7.io account and harvest a free API key.

    Flow (browser): dash.llm7.io -> sign-in wall probed for an email
    form (OAuth-only walls are reported honestly) -> email login
    (magic link or OTP) -> #/api-keys -> "Create key" -> the token is
    displayed -> scrape it and store it in the llm7 jar under api_key
    (the exact key the provider reads). The dashboard is an SPA, so
    the scrape is a generic token-shaped regex (>=24 chars of
    [A-Za-z0-9_-]) on line-anchored text.
    """
    if not mailgen.autogen_enabled():
        return False, 'mail autogen disabled (I4F_MAIL_AUTOGEN=false)'
    page = None
    try:
        page = _browser(headed=True)
        page.get('https://dash.llm7.io/#/api-keys')
        time.sleep(8)
        _dismiss_consents(page)
        # probe the sign-in form before spending a disposable mailbox
        if not _has_field(page, _CHATGPT_EMAIL_SELECTORS):
            return False, ('llm7 sign-in form not found (OAuth-only '
                           'wall likely) - create a free key at '
                           'dash.llm7.io/#/api-keys and set '
                           'LLM7_API_KEY manually')
        session, err = mailgen.create_email()
        if not session:
            return False, f'autogen mailbox unavailable: {err}'
        email = session['address']
        if not _fill_first(page, _CHATGPT_EMAIL_SELECTORS, email):
            return False, 'email field not found'
        _click_any(page, ['Continue with email', 'Continue', 'Sign in',
                          'Log in', 'Sign up', 'Continua con email',
                          'Continua', 'Prosegui', 'Accedi', 'Iscriviti',
                          '\u6ce8\u518c', '\u767b\u5f55'])
        # LLM7 verifies by emailed magic link or OTP code
        link = mailgen.fetch_magic_link(session,
                                        url_needle='llm7',
                                        max_wait_s=240)
        code = None
        if not link:
            code = mailgen.fetch_otp(session, max_wait_s=60,
                                     sender_needle='llm7')
        if not link and not code:
            return False, 'llm7 verification email not found'
        if link:
            page.get(link)
            time.sleep(8)
        else:
            if not _fill_first(page, ['css:input[name=code]',
                                      '@placeholder:code',
                                      'css:input[inputmode=numeric]',
                                      'css:input[type=tel]',
                                      'css:input[type=text]'], code):
                return False, 'code field not found'
            _click_any(page, ['Verify', 'Continue', 'Submit', 'Verifica',
                              'Conferma', 'Prosegui', '\u9a8c\u8bc1'])
        time.sleep(10)
        page.get('https://dash.llm7.io/#/api-keys')
        time.sleep(6)
        if not _click_any(page, ['Create key', 'Create API key',
                                 'New key', 'Create new key', 'Generate',
                                 'Crea chiave', 'Crea nuova chiave',
                                 'Crea']):
            return False, ('create-key button not found on '
                           'dash.llm7.io/#/api-keys')
        time.sleep(4)
        key = ''
        try:
            key = str(page.run_js(
                'let hit="";'
                'for (const el of document.querySelectorAll('
                '"input,textarea,code,pre,div,span")) {'
                'const t = el.value || el.textContent || "";'
                'const m = t.match(/^[A-Za-z0-9][A-Za-z0-9_-]{23,}$/m);'
                'if (m) { hit = m[0]; break; } }'
                'return hit;') or '')
        except Exception:  # noqa: BLE001
            pass
        if not key:
            return False, ('signup finished but no API key token '
                           'found (captcha or verification wall '
                           'likely) - create a key at '
                           'dash.llm7.io/#/api-keys and set '
                           'LLM7_API_KEY manually')
        _save_jar('llm7', {'api_key': key, 'email': email})
        _save_account('llm7', email, '', session.get('backend', ''))
        return True, (f'account created, API key saved '
                      f'({session.get("backend")}: {email})')
    except Exception as e:  # noqa: BLE001
        return False, f'llm7 signup failed: {type(e).__name__}: {e}'
    finally:
        if page is not None:
            _close_page(page)


def signup_google_ai_studio() -> Tuple[bool, str]:
    """Log into Google and harvest a free AI Studio API key.

    Flow (browser): aistudio.google.com/apikey -> Google sign-in wall
    (reuses GOOGLE_AI_STUDIO_LOGIN_EMAIL/PASSWORD, falling back to the
    GEMINI_LOGIN_* Google credentials; without them the rung reports
    honestly - a fresh Google account needs the gemini signup rung
    first) -> "Create API key" -> the AIzaSy... key is displayed ->
    scrape it and store it in the google_ai_studio jar under api_key
    (the exact key the provider reads).
    """
    page = None
    try:
        email = (os.getenv('GOOGLE_AI_STUDIO_LOGIN_EMAIL', '').strip()
                 or os.getenv('GEMINI_LOGIN_EMAIL', '').strip())
        password = (os.getenv('GOOGLE_AI_STUDIO_LOGIN_PASSWORD', '').strip()
                    or os.getenv('GEMINI_LOGIN_PASSWORD', '').strip())
        page = _browser(headed=True)
        page.get('https://aistudio.google.com/apikey')
        time.sleep(8)
        _dismiss_consents(page)
        # Google sign-in wall: log in when credentials are available
        if _fill_first(page, _GEMINI_EMAIL_SELECTORS,
                       email or 'probe@example.invalid'):
            if not email or not password:
                return False, ('Google sign-in required - set '
                               'GOOGLE_AI_STUDIO_LOGIN_EMAIL/PASSWORD '
                               '(or GEMINI_LOGIN_EMAIL/PASSWORD), or '
                               'create the Google account via the '
                               'gemini signup first, then create a '
                               'free key at aistudio.google.com/apikey '
                               'and set GOOGLE_AI_STUDIO_API_KEY '
                               'manually')
            _click_any(page, ['Next', 'Continue', 'Weiter'])
            time.sleep(5)
            if not _fill_first(page, _PASSWORD_SELECTORS, password):
                return False, 'google password field not found'
            _click_any(page, ['Next', 'Continue', 'Weiter'])
            # dismiss interstitials until the API-key UI responds
            deadline = time.time() + 120
            logged = False
            while time.time() < deadline:
                time.sleep(5)
                _click_any(page, ['I agree', 'Accept all', 'Got it',
                                  'Continue', 'Not now', 'Skip',
                                  "Yes, I'm in", 'Yes, continue'])
                body = str(_body_head(page) or '').lower()
                if (('api key' in body or 'get started' in body)
                        and 'sign in' not in body):
                    logged = True
                    break
            if not logged:
                page.get('https://aistudio.google.com/apikey')
                time.sleep(6)
                body = str(_body_head(page) or '').lower()
                if 'sign in' in body or not body:
                    return False, ('google login did not complete '
                                   '(anti-bot or 2FA) - create a free '
                                   'key at aistudio.google.com/apikey '
                                   'and set GOOGLE_AI_STUDIO_API_KEY '
                                   'manually')
        if not _click_any(page, ['Create API key', 'Create key',
                                 'Get API key', 'New API key']):
            return False, ('create-key button not found on '
                           'aistudio.google.com/apikey (Google login '
                           'wall likely) - create a free key at '
                           'aistudio.google.com/apikey and set '
                           'GOOGLE_AI_STUDIO_API_KEY manually')
        time.sleep(5)
        # the key lands behind a "create in new project" dialog
        _click_any(page, ['Create API key in new project', 'New project',
                          'Create API key', 'Create'])
        time.sleep(5)
        key = ''
        try:
            key = str(page.run_js(
                'let hit="";'
                'for (const el of document.querySelectorAll('
                '"input,textarea,code,pre,div,span")) {'
                'const t = el.value || el.textContent || "";'
                'const m = t.match(/AIzaSy[A-Za-z0-9_-]{30,}/);'
                'if (m) { hit = m[0]; break; } }'
                'return hit;') or '')
        except Exception:  # noqa: BLE001
            pass
        if not key:
            return False, ('signup finished but no AIzaSy key found '
                           '(login or quota wall likely) - create a '
                           'key at aistudio.google.com/apikey and set '
                           'GOOGLE_AI_STUDIO_API_KEY manually')
        _save_jar('google_ai_studio', {'api_key': key, 'email': email})
        _save_account('google_ai_studio', email, '', '')
        return True, (f'API key saved ({email or "google account"}): '
                      f'{key[:10]}...')
    except Exception as e:  # noqa: BLE001
        return False, (f'google_ai_studio signup failed: '
                       f'{type(e).__name__}: {e}')
    finally:
        if page is not None:
            _close_page(page)


def signup_cohere() -> Tuple[bool, str]:
    """Create a Cohere account and harvest a free trial API key.

    Flow (browser): dashboard.cohere.com/welcome/register -> email
    form probed (SSO-only walls are reported honestly) -> emailed
    verification code (magic-link fallback) -> dashboard.cohere.com/
    api-keys -> "Create key" -> the trial key is displayed -> scrape
    it and store it in the cohere jar under api_key (the exact key the
    provider reads). Trial keys are free, no credit card.
    """
    if not mailgen.autogen_enabled():
        return False, 'mail autogen disabled (I4F_MAIL_AUTOGEN=false)'
    page = None
    try:
        page = _browser(headed=True)
        page.get('https://dashboard.cohere.com/welcome/register')
        time.sleep(8)
        _dismiss_consents(page)
        # probe the registration form before spending a disposable mailbox
        if not _has_field(page, _CHATGPT_EMAIL_SELECTORS):
            return False, ('cohere registration form not found (SSO-'
                           'only wall likely) - create a free trial '
                           'key at dashboard.cohere.com/api-keys and '
                           'set COHERE_API_KEY manually')
        session, err = mailgen.create_email()
        if not session:
            return False, f'autogen mailbox unavailable: {err}'
        email = session['address']
        if not _fill_first(page, _CHATGPT_EMAIL_SELECTORS, email):
            return False, 'email field not found'
        _click_any(page, ['Continue', 'Sign up', 'Next', 'Register',
                          '注册', '登录'])
        # Cohere verifies by emailed code (magic-link fallback)
        code = mailgen.fetch_otp(session, max_wait_s=240,
                                 sender_needle='cohere')
        link = None
        if not code:
            link = mailgen.fetch_magic_link(session,
                                            url_needle='cohere',
                                            max_wait_s=240)
        if not code and not link:
            return False, 'cohere verification email not found'
        if code:
            if not _fill_first(page, ['css:input[name=code]',
                                      '@placeholder:code',
                                      'css:input[inputmode=numeric]',
                                      'css:input[type=tel]',
                                      'css:input[type=text]'], code):
                return False, 'code field not found'
            _click_any(page, ['Verify', 'Continue', 'Submit', '验证'])
        else:
            page.get(link)
        time.sleep(10)
        page.get('https://dashboard.cohere.com/api-keys')
        time.sleep(6)
        if not _click_any(page, ['Create key', 'Create API key',
                                 'New key', 'Generate API key']):
            return False, ('create-key button not found on '
                           'dashboard.cohere.com/api-keys')
        time.sleep(3)
        # optional key-name field in the creation dialog
        _fill_first(page, ['css:input[type=text]', '@placeholder:name',
                           '@placeholder:key name'], 'i4f-auto')
        _click_any(page, ['Generate', 'Create', 'Create key', 'Save'])
        time.sleep(4)
        key = ''
        try:
            key = str(page.run_js(
                'let hit="";'
                'for (const el of document.querySelectorAll('
                '"input,textarea")) {'
                'const t = (el.value || "").trim();'
                'const m = t.match(/^[A-Za-z0-9]{32,45}$/);'
                'if (m) { hit = m[0]; break; } }'
                'if (!hit) {'
                'for (const el of document.querySelectorAll('
                '"code,pre,div,span")) {'
                'const t = el.value || el.textContent || "";'
                'const m = t.match(/(?:^|[^A-Za-z0-9])'
                '([A-Za-z0-9]{32,45})(?:[^A-Za-z0-9]|$)/);'
                'if (m && m[1]) { hit = m[1]; break; } } }'
                'return hit;') or '')
        except Exception:  # noqa: BLE001
            pass
        if not key:
            return False, ('signup finished but no trial key found '
                           '(captcha or verification wall likely) - '
                           'create a key at '
                           'dashboard.cohere.com/api-keys and set '
                           'COHERE_API_KEY manually')
        _save_jar('cohere', {'api_key': key, 'email': email})
        _save_account('cohere', email, '', session.get('backend', ''))
        return True, (f'account created, trial key saved '
                      f'({session.get("backend")}: {email})')
    except Exception as e:  # noqa: BLE001
        return False, f'cohere signup failed: {type(e).__name__}: {e}'
    finally:
        if page is not None:
            _close_page(page)


def signup_cloudflare() -> Tuple[bool, str]:
    """Mint a Workers AI token from an existing Cloudflare account.

    Cloudflare signup is email-verified behind aggressive bot walls and
    the free tier only needs an EXISTING account, so this rung is a
    LOGIN (no mailgen): CLOUDFLARE_LOGIN_EMAIL + CLOUDFLARE_LOGIN_
    PASSWORD (or the cloudflare jar email/password) -> dash login ->
    profile/api-tokens -> custom token with Account > Workers AI >
    Edit -> scrape the 40-char token -> resolve the account id via
    GET /client/v4/accounts -> store both in the cloudflare jar.
    """
    jar0 = _load_jar('cloudflare') or {}
    email = ((os.getenv('CLOUDFLARE_LOGIN_EMAIL', '') or '').strip()
             or jar0.get('email') or '')
    password = ((os.getenv('CLOUDFLARE_LOGIN_PASSWORD', '') or '').strip()
                or jar0.get('password') or '')
    if not email or not password:
        return False, ('cloudflare signup needs an existing account - set '
                       'CLOUDFLARE_LOGIN_EMAIL + CLOUDFLARE_LOGIN_PASSWORD '
                       '(free account at dash.cloudflare.com/sign-up), then '
                       're-run; the rung mints a Workers AI token and saves '
                       'CLOUDFLARE_API_TOKEN + CLOUDFLARE_ACCOUNT_ID')
    page = None
    try:
        page = _browser(headed=True)
        page.get('https://dash.cloudflare.com/login')
        time.sleep(8)
        _dismiss_consents(page)
        if not _fill_first(page, _GEMINI_EMAIL_SELECTORS, email):
            return False, 'cloudflare login email field not found'
        _click_any(page, ['Continue', 'Next', 'Log in', 'Sign in'])
        time.sleep(5)
        if not _fill_first(page, _PASSWORD_SELECTORS, password):
            # the password step may sit behind an interstitial
            # (Turnstile / 2FA): poll the rendered body honestly
            for _ in range(24):
                time.sleep(5)
                head = _body_head(page)
                if any(k in (head or '') for k in
                       ('api tokens', 'home', 'dashboard', 'websites')):
                    break
            if not _fill_first(page, _PASSWORD_SELECTORS, password):
                return False, ('cloudflare login blocked (Turnstile/2FA '
                               'likely) - mint a token manually at '
                               'dash.cloudflare.com/profile/api-tokens '
                               '(Account > Workers AI > Edit) and set '
                               'CLOUDFLARE_API_TOKEN + '
                               'CLOUDFLARE_ACCOUNT_ID')
        _click_any(page, ['Log in', 'Sign in', 'Continue'])
        time.sleep(15)
        page.get('https://dash.cloudflare.com/profile/api-tokens')
        time.sleep(8)
        if not _click_any(page, ['Create Token', 'Create token']):
            return False, ('create-token button not found on '
                           'dash.cloudflare.com/profile/api-tokens '
                           '(login wall?) - mint a Workers AI token '
                           'manually (Account > Workers AI > Edit) and '
                           'set CLOUDFLARE_API_TOKEN + '
                           'CLOUDFLARE_ACCOUNT_ID')
        time.sleep(4)
        # custom-token template, then the guided form
        _click_any(page, ['Get started', 'Create Custom Token'])
        time.sleep(4)
        _fill_first(page, ['css:input[type=text]', '@placeholder:name',
                           '@aria-label:name'], 'i4f-workers-ai')
        # permissions: Account > Workers AI > Edit (best-effort; the
        # guided form is heavily frameworked and may need the manual
        # rung - every failure path reports the env names honestly)
        _select_first(page, ['css:select'], 'Account')
        _combobox_pick(page, 'Workers AI', 'Edit')
        _click_any(page, ['Continue to summary', 'Continue'])
        time.sleep(4)
        _click_any(page, ['Create Token', 'Create token'])
        time.sleep(6)
        token = ''
        try:
            token = str(page.run_js(
                'let hit="";'
                'for (const el of document.querySelectorAll('
                '"input,textarea")) {'
                'const t = (el.value || "").trim();'
                'const m = t.match(/^[A-Za-z0-9_-]{40}$/);'
                'if (m) { hit = m[0]; break; } }'
                'if (!hit) {'
                'for (const el of document.querySelectorAll('
                '"code,pre,div,span")) {'
                'const t = el.value || el.textContent || "";'
                'const m = t.match(/(?:^|[^A-Za-z0-9_-])'
                '([A-Za-z0-9_-]{40})(?:[^A-Za-z0-9_-]|$)/);'
                'if (m && m[1] && '
                '!/^[0-9a-f]{8}-[0-9a-f]{4}/.test(m[1])) '
                '{ hit = m[1]; break; } } }'
                'return hit;') or '')
        except Exception:  # noqa: BLE001
            pass
        if not token:
            return False, ('token created but not scraped (display wall '
                           'likely) - copy it from '
                           'dash.cloudflare.com/profile/api-tokens and set '
                           'CLOUDFLARE_API_TOKEN + CLOUDFLARE_ACCOUNT_ID')
        # resolve the account id with the fresh token
        aid = ''
        try:
            resp = _http_get('https://api.cloudflare.com/client/v4/accounts',
                             {}, headers={'Authorization': f'Bearer {token}',
                                          'Accept': 'application/json'})
            data = resp.json() if resp.status_code == 200 else {}
            found = ((data.get('result') or [])
                     if isinstance(data, dict) else [])
            if found and isinstance(found[0], dict):
                aid = str(found[0].get('id') or '')
        except Exception:  # noqa: BLE001
            aid = ''
        if not aid:
            return False, ('token scraped but account id unresolved - set '
                           'CLOUDFLARE_ACCOUNT_ID (dash right sidebar) '
                           'alongside CLOUDFLARE_API_TOKEN')
        _save_jar('cloudflare', {'api_key': token, 'account_id': aid,
                                 'email': email})
        _save_account('cloudflare', email, '', 'login')
        return True, ('Workers AI token minted and account id resolved '
                      f'({email})')
    except Exception as e:  # noqa: BLE001
        return False, f'cloudflare signup failed: {type(e).__name__}: {e}'
    finally:
        if page is not None:
            _close_page(page)


_MISTRAL_AUTH = 'https://auth.mistral.ai'


def _mistral_kratos_nodes(flow: Dict[str, Any]):
    """(action, nodes) from a Mistral custom-schema Kratos flow JSON."""
    ui = flow.get('ui') or {}
    if not isinstance(ui, dict):
        ui = {}
    return ui.get('action'), ui.get('nodes') or []


def _mistral_kratos_submit(nodes: List[Dict[str, Any]],
                           fields: Dict[str, str]) -> Dict[str, str]:
    """Build the form POST payload: hidden/inputs by name + submit method."""
    data: Dict[str, str] = {}
    for node in nodes:
        attrs = (node or {}).get('attributes') or {}
        name = attrs.get('name')
        if not name:
            continue
        if attrs.get('type') == 'submit' and name == 'method':
            data['method'] = attrs.get('value') or 'password'
            continue
        if name == 'csrf_token':
            data['csrf_token'] = attrs.get('value') or ''
        elif name in fields:
            data[name] = fields[name]
    if 'method' not in data:
        data['method'] = 'password'
    return data


def _mistral_session_token(s) -> Tuple[str, str]:
    """(value, cookie_name) of the Ory session cookie — the name is
    ``ory_session_<random>`` or ``ory_kratos_session`` depending on the
    deployment, and the whoami refresh check must replay the exact name."""
    try:
        for name, value in dict(s.cookies).items():
            if name.startswith('ory_') and len(value or '') > 40:
                return value, name
    except Exception:  # noqa: BLE001
        pass
    return '', ''


def _mistral_verify_email(email: str,
                          session: Optional[Dict[str, Any]]) -> Tuple[bool, str]:
    """Verify a mistral account's address via the Ory Kratos verification
    flow (pure HTTP, mailbox OTP). chat.mistral.ai gates model output on a
    VERIFIED address — a session over an unverified identity still hits the
    account-upsell wall (signup issues the session immediately, so this must
    be converged separately; whoami shows verifiable_addresses[].verified).

    Walks /self-service/verification/browser: submit the email to (re)send
    the code, poll the signup mailbox, submit the code. Returns (ok, detail).
    """
    from curl_cffi import requests as cffi
    s = cffi.Session(impersonate='chrome120')
    s.headers.update({'User-Agent': _UA})
    try:
        r = s.get(f'{_MISTRAL_AUTH}/self-service/verification/browser',
                  params={'return_to': 'https://chat.mistral.ai/'},
                  headers={'Accept': 'application/json'}, timeout=30)
        if r.status_code == 429:
            return False, 'verification rate-limited (429) on flow init'
        if r.status_code != 200:
            return False, f'verification flow HTTP {r.status_code}'
        flow = r.json()
        # Freshness cut: codes belong to the ADDRESS and are invalidated by
        # every new request — on the shared emailnator inbox, mails from
        # earlier requests linger and Kratos rejects their codes with
        # "invalid or has already been used". Only mails arriving AFTER our
        # request carry a live code.
        t_request: Optional[float] = None
        for _step in range(6):
            action, nodes = _mistral_kratos_nodes(flow)
            if not action:
                break
            names = [((n or {}).get('attributes') or {}).get('name')
                     for n in nodes]
            fields: Dict[str, str] = {}
            # 'code' FIRST: after the email step Kratos keeps BOTH nodes in
            # the UI ('email' for resending, 'code' for submitting) — checking
            # email first would re-send forever without ever submitting.
            if 'code' in names:
                if not session:
                    return False, 'code step reached but no signup mailbox known'
                t_request = t_request or time.time()
                code = (mailgen.fetch_otp(session, max_wait_s=240,
                                          sender_needle='mistral',
                                          after_ts=t_request)
                        or mailgen.fetch_otp(session, max_wait_s=60,
                                             sender_needle='mistral',
                                             after_ts=time.time()))
                if not code:
                    return False, 'verification code email not found'
                fields['code'] = code
            elif 'email' in names:
                # code request — the freshness clock starts right here
                t_request = time.time()
                fields['email'] = email
            if not fields:
                break  # flow finished without asking for anything more
            payload = _mistral_kratos_submit(nodes, fields)
            payload['method'] = 'code'  # verification uses the code strategy
            r = s.post(action, data=payload,
                       headers={'Accept': 'application/json'}, timeout=30)
            if r.status_code == 429:
                return False, ('verification rate-limited (429) — '
                               'backing off, next cycle retries')
            try:
                flow = r.json()
            except ValueError:
                flow = {}
            if isinstance(flow, dict) and flow.get('error'):
                err = flow['error'] or {}
                return False, ('verification blocked: '
                               f'{str(err.get("message") or err)[:120]}')
            if not flow:
                break  # redirected to return_to
            state = str(flow.get('state') or '')
            if state == 'passed_challenge':
                return True, 'address verified'
            if 'failed' in state or 'error' in state:
                return False, f'verification rejected (state={state})'
            # Kratos reports submit errors as flow-level ui.messages (not
            # node messages): a rejected code means the fetched mail was
            # stale — bump the cut so the next poll only takes newer mails
            fmsgs = [str(m.get('text') or '')
                     for m in (flow.get('ui') or {}).get('messages') or []]
            if any(('invalid' in m.lower()
                    or 'already been used' in m.lower()) for m in fmsgs):
                t_request = time.time()
        # no explicit state: accept only if the flow reports success nodes
        try:
            idn_msgs = [str(m.get('text', '')).lower()
                        for n in (flow.get('ui') or {}).get('nodes', [])
                        if isinstance(n, dict)
                        for m in (n.get('messages') if isinstance(n, dict)
                                  else []) or []]
        except Exception:  # noqa: BLE001
            idn_msgs = []
        if any('verified' in m or 'success' in m for m in idn_msgs):
            return True, 'address verified (flow messages)'
        return False, f'verification inconclusive (state={flow.get("state")})'
    except Exception as e:  # noqa: BLE001
        return False, f'kratos verification failed: {type(e).__name__}: {e}'


def _mistral_kratos_login(email: str, password: str
                          ) -> Tuple[Optional[str], str]:
    """Re-login on auth.mistral.ai (Ory Kratos, pure HTTP, no browser).

    Walks /self-service/login/browser: identifier-first, then password.
    Returns (session_token, detail)."""
    from curl_cffi import requests as cffi
    s = cffi.Session(impersonate='chrome120')
    s.headers.update({'User-Agent': _UA})
    try:
        r = s.get(f'{_MISTRAL_AUTH}/self-service/login/browser',
                 params={'return_to': 'https://chat.mistral.ai/'},
                 headers={'Accept': 'application/json'}, timeout=30)
        if r.status_code != 200:
            return None, f'login flow HTTP {r.status_code}'
        flow = r.json()
        password_sent = False
        for step in range(4):
            action, nodes = _mistral_kratos_nodes(flow)
            if not action:
                break
            names = [((n or {}).get('attributes') or {}).get('name')
                     for n in nodes]
            fields: Dict[str, str] = {}
            if 'identifier' in names:
                # the custom schema re-asks for the identifier on every step
                fields['identifier'] = email
            if 'password' in names and not password_sent:
                fields['password'] = password
                password_sent = True
            if not fields:
                break  # nothing to submit
            r = s.post(action, data=_mistral_kratos_submit(nodes, fields),
                       headers={'Accept': 'application/json'},
                       timeout=30)
            if r.status_code == 429:
                return None, 'login rate-limited (429)'
            try:
                flow = r.json()
            except ValueError:
                flow = {}
            if isinstance(flow, dict) and flow.get('error'):
                err = flow['error'] or {}
                return None, f'login blocked: {str(err.get("message") or err)[:120]}'
            if not flow:
                # redirected (302 to return_to) — the session cookie is set
                break
            if flow.get('session'):  # JSON success (no redirect follow)
                break
            if any('invalid' in str(m.get('text', '')).lower()
                   or 'credentials' in str(m.get('text', '')).lower()
                   for n in (flow.get('ui') or {}).get('nodes', [])
                   for m in (n.get('messages') if isinstance(n, dict)
                             else []) or []):
                return None, 'login rejected: invalid credentials'
        token, cookie_name = _mistral_session_token(s)
        if token:
            _save_jar('mistral', {'session_token': token,
                                  'session_cookie_name': cookie_name})
            return token, 'session token obtained'
        return None, 'login completed but no ory_* session cookie'
    except Exception as e:  # noqa: BLE001
        return None, f'kratos login failed: {type(e).__name__}: {e}'


def _mistral_kratos_signup(email: str, password: str,
                           session: Dict[str, Any]) -> Tuple[Optional[str], str]:
    """Create a chat.mistral.ai account via Ory Kratos (pure HTTP, no browser).

    Walks /self-service/registration/browser: submit identity+password (the
    session is issued immediately), then best-effort email verification
    (the code is polled from the mailgen session; an unverified session
    already serves the API).
    Returns (session_token, detail)."""
    from curl_cffi import requests as cffi
    s = cffi.Session(impersonate='chrome120')
    s.headers.update({'User-Agent': _UA})
    try:
        r = s.get(f'{_MISTRAL_AUTH}/self-service/registration/browser',
                 params={'return_to': 'https://chat.mistral.ai/'},
                 headers={'Accept': 'application/json'}, timeout=30)
        if r.status_code != 200:
            return None, f'registration flow HTTP {r.status_code}'
        flow = r.json()
        result = None
        for step in range(4):
            action, nodes = _mistral_kratos_nodes(flow)
            if not action:
                break
            fields = {'traits.email': email, 'password': password,
                      'traits.name.first': 'Alex', 'traits.name.last': 'Free'}
            if any(((n or {}).get('attributes') or {}).get('name')
                   == 'code' for n in nodes):
                code = (mailgen.fetch_otp(session, max_wait_s=240,
                                          sender_needle='mistral')
                        or mailgen.fetch_otp(session, max_wait_s=60,
                                            sender_needle=''))
                if not code:
                    return None, ('verification code email not found '
                                  '(bot wall/OTP?)')
                fields['code'] = code
            r = s.post(action, data=_mistral_kratos_submit(nodes, fields),
                       headers={'Accept': 'application/json'},
                       timeout=30)
            if r.status_code == 429:
                return None, 'signup rate-limited (429)'
            try:
                flow = r.json()
            except ValueError:
                flow = {}
            if isinstance(flow, dict) and flow.get('error'):
                err = flow['error'] or {}
                return None, f'signup blocked: {str(err.get("message") or err)[:120]}'
            if flow.get('session'):
                result = flow  # registered; session issued
                break
            if not flow:
                break  # redirected to return_to: session cookie is set
            state = str(flow.get('state') or '')
            if 'failed' in state or 'error' in state:
                return None, f'registration rejected (state={state})'
        token, cookie_name = _mistral_session_token(s)
        if not token and result:
            # exotic deployment: token only inside the JSON payload
            token = str((result.get('session') or {}).get('session_token')
                        or result.get('session_token') or '')
        if token:
            _save_jar('mistral', {'session_token': token,
                                  'session_cookie_name': cookie_name})
            # verification is REQUIRED upstream (chat.mistral.ai gates model
            # output on a verified address - an unverified session still hits
            # the account-upsell wall) but non-fatal here: refresh_mistral
            # converges it on the next cycle via the mailbox OTP.
            vok, vdetail = _mistral_verify_email(email, session)
            _log_history('mistral', 'verify',
                         vdetail if vok else f'pending: {vdetail}')
            return token, 'session token obtained'
        return None, 'signup completed but no ory_* session cookie'
    except Exception as e:  # noqa: BLE001
        return None, f'kratos signup failed: {type(e).__name__}: {e}'


def signup_mistral() -> Tuple[bool, str]:
    """Create a chat.mistral.ai account — pure HTTP first (Ory Kratos API,
    no browser, no bot wall), browser flow as the fallback. Exports the
    session under the ``session_token`` key the provider reads."""
    if not mailgen.autogen_enabled():
        return False, 'mail autogen disabled (I4F_MAIL_AUTOGEN=false)'
    session, err = mailgen.create_email()
    if not session:
        return False, f'autogen mailbox unavailable: {err}'
    email = session['address']
    password = session.get('password') or mailgen.gen_password()
    # 1) pure-HTTP Kratos registration (no browser, no wall)
    token, detail = _mistral_kratos_signup(email, password, session)
    if token:
        _save_account('mistral', email, password, session.get('backend', ''),
                      extra={'mail_session': session})
        return True, (f'account created via HTTP Kratos, session exported '
                     f'({session.get("backend")}: {email})')
    # deterministic upstream verdicts — the browser flow hits the same wall
    if 'rejected' in detail or 'code email not found' in detail:
        return False, f'kratos: {detail}'
    # 2) browser fallback (transport/flow failure only)
    page = None
    try:
        page = _browser(headed=True)
        page.get('https://auth.mistral.ai/ui/registration')
        time.sleep(6)
        _dismiss_consents(page)
        if not _fill_first(page, _CHATGPT_EMAIL_SELECTORS, email):
            return False, 'email field not found (bot wall?)'
        _fill_first(page, _PASSWORD_SELECTORS, password)
        _fill_first(page, ['css:input[name=reveal_password]',
                           'css:input[name=confirm_password]'], password)
        _click_any(page, ['Create an account', 'Sign up', 'Continue'])
        time.sleep(10)
        code = mailgen.fetch_otp(session, max_wait_s=240,
                                 sender_needle='mistral')
        if not code:
            code = mailgen.fetch_otp(session, max_wait_s=60, sender_needle='')
        if not code:
            return False, 'mistral verification email not found'
        if not _fill_first(page, ['css:input[name=code]', '@placeholder:code',
                                  'css:input[name=code*]',
                                  'css:input[inputmode=numeric]',
                                  'css:input[type=text]'], code):
            return False, 'code field not found'
        _click_any(page, ['Verify', 'Continue', 'Submit'])
        time.sleep(12)
        page.get('https://chat.mistral.ai/chat')
        time.sleep(6)
        jar_cookies = {}
        try:
            for c in (page.cookies(all_domains=True) or []):
                if 'mistral' in str(c.get('domain', '')):
                    jar_cookies[c['name']] = c['value']
        except TypeError:
            for c in (page.cookies() or []):
                if 'mistral' in str(c.get('domain', '')):
                    jar_cookies[c['name']] = c['value']
        if jar_cookies.get('ory_kratos_session'):
            # the provider reads the ``session_token`` jar key
            _save_jar('mistral', {'session_token': jar_cookies['ory_kratos_session'],
                                  'session_cookie_name': 'ory_kratos_session',
                                  'email': email})
            _save_account('mistral', email, password, session.get('backend', ''),
                          extra={'mail_session': session})
            return True, (f'account created, session cookie exported '
                          f'({session.get("backend")}: {email})')
        return False, 'signup finished but no Ory session cookie captured'
    except Exception as e:  # noqa: BLE001
        return False, f'mistral signup failed: {type(e).__name__}: {e}'
    finally:
        if page is not None:
            _close_page(page)


_QWEN_SIGNUP_URL = 'https://chat.qwen.ai/auth?action=signup'
_QWEN_NAME_SELECTORS = ['@placeholder:Full Name', 'css:input[type=text]']
_QWEN_EMAIL_SELECTORS = ['@placeholder:Email', 'css:input[type=email]']


def _largest_component(mask):
    """Bounding box + size of the largest 4-connected True region."""
    import numpy as np
    from collections import deque
    H, W = mask.shape
    lbl = np.zeros(mask.shape, dtype=int)
    best, bestn = None, 0
    cur = 0
    for y in range(H):
        for x in range(W):
            if mask[y, x] and lbl[y, x] == 0:
                cur += 1
                q = deque([(y, x)])
                lbl[y, x] = cur
                n = 0
                x0 = x1 = x
                y0 = y1 = y
                while q:
                    cy, cx = q.popleft()
                    n += 1
                    x0, x1 = min(x0, cx), max(x1, cx)
                    y0, y1 = min(y0, cy), max(y1, cy)
                    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        ny, nx = cy + dy, cx + dx
                        if (0 <= ny < H and 0 <= nx < W and mask[ny, nx]
                                and lbl[ny, nx] == 0):
                            lbl[ny, nx] = cur
                            q.append((ny, nx))
                if n > bestn:
                    bestn, best = n, (x0, x1, y0, y1, n)
    return best


def _masked_ncc(gray_b, gray_p, mask):
    """Best normalized cross-correlation of the masked piece patch over the
    background. Returns [(score, bx, by), ...] best-first."""
    import numpy as np
    ph, pw = gray_p.shape
    bh, bw = gray_b.shape
    m = mask.astype(float)
    n = m.sum()
    sp = (gray_p * m).sum()
    sp2 = (gray_p * gray_p * m).sum()
    var_p = max(sp2 - sp * sp / n, 1e-9)
    out = []
    for by in range(0, bh - ph + 1):
        for bx in range(0, bw - pw + 1):
            win = gray_b[by:by + ph, bx:bx + pw]
            sw = (win * m).sum()
            sw2 = (win * win * m).sum()
            var_w = max(sw2 - sw * sw / n, 1e-9)
            num = (win * gray_p * m).sum() - sp * sw / n
            out.append((num / np.sqrt(var_w * var_p), bx, by))
    out.sort(reverse=True)
    return out


# piece element-left as a function of handle travel D on the 300px embed
# track: left(D) = A*D^2 + B*D  (accelerating slider easing, measured)
_QWEN_TRACK = (0.0035503, 0.0769223)


def _qwen_widget_state(page) -> Optional[dict]:
    """Locate the Aliyun slider widget and dump what the solver needs.

    Returns None when no widget is on the page (not armed, or already
    solved and gone). The piece/bg <img> elements are found by displayed
    size; their src is either an inline data: URI or a static-captcha CDN
    URL — both are handled by _qwen_fetch_challenge_images. The knob has
    no DOM node (CSS-drawn); it sits at the left edge of the text-box.
    """
    js = (
        'const pick=(lo,hi,hlo,hhi)=>{'
        'for(const im of document.querySelectorAll("img")){'
        'const r=im.getBoundingClientRect();'
        'if(r.width>=lo&&r.width<=hi&&r.height>=hlo&&r.height<=hhi)'
        'return [im.src,r.x,r.y,r.width,r.height];}return null;};'
        'const bg=pick(285,315,185,215), piece=pick(45,60,100,220);'
        'const box=document.querySelector(".aliyunCaptcha-sliding-text-box");'
        'const br=box?box.getBoundingClientRect():null;'
        'const tx=document.querySelector(".aliyunCaptcha-sliding-text");'
        'return JSON.stringify({bg:bg,piece:piece,'
        'box:br?[br.x,br.y,br.width,br.height]:null,'
        'state:tx?tx.textContent:null});')
    try:
        raw = page.run_js(js)
    except Exception:  # noqa: BLE001
        return None
    if not raw:
        return None
    try:
        d = json.loads(raw)
    except Exception:  # noqa: BLE001
        return None
    if not d.get('bg') or not d.get('piece') or not d.get('box'):
        return None
    return d


def _qwen_fetch_challenge_images(state: dict) -> Tuple[bytes, bytes]:
    """Raw PNG bytes for (background, piece sprite) from widget state."""
    out = []
    for key in ('bg', 'piece'):
        src = state[key][0]
        if src.startswith('data:'):
            out.append(base64.b64decode(src.split(',', 1)[1]))
        else:
            import requests as _rq
            # public static CDN — no WAF, no cookies, egress irrelevant
            r = _rq.get(src, timeout=20, headers={'User-Agent': _UA})
            r.raise_for_status()
            out.append(r.content)
    return out[0], out[1]


def _qwen_solve_challenge(bg_bytes: bytes,
                          piece_bytes: bytes,
                          disp_w: float = 300.0) -> Optional[Tuple[float, str]]:
    """Closed-form slide target for one challenge.

    Handles both observed styles — a white ghost piece baked into the
    background (whiteness-blob detection) and a dark hole cut out of it
    (masked NCC of the piece sprite) — and arbitrates between them.
    Returns (L_star, method): L* is the piece element-left target on the
    DISPLAYED background in px, or None when the challenge is unsolvable.
    """
    try:
        import io as _io
        import numpy as np
        from PIL import Image
        bg = np.asarray(
            Image.open(_io.BytesIO(bg_bytes)).convert('RGB'), float)
        pc = np.asarray(
            Image.open(_io.BytesIO(piece_bytes)).convert('RGBA'), float)
    except Exception:  # noqa: BLE001
        return None
    A, B = _QWEN_TRACK
    alpha = pc[:, :, 3]
    pys, pxs = np.where(alpha > 128)
    if not len(pxs):
        return None
    px0, py0 = int(pxs.min()), int(pys.min())
    pw = int(pxs.max() - pxs.min() + 1)
    ph = int(pys.max() - pys.min() + 1)
    patch = pc[py0:py0 + ph, px0:px0 + pw, :3]
    pmask = alpha[py0:py0 + ph, px0:px0 + pw] > 128
    gray_p = patch.mean(axis=2)
    gray_b = bg.mean(axis=2)
    sx = disp_w / bg.shape[1]  # displayed/natural scale (300/296)
    # style 1: white ghost blob (largest bright low-saturation component)
    R, G, Bl = bg[:, :, 0], bg[:, :, 1], bg[:, :, 2]
    mx = np.maximum(np.maximum(R, G), Bl)
    mn = np.minimum(np.minimum(R, G), Bl)
    white = (mx > 150) & ((mx - mn) < 40)
    white[py0 + 3:py0 + ph - 3, px0 + 3:px0 + pw - 3] = False
    ghost = None
    try:
        comp = _largest_component(white)
        if comp and comp[4] > 200:
            x0, x1, y0, y1, _n = comp
            ghost = (x0, y0, x1 - x0 + 1, y1 - y0 + 1)
    except Exception:  # noqa: BLE001
        ghost = None
    # style 2: masked NCC of the piece sprite
    try:
        top = _masked_ncc(gray_b, gray_p, pmask)
        score, ndx, ndy = top[0]
    except Exception:  # noqa: BLE001
        return None
    if score >= 0.6:
        dx, dy, method = ndx, ndy, 'ncc'
    elif ghost and abs(ghost[2] - pw) < 14 and abs(ghost[3] - ph) < 14:
        dx, dy, method = ghost[0], ghost[1], 'ghost'
    elif score >= 0.35:
        dx, dy, method = ndx, ndy, 'ncc-weak'
    elif ghost:
        dx, dy, method = ghost[0], ghost[1], 'ghost-weak'
    else:
        return None
    return (dx * sx - px0, method)


def _qwen_drag_closed_loop(page, state: dict,
                           l_star: float) -> Optional[float]:
    """Humanized glide + vision-corrected micro-creep to the target.

    The piece <img> is read live from the DOM; each correction inverts the
    calibrated piece-left(D) easing, so the landing error converges below
    a pixel regardless of where the glide phase stopped. Returns the final
    piece element-left error in px (None if the piece could not be read).
    """
    A, B = _QWEN_TRACK
    bx = state['bg'][1]
    box = state['box']
    gx, gy = box[0] + 20, box[1] + box[3] / 2  # knob at the box's left edge
    d_est = (-B + (B * B + 4 * A * l_star) ** 0.5) / (2 * A)
    js = ('for(const im of document.querySelectorAll("img")){'
          'const r=im.getBoundingClientRect();'
          'if(r.width>=45&&r.width<=60&&r.height>=100)return r.x;}'
          'return null;')
    ac = page.actions
    # human approach: stray hovers before grabbing the knob
    ac.move(gx - random.randint(15, 35), gy + random.randint(15, 40),
            duration=.3)
    ac.move(gx - random.randint(3, 8), gy + random.randint(1, 4),
            duration=.2)
    time.sleep(random.uniform(.2, .5))
    ac.move_to((gx, gy)).hold()
    time.sleep(random.uniform(.15, .4))
    x = gx
    # glide to ~92% on a smoothstep velocity profile with jitter
    d1 = d_est * .92
    segs = random.randint(7, 10)
    for i in range(1, segs + 1):
        u = i / segs
        e = (3 * u * u - 2 * u * u * u) * d1
        nx = gx + e + random.uniform(-1, 1)
        ac.move(nx - x, random.uniform(-1.5, 1.5),
                duration=random.uniform(.04, .12))
        x = nx
        if random.random() < .25:
            time.sleep(random.uniform(.02, .09))
    time.sleep(random.uniform(.05, .2))
    # vision-corrected creep: read the piece, invert the easing, land
    err = None
    for _ in range(22):
        try:
            p = page.run_js(js)
        except Exception:  # noqa: BLE001
            p = None
        if p is None:
            break
        err = (bx + l_star) - p
        if abs(err) <= 0.8:
            break
        gain = 2 * A * (x - gx) + B
        dh = max(-12.0, min(12.0, err / gain))
        if abs(dh) < .25:
            dh = -0.25 if dh < 0 else 0.25
        x += dh
        ac.move(dh, 0, duration=random.uniform(.03, .08))
        time.sleep(random.uniform(.12, .25))
    time.sleep(random.uniform(.2, .45))
    ac.release()
    time.sleep(random.uniform(.8, 1.4))
    return err


def _qwen_rearm(page) -> None:
    """Click the slider's text area to fetch a fresh challenge (this also
    restarts the 20-minute init clock that expires with VerifyCode F014)."""
    try:
        el = page.ele('.aliyunCaptcha-sliding-text-box', timeout=3)
        if el:
            el.click(by_js=False)
            return
    except Exception:  # noqa: BLE001
        pass
    try:
        page.run_js('const e=document.querySelector('
                    '".aliyunCaptcha-sliding-text-box");if(e)e.click();')
    except Exception:  # noqa: BLE001
        pass


def _qwen_last_verify_code(page) -> str:
    """VerifyCode from the newest -verify.captcha response in the net log
    (T001 = pass; F001 = behavioral reject; F015 = wrong position; F014 =
    init expired)."""
    try:
        body = _net_log_read(page, needle='-verify.captcha')
        if body:
            last = body.split(' || ')[-1]
            raw = last.split(' -> ', 1)[1] if ' -> ' in last else ''
            return str((json.loads(raw).get('Result') or {})
                       .get('VerifyCode') or '')
    except Exception:  # noqa: BLE001
        pass
    return ''


def _qwen_slider_pass(page, attempts: int = 6) -> bool:
    """Solve Aliyun's noCaptcha slider with a closed-loop vision drag.

    chat.qwen.ai's signup POST is replayed by the WAF once the slider is
    solved. The widget's piece/background images are read straight from
    the DOM, the slide target is computed offline (_qwen_solve_challenge:
    ghost-blob / masked-NCC arbitration), and a humanized glide lands the
    piece within a pixel of the target (_qwen_drag_closed_loop). The whole
    arm-solve-drag cycle runs in seconds — well inside the challenge's
    20-minute init validity (an expired init fails with VerifyCode F014
    no matter how accurate the landing was).

    Position is never the bottleneck once landed accurately; the residual
    F001 rejects are behavioral/device-trust scoring, so each failed drag
    is followed by a re-arm (fresh challenge + fresh init clock) instead
    of grinding on one widget.
    Returns True when no widget is present (already passed / not armed) or
    a drag was accepted; False when every attempt was rejected.
    """
    for _ in range(attempts):
        state = _qwen_widget_state(page)
        if not state:
            return True  # no widget — the signup POST went through
        time.sleep(random.uniform(.4, 1.0))
        sol = None
        try:
            bg_b, piece_b = _qwen_fetch_challenge_images(state)
            sol = _qwen_solve_challenge(bg_b, piece_b)
        except Exception:  # noqa: BLE001
            sol = None
        if not sol:
            _qwen_rearm(page)
            time.sleep(1.5)
            continue
        try:
            err = _qwen_drag_closed_loop(page, state, sol[0])
        except Exception:  # noqa: BLE001
            err = None
        time.sleep(2.0)  # verify POST + verdict
        code = _qwen_last_verify_code(page)
        _log_history('qwen', 'stage',
                     f'slider method={sol[1]} err={err} verify={code or "?"}')
        if code == 'T001' or not _qwen_widget_state(page):
            time.sleep(3)  # let the WAF replay the original POST
            return True
        _qwen_rearm(page)
        time.sleep(random.uniform(1.0, 2.0))
    return False


def signup_qwen() -> Tuple[bool, str]:
    """Create a chat.qwen.ai account autonomously.

    chat.qwen.ai is an OpenWebUI-style deployment; the signup form asks for
    name/email/password (no phone). The signup API sits behind an Aliyun
    slide-captcha, so this runs in a real (windowed, humanized-warmup)
    browser and solves the slider with a closed-loop vision drag
    (_qwen_slider_pass: DOM image dump -> offline target solve ->
    humanized glide + sub-pixel creep). The mailbox comes
    from dsk/mailgen (emailnator's real gmail.com inboxes first). On success
    the JWT lands in the qwen jar and the credentials in data/accounts.json
    — every later renewal is then a plain HTTP re-login (see
    _qwen_signin/refresh_qwen). Accounts that only reach 'pending
    activation' still persist their credentials (never orphaned): the
    login rung polls the signup mailbox for the activation link on every
    later cycle, so activation converges with no operator action.
    """
    email, password = _creds('qwen')
    session = None
    preexisting = bool(_load_accounts().get('qwen'))
    if not email or not password:
        if not mailgen.autogen_enabled():
            return False, 'no QWEN_LOGIN_EMAIL/PASSWORD and mail autogen off'
        session, err = mailgen.create_email()
        if not session:
            return False, f'autogen mailbox unavailable: {err}'
        email = session['address']
        password = mailgen.gen_password()
    name = f'I4F {email.split("@")[0][:8]}'.strip()
    last_error = ''
    typed_pw: Dict[str, str] = {}  # password value actually in the form
    ladder: List[Optional[str]] = [None]
    try:
        # slider verdicts are per-IP-reputation: a burned direct IP never
        # passes, so sample FRESH pool exits directly (latency is irrelevant
        # for a one-off signup; reachability is pre-probed cheaply)
        from . import proxies as _proxies
        _proxies.ensure_pool()
        raw = list(_proxies.all_proxies())
        random.shuffle(raw)
        import requests as _rq
        for cand in raw[:24]:
            if len(ladder) >= 4:
                break
            if cand in ladder:
                continue
            try:
                _rq.get('https://chat.qwen.ai/', timeout=12,
                        proxies={'http': cand, 'https': cand},
                        headers={'User-Agent': _UA})
                ladder.append(cand)
            except Exception:  # noqa: BLE001 — dead exit, skip
                continue
    except Exception:  # noqa: BLE001 — direct remains the fallback
        pass
    def _fill_signup_form(page) -> bool:
        """Fill name/email/passwords, tick the custom agree widget, click
        Create Account. Returns False when the form or button is unusable."""
        if not _fill_first(page, _QWEN_NAME_SELECTORS, name):
            return False
        if not _fill_first(page, _QWEN_EMAIL_SELECTORS, email):
            return False
        pw_fields = page.eles('css:input[type=password]')
        if len(pw_fields) < 2:
            return False
        pw_fields[0].clear()
        pw_fields[0].input(password)
        pw_fields[1].clear()
        pw_fields[1].input(password)
        try:
            # keep what the form actually holds: page JS may normalize the
            # typed value, and re-login later must use the POSTed password
            v0 = str(pw_fields[0].attr('value') or '')
            v1 = str(pw_fields[1].attr('value') or '')
            typed_pw['v'] = v0 or v1
        except Exception:  # noqa: BLE001 — best-effort observation
            pass
        try:
            # the agree control is a custom widget: ARIA role=checkbox
            # (element-plus style) — no real input[type=checkbox] exists
            agreed = False
            for sel in ('css:[role=checkbox]', 'css:input[type=checkbox]'):
                try:
                    cb = page.ele(sel, timeout=2)
                except Exception:  # noqa: BLE001
                    cb = None
                if cb:
                    try:
                        cb.click(by_js=False)
                    except Exception:  # noqa: BLE001
                        cb.click(by_js=True)
                    agreed = True
                    break
            if not agreed:
                page.run_js(
                    'const el=document.querySelector('
                    '"[role=checkbox],[class*=checkbox],[class*=agree]");'
                    'if(el) el.click();')
        except Exception:  # noqa: BLE001
            pass
        time.sleep(1)
        btn_ready = page.run_js(
            '[...document.querySelectorAll("button")]'
            '.filter(b=>/create account/i.test(b.textContent))'
            '.map(b=>b.disabled)[0]')
        if btn_ready:
            return False  # agree toggle failed — button still disabled
        return bool(_click_any(page,
                               ['Create Account', 'Create account', '注册']))

    for proxy in ladder:
        page = None
        try:
            page = _browser(proxy=proxy, headed=True)
            # WAF reputation warmup: browse like a human before touching the
            # challenge (headless + cold single-page sessions score lowest)
            try:
                page.get('https://chat.qwen.ai/')
                time.sleep(random.uniform(6, 10))
                _dismiss_consents(page)
                ac = page.actions
                for _ in range(random.randint(2, 4)):
                    ac.move(random.randint(150, 900),
                            random.randint(120, 500),
                            duration=random.uniform(.2, .5))
                page.run_js('window.scrollBy(0, 400);')
                time.sleep(1)
            except Exception:  # noqa: BLE001 — best-effort warmup
                pass
            # slider scoring accumulates per WAF session: a few failed drags
            # poison the page — so limit drags per load and RELOAD for a
            # fresh verdict instead of grinding on one widget
            submitted = False
            for round_no in range(3):
                page.get(_QWEN_SIGNUP_URL)
                time.sleep(5)
                if 'qwen' not in (page.url or ''):
                    page.get(_QWEN_SIGNUP_URL)
                    time.sleep(4)
                _net_log_install(page)
                if not _fill_signup_form(page):
                    last_error = 'qwen signup form unusable (fields/agree)'
                    break
                submitted = True
                time.sleep(3)
                # the signup POST may be intercepted by Aliyun's slider
                slider_ok = _qwen_slider_pass(page, attempts=4)
                signup_resp = _net_log_read(page, needle='/signup')
                _log_history('qwen', 'stage',
                             f'round={round_no} slider_pass={slider_ok}'
                             + (f' api={signup_resp}' if signup_resp else ''))
            # never orphan a submitted account: persist the credentials now
            # so later cycles re-login (and poll the activation mail) instead
            # of minting yet another mailbox
            if submitted:
                stored_pw = password
                obs = (typed_pw.get('v') or '').strip()
                if obs and obs != password:
                    # The form holds a different password than generated —
                    # keep whichever one actually re-logins (INVALID_CRED on
                    # the generated value was the historical root cause of
                    # un-renewable qwen accounts).
                    ok_g, _d = _qwen_signin(email, password)
                    if not ok_g:
                        ok_o, det_o = _qwen_signin(email, obs)
                        if ok_o:
                            _log_history('qwen', 'stage',
                                         're-login works with the '
                                         'form-observed password; saving it')
                            stored_pw = obs
                        else:
                            _log_history('qwen', 'stage',
                                         f'observed-password signin failed: '
                                         f'{det_o}')
                _save_account('qwen', email, stored_pw,
                              (session or {}).get('backend', ''),
                              {'mail_session': session} if session else None)
            # the session JWT: scan localStorage for any JWT-shaped value
            # (the storage key name is fork-specific) and fall back to the
            # conventional 'token' key / cookies
            token = ''
            deadline = time.time() + 90
            while time.time() < deadline and not token:
                try:
                    token = str(page.run_js(
                        'let hit="";'
                        'const re=/^[A-Za-z0-9_-]{20,}\\.[A-Za-z0-9_-]{20,}'
                        '\\.[A-Za-z0-9_-]{20,}$/;'
                        'for(let i=0;i<localStorage.length;i++){'
                        'const k=localStorage.key(i);'
                        'const v=localStorage.getItem(k)||"";'
                        'if(k==="token"&&v){hit=v;break;}'
                        'if(v.length>80&&re.test(v)){hit=v;break;}}'
                        'return hit;') or '').strip()
                except Exception:  # noqa: BLE001
                    token = ''
                if not token:
                    time.sleep(3)
            # guard against analytics junk that merely LOOKS token-ish:
            # the session JWT must authenticate against /api/v1/auths
            import requests as _rq
            verified = False
            if token:
                try:
                    vcheck = _rq.get(
                        'https://chat.qwen.ai/api/v1/auths',
                        headers={**_qwen_headers(),
                                 'Authorization': f'Bearer {token}'},
                        timeout=30,
                        **_proxies_kwargs('https://chat.qwen.ai'))
                    verified = vcheck.status_code == 200
                except Exception:  # noqa: BLE001
                    verified = False
            _log_history('qwen', 'stage',
                         f'token={"yes" if token else "no"} verified={verified}'
                         + (f' body={_body_head(page)[:80]}' if not verified else ''))
            if not verified:
                # new accounts are "pending activation": the activation mail
                # lands in our mailbox — open its link, then re-login via the
                # WAF-free HTTP signin for a fresh, activated session token
                pending = 'pending activation' in _body_head(page).lower()
                if pending and session:
                    _log_history('qwen', 'stage', 'pending-activation detected')
                    link = mailgen.fetch_otp(
                        session, max_wait_s=180, sender_needle='qwen',
                        code_re=re.compile(
                            r'(https://chat\.qwen\.ai/[^\s"\'<>]*'
                            r'activate[^\s"\'<>]*)'))
                    _log_history('qwen', 'stage',
                                 'activation link '
                                 + ('found' if link else 'MISSING'))
                    if link:
                        try:
                            page.get(link)
                            time.sleep(6)
                        except Exception:  # noqa: BLE001
                            pass
                        ok2, detail2 = _qwen_signin(email, password)
                        if ok2:
                            try:
                                token = (_load_jar('qwen').get('token') or '')
                                vcheck = _rq.get(
                                    'https://chat.qwen.ai/api/v1/auths',
                                    headers={**_qwen_headers(),
                                             'Authorization':
                                                 f'Bearer {token}'},
                                    timeout=30,
                                    **_proxies_kwargs('https://chat.qwen.ai'))
                                verified = vcheck.status_code == 200
                            except Exception:  # noqa: BLE001
                                verified = False
            if not verified and preexisting:
                # the account likely already exists (this signup POST was a
                # duplicate): its activation mail is the missing piece — poll
                # the recorded mailbox, open the link, re-login
                _log_history('qwen', 'stage',
                             'preexisting creds; activation retry')
                if _qwen_activate(email, password)[0]:
                    token = (_load_jar('qwen').get('token') or '')
                    verified = bool(token)
            if not verified:
                last_error = ('qwen signup submitted but no working session '
                              f'captured via {proxy or "direct"} '
                              '(slider/WAF rejected the POST, or the '
                              'activation mail never arrived)')
                continue
            _save_jar('qwen', {'token': token, 'email': email})
            _save_account('qwen', email, password,
                          (session or {}).get('backend', ''),
                          {'mail_session': session} if session else None)
            return True, (f'qwen account ready for {email} (mailbox: '
                          f'{(session or {}).get("backend", "stored creds")})')
        except Exception as e:  # noqa: BLE001
            last_error = f'qwen signup failed: {type(e).__name__}: {e}'
        finally:
            if page is not None:
                _close_page(page)
    return False, last_error or 'all qwen signup egresses failed'


def signup_perplexity() -> Tuple[bool, str]:
    """Create a fresh perplexity.ai account via the NextAuth email magic link.

    2026-10: anonymous ``/rest/sse/perplexity_ask`` sessions are hard-walled
    (``fraud_authwall_upsell`` on every answer, on every IP — measured across
    direct AND several rotated pool egresses), so the provider needs a
    signed-in session. www.perplexity.ai runs NextAuth with a standard
    ``email`` (magic link) provider, so the whole flow is plain HTTP — no
    browser, same shape as the claude rung but without Chromium:

      1. mailgen inbox (real gmail via emailnator),
      2. POST ``/api/auth/signin/email`` ``{email, csrfToken}`` — the
         endpoint is aggressively IP-rate-limited (429 RATE_LIMITED, and
         free-proxy IPs arrive pre-flagged), so every attempt rides a fresh
         pool egress and a 429 rotates + backs off instead of failing the
         signup; the refresher simply retries next cycle,
      3. the emailed ``/api/auth/callback/email?token=...`` link is fetched
         from the same mailbox and opened IN THE SAME HTTP SESSION (NextAuth
         binds the token to this session's pending csrf/callback cookies),
      4. the resulting session cookies are saved to the perplexity jar.
    """
    if not mailgen.autogen_enabled():
        return False, 'mail autogen disabled (I4F_MAIL_AUTOGEN=false)'
    from curl_cffi import requests as cffi
    from . import proxies as _px

    session, err = mailgen.create_email()
    if not session:
        return False, f'autogen mailbox unavailable: {err}'
    email = session['address']

    try:
        attempts = max(1, int(os.getenv('I4F_PERPLEXITY_SIGNUP_ATTEMPTS',
                                        '6') or 6))
    except ValueError:
        attempts = 6
    backoff = max(5.0, float(os.getenv('I4F_PERPLEXITY_SIGNUP_BACKOFF',
                                       '20') or 20))
    ua = {
        'accept': '*/*',
        'accept-language': 'en-US,en;q=0.9',
        'origin': 'https://www.perplexity.ai',
        'referer': 'https://www.perplexity.ai/',
        'user-agent': ('Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 '
                       '(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'),
    }
    last = 'not attempted'
    sent_ts = time.time()
    accepted_body = ''
    http = None
    kw: Dict[str, Any] = {}
    for attempt in range(attempts):
        # proxies only: the host IP is usually already flagged by the
        # auth endpoint, and the pool gives a fresh roll every attempt —
        # except the FINAL attempt falls back to direct (429 flags decay,
        # and a direct roll beats skipping the cycle when the pool draws
        # only dead exits, which happens)
        egress = (None if attempt == attempts - 1
                  else _px.get_proxy('perplexity', direct_ok=False))
        kw = ({'proxies': {'http': egress, 'https': egress}}
              if egress else {})
        try:
            http = cffi.Session(impersonate='chrome120', timeout=25)
            # warm the CF/app cookies, then the NextAuth csrf handshake
            http.get('https://www.perplexity.ai/', headers=ua, **kw)
            csrf = ''
            try:
                csrf = str((http.get(
                    'https://www.perplexity.ai/api/auth/csrf',
                    headers={**ua, 'accept': 'application/json'},
                    **kw).json() or {}).get('csrfToken') or '')
            except ValueError:
                pass
            if not csrf:
                last = 'csrf endpoint gave no token'
                if egress:
                    _px.mark_failure(egress)
                continue
            sent_ts = time.time()
            resp = http.post(
                'https://www.perplexity.ai/api/auth/signin/email',
                data=json.dumps({'email': email, 'csrfToken': csrf,
                                 'callbackUrl': 'https://www.perplexity.ai/'}),
                headers={**ua, 'content-type': 'application/json'}, **kw)
            if resp.status_code in (200, 202):
                accepted_body = resp.text[:160]
                break
            if resp.status_code == 429:
                last = f'signin rate-limited (429) via {egress or "direct"}'
                if egress:
                    _px.mark_failure(egress)
                time.sleep(backoff + random.uniform(0.0, backoff))
                continue
            last = f'signin HTTP {resp.status_code}: {resp.text[:120]}'
            if egress:
                _px.mark_failure(egress)
        except Exception as e:  # noqa: BLE001 — rotate egress and retry
            last = f'{type(e).__name__}: {e}'
            if egress:
                _px.mark_failure(egress)
            http = None
    if http is None:
        return False, (f'perplexity signin never accepted '
                       f'({attempts} tries): {last}')

    magic, mail_body = mailgen.fetch_magic_link(
        session, url_needle='api/auth/callback/email',
        sender_needle='perplexity',
        # 300s expired before emailnator's gmail forward actually
        # delivered (claude precedent: observed ~4-5 min latency);
        # 480s covers it without changing the poll cadence
        max_wait_s=480, after_ts=sent_ts, with_body=True)
    if not magic:
        return False, (
            'perplexity magic-link email not found '
            f'({session.get("backend")}: @{email.rsplit("@", 1)[-1]}; '
            f'signin replied: {accepted_body or last}; '
            f'inbox: {mailgen._inbox_digest(session)}; cand: '
            f'{mailgen.debug_magic_candidates(session, "api/auth/callback/email")})')
    # open the callback IN THE SAME HTTP SESSION — NextAuth binds the token
    # to this session's pending csrf/callback cookies, not to any IP
    try:
        cb = http.get(magic, headers=ua, **kw)
        if cb.status_code >= 400:
            return False, (f'perplexity callback HTTP {cb.status_code}: '
                           f'{cb.text[:120]}')
    except Exception as e:  # noqa: BLE001
        return False, f'perplexity callback failed: {type(e).__name__}: {e}'
    try:
        ver = http.get('https://www.perplexity.ai/api/auth/session',
                       headers={**ua, 'accept': 'application/json'}, **kw)
        user = (ver.json() or {}).get('user') or {}
    except Exception as e:  # noqa: BLE001
        return False, f'perplexity session verify failed: {type(e).__name__}: {e}'
    if not (user.get('email') or user.get('id')):
        return False, (f'perplexity callback did not sign in '
                       f'(HTTP {ver.status_code}, body {ver.text[:120]})')
    try:
        cookies = dict(http.cookies.get_dict())
    except Exception:  # noqa: BLE001 — older curl_cffi fallback
        cookies = {c.name: c.value for c in getattr(http.cookies, 'jar', [])}
    cookies = {k: v for k, v in cookies.items() if k and v}
    if not cookies:
        return False, 'perplexity callback set no cookies'
    _save_jar('perplexity', {**cookies, 'email': email})
    egress_used = (kw.get('proxies') or {}).get('https', 'direct')
    return True, (f'signed in {email} via {egress_used} '
                  f'(user {user.get("email") or user.get("id")})')



def _scrape_api_key(page, min_len: int = 24) -> str:
    """Best-effort API-key scrape from the current page.

    Inputs first (a freshly created key is usually echoed into an input),
    then text nodes; keys are 24-64 chars of [A-Za-z0-9_-] with at least
    one digit or dash (so it never matches a username).
    """
    import re as _re
    pattern = ('[A-Za-z0-9][A-Za-z0-9_-]{%d,64}' % (min_len - 1))
    guard = _re.compile(r'^(?=.*[0-9_-])' + pattern + '$')
    js = (
        'let hit="";'
        'const RX=/(' + pattern + ')/;'
        'for (const el of document.querySelectorAll("input,textarea")) {'
        'const t=(el.value||"").trim();'
        'if (RX.test(t)) { hit=t.match(RX)[1]; break; } }'
        'if (!hit) {'
        'for (const el of document.querySelectorAll("code,pre,div,span,td")) {'
        'const t=(el.value||el.textContent||"");'
        'if (t.length<200 && RX.test(t)) { hit=t.match(RX)[1]; break; } } }'
        'return hit;')
    try:
        key = str(page.run_js(js) or '').strip()
    except Exception:  # noqa: BLE001
        return ''
    if key and guard.match(key):
        return key
    return ''


def signup_meta() -> Tuple[bool, str]:
    """Create a Meta developer account and harvest a Muse Spark API key.

    Flow (browser, US egress): developers.meta.ai -> login/register ->
    email form probed (SSO-only walls are reported honestly) -> emailed
    verification -> dashboard key creation -> the key is scraped into the
    meta jar under api_key. The preview is US-only, so the browser runs on
    the signup proxy ladder when one is configured.
    """
    if not mailgen.autogen_enabled():
        return False, 'mail autogen disabled (I4F_MAIL_AUTOGEN=false)'
    page = None
    try:
        page = _browser(headed=True)
        page.get('https://developers.meta.ai/')
        time.sleep(8)
        _dismiss_consents(page)
        if not _has_field(page, _CHATGPT_EMAIL_SELECTORS):
            return False, ('meta login form not found (SSO-only wall '
                           'likely) - create a key at developers.meta.ai '
                           'and set META_API_KEY manually')
        session, err = mailgen.create_email()
        if not session:
            return False, f'autogen mailbox unavailable: {err}'
        email = session['address']
        if not _fill_first(page, _CHATGPT_EMAIL_SELECTORS, email):
            return False, 'email field not found'
        _click_any(page, ['Continue', 'Sign up', 'Next', 'Log in',
                          'Register'])
        time.sleep(8)
        code = mailgen.fetch_otp(session, max_wait_s=240,
                                 sender_needle='meta')
        link = None
        if not code:
            link = mailgen.fetch_magic_link(session, url_needle='meta',
                                            max_wait_s=240)
        if not code and not link:
            return False, 'meta verification email not found'
        if code:
            if not _fill_first(page, ['css:input[name=code]',
                                      '@placeholder:code',
                                      'css:input[inputmode=numeric]',
                                      'css:input[type=tel]',
                                      'css:input[type=text]'], code):
                return False, 'code field not found'
            _click_any(page, ['Verify', 'Continue', 'Submit'])
        else:
            page.get(link)
        time.sleep(10)
        # key creation: probe the obvious dashboard controls, then scrape
        page.run_js('window.scrollBy(0, 600);')
        _click_any(page, ['Create API key', 'Create key', 'Create Key',
                          'Generate API key', 'New API key', 'API Keys',
                          'Get API key', 'Start building'])
        time.sleep(5)
        key = _scrape_api_key(page)
        if not key:
            return False, ('signup finished but no API key found '
                           '(dashboard layout changed or verification '
                           'wall) - create a key at developers.meta.ai '
                           'and set META_API_KEY manually')
        _save_jar('meta', {'api_key': key, 'email': email})
        _save_account('meta', email, '', session.get('backend', ''))
        return True, (f'account created, API key saved '
                      f'({session.get("backend")}: {email})')
    except Exception as e:  # noqa: BLE001
        return False, f'meta signup failed: {type(e).__name__}: {e}'
    finally:
        if page is not None:
            _close_page(page)


def signup_blackbox() -> Tuple[bool, str]:
    """Create a Blackbox account and harvest an inference API key.

    Flow (browser): blackbox.ai signup -> email + password (Google-only
    walls are reported honestly) -> emailed verification -> key creation
    -> the key is scraped into the blackbox jar under api_key.
    """
    if not mailgen.autogen_enabled():
        return False, 'mail autogen disabled (I4F_MAIL_AUTOGEN=false)'
    page = None
    try:
        page = _browser(headed=True)
        page.get('https://www.blackbox.ai/')
        time.sleep(8)
        _dismiss_consents(page)
        _click_any(page, ['Sign up', 'Sign Up', 'Get started', 'Login',
                          'Log in'])
        time.sleep(5)
        if not _has_field(page, _CHATGPT_EMAIL_SELECTORS):
            return False, ('blackbox signup form not found (Google-only '
                           'wall likely) - create a key per '
                           'docs.blackbox.ai and set BLACKBOX_API_KEY '
                           'manually')
        session, err = mailgen.create_email()
        if not session:
            return False, f'autogen mailbox unavailable: {err}'
        email = session['address']
        password = session.get('password') or mailgen.gen_password()
        if not _fill_first(page, _CHATGPT_EMAIL_SELECTORS, email):
            return False, 'email field not found'
        _fill_first(page, _PASSWORD_SELECTORS, password)
        _click_any(page, ['Continue', 'Sign up', 'Sign Up', 'Next',
                          'Register'])
        time.sleep(8)
        code = mailgen.fetch_otp(session, max_wait_s=240,
                                 sender_needle='blackbox')
        link = None
        if not code:
            link = mailgen.fetch_magic_link(session,
                                            url_needle='blackbox',
                                            max_wait_s=240)
        if not code and not link:
            return False, 'blackbox verification email not found'
        if code:
            if not _fill_first(page, ['css:input[name=code]',
                                      '@placeholder:code',
                                      'css:input[inputmode=numeric]',
                                      'css:input[type=text]'], code):
                return False, 'code field not found'
            _click_any(page, ['Verify', 'Continue', 'Submit'])
        else:
            page.get(link)
        time.sleep(10)
        page.get('https://www.blackbox.ai/settings/api-keys')
        time.sleep(6)
        _click_any(page, ['Create API key', 'Create key', 'Create Key',
                          'Generate API key', 'New key'])
        time.sleep(4)
        key = _scrape_api_key(page)
        if not key:
            return False, ('signup finished but no API key found '
                           '(dashboard layout changed) - create a key '
                           'per docs.blackbox.ai and set BLACKBOX_API_KEY '
                           'manually')
        _save_jar('blackbox', {'api_key': key, 'email': email})
        _save_account('blackbox', email, password,
                      session.get('backend', ''))
        return True, (f'account created, API key saved '
                      f'({session.get("backend")}: {email})')
    except Exception as e:  # noqa: BLE001
        return False, f'blackbox signup failed: {type(e).__name__}: {e}'
    finally:
        if page is not None:
            _close_page(page)


def signup_t3chat() -> Tuple[bool, str]:
    """Log into t3.chat with a disposable mailbox and save the session.

    Flow (browser): www.t3.chat/sign-in -> email -> emailed magic link
    (opened in the same session to pass the Vercel checkpoint) -> the full
    cookie set is CDP-dumped into the t3chat jar.
    """
    if not mailgen.autogen_enabled():
        return False, 'mail autogen disabled (I4F_MAIL_AUTOGEN=false)'
    page = None
    try:
        page = _browser(headed=True)
        page.get('https://www.t3.chat/sign-in')
        time.sleep(8)
        _dismiss_consents(page)
        if not _has_field(page, _CHATGPT_EMAIL_SELECTORS):
            return False, ('t3.chat sign-in form not found (Google-only '
                           'wall likely) - log in manually and save the '
                           'cookies to t3chat_cookies.json')
        session, err = mailgen.create_email()
        if not session:
            return False, f'autogen mailbox unavailable: {err}'
        email = session['address']
        if not _fill_first(page, _CHATGPT_EMAIL_SELECTORS, email):
            return False, 'email field not found'
        _click_any(page, ['Continue', 'Sign in', 'Sign In', 'Log in',
                          'Submit', 'Send'])
        time.sleep(8)
        link = mailgen.fetch_magic_link(session, url_needle='t3.chat',
                                        max_wait_s=240)
        code = None
        if not link:
            code = mailgen.fetch_otp(session, max_wait_s=60,
                                     sender_needle='t3')
        if not link and not code:
            return False, 't3.chat verification email not found'
        if link:
            page.get(link)
            time.sleep(10)
        else:
            if not _fill_first(page, ['css:input[name=code]',
                                      'css:input[inputmode=numeric]',
                                      'css:input[type=text]'], code):
                return False, 'verification code field not found'
            _click_any(page, ['Verify', 'Continue', 'Submit'])
            time.sleep(8)
        page.get('https://www.t3.chat/')
        time.sleep(6)
        saved = _export_cookies(page, 't3chat',
                                ('.t3.chat', 't3.chat'))
        if not saved:
            return False, ('login finished but no session cookies found '
                           '(checkpoint or email-domain blocklist likely)')
        _save_account('t3chat', email, '', session.get('backend', ''))
        return True, (f'login done, cookies saved '
                      f'({session.get("backend")}: {email})')
    except Exception as e:  # noqa: BLE001
        return False, f't3chat signup failed: {type(e).__name__}: {e}'
    finally:
        if page is not None:
            _close_page(page)


def signup_innerai() -> Tuple[bool, str]:
    """Log into inner.ai with a disposable mailbox and save the session.

    inner.ai resets raw TLS from this egress, so this rung is the ONLY
    path in (the shared browser renders fine). Cookies land in the
    innerai jar; the chat endpoint is then RE'd from the authenticated
    session and plugged via I4F_INNERAI_CHAT_URL.
    """
    if not mailgen.autogen_enabled():
        return False, 'mail autogen disabled (I4F_MAIL_AUTOGEN=false)'
    page = None
    try:
        page = _browser(headed=True)
        page.get('https://inner.ai/')
        time.sleep(8)
        _dismiss_consents(page)
        _click_any(page, ['Log in', 'Log In', 'Sign in', 'Sign In',
                          'Get started', 'Sign up'])
        time.sleep(5)
        if not _has_field(page, _CHATGPT_EMAIL_SELECTORS):
            return False, ('inner.ai login form not found (SSO-only wall '
                           'likely) - log in manually and save the cookies '
                           'to innerai_cookies.json')
        session, err = mailgen.create_email()
        if not session:
            return False, f'autogen mailbox unavailable: {err}'
        email = session['address']
        if not _fill_first(page, _CHATGPT_EMAIL_SELECTORS, email):
            return False, 'email field not found'
        _click_any(page, ['Continue', 'Next', 'Sign in', 'Log in',
                          'Submit'])
        time.sleep(8)
        code = mailgen.fetch_otp(session, max_wait_s=240,
                                 sender_needle='inner')
        link = None
        if not code:
            link = mailgen.fetch_magic_link(session, url_needle='inner',
                                            max_wait_s=240)
        if not code and not link:
            return False, 'inner.ai verification email not found'
        if code:
            if not _fill_first(page, ['css:input[name=code]',
                                      '@placeholder:code',
                                      'css:input[inputmode=numeric]',
                                      'css:input[type=text]'], code):
                return False, 'code field not found'
            _click_any(page, ['Verify', 'Continue', 'Submit'])
        else:
            page.get(link)
        time.sleep(10)
        saved = _export_cookies(page, 'innerai',
                                ('.inner.ai', 'inner.ai'))
        if not saved:
            return False, ('login finished but no session cookies found '
                           '(captcha or email-domain blocklist likely)')
        _save_account('innerai', email, '', session.get('backend', ''))
        return True, (f'login done, cookies saved '
                      f'({session.get("backend")}: {email}); next: RE the '
                      'chat endpoint and set I4F_INNERAI_CHAT_URL')
    except Exception as e:  # noqa: BLE001
        return False, f'innerai signup failed: {type(e).__name__}: {e}'
    finally:
        if page is not None:
            _close_page(page)


def signup_adapta() -> Tuple[bool, str]:
    """Log into adapta.org with a disposable mailbox and save the session.

    adapta.org is login-only (Framer marketing site fronts the app);
    cookies land in the adapta jar; the chat endpoint is then RE'd from
    the authenticated session and plugged via I4F_ADAPTA_CHAT_URL.
    """
    if not mailgen.autogen_enabled():
        return False, 'mail autogen disabled (I4F_MAIL_AUTOGEN=false)'
    page = None
    try:
        page = _browser(headed=True)
        page.get('https://adapta.org/')
        time.sleep(8)
        _dismiss_consents(page)
        _click_any(page, ['Log in', 'Log In', 'Sign in', 'Sign In',
                          'Entrar', 'Entrar na plataforma',
                          'Get started', 'Começar'])
        time.sleep(6)
        if not _has_field(page, _CHATGPT_EMAIL_SELECTORS):
            return False, ('adapta login form not found (SSO-only wall '
                           'likely) - log in manually and save the cookies '
                           'to adapta_cookies.json')
        session, err = mailgen.create_email()
        if not session:
            return False, f'autogen mailbox unavailable: {err}'
        email = session['address']
        password = session.get('password') or mailgen.gen_password()
        if not _fill_first(page, _CHATGPT_EMAIL_SELECTORS, email):
            return False, 'email field not found'
        _fill_first(page, _PASSWORD_SELECTORS, password)
        _click_any(page, ['Continue', 'Next', 'Entrar', 'Sign in',
                          'Log in', 'Submit'])
        time.sleep(8)
        code = mailgen.fetch_otp(session, max_wait_s=240,
                                 sender_needle='adapta')
        link = None
        if not code:
            link = mailgen.fetch_magic_link(session,
                                            url_needle='adapta',
                                            max_wait_s=240)
        if not code and not link:
            return False, 'adapta verification email not found'
        if code:
            if not _fill_first(page, ['css:input[name=code]',
                                      '@placeholder:code',
                                      'css:input[inputmode=numeric]',
                                      'css:input[type=text]'], code):
                return False, 'code field not found'
            _click_any(page, ['Verify', 'Continue', 'Submit',
                              'Confirmar'])
        else:
            page.get(link)
        time.sleep(10)
        saved = _export_cookies(page, 'adapta',
                                ('.adapta.org', 'adapta.org'))
        if not saved:
            return False, ('login finished but no session cookies found '
                           '(captcha or email-domain blocklist likely)')
        _save_account('adapta', email, password,
                      session.get('backend', ''))
        return True, (f'login done, cookies saved '
                      f'({session.get("backend")}: {email}); next: RE the '
                      'chat endpoint and set I4F_ADAPTA_CHAT_URL')
    except Exception as e:  # noqa: BLE001
        return False, f'adapta signup failed: {type(e).__name__}: {e}'
    finally:
        if page is not None:
            _close_page(page)


SIGNUP = {'deepseek': signup_deepseek, 'chatgpt': signup_chatgpt,
          'gemini': signup_gemini,
          'claude': signup_claude,
          'grok': signup_grok,
          'qwen': signup_qwen,
          'kimi': signup_kimi,
          'mistral': signup_mistral,
          'copilot': _anonymous('copilot'),
          'perplexity': signup_perplexity, 'glm': _anonymous('glm'),
          'duck': _anonymous('duck'),
          'pollinations': _anonymous('pollinations'),
          'huggingchat': signup_huggingchat,
          'groq': signup_groq,
          'cerebras': signup_cerebras,
          'modelscope': signup_modelscope,
          'mistral_api': signup_mistral_api,
          'openrouter': signup_openrouter,
          'llm7': signup_llm7,
          'google_ai_studio': signup_google_ai_studio,
          'cohere': signup_cohere,
          'cloudflare': signup_cloudflare,
          'meta': signup_meta,
          'blackbox': signup_blackbox,
          't3chat': signup_t3chat,
          'innerai': signup_innerai,
          'adapta': signup_adapta,
          'arena': _manual_only(
              'arena',
              'signup requires an allowlisted email domain '
              '(I4F_MAIL_DOMAIN catch-all) and a browser login')}


# ------------------------------------------------------------------ renew
def _seed_counts() -> None:
    """Seed today's per-provider attempt counts from history.jsonl.

    The counts live in memory, so a container restart would otherwise
    bypass the daily attempt cap; today's ``renew-start`` events make the
    budget continuous across restarts. Runs once per process.

    Only strict proactive attempts count against the budget: today's
    ``renew-start`` events whose reason is exactly ``proactive``
    (the TTL-cadence bootstrap sweep). Reactive attempts ('auth' from
    the self-heal probe, 'inline-auth' from the request path) and
    operator-initiated runs ('manual-*', CLI) are excluded — so a heavy
    debug or reactive day can never starve the daemon's own retries.
    """
    if _STATE.counts_seeded:
        return
    _STATE.counts_seeded = True
    try:
        path = _data_dir() / 'refresher' / 'history.jsonl'
        today = time.strftime('%Y-%m-%d')
        counts: Dict[str, int] = {}
        with path.open('r', encoding='utf-8') as fh:
            for line in fh:
                try:
                    entry = json.loads(line)
                except (ValueError, TypeError):
                    continue
                reason = str(entry.get('detail', ''))
                if (entry.get('event') == 'renew-start'
                        and str(entry.get('ts', '')).startswith(today)
                        and reason == 'proactive'):
                    counts[entry.get('provider', '')] = \
                        counts.get(entry.get('provider', ''), 0) + 1
        for name, n in counts.items():
            _STATE.counts[name] = (today, n)
        if counts:
            print(f'[refresher] daily budget seeded from history: {counts}')
    except OSError:
        pass  # no history yet


def renew(name: str, reason: str = '') -> Dict[str, Any]:
    """Run the full renewal ladder for one provider. Returns a status dict."""
    if not _env_bool('I4F_REFRESHER', True):
        return {'renewed': False, 'skipped': 'refresher disabled'}
    if not provider_enabled(name):
        return {'renewed': False, 'skipped': 'provider disabled (I4F_PROVIDERS)'}
    excl = {e.strip().lower() for e in
            os.getenv('I4F_REFRESHER_EXCLUDE', '').split(',') if e.strip()}
    if name in excl:
        return {'renewed': False, 'skipped': f'{name} excluded'}
    with _STATE.lock:
        if _STATE.renewing.get(name):
            return {'renewed': False, 'skipped': 'renewal already running'}
        last = _STATE.results.get(name, {})
        # 'escalate' = refresh_cycle handing a failed cheap-refresh to the
        # ladder. The failure was logged seconds ago, so the plain cooldown
        # check would skip the ladder with its own fresh timestamp — the
        # escalation would never run. Budget + rung breakers still apply.
        if last.get('ts') and time.time() - _entry_ts(last) < _cooldown() \
                and not (reason.startswith('manual') or reason == 'escalate'):
            return {'renewed': False, 'skipped': 'cooldown'}
        today = time.strftime('%Y-%m-%d')
        _seed_counts()
        day, n = _STATE.counts.get(name, (today, 0))
        n = n + 1 if day == today else 1
        manual = reason.startswith('manual')
        if n > _max_renews() and not manual:
            return {'renewed': False, 'skipped': 'daily attempt budget exhausted'}
        if not manual:  # manual/CLI attempts never consume the daemon budget
            _STATE.counts[name] = (day, n)
        _STATE.renewing[name] = True
    try:
        return _renew_locked(name, reason)
    finally:
        with _STATE.lock:
            _STATE.renewing[name] = False


def _entry_ts(entry: Dict[str, Any]) -> float:
    try:
        return time.mktime(time.strptime(entry['ts'][:19], '%Y-%m-%dT%H:%M:%S'))
    except Exception:
        return 0.0


def _breaker_n() -> int:
    return max(1, int(os.getenv('I4F_BREAKER_N', '3') or 3))


def _breaker_cooldown() -> float:
    return max(60.0, float(os.getenv('I4F_BREAKER_COOLDOWN_S', '1800') or 1800))


def _signup_breaker_cooldown() -> float:
    """Cooldown for the SIGNUP rung's breaker. A signup attempt is the most
    expensive kind of renewal (a multi-minute Chromium flow), so a rung that
    fails systematically must not retry every 30 minutes: after N consecutive
    failures it sleeps for hours instead (default 12h, floor at the plain
    breaker cooldown). One success resets it via _rung_result."""
    return max(_breaker_cooldown(),
               float(os.getenv('I4F_SIGNUP_BREAKER_COOLDOWN_S', '43200')
                     or 43200))


def _rung_open(name: str, rung: str) -> bool:
    with _STATE.lock:
        return time.time() < _STATE.rung_blocked_until.get((name, rung), 0.0)


def _rung_result(name: str, rung: str, ok: bool) -> None:
    opened = 0.0
    with _STATE.lock:
        key = (name, rung)
        if ok:
            _STATE.rung_fails[key] = 0
            _STATE.rung_blocked_until.pop(key, None)
            return
        n = _STATE.rung_fails.get(key, 0) + 1
        _STATE.rung_fails[key] = n
        if n >= _breaker_n():
            _STATE.rung_fails[key] = 0
            # signup failures cost a browser flow each — a rung that keeps
            # failing there sleeps for hours, not minutes
            cooldown = (_signup_breaker_cooldown() if rung == 'signup'
                        else _breaker_cooldown())
            _STATE.rung_blocked_until[key] = time.time() + cooldown
            opened = cooldown
    if opened:
        _log_history(name, 'breaker-open',
                     f'{rung}: {_breaker_n()} consecutive failures -> '
                     f'{opened:.0f}s cooldown')


_BAN_PATTERNS = ('account_banned', 'banned', 'suspended', 'deactivated')


def _permanent_auth_block(name: str) -> bool:
    """True when the live probe reports the account itself gone for good
    (ban/suspension/deactivation). A banned credential can never be
    refreshed or re-logged-into, so the ladder should rotate to a fresh
    signup instead of spending its rungs (and breaker history) on the
    corpse. HTTP-only: the probe is a cheap authenticated GET, no browser."""
    try:
        from . import selfheal
        status, detail = selfheal._probe_once(name)
    except Exception:  # noqa: BLE001
        return False
    if str(status) != 'auth':
        return False
    low = str(detail).lower()
    return any(p in low for p in _BAN_PATTERNS)


def _renew_locked(name: str, reason: str) -> Dict[str, Any]:
    steps: List[str] = []
    _log_history(name, 'renew-start', reason or 'proactive')

    # Anonymous providers have no credential, so the credential ladder below
    # can only ever report its no-op stub as a success. Their refusals are
    # egress-shaped (geo-block, datacenter-IP authwall), so rotate the exit
    # first and let the router retry from the new IP.
    rotate = name in _egress_rotate_providers()
    if rotate:
        if _rung_open(name, 'egress'):
            steps.append('egress: skipped (breaker open)')
        else:
            ok, detail = rotate_egress(name)
            steps.append(f'egress: {detail}')
            _log_history(name, 'egress-rotate', detail)
            _rung_result(name, 'egress', ok)
            if ok:
                _log_history(name, 'renewed', '; '.join(steps))
                return {'renewed': True, 'via': 'egress-rotation',
                        'steps': steps, 'egress': detail}

    if _rung_open(name, 'refresh'):
        steps.append('refresh: skipped (breaker open)')
        status = 'fail'
    else:
        ok, detail = REFRESH[name]()
        steps.append(f'refresh: {detail}')
        _log_history(name, 'refresh', detail)
        # The probe checks the *anonymous* surface (e.g. chatgpt's browser
        # relay), so it reports 'ok' even when the credential is dead —
        # gating on it here short-circuited the ladder and the provider
        # never re-logged in. The refresh rung's own success is the
        # credential verdict; only escalate to login/signup when it fails.
        _rung_result(name, 'refresh', ok)
        if ok and rotate:
            # 'anonymous access — nothing to refresh' proves nothing about
            # the wall that triggered this renewal; declaring it renewed is
            # what made copilot/perplexity spin: the same inline-auth error
            # recurred minutes later with a 'renewed' line in between.
            status = 'fail'
        elif ok:
            status = _verify(name)
            if status == 'ok':
                _log_history(name, 'renewed', '; '.join(steps))
                return {'renewed': True, 'via': 'http-refresh', 'steps': steps}
        else:
            status = 'fail'

    # A permanently dead account can never be refreshed back: rotate to a
    # fresh signup instead of burning the browser-login rung (and its
    # breaker history) against the corpse.
    banned = status == 'fail' and _permanent_auth_block(name)
    if banned:
        steps.append('ban detected — rotating to a fresh account')
        _log_history(name, 'ban-rotate',
                     'account banned/suspended — skipping refresh/login, '
                     'rotating straight to signup')
    if (not banned and _env_bool('I4F_REFRESHER_LOGIN', True)
            and not _rung_open(name, 'login')):
        ok, detail = browser_login(name)
        steps.append(f'login: {detail}')
        _log_history(name, 'browser-login', detail)
        # Same as the refresh rung: the probe reports the anonymous surface
        # healthy, so it must not gate the ladder. The rung's own success is
        # the credential verdict; a failed login escalates to signup.
        _rung_result(name, 'login', ok)
        if ok:
            status = _verify(name)
            if status == 'ok':
                _log_history(name, 'renewed', '; '.join(steps))
                return {'renewed': True, 'via': 'browser-login', 'steps': steps}
        else:
            status = 'fail'

    if _env_bool('I4F_REFRESHER_AUTOSIGNUP', True) \
            and not _rung_open(name, 'signup'):
        with _file_lock(_SIGNUP_LOCK_PATH):
            ok, detail = SIGNUP[name]()   # all providers: create what is missing
        steps.append(f'signup: {detail}')
        _log_history(name, 'autosignup', detail)
        # Same as the rungs above: a failed signup must not be masked by the
        # anonymous-surface probe.
        _rung_result(name, 'signup', ok)
        if ok:
            status = _verify(name)
            if status == 'ok':
                _log_history(name, 'renewed', '; '.join(steps))
                return {'renewed': True, 'via': 'autosignup', 'steps': steps}
        else:
            status = 'fail'

    _log_history(name, 'renew-failed', '; '.join(steps))
    return {'renewed': False, 'steps': steps,
            'hint': 'needs manual credential update' if status == 'auth' else status}


def _verify(name: str) -> str:
    try:
        from . import selfheal
        status, _ = selfheal._probe_once(name)
        return status
    except Exception as e:  # noqa: BLE001
        return f'probe-error: {e}'


def renew_inline(name: str, detail: str = '') -> Dict[str, Any]:
    """Request-path remediation: fire a background renewal on an auth error.

    Called from the router the moment a request classified as
    ``ProviderAuthError`` — the ladder (refresh -> browser re-login ->
    auto-signup) runs in a background thread so the failing request is
    not blocked; the NEXT request picks up the fresh credential.
    Guarded by an hourly per-provider attempt cap (I4F_REFRESHER_INLINE_HOURLY,
    default 2) and a 60s silence window after a completed attempt so a
    burst of failing requests cannot spin the ladder.
    """
    if not _env_bool('I4F_REFRESHER', True):
        return {'triggered': False, 'skipped': 'refresher disabled'}
    if not provider_enabled(name):
        return {'triggered': False, 'skipped': 'provider disabled'}
    hourly = max(1, int(os.getenv('I4F_REFRESHER_INLINE_HOURLY', '2') or 2))
    now = time.time()
    hour = time.strftime('%Y%m%d%H')
    with _STATE.lock:
        if _STATE.renewing.get(name):
            return {'triggered': False, 'skipped': 'renewal already running'}
        last = _STATE.inline_last.get(name)
        if last and now - last[1] < 60:
            return {'triggered': False, 'skipped': 'inline silence window'}
        (h, cnt) = _STATE.inline_counts.get(name, (hour, 0))
        if h == hour and cnt >= hourly:
            return {'triggered': False, 'skipped': 'inline hourly cap'}
        _STATE.inline_counts[name] = (hour, cnt + 1 if h == hour else 1)

    def _run() -> None:
        try:
            res = renew(name, reason='inline-auth')
            ok = bool(res.get('renewed'))
        except Exception:  # noqa: BLE001 — remediation must never raise
            ok = False
        with _STATE.lock:
            _STATE.inline_last[name] = ('ok' if ok else 'failed', time.time())

    t = threading.Thread(target=_run, name=f'inline-renew-{name}', daemon=True)
    with _STATE.lock:
        _STATE.inline_threads[name] = t
        # mark the silence window BEFORE the ladder starts: a browser re-login
        # can run for minutes, and without this a burst of failing requests
        # passes the 60s check repeatedly and stacks ladders on one provider
        _STATE.inline_last[name] = ('running', now)
    t.start()
    _log_history(name, 'inline-renew-triggered', detail[:200])
    return {'triggered': True, 'reason': detail[:120]}


# ------------------------------------------------------------------ daemon
def refresh_cycle() -> Dict[str, Any]:
    """Proactive daemon cycle: rotate refreshable cookies (gemini/chatgpt)
    and BOOTSTRAP any provider that has no credentials at all — the signup
    rung creates fresh ones unattended."""
    if not _env_bool('I4F_REFRESHER', True):
        return {'refresher': 'disabled'}
    out: Dict[str, Any] = {}
    for name in tuple(REFRESH):
        if not provider_enabled(name):
            continue  # disabled via I4F_PROVIDERS: no routes, no probes, no bot
        with _STATE.lock:
            if _STATE.renewing.get(name):
                # an inline/ladder renewal is running for this provider —
                # don't race a second ladder (its refresh would run the same
                # rungs concurrently and double-log)
                out[name] = 'skipped (renewal already running)'
                continue
        if not _has_creds(name):
            if not _env_bool('I4F_REFRESHER_AUTOSIGNUP', True):
                out[name] = 'skipped (no credentials, autosignup off)'
                continue
            try:
                out[name] = renew(name, reason='bootstrap')
            except Exception as e:  # noqa: BLE001
                out[name] = f'bootstrap error: {type(e).__name__}: {e}'
            continue
        if name == 'deepseek':
            continue  # token is verified live by the self-heal probe
        try:
            ok, detail = REFRESH[name]()
            _log_history(name, 'proactive-refresh' if ok else 'refresh-issue',
                         detail)
            if ok:
                out[name] = detail
                continue
            # The cheap refresh rung failed while credentials exist: hand the
            # provider to the full ladder (re-login -> signup). Logging the
            # issue and moving on left such providers stuck for weeks — this
            # is the "has credentials but never renews them" gap.
            try:
                out[name] = renew(name, reason='escalate')
            except Exception as e:  # noqa: BLE001
                out[name] = f'{detail}; ladder error: {type(e).__name__}: {e}'
        except Exception as e:  # noqa: BLE001
            out[name] = f'error: {e}'
    return out


def bootstrap_all() -> Dict[str, Any]:
    """Force credential creation for every provider that currently has
    none (CLI / manual trigger; bypasses the per-provider cooldown)."""
    out: Dict[str, Any] = {}
    for name in REFRESH:
        if not provider_enabled(name):
            out[name] = {'renewed': False, 'skipped': 'provider disabled (I4F_PROVIDERS)'}
            continue
        if _has_creds(name):
            out[name] = {'renewed': False, 'skipped': 'credentials present'}
            continue
        try:
            out[name] = renew(name, reason='manual-bootstrap')
        except Exception as e:  # noqa: BLE001
            out[name] = {'renewed': False, 'error': str(e)[:300]}
    return out


def start_daemon() -> bool:
    with _STATE.lock:
        if _STATE.started:
            return False
        _STATE.started = True

    def _loop() -> None:
        while True:
            try:
                refresh_cycle()
            except Exception:  # pragma: no cover
                pass
            time.sleep(_ttl())

    threading.Thread(target=_loop, name='refresher', daemon=True).start()
    return True


def status() -> Dict[str, Any]:
    with _STATE.lock:
        results = dict(_STATE.results)
        started = _STATE.started
    enabled = [p for p in REFRESH if provider_enabled(p)]
    return {'enabled': _env_bool('I4F_REFRESHER', True),
            'daemon': started,
            'ttl': _ttl(),
            'browser_login': _env_bool('I4F_REFRESHER_LOGIN', True),
            'autosignup': _env_bool('I4F_REFRESHER_AUTOSIGNUP', True),
            'autosignup_providers': sorted(SIGNUP),
            'mail_autogen': mailgen.autogen_enabled(),
            'mail_configured': bool(os.getenv('I4F_MAIL_IMAP_HOST', '').strip()),
            'providers': {'enabled': enabled,
                          'disabled': [p for p in REFRESH if p not in enabled]},
            'credentials': {p: bool(all(_creds(p))) for p in REFRESH},
            'has_credentials': {p: _has_creds(p) for p in REFRESH},
            'bootstrap': {'enabled': _env_bool('I4F_REFRESHER_AUTOSIGNUP', True),
                          'missing': [p for p in enabled if not _has_creds(p)]},
            'last_results': results}


def main(argv: List[str]) -> int:  # pragma: no cover - CLI
    cmd = argv[1] if len(argv) > 1 else 'status'
    if cmd == 'status':
        print(json.dumps(status(), indent=2, default=str))
        return 0
    if cmd == 'refresh' and len(argv) > 2:
        print(json.dumps({argv[2]: REFRESH[argv[2]]()[1]}, indent=2))
        return 0
    if cmd == 'login' and len(argv) > 2:
        print(json.dumps(dict(zip(('ok', 'detail'), browser_login(argv[2]))), indent=2))
        return 0
    if cmd == 'signup':
        prov = argv[2] if len(argv) > 2 else 'deepseek'
        with _file_lock(_SIGNUP_LOCK_PATH):
            result = SIGNUP[prov]()
        print(json.dumps(dict(zip(('ok', 'detail'), result)), indent=2))
        return 0
    if cmd == 'bootstrap':
        print(json.dumps(bootstrap_all(), indent=2, default=str))
        return 0
    if cmd == 'mailgen':
        session, err = mailgen.create_email()
        print(json.dumps({'ok': bool(session),
                          'detail': session or err,
                          'address': (session or {}).get('address')}, indent=2))
        return 0
    print(__doc__)
    return 1


if __name__ == '__main__':
    import sys
    sys.exit(main(sys.argv))
