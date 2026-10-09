"""Automatic email-address generation for unattended credential flows.

Used by the refresher (browser re-login / auto-signup) when no login e-mail
is configured: a throwaway mailbox is created on the fly, the verification
e-mail is fetched from it and the OTP is extracted — so the renewal ladder
stays fully unmanned with zero mail configuration.

Seven backends, tried in order (first that is *configured* wins, then the
first that *works*):

1. IMAP catch-all (``I4F_MAIL_IMAP_HOST`` + ``I4F_MAIL_DOMAIN``):
   a random local part is invented under your own domain
   (``i4f-<hex8>@<domain>``); the OTP is read through the existing IMAP
   poller. Most reliable — use this when you own a catch-all mailbox.

2. emailnator.com (https://www.emailnator.com, free, no key): the only
   free inbox service handing out REAL ``@gmail.com`` addresses (dot/plus
   variants of pooled master accounts). gmail.com is allowlisted
   essentially everywhere — this is the autonomous path that finally
   delivers DeepSeek OTP mail, which silently drops every disposable
   domain.

3. tempmail.lol (https://tempmail.lol, free public API, no key): a mailbox
   on a rotating pool of obscure domains — the best chance among the
   disposable pools against domain blocklists. No signup; the inbox is
   addressed by an opaque token stored in the session.

4. tempmail.plus (https://tempmail.plus, free public API, no key):
   an implicit inbox on @mailto.plus — any local part has a mailbox, the
   listing API needs no session. Semi-known domain: fewer blocklists than
   mail.tm, more than tempmail.lol's rotating pool.

5. temp-mail.io (https://temp-mail.io, free public API, no key): a mailbox
   on another rotating pool of obscure domains; the listing carries the
   body inline. Opaque token kept in the session for diagnostics.

6. Guerrilla Mail (https://guerrillamail.com, free public API, no key):
   the long-standing session-addressed inbox. A fresh random user is
   claimed per mailbox and the @sharklasers.com alias of the SAME inbox
   is handed out (every guerrilla domain delivers to one inbox, and
   sharklasers.com sits in far fewer disposable-domain blocklists than
   guerrillamailblock.com).

7. mail.tm / mail.gw (https://mail.tm, free public API, no key): a real
   throwaway account is created on a public temp-mail domain and its
   inbox is polled over HTTPS. Works out of the box, but public domains
   are often rejected by signup forms (DeepSeek silently drops them)
   — the caller treats failures as a normal renewal-ladder miss.

Env switches:
    I4F_MAIL_AUTOGEN   master switch for this module (default: true)
    I4F_MAIL_DOMAIN    domain for the IMAP catch-all backend (optional)

CLI smoke test:
    python -m dsk.mailgen
"""

import http.cookiejar
import json
import os
import re
import secrets
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

MAILTM_API = 'https://api.mail.tm'
MAILGW_API = 'https://api.mail.gw'
TEMPMAIL_API = 'https://api.tempmail.lol'

# ``after_ts`` guards against consuming a PREVIOUS signup attempt's
# already-used link/code, not against the submit round-trip: the form
# submit helper returns tens of seconds after the send actually fires, so
# the current run's own email can carry a timestamp slightly BEFORE
# ``sent_ts`` (observed 26-28s with emailnator). Mail younger than this
# grace must never be cut; a previous run's mail is minutes older anyway.
_AFTER_TS_GRACE_S = 120.0

# percent-encoded https URL inside click-tracker query strings
_ENC_URL_RE = re.compile(r'https?%3A%2F%2F[^\s"\'<>]+', re.I)
_EMAILNATOR_BASE = 'https://www.emailnator.com'
_EMAILNATOR_UA = ('Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 '
                  '(KHTML, like Gecko) Chrome/131.0 Safari/537.36')
_UA = 'inference4free-refresher/1.0 (+autonomous credential maintenance)'


def autogen_enabled() -> bool:
    raw = os.getenv('I4F_MAIL_AUTOGEN', '').strip().lower()
    if not raw:
        return True  # default ON
    return raw in ('1', 'true', 'yes', 'on')


def _gen_local_part() -> str:
    return f"i4f-{secrets.token_hex(4)}"


def _gen_password() -> str:
    # letter + digit prefix keeps even the pickiest signup forms happy
    return f"Aa1{secrets.token_urlsafe(12)}"


def gen_password() -> str:
    """Public alias — signup forms need a password even when the mailbox
    backend (tempmail.lol) is token-addressed and carries none."""
    return _gen_password()


def _http(method: str, url: str, body: Optional[Dict[str, Any]] = None,
          token: Optional[str] = None, timeout: int = 25) -> Tuple[int, Any]:
    """Minimal JSON HTTP client (stdlib only — no proxy deps)."""
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header('User-Agent', _UA)
    req.add_header('Accept', 'application/json')
    if data is not None:
        req.add_header('Content-Type', 'application/json')
    if token:
        req.add_header('Authorization', f'Bearer {token}')
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode('utf-8', 'replace')
            return resp.status, (json.loads(raw) if raw.strip() else {})
    except urllib.error.HTTPError as e:
        try:
            raw = e.read().decode('utf-8', 'replace')
            return e.code, json.loads(raw)
        except Exception:  # noqa: BLE001
            return e.code, {}
    except Exception as e:  # noqa: BLE001
        return 0, {'error': f'{type(e).__name__}: {e}'}


# ------------------------------------------------------------------ mail.tm
def _items(body: Any) -> List[Any]:
    """mail.tm returns either a JSON-LD object (hydra:member) or a plain
    list depending on the requested content type — accept both."""
    if isinstance(body, list):
        return body
    if isinstance(body, dict):
        return body.get('hydra:member') or body.get('member') or []
    return []


def _mailtm_domains(api: str) -> List[str]:
    code, body = _http('GET', f'{api}/domains?page=1')
    if code != 200:
        return []
    return [d['domain'] for d in _items(body)
            if isinstance(d, dict) and d.get('domain') and d.get('isActive', True)]


def _mailtm_extract(body: Any, code_re: re.Pattern) -> Optional[str]:
    """Extract the OTP from a mail.tm message payload (text or html)."""
    if not isinstance(body, dict):
        return None
    text = ''
    for key in ('text', 'intro'):
        if isinstance(body.get(key), str):
            text += body[key] + '\n'
    html = body.get('html')
    if isinstance(html, list):
        text += '\n'.join(str(h) for h in html)
    elif isinstance(html, str):
        text += html
    match = code_re.search(text)
    return match.group(1) or match.group(0) if match else None


def _mailtm_create() -> Optional[Dict[str, Any]]:
    """Create a throwaway account on the first working temp-mail service.

    mail.tm and mail.gw expose the same API shape but run different domain
    pools; ESP blocklists often cover one pool and not the other, so both
    are tried. The chosen service's API base is stored in the session so
    fetch_otp polls the right inbox.
    """
    for backend, api in (('mail.tm', MAILTM_API), ('mail.gw', MAILGW_API)):
        for domain in _mailtm_domains(api)[:3]:
            address = f"{_gen_local_part()}@{domain}"
            password = _gen_password()
            code, body = _http('POST', f'{api}/accounts',
                               {'address': address, 'password': password})
            if code in (200, 201) and isinstance(body, dict) and body.get('id'):
                code2, tok = _http('POST', f'{api}/token',
                                   {'address': address, 'password': password})
                if code2 == 200 and isinstance(tok, dict) and tok.get('token'):
                    return {'backend': backend, 'api': api, 'address': address,
                            'password': password, 'token': tok['token'],
                            'account_id': body.get('id')}
            # rate-limited / domain rejected → try the next domain
            time.sleep(1.5)
    return None


def _mailtm_fetch_otp(session: Dict[str, Any], sender_needle: str,
                      code_re: re.Pattern, max_age_min: float,
                      deadline: float, seen_ids: set,
                      after_ts: Optional[float] = None) -> Optional[str]:
    """Poll the temp-mail inbox (mail.tm or mail.gw) until a fresh OTP shows."""
    api = str(session.get('api') or MAILTM_API)
    while time.time() < deadline:
        code, body = _http('GET', f'{api}/messages?page=1',
                           token=session['token'])
        if code == 200:
            for msg in _items(body):
                if not isinstance(msg, dict) or msg.get('id') in seen_ids:
                    continue
                sender = str((msg.get('from') or {}).get('address', '')).lower()
                if sender_needle and sender_needle not in sender:
                    continue
                # age filter: mail.tm timestamps are ISO-8601
                age_min = _iso_age_min(msg.get('createdAt'))
                if age_min is not None and age_min > max_age_min:
                    continue
                # need the full message for the body
                mcode, full = _http('GET',
                                    f"{api}/messages/{msg.get('id')}",
                                    token=session['token'])
                if mcode == 200:
                    # blacklist only after a successful body fetch: a one-off
                    # fetch failure must not hide the message for the window
                    seen_ids.add(msg.get('id'))
                    otp = _mailtm_extract(full, code_re)
                    if otp:
                        return otp
        time.sleep(6)
    return None


def _iso_age_min(ts: Any) -> Optional[float]:
    if not ts:
        return None
    try:
        from datetime import datetime, timezone
        dt = datetime.fromisoformat(str(ts).replace('Z', '+00:00'))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - dt).total_seconds() / 60.0
    except Exception:
        return None


# --------------------------------------------------------------- tempmail.lol
def _tempmail_create() -> Optional[Dict[str, Any]]:
    """Mint a mailbox on tempmail.lol.

    The service rotates a large pool of obscure domains per mailbox, which
    dodges the disposable-domain blocklists that DeepSeek (and others) apply
    to the well-known mail.tm/mail.gw pools. No account signup needed — the
    inbox is addressed by an opaque token.
    """
    code, body = _http('GET', f'{TEMPMAIL_API}/generate')
    if code != 200 or not isinstance(body, dict):
        return None
    address = str(body.get('address') or '').strip()
    token = str(body.get('token') or '').strip()
    if not address or not token:
        return None
    return {'backend': 'tempmail.lol', 'address': address, 'token': token}


def _tempmail_fetch_otp(session: Dict[str, Any], sender_needle: str,
                        code_re: re.Pattern, max_age_min: float,
                        deadline: float, seen_ids: set,
                        after_ts: Optional[float] = None) -> Optional[str]:
    """Poll the tempmail.lol inbox for the verification code."""
    token = session.get('token') or ''
    while time.time() < deadline:
        code, body = _http('GET', f'{TEMPMAIL_API}/auth/{token}')
        if code == 200 and isinstance(body, dict):
            for msg in body.get('email') or []:
                if not isinstance(msg, dict):
                    continue
                mid = str(msg.get('date') or '') + str(msg.get('from') or '') \
                    + str(msg.get('subject') or '')
                if mid in seen_ids:
                    continue
                seen_ids.add(mid)
                sender = str(msg.get('from') or '').lower()
                if sender_needle and sender_needle not in sender:
                    continue
                age = _iso_age_min(msg.get('date'))
                if age is not None and age > max_age_min:
                    continue
                text = '\n'.join(str(msg.get(k) or '')
                                 for k in ('body', 'html', 'subject'))
                match = code_re.search(text)
                if match:
                    return match.group(1) or match.group(0)
        time.sleep(6)
    return None


# ------------------------------------------------- emailnator (real gmail.com)
class _Emailnator:
    """emailnator.com client — the only free inbox service that hands out
    REAL ``@gmail.com`` addresses (dot/plus variants of pooled master
    accounts). DeepSeek's mail pipeline silently drops every accessible
    disposable domain, but gmail.com is allowlisted essentially everywhere,
    so this is the first public backend for OTP delivery.

    Endpoints (reverse-engineered from the site bundle):
      POST /api/generate-email  {"ids": [2,3,8]}  -> {"email": "..."}
      POST /api/message-list    {"email": addr, "limit": 20} -> {"messages": [...]}
      GET  /api/message/{id}    -> message body
    CSRF: XSRF-TOKEN cookie echoed (URL-decoded) as X-XSRF-TOKEN; the pair
    is refreshed on 403/419.
    """

    def __init__(self) -> None:
        # emailnator fingerprints TLS: python urllib/requests get the page
        # but never receive session cookies — curl's fingerprint does, so
        # this client shells out to curl with a persistent cookie jar.
        self._jar = ''
        self._xsrf = ''

    def _jar_path(self) -> str:
        if not self._jar:
            import tempfile
            self._jar = os.path.join(tempfile.gettempdir(),
                                     f'i4f-emailnator-{os.getpid()}.jar')
        return self._jar

    def _curl(self, args: List[str], timeout: int = 40) -> Tuple[int, str]:
        cmd = (['curl', '-sS', '-m', '30',
                '-b', self._jar_path(), '-c', self._jar_path(),
                '-A', _EMAILNATOR_UA,
                '-H', 'Accept-Language: en-US,en;q=0.9'] + args)
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True,
                                  timeout=timeout)
        except (OSError, subprocess.TimeoutExpired):
            return 0, ''
        out = proc.stdout or ''
        status = 0
        if '\n' in out[-12:]:
            raw, _, code = out.rpartition('\n')
            try:
                status = int(code.strip())
                out = raw
            except ValueError:
                status = 0
        return (status or (200 if proc.returncode == 0 else 0)), out

    def _ensure(self) -> None:
        code, _ = self._curl(['-H', 'Accept: text/html,application/xhtml+xml',
                              _EMAILNATOR_BASE + '/'])
        xsrf = ''
        try:
            with open(self._jar_path(), 'r', encoding='utf-8') as fh:
                for line in fh:
                    parts = line.rstrip('\n').split('\t')
                    if len(parts) >= 7 and parts[5] == 'XSRF-TOKEN':
                        xsrf = urllib.parse.unquote(parts[6])
        except OSError:
            pass
        # the API currently works without any session/CSRF; XSRF is acquired
        # only when a 403/419 says it is needed — never fail hard here
        self._xsrf = xsrf

    def _call(self, method: str, path: str,
              body: Optional[Dict[str, Any]] = None,
              timeout: int = 25) -> Tuple[int, Any]:
        args: List[str] = []
        if method == 'POST':
            args += ['-X', 'POST', '-H', 'Content-Type: application/json',
                     '-d', json.dumps(body or {})]
        args += ['-H', 'Accept: application/json',
                 '-H', 'X-Requested-With: XMLHttpRequest',
                 '-H', 'Referer: https://www.emailnator.com/',
                 _EMAILNATOR_BASE + path]
        if self._xsrf:
            args += ['-H', f'X-XSRF-TOKEN: {self._xsrf}']
        code, out = self._curl(args, timeout)
        if code in (403, 419):  # CSRF challenge -> acquire pair, retry once
            try:
                self._ensure()
            except Exception:  # noqa: BLE001
                return code, {'error': 'session refresh failed'}
            args = [a for a in args if not a.startswith('X-XSRF-TOKEN:')]
            if self._xsrf:
                args += ['-H', f'X-XSRF-TOKEN: {self._xsrf}']
            code, out = self._curl(args, timeout)
        try:
            parsed = json.loads(out) if out.strip() else {}
        except ValueError:
            parsed = {'error': 'non-json response'}
        return code, parsed

    def generate(self) -> Optional[str]:
        # 3 = dotGmail, 8 = googleMail — real gmail.com inbox variants.
        # id 2 (plusGmail, user+tag@gmail.com) is deliberately EXCLUDED:
        # DeepSeek's mail pipeline silently drops plus-addressed recipients,
        # so the OTP never arrives (verified live 2026-09-16). Dot-variants
        # and googlemail.com aliases are delivered.
        code, body = self._call('POST', '/api/generate-email', {'ids': [3, 8]})
        if code == 200 and isinstance(body, dict) \
                and str(body.get('status') or '') == 'success':
            address = str(body.get('email') or '').strip()
            return address or None
        return None

    def messages(self, address: str) -> List[Dict[str, Any]]:
        code, body = self._call('POST', '/api/message-list',
                                {'email': address, 'limit': 20})
        if code == 200 and isinstance(body, dict) \
                and isinstance(body.get('messages'), list):
            return [m for m in body['messages'] if isinstance(m, dict)]
        return []

    def message_body(self, mid: str) -> str:
        code, body = self._call(
            'GET', f"/api/message/{urllib.parse.quote(mid, safe='')}")
        if code == 200:
            if isinstance(body, dict):
                return '\n'.join(str(body.get(k) or '')
                                 for k in ('html', 'text', 'body', 'content'))
            return body if isinstance(body, str) else ''
        return ''


_EMAILNATOR = _Emailnator()


def _emailnator_create() -> Optional[Dict[str, Any]]:
    """Mint a real @gmail.com inbox via emailnator (no account needed)."""
    try:
        address = _EMAILNATOR.generate()
    except Exception:  # noqa: BLE001
        return None
    if not address:
        return None
    return {'backend': 'emailnator-gmail', 'address': address}


def _emailnator_fetch_otp(session: Dict[str, Any], sender_needle: str,
                          code_re: re.Pattern, max_age_min: float,
                          deadline: float, seen_ids: set,
                          after_ts: Optional[float] = None
                          ) -> Optional[str]:
    """Poll the emailnator gmail inbox for the verification code."""
    address = session.get('address') or ''
    while time.time() < deadline:
        try:
            msgs = _EMAILNATOR.messages(address)
        except Exception:  # noqa: BLE001
            msgs = []
        for msg in msgs:
            mid = str(msg.get('id') or '')
            if not mid or mid in seen_ids or msg.get('locked'):
                continue
            ts = _msg_ts(msg)
            # only skip a provably stale email: backends whose listing
            # carries no epoch (emailnator) must NOT be cut by after_ts —
            # ``float(0) <= after_ts`` silently skipped EVERY message and
            # the fetch timed out while the wanted email sat in the inbox.
            if after_ts is not None and ts is not None \
                    and ts <= after_ts - _AFTER_TS_GRACE_S:
                continue  # stale email: its code was already invalidated
            sender = str(msg.get('from') or '').lower()
            subject = str(msg.get('subject') or '').lower()
            if sender_needle and sender_needle not in sender \
                    and sender_needle not in subject:
                continue
            age = ((time.time() - ts) / 60.0) if ts is not None else None
            if age is not None and age > max_age_min:
                continue
            body = _EMAILNATOR.message_body(mid)
            if not body:
                # transient empty fetch (backend hiccup): leave the message
                # unblacklisted so the next pass retries it — a one-off
                # failure must not silence the inbox for the whole window
                continue
            seen_ids.add(mid)
            match = code_re.search(body)
            if match:
                return match.group(1) or match.group(0)
        time.sleep(6)
    return None


# ------------------------------------------------------------- IMAP catch-all
def _imap_catchall_create() -> Optional[Dict[str, Any]]:
    domain = os.getenv('I4F_MAIL_DOMAIN', '').strip()
    host = os.getenv('I4F_MAIL_IMAP_HOST', '').strip()
    if not domain or not host:
        return None
    address = f"{_gen_local_part()}@{domain}"
    return {'backend': 'imap-catchall', 'address': address}


def _imap_fetch_otp(session: Dict[str, Any], sender_needle: str,
                    code_re: re.Pattern, max_age_min: float,
                    deadline: float, seen_ids: set,
                    after_ts: Optional[float] = None) -> Optional[str]:
    from . import refresher
    # late import avoids a circular dependency (refresher imports mailgen)
    return refresher.imap_otp(max_wait_s=max(1, int(deadline - time.time())),
                              to_needle=session['address'])


# --------------------------------------- tempmail.plus (implicit inbox)
TEMPMAILPLUS_API = 'https://tempmail.plus/api/mails/'
_TEMPMAILPLUS_DOMAIN = 'mailto.plus'


def _tempmailplus_create() -> Optional[Dict[str, Any]]:
    """Mint an inbox on tempmail.plus (@mailto.plus).

    The mailbox is implicit — any local part on the service's domain has an
    inbox — so creation is a reachability check of the listing API for the
    exact address being handed out (no account, no session token).
    """
    address = f'{_gen_local_part()}@{_TEMPMAILPLUS_DOMAIN}'
    code, body = _http('GET', f'{TEMPMAILPLUS_API}?email='
                       f'{urllib.parse.quote(address)}&first_id=0')
    if code == 200 and isinstance(body, dict) and body.get('result'):
        return {'backend': 'tempmail.plus', 'address': address}
    return None


def _tempmailplus_messages(session: Dict[str, Any]) -> List[Dict[str, Any]]:
    """List the tempmail.plus inbox in the common message shape.

    The listing carries text/html inline, so the body cache is filled here
    and ``_backend_body`` serves both OTP and magic-link flows. The display
    ``time`` field has no reliable timezone — emit timestamp None (the
    fresh random local part makes stale-mail cuts unnecessary).
    """
    address = session.get('address') or ''
    code, body = _http('GET', f'{TEMPMAILPLUS_API}?email='
                       f'{urllib.parse.quote(address)}&first_id=0')
    if code != 200 or not isinstance(body, dict):
        return []
    cache = _BODY_CACHE.setdefault(address, {})
    while len(_BODY_CACHE) > 32:  # throwaway mailboxes: bound the cache
        _BODY_CACHE.pop(next(iter(_BODY_CACHE)), None)
    out: List[Dict[str, Any]] = []
    for msg in body.get('mail_list') or []:
        if not isinstance(msg, dict):
            continue
        mid = str(msg.get('mail_id') or '')
        if not mid:
            continue
        cache[mid] = '\n'.join(str(msg.get(k) or '')
                                for k in ('text', 'html', 'subject'))
        out.append({'id': mid, 'from': msg.get('from') or '',
                    'subject': msg.get('subject') or '',
                    'timestamp': None, 'locked': False})
    return out


# ------------------------------------------------ temp-mail.io (v3 API)
TEMPMAILIO_API = 'https://api.internal.temp-mail.io/api/v3'


def _tempmailio_create() -> Optional[Dict[str, Any]]:
    """Mint a mailbox on temp-mail.io (rotating obscure-domain pool)."""
    code, body = _http('POST', f'{TEMPMAILIO_API}/email/new',
                       {'min_name_length': 10, 'max_name_length': 10})
    if code == 200 and isinstance(body, dict):
        address = str(body.get('email') or '').strip()
        if address:
            return {'backend': 'temp-mail.io', 'address': address,
                    'token': str(body.get('token') or '')}
    return None


def _tempmailio_messages(session: Dict[str, Any]) -> List[Dict[str, Any]]:
    """List the temp-mail.io inbox in the common message shape.

    GET /email/{email}/messages returns every message with the body inline
    ('body_text'/'body_html') and an ISO-8601 'created_at'.
    """
    address = session.get('address') or ''
    code, body = _http('GET', f'{TEMPMAILIO_API}/email/'
                       f'{urllib.parse.quote(address)}/messages')
    if code != 200 or not isinstance(body, list):
        return []
    cache = _BODY_CACHE.setdefault(address, {})
    while len(_BODY_CACHE) > 32:  # throwaway mailboxes: bound the cache
        _BODY_CACHE.pop(next(iter(_BODY_CACHE)), None)
    out: List[Dict[str, Any]] = []
    for msg in body:
        if not isinstance(msg, dict):
            continue
        mid = str(msg.get('id') or '')
        if not mid:
            continue
        cache[mid] = '\n'.join(str(msg.get(k) or '')
                                for k in ('body_text', 'body_html', 'subject'))
        age = _iso_age_min(msg.get('created_at'))
        frm = msg.get('from')
        sender = (frm.get('address') or '') if isinstance(frm, dict) \
            else str(frm or '')
        out.append({'id': mid, 'from': sender,
                    'subject': msg.get('subject') or '',
                    'timestamp': (time.time() - age * 60.0)
                    if age is not None else None,
                    'locked': False})
    return out


# ---------------------------------------------------- Guerrilla Mail API
GUERRILLA_API = 'https://api.guerrillamail.com/ajax.php'


def _guerrillamail_create() -> Optional[Dict[str, Any]]:
    """Mint a Guerrilla Mail inbox (session-addressed via sid_token).

    A fresh random user is claimed through ``set_email_user`` so the
    mailbox is not the shared default, and the ``@sharklasers.com`` alias
    of the SAME inbox is handed out: every guerrilla domain delivers to
    one inbox and sharklasers.com sits in far fewer disposable-domain
    blocklists than the default guerrillamailblock.com.
    """
    code, body = _http('GET', f'{GUERRILLA_API}?f=get_email_address&lang=en')
    if code != 200 or not isinstance(body, dict):
        return None
    sid = str(body.get('sid_token') or '')
    if not sid:
        return None
    local = _gen_local_part().replace('-', '')  # guerrilla users: alnum only
    code2, body2 = _http('GET', f'{GUERRILLA_API}?f=set_email_user'
                         f'&email_user={local}&lang=en&sid_token={sid}')
    address = ''
    if code2 == 200 and isinstance(body2, dict):
        address = str(body2.get('email_addr') or '')
    if not address:
        address = str(body.get('email_addr') or '')
    if not address:
        return None
    aliased = f"{address.split('@', 1)[0]}@sharklasers.com"
    return {'backend': 'guerrillamail', 'address': aliased,
            'address_native': address, 'sid': sid}


def _guerrillamail_messages(session: Dict[str, Any]) -> List[Dict[str, Any]]:
    """List the Guerrilla Mail inbox in the common message shape."""
    sid = session.get('sid') or ''
    if not sid:
        return []
    code, body = _http('GET', f'{GUERRILLA_API}?f=get_email_list'
                       f'&offset=0&sid_token={sid}')
    if code != 200 or not isinstance(body, dict):
        return []
    out: List[Dict[str, Any]] = []
    for msg in body.get('list') or []:
        if not isinstance(msg, dict):
            continue
        mid = str(msg.get('mail_id') or '')
        if not mid:
            continue
        ts = None
        try:
            cand = float(msg.get('mail_timestamp'))
            ts = cand if cand > 1_000_000_000 else None
        except (TypeError, ValueError):
            ts = None
        if ts is None:
            md = str(msg.get('mail_date') or '').strip()
            if re.fullmatch(r'\d{2}:\d{2}:\d{2}', md):
                # today's mail carries a time-only stamp (guerrilla puts a
                # full date only on older mail); the clock is UTC. A stamp
                # that lands a hair in the future (clock skew) is harmless:
                # a negative age passes the max-age cut and after_ts never
                # cuts mail newer than the request.
                from datetime import datetime, timedelta, timezone
                hh, mm, ss = (int(x) for x in md.split(':'))
                now = datetime.now(timezone.utc)
                midnight = now.replace(hour=0, minute=0, second=0,
                                       microsecond=0)
                ts = (midnight + timedelta(hours=hh, minutes=mm,
                                           seconds=ss)).timestamp()
            else:
                age = _iso_age_min(md)
                ts = (time.time() - age * 60.0) if age is not None else None
        out.append({'id': mid, 'from': str(msg.get('mail_from') or ''),
                    'subject': str(msg.get('mail_subject') or ''),
                    'timestamp': ts, 'locked': False})
    return out


def _guerrillamail_body(session: Dict[str, Any], mid: str) -> str:
    """Fetch one guerrilla message body (lazy, cached like the others)."""
    cache = _BODY_CACHE.setdefault(session.get('address') or '', {})
    while len(_BODY_CACHE) > 32:  # throwaway mailboxes: bound the cache
        _BODY_CACHE.pop(next(iter(_BODY_CACHE)), None)
    if mid in cache:
        return cache[mid]
    sid = session.get('sid') or ''
    if not sid:
        return ''
    code, body = _http('GET', f'{GUERRILLA_API}?f=fetch_email'
                       f'&email_id={urllib.parse.quote(mid, safe="")}'
                       f'&sid_token={sid}')
    if code != 200 or not isinstance(body, dict):
        return ''
    subject = ''
    for msg in _guerrillamail_messages(session):
        if str(msg.get('id') or '') == mid:
            subject = str(msg.get('subject') or '')
            break
    text = '\n'.join(x for x in (subject, str(body.get('mail_body') or ''))
                     if x)
    cache[mid] = text
    return text


# ------------------------------------------- common OTP fetch (new pools)
def _common_fetch_otp(session: Dict[str, Any], sender_needle: str,
                      code_re: re.Pattern, max_age_min: float,
                      deadline: float, seen_ids: set,
                      after_ts: Optional[float] = None) -> Optional[str]:
    """Backend-agnostic OTP poll over the common message shape.

    Serves every backend whose listing/body flows through
    ``_backend_messages``/``_backend_body`` (tempmail.plus, temp-mail.io,
    guerrillamail). Same shared-inbox hygiene as the emailnator fetcher:
    skip locked messages, honour ``after_ts`` so a previous request's
    already-spent code is never reused, sender/subject needle, max-age
    cut, and a transient empty body fetch leaves the message unblacklisted
    so the next pass retries it.
    """
    while time.time() < deadline:
        try:
            msgs = _backend_messages(session)
        except Exception:  # noqa: BLE001
            msgs = []
        for msg in msgs:
            mid = str(msg.get('id') or '')
            if not mid or mid in seen_ids or msg.get('locked'):
                continue
            ts = _msg_ts(msg)
            if after_ts is not None and ts is not None \
                    and ts <= after_ts - _AFTER_TS_GRACE_S:
                continue  # stale email: its code was already invalidated
            sender = str(msg.get('from') or '').lower()
            subject = str(msg.get('subject') or '').lower()
            if sender_needle and sender_needle not in sender \
                    and sender_needle not in subject:
                continue
            age = ((time.time() - ts) / 60.0) if ts is not None else None
            if age is not None and age > max_age_min:
                continue
            body = _backend_body(session, mid)
            if not body:
                # transient empty fetch (backend hiccup): leave the message
                # unblacklisted so the next pass retries it — a one-off
                # failure must not silence the inbox for the whole window
                continue
            seen_ids.add(mid)
            match = code_re.search(body)
            if match:
                return match.group(1) or match.group(0)
        time.sleep(6)
    return None


# -------------------------------------------------------------------- public
def available() -> bool:
    """True when at least one backend is plausibly configured."""
    if not autogen_enabled():
        return False
    if (os.getenv('I4F_MAIL_DOMAIN', '').strip()
            and os.getenv('I4F_MAIL_IMAP_HOST', '').strip()):
        return True
    return True  # mail.tm needs no configuration


def create_email(domain_suffixes: Optional[Tuple[str, ...]] = None,
                 no_gmail: bool = False,
                 exclude_backends: Tuple[str, ...] = ()) -> Tuple[Optional[Dict[str, Any]], str]:
    """Create a throwaway mailbox. Returns (session, error).

    Session is a dict with backend/address and (for mail.tm) credentials.
    The caller passes ``session`` to :func:`fetch_otp` once the signup form
    has asked for the verification code.

    Backend order: the operator's catch-all IMAP domain (most reliable,
    needs configuration), then emailnator's REAL gmail.com inboxes (gmail
    is allowlisted where disposable domains are dropped), then tempmail.lol
    (rotating obscure domains), then the well-known mail.tm/mail.gw pools
    (often blocklisted by big providers).

    ``domain_suffixes``: when set, only mailboxes whose address ends with
    one of these suffixes are accepted and others are regenerated (a few
    tries). Google's signup outright rejects ``@googlemail.com`` — a pool
    alias of the same inbox — so the gemini rung filters for @gmail.com.
    """
    if not autogen_enabled():
        return None, 'I4F_MAIL_AUTOGEN disabled'
    backends = (_imap_catchall_create, _emailnator_create,
                _tempmail_create, _tempmailplus_create, _tempmailio_create,
                _guerrillamail_create, _mailtm_create)
    if no_gmail:
        # the emailnator pool is gmail-only: Google's signup treats a gmail
        # address as a username claim ("That username is taken") and rejects
        # the googlemail.com alias outright, so its rungs need a non-gmail
        # disposable (tempmail.lol domains pass Google's blocklist).
        backends = tuple(b for b in backends if b is not _emailnator_create)
    if exclude_backends:
        # callers rotating across backends after a delivery failure name the
        # spent maker by function name (e.g. ('_tempmail_create',))
        backends = tuple(b for b in backends
                         if b.__name__ not in exclude_backends)
    errors: List[str] = []

    def _accepted(session: Optional[Dict[str, Any]]) -> bool:
        if not session:
            return False
        if not domain_suffixes:
            return True
        addr = str(session.get('address') or '').lower()
        return any(addr.endswith(s) for s in domain_suffixes)

    for make in backends:
        for _ in range(4):
            try:
                session = make()
            except Exception as e:  # noqa: BLE001
                session = None
                errors.append(f'{make.__name__}: {type(e).__name__}: {e}')
            if _accepted(session):
                return session, ''
        errors.append(f'{make.__name__}: no mailbox matching '
                      f'{list(domain_suffixes or [])}')
    # public temp-mail backends rate-limit in bursts: one retry pass after a
    # short pause usually gets a mailbox without failing the whole signup
    time.sleep(3.0)
    for make in backends[1:]:  # retry the non-IMAP backends once
        for _ in range(4):
            try:
                session = make()
            except Exception as e:  # noqa: BLE001
                session = None
                errors.append(f'{make.__name__} retry: '
                              f'{type(e).__name__}: {e}')
            if _accepted(session):
                return session, ''
    return None, '; '.join(errors[-6:]) or 'no backend produced a mailbox'


def fetch_otp(session: Dict[str, Any], max_wait_s: int = 180,
              sender_needle: Optional[str] = None,
              code_re: Optional[re.Pattern] = None,
              max_age_min: float = 30.0,
              after_ts: Optional[float] = None) -> Optional[str]:
    """Block until the OTP lands in the generated mailbox (or timeout).

    ``after_ts``: only accept messages that ARRIVED after this unix timestamp.
    Critical for shared inboxes (emailnator gmail dot-variants host every bot
    account's mail): a verification email from an earlier request is still
    sitting in the inbox but its code was already invalidated when a new one
    was requested — without the cut the fetcher grabs the stale code and the
    verification silently fails.
    """
    if not session:
        return None
    sender_needle = (sender_needle
                     or os.getenv('I4F_MAIL_OTP_SENDER', 'deepseek')).strip().lower()
    code_re = code_re or re.compile(os.getenv('I4F_MAIL_OTP_REGEX', r'\b(\d{6})\b'))
    max_age_min = float(os.getenv('I4F_MAIL_OTP_MAX_AGE', str(max_age_min)) or max_age_min)
    fetcher = {'imap-catchall': _imap_fetch_otp,
               'emailnator-gmail': _emailnator_fetch_otp,
               'tempmail.lol': _tempmail_fetch_otp,
               'tempmail.plus': _common_fetch_otp,
               'temp-mail.io': _common_fetch_otp,
               'guerrillamail': _common_fetch_otp}.get(session['backend']) \
        or _mailtm_fetch_otp
    try:
        return fetcher(session, sender_needle, code_re, max_age_min,
                       time.time() + max_wait_s, set(), after_ts)
    except Exception as e:  # noqa: BLE001
        print(f'[mailgen] fetch_otp failed: {type(e).__name__}: {e}',
              file=__import__('sys').stderr)
        return None


_BODY_CACHE: Dict[str, Dict[str, str]] = {}


def _msg_ts(msg: Dict[str, Any]) -> Optional[float]:
    """Epoch timestamp of a listing entry, or None when it has none.

    Only a plausible epoch (> 2001-09) counts: a missing or display-string
    field must read as "unknown", never as 0 — an after_ts cut against 0
    would drop every message the backend cannot date.
    """
    try:
        ts = float(msg.get('timestamp'))
    except (TypeError, ValueError):
        return None
    return ts if ts > 1_000_000_000 else None


def _tempmail_messages(session: Dict[str, Any]) -> List[Dict[str, Any]]:
    """List the tempmail.lol inbox in the common message shape.

    One GET returns every message with its body inline, so both the
    listing and the body cache are filled here. ``timestamp`` is derived
    from the ISO date so :func:`fetch_magic_link`'s ``after_ts`` cut works
    the same way it does for the emailnator listing.
    """
    token = session.get('token') or ''
    code, body = _http('GET', f'{TEMPMAIL_API}/auth/{token}')
    if code != 200 or not isinstance(body, dict):
        return []
    cache = _BODY_CACHE.setdefault(session.get('address') or '', {})
    while len(_BODY_CACHE) > 32:  # throwaway mailboxes: bound the cache
        _BODY_CACHE.pop(next(iter(_BODY_CACHE)), None)
    out: List[Dict[str, Any]] = []
    for msg in body.get('email') or []:
        if not isinstance(msg, dict):
            continue
        mid = str(msg.get('date') or '') + str(msg.get('from') or '') \
            + str(msg.get('subject') or '')
        if not mid:
            continue
        text = '\n'.join(str(msg.get(k) or '')
                         for k in ('body', 'html', 'subject'))
        cache[mid] = text
        age = _iso_age_min(msg.get('date'))
        out.append({'id': mid, 'from': msg.get('from') or '',
                    'subject': msg.get('subject') or '',
                    'timestamp': (time.time() - age * 60.0)
                    if age is not None else None,
                    'locked': False})
    return out


def _mailtm_messages(session: Dict[str, Any]) -> List[Dict[str, Any]]:
    """List the mail.tm / mail.gw inbox in the common message shape."""
    api = str(session.get('api') or MAILTM_API)
    code, body = _http('GET', f'{api}/messages?page=1',
                       token=session.get('token') or '')
    if code != 200:
        return []
    cache = _BODY_CACHE.setdefault(session.get('address') or '', {})
    out: List[Dict[str, Any]] = []
    for msg in _items(body):
        if not isinstance(msg, dict):
            continue
        mid = str(msg.get('id') or '')
        if not mid:
            continue
        text = ''
        mcode, full = _http('GET', f'{api}/messages/{mid}',
                            token=session.get('token') or '')
        if mcode == 200 and isinstance(full, dict):
            text = str(full.get('text') or '')
            html = full.get('html')
            if isinstance(html, list):
                text += '\n' + '\n'.join(str(h) for h in html)
            elif isinstance(html, str):
                text += '\n' + html
            if not text:
                text = str(full.get('intro') or '')
        cache[mid] = text
        frm = msg.get('from')
        sender = (frm.get('address') or '') if isinstance(frm, dict) \
            else str(frm or '')
        age = _iso_age_min(msg.get('createdAt'))
        out.append({'id': mid, 'from': sender,
                    'subject': msg.get('subject') or '',
                    'timestamp': (time.time() - age * 60.0)
                    if age is not None else None,
                    'locked': False})
    return out


def _backend_messages(session: Dict[str, Any]) -> List[Dict[str, Any]]:
    """List inbox messages for the session's backend, common shape.

    Returns ``{id, from, subject, timestamp, locked}`` dicts. Every public
    backend is served — magic-link providers were assumed gmail-only, but
    the disposable pools (tempmail.lol's rotating obscure domains, the
    mail.tm/mail.gw pools) still deliver for a meaningful subset of them,
    so the link/code fetch should look in those inboxes too instead of
    silently polling nothing for the full timeout.
    """
    backend = session.get('backend')
    try:
        if backend == 'emailnator-gmail':
            return _EMAILNATOR.messages(session.get('address') or '')
        if backend == 'tempmail.lol':
            return _tempmail_messages(session)
        if backend == 'tempmail.plus':
            return _tempmailplus_messages(session)
        if backend == 'temp-mail.io':
            return _tempmailio_messages(session)
        if backend == 'guerrillamail':
            return _guerrillamail_messages(session)
        if backend in ('mail.tm', 'mail.gw'):
            return _mailtm_messages(session)
    except Exception:  # noqa: BLE001
        return []
    return []


def _backend_body(session: Dict[str, Any], mid: str) -> str:
    if session.get('backend') == 'emailnator-gmail':
        try:
            return _EMAILNATOR.message_body(mid)
        except Exception:  # noqa: BLE001
            return ''
    if session.get('backend') == 'guerrillamail':
        try:
            return _guerrillamail_body(session, mid)
        except Exception:  # noqa: BLE001
            return ''
    return (_BODY_CACHE.get(session.get('address') or '') or {}).get(
        str(mid), '')


def _inbox_digest(session: Dict[str, Any], limit: int = 5) -> str:
    """One-line inbox summary for failure details (what DID arrive?)."""
    try:
        msgs = _backend_messages(session)
    except Exception:  # noqa: BLE001
        return 'unreadable'
    if not msgs:
        return 'empty'
    parts = []
    for m in msgs[:limit]:
        sender = str(m.get('from') or '')[:40]
        subject = str(m.get('subject') or '')[:60]
        parts.append(f'{sender}|{subject}')
    return '; '.join(parts)


def debug_magic_candidates(session: Dict[str, Any], url_needle: str,
                           sender_re: str = r'claude|anthropic',
                           limit: int = 2) -> str:
    """Diagnostics for a magic-link miss: for the candidate message(s),
    was a body fetched at all, did it hold the needle, which URLs showed?"""
    try:
        msgs = _backend_messages(session) or []
    except Exception:  # noqa: BLE001
        return 'listing failed'
    pat = re.compile(sender_re, re.I)
    cands = [m for m in msgs
             if pat.search(str(m.get('from') or '') + ' '
                           + str(m.get('subject') or ''))][-limit:]
    if not cands:
        return 'no candidate message'
    out = []
    url_re = re.compile(r'https?://[^\s"\'<>]+')
    for m in cands:
        mid = str(m.get('id') or '')
        body = str(_backend_body(session, mid)) if mid else ''
        urls = url_re.findall(body) + [urllib.parse.unquote(e)
                                       for e in _ENC_URL_RE.findall(body)]
        hits = [u for u in urls if url_needle.lower() in u.lower()]
        out.append(f'len={len(body)} needle={"claude.ai" in body} '
                   f'urls={len(urls)} hit={bool(hits)} '
                   f'first={(urls[0][:70] if urls else "-")}')
    return ' | '.join(out)


def fetch_magic_link(session: Dict[str, Any], url_needle: str,
                     sender_needle: str = '', max_wait_s: int = 180,
                     max_age_min: float = 30.0,
                     after_ts: Optional[float] = None,
                     body_needle: str = '',
                     with_body: bool = False):
    """Poll the mailbox for a magic-link / verification URL.

    Returns the URL string, or — with ``with_body=True`` — a
    ``(url, body_text)`` tuple (the body lets callers fall back to a
    numeric code printed next to the link, e.g. claude's
    ``fallback_code_configuration`` 6-digit code).
    """
    """Poll the mailbox for a magic-link / verification URL and return it.

    Claude (and a growing set of providers) sign up with an emailed *link*
    rather than a numeric OTP: the email carries a one-time
    ``https://<host>/magic-link#token`` URL that logs the browser in when
    opened in the requesting session. This is the link analogue of
    :func:`fetch_otp` — same shared-inbox hygiene (skip locked messages,
    honour ``after_ts`` so a previous request's already-used link is never
    reused, sender/subject needle), but it extracts a URL instead of a code.
    """
    if not session:
        return None
    needle = (url_needle or '').lower()
    sender_needle = (sender_needle or '').strip().lower()
    deadline = time.time() + max_wait_s
    seen: set = set()    # body fetched OK and did not match: stop refetching
    empty: set = set()   # body fetch failed (transient): retry next pass
    url_re = re.compile(r'https?://[^\s"\'<>]+')
    while time.time() < deadline:
        for msg in _backend_messages(session):
            mid = str(msg.get('id') or '')
            if not mid or mid in seen or msg.get('locked'):
                continue
            # same provably-stale rule as fetch_otp: backends without an
            # epoch timestamp (emailnator) must not be cut by after_ts
            ts = _msg_ts(msg)
            if after_ts is not None and ts is not None \
                    and ts <= after_ts - _AFTER_TS_GRACE_S:
                continue
            sender = str(msg.get('from') or '').lower()
            subject = str(msg.get('subject') or '').lower()
            if sender_needle and sender_needle not in sender \
                    and sender_needle not in subject:
                continue
            age = ((time.time() - ts) / 60.0) if ts is not None else None
            if age is not None and age > max_age_min:
                continue
            body = str(_backend_body(session, mid))
            if not body:
                # transient empty fetch (backend hiccup): retry next pass —
                # blacklisting here silenced the inbox for the whole window
                # while the wanted email sat readable in the listing
                empty.add(mid)
                continue
            empty.discard(mid)
            seen.add(mid)
            if body_needle and body_needle.lower() not in body.lower():
                continue
            for url in url_re.findall(body):
                url = url.rstrip('.,;)')
                if needle and needle in url.lower():
                    return (url, body) if with_body else url
            # click-tracker wrappers carry the real link percent-encoded in
            # the query — scan for the encoded form and hand back the inner
            # link (https%3A%2F%2Fclaude.ai%2Fmagic-link%23token=...), not
            # the redirector URL
            for enc in _ENC_URL_RE.findall(body):
                try:
                    inner = urllib.parse.unquote(enc).rstrip('.,;)')
                except Exception:  # noqa: BLE001
                    continue
                if needle and needle in inner.lower():
                    return (inner, body) if with_body else inner
        time.sleep(6)
    if with_body:
        return None, ''
    return None


def main(argv: List[str]) -> int:  # pragma: no cover - CLI smoke test
    session, err = create_email()
    if not session:
        print(json.dumps({'ok': False, 'error': err}, indent=2))
        return 1
    print(json.dumps({'ok': True, 'backend': session['backend'],
                      'address': session['address'],
                      'hint': 'send a code to this address, then rerun with fetch'
                      if len(argv) < 2 else ''}, indent=2))
    if len(argv) > 1 and argv[1] == 'wait':
        print('waiting up to 120s for any code…')
        otp = fetch_otp(session, max_wait_s=120)
        print(json.dumps({'otp': otp}, indent=2))
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main(sys.argv))
