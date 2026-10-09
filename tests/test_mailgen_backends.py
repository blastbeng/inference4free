"""Offline tests for the three mail backends added to the signup ladder
(NOTA work: more email providers so no signup is blocked on one pool).

  - tempmail.plus: implicit inbox on @mailto.plus, no session/token
  - temp-mail.io:  v3 API, opaque email+token
  - guerrillamail: session-addressed (sid_token), sharklasers.com alias
  - _common_fetch_otp: shared-inbox hygiene (after_ts cut, sender needle,
    locked skip, transient-body retry, stale-code protection)

No network: module-level ``_http`` is monkeypatched.

Run:  python tests/test_mailgen_backends.py     (or: pytest tests/)
"""
import re
import sys
import time
import types
import urllib.parse

ROOT = __file__.rsplit('/', 2)[0]
sys.path.insert(0, ROOT)

from dsk import mailgen as mg                        # noqa: E402

# originals captured BEFORE any test monkeypatches module attributes —
# tests run alphabetically and the hygiene tests would otherwise leave
# their lambdas in place for every test sorted after them
_REAL_MESSAGES = mg._backend_messages
_REAL_BODY = mg._backend_body


# ----------------------------------------------------------------- helpers
class _FakeHttp:
    """Replace mg._http with a canned-response router."""

    def __init__(self, routes):
        # routes: list of (predicate_fn, response) checked in order
        self.routes = routes
        self.calls = []

    def __call__(self, method, url, body=None, token=None, timeout=25):
        self.calls.append((method, url, body))
        for pred, resp in self.routes:
            if pred(method, url, body):
                code, payload = resp
                return code, types.SimpleNamespace(**{}) if False else payload
        return 0, {'error': 'no route'}


def _with_http(routes, fn):
    global _last_http
    orig = mg._http
    fake = _FakeHttp(routes)
    mg._http = fake
    _last_http = fake
    try:
        return fn()
    finally:
        mg._http = orig


# ------------------------------------------------------------ tempmail.plus
def test_tempmailplus_create():
    def run():
        s = mg._tempmailplus_create()
        assert s and s['backend'] == 'tempmail.plus'
        assert s['address'].endswith('@mailto.plus')
        assert s['address'].split('@')[0]
        return s
    s = _with_http(
        [(lambda m, u, b: 'api/mails' in u, (200, {'result': True,
                                                  'mail_list': []}))],
        run)
    assert s is not None
    # the listing query must have used the exact address handed out
    # ('@' is percent-encoded in the query string)
    q = urllib.parse.quote(s['address'])
    assert any(s['address'] in u or q in u
               for _, u, _ in _last_http.calls)


def test_tempmailplus_messages_shape():
    listing = {'result': True, 'mail_list': [
        {'mail_id': 42, 'from': 'noreply@llm7.io',
         'subject': 'Your code', 'text': 'use 135790', 'html': '',
         'time': '2026-10-09 10:00:00'}]}
    routes = [(lambda m, u, b: 'api/mails' in u, (200, listing))]

    def run():
        session = {'backend': 'tempmail.plus',
                   'address': 'i4fabc123@mailto.plus'}
        msgs = mg._backend_messages(session)
        assert len(msgs) == 1
        m = msgs[0]
        assert str(m['id']) == '42'
        assert m['from'] == 'noreply@llm7.io'
        assert m['subject'] == 'Your code'
        assert m['locked'] is False
        body = mg._backend_body(session, '42')
        assert '135790' in body
        return True
    assert _with_http(routes, run)


# ------------------------------------------------------------- temp-mail.io
def test_tempmailio_create():
    def run():
        s = mg._tempmailio_create()
        assert s and s['backend'] == 'temp-mail.io'
        assert '@' in s['address'] and s['token']
        return s
    _with_http(
        [(lambda m, u, b: u.endswith('/email/new'),
          (200, {'email': 'f05p71mo9x@tanpony.com', 'token': 'TOK'}))],
        run)


def test_tempmailio_messages_shape():
    msgs_payload = [
        {'id': 'a1', 'from': 'verify@groq.com', 'subject': 'Verify',
         'body_text': 'code 246810', 'body_html': '',
         'created_at': '2026-10-09T07:12:33.000+00:00'}]
    routes = [(lambda m, u, b: '/messages' in u, (200, msgs_payload))]

    def run():
        session = {'backend': 'temp-mail.io',
                   'address': 'f05p71mo9x@tanpony.com'}
        msgs = mg._backend_messages(session)
        assert len(msgs) == 1
        m = msgs[0]
        assert m['id'] == 'a1' and m['from'] == 'verify@groq.com'
        assert m['timestamp'] is not None      # ISO created_at parsed
        assert '246810' in mg._backend_body(session, 'a1')
        return True
    assert _with_http(routes, run)


def test_tempmailio_from_dict_shape():
    # some deployments return from as {address: ...} — must still map
    msgs_payload = [{'id': 'a2', 'from': {'address': 'x@y.com'},
                     'subject': 'S', 'body_text': '1', 'body_html': '',
                     'created_at': None}]
    routes = [(lambda m, u, b: '/messages' in u, (200, msgs_payload))]

    def run():
        msgs = mg._backend_messages({'backend': 'temp-mail.io',
                                     'address': 'a@b.c'})
        assert msgs[0]['from'] == 'x@y.com'
        return True
    assert _with_http(routes, run)


# ----------------------------------------------------------- guerrillamail
def test_guerrillamail_create_aliases_sharklasers():
    def run():
        s = mg._guerrillamail_create()
        assert s and s['backend'] == 'guerrillamail'
        assert s['sid'] == 'SID1'
        # handed-out address uses the blocklist-friendlier alias domain
        assert s['address'].endswith('@sharklasers.com')
        # same local part as the native inbox
        assert s['address'].split('@')[0] == \
            s['address_native'].split('@')[0]
        return s
    _with_http(
        [(lambda m, u, b: 'get_email_address' in u,
          (200, {'email_addr': 'rand0m@guerrillamailblock.com',
                 'sid_token': 'SID1'})),
         (lambda m, u, b: 'set_email_user' in u,
          (200, {'email_addr': 'i4fabc12@guerrillamailblock.com',
                 'sid_token': 'SID1'}))],
        run)


def test_guerrillamail_create_falls_back_to_native():
    # set_email_user failure must still yield a usable native mailbox
    def run():
        s = mg._guerrillamail_create()
        assert s and s['address_native'] == 'rand0m@guerrillamailblock.com'
        assert s['address'].endswith('@sharklasers.com')
        return s
    _with_http(
        [(lambda m, u, b: 'get_email_address' in u,
          (200, {'email_addr': 'rand0m@guerrillamailblock.com',
                 'sid_token': 'SID2'})),
         (lambda m, u, b: 'set_email_user' in u, (0, {'error': 'down'}))],
        run)


def test_guerrillamail_messages_and_body():
    listing = {'list': [
        {'mail_id': '7', 'mail_from': 'no-reply@openrouter.ai',
         'mail_subject': 'Welcome', 'mail_excerpt': '',
         'mail_timestamp': 1791540000, 'mail_date': '2026-10-09 07:00'}]}
    fetch = {'mail_body': '<p>Your key: abc</p>'}
    routes = [
        (lambda m, u, b: 'get_email_list' in u, (200, listing)),
        (lambda m, u, b: 'fetch_email' in u, (200, fetch)),
    ]

    def run():
        session = {'backend': 'guerrillamail',
                   'address': 'i4fabc12@sharklasers.com', 'sid': 'SID1'}
        msgs = mg._backend_messages(session)
        assert [m['id'] for m in msgs] == ['7']
        assert msgs[0]['from'] == 'no-reply@openrouter.ai'
        assert msgs[0]['timestamp'] == 1791540000   # epoch passthrough
        body = mg._backend_body(session, '7')
        assert 'Your key' in body and 'Welcome' in body  # subject prepended
        # second read served from the cache (fetch called once)
        calls_before = len(mg._http.calls) if hasattr(mg._http, 'calls') else 0
        mg._backend_body(session, '7')
        calls_after = len(mg._http.calls) if hasattr(mg._http, 'calls') else 0
        assert calls_before == calls_after
        return True
    assert _with_http(routes, run)


def test_guerrillamail_timeonly_stamp():
    # today's guerrilla mail carries a time-only stamp + mail_timestamp 0:
    # the parser must recover a plausible epoch (UTC today)
    listing = {'list': [{'mail_id': '9', 'mail_from': 'a@b.c',
                         'mail_subject': 'S', 'mail_date': '08:06:19',
                         'mail_timestamp': 0}]}
    routes = [(lambda m, u, b: 'get_email_list' in u, (200, listing))]

    def run():
        msgs = mg._guerrillamail_messages({'backend': 'guerrillamail',
                                           'sid': 'X',
                                           'address': 'a@sharklasers.com'})
        ts = msgs[0]['timestamp']
        assert ts is not None
        now = time.time()
        assert now - 3600 * 23 < ts <= now + 130
        return True
    assert _with_http(routes, run)


# ------------------------------------------------- _common_fetch_otp logic
def test_common_fetch_otp_hygiene():
    fresh_ts = time.time() - 60          # 1 min old: passes the age cut
    stale_ts = time.time() - 3600        # 1 h old: after_ts must cut it
    good = {'id': 'g', 'from': 'noreply@llm7.io', 'subject': 'Code',
            'timestamp': fresh_ts, 'locked': False}
    stale = {'id': 's', 'from': 'noreply@llm7.io', 'subject': 'Code',
             'timestamp': stale_ts, 'locked': False}
    locked = {'id': 'l', 'from': 'noreply@llm7.io', 'subject': 'Code',
              'timestamp': fresh_ts, 'locked': True}
    other = {'id': 'o', 'from': 'newsletter@elsewhere.io', 'subject': 'x',
             'timestamp': fresh_ts, 'locked': False}
    mg._backend_messages = lambda session: [stale, locked, other, good]
    bodies = {'g': 'your code is 918273', 's': 'old 111222',
              'l': 'locked 333444', 'o': 'nothing 555666'}
    mg._backend_body = lambda session, mid: bodies.get(mid, '')
    try:
        got = mg._common_fetch_otp(
            {'backend': 'tempmail.plus', 'address': 'a@mailto.plus'},
            'llm7', re.compile(r'\b(\d{6})\b'), 30.0,
            time.time() + 3, set(),
            after_ts=time.time() - 600)   # anything older than 10 min is stale
    finally:  # the monkeypatch must not leak into the other tests
        mg._backend_messages, mg._backend_body = _REAL_MESSAGES, _REAL_BODY
    assert got == '918273'


def test_common_fetch_otp_transient_body_retries():
    fresh_ts = time.time() - 60
    msg = {'id': 'g', 'from': 'noreply@llm7.io', 'subject': 'Code',
           'timestamp': fresh_ts, 'locked': False}
    mg._backend_messages = lambda session: [msg]
    state = {'n': 0}

    def flaky_body(session, mid):
        state['n'] += 1
        return 'code 765432' if state['n'] >= 2 else ''  # 1st pass: hiccup

    mg._backend_body = flaky_body
    try:
        got = mg._common_fetch_otp(
            {'backend': 'temp-mail.io', 'address': 'a@b.c'},
            'llm7', re.compile(r'\b(\d{6})\b'), 30.0,
            time.time() + 10, set())
    finally:  # the monkeypatch must not leak into the other tests
        mg._backend_messages, mg._backend_body = _REAL_MESSAGES, _REAL_BODY
    assert got == '765432' and state['n'] >= 2


# ------------------------------------------------------------- wiring
def test_wiring():
    src = open(ROOT + '/dsk/mailgen.py', encoding='utf-8').read()
    for needle in ('_tempmailplus_create', '_tempmailio_create',
                   '_guerrillamail_create'):
        assert needle in src, needle
        # every new maker is part of the create_email rotation
        assert needle in src.split('def create_email', 1)[1], needle
    # dispatcher maps all three backends to the common fetcher
    disp = src.split('fetcher = {', 1)[1].split("}.get", 1)[0]
    for b in ("'tempmail.plus'", "'temp-mail.io'", "'guerrillamail'"):
        assert b in disp, b
    assert "'guerrillamail': _common_fetch_otp" in disp


if __name__ == '__main__':
    fns = [v for k, v in sorted(globals().items())
           if k.startswith('test_') and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f'PASS {fn.__name__}')
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f'FAIL {fn.__name__}: {type(e).__name__}: {e}')
    print(f'{len(fns) - failed}/{len(fns)} passed')
    sys.exit(1 if failed else 0)
