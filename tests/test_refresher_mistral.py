"""Offline tests for the mistral credential ladder (dsk/refresher.py).

The bug this file pins down: ``_has_creds('mistral')`` validated the Ory
session against ``chat.mistral.ai/api/v1/usage``. Measured 2026-10-09, that
endpoint answers 401 "Invalid API Key" to ANY session token (it is the
API-key console endpoint) and 403 "Just a moment..." (Cloudflare) to the
fixed cookie name — so the check was PERMANENTLY False. The daemon reads
False as "no credentials" and calls ``renew(name, reason='bootstrap')``,
which re-logged the account in on EVERY cycle (data/renew_mistral.log shows
exactly that: "refresh: no mistral credentials" -> "login: re-logged in via
HTTP Kratos"), while the chat endpoint itself was serving 200 the whole time.

The check that works: ``auth.mistral.ai/sessions/whoami`` with the jar's
DYNAMIC cookie name (``ory_session_<rand>``) sent as a Cookie.

Covered here:
  - _has_creds: 200 -> True; 401 (jar token) -> False (must escalate);
    network error -> True (a hiccup must not burn a fresh identity);
    env token with no jar cookie name -> True (operator-supplied, validated
    at request time); no token at all -> False
  - _has_creds probes whoami, NOT the api/v1/usage decoy, and replays the
    dynamic cookie name (the fixed 'ory_kratos_session' answers 401)
  - refresh_mistral: 200 + verified addresses -> True 'session valid'
  - refresh_mistral: 401 with a jar-sourced token -> False (real expiry)
  - refresh_mistral: 401 with an env token and no cookie name -> True
  - refresh_mistral: a non-401/403 answer is 'unverified', not 'invalid'
  - refresh_mistral: an unverified address converges via _mistral_verify_email
  - _http_get's second positional is COOKIES: the whoami call must pass the
    cookie as {name: token}, never as a header named 'Cookie'

No network: curl_cffi and _http_get are faked.

Run:  python tests/test_refresher_mistral.py     (or: pytest tests/)
"""
import contextlib
import os
import sys
import types

ROOT = '/opt/docker/compose/inference4free'
if not os.path.isdir(os.path.join(ROOT, 'dsk')):
    ROOT = '/app'          # inside the container the code lives at /app
sys.path.insert(0, ROOT)

from dsk import refresher as R                            # noqa: E402

WHOAMI = 'https://auth.mistral.ai/sessions/whoami'
JAR = {'session_token': 'tok-123', 'session_cookie_name': 'ory_session_abc'}


class _Resp:
    def __init__(self, status, payload=None):
        self.status_code = status
        self._payload = payload if payload is not None else {}

    def json(self):
        return self._payload


def _install_fake_curl(status, recorder, payload=None):
    """Fake ``curl_cffi.requests.get`` and hand it back to refresher."""
    def get(url, headers=None, timeout=0, **kw):
        recorder.append({'url': url, 'headers': dict(headers or {})})
        return _Resp(status, payload)

    mod = types.ModuleType('curl_cffi')
    requests_mod = types.ModuleType('curl_cffi.requests')
    requests_mod.get = get
    mod.requests = requests_mod
    sys.modules['curl_cffi'] = mod
    sys.modules['curl_cffi.requests'] = requests_mod


@contextlib.contextmanager
def _ctx(jar=None, env_token=None, whoami_status=200, payload=None):
    """Patch the jar/env/HTTP surface; yield the request recorder."""
    recorder = []
    old_jar = R._load_jar
    old_env = os.getenv('MISTRAL_SESSION_TOKEN')
    old_get = R._http_get
    R._load_jar = lambda name: dict(jar) if jar is not None else {}
    if env_token is not None:
        os.environ['MISTRAL_SESSION_TOKEN'] = env_token
    _install_fake_curl(whoami_status, recorder, payload)
    try:
        yield recorder
    finally:
        R._load_jar = old_jar
        R._http_get = old_get
        os.environ.pop('MISTRAL_SESSION_TOKEN', None)
        if old_env is not None:
            os.environ['MISTRAL_SESSION_TOKEN'] = old_env


# --------------------------------------------------------------- _has_creds
def test_has_creds_accepts_a_whoami_validated_session():
    with _ctx(jar=JAR) as rec:
        assert R._has_creds('mistral') is True
    assert rec[0]['url'] == WHOAMI


def test_has_creds_never_probes_the_api_usage_decoy():
    # api/v1/usage answers 401 "Invalid API Key" to every session token:
    # probing it made _has_creds permanently False and the ladder re-logged
    # the account in on every daemon cycle.
    with _ctx(jar=JAR) as rec:
        R._has_creds('mistral')
    assert all('api/v1/usage' not in r['url'] for r in rec)


def test_has_creds_replays_the_dynamic_cookie_name():
    with _ctx(jar=JAR) as rec:
        R._has_creds('mistral')
    cookie = rec[0]['headers'].get('Cookie', '')
    assert cookie == f"{JAR['session_cookie_name']}={JAR['session_token']}"
    # the fixed name is a measured 401 — it must not be what we send
    assert 'ory_kratos_session=' not in cookie


def test_has_creds_rejects_an_expired_jar_session():
    with _ctx(jar=JAR, whoami_status=401):
        assert R._has_creds('mistral') is False


def test_has_creds_tolerates_the_fixed_cookie_name_for_env_tokens():
    # A hand-imported token has no jar cookie name: whoami cannot replay it,
    # so a 401 here is not evidence of expiry.
    with _ctx(jar={}, env_token='env-tok', whoami_status=401):
        assert R._has_creds('mistral') is True


def test_has_creds_does_not_burn_an_identity_on_a_network_hiccup():
    def boom(url, headers=None, timeout=0, **kw):
        raise OSError('connection reset')

    mod = types.ModuleType('curl_cffi')
    requests_mod = types.ModuleType('curl_cffi.requests')
    requests_mod.get = boom
    mod.requests = requests_mod
    old = sys.modules.get('curl_cffi')
    sys.modules['curl_cffi'] = mod
    sys.modules['curl_cffi.requests'] = requests_mod
    old_jar = R._load_jar
    R._load_jar = lambda name: dict(JAR)
    try:
        assert R._has_creds('mistral') is True
    finally:
        R._load_jar = old_jar
        sys.modules.pop('curl_cffi', None)
        sys.modules.pop('curl_cffi.requests', None)
        if old is not None:
            sys.modules['curl_cffi'] = old


def test_has_creds_without_any_token_is_false():
    with _ctx(jar={}):
        assert R._has_creds('mistral') is False


# ---------------------------------------------------------- refresh_mistral
def test_refresh_accepts_a_verified_session():
    payload = {'identity': {'verifiable_addresses': [{'verified': True}]}}
    calls = []

    def fake_get(url, cookies, timeout=30, headers=None):
        calls.append({'url': url, 'cookies': dict(cookies or {})})
        return _Resp(200, payload)

    with _ctx(jar=JAR) as _:
        R._http_get = fake_get
        ok, detail = R.refresh_mistral()
    assert ok is True and 'session valid' in detail
    # _http_get's second positional is COOKIES: passing {'Cookie': ...}
    # would send a cookie literally named "Cookie" and read as an expiry.
    assert calls[0]['cookies'] == {JAR['session_cookie_name']: JAR['session_token']}
    assert 'Cookie' not in calls[0]['cookies']


def test_refresh_escalates_on_a_rejected_jar_token():
    def fake_get(url, cookies, timeout=30, headers=None):
        return _Resp(401)

    with _ctx(jar=JAR):
        R._http_get = fake_get
        ok, detail = R.refresh_mistral()
    assert ok is False and 'rejected' in detail


def test_refresh_accepts_an_env_token_whoami_cannot_replay():
    def fake_get(url, cookies, timeout=30, headers=None):
        return _Resp(401)

    with _ctx(jar={}, env_token='env-tok'):
        R._http_get = fake_get
        ok, detail = R.refresh_mistral()
    assert ok is True and 'unverified' in detail


def test_refresh_treats_an_unreachable_whoami_as_unverified():
    def fake_get(url, cookies, timeout=30, headers=None):
        return _Resp(503)

    with _ctx(jar=JAR):
        R._http_get = fake_get
        ok, detail = R.refresh_mistral()
    assert ok is True and 'unverified' in detail


def test_refresh_converges_an_unverified_address():
    payload = {'identity': {'verifiable_addresses': [{'verified': False,
                                                       'value': 'a@b.c'}]}}
    seen = {}

    def fake_get(url, cookies, timeout=30, headers=None):
        return _Resp(200, payload)

    with _ctx(jar=JAR):
        old = {k: getattr(R, k) for k in ('_mistral_verify_email',
                                          '_load_accounts', '_log_history')}
        R._mistral_verify_email = lambda email, session: (
            seen.update(email=email) or (True, 'OTP verified'))
        R._load_accounts = lambda: {'mistral': {'email': 'a@b.c', 'password': 'x'}}
        R._http_get = fake_get
        R._log_history = lambda *a, **k: None
        try:
            ok, detail = R.refresh_mistral()
        finally:
            for k, v in old.items():
                setattr(R, k, v)
    assert ok is True and 'verified' in detail
    assert seen['email'] == 'a@b.c'


def test_refresh_reports_a_failed_verification():
    payload = {'identity': {'verifiable_addresses': [{'verified': False,
                                                       'value': 'a@b.c'}]}}
    with _ctx(jar=JAR):
        old = {k: getattr(R, k) for k in ('_mistral_verify_email',
                                          '_load_accounts')}
        R._mistral_verify_email = lambda email, session: (False, 'no OTP')
        R._load_accounts = lambda: {'mistral': {'email': 'a@b.c', 'password': 'x'}}
        R._http_get = lambda url, cookies, timeout=30, headers=None: _Resp(200, payload)
        try:
            ok, detail = R.refresh_mistral()
        finally:
            for k, v in old.items():
                setattr(R, k, v)
    assert ok is False and 'no OTP' in detail


if __name__ == '__main__':
    fns = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f'PASS {fn.__name__}')
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f'FAIL {fn.__name__}: {type(e).__name__}: {e}')
    sys.exit(1 if failed else 0)
