"""Offline tests for the credential-handling logic of chatgpt/gemini.

Covers the "providers never worked / never autorenewed" fixes:
  - honest refresh rungs (expired sessions no longer read as "valid")
  - _has_creds / _load_jar: one credential source (jar + env merged)
  - browser-ladder helpers (dead X socket detection, zombie reaping)
  - provider token preference (session cookie > pasted bearer)
  - the /providers/{name}/credentials endpoint (Cookie-header paste
    parsing, token, email/password merge)

No network, no browser: the HTTP layers are monkeypatched.

Run:  python tests/test_credential_logic.py     (or: pytest tests/)
"""
import contextlib
import json
import os
import sys
import tempfile
import time
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import dsk.refresher as refresher                      # noqa: E402
from dsk.providers import chatgpt_provider             # noqa: E402
from dsk.providers import jar as provider_jar          # noqa: E402

_ENV_KEYS = ('CHATGPT_ACCESS_TOKEN', 'CHATGPT_SESSION_TOKEN',
             'CHATGPT_SESSION_COOKIES',
             'GEMINI_1PSID', 'GEMINI_1PSIDTS', 'I4F_CHATGPT_RELAY')


@contextlib.contextmanager
def isolated(tmp: "os.PathLike"):
    """COOKIES_DIR pointed at a fresh temp dir, credential env cleared."""
    saved = {k: os.environ.get(k) for k in _ENV_KEYS + ('COOKIES_DIR',)}
    for k in saved:
        os.environ.pop(k, None)
    os.environ['COOKIES_DIR'] = str(tmp)
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def write_jar(tmp, name, cookies):
    path = tmp / f'{name}_cookies.json'
    path.write_text(json.dumps(cookies), encoding='utf-8')
    return path


def fake_resp(status_code=200, url='https://example.test/x', text='',
              cookies=None, json_data=None):
    return types.SimpleNamespace(
        status_code=status_code, url=url, text=text,
        cookies=cookies or {},
        json=lambda d=json_data or {}: d,
    )


# ------------------------------------------------- refresher._has_creds

def test_has_creds_chatgpt_cf_only_is_false(tmp):
    with isolated(tmp):
        write_jar(tmp, 'chatgpt', {'__cf_bm': 'x', 'oai-did': 'y'})
        assert refresher._has_creds('chatgpt') is False


def test_has_creds_chatgpt_session_token(tmp):
    with isolated(tmp):
        write_jar(tmp, 'chatgpt',
                  {'__Secure-next-auth.session-token': 's', '__cf_bm': 'x'})
        assert refresher._has_creds('chatgpt') is True


def test_has_creds_chatgpt_bearer(tmp):
    with isolated(tmp):
        write_jar(tmp, 'chatgpt', {'accessToken': 't'})
        assert refresher._has_creds('chatgpt') is True


def test_has_creds_gemini(tmp):
    with isolated(tmp):
        write_jar(tmp, 'gemini', {'NID': 'x'})
        assert refresher._has_creds('gemini') is False
        write_jar(tmp, 'gemini', {'__Secure-1PSID': 'p', 'NID': 'x'})
        assert refresher._has_creds('gemini') is True


# ------------------------------------------- refresher._load_jar (env merge)

def test_load_jar_chatgpt_env_merge(tmp):
    with isolated(tmp):
        write_jar(tmp, 'chatgpt', {'__cf_bm': 'x'})
        os.environ['CHATGPT_ACCESS_TOKEN'] = 'env-bearer'
        jar = refresher._load_jar('chatgpt')
        assert jar.get('accessToken') == 'env-bearer'
        assert jar.get('__cf_bm') == 'x'
        os.environ.pop('CHATGPT_ACCESS_TOKEN')
        os.environ['CHATGPT_SESSION_TOKEN'] = 'legacy-bearer'
        assert refresher._load_jar('chatgpt').get('accessToken') == 'legacy-bearer'
        os.environ.pop('CHATGPT_SESSION_TOKEN')


def test_load_jar_chatgpt_session_cookies_env(tmp):
    with isolated(tmp):
        write_jar(tmp, 'chatgpt', {})
        # dict JSON form
        os.environ['CHATGPT_SESSION_COOKIES'] = json.dumps(
            {'__Secure-next-auth.session-token': 'st1'})
        jar = refresher._load_jar('chatgpt')
        assert jar.get('__Secure-next-auth.session-token') == 'st1'
        # list form (browser export)
        os.environ['CHATGPT_SESSION_COOKIES'] = json.dumps(
            [{'name': '__Secure-next-auth.csrf-token', 'value': 'c1'}])
        jar = refresher._load_jar('chatgpt')
        assert jar.get('__Secure-next-auth.csrf-token') == 'c1'
        # malformed JSON is ignored, not fatal (and the env value is gone,
        # so the earlier env-merged value is gone too)
        os.environ['CHATGPT_SESSION_COOKIES'] = '{not json'
        jar = refresher._load_jar('chatgpt')  # no exception
        assert '__Secure-next-auth.csrf-token' not in jar


# ------------------------------------- honest refresh rungs (no network)

@contextlib.contextmanager
def http_get(resp):
    """Point refresher._http_get at a fixed fake response (restored after)."""
    real = refresher._http_get
    refresher._http_get = lambda url, cookies, timeout=30, headers=None, **kw: resp
    try:
        yield
    finally:
        refresher._http_get = real


def test_refresh_gemini_honest(tmp):
    with isolated(tmp):
        # no credentials at all
        ok, detail = refresher.refresh_gemini()
        assert not ok and 'no gemini credentials' in detail

        write_jar(tmp, 'gemini', {'__Secure-1PSID': 'p'})

        # /app answers 200 even logged-out: no SNlM0e => expired, NOT valid
        with http_get(fake_resp(200, text='<html>anonymous shell</html>')):
            ok, detail = refresher.refresh_gemini()
            assert not ok and 'SNlM0e' in detail

        # rejected session
        with http_get(fake_resp(
                403, url='https://accounts.google.com/v3/signin/')):
            ok, detail = refresher.refresh_gemini()
            assert not ok and 'rejected' in detail

        # valid session, no rotation offered (the check wants the quoted
        # token exactly as the page emits it)
        with http_get(fake_resp(200, text='data "SNlM0e"="tok" here')):
            ok, detail = refresher.refresh_gemini()
            assert ok and 'valid' in detail

        # valid session WITH rotated cookies -> written back to the jar
        with http_get(fake_resp(200, text='"SNlM0e"="tok"',
                                cookies={'__Secure-1PSIDTS': 'rotated'})):
            ok, detail = refresher.refresh_gemini()
            assert ok and 'rotated' in detail
        jar = json.loads((tmp / 'gemini_cookies.json').read_text())
        assert jar.get('__Secure-1PSIDTS') == 'rotated'


def test_refresh_chatgpt_honest(tmp):
    real_http = refresher._http_get
    with isolated(tmp):
        write_jar(tmp, 'chatgpt', {})
        ok, detail = refresher.refresh_chatgpt()
        assert not ok and 'no chatgpt credentials' in detail

        # pasted bearer: cannot be rotated over HTTP, but it IS validated
        # live against backend-api/me (an unchecked bearer made the ladder
        # claim "renewed" while every generation 403'd downstream)
        write_jar(tmp, 'chatgpt', {'accessToken': 't'})
        called = []

        def _record(url, cookies, timeout=30, headers=None, **kw):
            called.append(1)
            assert headers and 'Bearer' in headers.get('Authorization', '')
            return fake_resp(200)

        refresher._http_get = _record
        ok, detail = refresher.refresh_chatgpt()
        assert ok and 'bearer valid' in detail
        assert called
        # bearer rejected live -> honest failure, no silent success
        called.clear()
        with http_get(fake_resp(403)):
            ok, detail = refresher.refresh_chatgpt()
            assert not ok and 'bearer rejected' in detail
        refresher._http_get = real_http  # restored by later context managers

        # session cookies rejected
        write_jar(tmp, 'chatgpt', {'__Secure-next-auth.session-token': 's'})
        with http_get(fake_resp(401)):
            ok, detail = refresher.refresh_chatgpt()
            assert not ok and 'rejected' in detail

        # 200 WITHOUT a token: expired (the old lie), not valid
        with http_get(fake_resp(200, json_data={})):
            ok, detail = refresher.refresh_chatgpt()
            assert not ok and 'cookies expired' in detail

        # 200 WITH a token: valid + rotation persisted (jar still holds the
        # session cookie from the previous case — the rung needs it)
        with http_get(fake_resp(200, json_data={'accessToken': 'fresh'},
                                cookies={'__Secure-next-auth.session-token': 'rotated'})):
            ok, detail = refresher.refresh_chatgpt()
            assert ok and 'valid' in detail
        jar = json.loads((tmp / 'chatgpt_cookies.json').read_text())
        assert jar.get('__Secure-next-auth.session-token') == 'rotated'
        assert jar.get('accessToken') == 'fresh'


def test_chatgpt_restore_token(tmp):
    with isolated(tmp):
        # a working bearer clobbered by a login-stage token -> restore
        write_jar(tmp, 'chatgpt', {'accessToken': 'old', 'oai-did': 'x'})
        refresher._chatgpt_restore_token({'accessToken': 'new'}, 'old')
        jar = refresher._load_jar('chatgpt')
        assert jar.get('accessToken') == 'old'
        assert jar.get('oai-did') == 'x'  # other cookies untouched
        # no previous bearer + login-stage token present -> drop it
        write_jar(tmp, 'chatgpt', {'accessToken': 'stage'})
        refresher._chatgpt_restore_token({'accessToken': 'stage'}, '')
        assert not refresher._load_jar('chatgpt').get('accessToken')
        # same token -> left alone
        write_jar(tmp, 'chatgpt', {'accessToken': 'same'})
        refresher._chatgpt_restore_token({'accessToken': 'same'}, 'same')
        assert refresher._load_jar('chatgpt').get('accessToken') == 'same'


# ------------------------------------------- browser ladder helpers

def test_x_display_alive(tmp):
    import socket
    import glob
    sock_path = '/tmp/.X11-unix/X97'
    try:
        os.unlink(sock_path)
    except FileNotFoundError:
        pass
    assert refresher._x_display_alive(97) is False

    # a socket that accepts connections counts as a live display server
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(sock_path)
    srv.listen(1)
    try:
        assert refresher._x_display_alive(97) is True
    finally:
        srv.close()
        for p in glob.glob('/tmp/.X11-unix/X97*'):
            try:
                os.unlink(p)
            except OSError:
                pass
    assert refresher._x_display_alive(97) is False


def test_reap_dead_children(tmp=None):
    pid = os.fork()
    if pid == 0:
        os._exit(0)
    time.sleep(0.2)  # let the child exit -> zombie until reaped
    refresher._reap_dead_children()
    try:
        waited, _ = os.waitpid(pid, os.WNOHANG)
        assert waited == 0, 'child was not reaped by _reap_dead_children'
    except ChildProcessError:
        pass  # already reaped — exactly what we want


# ------------------------------------- provider token preference

def test_chatgpt_available_and_bearer_preference(tmp):
    with isolated(tmp):
        # the anonymous browser relay serves generation without credentials,
        # so chatgpt is available even with no real session (same contract
        # as qwen: available = credentials OR relay)
        os.environ['I4F_CHATGPT_RELAY'] = '1'
        write_jar(tmp, 'chatgpt', {'__cf_bm': 'x'})
        assert chatgpt_provider.ChatGPTProvider().available() is True

        # relay disabled: only real credentials count (a cf_bm cookie alone
        # is not a session)
        os.environ['I4F_CHATGPT_RELAY'] = '0'
        write_jar(tmp, 'chatgpt', {'__cf_bm': 'x'})
        assert chatgpt_provider.ChatGPTProvider().available() is False

        write_jar(tmp, 'chatgpt',
                  {'__Secure-next-auth.session-token': 's'})
        assert chatgpt_provider.ChatGPTProvider().available() is True

        # jar bearer only: used as-is, no session request issued
        write_jar(tmp, 'chatgpt', {'accessToken': 'past-bearer'})
        p = chatgpt_provider.ChatGPTProvider()
        assert p.available() is True
        assert p._get_access_token() == 'past-bearer'


def test_chatgpt_env_token_wins(tmp):
    with isolated(tmp):
        write_jar(tmp, 'chatgpt', {'accessToken': 'jar-bearer'})
        os.environ['CHATGPT_ACCESS_TOKEN'] = 'env-bearer'
        p = chatgpt_provider.ChatGPTProvider()
        assert p._get_access_token() == 'env-bearer'


# ------------- /providers/{name}/credentials endpoint (paste, merge, save)

class _FakeRequest:
    def __init__(self, body):
        self.headers = {}
        self._body = body

    async def json(self):
        return self._body


@contextlib.contextmanager
def _patched_endpoint(tmp):
    import dsk.openai_server as server
    with isolated(tmp):
        real_verify = refresher._verify
        real_rediscover = server.ROUTER.refresh_models
        real_api_key = server.API_KEY
        refresher._verify = lambda name: 'ok'
        server.ROUTER.refresh_models = lambda force=False: None
        server.API_KEY = ''   # endpoint check reads the module constant
        try:
            yield server
        finally:
            refresher._verify = real_verify
            server.ROUTER.refresh_models = real_rediscover
            server.API_KEY = real_api_key


def call_endpoint(server, name, body):
    import asyncio
    return asyncio.run(server.provider_credentials_set(name, _FakeRequest(body)))


def test_credentials_set_cookie_header_paste(tmp):
    with _patched_endpoint(tmp) as server:
        result = call_endpoint(server, 'chatgpt', {
            'cookies': ('oai-did=abc; __Secure-next-auth.session-token=st-1; '
                       '__cf_bm=noise\noai-did=abc')
        })
        assert result['saved'] is True
        assert result.get('verify') == 'ok'
        jar = provider_jar.load_jar('chatgpt')
        assert jar.get('__Secure-next-auth.session-token') == 'st-1'
        assert jar.get('oai-did') == 'abc'
        assert jar.get('__cf_bm') == 'noise'
        assert jar.get('noise') is None  # fragment without '=' is dropped


def test_credentials_set_token_and_dict_cookies(tmp):
    with _patched_endpoint(tmp) as server:
        result = call_endpoint(server, 'chatgpt', {
            'token': 'tok-1',
            'cookies': {'__Secure-next-auth.csrf-token': 'c-1'},
        })
        assert result['saved'] is True
        jar = provider_jar.load_jar('chatgpt')
        assert jar.get('accessToken') == 'tok-1'
        assert jar.get('__Secure-next-auth.csrf-token') == 'c-1'


def test_credentials_set_email_password_merge(tmp):
    with _patched_endpoint(tmp) as server:
        # full pair
        r1 = call_endpoint(server, 'chatgpt',
                           {'email': 'a@b.c', 'password': 'p1'})
        assert r1['saved'] is True
        # password only: must MERGE with the stored email, not orphan it
        r2 = call_endpoint(server, 'chatgpt', {'password': 'p2'})
        assert r2['saved'] is True
        email, password = refresher._creds('chatgpt')
        assert email == 'a@b.c' and password == 'p2'
        # email only: likewise
        r3 = call_endpoint(server, 'chatgpt', {'email': 'x@y.z'})
        assert r3['saved'] is True
        email, password = refresher._creds('chatgpt')
        assert email == 'x@y.z' and password == 'p2'


def test_credentials_set_empty_body_saves_nothing(tmp):
    with _patched_endpoint(tmp) as server:
        result = call_endpoint(server, 'chatgpt', {})
        assert result['saved'] is False
        assert 'hint' in result


# ------------------------------------- escalation vs. cooldown (regression)

@contextlib.contextmanager
def _ladder_stub(tmp, fail_for=('chatgpt',)):
    """Patch REFRESH (no network) and _renew_locked (records the reason the
    ladder was entered with); snapshot/restore the refresher singleton."""
    saved_refresh = dict(refresher.REFRESH)
    saved_state = (dict(refresher._STATE.results),
                   dict(refresher._STATE.counts),
                   refresher._STATE.counts_seeded,
                   dict(refresher._STATE.renewing))
    calls = []

    def _stub(name):
        def rung():
            if name in fail_for:
                return False, 'stub: credential expired'
            return True, 'stub: valid'
        return rung

    def _locked(name, reason):
        calls.append((name, reason))
        return {'renewed': True, 'via': 'stub', 'reason': reason}

    refresher.REFRESH = {n: _stub(n) for n in refresher.REFRESH}
    refresher._renew_locked = _locked
    refresher._STATE.results.pop('chatgpt', None)
    refresher._STATE.renewing.clear()
    try:
        yield calls
    finally:
        refresher.REFRESH = saved_refresh
        (refresher._STATE.results, refresher._STATE.counts,
         refresher._STATE.counts_seeded,
         refresher._STATE.renewing) = saved_state


def test_renew_escalate_bypasses_self_armed_cooldown(tmp):
    """A failed cheap-refresh logs 'refresh-issue', which stamps the provider
    result with a FRESH timestamp; the ladder escalation must not be skipped
    by the cooldown that failure itself armed (it silently defeated the
    'has credentials but never renews' fix)."""
    with isolated(tmp):
        write_jar(tmp, 'chatgpt', {'accessToken': 't'})
        with _ladder_stub(tmp, fail_for=('chatgpt',)) as calls:
            # plain proactive reason keeps the cooldown throttle
            refresher._STATE.results['chatgpt'] = {
                'ts': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
                'provider': 'chatgpt', 'event': 'refresh-issue',
                'detail': 'prior failure'}
            res = refresher.renew('chatgpt', reason='proactive')
            assert res.get('skipped') == 'cooldown'
            assert not calls

            # escalate = the daemon handing ITS OWN failure to the ladder:
            # must run despite the fresh timestamp
            res = refresher.renew('chatgpt', reason='escalate')
            assert res.get('renewed') is True
            assert calls == [('chatgpt', 'escalate')]


def test_refresh_cycle_escalates_failed_refresh(tmp):
    with isolated(tmp):
        write_jar(tmp, 'chatgpt', {'accessToken': 't'})
        # qwen WITH credentials: its stub refresh succeeds, so it must
        # never enter the ladder (credentialed + healthy = refresh only)
        write_jar(tmp, 'qwen', {'token': 't'})
        with _ladder_stub(tmp, fail_for=('chatgpt',)) as calls:
            out = refresher.refresh_cycle()
            assert out.get('chatgpt', {}).get('renewed') is True, out.get('chatgpt')
            assert ('chatgpt', 'escalate') in calls
            assert out.get('qwen') == 'stub: valid', out.get('qwen')
            assert not any(n == 'qwen' for n, _ in calls)


# ------------------------------------------------------------- runner

def _main() -> int:
    import shutil
    import traceback
    from pathlib import Path
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith('test_') and callable(f)]
    failed = 0
    for name, fn in tests:
        tmp = tempfile.mkdtemp(prefix='i4f-test-')
        try:
            fn(Path(tmp))
            print(f'PASS  {name}')
        except Exception:
            failed += 1
            print(f'FAIL  {name}')
            traceback.print_exc()
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    print(f'\n{len(tests) - failed}/{len(tests)} passed')
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(_main())
