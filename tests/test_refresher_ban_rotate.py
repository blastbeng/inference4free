"""Offline tests for the deepseek ban/mute rotation in the renewal ladder.

The bug this file pins down (measured live 2026-10-09): the bot's deepseek
account was muted upstream (probe detail "deepseek account muted
(biz_code=5: user is muted)") and the ladder never recovered:

  - ``_permanent_auth_block`` matched only banned/suspended/deactivated —
    "muted" (deepseek's account-dead synonym) fell through, so the ladder
    burned its browser-login rung on a corpse that can never serve again;
  - ``signup_deepseek`` re-used the stored (bot-created, already-registered)
    email: a signup against a registered address can never create anything
    (observed live: the Send Code click fires NO send-code API call), so no
    OTP ever arrived and the rung failed forever.

Covered here:
  - _permanent_auth_block: muted probe detail -> True (rotate to signup);
    transient auth details ("token expired", "401") -> False; non-auth
    probe -> False; probe crash -> False
  - _deepseek_signup_creds: operator env pair wins and is never rotated;
    bot-stored creds rotate to a FRESH mailbox (generated=True, new
    address, mail session carried); no creds + autogen -> fresh mailbox;
    no creds + autogen off -> the explicit misconfiguration error;
    create_email failure -> the backend error, empty creds

No network, no browser: mailgen and the selfheal probe are faked.

Run:  python tests/test_refresher_ban_rotate.py     (or: pytest tests/)
"""
import contextlib
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if not os.path.isdir(os.path.join(ROOT, 'dsk')):
    ROOT = '/app'          # inside the container the code lives at /app
sys.path.insert(0, ROOT)

from dsk import mailgen                                  # noqa: E402
from dsk import refresher as R                           # noqa: E402
import dsk.selfheal as selfheal                          # noqa: E402

MUTED_DETAIL = 'deepseek account muted (biz_code=5: user is muted)'

_ENV_KEYS = ('DEEPSEEK_LOGIN_EMAIL', 'DEEPSEEK_LOGIN_PASSWORD',
             'I4F_MAIL_AUTOGEN', 'COOKIES_DIR')


@contextlib.contextmanager
def isolated():
    """Fresh COOKIES_DIR + deepseek credential env cleared."""
    with tempfile.TemporaryDirectory() as tmp:
        saved = {k: os.environ.get(k) for k in _ENV_KEYS}
        for k in _ENV_KEYS:
            os.environ.pop(k, None)
        os.environ['COOKIES_DIR'] = tmp
        try:
            yield tmp
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v


@contextlib.contextmanager
def probe(status, detail):
    """Fake the selfheal probe the ladder consults for ban detection."""
    real = selfheal._probe_once
    selfheal._probe_once = lambda name: (status, detail)
    try:
        yield
    finally:
        selfheal._probe_once = real


# ---------------------------------------------------------------- ban rotate

def test_muted_account_is_a_permanent_block():
    with isolated(), probe('auth', MUTED_DETAIL):
        assert R._permanent_auth_block('deepseek') is True


def test_transient_auth_is_not_a_permanent_block():
    with isolated(), probe('auth', 'deepseek token expired (401)'):
        assert R._permanent_auth_block('deepseek') is False


def test_non_auth_probe_is_not_a_permanent_block():
    with isolated(), probe('rate', '429 too many requests'):
        assert R._permanent_auth_block('deepseek') is False


def test_probe_crash_is_not_a_permanent_block():
    with isolated():
        real = selfheal._probe_once

        def boom(name):
            raise RuntimeError('probe exploded')

        selfheal._probe_once = boom
        try:
            assert R._permanent_auth_block('deepseek') is False
        finally:
            selfheal._probe_once = real


# ------------------------------------------------------------- signup creds

@contextlib.contextmanager
def _fake_mailgen(monkey_session):
    """Point mailgen.create_email/autogen_enabled at fakes."""
    real_create, real_auto = mailgen.create_email, mailgen.autogen_enabled
    mailgen.create_email = lambda *a, **k: monkey_session
    try:
        yield
    finally:
        mailgen.create_email, mailgen.autogen_enabled = real_create, real_auto


def test_operator_env_pair_is_never_rotated():
    with isolated():
        os.environ['DEEPSEEK_LOGIN_EMAIL'] = 'op@example.com'
        os.environ['DEEPSEEK_LOGIN_PASSWORD'] = 'pw'
        with _fake_mailgen(({'address': 'fresh@tm.io', 'password': 'x'},
                            None)):
            mailgen.autogen_enabled = lambda: True
            email, pw, session, generated, err = R._deepseek_signup_creds()
        assert (email, pw, generated, err) == ('op@example.com', 'pw',
                                               False, '')
        assert session is None            # nothing minted, nothing to save


def test_stored_bot_account_rotates_to_fresh_mailbox():
    with isolated():
        R._save_account('deepseek', 'corpse@gmail.com', 'deadpw',
                        'emailnator-gmail')
        fresh = {'backend': 'emailnator-gmail',
                 'address': 'fresh@gmail.com', 'password': 'newpw'}
        with _fake_mailgen((fresh, None)):
            mailgen.autogen_enabled = lambda: True
            email, pw, session, generated, err = R._deepseek_signup_creds()
        assert err == ''
        assert generated is True
        assert email == 'fresh@gmail.com'          # corpse NOT re-registered
        assert pw == 'newpw'
        assert session == fresh                    # OTP can be read later


def test_no_creds_with_autogen_mints_fresh_mailbox():
    with isolated():
        fresh = {'backend': 'tempmail.lol', 'address': 'a@t.lol',
                 'password': None}
        with _fake_mailgen((fresh, None)):
            mailgen.autogen_enabled = lambda: True
            email, pw, session, generated, err = R._deepseek_signup_creds()
        assert err == '' and generated is True
        assert email == 'a@t.lol'
        assert pw                                  # form password minted


def test_no_creds_without_autogen_reports_misconfiguration():
    with isolated():
        with _fake_mailgen(({'address': 'x@y'}, None)):
            mailgen.autogen_enabled = lambda: False
            email, pw, session, generated, err = R._deepseek_signup_creds()
        assert email == ''
        assert 'mail autogen off' in err


def test_mailgen_failure_surfaces_backend_error():
    with isolated():
        with _fake_mailgen((None, 'all backends down')):
            mailgen.autogen_enabled = lambda: True
            email, pw, session, generated, err = R._deepseek_signup_creds()
        assert email == ''
        assert 'all backends down' in err


if __name__ == '__main__':
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            try:
                fn()
                print(f'PASS {name}')
            except Exception as e:  # noqa: BLE001
                fails += 1
                print(f'FAIL {name}: {type(e).__name__}: {e}')
    print('ALL PASS' if not fails else f'{fails} FAILURES')
    sys.exit(1 if fails else 0)
