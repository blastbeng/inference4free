"""Offline tests for the dormant Arena.ai (ex LMArena) provider.

Covers what the live recon established (2026-10-08):
  - catalog parsing: arena='text' block filter, userSelectable-only,
    non-dict entries tolerated, publicName fallbacks, vision capability
  - dormant semantics: no session → available() False, list_models raises
    ProviderAuthError, stream raises before any HTTP call
  - session sources: ARENA_COOKIES env JSON (jar file read elsewhere)
  - uuid7 shape (RFC 9562 version/variant bits)
  - create-evaluation payload: mode direct, modelAId from catalog,
    recaptchaV3Token null, uuid7 ids
  - AI SDK UI message stream parsing: 0: text, g: reasoning, 3: error,
    d: finish, unknown codes ignored, empty stream → unavailable
  - error classification: 401/403 → auth, 429 → rate limit
  - wiring: router/selfheal/refresher registries

No network: http_get/http_post_stream are monkeypatched.

Run:  python tests/test_arena_provider.py     (or: pytest tests/)
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from dsk.providers import arena_provider as ap          # noqa: E402
from dsk.providers import router                        # noqa: E402
from dsk.providers.base import (                        # noqa: E402
    ProviderAuthError,
    ProviderError,
    ProviderRateLimitError,
    ProviderUnavailableError,
)
from dsk.refresher import REFRESH, SIGNUP, _has_creds, _jar_path  # noqa: E402
from dsk.selfheal import (                              # noqa: E402
    _EVIDENCE_PATTERNS,
    _MODULE_NAMES,
    _PROVIDER_CLASSES,
    _PROVIDER_MODULES,
    HEALABLE,
)

_RE = ap._RE_UUID


class _FakeResponse:
    def __init__(self, status_code=200, json_data=None, lines=None,
                 text='', headers=None):
        self.status_code = status_code
        self._json = json_data
        self._lines = lines or []
        self.text = text
        self.headers = headers or {}

    def json(self):
        if self._json is None:
            raise ValueError('no json')
        return self._json

    def iter_lines(self):
        for line in self._lines:
            yield line


CATALOG = [{
    'arena': 'text',
    'models': [
        {'id': 'uuid-gemini', 'publicName': 'gemini-3.6-flash',
         'organization': 'google', 'userSelectable': True,
         'capabilities': {'inputCapabilities': {'text': True, 'image': True},
                          'outputCapabilities': {'text': True}}},
        {'id': 'uuid-gpt', 'publicName': 'gpt-5.5-instant',
         'organization': 'openai', 'userSelectable': True,
         'capabilities': {'inputCapabilities': {'text': True}}},
        {'id': 'uuid-hidden', 'publicName': 'internal-only',
         'organization': 'x', 'userSelectable': False, 'capabilities': {}},
        'a stray string element (seen live)',
        {'id': 'uuid-ghost', 'publicName': 'ghost'},   # no userSelectable key
    ],
}, {
    'arena': 'webdev',  # non-text block must be skipped
    'models': [{'id': 'uuid-webdev', 'publicName': 'web-model',
                'userSelectable': True}],
}, 'another stray string']


def _set_env(value=None):
    """Point ARENA_COOKIES at value (or clear it). Returns the old value."""
    old = os.environ.get('ARENA_COOKIES')
    if value is None:
        os.environ.pop('ARENA_COOKIES', None)
    else:
        os.environ['ARENA_COOKIES'] = value
    return old


def _unset(old):
    if old is None:
        os.environ.pop('ARENA_COOKIES', None)
    else:
        os.environ['ARENA_COOKIES'] = old


def test_catalog_parse():
    models = ap._parse_catalog(CATALOG)
    names = [m['public_name'] for m in models]
    # userSelectable-only: hidden entry AND the key-less ghost are dropped
    assert names == ['gemini-3.6-flash', 'gpt-5.5-instant'], names
    gem = models[0]
    assert gem['upstream'] == 'uuid-gemini'
    assert gem['vision'] is True
    assert gem['organization'] == 'google'
    assert models[1]['vision'] is False
    assert ap._parse_catalog(None) == []
    assert ap._parse_catalog('nope') == []
    assert ap._parse_catalog([42, 'x']) == []


def test_uuid7_shape():
    for _ in range(50):
        u = ap.uuid7()
        assert _RE.match(u), u


def test_dormant_without_session():
    old = _set_env(None)
    try:
        p = ap.ArenaProvider()
        assert p.available() is False
        try:
            p.list_models()
            raise AssertionError('list_models should raise ProviderAuthError')
        except ProviderAuthError:
            pass
        try:
            list(p.stream('hi', model='gemini-3.6-flash'))
            raise AssertionError('stream should raise ProviderAuthError')
        except ProviderAuthError:
            pass
    finally:
        _unset(old)


def test_env_session_enables(monkeypatch=None):
    old = _set_env(json.dumps({'session': 'abc', 'cf_bm': 'x'}))
    orig_get = ap.http_get
    ap.http_get = lambda *a, **k: _FakeResponse(json_data=CATALOG,
                                                status_code=200)
    try:
        p = ap.ArenaProvider()
        assert p.available() is True
        models = p.list_models()
        ids = [m['id'] for m in models]
        assert ids == ['gemini-3.6-flash', 'gpt-5.5-instant']
        gem = models[0]
        assert gem['upstream_model'] == 'uuid-gemini'
        assert gem['vision'] is True and gem['image_gen'] is False
        # upstream model id (uuid) also resolves
        assert p._find_model('uuid-gpt')['public_name'] == 'gpt-5.5-instant'
        # catalog parsed once, cached on the instance
        assert p._find_model('nope') is None
    finally:
        ap.http_get = orig_get
        _unset(old)


def test_stream_payload_and_parsing(monkeypatch=None):
    old = _set_env(json.dumps({'session': 'abc'}))
    orig_get, orig_post = ap.http_get, ap.http_post_stream
    ap.http_get = lambda *a, **k: _FakeResponse(json_data=CATALOG,
                                                status_code=200)
    captured = {}

    def fake_post(url, headers=None, json_body=None, cookies=None,
                  timeout=600, proxies=None, no_proxy=False):
        captured.update({'url': url, 'headers': headers,
                         'body': json_body, 'cookies': cookies})
        return _FakeResponse(lines=[
            'f:{"messageId":"x"}',
            '0:"he"',
            '0:"llo"',
            'g:"thinking bit"',
            '2:[{"type":"data-something"}]',   # data frames ignored
            '9:{"toolCallId":"t"}',            # tool frames ignored
            'd:{"finishReason":"stop"}',
        ])

    ap.http_post_stream = fake_post
    try:
        p = ap.ArenaProvider()
        chunks = list(p.stream('say hi', model='gemini-3.6-flash'))

        # ---- payload shape (live recon contract)
        body = captured['body']
        assert captured['url'] == ap.CREATE_EVAL_URL
        assert body['mode'] == 'direct'
        assert body['modality'] == 'chat'
        assert body['modelAId'] == 'uuid-gemini'
        assert body['recaptchaV3Token'] is None
        assert body['userMessage']['content'] == 'say hi'
        assert body['userMessage']['experimental_attachments'] == []
        for key in ('id', 'userMessageId', 'modelAMessageId'):
            assert _RE.match(body[key]), (key, body[key])
        assert captured['cookies'] == {'session': 'abc'}
        assert captured['headers']['Origin'] == ap.ARENA_BASE_URL
        # ---- chunk stream
        assert [c['content'] for c in chunks[:3]] == ['he', 'llo',
                                                      'thinking bit']
        assert chunks[0]['type'] == 'text'
        assert chunks[2]['type'] == 'thinking'
        assert chunks[-1]['finish_reason'] == 'stop'
    finally:
        ap.http_get, ap.http_post_stream = orig_get, orig_post
        _unset(old)


def test_stream_error_frame_raises(monkeypatch=None):
    old = _set_env(json.dumps({'session': 'abc'}))
    orig_get, orig_post = ap.http_get, ap.http_post_stream
    ap.http_get = lambda *a, **k: _FakeResponse(json_data=CATALOG,
                                                status_code=200)

    def fake_post(*a, **k):
        return _FakeResponse(lines=['0:"partial"', '3:"quota exceeded"'])

    ap.http_post_stream = fake_post
    try:
        p = ap.ArenaProvider()
        try:
            list(p.stream('hi', model='gemini-3.6-flash'))
            raise AssertionError('3: error frame must raise')
        except ProviderError as e:
            assert 'quota exceeded' in str(e)
    finally:
        ap.http_get, ap.http_post_stream = orig_get, orig_post
        _unset(old)


def test_stream_empty_raises_unavailable(monkeypatch=None):
    old = _set_env(json.dumps({'session': 'abc'}))
    orig_get, orig_post = ap.http_get, ap.http_post_stream
    ap.http_get = lambda *a, **k: _FakeResponse(json_data=CATALOG,
                                                status_code=200)

    def fake_post(*a, **k):
        return _FakeResponse(lines=['2:{"junk":1}', 'f:{}'])

    ap.http_post_stream = fake_post
    try:
        p = ap.ArenaProvider()
        try:
            list(p.stream('hi', model='gemini-3.6-flash'))
            raise AssertionError('empty stream must raise Unavailable')
        except ProviderUnavailableError:
            pass
    finally:
        ap.http_get, ap.http_post_stream = orig_get, orig_post
        _unset(old)


def test_error_classification(monkeypatch=None):
    old = _set_env(json.dumps({'session': 'abc'}))
    orig_get, orig_post = ap.http_get, ap.http_post_stream
    ap.http_get = lambda *a, **k: _FakeResponse(json_data=CATALOG,
                                                status_code=200)
    try:
        p = ap.ArenaProvider()

        def fake_post_401(*a, **k):
            return _FakeResponse(status_code=401,
                                 text='{"message":"User not found"}')

        ap.http_post_stream = fake_post_401
        try:
            list(p.stream('hi', model='gemini-3.6-flash'))
            raise AssertionError('401 must raise ProviderAuthError')
        except ProviderAuthError as e:
            assert 'User not found' in str(e)

        def fake_post_429(*a, **k):
            return _FakeResponse(status_code=429, text='slow down')

        ap.http_post_stream = fake_post_429
        try:
            list(p.stream('hi', model='gemini-3.6-flash'))
            raise AssertionError('429 must raise ProviderRateLimitError')
        except ProviderRateLimitError:
            pass

        # unknown model in catalog → plain ProviderError, no HTTP call
        try:
            list(p.stream('hi', model='nonexistent-model'))
            raise AssertionError('unknown model must raise')
        except ProviderError as e:
            assert 'not in catalog' in str(e)

        # images unsupported (attachment format unverified upstream)
        try:
            list(p.stream('hi', model='gemini-3.6-flash',
                          images=[{'mime': 'image/png', 'data': b'x'}]))
            raise AssertionError('images must raise')
        except ProviderError as e:
            assert 'image' in str(e).lower()
    finally:
        ap.http_get, ap.http_post_stream = orig_get, orig_post
        _unset(old)


def test_wiring():
    assert ('arena', '.arena_provider', 'ArenaProvider') in \
        router.PROVIDER_MODULES
    assert router.OWNED_BY['arena'] == 'arena'
    assert router.PUBLIC_PREFIX['arena'] == 'arena'
    assert 'arena' in router.AUTO_CATEGORIES['general']
    assert HEALABLE['arena'].name == 'arena_provider.py'
    assert _PROVIDER_MODULES['arena'] == \
        _MODULE_NAMES['arena'] == 'dsk.providers.arena_provider'
    assert _PROVIDER_CLASSES['arena'] == 'ArenaProvider'
    assert any('create-evaluation' in pat
               for pat in _EVIDENCE_PATTERNS['arena'])
    old = _set_env(None)
    try:
        assert _has_creds('arena') is False      # dormant: no session yet
    finally:
        _unset(old)
    assert _jar_path('arena').name == 'arena_cookies.json'
    assert 'arena' in REFRESH and 'arena' in SIGNUP
    ok, detail = REFRESH['arena']()
    assert ok is False and 'arena_cookies.json' in detail
    ok2, detail2 = SIGNUP['arena']()
    assert ok2 is False and 'I4F_MAIL_DOMAIN' in detail2


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
