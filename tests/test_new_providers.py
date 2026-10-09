"""Offline tests for the 5 new providers (meta, blackbox, t3chat,
innerai, adapta) — no network, fake HTTP only."""

import json
import os
import sys
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from dsk.providers import meta_provider as mp          # noqa: E402
from dsk.providers import blackbox_provider as bp      # noqa: E402
from dsk.providers import t3chat_provider as tp        # noqa: E402
from dsk.providers import innerai_provider as ip       # noqa: E402
from dsk.providers import adapta_provider as ap        # noqa: E402
from dsk.providers.base import (                        # noqa: E402
    ProviderAuthError,
    ProviderUnavailableError,
)


class _Resp:
    def __init__(self, status=200, text='', lines=None, data=None):
        self.status_code = status
        self.text = text
        self._lines = lines or []
        self.headers = {}
        self._data = data

    def json(self):
        if self._data is None:
            raise ValueError('no json')
        return self._data

    def iter_lines(self):
        for line in self._lines:
            yield line


def _env(name, value):
    old = os.environ.get(name)
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value
    return old


def _restore(name, old):
    if old is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = old


# ------------------------------------------------------------------- meta
def test_meta_catalog_parse_and_flags():
    data = {'data': [
        {'id': 'muse-spark-1.1', 'owned_by': 'meta',
         'context_window': 128000},
        {'id': 'muse-spark-1.1-vision', 'owned_by': 'meta'},
        {'id': 'muse-spark-r1-thinking'},
        {'id': '', 'owned_by': 'meta'},           # no id -> skipped
        {'active': False, 'id': 'retired'},       # inactive -> skipped
        'not-a-dict',
    ]}
    models = mp._parse_models(data)
    ids = [m['id'] for m in models]
    assert ids == ['muse-spark-1.1', 'muse-spark-1.1-vision',
                   'muse-spark-r1-thinking'], ids
    assert models[0]['context'] == 128000
    assert models[2]['owned_by'] == 'meta'
    print('PASS test_meta_catalog_parse_and_flags')


def test_meta_dormant_and_liveness():
    old = _env('META_API_KEY', None)
    try:
        p = mp.MetaProvider()
        assert not p.available()
        try:
            p.list_models()
            raise AssertionError('expected ProviderAuthError')
        except ProviderAuthError:
            pass
        # verify_key status mapping via fake http_get
        real = mp.http_get
        try:
            mp.http_get = lambda *a, **k: _Resp(
                status=401, data={'error': {'code': 'invalid_api_key'}})
            ok, detail = mp.verify_key('k')
            assert not ok and 'rejected' in detail
            mp.http_get = lambda *a, **k: _Resp(status=429)
            ok, detail = mp.verify_key('k')
            assert ok and '429' in detail
            mp.http_get = lambda *a, **k: _Resp(
                status=200, data={'data': [{'id': 'muse-spark-1.1'}]})
            ok, detail = mp.verify_key('k')
            assert ok and '1 models' in detail
        finally:
            mp.http_get = real
    finally:
        _restore('META_API_KEY', old)
    print('PASS test_meta_dormant_and_liveness')


def test_meta_stream_parsing():
    old = _env('META_API_KEY', 'test-key-123')
    real = mp.http_post_stream
    try:
        resp = _Resp(lines=[
            b'data: {"choices":[{"delta":{"content":"he"}}]}',
            b'data: {"choices":[{"delta":{"reasoning_content":"th"}}]}',
            b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}',
            b'data: [DONE]',
        ])
        mp.http_post_stream = lambda *a, **k: resp
        p = mp.MetaProvider()
        chunks = list(p.stream('hi', model='muse-spark-1.1'))
        kinds = [c['type'] for c in chunks]
        assert kinds == ['text', 'thinking', 'text', 'text'], kinds
        assert chunks[0]['content'] == 'he'
        assert chunks[1]['content'] == 'th'
        assert chunks[-1]['finish_reason'] == 'stop'
    finally:
        mp.http_post_stream = real
        _restore('META_API_KEY', old)
    print('PASS test_meta_stream_parsing')


# ---------------------------------------------------------------- blackbox
def test_blackbox_catalog_flags_and_dormant():
    p = bp.BlackboxProvider()
    old = _env('BLACKBOX_API_KEY', None)
    try:
        assert not p.available()
        try:
            p.list_models()
            raise AssertionError('expected ProviderAuthError')
        except ProviderAuthError:
            pass
        _env('BLACKBOX_API_KEY', 'bb-key-1')
        assert p.available()
        models = p.list_models()
        ids = [m['id'] for m in models]
        assert 'deepseek-reasoner' in ids and 'deepseek-chat' in ids
        by_id = {m['id']: m for m in models}
        assert by_id['deepseek-reasoner']['thinking_enabled'] is True
        assert by_id['deepseek-chat']['thinking_enabled'] is False
        assert by_id['deepseek-reasoner']['upstream_model'] == \
            'deepseek-reasoner'
    finally:
        _restore('BLACKBOX_API_KEY', old)
    print('PASS test_blackbox_catalog_flags_and_dormant')


def test_blackbox_verify_and_stream():
    real = bp.http_post_stream
    try:
        bp.http_post_stream = lambda *a, **k: _Resp(status=403)
        ok, detail = bp.verify_key('k')
        assert not ok and 'rejected' in detail
        bp.http_post_stream = lambda *a, **k: _Resp(status=429)
        ok, detail = bp.verify_key('k')
        assert ok and '429' in detail

        resp = _Resp(lines=[
            b'data: {"choices":[{"delta":{"reasoning":"pre"}}]}',
            b'data: {"choices":[{"delta":{"content":"ok"}}]}',
        ])
        bp.http_post_stream = lambda *a, **k: resp
        old = _env('BLACKBOX_API_KEY', 'bb-key-1')
        try:
            chunks = list(bp.BlackboxProvider().stream(
                'hi', model='deepseek-reasoner'))
            assert [c['type'] for c in chunks] == \
                ['thinking', 'text', 'text']
        finally:
            _restore('BLACKBOX_API_KEY', old)
    finally:
        bp.http_post_stream = real
    print('PASS test_blackbox_verify_and_stream')


# ------------------------------------------------------------------ t3chat
def test_t3chat_stream_protocols():
    # v5 SSE typed frames
    assert tp._parse_stream_line(
        b'data: {"type":"text-delta","delta":"hi"}') == ('text', 'hi')
    assert tp._parse_stream_line(
        b'data: {"type":"reasoning-delta","delta":"why"}') == \
        ('thinking', 'why')
    assert tp._parse_stream_line(
        b'data: {"type":"error","error":{"message":"boom"}}') == \
        ('error', "{'message': 'boom'}")
    # legacy custom-data lines
    assert tp._parse_stream_line(b'0:"Hel"') == ('text', 'Hel')
    assert tp._parse_stream_line(b'0:"He\",\"llo"') == ('text', 'Hello')
    assert tp._parse_stream_line(b'2:"[meta]"') is None
    assert tp._parse_stream_line(b'data: [DONE]') is None
    assert tp._parse_stream_line(b'') is None
    print('PASS test_t3chat_stream_protocols')


def test_t3chat_checkpoint_and_stream():
    assert tp._is_checkpoint(429, '<html>vercel checkpoint</html>')
    assert not tp._is_checkpoint(200, '{}')
    old_env = _env('T3CHAT_COOKIES', '{"session-token": "abc"}')
    old_post = tp.http_post_stream
    try:
        p = tp.T3ChatProvider()
        assert p.available()
        resp = _Resp(lines=[
            b'0:"Hel"', b'0:"lo"', b'2:"[done]"',
        ])
        tp.http_post_stream = lambda *a, **k: resp
        chunks = list(p.stream('hi', model='gpt-4o'))
        joined = ''.join(c['content'] for c in chunks
                         if c['type'] == 'text')
        assert 'Hello' in joined
        # checkpoint: 429 + HTML -> ProviderUnavailableError (not auth)
        tp.http_post_stream = lambda *a, **k: _Resp(
            status=429, text='<html>challenge</html>')
        try:
            list(p.stream('hi', model='gpt-4o'))
            raise AssertionError('expected ProviderUnavailableError')
        except ProviderUnavailableError as e:
            assert 'Checkpoint' in str(e)
    finally:
        tp.http_post_stream = old_post
        _restore('T3CHAT_COOKIES', old_env)
    print('PASS test_t3chat_checkpoint_and_stream')


def test_t3chat_dormant_and_catalog_scrape():
    old = _env('T3CHAT_COOKIES', None)
    try:
        p = tp.T3ChatProvider()
        assert not p.available()
        try:
            p.list_models()
            raise AssertionError('expected ProviderAuthError')
        except ProviderAuthError:
            pass
        # catalog scrape: page ids merged over the static fallback
        html = '"gpt-4o","gemini-2.5-flash","weird-new-model-1",' \
               '"icon.png","bundle.js"'
        models = tp._scrape_models(html)
        ids = [m['id'] for m in models]
        # only known-family ids are scraped (defensible anchor); unknown
        # families and static assets are ignored
        assert 'gpt-4o' in ids and 'gemini-2.5-flash' in ids
        assert 'weird-new-model-1' not in ids
        assert 'icon.png' not in ids and 'bundle.js' not in ids
        # static fallback survives a scrape that misses entries
        assert 'deepseek-r1' in ids
        models2 = tp._scrape_models('')
        assert {m['id'] for m in models2} == \
            {m['id'] for m in tp._STATIC_MODELS}
    finally:
        _restore('T3CHAT_COOKIES', old)
    print('PASS test_t3chat_dormant_and_catalog_scrape')


# ------------------------------------------------------- innerai / adapta
def test_skeletons_dormant_states():
    for mod, env_name in ((ip, 'INNERAI_COOKIES'), (ap, 'ADAPTA_COOKIES')):
        old = _env(env_name, None)
        try:
            p = mod.InnerAiProvider() if mod is ip else mod.AdaptaProvider()
            assert not p.available()
            try:
                p.list_models()
                raise AssertionError('expected ProviderAuthError')
            except ProviderAuthError:
                pass
            # cookies but no endpoint: still dormant, honest error
            _env(env_name, '{"session": "x"}')
            assert not p.available()
            try:
                p.list_models()
                raise AssertionError('expected ProviderUnavailableError')
            except ProviderUnavailableError:
                pass
        finally:
            _restore(env_name, old)
    print('PASS test_skeletons_dormant_states')


def test_skeletons_active_with_endpoint():
    old_c = _env('ADAPTA_COOKIES', '{"session": "x"}')
    old_u = _env('I4F_ADAPTA_CHAT_URL', 'https://adapta.org/api/chat')
    real = ap.http_post_stream
    try:
        p = ap.AdaptaProvider()
        assert p.available()
        # empty catalog still blocks a live call honestly
        try:
            p.list_models()
            raise AssertionError('expected ProviderUnavailableError')
        except ProviderUnavailableError:
            pass
        try:
            list(p.stream('hi', model='anything'))
            raise AssertionError('expected ProviderError (empty catalog)')
        except Exception as e:  # noqa: BLE001
            assert 'not in catalog' in str(e)
    finally:
        ap.http_post_stream = real
        _restore('ADAPTA_COOKIES', old_c)
        _restore('I4F_ADAPTA_CHAT_URL', old_u)
    print('PASS test_skeletons_active_with_endpoint')


# ----------------------------------------------------------------- wiring
def test_wiring():
    from dsk.providers import router as router_mod
    mods = {name for name, _m, _c in router_mod.PROVIDER_MODULES}
    for n in ('meta', 'blackbox', 't3chat', 'innerai', 'adapta'):
        assert n in mods, n
        assert router_mod.OWNED_BY.get(n)
        assert router_mod.PUBLIC_PREFIX.get(n)
    assert router_mod.PUBLIC_PREFIX['innerai'] == 'inner-ai'
    assert 'meta' in router_mod.AUTO_CATEGORIES['general']
    from dsk import refresher
    for n in ('meta', 'blackbox', 't3chat', 'innerai', 'adapta'):
        assert n in refresher.REFRESH, n
        assert n in refresher.SIGNUP, n
    from dsk import selfheal
    for n in ('meta', 'blackbox', 't3chat', 'innerai', 'adapta'):
        assert selfheal.HEALABLE.get(n), n
        assert selfheal._PROVIDER_CLASSES.get(n), n
        assert selfheal._EVIDENCE_PATTERNS.get(n), n
    print('PASS test_wiring')


if __name__ == '__main__':
    test_meta_catalog_parse_and_flags()
    test_meta_dormant_and_liveness()
    test_meta_stream_parsing()
    test_blackbox_catalog_flags_and_dormant()
    test_blackbox_verify_and_stream()
    test_t3chat_stream_protocols()
    test_t3chat_checkpoint_and_stream()
    test_t3chat_dormant_and_catalog_scrape()
    test_skeletons_dormant_states()
    test_skeletons_active_with_endpoint()
    test_wiring()
    print('ALL OK')
