"""Offline tests for the HuggingChat provider (no network)."""

import contextlib
import io
import json
import os
import sys
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from dsk.providers import huggingchat_provider as hp  # noqa: E402
from dsk.providers.base import (  # noqa: E402
    ProviderAuthError,
    ProviderError,
    ProviderUnavailableError,
)


def _set_env(value):
    old = os.environ.get('HUGGINGCHAT_COOKIES')
    os.environ['HUGGINGCHAT_COOKIES'] = value
    return old


def _unset(old):
    if old is None:
        os.environ.pop('HUGGINGCHAT_COOKIES', None)
    else:
        os.environ['HUGGINGCHAT_COOKIES'] = old


def test_catalog_parse():
    data = {'json': [
        {'id': 'openai/gpt-oss-120b', 'displayName': 'gpt-oss-120b',
         'multimodal': False, 'supportsReasoning': True,
         'supportsTools': True, 'unlisted': False, 'isRouter': False},
        {'id': 'omni', 'displayName': 'Omni', 'isRouter': True},
        {'id': 'legacy/model', 'displayName': 'legacy', 'unlisted': True},
        {'id': 'google/gemma-4-31b-it', 'displayName': 'gemma',
         'multimodal': True, 'supportsReasoning': False,
         'unlisted': False, 'isRouter': False},
        'not-a-dict',
        {'displayName': 'no-id'},
    ]}
    models = hp._parse_catalog(data)
    assert [m['upstream'] for m in models] == ['openai/gpt-oss-120b',
                                               'google/gemma-4-31b-it']
    assert models[0]['thinking'] is True and models[0]['vision'] is False
    assert models[1]['vision'] is True and models[1]['thinking'] is False
    print('PASS test_catalog_parse')


def test_dormant_without_session():
    old = _set_env('')
    hp.load_jar_orig = None
    try:
        # no env, empty jar
        import dsk.providers.jar as jar_mod
        orig = jar_mod.load_jar
        jar_mod.load_jar = lambda name: {}
        hp.load_jar = jar_mod.load_jar  # rebind the imported name
        try:
            p = hp.HuggingChatProvider()
            assert p.available() is False
            try:
                p.list_models()
                raise AssertionError('expected ProviderAuthError')
            except ProviderAuthError as e:
                assert 'huggingchat_cookies.json' in str(e)
        finally:
            jar_mod.load_jar = orig
            hp.load_jar = orig
    finally:
        _unset(old)
    print('PASS test_dormant_without_session')


def test_env_session_enables():
    old = _set_env(json.dumps({'hf-chat': 'abc', 'token': 'hf_x'}))
    try:
        p = hp.HuggingChatProvider()
        assert p.available() is True
    finally:
        _unset(old)
    print('PASS test_env_session_enables')


def _fake_create_response(status=200, payload=None, text=''):
    resp = types.SimpleNamespace()
    resp.status_code = status
    resp.text = text or json.dumps(payload or {})
    resp.headers = {}

    def _json():
        return json.loads(resp.text)

    resp.json = _json
    resp.content = b''
    return resp


def test_stream_payload_and_parsing():
    old = _set_env(json.dumps({'hf-chat': 'abc'}))
    import dsk.providers.jar as jar_mod
    orig_load = jar_mod.load_jar
    jar_mod.load_jar = lambda name: {}
    hp.load_jar = jar_mod.load_jar
    orig_post = hp.http_post_raw
    orig_get = hp.http_get
    captured = {}

    def fake_get(url, headers=None, **kw):
        captured['get_url'] = url
        return _fake_create_response(payload={'json': [
            {'id': 'openai/gpt-oss-120b', 'displayName': 'gpt-oss-120b',
             'multimodal': False, 'supportsReasoning': True,
             'supportsTools': True, 'unlisted': False, 'isRouter': False}]})

    def fake_post(url, data, headers=None, timeout=300, no_proxy=False):
        if url.endswith('/conversation'):
            captured['create_body'] = json.loads(data.decode())
            return _fake_create_response(payload={'conversationId': 'abc123'})
        captured['msg_url'] = url
        captured['msg_headers'] = headers
        body = data.decode()
        assert body.startswith('--i4f-') and 'name="data"' in body
        ndjson = '\n'.join([
            json.dumps({'type': 'status', 'status': 'started'}),
            json.dumps({'messageId': 'm1'}),
            json.dumps({'type': 'stream', 'token': 'Hello'}),
            json.dumps({'type': 'reasoning', 'token': 'thinking...'}),
            json.dumps({'type': 'stream', 'token': ' world'}),
            json.dumps({'type': 'finalAnswer', 'text': 'Hello world!',
                        'len': 12}),
            json.dumps({'type': 'status', 'status': 'finished'}),
        ]).encode()
        resp = _fake_create_response()
        resp.content = ndjson
        resp.text = ''
        return resp

    try:
        hp.http_get = fake_get
        hp.http_post_raw = fake_post
        p = hp.HuggingChatProvider()
        chunks = list(p.stream('hi', model='openai/gpt-oss-120b'))
        assert captured['get_url'] == hp.MODELS_URL
        assert captured['create_body']['model'] == 'openai/gpt-oss-120b'
        assert captured['create_body']['mlAssistant'] is False
        assert captured['msg_url'] == f'{hp.CREATE_URL}/abc123'
        assert 'hf-chat=abc' in captured['msg_headers']['Cookie']
        kinds = [(c['type'], c['content']) for c in chunks
                 if c['type'] != 'text' or c['content']]
        assert ('thinking', 'thinking...') in kinds
        # stream deltas + finalAnswer suffix only (no duplication)
        text = ''.join(c['content'] for c in chunks
                       if c['type'] == 'text' and c['content'])
        assert text == 'Hello world!', text
        assert chunks[-1]['finish_reason'] == 'stop'
    finally:
        hp.http_post_raw = orig_post
        hp.http_get = orig_get
        jar_mod.load_jar = orig_load
        hp.load_jar = orig_load
        _unset(old)
    print('PASS test_stream_payload_and_parsing')


def test_stream_error_frame_raises():
    old = _set_env(json.dumps({'hf-chat': 'abc'}))
    orig_post = hp.http_post_raw
    orig_get = hp.http_get

    def fake_get(url, headers=None, **kw):
        return _fake_create_response(payload={'json': [
            {'id': 'm1', 'displayName': 'm1', 'unlisted': False,
             'isRouter': False}]})

    def fake_post(url, data, headers=None, timeout=300, no_proxy=False):
        if url.endswith('/conversation'):
            return _fake_create_response(payload={'conversationId': 'abc'})
        resp = _fake_create_response()
        resp.content = json.dumps({
            'type': 'status', 'status': 'error',
            'message': 'quota exceeded'}).encode()
        resp.text = ''
        return resp

    try:
        hp.http_get = fake_get
        hp.http_post_raw = fake_post
        p = hp.HuggingChatProvider()
        try:
            list(p.stream('hi', model='m1'))
            raise AssertionError('expected ProviderError')
        except ProviderError as e:
            assert 'quota exceeded' in str(e)
    finally:
        hp.http_post_raw = orig_post
        hp.http_get = orig_get
        _unset(old)
    print('PASS test_stream_error_frame_raises')


def test_stream_empty_raises_unavailable():
    old = _set_env(json.dumps({'hf-chat': 'abc'}))
    orig_post = hp.http_post_raw
    orig_get = hp.http_get

    def fake_get(url, headers=None, **kw):
        return _fake_create_response(payload={'json': [
            {'id': 'm1', 'displayName': 'm1', 'unlisted': False,
             'isRouter': False}]})

    def fake_post(url, data, headers=None, timeout=300, no_proxy=False):
        if url.endswith('/conversation'):
            return _fake_create_response(payload={'conversationId': 'abc'})
        resp = _fake_create_response()
        resp.content = b''
        resp.text = ''
        return resp

    try:
        hp.http_get = fake_get
        hp.http_post_raw = fake_post
        p = hp.HuggingChatProvider()
        try:
            list(p.stream('hi', model='m1'))
            raise AssertionError('expected ProviderUnavailableError')
        except ProviderUnavailableError:
            pass
    finally:
        hp.http_post_raw = orig_post
        hp.http_get = orig_get
        _unset(old)
    print('PASS test_stream_empty_raises_unavailable')


def test_error_classification():
    classify = hp._classify
    e = classify(401, '{"error":"You have to be logged in."}')
    assert isinstance(e, ProviderAuthError) and 'logged in' in str(e)
    e = classify(403, 'challenge')
    assert isinstance(e, ProviderAuthError)
    e = classify(429, 'slow down')
    assert type(e).__name__ == 'ProviderRateLimitError'
    print('PASS test_error_classification')


def test_unknown_model_and_images():
    old = _set_env(json.dumps({'hf-chat': 'abc'}))
    orig_get = hp.http_get
    try:
        hp.http_get = lambda url, headers=None, **kw: _fake_create_response(
            payload={'json': []})
        p = hp.HuggingChatProvider()
        try:
            p.stream('hi', model='nope/nope')
            raise AssertionError('expected ProviderError')
        except ProviderError as e:
            assert 'not in catalog' in str(e)
        try:
            p.stream('hi', model='x', images=[{'b64': 'x'}])
            raise AssertionError('expected ProviderError')
        except ProviderError as e:
            assert 'image input' in str(e)
    finally:
        hp.http_get = orig_get
        _unset(old)
    print('PASS test_unknown_model_and_images')


def test_wiring():
    from dsk.providers import router as router_mod
    assert ('huggingchat', '.huggingchat_provider',
            'HuggingChatProvider') in router_mod.PROVIDER_MODULES
    assert router_mod.PUBLIC_PREFIX['huggingchat'] == 'huggingchat'
    assert router_mod.OWNED_BY['huggingchat'] == 'huggingface'
    assert 'huggingchat' in router_mod.AUTO_CATEGORIES['general']
    from dsk import selfheal
    assert selfheal.HEALABLE['huggingchat'].name == 'huggingchat_provider.py'
    assert selfheal._PROVIDER_CLASSES['huggingchat'] == 'HuggingChatProvider'
    from dsk import refresher
    assert refresher.REFRESH['huggingchat'] is not None
    assert refresher.SIGNUP['huggingchat'] is not None
    print('PASS test_wiring')


if __name__ == '__main__':
    test_catalog_parse()
    test_dormant_without_session()
    test_env_session_enables()
    test_stream_payload_and_parsing()
    test_stream_error_frame_raises()
    test_stream_empty_raises_unavailable()
    test_error_classification()
    test_unknown_model_and_images()
    test_wiring()
    print('ALL OK')
