"""Offline tests for the Cerebras provider (no network)."""

import json
import os
import sys
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from dsk.providers import cerebras_provider as cp  # noqa: E402
from dsk.providers.base import (  # noqa: E402
    ProviderAuthError,
    ProviderError,
    ProviderUnavailableError,
)


def _set_env(value):
    old = os.environ.get('CEREBRAS_API_KEY')
    os.environ['CEREBRAS_API_KEY'] = value
    return old


def _unset(old):
    if old is None:
        os.environ.pop('CEREBRAS_API_KEY', None)
    else:
        os.environ['CEREBRAS_API_KEY'] = old


def test_catalog_parse():
    data = {'data': [
        {'id': 'llama-3.3-70b', 'owned_by': 'Meta',
         'context_window': 128000, 'max_completion_tokens': 8192,
         'active': True},
        {'id': 'qwen/qwen3-32b', 'context_window': 128000,
         'max_completion_tokens': 16384, 'active': True},
        {'id': 'retired-model', 'active': False},
        'not-a-dict',
        {'displayName': 'no-id'},
        {'id': 'no-limits-model'},
        {'id': 'max-over-context', 'context_window': 8192,
         'max_completion_tokens': 999999},
    ]}
    models = cp._parse_models(data)
    assert [m['id'] for m in models] == ['llama-3.3-70b', 'qwen/qwen3-32b',
                                         'no-limits-model',
                                         'max-over-context']
    assert models[0]['context'] == 128000
    assert models[0]['max_out'] == 8192
    assert models[2]['context'] == cp.CEREBRAS_CONTEXT_FALLBACK
    assert models[2]['max_out'] == cp.CEREBRAS_MAX_OUTPUT_FALLBACK
    # max_completion_tokens is capped at the context window
    assert models[3]['max_out'] == 8192
    # list payload (no {"data": ...} wrapper) also parses
    assert len(cp._parse_models([{'id': 'bare'}])) == 1
    assert cp._parse_models({'nope': 1}) == []
    print('PASS test_catalog_parse')


def test_dormant_without_key():
    old = _set_env('')
    import dsk.providers.jar as jar_mod
    orig = jar_mod.load_jar
    jar_mod.load_jar = lambda name: {}
    cp.load_jar = jar_mod.load_jar  # rebind the imported name
    try:
        p = cp.CerebrasProvider()
        assert p.available() is False
        try:
            p.list_models()
            raise AssertionError('expected ProviderAuthError')
        except ProviderAuthError as e:
            assert 'cerebras' in str(e).lower()
    finally:
        jar_mod.load_jar = orig
        cp.load_jar = orig
        _unset(old)
    print('PASS test_dormant_without_key')


def test_env_key_enables():
    old = _set_env('csk_testkey1234567890')
    try:
        p = cp.CerebrasProvider()
        assert p.available() is True
        hdr = cp._headers()
        assert hdr['Authorization'] == 'Bearer csk_testkey1234567890'
    finally:
        _unset(old)
    print('PASS test_env_key_enables')


def _fake_response(status=200, payload=None, lines=()):
    resp = types.SimpleNamespace()
    resp.status_code = status
    resp.text = json.dumps(payload or {})
    resp.headers = {}

    def _json():
        return json.loads(resp.text)

    resp.json = _json
    resp.iter_lines = lambda: iter(lines)
    return resp


MODELS_FIXTURE = [
    {'id': 'llama-3.3-70b', 'context_window': 128000,
     'max_completion_tokens': 8192, 'active': True},
    {'id': 'qwen/qwen3-32b', 'context_window': 128000,
     'max_completion_tokens': 16384, 'active': True},
    {'id': 'meta-llama/Llama-4-Scout-17B-16E-Instruct',
     'context_window': 128000, 'max_completion_tokens': 8192,
     'active': True},
]


def test_stream_payload_and_parsing():
    old = _set_env('csk_kkkk')
    orig_get = cp.http_get
    orig_post = cp.http_post_stream
    captured = {}

    def fake_get(url, headers=None, **kw):
        captured['get_url'] = url
        captured['get_auth'] = (headers or {}).get('Authorization')
        return _fake_response(payload={'data': MODELS_FIXTURE})

    def fake_post(url, headers=None, json_body=None, **kw):
        captured['post_url'] = url
        captured['post_auth'] = (headers or {}).get('Authorization')
        captured['body'] = json_body
        sse = '\n'.join([
            json.dumps({'choices': [{'delta': {'reasoning': 'thinking...'},
                                     'finish_reason': None}]}),
            json.dumps({'choices': [{'delta': {'content': 'Hello'},
                                     'finish_reason': None}]}),
            json.dumps({'choices': [{'delta': {'content': ' world'},
                                     'finish_reason': None}]}),
            json.dumps({'choices': [{'delta': {},
                                     'finish_reason': 'stop'}]}),
            '[DONE]',
        ]).encode()
        return _fake_response(lines=[b'data: ' + l for l in sse.split(b'\n')])

    try:
        cp.http_get = fake_get
        cp.http_post_stream = fake_post
        p = cp.CerebrasProvider()
        chunks = list(p.stream('hi', model='qwen/qwen3-32b',
                               max_tokens=512, temperature=0.7))
        assert captured['get_url'] == cp.MODELS_URL
        assert captured['get_auth'] == 'Bearer csk_kkkk'
        assert captured['post_url'] == cp.CHAT_URL
        assert captured['post_auth'] == 'Bearer csk_kkkk'
        body = captured['body']
        assert body['model'] == 'qwen/qwen3-32b'
        assert body['stream'] is True
        assert body['max_completion_tokens'] == 512
        assert body['temperature'] == 0.7
        # Cerebras has no reasoning_format parameter — gpt-oss streams
        # delta.reasoning natively, qwen3 inlines <think/> tags.
        assert 'reasoning_format' not in body
        kinds = [(c['type'], c['content']) for c in chunks
                 if c['type'] != 'text' or c['content']]
        assert ('thinking', 'thinking...') in kinds
        text = ''.join(c['content'] for c in chunks
                       if c['type'] == 'text' and c['content'])
        assert text == 'Hello world', text
        assert chunks[-1]['finish_reason'] == 'stop'
    finally:
        cp.http_get = orig_get
        cp.http_post_stream = orig_post
        _unset(old)
    print('PASS test_stream_payload_and_parsing')


def test_stream_error_frame_raises():
    old = _set_env('csk_kkkk')
    orig_get = cp.http_get
    orig_post = cp.http_post_stream

    def fake_get(url, headers=None, **kw):
        return _fake_response(payload={'data': MODELS_FIXTURE})

    def fake_post(url, headers=None, json_body=None, **kw):
        return _fake_response(lines=[
            b'data: ' + json.dumps(
                {'error': {'message': 'rate limit exceeded',
                           'code': 'rate_limit_exceeded'}}).encode(),
        ])

    try:
        cp.http_get = fake_get
        cp.http_post_stream = fake_post
        p = cp.CerebrasProvider()
        try:
            list(p.stream('hi', model='llama-3.3-70b'))
            raise AssertionError('expected ProviderError')
        except ProviderError as e:
            assert 'rate limit exceeded' in str(e)
    finally:
        cp.http_get = orig_get
        cp.http_post_stream = orig_post
        _unset(old)
    print('PASS test_stream_error_frame_raises')


def test_stream_type_error_frame_raises():
    # Cerebras also emits bare {"type": "error", "message": ...} frames
    old = _set_env('csk_kkkk')
    orig_get = cp.http_get
    orig_post = cp.http_post_stream

    def fake_get(url, headers=None, **kw):
        return _fake_response(payload={'data': MODELS_FIXTURE})

    def fake_post(url, headers=None, json_body=None, **kw):
        return _fake_response(lines=[
            b'data: ' + json.dumps(
                {'type': 'error', 'message': 'model overloaded'}).encode(),
        ])

    try:
        cp.http_get = fake_get
        cp.http_post_stream = fake_post
        p = cp.CerebrasProvider()
        try:
            list(p.stream('hi', model='llama-3.3-70b'))
            raise AssertionError('expected ProviderError')
        except ProviderError as e:
            assert 'model overloaded' in str(e)
    finally:
        cp.http_get = orig_get
        cp.http_post_stream = orig_post
        _unset(old)
    print('PASS test_stream_type_error_frame_raises')


def test_stream_empty_raises_unavailable():
    old = _set_env('csk_kkkk')
    orig_get = cp.http_get
    orig_post = cp.http_post_stream

    def fake_get(url, headers=None, **kw):
        return _fake_response(payload={'data': MODELS_FIXTURE})

    def fake_post(url, headers=None, json_body=None, **kw):
        return _fake_response(lines=[b'', b'data: [DONE]'])

    try:
        cp.http_get = fake_get
        cp.http_post_stream = fake_post
        p = cp.CerebrasProvider()
        try:
            list(p.stream('hi', model='llama-3.3-70b'))
            raise AssertionError('expected ProviderUnavailableError')
        except ProviderUnavailableError:
            pass
    finally:
        cp.http_get = orig_get
        cp.http_post_stream = orig_post
        _unset(old)
    print('PASS test_stream_empty_raises_unavailable')


def test_http_error_classification():
    flat = ('{"message":"Wrong API Key","type":"invalid_request_error",'
            '"param":"api_key","code":"wrong_api_key"}')
    e = cp._classify(401, flat)
    assert isinstance(e, ProviderAuthError) and 'rejected' in str(e)
    e = cp._classify(403, 'forbidden')
    assert isinstance(e, ProviderAuthError)
    e = cp._classify(429, 'slow down', {'Retry-After': '3'})
    assert type(e).__name__ == 'ProviderRateLimitError'
    # non-auth statuses fall through to the shared classifier
    e = cp._classify(503, 'overloaded')
    assert type(e).__name__ == 'ProviderUnavailableError'
    print('PASS test_http_error_classification')


def test_unknown_model_and_images():
    old = _set_env('csk_kkkk')
    orig_get = cp.http_get
    try:
        cp.http_get = lambda url, headers=None, **kw: _fake_response(
            payload={'data': MODELS_FIXTURE})
        p = cp.CerebrasProvider()
        try:
            p.stream('hi', model='nope/nope')
            raise AssertionError('expected ProviderError')
        except ProviderError as e:
            assert 'not in catalog' in str(e)
        try:
            p.stream('hi', model='llama-3.3-70b',
                     images=[{'mime': 'image/png', 'data': b'xx'}])
            raise AssertionError('expected ProviderError')
        except ProviderError as e:
            assert 'not vision-capable' in str(e)
        try:
            p.stream('hi', model='llama-3.3-70b',
                     image_generation=True)
            raise AssertionError('expected ProviderError')
        except ProviderError as e:
            assert 'image generation' in str(e)
    finally:
        cp.http_get = orig_get
        _unset(old)
    print('PASS test_unknown_model_and_images')


def test_vision_payload():
    old = _set_env('csk_kkkk')
    orig_get = cp.http_get
    orig_post = cp.http_post_stream
    captured = {}

    def fake_get(url, headers=None, **kw):
        return _fake_response(payload={'data': MODELS_FIXTURE})

    def fake_post(url, headers=None, json_body=None, **kw):
        captured['body'] = json_body
        return _fake_response(lines=[
            b'data: ' + json.dumps(
                {'choices': [{'delta': {'content': 'pic'},
                              'finish_reason': 'stop'}]}).encode(),
        ])

    try:
        cp.http_get = fake_get
        cp.http_post_stream = fake_post
        p = cp.CerebrasProvider()
        list(p.stream('what is this',
                      model='meta-llama/Llama-4-Scout-17B-16E-Instruct',
                      images=[{'mime': 'image/png', 'data': b'png-bytes'}]))
        msgs = captured['body']['messages']
        parts = msgs[0]['content']
        assert parts[0] == {'type': 'text', 'text': 'what is this'}
        assert parts[1]['type'] == 'image_url'
        assert parts[1]['image_url']['url'].startswith(
            'data:image/png;base64,')
    finally:
        cp.http_get = orig_get
        cp.http_post_stream = orig_post
        _unset(old)
    print('PASS test_vision_payload')


def test_refresh_liveness():
    from dsk import refresher as rf
    old = _set_env('csk_testkey1234567890')
    orig_get = rf._http_get
    try:
        # 429 still proves the key is accepted (auth runs first)
        rf._http_get = lambda *a, **kw: _fake_response(429, {})
        ok, msg = rf.refresh_cerebras()
        assert ok is True and '429' in msg, msg
        # flat 401 wrong_api_key -> rejected
        rf._http_get = lambda *a, **kw: _fake_response(
            401, {'message': 'Wrong API Key', 'code': 'wrong_api_key'})
        ok, msg = rf.refresh_cerebras()
        assert ok is False and 'rejected' in msg, msg
        # 200 with catalog
        rf._http_get = lambda *a, **kw: _fake_response(
            200, {'data': [{'id': 'llama-3.3-70b'}]})
        ok, msg = rf.refresh_cerebras()
        assert ok is True and '1 models visible' in msg, msg
    finally:
        rf._http_get = orig_get
        _unset(old)
    print('PASS test_refresh_liveness')


def test_wiring():
    from dsk.providers import router as router_mod
    assert ('cerebras', '.cerebras_provider', 'CerebrasProvider') \
        in router_mod.PROVIDER_MODULES
    assert router_mod.PUBLIC_PREFIX['cerebras'] == 'cerebras'
    assert router_mod.OWNED_BY['cerebras'] == 'cerebras'
    assert 'cerebras' in router_mod.AUTO_CATEGORIES['general']
    from dsk import selfheal
    assert selfheal.HEALABLE['cerebras'].name == 'cerebras_provider.py'
    assert selfheal._PROVIDER_CLASSES['cerebras'] == 'CerebrasProvider'
    assert selfheal._MODULE_NAMES['cerebras'] == \
        'dsk.providers.cerebras_provider'
    import re
    pats = [re.compile(x) for x in selfheal._EVIDENCE_PATTERNS['cerebras']]
    assert pats[0].search('api.cerebras.ai/v1/models')
    from dsk import refresher
    assert refresher.REFRESH['cerebras'] is not None
    assert refresher.SIGNUP['cerebras'] is not None
    assert refresher._jar_path('cerebras').name == 'cerebras_cookies.json'
    old = _set_env('csk_testkey1234567890')
    try:
        assert refresher._has_creds('cerebras') is True
        assert refresher._load_jar('cerebras').get('api_key') == \
            'csk_testkey1234567890'
    finally:
        _unset(old)
    print('PASS test_wiring')


if __name__ == '__main__':
    test_catalog_parse()
    test_dormant_without_key()
    test_env_key_enables()
    test_stream_payload_and_parsing()
    test_stream_error_frame_raises()
    test_stream_type_error_frame_raises()
    test_stream_empty_raises_unavailable()
    test_http_error_classification()
    test_unknown_model_and_images()
    test_vision_payload()
    test_refresh_liveness()
    test_wiring()
    print('ALL OK')
