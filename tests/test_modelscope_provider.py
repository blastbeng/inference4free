"""Offline tests for the ModelScope provider (no network)."""

import json
import os
import sys
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from dsk.providers import modelscope_provider as mp  # noqa: E402
from dsk.providers.base import (  # noqa: E402
    ProviderAuthError,
    ProviderError,
    ProviderUnavailableError,
)


def _set_env(value):
    old = os.environ.get('MODELSCOPE_API_KEY')
    os.environ['MODELSCOPE_API_KEY'] = value
    return old


def _unset(old):
    if old is None:
        os.environ.pop('MODELSCOPE_API_KEY', None)
    else:
        os.environ['MODELSCOPE_API_KEY'] = old


def test_catalog_parse():
    data = {'data': [
        {'id': 'Qwen/Qwen3.8-27B', 'owned_by': 'system', 'created': 1},
        {'id': 'deepseek-ai/DeepSeek-V4-Pro', 'owned_by': 'system',
         'created': 2},
        {'id': 'Qwen/Qwen-Image-Edit', 'owned_by': 'system', 'created': 3},
        {'id': 'OpenGVLab/InternVL3_5-241B-A28B', 'owned_by': 'system'},
        'not-a-dict',
        {'displayName': 'no-id'},
    ]}
    models = mp._parse_models(data)
    # image-edit entries are skipped — no chat endpoint
    assert [m['id'] for m in models] == ['Qwen/Qwen3.8-27B',
                                         'deepseek-ai/DeepSeek-V4-Pro',
                                         'OpenGVLab/InternVL3_5-241B-A28B']
    # ModelScope entries carry no context/output limits — fallbacks apply
    assert models[0]['context'] == mp.MODELSCOPE_CONTEXT_FALLBACK
    assert models[0]['max_out'] == mp.MODELSCOPE_MAX_OUTPUT_FALLBACK
    assert models[0]['owned_by'] == 'system'
    # list payload (no {"data": ...} wrapper) also parses
    assert len(mp._parse_models([{'id': 'bare'}])) == 1
    assert mp._parse_models({'nope': 1}) == []
    print('PASS test_catalog_parse')


def test_dormant_without_token():
    old = _set_env('')
    import dsk.providers.jar as jar_mod
    orig = jar_mod.load_jar
    jar_mod.load_jar = lambda name: {}
    mp.load_jar = jar_mod.load_jar  # rebind the imported name
    try:
        p = mp.ModelScopeProvider()
        assert p.available() is False
        try:
            p.list_models()
            raise AssertionError('expected ProviderAuthError')
        except ProviderAuthError as e:
            assert 'modelscope' in str(e).lower()
    finally:
        jar_mod.load_jar = orig
        mp.load_jar = orig
        _unset(old)
    print('PASS test_dormant_without_token')


def test_env_token_enables():
    old = _set_env('ms-01234567-89ab-cdef-0123-456789abcdef')
    try:
        p = mp.ModelScopeProvider()
        assert p.available() is True
        hdr = mp._headers()
        assert hdr['Authorization'] == \
            'Bearer ms-01234567-89ab-cdef-0123-456789abcdef'
    finally:
        _unset(old)
    print('PASS test_env_token_enables')


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
    {'id': 'Qwen/Qwen3.8-27B', 'owned_by': 'system', 'created': 1},
    {'id': 'deepseek-ai/DeepSeek-V4-Pro', 'owned_by': 'system', 'created': 2},
    {'id': 'OpenGVLab/InternVL3_5-241B-A28B', 'owned_by': 'system',
     'created': 3},
]


def test_stream_payload_and_parsing():
    old = _set_env('ms-01234567-89ab-cdef-0123-456789abcdef')
    orig_get = mp.http_get
    orig_post = mp.http_post_stream
    captured = {}

    def fake_get(url, headers=None, **kw):
        captured['get_url'] = url
        # /models is fetched WITHOUT the bearer (public endpoint)
        captured['get_auth'] = (headers or {}).get('Authorization')
        return _fake_response(payload={'data': MODELS_FIXTURE})

    def fake_post(url, headers=None, json_body=None, **kw):
        captured['post_url'] = url
        captured['post_auth'] = (headers or {}).get('Authorization')
        captured['body'] = json_body
        sse = '\n'.join([
            json.dumps({'choices': [
                {'delta': {'reasoning_content': 'thinking...'},
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
        mp.http_get = fake_get
        mp.http_post_stream = fake_post
        p = mp.ModelScopeProvider()
        chunks = list(p.stream('hi', model='Qwen/Qwen3.8-27B',
                               max_tokens=512, temperature=0.7))
        assert captured['get_url'] == mp.MODELS_URL
        # catalog is public: no Authorization header on the models fetch
        assert captured['get_auth'] is None
        assert captured['post_url'] == mp.CHAT_URL
        assert captured['post_auth'].startswith('Bearer ms-')
        body = captured['body']
        assert body['model'] == 'Qwen/Qwen3.8-27B'
        assert body['stream'] is True
        assert body['max_tokens'] == 512
        assert body['temperature'] == 0.7
        # reasoning streams as delta.reasoning_content (DeepSeek-style)
        assert 'reasoning_format' not in body
        kinds = [(c['type'], c['content']) for c in chunks
                 if c['type'] != 'text' or c['content']]
        assert ('thinking', 'thinking...') in kinds
        text = ''.join(c['content'] for c in chunks
                       if c['type'] == 'text' and c['content'])
        assert text == 'Hello world', text
        assert chunks[-1]['finish_reason'] == 'stop'
    finally:
        mp.http_get = orig_get
        mp.http_post_stream = orig_post
        _unset(old)
    print('PASS test_stream_payload_and_parsing')


def test_stream_error_frame_raises():
    old = _set_env('ms-kkkk')
    orig_get = mp.http_get
    orig_post = mp.http_post_stream

    def fake_get(url, headers=None, **kw):
        return _fake_response(payload={'data': MODELS_FIXTURE})

    def fake_post(url, headers=None, json_body=None, **kw):
        return _fake_response(lines=[
            b'data: ' + json.dumps(
                {'error': {'message': 'rate limit exceeded',
                           'request_id': 'abc-123'}}).encode(),
        ])

    try:
        mp.http_get = fake_get
        mp.http_post_stream = fake_post
        p = mp.ModelScopeProvider()
        try:
            list(p.stream('hi', model='Qwen/Qwen3.8-27B'))
            raise AssertionError('expected ProviderError')
        except ProviderError as e:
            assert 'rate limit exceeded' in str(e)
    finally:
        mp.http_get = orig_get
        mp.http_post_stream = orig_post
        _unset(old)
    print('PASS test_stream_error_frame_raises')


def test_stream_empty_raises_unavailable():
    old = _set_env('ms-kkkk')
    orig_get = mp.http_get
    orig_post = mp.http_post_stream

    def fake_get(url, headers=None, **kw):
        return _fake_response(payload={'data': MODELS_FIXTURE})

    def fake_post(url, headers=None, json_body=None, **kw):
        return _fake_response(lines=[b'', b'data: [DONE]'])

    try:
        mp.http_get = fake_get
        mp.http_post_stream = fake_post
        p = mp.ModelScopeProvider()
        try:
            list(p.stream('hi', model='Qwen/Qwen3.8-27B'))
            raise AssertionError('expected ProviderUnavailableError')
        except ProviderUnavailableError:
            pass
    finally:
        mp.http_get = orig_get
        mp.http_post_stream = orig_post
        _unset(old)
    print('PASS test_stream_empty_raises_unavailable')


def test_http_error_classification():
    nested = ('{"error":{"message":"Authentication failed, please make '
              'sure that a valid ModelScope token is supplied.",'
              '"request_id":"75a0fb11"}}')
    e = mp._classify(401, nested)
    assert isinstance(e, ProviderAuthError) and 'rejected' in str(e)
    e = mp._classify(403, 'forbidden')
    assert isinstance(e, ProviderAuthError)
    e = mp._classify(429, 'slow down', {'Retry-After': '3'})
    assert type(e).__name__ == 'ProviderRateLimitError'
    # non-auth statuses fall through to the shared classifier
    e = mp._classify(503, 'overloaded')
    assert type(e).__name__ == 'ProviderUnavailableError'
    print('PASS test_http_error_classification')


def test_unknown_model_and_images():
    old = _set_env('ms-kkkk')
    orig_get = mp.http_get
    try:
        mp.http_get = lambda url, headers=None, **kw: _fake_response(
            payload={'data': MODELS_FIXTURE})
        p = mp.ModelScopeProvider()
        try:
            p.stream('hi', model='nope/nope')
            raise AssertionError('expected ProviderError')
        except ProviderError as e:
            assert 'not in catalog' in str(e)
        try:
            p.stream('hi', model='Qwen/Qwen3.8-27B',
                     images=[{'mime': 'image/png', 'data': b'xx'}])
            raise AssertionError('expected ProviderError')
        except ProviderError as e:
            assert 'not vision-capable' in str(e)
        try:
            p.stream('hi', model='Qwen/Qwen3.8-27B',
                     image_generation=True)
            raise AssertionError('expected ProviderError')
        except ProviderError as e:
            assert 'image generation' in str(e)
    finally:
        mp.http_get = orig_get
        _unset(old)
    print('PASS test_unknown_model_and_images')


def test_vision_payload():
    old = _set_env('ms-kkkk')
    orig_get = mp.http_get
    orig_post = mp.http_post_stream
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
        mp.http_get = fake_get
        mp.http_post_stream = fake_post
        p = mp.ModelScopeProvider()
        list(p.stream('what is this',
                      model='OpenGVLab/InternVL3_5-241B-A28B',
                      images=[{'mime': 'image/png', 'data': b'png-bytes'}]))
        msgs = captured['body']['messages']
        parts = msgs[0]['content']
        assert parts[0] == {'type': 'text', 'text': 'what is this'}
        assert parts[1]['type'] == 'image_url'
        assert parts[1]['image_url']['url'].startswith(
            'data:image/png;base64,')
    finally:
        mp.http_get = orig_get
        mp.http_post_stream = orig_post
        _unset(old)
    print('PASS test_vision_payload')


def test_refresh_liveness():
    from dsk import refresher as rf
    old = _set_env('ms-01234567-89ab-cdef-0123-456789abcdef')
    orig_get = rf._http_get
    try:
        # 429 still proves the token is accepted (auth runs first)
        rf._http_get = lambda *a, **kw: _fake_response(429, {})
        ok, msg = rf.refresh_modelscope()
        assert ok is True and '429' in msg, msg
        # nested 401 Authentication failed -> rejected
        rf._http_get = lambda *a, **kw: _fake_response(401, {
            'error': {'message': 'Authentication failed, please make '
                      'sure that a valid ModelScope token is supplied.',
                      'request_id': 'x'}})
        ok, msg = rf.refresh_modelscope()
        assert ok is False and 'rejected' in msg, msg
        # 200 with catalog
        rf._http_get = lambda *a, **kw: _fake_response(
            200, {'data': [{'id': 'Qwen/Qwen3.8-27B'}]})
        ok, msg = rf.refresh_modelscope()
        assert ok is True and '1 models visible' in msg, msg
    finally:
        rf._http_get = orig_get
        _unset(old)
    print('PASS test_refresh_liveness')


def test_wiring():
    from dsk.providers import router as router_mod
    assert ('modelscope', '.modelscope_provider', 'ModelScopeProvider') \
        in router_mod.PROVIDER_MODULES
    assert router_mod.PUBLIC_PREFIX['modelscope'] == 'modelscope'
    assert router_mod.OWNED_BY['modelscope'] == 'modelscope'
    assert 'modelscope' in router_mod.AUTO_CATEGORIES['general']
    from dsk import selfheal
    assert selfheal.HEALABLE['modelscope'].name == 'modelscope_provider.py'
    assert selfheal._PROVIDER_CLASSES['modelscope'] == 'ModelScopeProvider'
    assert selfheal._MODULE_NAMES['modelscope'] == \
        'dsk.providers.modelscope_provider'
    import re
    pats = [re.compile(x) for x in selfheal._EVIDENCE_PATTERNS['modelscope']]
    assert pats[0].search('api-inference.modelscope.cn/v1/models')
    assert pats[1].search('token ms-12345678-abcd-1234-abcd-1234567890ab')
    from dsk import refresher
    assert refresher.REFRESH['modelscope'] is not None
    assert refresher.SIGNUP['modelscope'] is not None
    assert refresher._jar_path('modelscope').name == \
        'modelscope_cookies.json'
    old = _set_env('ms-01234567-89ab-cdef-0123-456789abcdef')
    try:
        assert refresher._has_creds('modelscope') is True
        assert refresher._load_jar('modelscope').get('api_key') == \
            'ms-01234567-89ab-cdef-0123-456789abcdef'
    finally:
        _unset(old)
    print('PASS test_wiring')


if __name__ == '__main__':
    test_catalog_parse()
    test_dormant_without_token()
    test_env_token_enables()
    test_stream_payload_and_parsing()
    test_stream_error_frame_raises()
    test_stream_empty_raises_unavailable()
    test_http_error_classification()
    test_unknown_model_and_images()
    test_vision_payload()
    test_refresh_liveness()
    test_wiring()
    print('ALL OK')
