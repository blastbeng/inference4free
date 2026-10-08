"""Offline tests for the Groq provider (no network)."""

import json
import os
import sys
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from dsk.providers import groq_provider as gp  # noqa: E402
from dsk.providers.base import (  # noqa: E402
    ProviderAuthError,
    ProviderError,
    ProviderUnavailableError,
)


def _set_env(value):
    old = os.environ.get('GROQ_API_KEY')
    os.environ['GROQ_API_KEY'] = value
    return old


def _unset(old):
    if old is None:
        os.environ.pop('GROQ_API_KEY', None)
    else:
        os.environ['GROQ_API_KEY'] = old


def test_catalog_parse():
    data = {'data': [
        {'id': 'llama-3.3-70b-versatile', 'owned_by': 'Meta',
         'context_window': 131072, 'max_completion_tokens': 32768,
         'active': True},
        {'id': 'qwen/qwen3-32b', 'context_window': 131072,
         'max_completion_tokens': 40960, 'active': True},
        {'id': 'retired-model', 'active': False},
        'not-a-dict',
        {'displayName': 'no-id'},
        {'id': 'no-limits-model'},
        {'id': 'max-over-context', 'context_window': 8192,
         'max_completion_tokens': 999999},
    ]}
    models = gp._parse_models(data)
    assert [m['id'] for m in models] == ['llama-3.3-70b-versatile',
                                         'qwen/qwen3-32b',
                                         'no-limits-model',
                                         'max-over-context']
    assert models[0]['context'] == 131072
    assert models[0]['max_out'] == 32768
    assert models[2]['context'] == gp.GROQ_CONTEXT_FALLBACK
    assert models[2]['max_out'] == gp.GROQ_MAX_OUTPUT_FALLBACK
    # max_completion_tokens is capped at the context window
    assert models[3]['max_out'] == 8192
    # list payload (no {"data": ...} wrapper) also parses
    assert len(gp._parse_models([{'id': 'bare'}])) == 1
    assert gp._parse_models({'nope': 1}) == []
    print('PASS test_catalog_parse')


def test_dormant_without_key():
    old = _set_env('')
    import dsk.providers.jar as jar_mod
    orig = jar_mod.load_jar
    jar_mod.load_jar = lambda name: {}
    gp.load_jar = jar_mod.load_jar  # rebind the imported name
    try:
        p = gp.GroqProvider()
        assert p.available() is False
        try:
            p.list_models()
            raise AssertionError('expected ProviderAuthError')
        except ProviderAuthError as e:
            assert 'console.groq.com' in str(e)
    finally:
        jar_mod.load_jar = orig
        gp.load_jar = orig
        _unset(old)
    print('PASS test_dormant_without_key')


def test_env_key_enables():
    old = _set_env('gsk_testkey1234567890')
    try:
        p = gp.GroqProvider()
        assert p.available() is True
        hdr = gp._headers()
        assert hdr['Authorization'] == 'Bearer gsk_testkey1234567890'
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
    {'id': 'llama-3.3-70b-versatile', 'context_window': 131072,
     'max_completion_tokens': 32768, 'active': True},
    {'id': 'qwen/qwen3-32b', 'context_window': 131072,
     'max_completion_tokens': 40960, 'active': True},
    {'id': 'meta-llama/llama-4-scout-17b-16e-instruct',
     'context_window': 131072, 'max_completion_tokens': 8192,
     'active': True},
]


def test_stream_payload_and_parsing():
    old = _set_env('gsk_kkkk')
    orig_get = gp.http_get
    orig_post = gp.http_post_stream
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
        gp.http_get = fake_get
        gp.http_post_stream = fake_post
        p = gp.GroqProvider()
        chunks = list(p.stream('hi', model='qwen/qwen3-32b',
                               max_tokens=512, temperature=0.7))
        assert captured['get_url'] == gp.MODELS_URL
        assert captured['get_auth'] == 'Bearer gsk_kkkk'
        assert captured['post_url'] == gp.CHAT_URL
        assert captured['post_auth'] == 'Bearer gsk_kkkk'
        body = captured['body']
        assert body['model'] == 'qwen/qwen3-32b'
        assert body['stream'] is True
        assert body['max_completion_tokens'] == 512
        assert body['temperature'] == 0.7
        # reasoning model heuristic attaches reasoning_format:parsed
        assert body['reasoning_format'] == 'parsed'
        kinds = [(c['type'], c['content']) for c in chunks
                 if c['type'] != 'text' or c['content']]
        assert ('thinking', 'thinking...') in kinds
        text = ''.join(c['content'] for c in chunks
                       if c['type'] == 'text' and c['content'])
        assert text == 'Hello world', text
        assert chunks[-1]['finish_reason'] == 'stop'
    finally:
        gp.http_get = orig_get
        gp.http_post_stream = orig_post
        _unset(old)
    print('PASS test_stream_payload_and_parsing')


def test_non_reasoning_model_omits_reasoning_format():
    old = _set_env('gsk_kkkk')
    orig_get = gp.http_get
    orig_post = gp.http_post_stream
    captured = {}

    def fake_get(url, headers=None, **kw):
        return _fake_response(payload={'data': MODELS_FIXTURE})

    def fake_post(url, headers=None, json_body=None, **kw):
        captured['body'] = json_body
        return _fake_response(lines=[
            b'data: ' + json.dumps(
                {'choices': [{'delta': {'content': 'ok'},
                              'finish_reason': 'stop'}]}).encode(),
        ])

    try:
        gp.http_get = fake_get
        gp.http_post_stream = fake_post
        p = gp.GroqProvider()
        chunks = list(p.stream('hi', model='llama-3.3-70b-versatile'))
        # a non-reasoning model would 400 on reasoning_format — omitted
        assert 'reasoning_format' not in captured['body']
        assert chunks[-1]['finish_reason'] == 'stop'
    finally:
        gp.http_get = orig_get
        gp.http_post_stream = orig_post
        _unset(old)
    print('PASS test_non_reasoning_model_omits_reasoning_format')


def test_stream_error_frame_raises():
    old = _set_env('gsk_kkkk')
    orig_get = gp.http_get
    orig_post = gp.http_post_stream

    def fake_get(url, headers=None, **kw):
        return _fake_response(payload={'data': MODELS_FIXTURE})

    def fake_post(url, headers=None, json_body=None, **kw):
        return _fake_response(lines=[
            b'data: ' + json.dumps(
                {'error': {'message': 'rate limit exceeded',
                           'code': 'rate_limit_exceeded'}}).encode(),
        ])

    try:
        gp.http_get = fake_get
        gp.http_post_stream = fake_post
        p = gp.GroqProvider()
        try:
            list(p.stream('hi', model='llama-3.3-70b-versatile'))
            raise AssertionError('expected ProviderError')
        except ProviderError as e:
            assert 'rate limit exceeded' in str(e)
    finally:
        gp.http_get = orig_get
        gp.http_post_stream = orig_post
        _unset(old)
    print('PASS test_stream_error_frame_raises')


def test_stream_empty_raises_unavailable():
    old = _set_env('gsk_kkkk')
    orig_get = gp.http_get
    orig_post = gp.http_post_stream

    def fake_get(url, headers=None, **kw):
        return _fake_response(payload={'data': MODELS_FIXTURE})

    def fake_post(url, headers=None, json_body=None, **kw):
        return _fake_response(lines=[b'', b'data: [DONE]'])

    try:
        gp.http_get = fake_get
        gp.http_post_stream = fake_post
        p = gp.GroqProvider()
        try:
            list(p.stream('hi', model='llama-3.3-70b-versatile'))
            raise AssertionError('expected ProviderUnavailableError')
        except ProviderUnavailableError:
            pass
    finally:
        gp.http_get = orig_get
        gp.http_post_stream = orig_post
        _unset(old)
    print('PASS test_stream_empty_raises_unavailable')


def test_http_error_classification():
    e = gp._classify(401, '{"error":{"code":"invalid_api_key"}}')
    assert isinstance(e, ProviderAuthError) and 'rejected' in str(e)
    e = gp._classify(403, 'forbidden')
    assert isinstance(e, ProviderAuthError)
    e = gp._classify(429, 'slow down', {'Retry-After': '3'})
    assert type(e).__name__ == 'ProviderRateLimitError'
    # non-auth statuses fall through to the shared classifier
    e = gp._classify(503, 'overloaded')
    assert type(e).__name__ == 'ProviderUnavailableError'
    print('PASS test_http_error_classification')


def test_unknown_model_and_images():
    old = _set_env('gsk_kkkk')
    orig_get = gp.http_get
    try:
        gp.http_get = lambda url, headers=None, **kw: _fake_response(
            payload={'data': MODELS_FIXTURE})
        p = gp.GroqProvider()
        try:
            p.stream('hi', model='nope/nope')
            raise AssertionError('expected ProviderError')
        except ProviderError as e:
            assert 'not in catalog' in str(e)
        try:
            p.stream('hi', model='llama-3.3-70b-versatile',
                     images=[{'mime': 'image/png', 'data': b'xx'}])
            raise AssertionError('expected ProviderError')
        except ProviderError as e:
            assert 'not vision-capable' in str(e)
        try:
            p.stream('hi', model='llama-3.3-70b-versatile',
                     image_generation=True)
            raise AssertionError('expected ProviderError')
        except ProviderError as e:
            assert 'image generation' in str(e)
    finally:
        gp.http_get = orig_get
        _unset(old)
    print('PASS test_unknown_model_and_images')


def test_vision_payload():
    old = _set_env('gsk_kkkk')
    orig_get = gp.http_get
    orig_post = gp.http_post_stream
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
        gp.http_get = fake_get
        gp.http_post_stream = fake_post
        p = gp.GroqProvider()
        list(p.stream('what is this',
                      model='meta-llama/llama-4-scout-17b-16e-instruct',
                      images=[{'mime': 'image/png', 'data': b'png-bytes'}]))
        msgs = captured['body']['messages']
        parts = msgs[0]['content']
        assert parts[0] == {'type': 'text', 'text': 'what is this'}
        assert parts[1]['type'] == 'image_url'
        assert parts[1]['image_url']['url'].startswith(
            'data:image/png;base64,')
    finally:
        gp.http_get = orig_get
        gp.http_post_stream = orig_post
        _unset(old)
    print('PASS test_vision_payload')


def test_wiring():
    from dsk.providers import router as router_mod
    assert ('groq', '.groq_provider', 'GroqProvider') \
        in router_mod.PROVIDER_MODULES
    assert router_mod.PUBLIC_PREFIX['groq'] == 'groq'
    assert router_mod.OWNED_BY['groq'] == 'groq'
    assert 'groq' in router_mod.AUTO_CATEGORIES['general']
    from dsk import selfheal
    assert selfheal.HEALABLE['groq'].name == 'groq_provider.py'
    assert selfheal._PROVIDER_CLASSES['groq'] == 'GroqProvider'
    assert selfheal._MODULE_NAMES['groq'] == 'dsk.providers.groq_provider'
    import re
    pats = [re.compile(x) for x in selfheal._EVIDENCE_PATTERNS['groq']]
    assert pats[0].search('api.groq.com/openai/v1/models')
    from dsk import refresher
    assert refresher.REFRESH['groq'] is not None
    assert refresher.SIGNUP['groq'] is not None
    assert refresher._jar_path('groq').name == 'groq_cookies.json'
    old = _set_env('gsk_kkkk')
    try:
        assert refresher._has_creds('groq') is True
        assert refresher._load_jar('groq').get('api_key') == 'gsk_kkkk'
    finally:
        _unset(old)
    print('PASS test_wiring')


if __name__ == '__main__':
    test_catalog_parse()
    test_dormant_without_key()
    test_env_key_enables()
    test_stream_payload_and_parsing()
    test_non_reasoning_model_omits_reasoning_format()
    test_stream_error_frame_raises()
    test_stream_empty_raises_unavailable()
    test_http_error_classification()
    test_unknown_model_and_images()
    test_vision_payload()
    test_wiring()
    print('ALL OK')
