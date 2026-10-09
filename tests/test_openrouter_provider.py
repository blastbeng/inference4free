"""Offline tests for the OpenRouter provider (no network)."""

import json
import os
import sys
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from dsk.providers import openrouter_provider as op  # noqa: E402
from dsk.providers.base import (  # noqa: E402
    ProviderAuthError,
    ProviderError,
    ProviderUnavailableError,
)


def _set_env(value):
    old = os.environ.get('OPENROUTER_API_KEY')
    os.environ['OPENROUTER_API_KEY'] = value
    return old


def _unset(old):
    if old is None:
        os.environ.pop('OPENROUTER_API_KEY', None)
    else:
        os.environ['OPENROUTER_API_KEY'] = old


def _entry(mid, context=262144, max_out=4096, inputs=None, sp=None):
    return {
        'id': mid,
        'context_length': context,
        'architecture': {'modality': 'text->text',
                         'input_modalities': inputs or ['text'],
                         'output_modalities': ['text'],
                         'tokenizer': 'Other', 'instruct_type': None},
        'top_provider': {'context_length': context,
                         'max_completion_tokens': max_out,
                         'is_moderated': False},
        'supported_parameters': sp if sp is not None else
        ['temperature', 'max_tokens', 'stop'],
        'pricing': {'prompt': '0', 'completion': '0'},
    }


def test_catalog_parse():
    data = {'data': [
        _entry('nvidia/nemotron-3.5-lightning:free'),
        _entry('qwen/qwen3-vision:free', inputs=['text', 'image']),
        _entry('deepseek/deepseek-r1:free',
               sp=['temperature', 'reasoning', 'include_reasoning']),
        _entry('openai/gpt-5.6-mini'),  # not :free -> skipped
        'not-a-dict',
        {'displayName': 'no-id'},
        # no metadata at all -> fallbacks apply, regex thinking fallback
        {'id': 'someone/deepseek-r1-distill:free'},
    ]}
    models = op._parse_models(data)
    # only :free ids survive
    assert [m['id'] for m in models] == [
        'nvidia/nemotron-3.5-lightning:free',
        'qwen/qwen3-vision:free',
        'deepseek/deepseek-r1:free',
        'someone/deepseek-r1-distill:free',
    ]
    m0 = models[0]
    assert m0['context'] == 262144
    assert m0['max_out'] == 4096
    assert m0['owned_by'] == 'nvidia'
    assert m0['vision'] is False
    assert m0['thinking'] is False
    # vision from architecture.input_modalities
    assert models[1]['vision'] is True
    # thinking from supported_parameters
    assert models[2]['thinking'] is True
    # no metadata: fallback context/max_output, id-regex thinking fallback
    m3 = models[3]
    assert m3['context'] == op.OPENROUTER_CONTEXT_FALLBACK
    assert m3['max_out'] == op.OPENROUTER_MAX_OUTPUT_FALLBACK
    assert m3['thinking'] is True
    assert m3['owned_by'] == 'someone'
    # list payload also parses; non-list shapes give no models
    assert len(op._parse_models([_entry('x/y:free')])) == 1
    assert op._parse_models({'nope': 1}) == []
    print('PASS test_catalog_parse')


def test_dormant_without_key():
    old = _set_env('')
    import dsk.providers.jar as jar_mod
    orig = jar_mod.load_jar
    jar_mod.load_jar = lambda name: {}
    op.load_jar = jar_mod.load_jar  # rebind the imported name
    try:
        p = op.OpenRouterProvider()
        assert p.available() is False
        try:
            p.list_models()
            raise AssertionError('expected ProviderAuthError')
        except ProviderAuthError as e:
            assert 'openrouter.ai/settings/keys' in str(e)
    finally:
        jar_mod.load_jar = orig
        op.load_jar = orig
        _unset(old)
    print('PASS test_dormant_without_key')


def test_env_key_enables():
    old = _set_env('sk-or-v1-0123456789abcdef0123456789abcdef')
    try:
        p = op.OpenRouterProvider()
        assert p.available() is True
        hdr = op._headers()
        assert hdr['Authorization'] == \
            'Bearer sk-or-v1-0123456789abcdef0123456789abcdef'
        # attribution headers OpenRouter asks for
        assert hdr['HTTP-Referer']
        assert hdr['X-Title']
    finally:
        _unset(old)
    print('PASS test_env_key_enables')


def _fake_response(status=200, payload=None, lines=()):
    resp = types.SimpleNamespace()
    resp.status_code = status
    resp.text = json.dumps(payload if payload is not None else {})
    resp.headers = {}

    def _json():
        return json.loads(resp.text)

    resp.json = _json
    resp.iter_lines = lambda: iter(lines)
    return resp


MODELS_FIXTURE = {'data': [
    _entry('nvidia/nemotron-3.5-lightning:free'),
    _entry('qwen/qwen3-vision:free', inputs=['text', 'image']),
    _entry('deepseek/deepseek-r1:free',
           sp=['temperature', 'reasoning', 'include_reasoning']),
]}


def test_stream_payload_and_parsing():
    old = _set_env('sk-or-v1-0123456789abcdef0123456789abcdef')
    orig_get = op.http_get
    orig_post = op.http_post_stream
    captured = {}

    def fake_get(url, headers=None, **kw):
        captured['get_url'] = url
        # /models is fetched WITHOUT the bearer (public endpoint)
        captured['get_auth'] = (headers or {}).get('Authorization')
        return _fake_response(payload=MODELS_FIXTURE)

    def fake_post(url, headers=None, json_body=None, **kw):
        captured['post_url'] = url
        captured['post_auth'] = (headers or {}).get('Authorization')
        captured['body'] = json_body
        sse = '\n'.join([
            json.dumps({'choices': [
                {'delta': {'reasoning': 'thinking...'},
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
        p = op.OpenRouterProvider()
        op.http_get = fake_get
        op.http_post_stream = fake_post
        chunks = list(p.stream('hi', model='nvidia/nemotron-3.5-lightning:free',
                               max_tokens=512, temperature=0.7))
        assert captured['get_url'] == op.MODELS_URL
        assert captured['get_auth'] is None
        assert captured['post_url'] == op.CHAT_URL
        assert captured['post_auth'].startswith('Bearer sk-or-v1-')
        body = captured['body']
        assert body['model'] == 'nvidia/nemotron-3.5-lightning:free'
        assert body['stream'] is True
        assert body['max_tokens'] == 512
        assert body['temperature'] == 0.7
        # reasoning streams as delta.reasoning
        kinds = [(c['type'], c['content']) for c in chunks
                 if c['type'] != 'text' or c['content']]
        assert ('thinking', 'thinking...') in kinds
        text = ''.join(c['content'] for c in chunks
                       if c['type'] == 'text' and c['content'])
        assert text == 'Hello world', text
        assert chunks[-1]['finish_reason'] == 'stop'
    finally:
        op.http_get = orig_get
        op.http_post_stream = orig_post
        _unset(old)
    print('PASS test_stream_payload_and_parsing')


def test_stream_error_frame_raises():
    old = _set_env('sk-or-v1-0123456789abcdef0123456789abcdef')
    orig_get = op.http_get
    orig_post = op.http_post_stream

    def fake_get(url, headers=None, **kw):
        return _fake_response(payload=MODELS_FIXTURE)

    def fake_post(url, headers=None, json_body=None, **kw):
        return _fake_response(lines=[
            b'data: ' + json.dumps(
                {'error': {'message': 'Rate limit exceeded: free-models-per-day',
                           'code': 429}}).encode(),
        ])

    try:
        p = op.OpenRouterProvider()
        op.http_get = fake_get
        op.http_post_stream = fake_post
        try:
            list(p.stream('hi', model='nvidia/nemotron-3.5-lightning:free'))
            raise AssertionError('expected ProviderError')
        except ProviderError as e:
            assert 'Rate limit exceeded' in str(e)
    finally:
        op.http_get = orig_get
        op.http_post_stream = orig_post
        _unset(old)
    print('PASS test_stream_error_frame_raises')


def test_stream_empty_raises_unavailable():
    old = _set_env('sk-or-v1-0123456789abcdef0123456789abcdef')
    orig_get = op.http_get
    orig_post = op.http_post_stream

    def fake_get(url, headers=None, **kw):
        return _fake_response(payload=MODELS_FIXTURE)

    def fake_post(url, headers=None, json_body=None, **kw):
        return _fake_response(lines=[b'', b'data: [DONE]'])

    try:
        p = op.OpenRouterProvider()
        op.http_get = fake_get
        op.http_post_stream = fake_post
        try:
            list(p.stream('hi', model='nvidia/nemotron-3.5-lightning:free'))
            raise AssertionError('expected ProviderUnavailableError')
        except ProviderUnavailableError:
            pass
    finally:
        op.http_get = orig_get
        op.http_post_stream = orig_post
        _unset(old)
    print('PASS test_stream_empty_raises_unavailable')


def test_http_error_classification():
    nested = ('{"error":{"message":"No cookie auth credentials found",'
              '"code":401}}')
    e = op._classify(401, nested)
    assert isinstance(e, ProviderAuthError) and 'rejected' in str(e)
    e = op._classify(403, 'forbidden')
    assert isinstance(e, ProviderAuthError)
    e = op._classify(429, 'slow down', {'Retry-After': '3'})
    assert type(e).__name__ == 'ProviderRateLimitError'
    # non-auth statuses fall through to the shared classifier
    e = op._classify(503, 'overloaded')
    assert type(e).__name__ == 'ProviderUnavailableError'
    print('PASS test_http_error_classification')


def test_unknown_model_and_images():
    old = _set_env('sk-or-v1-0123456789abcdef0123456789abcdef')
    orig_get = op.http_get
    try:
        p = op.OpenRouterProvider()
        op.http_get = lambda url, headers=None, **kw: _fake_response(
            payload=MODELS_FIXTURE)
        try:
            p.stream('hi', model='nope/nope')
            raise AssertionError('expected ProviderError')
        except ProviderError as e:
            assert 'not in catalog' in str(e)
        try:
            p.stream('hi', model='nvidia/nemotron-3.5-lightning:free',
                     images=[{'mime': 'image/png', 'data': b'xx'}])
            raise AssertionError('expected ProviderError')
        except ProviderError as e:
            assert 'not vision-capable' in str(e)
        try:
            p.stream('hi', model='nvidia/nemotron-3.5-lightning:free',
                     image_generation=True)
            raise AssertionError('expected ProviderError')
        except ProviderError as e:
            assert 'image generation' in str(e)
    finally:
        op.http_get = orig_get
        _unset(old)
    print('PASS test_unknown_model_and_images')


def test_vision_payload():
    old = _set_env('sk-or-v1-0123456789abcdef0123456789abcdef')
    orig_get = op.http_get
    orig_post = op.http_post_stream
    captured = {}

    def fake_get(url, headers=None, **kw):
        return _fake_response(payload=MODELS_FIXTURE)

    def fake_post(url, headers=None, json_body=None, **kw):
        captured['body'] = json_body
        return _fake_response(lines=[
            b'data: ' + json.dumps(
                {'choices': [{'delta': {'content': 'pic'},
                              'finish_reason': 'stop'}]}).encode(),
        ])

    try:
        p = op.OpenRouterProvider()
        op.http_get = fake_get
        op.http_post_stream = fake_post
        list(p.stream('what is this',
                      model='qwen/qwen3-vision:free',
                      images=[{'mime': 'image/png', 'data': b'png-bytes'}]))
        msgs = captured['body']['messages']
        parts = msgs[0]['content']
        assert parts[0] == {'type': 'text', 'text': 'what is this'}
        assert parts[1]['type'] == 'image_url'
        assert parts[1]['image_url']['url'].startswith(
            'data:image/png;base64,')
    finally:
        op.http_get = orig_get
        op.http_post_stream = orig_post
        _unset(old)
    print('PASS test_vision_payload')


def test_refresh_liveness():
    from dsk import refresher as rf
    old = _set_env('sk-or-v1-0123456789abcdef0123456789abcdef')
    orig_get = rf._http_get
    try:
        # 429 still proves the key is accepted (auth runs first)
        rf._http_get = lambda *a, **kw: _fake_response(429, {})
        ok, msg = rf.refresh_openrouter()
        assert ok is True and '429' in msg, msg
        # nested 401 No cookie auth credentials -> rejected
        rf._http_get = lambda *a, **kw: _fake_response(401, {
            'error': {'message': 'No cookie auth credentials found',
                      'code': 401}})
        ok, msg = rf.refresh_openrouter()
        assert ok is False and 'rejected' in msg, msg
        # 200 with the usage/limit record
        rf._http_get = lambda *a, **kw: _fake_response(
            200, {'data': {'label': 'free-key', 'usage': 0.0,
                           'limit': None, 'is_free_tier': True}})
        ok, msg = rf.refresh_openrouter()
        assert ok is True and 'valid' in msg, msg
    finally:
        rf._http_get = orig_get
        _unset(old)
    print('PASS test_refresh_liveness')


def test_wiring():
    from dsk.providers import router as router_mod
    assert ('openrouter', '.openrouter_provider', 'OpenRouterProvider') \
        in router_mod.PROVIDER_MODULES
    assert router_mod.PUBLIC_PREFIX['openrouter'] == 'openrouter'
    assert router_mod.OWNED_BY['openrouter'] == 'openrouter'
    assert 'openrouter' in router_mod.AUTO_CATEGORIES['general']
    from dsk import selfheal
    assert selfheal.HEALABLE['openrouter'].name == 'openrouter_provider.py'
    assert selfheal._PROVIDER_CLASSES['openrouter'] == 'OpenRouterProvider'
    assert selfheal._MODULE_NAMES['openrouter'] == \
        'dsk.providers.openrouter_provider'
    import re
    pats = [re.compile(x) for x in selfheal._EVIDENCE_PATTERNS['openrouter']]
    assert pats[0].search('api.openrouter.ai/api/v1/key')
    assert pats[1].search('sk-or-v1-1234567890abcdef')
    assert pats[2].search('No cookie auth credentials found')
    assert pats[3].search('nemotron-3.5-lightning:free')
    from dsk import refresher
    assert refresher.REFRESH['openrouter'] is not None
    assert refresher.SIGNUP['openrouter'] is not None
    assert refresher._jar_path('openrouter').name == \
        'openrouter_cookies.json'
    old = _set_env('sk-or-v1-0123456789abcdef0123456789abcdef')
    try:
        assert refresher._has_creds('openrouter') is True
        assert refresher._load_jar('openrouter').get('api_key') == \
            'sk-or-v1-0123456789abcdef0123456789abcdef'
    finally:
        _unset(old)
    print('PASS test_wiring')


if __name__ == '__main__':
    test_catalog_parse()
    test_dormant_without_key()
    test_env_key_enables()
    test_stream_payload_and_parsing()
    test_stream_error_frame_raises()
    test_stream_empty_raises_unavailable()
    test_http_error_classification()
    test_unknown_model_and_images()
    test_vision_payload()
    test_refresh_liveness()
    test_wiring()
    print('ALL OK')
