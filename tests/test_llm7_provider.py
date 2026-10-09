"""Offline tests for the LLM7.io provider (no network)."""

import json
import os
import sys
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from dsk.providers import llm7_provider as lp  # noqa: E402
from dsk.providers.base import (  # noqa: E402
    ProviderAuthError,
    ProviderError,
    ProviderUnavailableError,
)


def _set_env(value):
    old = os.environ.get('LLM7_API_KEY')
    os.environ['LLM7_API_KEY'] = value
    return old


def _unset(old):
    if old is None:
        os.environ.pop('LLM7_API_KEY', None)
    else:
        os.environ['LLM7_API_KEY'] = old


def _entry(mid, usage_based_only=False, model_type='chat', stream=True,
           caps=None, context=131072, modalities=None, owned_by='',
           reasoning=False):
    return {
        'id': mid,
        'model_type': model_type,
        'tier': 'turbo',
        'stream': stream,
        'usage_based_only': usage_based_only,
        'owned_by': owned_by,
        'capabilities': caps if caps is not None else
        {'vision': False, 'tools': True, 'reasoning': False,
         'json_mode': True, 'stream': True},
        'context_window': {'tokens': context, 'chars': context * 4},
        'modalities': modalities if modalities is not None else
        {'input': ['text'], 'output': ['text']},
        'reasoning': reasoning,
    }


def test_catalog_parse():
    data = {'data': [
        _entry('gpt-oss:20b'),
        _entry('minimax-m3', caps={'vision': True, 'reasoning': False,
                                   'stream': True},
               modalities={'input': ['text', 'image'],
                           'output': ['text']}),
        _entry('deepseek-v4-flash', caps={'vision': False,
                                          'reasoning': True,
                                          'stream': True}),
        # regex fallback on the id, no capability flags at all
        {'id': 'kimi/deepseek-r1-distill', 'model_type': 'chat'},
        # usage_based_only needs paid credit -> skipped
        _entry('gpt-5-paid', usage_based_only=True),
        # embeddings/rerank have no chat endpoint -> skipped
        _entry('bge-m3-embed', model_type='embedding'),
        # non-streaming entries cannot satisfy this contract -> skipped
        _entry('slow-thinker', stream=False),
        'not-a-dict',
        {'tier': 'turbo'},  # no id -> skipped
    ]}
    models = lp._parse_models(data)
    assert [m['id'] for m in models] == [
        'gpt-oss:20b',
        'minimax-m3',
        'deepseek-v4-flash',
        'kimi/deepseek-r1-distill',
    ]
    m0 = models[0]
    assert m0['context'] == 131072
    assert m0['max_out'] == lp.LLM7_MAX_OUTPUT_FALLBACK
    # empty owned_by falls back to the provider name
    assert m0['owned_by'] == 'llm7'
    assert m0['vision'] is False
    assert m0['thinking'] is False
    # vision from capabilities.vision OR image in modalities.input
    assert models[1]['vision'] is True
    # thinking from capabilities.reasoning
    assert models[2]['thinking'] is True
    # no metadata: fallback context + id-regex thinking fallback
    m3 = models[3]
    assert m3['context'] == lp.LLM7_CONTEXT_FALLBACK
    assert m3['max_out'] == lp.LLM7_MAX_OUTPUT_FALLBACK
    assert m3['thinking'] is True
    # list payload also parses; non-list shapes give no models
    assert len(lp._parse_models([_entry('x/y')])) == 1
    assert lp._parse_models({'nope': 1}) == []
    # a per-model max_output is clamped to the context window
    big = _entry('big', context=8192)
    big['max_completion_tokens'] = 999999
    assert lp._parse_models([big])[0]['max_out'] == 8192
    print('PASS test_catalog_parse')


def test_dormant_without_key():
    old = _set_env('')
    import dsk.providers.jar as jar_mod
    orig = jar_mod.load_jar
    jar_mod.load_jar = lambda name: {}
    lp.load_jar = jar_mod.load_jar  # rebind the imported name
    try:
        p = lp.Llm7Provider()
        assert p.available() is False
        try:
            p.list_models()
            raise AssertionError('expected ProviderAuthError')
        except ProviderAuthError as e:
            assert 'dash.llm7.io' in str(e)
    finally:
        jar_mod.load_jar = orig
        lp.load_jar = orig
        _unset(old)
    print('PASS test_dormant_without_key')


def test_env_key_enables():
    old = _set_env('llm7-test-key-0123456789abcdef')
    try:
        p = lp.Llm7Provider()
        assert p.available() is True
        hdr = lp._headers()
        assert hdr['Authorization'] == \
            'Bearer llm7-test-key-0123456789abcdef'
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
    _entry('gpt-oss:20b'),
    _entry('minimax-m3', caps={'vision': True, 'reasoning': False,
                               'stream': True},
           modalities={'input': ['text', 'image'],
                       'output': ['text']}),
    _entry('deepseek-v4-flash', caps={'vision': False,
                                      'reasoning': True,
                                      'stream': True}),
]}


def test_stream_payload_and_parsing():
    old = _set_env('llm7-test-key-0123456789abcdef')
    orig_get = lp.http_get
    orig_post = lp.http_post_stream
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
        p = lp.Llm7Provider()
        lp.http_get = fake_get
        lp.http_post_stream = fake_post
        chunks = list(p.stream('hi', model='gpt-oss:20b',
                               max_tokens=512, temperature=0.7))
        assert captured['get_url'] == lp.MODELS_URL
        assert captured['get_auth'] is None
        assert captured['post_url'] == lp.CHAT_URL
        assert captured['post_auth'].startswith('Bearer llm7-test-key')
        body = captured['body']
        assert body['model'] == 'gpt-oss:20b'
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
        lp.http_get = orig_get
        lp.http_post_stream = orig_post
        _unset(old)
    print('PASS test_stream_payload_and_parsing')


def test_stream_error_frame_raises():
    old = _set_env('llm7-test-key-0123456789abcdef')
    orig_get = lp.http_get
    orig_post = lp.http_post_stream
    try:
        p = lp.Llm7Provider()
        lp.http_get = lambda url, headers=None, **kw: _fake_response(
            payload=MODELS_FIXTURE)

        def fake_post(url, headers=None, json_body=None, **kw):
            return _fake_response(lines=[
                b'data: ' + json.dumps(
                    {'error': {'message': 'Daily token quota exhausted',
                               'code': 429}}).encode(),
            ])
        lp.http_post_stream = fake_post
        try:
            list(p.stream('hi', model='gpt-oss:20b'))
            raise AssertionError('expected ProviderError')
        except ProviderError as e:
            assert 'Daily token quota' in str(e)
    finally:
        lp.http_get = orig_get
        lp.http_post_stream = orig_post
        _unset(old)
    print('PASS test_stream_error_frame_raises')


def test_stream_empty_raises_unavailable():
    old = _set_env('llm7-test-key-0123456789abcdef')
    orig_get = lp.http_get
    orig_post = lp.http_post_stream
    try:
        p = lp.Llm7Provider()
        lp.http_get = lambda url, headers=None, **kw: _fake_response(
            payload=MODELS_FIXTURE)
        lp.http_post_stream = lambda url, headers=None, json_body=None, **kw: \
            _fake_response(lines=[b'', b'data: [DONE]'])
        try:
            list(p.stream('hi', model='gpt-oss:20b'))
            raise AssertionError('expected ProviderUnavailableError')
        except ProviderUnavailableError:
            pass
    finally:
        lp.http_get = orig_get
        lp.http_post_stream = orig_post
        _unset(old)
    print('PASS test_stream_empty_raises_unavailable')


def test_http_error_classification():
    missing = ('{"error":{"message":"Missing API key.",'
               '"type":"authentication_error","code":"missing_api_key"}}')
    e = lp._classify(401, missing)
    assert isinstance(e, ProviderAuthError) and 'rejected' in str(e)
    invalid = ('{"error":{"message":"Your API key is invalid, expired, or '
               'revoked.","code":"invalid_api_key"}}')
    e = lp._classify(401, invalid)
    assert isinstance(e, ProviderAuthError)
    e = lp._classify(403, 'forbidden')
    assert isinstance(e, ProviderAuthError)
    e = lp._classify(429, 'slow down', {'Retry-After': '3'})
    assert type(e).__name__ == 'ProviderRateLimitError'
    # non-auth statuses fall through to the shared classifier
    e = lp._classify(503, 'overloaded')
    assert type(e).__name__ == 'ProviderUnavailableError'
    print('PASS test_http_error_classification')


def test_unknown_model_and_images():
    old = _set_env('llm7-test-key-0123456789abcdef')
    orig_get = lp.http_get
    try:
        p = lp.Llm7Provider()
        lp.http_get = lambda url, headers=None, **kw: _fake_response(
            payload=MODELS_FIXTURE)
        try:
            p.stream('hi', model='nope/nope')
            raise AssertionError('expected ProviderError')
        except ProviderError as e:
            assert 'not in catalog' in str(e)
        try:
            p.stream('hi', model='gpt-oss:20b',
                     images=[{'mime': 'image/png', 'data': b'xx'}])
            raise AssertionError('expected ProviderError')
        except ProviderError as e:
            assert 'not vision-capable' in str(e)
        try:
            p.stream('hi', model='gpt-oss:20b', image_generation=True)
            raise AssertionError('expected ProviderError')
        except ProviderError as e:
            assert 'image generation' in str(e)
    finally:
        lp.http_get = orig_get
        _unset(old)
    print('PASS test_unknown_model_and_images')


def test_vision_payload():
    old = _set_env('llm7-test-key-0123456789abcdef')
    orig_get = lp.http_get
    orig_post = lp.http_post_stream
    captured = {}
    try:
        p = lp.Llm7Provider()
        lp.http_get = lambda url, headers=None, **kw: _fake_response(
            payload=MODELS_FIXTURE)

        def fake_post(url, headers=None, json_body=None, **kw):
            captured['body'] = json_body
            return _fake_response(lines=[
                b'data: ' + json.dumps(
                    {'choices': [{'delta': {'content': 'pic'},
                                  'finish_reason': 'stop'}]}).encode(),
            ])
        lp.http_post_stream = fake_post
        list(p.stream('what is this',
                      model='minimax-m3',
                      images=[{'mime': 'image/png', 'data': b'png-bytes'}]))
        msgs = captured['body']['messages']
        parts = msgs[0]['content']
        assert parts[0] == {'type': 'text', 'text': 'what is this'}
        assert parts[1]['type'] == 'image_url'
        assert parts[1]['image_url']['url'].startswith(
            'data:image/png;base64,')
    finally:
        lp.http_get = orig_get
        lp.http_post_stream = orig_post
        _unset(old)
    print('PASS test_vision_payload')


def test_refresh_liveness():
    from dsk import refresher as rf
    old = _set_env('llm7-test-key-0123456789abcdef')
    orig_get = rf._http_get
    try:
        # 429 still proves the key is accepted (auth runs first)
        rf._http_get = lambda *a, **kw: _fake_response(429, {})
        ok, msg = rf.refresh_llm7()
        assert ok is True and '429' in msg, msg
        # invalid key -> rejected with the dash.llm7.io pointer
        rf._http_get = lambda *a, **kw: _fake_response(401, {
            'error': {'message': 'Your API key is invalid, expired, or '
                      'revoked.', 'code': 'invalid_api_key'}})
        ok, msg = rf.refresh_llm7()
        assert ok is False and 'rejected' in msg, msg
        # 200 with the balance record
        rf._http_get = lambda *a, **kw: _fake_response(
            200, {'balance': 95000, 'currency': 'tokens'})
        ok, msg = rf.refresh_llm7()
        assert ok is True and 'balance returned' in msg, msg
    finally:
        rf._http_get = orig_get
        _unset(old)
    print('PASS test_refresh_liveness')


def test_wiring():
    from dsk.providers import router as router_mod
    assert ('llm7', '.llm7_provider', 'Llm7Provider') \
        in router_mod.PROVIDER_MODULES
    assert router_mod.PUBLIC_PREFIX['llm7'] == 'llm7'
    assert router_mod.OWNED_BY['llm7'] == 'llm7'
    assert 'llm7' in router_mod.AUTO_CATEGORIES['general']
    from dsk import selfheal
    assert selfheal.HEALABLE['llm7'].name == 'llm7_provider.py'
    assert selfheal._PROVIDER_CLASSES['llm7'] == 'Llm7Provider'
    assert selfheal._MODULE_NAMES['llm7'] == 'dsk.providers.llm7_provider'
    import re
    pats = [re.compile(x) for x in selfheal._EVIDENCE_PATTERNS['llm7']]
    assert pats[0].search('api.llm7.io/v1/chat/completions')
    assert pats[1].search('missing_api_key')
    assert pats[2].search('invalid_api_key')
    assert pats[3].search('dash.llm7.io/#/api-keys')
    from dsk import refresher
    assert refresher.REFRESH['llm7'] is not None
    assert refresher.SIGNUP['llm7'] is not None
    assert refresher._jar_path('llm7').name == 'llm7_cookies.json'
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
