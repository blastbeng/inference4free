"""Offline tests for the Cohere provider (no network)."""

import json
import os
import sys
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from dsk.providers import cohere_provider as cp  # noqa: E402
from dsk.providers.base import (  # noqa: E402
    ProviderAuthError,
    ProviderError,
    ProviderUnavailableError,
)


def _set_env(value):
    old = os.environ.get('COHERE_API_KEY')
    os.environ['COHERE_API_KEY'] = value
    return old


def _unset(old):
    if old is None:
        os.environ.pop('COHERE_API_KEY', None)
    else:
        os.environ['COHERE_API_KEY'] = old


def _entry_native(name, methods=None, context=128000, vision=False,
                  features=None):
    return {
        'name': name,
        'endpoints': methods if methods is not None else
        ['chat', 'v2/chat'],
        'context_length': context,
        'supports_vision': vision,
        'features': features if features is not None else
        ['json_mode', 'tools'],
    }


def test_catalog_parse():
    data = {'models': [
        _entry_native('command-a-03-2025', context=256000),
        _entry_native('command-a-reasoning-08-2025'),
        _entry_native('aya-vision-32b', vision=True),
        # embeddings/rerank are filtered by endpoints
        _entry_native('embed-english-v3.0', methods=['embed']),
        _entry_native('rerank-v3.5', methods=['rerank']),
        # belt-and-braces id filter when endpoints are absent
        _entry_native('embed-multilingual-v3.0', methods=None),
        # features-based thinking flag
        _entry_native('command-r7b-12-2024',
                      features=['json_mode', 'thinking', 'tools']),
        'not-a-dict',
        {'endpoints': ['chat']},  # no name -> skipped
    ]}
    models = cp._parse_models(data)
    assert [m['id'] for m in models] == [
        'command-a-03-2025',
        'command-a-reasoning-08-2025',
        'aya-vision-32b',
        'command-r7b-12-2024',
    ]
    m0 = models[0]
    assert m0['context'] == 256000
    assert m0['max_out'] == cp.COHERE_MAX_OUTPUT_FALLBACK
    assert m0['owned_by'] == 'cohere'
    assert m0['vision'] is False
    assert m0['thinking'] is False
    # reasoning id regex
    assert models[1]['thinking'] is True
    # aya-vision: supports_vision flag OR vision in the id
    assert models[2]['vision'] is True
    # features carry the thinking flag for r7b
    assert models[3]['thinking'] is True
    # OpenAI-shape fallback (bare ids)
    compat = cp._parse_models({'data': [
        {'id': 'command-r-plus-08-2024', 'owned_by': 'cohere'},
        {'id': 'embed-english-light-v3.0'},
        {'object': 'model'},
    ]})
    assert [m['id'] for m in compat] == ['command-r-plus-08-2024']
    # compat shape has no limits -> module fallbacks
    assert compat[0]['context'] == cp.COHERE_CONTEXT_FALLBACK
    assert compat[0]['max_out'] == cp.COHERE_MAX_OUTPUT_FALLBACK
    # non-list shapes give no models
    assert cp._parse_models({'nope': 1}) == []
    # a per-model output limit is clamped to the context window
    big = _entry_native('command-a-03-2025', context=8192)
    big['max_output_tokens'] = 999999
    assert cp._parse_models({'models': [big]})[0]['max_out'] == 8192
    print('PASS test_catalog_parse')


def test_dormant_without_key():
    old = _set_env('')
    import dsk.providers.jar as jar_mod
    orig = jar_mod.load_jar
    jar_mod.load_jar = lambda name: {}
    cp.load_jar = jar_mod.load_jar  # rebind the imported name
    try:
        p = cp.CohereProvider()
        assert p.available() is False
        try:
            p.list_models()
            raise AssertionError('expected ProviderAuthError')
        except ProviderAuthError as e:
            assert 'dashboard.cohere.com' in str(e)
    finally:
        jar_mod.load_jar = orig
        cp.load_jar = orig
        _unset(old)
    print('PASS test_dormant_without_key')


def test_env_key_enables():
    old = _set_env('cohere-trial-key-0123456789abcdef0123456789')
    try:
        p = cp.CohereProvider()
        assert p.available() is True
        hdr = cp._headers()
        assert hdr['Authorization'] == \
            'Bearer cohere-trial-key-0123456789abcdef0123456789'
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


PAGE1 = {'models': [
    _entry_native('command-a-03-2025', context=256000),
    _entry_native('aya-vision-32b', vision=True),
], 'next_page_token': 'tok7'}
PAGE2 = {'models': [
    _entry_native('command-a-reasoning-08-2025'),
]}


def test_stream_payload_and_parsing():
    old = _set_env('cohere-trial-key-0123456789abcdef0123456789')
    orig_get = cp.http_get
    orig_post = cp.http_post_stream
    captured = {}
    calls = []

    def fake_get(url, headers=None, **kw):
        calls.append(url)
        captured['get_url'] = url
        captured['get_auth'] = (headers or {}).get('Authorization')
        return _fake_response(payload=PAGE1 if len(calls) == 1 else PAGE2)

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
        p = cp.CohereProvider()
        cp.http_get = fake_get
        cp.http_post_stream = fake_post
        chunks = list(p.stream('hi', model='command-a-03-2025',
                               max_tokens=512, temperature=0.7))
        # paginated native catalog: two GETs, second follows the token
        assert len(calls) == 2, calls
        assert 'page_token=tok7' in calls[1]
        assert captured['get_url'].startswith(cp.MODELS_URL)
        assert captured['get_auth'] == 'Bearer cohere-trial-key-0123456789abcdef0123456789'
        assert captured['post_url'] == cp.CHAT_URL
        assert captured['post_auth'].startswith('Bearer cohere-trial-key')
        body = captured['body']
        assert body['model'] == 'command-a-03-2025'
        assert body['stream'] is True
        assert body['max_tokens'] == 512
        assert body['temperature'] == 0.7
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


def test_stream_error_frames_raise():
    old = _set_env('cohere-trial-key-0123456789abcdef0123456789')
    orig_get = cp.http_get
    orig_post = cp.http_post_stream
    try:
        p = cp.CohereProvider()
        gp_get = lambda url, headers=None, **kw: _fake_response(payload=PAGE1)
        cp.http_get = gp_get

        # nested error frame
        def fake_post(url, headers=None, json_body=None, **kw):
            return _fake_response(lines=[
                b'data: ' + json.dumps(
                    {'error': {'message': 'too many requests'}}).encode(),
            ])
        cp.http_post_stream = fake_post
        try:
            list(p.stream('hi', model='command-a-03-2025'))
            raise AssertionError('expected ProviderError')
        except ProviderError as e:
            assert 'too many requests' in str(e)

        # FLAT error frame (Cohere style: top-level message, no choices)
        def fake_post_flat(url, headers=None, json_body=None, **kw):
            return _fake_response(lines=[
                b'data: ' + json.dumps(
                    {'id': 'abc', 'message': 'invalid model'}).encode(),
            ])
        cp.http_post_stream = fake_post_flat
        try:
            list(p.stream('hi', model='command-a-03-2025'))
            raise AssertionError('expected ProviderError')
        except ProviderError as e:
            assert 'invalid model' in str(e)
    finally:
        cp.http_get = orig_get
        cp.http_post_stream = orig_post
        _unset(old)
    print('PASS test_stream_error_frames_raise')


def test_stream_empty_raises_unavailable():
    old = _set_env('cohere-trial-key-0123456789abcdef0123456789')
    orig_get = cp.http_get
    orig_post = cp.http_post_stream
    try:
        p = cp.CohereProvider()
        cp.http_get = lambda url, headers=None, **kw: _fake_response(
            payload=PAGE1)
        cp.http_post_stream = lambda url, headers=None, json_body=None, **kw: \
            _fake_response(lines=[b'', b'data: [DONE]'])
        try:
            list(p.stream('hi', model='command-a-03-2025'))
            raise AssertionError('expected ProviderUnavailableError')
        except ProviderUnavailableError:
            pass
    finally:
        cp.http_get = orig_get
        cp.http_post_stream = orig_post
        _unset(old)
    print('PASS test_stream_empty_raises_unavailable')


def test_http_error_classification():
    # flat bodies: classification keys off the 401 status
    e = cp._classify(401, '{"id":"x","message":"no api key supplied"}')
    assert isinstance(e, ProviderAuthError) and 'rejected' in str(e)
    e = cp._classify(401, '{"id":"x","message":"Incorrect API key '
                          'provided: ***6789. You can find your API key '
                          'at https://dashboard.cohere.com/api-keys."}')
    assert isinstance(e, ProviderAuthError)
    e = cp._classify(403, 'forbidden')
    assert isinstance(e, ProviderAuthError)
    e = cp._classify(429, '{"message":"rate limited"}',
                     {'Retry-After': '3'})
    assert type(e).__name__ == 'ProviderRateLimitError'
    # non-auth statuses fall through to the shared classifier
    e = cp._classify(503, 'overloaded')
    assert type(e).__name__ == 'ProviderUnavailableError'
    print('PASS test_http_error_classification')


def test_unknown_model_and_images():
    old = _set_env('cohere-trial-key-0123456789abcdef0123456789')
    orig_get = cp.http_get
    try:
        p = cp.CohereProvider()
        cp.http_get = lambda url, headers=None, **kw: _fake_response(
            payload=PAGE1)
        try:
            p.stream('hi', model='nope/nope')
            raise AssertionError('expected ProviderError')
        except ProviderError as e:
            assert 'not in catalog' in str(e)
        try:
            p.stream('hi', model='command-a-03-2025',
                     images=[{'mime': 'image/png', 'data': b'xx'}])
            raise AssertionError('expected ProviderError')
        except ProviderError as e:
            assert 'not vision-capable' in str(e)
        try:
            p.stream('hi', model='command-a-03-2025',
                     image_generation=True)
            raise AssertionError('expected ProviderError')
        except ProviderError as e:
            assert 'image generation' in str(e)
    finally:
        cp.http_get = orig_get
        _unset(old)
    print('PASS test_unknown_model_and_images')


def test_vision_payload():
    old = _set_env('cohere-trial-key-0123456789abcdef0123456789')
    orig_get = cp.http_get
    orig_post = cp.http_post_stream
    captured = {}
    try:
        p = cp.CohereProvider()
        cp.http_get = lambda url, headers=None, **kw: _fake_response(
            payload=PAGE1)

        def fake_post(url, headers=None, json_body=None, **kw):
            captured['body'] = json_body
            return _fake_response(lines=[
                b'data: ' + json.dumps(
                    {'choices': [{'delta': {'content': 'pic'},
                                  'finish_reason': 'stop'}]}).encode(),
            ])
        cp.http_post_stream = fake_post
        list(p.stream('what is this',
                      model='aya-vision-32b',
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
    old = _set_env('cohere-trial-key-0123456789abcdef0123456789')
    orig_get = rf._http_get
    try:
        # 200 with a model page -> valid, count reported
        rf._http_get = lambda *a, **kw: _fake_response(
            200, {'models': [{'name': 'command-a-03-2025'}] * 7})
        ok, msg = rf.refresh_cohere()
        assert ok is True and '7 models visible' in msg, msg
        # flat 401 no api key supplied -> rejected
        rf._http_get = lambda *a, **kw: _fake_response(401, {
            'id': 'x', 'message': 'no api key supplied'})
        ok, msg = rf.refresh_cohere()
        assert ok is False and 'rejected' in msg, msg
        # flat 401 incorrect key -> rejected
        rf._http_get = lambda *a, **kw: _fake_response(401, {
            'id': 'x', 'message': 'Incorrect API key provided: ***6789.'})
        ok, msg = rf.refresh_cohere()
        assert ok is False and 'rejected' in msg, msg
        # 429 still proves the key is accepted (auth runs first)
        rf._http_get = lambda *a, **kw: _fake_response(429, {})
        ok, msg = rf.refresh_cohere()
        assert ok is True and '429' in msg, msg
    finally:
        rf._http_get = orig_get
        _unset(old)
    print('PASS test_refresh_liveness')


def test_wiring():
    from dsk.providers import router as router_mod
    assert ('cohere', '.cohere_provider', 'CohereProvider') \
        in router_mod.PROVIDER_MODULES
    assert router_mod.PUBLIC_PREFIX['cohere'] == 'cohere'
    assert router_mod.OWNED_BY['cohere'] == 'cohere'
    assert 'cohere' in router_mod.AUTO_CATEGORIES['general']
    from dsk import selfheal
    assert selfheal.HEALABLE['cohere'].name == 'cohere_provider.py'
    assert selfheal._PROVIDER_CLASSES['cohere'] == 'CohereProvider'
    assert selfheal._MODULE_NAMES['cohere'] == 'dsk.providers.cohere_provider'
    import re
    pats = [re.compile(x) for x in selfheal._EVIDENCE_PATTERNS['cohere']]
    assert pats[0].search('api.cohere.com/v1/models')
    assert pats[0].search('api.cohere.ai/compatibility/v1/chat/completions')
    assert pats[1].search('no api key supplied')
    assert pats[2].search('Incorrect API key provided: ***6789.')
    assert pats[3].search('dashboard.cohere.com/api-keys')
    from dsk import refresher
    assert refresher.REFRESH['cohere'] is not None
    assert refresher.SIGNUP['cohere'] is not None
    assert refresher._jar_path('cohere').name == 'cohere_cookies.json'
    print('PASS test_wiring')


if __name__ == '__main__':
    test_catalog_parse()
    test_dormant_without_key()
    test_env_key_enables()
    test_stream_payload_and_parsing()
    test_stream_error_frames_raise()
    test_stream_empty_raises_unavailable()
    test_http_error_classification()
    test_unknown_model_and_images()
    test_vision_payload()
    test_refresh_liveness()
    test_wiring()
    print('ALL OK')
