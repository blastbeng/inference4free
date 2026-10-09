"""Offline tests for the Mistral API provider (no network)."""

import json
import os
import sys
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from dsk.providers import mistral_api_provider as mp  # noqa: E402
from dsk.providers.base import (  # noqa: E402
    ProviderAuthError,
    ProviderError,
    ProviderUnavailableError,
)


def _set_env(value):
    old = os.environ.get('MISTRAL_API_KEY')
    os.environ['MISTRAL_API_KEY'] = value
    return old


def _unset(old):
    if old is None:
        os.environ.pop('MISTRAL_API_KEY', None)
    else:
        os.environ['MISTRAL_API_KEY'] = old


def test_catalog_parse():
    # Mistral's catalog is a BARE JSON array (no {"data": ...} wrapper)
    data = [
        {'id': 'mistral-large-latest', 'owned_by': 'mistral',
         'max_context_length': 131072,
         'capabilities': {'completion_chat': True, 'vision': False,
                          'completion_fim': False}},
        {'id': 'mistral-old-0613', 'owned_by': 'mistral',
         'max_context_length': 32768, 'deprecation': {'date': '2025-01-01'},
         'capabilities': {'completion_chat': True}},
        {'id': 'codestral-latest', 'owned_by': 'mistral',
         'max_context_length': 262144,
         'capabilities': {'completion_chat': False, 'completion_fim': True}},
        {'id': 'pixtral-large-latest', 'owned_by': 'mistral',
         'max_context_length': 131072,
         'capabilities': {'completion_chat': True, 'vision': True}},
        {'id': 'mistral-vision-x', 'owned_by': 'mistral',
         'capabilities': {'completion_chat': True, 'vision': False}},
        'not-a-dict',
        {'displayName': 'no-id'},
    ]
    models = mp._parse_models(data)
    # deprecated and FIM-only entries are skipped
    assert [m['id'] for m in models] == ['mistral-large-latest',
                                         'pixtral-large-latest',
                                         'mistral-vision-x']
    assert models[0]['context'] == 131072
    assert models[0]['max_out'] == mp.MISTRAL_MAX_OUTPUT_FALLBACK
    # vision from capabilities and from the id regex
    assert models[1]['vision'] is True
    assert models[2]['vision'] is True
    assert models[0]['vision'] is False
    # wrapped payload also parses; non-list shapes give no models
    assert len(mp._parse_models({'data': [{'id': 'wrapped'}]})) == 1
    assert mp._parse_models({'nope': 1}) == []
    print('PASS test_catalog_parse')


def test_dormant_without_key():
    old = _set_env('')
    import dsk.providers.jar as jar_mod
    orig = jar_mod.load_jar
    jar_mod.load_jar = lambda name: {}
    mp.load_jar = jar_mod.load_jar  # rebind the imported name
    try:
        p = mp.MistralApiProvider()
        assert p.available() is False
        try:
            p.list_models()
            raise AssertionError('expected ProviderAuthError')
        except ProviderAuthError as e:
            assert 'console.mistral.ai' in str(e)
    finally:
        jar_mod.load_jar = orig
        mp.load_jar = orig
        _unset(old)
    print('PASS test_dormant_without_key')


def test_env_key_enables():
    old = _set_env('msk-abcdef0123456789abcdef0123456789')
    try:
        p = mp.MistralApiProvider()
        assert p.available() is True
        hdr = mp._headers()
        assert hdr['Authorization'] == \
            'Bearer msk-abcdef0123456789abcdef0123456789'
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


MODELS_FIXTURE = [
    {'id': 'mistral-large-latest', 'owned_by': 'mistral',
     'max_context_length': 131072,
     'capabilities': {'completion_chat': True, 'vision': False}},
    {'id': 'magistral-medium-latest', 'owned_by': 'mistral',
     'max_context_length': 131072,
     'capabilities': {'completion_chat': True, 'vision': False}},
    {'id': 'pixtral-large-latest', 'owned_by': 'mistral',
     'max_context_length': 131072,
     'capabilities': {'completion_chat': True, 'vision': True}},
]


def test_stream_payload_and_parsing():
    old = _set_env('msk-abcdef0123456789abcdef0123456789')
    orig_get = mp.http_get
    orig_post = mp.http_post_stream
    captured = {}

    def fake_get(url, headers=None, **kw):
        captured['get_url'] = url
        # /models IS authenticated on Mistral (unlike ModelScope)
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
        p = mp.MistralApiProvider()
        mp.http_get = fake_get
        mp.http_post_stream = fake_post
        chunks = list(p.stream('hi', model='mistral-large-latest',
                               max_tokens=512, temperature=0.7))
        assert captured['get_url'] == mp.MODELS_URL
        assert captured['get_auth'].startswith('Bearer msk-')
        assert captured['post_url'] == mp.CHAT_URL
        assert captured['post_auth'].startswith('Bearer msk-')
        body = captured['body']
        assert body['model'] == 'mistral-large-latest'
        assert body['stream'] is True
        assert body['max_tokens'] == 512
        assert body['temperature'] == 0.7
        # reasoning streams as delta.reasoning — no opt-in parameter
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


def test_think_tags_inline():
    p = mp.MistralApiProvider()

    def run(parts):
        state = {'in_think': False, 'after_close': False, 'carry': ''}
        out = []
        for part in parts:
            out.extend(p._split_thinks(part, state))
        out.extend(p._flush_thinks(state))
        return out

    def merge(pairs):
        merged = []
        for kind, seg in pairs:
            if merged and merged[-1][0] == kind:
                merged[-1] = (kind, merged[-1][1] + seg)
            else:
                merged.append((kind, seg))
        return merged

    # tag split across chunks
    assert merge(run(['<thi', 'nk>reason here</th', 'ink>Answer'])) == \
        [('thinking', 'reason here'), ('text', 'Answer')]
    # empty <think/> opener (magistral) — thinking may span chunks
    assert merge(run(['<think/>deep thought', ' here</think', '>done'])) == \
        [('thinking', 'deep thought here'), ('text', 'done')]
    # literal '<think' that is not a tag
    assert merge(run(['the <thinker is out'])) == \
        [('text', 'the <thinker is out')]
    # attributes in the opener
    assert merge(run(['<think mode="x">why</think', '>A'])) == \
        [('thinking', 'why'), ('text', 'A')]
    # unclosed think block (opened, never closed) flushes as thinking
    assert merge(run(['Answer. ' + '<think' + '>' + 'still',
                     ' thinking'])) == \
        [('text', 'Answer. '), ('thinking', 'still thinking')]
    # '<think' immediately followed by a letter is literal text
    assert merge(run(['Answer. <think', 'still'])) == \
        [('text', 'Answer. <thinkstill')]
    # closing '>' split into its own chunk is consumed, not leaked
    assert merge(run(['<think', '>r</think', '\n>', 'A'])) == \
        [('thinking', 'r'), ('text', 'A')]
    print('PASS test_think_tags_inline')


def test_stream_error_frame_raises():
    old = _set_env('msk-abcdef0123456789abcdef0123456789')
    orig_post = mp.http_post_stream

    def fake_post(url, headers=None, json_body=None, **kw):
        return _fake_response(lines=[
            b'data: ' + json.dumps(
                {'error': {'message': 'rate limit exceeded',
                           'request_id': 'abc-123'}}).encode(),
        ])

    orig_get = mp.http_get

    def fake_get(url, headers=None, **kw):
        return _fake_response(payload=MODELS_FIXTURE)

    try:
        p = mp.MistralApiProvider()
        mp.http_get = fake_get
        mp.http_post_stream = fake_post
        try:
            list(p.stream('hi', model='mistral-large-latest'))
            raise AssertionError('expected ProviderError')
        except ProviderError as e:
            assert 'rate limit exceeded' in str(e)
    finally:
        mp.http_get = orig_get
        mp.http_post_stream = orig_post
        _unset(old)
    print('PASS test_stream_error_frame_raises')


def test_stream_object_error_frame_raises():
    old = _set_env('msk-abcdef0123456789abcdef0123456789')
    orig_post = mp.http_post_stream

    def fake_post(url, headers=None, json_body=None, **kw):
        return _fake_response(lines=[
            b'data: ' + json.dumps(
                {'object': 'error', 'message': 'quota exhausted',
                 'type': 'limit_reached'}).encode(),
        ])

    orig_get = mp.http_get

    def fake_get(url, headers=None, **kw):
        return _fake_response(payload=MODELS_FIXTURE)

    try:
        p = mp.MistralApiProvider()
        mp.http_get = fake_get
        mp.http_post_stream = fake_post
        try:
            list(p.stream('hi', model='mistral-large-latest'))
            raise AssertionError('expected ProviderError')
        except ProviderError as e:
            assert 'quota exhausted' in str(e)
    finally:
        mp.http_get = orig_get
        mp.http_post_stream = orig_post
        _unset(old)
    print('PASS test_stream_object_error_frame_raises')


def test_stream_empty_raises_unavailable():
    old = _set_env('msk-abcdef0123456789abcdef0123456789')
    orig_post = mp.http_post_stream

    def fake_post(url, headers=None, json_body=None, **kw):
        return _fake_response(lines=[b'', b'data: [DONE]'])

    orig_get = mp.http_get

    def fake_get(url, headers=None, **kw):
        return _fake_response(payload=MODELS_FIXTURE)

    try:
        p = mp.MistralApiProvider()
        mp.http_get = fake_get
        mp.http_post_stream = fake_post
        try:
            list(p.stream('hi', model='mistral-large-latest'))
            raise AssertionError('expected ProviderUnavailableError')
        except ProviderUnavailableError:
            pass
    finally:
        mp.http_get = orig_get
        mp.http_post_stream = orig_post
        _unset(old)
    print('PASS test_stream_empty_raises_unavailable')


def test_http_error_classification():
    flat = '{"detail":"Invalid API Key"}'
    e = mp._classify(401, flat)
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
    old = _set_env('msk-abcdef0123456789abcdef0123456789')
    orig_get = mp.http_get
    try:
        p = mp.MistralApiProvider()
        mp.http_get = lambda url, headers=None, **kw: _fake_response(
            payload=MODELS_FIXTURE)
        try:
            p.stream('hi', model='nope/nope')
            raise AssertionError('expected ProviderError')
        except ProviderError as e:
            assert 'not in catalog' in str(e)
        try:
            p.stream('hi', model='mistral-large-latest',
                     images=[{'mime': 'image/png', 'data': b'xx'}])
            raise AssertionError('expected ProviderError')
        except ProviderError as e:
            assert 'not vision-capable' in str(e)
        try:
            p.stream('hi', model='mistral-large-latest',
                     image_generation=True)
            raise AssertionError('expected ProviderError')
        except ProviderError as e:
            assert 'image generation' in str(e)
    finally:
        mp.http_get = orig_get
        _unset(old)
    print('PASS test_unknown_model_and_images')


def test_vision_payload():
    old = _set_env('msk-abcdef0123456789abcdef0123456789')
    orig_get = mp.http_get
    orig_post = mp.http_post_stream
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
        p = mp.MistralApiProvider()
        mp.http_get = fake_get
        mp.http_post_stream = fake_post
        list(p.stream('what is this',
                      model='pixtral-large-latest',
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
    old = _set_env('msk-abcdef0123456789abcdef0123456789')
    orig_get = rf._http_get
    try:
        # 429 still proves the key is accepted (auth runs first)
        rf._http_get = lambda *a, **kw: _fake_response(429, {})
        ok, msg = rf.refresh_mistral_api()
        assert ok is True and '429' in msg, msg
        # flat 401 Invalid API Key -> rejected
        rf._http_get = lambda *a, **kw: _fake_response(
            401, {'detail': 'Invalid API Key'})
        ok, msg = rf.refresh_mistral_api()
        assert ok is False and 'rejected' in msg, msg
        # 200 with the bare-array catalog
        rf._http_get = lambda *a, **kw: _fake_response(
            200, [{'id': 'mistral-large-latest'}])
        ok, msg = rf.refresh_mistral_api()
        assert ok is True and '1 models visible' in msg, msg
    finally:
        rf._http_get = orig_get
        _unset(old)
    print('PASS test_refresh_liveness')


def test_wiring():
    from dsk.providers import router as router_mod
    assert ('mistral_api', '.mistral_api_provider', 'MistralApiProvider') \
        in router_mod.PROVIDER_MODULES
    assert router_mod.PUBLIC_PREFIX['mistral_api'] == 'mistral-api'
    assert router_mod.OWNED_BY['mistral_api'] == 'mistral'
    assert 'mistral_api' in router_mod.AUTO_CATEGORIES['general']
    from dsk import selfheal
    assert selfheal.HEALABLE['mistral_api'].name == 'mistral_api_provider.py'
    assert selfheal._PROVIDER_CLASSES['mistral_api'] == 'MistralApiProvider'
    assert selfheal._MODULE_NAMES['mistral_api'] == \
        'dsk.providers.mistral_api_provider'
    import re
    pats = [re.compile(x) for x in selfheal._EVIDENCE_PATTERNS['mistral_api']]
    assert pats[0].search('api.mistral.ai/v1/models')
    assert pats[1].search('{"detail":"Invalid API Key"}')
    from dsk import refresher
    assert refresher.REFRESH['mistral_api'] is not None
    assert refresher.SIGNUP['mistral_api'] is not None
    assert refresher._jar_path('mistral_api').name == \
        'mistral_api_cookies.json'
    old = _set_env('msk-abcdef0123456789abcdef0123456789')
    try:
        assert refresher._has_creds('mistral_api') is True
        assert refresher._load_jar('mistral_api').get('api_key') == \
            'msk-abcdef0123456789abcdef0123456789'
    finally:
        _unset(old)
    print('PASS test_wiring')


if __name__ == '__main__':
    test_catalog_parse()
    test_dormant_without_key()
    test_env_key_enables()
    test_stream_payload_and_parsing()
    test_think_tags_inline()
    test_stream_error_frame_raises()
    test_stream_object_error_frame_raises()
    test_stream_empty_raises_unavailable()
    test_http_error_classification()
    test_unknown_model_and_images()
    test_vision_payload()
    test_refresh_liveness()
    test_wiring()
    print('ALL OK')
