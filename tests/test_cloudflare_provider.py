"""Offline tests for the Cloudflare Workers AI provider (no network)."""

import json
import os
import re
import sys
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from dsk.providers import cloudflare_provider as cp  # noqa: E402
from dsk.providers.base import (  # noqa: E402
    ProviderAuthError,
    ProviderError,
    ProviderRateLimitError,
    ProviderUnavailableError,
)

TOKEN = 'cf-test-token-0123456789abcdef0123456789abcdef'
ACCOUNT = '0123456789abcdef0123456789abcdef'


def _set_env(token=None, account=None):
    olds = (os.environ.get('CLOUDFLARE_API_TOKEN'),
            os.environ.get('CLOUDFLARE_ACCOUNT_ID'))
    if token is not None:
        os.environ['CLOUDFLARE_API_TOKEN'] = token
    if account is not None:
        os.environ['CLOUDFLARE_ACCOUNT_ID'] = account
    return olds


def _unset(olds):
    for name, old in zip(('CLOUDFLARE_API_TOKEN',
                          'CLOUDFLARE_ACCOUNT_ID'), olds):
        if old is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = old


def _entry_native(name, task='Text Generation', context=None,
                  author='meta'):
    entry = {'name': name, 'tags': {'author': author}}
    if task is not None:
        entry['task'] = {'name': task}
    if context is not None:
        entry['properties'] = [{'property_id': 'context_window',
                                'value': str(context)}]
    return entry


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


PAGE1 = {'result': [
    _entry_native('@cf/meta/llama-3.1-8b-instruct-fp8', context=131072),
    _entry_native('@cf/llava-1.5-7b-hf', task='Image-to-Text'),
], 'result_info': {'page': 1, 'per_page': 100, 'count': 2,
                   'total_count': 3}}
PAGE2 = {'result': [
    _entry_native('@cf/deepseek-ai/deepseek-r1-distill-qwen-32b',
                  author='deepseek-ai'),
], 'result_info': {'page': 2, 'per_page': 100, 'count': 1,
                   'total_count': 3}}


def test_catalog_parse():
    data = {'result': [
        _entry_native('@cf/meta/llama-3.1-8b-instruct-fp8',
                      context=131072),
        _entry_native('@cf/llava-1.5-7b-hf', task='Image-to-Text'),
        # non-chat tasks are skipped
        _entry_native('@cf/baai/bge-base-en-v1.5',
                      task='Text Embeddings'),
        _entry_native('@cf/black-forest-labs/flux-1-schnell',
                      task='Text-to-Image'),
        _entry_native('@cf/deepseek-ai/deepseek-r1-distill-qwen-32b',
                      author='deepseek-ai'),
        # properties as a dict are tolerated too
        {'name': '@cf/qwen/qwen2.5-coder-32b-instruct',
         'task': {'name': 'Text Generation'},
         'properties': {'context_window': '32768'},
         'tags': {'author': 'qwen'}},
        'not-a-dict',
        {'task': {'name': 'Text Generation'}},  # no name -> skipped
    ]}
    models = cp._parse_models(data)
    assert [m['id'] for m in models] == [
        '@cf/meta/llama-3.1-8b-instruct-fp8',
        '@cf/llava-1.5-7b-hf',
        '@cf/deepseek-ai/deepseek-r1-distill-qwen-32b',
        '@cf/qwen/qwen2.5-coder-32b-instruct',
    ]
    m0 = models[0]
    assert m0['context'] == 131072
    assert m0['max_out'] == cp.CF_MAX_OUTPUT_FALLBACK
    assert m0['owned_by'] == 'meta'
    assert m0['vision'] is False
    assert m0['thinking'] is False
    # Image-to-Text task -> vision chat model
    assert models[1]['vision'] is True
    # distill id carries the thinking flag
    assert models[2]['thinking'] is True
    assert models[2]['owned_by'] == 'deepseek-ai'
    # dict-shaped properties parse as well
    assert models[3]['context'] == 32768
    # bare OpenAI-shape fallback: id filter only (bge is non-chat)
    compat = cp._parse_models({'data': [
        {'id': '@cf/meta/llama-3.1-8b-instruct-fp8',
         'owned_by': 'meta'},
        {'id': '@cf/baai/bge-base-en-v1.5'},
        {'object': 'model'},
    ]})
    assert [m['id'] for m in compat] == \
        ['@cf/meta/llama-3.1-8b-instruct-fp8']
    # compat shape has no limits -> module fallbacks
    assert compat[0]['context'] == cp.CF_CONTEXT_FALLBACK
    assert compat[0]['max_out'] == cp.CF_MAX_OUTPUT_FALLBACK
    # non-list shapes give no models
    assert cp._parse_models({'nope': 1}) == []
    # a per-model output limit is clamped to the context window
    big = _entry_native('@cf/meta/llama-3.1-8b-instruct-fp8',
                        context=8192)
    big['max_output_tokens'] = 999999
    assert cp._parse_models({'result': [big]})[0]['max_out'] == 8192
    print('PASS test_catalog_parse')


def test_dormant_without_credentials():
    olds = _set_env('', '')
    import dsk.providers.jar as jar_mod
    orig = jar_mod.load_jar
    jar_mod.load_jar = lambda name: {}
    cp.load_jar = jar_mod.load_jar  # rebind the imported name
    try:
        p = cp.CloudflareProvider()
        assert p.available() is False
        try:
            p.list_models()
            raise AssertionError('expected ProviderAuthError')
        except ProviderAuthError as e:
            assert 'dash.cloudflare.com' in str(e)
            assert 'CLOUDFLARE_API_TOKEN' in str(e)
    finally:
        jar_mod.load_jar = orig
        cp.load_jar = orig
        _unset(olds)
    print('PASS test_dormant_without_credentials')


def test_env_pair_enables():
    # token alone is NOT enough - the account id is required too
    olds = _set_env(TOKEN, None)
    import dsk.providers.jar as jar_mod
    orig = jar_mod.load_jar
    jar_mod.load_jar = lambda name: {}
    cp.load_jar = jar_mod.load_jar
    try:
        p = cp.CloudflareProvider()
        assert p.available() is False
        os.environ['CLOUDFLARE_ACCOUNT_ID'] = ACCOUNT
        assert p.available() is True
        hdr = cp._headers()
        assert hdr['Authorization'] == f'Bearer {TOKEN}'
        assert cp._models_url(ACCOUNT, page=2).endswith(
            f'/accounts/{ACCOUNT}/ai/models/search'
            '?per_page=100&page=2')
        assert cp._chat_url(ACCOUNT).endswith(
            f'/accounts/{ACCOUNT}/ai/v1/chat/completions')
    finally:
        jar_mod.load_jar = orig
        cp.load_jar = orig
        _unset(olds)
    print('PASS test_env_pair_enables')


def test_stream_payload_and_parsing():
    olds = _set_env(TOKEN, ACCOUNT)
    p = cp.CloudflareProvider()
    orig_get = cp.http_get
    orig_post = cp.http_post_stream
    captured = {}
    calls = []

    def fake_get(url, headers=None, **kw):
        calls.append(url)
        captured['get_url'] = url
        captured['get_auth'] = (headers or {}).get('Authorization')
        return _fake_response(
            payload=PAGE1 if len(calls) == 1 else PAGE2)

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
        return _fake_response(
            lines=[b'data: ' + l for l in sse.split(b'\n')])

    try:
        cp.http_get = fake_get
        cp.http_post_stream = fake_post
        chunks = list(p.stream(
            'hi', model='@cf/meta/llama-3.1-8b-instruct-fp8',
            max_tokens=512, temperature=0.7))
        # paginated v4 catalog: two GETs, the second follows page 2
        assert len(calls) == 2, calls
        assert 'page=2' in calls[1], calls
        assert captured['get_url'].startswith(
            f'{cp.CF_API_BASE}/accounts/{ACCOUNT}/ai/models/search')
        assert captured['get_auth'] == f'Bearer {TOKEN}'
        assert captured['post_url'] == \
            f'{cp.CF_API_BASE}/accounts/{ACCOUNT}/ai/v1/chat/completions'
        assert captured['post_auth'] == f'Bearer {TOKEN}'
        body = captured['body']
        assert body['model'] == '@cf/meta/llama-3.1-8b-instruct-fp8'
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
        _unset(olds)
    print('PASS test_stream_payload_and_parsing')


def test_stream_error_frames_raise():
    olds = _set_env(TOKEN, ACCOUNT)
    p = cp.CloudflareProvider()
    orig_get = cp.http_get
    orig_post = cp.http_post_stream
    try:
        cp.http_get = lambda url, headers=None, **kw: \
            _fake_response(payload=PAGE1)

        # v4 flat errors array (rate limit mid-stream)
        def fake_post_v4(url, headers=None, json_body=None, **kw):
            return _fake_response(lines=[
                b'data: ' + json.dumps(
                    {'errors': [{'code': 7502,
                                 'message': 'rate limit exceeded'}]}
                ).encode(),
            ])
        cp.http_post_stream = fake_post_v4
        try:
            list(p.stream('hi',
                          model='@cf/meta/llama-3.1-8b-instruct-fp8'))
            raise AssertionError('expected ProviderError')
        except ProviderError as e:
            assert 'rate limit exceeded' in str(e)

        # OpenAI nested error frame
        def fake_post_nested(url, headers=None, json_body=None, **kw):
            return _fake_response(lines=[
                b'data: ' + json.dumps(
                    {'error': {'message': 'model not found'}}).encode(),
            ])
        cp.http_post_stream = fake_post_nested
        try:
            list(p.stream('hi',
                          model='@cf/meta/llama-3.1-8b-instruct-fp8'))
            raise AssertionError('expected ProviderError')
        except ProviderError as e:
            assert 'model not found' in str(e)

        # flat v4 failure envelope (success:false, no choices)
        def fake_post_flat(url, headers=None, json_body=None, **kw):
            return _fake_response(lines=[
                b'data: ' + json.dumps(
                    {'message': 'quota exhausted',
                     'success': False}).encode(),
            ])
        cp.http_post_stream = fake_post_flat
        try:
            list(p.stream('hi',
                          model='@cf/meta/llama-3.1-8b-instruct-fp8'))
            raise AssertionError('expected ProviderError')
        except ProviderError as e:
            assert 'quota exhausted' in str(e)
    finally:
        cp.http_get = orig_get
        cp.http_post_stream = orig_post
        _unset(olds)
    print('PASS test_stream_error_frames_raise')


def test_stream_empty_raises_unavailable():
    olds = _set_env(TOKEN, ACCOUNT)
    p = cp.CloudflareProvider()
    orig_get = cp.http_get
    orig_post = cp.http_post_stream
    try:
        cp.http_get = lambda url, headers=None, **kw: \
            _fake_response(payload=PAGE1)
        cp.http_post_stream = lambda url, headers=None, json_body=None, **kw: \
            _fake_response(lines=[b'', b'data: [DONE]'])
        try:
            list(p.stream('hi',
                          model='@cf/meta/llama-3.1-8b-instruct-fp8'))
            raise AssertionError('expected ProviderUnavailableError')
        except ProviderUnavailableError:
            pass
    finally:
        cp.http_get = orig_get
        cp.http_post_stream = orig_post
        _unset(olds)
    print('PASS test_stream_empty_raises_unavailable')


def test_http_error_classification():
    # v4 auth errors arrive under several statuses; all read as auth
    e = cp._classify(401, json.dumps(
        {'success': False,
         'errors': [{'code': 10000, 'message': 'Authentication error'}]}))
    assert isinstance(e, ProviderAuthError) and 'rejected' in str(e)
    assert 'Authentication error' in str(e)
    e = cp._classify(403, 'forbidden')
    assert isinstance(e, ProviderAuthError)
    # 400 code 9106: missing auth headers
    e = cp._classify(400, json.dumps(
        {'success': False,
         'errors': [{'code': 9106,
                     'message': 'Missing X-Auth-Key, X-Auth-Email or '
                                'Authorization headers'}]}))
    assert isinstance(e, ProviderAuthError), e
    # 404 code 7003: wrong account id (routing precedes auth)
    e = cp._classify(404, json.dumps(
        {'success': False,
         'errors': [{'code': 7003,
                     'message': 'Could not route to /accounts/x/ai - '
                                'perhaps your object identifier is '
                                'invalid?'}]}))
    assert isinstance(e, ProviderAuthError), e
    # a plain 400 without auth-shaped text falls through
    e = cp._classify(400, '{"error": "bad request"}')
    assert not isinstance(e, ProviderAuthError)
    # free-tier 429 maps to the rate-limit class
    e = cp._classify(429, '{"message": "rate limited"}',
                     {'Retry-After': '3'})
    assert isinstance(e, ProviderRateLimitError)
    # 5xx falls through to the shared classifier
    e = cp._classify(503, 'overloaded')
    assert isinstance(e, ProviderUnavailableError)
    # _v4_errors summarizes the errors array; raw text otherwise
    assert cp._v4_errors('{"errors":[{"message":"a"},{"message":"b"}]}') \
        == 'a; b'
    assert cp._v4_errors('plain text') == 'plain text'
    print('PASS test_http_error_classification')


def test_unknown_model_and_images():
    olds = _set_env(TOKEN, ACCOUNT)
    p = cp.CloudflareProvider()
    orig_get = cp.http_get
    orig_post = cp.http_post_stream
    try:
        cp.http_get = lambda url, headers=None, **kw: \
            _fake_response(payload=PAGE1)
        cp.http_post_stream = lambda url, headers=None, json_body=None, **kw: \
            _fake_response(lines=[b'data: [DONE]'])
        try:
            p.stream('hi', model='nope/nope')
            raise AssertionError('expected ProviderError')
        except ProviderError as e:
            assert 'not in catalog' in str(e)
        # llama is not vision-capable: images are refused
        try:
            p.stream('hi', model='@cf/meta/llama-3.1-8b-instruct-fp8',
                     images=[{'mime': 'image/png', 'data': b'xx'}])
            raise AssertionError('expected ProviderError')
        except ProviderError as e:
            assert 'not vision-capable' in str(e)
        try:
            p.stream('hi', model='@cf/meta/llama-3.1-8b-instruct-fp8',
                     image_generation=True)
            raise AssertionError('expected ProviderError')
        except ProviderError as e:
            assert 'image generation' in str(e)
    finally:
        cp.http_get = orig_get
        cp.http_post_stream = orig_post
        _unset(olds)
    print('PASS test_unknown_model_and_images')


def test_vision_payload():
    olds = _set_env(TOKEN, ACCOUNT)
    p = cp.CloudflareProvider()
    orig_get = cp.http_get
    orig_post = cp.http_post_stream
    captured = {}
    try:
        cp.http_get = lambda url, headers=None, **kw: \
            _fake_response(payload=PAGE1)

        def fake_post(url, headers=None, json_body=None, **kw):
            captured['body'] = json_body
            return _fake_response(lines=[
                b'data: ' + json.dumps(
                    {'choices': [{'delta': {'content': 'pic'},
                                  'finish_reason': 'stop'}]}).encode(),
            ])
        cp.http_post_stream = fake_post
        list(p.stream('what is this',
                      model='@cf/llava-1.5-7b-hf',
                      images=[{'mime': 'image/png',
                               'data': b'png-bytes'}]))
        msgs = captured['body']['messages']
        parts = msgs[0]['content']
        assert parts[0] == {'type': 'text', 'text': 'what is this'}
        assert parts[1]['type'] == 'image_url'
        assert parts[1]['image_url']['url'].startswith(
            'data:image/png;base64,')
    finally:
        cp.http_get = orig_get
        cp.http_post_stream = orig_post
        _unset(olds)
    print('PASS test_vision_payload')


def test_refresh_liveness():
    from dsk import refresher as rf
    olds = _set_env(TOKEN, ACCOUNT)
    orig_get = rf._http_get
    captured = {}

    def fake(url, cookies=None, timeout=30, headers=None, **kw):
        captured['url'] = url
        captured['auth'] = (headers or {}).get('Authorization')
        return _fake_response(
            200, {'success': True,
                  'result': [{'name': '@cf/meta/llama-3.1-8b-instruct-fp8'}],
                  'result_info': {'page': 1, 'per_page': 1, 'count': 1,
                                  'total_count': 42}})

    try:
        # 200 success envelope with total_count -> valid, count reported
        rf._http_get = fake
        ok, msg = rf.refresh_cloudflare()
        assert ok is True and '42 models visible' in msg, msg
        assert captured['url'] == (
            'https://api.cloudflare.com/client/v4/accounts/'
            f'{ACCOUNT}/ai/models/search?per_page=1&page=1'), captured
        assert captured['auth'] == f'Bearer {TOKEN}'
        # 401 code 10000 -> rejected
        rf._http_get = lambda *a, **kw: _fake_response(401, {
            'success': False,
            'errors': [{'code': 10000,
                        'message': 'Authentication error'}]})
        ok, msg = rf.refresh_cloudflare()
        assert ok is False and 'rejected' in msg, msg
        # 404 code 7003 (wrong account id) -> rejected
        rf._http_get = lambda *a, **kw: _fake_response(404, {
            'success': False,
            'errors': [{'code': 7003,
                        'message': 'object identifier is invalid'}]})
        ok, msg = rf.refresh_cloudflare()
        assert ok is False and 'rejected' in msg, msg
        # success:false envelope on HTTP 200 -> rejected with the reason
        rf._http_get = lambda *a, **kw: _fake_response(200, {
            'success': False,
            'errors': [{'code': 10000, 'message': 'Authentication error'}]})
        ok, msg = rf.refresh_cloudflare()
        assert ok is False and 'Authentication error' in msg, msg
        # 429 still proves the token routed and was accepted
        rf._http_get = lambda *a, **kw: _fake_response(429, {})
        ok, msg = rf.refresh_cloudflare()
        assert ok is True and '429' in msg, msg
    finally:
        rf._http_get = orig_get
        _unset(olds)
    print('PASS test_refresh_liveness')


def test_has_creds_requires_both():
    from dsk import refresher as rf
    olds = _set_env('', '')
    orig_load = rf._load_jar
    rf._load_jar = lambda name: ({} if name == 'cloudflare'
                                 else orig_load(name))
    try:
        # nothing -> False
        assert rf._has_creds('cloudflare') is False
        # token alone -> False (account id required)
        os.environ['CLOUDFLARE_API_TOKEN'] = TOKEN
        assert rf._has_creds('cloudflare') is False
        # both env -> True
        os.environ['CLOUDFLARE_ACCOUNT_ID'] = ACCOUNT
        assert rf._has_creds('cloudflare') is True
        os.environ.pop('CLOUDFLARE_API_TOKEN')
        os.environ.pop('CLOUDFLARE_ACCOUNT_ID')
        # jar with both fields -> True
        rf._load_jar = lambda name: ({'api_key': TOKEN,
                                      'account_id': ACCOUNT}
                                     if name == 'cloudflare'
                                     else orig_load(name))
        assert rf._has_creds('cloudflare') is True
        # jar with only one field -> False
        rf._load_jar = lambda name: ({'api_key': TOKEN}
                                     if name == 'cloudflare'
                                     else orig_load(name))
        assert rf._has_creds('cloudflare') is False
    finally:
        rf._load_jar = orig_load
        _unset(olds)
    print('PASS test_has_creds_requires_both')


def test_wiring():
    from dsk.providers import router as router_mod
    assert ('cloudflare', '.cloudflare_provider',
            'CloudflareProvider') in router_mod.PROVIDER_MODULES
    assert router_mod.PUBLIC_PREFIX['cloudflare'] == 'cloudflare'
    assert router_mod.OWNED_BY['cloudflare'] == 'cloudflare'
    assert 'cloudflare' in router_mod.AUTO_CATEGORIES['general']
    from dsk import selfheal
    assert selfheal.HEALABLE['cloudflare'].name == \
        'cloudflare_provider.py'
    assert selfheal._PROVIDER_CLASSES['cloudflare'] == \
        'CloudflareProvider'
    assert selfheal._MODULE_NAMES['cloudflare'] == \
        'dsk.providers.cloudflare_provider'
    pats = [re.compile(x)
            for x in selfheal._EVIDENCE_PATTERNS['cloudflare']]
    assert pats[0].search(
        'api.cloudflare.com/client/v4/accounts/x/ai/models/search')
    assert pats[0].search(
        'api.cloudflare.com/client/v4/accounts/x/ai/v1/chat/completions')
    assert pats[1].search('Authentication error')
    assert pats[2].search('object identifier is invalid')
    assert pats[3].search('dash.cloudflare.com/profile/api-tokens')
    from dsk import refresher
    assert refresher.REFRESH['cloudflare'] is not None
    assert refresher.SIGNUP['cloudflare'] is not None
    assert refresher._jar_path('cloudflare').name == \
        'cloudflare_cookies.json'
    print('PASS test_wiring')


if __name__ == '__main__':
    test_catalog_parse()
    test_dormant_without_credentials()
    test_env_pair_enables()
    test_stream_payload_and_parsing()
    test_stream_error_frames_raise()
    test_stream_empty_raises_unavailable()
    test_http_error_classification()
    test_unknown_model_and_images()
    test_vision_payload()
    test_refresh_liveness()
    test_has_creds_requires_both()
    test_wiring()
    print('ALL OK')
