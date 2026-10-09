"""Offline tests for the Google AI Studio provider (no network)."""

import json
import os
import sys
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from dsk.providers import google_ai_studio_provider as gp  # noqa: E402
from dsk.providers.base import (  # noqa: E402
    ProviderAuthError,
    ProviderError,
    ProviderUnavailableError,
)


def _set_env(value):
    old = os.environ.get('GOOGLE_AI_STUDIO_API_KEY')
    os.environ['GOOGLE_AI_STUDIO_API_KEY'] = value
    return old


def _unset(old):
    if old is None:
        os.environ.pop('GOOGLE_AI_STUDIO_API_KEY', None)
    else:
        os.environ['GOOGLE_AI_STUDIO_API_KEY'] = old


def _entry_native(name, methods=None, context=1048576, max_out=65536,
                  display='', desc=''):
    return {
        'name': name,
        'displayName': display or name.split('/')[-1],
        'description': desc,
        'inputTokenLimit': context,
        'outputTokenLimit': max_out,
        'supportedGenerationMethods': methods
        if methods is not None else
        ['generateContent', 'countTokens'],
    }


def test_catalog_parse():
    data = {'models': [
        _entry_native('models/gemini-2.5-flash',
                      desc='image and video input, multimodal'),
        _entry_native('models/gemini-3-pro-preview'),
        _entry_native('models/gemini-2.0-flash-thinking-exp-1219'),
        # embeddings are filtered by supportedGenerationMethods
        _entry_native('models/text-embedding-004',
                      methods=['embedContent']),
        # imagen/veo likewise
        _entry_native('models/imagen-4.0-generate-001',
                      methods=['predictImages']),
        # belt-and-braces id filter when methods are absent
        _entry_native('models/gemini-2.5-flash-native-audio',
                      methods=None),
        # text-only non-gemini model: no vision heuristic hits
        _entry_native('models/gemma-2-2b-it', context=8192, max_out=4096),
        'not-a-dict',
        {'displayName': 'no-name'},
    ]}
    models = gp._parse_models(data)
    assert [m['id'] for m in models] == [
        'gemini-2.5-flash',
        'gemini-3-pro-preview',
        'gemini-2.0-flash-thinking-exp-1219',
        'gemma-2-2b-it',
    ]
    m0 = models[0]
    assert m0['context'] == 1048576
    assert m0['max_out'] == 65536
    assert m0['owned_by'] == 'google'
    assert m0['vision'] is True
    assert m0['thinking'] is True
    # gemini-3 -> thinking; no desc needed
    assert models[1]['thinking'] is True
    # -thinking id regex fallback
    assert models[2]['thinking'] is True
    # text-only gemma: no vision, no thinking, small limits
    m3 = models[3]
    assert m3['vision'] is False
    assert m3['thinking'] is False
    assert m3['context'] == 8192
    assert m3['max_out'] == 4096
    # OpenAI-shape fallback (bare ids, models/ prefix stripped)
    compat = gp._parse_models({'data': [
        {'id': 'models/gemini-2.5-flash', 'owned_by': 'google'},
        {'id': 'models/text-embedding-004'},
        {'object': 'model'},
    ]})
    assert [m['id'] for m in compat] == ['gemini-2.5-flash']
    # compat shape has no limits -> module fallbacks
    assert compat[0]['context'] == gp.GAS_CONTEXT_FALLBACK
    assert compat[0]['max_out'] == gp.GAS_MAX_OUTPUT_FALLBACK
    # non-list shapes give no models
    assert gp._parse_models({'nope': 1}) == []
    # a per-model output limit is clamped to the context window
    big = _entry_native('models/gemini-2.5-flash', context=8192,
                        max_out=999999)
    assert gp._parse_models({'models': [big]})[0]['max_out'] == 8192
    print('PASS test_catalog_parse')


def test_dormant_without_key():
    old = _set_env('')
    os.environ.pop('GEMINI_API_KEY', None)
    import dsk.providers.jar as jar_mod
    orig = jar_mod.load_jar
    jar_mod.load_jar = lambda name: {}
    gp.load_jar = jar_mod.load_jar  # rebind the imported name
    try:
        p = gp.GoogleAiStudioProvider()
        assert p.available() is False
        try:
            p.list_models()
            raise AssertionError('expected ProviderAuthError')
        except ProviderAuthError as e:
            assert 'aistudio.google.com' in str(e)
    finally:
        jar_mod.load_jar = orig
        gp.load_jar = orig
        _unset(old)
    print('PASS test_dormant_without_key')


def test_env_key_enables():
    old = _set_env('AIzaSyTESTKEY0123456789abcdef012345678')
    try:
        p = gp.GoogleAiStudioProvider()
        assert p.available() is True
        # native catalog uses the x-goog-api-key header
        hdr = {'x-goog-api-key': _set_env('AIzaSyTESTKEY0123456789abcdef012345678')}
        assert hdr['x-goog-api-key'].startswith('AIzaSy')
        # GEMINI_API_KEY is accepted as the official SDK alias
        os.environ.pop('GOOGLE_AI_STUDIO_API_KEY', None)
        os.environ['GEMINI_API_KEY'] = 'AIzaSyALIAS0123456789abcdef0123456'
        assert gp._api_key() == 'AIzaSyALIAS0123456789abcdef0123456'
    finally:
        os.environ.pop('GEMINI_API_KEY', None)
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
    _entry_native('models/gemini-2.5-flash'),
    _entry_native('models/gemma-2-2b-it', context=8192, max_out=4096),
], 'nextPageToken': 'tok42'}
PAGE2 = {'models': [
    _entry_native('models/gemini-3-pro-preview'),
]}


def test_stream_payload_and_parsing():
    old = _set_env('AIzaSyTESTKEY0123456789abcdef012345678')
    orig_get = gp.http_get
    orig_post = gp.http_post_stream
    captured = {}
    calls = []

    def fake_get(url, headers=None, **kw):
        calls.append(url)
        captured['get_url'] = url
        captured['get_key'] = (headers or {}).get('x-goog-api-key')
        captured['get_bearer'] = (headers or {}).get('Authorization')
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
        p = gp.GoogleAiStudioProvider()
        gp.http_get = fake_get
        gp.http_post_stream = fake_post
        chunks = list(p.stream('hi', model='gemini-2.5-flash',
                               max_tokens=512, temperature=0.7))
        # paginated native catalog: two GETs, second follows the token
        assert len(calls) == 2, calls
        assert 'pageToken=tok42' in calls[1]
        assert captured['get_key'] == 'AIzaSyTESTKEY0123456789abcdef012345678'
        assert captured['get_bearer'] is None
        assert captured['post_url'] == gp.OPENAI_CHAT_URL
        assert captured['post_auth'].startswith('Bearer AIzaSy')
        body = captured['body']
        assert body['model'] == 'gemini-2.5-flash'
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
        gp.http_get = orig_get
        gp.http_post_stream = orig_post
        _unset(old)
    print('PASS test_stream_payload_and_parsing')


def test_stream_error_frames_raise():
    old = _set_env('AIzaSyTESTKEY0123456789abcdef012345678')
    orig_get = gp.http_get
    orig_post = gp.http_post_stream
    try:
        p = gp.GoogleAiStudioProvider()
        gp.http_get = lambda url, headers=None, **kw: _fake_response(
            payload=PAGE1)

        def fake_post(url, headers=None, json_body=None, **kw):
            return _fake_response(lines=[
                b'data: ' + json.dumps(
                    {'error': {'code': 429,
                               'message': 'Resource has been exhausted',
                               'status': 'RESOURCE_EXHAUSTED'}}).encode(),
            ])
        gp.http_post_stream = fake_post
        try:
            list(p.stream('hi', model='gemini-2.5-flash'))
            raise AssertionError('expected ProviderError')
        except ProviderError as e:
            assert 'exhausted' in str(e)

        # the OpenAI shim wraps some error frames in a JSON array
        def fake_post_list(url, headers=None, json_body=None, **kw):
            return _fake_response(lines=[
                b'data: ' + json.dumps([
                    {'error': {'code': 400,
                               'message': 'Missing or invalid '
                                          'Authorization header.',
                               'status': 'INVALID_ARGUMENT'}}]).encode(),
            ])
        gp.http_post_stream = fake_post_list
        try:
            list(p.stream('hi', model='gemini-2.5-flash'))
            raise AssertionError('expected ProviderError')
        except ProviderError as e:
            assert 'Authorization header' in str(e)
    finally:
        gp.http_get = orig_get
        gp.http_post_stream = orig_post
        _unset(old)
    print('PASS test_stream_error_frames_raise')


def test_stream_empty_raises_unavailable():
    old = _set_env('AIzaSyTESTKEY0123456789abcdef012345678')
    orig_get = gp.http_get
    orig_post = gp.http_post_stream
    try:
        p = gp.GoogleAiStudioProvider()
        gp.http_get = lambda url, headers=None, **kw: _fake_response(
            payload=PAGE1)
        gp.http_post_stream = lambda url, headers=None, json_body=None, **kw: \
            _fake_response(lines=[b'', b'data: [DONE]'])
        try:
            list(p.stream('hi', model='gemini-2.5-flash'))
            raise AssertionError('expected ProviderUnavailableError')
        except ProviderUnavailableError:
            pass
    finally:
        gp.http_get = orig_get
        gp.http_post_stream = orig_post
        _unset(old)
    print('PASS test_stream_empty_raises_unavailable')


def test_http_error_classification():
    # bad key arrives as 400 with a key-specific message (not 401!)
    e = gp._classify(400, '{"error":{"code":400,"message":"API key not '
                          'valid. Please pass a valid API key.",'
                          '"status":"INVALID_ARGUMENT"}}')
    assert isinstance(e, ProviderAuthError) and 'rejected' in str(e)
    e = gp._classify(400, '{"error":{"code":400,"message":"Missing or '
                          'invalid Authorization header.",'
                          '"status":"INVALID_ARGUMENT"}}')
    assert isinstance(e, ProviderAuthError)
    e = gp._classify(403, '{"error":{"code":403,"message":"Method doesn\'t '
                          'allow unregistered callers.","status":'
                          '"PERMISSION_DENIED"}}')
    assert isinstance(e, ProviderAuthError)
    e = gp._classify(429, 'slow down', {'Retry-After': '3'})
    assert type(e).__name__ == 'ProviderRateLimitError'
    # a non-auth 400 stays a plain error
    e = gp._classify(400, '{"error":{"message":"Invalid JSON payload"}}')
    assert not isinstance(e, ProviderAuthError)
    e = gp._classify(503, 'overloaded')
    assert type(e).__name__ == 'ProviderUnavailableError'
    print('PASS test_http_error_classification')


def test_unknown_model_and_images():
    old = _set_env('AIzaSyTESTKEY0123456789abcdef012345678')
    orig_get = gp.http_get
    try:
        p = gp.GoogleAiStudioProvider()
        gp.http_get = lambda url, headers=None, **kw: _fake_response(
            payload=PAGE1)
        try:
            p.stream('hi', model='nope/nope')
            raise AssertionError('expected ProviderError')
        except ProviderError as e:
            assert 'not in catalog' in str(e)
        try:
            p.stream('hi', model='gemma-2-2b-it',
                     images=[{'mime': 'image/png', 'data': b'xx'}])
            raise AssertionError('expected ProviderError')
        except ProviderError as e:
            assert 'not vision-capable' in str(e)
        try:
            p.stream('hi', model='gemini-2.5-flash', image_generation=True)
            raise AssertionError('expected ProviderError')
        except ProviderError as e:
            assert 'image generation' in str(e)
    finally:
        gp.http_get = orig_get
        _unset(old)
    print('PASS test_unknown_model_and_images')


def test_vision_payload():
    old = _set_env('AIzaSyTESTKEY0123456789abcdef012345678')
    orig_get = gp.http_get
    orig_post = gp.http_post_stream
    captured = {}
    try:
        p = gp.GoogleAiStudioProvider()
        gp.http_get = lambda url, headers=None, **kw: _fake_response(
            payload=PAGE1)

        def fake_post(url, headers=None, json_body=None, **kw):
            captured['body'] = json_body
            return _fake_response(lines=[
                b'data: ' + json.dumps(
                    {'choices': [{'delta': {'content': 'pic'},
                                  'finish_reason': 'stop'}]}).encode(),
            ])
        gp.http_post_stream = fake_post
        list(p.stream('what is this',
                      model='gemini-2.5-flash',
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


def test_refresh_liveness():
    from dsk import refresher as rf
    old = _set_env('AIzaSyTESTKEY0123456789abcdef012345678')
    os.environ.pop('GEMINI_API_KEY', None)
    orig_get = rf._http_get
    try:
        # 200 with a model page -> valid, count reported
        rf._http_get = lambda *a, **kw: _fake_response(
            200, {'models': [{'name': 'models/gemini-2.5-flash'}] * 5})
        ok, msg = rf.refresh_google_ai_studio()
        assert ok is True and '5 models visible' in msg, msg
        # bad key -> 400 "API key not valid" -> rejected
        rf._http_get = lambda *a, **kw: _fake_response(400, {
            'error': {'code': 400,
                      'message': 'API key not valid. Please pass a valid '
                                 'API key.',
                      'status': 'INVALID_ARGUMENT'}})
        ok, msg = rf.refresh_google_ai_studio()
        assert ok is False and 'rejected' in msg, msg
        # no identity -> 403 PERMISSION_DENIED -> rejected
        rf._http_get = lambda *a, **kw: _fake_response(403, {
            'error': {'code': 403,
                      'message': "Method doesn't allow unregistered "
                                 "callers.",
                      'status': 'PERMISSION_DENIED'}})
        ok, msg = rf.refresh_google_ai_studio()
        assert ok is False and 'rejected' in msg, msg
        # 429 still proves the key is accepted (auth runs first)
        rf._http_get = lambda *a, **kw: _fake_response(429, {})
        ok, msg = rf.refresh_google_ai_studio()
        assert ok is True and '429' in msg, msg
    finally:
        rf._http_get = orig_get
        _unset(old)
    print('PASS test_refresh_liveness')


def test_wiring():
    from dsk.providers import router as router_mod
    assert ('google_ai_studio', '.google_ai_studio_provider',
            'GoogleAiStudioProvider') in router_mod.PROVIDER_MODULES
    assert router_mod.PUBLIC_PREFIX['google_ai_studio'] == \
        'google-ai-studio'
    assert router_mod.OWNED_BY['google_ai_studio'] == 'google'
    assert 'google_ai_studio' in router_mod.AUTO_CATEGORIES['general']
    from dsk import selfheal
    assert selfheal.HEALABLE['google_ai_studio'].name == \
        'google_ai_studio_provider.py'
    assert selfheal._PROVIDER_CLASSES['google_ai_studio'] == \
        'GoogleAiStudioProvider'
    assert selfheal._MODULE_NAMES['google_ai_studio'] == \
        'dsk.providers.google_ai_studio_provider'
    import re
    pats = [re.compile(x) for x in
            selfheal._EVIDENCE_PATTERNS['google_ai_studio']]
    assert pats[0].search('generativelanguage.googleapis.com/v1beta/models')
    assert pats[1].search('API key not valid. Please pass a valid API key.')
    assert pats[2].search('PERMISSION_DENIED')
    assert pats[3].search('aistudio.google.com/apikey')
    from dsk import refresher
    assert refresher.REFRESH['google_ai_studio'] is not None
    assert refresher.SIGNUP['google_ai_studio'] is not None
    assert refresher._jar_path('google_ai_studio').name == \
        'google_ai_studio_cookies.json'
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
