"""Offline tests for the anonymous Pollinations.ai provider.

Covers the pieces that broke or could regress during development:
  - live catalog parsing: text models (tier filter, reasoning → thinking,
    input/output modalities → vision/image_gen flags) + image models,
    including the '-image' suffix when an id collides with a text model
  - fallback catalogs when both discovery endpoints fail
  - 'img:' upstream prefix: image routes are dispatched by prefix, so a
    text model sharing the name never hits the image renderer (and the
    prefix is stripped before the upstream render call)
  - vision pre-check: attachments on a non-vision model fail fast
  - empty 200 stream surfaces as ProviderUnavailableError (router fallback)
  - OpenAI-shim SSE parsing: delta.content → text, delta.reasoning (their
    non-standard thinking field) AND reasoning_content → thinking,
    [DONE]/junk tolerated, terminal stop chunk
  - vision payload: byte attachments wrapped as image_url data URIs
  - image generation: canonical seeded URL, markdown image chunk, stop
    chunk, prompt URL-encoding, magic-byte sniffing of the rendered body
  - error classification passthrough (429 → rate limit)
  - wiring: router/selfheal/refresher registries, anonymous availability

No network: http_get/http_post_stream are monkeypatched.

Run:  python tests/test_pollinations_provider.py     (or: pytest tests/)
"""
import sys
import types

import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from dsk.providers import pollinations_provider as pp     # noqa: E402
from dsk.providers.base import (                          # noqa: E402
    ProviderError,
    ProviderRateLimitError,
    ProviderUnavailableError,
)
from dsk.refresher import REFRESH, SIGNUP, _has_creds, _jar_path  # noqa: E402
from dsk.selfheal import (                                # noqa: E402
    _EVIDENCE_PATTERNS,
    _MODULE_NAMES,
    _PROVIDER_CLASSES,
    _PROVIDER_MODULES,
    HEALABLE,
)


class _FakeResponse:
    def __init__(self, status_code=200, json_data=None, lines=None,
                 content=b'\xff\xd8\xe0\x00', text='', headers=None):
        self.status_code = status_code
        self._json = json_data
        self._lines = lines or []
        self.content = content
        self.text = text
        self.headers = headers or {}

    def json(self):
        if self._json is None:
            raise ValueError('no json')
        return self._json

    def iter_lines(self):
        for line in self._lines:
            yield line


def _fresh():
    return pp.PollinationsProvider()


def _seed_catalog(p, text=True, images=True, vision=False):
    p._text_models = ([{
        'id': 'openai-fast', 'name': 'GPT-OSS 20B (OVH)',
        'thinking_enabled': True, 'search_enabled': False,
        'vision': vision, 'image_gen': False,
        'context_length': 131072, 'max_output_tokens': 32768,
    }] if text else [])
    p._image_models = (['sana'] if images else [])
    import time
    p._catalog_ts = time.monotonic()


# ------------------------------------------------------------------ catalog

def test_catalog_parsing(monkeypatch=None):
    p = _fresh()
    fake = _FakeResponse(json_data=[{
        'name': 'openai-fast', 'description': 'GPT-OSS 20B Reasoning LLM',
        'reasoning': True, 'tier': 'anonymous', 'community': False,
        'input_modalities': ['text'], 'output_modalities': ['text'],
        'tools': True, 'aliases': ['openai'], 'vision': False, 'audio': False,
    }, {
        'name': 'vision-pro', 'description': 'seedy vision model',
        'reasoning': False, 'tier': 'seed',                 # NOT anonymous
        'input_modalities': ['text', 'image'], 'output_modalities': ['text'],
        'vision': True,
    }, {
        'name': 'multimodal', 'description': 'anonymous image-out',
        'tier': 'anonymous',
        'input_modalities': ['text', 'image'],              # vision in
        'output_modalities': ['text', 'image'],             # image gen out
    }, 'junk-not-a-dict', {'noname': True}])
    fake_img = _FakeResponse(json_data=['sana', '  ', {'name': 'flux'},
                                        {'noname': True}])
    orig_get = pp.http_get

    def fake_get(url, **kwargs):
        return fake_img if 'image.pollinations' in url else fake

    pp.http_get = fake_get
    try:
        entries = p.list_models()
    finally:
        pp.http_get = orig_get
    by_id = {e['id']: e for e in entries}
    # anonymous tier only; seed-tier excluded
    assert set(by_id) == {'openai-fast', 'multimodal', 'sana', 'flux'}
    assert by_id['openai-fast']['thinking_enabled'] is True
    assert by_id['openai-fast']['vision'] is False
    assert by_id['openai-fast']['image_gen'] is False
    assert by_id['multimodal']['vision'] is True            # input modality
    assert by_id['multimodal']['image_gen'] is True         # output modality
    # image models: dedicated routes with image_gen=True
    assert by_id['sana']['image_gen'] is True
    assert by_id['sana']['vision'] is False
    assert by_id['sana']['extra'].get('image_model') is True
    assert by_id['sana']['upstream_model'] == 'img:sana'
    assert by_id['openai-fast']['upstream_model'] == 'openai-fast'
    # collision suffix (no text/image name clash in this catalog, both kept)
    assert by_id['flux']['image_gen'] is True
    assert by_id['flux']['upstream_model'] == 'img:flux'


def test_catalog_id_collision_gets_suffix():
    p = _fresh()
    p._text_models = [{'id': 'sana', 'name': 'x', 'thinking_enabled': False,
                       'search_enabled': False, 'vision': False,
                       'image_gen': False, 'context_length': 131072,
                       'max_output_tokens': 32768}]
    p._image_models = ['sana']
    import time
    p._catalog_ts = time.monotonic()
    entries = {e['id']: e for e in p.list_models()}
    assert 'sana' in entries and 'sana-image' in entries
    assert entries['sana']['image_gen'] is False
    assert entries['sana-image']['image_gen'] is True
    # the two routes can never be confused: bare upstream vs img: prefix
    assert entries['sana']['upstream_model'] == 'sana'
    assert entries['sana-image']['upstream_model'] == 'img:sana'


def test_catalog_fallback_when_discovery_fails():
    p = _fresh()
    orig_get = pp.http_get
    pp.http_get = lambda url, **kw: (_ for _ in ()).throw(OSError('down'))
    try:
        entries = p.list_models()
    finally:
        pp.http_get = orig_get
    ids = {e['id'] for e in entries}
    assert 'openai-fast' in ids                 # static text fallback
    assert 'flux' in ids                        # static image fallback
    flux = next(e for e in entries if e['id'] == 'flux')
    assert flux['image_gen'] is True


# -------------------------------------------------------------------- text

def test_stream_text_sse_parsing():
    p = _fresh()
    _seed_catalog(p)
    lines = [
        b'data: {"choices":[{"delta":{"role":"assistant","content":""}}]}',
        b'data: {"choices":[{"delta":{"reasoning":"User asks"}}]}',
        b'data: {"choices":[{"delta":{"reasoning_content":" alt field"}}]}',
        b'data: {"choices":[{"delta":{"content":"Hello"}}]}',
        b'data: {"choices":[{"delta":{"content":" world"},"finish_reason":null}]}',
        b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}',
        b'data: [DONE]',
        b'event: ping',
        b'data: {broken json',
    ]
    orig = pp.http_post_stream
    captured = {}

    def fake_post(url, **kwargs):
        captured.update(kwargs)
        assert url == pp.TEXT_CHAT_URL
        return _FakeResponse(lines=lines)

    pp.http_post_stream = fake_post
    try:
        chunks = list(p.stream('hi', model='openai-fast'))
    finally:
        pp.http_post_stream = orig
    kinds = [(c['type'], c['content']) for c in chunks if c['content']]
    assert kinds == [('thinking', 'User asks'),
                     ('thinking', ' alt field'),
                     ('text', 'Hello'),
                     ('text', ' world')]
    assert chunks[-1] == {'content': '', 'type': 'text', 'finish_reason': 'stop'}
    body = captured['json_body']
    assert body['model'] == 'openai-fast' and body['stream'] is True
    # text-only: plain string content — the parts-array form hangs the
    # upstream's text-only backends (measured live)
    assert body['messages'][0]['content'] == 'hi'


def test_stream_text_params_forwarded():
    p = _fresh()
    _seed_catalog(p)
    captured = {}

    def fake_post(url, **kwargs):
        captured.update(kwargs)
        return _FakeResponse(lines=[b'data: {"choices":[{"delta":{"content":"x"}}]}'])

    orig = pp.http_post_stream
    pp.http_post_stream = fake_post
    try:
        list(p.stream('hi', model='openai-fast', temperature=0.3,
                      max_tokens=512))
    finally:
        pp.http_post_stream = orig
    assert captured['json_body']['temperature'] == 0.3
    assert captured['json_body']['max_tokens'] == 512


def test_stream_text_http_error():
    p = _fresh()
    _seed_catalog(p)
    orig = pp.http_post_stream
    pp.http_post_stream = lambda url, **kw: _FakeResponse(
        status_code=429, text='rate limited', headers={'retry-after': '7'})
    try:
        try:
            list(p.stream('hi', model='openai-fast'))
            raise AssertionError('expected ProviderRateLimitError')
        except ProviderRateLimitError as e:
            assert getattr(e, 'retry_after', None) == 7.0
    finally:
        pp.http_post_stream = orig


# ------------------------------------------------------------------ vision

def test_vision_images_become_data_uris():
    p = _fresh()
    _seed_catalog(p, vision=True)   # attachments need a vision-capable model
    captured = {}

    def fake_post(url, **kwargs):
        captured.update(kwargs)
        return _FakeResponse(lines=[b'data: {"choices":[{"delta":{"content":"ok"}}]}'])

    orig = pp.http_post_stream
    pp.http_post_stream = fake_post
    try:
        list(p.stream('describe', model='openai-fast', images=[
            {'mime': 'image/png', 'data': b'\x89PNG-fake-bytes'}]))
    finally:
        pp.http_post_stream = orig
    content = captured['json_body']['messages'][0]['content']
    # attachments force the multimodal parts shape (string otherwise)
    assert content[0] == {'type': 'text', 'text': 'describe'}
    img_part = content[1]
    assert img_part['type'] == 'image_url'
    uri = img_part['image_url']['url']
    assert uri.startswith('data:image/png;base64,')
    import base64
    assert base64.b64decode(uri.split(',', 1)[1]) == b'\x89PNG-fake-bytes'


# ------------------------------------------------------------------ images

def test_image_generation_stream():
    p = _fresh()
    _seed_catalog(p)
    captured = {}

    def fake_get(url, **kwargs):
        captured['url'] = url
        return _FakeResponse(content=b'\xff\xd8\xff\xe0' + b'x' * 64)

    orig = pp.http_get
    pp.http_get = fake_get
    try:
        chunks = list(p.stream('a red cube', model='sana',
                               image_generation=True))
    finally:
        pp.http_get = orig
    img = [c for c in chunks if c.get('type') == 'image']
    assert len(img) == 1
    url = img[0]['url']
    assert url.startswith('https://image.pollinations.ai/prompt/')
    assert 'model=sana' in url and 'nologo=true' in url and 'seed=' in url
    assert 'a%20red%20cube' in url
    assert img[0]['content'] == f'![image]({url})'
    assert chunks[-1]['finish_reason'] == 'stop'
    # magic bytes were sniffed on the downloaded render
    assert captured['url'] == url


def test_image_model_selected_without_flag():
    p = _fresh()
    _seed_catalog(p)
    orig_get, orig_post = pp.http_get, pp.http_post_stream
    pp.http_get = lambda url, **kw: _FakeResponse(content=b'\x89PNG\r\n')
    called = {'post': False}

    def fake_post(url, **kw):
        called['post'] = True
        return _FakeResponse(lines=[])

    pp.http_post_stream = fake_post
    try:
        # router-style upstream_model ('img:sana'), no image_generation flag
        chunks = list(p.stream('a cat', model='img:sana'))
    finally:
        pp.http_get, pp.http_post_stream = orig_get, orig_post
    # the image model routes to the image path, never to the text shim
    assert called['post'] is False
    assert any(c.get('type') == 'image' for c in chunks)


def test_img_prefix_stripped_in_image_url():
    p = _fresh()
    _seed_catalog(p)
    captured = {}
    orig = pp.http_get
    pp.http_get = lambda url, **kw: (captured.update(url=url),
                                     _FakeResponse(content=b'\xff\xd8'))[1]
    try:
        list(p.stream('x', model='img:sana'))   # router-style upstream_model
    finally:
        pp.http_get = orig
    # the img: prefix is stripped for the upstream render call
    assert 'model=sana' in captured['url']
    assert 'img%3Asana' not in captured['url']
    assert 'img:sana' not in captured['url']


def test_collision_name_text_stays_text():
    # regression: a text model whose id also exists in the image catalog must
    # keep going through the text shim — only the 'img:'-prefixed route is
    # dispatched to the image renderer
    p = _fresh()
    p._text_models = [{'id': 'sana', 'name': 'x', 'thinking_enabled': False,
                       'search_enabled': False, 'vision': False,
                       'image_gen': False, 'context_length': 131072,
                       'max_output_tokens': 32768}]
    p._image_models = ['sana']
    import time
    p._catalog_ts = time.monotonic()
    orig_get, orig_post = pp.http_get, pp.http_post_stream
    pp.http_get = lambda url, **kw: (_ for _ in ()).throw(
        AssertionError('image path must not be hit'))

    def fake_post(url, **kw):
        return _FakeResponse(
            lines=[b'data: {"choices":[{"delta":{"content":"t"}}]}'])

    pp.http_post_stream = fake_post
    try:
        chunks = list(p.stream('hi', model='sana'))
    finally:
        pp.http_get, pp.http_post_stream = orig_get, orig_post
    assert any(c.get('type') == 'text' and c['content'] == 't' for c in chunks)


def test_vision_precheck_rejects_non_vision_model():
    # seeded catalog: openai-fast has vision=False → attachments are rejected
    # BEFORE any HTTP call (the router capability filter is the first line)
    p = _fresh()
    _seed_catalog(p)
    orig = pp.http_post_stream
    pp.http_post_stream = lambda url, **kw: (_ for _ in ()).throw(
        AssertionError('no request expected'))
    try:
        try:
            list(p.stream('what is this', model='openai-fast', images=[
                {'mime': 'image/png', 'data': b'xx'}]))
            raise AssertionError('expected ProviderError')
        except ProviderError as e:
            assert 'vision' in str(e) and 'openai-fast' in str(e)
    finally:
        pp.http_post_stream = orig


def test_empty_stream_raises_unavailable():
    # 200 with zero deltas must surface as ProviderUnavailableError (router
    # retry/fallback), never as a silent empty 200 response
    p = _fresh()
    _seed_catalog(p)
    orig = pp.http_post_stream
    pp.http_post_stream = lambda url, **kw: _FakeResponse(
        lines=[b'data: {"choices":[{"delta":{"role":"assistant"}}]}',
               b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}'])
    try:
        try:
            list(p.stream('hi', model='openai-fast'))
            raise AssertionError('expected ProviderUnavailableError')
        except ProviderUnavailableError as e:
            assert 'openai-fast' in str(e)
    finally:
        pp.http_post_stream = orig


def test_image_non_image_body_rejected():
    p = _fresh()
    _seed_catalog(p)
    orig = pp.http_get
    pp.http_get = lambda url, **kw: _FakeResponse(
        content=b'<html>oops</html>', status_code=200)
    try:
        try:
            list(p.stream('x', model='sana', image_generation=True))
            raise AssertionError('expected ProviderUnavailableError')
        except ProviderUnavailableError:
            pass
    finally:
        pp.http_get = orig


def test_unicode_prompt_encoded():
    p = _fresh()
    _seed_catalog(p)
    captured = {}
    orig = pp.http_get
    pp.http_get = lambda url, **kw: (captured.update(url=url),
                                     _FakeResponse(content=b'\xff\xd8'))[1]
    try:
        list(p.stream('un gatto & una pizza 🍕', model='sana',
                      image_generation=True))
    finally:
        pp.http_get = orig
    from urllib.parse import urlparse, unquote
    path = urlparse(captured['url']).path
    assert unquote(path).endswith('un gatto & una pizza 🍕')


def test_tier_exhaustion_402_maps_to_rate_limit():
    from dsk.providers.base import ProviderRateLimitError as RL
    # Pollinations answers an HTML 402 page when the anonymous budget is
    # gone — a quota condition: the router must cooldown, not retry every
    # sibling model against the same wall.
    err = pp._classify(402, '<html>Payment Required - Pollinations</html>')
    assert isinstance(err, RL)
    # everything else keeps the shared taxonomy
    assert isinstance(pp._classify(500, 'boom'), ProviderUnavailableError)


# ------------------------------------------------------------------ wiring

def test_wiring():
    import dsk.providers.router as router
    assert ('pollinations', '.pollinations_provider',
            'PollinationsProvider') in router.PROVIDER_MODULES
    assert router.OWNED_BY['pollinations'] == 'pollinations'
    assert router.PUBLIC_PREFIX['pollinations'] == 'pollinations'
    assert 'pollinations' in router.AUTO_CATEGORIES['image_gen']
    assert 'pollinations' in router.AUTO_CATEGORIES['vision']
    assert HEALABLE['pollinations'].name == 'pollinations_provider.py'
    assert _PROVIDER_MODULES['pollinations'] == \
        _MODULE_NAMES['pollinations'] == 'dsk.providers.pollinations_provider'
    assert _PROVIDER_CLASSES['pollinations'] == 'PollinationsProvider'
    assert any('pollinations' in pat for pat in _EVIDENCE_PATTERNS['pollinations'])
    assert _has_creds('pollinations') is True      # anonymous provider
    assert _jar_path('pollinations').name == 'pollinations_cookies.json'
    assert 'pollinations' in REFRESH and 'pollinations' in SIGNUP


def test_available_and_provider_shape():
    p = _fresh()
    assert p.name == 'pollinations'
    assert p.available() is True
    assert issubclass(type(p), pp.Provider)


if __name__ == '__main__':
    fns = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f'PASS {fn.__name__}')
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f'FAIL {fn.__name__}: {type(e).__name__}: {e}')
    sys.exit(1 if failed else 0)
