"""Offline tests for the duck.ai provider (phase 1).

Covers the pieces that broke or regressed during development:
  - payload builder: reasoningEffort is REQUIRED (omission answered 400),
    durableStream carries an RSA JWK, toolChoice all-false metadata
  - SSE line parsing: content/message deltas, [DONE], junk tolerated
  - fully-dynamic model resolution: the default follows the LIVE catalog
    (never a hardcoded id), aliases remap retired ids, unknown ids route
    to the live default
  - error classification: 418 ERR_BN_LIMIT (IP ban, no retry) vs
    ERR_CHALLENGE (fresh VQD retry) vs 429 (Retry-After) vs 400 (no retry)
  - reasoningEffort selection incl. the forced-low-effort models
  - x-fe-version scrape regex
  - refresher/selfheal wiring (jar path, anonymous creds, registries)

No network: the catalog fetch and the node solver are monkeypatched.

Run:  python tests/test_duck_provider.py     (or: pytest tests/)
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from dsk.providers import duck_provider as dp            # noqa: E402
from dsk.providers.base import (                         # noqa: E402
    ProviderError,
    ProviderRateLimitError,
    ProviderUnavailableError,
)
from dsk.refresher import REFRESH, SIGNUP, _has_creds, _jar_path  # noqa: E402
from dsk.selfheal import (                               # noqa: E402
    _EVIDENCE_PATTERNS,
    _MODULE_NAMES,
    _PROVIDER_CLASSES,
    _PROVIDER_MODULES,
    HEALABLE,
)


def _provider() -> "dp.DuckProvider":
    p = dp.DuckProvider()
    # A stable JWK so _build_payload never shells out to node.
    p._durable_public_key = {
        'kty': 'RSA', 'alg': 'RSA-OAEP-256', 'use': 'enc',
        'n': 'x', 'e': 'AQAB', 'key_ops': ['encrypt'],
    }
    return p


def test_payload_builder():
    p = _provider()
    payload = p._build_payload('hi', 'gpt-5.4-mini', 'none')
    assert payload['model'] == 'gpt-5.4-mini'
    # reasoningEffort is REQUIRED — the live A/B proved omission answers 400.
    assert payload['reasoningEffort'] == 'none'
    assert payload['messages'] == [{'role': 'user', 'content': 'hi'}]
    assert payload['canUseTools'] is True
    assert payload['metadata']['toolChoice'] == {
        'NewsSearch': False, 'VideosSearch': False,
        'LocalSearch': False, 'WeatherForecast': False}
    durable = payload['durableStream']
    assert durable['publicKey']['kty'] == 'RSA'
    assert durable['messageId'] != durable['conversationId']


def test_parse_line():
    parse = dp.DuckProvider._parse_line
    assert parse('data: {"message": "hello"}') == {
        'content': 'hello', 'type': 'text', 'finish_reason': None}
    assert parse('data: {"content": "world"}') == {
        'content': 'world', 'type': 'text', 'finish_reason': None}
    assert parse('data: [DONE]') is None
    assert parse('[DONE]') is None
    assert parse('data: {not json') is None
    assert parse('event: ping') is None
    assert parse('') is None
    assert parse('data: {"content": ""}') is None  # empty delta ignored
    assert parse('data: {"role": "assistant"}') is None


def test_dynamic_default_follows_live_catalog():
    p = _provider()
    # Default comes from the LIVE catalog, cheapest/general first — never a
    # hardcoded id that a rename would strand.
    p._live_models = (1e12, {'claude-haiku-4-5': {'name': 'x', 'efforts': ['none']},
                             'gpt-5.4-mini': {'name': 'x', 'efforts': ['none']}})
    assert p._resolve_model(None) == 'gpt-5.4-mini'
    # Catalog without any gpt*: the default moves to what IS live.
    p._live_models = (1e12, {'mistral-small-2603': {'name': 'x', 'efforts': ['none']}})
    assert p._resolve_model(None) == 'mistral-small-2603'
    assert p._resolve_model('duck/retired-id') == 'mistral-small-2603'


def test_resolve_model_aliases_and_passthrough():
    p = _provider()
    p._live_models = (1e12, {'gpt-5.4-mini': {'name': 'x', 'efforts': ['none']},
                             'gpt-6-luna': {'name': 'x', 'efforts': ['none']},
                             'tinfoil/gpt-oss-120b': {'name': 'x', 'efforts': ['none']}})
    # Public namespace prefix stripped, retired ids remapped via aliases.
    assert p._resolve_model('duck/gpt-4o-mini') == 'gpt-5.4-mini'
    assert p._resolve_model('gpt-oss-120b') == 'tinfoil/gpt-oss-120b'
    # A live id passes through untouched.
    assert p._resolve_model('gpt-6-luna') == 'gpt-6-luna'


def test_default_last_resort_is_static():
    p = _provider()
    # Catalog unreachable: the static fallback table supplies the ids and
    # the preference order still picks the cheap general one.
    p._live_models = (0.0, {})
    assert p._default_model() == dp.DEFAULT_DUCK_MODEL == 'gpt-5.4-mini'
    assert p._resolve_model('duck/unknown') == 'gpt-5.4-mini'


def test_classify_chat_error():
    classify = dp.DuckProvider._classify_chat_error
    # 418 + ERR_BN_LIMIT: the exit IP is banned — retrying cannot help.
    err = classify(418, '{"type": "ERR_BN_LIMIT"}')
    assert isinstance(err, ProviderRateLimitError)
    # 418 + anything else: challenge rejected — a fresh VQD retry may fix it.
    err = classify(418, '{"type": "ERR_CHALLENGE"}')
    assert isinstance(err, ProviderUnavailableError)
    # 429 carries the Retry-After through.
    class _H:
        def get(self, k):
            return {'Retry-After': '12'}.get(k)
    err = classify(429, 'slow down', _H())
    assert isinstance(err, ProviderRateLimitError) and err.retry_after == 12.0
    # 400 = schema drift — a code fix, never a retry.
    assert isinstance(classify(400, 'ERR_BAD_REQUEST'), ProviderError)
    assert isinstance(classify(401, 'nope'), ProviderUnavailableError)
    assert isinstance(classify(403, 'nope'), ProviderUnavailableError)


def test_effort_selection():
    effort = dp._effort_for
    assert effort('gpt-5.4-mini', False) == 'none'
    assert effort('gpt-5.4-mini', True, ['none', 'low', 'medium']) == 'low'
    # Live A/B: these two always carry 'low' regardless of advertised tiers.
    assert effort('claude-haiku-4-5', False) == 'low'
    assert effort('tinfoil/gpt-oss-120b', False) == 'low'
    # No live efforts known: thinking falls back to 'low'.
    assert effort('gpt-5.4-mini', True) == 'low'


def test_fe_version_regex():
    assert dp._FE_VERSION_RE.search('x-fe-version: serp_20260424_180649_ET-abcdef0123456789abcd')
    assert dp._FE_VERSION_RE.search("('serp_20251001_090000_ET-0123456789abcdef0123',)")
    assert not dp._FE_VERSION_RE.search('serp_20260424')  # truncated build id


def test_wiring():
    assert ('duck', '.duck_provider', 'DuckProvider') in \
        __import__('dsk.providers.router', fromlist=['x']).PROVIDER_MODULES
    assert HEALABLE['duck'].name == 'duck_provider.py'
    assert _PROVIDER_MODULES['duck'] == _MODULE_NAMES['duck'] == \
        'dsk.providers.duck_provider'
    assert _PROVIDER_CLASSES['duck'] == 'DuckProvider'
    assert any('duckchat' in p for p in _EVIDENCE_PATTERNS['duck'])
    assert _has_creds('duck') is True          # anonymous provider
    assert _jar_path('duck').name == 'duck_cookies.json'
    assert 'duck' in REFRESH and 'duck' in SIGNUP


def test_parse_reasoning_summary_events():
    # ``role: "reasoning"`` events carry ``summaryText``: emitted as thinking
    # pieces, empty summaries skipped, plain content/message untouched.
    line = json.dumps({'role': 'reasoning', 'state': 'done',
                       'summaryText': 'thinking about 19*21'})
    piece = dp.DuckProvider._parse_line(f'data: {line}')
    assert piece == {'content': 'thinking about 19*21', 'type': 'thinking',
                     'finish_reason': None, 'cumulative': True}
    listed = dp.DuckProvider._parse_line('data: ' + json.dumps(
        {'role': 'reasoning', 'state': 'done',
         'summaryText': ['part one, ', 'part two']}))
    assert listed == {'content': 'part one, part two', 'type': 'thinking',
                      'finish_reason': None, 'cumulative': True}
    empty = dp.DuckProvider._parse_line('data: ' + json.dumps(
        {'role': 'reasoning', 'state': 'done', 'summaryText': ''}))
    assert empty is None
    empty_list = dp.DuckProvider._parse_line('data: ' + json.dumps(
        {'role': 'reasoning', 'state': 'done', 'summaryText': []}))
    assert empty_list is None
    text = dp.DuckProvider._parse_line('data: ' + json.dumps(
        {'role': 'assistant', 'message': '399'}))
    assert text == {'content': '399', 'type': 'text', 'finish_reason': None}


def test_iter_chunks_dedupes_cumulative_summary():
    # summaryText is the summary SO FAR: _iter_chunks forwards only the new
    # suffix, so a growing summary never repeats in the reasoning channel.
    class _Resp:
        def iter_content(self, chunk_size=None):
            lines = [
                'data: ' + json.dumps({'role': 'reasoning', 'summaryText': 'step one'}),
                'data: ' + json.dumps({'role': 'reasoning', 'summaryText': 'step one, step two'}),
                'data: ' + json.dumps({'role': 'assistant', 'message': 'done'}),
                '[DONE]',
            ]
            yield ('\n'.join(lines) + '\n').encode()
    provider = dp.DuckProvider.__new__(dp.DuckProvider)
    pieces = list(dp.DuckProvider._iter_chunks(provider, _Resp()))
    thinking = [p for p in pieces if p.get('type') == 'thinking']
    assert [p['content'] for p in thinking] == ['step one', ', step two']
    text = [p['content'] for p in pieces if p.get('type') == 'text']
    assert 'done' in text


def test_fallback_catalog_shape():
    # The static last-resort table must match the live-catalog value shape.
    for entry in dp._FALLBACK_MODELS:
        assert set(entry) >= {'id', 'name', 'efforts'}
        assert entry['efforts'] and 'none' in entry['efforts']


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
    print(f'{len(fns) - failed}/{len(fns)} passed')
    sys.exit(1 if failed else 0)
