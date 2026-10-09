"""Contract tests for the OpenAI-compatible shim (dsk/openai_server.py).

The shim is where the provider-agnostic chunk protocol becomes the wire
format clients read. Its one rule is the one that broke in production:

    type == 'thinking'  ->  delta.reasoning_content   (the thinking panel)
    type == 'text'      ->  delta.content             (the message body)

Emitting a chain-of-thought summary as ``text`` puts it in the body, and a
client (OpenWebUI through litellm) then shows the reasoning as the whole
reply with the answer missing. These tests lock the mapping, the ordering,
the terminal chunks, and the failure contract (an upstream failure must
arrive as an SSE error event, and must NOT be followed by a normal stop —
that is what lets a truncated stream pass as a successful completion).

Run:  python tests/test_openai_shim_contract.py     (or: pytest tests/)
"""
import asyncio
import json
import os
import sys

ROOT = '/opt/docker/compose/inference4free'
if not os.path.isdir(os.path.join(ROOT, 'dsk')):
    ROOT = '/app'          # inside the container the code lives at /app
sys.path.insert(0, ROOT)

from dsk.openai_server import _stream_completion                        # noqa: E402
from dsk.providers.base import (                                        # noqa: E402
    ProviderAuthError,
    ProviderError,
    ProviderRateLimitError,
)


def _events(chunks, **kw):
    """Drive the shim over a chunk generator; return the parsed SSE events.

    ``chunks`` is passed through untouched: a generator that raises must
    raise INSIDE the shim's worker (that is the failure path under test),
    so materializing it here would test the test, not the shim.
    """
    async def run():
        raw = []
        gen = _stream_completion(iter(chunks), 1760000000,
                                 'test/model', **kw)
        async for item in gen:
            raw.append(item)
        return raw

    out = []
    for item in asyncio.run(run()):
        payload = item.replace('data:', '', 1).strip()
        if not payload or payload == '[DONE]':
            out.append({'done': True})
            continue
        out.append(json.loads(payload))
    return out


def _deltas(events):
    return [c.get('delta') or {}
            for e in events for c in (e.get('choices') or [])]


def _text_of(events):
    return ''.join(d.get('content') or '' for d in _deltas(events))


def _reasoning_of(events):
    return ''.join(d.get('reasoning_content') or '' for d in _deltas(events))


# ------------------------------------------------------------------ the mapping
def test_thinking_goes_to_reasoning_content_not_the_body():
    events = _events([{'content': 'The user asks who I am.',
                       'type': 'thinking', 'finish_reason': None}])
    assert _reasoning_of(events) == 'The user asks who I am.'
    assert _text_of(events) == ''
    # the summary must never ride along in a content delta
    assert all('content' not in d for d in _deltas(events)
               if d.get('reasoning_content'))


def test_text_goes_to_the_body_and_never_to_reasoning():
    events = _events([{'content': 'Ciao! Sono Vibe.',
                       'type': 'text', 'finish_reason': None}])
    assert _text_of(events) == 'Ciao! Sono Vibe.'
    assert _reasoning_of(events) == ''


def test_thinking_and_text_keep_their_own_channels():
    events = _events([
        {'content': 'The user asks who I am.', 'type': 'thinking',
         'finish_reason': None},
        {'content': 'Ciao! Sono Vibe.', 'type': 'text', 'finish_reason': None},
    ])
    assert _reasoning_of(events) == 'The user asks who I am.'
    assert _text_of(events) == 'Ciao! Sono Vibe.'


def test_image_chunks_are_body_content():
    events = _events([{'content': '![gen](https://x/y.png)', 'type': 'image',
                       'finish_reason': None}])
    assert _text_of(events) == '![gen](https://x/y.png)'
    assert _reasoning_of(events) == ''


def test_stream_ends_with_a_stop_chunk_and_done():
    events = _events([{'content': 'Ciao.', 'type': 'text',
                       'finish_reason': None}])
    finishes = [c.get('finish_reason')
                for e in events for c in (e.get('choices') or [])]
    assert finishes[-1] == 'stop'
    assert events[-1].get('done') is True


def test_empty_stream_still_terminates_cleanly():
    events = _events([])
    finishes = [c.get('finish_reason')
                for e in events for c in (e.get('choices') or [])]
    assert finishes[-1] == 'stop'


# ------------------------------------------------------------------- failures
def test_rate_limit_arrives_as_an_error_event_without_a_stop():
    def gen():
        yield {'content': 'Parte di ', 'type': 'text', 'finish_reason': None}
        raise ProviderRateLimitError('mistral anonymous quota reached')

    events = _events(gen())
    errs = [e['error'] for e in events if 'error' in e]
    assert errs and errs[0]['type'] == 'rate_limit_error'
    # prose received before the failure is handed back, nothing is invented
    assert _text_of(events) == 'Parte di '
    # and the stream is NOT closed as a successful completion
    assert all(c.get('finish_reason') is None
               for e in events for c in (e.get('choices') or []))


def test_rate_limit_carries_the_retry_hint():
    # SSE has no headers, so the quota window travels in the error object:
    # clients must be able to back off instead of burning the whole day.
    def gen():
        # a yield is what makes this a GENERATOR: without it the raise fires
        # at call time, outside the shim, and the test proves nothing.
        yield {'content': '', 'type': 'text', 'finish_reason': None}
        raise ProviderRateLimitError('Message rate limit reached', retry_after=1830.4)

    events = _events(gen())
    errs = [e['error'] for e in events if 'error' in e]
    assert errs[0].get('retry_after') == 1831


def test_rate_limit_without_a_hint_sends_no_retry_after():
    def gen():
        yield {'content': '', 'type': 'text', 'finish_reason': None}
        raise ProviderRateLimitError('rate limited')

    events = _events(gen())
    errs = [e['error'] for e in events if 'error' in e]
    assert 'retry_after' not in errs[0]


def test_auth_failure_is_reported_as_invalid_token():
    def gen():
        yield {'content': 'x', 'type': 'thinking', 'finish_reason': None}
        raise ProviderAuthError('session token expired')

    events = _events(gen())
    errs = [e['error'] for e in events if 'error' in e]
    assert errs and errs[0]['code'] == 'invalid_token'
    assert _reasoning_of(events) == 'x'


def test_unclassified_failure_is_reported_as_api_error():
    def gen():
        raise PageGone('The connection to the page has been disconnected.')
        yield  # pragma: no cover — makes this a generator, not a call

    events = _events(gen())
    errs = [e['error'] for e in events if 'error' in e]
    assert errs and errs[0]['type'] == 'api_error'
    assert 'PageGone' in errs[0]['message']


class PageGone(RuntimeError):
    """Stand-in for a provider crash (browser tab died)."""


# ------------------------------------------------------------------ tool calls
def test_tool_call_markers_become_tool_calls_not_prose():
    text = ('Controllo il meteo. TOOL_CALL: {"name": "get_weather", '
            '"arguments": {"city": "Roma"}}')
    events = _events([{'content': text, 'type': 'text',
                       'finish_reason': None}], use_tools=True)
    calls = [c for d in _deltas(events) for c in (d.get('tool_calls') or [])]
    assert calls, events
    assert calls[0]['function']['name'] == 'get_weather'
    assert 'Roma' in calls[0]['function']['arguments']
    assert _text_of(events).startswith('Controllo il meteo.')
    finishes = [c.get('finish_reason')
                for e in events for c in (e.get('choices') or [])]
    assert finishes[-1] == 'tool_calls'


def test_usage_chunk_is_only_emitted_when_asked():
    chunks = [{'content': 'Ciao.', 'type': 'text', 'finish_reason': None}]
    assert not [e for e in _events(chunks) if 'usage' in e]
    with_usage = [e for e in _events(chunks, include_usage=True,
                                      prompt_len=8) if 'usage' in e]
    assert with_usage and 'completion_tokens' in with_usage[0]['usage']


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
