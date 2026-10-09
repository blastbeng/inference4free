"""Offline tests for the chat.mistral.ai (Le Chat / Vibe "Work") provider.

The bug this file pins down: the Work harness streams the model's REASONING
SUMMARY alongside the answer, and the parser used to (a) emit everything as
``text`` — so the summary landed in ``delta.content`` and clients showed the
thinking as the whole reply — and (b) drop every patch shape it did not know
(``op: add`` of a new chunk, ``replace`` with a dict, non-``text`` chunk
types, non-``/contentChunks`` paths), which is how the real answer vanished.

Covered here:
  - append deltas on ``/contentChunks/N/text`` stay text
  - reasoning fields/paths/chunk types become ``type: 'thinking'``
  - ``op: add`` (new chunk) and ``op: replace`` with a dict emit their text
  - snapshots are diffed per chunk field: a repeated snapshot emits the tail
    only, and appends + snapshots on the same chunk never double-send
  - JSON-Pointer ``/-`` appends get a real index (so the deltas that follow
    diff against the chunk they extend)
  - tool/plugin/citation chunks never reach the text stream
  - the account upsell raises ProviderAuthError from both patch shapes
  - the full line-framed stream: order preserved, auth-wall hold window,
    terminal stop chunk, quota frame -> ProviderRateLimitError
  - _parse_line stays usable without diff state (back-compat)

No network: the response object is faked.

Run:  python tests/test_mistral_provider.py     (or: pytest tests/)
"""
import json
import os
import sys

ROOT = '/opt/docker/compose/inference4free'
if not os.path.isdir(os.path.join(ROOT, 'dsk')):
    ROOT = '/app'          # inside the container the code lives at /app
sys.path.insert(0, ROOT)

from dsk.providers import mistral_provider as mp      # noqa: E402
from dsk.providers.base import ProviderAuthError, ProviderRateLimitError  # noqa: E402


class _FakeResponse:
    def __init__(self, lines):
        self._lines = lines
        self.status_code = 200
        self.headers = {}
        self.text = ''

    def iter_content(self, chunk_size=None):
        for line in self._lines:
            yield (line + '\n').encode('utf-8')


def _frame(patches, msg_type='message'):
    return '15:' + json.dumps({'json': {'type': msg_type, 'patches': patches}})


def _provider():
    return mp.MistralProvider.__new__(mp.MistralProvider)


def _drain(lines):
    p = _provider()
    return list(p._iter_chunks(_FakeResponse(lines)))


def _text_of(chunks, kind):
    return ''.join(c['content'] for c in chunks if c['type'] == kind)


# --------------------------------------------------------------- patch shapes
def test_append_deltas_are_text():
    p = _provider()
    state = {}
    out = p._parse_line(_frame([{'op': 'append', 'path': '/contentChunks/0/text',
                                 'value': 'Ciao'}]), state)
    assert out == [{'content': 'Ciao', 'type': 'text', 'finish_reason': None}]
    assert state['/contentChunks/0/text'] == 4


def test_reasoning_field_is_thinking():
    p = _provider()
    state = {}
    out = p._parse_line(_frame([{'op': 'append',
                                 'path': '/contentChunks/0/reasoning',
                                 'value': 'the user asks who I am'}]), state)
    assert out[0]['type'] == 'thinking'
    assert out[0]['content'] == 'the user asks who I am'


def test_reasoning_chunk_type_is_thinking():
    p = _provider()
    out = p._parse_line(_frame([{'op': 'replace', 'path': '/contentChunks',
                                 'value': [{'type': 'reasoning',
                                            'text': 'plan: answer briefly'}]}]))
    assert out == [{'content': 'plan: answer briefly', 'type': 'thinking',
                    'finish_reason': None}]


def test_add_new_chunk_emits_the_answer():
    # The shape that used to be dropped: the answer arrives as a NEW chunk.
    p = _provider()
    state = {}
    out = p._parse_line(_frame([{'op': 'add', 'path': '/contentChunks/1',
                                 'value': {'type': 'text',
                                           'text': 'Sono Vibe.'}}]), state)
    assert out == [{'content': 'Sono Vibe.', 'type': 'text',
                    'finish_reason': None}]
    # ...and the deltas that follow extend the same baseline (no re-send).
    more = p._parse_line(_frame([{'op': 'append', 'path': '/contentChunks/1/text',
                                  'value': ' Come posso aiutarti?'}]), state)
    assert more == [{'content': ' Come posso aiutarti?', 'type': 'text',
                     'finish_reason': None}]
    assert _text_of(out + more, 'text') == 'Sono Vibe. Come posso aiutarti?'


def test_replace_dict_chunk_emits_the_answer():
    p = _provider()
    out = p._parse_line(_frame([{'op': 'replace', 'path': '/contentChunks/2',
                                 'value': {'type': 'text',
                                           'text': 'la risposta'}}]))
    assert out[0]['content'] == 'la risposta' and out[0]['type'] == 'text'


def test_snapshot_is_diffed_per_chunk():
    p = _provider()
    state = {}
    first = p._parse_line(_snapshot([{'type': 'text', 'text': 'Ciao'}]), state)
    repeat = p._parse_line(_snapshot([{'type': 'text', 'text': 'Ciao'}]), state)
    tail = p._parse_line(_snapshot([{'type': 'text',
                                     'text': 'Ciao! Sono Vibe.'}]), state)
    assert first[0]['content'] == 'Ciao'
    # A snapshot repeating what was already sent emits nothing.
    assert repeat == []
    assert tail[0]['content'] == '! Sono Vibe.'


def _snapshot(value):
    return _frame([{'op': 'replace', 'path': '/contentChunks', 'value': value}])


def test_append_then_snapshot_never_double_sends():
    p = _provider()
    state = {}
    p._parse_line(_frame([{'op': 'append', 'path': '/contentChunks/0/text',
                           'value': 'Ciao'}]), state)
    tail = p._parse_line(_snapshot([{'type': 'text', 'text': 'Ciao!'}]), state)
    assert tail == [{'content': '!', 'type': 'text', 'finish_reason': None}]


def test_pointer_append_gets_a_real_index():
    p = _provider()
    state, counts = {}, {}
    added = p._parse_line(_frame([{'op': 'add', 'path': '/contentChunks/-',
                                   'value': {'type': 'text', 'text': 'Ciao'}}]),
                          state, counts)
    delta = p._parse_line(_frame([{'op': 'append', 'path': '/contentChunks/0/text',
                                   'value': ' mondo'}]), state, counts)
    assert added[0]['content'] == 'Ciao'
    assert delta[0]['content'] == ' mondo'


def test_tool_and_citation_chunks_are_skipped():
    p = _provider()
    out = p._parse_line(_snapshot([
        {'type': 'tool_call', 'text': 'websearch(...)'},
        {'type': 'citation', 'text': 'https://example.com'},
        {'type': 'text', 'text': 'Ciao'},
    ]))
    assert [c['content'] for c in out] == ['Ciao']


def test_non_contentchunks_paths_are_ignored():
    p = _provider()
    out = p._parse_line(_frame([{'op': 'append', 'path': '/title',
                                 'value': 'New chat'}]))
    assert out == []


def test_bootstrap_frame_never_echoes_the_prompt():
    line = '15:' + json.dumps({'json': {'type': 'bootstrap',
                                        'messages': [{'role': 'user',
                                                      'content': 'ciao'}]}})
    assert _provider()._parse_line(line) == []


# ------------------------------------------------------------------ auth wall
def test_auth_wall_from_append_delta():
    p = _provider()
    try:
        p._parse_line(_frame([{'op': 'append', 'path': '/contentChunks/0/text',
                               'value': 'An account is now required to use Vibe'}]))
    except ProviderAuthError:
        return
    raise AssertionError('account upsell must raise ProviderAuthError')


def test_auth_wall_from_snapshot():
    p = _provider()
    try:
        p._parse_line(_snapshot([{'type': 'text',
                                  'text': '## An account is now required'}]))
    except ProviderAuthError:
        return
    raise AssertionError('account upsell must raise ProviderAuthError')


# ---------------------------------------------------------------- full stream
def test_full_stream_order_and_stop_chunk():
    lines = [
        '15:' + json.dumps({'json': {'type': 'bootstrap', 'chat': {'id': 'x'}}}),
        _frame([{'op': 'append', 'path': '/contentChunks/0/reasoning',
                 'value': 'the user asks who I am; '}]),
        _frame([{'op': 'append', 'path': '/contentChunks/0/reasoning',
                 'value': 'answer briefly.'}]),
        _frame([{'op': 'add', 'path': '/contentChunks/1',
                 'value': {'type': 'text', 'text': 'Ciao! '}}]),
        _frame([{'op': 'append', 'path': '/contentChunks/1/text',
                 'value': 'Sono Vibe.'}]),
        '8:null',
    ]
    chunks = _drain(lines)
    assert chunks[-1] == {'content': '', 'type': 'text', 'finish_reason': 'stop'}
    assert _text_of(chunks, 'thinking') == 'the user asks who I am; answer briefly.'
    assert _text_of(chunks, 'text') == 'Ciao! Sono Vibe.'
    # reasoning must never be emitted as prose
    assert all(c['type'] != 'text' for c in chunks
               if 'user asks' in c['content'])


def test_quota_frame_is_rate_limit():
    lines = ['6:' + json.dumps({'json': {'message': 'Message rate limit reached',
                                         'internalCode': 6200,
                                         'retryAfterSeconds': 7405}})]
    try:
        _drain(lines)
    except ProviderRateLimitError as e:
        assert e.retry_after == 7405.0
        return
    raise AssertionError('code 6200 must raise ProviderRateLimitError')


def test_auth_wall_is_held_not_leaked():
    # The upsell arrives split across tiny deltas: nothing may be emitted
    # before the phrase is recognised.
    lines = ['15:' + json.dumps({'json': {'type': 'message', 'patches': [
        {'op': 'append', 'path': '/contentChunks/0/text', 'value': '## '}]}},
    )]
    for frag in ('An ', 'account ', 'is now ', 'required', ' to use Vibe'):
        lines.append(_frame([{'op': 'append', 'path': '/contentChunks/0/text',
                              'value': frag}]))
    gen = _provider()._iter_chunks(_FakeResponse(lines))
    emitted = []
    try:
        for chunk in gen:
            emitted.append(chunk)
    except ProviderAuthError:
        pass
    assert emitted == [], f'held pieces leaked: {emitted}'


def test_parse_line_without_state_still_works():
    p = _provider()
    out = p._parse_line(_frame([{'op': 'append', 'path': '/contentChunks/0/text',
                                 'value': 'Ciao'}]))
    assert out[0]['content'] == 'Ciao'


def test_junk_lines_are_tolerated():
    p = _provider()
    assert p._parse_line('') == []
    assert p._parse_line('no-colon') == []
    assert p._parse_line('x:{"json":{}}') == []
    assert p._parse_line('15:not-json') == []
    assert p._parse_line('15:null') == []
    assert p._parse_line('16:' + json.dumps({'json': {'type': 'metadata'}})) == []


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
