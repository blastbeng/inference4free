"""Offline tests for the chatgpt browser relay's reply parsing.

The anonymous chatgpt UI is scraped from ``document.body.innerText``, and its
collapsed chain-of-thought panel ("Thought · 3s" + summary) lives INSIDE the
same text region as the answer. Emitting that region verbatim puts the
reasoning summary in the message body, which is what a client then reads as
"the reply is only the thinking" — with the answer never arriving at all.

The contract these tests pin down:
  - the ANSWER always reaches the text stream (never hidden by the panel)
  - panel chrome ("Thought · 3s", "Thinking") never reaches the text stream
  - nothing is emitted twice when the split is decided late (the summary is
    first prose, then re-read as the panel: see ``_new_delta``)
  - when the UI really separates the panel, the summary goes out as
    ``type: 'thinking'`` and the answer as ``type: 'text'``
  - the model's own disclaimer footer is cut out of the reply region
  - dead-UI states surface as RelayBlocked, not as an empty answer

No browser: ``_body_text`` is scripted, and the drain's timing constants are
shrunk on the instance.

Run:  python tests/test_chatgpt_relay.py     (or: pytest tests/)
"""
import os
import sys

ROOT = '/opt/docker/compose/inference4free'
if not os.path.isdir(os.path.join(ROOT, 'dsk')):
    ROOT = '/app'          # inside the container the code lives at /app
sys.path.insert(0, ROOT)

from dsk.chatgpt_relay import (                            # noqa: E402
    MARKER,
    REPLY_TERMINATORS,
    ChatGPTRelay,
    RelayBlocked,
    THOUGHT_HEAD_RE,
    _is_page_death,
    _new_delta,
    _region_delta,
    _reply_of,
    _split_thinking,
    _strip_headers,
    _thinking_only,
    get_relay,
    norm_model,
)

SUMMARY = 'The user asks who I am in Italian.'
ANSWER = 'Ciao! Sono Vibe.'


# --------------------------------------------------------------- region scrape
def test_reply_of_cuts_at_the_model_disclaimer():
    body = (f'Sidebar\n{MARKER}\n{ANSWER}\n'
            'ChatGPT is AI and can make mistakes.\nNew chat')
    assert _reply_of(body) == ANSWER


def test_reply_of_keeps_a_reply_that_mentions_the_terminators():
    body = f'{MARKER}\nPuoi usare Log in per salvare la chat.'
    assert _reply_of(body) == 'Puoi usare Log in per salvare la chat.'


def test_reply_of_is_empty_until_the_marker_appears():
    assert _reply_of('Thinking') == ''
    assert _reply_of(f'{MARKER}\n\nThinking') == ''


def test_the_disclaimer_is_a_registered_terminator():
    assert 'ChatGPT is AI and can make mistakes.' in REPLY_TERMINATORS


# ------------------------------------------------------------------- the header
def test_thought_head_matches_panel_headers():
    for line in ('Thought · 3s', 'Thought for 12 seconds', 'Thought · 250ms',
                 'Thinking', 'Reasoning · 1 min', 'thought · 4s'):
        assert THOUGHT_HEAD_RE.match(line), line


def test_thought_head_rejects_prose():
    for line in (SUMMARY, ANSWER, 'Thoughts on the topic',
                 'Thinking about your question in detail', 'Thank you'):
        assert not THOUGHT_HEAD_RE.match(line), line


# ------------------------------------------------------------------ the split
def test_split_thinking_separates_the_panel_from_the_answer():
    region = f'Thought · 3s\n{SUMMARY}\n\n{ANSWER}'
    assert _split_thinking(region) == (SUMMARY, ANSWER)


def test_split_thinking_skips_the_gap_under_the_header():
    # Block elements make innerText emit a blank line between header and
    # summary; treating THAT blank line as the separator is what put the
    # summary in the message body.
    region = f'Thought · 3s\n\n{SUMMARY}\n\n{ANSWER}'
    assert _split_thinking(region) == (SUMMARY, ANSWER)


def test_split_thinking_before_the_answer_starts_keeps_prose():
    # No separator after the summary: where the answer will begin is not
    # knowable, so nothing may be hidden in a panel — the header is dropped
    # and the rest stays prose.
    assert _split_thinking(f'Thought · 3s\n{SUMMARY}') == ('', SUMMARY)
    assert _split_thinking(f'Thought · 3s\n\n{SUMMARY}') == ('', SUMMARY)


def test_split_thinking_header_only_is_not_content():
    assert _split_thinking('Thought · 3s') == ('', '')
    assert _split_thinking('Thought · 3s\n\n') == ('', '')


def test_split_thinking_without_a_panel_is_all_prose():
    assert _split_thinking(ANSWER) == ('', ANSWER)
    assert _split_thinking('') == ('', '')


def test_split_thinking_leaks_a_multiline_summary_to_the_body():
    # Conservative by design: guessing the end of a multi-paragraph summary
    # would hide the answer, so only the first block becomes reasoning.
    region = f'Thought · 3s\n\nPara one.\n\nPara two.\n\n{ANSWER}'
    assert _split_thinking(region) == ('Para one.', f'Para two.\n\n{ANSWER}')


def test_strip_headers_removes_only_leading_chrome():
    assert _strip_headers(f'Thought · 3s\n\n{SUMMARY}') == SUMMARY
    assert _strip_headers(f'Thought · 3s\nThinking\n{SUMMARY}') == SUMMARY
    assert _strip_headers(f'{ANSWER}\nThought · 3s') == f'{ANSWER}\nThought · 3s'


def test_thinking_only_states():
    for region in ('', 'Thinking', 'Thought · 3s', 'Thought · 3s\n\nThinking'):
        assert _thinking_only(region), region
    for region in (f'Thought · 3s\n{SUMMARY}', ANSWER,
                   f'{SUMMARY}\n\n{ANSWER}'):
        assert not _thinking_only(region), region


# ------------------------------------------------------------------- the delta
def test_region_delta_growth_and_stability():
    assert _region_delta('abc', 'ab') == 'c'
    assert _region_delta('ab', 'ab') == ''
    assert _region_delta('', 'ab') == ''
    assert _region_delta(f'{ANSWER}!', ANSWER) == '!'


def test_region_delta_of_a_rewrite_emits_the_new_text():
    # Re-basing on the rewritten region without emitting it is how an answer
    # disappears, so everything past the common prefix is emitted.
    assert _region_delta(ANSWER, SUMMARY) == ANSWER
    assert _region_delta('The answer X', 'The answer') == ' X'


def test_new_delta_skips_what_the_other_region_already_sent():
    assert _new_delta(SUMMARY, '', f'Thought · 3s\n{SUMMARY}') == ''
    assert _new_delta(SUMMARY, '', '') == SUMMARY


def test_new_delta_keeps_a_short_answer_inside_a_longer_body():
    assert _new_delta('Ciao!', SUMMARY, '') == 'Ciao!'
    assert _new_delta(ANSWER, SUMMARY, '') == ANSWER


# ----------------------------------------------------------------- the drain
def _scripted(bodies, **timeouts) -> ChatGPTRelay:
    """A relay whose browser is replaced by a scripted ``body.innerText``."""
    relay = ChatGPTRelay()
    seq = list(bodies)
    state = {'i': 0}

    def _body_text() -> str:
        i = min(state['i'], len(seq) - 1)
        state['i'] += 1
        return seq[i]

    relay._body_text = _body_text
    relay.POLL_S = 0.0
    relay.STREAM_TIMEOUT = timeouts.get('stream', 5.0)
    relay.FIRST_MARK_TIMEOUT = timeouts.get('first_mark', 0.5)
    relay.THINKING_TIMEOUT = timeouts.get('thinking', 3.0)
    relay.REPLY_STALL_TIMEOUT = timeouts.get('stall', 0.05)
    return relay


def _drain(bodies, **timeouts):
    return list(_scripted(bodies, **timeouts)._drain_reply())


def _parts(chunks, kind):
    return ''.join(c['content'] for c in chunks if c['type'] == kind)


def test_drain_emits_the_panel_as_thinking_and_the_answer_as_text():
    chunks = _drain([f'{MARKER}\nThought · 3s\n{SUMMARY}\n\n{ANSWER}'])
    assert _parts(chunks, 'thinking') == SUMMARY
    assert _parts(chunks, 'text') == ANSWER
    kinds = [c['type'] for c in chunks]
    assert kinds == ['thinking', 'text', 'text']       # stop chunk last
    assert chunks[-1] == {'content': '', 'type': 'text',
                          'finish_reason': 'stop'}


def test_drain_streams_the_answer_and_keeps_the_panel_separate():
    bodies = [f'{MARKER}\nThinking',
              'Thought · 3s',
              f'{MARKER}\nThought · 3s\n{SUMMARY}\n\nCiao!',
              f'{MARKER}\nThought · 3s\n{SUMMARY}\n\n{ANSWER}']
    chunks = _drain(bodies)
    assert _parts(chunks, 'thinking') == SUMMARY
    # streamed in two text deltas ('Ciao!' + ' Sono Vibe.') == the answer
    assert _parts(chunks, 'text') == ANSWER
    assert [c['content'] for c in chunks if c['type'] == 'text'] == \
        ['Ciao!', ' Sono Vibe.', '']
    assert chunks[-1]['finish_reason'] == 'stop'


def test_drain_delivers_the_answer_when_the_panel_never_separates():
    # The leak shape: header, blank line, summary — then the answer. The
    # summary is prose at that point and goes out as text; the answer must
    # follow, and the summary must NOT be re-sent as thinking.
    bodies = [f'{MARKER}\nThought · 3s\n\n{SUMMARY}',
              f'{MARKER}\nThought · 3s\n\n{SUMMARY}\n\n{ANSWER}']
    chunks = _drain(bodies)
    assert _parts(chunks, 'text') == f'{SUMMARY}{ANSWER}'
    assert _parts(chunks, 'thinking') == ''
    assert 'Thought' not in _parts(chunks, 'text')


def test_drain_never_hides_the_answer_behind_the_panel():
    # The UI drops the panel and rewrites the region as the answer: the
    # rewrite must emit the answer, not re-base on it silently.
    bodies = [f'{MARKER}\nThought · 3s\n{SUMMARY}',
              f'{MARKER}\n{ANSWER}']
    chunks = _drain(bodies)
    assert ANSWER in _parts(chunks, 'text')
    assert chunks[-1]['finish_reason'] == 'stop'


def test_drain_answer_only_reply_is_all_text():
    chunks = _drain([f'{MARKER}\n{ANSWER}'])
    assert _parts(chunks, 'text') == ANSWER
    assert _parts(chunks, 'thinking') == ''


def test_drain_no_reply_surfaces_as_blocked():
    try:
        _drain(['Loading…'], first_mark=0.05)
    except RelayBlocked as e:
        assert 'no reply started' in str(e)
        return
    raise AssertionError('expected RelayBlocked')


def test_drain_placeholder_only_surfaces_as_blocked():
    try:
        _drain([f'{MARKER}\nThinking'], first_mark=0.05)
    except RelayBlocked as e:
        assert 'no reply started' in str(e)
        return
    raise AssertionError('expected RelayBlocked')


def test_drain_stuck_in_the_panel_surfaces_as_blocked():
    try:
        _drain([f'{MARKER}\nThought · 3s'], first_mark=0.5, thinking=0.05)
    except RelayBlocked as e:
        assert 'stuck in thinking' in str(e)
        return
    raise AssertionError('expected RelayBlocked')


# ------------------------------------------------------------------- wiring
def test_relay_singleton_and_model_normalization():
    assert get_relay() is get_relay()
    assert isinstance(get_relay(), ChatGPTRelay)
    assert norm_model('GPT-5.5') == 'gpt55'
    assert norm_model('gpt_5_5') == 'gpt55'


def test_provider_uses_the_relay_as_the_credential_free_transport():
    from dsk.providers import chatgpt_provider as cp
    assert callable(cp._relay_enabled)
    assert callable(cp.get_relay_default_title)


# ------------------------------------------------------ dead-tab retry policy
class ContextLostError(RuntimeError):
    """Stand-in for DrissionPage's mid-reload error."""


class PageDisconnectedError(RuntimeError):
    """Stand-in for DrissionPage's disconnected-tab error."""


def test_page_death_is_recognized_by_name_and_by_text():
    assert _is_page_death(ContextLostError(''))
    assert _is_page_death(PageDisconnectedError(''))
    assert _is_page_death(RuntimeError(
        'The connection to the page has been disconnected.\nVersion: 4.1.1.4'))
    assert _is_page_death(RuntimeError('page was refreshed'))
    assert not _is_page_death(RuntimeError('composer not found'))
    assert not _is_page_death(ValueError('model not offered'))


def _relay_without_browser():
    """A relay whose every browser step is stubbed out."""
    relay = ChatGPTRelay()
    relay.rebuilds = 0
    relay._ensure = lambda: None
    relay._reset_chat = lambda: None
    relay._select_model = lambda model: True
    relay._type_and_send = lambda prompt: None
    relay._resend_if_swallowed = lambda prompt: None
    relay._build = lambda: setattr(relay, 'rebuilds', relay.rebuilds + 1)
    relay.POLL_S = 0.0
    return relay


def test_a_dead_tab_pre_stream_is_retried_once():
    relay = _relay_without_browser()
    state = {'calls': 0}

    def _drain():
        state['calls'] += 1
        if state['calls'] == 1:
            raise PageDisconnectedError('The connection to the page has '
                                        'been disconnected.')
        yield {'content': ANSWER, 'type': 'text', 'finish_reason': None}
        yield {'content': '', 'type': 'text', 'finish_reason': 'stop'}

    relay._drain_reply = _drain
    chunks = list(relay.stream('ciao'))
    assert _parts(chunks, 'text') == ANSWER
    assert relay.rebuilds == 1
    assert state['calls'] == 2


def test_a_dead_tab_mid_stream_is_not_retried():
    # Replaying the prompt after content reached the client would duplicate
    # the answer, so the failure is surfaced as-is.
    relay = _relay_without_browser()
    state = {'calls': 0}

    def _drain():
        state['calls'] += 1
        yield {'content': 'partial', 'type': 'text', 'finish_reason': None}
        raise PageDisconnectedError('The connection to the page has '
                                    'been disconnected.')

    relay._drain_reply = _drain
    got = []
    try:
        for chunk in relay.stream('ciao'):
            got.append(chunk)
    except PageDisconnectedError:
        assert relay.rebuilds == 0
        assert state['calls'] == 1
        assert _parts(got, 'text') == 'partial'
        return
    raise AssertionError('expected the disconnect to surface')


def test_a_non_page_failure_is_not_retried():
    relay = _relay_without_browser()

    def _drain():
        raise RuntimeError('composer not found')
        yield  # pragma: no cover — generator marker

    relay._drain_reply = _drain
    try:
        list(relay.stream('ciao'))
    except RuntimeError as e:
        assert 'composer not found' in str(e)
        assert relay.rebuilds == 0
        return
    raise AssertionError('expected the failure to surface')


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
