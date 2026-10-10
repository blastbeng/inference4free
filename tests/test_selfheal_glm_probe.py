"""Offline tests for the honest glm (z.ai) self-heal probe.

Regression: the generic ``_probe_once`` branch trusted ``list_models()``, so a
z.ai outage — the anonymous tier gated with a 403 "Model not available for
current user level", or a stale guest session that streams nothing — was
reported as ``ok`` on ``/health`` and never surfaced to the refresher. The glm
branch must validate a tiny REAL completion, and must never classify a
transient empty-session / busy-browser condition as ``structural`` (that would
trigger an LLM auto-patch, which cannot fix a gated account).

Run:  python tests/test_selfheal_glm_probe.py     (or: pytest tests/)
"""
import contextlib
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from dsk import selfheal                        # noqa: E402
from dsk.providers import glm_provider as gp    # noqa: E402
from dsk.providers.base import (                # noqa: E402
    ProviderAuthError,
    ProviderError,
    ProviderRateLimitError,
)


class _Gen:
    """Minimal stream generator that records whether it was closed."""

    def __init__(self, events=None, exc=None):
        self._events = list(events or [])
        self._exc = exc
        self.closed = False

    def __iter__(self):
        return self

    def __next__(self):
        if self._exc is not None:
            raise self._exc
        if self._events:
            return self._events.pop(0)
        raise StopIteration

    def close(self):
        self.closed = True


class _FakeGlm:
    events = [{'content': 'ok', 'type': 'text', 'finish_reason': 'stop'}]
    exc = None
    models = [{'id': 'glm-4.7', 'backend': 'zai'}]
    avail = True
    list_exc = None
    kwargs = None
    last_gen = None

    def __init__(self, *a, **k):
        pass

    def available(self, auth_key=None):
        return type(self).avail

    def list_models(self, auth_key=None):
        if type(self).list_exc is not None:
            raise type(self).list_exc
        return type(self).models

    def stream(self, prompt, **kw):
        type(self).kwargs = kw
        gen = _Gen(type(self).events, type(self).exc)
        type(self).last_gen = gen
        return gen


@contextlib.contextmanager
def _patched():
    orig = gp.GlmProvider
    gp.GlmProvider = _FakeGlm
    try:
        yield _FakeGlm
    finally:
        gp.GlmProvider = orig


def _reset(**over):
    _FakeGlm.events = [{'content': 'ok', 'type': 'text', 'finish_reason': 'stop'}]
    _FakeGlm.exc = None
    _FakeGlm.models = [{'id': 'glm-4.7', 'backend': 'zai'}]
    _FakeGlm.avail = True
    _FakeGlm.list_exc = None
    _FakeGlm.kwargs = None
    _FakeGlm.last_gen = None
    for k, v in over.items():
        setattr(_FakeGlm, k, v)


def test_ok_on_real_completion():
    with _patched():
        _reset()
        status, detail = selfheal._probe_once('glm')
        assert status == 'ok', (status, detail)
        assert detail == 'completion responds'
        # a tiny real completion, thinking/search off, anonymous route
        assert _FakeGlm.kwargs['model'] == 'glm-4.7'
        assert _FakeGlm.kwargs['thinking_enabled'] is False
        assert _FakeGlm.kwargs['search_enabled'] is False
        assert _FakeGlm.kwargs['max_tokens'] == 8
        # the stream must be closed, not abandoned (shared browser session)
        assert _FakeGlm.last_gen.closed is True


def test_gated_upsell_403_is_auth():
    with _patched():
        _reset(exc=ProviderError(
            "z.ai stream error: {'detail': 'Model not available for current "
            "user level', 'code': 403}"))
        status, detail = selfheal._probe_once('glm')
        assert status == 'auth', (status, detail)


def test_empty_stale_session_is_network_not_structural():
    # The critical guard: an empty anonymous session must NOT trigger an
    # LLM auto-patch (structural).
    with _patched():
        _reset(exc=ProviderError(
            'z.ai browser returned no content (status=200)'))
        status, detail = selfheal._probe_once('glm')
        assert status == 'network', (status, detail)
        assert _FakeGlm.last_gen.closed is True


def test_busy_browser_is_network():
    with _patched():
        _reset(exc=ProviderError(
            'z.ai browser session is busy with another request'))
        status, _ = selfheal._probe_once('glm')
        assert status == 'network'


def test_no_events_is_network():
    with _patched():
        _reset(events=[])
        status, detail = selfheal._probe_once('glm')
        assert status == 'network', (status, detail)


def test_auth_error_from_stream_is_auth():
    with _patched():
        _reset(exc=ProviderAuthError('z.ai anonymous auth returned no token'))
        status, _ = selfheal._probe_once('glm')
        assert status == 'auth'


def test_rate_limit_from_discovery():
    with _patched():
        _reset(list_exc=ProviderRateLimitError('429 rate limited', 30))
        status, _ = selfheal._probe_once('glm')
        assert status == 'rate'


def test_unavailable_transport_is_auth():
    with _patched():
        _reset(avail=False)
        status, detail = selfheal._probe_once('glm')
        assert status == 'auth', (status, detail)


def test_chatglm_only_catalog_falls_back_to_its_first_model():
    with _patched():
        _reset(models=[{'id': 'glm-4.6', 'backend': 'chatglm'}])
        status, _ = selfheal._probe_once('glm')
        assert status == 'ok'
        assert _FakeGlm.kwargs['model'] == 'glm-4.6'


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
