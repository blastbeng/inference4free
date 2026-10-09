"""Offline tests for the 'auto' smart router: it must route across ALL providers.

Covers the "auto only ever serves z.ai GLM models" bug and its two causes:

  1. namespace collision — chatgpt's upstream catalog exposes a model literally
     named ``auto``, which used to be written over the synthetic smart-router
     route, so ``model: "auto"`` stopped being routed at all (one upstream
     model, one auth error, no chain).
  2. health signal — the chain was built from ``provider.available()`` alone,
     which only proves credentials EXIST. Every request therefore led with the
     same broken providers, walked through them and landed on the one provider
     that answers, which reads to the user as "auto only uses z.ai GLM".

No network, no browser: fake providers only.

Run:  python tests/test_auto_router.py     (or: pytest tests/)
"""
import contextlib
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from dsk.providers import router as router_mod          # noqa: E402
from dsk.providers.base import (                        # noqa: E402
    ProviderAuthError,
    ProviderError,
    ProviderRateLimitError,
)


class FakeProvider:
    """Minimal stand-in: the router only calls list_models/available/stream."""

    def __init__(self, name, models, healthy=True, error=None):
        self.name = name
        self._models = list(models)
        self.healthy = healthy
        self.error = error            # raised by stream(), when set
        self.calls = []

    def available(self, auth_key=None):
        return self.healthy

    def list_models(self, auth_key=None):
        return list(self._models)

    def stream(self, prompt, model=None, **kwargs):
        self.calls.append(model)
        if self.error is not None:
            raise self.error
        yield {'content': f'{self.name}:{model}', 'type': 'text',
               'finish_reason': 'stop'}


def build_router(specs):
    """A Router holding only fake providers (no real provider is instantiated)."""
    saved = router_mod.provider_enabled
    router_mod.provider_enabled = lambda name: False
    try:
        router = router_mod.Router()
    finally:
        router_mod.provider_enabled = saved
    router.providers = dict(specs)
    for name, provider in specs.items():
        router._apply_provider_models(name, provider.list_models())
    router._auto_routes()
    return router


@contextlib.contextmanager
def no_refresher():
    """Keep the auth-error remediation ladder from firing in tests."""
    saved = os.environ.get('I4F_REFRESHER')
    os.environ['I4F_REFRESHER'] = '0'
    try:
        yield
    finally:
        if saved is None:
            os.environ.pop('I4F_REFRESHER', None)
        else:
            os.environ['I4F_REFRESHER'] = saved


@contextlib.contextmanager
def no_retries():
    """Skip the retry ladder (it sleeps) in tests that exhaust it."""
    saved = router_mod.MAX_RETRIES
    router_mod.MAX_RETRIES = 0
    try:
        yield
    finally:
        router_mod.MAX_RETRIES = saved


# ------------------------------------------- 1. reserved router-owned ids

def test_provider_model_named_auto_cannot_hijack_the_smart_router(tmp):
    chatgpt = FakeProvider('chatgpt', [{'id': 'auto'}, {'id': 'gpt-5'}])
    glm = FakeProvider('glm', [{'id': 'glm-4.7'}])
    router = build_router({'chatgpt': chatgpt, 'glm': glm})

    # the synthetic route keeps the 'auto' id, and resolve() reaches the router
    assert router.routes['auto'].provider_name == 'router'
    assert router.resolve('auto').provider_name == 'router'

    # the colliding upstream model stays reachable, under a namespaced id
    assert router.routes['chatgpt-auto'].provider_name == 'chatgpt'
    assert router.routes['chatgpt-auto'].upstream_model == 'auto'

    # the provider-scoped router id is not shadowed by it either
    assert router.routes['openai/auto'].provider_name == 'router'
    assert router.resolve('openai/auto').upstream_model == 'chatgpt'
    ids = [entry['id'] for entry in router.list_models()]
    assert ids.count('openai/auto') == 1, ids

    # no chain position is ever a router-owned id
    chain = router._auto_chain('general')
    assert 'auto' not in chain
    assert 'chatgpt-auto' in chain and 'gpt-5' in chain

    # re-discovery is idempotent: the namespaced route keeps its real upstream
    # model and the smart router still owns 'auto'
    router._apply_provider_models('chatgpt', chatgpt.list_models())
    assert router.routes['chatgpt-auto'].upstream_model == 'auto'
    assert router.routes['auto'].provider_name == 'router'
    # serving it asks the provider for the real upstream model, not the alias
    list(router.stream(router.resolve('chatgpt-auto'), 'hi'))
    assert chatgpt.calls[-1] == 'auto'


def test_auto_route_reclaims_a_clobbered_id(tmp):
    router = build_router({'chatgpt': FakeProvider('chatgpt',
                                                   [{'id': 'gpt-5'}])})
    # simulate the old bug: a provider route sitting on the reserved id
    router.routes['auto'] = router.routes['gpt-5']
    assert router._auto_route().provider_name == 'router'
    assert router.routes['auto'].provider_name == 'router'


# ------------------------------------- 2. spread across every provider

def test_auto_chain_rotates_across_every_healthy_provider(tmp):
    router = build_router({
        # one provider with many models must not lead every call
        'glm': FakeProvider('glm', [{'id': f'glm-{i}'} for i in range(4)]),
        'deepseek': FakeProvider('deepseek', [{'id': 'deepseek-chat'}]),
        'qwen': FakeProvider('qwen', [{'id': 'qwen-max'}]),
    })
    heads = []
    for _ in range(6):
        chain = router._auto_chain('general')
        assert len(chain) == 6, chain          # every model of every provider
        heads.append(router.routes[chain[0]].provider_name)
    assert set(heads) == {'glm', 'deepseek', 'qwen'}, heads


def test_auto_chain_covers_providers_outside_the_category_tables(tmp):
    router = build_router({
        'glm': FakeProvider('glm', [{'id': 'glm-4.7'}]),
        'grok': FakeProvider('grok', [{'id': 'grok-4'}]),
        'perplexity': FakeProvider('perplexity', [{'id': 'perplexity-turbo'}]),
    })
    chain = router._auto_chain('general')
    served = {router.routes[mid].provider_name for mid in chain}
    assert served == {'glm', 'grok', 'perplexity'}, chain


# ------------------------------- 3. real outcomes complete the health signal

def test_failed_provider_is_demoted_out_of_the_chain_front(tmp):
    with no_refresher():
        chatgpt = FakeProvider('chatgpt', [{'id': 'gpt-5'}],
                               error=ProviderAuthError('HTTP 403'))
        glm = FakeProvider('glm', [{'id': 'glm-4.7'}])
        router = build_router({'chatgpt': chatgpt, 'glm': glm})

        # credentials exist, so chatgpt is considered healthy up front
        assert router._healthy_model('gpt-5') is True

        chunks = list(router.stream(router.resolve('auto'), 'hello'))
        assert chunks[0]['served_by'] == 'glm/glm-4.7'

        # the outcome is remembered: chatgpt no longer leads the chain ...
        assert router._healthy_model('gpt-5') is False
        chain = router._auto_chain('general')
        assert router.routes[chain[0]].provider_name == 'glm'
        # ... but it stays at the back, so it is still probed and recovers
        assert router.routes[chain[-1]].provider_name == 'chatgpt'


def test_demotion_expires_and_the_provider_comes_back(tmp):
    router = build_router({'chatgpt': FakeProvider('chatgpt',
                                                   [{'id': 'gpt-5'}]),
                           'glm': FakeProvider('glm', [{'id': 'glm-4.7'}])})
    router._mark_failed('chatgpt', 'test')
    assert router._healthy_model('gpt-5') is False
    with router._rt_lock:
        router._rt_fail['chatgpt'] = time.time() - 1  # window elapsed
    assert router._healthy_model('gpt-5') is True


def test_rate_limit_does_not_demote_a_working_provider(tmp):
    with no_refresher(), no_retries():
        chatgpt = FakeProvider('chatgpt', [{'id': 'gpt-5'}],
                               error=ProviderRateLimitError('429'))
        glm = FakeProvider('glm', [{'id': 'glm-4.7'}])
        router = build_router({'chatgpt': chatgpt, 'glm': glm})
        chunks = list(router.stream(router.resolve('auto'), 'hello'))
        assert chunks[0]['served_by'] == 'glm/glm-4.7'
        assert router._healthy_model('gpt-5') is True


def test_provider_that_served_is_healthy_even_if_available_disagrees(tmp):
    glm = FakeProvider('glm', [{'id': 'glm-4.7'}], healthy=False)
    router = build_router({'glm': glm})
    assert router._healthy_model('glm-4.7') is False
    router._mark_served('glm')
    assert router._healthy_model('glm-4.7') is True


def test_auto_router_serves_the_next_provider_not_the_first_one(tmp):
    """The reported symptom: only z.ai GLM answered, because every other
    provider in front of it was broken. With runtime health the working
    providers lead and take turns."""
    with no_refresher():
        broken = {name: FakeProvider(name, [{'id': f'{name}-m'}],
                                     error=ProviderAuthError('wall'))
                  for name in ('chatgpt', 'gemini', 'mistral')}
        working = {name: FakeProvider(name, [{'id': f'{name}-m'}])
                   for name in ('glm', 'deepseek', 'qwen')}
        router = build_router({**broken, **working})

        served = []
        for _ in range(6):
            chunks = list(router.stream(router.resolve('auto'), 'hello'))
            served.append(chunks[0]['served_by'].split('/')[0])
        assert set(served) == {'glm', 'deepseek', 'qwen'}, served
        assert 'chatgpt' not in served and 'gemini' not in served


# ------------------------------------------- 4. auto-fast / auto-thinking pools

class RecordingProvider(FakeProvider):
    """FakeProvider that also records every stream() kwargs dict."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stream_kwargs = []

    def stream(self, prompt, model=None, **kwargs):
        self.stream_kwargs.append(dict(kwargs))
        yield from super().stream(prompt, model=model, **kwargs)


def _pool_router_specs():
    """Two providers whose catalogs split cleanly across the thinking pools."""
    glm = FakeProvider('glm', [{'id': 'glm-4.7', 'thinking_enabled': True},
                               {'id': 'glm-4.6'}])
    deepseek = FakeProvider('deepseek', [
        {'id': 'deepseek-chat'},
        {'id': 'deepseek-reasoner', 'thinking_enabled': True},
        {'id': 'deepseek-search', 'search_enabled': True},
    ])
    return {'glm': glm, 'deepseek': deepseek}


def test_pool_routers_registered_and_resolvable(tmp):
    router = build_router(_pool_router_specs())
    # every kind resolves globally, via the public prefix and via the bare
    # provider name — always to a synthetic router-owned route
    for mid in ('auto', 'auto-fast', 'auto-thinking', 'z.ai/auto',
                'deepseek/auto', 'z.ai/auto-fast', 'deepseek/auto-fast',
                'z.ai/auto-thinking', 'deepseek/auto-thinking',
                'glm/auto-fast'):
        assert router.resolve(mid).provider_name == 'router', mid
    assert router.resolve('AUTO-FAST').model_id == 'auto-fast'  # lowercased
    assert router.resolve('deepseek/auto-thinking').upstream_model == 'deepseek'

    # /v1/models lists each router id exactly once
    ids = [entry['id'] for entry in router.list_models()]
    for mid in ('auto', 'auto-fast', 'auto-thinking', 'z.ai/auto',
                'deepseek/auto', 'z.ai/auto-fast', 'deepseek/auto-fast',
                'z.ai/auto-thinking', 'deepseek/auto-thinking'):
        assert ids.count(mid) == 1, (mid, ids)


def test_upstream_model_named_auto_fast_cannot_hijack_the_router(tmp):
    chatgpt = FakeProvider('chatgpt', [{'id': 'auto-fast'}, {'id': 'gpt-5'}])
    router = build_router({'chatgpt': chatgpt})

    # the synthetic route keeps the 'auto-fast' id …
    assert router.routes['auto-fast'].provider_name == 'router'
    assert router.resolve('auto-fast').provider_name == 'router'
    # … and the colliding upstream model stays reachable, namespaced
    assert router.routes['chatgpt-auto-fast'].upstream_model == 'auto-fast'
    ids = [entry['id'] for entry in router.list_models()]
    assert ids.count('openai/auto-fast') == 1, ids


def test_pools_partition_models_strictly(tmp):
    router = build_router(_pool_router_specs())

    # auto-fast: ONLY never-thinking models — thinking models AND search
    # modes are excluded, even though both are served by plain 'auto'
    fast = router._auto_chain('general', pool='fast')
    assert set(fast) == {'glm-4.6', 'deepseek-chat'}, fast

    # auto-thinking: ONLY models that can think (deepseek-search does not)
    think = router._auto_chain('general', pool='thinking')
    assert set(think) == {'glm-4.7', 'deepseek-reasoner'}, think

    # plain auto keeps the legacy full chain
    full = router._auto_chain('general')
    assert set(full) == {'glm-4.7', 'glm-4.6', 'deepseek-chat',
                         'deepseek-reasoner', 'deepseek-search'}, full


def test_provider_pool_chains_restrict_to_one_provider(tmp):
    router = build_router(_pool_router_specs())
    assert router._provider_chain('glm', pool='thinking') == ['glm-4.7']
    assert router._provider_chain('glm', pool='fast') == ['glm-4.6']
    assert router._provider_chain('deepseek', pool='fast') == ['deepseek-chat']
    assert (router._provider_chain('deepseek', pool='thinking')
            == ['deepseek-reasoner'])
    # search models are in NEITHER pool
    assert router._provider_chain('deepseek') == ['deepseek-chat',
                                                  'deepseek-reasoner',
                                                  'deepseek-search']


def test_stream_forces_thinking_flag_by_pool_contract(tmp):
    glm = RecordingProvider('glm', [{'id': 'glm-4.7', 'thinking_enabled': True},
                                    {'id': 'glm-4.6'}])
    router = build_router({'glm': glm})

    # auto-thinking forces thinking ON, even when the request asks for False
    chunks = list(router.stream(router.resolve('auto-thinking'), 'hi',
                                thinking_override=False))
    assert chunks[0]['served_by'] == 'glm/glm-4.7'
    assert glm.stream_kwargs[-1]['thinking_enabled'] is True

    # auto-fast forces thinking OFF, even when the request asks for True —
    # and it can only have been served by the never-thinking model
    chunks = list(router.stream(router.resolve('auto-fast'), 'hi',
                                thinking_override=True))
    assert chunks[0]['served_by'] == 'glm/glm-4.6'
    assert glm.stream_kwargs[-1]['thinking_enabled'] is False


def test_empty_pool_raises_clean_error(tmp):
    # glm lists ONLY a thinking model; deepseek ONLY a non-thinking one
    glm = FakeProvider('glm', [{'id': 'glm-4.7', 'thinking_enabled': True}])
    deepseek = FakeProvider('deepseek', [{'id': 'deepseek-chat'}])
    router = build_router({'glm': glm, 'deepseek': deepseek})

    # per-provider pool routers fail cleanly when that provider's pool is
    # empty — they must never fall back to the other pool's models
    with no_refresher():
        for model_id, needle in (('z.ai/auto-fast', 'non-thinking'),
                                 ('deepseek/auto-thinking', 'thinking')):
            try:
                list(router.stream(router.resolve(model_id), 'hi'))
                raise AssertionError(f'{model_id}: expected ProviderError')
            except ProviderError as e:
                assert needle in str(e), (model_id, e)

    # the GLOBAL pool routers fail cleanly only when NO provider at all has
    # a model in the pool: a glm-only registry has no never-thinking model …
    thinking_only = build_router(
        {'glm': FakeProvider('glm',
                             [{'id': 'glm-4.7', 'thinking_enabled': True}])})
    with no_refresher():
        try:
            list(thinking_only.stream(thinking_only.resolve('auto-fast'), 'hi'))
            raise AssertionError('expected ProviderError')
        except ProviderError as e:
            assert 'non-thinking' in str(e), e
        # … and a deepseek-only registry has no thinking model
        fast_only = build_router(
            {'deepseek': FakeProvider('deepseek', [{'id': 'deepseek-chat'}])})
        try:
            list(fast_only.stream(fast_only.resolve('auto-thinking'), 'hi'))
            raise AssertionError('expected ProviderError')
        except ProviderError as e:
            assert 'thinking' in str(e), e

    # and the mixed registry DOES serve both global pools (no false errors):
    # auto-fast lands on the only never-thinking model, auto-thinking on the
    # only thinking one
    with no_refresher():
        fast_chunks = list(router.stream(router.resolve('auto-fast'), 'hi'))
        assert fast_chunks[0]['served_by'] == 'deepseek/deepseek-chat'
        think_chunks = list(router.stream(router.resolve('auto-thinking'),
                                          'hi'))
        assert think_chunks[0]['served_by'] == 'glm/glm-4.7'


def test_list_models_advertises_pool_contracts(tmp):
    glm = FakeProvider('glm', [{'id': 'glm-4.7', 'thinking_enabled': True,
                                'vision': True},
                               {'id': 'glm-4.6'}])
    router = build_router({'glm': glm})
    entries = {entry['id']: entry for entry in router.list_models()}

    # global routers advertise their contract flags
    assert entries['auto-fast']['thinking_enabled'] is False
    assert entries['auto-thinking']['thinking_enabled'] is True
    # per-provider variants: capability unions over the POOL's models only
    assert entries['z.ai/auto-fast']['thinking_enabled'] is False
    assert entries['z.ai/auto-fast']['vision'] is False     # glm-4.6 has none
    assert entries['z.ai/auto-thinking']['thinking_enabled'] is True
    assert entries['z.ai/auto-thinking']['vision'] is True  # glm-4.7 has it


def test_pool_rotations_use_independent_cursors(tmp):
    specs = _pool_router_specs()
    # a provider with two never-thinking models: its provider-scoped fast
    # router has something to rotate (single-target cursors return early)
    specs['qwen'] = FakeProvider('qwen', [{'id': 'qwen-omni-flash'},
                                          {'id': 'qwen-turbo'}])
    router = build_router(specs)
    router._auto_chain('general')
    router._auto_chain('general', pool='fast')
    router._auto_chain('general', pool='thinking')
    router._provider_chain('qwen', pool='fast')
    keys = set(router._rr)
    assert any(k.startswith('auto:') for k in keys), keys
    assert any(k.startswith('auto-fast:') for k in keys), keys
    assert any(k.startswith('auto-thinking:') for k in keys), keys
    assert any(k.startswith('pauto:auto-fast:qwen') for k in keys), keys


# ----------------------- 6. router-id recognition & /v1/models tools flag

def test_router_id_helpers(tmp):
    """router_id_parts/is_router_model_id cover ALL router kinds — including
    '<prefix>/auto' (regression: the tail check used to know only about the
    two pool kinds, so deepseek/auto stopped being recognized as a router id
    by the /v1 gates and the toolprobe)."""
    parts = router_mod.router_id_parts
    is_rid = router_mod.is_router_model_id

    # global routers
    assert parts('auto') == ('auto', '')
    assert parts('auto-fast') == ('auto-fast', '')
    assert parts('auto-thinking') == ('auto-thinking', '')
    # provider-scoped routers — every kind, public prefix or bare name
    assert parts('deepseek/auto') == ('auto', 'deepseek')
    assert parts('z.ai/auto-fast') == ('auto-fast', 'z.ai')
    assert parts('alibaba/auto-thinking') == ('auto-thinking', 'alibaba')
    # case-insensitive, exactly like resolve()
    assert parts('AUTO') == ('auto', '')
    assert parts('DeepSeek/Auto') == ('auto', 'deepseek')
    assert is_rid('Z.AI/AUTO-FAST')
    # leaf ids never match
    assert parts('deepseek-chat') is None
    assert parts('deepseek/deepseek-chat') is None
    assert parts('z.ai/glm-4.7') is None
    assert parts('') is None
    assert parts(None) is None
    assert parts('auto/custom') is None
    assert parts('x/auto-fast-turbo') is None
    assert not is_rid('deepseek/deepseek-chat')
    # the regression itself: '<prefix>/auto' must stay a router id
    assert is_rid('deepseek/auto')
    assert is_rid('openai/auto')


def test_annotate_models_router_tools_flag(tmp):
    """The /v1/models 'tools' flag on router entries is pool-aware: global
    routers look at every provider's pool leaves, provider-scoped routers
    only at their own, and auto-fast/auto-thinking only at their pool's
    leaves (same strict partition the router serves with)."""
    from dsk import toolprobe
    saved_state, saved_loaded = toolprobe._state, toolprobe._loaded
    saved_hide = os.environ.get('I4F_HIDE_TOOLLESS')
    toolprobe._state = {}
    toolprobe._loaded = True          # _load() becomes a no-op: no disk I/O
    os.environ['I4F_HIDE_TOOLLESS'] = '0'
    try:
        toolprobe._state.update({
            'deepseek/deepseek-chat': {'status': 'ok'},
            'deepseek/deepseek-reasoner': {'status': 'failed'},
            'z.ai/glm-4.6': {'status': 'ok'},
            'z.ai/glm-4.7': {'status': 'unknown'},
        })
        data = [
            {'id': 'auto'},
            {'id': 'auto-fast'},
            {'id': 'auto-thinking'},
            {'id': 'deepseek/auto'},
            {'id': 'deepseek/auto-fast'},
            {'id': 'deepseek/auto-thinking'},
            {'id': 'deepseek/deepseek-chat',
             'thinking_enabled': False, 'search_enabled': False},
            {'id': 'deepseek/deepseek-reasoner',
             'thinking_enabled': True, 'search_enabled': False},
            {'id': 'z.ai/glm-4.6',
             'thinking_enabled': False, 'search_enabled': False},
            {'id': 'z.ai/glm-4.7',
             'thinking_enabled': True, 'search_enabled': False},
        ]
        out = {e['id']: e for e in toolprobe.annotate_models(data)}
        assert out['auto']['tools'] is True            # any leaf ok
        assert out['auto-fast']['tools'] is True       # glm-4.6 ok in pool
        assert out['auto-thinking']['tools'] is True   # glm-4.7 unknown
        assert out['deepseek/auto']['tools'] is True   # deepseek-chat ok
        assert out['deepseek/auto-fast']['tools'] is True
        assert out['deepseek/auto-thinking']['tools'] is False  # all failed
        assert out['deepseek/deepseek-reasoner']['tools'] is False

        # every deepseek leaf failed: the provider routers flip to False,
        # the global routers survive on z.ai
        toolprobe._state['deepseek/deepseek-chat']['status'] = 'failed'
        out = {e['id']: e for e in toolprobe.annotate_models(data)}
        assert out['deepseek/auto']['tools'] is False
        assert out['deepseek/auto-fast']['tools'] is False
        assert out['auto']['tools'] is True
        assert out['auto-fast']['tools'] is True       # z.ai/glm-4.6 ok

        # all thinking leaves failed -> auto-thinking False, auto still True
        toolprobe._state['z.ai/glm-4.7']['status'] = 'failed'
        out = {e['id']: e for e in toolprobe.annotate_models(data)}
        assert out['auto-thinking']['tools'] is False
        assert out['auto']['tools'] is True

        # hide_toolless drops failed LEAVES but never router entries
        os.environ['I4F_HIDE_TOOLLESS'] = '1'
        ids = [e['id'] for e in toolprobe.annotate_models(data)]
        assert 'deepseek/deepseek-reasoner' not in ids
        assert 'z.ai/glm-4.7' not in ids
        for rid in ('auto', 'auto-fast', 'auto-thinking',
                    'deepseek/auto', 'deepseek/auto-fast',
                    'deepseek/auto-thinking'):
            assert rid in ids, ids
    finally:
        toolprobe._state = saved_state
        toolprobe._loaded = saved_loaded
        if saved_hide is None:
            os.environ.pop('I4F_HIDE_TOOLLESS', None)
        else:
            os.environ['I4F_HIDE_TOOLLESS'] = saved_hide


# ------------------------------------- chain-exhaustion error aggregation

def test_chain_exhaustion_names_every_failed_provider(tmp):
    """The client must see WHY its model failed, not the last provider's
    complaint: a chatgpt request that walked a busy relay into a muted
    deepseek must not read as "deepseek muted" alone."""
    a = FakeProvider('chatgpt', [{'id': 'gpt-5'}],
                     error=ProviderError('chatgpt browser relay: relay busy '
                                         'with another stream'))
    b = FakeProvider('deepseek', [{'id': 'deepseek-chat'}],
                     error=ProviderAuthError('deepseek account muted '
                                             '(biz_code=5: user is muted)'))
    router = build_router({'chatgpt': a, 'deepseek': b})
    route = router.resolve('auto')
    with no_refresher(), no_retries():
        try:
            list(router.stream(route, 'hi'))
            raise AssertionError('expected the chain to exhaust into an error')
        except ProviderAuthError as e:
            msg = str(e)
    # the LAST provider's exception type is preserved (auth -> 401 mapping)
    assert 'deepseek account muted' in msg
    # and the full chain history travels with it
    assert 'relay busy with another stream' in msg
    assert 'fallback chain exhausted' in msg


def test_single_provider_failure_message_stays_untouched(tmp):
    """One provider, one failure: no chain summary is appended — the
    message must stay byte-identical to the provider's own error."""
    a = FakeProvider('chatgpt', [{'id': 'gpt-5'}],
                     error=ProviderError('relay busy with another stream'))
    router = build_router({'chatgpt': a})
    route = router.resolve('chatgpt-gpt-5')
    with no_refresher(), no_retries():
        try:
            list(router.stream(route, 'hi'))
            raise AssertionError('expected the provider error to surface')
        except ProviderError as e:
            assert str(e) == 'relay busy with another stream'


# ------------------------------------------------------------- runner

def _main() -> int:
    import shutil
    import tempfile
    import traceback
    from pathlib import Path
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith('test_') and callable(f)]
    failed = 0
    for name, fn in tests:
        tmp = Path(tempfile.mkdtemp(prefix='i4f-test-'))
        try:
            fn(tmp)
            print(f'PASS  {name}')
        except Exception:
            failed += 1
            print(f'FAIL  {name}')
            traceback.print_exc()
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    print(f'\n{len(tests) - failed}/{len(tests)} passed')
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(_main())
