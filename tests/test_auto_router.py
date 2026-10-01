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
