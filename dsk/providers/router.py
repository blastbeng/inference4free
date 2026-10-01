"""Model router: dynamic discovery, retries and cross-provider fallback chains.

The router owns every model id exposed on ``/v1/models``. Models are
DYNAMICALLY discovered from each provider's web session — no model lists are
hardcoded anywhere:

    deepseek  the web app has three fixed chat modes (plain/think/search); the
              exposed ids follow the operator's I4F_MODEL_* configuration
    gemini    discovered live from the gemini.google.com web app (batchexecute
              user-status RPC) using browser cookies
    chatgpt   discovered live from chatgpt.com/backend-api/models using the
              web session access token

``Router.refresh_models`` re-runs discovery (TTL-cached, thread-safe). A
provider whose discovery fails keeps its previously known routes, so a
transient outage never empties the registry. ``Router.stream`` transparently
retries rate limits / network failures with backoff (honoring ``Retry-After``)
and then falls back down the chain, so a single OpenAI request is always served
by the best available backend.

Configuration (env):
    I4F_MODEL_THINKER      exposed id of the DeepSeek thinking mode
                           (default deepseek-reasoner)
    I4F_MODEL_FAST         exposed id of the DeepSeek fast mode (default deepseek-chat)
    I4F_MODEL_SEARCH       exposed id of the DeepSeek search mode (default deepseek-search)
    I4F_MODELS_TTL         seconds between model re-discoveries (default 300)
    I4F_MAX_RETRIES        retries per provider before falling back (default 2)
    I4F_RETRY_BACKOFF      base backoff seconds, doubled each retry (default 2.0)
    I4F_FALLBACKS          JSON object {model_id: [fallback_id, ...]}
    I4F_DEFAULT_FALLBACKS  comma list applied to routes without explicit fallbacks
    I4F_AUTO_DEMOTE_S      seconds a provider stays out of the auto chain front
                           after a real request failure (default 300, 0 disables)
    I4F_AUTO_PROVE_S       seconds a provider that just served counts as healthy
                           even when its credential probe disagrees (default 900)

The synthetic ``auto`` model is the smart router: every request is classified
(coding / general / translation / summarize / vision / image generation) and
served by the best available provider, falling back through all other healthy
models on rate limits, auth failures (which also trigger an inline credential
renewal), outages or blocks. ``auto`` is also the default when a client omits
the model field.
"""

import importlib
import json
import logging
import os
import queue
import re
import threading
import time
from typing import Any, Dict, Generator, List, Optional, Set

from .base import (
    Route,
    Provider,
    FirstTokenTimeoutError,
    ProviderAuthError,
    ProviderError,
    ProviderRateLimitError,
    ProviderUnavailableError,
    provider_enabled,
)

# Provider classes are imported LAZILY via PROVIDER_MODULES below — a broken
# module then only disables its own provider instead of crashing startup.

logger = logging.getLogger('dsk.router')

# Registry master list: name -> (module, provider class). Both __init__ and
# the self-healing reload walk this single map so no provider can be wired
# in one place and forgotten in the other.
PROVIDER_MODULES = (
    ('deepseek', '.deepseek_provider', 'DeepSeekProvider'),
    ('gemini', '.gemini_provider', 'GeminiWebProvider'),
    ('chatgpt', '.chatgpt_provider', 'ChatGPTProvider'),
    ('claude', '.claude_provider', 'ClaudeWebProvider'),
    ('grok', '.grok_provider', 'GrokProvider'),
    ('mistral', '.mistral_provider', 'MistralProvider'),
    ('qwen', '.qwen_provider', 'QwenProvider'),
    ('kimi', '.kimi_provider', 'KimiProvider'),
    ('copilot', '.copilot_provider', 'CopilotProvider'),
    ('perplexity', '.perplexity_provider', 'PerplexityProvider'),
    ('glm', '.glm_provider', 'GlmProvider'),
)

OWNED_BY = {
    'deepseek': 'inference4free',
    'gemini': 'google',
    'chatgpt': 'openai',
    'claude': 'anthropic',
    'grok': 'xai',
    'mistral': 'mistral',
    'qwen': 'alibaba',
    'kimi': 'moonshot',
    'copilot': 'microsoft',
    'perplexity': 'perplexity',
    'glm': 'zai',
}

# Public model-id namespaces: every model surfaced via /v1/models and the
# playground carries its provider prefix (deepseek/deepseek-chat,
# z.ai/glm-4.7, alibaba/qwen-3-max, …). Routes stay
# keyed by the bare internal id — fallback chains, the auto router and the
# providers themselves never see prefixed ids; resolve() maps them back.
PUBLIC_PREFIX = {
    'deepseek': 'deepseek',
    'gemini': 'google',
    'chatgpt': 'openai',
    'claude': 'anthropic',
    'grok': 'xai',
    'mistral': 'mistral',
    'qwen': 'alibaba',
    'kimi': 'moonshot',
    'copilot': 'microsoft',
    'perplexity': 'perplexity',
    'glm': 'z.ai',
}

# Reverse of PUBLIC_PREFIX: 'z.ai' -> 'glm', used by resolve() so both the
# public namespace prefix and the bare provider name select a provider's
# auto router (z.ai/auto and glm/auto both work).
_PREFIX_TO_PROVIDER = {prefix: name for name, prefix in PUBLIC_PREFIX.items()}


def public_model_id(provider_name: str, model_id: str) -> str:
    """Namespace a provider model id for public (API/UI) consumption.

    Unknown providers pass ids through unchanged so custom setups stay
    visible.
    """
    prefix = PUBLIC_PREFIX.get(provider_name)
    if not prefix:
        return model_id
    return f'{prefix}/{model_id}'


def provider_auto_id(provider_name: str) -> str:
    """Public id of a provider-scoped auto router (deepseek/auto, z.ai/auto,
    alibaba/auto, …). Providers without a namespace prefix use
    ``<provider_name>/auto`` so the id can never collide with the global
    ``auto`` smart router."""
    prefix = PUBLIC_PREFIX.get(provider_name)
    return f'{prefix}/auto' if prefix else f'{provider_name}/auto'

MAX_RETRIES = int(os.getenv('I4F_MAX_RETRIES', '2'))
RETRY_BACKOFF = float(os.getenv('I4F_RETRY_BACKOFF', '2.0'))
# Cap for a single retry wait (honored Retry-After included): a request must
# never stall tens of seconds on one backoff — better to fall back quickly.
RETRY_CAP = max(1.0, float(os.getenv('I4F_RETRY_CAP', '10') or 10))
MODELS_TTL = float(os.getenv('I4F_MODELS_TTL', '300'))
# Deadline for the FIRST stream chunk (seconds). A provider that connects but
# yields nothing within this window (hung proxy, busy upstream, stuck session)
# is treated as unavailable so the router retries/falls back instead of
# blocking until the full stream timeout. 0 disables. Default covers the
# cold-start of browser-backed providers (z.ai).
FIRST_TOKEN_TIMEOUT = max(0.0, float(os.getenv('I4F_FIRST_TOKEN_TIMEOUT', '180') or 180))
# After a first-token stall the target is skipped for this many seconds so a
# fallback chain never pays the full deadline once per stalled model.
PROVIDER_STALL_COOLDOWN = max(0.0,
                              float(os.getenv('I4F_PROVIDER_STALL_COOLDOWN', '120') or 120))
# Runtime health for the auto routers.
#
# ``provider.available()`` only proves credentials EXIST — it cannot tell an
# expired session, a geo-blocked anonymous identity or an auth-walled account
# from a working one. A chain built from that signal alone therefore leads with
# the same broken providers on every request, every request walks through them
# and lands on the one provider that actually answers — which reads to the user
# as "auto only uses z.ai GLM models". Real request outcomes are the missing
# signal: a provider that just failed a request (bad credentials, wall, outage)
# is demoted to the back of the auto chain for AUTO_DEMOTE_S seconds, and a
# provider that just SERVED counts as available even when its credential probe
# disagrees (some providers keep working from a token file their available()
# does not see). Demoted targets stay at the very back, so they are still
# probed and recover on their own once the renewal bot fixes them.
AUTO_DEMOTE_S = max(0.0, float(os.getenv('I4F_AUTO_DEMOTE_S', '300') or 300))
AUTO_PROVE_S = max(0.0, float(os.getenv('I4F_AUTO_PROVE_S', '900') or 900))


def _first_chunk(gen, timeout: float) -> Optional[Dict[str, Any]]:
    """Pull the generator's FIRST chunk under a deadline (thread-assisted).

    Returns the first chunk, or None when the stream ended without emitting
    anything. On timeout the stalled generator is abandoned and
    ProviderUnavailableError is raised so the router retries / falls back —
    instead of the request hanging until the full stream timeout.
    Non-timeout exceptions raised by the generator propagate unchanged.
    """
    box: queue.Queue = queue.Queue()

    def _pull():
        try:
            box.put(next(gen))
        except StopIteration:
            box.put(None)
        except BaseException as exc:  # noqa: BLE001 — re-raised in caller
            box.put(exc)

    threading.Thread(target=_pull, name='first-token', daemon=True).start()
    try:
        item = box.get(timeout=timeout)
    except queue.Empty:
        try:
            gen.close()  # no-op when the frame is executing in the worker
        except Exception:  # noqa: BLE001 — best-effort cleanup
            pass
        raise FirstTokenTimeoutError(
            f'no first token within {timeout:.0f}s (stalled upstream/proxy)')
    if item is None:
        return None
    if isinstance(item, BaseException):
        raise item
    return item


def _csv_env(name: str, default: str) -> List[str]:
    raw = os.getenv(name, '') or default
    return [item.strip() for item in raw.split(',') if item.strip()]


def _parse_fallbacks() -> Dict[str, List[str]]:
    """Parse I4F_FALLBACKS JSON ({model_id: [fallback, ...]})."""
    raw = (os.getenv('I4F_FALLBACKS', '') or '').strip()
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except ValueError:
        logger.warning('I4F_FALLBACKS is not valid JSON, ignoring: %.100s', raw)
        return {}
    out: Dict[str, List[str]] = {}
    if isinstance(data, dict):
        for model_id, chain in data.items():
            if isinstance(chain, list):
                out[str(model_id)] = [str(f) for f in chain]
    return out


# ---------------------------------------------------------------------------
# 'auto' smart router: request classification + category preference tables.
# The tables list PREFERRED PROVIDERS per request category (never concrete
# model ids): whatever each web provider discovers dynamically is picked up
# automatically. At serve time the chain is filtered by live credential state
# (available()) and every other discovered model is appended as a safety net.
# ---------------------------------------------------------------------------
AUTO_MODEL_ID = 'auto'

AUTO_CATEGORIES: Dict[str, List[str]] = {
    'image_gen':   ['chatgpt', 'gemini', 'glm'],
    'vision':      ['chatgpt', 'gemini', 'glm'],
    'translation': ['gemini', 'chatgpt', 'deepseek', 'glm', 'qwen', 'mistral'],
    'summarize':   ['chatgpt', 'gemini', 'glm', 'qwen', 'mistral', 'deepseek'],
    'coding':      ['deepseek', 'glm', 'qwen', 'kimi', 'mistral', 'chatgpt'],
    'general':     ['chatgpt', 'gemini', 'glm', 'deepseek', 'qwen', 'mistral',
                    'kimi'],
}

_RE_CODE_FENCE = re.compile(
    r'```|\bdef\s+\w+\s*\(|\bclass\s+\w+\s*[(:]|\bfunction\s+\w+\s*\('
    r'|\bconsole\.log\s*\(|^\s*(?:import|from)\s+\w+', re.MULTILINE)
_RE_CODE_HINTS = re.compile(
    r'\b(?:python|javascript|typescript|golang|rust|sql|regex|json|yaml|html|css|'
    r'bug|debug|traceback|exception|compile|refactor|npm|pytest|docker|bash|'
    r'script|snippet|function|algorithm|program|code|api|endpoint|database|'
    r'java|c\+\+|c#|php|swift|kotlin)\b', re.IGNORECASE)
_RE_CODE_ASK = re.compile(
    r'\b(?:write|create|fix|debug|refactor|optimize|implement|generate|convert|'
    r'explain|review)\b[^.?!]{0,80}\b(?:function|script|class|code|program|'
    r'query|regex|component|endpoint|algorithm)\b', re.IGNORECASE)
_RE_EXPLAIN_LANG = re.compile(
    r'\b(?:explain|how)\b[^.?!]{0,60}\b(?:in|with|using)\s+'
    r'(?:python|javascript|typescript|java|golang|rust|c\+\+|php|sql|bash)\b',
    re.IGNORECASE)
_RE_TRANSLATE = re.compile(
    r'\btranslat(?:e|ion|ing)\b|tradu[cz]|\u00fcbersetz|\u7ffb\u8bd1',
    re.IGNORECASE)
_RE_SUMMARIZE = re.compile(
    r'\bsummari[sz]e\b|\bsummary\b|\btldr\b|\btl;dr\b|\bkey points\b'
    r'|\bin brief\b|\bcondense\b', re.IGNORECASE)


def classify_request(prompt: str, has_images: bool = False,
                     image_generation: bool = False) -> str:
    """Cheap heuristic classification of a request into an 'auto' category.

    Regex-only (no LLM roundtrip): image payloads win first, then
    translation/summarization phrasings, code fences/definitions and finally
    coding keywords (two or more). Everything else is general chat; a wrong
    guess is harmless because the fallback chain still serves the request.
    """
    if image_generation:
        return 'image_gen'
    if has_images:
        return 'vision'
    text = prompt or ''
    if _RE_TRANSLATE.search(text):
        return 'translation'
    if _RE_SUMMARIZE.search(text):
        return 'summarize'
    if (_RE_CODE_FENCE.search(text) or _RE_CODE_ASK.search(text)
            or _RE_EXPLAIN_LANG.search(text)
            or len(_RE_CODE_HINTS.findall(text)) >= 2):
        return 'coding'
    return 'general'


class Router:
    """Registry of providers and routes with retry/fallback orchestration.

    Routes are (re)built dynamically from each provider's ``list_models``;
    see ``refresh_models``.
    """

    def __init__(self) -> None:
        self.providers: Dict[str, Provider] = {
            name: getattr(importlib.import_module(module, __package__), cls)()
            for name, module, cls in PROVIDER_MODULES
            if provider_enabled(name)
        }
        self.routes: Dict[str, Route] = {}
        # provider name -> {public prefixed id -> internal route id}; rebuilt
        # per provider in _apply_provider_models, consumed by resolve().
        self._aliases: Dict[str, Dict[str, str]] = {}
        # target ('provider/model') -> epoch until which it is skipped after
        # a first-token stall: a stalled upstream rarely recovers in seconds,
        # so later chain positions are served immediately instead of
        # re-waiting the full first-token deadline per target.
        self._stall_until: Dict[str, float] = {}
        # Round-robin cursors for the auto routers: each request starts its
        # chain at the NEXT healthy target, cycling among all of them instead
        # of always serving the same first model (e.g. glm-4.7 on every call).
        self._rr: Dict[str, int] = {}
        self._rr_lock = threading.Lock()
        # Runtime health for the auto chains (see AUTO_DEMOTE_S / AUTO_PROVE_S).
        # provider -> epoch until which it is demoted out of the healthy front
        # after a real request failure (auth wall, outage, stall).
        self._rt_fail: Dict[str, float] = {}
        # provider -> epoch of its last successful first token: a provider that
        # just served is healthy even when its credential probe disagrees.
        self._rt_ok: Dict[str, float] = {}
        self._rt_lock = threading.Lock()
        self._lock = threading.Lock()
        # Serializes whole discovery runs; request paths must never wait on
        # a slow provider's list_models (HF cold discovery takes minutes),
        # so network I/O happens BEFORE _lock is taken below.
        self._refresh_lock = threading.Lock()
        self._refreshed_at = 0.0
        # The DeepSeek modes are configuration-derived (no network involved),
        # so the registry is never empty, even before web discovery completes.
        try:
            if 'deepseek' in self.providers:
                self._apply_provider_models(
                    'deepseek', self.providers['deepseek'].list_models())
            self._apply_fallbacks()
        except Exception as e:  # pragma: no cover - defensive
            logger.warning('deepseek route bootstrap failed: %s', e)
        self._auto_routes()

    # -------------------------------------------------------------- discovery
    def refresh_models(self, auth_key: Optional[str] = None,
                       force: bool = False) -> bool:
        """(Re-)discover models from every provider. Thread-safe, TTL-cached.

        Per-provider failures are tolerated: a provider that fails discovery
        keeps its previously known routes. Returns True when the registry
        changed.
        """
        with self._refresh_lock:
            if not force and time.time() - self._refreshed_at < MODELS_TTL:
                return False
            changed = False
            discovered: Dict[str, List[Dict[str, Any]]] = {}
            for name, provider in self.providers.items():
                try:
                    # Only DeepSeek can authenticate per-request (userToken as
                    # API key); the web providers use operator cookies.
                    # OUTSIDE _lock: a slow discovery (cold provider start)
                    # must not stall request routing for minutes.
                    discovered[name] = provider.list_models(
                        auth_key if name == 'deepseek' else None)
                except ProviderAuthError as e:
                    logger.info('%s: no credentials for model discovery (%s)',
                                name, e)
                except ProviderError as e:
                    logger.warning('%s model discovery failed: %s', name, e)
                except Exception as e:  # never let discovery kill the registry
                    logger.warning('%s model discovery crashed: %s', name, e)
            with self._lock:
                for name, models in discovered.items():
                    if self._apply_provider_models(name, models):
                        changed = True
                self._apply_fallbacks()
            self._refreshed_at = time.time()
            if changed:
                logger.info('model registry updated: %d models available',
                            len(self.routes))
            return changed

    def stale(self) -> bool:
        """True when the registry is older than MODELS_TTL (lock-free hint)."""
        return time.time() - self._refreshed_at >= MODELS_TTL

    def maybe_refresh_async(self) -> bool:
        """Kick off a background re-discovery when the registry is stale.

        Used by /v1/models (stale-while-revalidate): the endpoint answers
        instantly with the current registry while a daemon thread refreshes
        stale providers. Blocking the request on discovery instead made the
        playground's first model load sit on ``loading…`` for up to a minute
        (browser-warming providers) and pushed users to the reload button.
        Concurrent calls dedupe on refresh_models' own TTL check.
        """
        if not self.stale():
            return False
        threading.Thread(target=self.refresh_models,
                         name='model-refresh', daemon=True).start()
        return True

    def _reserved_ids(self) -> Set[str]:
        """Model ids owned by the router itself.

        The global ``auto`` smart router and every ``<prefix>/auto``
        provider-scoped router are synthetic routes, not upstream models. An
        upstream catalog that happens to expose a model literally named
        ``auto`` (chatgpt's /backend-api/models does exactly that) must never
        take one of these ids over: writing it into ``routes`` silently
        replaces the smart router with that single upstream model, so every
        ``model: "auto"`` request stops being routed at all.
        """
        reserved = {AUTO_MODEL_ID}
        reserved.update(provider_auto_id(name) for name in self.providers)
        return reserved

    def _apply_provider_models(self, name: str,
                               models: List[Dict[str, Any]]) -> bool:
        """Replace one provider's routes with its discovered models."""
        changed = False
        wanted = set()
        aliases: Dict[str, str] = {}
        reserved = self._reserved_ids()
        for entry in models:
            model_id = str(entry.get('id') or '').strip()
            if not model_id:
                continue
            upstream = str(entry.get('upstream_model') or model_id)
            # Never let a discovered model take a router-owned id. Keep it
            # reachable under a namespaced id instead of dropping it (the
            # provider is still asked for the ORIGINAL upstream model).
            if model_id in reserved or public_model_id(name, model_id) in reserved:
                namespaced = f'{name}-{model_id}'
                owner = self.routes.get(namespaced)
                if namespaced in reserved or (owner is not None
                                              and owner.provider_name != name):
                    logger.info('%s: discovered model %r skipped — its id '
                                'belongs to the smart router', name, model_id)
                    continue
                if owner is None:
                    logger.info('%s: discovered model %r registered as %r — the '
                                'bare id belongs to the smart router',
                                name, model_id, namespaced)
                model_id = namespaced
            wanted.add(model_id)
            route = Route(
                model_id=model_id,
                provider_name=name,
                upstream_model=upstream,
                thinking_enabled=bool(entry.get('thinking_enabled')),
                search_enabled=bool(entry.get('search_enabled')),
                vision=bool(entry.get('vision')),
                image_gen=bool(entry.get('image_gen')),
                context_length=int(entry.get('context_length') or 131072),
                max_output_tokens=int(entry.get('max_output_tokens') or 32768),
                extra=dict(entry.get('extra') or {}),
            )
            if self.routes.get(model_id) != route:
                changed = True
            self.routes[model_id] = route
            # Public namespace alias: the canonical prefixed id maps back.
            pub = public_model_id(name, model_id)
            aliases[pub] = model_id
        self._aliases[name] = aliases
        # Drop models of this provider that disappeared upstream. Router-owned
        # synthetic routes (the 'auto' smart routers) never match a real
        # provider name, so re-discovery can never drop them.
        for model_id in [m for m, r in self.routes.items()
                         if r.provider_name == name and m not in wanted]:
            del self.routes[model_id]
            changed = True
        return changed

    def _apply_fallbacks(self) -> None:
        """Attach the configured fallback chains to every route.

        Fallback ids are kept raw: unknown targets are skipped at serve time,
        so chains may reference providers whose discovery has not completed
        yet.
        """
        explicit = _parse_fallbacks()
        default_chain = _csv_env('I4F_DEFAULT_FALLBACKS', '')
        for model_id, route in self.routes.items():
            if route.provider_name == 'router':
                continue  # auto routers: dynamic chain, rebuilt per request
            chain = explicit.get(model_id) or default_chain
            route.fallbacks = []
            for fallback in chain:
                if fallback != model_id and fallback not in route.fallbacks:
                    route.fallbacks.append(fallback)

    # ----------------------------------------------------------- auto routing
    def _auto_route(self) -> Route:
        """Return (registering on first use) the synthetic 'auto' route.

        Ownership is re-asserted, not just checked for absence: if a provider
        ever leaves a route of its own under the reserved id (an older registry,
        a hand-written ``register()`` call), the smart router takes it back
        instead of silently deferring to it forever.
        """
        route = self.routes.get(AUTO_MODEL_ID)
        if route is None or route.provider_name != 'router':
            route = Route(
                model_id=AUTO_MODEL_ID,
                provider_name='router',
                upstream_model='auto',
                vision=True, image_gen=True,
            )
            self.routes[AUTO_MODEL_ID] = route
        return route

    def _provider_auto_route(self, provider_name: str) -> Route:
        """Return (registering on first use) the synthetic '<prefix>/auto'
        route of one provider: a smart router restricted to that provider's
        own discovered models. Registered for every enabled provider so the
        routes survive re-discovery (``_apply_provider_models`` never touches
        router-owned routes)."""
        model_id = provider_auto_id(provider_name)
        route = self.routes.get(model_id)
        if route is None or route.provider_name != 'router':
            route = Route(
                model_id=model_id,
                provider_name='router',
                upstream_model=provider_name,
                vision=True, image_gen=True,
            )
            self.routes[model_id] = route
        return route

    def _auto_routes(self) -> None:
        """(Re-)register every synthetic auto route: the global smart router
        plus one per-provider router. Called at init and after provider
        reloads; re-discovery preserves them."""
        self._auto_route()
        for name in self.providers:
            self._provider_auto_route(name)

    def _auto_chain(self, category: str,
                    thinking_override: Optional[bool] = None,
                    search_override: Optional[bool] = None) -> List[str]:
        """Build the ordered model chain for an 'auto' request.

        Preferred targets for the classified category come first (a provider
        entry expands to every discovered model of that provider), then every
        other discovered model as a safety net. Providers without usable
        credentials (``available() == False``) are demoted out of the front;
        when nothing is healthy the full ordered list is kept — the stream
        loop still probes every target and falls back on auth/rate/offline
        errors. Vision/image-gen requests are restricted to capable targets;
        explicit thinking/search requests to routes supporting the mode.
        The healthy front is round-robin rotated per request, at the
        PROVIDER level: the group order is cycled so every provider gets its
        turn at the head of the chain (a many-model provider such as z.ai GLM
        must not lead every call), and within each provider its own models are
        rotated too so consecutive calls from the same provider land on
        different models. Unhealthy (credential-less) targets stay at the very
        back so the stream loop still probes them.
        """
        ordered: List[str] = []

        def _expand(pref: str) -> None:
            for model_id, route in self.routes.items():
                if route.provider_name == 'router' or model_id in ordered:
                    continue
                if (route.provider_name == pref
                        or route.model_id.startswith(pref + '-')):
                    ordered.append(model_id)

        for pref in AUTO_CATEGORIES.get(category, AUTO_CATEGORIES['general']):
            _expand(pref)
        for model_id, route in self.routes.items():
            if route.provider_name != 'router' and model_id not in ordered:
                ordered.append(model_id)

        for flag, requested in (('thinking_enabled', thinking_override),
                                ('search_enabled', search_override)):
            if requested:
                kept = [mid for mid in ordered
                        if getattr(self.routes[mid], flag)]
                if kept:
                    ordered = kept

        flags = [(mid, self._healthy_model(mid)) for mid in ordered]
        # Healthy targets first (category order preserved); credential-less
        # or offline providers stay at the very back so the stream loop
        # still probes them — credentials can appear at any moment (the
        # renewal bot runs continuously).
        healthy = [mid for mid, ok in flags if ok]
        unhealthy = [mid for mid, ok in flags if not ok]
        # Round-robin the healthy front by PROVIDER, not by individual model.
        #
        # A single provider often exposes far more models than the others
        # (e.g. z.ai GLM lists a dozen models and sits early in every category
        # preference), so rotating the flat model list just cycled through one
        # provider's models in a row — the first ~N calls always led with GLM,
        # which was reported as "auto only uses z.ai GLM models". Grouping the
        # healthy targets by provider and rotating the group order gives every
        # provider its turn at the head of the chain; each provider's own model
        # list is still rotated so successive calls from the same provider land
        # on different models of that provider.
        prov_order = list(dict.fromkeys(self.routes[mid].provider_name
                                        for mid in healthy))
        prov_off = self._rr_next(f'auto:{category}', len(prov_order))
        prov_order = prov_order[prov_off:] + prov_order[:prov_off]
        chain: List[str] = []
        for provider in prov_order:
            p_models = [mid for mid in healthy
                        if self.routes[mid].provider_name == provider]
            m_off = self._rr_next(f'auto:{category}:{provider}',
                                  len(p_models))
            chain.extend(p_models[m_off:] + p_models[:m_off])
        chain.extend(unhealthy)
        if category in ('vision', 'image_gen'):
            cap = 'image_gen' if category == 'image_gen' else 'vision'
            capable = [mid for mid in chain if getattr(self.routes[mid], cap)]
            if capable:
                chain = capable
        return [mid for mid in chain
                if self.routes.get(mid) is not None
                and self.routes[mid].provider_name != 'router']

    def _healthy_model(self, model_id: str) -> bool:
        """Live health for one route target: credential state PLUS outcomes.

        ``available()`` only proves credentials EXIST — it cannot tell an
        expired session, a geo-blocked anonymous identity or an auth-walled
        account from a working one. A chain built from that signal alone leads
        with the same broken providers on every request, every request walks
        through them and lands on the one provider that actually answers, which
        is exactly what "auto only uses z.ai GLM models" looked like. So the
        signal is completed with real request outcomes: a provider that just
        failed is demoted out of the healthy front, and one that just served
        counts as healthy even when its credential probe disagrees.
        """
        route = self.routes.get(model_id)
        provider = self.providers.get(route.provider_name) if route else None
        if provider is None:
            return False
        name = route.provider_name
        if self._demoted(name):
            return False
        try:
            ok = bool(provider.available())
        except Exception:  # noqa: BLE001 — a broken probe means unproven
            ok = False
        return ok or self._proven(name)

    def _mark_served(self, provider_name: str) -> None:
        """A provider produced a first token: it is proven working."""
        if AUTO_PROVE_S <= 0:
            return
        with self._rt_lock:
            self._rt_ok[provider_name] = time.time()

    def _mark_failed(self, provider_name: str, reason: str = '') -> None:
        """A provider failed a request: demote it out of the healthy front.

        Only the auto chains are affected, and only for AUTO_DEMOTE_S seconds:
        the demoted targets stay at the very back of the chain, so they are
        still probed and climb back on their own as soon as the renewal bot
        fixes their credentials (or the demotion simply expires).
        """
        if AUTO_DEMOTE_S <= 0:
            return
        fresh = False
        with self._rt_lock:
            if self._rt_fail.get(provider_name, 0.0) <= time.time():
                fresh = True
            self._rt_fail[provider_name] = time.time() + AUTO_DEMOTE_S
        if fresh:
            logger.info('%s demoted out of the auto chain front for %.0fs (%s)',
                        provider_name, AUTO_DEMOTE_S, reason or 'request failure')

    def _demoted(self, provider_name: str) -> bool:
        """True while a recent failure keeps this provider out of the front."""
        now = time.time()
        with self._rt_lock:
            until = self._rt_fail.get(provider_name, 0.0)
            if until > now:
                return True
            if until:
                del self._rt_fail[provider_name]  # expired: forget it
            return False

    def _proven(self, provider_name: str) -> bool:
        """True when this provider served a request very recently."""
        if AUTO_PROVE_S <= 0:
            return False
        with self._rt_lock:
            since = self._rt_ok.get(provider_name, 0.0)
        return bool(since) and (time.time() - since) < AUTO_PROVE_S

    def _provider_chain(self, provider_name: str,
                        thinking_override: Optional[bool] = None,
                        search_override: Optional[bool] = None) -> List[str]:
        """Build the ordered chain for a '<prefix>/auto' provider router.

        Every discovered model of that single provider, healthy targets
        first (the rest stay at the back so the stream loop still probes
        them — credentials can appear at any moment). Explicit thinking/
        search requests restrict the chain to routes supporting the mode.
        The healthy front of the chain is round-robin rotated per request:
        successive calls cycle through every healthy model.
        """
        ordered: List[str] = [model_id for model_id, route in self.routes.items()
                              if route.provider_name == provider_name]
        for flag, requested in (('thinking_enabled', thinking_override),
                                ('search_enabled', search_override)):
            if requested:
                kept = [mid for mid in ordered
                        if getattr(self.routes[mid], flag)]
                if kept:
                    ordered = kept
        flags = [(mid, self._healthy_model(mid)) for mid in ordered]
        healthy = [mid for mid, ok in flags if ok]
        unhealthy = [mid for mid, ok in flags if not ok]
        offset = self._rr_next(f'pauto:{provider_name}', len(healthy))
        return healthy[offset:] + healthy[:offset] + unhealthy

    def _rr_next(self, key: str, n: int) -> int:
        """Next round-robin offset (0..n-1) for a router key.

        Returns 0 when there is nothing to rotate (no/one healthy target);
        the cursor keeps growing under its own lock so concurrent requests
        never share the same offset.
        """
        if n <= 1:
            return 0
        with self._rr_lock:
            pos = self._rr.get(key, 0)
            self._rr[key] = pos + 1
        return pos % n

    def register(self, route: Route) -> None:
        """Add/replace a route (used by tests and custom setups)."""
        self.routes[route.model_id] = route

    def reload_providers(self) -> bool:
        """Re-import provider modules and rebuild every provider instance.

        Used by the self-healing layer after patching a provider module on
        disk (upstream web-app changed). Thread-safe: swaps the instances
        under the lock, then re-bootstraps DeepSeek's config-derived routes
        and forces a full re-discovery of the web providers.
        """
        with self._lock:
            new_providers = {}
            for name, module_name, class_name in PROVIDER_MODULES:
                module = importlib.import_module(module_name, __package__)
                importlib.reload(module)
                new_providers[name] = getattr(module, class_name)()
            self.providers = new_providers
            self._refreshed_at = 0.0
        changed = False
        try:
            changed = self._apply_provider_models(
                'deepseek', self.providers['deepseek'].list_models())
            self._apply_fallbacks()
        except Exception as e:  # pragma: no cover - defensive
            logger.warning('deepseek route bootstrap failed after reload: %s', e)
        self._auto_routes()
        threading.Thread(target=self.refresh_models, kwargs={'force': True},
                         name='model-re-discovery', daemon=True).start()
        return changed

    # ------------------------------------------------------------- inspection
    def resolve(self, model_id: str, auth_key: Optional[str] = None) -> Route:
        """Resolve a model id to its route.

        Unknown ids trigger a best-effort re-discovery (the upstream may have
        added models since the last refresh); ids that are still unknown fall
        back to the default fast route, so clients sending an arbitrary name
        still get served. An empty id or ``'auto'`` selects the smart router:
        its serving chain is built per request from live provider state.
        """
        model_id = (model_id or '').strip().lower()
        if not model_id or model_id == AUTO_MODEL_ID:
            return self._auto_route()
        route = self.routes.get(model_id)
        if route is not None:
            return route
        # '<prefix>/auto' or '<provider>/auto' selects the provider-scoped
        # smart router (deepseek/auto, z.ai/auto, alibaba/auto, glm/auto…).
        if model_id.endswith('/auto'):
            prefix = model_id[:-len('/auto')]
            name = _PREFIX_TO_PROVIDER.get(prefix, prefix)
            if name in self.providers:
                return self._provider_auto_route(name)
        route = self._resolve_alias(model_id)
        if route is not None:
            return route
        try:
            self.refresh_models(auth_key=auth_key)
        except Exception as e:  # pragma: no cover - defensive
            logger.warning('re-discovery on unknown model failed: %s', e)
        route = self.routes.get(model_id) or self._resolve_alias(model_id)
        if route is not None:
            return route
        fast_id = os.getenv('I4F_MODEL_FAST', 'deepseek-chat').strip()
        return self.routes.get(fast_id) or next(iter(self.routes.values()))

    def _resolve_alias(self, model_id: str) -> Optional[Route]:
        """Map a provider-prefixed public id (deepseek/deepseek-chat,
        z.ai/glm-4.7, …) back to its internal route."""
        for aliases in self._aliases.values():
            mid = aliases.get(model_id)
            if mid:
                route = self.routes.get(mid)
                if route is not None:
                    return route
        return None

    def available(self, route: Route, auth_key: Optional[str] = None) -> bool:
        provider = self.providers.get(route.provider_name)
        return bool(provider and provider.available(auth_key))

    def list_models(self) -> List[Dict[str, Any]]:
        """OpenAI-style /v1/models payload with agent-tooling metadata.

        Every id is provider-prefixed (deepseek/deepseek-chat, z.ai/glm-4.7,
        alibaba/qwen-3-max, …); resolve() accepts both
        the prefixed and the bare internal form.
        """
        auto = self.routes.get(AUTO_MODEL_ID)
        entries: List[Dict[str, Any]] = []

        def _pub(mid: str) -> str:
            r = self.routes.get(mid)
            return public_model_id(r.provider_name, mid) if r else mid

        if auto is not None:
            # Listed first: the smart router handles every capability (it
            # re-routes to a capable model at serve time), so clients must
            # not pre-gate vision/image requests on its behalf.
            entries.append({
                'id': auto.model_id,
                'object': 'model',
                'created': 1700000000,
                'owned_by': 'inference4free',
                'context_length': 131072,
                'max_model_len': 131072,
                'max_completion_tokens': 32768,
                'max_tokens': 32768,
                'thinking_enabled': True,
                'search_enabled': True,
                'vision': True,
                'image_gen': True,
                'fallbacks': [],
            })
        # Per-provider smart routers (deepseek/auto, z.ai/auto, …): listed
        # right after the global auto. Capability flags are the union of the
        # provider's discovered models (all-True before discovery completes
        # so clients do not pre-gate — the router filters at serve time).
        for r in self.routes.values():
            if r.provider_name != 'router' or r.model_id == AUTO_MODEL_ID:
                continue
            siblings = [s for s in self.routes.values()
                        if s.provider_name == r.upstream_model]
            entries.append({
                'id': r.model_id,
                'object': 'model',
                'created': 1700000000,
                'owned_by': OWNED_BY.get(r.upstream_model, r.upstream_model),
                'context_length': 131072,
                'max_model_len': 131072,
                'max_completion_tokens': 32768,
                'max_tokens': 32768,
                'thinking_enabled': (not siblings
                                     or any(s.thinking_enabled for s in siblings)),
                'search_enabled': (not siblings
                                   or any(s.search_enabled for s in siblings)),
                'vision': (not siblings or any(s.vision for s in siblings)),
                'image_gen': (not siblings
                              or any(s.image_gen for s in siblings)),
                'fallbacks': [],
            })
        entries.extend(
            {
                'id': _pub(r.model_id),
                'object': 'model',
                'created': 1700000000,
                'owned_by': OWNED_BY.get(r.provider_name,
                                         r.provider_name),
                'context_length': r.context_length,
                'max_model_len': r.context_length,
                'max_completion_tokens': r.max_output_tokens,
                'max_tokens': r.max_output_tokens,
                'thinking_enabled': r.thinking_enabled,
                'search_enabled': r.search_enabled,
                'vision': r.vision,
                'image_gen': r.image_gen,
                'fallbacks': [_pub(f) for f in r.fallbacks],
            }
            for r in self.routes.values() if r.provider_name != 'router'
        )
        return entries

    # ---------------------------------------------------------------- serving
    def stream(self, route: Route, prompt: str, *, temperature: Optional[float] = None,
               max_tokens: Optional[int] = None, auth_key: Optional[str] = None,
               thinking_override: Optional[bool] = None,
               search_override: Optional[bool] = None,
               images: Optional[List[Dict[str, Any]]] = None,
               image_generation: bool = False,
               no_proxy: bool = False,
               ) -> Generator[Dict[str, Any], None, None]:
        """Yield unified chunks, retrying rate limits/network errors and
        falling back through the route's chain when a provider keeps failing.

        Fallbacks happen on request-level failures. Errors raised mid-stream
        (after content was already emitted) are surfaced as-is to avoid
        duplicating partial output.

        ``images``/``image_generation`` restrict the fallback chain to
        vision/image-gen capable targets and are forwarded to the provider.
        """
        thinking = route.thinking_enabled if thinking_override is None else thinking_override
        search = route.search_enabled if search_override is None else search_override

        if route.provider_name == 'router':
            if route.model_id == AUTO_MODEL_ID:
                # Global smart router: classify the request and build the
                # chain from live provider state (preferred category models
                # first, the rest as safety net). The loop below still
                # handles rate limits, auth failures (with inline renewal),
                # outages and blocks.
                category = classify_request(prompt, bool(images),
                                            image_generation)
                chain = self._auto_chain(category, thinking_override,
                                         search_override)
                if not chain:
                    raise ProviderError(
                        'auto router found no available model — providers '
                        'are still discovering or credentials are renewed')
                logger.info('auto router: category=%s chain=%s', category,
                            ' -> '.join(chain[:5])
                            + ('…' if len(chain) > 5 else ''))
            else:
                # Provider-scoped smart router ('<prefix>/auto'): the same
                # serve-time machinery restricted to one provider's models.
                chain = self._provider_chain(route.upstream_model,
                                             thinking_override,
                                             search_override)
                if not chain:
                    raise ProviderError(
                        f'{route.model_id}: provider {route.upstream_model!r} '
                        'has no discovered models yet — still discovering or '
                        'credentials are being renewed')
                logger.info('%s: chain=%s', route.model_id,
                            ' -> '.join(chain[:5])
                            + ('…' if len(chain) > 5 else ''))
        else:
            chain = [route.model_id] + [f for f in route.fallbacks
                                        if f != route.model_id]
        last_error: Optional[ProviderError] = None

        if images or image_generation:
            # Vision/image-gen requests may only be served by capable targets.
            capability = 'image_gen' if image_generation else 'vision'
            capable = [mid for mid in chain
                       if (target := self.routes.get(mid)) and getattr(target, capability)]
            if not capable:
                what = 'image generation' if image_generation else 'vision'
                raise ProviderError(
                    f'No model in the fallback chain of {route.model_id} '
                    f'supports {what} (vision/image-capable providers: '
                    f'chatgpt, gemini — they need credentials; the bot '
                    f'creates them automatically when possible)')
            chain = capable

        for position, model_id in enumerate(chain):
            target = self.routes.get(model_id)
            # Router-owned synthetic targets can only enter a chain via a
            # hand-written I4F_FALLBACKS entry — they must never be served
            # directly (their provider is the router itself).
            if target is None or target.provider_name == 'router':
                continue
            provider = self.providers.get(target.provider_name)
            if provider is None:
                continue
            served_by = f'{target.provider_name}/{target.upstream_model}'
            stall_key = f'{target.provider_name}/{target.upstream_model}'
            stall_until = getattr(self, '_stall_until', None) or {}
            if stall_until.get(stall_key, 0) > time.time():
                # stalled moments ago: skip instead of re-waiting the full
                # first-token deadline (the last chain target is still tried)
                if position < len(chain) - 1:
                    logger.info('skipping %s: stalled recently (cooldown)',
                                served_by)
                    continue
            attempt = 0
            while True:
                attempt += 1
                emitted = False
                try:
                    gen = provider.stream(
                        prompt, model=target.upstream_model,
                        thinking_enabled=thinking, search_enabled=search,
                        temperature=temperature, max_tokens=max_tokens,
                        images=images, image_generation=image_generation,
                        no_proxy=no_proxy,
                        auth_key=auth_key,
                    )
                    if FIRST_TOKEN_TIMEOUT > 0:
                        # stall watchdog: a provider that connects but never
                        # yields is retried/fallen back, never hung-up-on
                        first = _first_chunk(gen, FIRST_TOKEN_TIMEOUT)
                        if first is not None:
                            emitted = True
                            try:
                                self._stall_until.pop(stall_key, None)
                            except Exception:  # noqa: BLE001 — bookkeeping
                                pass
                            self._mark_served(target.provider_name)
                            if isinstance(first, dict):
                                first.setdefault('served_by', served_by)
                            yield first
                    for chunk in gen:
                        if not emitted:  # first chunk: target recovered
                            emitted = True
                            try:
                                self._stall_until.pop(stall_key, None)
                            except Exception:  # noqa: BLE001 — bookkeeping
                                pass
                            self._mark_served(target.provider_name)
                        if isinstance(chunk, dict):
                            chunk.setdefault('served_by', served_by)
                        yield chunk
                    return
                except ProviderRateLimitError as e:
                    last_error = e
                    if emitted:
                        raise  # mid-stream failure: fallback would duplicate output
                    # Deliberately NOT demoted: a quota hit means "busy now", not
                    # "broken", and the retry ladder plus the round-robin chain
                    # already spread the load. Demoting here would starve the
                    # providers that are actually healthy.
                    if attempt <= MAX_RETRIES:
                        wait = min(e.retry_after if e.retry_after
                                   else RETRY_BACKOFF * (2 ** (attempt - 1)), RETRY_CAP)
                        logger.warning('%s rate limited (attempt %d/%d), retrying in %.1fs: %s',
                                       served_by, attempt, MAX_RETRIES + 1, wait, e)
                        time.sleep(wait)
                        continue
                    logger.warning('%s rate limited after %d attempts: %s',
                                   served_by, attempt - 1, e)
                    break
                except FirstTokenTimeoutError as e:
                    # a stalled upstream does not recover within seconds:
                    # skip the retry ladder, fall back right away and put the
                    # target on a stall cooldown so later chain positions are
                    # not re-waited either
                    last_error = e
                    if emitted:
                        raise  # mid-stream failure: fallback would duplicate output
                    try:
                        self._stall_until[stall_key] = (time.time()
                                                        + PROVIDER_STALL_COOLDOWN)
                    except Exception:  # noqa: BLE001 — bookkeeping only
                        pass
                    self._mark_failed(target.provider_name, 'first-token stall')
                    logger.warning('%s stalled without first token, skipping '
                                   'to fallback: %s', served_by, e)
                    break
                except ProviderUnavailableError as e:
                    last_error = e
                    if emitted:
                        raise  # mid-stream failure: fallback would duplicate output
                    if attempt <= MAX_RETRIES:
                        wait = min(RETRY_BACKOFF * (2 ** (attempt - 1)), RETRY_CAP)
                        logger.warning('%s unavailable (attempt %d/%d), retrying in %.1fs: %s',
                                       served_by, attempt, MAX_RETRIES + 1, wait, e)
                        time.sleep(wait)
                        continue
                    logger.warning('%s unavailable after %d attempts: %s',
                                   served_by, attempt - 1, e)
                    self._mark_failed(target.provider_name, 'unreachable')
                    break
                except ProviderAuthError as e:
                    # Credentials rejected/missing: retrying cannot help.
                    last_error = e
                    if emitted:
                        raise  # mid-stream failure: fallback would duplicate output
                    logger.warning('%s auth failed, skipping to fallback: %s', served_by, e)
                    # Credentials are provider-wide, so the whole provider is
                    # demoted: without this every auto request re-probes the
                    # same auth-walled provider from the front of the chain and
                    # ends up served by whichever provider happens to work.
                    self._mark_failed(target.provider_name, 'auth wall')
                    # Request-path remediation: fire a background renewal
                    # ladder so the NEXT request can use fresh credentials.
                    try:
                        from dsk import refresher as _refresher
                        _refresher.renew_inline(target.provider_name, str(e)[:120])
                    except Exception:  # noqa: BLE001 — never break the request
                        pass
                    break
                except ProviderError as e:
                    last_error = e
                    if emitted:
                        raise  # mid-stream failure: fallback would duplicate output
                    logger.warning('%s failed, skipping to fallback: %s', served_by, e)
                    self._mark_failed(target.provider_name, str(e)[:80])
                    break
                except Exception as e:  # noqa: BLE001 — unclassified provider
                    # crash (upstream format change, provider bug): keep the
                    # request alive by falling through the chain. Mid-stream
                    # failures still re-raise: output was already emitted.
                    if emitted:
                        raise
                    last_error = ProviderError(f'{type(e).__name__}: {e}')
                    logger.warning('%s crashed, skipping to fallback: %s',
                                   served_by, e)
                    self._mark_failed(target.provider_name,
                                      f'{type(e).__name__}')
                    break

            if position < len(chain) - 1:
                logger.info('falling back: %s -> %s', route.model_id, chain[position + 1])

        if last_error is not None:
            raise last_error
        raise ProviderError(f'No provider available for model {route.model_id}')
