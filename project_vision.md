# project_vision.md — Inference4Free

> **What it is, how it works, and what it MUST be.**
> Vision document, updated 2026-10-09, derived from a complete analysis of the
> code (≈31,000 lines in `dsk/`, 29 registered providers, 21 test suites).

---

## 1. Project identity (what it IS)

**Inference4Free is a self-hosted server that exposes free LLM models — taken
from the providers' chat web apps, their anonymous surfaces and their free API
tiers — behind a standard OpenAI-compatible endpoint, in a fully autonomous
way: the service creates accounts by itself, renews sessions by itself, and
repairs its own code when an upstream site changes.**

It is a fork of [`xtekky/deepseek4free`](https://github.com/xtekky/deepseek4free)
(from which it inherits the reverse-engineered DeepSeek client, the
proof-of-work WASM and the Cloudflare bypass), turned into something much
bigger: a **multi-provider aggregator with intelligent routing**, in the same
conceptual family as gpt4free — free access to frontier models without paid
keys — but with three ambitions gpt4free does not have:

1. **Total OpenAI fidelity** — any client that speaks the OpenAI protocol
   (aider, AiderDesk, OpenWebUI, LiteLLM, LibreChat, the `openai` SDK) must
   work without knowing it: SSE streaming, `reasoning_content`, tool calling,
   vision, image generation, errors in the OpenAI shape.
2. **Zero human credentials** — no token to paste, no cookie to copy. The
   credential bot (`dsk/refresher.py`) generates throwaway mailboxes, fills in
   the signup forms, reads the OTPs and deposits the sessions in `./data/`,
   renewing them autonomously.
3. **Self-maintenance** — when an upstream provider changes its API, the
   module is probed, the breakage classified and (for structural breakages) the
   provider code is **rewritten by an LLM, validated in a subprocess and
   hot-reloaded** (`dsk/selfheal.py`), with backups and automatic rollback.

```
┌──────────────────┐   OpenAI protocol     ┌─────────────────────────────┐   heterogeneous transports ┌────────────────────────┐
│ aider/AiderDesk, │  /v1/chat/completions │   dsk/openai_server.py      │  reverse-engineered HTTP   │ provider chat web apps │
│ openai SDK,      │  /v1/models           │   FastAPI + SSE + tool-call │  browser relays (real UI)  │ free API tiers         │
│ OpenWebUI, etc.  │  /v1/images/...       │   + playground + selfheal   │  free API keys             │ anonymous surfaces     │
└──────────────────┘ ◄───────────────────── └─────────────────────────────┘ ◄──────────────────────  └────────────────────────┘
```

---

## 2. Current architecture (what is there, file by file)

### 2.1 API layer — `dsk/openai_server.py` (≈2,000 lines)

FastAPI, entrypoint `python -m dsk.openai_server`. It exposes:

| Endpoint | Function |
|---|---|
| `GET /v1/models` | Dynamic registry discovered from live sessions; the router ids (`auto`, `auto-fast`, `auto-thinking` — global and per provider); `thinking_enabled` / `search_enabled` / `vision` / `image_gen` capability flags; tool-less models hidden by toolprobe (router entries never hidden — their `tools` flag mirrors their pool) |
| `POST /v1/chat/completions` | Streaming + non-streaming; multi-protocol tool calling; vision via `image_url` content parts; extra parameters `thinking`, `search_enabled`, `disable_proxy` |
| `POST /v1/images/generations` | OpenAI Images API (`url` / `b64_json`), reserved to models with `image_gen` |
| `GET /health` | Proxy, selfheal, refresher, toolprobe state (liveness + diagnostics) |
| `GET /selfheal/status`, `POST /selfheal/probe`, `POST /selfheal/refresh` | Control of the automatic maintenance |
| `GET /providers`, `GET /providers/{name}`, `POST /providers/{name}/test|renew`, `POST /providers/{name}/credentials[/clear]`, `POST /providers/renew-all` | Live provider and credential management |
| `GET /toolcall/status`, `POST /toolcall/reprobe` | State and re-probe of the tool-calling capability per model |
| `GET /`, `/playground` | llama.cpp-style web playground (`dsk/static/index.html`): streaming chat, thinking panel, inline images, 24 h saved chats |

The startup lifecycle launches in separate threads: model discovery, the
selfheal daemon, the refresher daemon, the tool-calling prober and the z.ai
browser session pre-warm. Client authentication is optional (`I4F_API_KEY`);
errors always follow the OpenAI shape `{"error": {message, type, param, code}}`.

**Chat request pipeline:**
`LLM CALL → llmtrim (history trimmed to the route's context budget) → router (classification + fallback chain) → proxies (rotating egress) → provider → OpenAI SSE bridge` — with tool-call emulation in the middle.

### 2.2 Routing — `dsk/providers/router.py` (≈1,690 lines)

The heart of the aggregation. Prime rule: **no model is hardcoded** — every
route is born from the provider's `list_models()` on the live session, with a
TTL (`I4F_MODELS_TTL`, 300 s) and on-demand re-discovery on unknown ids. A
provider that fails discovery keeps its already-known routes.

- **Registry of 29 providers** (`PROVIDER_MODULES`, lazy imports: a broken
  module disables only itself).
- **Public namespacing**: every model on `/v1/models` carries the provider
  prefix (`deepseek/deepseek-chat`, `z.ai/glm-4.7`, `alibaba/qwen-3-max`, …);
  internal fallback chains use bare ids, `resolve()` maps them back.
- **`auto` smart router** (the default model when the client omits `model`):
  - regex-only heuristic classification of the request into
    `image_gen | vision | translation | summarize | coding | general`;
  - per-category preference tables (`AUTO_CATEGORIES`: e.g. `image_gen`
    leads with pollinations, `vision` with chatgpt, `coding` with deepseek);
  - every category still includes *all* the other discovered models as a
    safety net;
  - orders the "healthy front" by **measured latency** (EWMA of the first
    token, fast-tier = within 2.5× of the fastest + 5 s), round-robin rotation
    **at provider level** and at model level, so rotation is fair and never
    dominated by a single vendor;
  - runtime state: demotion after failure (`I4F_AUTO_DEMOTE_S`), "proven"
    after success (`I4F_AUTO_PROVE_S`), quota cooldown on rate limits, stall
    cooldown on first-token timeouts.
- **Router family** — three synthetic router kinds, each global AND
  per-provider (`<prefix>/<kind>`, registered for each of the 29):
  - **`auto` / `<provider>/auto`**: the smart router (and default model when
    the client omits `model`) — serves every discovered model;
  - **`auto-fast` / `<provider>/auto-fast`**: strict pool of the models that
    can **never** think (metadata `thinking_enabled` False — no thinking
    capability or no way to activate it; web-search modes excluded too),
    streamed with thinking forced **off**: zero `reasoning_content`, by
    contract;
  - **`auto-thinking` / `<provider>/auto-thinking`**: strict pool of the
    thinking-capable models (always-on ones and models callable with the
    thinking flag active), streamed with thinking forced **on**;
  - the pools are a disjoint partition of the registry (`_filter_pool`,
    strict: an empty pool is a clean `ProviderError`, never relaxed into a
    contract violation); the per-request `thinking` body flag is ignored on
    the two pool routers — the router id IS the contract;
  - each pool rotates on its own round-robin/tier cursors
    (`auto-fast:{category}:{provider}`, `pauto:auto-fast:{provider}`, …);
  - `/v1/models` advertises each router's contract flags and unions the
    capability flags over its own pool's models; the toolprobe `tools` flag
    on a router entry mirrors the same pool, and `is_router_model_id`
    (shared with the `/v1` gates) keeps router ids out of the leaf probes;
  - the metadata invariant that keeps the pools honest:
    `thinking_enabled = True` ⇔ the route can/will be called with thinking
    active (the thinking itself may surface as separate reasoning frames or
    be folded into the answer by the transport — the flag says the route is
    CALLED thinking-active). Every provider's `list_models()` must respect
    it (arena was fixed for this — it hardcoded False while some of its
    models stream reasoning frames); new providers are audited against it
    before registration.
- **Fallback and retry**: per-provider retries with exponential backoff
  (honoring `Retry-After`, cap `I4F_RETRY_CAP`), then the chain is walked;
  mid-stream failures do not re-emit content; on `auth` the credential renewal
  is triggered inline; first-token timeout (`I4F_FIRST_TOKEN_TIMEOUT`) with
  stall cooldown, plus a mid-stream **silence guard**
  (`I4F_STREAM_SILENCE_TIMEOUT`, 180 s) that aborts and falls back when a
  stream goes quiet after it started.

### 2.3 The providers — `dsk/providers/` (29, in four classes)

Common contract (`base.py`): every provider inherits `Provider` and produces a
generator of chunks `{'content': str, 'type': 'text'|'thinking', 'finish_reason': …}`,
the same shape as the original DeepSeek stream, so the server consumes all
providers through a single code path. `base.py` also provides the typed error
taxonomy (`ProviderAuthError`, `ProviderRateLimitError` (with `Retry-After`),
`ProviderUnavailableError`, `FirstTokenTimeoutError`), the
`classify_http_error` classification, resilient HTTP clients (retry, proxy,
curl_cffi), image helpers (data-URI, download, MIME sniffing, dimensions) and
the cookie jars.

| Class | Providers | Credentials |
|---|---|---|
| **Reverse-engineered web** | `deepseek` (token + PoW WASM + CF bypass), `gemini` (1PSID cookies, batchexecute RPC, vision via multipart upload to `content-push.googleapis.com`), `chatgpt` (backend-api / browser relay), `claude` (sessionKey, temporary conversations), `grok` (sso cookie, images via Aurora), `mistral` (Le Chat, session token), `qwen` (bearer + bx-ua fingerprint / guest relay), `kimi` (token cookie, gRPC-web frames) | created/renewed by the bot |
| **Anonymous / keyless** | `duck` (duck.ai `duckchat/v1`, Node.js challenge solver `duck_solver.js`), `pollinations` (text + image, live catalogs), `glm` (anonymous z.ai + signed chatglm.cn; Chromium browser transport), `copilot` (websocket, harvested anonymous identity), `perplexity` (SSE ask, Sonar backend) | none (always available) |
| **Reverse-eng. dormant** | `arena` (ex-LMArena, login wall + reCAPTCHA v3), `huggingchat` (cookie jar; public catalog), `t3chat` (free LLM-chat web app), `innerai`, `adapta` (login workspaces) | dormant until a session exists |
| **Free API tiers** | `groq`, `cerebras`, `modelscope`, `mistral_api`, `openrouter` (`:free` models), `llm7` (free dash key, bot-automated signup), `google_ai_studio`, `cohere`, `cloudflare` (Workers AI: token + account id), `meta` (Muse Spark), `blackbox` | free API key, verified by the refresh rung without burning quota |

**Browser relays** (`dsk/chatgpt_relay.py`, `dsk/qwen_relay.py`): where the
HTTP transport is walled (ChatGPT 2026 Sentinel/Turnstile, Qwen Aliyun WAF),
the provider drives the **real UI** in a shared Chromium (DrissionPage), hooks
the SPA's stream and diffs the reply into chunks — used as the anonymous
surface when credentials are missing.

### 2.4 Autonomous credentials — `dsk/refresher.py` (≈5,800 lines) + `dsk/mailgen.py`

The renewal ladder, from cheapest to most invasive, per provider:

1. **HTTP refresh** of the bot-managed jars in `./data/` (Gemini's 1PSIDTS
   rotation, ChatGPT's session→accessToken, live DeepSeek probe);
2. **headless re-login** with the accounts created by the bot
   (`data/accounts.json`) in the shared Chromium;
3. **full auto-signup**: a mailbox is generated on the fly (7 backends in
   `mailgen.py`: IMAP catch-all → emailnator (real gmail addresses) →
   tempmail.lol → tempmail.plus → temp-mail.io → Guerrilla → mail.tm/mail.gw),
   the form is filled in (CMP/consent interstitials — including the EU consent
   wall — are dismissed automatically), the OTP is read automatically.

Proactive daemon every `I4F_REFRESHER_TTL` (6 h) + on-demand trigger from
selfheal on `auth` outcomes. Per-provider cooldowns, daily budget, per-rung
circuit breaker (`I4F_BREAKER_N`), egress rotation for walled anonymous
providers (`I4F_EGRESS_ROTATE`) and a dedicated signup egress
(`I4F_SIGNUP_PROXY`, residential recommended: datacenter IPs are often
blocked by the signup walls), JSONL audit in `data/refresher/history.jsonl`.
CLI:
`status | refresh | login | signup | bootstrap | mailgen`. Bot files in
`./data/` **win** over env vars; `.env` never contains LLM credentials.

### 2.5 Self-maintenance — `dsk/selfheal.py` (≈960 lines)

Cycle: **PROBE** (600 s TTL, cheap calls) → classification
`ok | network | rate | auth | structural` → after ≥3 consecutive structural
failures: **HEAL** = evidence collection (module source, recent errors, live
JS bundles from the site) → an LLM fixer (custom OpenAI-compatible endpoint or
this very server via loopback) rewrites the module → validation in a
subprocess (real import + real probe against the site) → replacement on disk
with backup, hot-reload, re-probe; automatic rollback at the first problem.
File whitelist, attempt/incident caps, JSONL audit, `I4F_SELFHEAL=false` to
disable. **The LLM fixer never sees credentials.**

Probes are **honest per provider** (`_probe_once`): DeepSeek gets a real
completion through its PoW pipeline, qwen a `validate_token` call, ChatGPT the
sentinel gate, gemini real cookies + `list_models`, perplexity a real
completion (its `available()` is unconditional and the anonymous catalog
lists models that are auth-walled — the generic JS-bundle evidence fallback
would report a misleading `ok`); only providers without a dedicated probe
fall back to bundle-evidence grep.

### 2.6 Tool calling for agent coding — `dsk/openai_server.py` + `dsk/toolprobe.py`

The web providers have no native function calling: the server emulates it.

- The flat prompt (the providers accept a single prompt per turn: the OpenAI
  message list is rendered as `[System]/[Assistant]/[Tool result]`) is enriched
  with the tool protocol; **multi-protocol** on the reading side:
  `TOOL_CALL: {json}` (taught, tolerant of mangling),
  DeepSeek's native **DSML** (`<｜DSML｜ invoke>`, recognized and re-rendered in
  history in the same format), **Hermes/Qwen/GLM** `<tool_call{…}>` blocks,
  the **Mistral** `[TOOL_CALLS][…]` marker; JSON parsing with repair.
- `tool_choice: "none"`/forced honored; replies are converted back into real
  OpenAI `tool_calls` with streaming deltas; tool results return into the
  prompt with an explicit anti-loop instruction ("the call is DONE…").
- **`toolprobe`**: periodic canary (`get_weather`) for EVERY model through the
  full real stack; models that cannot call tools are excluded from
  `/v1/models` (re-entering on their own if they recover); real traffic feeds
  the same state (soft-fail only if `tool_choice='required'` produced no
  calls); infrastructure failures do not count. Persistent state in
  `data/toolcall_state.json`.
- **Router-aware annotation** (`annotate_models`): router ids are recognized
  via `is_router_model_id` and never probed or tracked as leaf models (the
  per-model probe endpoint refuses them with a 400); on `/v1/models` a
  router's `tools` flag mirrors its pool — global routers weigh every
  provider's pool leaves, provider-scoped ones only their own, with
  `auto-fast` / `auto-thinking` restricted to the pool partition — `false`
  only when every pool leaf is known-failed, and router entries are never
  hidden by `I4F_HIDE_TOOLLESS`.

### 2.7 Cross-cutting modules

- **`dsk/llmtrim.py`** — ALWAYS-ON inbound payload trimmer: measures the
  history against the route's context budget (per-provider overridable),
  drops the oldest history (system and latest turns always kept), middle-out
  truncates oversized messages. Goal: faster first byte and less per-IP
  rate-limit pressure.
- **`dsk/proxies.py`** (≈760 lines) — fully dynamic proxy pool: static +
  aggregation of public lists (TheSpeedX, monosans, proxifly, proxyscrape,
  roosterkid, geonode — with country slices US/IN/BR/JP) + extra URLs;
  **quality gates** on JSON sources that publish metadata (latency, upTime)
  so entries known to be slow/flaky are pre-filtered, **stratified sampling**
  to `I4F_PROXY_MAX_POOL`; concurrent health check, **only fast proxies**
  (latency budget) get traffic; latency-ranked selection
  (`I4F_PROXY_TOP_K`), **no-proxy as a first-class route**, sticky
  per-provider assignment with TTL rotation, cooldown on failures, never Tor.
- **`dsk/browser.py`** — RAM rule: **one shared Chromium process per
  profile**, consumers are tabs (relays, z.ai, signup rungs, copilot
  harvest), idle reaper (default 180 s), hard instance cap
  (`I4F_BROWSER_MAX_INSTANCES`, default 3) with soft oldest-idle eviction
  (a busy profile is never killed), wedge hygiene (kill-by-port, profile
  lock, zombies, Xvfb).
- **`dsk/pow.py` + `dsk/wasm/`** — DeepSeek proof-of-work solved with **its
  own** sha3 WASM module.
- **`dsk/bypass.py` + `dsk/CloudflareBypasser.py` + `dsk/server.py`** —
  Cloudflare bypass with a real Chromium, `cf_clearance` cache in
  `./data/cookies.json`.
- **Packaging** — `Dockerfile` (python:3.11-slim + Chromium/Xvfb/Node),
  `docker-compose.yml` (healthcheck, log cap, `./data`→`/data` volume,
  env_file), lifecycle managed by systemd on the host.

### 2.8 State, data and tests

- All persistent state lives in `./data/`: per-provider cookie jars,
  `accounts.json`, `toolcall_state.json`, `refresher/history.jsonl`,
  `selfheal/` (backups + audit), `cookies.json` (cf_clearance).
- **Tests**: 21 suites / 293 tests in `tests/` covering the API-key provider
  contract (catalog parsing, dormancy without keys, env activation, stream
  parsing, error frame → exceptions), the auto routers (fair rotation,
  demotion, anti-hijack of the router ids, the `auto-fast`/`auto-thinking`
  pool partition, router-id recognition, pool-aware `/v1/models`
  annotation), credential/jar logic, the ChatGPT relay parsers, the mailgen
  backends. Tests run against fake HTTP, no network.

### 2.9 Configuration surface

~170 `I4F_*` environment variables tune the service. This document describes
behaviors, not the full var reference — the authoritative list lives in
`.env.example` (commented) and the README table. Conventions:

- **Global knobs** `I4F_*`; **per-provider overrides** follow
  `I4F_<PROVIDER>_CONTEXT_LENGTH` / `_MAX_OUTPUT` / `_API_BASE` (the latter
  lets any API-tier provider be repointed at a mirror or compatible gateway);
  **generic cookie fallback** `<PROVIDER>_COOKIES` (JSON jar) for every
  provider; **login seeds** `<PROVIDER>_LOGIN_EMAIL/_PASSWORD` where a signup
  flow exists.
- Behavioral groups not covered elsewhere above: transport-specific
  first-token budgets (`I4F_HTTP_FIRST_TOKEN_TIMEOUT` 60 s for HTTP routes,
  `I4F_IMAGE_FIRST_TOKEN_TIMEOUT` 180 s for image generation, on top of the
  generic `I4F_FIRST_TOKEN_TIMEOUT`), total fallback-chain walk budget
  (`I4F_AUTO_WALK_BUDGET`, 420 s), quota cooldown (`I4F_QUOTA_COOLDOWN_S`),
  unknown-provider probing threshold (`I4F_PROBE_UNKNOWN_ABOVE_S`),
  relay enable/default models (`I4F_CHATGPT_RELAY`,
  `I4F_CHATGPT_RELAY_DEFAULT`, `I4F_QWEN_RELAY`), duck solver knobs
  (`I4F_DUCK_NODE`, `I4F_DUCK_SOLVER_TIMEOUT`, `I4F_DUCK_WARM_TTL`),
  pollinations image size (`I4F_POLLINATIONS_IMAGE_WIDTH/HEIGHT`), inline
  renewal hourly cap (`I4F_REFRESHER_INLINE_HOURLY`), legacy bypass throttle
  (`I4F_BYPASS_THROTTLE_S`), connect/read timeouts
  (`I4F_HTTP_CONNECT_TIMEOUT`, `I4F_PROXY_READ_TIMEOUT`).

---

## 3. Fundamental principles (the rules the project MUST respect)

These are invariants: every new feature and every refactoring must preserve
them.

1. **No hardcoded models.** Catalogs are discovered from live sessions; the
   `AUTO_CATEGORIES` tables list *providers*, never model ids: a new upstream
   model must appear on `/v1/models` without touching the code.
2. **No mandatory human credentials.** Zero-credential is the normal state:
   keyless when possible, bot-autonomous elsewhere, manual override only as an
   option. `.env` never contains LLM credentials; credentials live only in the
   `./data` volume and bot files win over env vars.
3. **A broken provider must never degrade the others.** Lazy imports,
   tolerant discovery, fallback chains, cooldowns: failure isolation is an
   architectural requirement, not an optimization.
4. **OpenAI fidelity as the external contract.** Any standard field must be
   accepted (ignored if not supportable, never a 422), any error in the OpenAI
   shape, streaming always with `finish_reason` and `data: [DONE]`, real tool
   calling for the agent. The client agent must be able to trust it.
5. **Operational autonomy.** Credential renewal, signup, provider re-patching,
   capability probing and proxy health must run unattended for weeks; every
   action is tracked (JSONL) and can be turned off via env.
6. **One Chromium only.** Every new browser consumer goes through
   `dsk/browser.py` (tabs, not processes).
7. **Fast first byte.** Fast-only proxies, latency-ranked routing, llmtrim,
   first-token timeout with immediate fallback: the user-perceived latency is a
   design criterion on a par with correctness.
8. **Defensive security.** Keys/cookies never leave the data volume nor end up
   in prompts (not even in the selfheal fixer); public proxies are
   opportunistic (end-to-end TLS for HTTPS, exclude-list for sensitive
   providers); never Tor.
9. **Everything switchable via env, with sane defaults.** `I4F_PROVIDERS`,
   `I4F_SELFHEAL`, `I4F_REFRESHER*`, `I4F_LLMTRIM`, `I4F_PROXY*`, …: every
   subsystem has a switch and no default requires configuration.
10. **Router ids and pool contracts are public API.** `auto`, `auto-fast`,
    `auto-thinking` and every `<prefix>/<kind>` variant are reserved and
    their thinking contract is guaranteed by construction: the metadata
    invariant `thinking_enabled = True` ⇔ the route can stream thinking
    holds for every provider catalog (audited at registration), the pools
    are filtered strictly and never relaxed, and no per-request flag can
    break the router id's promise.

---

## 4. What it MUST be (vision)

**Inference4Free must be THE reference free, self-hosted OpenAI endpoint for
agent coding and for everyday use: you start it once and forever; the catalog
updates by itself, accounts are created, die and are re-created by themselves,
providers repair themselves, and the client never sees any of this — it only
sees a reliable OpenAI endpoint.**

Concretely, the project must keep growing along five directions, all already
impressed in the code:

1. **Provider coverage = coverage of the web's free value.** Every new free
   surface — reverse-engineerable chat web, anonymous tier, free API key,
   browser relay — is one more provider in the same contract class. The
   integration pattern is established ("Auth model" docstring + catalog/
   dormant/env/stream tests + registration in `PROVIDER_MODULES`): a new
   provider is one file, one registry line, and — if it has a free tier — it
   enters the `auto` tables.
2. **The router family as the main product.** `auto`, `auto-fast`,
   `auto-thinking` — global and per provider — are the most important user
   surface: classification, category preferences, pool contracts, fair and
   latency-ranked rotation must keep being refined (real measurements, not
   theory), because that is where "aggregator" becomes "experience".
3. **Agent coding first-class.** Multi-protocol tool calling, per-model
   capability probing, `reasoning_content`, honest published context limits:
   any agent framework must work on the first try. Every new tool-call
   protocol observed in the wild enters the parsers.
4. **Resilience as identity.** Selfheal (validated LLM auto-patch), refresher
   (3-rung renewal ladder), toolprobe (dynamic capability), proxy health
   (fast-only), fallback chains (every request served by the best alive
   backend): the service must survive the adversarial nature of its upstreams
   — sites that change, accounts that die, proxies that rot — without human
   intervention.
5. **Resource honesty.** One browser, minimal payloads, quota consumed only
   when needed, credential checks that do not burn quota, sampled proxy pool:
   the project runs on modest hardware and on free tiers, and every waste is a
   design failure.

### Non-goals (what it is NOT and must not become)

- **It is not a paid-API proxy** and never uses official paid APIs; it only
  leverages free surfaces (chat web apps, anonymous tiers, free API tiers).
- **It is not a multi-tenant/SaaS service**: self-hosted, one operator, one
  data volume.
- **It is not an abuse tool**: personal/educational use, upstream ToS at the
  operator's own risk, no paywall bypass, no resale.
- **It never embeds credentials in code, logs or prompts.**
- **It never uses Tor** on any path.

---

## 5. Reference technical contracts (must not be broken)

- **Provider stream chunk**: `{'content': str, 'type': 'text'|'thinking',
  'finish_reason': None|'stop'}` — every new provider speaks this language.
- **`Route`** (`base.py`): internal id, provider, upstream model, flags
  `thinking_enabled` / `search_enabled` / `vision` / `image_gen` / tool,
  `context_length` / `max_output`, fallback chain.
- **Error taxonomy**: `ProviderAuthError` → refresher; `ProviderRateLimitError`
  (with `Retry-After`) → retry + quota cooldown; `ProviderUnavailableError` /
  `FirstTokenTimeoutError` → fallback; generic `ProviderError` → chain.
- **Endpoint shape**: provider-prefixed models on `/v1/models`; chat body with
  tolerated extras (`thinking`, `search_enabled`, `disable_proxy`); errors in
  OpenAI shape.
- **Router ids are owned by the Router**: `auto`, `auto-fast`,
  `auto-thinking` and every `<prefix>/<kind>` variant are reserved synthetic
  routes (`_reserved_ids`); an upstream model with a colliding id is
  namespaced, never allowed to hijack the router. The pool routers have a
  hard contract: `auto-fast` never streams thinking, `auto-thinking` always
  does — the per-request `thinking` flag is ignored on them.
- **Provider registration**: tuple `(name, module, class)` in
  `PROVIDER_MODULES` (router) + public prefix in `PUBLIC_PREFIX` +
  ownership in `OWNED_BY`; lazy imports mandatory.
- **On-disk state**: only in `./data/`; no useful state inside the ephemeral
  container or the repo.

---

## 6. Complete module map

| Path | Role |
|---|---|
| `dsk/openai_server.py` | OpenAI-compatible server, multi-protocol tool-call emulation, playground, management endpoints |
| `dsk/providers/router.py` | Dynamic 29-provider registry, router family (`auto` / `auto-fast` / `auto-thinking`, global + per provider), retry/fallback, runtime health |
| `dsk/providers/base.py` | `Provider`/`Route` contract, typed errors, resilient HTTP, images |
| `dsk/providers/jar.py` | Shared bot-managed cookie-jar helpers (lock-safe read/merge/write, `COOKIES_DIR` or project root, dict/list/bypass formats) |
| `dsk/providers/__init__.py` | Multi-provider package docstring: the unified stream-chunk contract |
| `dsk/providers/*_provider.py` | The 29 providers (4 classes: web, keyless, dormant, free-API) |
| `dsk/providers/duck_solver.js` | Node.js solver (sandboxed `vm`) of duck.ai's anti-abuse challenges |
| `dsk/chatgpt_relay.py`, `dsk/qwen_relay.py` | Browser relays over the real UIs (Sentinel/WAF walls) |
| `tools/qwen/` | Aliyun slide-captcha solver (`solve_fast.py`: whiteness-blob + masked-NCC target detection with track calibration; `fetch_solve.sh` challenge fetcher) used by the qwen signup rung |
| `dsk/refresher.py` | Credential bot: HTTP refresh → re-login → auto-signup, daemon + CLI |
| `dsk/mailgen.py` | 7 throwaway mailbox backends with OTP extraction |
| `dsk/selfheal.py` | Periodic probing + validated LLM auto-patch of providers |
| `dsk/toolprobe.py` | Dynamic tool-calling capability probing per model; pool-aware `tools` annotation of the router entries |
| `dsk/llmtrim.py` | Always-on payload trimmer to the context budget |
| `dsk/proxies.py` | Dynamic proxy pool, fast-only, per-provider, no-proxy first-class |
| `dsk/browser.py` | Shared Chromium (1 process, N tabs, reaper, hygiene) |
| `dsk/api.py`, `dsk/pow.py`, `dsk/wasm/` | Reverse-engineered DeepSeek client + PoW WASM |
| `dsk/bypass.py`, `dsk/CloudflareBypasser.py`, `dsk/server.py`, `dsk/run_and_get_cookies.py` | Cloudflare bypass and legacy of the original project |
| `dsk/static/index.html` | Web playground |
| `example.py` | Legacy usage example of the original `dsk` library (kept from upstream) |
| `tests/` | 21 offline suites (provider contracts, router family, credentials, relay, mailgen) |
| `Dockerfile`, `docker-compose.yml`, `.env.example` | Docker packaging with data volume and complete env surface |

---

*This document describes the real state of the code and the intended
direction: in case of conflict between this file and the code, the code wins —
and this file must be updated.*
