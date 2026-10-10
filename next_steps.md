# next_steps.md — Inference4Free

> Living work queue. Order = priority (README priorities: live breakages → OpenAI
> conformance → credential bot → discovery/routing → streaming/reasoning/tools →
> performance → tech debt).
>
> **`README.md` is authoritative; `project_vision.md` wins if the two ever
> conflict.** Every item below states (a) the *measured* evidence, (b) the files
> involved, (c) the acceptance test. Pick **one** item, finish it fully, verify
> live, commit atomically, then update this file.

---

## Status snapshot — 2026-10-10 (commit `8802607`)

- Stack: **healthy**, host `:18010` → container `:8000`; `/v1/models` = **141** routes.
- Working web/session providers: `deepseek` (muted), `chatgpt` (relay), `mistral`,
  `qwen` (HTTP → relay), `copilot`, `glm`.
- **Keyless tiers serving (must never regress):** `duck`, `pollinations`, `llm7`.
- Free-API tiers with credentials: `groq`, `openrouter`, `llm7`.
- Cross-provider fallback net is carrying the broken providers (requests still
  answer), so none of the P0 items below is a total outage — they are degraded
  quality/latency, not darkness.
- Local lan llama-swap (`http://192.168.1.29:11435/v1`, model `local-model`) is
  **online** — see *Balance check* below.

---

## P0 — Live breakages (upstream/environmental walls, not code bugs)

These are the priority-1 class, but each is **server-side gated**; the fix is
environmental (egress / fresh account), not a patch. Do not "fix" them by
weakening a working provider. Document any new evidence in README Troubleshooting.

1. **deepseek — account muted (`biz_code=5`).**
   Evidence: `/health` → selfheal `deepseek: auth` "account muted"; auto-signup
   blocked by Cloudflare **Turnstile** from datacenter egress
   (`users/create_email_verification_code` → `biz_code:2 RECAPTCHA_VERIFY_FAILED`).
   Fix: residential `I4F_SIGNUP_PROXY`, or seed `DEEPSEEK_LOGIN_EMAIL/_PASSWORD`.
   Files: `dsk/refresher.py`, README Troubleshooting.
   *Acceptance:* a real `deepseek/deepseek-chat` completion succeeds (not via net).

2. **claude — `account_banned` (HTTP 403 `account_banned`).**
   Evidence: `/health` selfheal `claude: auth`; error body
   `{"error":{"type":"permission_error","message":"account_banned"}}`.
   Prior wall: "couldn't verify your browser" at the magic-link exchange.
   Fix: fresh account on clean egress; the ban is permanent for that identity
   (refresher should ban-rotate, verify `_BAN_PATTERNS` covers `account_banned`).
   Files: `dsk/refresher.py`, `dsk/selfheal.py`.
   *Acceptance:* a real `anthropic/*` chat completion succeeds.

3. **perplexity — anonymous session auth-walled (`fraud_authwall_upsell`).**
   Evidence: `/health` selfheal `perplexity: auth` "anonymous session auth-walled
   — signup rung must create a signed-in jar". `available()` is unconditional.
   Fix: a signup/login rung that produces a **signed-in** cookie jar, or a
   working proxy pool. Files: `dsk/providers/perplexity_provider.py`,
   `dsk/refresher.py`.
   *Acceptance:* a real `perplexity/*` completion succeeds.

4. **z.ai (glm-anon) — anonymous tier gated.**
   Evidence (live, 2026-10-10): `z.ai/glm-4.5` direct returned no completion;
   `z.ai/glm-4.5-thinking` was served by `llm7/…` (net). Prior probe: HTTP 403
   "Model not available for current user level". The separate `glm` provider
   self-heal probe reports `ok` — **re-measure** which surface is alive before
   acting (z.ai anonymous vs chatglm.cn signed). Files:
   `dsk/providers/glm_provider.py`, `dsk/selfheal.py`.
   *Acceptance:* a real `z.ai/glm-*` completion succeeds from the provider, not
   the net.

---

## P1 — OpenAI conformance

- **None open** as of `8802607` (the provider-scoped `auto-fast`/`auto-thinking`
  pool-contract metadata bug was fixed — see commit message). Keep the
  `/v1/models` contract checks in the battery below.

---

## P2 — Credential bot / refresher

1. **DeepSeek signup Turnstile wall** (blocked by P0#1) — track until egress is
   fixed; confirm the captcha verdict path still fails fast (no 180 s OTP stall).
2. **Activate dormant free-API tiers** once any key exists — catalog must enter
   the `auto` tables dynamically: `groq`, `cerebras`, `modelscope`, `mistral_api`,
   `openrouter`, `google_ai_studio`, `cohere`, `cloudflare`, `meta`, `blackbox`,
   plus session surfaces `t3chat`, `innerai`, `adapta`, `arena`, `huggingchat`.
   Files: `dsk/providers/*_provider.py` (each already has a dormant contract +
   tests); activation is env-only (`<PROVIDER>_API_KEY` / jar).
   *Acceptance:* the provider's `<prefix>/auto` serves a real completion and its
   leaves appear on `/v1/models`.

---

## P3 — Model discovery / routing

1. **gemini (`google/*`) discovers ZERO leaf models.**
   Evidence (live): `/v1/models` lists only `google/auto`, `google/auto-fast`,
   `google/auto-thinking`; `/providers/gemini` → `has_credentials:false`,
   `renew-failed: refresh: no gemini credential`. Investigate the discovery
   path (`list_models` over `__Secure-1PSID` RPC) and whether cookie rotation
   can recover, vs. the known Google phone-verification signup wall.
   Files: `dsk/providers/gemini_provider.py`, `dsk/refresher.py`.
   *Acceptance:* ≥1 `google/gemini-*` leaf on `/v1/models` serving a completion.

2. **qwen relay cold-start inflates `alibaba/auto` latency.**
   Evidence: `alibaba/auto` measured **36 s** (and one >120 s) — the HTTP
   transport served an HTML interstitial → browser relay "page never settled
   after reload" → demote → net. Not an outage, but a UX regression.
   Files: `dsk/qwen_relay.py`, `dsk/providers/router.py` (first-token bound on
   relay attempts). *Acceptance:* `alibaba/auto` text request returns in < ~15 s
   or fails over to the net promptly.

---

## P4 — Streaming / reasoning / tool-call emulation

1. **Tool-less canary flip-flop on search modes.**
   Evidence: toolcall/status → `deepseek/deepseek-chat` **failed** "answered
   without tool call: '…Paris is around 13 °C…<cita…'" and
   `openai/chatgptauto` failed 2× "answered without tool call: 'Searching the
   web…'". Web-search preludes / citations leak into the answer and suppress the
   tool directive. Consider: exclude search modes from the tool canary, and/or
   strip citation/search-prelude frames before tool-call parsing.
   Files: `dsk/toolprobe.py`, `dsk/openai_server.py` (tool protocol),
   `dsk/providers/duck_provider.py` (citation frames).
   *Acceptance:* the canary is stable across repeated runs; real
   `tool_choice:"required"` round-trips still produce `finish=tool_calls`.

---

## P5 — Performance / latency

1. **Proxy pool is thin.** `/health` → `pool=400, healthy=18`. With
   `I4F_PROXY_TOP_K=5` the traffic draws from a small set; widen sources or
   relax `I4F_PROXY_MAX_LATENCY`/top-k so P0 fallbacks have fresher egress.
   Files: `dsk/proxies.py`, `.env`.
   *Acceptance:* `healthy` count improves and first-token latency does not regress.

---

## P6 — Technical debt

1. `tests/test_mailgen_backends.py::test_guerrillamail_timeonly_stamp` is
   **time-of-day fragile** (passes/fails depending on the clock). Pin the
   timestamp instead of asserting against "now". Files: `tests/test_mailgen_backends.py`.
2. Pre-existing host quirk: `/app/tests` disappears after a container restart
   (`docker exec inference4free mkdir -p /app/tests` before `docker cp`-ing tests).

---

## Operational commands

```bash
# rebuild + restart (code is baked into the image)
cd /opt/docker/compose/inference4free
docker compose build
sudo -n systemctl restart docker-compose@inference4free   # or: docker compose up -d

# health + inventory (host port 18010 -> container 8000)
curl -s http://localhost:18010/health | python3 -m json.tool
curl -s http://localhost:18010/v1/models | python3 -c 'import sys,json;print(len(json.load(sys.stdin)["data"]))'

# tests (pytest lives in the project venv)
.venv/bin/python -m pytest tests/ -q
```

## Balance check (before starting any session)

```bash
# local lan llama-swap online? if yes, we are good to go
curl -s -m5 http://192.168.1.29:11435/v1/models \
  -H 'Authorization: Bearer sk-Xi8iefeBeitie7kieza5weiDuiphae5oji8ogaishoh2eeweiche8aiGoo3zoove'

# otherwise check remaining quota; STOP if weeklyTokenLimit.percentRemaining < 5
curl -s https://api.synthetic.new/v2/quotas \
  -H 'Authorization: Bearer syn_05a5cae50fb5f5fab572e5efceccb543'
```

## Validation battery (required for any change to server/provider/refresher/bypass/selfheal/llmtrim)

1. `docker compose build` + restart the stack, wait for `/health` = `ok`.
2. **Non-streaming** `POST /v1/chat/completions` (`model:"auto"`) → has `choices`
   and `served_by`.
3. **Streaming** (`stream:true`, `model:"auto-fast"`) → `finish_reason` +
   `data: [DONE]`, **zero** `reasoning_content` (contract).
4. **`/v1/models`** → expected count; **no** `auto-fast` entry with
   `thinking_enabled:true` / `search_enabled:true`, **no** `auto-thinking` entry
   with `thinking_enabled:false`.
5. **Tool round-trip** (`tools:[get_weather]`) → `finish_reason:"tool_calls"`
   with populated `arguments`.
6. **Keyless tiers** `duck`, `pollinations`, `llm7` still serve (never regress).
7. `/v1/images/generations` only if image paths were touched.
8. Full suite green (`.venv/bin/python -m pytest tests/ -q`).
