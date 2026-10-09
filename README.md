# Inference4Free (OpenAI-compatible fork)

**Free access to 29 providers — DeepSeek, Gemini (Web), ChatGPT (Web), Claude, Grok, Mistral, Qwen, Kimi, Copilot, Perplexity, GLM and more, plus anonymous and free-tier tiers — through their own web APIs — exposed as a standard OpenAI-compatible server, packaged in Docker, and built for agent coding.**

This project talks directly to the chat web apps — `chat.deepseek.com`, `gemini.google.com`, `chatgpt.com`, `claude.ai`, `grok.com`, `chat.mistral.ai`, `chat.qwen.ai`, `kimi.com`, `copilot.microsoft.com`, `perplexity.ai`, `chat.z.ai` (GLM), `duck.co`, `huggingface.co/chat`, `arena.ai`, `t3.chat`, `inner.ai`, Adapta and `pollinations.ai` (anonymous keyless tier: text chat + image generation) — plus the **free-tier API keys** of Groq, Cerebras, ModelScope, Mistral API, OpenRouter, Google AI Studio, Cohere, Cloudflare Workers AI, Meta and Blackbox: **29 providers**, none of them a paid API. Everything is re-exposed behind the familiar OpenAI endpoints (`/v1/chat/completions`, `/v1/models`, `/v1/images/generations`). That means any tool that speaks the OpenAI API — [aider](https://aider.chat) / **AiderDesk agent mode**, OpenWebUI, LiteLLM, LibreChat, the `openai` SDK, anything else — can use these models for free.

**Zero-credential operation:** a background credential bot *creates every account itself* (auto-signup with auto-generated mailboxes, verification OTPs read automatically), stores the sessions in `./data/` and renews them automatically. Nothing to paste, nothing to maintain.

```
┌──────────────┐   OpenAI API    ┌────────────────────────────┐   web API    ┌──────────────────────────┐
│ aider /      │ ──────────────► │  dsk/openai_server.py      │ ───────────► │ chat.deepseek.com (token)│
│ AiderDesk /  │  /v1/chat/...   │  FastAPI + SSE + tool-call │  reverse-    │ gemini.google.com (cookie│
│ any OpenAI   │ ◄────────────── │  emulation + playground    │  engineered  │ chatgpt.com backend-api  │
│ client       │  SSE chunks     └────────────────────────────┘              └──────────────────────────┘
└──────────────┘
```

**No hardcoded models.** Model lists are discovered *dynamically* from each provider's live web session: DeepSeek exposes its three web-app modes, ChatGPT is discovered via `/backend-api/models`, Gemini via its own web RPC — so new upstream models appear on `/v1/models` automatically (refreshed every `I4F_MODELS_TTL` seconds, with on-demand re-discovery when an unknown model id is requested). Unknown providers are skipped gracefully and previously discovered routes are kept on refresh failures.

## 🍴 Fork notice & credits

> This repository is a **fork of [xtekky/deepseek4free](https://github.com/xtekky/deepseek4free)**.
> All credit for the original reverse-engineered DeepSeek client — the `dsk` library, the WASM proof-of-work implementation, and the Cloudflare bypass — goes to **[@xtekky](https://github.com/xtekky)** and the original project's contributors. Huge thanks! 🙏
>
> **What this fork adds on top of the original:**
> - a full **OpenAI-compatible API server** (`dsk/openai_server.py`): `/v1/chat/completions` (streaming + non-streaming), `/v1/models`, `/health`
> - **tool-calling emulation** so function-calling clients (aider / AiderDesk agent mode) work end-to-end
> - `reasoning_content` streaming for the thinking model
> - complete **Docker / docker-compose packaging** with persistent Cloudflare-cookie storage
> - **autonomous credential bot** (`dsk/refresher.py`): creates accounts with auto-generated
>   mailboxes (catch-all IMAP or any of seven throwaway backends — emailnator real-Gmail, tempmail.lol/plus, temp-mail.io, Guerrilla, mail.tm), reads the verification OTPs, stores the
>   sessions in `./data/` and renews them automatically (refresh → re-login → fresh signup)
> - **29-provider router** (incl. the anonymous keyless `duck` and `pollinations` tiers) with dynamic model discovery, fallback chains and a
>   `I4F_PROVIDERS` allowlist; **self-healing** provider patches (`dsk/selfheal.py`)
> - **synthetic model routers** `auto` / `auto-fast` / `auto-thinking` — global and per provider — with strict thinking/reasoning contracts (§7)
> - **anonymous browser relays** (`dsk/chatgpt_relay.py`, `dsk/qwen_relay.py`): ChatGPT/Qwen keep serving through the public web UI when session credentials expire, on a **shared Chromium manager** (`dsk/browser.py`)
> - **strict reasoning/answer separation**: provider reasoning streams (ChatGPT's collapsed `Thought · Ns` panel, Mistral's reasoning chunks) are routed to `reasoning_content` and never leak into the answer text
> - **`Retry-After` propagation** on every 429 (HTTP header + streaming error payload)
> - **tool-calling probe** (`dsk/toolprobe.py`): the `tools` flag on `/v1/models` is measured, not assumed
> - **llmtrim** request stage (`dsk/llmtrim.py`): LLM CALL → llmtrim → proxy rotator → LLM response
> - **fast-only proxy rotation** with the no-proxy route as a first-class candidate
> - a built-in **ChatGPT-style playground** (`/`) with saved chats (24 h expiry)

---

## 📖 How it works

This section explains the whole pipeline, from account creation to the model's answer.

### 1. Credentials are created and renewed automatically

DeepSeek's web app authenticates every request with a bearer `userToken`; the other providers use session cookies. **You never paste any of them**: the credential bot (`dsk/refresher.py`) runs a renewal ladder on its own —

1. **HTTP refresh** of the bot-managed session jars in `./data/` (`deepseek_token`, `gemini_cookies.json`, `chatgpt_cookies.json`, …);
2. **headless-Chromium re-login** with the accounts the bot created itself (`data/accounts.json`);
3. **fully automatic signup**: a throwaway mailbox is generated (catch-all IMAP domain via `I4F_MAIL_DOMAIN`, otherwise the best of seven throwaway backends — emailnator real-Gmail addresses, tempmail.lol, tempmail.plus, temp-mail.io, Guerrilla, mail.tm/mail.gw), the signup form is filled in a real browser (CMP/consent interstitials dismissed automatically), the verification OTP is read from that mailbox and the fresh account is stored.

Credentials therefore live exclusively in the `./data` volume — the `.env` file contains **no LLM credentials at all** (the server still honours a `userToken` sent as the client's API key, and legacy env vars if present, but nothing requires them). Providers whose web apps have no signup flow (claude, grok, qwen, kimi, mistral) stay dormant until one of their tokens exists in `data/`; `duck` and `pollinations` need no credentials at all (anonymous, keyless); none of them ever block the other providers, and `I4F_PROVIDERS` can hide any of them entirely.

### 2. Cloudflare bypass (`dsk/bypass.py`, `dsk/CloudflareBypasser.py`)

`chat.deepseek.com` sits behind Cloudflare. If requests start being challenged ("Just a moment…"), the bypass module spins up a real (undetected) Chromium via [DrissionPage](https://github.com/g1879/DrissionPage), visits the site, clicks through the challenge with `CloudflareBypasser`, and captures the resulting `cf_clearance` cookie. The cookie is saved to `cookies.json` (in Docker: `/data/cookies.json`, bind-mounted to the project's `./data` directory) and silently attached to every subsequent API request until it expires.

You normally don't have to do anything — but if you ever see Cloudflare errors, run the helper once:

```bash
python -m dsk.bypass
```

### 3. Proof-of-Work via WASM (`dsk/pow.py`)

Before certain calls, DeepSeek's web API requires a **proof-of-work**: it hands out a challenge (algorithm, challenge string, salt, difficulty) and expects a hash answer back. The site itself solves this in WebAssembly — so this project does the same: `dsk/pow.py` loads the actual `sha3_wasm_bg.*.wasm` binary shipped from DeepSeek's frontend, writes the challenge into WASM memory, and executes DeepSeek's own hashing code (`DeepSeekHash`) to produce a valid answer (`DeepSeekPOW.solve_challenge`). Because it's the genuine WASM module, the answers are always accepted and never break when the algorithm changes.

### 4. The reverse-engineered client (`dsk/api.py`)

`DeepSeekAPI` wraps `https://chat.deepseek.com/api/v0`:

- `create_chat_session()` opens a chat session
- `chat_completion(session_id, prompt, thinking_enabled, search_enabled, parent_message_id)` posts the prompt and **streams Server-Sent-Event chunks** back as a generator
- each chunk is a dict like `{'type': 'thinking' | 'text', 'content': ...}` (plus message ids for threading)
- it also handles header building, cookie refreshing, PoW challenge solving, retries, and maps failures to typed exceptions (`AuthenticationError`, `RateLimitError`, `NetworkError`, `CloudflareError`, `APIError`)

### 5. The OpenAI translation layer (`dsk/openai_server.py`)

This is the main addition of this fork. A FastAPI server translates between the OpenAI wire format and the `dsk` client:

- **Prompt flattening** — the DeepSeek web API only accepts a single flat prompt per turn, while OpenAI clients send a full message list. The server therefore renders the whole conversation into one prompt: `[System]`, `[Assistant]`, `[Tool result]` turns, including assistant messages that contain `tool_calls`.
- **Model mapping** — the OpenAI `model` field is mapped to DeepSeek capabilities:

  | OpenAI model name | DeepSeek behaviour |
  |---|---|
  | `deepseek-reasoner` | thinking process **enabled** |
  | `deepseek-chat` | thinking **disabled** |
  | `deepseek-search` | thinking disabled + **web search** enabled |

  (names are overridable via `I4F_MODEL_THINKER` / `I4F_MODEL_FAST` / `I4F_MODEL_SEARCH`)

  The DeepSeek web API does **not** expose per-model metadata, so `/v1/models` advertises DeepSeek's documented limits — 128K context (`context_length` / `max_model_len`) and max output of 64K for `deepseek-reasoner` / 32K otherwise (`max_completion_tokens` / `max_tokens`). Agent tools like AiderDesk and aider read these fields to size the context window and max output tokens; adjust them via `I4F_CONTEXT_LENGTH`, `I4F_MAX_OUTPUT_THINKING` and `I4F_MAX_OUTPUT` if DeepSeek changes its limits.
- **Streaming** — the synchronous DeepSeek generator runs in a worker thread and is bridged into an async SSE response. `thinking` chunks are re-emitted as OpenAI `reasoning_content` deltas; `text` chunks become normal `delta.content` deltas; the stream ends with a proper `finish_reason` chunk and `data: [DONE]`.
- **Reasoning separation** — providers that fold reasoning into the answer stream are separated **before** the shim: the ChatGPT relay splits the collapsed `Thought · Ns` panel out of the reply region as `thinking` chunks, and the Mistral stream classifier tags every chunk with the type its `contentChunk` object declared (the reasoning summary arrives as plain appends — only the chunk-type memory says which chunk is the reasoning one). The answer text stays clean; the non-streaming shape carries `message.reasoning_content` alongside `message.content`.
- **Tool-calling emulation** — see the next section.
- **Compatibility** — standard OpenAI parameters (`temperature`, `top_p`, `max_tokens`, `stop`, `seed`, `frequency_penalty`, `presence_penalty`, `response_format`, `stream_options.include_usage`, …) are accepted; the ones DeepSeek cannot honour are tolerated and ignored, so exotic clients never get validation errors.
- **Vision & image generation** — models whose provider supports them accept OpenAI-style multimodal `content` parts (`{"type": "image_url", "image_url": {"url": "data:image/png;base64,…"}}`) and are advertised in `/v1/models` via `vision: true` / `image_gen: true` capability flags. Vision-capable models also power `POST /v1/images/generations` (OpenAI Images API shape, `b64_json` responses). DeepSeek web chat is text-only — sending images to it returns `400 model_does_not_support_vision`; Gemini-web and ChatGPT-web models support both when configured; Pollinations adds keyless image generation (`pollinations/sana` on `/v1/images/generations` and in-chat via markdown images) plus vision on whatever models its anonymous tier currently advertises as image-input capable. The playground renders generated/linked images inline (click to open the full-size original) and the `🎨 image gen` badge marks capable models.
- **Per-request direct connection** — non-standard `disable_proxy: true` (top level of the request body, chat and image-generation endpoints) forces the request to skip the rotating proxy pool and connect directly. Handy for low-latency testing when you don't want a random free proxy in the path.

### 6. Tool calling (agent coding)

DeepSeek's web API has **no native function calling**, which normally breaks agent tools. The server emulates it:

1. When a request includes `tools`, the server appends a **tool protocol** to the flattened prompt: the available functions with their JSON schemas, plus the instruction that the model must answer with exactly one line:

   ```
   TOOL_CALL: {"name": "<tool name>", "arguments": {...}}
   ```

2. The streamed answer is buffered and parsed. If it contains a `TOOL_CALL:` directive, the server converts it back into a **genuine OpenAI tool-call response**: `choices[0].message.tool_calls` (with `call_…` ids and JSON `arguments`), `finish_reason: "tool_calls"`, and the matching tool-call delta chunks in streaming mode.
3. When the conversation comes back with the assistant's `tool_calls` history and `[Tool result]` messages, they are rendered into the next prompt so the model can see its own calls and their results.

The result: aider, **AiderDesk agent mode**, and every other function-calling client work end-to-end — the agent believes it is talking to a real tool-calling model. `tool_choice: "none"` disables the protocol; a forced `tool_choice: {"function": {"name": …}}` is honoured too.

### 7. Model routers (`auto`, `auto-fast`, `auto-thinking`)

Besides the concrete models discovered from each provider, the server exposes synthetic **router** ids — they own no upstream model, they build a serving chain per request from live provider state (category preferences, credential health, measured first-token latency, provider-level round-robin) and fall back across it on rate limits, auth walls, stalls and outages. Each router kind exists **globally** and **per provider**:

| Router id (global) | Per provider (examples) | Serves |
|---|---|---|
| `auto` | `deepseek/auto`, `z.ai/auto`, `alibaba/auto`, … | Every discovered model — the smart router, default when `model` is omitted. Classifies each request (coding / general / translation / summarize / vision / image_gen) and orders the chain accordingly. |
| `auto-fast` | `deepseek/auto-fast`, `z.ai/auto-fast`, … | **Only models that can never think** — no thinking capability, or no way to activate it (web-search modes excluded too). Served with thinking forced **off**: zero `reasoning_content` output, by contract. |
| `auto-thinking` | `deepseek/auto-thinking`, `z.ai/auto-thinking`, … | **Only thinking-capable models** — always-on ones and models callable with the thinking flag active. Served with thinking forced **on**. |

- The pools are a strict, disjoint partition of every discovered model; a provider whose pool is empty (e.g. a thinking-only catalog under `…/auto-fast`) answers with a clean error instead of breaking the contract.
- The per-request `thinking` body flag is **ignored** on `auto-fast` / `auto-thinking`: the router id is the contract. On `auto` (and on concrete models) it keeps working as a toggle.
- Search modes (`deepseek-search`, `grok-deepsearch`, …) are reachable directly and via `auto`; they never serve `auto-fast` (a web crawl is not fast) and enter `auto-thinking` only when the model also thinks.
- Pool capability metadata on `/v1/models` reflects each router's contract (`auto-fast` advertises `thinking_enabled: false`), so gating clients show the right toggles.
- Router ids are **owned by the server**: the tool-calling probe and the per-model probe endpoint never track a router id as a leaf model, and on `/v1/models` a router's dynamic `tools` flag mirrors its pool (`true` while any leaf in the pool can still take tools, `false` only when every pool leaf is known-failed). Router entries are never hidden by `I4F_HIDE_TOOLLESS`.
- When the **whole chain fails**, the final error names every provider's failure — `[fallback chain exhausted: chatgpt/chatgptauto: relay busy with another stream | deepseek: account muted]` — so a request never reads as the *last* provider's complaint. The last failure's exception type is preserved (a 429 stays a 429, `Retry-After` included).

```bash
# fast, never any reasoning output, across all providers:
curl http://localhost:8000/v1/chat/completions -d '{"model":"auto-fast","messages":[{"role":"user","content":"summarize this in one line"}]}'

# deep reasoning, thinking forced on, best thinking model of all providers:
curl http://localhost:8000/v1/chat/completions -d '{"model":"auto-thinking","messages":[{"role":"user","content":"solve this step by step"}]}'

# same contracts, restricted to one provider:
curl http://localhost:8000/v1/chat/completions -d '{"model":"z.ai/auto-thinking","messages":[…]}'
```

### 8. Anonymous browser relays (`chatgpt`, `qwen`)

When a ChatGPT/Qwen session expires, the providers do **not** go dark: they fall back to an **anonymous browser relay** that drives the public web UI (`chatgpt.com`, `chat.qwen.ai`) in a real Chromium — no credentials needed at all. Model titles are read from the site's own picker; a picker that won't open degrades gracefully to the family default. The relay serves **one stream at a time** (the web UI is a single chat): a concurrent request fails fast with `relay busy with another stream` and the router falls back to the next chain position. Model discovery, the picker catalog and the relay state feed `/v1/models` exactly like the HTTP transport.

All relay/tab consumers share **one Chromium process** (`dsk/browser.py`): per-profile tabs, idle reaping (`I4F_BROWSER_IDLE_REAP`), and stale-page recycling (`I4F_CHATGPT_RELAY_STALE_AFTER`, `I4F_QWEN_RELAY_STALE_AFTER`) so anonymous sessions — which expire server-side — never serve from a dead page.

### Module map

| File | Role |
|---|---|
| `dsk/api.py` | Reverse-engineered DeepSeek client (sessions, streaming, errors, cookies) |
| `dsk/pow.py` | WASM proof-of-work solver using DeepSeek's own WASM binary |
| `dsk/bypass.py` | Cloudflare `cf_clearance` cookie fetcher/validator |
| `dsk/CloudflareBypasser.py` | Browser automation that clicks through Cloudflare challenges |
| `dsk/run_and_get_cookies.py` | Standalone helper to grab cookies from a running bypass server |
| `dsk/server.py` | Upstream's original bypass HTTP service (unchanged from the original repo) |
| `dsk/openai_server.py` | **This fork:** OpenAI-compatible API server (aider/AiderDesk entry point) + playground web UI |
| `dsk/providers/base.py` | Provider abstraction, dynamic `Route` model registry, retry/fallback error taxonomy |
| `dsk/providers/deepseek_provider.py` | DeepSeek web provider (PoW + Cloudflare bypass) |
| `dsk/providers/gemini_provider.py` | Gemini web-chat provider (`gemini.google.com` session cookies, no official API) |
| `dsk/providers/chatgpt_provider.py` | ChatGPT web provider (`chatgpt.com/backend-api` session cookies, no official API) |
| `dsk/providers/jar.py` | Shared bot-managed cookie-jar helpers (`<name>_cookies.json` + `<NAME>_COOKIES` env fallback) |
| `dsk/providers/claude_provider.py` | Claude web provider (`claude.ai` temporary conversations, `sessionKey` cookie) |
| `dsk/providers/grok_provider.py` | Grok web provider (`grok.com` REST app-chat, `sso` cookie, auto image-gen via Aurora) |
| `dsk/providers/mistral_provider.py` | Mistral Le Chat provider (`chat.mistral.ai`, single create-mode stream call; session-token gated) |
| `dsk/providers/qwen_provider.py` | Qwen web provider (`chat.qwen.ai` api/v2, bundled `bx-ua` anti-bot fingerprint) |
| `dsk/providers/kimi_provider.py` | Kimi web provider (`kimi.com` connect+json gRPC-web frames) |
| `dsk/providers/copilot_provider.py` | Microsoft Copilot provider (`copilot.microsoft.com` websocket, anonymous by default) |
| `dsk/providers/perplexity_provider.py` | Perplexity provider (`perplexity.ai` SSE ask, Sonar-backed routes) |
| `dsk/providers/glm_provider.py` | GLM provider (dual backend: anonymous `chat.z.ai` + `chatglm.cn` signed stream) |
| `dsk/providers/*_provider.py` | The remaining providers — web (`duck`, `pollinations`, `arena`, `huggingchat`), session-cookie (`t3chat`, `innerai`, `adapta`) and free-API tiers (`groq`, `cerebras`, `modelscope`, `mistral_api`, `openrouter`, `llm7`, `google_ai_studio`, `cohere`, `cloudflare`, `meta`, `blackbox`) |
| `dsk/providers/router.py` | Dynamic model discovery (TTL cache) + retry/fallback orchestration + the synthetic routers (`auto`, `auto-fast`, `auto-thinking` — global and per provider) |
| `dsk/chatgpt_relay.py` | Anonymous ChatGPT browser relay (serves the public UI when session cookies expire; splits the `Thought · Ns` panel into `thinking` chunks) |
| `dsk/qwen_relay.py` | Anonymous Qwen browser relay (same pattern for `chat.qwen.ai`) |
| `dsk/browser.py` | Shared Chromium manager — ONE browser process for every relay/tab consumer, idle reaping |
| `dsk/mailgen.py` | Signup mailbox backends: catch-all IMAP + seven throwaway services (emailnator real-Gmail, tempmail.lol/plus, temp-mail.io, Guerrilla, mail.tm/mail.gw) with OTP fetch |
| `dsk/refresher.py` | Credential bot: HTTP refresh → headless re-login → fully automatic signup ladder |
| `dsk/selfheal.py` | Upstream-breakage detection + LLM auto-patch with audit trail (`/data/selfheal`) |
| `dsk/llmtrim.py` | Request-trimming stage — context budgeting before the proxy rotator |
| `dsk/proxies.py` | Outbound proxy pool: list aggregation, concurrent health probing, latency ranking, per-provider assignment |
| `dsk/toolprobe.py` | Tool-calling probe: the dynamic `tools` flag per model/router on `/v1/models` is measured, not assumed |
| `dsk/static/index.html` | llama.cpp-style playground web UI (served at `/` and `/playground`) |
| `dsk/wasm/` | DeepSeek's SHA3 WASM module used for PoW |

---

## 🔀 Providers (reverse-engineered surfaces + free tiers — no paid APIs)

Twenty-nine providers are supported. Each one borrows the credentials of a
normal browser session, runs fully anonymous, or uses a **free-tier** API key —
never a paid API. Providers without configured credentials are simply skipped;
the server runs
with whichever are available. Configure them in `.env` (see
`cp .env.example .env`) and restart the stack
(`sudo systemctl restart docker-compose@inference4free` or `docker compose up -d`).

### 1. DeepSeek (`deepseek-chat`, `deepseek-reasoner`, `deepseek-search`)

Uses your free-account `userToken` from the DeepSeek web app:

1. Log in at [chat.deepseek.com](https://chat.deepseek.com) (a free account is enough).
2. Open DevTools (F12) → **Console** and run:
   `JSON.parse(localStorage.getItem("userToken")).value`
   (alternative: Network tab → send any chat → copy the `authorization`
   header value without the `Bearer ` prefix).
3. Put it in `.env`:
   ```bash
   DEEPSEEK_AUTH_TOKEN=userToken value from step 2
   ```
4. Cloudflare: nothing to configure — the bypass module solves challenges
   with a headless Chromium and caches `cf_clearance` in `./data/cookies.json`
   automatically. If you ever see Cloudflare errors, run `python -m dsk.bypass`
   once (see *Cloudflare cookies* below).

Notes: the `userToken` expires when you log out or rotate sessions — if
requests start returning 401, repeat step 2.

### 2. Gemini web (`gemini.google.com`)

Uses the two session cookies of your Google account:

1. Log in at [gemini.google.com](https://gemini.google.com).
2. DevTools (F12) → **Application** → Cookies → `https://gemini.google.com`.
3. Copy `__Secure-1PSID` and `__Secure-1PSIDTS` into `.env`:
   ```bash
   GEMINI_1PSID=<value of __Secure-1PSID>
   GEMINI_1PSIDTS=<value of __Secure-1PSIDTS>
   ```
4. Models are discovered live from the session (e.g. `gemini-2.5-flash`,
   `gemini-2.5-pro`) — no model list to configure.

Notes: `__Secure-1PSIDTS` rotates periodically; if discovery fails or
requests 401, re-copy **both** cookies. A cookie jar can also be dropped
as a file into the `./data` volume as `gemini_cookies.json`.

### 3. ChatGPT web (`chatgpt.com` backend-api)

Two options — the cookie jar is preferred (the access token is refreshed
automatically):

1. Log in at [chatgpt.com](https://chatgpt.com).
2. Either
   - fetch `https://chatgpt.com/api/auth/session` in the same browser
     (or DevTools → Network) and copy `accessToken` into `.env`:
     ```bash
     CHATGPT_ACCESS_TOKEN=<accessToken>
     ```
   - or export the cookie jar (DevTools → Application → Cookies → export,
     or a JSON array/object of cookies) into `.env`:
     ```bash
     CHATGPT_SESSION_COOKIES=<JSON string>
     ```
3. Alternatively drop the jar as a file into the `./data` volume as
   `chatgpt_cookies.json`.
4. Models are discovered live via `/backend-api/models`.

### 4. Claude web (`claude.ai`)

Borrows the `sessionKey` cookie of a logged-in browser session. Every request
runs in an ephemeral *temporary* conversation that is never persisted to your
account history.

1. Log in at [claude.ai](https://claude.ai).
2. DevTools (F12) → **Application** → Cookies → `https://claude.ai` → copy `sessionKey`.
3. Put it in `.env`: `CLAUDE_SESSION_KEY=sk-ant-sid01-...` (or
   `CLAUDE_COOKIES={"sessionKey": "..."}`).
4. Models: `claude-sonnet-4-6`, `claude-opus-4-6`, `claude-haiku-4-5` (thinking included).

### 5. Grok web (`grok.com`)

Borrows the `sso` cookie of a logged-in X/Grok session.

1. Log in at [grok.com](https://grok.com).
2. DevTools → **Application** → Cookies → `https://grok.com` → copy `sso`.
3. Put it in `.env`: `GROK_SSO=<value>` (or `GROK_COOKIES={"sso": "..."}`).
4. Models: `grok-4`, `grok-4-reasoning`, `grok-4-heavy`, `grok-3`, `grok-3-mini`,
   `grok-deepsearch`; image generation is available via Aurora.

### 6. Mistral Le Chat (`chat.mistral.ai`) — session token required

Anonymous access is now **account-gated upstream** (it returns an "An account
is now required" upsell instead of model output), so a Le Chat session token
is **required**: DevTools (F12) → Application → Cookies →
`https://chat.mistral.ai` → `session_token` → set
`MISTRAL_SESSION_TOKEN=<value>`. The provider makes a single `POST /api/chat`
`mode: 'create'` call that both starts the conversation and streams the answer
(upstream model `mistral-large-2411`, exposed as route `mistral-large`). The
stream's reasoning summary is surfaced as `reasoning_content` deltas (a
chunk-type classifier keeps it out of the answer text).

### 7. Qwen web (`chat.qwen.ai`)

Borrows the Bearer token of a logged-in session:

1. Log in at [chat.qwen.ai](https://chat.qwen.ai).
2. DevTools → **Network** → send any message → open an `api/v2/*` request →
   copy the `Authorization` header value without `Bearer `.
3. Put it in `.env`: `QWEN_TOKEN=<value>`.
4. Models: `qwen3.7-max`, `qwen3.6-plus`, `qwen3-coder-plus`, `qwen3.6-35b-a3b`,
   `qwen3.6-27b`. The bundled `bx-ua` anti-bot fingerprint is configurable via
   `QWEN_BX_UA`/`QWEN_UMID` if Alibaba changes it.

### 8. Kimi web (`kimi.com`)

Borrows the `token` cookie of a logged-in session:

1. Log in at [kimi.com](https://www.kimi.com).
2. DevTools → **Application** → Cookies → `https://www.kimi.com` → copy `token`.
3. Put it in `.env`: `KIMI_TOKEN=<value>` (or `KIMI_COOKIES={"token": "..."}`).
4. Models: `kimi-k2.6`, `kimi-k2.5` (thinking included).

### 9. Microsoft Copilot (`copilot.microsoft.com`) — anonymous

Works with **zero credentials** (anonymous tier with synthetic device
cookies, generated and persisted automatically). Paste a whole cookie jar
JSON for authenticated quality: `COPILOT_COOKIES=<JSON>`. Models:
`copilot-chat`, `copilot-think`, `copilot-smart` (streams over websocket).

### 10. Perplexity (`perplexity.ai`) — anonymous

Works with **zero credentials** (anonymous search-backed answers). Paste a
logged-in cookie jar JSON for pro-tier models: `PERPLEXITY_COOKIES=<JSON>`.
Models: `perplexity-turbo`, `perplexity-pro`, `perplexity-reasoning`,
`perplexity-gpt5`, `perplexity-claude-4.5-sonnet`, `perplexity-o3`.

### 11. GLM — Z.ai / ChatGLM (`chat.z.ai`, `chatglm.cn`)

Dual backend, anonymous by default: z.ai serves `glm-4.5` and
`glm-4.5-thinking` with no setup (the access token is fetched automatically).
Setting `GLM_REFRESH_TOKEN` (chatglm.cn → DevTools → Application → Local
Storage → `refresh_token`) unlocks `glm-4.6` and `glm-4.6-thinking` routes.

### 12. Free-API tiers (free key, never a payment)

OpenAI-compatible free tiers — set the key and the live catalog is discovered
automatically and enters the `auto` routers. Dormant until the key exists:

| Provider | Env | Notes |
|---|---|---|
| Groq | `GROQ_API_KEY` | fast open models (Llama, Kimi, …) |
| Cerebras | `CEREBRAS_API_KEY` | wafer-scale inference, open models |
| ModelScope | `MODELSCOPE_API_KEY` | open-model catalog (Qwen family, …) |
| Mistral API | `MISTRAL_API_KEY` | official free API tier (distinct from the Le Chat web provider above) |
| OpenRouter | `OPENROUTER_API_KEY` | the aggregator's `:free` models |
| Google AI Studio | `GOOGLE_AI_STUDIO_API_KEY` (or `GEMINI_API_KEY`) | Gemini free tier |
| Cohere | `COHERE_API_KEY` | command models on free trial keys |
| Cloudflare Workers AI | `CLOUDFLARE_API_TOKEN` + `CLOUDFLARE_ACCOUNT_ID` | Workers AI catalog |
| Meta Muse Spark | `META_API_KEY` | `api.meta.ai/v1` (`muse-spark`, `muse-glimmer`) |
| Blackbox AI | `BLACKBOX_API_KEY` | OpenAI-compatible free tier |
| llm7 | *(none — keyless)* | free endpoint, no key at all |

### 13. Session-cookie surfaces (`t3chat`, `innerai`, `adapta`)

Three more web surfaces of the cookie-jar class — dormant until a session
exists, via the generic JSON jar (env `T3CHAT_COOKIES` / `INNERAI_COOKIES` /
`ADAPTA_COOKIES`, or the bot-managed `data/<provider>_cookies.json`):

- **t3chat** (`t3.chat`) — free "LLM chat" web app (`gpt-4o`, `gpt-4.1`, `o4-mini`, …);
- **Inner.ai** — login-only AI workspace;
- **Adapta** — login-based multi-model workspace.

### Verifying a provider

```bash
# which routes/models are live (discovered from your sessions):
curl -s http://localhost:${I4F_PORT:-8000}/v1/models | python3 -m json.tool

# quick smoke test (streaming):
curl -N -X POST http://localhost:${I4F_PORT:-8000}/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"deepseek-chat","messages":[{"role":"user","content":"ping"}],"stream":true}'

# per-provider proxy assignments + pool state:
curl -s http://localhost:${I4F_PORT:-8000}/health
```

Unknown model ids are routed by fuzzy match, and per-model fallback
chains (`I4F_FALLBACKS` / `I4F_DEFAULT_FALLBACKS`) kick in automatically
when a provider fails.

When adding or auditing a provider, keep the pool metadata honest: every
route must advertise `thinking_enabled: true` **iff** it can — or is called
so as to — stream thinking. The `auto-fast` / `auto-thinking` pools are
filtered strictly on that flag.

---

## 🚀 Quick start (Docker — recommended)

### 1. Get your token

Visit [chat.deepseek.com](https://chat.deepseek.com), log in, then either:

- run `JSON.parse(localStorage.getItem("userToken")).value` in the browser console (**recommended**), or
- open DevTools → Network tab, send any chat message, and copy the `authorization` header value (without the `Bearer ` prefix).

### 2. Configure

```bash
cp .env.example .env
# edit .env and set DEEPSEEK_AUTH_TOKEN=<paste your token>
```

`.env.example` contains every supported variable (DeepSeek token, optional Gemini/ChatGPT web-session credentials, `I4F_API_KEY`, `I4F_PORT`, model-name overrides, discovery TTL and fallback chains) with comments.

### 3. Run

```bash
docker compose up -d --build
```

The API is now available at **`http://localhost:8000/v1`**.

> ℹ️ The compose file intentionally sets `restart: no` — on this machine the container's lifecycle is managed by the systemd unit `docker-compose@.service`, so Docker itself must not restart it.

### 4. Point your tools at it

**aider / AiderDesk** (`~/.aider.conf.yml` or environment):

```bash
export OPENAI_API_BASE=http://localhost:8000/v1
export OPENAI_API_KEY=anything          # or your I4F_API_KEY value
aider --model openai/deepseek-chat
```

For agent mode, just enable it in AiderDesk — tool calling is emulated transparently (see *Tool calling* above).

**openai Python SDK:**

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="anything")
resp = client.chat.completions.create(
    model="deepseek-reasoner",
    messages=[{"role": "user", "content": "Hello!"}],
    stream=True,
)
for chunk in resp:
    delta = chunk.choices[0].delta
    print(delta.reasoning_content or delta.content or "", end="", flush=True)
```

**LiteLLM / OpenWebUI:**

```yaml
# litellm config.yaml — every model, dynamically (the catalog changes; litellm stays current)
  - model_name: inference4free/*
    litellm_params:
      model: openai/*
      api_base: http://localhost:8000/v1
      api_key: anything
```

Point OpenWebUI (OpenAI-compatible provider) at LiteLLM — or at this server directly. `reasoning_content` arrives as a separate delta, so OpenWebUI renders it in its collapsible *thinking* panel and the answer text stays clean; `auto` / `auto-fast` / `auto-thinking` work as model ids (`inference4free/auto`, …), and 429s carry `Retry-After` for the client's backoff.

### Alternative: token as API key

You can skip `DEEPSEEK_AUTH_TOKEN` entirely and pass your DeepSeek `userToken` **as the OpenAI API key** — the server uses it directly:

```bash
aider --model openai/deepseek-chat \
      --openai-api-base http://localhost:8000/v1 \
      --openai-api-key <your_userToken>
```

---

## 🖥️ Playground

A llama.cpp-style chat playground is served at `http://localhost:${I4F_PORT:-8000}/` (and `/playground`): pick any discovered model from the dropdown, stream responses (with a collapsible thinking panel for `reasoning_content`), tweak system message / temperature / max tokens / web-search toggle, and stop generations mid-stream. It talks to the same OpenAI endpoints your agent tools use, so it doubles as an end-to-end test harness. The three global routers sit in their own **⚡ smart routers** group at the top of the dropdown (`auto` is the default pick); the per-provider routers (`deepseek/auto`, `z.ai/auto-fast`, …) are listed inside their provider's group, marked ⚡.

## 🔌 API reference

| Endpoint | Description |
|---|---|
| `GET /v1/models` | Lists every model discovered from all configured providers (dynamic, TTL-cached) |
| `POST /v1/chat/completions` | Chat completions, streaming (`stream: true`) and non-streaming; supports `tools` |
| `POST /v1/images/generations` | Image generation on capable models (OpenAI Images shape, `b64_json`) |
| `GET /` or `/playground` | Chat playground web UI |
| `GET /health` | Liveness probe (proxy pool, selfheal, refresher, toolprobe state) |
| `GET /providers` · `GET /providers/{name}` | Provider inventory + per-provider detail |
| `POST /providers/{name}/test` · `/renew` · `/credentials` · `/credentials/clear` · `POST /providers/renew-all` | Manual provider test, renewal ladder, credential set/clear |
| `GET /selfheal/status` · `POST /selfheal/probe` · `POST /selfheal/refresh` | Self-heal state and manual triggers |
| `GET /toolcall/status` · `POST /toolcall/reprobe` | Tool-calling probe state and forced re-probe |

- Requests may include any standard OpenAI field; unsupported ones are ignored.
- `deepseek-reasoner` streams reasoning as `reasoning_content` deltas.
- Every completion reports what actually served it: `served_by` (`"<provider>/<model>"`, e.g. `glm/glm-4.7`) on the non-streaming response object and on the final streaming chunk (the one carrying `finish_reason`) — routers and fallback chains stay transparent.
- `reasoning_content` follows the router contract: `auto-fast` never emits it; `auto-thinking` always calls with thinking forced **on** and emits it whenever the served provider streams reasoning as separate frames (most do; a few transports fold it into the answer text); `auto` and concrete models exactly when the served model thinks.
- `/v1/models` entries carry the capability flags the routers pool on (`thinking_enabled`, `search_enabled`, `vision`, `image_gen`, `fallbacks`) plus a dynamic `tools` flag from the tool-calling probe (router entries: flag of their pool, never hidden).
- An empty pool (e.g. `…/auto-fast` on a thinking-only provider) answers with the OpenAI error shape, HTTP 502, `code: "upstream_error"` — never with a contract violation.
- Errors follow the OpenAI error shape: `{"error": {"message", "type", "param", "code"}}`.
- **429s carry `Retry-After`**: non-streaming as the HTTP header, streaming inside the error payload (`"retry_after": <seconds>`). Free-tier quotas are day/IP-bound — honour the hint; a retry storm burns the whole window.
- **Chain-exhaustion errors name every provider** that failed, not just the last one (see §7).
- Authentication: `Authorization: Bearer <I4F_API_KEY>` if you set one, otherwise any key (including your `userToken`) is accepted.

### Environment variables

| Variable | Default | Description |
|---|---|---|
| `DEEPSEEK_AUTH_TOKEN` | — | Your DeepSeek `userToken` (required unless the client sends it as API key) |
| `I4F_API_KEY` | *(none)* | If set, clients must send this as `Authorization: Bearer <key>` |
| `I4F_PORT` | `8000` | Host port binding (compose maps it to container port 8000) |
| `I4F_HOST` | `0.0.0.0` | Server bind host (non-Docker runs) |
| `I4F_MODEL_THINKER` | `deepseek-reasoner` | Name exposed for the thinking-enabled model |
| `I4F_MODEL_FAST` | `deepseek-chat` | Name exposed for the fast model |
| `I4F_MODEL_SEARCH` | `deepseek-search` | Name exposed for the web-search model |
| `I4F_CONTEXT_LENGTH` | `131072` | Context length advertised on `/v1/models` (DeepSeek's documented 128K) |
| `I4F_MAX_OUTPUT_THINKING` | `65536` | Max output tokens advertised for the thinking model (documented 64K) |
| `I4F_MAX_OUTPUT` | `32768` | Max output tokens advertised for the other models (documented 32K) |
| `GEMINI_1PSID` | *(none)* | `__Secure-1PSID` cookie from `gemini.google.com` (enables the Gemini web provider) |
| `GEMINI_1PSIDTS` | *(none)* | `__Secure-1PSIDTS` cookie from `gemini.google.com` |
| `CHATGPT_ACCESS_TOKEN` | *(none)* | ChatGPT web `accessToken` (from `/api/auth/session`) |
| `CHATGPT_SESSION_COOKIES` | *(none)* | ChatGPT session cookie jar as JSON (preferred over the raw token; auto-refresh) |
| `CLAUDE_SESSION_KEY` | *(none)* | `sessionKey` cookie from `claude.ai` (enables the Claude web provider) |
| `GROK_SSO` | *(none)* | `sso` cookie from `grok.com` (enables the Grok web provider) |
| `MISTRAL_SESSION_TOKEN` | *(none)* | `session_token` cookie from `chat.mistral.ai` (**required** — anonymous access is account-gated) |
| `QWEN_TOKEN` | *(none)* | Bearer token from `chat.qwen.ai` (enables the Qwen web provider) |
| `QWEN_LOGIN_EMAIL` / `QWEN_LOGIN_PASSWORD` | *(none)* | Operator seed: enables fully automatic qwen re-login on every renewal cycle (preferred over `QWEN_TOKEN`) |
| `QWEN_UMID` / `QWEN_BX_UA` | *(bundled)* | Override Qwen's anti-bot fingerprint headers if upstream changes them |
| `KIMI_TOKEN` | *(none)* | `token` cookie from `kimi.com` (enables the Kimi web provider) |
| `COPILOT_COOKIES` | *(none)* | Cookie jar JSON for `copilot.microsoft.com` (anonymous works without it) |
| `PERPLEXITY_COOKIES` | *(none)* | Cookie jar JSON for `perplexity.ai` (anonymous works without it) |
| `GLM_REFRESH_TOKEN` | *(none)* | `refresh_token` from `chatglm.cn` (unlocks glm-4.6 routes; z.ai anonymous always on) |
| `GROQ_API_KEY` / `CEREBRAS_API_KEY` / `MODELSCOPE_API_KEY` | *(none)* | Free API-tier providers (dormant until set) |
| `MISTRAL_API_KEY` | *(none)* | Mistral official free API tier (distinct from `MISTRAL_SESSION_TOKEN`) |
| `OPENROUTER_API_KEY` | *(none)* | OpenRouter `:free` models |
| `GOOGLE_AI_STUDIO_API_KEY` (or `GEMINI_API_KEY`) | *(none)* | Gemini free API tier |
| `COHERE_API_KEY` | *(none)* | Cohere free trial tier |
| `CLOUDFLARE_API_TOKEN` + `CLOUDFLARE_ACCOUNT_ID` | *(none)* | Cloudflare Workers AI |
| `META_API_KEY` / `BLACKBOX_API_KEY` | *(none)* | Meta Muse Spark / Blackbox AI free tiers |
| `<PROVIDER>_COOKIES` | *(none)* | Generic JSON cookie-jar fallback for every provider (e.g. `CLAUDE_COOKIES`, `T3CHAT_COOKIES`, `INNERAI_COOKIES`, `ADAPTA_COOKIES`) |
| `I4F_<PROVIDER>_CONTEXT_LENGTH` / `I4F_<PROVIDER>_MAX_OUTPUT` | *(per-provider defaults)* | Advertised limits for claude/grok/mistral/qwen/kimi/copilot/perplexity/glm routes |
| `I4F_MODELS_TTL` | `300` | Seconds between dynamic model re-discovery across providers |
| `I4F_FALLBACKS` | *(none)* | JSON map of per-model fallback chains, e.g. `{"deepseek-chat": ["deepseek-reasoner"]}` |
| `I4F_DEFAULT_FALLBACKS` | *(none)* | Comma-separated fallbacks applied to every route |
| `I4F_MAX_RETRIES` / `I4F_RETRY_BACKOFF` | `2` / `2.0` | Retries per provider before falling back + exponential backoff base (seconds) |
| `I4F_RETRY_CAP` | `10` | Cap for a single retry wait (honored `Retry-After` included) — lower = faster fallback |
| `I4F_FIRST_TOKEN_TIMEOUT` | `180` | Seconds a provider may take to emit its FIRST chunk before the router gives up on it and falls back immediately (0 disables) |
| `I4F_PROVIDER_STALL_COOLDOWN` | `120` | Seconds a stalled target is skipped by the fallback chain after a first-token timeout |
| `I4F_HTTP_FIRST_TOKEN_TIMEOUT` | `60` | Stricter first-token bound for HTTP providers on the `auto` routers (no browser cold-start excuse) |
| `I4F_AUTO_WALK_BUDGET` | `420` | Seconds one request may walk the fallback chain before failing fast |
| `I4F_STREAM_SILENCE_TIMEOUT` | `180` | Silence that kills a stalled stream |
| `I4F_QUOTA_COOLDOWN_S` | `120` | Seconds a provider sits quota-cooled after a rate limit |
| `I4F_EGRESS_ROTATE` | `copilot,duck` | Providers whose refusals rotate the proxy assignment (egress-shaped blocks) instead of running the credential ladder |
| `I4F_BREAKER_N` / `I4F_BREAKER_COOLDOWN_S` | `3` / `1800` | Consecutive-failure circuit breaker per provider |
| `I4F_PROVIDERS` | *(all)* | Comma-separated provider allowlist — everything else is hidden |
| `I4F_HIDE_TOOLLESS` | `true` | Hide probe-confirmed tool-less models from `/v1/models` |
| `I4F_LLMTRIM` | `true` | Enable the llmtrim request stage |
| `I4F_SELFHEAL` | `true` | Self-heal daemon (probe TTL / trigger / cooldown / max attempts: `I4F_SELFHEAL_PROBE_TTL=600`, `I4F_SELFHEAL_TRIGGER=3`, `I4F_SELFHEAL_COOLDOWN=3600`, `I4F_SELFHEAL_MAX_ATTEMPTS=3`) |
| `I4F_BROWSER_PORT` / `I4F_BROWSER_HEADLESS` / `I4F_BROWSER_IDLE_REAP` | `9333` / *(auto)* / `300` | Shared Chromium manager: debug port, headless override, idle reap |
| `I4F_CHATGPT_RELAY_STALE_AFTER` / `I4F_QWEN_RELAY_STALE_AFTER` | `600` | Seconds after which an idle relay page is recycled (anonymous sessions expire server-side) |
| `I4F_SIGNUP_PROXY` | *(pool)* | Fixed proxy for signup browser flows |
| `I4F_SIGNUP_BREAKER_COOLDOWN_S` | `43200` | Sleep after 3 consecutive signup failures (12 h, not 30 min — a never-succeeding signup must not burn RAM all day) |
| `COOKIES_DIR` | *(none)* (Docker: `/data`) | Directory where provider cookie files are persisted |
| `I4F_PROXY` / `I4F_PROXIES` | *(none)* | Single / comma-separated proxy URLs (always in the pool) |
| `I4F_PROXY_LIST_URL` | *(none)* | URL fetching a dynamic proxy list (text/JSON), TTL-refreshed |
| `I4F_PROXY_LIST_TTL` | `3600` | Seconds between dynamic proxy list refreshes |
| `I4F_PROXY_MODE` | `random` | Proxy selection: `random`, `round`, or `single` |
| `I4F_PROXY_EXCLUDE` | *(none)* | Providers that always go direct (e.g. `deepseek`) |
| `I4F_PROXY_COOLDOWN` | `120` | Seconds a failing proxy is skipped |
| `I4F_PROXY_ROTATE_TTL` | `300` | Seconds a provider keeps its assigned proxy before re-randomizing |
| `I4F_PROXY_AUTO` | `false` | Aggregate public free-proxy lists from the web automatically |
| `I4F_PROXY_SOURCES` | *(built-in)* | Override the auto source list (`socks5=<url>`, ... ) |
| `I4F_PROXY_MAX_POOL` | `250` | Random sample cap for the aggregated pool |
| `I4F_PROXY_LIST_URLS` | *(none)* | Extra list URL(s) (text/JSON) fetched with `I4F_PROXY_LIST_TTL` |
| `I4F_PROXY_CHECK` | `false` | Background health probing; traffic only uses alive proxies |
| `I4F_PROXY_CHECK_TTL` | `1800` | Healthy-lease duration / re-check interval |
| `I4F_PROXY_CHECK_TIMEOUT` | `8` | Per-probe timeout in seconds |
| `I4F_PROXY_CHECK_CONCURRENCY` | `24` | Concurrent health probes |
| `I4F_PROXY_MAX_LATENCY` | `1200` | ms — only proxies answering this fast get traffic (fast-only rotation) |
| `I4F_PROXY_TOP_K` | `5` | Traffic is drawn from the K fastest proxies (by probe + runtime latency) instead of the whole pool |
| `I4F_PROXY_DIRECT` | `true` | Include the no-proxy route in the rotation (`round` mode: direct first) |
| `I4F_PROXY_ENSURE_TIMEOUT` | `90` | Seconds CLI one-shots wait for the warm-up health pass |
| `I4F_ZAI_CONTEXT_LENGTH` | `10000` | Prompt budget (tokens) for the z.ai web transport — its browser input silently fails beyond ~40k characters |
| `I4F_ZAI_PARALLEL` | `1` | Independent z.ai browser sessions (2-4 enable parallel streams; each spawns a Chromium only when requests overlap) |
| `I4F_ZAI_STALE_AFTER` | `600` | s — idle z.ai pages are recycled proactively (anonymous sessions expire server-side) |
| `I4F_ZAI_MAX_OUTPUT` | `2048` | Output reservation advertised for z.ai routes — keeps most of the 10k-token context for the conversation (llmtrim budget) |

**Per-request flags** (chat + image endpoints, non-standard): `"thinking": false` skips a thinking route's reasoning pass (much faster for simple tasks — the playground's *fast mode* toggle sends it), `"search_enabled": true` forces web search, `"disable_proxy": true` connects directly.

---

## 🔄 Credential refresher (default ON)

The service keeps its web sessions alive **autonomously** — no human re-copying cookies. `GET /health` exposes the live state under `refresher`.

**Renewal ladder** (run when a self-heal probe classifies a provider as `auth`, or proactively every `I4F_REFRESHER_TTL`):

1. **HTTP cookie refresh** (always) — Gemini/ChatGPT cookie jars are rotated over plain HTTP; DeepSeek is verified with a live probe (its `userToken` only changes on login).
2. **Headless-browser re-login** (default ON) — re-signs-in with the per-provider login credentials below inside the container's Chromium and exports the fresh token/cookies.
3. **Auto-signup** (default ON, **all providers**) — when even the login is dead — or a provider has **no credentials at all** — a brand-new free account is **created** (the refresher daemon bootstraps every missing provider on its first cycle). If no `<PROVIDER>_LOGIN_EMAIL` is configured, the address is **auto-generated**:
   - your own **catch-all IMAP domain** (`I4F_MAIL_DOMAIN` + IMAP settings) — random local parts, OTP read from your mailbox; or
   - one of **seven throwaway backends** (emailnator real-Gmail addresses, tempmail.lol, tempmail.plus, temp-mail.io, Guerrilla, mail.tm/mail.gw) — zero configuration, picked automatically with fallback.

   Created accounts are persisted to `data/accounts.json` so later renewals can re-login with them. Google/OpenAI may still hit captcha or phone-verification walls — those rungs are **best effort** and their failures surface in the history log.

All rungs respect per-provider cooldowns and daily attempt caps; every action is logged to `data/refresher/history.jsonl` (`python -m dsk.refresher status`, `python -m dsk.refresher bootstrap` to force-create missing credentials now). Everything can be disabled: `I4F_REFRESHER=false`, `I4F_REFRESHER_LOGIN=false`, `I4F_REFRESHER_AUTOSIGNUP=false`, `I4F_MAIL_AUTOGEN=false`.

Bot-written credential files (`data/deepseek_token`, `data/gemini_cookies.json`, `data/chatgpt_cookies.json`) **win over** the env vars — delete a file to hand control back to the environment.

```bash
# environment variables (all shown with their defaults)
I4F_REFRESHER=true
I4F_REFRESHER_TTL=21600        # proactive refresh cycle, seconds
I4F_REFRESHER_COOLDOWN=1800    # per-provider cooldown after an attempt
I4F_REFRESHER_MAX_RENEWS=6     # daily attempt budget per provider
I4F_REFRESHER_LOGIN=true       # rung 2: headless re-login
I4F_REFRESHER_AUTOSIGNUP=true  # rung 3: create fresh accounts
I4F_MAIL_AUTOGEN=true          # auto-create throwaway mailboxes
I4F_MAIL_DOMAIN=               # your catch-all domain (optional; else mail.tm)
I4F_MAIL_IMAP_HOST=            # IMAP mailbox for OTP delivery
I4F_MAIL_IMAP_PORT=993
I4F_MAIL_IMAP_USER=
I4F_MAIL_IMAP_PASS=
I4F_MAIL_OTP_SENDER=deepseek   # sender substring filter
I4F_MAIL_OTP_MAX_AGE=30        # ignore older mail, minutes
I4F_MAIL_OTP_REGEX=\b(\d{6})\b # code extraction

DEEPSEEK_LOGIN_EMAIL=          # optional; autogen e-mail is used when empty
DEEPSEEK_LOGIN_PASSWORD=
GEMINI_LOGIN_EMAIL=
GEMINI_LOGIN_PASSWORD=
CHATGPT_LOGIN_EMAIL=
CHATGPT_LOGIN_PASSWORD=
QWEN_LOGIN_EMAIL=              # register once at chat.qwen.ai; re-login is then automatic
QWEN_LOGIN_PASSWORD=
```

---

## ☁️ Cloudflare cookies

In normal operation cookies are fetched and refreshed automatically. If you hit persistent Cloudflare errors:

1. Run `python -m dsk.bypass` (outside Docker, or with `DOCKERMODE=true` which uses Xvfb). It opens a browser, solves the challenge and writes `dsk/cookies.json`.
2. In Docker the `./data` directory (bind-mounted to `/data`) persists cookies at `/data/cookies.json` across restarts — `dsk/api.py` picks them up automatically.

You only need this when you see Cloudflare challenges, your `cf_clearance` cookie expired, or you get "Please wait a few minutes before trying again".

---

## 🔀 Proxy rotation (rate-limit friendly)

All outbound provider traffic can be routed through one or more **outbound proxies** to spread requests across exit IPs and soften per-IP rate limiting. Configure in `.env`:

```bash
# Any single proxy / comma-separated list (always kept in the pool):
I4F_PROXY=socks5://1.2.3.4:1080
I4F_PROXIES=socks5://1.2.3.4:1080,http://5.6.7.8:8080

# FULLY DYNAMIC: automatically aggregate public free-proxy lists from the web
# (TheSpeedX, monosans, proxifly, proxyscrape, roosterkid, geonode — verified
# working sources), refreshed every 30 min and randomly sampled to 400:
I4F_PROXY_AUTO=true
I4F_PROXY_MAX_POOL=400
# Optional custom sources ("socks5=<url>" sets the scheme):
I4F_PROXY_SOURCES=socks5=https://example.com/socks5.txt
# Optional extra list URL(s) (plain text or JSON):
I4F_PROXY_LIST_URL=https://example.com/proxy-list.txt
```

**Health checking** (`I4F_PROXY_CHECK=true`) — strongly recommended with free lists, where typically only ~10% of published proxies are alive at any moment: a background worker probes every pooled proxy concurrently and **only proxies that answer within `I4F_PROXY_MAX_LATENCY` (1200 ms) receive traffic** — slow exits never slow down responses. Selection is **latency-ranked** (`I4F_PROXY_TOP_K`): the fastest measured proxies get most of the traffic, and a proxy that turns out slow on real payloads is demoted at runtime (EMA of observed latency vs the same budget) until the next health pass re-validates it. Combined with `I4F_PROXY_COOLDOWN`, a proxy that dies mid-session is skipped and traffic falls back to direct, so a dead pool never breaks the service.

**Selection is `random` by default** (`I4F_PROXY_MODE=round|single` also available). The no-proxy route is a first-class rotation candidate (`I4F_PROXY_DIRECT=true`); in `round` mode the direct route comes first. A proxy that fails is put on cooldown (`I4F_PROXY_COOLDOWN`, 120s). Providers that misbehave behind proxies (e.g. bot-protection false positives) can be pinned to direct with `I4F_PROXY_EXCLUDE=deepseek`.

**Per-provider randomization:** every provider (`deepseek`, `gemini`, `chatgpt`, …) gets its *own* proxy, chosen randomly and — while the pool is large enough — distinct from the proxies already used by the other providers, so concurrent providers are spread across different exit IPs instead of hammering one shared proxy. Each assignment is sticky for `I4F_PROXY_ROTATE_TTL` seconds (default 300), then the provider is re-randomized; a runtime failure releases the assignment immediately so the next request picks a fresh random proxy. `GET /health` reports current assignments (`assigned=deepseek->1.2.3.4:8080,...`).

Short-lived CLI one-shots (`python -m dsk.refresher …`) start with an empty pool; `ensure_pool()` refreshes the sources and awaits one bounded health pass (`I4F_PROXY_ENSURE_TIMEOUT`, 90 s) so the first ladder already draws fast, validated exits.

Sanity-check your setup from the host: `python -m dsk.proxies` — it prints each pooled proxy and the exit IP it reaches.

---

## 💻 Local (non-Docker) run

```bash
git clone https://github.com/blastbeng/inference4free.git
cd inference4free
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt "setuptools<81"
DEEPSEEK_AUTH_TOKEN=yourtoken .venv/bin/python -m dsk.openai_server
```

---

## 📚 Original `dsk` library usage

The underlying library from the original repo can also be used directly:

### Basic example

```python
from dsk.api import DeepSeekAPI

api = DeepSeekAPI("YOUR_AUTH_TOKEN")
chat_id = api.create_chat_session()

for chunk in api.chat_completion(chat_id, "What is Python?"):
    if chunk['type'] == 'text':
        print(chunk['content'], end='', flush=True)
```

### Thinking process & web search

```python
for chunk in api.chat_completion(
    chat_id,
    "What are the latest developments in AI?",
    thinking_enabled=True,
    search_enabled=True,
):
    if chunk['type'] == 'thinking':
        print(f"🔍 Thinking: {chunk['content']}")
    elif chunk['type'] == 'text':
        print(chunk['content'], end='', flush=True)
```

### Threaded conversations

```python
chat_id = api.create_chat_session()
parent_id = None
for chunk in api.chat_completion(chat_id, "Tell me about neural networks"):
    if chunk['type'] == 'text':
        print(chunk['content'], end='', flush=True)
    elif 'message_id' in chunk:
        parent_id = chunk['message_id']

for chunk in api.chat_completion(
    chat_id,
    "How do they compare to other ML models?",
    parent_message_id=parent_id,
):
    if chunk['type'] == 'text':
        print(chunk['content'], end='', flush=True)
```

### Error handling

```python
from dsk.api import (
    DeepSeekAPI,
    AuthenticationError,
    RateLimitError,
    NetworkError,
    CloudflareError,
    APIError,
)

try:
    api = DeepSeekAPI("YOUR_AUTH_TOKEN")
    chat_id = api.create_chat_session()
    for chunk in api.chat_completion(chat_id, "Your prompt here"):
        if chunk['type'] == 'text':
            print(chunk['content'], end='', flush=True)
except AuthenticationError:
    print("Authentication failed. Please check your token.")
except RateLimitError:
    print("Rate limit exceeded. Please wait before making more requests.")
except CloudflareError as e:
    print(f"Cloudflare protection encountered: {e}")
except NetworkError:
    print("Network error occurred. Check your internet connection.")
except APIError as e:
    print(f"API error occurred: {e}")
```

---

## 🧪 Tests

```bash
pip install pytest
pytest tests -q
```

21 offline suites (~300 tests, no network, no browser — fake transports only): every provider has one, plus the OpenAI shim contract (SSE shapes, `reasoning_content` separation, 429 `Retry-After`, chain-exhaustion messages), the `auto` routers (pool partition, contracts, reserved ids), the ChatGPT relay thinking-split, the mail backends, the refresher ladder and the credential logic. Each suite also runs standalone: `python tests/test_auto_router.py`.

---

## 🛠️ Troubleshooting

| Symptom | Fix |
|---|---|
| `401` / `invalid_token` | Your `userToken` expired or is wrong — grab a fresh one (step 1 of Quick start) |
| Cloudflare / "Just a moment…" | Run `python -m dsk.bypass` once to refresh `cf_clearance` |
| "Please wait a few minutes before trying again" | DeepSeek rate limiting — wait or switch models |
| Tools never get called / agent misbehaves | Ensure the client sends `tools`; tool calling only activates when tools are present |
| API changes break the client | Update to the latest version — DeepSeek's web API changes frequently |
| `chatgpt relay busy with another stream` | The anonymous relay serves one stream at a time (the web UI is a single chat) — concurrent ChatGPT requests fall back; retry shortly or route through `auto` |
| `fallback chain exhausted: …` | Every provider in the chain failed and the message names each one — fix the first entry (usually credentials or quota), the rest follow |
| 429 with `Retry-After` | Free-tier quota window — honour the header; hammering burns the whole window |
| Thinking shows up as the answer in your client | Your client is reading `delta.content` only — read `delta.reasoning_content` (streaming) / `message.reasoning_content` (non-streaming) for the thinking panel, as OpenWebUI/LiteLLM do |

## ⚠️ Disclaimer

This project uses the providers' **web** interfaces (and free API tiers), not paid APIs. It is intended for personal, educational use. Every upstream may change its interface at any time, may rate-limit or block automated access, and you remain subject to each provider's terms of service.

## 🙏 Credits

- **[xtekky/deepseek4free](https://github.com/xtekky/deepseek4free)** — the original project: reverse-engineered API client, WASM proof-of-work, Cloudflare bypass.
- **[blastbeng/inference4free](https://github.com/blastbeng/inference4free)** — this fork: OpenAI-compatible server, tool-calling emulation, Docker packaging, documentation.
- [aider](https://aider.chat) / [AiderDesk](https://github.com/hotstepper23/aiderdesk) — the agent coding tools this fork targets.
