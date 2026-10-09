"""Mistral API provider (api.mistral.ai) — free-tier OpenAI-compatible API.

Mistral's official API has a free "Experiment" plan (console.mistral.ai
account -> API keys): shared-capacity access to the current model lineup
(mistral-small/large, magistral, pixtral, devstral, codestral, …) at no
charge, keyed by an opaque token. This is the "free API key" provider
class alongside the cookie-jar web providers — distinct from the
reverse-engineered Le Chat provider (mistral_provider.py).

Auth model
----------
    MISTRAL_API_KEY env
        ->  ``mistral_api`` jar (mistral_api_cookies.json, ``api_key`` field)
        ->  MISTRAL_COOKIES env JSON fallback

``available()`` is just key presence; the refresh rung verifies liveness
with an authenticated GET /v1/models (cheap, no quota burn).

Endpoints (standard OpenAI shapes)
----------------------------------
    GET  /v1/models           AUTH REQUIRED (unlike ModelScope): a bare
                              JSON array of {id, owned_by, capabilities:
                              {completion_chat, vision, completion_fim},
                              max_context_length, deprecation, ...}
    POST /v1/chat/completions
        {model, messages, stream: true, max_tokens?, temperature?}
        -> SSE ``data: {choices:[{delta:{content|reasoning}}]}``, [DONE]

Differences vs Groq/Cerebras/ModelScope
---------------------------------------
    - the catalog is a BARE JSON array (no {"data": ...} wrapper) and
      needs the bearer: a stale key 401s the catalog away (the refresh
      rung then surfaces the rejected-key message).
    - deprecated models carry a truthy ``deprecation`` object — skipped.
    - codestral is completion_fim-only (capabilities.completion_chat is
      False) — no chat endpoint, skipped from the catalog.
    - context comes from ``max_context_length``; there is no per-model
      max_completion_tokens field — the module fallback applies.
    - magistral (and any thinking model) streams reasoning as
      ``delta.reasoning`` AND/OR inline ``<think ...>...`` tags in
      ``content`` — both are mapped to ``thinking`` chunks here (the
      splitter tolerates tags split across SSE chunks, the empty
      ``<think/>`` opener and literal ``<thinker``-style text).

Error shapes: FastAPI flat bodies on both endpoints —
    401 {"detail":"Invalid API Key"} (nothing nested under ``error``).
Free-tier 429s map to ProviderRateLimitError via the shared classifier,
letting the router park the model and fall through.
"""

import base64
import json
import logging
import os
import re
import threading
import time
from typing import Any, Dict, Generator, List, Optional, Tuple

from .base import (
    Provider,
    ProviderAuthError,
    ProviderError,
    ProviderUnavailableError,
    classify_http_error,
    http_get,
    http_post_stream,
    parse_sse_data,
)
from .jar import env_cookies, load_jar

logger = logging.getLogger('dsk.providers.mistral_api')

MISTRAL_API_BASE = (os.getenv('I4F_MISTRAL_API_BASE', '') or
                    'https://api.mistral.ai/v1').rstrip('/')
MODELS_URL = f'{MISTRAL_API_BASE}/models'
CHAT_URL = f'{MISTRAL_API_BASE}/chat/completions'

_USER_AGENT = ('Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 '
               '(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')

_MODELS_TTL = 300.0
MISTRAL_CONTEXT_FALLBACK = 131072
MISTRAL_MAX_OUTPUT_FALLBACK = 8192

# Reasoning-capable families (magistral, *-thinking, deepseek-r1): they
# stream delta.reasoning and/or inline <think ...> tags.
_RE_THINKING = re.compile(r'(?:magistral|.*thinking|deepseek[-_]?r1)',
                          re.IGNORECASE)
# Vision-capable families (pixtral, *-vision ids).
_RE_VISION = re.compile(r'(?:pixtral|vision)', re.IGNORECASE)

# Inline <think ...> ... </think ...> tag handling. Only the tag PREFIX is
# matched so a closing '>' split across chunks (``</think\n>``) is still
# consumed; the empty <think/> form opens the block (magistral convention).
_THINK_OPEN = '<think'
_THINK_CLOSE = '</think'
_CLOSE_TAIL_RE = re.compile(r'\s*>')


def _api_key() -> str:
    raw = (os.getenv('MISTRAL_API_KEY', '') or '').strip()
    if raw:
        return raw
    jar = load_jar('mistral_api') or env_cookies('MISTRAL')
    return (jar.get('api_key') or jar.get('key') or jar.get('token')
            or '').strip()


def _headers(extra: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    hdr = {
        'User-Agent': _USER_AGENT,
        'Accept': '*/*',
        'Authorization': f'Bearer {_api_key()}',
    }
    if extra:
        hdr.update(extra)
    return hdr


def _classify(status: int, text: str,
              headers: Optional[Any] = None) -> ProviderError:
    if status in (401, 403):
        # FastAPI flat body: {"detail":"Invalid API Key"}
        return ProviderAuthError(
            f'mistral API key rejected (HTTP {status}): {(text or "")[:200]}')
    return classify_http_error(status, text, headers)


def _parse_models(data: Any) -> List[Dict[str, Any]]:
    """Defensive parse of the /models payload (bare array or wrapped).

    Deprecated entries (truthy ``deprecation``) and FIM-only models
    (capabilities.completion_chat is False — codestral) are skipped:
    there is no chat endpoint behind them. Context comes from
    ``max_context_length``; max output has no field — module fallback.
    """
    entries: Any = data.get('data') if isinstance(data, dict) else data
    if not isinstance(entries, list):
        return []
    out: List[Dict[str, Any]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        mid = str(entry.get('id') or '').strip()
        if not mid or entry.get('deprecation'):
            continue
        caps = entry.get('capabilities')
        caps = caps if isinstance(caps, dict) else {}
        if caps.get('completion_chat') is False:
            continue  # FIM-only (codestral): no chat endpoint
        try:
            context = int(entry.get('max_context_length')
                          or entry.get('context_length') or 0)
        except (TypeError, ValueError):
            context = 0
        try:
            max_out = int(entry.get('max_completion_tokens')
                          or entry.get('max_tokens') or 0)
        except (TypeError, ValueError):
            max_out = 0
        vision = bool(caps.get('vision')) or bool(_RE_VISION.search(mid))
        out.append({
            'id': mid,
            'owned_by': str(entry.get('owned_by') or 'mistral'),
            'vision': vision,
            'context': context or MISTRAL_CONTEXT_FALLBACK,
            'max_out': min(max_out, context) if max_out else
            MISTRAL_MAX_OUTPUT_FALLBACK,
        })
    return out


class MistralApiProvider(Provider):
    """api.mistral.ai free "Experiment" tier behind the unified contract.

    Dormant until an API key is provisioned (MISTRAL_API_KEY env, the
    ``mistral_api`` jar, or the refresher signup rung which automates
    console.mistral.ai sign-up + key creation).
    """

    name = 'mistral_api'

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._catalog_ts = 0.0
        self._models: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------ auth
    def available(self, auth_key: Optional[str] = None) -> bool:
        return bool(_api_key())

    # ---------------------------------------------------------------- models
    def _refresh_catalog(self, no_proxy: bool = False) -> None:
        now = time.time()
        with self._lock:
            if self._models and now - self._catalog_ts < _MODELS_TTL:
                return
        try:
            # /models is AUTH on Mistral (unlike ModelScope) — fetched WITH
            # the bearer; a stale key answers 401 and the cache survives.
            resp = http_get(MODELS_URL, headers=_headers({
                'Accept': 'application/json'}), timeout=30,
                no_proxy=no_proxy)
            if resp.status_code == 200:
                parsed = _parse_models(resp.json())
                if parsed:
                    with self._lock:
                        self._models = parsed
                        self._catalog_ts = now
                    return
            logger.warning('mistral API models HTTP %s — keeping %s cached '
                           'models', resp.status_code, len(self._models))
        except Exception as e:  # noqa: BLE001 — best-effort refresh
            logger.warning('mistral API models refresh failed: %s', e)

    def _find_model(self, key: str) -> Optional[Dict[str, Any]]:
        for m in self._models:
            if m['id'] == key:
                return m
        lowered = key.lower()
        for m in self._models:
            if m['id'].lower() == lowered or \
                    m['id'].lower().endswith('/' + lowered):
                return m
        return None

    # ------------------------------------------------------- inline <think
    def _split_thinks(self, text: str,
                      state: Dict[str, Any]) -> List[Tuple[str, str]]:
        """Split streamed ``content`` into ('thinking'|'text', chunk) pairs.

        Handles ``<think ...>``/``</think ...>`` tags split across SSE
        chunks, the empty ``<think/>`` opener (opens the block), literal
        ``<think`` text that is not a tag (``<thinker`` …) and unterminated
        blocks (flushed at stream end by :meth:`_flush_thinks`). ``state``
        carries ``{'in_think': bool, 'after_close': bool, 'carry': str}``
        across calls.
        """
        buf = state.get('carry', '') + (text or '')
        state['carry'] = ''
        out: List[Tuple[str, str]] = []

        def _emit(kind: str, seg: str) -> None:
            if seg:
                out.append((kind, seg))

        while buf:
            if state.get('after_close'):
                m = _CLOSE_TAIL_RE.match(buf)
                if m:
                    buf = buf[m.end():]
                    state['after_close'] = False
                    continue
                if not buf.strip():
                    # whitespace only: the closing '>' may still come in
                    # the next chunk — hold it back
                    state['carry'] = buf
                    return out
                state['after_close'] = False
                # fall through: no closing bracket, treat as text
            if state.get('in_think'):
                idx = buf.find(_THINK_CLOSE)
                if idx >= 0:
                    _emit('thinking', buf[:idx])
                    buf = buf[idx + len(_THINK_CLOSE):]
                    state['in_think'] = False
                    state['after_close'] = True
                    continue
                keep = 0
                for n in range(min(len(buf), len(_THINK_CLOSE) - 1), 0, -1):
                    if _THINK_CLOSE.startswith(buf[-n:]):
                        keep = n
                        break
                if keep:
                    _emit('thinking', buf[:-keep])
                    state['carry'] = buf[-keep:]
                else:
                    _emit('thinking', buf)
                return out
            idx = buf.find(_THINK_OPEN)
            if idx < 0:
                keep = 0
                for n in range(min(len(buf), len(_THINK_OPEN) - 1), 0, -1):
                    if _THINK_OPEN.startswith(buf[-n:]):
                        keep = n
                        break
                if keep:
                    _emit('text', buf[:-keep])
                    state['carry'] = buf[-keep:]
                else:
                    _emit('text', buf)
                return out
            _emit('text', buf[:idx])
            rest = buf[idx + len(_THINK_OPEN):]
            if rest.startswith('/>'):
                state['in_think'] = True
                buf = rest[2:]
                continue
            if rest.startswith('>'):
                state['in_think'] = True
                buf = rest[1:]
                continue
            if rest[:1] in ('', ' ', '\n', '\t', '\r'):
                if not rest:
                    # tag boundary is ambiguous — wait for more input
                    state['carry'] = buf[idx:]
                    return out
                gt = rest.find('>')
                if gt >= 0 and '<' not in rest[:gt]:
                    state['in_think'] = True
                    buf = rest[gt + 1:]
                    continue
                if rest.strip() == '':
                    # attributes still streaming in — hold the tag back
                    state['carry'] = buf[idx:]
                    return out
            # literal '<think' that is not a tag (e.g. '<thinker') — text
            _emit('text', _THINK_OPEN)
            buf = rest
            continue
        return out

    def _flush_thinks(self, state: Dict[str, Any]) -> List[Tuple[str, str]]:
        """Flush any held-back carry at finish/end-of-stream."""
        carry = state.get('carry', '')
        out: List[Tuple[str, str]] = []
        if not carry:
            return out
        if state.get('after_close'):
            state['after_close'] = False
            if _CLOSE_TAIL_RE.fullmatch(carry):
                return out  # bare closing bracket — drop
            lead = _CLOSE_TAIL_RE.match(carry)
            if lead:
                carry = carry[lead.end():]
        state['carry'] = ''
        if carry:
            out.append(('thinking' if state.get('in_think') else 'text',
                        carry))
        return out

    def list_models(self, auth_key: Optional[str] = None) -> List[Dict[str, Any]]:
        if not self.available():
            raise ProviderAuthError(
                'no mistral API key configured (MISTRAL_API_KEY env or '
                'mistral_api_cookies.json): free keys come from '
                'console.mistral.ai')
        self._refresh_catalog()
        out: List[Dict[str, Any]] = []
        for m in self._models:
            out.append({
                'id': m['id'],
                'upstream_model': m['id'],
                'thinking_enabled': bool(_RE_THINKING.search(m['id'])),
                'search_enabled': False,
                'vision': m['vision'],
                'image_gen': False,
                'context_length': m['context'],
                'max_output_tokens': m['max_out'],
                'extra': {'owned_by': m['owned_by']},
            })
        return out

    # ---------------------------------------------------------------- stream
    def stream(self, prompt: str, *, model: str, thinking_enabled: bool = False,
               search_enabled: bool = False, temperature: Optional[float] = None,
               max_tokens: Optional[int] = None,
               images: Optional[List[Dict[str, Any]]] = None,
               image_generation: bool = False,
               no_proxy: bool = False,
               auth_key: Optional[str] = None) -> Generator[Dict[str, Any], None, None]:
        if not self.available():
            raise ProviderAuthError('no mistral API key configured')
        if image_generation:
            raise ProviderError(
                'mistral API has no image generation endpoint')
        self._refresh_catalog(no_proxy=no_proxy)
        entry = self._find_model(model)
        if not entry:
            raise ProviderError(f'mistral API model not in catalog: {model}')

        content: Any = prompt
        if images:
            if not entry.get('vision'):
                raise ProviderError(
                    f'mistral API model {entry["id"]!r} is not '
                    'vision-capable')
            parts: List[Dict[str, Any]] = [{'type': 'text', 'text': prompt}]
            for img in images:
                mime = img.get('mime') or 'image/png'
                data = img.get('data') or b''
                if not data:
                    continue
                uri = f'data:{mime};base64,' + base64.b64encode(data).decode()
                parts.append({'type': 'image_url', 'image_url': {'url': uri}})
            if len(parts) > 1:
                content = parts

        payload: Dict[str, Any] = {
            'model': entry['id'],
            'messages': [{'role': 'user', 'content': content}],
            'stream': True,
        }
        # no reasoning opt-in/out parameter exists: magistral streams its
        # reasoning block unconditionally; thinking_enabled is a no-op
        if temperature is not None:
            payload['temperature'] = temperature
        if max_tokens:
            payload['max_tokens'] = max_tokens

        resp = http_post_stream(
            CHAT_URL,
            headers=_headers({'Content-Type': 'application/json',
                              'Accept': 'text/event-stream'}),
            json_body=payload,
            timeout=300,
            no_proxy=no_proxy,
        )
        if resp.status_code != 200:
            try:
                error_text = resp.text or ''
            except Exception:  # pragma: no cover
                error_text = f'HTTP {resp.status_code}'
            raise _classify(resp.status_code, error_text, resp.headers)
        return self._iter_chunks(resp)

    def _iter_chunks(self, resp: Any) -> Generator[Dict[str, Any], None, None]:
        saw_content = False
        state: Dict[str, Any] = {'in_think': False, 'after_close': False,
                                 'carry': ''}
        for line in resp.iter_lines():
            data = parse_sse_data(line)
            if not data:
                continue
            error = data.get('error')
            if error:
                message = (error.get('message') if isinstance(error, dict)
                           else str(error)) or json.dumps(error)[:300]
                raise ProviderError(f'mistral API stream error: {message}')
            if data.get('object') == 'error':
                raise ProviderError(
                    f"mistral API stream error: "
                    f"{data.get('message') or json.dumps(data)[:300]}")
            choices = data.get('choices') or []
            if not choices:
                continue
            delta = (choices[0] or {}).get('delta') or {}
            reasoning = delta.get('reasoning') or delta.get('reasoning_content')
            if reasoning:
                saw_content = True
                yield {'content': str(reasoning), 'type': 'thinking',
                       'finish_reason': None}
            if delta.get('content'):
                saw_content = True
                for kind, seg in self._split_thinks(str(delta['content']),
                                                    state):
                    yield {'content': seg, 'type': kind,
                           'finish_reason': None}
            if choices[0].get('finish_reason'):
                for kind, seg in self._flush_thinks(state):
                    saw_content = True
                    yield {'content': seg, 'type': kind,
                           'finish_reason': None}
                yield {'content': '', 'type': 'text', 'finish_reason': 'stop'}
                return
        for kind, seg in self._flush_thinks(state):
            saw_content = True
            yield {'content': seg, 'type': kind, 'finish_reason': None}
        if saw_content:
            yield {'content': '', 'type': 'text', 'finish_reason': 'stop'}
        else:
            raise ProviderUnavailableError(
                'mistral API stream produced no output')
