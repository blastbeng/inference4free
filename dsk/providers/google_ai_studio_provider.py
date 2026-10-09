"""Google AI Studio provider (generativelanguage.googleapis.com) — free-tier API.

Google AI Studio hands out free API keys (aistudio.google.com/apikey)
with per-model free-tier rate limits; the keys are plain ``AIzaSy...``
tokens accepted by the Gemini API. This is the API-key sibling of the
cookie-jar ``gemini`` web provider — same Google account, different
transport (no anti-bot scraping, just the documented REST surface).

Auth model
----------
    GOOGLE_AI_STUDIO_API_KEY env
        (fallback: GEMINI_API_KEY — the official SDK name)
        ->  ``google_ai_studio`` jar (google_ai_studio_cookies.json,
            ``api_key`` field)
        ->  GOOGLE_AI_STUDIO_COOKIES env JSON fallback

``available()`` is just key presence; the refresh rung verifies
liveness with an authenticated native ListModels call (no quota burn).

Endpoints (mixed surface, both documented)
------------------------------------------
    GET  v1beta/models        NATIVE catalog — needs the key via
                              ``x-goog-api-key``. Entries carry the
                              useful metadata the OpenAI shim hides:
                              {name: "models/gemini-2.5-flash",
                              displayName, description, inputTokenLimit,
                              outputTokenLimit, supportedGenerationMethods}.
                              Paginated via nextPageToken (default 50/page).
    POST v1beta/openai/chat/completions
        OpenAI-compatible SSE (Authorization: Bearer):
        {model, messages, stream: true, max_tokens?, temperature?}
        -> ``data: {choices:[{delta:{content|reasoning}}]}``, [DONE]
    (image input via standard OpenAI image_url data URIs)

Catalog filter
--------------
    - native entries without ``generateContent`` in
      supportedGenerationMethods are not chat models (embeddings,
      imagen, veo, tts, live) — skipped.
    - belt-and-braces id filter for the bare OpenAI-shape fallback
      (embedding|aqa|imagen|veo|lyria|tts|native-audio|live).
    - ``vision``: every gemini-* generation is multimodal; desc regex
      fallback for the compat shape. ``thinking``: gemini-2.5/3 (+ the
      -thinking experiments and omni) think by default — id/desc regex.
    - context from inputTokenLimit, max output from outputTokenLimit
      (clamped to context); module fallbacks otherwise.

Error shapes (single flat ``error`` object; the OpenAI shim may wrap
it in a JSON array)
-------------------------------------------------------------------
    403 {"error":{"code":403,"message":"Method doesn't allow unregistered
         callers...","status":"PERMISSION_DENIED"}}          (no key)
    400 {"error":{"code":400,"message":"API key not valid. Please pass a
         valid API key.","status":"INVALID_ARGUMENT"}}       (bad key)
    400 [{"error":{"code":400,"message":"Missing or invalid
         Authorization header.","status":"INVALID_ARGUMENT"}}]  (compat)
Note auth failures arrive as 400 (not 401) — classified by message.
Free-tier 429s map to ProviderRateLimitError via the shared
classifier, letting the router park the model and fall through.
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

logger = logging.getLogger('dsk.providers.google_ai_studio')

GAS_API_BASE = (os.getenv('I4F_GOOGLE_AI_STUDIO_API_BASE', '') or
                'https://generativelanguage.googleapis.com').rstrip('/')
NATIVE_MODELS_URL = f'{GAS_API_BASE}/v1beta/models'
OPENAI_CHAT_URL = f'{GAS_API_BASE}/v1beta/openai/chat/completions'

_USER_AGENT = ('Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 '
               '(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')

_MODELS_TTL = 300.0
GAS_CONTEXT_FALLBACK = 131072
GAS_MAX_OUTPUT_FALLBACK = 8192
_MAX_CATALOG_PAGES = 5

# Auth failures arrive as 400 with key-specific messages (or 403
# PERMISSION_DENIED for the keyless native call) — classified by text.
_RE_AUTH_MSG = re.compile(
    r'(?:API key not valid|Please pass a valid API key|'
    r'Missing or invalid Authorization|unregistered callers|'
    r'API_KEY_INVALID|PERMISSION_DENIED)', re.IGNORECASE)

# id/description fallbacks when the catalog shape is the bare OpenAI one
_RE_NON_CHAT = re.compile(
    r'(?:embedding|aqa|imagen|veo|lyria|tts|native-audio|-live|robotics)',
    re.IGNORECASE)
_RE_THINKING = re.compile(r'(?:gemini-(?:2\.5|3)|-thinking|omni)',
                          re.IGNORECASE)


def _api_key() -> str:
    raw = ((os.getenv('GOOGLE_AI_STUDIO_API_KEY', '') or
            os.getenv('GEMINI_API_KEY', '') or '')).strip()
    if raw:
        return raw
    jar = load_jar('google_ai_studio') or env_cookies('GOOGLE_AI_STUDIO')
    return (jar.get('api_key') or jar.get('key') or jar.get('token')
            or '').strip()


def _classify(status: int, text: str,
              headers: Optional[Any] = None) -> ProviderError:
    if status in (401, 403) or (status == 400 and
                                _RE_AUTH_MSG.search(text or '')):
        # {"error":{"message":"API key not valid. Please pass a valid
        #  API key.","status":"INVALID_ARGUMENT"}}
        return ProviderAuthError(
            f'Google AI Studio API key rejected (HTTP {status}): '
            f'{(text or "")[:200]}')
    return classify_http_error(status, text, headers)


def _strip_model_prefix(mid: str) -> str:
    return mid[len('models/'):] if mid.startswith('models/') else mid


def _parse_models(data: Any) -> List[Dict[str, Any]]:
    """Defensive parse of both catalog shapes.

    Native ListModels: ``{"models":[{name, displayName, description,
    inputTokenLimit, outputTokenLimit, supportedGenerationMethods}]}``.
    OpenAI shim fallback: ``{"data":[{id, owned_by}]}`` (bare). Native
    wins when both are present.
    """
    entries: Any = None
    if isinstance(data, dict):
        entries = data.get('models')
        if not isinstance(entries, list):
            entries = data.get('data')
    elif isinstance(data, list):
        entries = data
    if not isinstance(entries, list):
        return []
    out: List[Dict[str, Any]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        mid = _strip_model_prefix(str(entry.get('name')
                                      or entry.get('id') or '').strip())
        if not mid:
            continue
        methods = entry.get('supportedGenerationMethods')
        if isinstance(methods, list) and \
                'generateContent' not in [str(m) for m in methods]:
            continue  # embeddings/imagen/veo/tts/live — not chat models
        if _RE_NON_CHAT.search(mid):
            continue
        desc = (f'{entry.get("displayName") or ""} '
                f'{entry.get("description") or ""}').lower()
        try:
            context = int(entry.get('inputTokenLimit')
                          or entry.get('context_length') or 0)
        except (TypeError, ValueError):
            context = 0
        try:
            max_out = int(entry.get('outputTokenLimit')
                          or entry.get('max_completion_tokens') or 0)
        except (TypeError, ValueError):
            max_out = 0
        vision = (mid.startswith('gemini') or 'image' in desc
                  or 'multimodal' in desc or 'vision' in desc)
        thinking = (bool(_RE_THINKING.search(mid)) or 'thinking' in desc
                    or 'thought' in desc)
        out.append({
            'id': mid,
            'owned_by': str(entry.get('owned_by') or 'google') or 'google',
            'vision': vision,
            'thinking': thinking,
            'context': context or GAS_CONTEXT_FALLBACK,
            'max_out': min(max_out, context) if max_out else
            GAS_MAX_OUTPUT_FALLBACK,
        })
    return out


def _fetch_catalog_pages(key: str, no_proxy: bool = False
                         ) -> Tuple[Optional[List[Any]], Optional[Any]]:
    """Follow the ListModels pagination (default 50/page, capped).

    Returns (entries, None) on success or (None, response) when a page
    fails, so the caller can classify the failing response.
    """
    entries: List[Any] = []
    token = ''
    for _ in range(_MAX_CATALOG_PAGES):
        url = f'{NATIVE_MODELS_URL}?pageSize=50'
        if token:
            url += f'&pageToken={token}'
        resp = http_get(url, headers={
            'x-goog-api-key': key,
            'User-Agent': _USER_AGENT,
            'Accept': 'application/json'}, timeout=30, no_proxy=no_proxy)
        if resp.status_code != 200:
            return None, resp
        data = resp.json()
        page = data.get('models') if isinstance(data, dict) else data
        if isinstance(page, list):
            entries.extend(page)
        token = str(data.get('nextPageToken') or '') \
            if isinstance(data, dict) else ''
        if not token:
            break
    return entries, None


def _parse_sse_payload(line: bytes) -> Dict[str, Any]:
    """parse_sse_data drops non-dict payloads; the Google OpenAI shim
    wraps some error frames in a JSON array — recover those locally
    instead of changing the shared parser's contract."""
    data = parse_sse_data(line)
    if isinstance(data, dict):
        return data
    try:
        text = line.decode('utf-8', 'ignore').strip()
    except AttributeError:
        text = str(line or '').strip()
    if not text.startswith('data:'):
        return {}
    payload = text[5:].strip()
    if not payload or payload == '[DONE]':
        return {}
    try:
        obj = json.loads(payload)
    except ValueError:
        return {}
    if isinstance(obj, list):
        obj = obj[0] if obj and isinstance(obj[0], dict) else {}
    return obj if isinstance(obj, dict) else {}


class GoogleAiStudioProvider(Provider):
    """Google AI Studio free tier behind the unified provider contract.

    Dormant until an API key is provisioned (GOOGLE_AI_STUDIO_API_KEY
    env, the ``google_ai_studio`` jar, or the refresher signup rung
    which logs into Google and harvests a key from
    aistudio.google.com/apikey).
    """

    name = 'google_ai_studio'

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
            entries, failed = _fetch_catalog_pages(_api_key(),
                                                   no_proxy=no_proxy)
            if entries is not None:
                parsed = _parse_models({'models': entries})
                if parsed:
                    with self._lock:
                        self._models = parsed
                        self._catalog_ts = now
                    return
            status = failed.status_code if failed is not None else '?'
            logger.warning('google_ai_studio models HTTP %s — keeping %s '
                           'cached models', status, len(self._models))
        except Exception as e:  # noqa: BLE001 — best-effort refresh
            logger.warning('google_ai_studio models refresh failed: %s', e)

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

    def list_models(self, auth_key: Optional[str] = None) -> List[Dict[str, Any]]:
        if not self.available():
            raise ProviderAuthError(
                'no Google AI Studio API key configured '
                '(GOOGLE_AI_STUDIO_API_KEY env or '
                'google_ai_studio_cookies.json): free keys come from '
                'aistudio.google.com/apikey')
        self._refresh_catalog()
        out: List[Dict[str, Any]] = []
        for m in self._models:
            out.append({
                'id': m['id'],
                'upstream_model': m['id'],
                'thinking_enabled': m['thinking'],
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
            raise ProviderAuthError('no Google AI Studio API key configured')
        if image_generation:
            raise ProviderError(
                'google_ai_studio has no image generation endpoint')
        self._refresh_catalog(no_proxy=no_proxy)
        entry = self._find_model(model)
        if not entry:
            raise ProviderError(
                f'google_ai_studio model not in catalog: {model}')

        content: Any = prompt
        if images:
            if not entry.get('vision'):
                raise ProviderError(
                    f'google_ai_studio model {entry["id"]!r} is not '
                    'vision-capable')
            parts: List[Dict[str, Any]] = [{'type': 'text', 'text': prompt}]
            for img in images:
                mime = img.get('mime') or 'image/png'
                data = img.get('data') or b''
                if not data:
                    continue
                uri = f'data:{mime};base64,' + base64.b64encode(data).decode()
                parts.append({'type': 'image_url',
                              'image_url': {'url': uri}})
            if len(parts) > 1:
                content = parts

        payload: Dict[str, Any] = {
            'model': entry['id'],
            'messages': [{'role': 'user', 'content': content}],
            'stream': True,
        }
        if temperature is not None:
            payload['temperature'] = temperature
        if max_tokens:
            payload['max_tokens'] = max_tokens

        resp = http_post_stream(
            OPENAI_CHAT_URL,
            headers={'Authorization': f'Bearer {_api_key()}',
                     'User-Agent': _USER_AGENT,
                     'Content-Type': 'application/json',
                     'Accept': 'text/event-stream'},
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
        for line in resp.iter_lines():
            data = _parse_sse_payload(line)
            if not data:
                continue
            error = data.get('error')
            if error:
                message = (error.get('message') if isinstance(error, dict)
                           else str(error)) or json.dumps(error)[:300]
                raise ProviderError(
                    f'google_ai_studio stream error: {message}')
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
                yield {'content': str(delta['content']), 'type': 'text',
                       'finish_reason': None}
            if choices[0].get('finish_reason'):
                yield {'content': '', 'type': 'text', 'finish_reason': 'stop'}
                return
        if saw_content:
            yield {'content': '', 'type': 'text', 'finish_reason': 'stop'}
        else:
            raise ProviderUnavailableError(
                'google_ai_studio stream produced no output')
