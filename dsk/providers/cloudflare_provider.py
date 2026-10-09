"""Cloudflare Workers AI provider — free daily neuron allocation.

Workers AI gives every Cloudflare account a free daily allocation
(~10k neurons/day, no credit card): the catalog (@cf/... models:
llama, qwen-coder, mistral, deepseek-r1-distill, gemma, phi...) is
keyed by an Account-scoped API token PLUS the account id. This is the
"free API key" provider class alongside the cookie-jar web providers.

Auth model (BOTH credentials required)
--------------------------------------
    CLOUDFLARE_API_TOKEN env   Account token with Workers AI permission
    CLOUDFLARE_ACCOUNT_ID env  32-hex account identifier
        ->  ``cloudflare`` jar (cloudflare_cookies.json: ``api_key``
            + ``account_id`` fields)
        ->  CLOUDFLARE_COOKIES env JSON fallback

``available()`` is key AND account presence; the refresh rung
verifies liveness with an authenticated models/search call (no
neuron burn).

Endpoints (mixed surface, both documented)
------------------------------------------
    GET  /accounts/{aid}/ai/models/search?per_page=100&page=N
        NATIVE catalog (Bearer) — v4 envelope {result, success,
        result_info:{page,per_page,count,total_count}}. Entries carry
        {name: "@cf/meta/llama-3.1-8b-instruct-fp8", description,
        task: {name: "Text Generation"|"Image-to-Text"|...},
        properties: [{property_id: "context_window", value: "131072"}],
        tags: {author: "meta"}}.
    POST /accounts/{aid}/ai/v1/chat/completions
        OpenAI-compatible SSE (Bearer):
        {model: "@cf/...", messages, stream: true, max_tokens?,
         temperature?} -> standard ``data: {choices:[{delta}]} rockets``
        with ``data: [DONE]``. The compat layer keeps the @cf/ model
        ids in the body, so no URL-encoding is ever needed.
    (image input via standard OpenAI image_url data URIs for the
    Image-to-Text / vision-capable entries)

Catalog filter
--------------
    - task ``Text Generation`` = chat; task ``Image-to-Text`` = the
      vision chat models (llama-3.2-11b-vision, llava) — kept with
      ``vision=True``. Every other task (embeddings, image gen, tts,
      asr, rerank...) — skipped; belt-and-braces id filter covers the
      bare OpenAI-shape fallback.
    - context from the ``context_window`` property; there is no
      per-model max_completion_tokens — module fallback applies.
    - thinking: id regex (deepseek-r1 distills; reasoning/thinking).
    - owned_by from tags.author.

Error shapes (Cloudflare v4 envelope — errors ARRAY, and auth can
arrive as 400, not only 401)
-------------------------------------------------------------------
    400 {"success":false,"errors":[{"code":9106,"message":"Missing
         X-Auth-Key, X-Auth-Email or Authorization headers"}]}
    401 {"success":false,"errors":[{"code":10000,"message":
         "Authentication error"}]}
    404 {"success":false,"errors":[{"code":7003,"message":"Could not
         route to ... perhaps your object identifier is invalid?"}]}
         (wrong account id reads as routing failure)
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

logger = logging.getLogger('dsk.providers.cloudflare')

CF_API_BASE = (os.getenv('I4F_CF_API_BASE', '') or
               'https://api.cloudflare.com/client/v4').rstrip('/')
ACCOUNTS_URL = f'{CF_API_BASE}/accounts'

_USER_AGENT = ('Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 '
               '(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')

_MODELS_TTL = 300.0
CF_CONTEXT_FALLBACK = 8192
CF_MAX_OUTPUT_FALLBACK = 4096
_MAX_CATALOG_PAGES = 5

# chat-capable tasks; Image-to-Text entries are the vision chat models
_CHAT_TASKS = {'text generation', 'image-to-text'}
# belt-and-braces id filter for the bare OpenAI-shape fallback
_RE_NON_CHAT = re.compile(
    r'(?:embed|whisper|stable-|flux|sdxl|dreamshaper|bge|rerank|tts|'
    r'melotts|yolo|resnet|deoldify|img2txt|ocr|udop|nebula|flowfit|'
    r'3d|tarot|bacteria|rnnoise|speech)', re.IGNORECASE)
_RE_THINKING = re.compile(r'(?:deepseek-r1|[-_/]r1\b|reasoning|thinking)',
                          re.IGNORECASE)
# auth-shaped v4 error codes / messages (auth can arrive as 400!)
_RE_AUTH_MSG = re.compile(
    r'(?:authentication|authorization|x-auth-|api token|permission|'
    r'object identifier is invalid)', re.IGNORECASE)


def _api_key() -> str:
    raw = (os.getenv('CLOUDFLARE_API_TOKEN', '') or '').strip()
    if raw:
        return raw
    jar = load_jar('cloudflare') or env_cookies('CLOUDFLARE')
    return (jar.get('api_key') or jar.get('key') or jar.get('token')
            or '').strip()


def _account_id() -> str:
    raw = (os.getenv('CLOUDFLARE_ACCOUNT_ID', '') or '').strip()
    if raw:
        return raw
    jar = load_jar('cloudflare') or {}
    return (jar.get('account_id') or jar.get('account')
            or jar.get('accountid') or '').strip()


def _models_url(account: str, page: int = 1, per_page: int = 100) -> str:
    return (f'{CF_API_BASE}/accounts/{account}/ai/models/search'
            f'?per_page={per_page}&page={page}')


def _chat_url(account: str) -> str:
    return f'{CF_API_BASE}/accounts/{account}/ai/v1/chat/completions'


def _headers(extra: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    hdr = {
        'User-Agent': _USER_AGENT,
        'Accept': '*/*',
        'Authorization': f'Bearer {_api_key()}',
    }
    if extra:
        hdr.update(extra)
    return hdr


def _v4_errors(text: str) -> str:
    """Human summary of a v4 envelope's errors array."""
    try:
        data = json.loads(text or '{}')
        errors = data.get('errors') if isinstance(data, dict) else None
        if isinstance(errors, list) and errors:
            msgs = [str(e.get('message') or e) if isinstance(e, dict)
                    else str(e) for e in errors]
            return '; '.join(msgs)
    except ValueError:
        pass
    return (text or '')[:300]


def _is_auth_error(status: int, text: str) -> bool:
    """v4 auth failures arrive as 400 (9106), 401/403 (10000), or 404
    7003 for a wrong account id (routing precedes auth)."""
    if status in (401, 403):
        return True
    if status in (400, 404) and _RE_AUTH_MSG.search(_v4_errors(text) or ''):
        return True
    return False


def _classify(status: int, text: str,
              headers: Optional[Any] = None) -> ProviderError:
    if _is_auth_error(status, text):
        return ProviderAuthError(
            f'Cloudflare Workers AI credentials rejected (HTTP {status}): '
            f'{_v4_errors(text)[:200]}')
    return classify_http_error(status, text, headers)


def _property_value(entry: Dict[str, Any], prop: str) -> str:
    """Read a model property: properties is a list of
    {property_id, value} in the documented shape; tolerate a dict."""
    props = entry.get('properties')
    if isinstance(props, dict):
        return str(props.get(prop) or '')
    if isinstance(props, list):
        for p in props:
            if isinstance(p, dict) and \
                    str(p.get('property_id') or '').lower() == prop:
                return str(p.get('value') or '')
    return ''


def _parse_models(data: Any) -> List[Dict[str, Any]]:
    """Defensive parse of the v4 catalog (and the bare compat one).

    Native: ``{"result":[{name, task:{name}, properties, tags}]}``.
    OpenAI shim fallback: ``{"data":[{id, owned_by}]}``. Native wins
    when both are present.
    """
    entries: Any = None
    if isinstance(data, dict):
        entries = data.get('result')
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
        mid = str(entry.get('name') or entry.get('id') or '').strip()
        if not mid:
            continue
        task = entry.get('task')
        task_name = ''
        if isinstance(task, dict):
            task_name = str(task.get('name') or '').lower()
        elif isinstance(task, str):
            task_name = task.lower()
        compat_only = 'result' not in (data if isinstance(data, dict)
                                       else {})
        if task_name:
            if task_name not in _CHAT_TASKS:
                continue  # embeddings / image gen / tts / asr ...
        elif compat_only and _RE_NON_CHAT.search(mid):
            continue  # bare compat fallback: id filter only
        tags = entry.get('tags')
        tags = tags if isinstance(tags, dict) else {}
        try:
            context = int(_property_value(entry, 'context_window')
                          or entry.get('context_length') or 0)
        except (TypeError, ValueError):
            context = 0
        try:
            max_out = int(entry.get('max_output_tokens')
                          or entry.get('max_completion_tokens') or 0)
        except (TypeError, ValueError):
            max_out = 0
        vision = (task_name == 'image-to-text' or 'vision' in mid.lower())
        thinking = bool(_RE_THINKING.search(mid))
        out.append({
            'id': mid,
            'owned_by': str(tags.get('author') or 'cloudflare') or 'cloudflare',
            'vision': vision,
            'thinking': thinking,
            'context': context or CF_CONTEXT_FALLBACK,
            'max_out': min(max_out, context) if max_out else
            CF_MAX_OUTPUT_FALLBACK,
        })
    return out


def _fetch_catalog_pages(key: str, account: str, no_proxy: bool = False
                         ) -> Tuple[Optional[List[Any]], Optional[Any]]:
    """Follow the models/search pagination (per_page=100, page N).

    Returns (entries, None) on success or (None, response) when a page
    fails, so the caller can classify the failing response.
    """
    entries: List[Any] = []
    total: Optional[int] = None
    for page in range(1, _MAX_CATALOG_PAGES + 1):
        resp = http_get(_models_url(account, page=page), headers=_headers({
            'Accept': 'application/json'}), timeout=30, no_proxy=no_proxy)
        if resp.status_code != 200:
            return None, resp
        data = resp.json()
        if isinstance(data, dict) and data.get('success') is False:
            return None, resp
        page_entries = data.get('result') if isinstance(data, dict) else data
        if not isinstance(page_entries, list):
            break
        entries.extend(page_entries)
        info = data.get('result_info') if isinstance(data, dict) else {}
        info = info if isinstance(info, dict) else {}
        try:
            total = int(info.get('total_count') or 0)
        except (TypeError, ValueError):
            total = None
        if total is not None and len(entries) >= total:
            break
        if not page_entries:
            break
    return entries, None


class CloudflareProvider(Provider):
    """Workers AI free allocation behind the unified provider contract.

    Dormant until BOTH an API token and the account id are provisioned
    (CLOUDFLARE_API_TOKEN + CLOUDFLARE_ACCOUNT_ID env, the
    ``cloudflare`` jar, or the refresher signup rung which logs into
    dash.cloudflare.com, creates a custom Workers AI token and
    resolves the account id via /accounts).
    """

    name = 'cloudflare'

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._catalog_ts = 0.0
        self._models: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------ auth
    def available(self, auth_key: Optional[str] = None) -> bool:
        return bool(_api_key()) and bool(_account_id())

    # ---------------------------------------------------------------- models
    def _refresh_catalog(self, no_proxy: bool = False) -> None:
        now = time.time()
        with self._lock:
            if self._models and now - self._catalog_ts < _MODELS_TTL:
                return
        try:
            entries, failed = _fetch_catalog_pages(
                _api_key(), _account_id(), no_proxy=no_proxy)
            if entries is not None:
                parsed = _parse_models({'result': entries})
                if parsed:
                    with self._lock:
                        self._models = parsed
                        self._catalog_ts = now
                    return
            status = failed.status_code if failed is not None else '?'
            logger.warning('cloudflare models HTTP %s — keeping %s '
                           'cached models', status, len(self._models))
        except Exception as e:  # noqa: BLE001 — best-effort refresh
            logger.warning('cloudflare models refresh failed: %s', e)

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
                'no Cloudflare Workers AI credentials configured '
                '(CLOUDFLARE_API_TOKEN + CLOUDFLARE_ACCOUNT_ID env or '
                'cloudflare_cookies.json): free accounts get a daily '
                'neuron allocation; create a token at '
                'dash.cloudflare.com/profile/api-tokens')
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
            raise ProviderAuthError(
                'no Cloudflare Workers AI credentials configured')
        if image_generation:
            raise ProviderError(
                'cloudflare has no image generation endpoint')
        self._refresh_catalog(no_proxy=no_proxy)
        entry = self._find_model(model)
        if not entry:
            raise ProviderError(
                f'cloudflare model not in catalog: {model}')

        content: Any = prompt
        if images:
            if not entry.get('vision'):
                raise ProviderError(
                    f'cloudflare model {entry["id"]!r} is not '
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
            _chat_url(_account_id()),
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
        for line in resp.iter_lines():
            data = parse_sse_data(line)
            if not data:
                continue
            # v4 flat errors array or OpenAI nested error frame
            errors = data.get('errors')
            if isinstance(errors, list) and errors:
                raise ProviderError(
                    'cloudflare stream error: '
                    f'{_v4_errors(json.dumps(data))[:300]}')
            error = data.get('error')
            if error:
                message = (error.get('message') if isinstance(error, dict)
                           else str(error)) or json.dumps(error)[:300]
                raise ProviderError(
                    f'cloudflare stream error: {message}')
            if data.get('message') and not data.get('choices') and \
                    data.get('success') is False:
                raise ProviderError(
                    f'cloudflare stream error: {data["message"]}')
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
                'cloudflare stream produced no output')
