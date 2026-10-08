"""Anonymous Pollinations.ai provider: keyless text chat + image generation.

Pollinations serves an anonymous tier that needs NO credentials at all —
no signup, no cookies, no keys — so like ``duck`` it is always available
and the refresher bot has nothing to renew. Two live discovery endpoints
drive the exposed model list (nothing is hardcoded; the rotating anonymous
catalog is picked up on every TTL refresh):

    GET https://text.pollinations.ai/models     text/vision chat models
        [{name, description, reasoning, tier, input_modalities,
          output_modalities, tools, aliases, vision, audio}, …]
    GET https://image.pollinations.ai/models    image-generation models
        ["sana", "flux", …]

Text streams through their OpenAI-compatible shim (SSE, OpenAI chunk
shape — the thinking deltas arrive as ``delta.reasoning``, not
``reasoning_content``):

    POST https://text.pollinations.ai/openai
        {"model": "openai-fast",
         "messages": [{"role": "user",
                       "content": [{"type": "text", "text": …},
                                    {"type": "image_url", …}]}],
         "stream": true}

Vision: text models that advertise ``image`` in ``input_modalities``
accept OpenAI ``image_url`` parts; the router's byte attachments are
wrapped as ``data:`` URIs.

Image generation renders on a plain GET (the URL itself is the image —
stable and seeded, so /v1/images/generations can hand it out directly and
the server can fetch the bytes for ``b64_json``):

    GET https://image.pollinations.ai/prompt/<urlencoded prompt>
        ?model=sana&width=1024&height=1024&seed=<n>&nologo=true
"""

import base64
import logging
import os
import random
import threading
import time
from typing import Any, Dict, Generator, List, Optional
from urllib.parse import quote

from .base import (
    Provider,
    ProviderError,
    ProviderRateLimitError,
    ProviderUnavailableError,
    classify_http_error,
    http_get,
    http_post_stream,
    parse_sse_data,
)

logger = logging.getLogger('dsk.providers.pollinations')

# Pollinations answers HTTP 402 (HTML "Payment Required") when the anonymous
# tier's per-IP budget is gone or a model needs a seed token — a QUOTA
# condition, not a code bug: mapped to a rate limit so the router applies
# its provider-wide cooldown instead of re-paying the same 402 model by model.
_TIER_EXHAUSTED_STATUSES = {402}


def _classify(status: int, text: str, headers: Optional[Any] = None) -> ProviderError:
    if status in _TIER_EXHAUSTED_STATUSES:
        return ProviderRateLimitError(
            f'pollinations anonymous tier exhausted (HTTP {status}): '
            f'{(text or "")[:200]}')
    return classify_http_error(status, text, headers)


TEXT_BASE_URL = 'https://text.pollinations.ai'
TEXT_CHAT_URL = f'{TEXT_BASE_URL}/openai'
TEXT_MODELS_URL = f'{TEXT_BASE_URL}/models'
IMAGE_BASE_URL = 'https://image.pollinations.ai'
IMAGE_MODELS_URL = f'{IMAGE_BASE_URL}/models'

_USER_AGENT = ('Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 '
               '(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')

# The anonymous catalog rotates: cache it briefly so /v1/models refreshes
# and per-request discovery don't hammer the free endpoint.
_MODELS_TTL = 300.0

# Last-resort catalogs (verified live 2026-10): used only when BOTH
# discovery endpoints fail so the provider still surfaces something.
_FALLBACK_TEXT_MODELS: List[Dict[str, Any]] = [
    {'id': 'openai-fast', 'name': 'GPT-OSS 20B (anonymous tier)',
     'thinking_enabled': True, 'vision': False, 'image_gen': False},
]
_FALLBACK_IMAGE_MODELS: List[str] = ['flux']

# Image generation defaults (Pollinations renders on request; these match
# the OpenAI Images API square default).
_IMAGE_WIDTH = int(os.getenv('I4F_POLLINATIONS_IMAGE_WIDTH', '1024'))
_IMAGE_HEIGHT = int(os.getenv('I4F_POLLINATIONS_IMAGE_HEIGHT', '1024'))


class PollinationsProvider(Provider):
    """Keyless Pollinations.ai models behind the unified provider contract."""

    name = 'pollinations'

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._catalog_ts = 0.0
        self._text_models: List[Dict[str, Any]] = []
        self._image_models: List[str] = []

    # ------------------------------------------------------------------ auth
    def available(self, auth_key: Optional[str] = None) -> bool:
        """Anonymous provider: always available (the router health-probes it)."""
        return True

    # ---------------------------------------------------------------- models
    def _refresh_catalog(self, no_proxy: bool = False) -> None:
        """Re-read both discovery endpoints into the TTL cache (best-effort:
        a failing endpoint keeps the previously known models)."""
        text_models: List[Dict[str, Any]] = []
        try:
            resp = http_get(TEXT_MODELS_URL, timeout=20, no_proxy=no_proxy)
            if resp.status_code == 200:
                for entry in resp.json() or []:
                    if not isinstance(entry, dict):
                        continue
                    model_id = str(entry.get('name') or '').strip()
                    if not model_id:
                        continue
                    # Seed/authenticated tiers answer 402 for us — anonymous only.
                    tier = str(entry.get('tier') or 'anonymous').strip().lower()
                    if tier not in ('', 'anonymous', 'seed-optional'):
                        continue
                    modalities_in = [str(m).lower() for m in
                                     (entry.get('input_modalities') or [])]
                    modalities_out = [str(m).lower() for m in
                                      (entry.get('output_modalities') or [])]
                    text_models.append({
                        'id': model_id,
                        'name': str(entry.get('description') or model_id),
                        'thinking_enabled': bool(entry.get('reasoning')),
                        'search_enabled': False,
                        'vision': ('image' in modalities_in
                                   or bool(entry.get('vision'))),
                        'image_gen': 'image' in modalities_out,
                        'context_length': 131072,
                        'max_output_tokens': 32768,
                    })
        except Exception as e:  # noqa: BLE001 — catalog is best-effort
            logger.debug('pollinations text catalog unavailable: %s', e)

        image_models: List[str] = []
        try:
            resp = http_get(IMAGE_MODELS_URL, timeout=20, no_proxy=no_proxy)
            if resp.status_code == 200:
                for entry in resp.json() or []:
                    if isinstance(entry, str) and entry.strip():
                        image_models.append(entry.strip())
                    elif isinstance(entry, dict) and str(entry.get('name') or '').strip():
                        image_models.append(str(entry['name']).strip())
        except Exception as e:  # noqa: BLE001 — catalog is best-effort
            logger.debug('pollinations image catalog unavailable: %s', e)

        with self._lock:
            if text_models:
                self._text_models = text_models
            if image_models:
                self._image_models = image_models
            self._catalog_ts = time.monotonic()

    def _catalog(self, no_proxy: bool = False) -> tuple:
        now = time.monotonic()
        if not self._text_models and not self._image_models \
                or now - self._catalog_ts > _MODELS_TTL:
            self._refresh_catalog(no_proxy=no_proxy)
        with self._lock:
            text = list(self._text_models) or [dict(m) for m in _FALLBACK_TEXT_MODELS]
            images = list(self._image_models) or list(_FALLBACK_IMAGE_MODELS)
        return text, images

    def list_models(self, auth_key: Optional[str] = None) -> List[Dict[str, Any]]:
        """Live anonymous catalog: text/vision models + image-generation models."""
        text_models, image_models = self._catalog()
        # Text models keep their bare id ('pollinations/openai-fast' publicly);
        # image routes carry an 'img:' upstream prefix so a text model sharing
        # the same name can never be dispatched to the image path (and vice
        # versa); the '-image' public suffix keeps both ids reachable.
        text_ids = {m['id'] for m in text_models}
        entries: List[Dict[str, Any]] = []
        for m in text_models:
            entries.append({
                'id': m['id'],
                'upstream_model': m['id'],
                'thinking_enabled': bool(m.get('thinking_enabled')),
                'search_enabled': bool(m.get('search_enabled')),
                'vision': bool(m.get('vision')),
                'image_gen': bool(m.get('image_gen')),
                'context_length': int(m.get('context_length') or 131072),
                'max_output_tokens': int(m.get('max_output_tokens') or 32768),
            })
        for name in image_models:
            model_id = name if name not in text_ids else f'{name}-image'
            entries.append({
                'id': model_id,
                'upstream_model': f'img:{name}',
                'thinking_enabled': False,
                'search_enabled': False,
                'vision': False,
                'image_gen': True,
                'context_length': 8192,
                'max_output_tokens': 0,
                'extra': {'image_model': True},
            })
        if not entries:
            raise ProviderUnavailableError('pollinations catalog is empty')
        return entries

    # ---------------------------------------------------------------- stream
    def stream(self, prompt: str, *, model: str, thinking_enabled: bool = False,
               search_enabled: bool = False, temperature: Optional[float] = None,
               max_tokens: Optional[int] = None,
               images: Optional[List[Dict[str, Any]]] = None,
               image_generation: bool = False,
               no_proxy: bool = False,
               auth_key: Optional[str] = None) -> Generator[Dict[str, Any], None, None]:
        text_models, image_models = self._catalog(no_proxy=no_proxy)
        # image routes carry an 'img:' upstream prefix (see list_models) —
        # membership in the image catalog alone is ambiguous when a text
        # model shares the name
        wants_image = image_generation or model.startswith('img:')
        if wants_image:
            image_name = model[4:] if model.startswith('img:') else model
            image_model = image_name if image_name in image_models else \
                (image_models[0] if image_models else 'flux')
            yield from self._stream_image(prompt, image_model, no_proxy=no_proxy)
            return
        if images:
            # second line of defense behind the router's capability filter:
            # a stale vision flag would otherwise let Pollinations silently
            # ignore the attachments
            entry = next((m for m in text_models if m['id'] == model), None)
            if entry is not None and not entry.get('vision'):
                raise ProviderError(
                    f"pollinations model '{model}' has no vision input — "
                    'attach images only to vision models')
        yield from self._stream_text(prompt, model, thinking_enabled,
                                     temperature, max_tokens, images,
                                     no_proxy=no_proxy)

    # ------------------------------------------------------------ image path
    def _stream_image(self, prompt: str, image_model: str,
                      no_proxy: bool = False) -> Generator[Dict[str, Any], None, None]:
        """Render the image, then hand out its canonical seeded URL.

        The URL is deterministic (same prompt+seed → same image) and
        keyless, so it can be embedded in markdown or downloaded later by
        /v1/images/generations for ``b64_json``.
        """
        seed = random.randint(1, 2_147_483_647)
        url = (f'{IMAGE_BASE_URL}/prompt/{quote(prompt, safe="")}'
               f'?width={_IMAGE_WIDTH}&height={_IMAGE_HEIGHT}'
               f'&model={quote(image_model)}&seed={seed}&nologo=true')
        try:
            resp = http_get(url, timeout=300, no_proxy=no_proxy)
        except Exception as e:  # noqa: BLE001 — transport → typed error
            raise ProviderUnavailableError(
                f'pollinations image render failed: {e}') from e
        if resp.status_code != 200:
            raise _classify(resp.status_code,
                            (getattr(resp, 'text', '') or '')[:400],
                            getattr(resp, 'headers', None))
        # an HTML/JSON error page can slip through a bare 200 — sniff the
        # magic bytes (jpeg/png/webp/gif) before handing the URL out
        head = (resp.content or b'')[:12]
        if not (head.startswith(b'\xff\xd8') or head.startswith(b'\x89PNG')
                or head.startswith(b'RIFF') or head.startswith(b'GIF8')):
            raise ProviderUnavailableError(
                'pollinations returned a non-image body for the render')
        yield {'content': f'![image]({url})', 'type': 'image', 'url': url,
               'finish_reason': None}
        yield {'content': '', 'type': 'text', 'finish_reason': 'stop'}

    # ------------------------------------------------------------- text path
    def _payload(self, prompt: str, model: str, thinking_enabled: bool,
                 temperature: Optional[float], max_tokens: Optional[int],
                 images: Optional[List[Dict[str, Any]]]) -> Dict[str, Any]:
        image_parts: List[Dict[str, Any]] = []
        if images:
            for img in images:
                mime = img.get('mime') or 'image/png'
                data = img.get('data') or b''
                if not data:
                    continue
                uri = f'data:{mime};base64,' + base64.b64encode(data).decode()
                image_parts.append({'type': 'image_url',
                                    'image_url': {'url': uri}})
        if image_parts:
            # multimodal shape is required to carry attachments
            content: Any = [{'type': 'text', 'text': prompt}] + image_parts
        else:
            # text-only: the plain string form is what the upstream's OpenAI
            # shim actually serves — the parts-array form hangs the text-only
            # backends (measured live: string 200 first-token 0.2s, array
            # 0 bytes for 46s until the client gave up)
            content = prompt
        payload: Dict[str, Any] = {
            'model': model,
            'messages': [{'role': 'user', 'content': content}],
            'stream': True,
        }
        if temperature is not None:
            payload['temperature'] = temperature
        if max_tokens:
            payload['max_tokens'] = max_tokens
        return payload

    def _stream_text(self, prompt: str, model: str, thinking_enabled: bool,
                     temperature: Optional[float], max_tokens: Optional[int],
                     images: Optional[List[Dict[str, Any]]],
                     no_proxy: bool = False) -> Generator[Dict[str, Any], None, None]:
        response = http_post_stream(
            TEXT_CHAT_URL,
            headers={'Content-Type': 'application/json',
                     'Accept': 'text/event-stream',
                     'User-Agent': _USER_AGENT},
            json_body=self._payload(prompt, model, thinking_enabled,
                                    temperature, max_tokens, images),
            timeout=600, no_proxy=no_proxy)
        if response.status_code != 200:
            try:
                error_text = response.text or ''
            except Exception:  # pragma: no cover
                error_text = f'HTTP {response.status_code}'
            raise _classify(response.status_code, error_text,
                            response.headers)
        saw_content = False
        for line in response.iter_lines():
            obj = parse_sse_data(line if isinstance(line, bytes)
                                 else str(line).encode())
            if not obj:
                continue
            choices = obj.get('choices') or []
            delta: Dict[str, Any] = {}
            if choices:
                first = choices[0] or {}
                delta = first.get('delta') or {}
            reasoning = delta.get('reasoning') or delta.get('reasoning_content')
            if reasoning:
                saw_content = True
                yield {'content': reasoning, 'type': 'thinking',
                       'finish_reason': None}
            if delta.get('content'):
                saw_content = True
                yield {'content': delta['content'], 'type': 'text',
                       'finish_reason': None}
            if choices and choices[0].get('finish_reason'):
                break
        if not saw_content:
            # 200 with zero deltas: the upstream answered but produced no
            # tokens — raising (instead of yielding a bare stop) lets the
            # router retry / fall back instead of returning a silent 200.
            raise ProviderUnavailableError(
                f'pollinations stream for {model!r} ended without any content')
        yield {'content': '', 'type': 'text', 'finish_reason': 'stop'}
