"""Dynamic tool-calling capability probing for every served model.

Every model is probed with a canary request (a tiny ``get_weather`` tool)
through the FULL server stack — the same ``/v1/chat/completions`` path real
tool traffic takes, so the taught protocol, the parsers and the provider
transport are all exercised exactly as a client would.

Models that repeatedly answer without producing a tool call are excluded
from ``/v1/models``; excluded models keep being re-probed on the same
schedule, so one that starts working again (quota window reset, fresh
credentials from the renewal bot) reappears on its own. Real traffic feeds
the same state: a tools-request that produced a tool call is a positive
signal, one that answered without calling under ``tool_choice='required'``
is an unambiguous soft failure (K consecutive -> excluded).

State lives in ``<data>/toolcall_state.json`` and survives restarts.
Infra failures (timeouts, 5xx, auth walls) never count toward exclusion:
a muted account or a crashed browser says nothing about tool capability.

Env:
  I4F_TOOLPROBE           enable the background prober (default on)
  I4F_TOOLPROBE_INTERVAL  seconds between full canary cycles (default 21600)
  I4F_TOOLPROBE_BUDGET_S  min seconds between probes of ONE provider (3600)
  I4F_TOOLPROBE_SOFT_K    soft failures before exclusion (default 3)
  I4F_TOOLPROBE_TIMEOUT   canary request timeout (default 120)
  I4F_HIDE_TOOLLESS       hide failed models from /v1/models (default on)
"""
import json
import os
import random
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

_lock = threading.Lock()
_state: Dict[str, Dict[str, Any]] = {}
_loaded = False
_dirty: set = set()          # providers whose models need an immediate probe
_probe_now: set = set()      # model ids forced by /toolcall/reprobe
_pending_dirty_providers: set = set()


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, '') or default)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, '') or default)
    except ValueError:
        return default


def enabled() -> bool:
    return os.getenv('I4F_TOOLPROBE', '1').strip().lower() not in (
        '0', 'false', 'no', 'off')


def interval() -> float:
    return max(300.0, _env_float('I4F_TOOLPROBE_INTERVAL', 21600.0))


def provider_budget_s() -> float:
    return max(60.0, _env_float('I4F_TOOLPROBE_BUDGET_S', 3600.0))


def soft_k() -> int:
    return max(1, _env_int('I4F_TOOLPROBE_SOFT_K', 3))


def probe_timeout() -> float:
    return max(30.0, _env_float('I4F_TOOLPROBE_TIMEOUT', 120.0))


def hide_toolless() -> bool:
    return os.getenv('I4F_HIDE_TOOLLESS', '1').strip().lower() not in (
        '0', 'false', 'no', 'off')


def _data_dir() -> Path:
    d = os.getenv('COOKIES_DIR', '')
    if d and Path(d).is_dir():
        return Path(d)
    if Path('/data').is_dir():
        return Path('/data')
    return Path(__file__).resolve().parent


def _state_path() -> Path:
    return _data_dir() / 'toolcall_state.json'


def _load() -> None:
    global _loaded
    if _loaded:
        return
    try:
        raw = json.loads(_state_path().read_text(encoding='utf-8'))
        models = raw.get('models') if isinstance(raw, dict) else None
        if isinstance(models, dict):
            _state.update({str(k): dict(v) for k, v in models.items()
                           if isinstance(v, dict)})
    except Exception:  # noqa: BLE001 — missing/corrupt state = fresh start
        pass
    _loaded = True


def _save() -> None:
    try:
        tmp = _state_path().with_suffix('.json.tmp')
        tmp.write_text(json.dumps(
            {'models': _state, 'saved_at': time.time()},
            ensure_ascii=False, indent=1), encoding='utf-8')
        tmp.replace(_state_path())
    except Exception:  # noqa: BLE001 — state is best-effort
        pass


def _entry(model_id: str) -> Dict[str, Any]:
    return _state.setdefault(model_id, {
        'status': 'unknown', 'failures': 0,
        'last_checked': 0.0, 'last_ok': 0.0, 'detail': ''})


def capability(model_id: str) -> str:
    """'ok' | 'failed' | 'unknown' for one PUBLIC model id."""
    with _lock:
        _load()
        return str(_entry(model_id).get('status') or 'unknown')


def status_snapshot() -> Dict[str, Any]:
    with _lock:
        _load()
        return {mid: dict(e) for mid, e in _state.items()}


def record_success(model_id: str, detail: str = '') -> None:
    """Real traffic (or a canary) produced a tool call: re-include instantly."""
    if not model_id:
        return
    with _lock:
        _load()
        e = _entry(model_id)
        e.update({'status': 'ok', 'failures': 0,
                  'last_checked': time.time(), 'last_ok': time.time(),
                  'detail': detail or 'tool call produced'})
        _save()


def record_soft_failure(model_id: str, detail: str = '') -> bool:
    """A tools-request answered without calling under tool_choice='required'.

    Returns True when this failure flipped the model to 'failed'."""
    if not model_id:
        return False
    with _lock:
        _load()
        e = _entry(model_id)
        e['failures'] = int(e.get('failures') or 0) + 1
        e['last_checked'] = time.time()
        e['detail'] = (detail or 'no tool call')[:300]
        flipped = False
        if e['failures'] >= soft_k():
            e['status'] = 'failed'
            flipped = True
        _save()
        return flipped


def record_probe(model_id: str, verdict: str, detail: str = '') -> None:
    """Canary outcome: 'ok' re-includes, 'capfail' counts toward exclusion,
    'infra' only refreshes the checked timestamp (capability unproven)."""
    with _lock:
        _load()
        e = _entry(model_id)
        e['last_checked'] = time.time()
        e['detail'] = (detail or verdict)[:300]
        if verdict == 'ok':
            e['status'] = 'ok'
            e['failures'] = 0
            e['last_ok'] = time.time()
        elif verdict == 'capfail':
            e['failures'] = int(e.get('failures') or 0) + 1
            if e['failures'] >= soft_k():
                e['status'] = 'failed'
        # 'infra': status untouched
        _save()


def mark_provider_dirty(provider_name: str) -> None:
    """Credentials were just renewed: drop cached verdicts and re-probe soon.

    Models keep their visibility until the canary runs — a renewal that made
    things WORSE is caught by the next cycle, one that fixed them surfaces
    within minutes instead of a full interval."""
    if not provider_name:
        return
    with _lock:
        _pending_dirty_providers.add(provider_name)
    _kick_event.set()


_kick_event = threading.Event()
_probe_thread: Optional[threading.Thread] = None


def force_probe(model_id: str) -> None:
    """Schedule an immediate canary for one model (/toolcall/reprobe)."""
    with _lock:
        _probe_now.add(model_id)
    _kick_event.set()


def _provider_of(model_id: str) -> str:
    return model_id.split('/', 1)[0].lower() if '/' in model_id else model_id


# --------------------------------------------------------------------- canary
CANARY_TOOL = {
    'type': 'function',
    'function': {
        'name': 'get_weather',
        'description': 'Get the current weather for a city',
        'parameters': {
            'type': 'object',
            'properties': {
                'city': {'type': 'string', 'description': 'City name'},
                'unit': {'type': 'string',
                         'enum': ['celsius', 'fahrenheit']},
            },
            'required': ['city'],
        },
    },
}
CANARY_PROMPT = ('What is the weather in Paris right now? '
                 'Use the get_weather tool.')


def _base_url() -> str:
    port = os.getenv('I4F_PORT', '8000').strip() or '8000'
    return f'http://127.0.0.1:{port}'


def probe_model(model_id: str, timeout: Optional[float] = None
                ) -> Tuple[str, str]:
    """One canary request through the real server. Returns (verdict, detail)
    with verdict in ok | capfail | infra."""
    body = {
        'model': model_id,
        'max_tokens': 300,
        'messages': [{'role': 'user', 'content': CANARY_PROMPT}],
        'tools': [CANARY_TOOL],
    }
    headers = {'Content-Type': 'application/json',
               'X-I4F-Toolprobe': '1'}  # bypasses the /v1 gate for failed
    api_key = os.getenv('I4F_API_KEY', '')
    if api_key:
        headers['Authorization'] = f'Bearer {api_key}'
    req = urllib.request.Request(
        f'{_base_url()}/v1/chat/completions',
        data=json.dumps(body).encode(),
        headers=headers, method='POST')
    try:
        with urllib.request.urlopen(
                req, timeout=timeout or probe_timeout()) as resp:
            raw = resp.read().decode('utf-8', 'replace')
    except urllib.error.HTTPError as e:
        return 'infra', f'HTTP {e.code}'
    except Exception as e:  # noqa: BLE001 — transport trouble is infra
        return 'infra', f'{type(e).__name__}: {e}'
    try:
        d = json.loads(raw)
    except ValueError:
        return 'infra', 'non-JSON response'
    if not isinstance(d, dict) or d.get('error'):
        msg = (d.get('error') or {}).get('message', '') \
            if isinstance(d, dict) else ''
        return 'infra', f'error response: {str(msg)[:150]}'
    try:
        msg = d['choices'][0]['message']
    except (KeyError, IndexError, TypeError):
        return 'infra', 'malformed completion payload'
    calls = msg.get('tool_calls') or []
    if calls:
        name = ''
        try:
            name = calls[0]['function']['name']
        except Exception:  # noqa: BLE001
            pass
        return 'ok', f'canary tool call: {name or "?"}'
    content = (msg.get('content') or '')[:120]
    return 'capfail', f'answered without tool call: {content!r}'


# -------------------------------------------------------------------- prober
def start_prober(get_models: Callable[[], List[str]]) -> bool:
    """Start the background canary loop (idempotent). ``get_models`` returns
    the PUBLIC ids of all leaf models currently served."""
    global _probe_thread
    if not enabled():
        return False
    with _lock:
        if _probe_thread is not None and _probe_thread.is_alive():
            return False
        _probe_thread = threading.Thread(
            target=_loop, args=(get_models,), daemon=True,
            name='toolprobe')
        _probe_thread.start()
        return True


def _loop(get_models: Callable[[], List[str]]) -> None:
    while True:
        try:
            _cycle(get_models)
        except Exception:  # noqa: BLE001 — daemon must never die
            pass
        _kick_event.wait(timeout=max(60.0, interval() / 12.0))
        _kick_event.clear()


def _cycle(get_models: Callable[[], List[str]]) -> None:
    with _lock:
        dirty = set(_pending_dirty_providers)
        _pending_dirty_providers.clear()
        forced = set(_probe_now)
        _probe_now.clear()
    models = []
    try:
        models = list(get_models() or [])
    except Exception:  # noqa: BLE001 — registry not ready yet
        return
    now = time.time()
    due: List[str] = []
    for mid in forced:
        if mid in models:
            due.append(mid)
    dirty_provs = {_provider_of(m) for m in models} & dirty \
        if dirty else set()
    budget = provider_budget_s()
    with _lock:
        _load()
        prov_last: Dict[str, float] = {}
        for mid in models:
            if _is_router_model(mid):
                continue
            e = _entry(mid)
            prov = _provider_of(mid)
            last = max(float(e.get('last_checked') or 0.0),
                       prov_last.get(prov, 0.0))
            prov_last[prov] = last
            interval_due = now - last >= interval()
            fresh_provider = prov in dirty_provs and now - last >= 60.0
            if (interval_due or fresh_provider) and mid not in due:
                due.append(mid)
    if not due:
        return
    random.shuffle(due)
    prov_last: Dict[str, float] = {}
    for mid in due:
        prov = _provider_of(mid)
        with _lock:
            _load()
            last = float(_entry(mid).get('last_checked') or 0.0)
        prov_last[prov] = max(prov_last.get(prov, 0.0), last)
        if mid not in forced and now - prov_last[prov] < budget:
            continue  # provider quota protection: 1 probe/h/provider
        verdict, detail = probe_model(mid)
        record_probe(mid, verdict, detail)
        with _lock:
            _load()
            _entry(mid)['last_checked'] = time.time()
            prov_last[prov] = time.time()
            _save()


def _is_router_model(model_id: str) -> bool:
    return model_id == 'auto' or model_id.endswith('/auto')


# ------------------------------------------------------------ /v1/models glue
def annotate_models(data: List[Dict[str, Any]]
                    ) -> List[Dict[str, Any]]:
    """Set the ``tools`` flag on /v1/models entries and drop toolless ones.

    Leaf models: flag from their own probe state ('failed' -> False).
    Router entries ('auto', '<prefix>/auto'): True while ANY sibling leaf is
    ok or unknown (they re-route at serve time), False only when every
    sibling leaf is known-failed; router entries are never hidden — they can
    still serve plain chat."""
    hide = hide_toolless()
    out: List[Dict[str, Any]] = []
    leaf_status: Dict[str, str] = {}
    for e in data:
        mid = str(e.get('id') or '')
        if not _is_router_model(mid):
            leaf_status[mid] = capability(mid)
    for e in data:
        mid = str(e.get('id') or '')
        if _is_router_model(mid):
            prefix = mid[:-len('/auto')] if mid != 'auto' else ''
            statuses = [s for m, s in leaf_status.items()
                        if (m.split('/', 1)[0] == prefix) or not prefix]
            e['tools'] = (True if any(s != 'failed' for s in statuses)
                          else (False if statuses else 'unknown'))
            out.append(e)
            continue
        st = leaf_status.get(mid, 'unknown')
        e['tools'] = {'ok': True, 'failed': False}.get(st, 'unknown')
        if hide and st == 'failed':
            continue
        out.append(e)
    return out
