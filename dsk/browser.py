"""One shared Chromium for the whole service (RAM rule).

Historically every consumer spawned its own Chromium (~300-500 MB each:
chatgpt relay, qwen relay, z.ai pool sessions, every signup/refresh rung).
This module gives all of them TABS of a single browser process instead.

Design
------
* One :class:`_SharedBrowser` per ``(profile, proxy)`` key. Consumers that
  need a persistent login profile (chatgpt relay + its login rung) key on
  their profile dir; everyone else shares the default instance, which keeps
  a persistent profile at ``I4F_BROWSER_PROFILE`` (Cloudflare/Google score
  returning browsers far higher).
* :func:`acquire` hands out a :class:`SharedPage` — a tab-shaped wrapper.
  ``quit()``/``close()`` on it close ONLY the tab; every other attribute is
  delegated to DrissionPage, so callers (relays, z.ai, rungs) need no
  changes beyond how they obtained the page.
* A central reaper quits the whole Chromium once every tab is gone and the
  idle window (``I4F_BROWSER_IDLE_REAP``, default 600 s) has elapsed. The
  next acquire cold-starts it back on the same profile.
* Refresher rungs that used to get an ephemeral auto-port profile now take
  a tab with ``fresh=True``: the manager clears cookies + cache of the
  shared profile via CDP before handing the tab over. Signup rungs are
  serialized by the process-wide flock, and the fresh gate additionally
  serializes all fresh-tab lifetimes (with a steal timeout so a leaked
  handle can never block renewals forever) — two concurrent rungs can no
  longer wipe each other's cookies out from under them.
* Wedge hygiene from refresher (kill-stale-by-port, profile lock cleanup,
  zombie reaping, Xvfb display) moved here so every keyed instance gets it.
"""
import atexit
import hashlib
import logging
import os
import signal
import threading
import time
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger('shared-browser')

DEFAULT_PORT = int(os.getenv('I4F_BROWSER_PORT', '9333') or 9333)
DEFAULT_PROFILE = os.getenv('I4F_BROWSER_PROFILE', '/data/shared_browser')
IDLE_REAP = max(60.0, float(os.getenv('I4F_BROWSER_IDLE_REAP', '600') or 600))
# A fresh-state rung holding the gate longer than this is assumed wedged;
# the next waiter steals the gate and closes its tab.
FRESH_STEAL_AFTER = max(300.0, float(
    os.getenv('I4F_BROWSER_FRESH_STEAL_AFTER', '1800') or 1800))


def _env_bool(name: str, default: bool) -> bool:
    raw = (os.getenv(name, '') or '').strip().lower()
    if not raw:
        return default
    return raw in ('1', 'true', 'yes', 'on', 'y')


def _headless_default() -> bool:
    """Headed under Xvfb by default (anti-bot walls score headless lower).

    ``I4F_BROWSER_HEADLESS`` wins when set; ``I4F_ZAI_HEADLESS`` is the
    legacy fallback so existing .env files keep working."""
    if (os.getenv('I4F_BROWSER_HEADLESS', '') or '').strip():
        return _env_bool('I4F_BROWSER_HEADLESS', False)
    return _env_bool('I4F_ZAI_HEADLESS', False)


# ----------------------------------------------------------------- display
_DISPLAY = None  # pyvirtualdisplay handle kept alive for headed runs


def _x_display_alive(number: int) -> bool:
    """True when an X server is actually listening on display ``:number``.

    A dead Xvfb leaves its /tmp/.X11-unix socket behind; trusting the
    socket alone makes Chromium die on connect."""
    sock = Path(f'/tmp/.X11-unix/X{number}')
    if not sock.exists():
        return False
    import socket
    try:
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            s.settimeout(2)
            s.connect(str(sock))
            return True
        finally:
            s.close()
    except OSError:
        return False


def ensure_display() -> bool:
    """Best-effort X server for headed runs. Returns True when a LIVE
    display is available (env, an existing server, or one we start)."""
    global _DISPLAY
    if _DISPLAY is not None:
        try:
            num = str(_DISPLAY.display).lstrip(':').split('.')[0]
            if not (num.isdigit() and _x_display_alive(int(num))):
                _DISPLAY = None  # dead — start a fresh one below
                return False
        except Exception:  # noqa: BLE001
            return True  # cannot verify — trust it
        return True
    disp = os.environ.get('DISPLAY', '')
    if disp:
        num = disp.lstrip(':').split('.')[0]
        if num.isdigit() and _x_display_alive(int(num)):
            return True
        os.environ.pop('DISPLAY', None)  # dead socket — don't trust it
    import glob as _glob
    for sock in sorted(_glob.glob('/tmp/.X11-unix/X[0-9]*')):
        try:
            num = int(sock.rsplit('X', 1)[-1])
        except ValueError:
            continue
        if _x_display_alive(num):
            os.environ['DISPLAY'] = f':{num}'
            return True
    try:
        from pyvirtualdisplay import Display
        _DISPLAY = Display(visible=False, size=(1440, 900))
        _DISPLAY.start()
        os.environ['DISPLAY'] = _DISPLAY.new_display_var
        return True
    except Exception:  # noqa: BLE001
        return False


# ------------------------------------------------------------ wedge hygiene
def reap_dead_children() -> None:
    """Reap exited child processes (zombie chromium after failed launches).

    DrissionPage spawns chromium through short-lived intermediates; when
    the browser dies the zombie is reparented to PID 1 (this process),
    which never wait()s — without reaping the container accumulates one
    zombie pair per failed browser attempt."""
    while True:
        try:
            pid, _ = os.waitpid(-1, os.WNOHANG)
        except (ChildProcessError, OSError):
            return
        if pid <= 0:
            return


def kill_stale_browsers(user_data_path: Optional[str] = None,
                        port: Optional[int] = None) -> int:
    """Kill chromium processes wedged on a profile directory / debug port.

    DrissionPage's ``quit()`` fails silently on a wedged tab, so the Chrome
    it spawned stays alive and keeps owning the profile/port. Only
    processes whose own command line names this profile/port are killed,
    so unrelated browsers are untouched."""
    needles = []
    if user_data_path:
        needles.append(f'--user-data-dir={user_data_path}')
    if port:
        needles.append(f'--remote-debugging-port={port}')
    if not needles:
        return 0
    killed = 0
    try:
        entries = os.listdir('/proc')
    except OSError:
        return 0
    for entry in entries:
        if not entry.isdigit() or int(entry) == os.getpid():
            continue
        try:
            with open(f'/proc/{entry}/cmdline', 'rb') as fh:
                cmd = fh.read().decode('utf-8', 'ignore')
        except OSError:
            continue
        if 'chrom' not in cmd:
            continue
        if not any(needle in cmd for needle in needles):
            continue
        try:
            os.kill(int(entry), signal.SIGKILL)
            killed += 1
        except OSError:
            continue
    if killed:
        reap_dead_children()
        logger.info('reaped %d stale browser process(es) for %s',
                    killed, needles[0])
    return killed


def profile_port(profile: str) -> int:
    """Deterministic debug port for a persistent browser profile.

    DrissionPage's ``set_user_data_path()`` clears ``auto_port`` while
    leaving the address empty, so every profile-based launch must carry an
    explicit port. Deriving it from the profile path keeps one stable port
    per profile."""
    digest = int(hashlib.sha1(
        os.path.abspath(profile).encode('utf-8')).hexdigest(), 16)
    return 19300 + digest % 40000  # 19300..59299, clear of auto_port picks


def clear_profile_lock(profile: str) -> None:
    """Remove Chrome singleton locks orphaned by a dead/foreign owner.

    A container restart leaves the profile's SingletonLock pointing at the
    old container's hostname+pid; every new Chrome then refuses the
    profile and starts WITHOUT binding the DevTools port. Called only
    after kill_stale_browsers, which guarantees no live local owner is
    holding the profile."""
    try:
        p = Path(profile)
        if not p.is_dir():
            return
        for name in ('SingletonLock', 'SingletonSocket', 'SingletonCookie'):
            try:
                (p / name).unlink()
            except OSError:
                continue
    except Exception:  # noqa: BLE001
        pass


# ------------------------------------------------------------------ gate
class _FreshClaim:
    """Owner token of the fresh-state gate (one per fresh acquire)."""

    def __init__(self) -> None:
        self.acquired_at = time.time()
        self.tab = None  # set once the tab exists — steal closes it

    def close_tab(self) -> None:
        tab = self.tab
        if tab is not None:
            try:
                tab.close()
            except Exception:  # noqa: BLE001 — already gone
                pass
            self.tab = None


class _FreshGate:
    """Serialize fresh-state rungs; steal the gate from runs wedged past
    ``FRESH_STEAL_AFTER`` so a leaked handle cannot block renewals forever.

    Consumers without a fresh wipe (relays, z.ai) never touch this gate
    and can never be blocked by it."""

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._owner: Optional[_FreshClaim] = None

    def acquire(self, claim: _FreshClaim) -> bool:
        deadline = time.time() + FRESH_STEAL_AFTER
        with self._cond:
            while True:
                if self._owner is None:
                    self._owner = claim
                    return True
                if time.time() - self._owner.acquired_at > FRESH_STEAL_AFTER:
                    stale, self._owner = self._owner, None
                    self._cond.notify_all()
                    logger.warning('fresh gate stolen from a run wedged '
                                   'past %.0fs', FRESH_STEAL_AFTER)
                    stale.close_tab()
                    continue
                remaining = deadline - time.time()
                if remaining <= 0:
                    return False  # proceed without the wipe, never hang
                self._cond.wait(min(remaining, 5.0))

    def release(self, claim: _FreshClaim) -> None:
        with self._cond:
            if self._owner is claim:
                self._owner = None
                self._cond.notify_all()


_FRESH_GATE = _FreshGate()


# ------------------------------------------------------------------ page
class SharedPage:
    """Tab-shaped handle of the shared Chromium.

    ``quit()``/``close()`` close ONLY this tab and release it back to the
    manager — the whole browser lives and dies with the central reaper.
    Every other attribute is delegated to the DrissionPage tab, so code
    written against ``ChromiumPage`` keeps working."""

    def __init__(self, owner: '_SharedBrowser', tab, fresh: bool = False):
        self._owner = owner
        self._tab = tab
        self.fresh = fresh
        self._claim = None  # _FreshClaim when fresh
        self.released = False
        self.acquired_at = time.time()
        # refresher._close_page() probes _port/port to kill a wedged chrome
        # by its debug port — on a SHARED tab that would kill the whole
        # shared browser with every other consumer in it. Keep them None.
        self._port = None
        self.port = None

    def quit(self, *args, **kwargs) -> None:
        self._owner.release(self)

    close = quit

    def __getattr__(self, name):
        if name.startswith('__'):
            raise AttributeError(name)
        tab = self.__dict__.get('_tab')
        if tab is None:
            raise AttributeError(name)
        return getattr(tab, name)

    def __enter__(self) -> 'SharedPage':
        return self

    def __exit__(self, *exc) -> None:
        self.quit()


# ---------------------------------------------------------------- browser
class _SharedBrowser:
    """One Chromium process shared by every consumer of a (profile, proxy)
    key. Consumers hold tabs; the process quits when all tabs are gone."""

    def __init__(self, key: str, profile: Optional[str] = None,
                 port: Optional[int] = None, proxy: Optional[str] = None):
        self.key = key
        self.profile = profile
        self.port = int(port) if port else (
            profile_port(profile) if profile else DEFAULT_PORT)
        self.proxy = proxy
        self._browser = None
        self._handles = []  # live SharedPage handles
        self._lock = threading.RLock()
        self._last_release = time.time()
        self._headless: Optional[bool] = None  # decided at first build

    # ------------------------------------------------------------ lifecycle
    def _alive(self) -> bool:
        b = self._browser
        if b is None:
            return False
        try:
            b.tabs_count  # CDP round-trip — raises on a dead browser
            return True
        except Exception:  # noqa: BLE001
            return False

    def _build(self) -> None:
        from DrissionPage import Chromium, ChromiumOptions  # heavy import
        options = ChromiumOptions().set_local_port(self.port)
        if self.profile:
            Path(self.profile).mkdir(parents=True, exist_ok=True)
            options.set_user_data_path(self.profile)
        options.set_argument('--no-sandbox')
        options.set_argument('--disable-gpu')
        # Docker's default /dev/shm is 64MB: Chrome dies mid-navigation
        # there (renderer killed) — observed on signup rungs.
        options.set_argument('--disable-dev-shm-usage')
        # Anti-bot services (Aliyun slider, Cloudflare, Google) score the
        # client: hide automation and run windowed under Xvfb.
        options.set_argument('--disable-blink-features=AutomationControlled')
        options.set_argument('--window-size=1440,900')
        # Memory trims (8 GB SBC): no BFCache renderers held behind the
        # active tab, no audio utility process — both showed up in RSS.
        options.set_argument('--disable-back-forward-cache')
        options.set_argument('--mute-audio')
        # Signup rungs click ENGLISH button texts ('Continue', 'Sign in',
        # 'Verify'): a page that follows the host IP's locale (observed:
        # dash.llm7.io rendered Italian for an IT egress) matches none of
        # them and the rung stalls before the form is ever submitted.
        # Pin the UI language so the English click lists stay valid.
        options.set_argument('--lang=en-US')
        try:
            options.set_pref('intl.accept_languages', 'en-US,en')
        except Exception:  # noqa: BLE001 - pref API drift; --lang still
            pass           # covers navigator.language
        if self.proxy:
            if self.proxy.startswith('socks'):
                # Chromium accepts the plain "socks5://" scheme only
                # ("socks5h://" is a curl-ism → ERR_NO_SUPPORTED_PROXIES);
                # DNS is forced through the proxy so the exit stays
                # consistent.
                _, _, hostport = self.proxy.partition('://')
                host = hostport.split('/')[0].split(':')[0]
                options.set_argument(f'--proxy-server=socks5://{hostport}')
                options.set_argument(
                    f'--host-resolver-rules=MAP * ~NOTFOUND , '
                    f'EXCLUDE {host}')
            else:
                options.set_proxy(self.proxy)
        if self._headless is None:
            self._headless = _headless_default()
        headless = self._headless
        if headless:
            options.headless(True)
        elif not ensure_display():
            # Xvfb may still be booting at container start (the first
            # consumer starts it) — one short retry before degrading:
            # headless scores lower on every anti-bot wall (Aliyun,
            # Cloudflare, Google) and the keyed instances boot at the
            # same time as the first shared one.
            time.sleep(2)
            if not ensure_display():
                options.headless(True)
                headless = True
        reap_dead_children()
        # A wedged Chrome still holding this profile/port makes the new
        # launch fail in ways that look like a bot wall — clear it first.
        kill_stale_browsers(self.profile, self.port)
        if self.profile:
            clear_profile_lock(self.profile)
        try:
            self._browser = Chromium(options)
        except Exception:
            # Slow boot / half-open CDP (observed on a loaded SBC: the
            # previous chromium was still dying when the new one bound the
            # port, and DrissionPage's handshake answered 404 mid-boot).
            # Reap, wait for the port to actually close, retry once — only
            # then degrade the window, which would mask a real bot-wall
            # signal if done prematurely.
            import socket
            reap_dead_children()
            kill_stale_browsers(self.profile, self.port)
            clear_profile_lock(self.profile)
            for _ in range(12):
                s = socket.socket()
                s.settimeout(0.3)
                busy = self.port is not None and \
                    s.connect_ex(('127.0.0.1', self.port)) == 0
                s.close()
                if not busy:
                    break
                time.sleep(1)
            try:
                self._browser = Chromium(options)
            except Exception:
                if not headless:
                    # windowed spawn failed (e.g. the X server died between
                    # the liveness check and the spawn) — degrade to headless
                    options.headless(True)
                    self._headless = True
                    self._browser = Chromium(options)
                else:
                    raise
        logger.info('shared browser %s: chromium ready on port %d '
                    '(profile=%s, proxy=%s, headless=%s)',
                    self.key, self.port, self.profile or 'default',
                    self.proxy or 'none', headless)

    def _clear_state(self, tab) -> None:
        """Wipe cookies + cache of the shared profile (fresh-tab semantics).

        Reproduces the old ephemeral auto-port profile inside ONE process:
        each refresher rung starts logged-out without paying a new browser.
        localStorage is deliberately kept (z.ai guest token survives)."""
        for call in ('Network.clearBrowserCookies',
                     'Network.clearBrowserCache'):
            try:
                tab.run_cdp(call)
            except Exception as exc:  # noqa: BLE001 — best effort
                logger.debug('shared browser %s: %s failed: %s',
                             self.key, call, exc)

    # ------------------------------------------------------------- acquire
    def acquire(self, url: Optional[str] = None, fresh: bool = False,
                headed_hint: bool = False,
                url_timeout: float = 90.0) -> SharedPage:
        claim = _FreshClaim() if fresh else None
        if claim is not None and not _FRESH_GATE.acquire(claim):
            claim = None  # proceed without the wipe rather than hang
        try:
            with self._lock:
                if not self._alive():
                    try:
                        if self._browser is not None:
                            self._browser.quit(timeout=8, force=True)
                    except Exception:  # noqa: BLE001 — already dead
                        pass
                    self._browser = None
                    self._build()
                tab = self._browser.new_tab()
                handle = SharedPage(self, tab, fresh=claim is not None)
                if claim is not None:
                    claim.tab = tab
                    handle._claim = claim
                    self._clear_state(tab)
                if url:
                    try:
                        tab.get(url, timeout=url_timeout)
                    except Exception as exc:  # noqa: BLE001
                        # surface to the caller's own load path; the tab
                        # itself is usable
                        logger.debug('shared browser %s: initial get(%s) '
                                     'failed: %s', self.key, url, exc)
                if len(self._handles) == 1:
                    self._close_initial_tab(tab)
                self._handles.append(handle)
                _ensure_reaper()
                return handle
        except Exception:
            if claim is not None:
                _FRESH_GATE.release(claim)
            raise

    def _close_initial_tab(self, keep_tab) -> None:
        """Close the launch-time ``chrome://newtab`` tab.

        Chromium boots with a starter tab that DrissionPage never reuses —
        one idle renderer (~50-80 MB headed) per browser, for nothing. The
        first consumer's own tab is never a candidate: fresh tabs sit on
        ``about:blank`` before their first navigation, not ``newtab``."""
        try:
            keep_id = getattr(keep_tab, 'tab_id', None)
            for tid in list(self._browser.tab_ids):
                if tid == keep_id:
                    continue
                t = self._browser.get_tab(tid)
                if (t.url or '').startswith('chrome://newtab'):
                    t.close()
        except Exception:  # noqa: BLE001 — trim is best effort
            pass

    def release(self, handle: SharedPage) -> None:
        if getattr(handle, 'released', False):
            return
        handle.released = True
        with self._lock:
            try:
                self._handles.remove(handle)
            except ValueError:
                pass
            try:
                handle._tab.close()
            except Exception:  # noqa: BLE001 — tab already gone
                pass
            self._last_release = time.time()
        claim = getattr(handle, '_claim', None)
        if claim is not None:
            _FRESH_GATE.release(claim)

    def reap_if_idle(self, idle_s: float) -> None:
        with self._lock:
            if self._handles or self._browser is None:
                return
            if time.time() - self._last_release < idle_s:
                return
            logger.info('shared browser %s idle %.0fs — quitting chrome '
                        '(profile=%s)', self.key,
                        time.time() - self._last_release,
                        self.profile or 'default')
            try:
                self._browser.quit(timeout=8, force=True)
            except Exception:  # noqa: BLE001 — kill below guarantees it
                pass
            self._browser = None
            self._last_release = time.time()
        kill_stale_browsers(self.profile, self.port)
        reap_dead_children()


# -------------------------------------------------------------- registry
_REGISTRY: Dict[str, _SharedBrowser] = {}
_REGISTRY_LOCK = threading.Lock()
_REAPER_THREAD: Optional[threading.Thread] = None


def _instance_key(profile: Optional[str], proxy: Optional[str]) -> str:
    p = os.path.abspath(profile) if profile else 'default'
    return f'{p}|{proxy or ""}'


def acquire(proxy: Optional[str] = None, headed: bool = False,
            profile: Optional[str] = None, port: Optional[int] = None,
            fresh: bool = False, url: Optional[str] = None,
            url_timeout: float = 90.0) -> SharedPage:
    """Acquire a tab of the shared Chromium for this (profile, proxy) key.

    ``fresh=True`` wipes cookies + cache first (the old ephemeral-profile
    semantics, inside one process). ``url`` navigates the new tab
    immediately; ``headed`` is advisory (the instance decides its mode
    once, at first build)."""
    key = _instance_key(profile, proxy)
    with _REGISTRY_LOCK:
        inst = _REGISTRY.get(key)
        if inst is None:
            inst = _SharedBrowser(
                key, profile=profile, port=port, proxy=proxy)
            _REGISTRY[key] = inst
    return inst.acquire(url=url, fresh=fresh, headed_hint=headed,
                        url_timeout=url_timeout)


def _reaper_loop() -> None:
    while True:
        time.sleep(30)
        with _REGISTRY_LOCK:
            instances = list(_REGISTRY.values())
        for inst in instances:
            try:
                inst.reap_if_idle(IDLE_REAP)
            except Exception:  # noqa: BLE001 — reaping is best effort
                logger.debug('shared-browser reaper failed on %s',
                             inst.key, exc_info=True)


def _ensure_reaper() -> None:
    global _REAPER_THREAD
    with _REGISTRY_LOCK:
        if _REAPER_THREAD is not None and _REAPER_THREAD.is_alive():
            return
        _REAPER_THREAD = threading.Thread(
            target=_reaper_loop, name='shared-browser-reaper', daemon=True)
        _REAPER_THREAD.start()


def shutdown_all() -> None:
    """atexit: quit every shared Chromium."""
    with _REGISTRY_LOCK:
        instances = list(_REGISTRY.values())
    for inst in instances:
        try:
            inst.reap_if_idle(0)
        except Exception:  # noqa: BLE001 — shutdown best effort
            pass


atexit.register(shutdown_all)
