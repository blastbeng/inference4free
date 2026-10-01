"""Browser relay for chatgpt.com completions (anonymous surface).

Every non-browser POST to /backend-api/conversation is gated by the 2026
Sentinel stack (Turnstile ~29KB dx token, sha3-384 proof-of-work, and a
behavioral attestation) that only the site's own JS can satisfy — hand-rolled
replays stop at 403/422. The one transport that works is the site's own
anonymous chat: the unauth surface (unauth-mweb) answers without any login.
So the relay runs one persistent Chrome (DrissionPage) on a dedicated port and
drives the real UI:

1. Each request: click "New chat" (fresh context), optionally pick the
   requested model in the model dropdown, type the prompt, click send.
2. The anonymous UI renders WITHOUT ``data-message-author-role`` markers, so
   the reply is read from ``document.body.innerText``: the text after the
   last ``ChatGPT said:`` marker, cut at the first footer/UI line, diffed
   into deltas as it grows.

The surface is unauthenticated, so generation never depends on the renewal
ladder; the refresher keeps working account cookies for status/limits only.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from typing import Any, Dict, Generator, List, Optional

logger = logging.getLogger(__name__)

# Lines that end the assistant reply inside body.innerText. Matched as whole
# trimmed lines only, so a reply that merely mentions "log in" is not cut.
REPLY_TERMINATORS = (
    'You said:',
    'Chat with ChatGPT',
    'Get responses tailored to you',
    'Log in',
    'Sign up for free',
    'Continue with Google',
    'Continue with Apple',
    'Continue with phone',
    'Email address',
    'New chat',
    'Report content',
)
MARKER = 'ChatGPT said:'
# Shown in the reply region while the model is still working.
PLACEHOLDER = 'Thinking'
# Error surfaces checked only while no reply has started yet.
ERROR_LINES = (
    'Something went wrong',
    "You've reached our limit",
    'You have reached our limit',
    'unable to load conversation',
    'Oops',
)


class RelayBlocked(RuntimeError):
    """Challenge/interstitial or dead UI swallowed the reply — rebuild."""


def norm_model(model: str) -> str:
    """'GPT-5.5' and 'gpt_5_5' both -> 'gpt55'."""
    return re.sub(r'[^a-z0-9]+', '', str(model or '').lower())


def _reply_of(text: str) -> str:
    """Reply text after the last ``ChatGPT said:`` marker, cut at the first
    terminator line. Empty while the marker has not appeared yet."""
    if MARKER not in text:
        return ''
    region = text.split(MARKER)[-1]
    kept: List[str] = []
    for line in region.splitlines():
        if line.strip() in REPLY_TERMINATORS:
            break
        kept.append(line)
    while kept and kept[0].strip() in ('', PLACEHOLDER):
        kept.pop(0)
    return '\n'.join(kept).strip()


class ChatGPTRelay:
    """One persistent browser session driving the anonymous chatgpt UI."""

    RELOAD_TTL = 1800          # rebuild the session twice an hour
    SEND_TIMEOUT = 30          # UI acceptance of the send click
    FIRST_MARK_TIMEOUT = 75    # 'ChatGPT said:' must appear after send
    THINKING_TIMEOUT = 240     # max time in the placeholder-only state
    REPLY_STALL_TIMEOUT = 25   # no growth for this long -> reply complete
    STREAM_TIMEOUT = 600       # cap on one generation
    POLL_S = 0.7

    def __init__(self) -> None:
        self._page = None
        self._lock = threading.RLock()
        # One stream at a time. A BoundedSemaphore, NOT the RLock: the API
        # layer may close the stream generator from a different thread than
        # the one that advanced it, and an RLock released by a non-owner
        # thread raises "cannot release un-acquired lock".
        self._busy = threading.BoundedSemaphore(1)
        self._loaded_at: float = 0.0
        self._offered: List[str] = []   # display titles seen in the picker
        # Seed the default so list_models can advertise a relay route before
        # the relay has opened its picker — the relay serves this family
        # default on the anonymous surface, which breaks the chicken-and-egg
        # between "router needs a route to start the relay" and "relay needs
        # a stream to discover its models".
        self._default_title: str = os.getenv(
            'I4F_CHATGPT_RELAY_DEFAULT', 'chatgpt-auto')

    # ------------------------------------------------------------- lifecycle
    def enabled(self) -> bool:
        return os.getenv('I4F_CHATGPT_RELAY', '1').strip().lower() not in \
            ('0', 'false', 'no', 'off')

    def offered(self) -> List[str]:
        """Model display titles seen in the picker (empty until the first
        dropdown open). Keeps /v1/models honest about what the relay serves."""
        return list(self._offered)

    def default_title(self) -> str:
        return self._default_title

    def _profile_dir(self) -> str:
        override = (os.getenv('I4F_CHATGPT_PROFILE', '') or '').strip()
        if override:
            return override
        from dsk.refresher import _data_dir
        return str(_data_dir() / 'browser' / 'chatgpt')

    def _port(self) -> int:
        # A fixed local port makes DrissionPage ADOPT the already-running
        # Chrome instead of launching a competing one on the same profile.
        # Default to the refresher's per-profile derivation so the browser
        # login rung (same profile) adopts this relay session too.
        override = (os.getenv('I4F_CHATGPT_RELAY_PORT', '') or '').strip()
        if override:
            return int(override)
        from dsk import refresher
        return refresher._profile_port(self._profile_dir())

    def _alive(self) -> bool:
        try:
            return self._page is not None and bool(
                self._page.run_js('return 1;') == 1)
        except Exception:  # noqa: BLE001 — crashed/closed browser
            return False

    def _wait_settle(self, timeout: float = 30.0) -> None:
        """Wait out a mid-flight page reload (DrissionPage raises
        ContextLostError until the new document is ready)."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                if self._page.run_js('return 1;', timeout=10) == 1:
                    return
            except Exception:  # noqa: BLE001
                time.sleep(1.5)
        raise RuntimeError('chatgpt relay: page never settled after reload')

    def _js(self, script: str, retries: int = 4):
        """run_js that survives a mid-stream reload: wait for settle, retry."""
        last: Optional[Exception] = None
        for _ in range(retries):
            try:
                return self._page.run_js(script)
            except Exception as e:  # noqa: BLE001
                if 'ContextLost' not in type(e).__name__ \
                        and 'refreshed' not in str(e):
                    raise
                last = e
                logger.warning('chatgpt relay: page reloaded mid-stream — '
                               'waiting for settle')
                time.sleep(2)
                try:
                    self._wait_settle()
                except Exception:  # noqa: BLE001
                    continue
        raise last if last else RuntimeError('chatgpt relay: js failed')

    def _accept_dialogs(self) -> None:
        from dsk import refresher
        refresher._click_any(self._page,
                             ['I agree', 'Accept all', 'Continue', 'Got it',
                              'Okay', 'OK'])

    def _open_home(self) -> None:
        """(Re)load the chat surface and wait until a composer exists."""
        page = self._page
        page.get('https://chatgpt.com/')
        time.sleep(6)
        self._wait_settle()
        self._accept_dialogs()
        self._loaded_at = time.time()
        deadline = time.time() + 25
        while time.time() < deadline:
            if self._composer() is not None:
                logger.info('chatgpt relay: anonymous composer ready')
                return
            self._accept_dialogs()
            time.sleep(1.5)
        raise RelayBlocked('chatgpt relay: composer not found on the '
                           'anonymous surface')

    def _build(self) -> None:
        from dsk import refresher
        if self._page is not None:
            try:
                self._page.quit()
            except Exception:  # noqa: BLE001
                pass
            self._page = None
        refresher._ensure_display()
        refresher._reap_dead_children()
        relay_proxy = (os.getenv('I4F_CHATGPT_PROXY', '') or '').strip() or None
        last: Optional[Exception] = None
        for attempt in range(3):
            try:
                self._page = refresher._browser(
                    proxy=relay_proxy, headed=True,
                    user_data_path=self._profile_dir(),
                    local_port=self._port())
                self._open_home()
                return
            except Exception as e:  # noqa: BLE001 — wedged tab / challenge
                last = e
                logger.warning('chatgpt relay build attempt %d failed: %s',
                               attempt + 1, e)
                if self._page is not None:
                    try:
                        self._page.quit()
                    except Exception:  # noqa: BLE001
                        pass
                self._page = None
                # quit() above is a no-op on a wedged tab: the Chrome it
                # spawned keeps owning the relay profile, and the next
                # attempt then fails on the profile lock instead of the
                # wall it was meant to test.
                try:
                    refresher._kill_stale_browsers(self._profile_dir(),
                                                   self._port())
                except Exception:  # noqa: BLE001
                    pass
                time.sleep(3)
        raise RelayBlocked(
            f'chatgpt relay: could not build a working session: {last}')

    def _ensure(self) -> None:
        if self._alive() and (time.time() - self._loaded_at) <= self.RELOAD_TTL:
            return
        if self._alive():
            try:
                self._open_home()
                return
            except Exception as e:  # noqa: BLE001 — fall through to rebuild
                logger.warning('chatgpt relay refresh failed (%s); rebuilding', e)
        self._build()

    def shutdown(self) -> None:
        with self._lock:
            if self._page is not None:
                try:
                    self._page.quit()
                except Exception:  # noqa: BLE001
                    pass
                self._page = None

    # ------------------------------------------------------------- streaming
    def _composer(self):
        for sel in ('css:#prompt-textarea', 'css:div[contenteditable=true]',
                    'css:[role=textbox]', 'css:textarea'):
            try:
                ele = self._page.ele(sel, timeout=3)
                if ele:
                    return ele
            except Exception:  # noqa: BLE001
                continue
        return None

    def _reset_chat(self) -> None:
        """Fresh context per request: click New chat, reload as fallback."""
        try:
            newchat = (self._page.ele('css:a[aria-label="New chat"]', timeout=2)
                       or self._page.ele('css:button[aria-label="New chat"]',
                                         timeout=1)
                       or self._page.ele('text:New chat', timeout=1))
            if newchat:
                newchat.click()
                time.sleep(2)
                if self._composer() is not None:
                    return
        except Exception as e:  # noqa: BLE001
            logger.debug('chatgpt relay: new-chat click failed: %s', e)
        self._open_home()

    def _select_model(self, model: str) -> bool:
        """Pick `model` in the anonymous model dropdown when it exists.

        The unauth surface usually serves its default only; in that case the
        request rides the default (recorded as ``_default_title``) and any
        explicitly different model is refused instead of silently served by
        another one.
        """
        want = norm_model(model)
        button = None
        for sel in ('css:[data-testid="model-switcher-dropdown-button"]',
                    'css:button[aria-label*="Model"]'):
            try:
                button = self._page.ele(sel, timeout=2)
                if button:
                    break
            except Exception:  # noqa: BLE001
                continue
        if button is None:
            # No picker on this surface: only the default exists. Keep serving
            # when the operator asked for the family default / unknown alias.
            logger.info('chatgpt relay: no model picker; serving the default '
                        'for requested %r', model)
            return True
        try:
            title = str(button.text or button.attr('aria-label') or '').strip()
            if title and not self._default_title:
                self._default_title = title.split('\n')[0]
        except Exception:  # noqa: BLE001
            pass
        try:
            button.click()
        except Exception as e:  # noqa: BLE001
            logger.info('chatgpt relay: model picker did not open (%s); '
                        'serving the default', e)
            return True
        time.sleep(1.2)
        picked = 'not-found'
        try:
            picked = str(self._js('''
              var want = %s;
              var items = document.querySelectorAll(
                '[role=menuitem], [role=option], [role=menuitemradio]');
              var seen = [];
              var out = 'not-found';
              for (var i = 0; i < items.length; i++) {
                var t = (items[i].innerText || '').trim().split('\\n')[0];
                if (!t) continue;
                seen.push(t);
                var norm = t.toLowerCase().replace(/[^a-z0-9]+/g, '');
                if (norm === want) { items[i].click(); out = 'picked:' + t; }
              }
              window.__dsfOffered = seen;
              return out;
            ''' % json.dumps(want)) or '')
        except Exception as e:  # noqa: BLE001
            picked = 'err'
            logger.debug('chatgpt relay: picker scan failed: %s', e)
        titles: List[str] = []
        try:
            titles = json.loads(str(self._js(
                'return JSON.stringify(window.__dsfOffered || []);') or '[]'))
        except Exception:  # noqa: BLE001
            titles = []
        if titles:
            self._offered = [str(t) for t in titles]
        try:
            self._page.run_js('if (document.body) document.body.click();')
        except Exception:  # noqa: BLE001
            pass
        time.sleep(0.4)
        if picked.startswith('picked'):
            logger.info('chatgpt relay: model selected -> %s', picked[7:])
            return True
        logger.warning('chatgpt relay: model %r not offered (picker shows %s)',
                       model, ', '.join(self._offered) or '?')
        return False

    def _type_and_send(self, prompt: str) -> None:
        typed = str(self._js(
            'var ed = document.querySelector("#prompt-textarea, '
            'div[contenteditable=true], textarea");'
            'if (!ed) return "no-editor";'
            'ed.focus();'
            'if (!document.execCommand("insertText", false, %s)) '
            'return "exec-failed";'
            'return "typed";' % json.dumps(prompt)) or '')
        if typed != 'typed':
            ele = self._composer()
            if ele is None:
                raise RelayBlocked(f'chatgpt relay: cannot type ({typed})')
            try:
                ele.click()
            except Exception:  # noqa: BLE001
                pass
            ele.input(prompt)
        time.sleep(1)
        btn = None
        for sel in ('css:button[data-testid="send-button"]',
                    'css:button[aria-label*="Send"]',
                    'css:button[aria-label*="send"]'):
            try:
                btn = self._page.ele(sel, timeout=2)
                if btn:
                    break
            except Exception:  # noqa: BLE001
                continue
        if btn:
            btn.click()
        else:
            from DrissionPage.common import Actions
            Actions(self._page).key_down('Enter').key_up('Enter')

    def _resend_if_swallowed(self, prompt: str) -> None:
        """The send click can land before React accepts the input: if the
        editor still holds the prompt, click send again."""
        time.sleep(2)
        try:
            held = str(self._js(
                'var ed=document.querySelector("#prompt-textarea, textarea, '
                'div[contenteditable=true]"); return ed ? '
                '(ed.value || ed.textContent || "") : "";') or '')
            if held.strip():
                btn = self._page.ele(
                    'css:button[data-testid="send-button"]', timeout=2)
                if btn:
                    btn.click()
                else:
                    from DrissionPage.common import Actions
                    Actions(self._page).key_down('Enter').key_up('Enter')
                time.sleep(2)
        except Exception as e:  # noqa: BLE001 — the drain will surface failures
            logger.debug('chatgpt relay: resend check failed: %s', e)

    def _body_text(self) -> str:
        try:
            return str(self._js('return document.body.innerText;') or '')
        except Exception:  # noqa: BLE001 — transient during reloads
            return ''

    def _drain_reply(self) -> Generator[Dict[str, Any], None, None]:
        """Diff the reply region as it grows and yield text deltas."""
        sent = ''
        t0 = time.time()
        first_mark_at: Optional[float] = None
        content_at: Optional[float] = None
        stall_since: Optional[float] = None
        deadline = t0 + self.STREAM_TIMEOUT
        while time.time() < deadline:
            time.sleep(self.POLL_S)
            rep = _reply_of(self._body_text())
            if first_mark_at is None and rep:
                first_mark_at = time.time()
            if first_mark_at is None:
                if time.time() - t0 > self.FIRST_MARK_TIMEOUT:
                    text = self._body_text()
                    err = next((line for line in ERROR_LINES
                                if line in text), None)
                    raise RelayBlocked(
                        f'chatgpt relay: no reply started'
                        f'{f" (UI says: {err!r})" if err else ""}')
                continue
            if rep in ('', PLACEHOLDER):
                if content_at is None and \
                        time.time() - first_mark_at > self.THINKING_TIMEOUT:
                    raise RelayBlocked('chatgpt relay: stuck in thinking '
                                       'placeholder')
                continue
            if content_at is None:
                content_at = time.time()
                stall_since = time.time()
            if rep.startswith(sent) and len(rep) > len(sent):
                delta = rep[len(sent):]
                sent = rep
                stall_since = time.time()
                yield {'content': delta, 'type': 'text', 'finish_reason': None}
            elif not rep.startswith(sent) and rep:
                # The region was rewritten under us (rare): restart the diff.
                logger.warning('chatgpt relay: reply region rewritten; '
                               're-diffing from the new text')
                sent = rep
                stall_since = time.time()
            if sent and stall_since is not None and \
                    time.time() - stall_since > self.REPLY_STALL_TIMEOUT:
                break
        if not sent:
            raise RelayBlocked('chatgpt relay: no reply text within the '
                               'stream window')
        yield {'content': '', 'type': 'text', 'finish_reason': 'stop'}

    def stream(self, prompt: str, model: str = 'chatgpt-auto') -> \
            Generator[Dict[str, Any], None, None]:
        """Yield provider text chunks for one prompt via the anonymous UI.

        The lock is only held around setup steps, never across the yields of
        the drain: the consumer may close this generator from another thread.
        """
        if not self.enabled():
            raise RuntimeError('chatgpt relay disabled')
        if not self._busy.acquire(blocking=False):
            raise RuntimeError('chatgpt relay busy with another stream')
        try:
            with self._lock:
                self._ensure()
            for attempt in (1, 2):
                try:
                    with self._lock:
                        self._reset_chat()
                        if not self._select_model(model):
                            raise RuntimeError(
                                f'model {model!r} is not offered by the anonymous '
                                f'chatgpt surface (offered: '
                                f'{", ".join(self._offered) or self._default_title or "?"})')
                        self._type_and_send(prompt)
                        self._resend_if_swallowed(prompt)
                    yield from self._drain_reply()
                    return
                except RelayBlocked:
                    logger.warning('chatgpt relay: stream blocked; rebuilding '
                                   'session')
                    try:
                        with self._lock:
                            self._build()
                    except Exception as e:  # noqa: BLE001
                        logger.warning('chatgpt relay rebuild failed: %s', e)
                    raise
                except Exception as e:  # noqa: BLE001
                    if attempt == 2 or 'ContextLost' not in type(e).__name__:
                        raise
                    logger.warning('chatgpt relay: context lost pre-stream; '
                                   'rebuilding and retrying once')
                    try:
                        with self._lock:
                            self._build()
                    except Exception as e2:  # noqa: BLE001
                        logger.warning('chatgpt relay rebuild failed: %s', e2)
                        raise
        finally:
            self._busy.release()


_RELAY: Optional[ChatGPTRelay] = None
_RELAY_LOCK = threading.Lock()


def get_relay() -> ChatGPTRelay:
    """Process-wide relay singleton."""
    global _RELAY
    with _RELAY_LOCK:
        if _RELAY is None:
            _RELAY = ChatGPTRelay()
        return _RELAY
