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
3. The collapsed chain-of-thought panel ("Thought · 3s" + summary + blank
   line) is split out of that region and streamed as ``type: 'thinking'``,
   so the summary reaches the client as ``reasoning_content`` and the
   message body stays the answer.

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
from typing import Any, Dict, Generator, List, Optional, Tuple

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
    # Rendered inside the reply region as the model's own disclaimer.
    'ChatGPT is AI and can make mistakes.',
)
MARKER = 'ChatGPT said:'
# Shown in the reply region while the model is still working.
PLACEHOLDER = 'Thinking'
# Header line of the collapsed chain-of-thought panel ("Thought · 3s",
# "Thought for 12 seconds", "Thinking"). The summary under it is NOT the
# answer: emitting it as text puts it in the message body, and clients read
# the thinking as the whole reply.
THOUGHT_HEAD_RE = re.compile(
    r'^(thought|reasoning|thinking)\s*(·|:|\bfor\b)?\s*[\d.]*\s*'
    r'(ms|s|sec|secs|second|seconds|min|mins|minutes)?\s*[.…]*$',
    re.IGNORECASE)

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


# DrissionPage names a dead (or mid-reload) tab several ways: ContextLostError,
# PageDisconnectedError, and the bare "The connection to the page has been
# disconnected." text. For the relay they all mean the same thing — the tab is
# gone, so the session must be rebuilt and the request retried. Surfacing it
# instead is what makes ONE flaky reload cost a failed answer: the client gets
# an api_error for a request that succeeds on the next attempt.
_PAGE_DEATH_NAMES = ('ContextLost', 'PageDisconnected', 'TargetClosed',
                     'SessionDisconnected', 'BrowserDisconnected')
_PAGE_DEATH_TEXTS = ('connection to the page has been disconnected',
                     'was refreshed', 'no such window', 'target closed',
                     'session deleted because of page crash')


def _is_page_death(e: BaseException) -> bool:
    """True when an exception means "the tab died", i.e. it is retryable."""
    name = type(e).__name__
    blob = str(e).lower()
    return any(n in name for n in _PAGE_DEATH_NAMES) \
        or any(t in blob for t in _PAGE_DEATH_TEXTS)


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


def _strip_headers(region: str) -> str:
    """Region with the panel chrome (blank lines + "Thought …" headers)
    removed from its head.

    Used when the panel cannot be separated from the answer: the header is UI
    chrome, never prose, so it must not reach the message body."""
    lines = region.split('\n')
    i = 0
    while i < len(lines) and (not lines[i].strip() or
                              THOUGHT_HEAD_RE.match(lines[i].strip())):
        i += 1
    return '\n'.join(lines[i:]).strip()


def _split_thinking(region: str) -> Tuple[str, str]:
    """Split the reply region into ``(thinking, text)``.

    The panel is only treated as reasoning when the UI really separated it
    from the answer: a "Thought …" header, a summary block, a blank line, and
    content after that blank line. The blank line is the requirement that
    matters — without it the summary and the answer are one undivided block,
    and guessing where the panel ends would hide the answer inside the
    thinking panel, so the whole region stays prose.

    The gap between header and summary is skipped: block elements make
    ``innerText`` emit "Thought · 3s\\n\\n<summary>", and treating that blank
    line as the separator is what put the summary in the message body.
    """
    if not region:
        return '', ''
    lines = region.split('\n')
    i = 0
    while i < len(lines) and not lines[i].strip():
        i += 1
    if i >= len(lines) or not THOUGHT_HEAD_RE.match(lines[i].strip()):
        return '', _strip_headers(region)
    k = i + 1
    while k < len(lines) and not lines[k].strip():
        k += 1                      # blank gap between header and summary
    if k >= len(lines):
        return '', ''              # header only: the model is still working
    j = k
    while j < len(lines) and lines[j].strip():
        j += 1                     # end of the summary block
    answer = '\n'.join(lines[j:]).strip()
    if j >= len(lines) or not answer:
        # Nothing after the summary block: the answer has not started, and
        # where it will begin is not knowable yet — keep it all as prose.
        return '', _strip_headers(region)
    return '\n'.join(lines[k:j]).strip(), answer


def _thinking_only(region: str) -> bool:
    """True while the region holds nothing but the placeholder and/or the
    panel header ("Thinking", "Thought · 3s") — the answer has not started."""
    if region in ('', PLACEHOLDER):
        return True
    lines = [ln.strip() for ln in region.split('\n') if ln.strip()]
    return bool(lines) and all(THOUGHT_HEAD_RE.match(ln) for ln in lines)


def _region_delta(cur: str, base: str) -> str:
    """Text to send for a region that grew (or was rewritten) since ``base``.

    The UI rewrites the region when the thinking panel collapses and the
    answer replaces it. Re-basing on the new text without emitting it is how
    an answer disappears, so a rewrite emits everything past the common
    prefix instead of nothing."""
    if not cur or cur == base:
        return ''
    if cur.startswith(base):
        return cur[len(base):]
    n = 0
    for a, b in zip(base, cur):
        if a != b:
            break
        n += 1
    return cur[n:]


# Shortest span considered "already delivered": a summary is a sentence, so
# suppression only ever fires on real prose, never on a one-word answer that
# happens to be quoted inside the summary.
_MIN_RECLASS_SPAN = 20


def _new_delta(cur: str, base: str, other_sent: str) -> str:
    """Delta for ``cur``, skipping what the OTHER region already delivered.

    The split is a judgement made on a snapshot, and the judgement can arrive
    late: the UI first shows "Thought · 3s\\n\\n<summary>" (no separator after
    the summary, so the summary is prose and goes out as ``text``), then the
    answer appears and the same summary is re-read as the thinking panel.
    Diffing per region would then re-send it — the client sees the reasoning
    twice, once in the body and once in ``reasoning_content``.
    """
    delta = _region_delta(cur, base)
    if not delta:
        return ''
    if len(delta) >= _MIN_RECLASS_SPAN and delta in other_sent:
        return ''
    return delta


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
        self._last_used: float = 0.0
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
                if not _is_page_death(e):
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

    def reap_if_idle(self, stale_after: float) -> None:
        """Close the browser when idle past ``stale_after`` seconds.

        Without this the ~300-500 MB Chromium footprint of the anonymous
        session stays resident forever after the first stream (the next
        stream cold-starts a fresh session via _build, same as the wedged-
        tab recovery path). Zero-timeout busy acquire: a reap can never
        race an in-flight stream."""
        if self._page is None or self._last_used <= 0:
            return
        if not self._busy.acquire(blocking=False):
            return
        try:
            if (self._page is not None and self._last_used > 0
                    and time.time() - self._last_used > stale_after):
                logger.debug('chatgpt relay idle %.0fs — reaper closing',
                             time.time() - self._last_used)
                self.shutdown()
        finally:
            self._busy.release()

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
        """Diff the reply region as it grows and yield text/thinking deltas.

        The reasoning panel and the answer are diffed as SEPARATE regions: the
        summary goes out as ``type: 'thinking'`` (the OpenAI shim turns it into
        ``delta.reasoning_content``) and only the answer becomes ``content``.
        Each region is diffed against its own baseline, so a rewrite of one
        cannot swallow the other, and a span the other region already
        delivered is not sent twice (see ``_new_delta``).
        """
        sent_text = ''
        sent_think = ''
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
            if _thinking_only(rep):
                # Panel header (or the bare placeholder) and nothing else:
                # the model is still working — no answer has started.
                if content_at is None and \
                        time.time() - first_mark_at > self.THINKING_TIMEOUT:
                    raise RelayBlocked('chatgpt relay: stuck in thinking '
                                       'placeholder')
                continue
            think, body = _split_thinking(rep)
            grew = False
            for kind, cur, base in (('thinking', think, sent_think),
                                    ('text', body, sent_text)):
                other = sent_text if kind == 'thinking' else sent_think
                delta = _new_delta(cur, base, other)
                if len(cur) < len(base):
                    logger.info('chatgpt relay: %s region rewritten; emitting '
                                'everything past the common prefix', kind)
                # Re-base on every snapshot, delta or not: the baseline is
                # what we have SEEN, so a late reclassification diffs the
                # next snapshot against it instead of re-sending it whole.
                if kind == 'thinking':
                    sent_think = cur
                else:
                    sent_text = cur
                if not delta:
                    continue
                grew = True
                yield {'content': delta, 'type': kind, 'finish_reason': None}
            if grew:
                stall_since = time.time()
                if content_at is None:
                    content_at = time.time()
            if (sent_text or sent_think) and stall_since is not None and \
                    time.time() - stall_since > self.REPLY_STALL_TIMEOUT:
                break
        if not (sent_text or sent_think):
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
        _start_reaper()
        try:
            self._last_used = time.time()
            with self._lock:
                self._ensure()
            for attempt in (1, 2):
                emitted = 0
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
                    for chunk in self._drain_reply():
                        emitted += 1
                        yield chunk
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
                    # A dead tab is retryable, but ONLY pre-stream: once
                    # chunks reached the client, replaying the prompt would
                    # duplicate the answer (the router has the same rule for
                    # provider fallbacks).
                    if attempt == 2 or emitted or not _is_page_death(e):
                        raise
                    logger.warning('chatgpt relay: %s pre-stream; rebuilding '
                                   'and retrying once', type(e).__name__)
                    try:
                        with self._lock:
                            self._build()
                    except Exception as e2:  # noqa: BLE001
                        logger.warning('chatgpt relay rebuild failed: %s', e2)
                        raise
        finally:
            self._last_used = time.time()
            self._busy.release()


_RELAY: Optional[ChatGPTRelay] = None
_RELAY_LOCK = threading.Lock()
_RELAY_REAPER: Optional[threading.Thread] = None


def _stale_after() -> float:
    """Idle seconds after which the reaper closes the relay browser
    (``I4F_CHATGPT_RELAY_STALE_AFTER``, default 10 min — matches the z.ai
    session reaper; 0/empty keeps the browser resident forever)."""
    raw = (os.getenv('I4F_CHATGPT_RELAY_STALE_AFTER', '600') or '').strip()
    try:
        val = float(raw) if raw else 0.0
    except ValueError:
        return 600.0
    return max(60.0, val) if val > 0 else float('inf')


def _reaper_loop(stale_after: float) -> None:
    """Periodically close the relay browser idle beyond ``stale_after``."""
    interval = max(30.0, min(60.0, stale_after / 4)) \
        if stale_after != float('inf') else 60.0
    while True:
        time.sleep(interval)
        try:
            get_relay().reap_if_idle(stale_after)
        except Exception:  # noqa: BLE001 — reaping is best effort
            logger.debug('chatgpt relay reaper failed', exc_info=True)


def _start_reaper() -> None:
    """Start the idle-session reaper once, when the relay is first used."""
    global _RELAY_REAPER
    with _RELAY_LOCK:
        if _RELAY_REAPER is not None and _RELAY_REAPER.is_alive():
            return
        _RELAY_REAPER = threading.Thread(
            target=_reaper_loop, args=(_stale_after(),),
            name='chatgpt-relay-reaper', daemon=True)
        _RELAY_REAPER.start()


def get_relay() -> ChatGPTRelay:
    """Process-wide relay singleton."""
    global _RELAY
    with _RELAY_LOCK:
        if _RELAY is None:
            _RELAY = ChatGPTRelay()
        return _RELAY
