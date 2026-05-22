"""
Patchright browser controller — wraps the sync API for clean tool dispatch.

Uses patchright (a drop-in Playwright fork) which patches Chromium at the binary
level to remove automation fingerprints: navigator.webdriver, CDP exposure,
automation extension flags, and dozens of other bot-detection signals.

Additional hardening applied here:
  - Launch args that suppress remaining automation tells
  - Realistic user-agent matching the installed Chromium build
  - Persistent browser profile (cookies / localStorage survive across sessions)
  - Extra navigator overrides via init script as belt-and-suspenders
"""

import base64
import logging
import random
from pathlib import Path

from patchright.sync_api import Browser, BrowserContext, Page, Playwright, sync_playwright

logger = logging.getLogger(__name__)

VIEWPORT_WIDTH  = 1280
VIEWPORT_HEIGHT = 800

# Persistent profile directory — keeps cookies/history across sessions so the
# browser looks like a returning user rather than a fresh install every time.
_PROFILE_DIR = Path(__file__).parent / ".browser_profile"

# Stealth init script injected into every page before any site JS runs.
# Patchright already handles navigator.webdriver; these cover additional signals
# that some advanced fingerprinters check.
_STEALTH_INIT_SCRIPT = """
(() => {
  // Mask automation-related chrome runtime properties
  if (window.chrome && window.chrome.runtime) {
    Object.defineProperty(window.chrome.runtime, 'connect', { get: () => undefined });
  }

  // Ensure plugins array is non-empty (empty = headless tell)
  if (navigator.plugins.length === 0) {
    Object.defineProperty(navigator, 'plugins', {
      get: () => [
        { name: 'PDF Viewer', filename: 'internal-pdf-viewer', description: 'Portable Document Format' },
        { name: 'Chrome PDF Viewer', filename: 'mhjfbmdgcfjbbpaeojofohoefgiehjai', description: '' },
      ],
    });
  }

  // Languages — headless often reports empty or just ['en-US']
  Object.defineProperty(navigator, 'languages', { get: () => ['en-US', 'en'] });

  // Permissions API — automation environments behave differently
  const _origQuery = window.navigator.permissions && window.navigator.permissions.query;
  if (_origQuery) {
    window.navigator.permissions.query = (params) =>
      params.name === 'notifications'
        ? Promise.resolve({ state: Notification.permission })
        : _origQuery.call(window.navigator.permissions, params);
  }
})();
"""

# Launch args that strip the remaining automation signals Patchright doesn't
# patch at the binary level.
_LAUNCH_ARGS = [
    "--start-maximized",
    # Remove the "Chrome is being controlled by automated software" banner
    "--disable-blink-features=AutomationControlled",
    # Disable various features that fingerprinters use to detect headless/CDP
    "--disable-features=IsolateOrigins,site-per-process,AutomationControlled",
    "--disable-site-isolation-trials",
    # Reduce WebGL/Canvas fingerprint consistency (minor help)
    "--disable-reading-from-canvas",
    # Miscellaneous hardening
    "--no-first-run",
    "--no-default-browser-check",
    "--disable-default-apps",
    "--disable-extensions-except=",
]


class BrowserController:
    """Manages a headed Chromium browser session for the agent."""

    def __init__(
        self,
        headless: bool = False,
        screenshot_format: str = "png",
        screenshot_quality: int = 70,
    ) -> None:
        self._headless = headless
        self._screenshot_format = self._normalise_screenshot_format(screenshot_format)
        self._screenshot_quality = self._clamp_quality(screenshot_quality)
        self._playwright: Playwright | None = None
        self._browser: Browser | None = None
        self._context: BrowserContext | None = None
        self._page: Page | None = None
        # Accessibility snapshot state — populated by get_accessibility_snapshot()
        self._last_snapshot: list[dict] = []

    @property
    def headless(self) -> bool:
        return self._headless

    @staticmethod
    def _normalise_screenshot_format(value: str) -> str:
        fmt = (value or "").strip().lower()
        if fmt in ("jpg", "jpeg"):
            return "jpeg"
        if fmt == "png":
            return "png"
        return "png"

    @staticmethod
    def _clamp_quality(value: int) -> int:
        try:
            q = int(value)
        except Exception:
            return 70
        return max(30, min(q, 95))

    def set_screenshot_settings(self, fmt: str | None, quality: int | None) -> None:
        if fmt is not None:
            self._screenshot_format = self._normalise_screenshot_format(fmt)
        if quality is not None:
            self._screenshot_quality = self._clamp_quality(quality)
        logger.info(
            "Screenshot settings updated: format=%s quality=%d",
            self._screenshot_format,
            self._screenshot_quality,
        )

    # ------------------------------------------------------------------ lifecycle

    def start(self) -> None:
        """Launch a Chromium browser with stealth configuration."""
        if self._browser is not None:
            return

        _PROFILE_DIR.mkdir(parents=True, exist_ok=True)

        self._playwright = sync_playwright().start()
        self._browser = self._playwright.chromium.launch(
            headless=self._headless,
            args=_LAUNCH_ARGS,
        )

        # Use the browser's actual user-agent rather than a hardcoded one.
        # Hardcoded UAs diverge from the real Chromium version over time and
        # become a detection signal themselves.
        real_ua = self._browser.new_browser_cdp_session().send(
            "Browser.getVersion"
        ).get("userAgent", "")

        self._context = self._browser.new_context(
            viewport={"width": VIEWPORT_WIDTH, "height": VIEWPORT_HEIGHT},
            user_agent=real_ua or None,
            # Locale/timezone signals should match the UA
            locale="en-US",
            timezone_id="America/New_York",
            # Pretend to have color-depth of a normal monitor
            color_scheme="light",
            # Accept common media types so the site doesn't see an odd profile
            extra_http_headers={"Accept-Language": "en-US,en;q=0.9"},
            # Persistent storage path — keeps cookies/localStorage across runs
            storage_state=str(_PROFILE_DIR / "storage.json")
            if ((_PROFILE_DIR / "storage.json").exists()) else None,
        )

        # Inject stealth overrides before any page JS runs
        self._context.add_init_script(_STEALTH_INIT_SCRIPT)

        self._page = self._context.new_page()
        logger.info("Browser started (patchright stealth Chromium, headless=%s).", self._headless)

    def stop(self) -> None:
        """Persist cookies/storage, then close the browser."""
        if self._context:
            try:
                # Save cookies and localStorage so the next session looks like a
                # returning user rather than a fresh install (major stealth benefit).
                _PROFILE_DIR.mkdir(parents=True, exist_ok=True)
                self._context.storage_state(path=str(_PROFILE_DIR / "storage.json"))
                logger.info("Browser profile saved to %s", _PROFILE_DIR)
            except Exception as exc:
                logger.warning("Failed to save browser profile: %s", exc)
            self._context = None
        if self._browser:
            self._browser.close()
            self._browser = None
        if self._playwright:
            self._playwright.stop()
            self._playwright = None
        self._page = None
        logger.info("Browser stopped.")

    @property
    def page(self) -> Page:
        if self._page is None:
            raise RuntimeError("Browser not started. Call start() first.")
        return self._page

    # ------------------------------------------------------------------ lifecycle helpers

    def _is_connected(self) -> bool:
        """Return True if the browser process is still alive and the page is usable."""
        try:
            if self._browser is None or not self._browser.is_connected():
                return False
            if self._context is None:
                return False
            if self._page is None or self._page.is_closed():
                return False
            return True
        except Exception:
            return False

    def _ensure_alive(self) -> None:
        """Restart the browser if it has crashed or been closed externally."""
        if not self._is_connected():
            logger.warning("Browser disconnected — restarting automatically.")
            try:
                self.stop()
            except Exception:
                pass
            self.start()

    # ------------------------------------------------------------------ actions

    def navigate(self, url: str) -> str:
        """Navigate to a URL. Returns the page title after load."""
        self._ensure_alive()
        if not url.startswith(("http://", "https://", "about:", "chrome:", "data:")):
            url = "https://" + url
        logger.info("Navigating to: %s", url)
        self.page.goto(url, wait_until="domcontentloaded", timeout=30_000)
        self.page.wait_for_timeout(1500)
        title = self.page.title()
        return f"Navigated to: {url} | Title: {title}"

    def screenshot(self) -> str:
        """Take a full-viewport screenshot and return it as base64."""
        self._ensure_alive()
        if self._screenshot_format == "jpeg":
            img_bytes = self.page.screenshot(
                full_page=False,
                type="jpeg",
                quality=self._screenshot_quality,
            )
        else:
            img_bytes = self.page.screenshot(full_page=False)
        encoded = base64.standard_b64encode(img_bytes).decode("utf-8")
        logger.debug(
            "Screenshot taken (%d bytes) format=%s quality=%d.",
            len(img_bytes),
            self._screenshot_format,
            self._screenshot_quality,
        )
        return encoded

    def click(self, x: int, y: int) -> str:
        """Click at the given viewport coordinates."""
        self._ensure_alive()
        logger.info("Clicking at (%d, %d).", x, y)
        # Small random offset and variable delay mimic human imprecision
        self.page.mouse.click(
            x + random.randint(-2, 2),
            y + random.randint(-2, 2),
        )
        self.page.wait_for_timeout(random.randint(600, 1100))
        return f"Clicked at ({x}, {y})."

    def type_text(self, text: str) -> str:
        """Type text at the current focus / active element."""
        self._ensure_alive()
        logger.info("Typing: %r", text)
        # Variable inter-key delay (30–90 ms) mimics human typing rhythm
        self.page.keyboard.type(text, delay=random.randint(30, 90))
        return f"Typed: {text!r}"

    def key_press(self, key: str) -> str:
        """Press a keyboard key (e.g. Enter, Tab, Escape, ArrowDown)."""
        self._ensure_alive()
        logger.info("Key press: %s", key)
        self.page.keyboard.press(key)
        self.page.wait_for_timeout(600)
        return f"Pressed key: {key}"

    def scroll(self, direction: str, amount: int = 300) -> str:
        """Scroll the page. direction='down' or 'up'. amount in pixels."""
        self._ensure_alive()
        delta = amount if direction.lower() == "down" else -amount
        self.page.mouse.wheel(0, delta)
        self.page.wait_for_timeout(500)
        return f"Scrolled {direction} by {amount}px."

    def hover(self, x: int, y: int) -> str:
        """Move the mouse to (x, y) without clicking (reveals tooltips/dropdowns)."""
        self._ensure_alive()
        self.page.mouse.move(x, y)
        self.page.wait_for_timeout(400)
        return f"Hovered at ({x}, {y})."

    def wait(self, seconds: float) -> str:
        """Pause execution for the specified number of seconds."""
        self._ensure_alive()
        ms = int(seconds * 1000)
        self.page.wait_for_timeout(ms)
        return f"Waited {seconds}s."

    def get_url(self) -> str:
        """Return the current page URL."""
        self._ensure_alive()
        return self.page.url

    def get_title(self) -> str:
        """Return the current page title."""
        self._ensure_alive()
        return self.page.title()

    def go_back(self) -> str:
        """Navigate back in browser history."""
        self._ensure_alive()
        self.page.go_back(wait_until="domcontentloaded", timeout=15_000)
        self.page.wait_for_timeout(800)
        return f"Went back. Now at: {self.page.url}"

    # ------------------------------------------------------------------ accessibility tree

    _SNAPSHOT_JS = """
    () => {
        const results = [];
        let ref = 1;
        const seen = new WeakSet();

        const SELECTORS = [
            'a[href]', 'button:not([disabled])',
            'input:not([type="hidden"])', 'textarea', 'select',
            '[role="button"]', '[role="link"]', '[role="textbox"]',
            '[role="checkbox"]', '[role="radio"]', '[role="combobox"]',
            '[role="listbox"]', '[role="option"]', '[role="menuitem"]',
            '[role="tab"]', '[role="switch"]', '[role="slider"]',
            'h1', 'h2', 'h3', 'h4', 'h5',
            'img[alt]:not([alt=""])', '[aria-label]:not(script)',
        ];

        const allEls = [];
        for (const sel of SELECTORS) {
            for (const el of document.querySelectorAll(sel)) {
                if (!seen.has(el)) { seen.add(el); allEls.push(el); }
            }
        }

        // Sort top-to-bottom, left-to-right by DOM position
        allEls.sort((a, b) => {
            const ar = a.getBoundingClientRect(), br = b.getBoundingClientRect();
            return Math.abs(ar.top - br.top) > 10 ? ar.top - br.top : ar.left - br.left;
        });

        for (const el of allEls) {
            const rect = el.getBoundingClientRect();
            if (rect.width < 2 || rect.height < 2) continue;
            if (rect.bottom < 0 || rect.top > window.innerHeight) continue;
            if (rect.right < 0 || rect.left > window.innerWidth) continue;
            const st = window.getComputedStyle(el);
            if (st.visibility === 'hidden' || st.display === 'none' || st.opacity === '0') continue;

            const tag  = el.tagName.toLowerCase();
            const role = el.getAttribute('role') || tag;
            const text = (
                el.getAttribute('aria-label') ||
                el.getAttribute('alt') ||
                el.getAttribute('placeholder') ||
                (el.innerText || el.textContent || '').trim()
            ).replace(/\\s+/g, ' ').slice(0, 100);

            results.push({
                ref:     ref++,
                role:    role,
                tag:     tag,
                text:    text,
                type:    el.getAttribute('type') || '',
                href:    (el.getAttribute('href') || '').slice(0, 120),
                cx:      Math.round(rect.left + rect.width  / 2),
                cy:      Math.round(rect.top  + rect.height / 2),
                focused: document.activeElement === el,
                checked: el.type === 'checkbox' || el.type === 'radio'
                         ? el.checked : null,
            });
        }
        return results;
    }
    """

    def get_accessibility_snapshot(self) -> tuple[str, list[dict]]:
        """
        Extract a compact accessibility tree of the current viewport.
        Stores elements in self._last_snapshot for use by click_element / hover_element.
        Returns (formatted_text, elements).  Falls back to ("", []) on error.
        """
        self._ensure_alive()
        try:
            elements: list[dict] = self.page.evaluate(self._SNAPSHOT_JS)
        except Exception as exc:
            logger.warning("Accessibility snapshot JS failed: %s", exc)
            return "", []

        self._last_snapshot = elements
        if not elements:
            return "", []

        lines = [f"[Page: {self.page.title()!r} | {self.page.url}]"]
        for el in elements:
            ref  = el["ref"]
            role = el["role"]
            text = el.get("text", "")
            label = f'"{text}"' if text else "(no label)"

            extras: list[str] = []
            if el.get("focused"):
                extras.append("focused")
            t = el.get("type", "")
            if t and t not in ("text", "submit", ""):
                extras.append(f"type={t}")
            href = el.get("href", "")
            if href.startswith("http") and len(href) < 70:
                extras.append(f"→{href}")
            checked = el.get("checked")
            if checked is True:
                extras.append("checked")
            elif checked is False:
                extras.append("unchecked")

            suffix = f" ({', '.join(extras)})" if extras else ""
            lines.append(f"#{ref} {role} {label}{suffix}")

        return "\n".join(lines), elements

    def click_element(self, ref: int) -> str:
        """Click an element by its ref ID from the last accessibility snapshot."""
        self._ensure_alive()
        el = next((e for e in self._last_snapshot if e["ref"] == ref), None)
        if el is None:
            return (
                f"Error: ref #{ref} not found in last snapshot. "
                "Call observe first to refresh the element list."
            )
        x, y = el["cx"], el["cy"]
        self.page.mouse.click(x + random.randint(-2, 2), y + random.randint(-2, 2))
        self.page.wait_for_timeout(random.randint(600, 1100))
        return f"Clicked #{ref} {el['role']} {el['text']!r} at ({x}, {y})."

    def hover_element(self, ref: int) -> str:
        """Hover over an element by its ref ID from the last accessibility snapshot."""
        self._ensure_alive()
        el = next((e for e in self._last_snapshot if e["ref"] == ref), None)
        if el is None:
            return (
                f"Error: ref #{ref} not found in last snapshot. "
                "Call observe first to refresh the element list."
            )
        x, y = el["cx"], el["cy"]
        self.page.mouse.move(x, y)
        self.page.wait_for_timeout(400)
        return f"Hovered #{ref} {el['role']} {el['text']!r} at ({x}, {y})."

    def is_running(self) -> bool:
        return self._is_connected()
