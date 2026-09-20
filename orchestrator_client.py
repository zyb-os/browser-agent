"""
orchestrator_client.py

Connects the browser-agent to the agent-orchestrator, implementing the full
protocol defined in AGENT_MANIFEST.md.

Protocol checklist (§14):
  ✓ POST /api/v1/agents/register — capability schema + required_settings
  ✓ WS /ws/{agent_id} — connect immediately after registration
  ✓ Close code 4004 — re-register then reconnect
  ✓ Exponential-backoff auto-reconnect (cap: 60 s)
  ✓ Heartbeat every 15 s — status, current_load, active_tasks, metrics
  ✓ task_request → dedicated browser thread (run_task) → task_response
  ✓ Respects task timeout_ms hint
  ✓ status_update sent on task start / finish (available ↔ busy)
  ✓ Status machine: starting → available → busy → draining → offline
  ✓ Metrics: tasks_completed, tasks_failed, avg_response_time_ms, uptime_seconds
  ✓ agent_registered / agent_offline / error / broadcast / discovery_response handlers
  ✓ Graceful shutdown on SIGINT/SIGTERM: draining → wait → DELETE → WS close
  ✓ Anthropic API key read from orchestrator settings; falls back to env var
  ✓ ask_user tool: returns structured followup_request to the planner

Usage:
    python orchestrator_client.py [--orchestrator-url http://localhost:8000]
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import json
import logging
import queue as _queue
import signal
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import httpx
import websockets
import websockets.exceptions

import functools

from agent import BrowserAgent, BudgetExhausted, update_system_prompt_template
from browser import BrowserController

# ── Stable agent identity ──────────────────────────────────────────────────
# Generate once on first run, reuse forever so re-registrations upsert the
# existing record instead of creating a duplicate.  See §3 of AGENT_MANIFEST.

_AGENT_ID_FILE = Path(".agent_id")


def _stable_agent_id() -> str:
    """Read the persisted agent UUID from disk, or generate and save a new one."""
    if _AGENT_ID_FILE.exists():
        return _AGENT_ID_FILE.read_text().strip()
    new_id = str(uuid.uuid4())
    _AGENT_ID_FILE.write_text(new_id)
    logger.info("Generated new stable agent ID: %s → %s", new_id, _AGENT_ID_FILE)
    return new_id

logger = logging.getLogger(__name__)

# ── Agent identity ─────────────────────────────────────────────────────────

AGENT_NAME = "browser-agent"
AGENT_VERSION = "1.0.0"
AGENT_DESCRIPTION = "AI-powered browser agent for web research and task completion."

# ── Registration payload ───────────────────────────────────────────────────

def _load_default_prompt() -> str:
    _pf = Path(__file__).parent / "prompts" / "system_prompt.md"
    try:
        return _pf.read_text(encoding="utf-8").strip()
    except Exception:
        return ""


REGISTRATION_PAYLOAD: dict = {
    "name":           AGENT_NAME,
    "description":    AGENT_DESCRIPTION,
    "version":        AGENT_VERSION,
    "default_prompt": _load_default_prompt(),
    "capabilities": [
        {
            "name": "browse_web",
            "description": "Navigate websites, perform web research, extract and scrape content, fill and submit forms, complete shopping and e-commerce tasks, take screenshots of web pages, locate and interact with specific elements, visit URLs, click buttons and links, and automate complex multi-step browsing workflows from natural language instructions. OUTPUT: returns {summary: str} only — a plain-text summary of what was done. Never reference specific fields like .order_number or .delivery_date from browse_web output.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": "Natural language browsing task description",
                    },
                    "followup_answers": {
                        "type": "object",
                        "description": "Optional follow-up answers provided by planner for resumed execution.",
                    },
                },
                "required": ["task"],
            },
            "tags": ["browser", "web", "research", "shopping", "ai", "navigate", "scrape", "automate", "visit", "screenshot", "website", "extract", "click", "form", "find", "open", "internet", "crawl"],
            "cost": {"type": "per_call", "estimated_cost_usd": 0.015},
        }
    ],
    "tags": ["browser", "web", "ai", "claude", "playwright", "chrome-mcp"],
    "required_settings": [
        {
            "key": "browser_backend",
            "label": "Browser Backend",
            "type": "string",
            "required": False,
            "description": (
                "chrome-mcp — control a real Chrome browser via the Chrome MCP extension (default). "
                "playwright — launch a headless/visible Chromium instance managed by Playwright."
            ),
            "default": "chrome-mcp",
            "options": ["chrome-mcp", "playwright"],
        },
        {
            "key": "chrome_agent_ws_port",
            "label": "Chrome Extension WebSocket Port",
            "type": "integer",
            "required": False,
            "description": (
                "Port the Chrome MCP extension connects to (default: 8765). "
                "Only used when browser_backend = chrome-mcp."
            ),
            "default": 8765,
        },
        {
            "key": "anthropic_api_key",
            "label": "Anthropic API Key",
            "type": "secret",
            "required": False,
            "description": "API key for Claude",
        },
        {
            "key": "openai_api_key",
            "label": "OpenAI API Key",
            "type": "secret",
            "required": False,
            "description": "API key for GPT models",
        },
        {
            "key": "google_api_key",
            "label": "Google API Key",
            "type": "secret",
            "required": False,
            "description": "API key for Gemini models",
        },
        {
            "key": "provider",
            "label": "LLM Provider",
            "type": "string",
            "required": False,
            "description": "anthropic | openai | gemini | ollama | lmstudio | openai-compatible | groq | xai | mistral | deepseek",
            "default": "anthropic",
            "options": ["anthropic", "openai", "gemini", "ollama", "lmstudio", "openai-compatible", "groq", "xai", "mistral", "deepseek"],
        },
        {
            "key": "model",
            "label": "Model Name",
            "type": "string",
            "required": False,
            "description": "Model override. Uses provider default if blank.",
        },
        {
            "key": "headless",
            "label": "Run Browser Headless",
            "type": "boolean",
            "required": False,
            "description": "Run Chromium without a visible window (default: false)",
            "default": False,
        },
        {
            "key": "browser_llm_retry_count",
            "label": "LLM Retry Count",
            "type": "integer",
            "required": False,
            "description": "Number of times to retry a failed LLM call before giving up and requesting a replan (default: 3).",
            "default": 3,
        },
        {
            "key": "browser_llm_retry_delay_s",
            "label": "LLM Retry Delay (seconds)",
            "type": "integer",
            "required": False,
            "description": "Seconds to wait between LLM retry attempts (default: 5).",
            "default": 5,
        },
        {
            "key": "screenshot_format",
            "label": "Screenshot Format",
            "type": "string",
            "required": False,
            "description": "Screenshot image format: png (lossless) or jpeg (smaller).",
            "default": "png",
            "options": ["png", "jpeg"],
        },
        {
            "key": "screenshot_quality",
            "label": "Screenshot JPEG Quality",
            "type": "integer",
            "required": False,
            "description": "JPEG quality for screenshots (30-95). Ignored for PNG.",
            "default": 70,
        },
        {
            "key": "browser_observe_mode",
            "label": "Observation Mode",
            "type": "string",
            "required": False,
            "description": (
                "How the agent observes each page. "
                "'tree' (default) extracts an accessibility tree (~150-300 tokens, no image). "
                "'screenshot' sends a full PNG (~2000 tokens) — use for canvas or visual-only pages."
            ),
            "default": "tree",
            "options": ["tree", "screenshot"],
        },
    ],
}

# ── Constants ──────────────────────────────────────────────────────────────

HEARTBEAT_INTERVAL_S: int = 15       # heartbeat cadence
MAX_BACKOFF_S: int = 60              # reconnection backoff ceiling
DRAIN_TIMEOUT_S: int = 120           # max seconds to wait for tasks when draining


# ── Helpers ────────────────────────────────────────────────────────────────

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _envelope(
    sender_id: str,
    msg_type: str,
    payload: dict,
    recipient_id: str | None = None,
    correlation_id: str | None = None,
) -> str:
    return json.dumps({
        "id": str(uuid.uuid4()),
        "type": msg_type,
        "sender_id": sender_id,
        "recipient_id": recipient_id,
        "payload": payload,
        "timestamp": _now_iso(),
        "correlation_id": correlation_id,
    })


class FollowupRequired(Exception):
    """Raised from browser thread when ask_user requires planner-mediated follow-up."""

    def __init__(
        self,
        question: str,
        field_name: str | None = None,
        agent_capability: str | None = None,
        agent_task: str | None = None,
        intent: str | None = None,
    ) -> None:
        self.question = question
        self.field_name = field_name
        self.agent_capability = agent_capability
        # Default agent_task to the question itself so the dispatched agent
        # always has something actionable even if the caller didn't specify one.
        self.agent_task = agent_task or question
        self.intent = intent
        super().__init__(question)


# ── Main client ────────────────────────────────────────────────────────────

class OrchestratorClient:
    """
    Registers the browser-agent with the orchestrator and maintains the
    persistent WebSocket connection for the lifetime of the process.
    """

    def __init__(
        self,
        orchestrator_url: str  = "http://localhost:8000",
        provider_name:    str  = "anthropic",
        model:            str | None = None,
        backend:          str = "chrome-mcp",
    ) -> None:
        self._base          = orchestrator_url.rstrip("/")
        self._http          = httpx.AsyncClient(timeout=15)
        self._provider_name = provider_name
        self._model         = model
        self._backend       = backend          # resolved after _register(); CLI value is default

        # Identity — populated after registration
        self._agent_id: str = ""
        self._ws_url: str = ""

        # Status / metrics
        self._status: str = "starting"
        self._active_tasks: int = 0
        self._tasks_completed: int = 0
        self._tasks_failed: int = 0
        self._total_duration_ms: float = 0.0
        self._start_time: float = time.monotonic()

        self._common_settings: dict = {}

        # Browser supports exactly 1 concurrent task (single tab)
        self._task_sem = asyncio.Semaphore(1)
        self._shutting_down: bool = False
        # req_id → asyncio.Task: for task_cancel support
        self._running_tasks: dict[str, asyncio.Task] = {}

        # Pause/resume placeholders retained for backward compatibility.
        self._pause_event:    threading.Event = threading.Event()
        self._pause_response: str             = ""
        self._pause_sender:   str             = ""
        self._pause_req_id:   str             = ""
        self._loop:           asyncio.AbstractEventLoop | None = None
        self._active_ws = None

        # Browser components — initialised after settings are fetched.
        self._llm_provider = None  # ProxyProvider, set in _init_browser
        self._browser: BrowserController | None = None
        self._agent: BrowserAgent | None = None
        self._browser_thread = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="playwright"
        )

        # Chrome MCP backend components (used when _backend == "chrome-mcp")
        self._chrome = None          # ChromeController
        self._chrome_mcp_agent = None  # ChromeMcpAgent

    # ── Lifecycle ──────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Register, initialise the browser, then run the WS loop. Blocks until shutdown."""
        self._loop = asyncio.get_running_loop()
        loop = self._loop
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(
                sig, lambda: asyncio.create_task(self._graceful_shutdown())
            )

        await self._register()
        await self._init_browser()
        await self._connect_loop()

    # ── Registration ───────────────────────────────────────────────────────

    async def _register(self) -> None:
        """POST /api/v1/agents/register; store agent_id and ws_url."""
        url = f"{self._base}/api/v1/agents/register"
        logger.info("Registering with orchestrator at %s …", url)
        payload = {**REGISTRATION_PAYLOAD, "id": _stable_agent_id()}
        resp = await self._http.post(url, json=payload)
        resp.raise_for_status()
        data = resp.json()
        self._agent_id = data["agent_id"]
        self._ws_url = data["ws_url"]
        # Merge common settings first, then agent-specific settings on top so
        # dashboard-configured values override orchestrator-wide defaults.
        self._common_settings = {
            **data.get("common_settings", {}),
            **data.get("agent_settings", {}),
        }
        system_prompt = data.get("system_prompt", "")
        if system_prompt:
            update_system_prompt_template(system_prompt)
        logger.info("Registered — agent_id=%s  ws=%s", self._agent_id, self._ws_url)

    # ── Browser / backend init ─────────────────────────────────────────────

    async def _init_browser(self) -> None:
        """
        Resolve the browser backend, initialise the LLM provider, then start
        whichever backend was selected.

        chrome-mcp (default):
            Starts a ChromeController WebSocket server so the Chrome extension
            can connect.  Fully async — no dedicated thread needed.

        playwright:
            Lazily initialised — the Playwright browser is NOT started here.
            It starts on the first incoming task so the process stays lightweight
            until work actually arrives.
        """
        from providers import LlmRetryExhausted, ProxyProvider  # noqa: F401

        # Resolve backend: dashboard setting beats CLI flag (dashboard can override after registration)
        backend = self._common_settings.get("browser_backend") or self._backend or "chrome-mcp"
        self._backend = backend

        # ── LLM provider (shared by both backends) ─────────────────────────
        # Determine model — agent-specific "model" overrides common "default_model"
        model = (
            self._common_settings.get("model")
            or self._common_settings.get("default_model")
            or self._model
            or "claude-haiku-4-5"
        )
        provider      = self._common_settings.get("provider") or self._provider_name or ""
        proxy_url     = f"{self._base}/api/v1/llm/complete"
        retry_count   = int(self._common_settings.get("browser_llm_retry_count") or 3)
        retry_delay_s = float(self._common_settings.get("browser_llm_retry_delay_s") or 5)
        self._llm_provider = ProxyProvider(
            proxy_url=proxy_url,
            agent_id=str(self._agent_id),
            model=model,
            retry_count=retry_count,
            retry_delay_s=retry_delay_s,
            provider=provider,
        )
        logger.info("LLM provider: proxy  model: %s  backend: %s", self._llm_provider.model, backend)

        if backend == "chrome-mcp":
            await self._init_chrome_mcp()
        else:
            # playwright — lazy: browser starts on first task
            logger.info("Playwright backend — browser will start on the first incoming task.")

    async def _init_chrome_mcp(self) -> None:
        """Start the ChromeController WebSocket server and create the ChromeMcpAgent."""
        from chrome_controller import ChromeController
        from chrome_agent import ChromeMcpAgent

        port = int(self._common_settings.get("chrome_agent_ws_port") or 8765)
        self._chrome = ChromeController(port=port)
        await self._chrome.start()
        self._chrome_mcp_agent = ChromeMcpAgent(self._chrome, self._llm_provider)
        logger.info(
            "Chrome MCP backend ready — extension WebSocket listening on port %d. "
            "Install the Chrome MCP extension and it will connect automatically.",
            port,
        )

    def _start_browser_sync(self) -> None:
        """Runs inside the dedicated browser thread — safe to call Playwright sync API."""
        headless = str(self._common_settings.get("headless", "false")).lower() in ("true", "1", "yes")
        screenshot_format = self._common_settings.get("screenshot_format") or "jpeg"
        screenshot_quality = self._common_settings.get("screenshot_quality") or 50
        self._browser = BrowserController(
            headless=headless,
            screenshot_format=str(screenshot_format),
            screenshot_quality=int(screenshot_quality),
        )
        self._browser.start()
        self._agent = BrowserAgent(
            self._browser,
            self._llm_provider,
            human_input_fn=self._human_input_fn,
        )
        # Apply observe mode from common settings (default "tree")
        observe_mode = str(self._common_settings.get("browser_observe_mode") or "tree").strip().lower()
        self._agent.set_observe_mode(observe_mode)

    def _restart_browser_sync(self) -> None:
        """Stop and restart the browser with current settings (called from executor thread)."""
        if self._browser is not None:
            try:
                self._browser.stop()
            except Exception as exc:
                logger.warning("Browser stop during restart failed: %s", exc)
        self._start_browser_sync()
        logger.info("Browser restarted with headless=%s", self._browser._headless if self._browser else "?")

    def _human_input_fn(self, question: str, **kwargs) -> str:
        """
        Called from the browser thread when the agent needs user input.
        In orchestrator mode, convert this into a structured follow-up
        request handled by the planner; do not block waiting for task_resume.
        kwargs forwarded from ask_user tool_input (field_name, agent_capability, etc.)
        """
        logger.info("Agent paused — question: %s", question)
        raise FollowupRequired(question, **kwargs)

    # ── WebSocket connection loop ──────────────────────────────────────────

    async def _connect_loop(self) -> None:
        """Connect and reconnect with exponential backoff until shutdown."""
        backoff = 1.0
        while not self._shutting_down:
            try:
                logger.info("Connecting to %s …", self._ws_url)
                async with websockets.connect(self._ws_url) as ws:
                    backoff = 1.0  # reset on successful connect
                    await self._run_session(ws)

            except websockets.exceptions.ConnectionClosed as exc:
                code = exc.rcvd.code if exc.rcvd else None
                if code == 4004:
                    # Orchestrator doesn't recognise this agent_id — re-register
                    logger.warning("Orchestrator: unknown agent_id (4004) — re-registering …")
                    try:
                        await self._register()
                    except Exception as reg_exc:
                        logger.error("Re-registration failed: %s", reg_exc)
                elif code == 4003:
                    logger.info("Agent is disabled by orchestrator (4003) — will retry so dashboard enable can restore connection")
                    backoff = max(backoff, 10.0)
                elif self._shutting_down:
                    break
                else:
                    logger.warning("WS closed (code=%s) — retry in %.0fs", code, backoff)

            except (OSError, Exception) as exc:
                if self._shutting_down:
                    break
                logger.warning("WS error (%s) — retry in %.0fs", exc, backoff)

            if not self._shutting_down:
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, MAX_BACKOFF_S)

    async def _run_session(self, ws) -> None:
        """Run heartbeat + receive loop concurrently for one WS session."""
        self._active_ws = ws
        self._status = "available"
        logger.info("WebSocket session active — status: available")
        try:
            await asyncio.gather(
                self._heartbeat_loop(ws),
                self._recv_loop(ws),
            )
        finally:
            self._active_ws = None
            self._status = "offline"

    # ── Heartbeat ─────────────────────────────────────────────────────────

    async def _heartbeat_loop(self, ws) -> None:
        while True:
            await self._ws_send(ws, self._msg(
                "heartbeat",
                {
                    "status": self._status,
                    "current_load": float(min(self._active_tasks, 1)),
                    "active_tasks": self._active_tasks,
                    "expected_wait_time_ms": 30_000 if self._active_tasks else 0,
                    "metrics": self._metrics(),
                },
            ))
            await asyncio.sleep(HEARTBEAT_INTERVAL_S)

    # ── Receive loop ───────────────────────────────────────────────────────

    async def _recv_loop(self, ws) -> None:
        async for raw in ws:
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                logger.warning("Non-JSON WS frame ignored")
                continue
            mtype = msg.get("type", "?")
            sender = msg.get("sender_id", "?")
            payload_preview = json.dumps(msg.get("payload", {}))[:200]
            _lvl = logging.DEBUG if mtype in ("agent_registered", "agent_offline", "heartbeat_ack", "settings_push") else logging.INFO
            logger.log(_lvl, "← [%s] from=%s  %s", mtype, sender, payload_preview)
            await self._dispatch(ws, msg)

    async def _dispatch(self, ws, msg: dict) -> None:
        mtype = msg.get("type", "")
        payload = msg.get("payload", {})

        if mtype == "task_request":
            if self._backend == "chrome-mcp":
                asyncio.create_task(self._handle_chrome_task_request(ws, msg))
            else:
                asyncio.create_task(self._handle_task_request(ws, msg))

        elif mtype == "task_resume":
            # Legacy compatibility path; planner now uses followup_request.
            response = payload.get("response", "")
            logger.info("task_resume received (legacy): %r", response[:80])
            self._pause_response = response
            self._pause_event.set()

        elif mtype == "prompt_push":
            content = payload.get("content", "")
            if content:
                update_system_prompt_template(content)
                logger.info("Prompt push received (%d chars)", len(content))

        elif mtype == "agent_registered":
            logger.info("Peer joined: %s", payload.get("agent_id"))

        elif mtype == "agent_offline":
            logger.info(
                "Peer left: %s (reason: %s)",
                payload.get("agent_id"),
                payload.get("reason"),
            )

        elif mtype == "error":
            logger.error(
                "Orchestrator error [%s]: %s",
                payload.get("code"),
                payload.get("detail"),
            )

        elif mtype == "settings_push":
            settings: dict = payload.get("settings", {})
            if settings:
                self._common_settings.update(settings)

                # ── Backend change (requires restart) ──────────────────────
                if "browser_backend" in settings:
                    new_backend = settings["browser_backend"]
                    if new_backend != self._backend:
                        logger.warning(
                            "settings_push: browser_backend changed to %r "
                            "(currently %r) — restart the agent for this to take effect.",
                            new_backend, self._backend,
                        )

                # ── Model / provider hot-reload (works for both backends) ────
                new_model = (
                    self._common_settings.get("model")
                    or self._common_settings.get("default_model")
                )
                new_provider = self._common_settings.get("provider") or ""
                if self._llm_provider is not None:
                    if new_model:
                        self._llm_provider.model = new_model
                        logger.info("settings_push: model updated → %s", new_model)
                    if new_provider:
                        self._llm_provider._provider = new_provider
                        logger.info("settings_push: provider updated → %s", new_provider)
                    if not new_model and not new_provider:
                        logger.info("settings_push: %d setting(s) applied (no model/provider change)", len(settings))

                # ── Retry settings hot-reload ───────────────────────────────
                if self._llm_provider is not None:
                    new_retry_count = settings.get("browser_llm_retry_count")
                    new_retry_delay = settings.get("browser_llm_retry_delay_s")
                    if new_retry_count is not None or new_retry_delay is not None:
                        rc = int(new_retry_count) if new_retry_count is not None else self._llm_provider._retry_count
                        rd = float(new_retry_delay) if new_retry_delay is not None else self._llm_provider._retry_delay_s
                        self._llm_provider.update_retry_settings(rc, rd)

                # ── chrome_agent_ws_port change (requires restart) ──────────
                if "chrome_agent_ws_port" in settings and self._backend == "chrome-mcp":
                    logger.warning(
                        "settings_push: chrome_agent_ws_port changed — "
                        "restart the agent for the new port to take effect."
                    )

                # ── Playwright-only settings (ignored for chrome-mcp) ───────
                if self._backend == "playwright":
                    # Hot-reload headless: restart the browser if toggled
                    if "headless" in settings and self._browser is not None:
                        new_headless = str(settings["headless"]).lower() in ("true", "1", "yes")
                        if new_headless != self._browser.headless:
                            logger.info("settings_push: headless toggled → %s, restarting browser", new_headless)
                            loop = asyncio.get_running_loop()
                            asyncio.create_task(
                                loop.run_in_executor(self._browser_thread, self._restart_browser_sync),
                                name="browser-restart-headless",
                            )
                    # Hot-reload screenshot settings without restart
                    if self._browser is not None and (
                        "screenshot_format" in settings or "screenshot_quality" in settings
                    ):
                        fmt = settings.get("screenshot_format")
                        qual = settings.get("screenshot_quality")
                        loop = asyncio.get_running_loop()
                        asyncio.create_task(
                            loop.run_in_executor(
                                self._browser_thread,
                                lambda: self._browser.set_screenshot_settings(fmt, qual),
                            ),
                            name="browser-update-screenshot-settings",
                        )
                    # Hot-reload observe mode — applies immediately on the next task
                    if "browser_observe_mode" in settings and self._agent is not None:
                        new_mode = str(settings["browser_observe_mode"]).strip().lower()
                        self._agent.set_observe_mode(new_mode)

        elif mtype == "broadcast":
            logger.debug("Broadcast from %s: %s", msg.get("sender_id"), payload.get("content"))

        elif mtype == "discovery_response":
            agents = payload.get("agents", [])
            logger.debug("Discovery response: %d agent(s)", len(agents))

        elif mtype == "task_cancel":
            req_id = payload.get("task_id", "")
            task = self._running_tasks.get(req_id)
            if task and not task.done():
                task.cancel()
                logger.info("Cancelling browser task req_id=%s", req_id)
            else:
                logger.warning("task_cancel for unknown/completed browser task req_id=%s", req_id)

        elif mtype == "agent_restart":
            logger.info("Restart requested by orchestrator — shutting down for restart")
            asyncio.create_task(self._graceful_shutdown())
            import sys
            asyncio.get_event_loop().call_later(1.0, lambda: sys.exit(0))

        else:
            logger.debug("Unhandled message type: %r", mtype)

    # ── Task execution ─────────────────────────────────────────────────────

    async def _handle_chrome_task_request(self, ws, msg: dict) -> None:
        """
        Handle a browse_web task using the Chrome MCP backend.
        Runs entirely in the asyncio event loop — no thread pool needed.
        """
        req_id    = msg.get("id")
        sender_id = msg.get("sender_id")
        payload   = msg.get("payload", {})
        capability  = payload.get("capability")
        input_data  = payload.get("input_data", {})
        timeout_ms  = float(payload.get("timeout_ms") or 120_000)

        if capability != "browse_web":
            await self._ws_send(ws, self._msg(
                "task_response",
                {"success": False, "error": f"Unknown capability: {capability!r}"},
                recipient_id=sender_id, correlation_id=req_id,
            ))
            return

        task_text: str = input_data.get("task", "").strip()
        if not task_text:
            await self._ws_send(ws, self._msg(
                "task_response",
                {"success": False, "error": "input_data.task must be a non-empty string"},
                recipient_id=sender_id, correlation_id=req_id,
            ))
            return

        followup_answers: dict | None = input_data.get("followup_answers")
        if not isinstance(followup_answers, dict) or not followup_answers:
            followup_answers = None

        # Register task for cancellation (task_cancel handler looks up by req_id)
        _my_task = asyncio.current_task()
        if req_id and _my_task:
            self._running_tasks[req_id] = _my_task

        # Fast-fail if Chrome extension is not connected — avoids wasting 60
        # LLM iterations returning "Chrome extension is not connected" errors.
        if self._chrome is None or not self._chrome.is_connected:
            port = int(self._common_settings.get("chrome_agent_ws_port") or 8765)
            await self._ws_send(ws, self._msg(
                "task_response",
                {
                    "success": False,
                    "error": (
                        f"Chrome extension is not connected (WebSocket port {port}). "
                        "Open Chrome, install the Chrome MCP extension, and ensure it shows 'Connected'."
                    ),
                },
                recipient_id=sender_id, correlation_id=req_id,
            ))
            return

        async with self._task_sem:
            self._active_tasks += 1
            self._status = "busy"
            await self._send_status_update(ws)

            t0 = time.monotonic()
            try:
                assert self._chrome_mcp_agent is not None, "ChromeMcpAgent not initialised"
                timeout_s = timeout_ms / 1000.0
                result = await asyncio.wait_for(
                    self._chrome_mcp_agent.run_task_async(task_text, followup_answers, timeout_s),
                    timeout=timeout_s,
                )
                duration_ms = (time.monotonic() - t0) * 1000

                if result.get("followup_required"):
                    self._tasks_completed += 1
                    followup_payload: dict = {
                        "question":      result["question"],
                        "question_id":   str(uuid.uuid4()),
                        "answer_format": "text",
                    }
                    if result.get("field_name"):
                        followup_payload["field"] = result["field_name"]
                    if result.get("agent_capability"):
                        followup_payload["agent_capability"] = result["agent_capability"]
                        followup_payload["agent_task"] = result.get("agent_task", result["question"])
                    if result.get("intent"):
                        followup_payload["intent"] = result["intent"]
                    await self._ws_send(ws, self._msg(
                        "task_response",
                        {
                            "success": True,
                            "output_data": {"followup_request": followup_payload},
                            "duration_ms": round(duration_ms, 1),
                        },
                        recipient_id=sender_id, correlation_id=req_id,
                    ))

                elif result.get("success"):
                    self._tasks_completed += 1
                    self._total_duration_ms += duration_ms
                    logger.info(
                        "browse_web (chrome-mcp) completed in %.0f ms: %s…",
                        duration_ms, result.get("summary", "")[:80],
                    )
                    await self._ws_send(ws, self._msg(
                        "task_response",
                        {
                            "success": True,
                            "output_data": {"summary": result.get("summary", "")},
                            "duration_ms": round(duration_ms, 1),
                        },
                        recipient_id=sender_id, correlation_id=req_id,
                    ))

                else:
                    self._tasks_failed += 1
                    err = result.get("error", "Task failed")
                    logger.warning("browse_web (chrome-mcp) failed: %s", err)
                    await self._ws_send(ws, self._msg(
                        "task_response",
                        {"success": False, "error": err, "duration_ms": round(duration_ms, 1)},
                        recipient_id=sender_id, correlation_id=req_id,
                    ))

            except asyncio.TimeoutError:
                self._tasks_failed += 1
                duration_ms = (time.monotonic() - t0) * 1000
                err = f"Task timed out after {timeout_ms:.0f} ms"
                logger.warning(err)
                await self._ws_send(ws, self._msg(
                    "task_response",
                    {"success": False, "error": err, "duration_ms": round(duration_ms, 1)},
                    recipient_id=sender_id, correlation_id=req_id,
                ))

            except asyncio.CancelledError:
                logger.info("browse_web (chrome-mcp) task cancelled (req_id=%s)", req_id)
                raise

            except Exception as exc:
                self._tasks_failed += 1
                duration_ms = (time.monotonic() - t0) * 1000
                logger.exception("browse_web (chrome-mcp) raised an unhandled exception")
                await self._ws_send(ws, self._msg(
                    "task_response",
                    {"success": False, "error": str(exc), "duration_ms": round(duration_ms, 1)},
                    recipient_id=sender_id, correlation_id=req_id,
                ))

            finally:
                self._active_tasks -= 1
                self._status = "draining" if self._shutting_down else "available"
                await self._send_status_update(ws)
                if req_id:
                    self._running_tasks.pop(req_id, None)

    async def _handle_task_request(self, ws, msg: dict) -> None:
        """
        Receive a browse_web task_request, run BrowserAgent.run_task in a thread
        (it is synchronous / Playwright), and reply with a task_response.
        """
        req_id = msg.get("id")
        sender_id = msg.get("sender_id")
        payload = msg.get("payload", {})
        capability = payload.get("capability")
        input_data = payload.get("input_data", {})
        timeout_ms: float | None = payload.get("timeout_ms")

        # ── validate ──────────────────────────────────────────────────────
        if capability != "browse_web":
            await self._ws_send(ws, self._msg(
                "task_response",
                {"success": False, "error": f"Unknown capability: {capability!r}"},
                recipient_id=sender_id,
                correlation_id=req_id,
            ))
            return

        task_text: str = input_data.get("task", "").strip()
        if not task_text:
            await self._ws_send(ws, self._msg(
                "task_response",
                {"success": False, "error": "input_data.task must be a non-empty string"},
                recipient_id=sender_id,
                correlation_id=req_id,
            ))
            return

        # Register task for cancellation (task_cancel handler looks up by req_id)
        _my_task = asyncio.current_task()
        if req_id and _my_task:
            self._running_tasks[req_id] = _my_task

        # Legacy pause metadata retained for compatibility.
        self._pause_sender = sender_id or ""
        self._pause_req_id = req_id or ""

        followup_answers: dict | None = input_data.get("followup_answers")
        if not isinstance(followup_answers, dict) or not followup_answers:
            followup_answers = None

        # ── Lazy Playwright browser start (first task only) ───────────────
        if self._browser is None:
            logger.info("First task received — starting Playwright browser now …")
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(self._browser_thread, self._start_browser_sync)
            logger.info("Playwright browser started — agent ready.")

        # ── acquire single-tab semaphore ──────────────────────────────────
        async with self._task_sem:
            self._active_tasks += 1
            self._status = "busy"
            await self._send_status_update(ws)

            reply_context: dict | None = input_data.get("_reply_context") if isinstance(input_data, dict) else None

            t0 = time.monotonic()
            try:
                logger.info("browse_web task received: %r", task_text[:120])

                loop = asyncio.get_running_loop()

                # ── Budget parameters ─────────────────────────────────────────
                # Trigger graceful wrap-up at 80 % of the allowed time window,
                # leaving the remaining 20 % for the wrap-up LLM call + response.
                _soft_deadline: float | None = None
                if timeout_ms:
                    _soft_deadline = time.monotonic() + (timeout_ms / 1000) * 0.80
                # Max context tokens: read from agent settings, fall back to 100 K.
                _max_ctx = int(self._common_settings.get("browser_max_context_tokens") or 0) or 100_000

                # ── Notification bridge: sync callbacks → asyncio queue ────────
                _notify_q: _queue.SimpleQueue = _queue.SimpleQueue()

                def _plan_cb(text: str) -> None:
                    _notify_q.put(("plan", text))

                def _step_cb(tool_name: str, summary: str) -> None:
                    _notify_q.put(("step", tool_name, summary))

                if reply_context and self._agent:
                    self._agent.plan_callback = _plan_cb
                    self._agent.step_callback = _step_cb

                # ── Async consumer that drains _notify_q while task runs ───────
                _SENTINEL = object()

                async def _consume_notifications() -> None:
                    while True:
                        try:
                            item = await loop.run_in_executor(None, _notify_q.get)
                        except Exception:
                            break
                        if item is _SENTINEL:
                            break
                        if item[0] == "plan" and reply_context:
                            msg = f"🗺️ *Browser plan:*\n{item[1]}"
                            await self._notify_user(reply_context, msg)
                        elif item[0] == "step" and reply_context:
                            await self._notify_user(reply_context, f"⚙️ {item[2]}")

                consumer_task: asyncio.Task | None = None
                if reply_context:
                    consumer_task = asyncio.create_task(_consume_notifications())

                coro = loop.run_in_executor(
                    self._browser_thread,
                    functools.partial(
                        self._agent.run_task,  # type: ignore[union-attr]
                        task_text,
                        followup_answers,
                        soft_deadline=_soft_deadline,
                        max_context_tokens=_max_ctx,
                    ),
                )
                if timeout_ms:
                    summary = await asyncio.wait_for(coro, timeout=timeout_ms / 1000)
                else:
                    summary = await coro

                # Stop consumer
                if consumer_task:
                    _notify_q.put(_SENTINEL)
                    await consumer_task

                # Clear callbacks
                if self._agent:
                    self._agent.plan_callback = None
                    self._agent.step_callback = None

                duration_ms = (time.monotonic() - t0) * 1000
                self._tasks_completed += 1
                self._total_duration_ms += duration_ms

                logger.info(
                    "browse_web completed in %.0f ms: %s…",
                    duration_ms,
                    summary[:80],
                )

                # Notify user of final result
                if reply_context:
                    result_preview = summary[:300] + ("…" if len(summary) > 300 else "")
                    await self._notify_user(reply_context, f"✅ *Browser task complete:*\n{result_preview}")

                await self._ws_send(ws, self._msg(
                    "task_response",
                    {
                        "success": True,
                        "output_data": {"summary": summary},
                        "duration_ms": round(duration_ms, 1),
                    },
                    recipient_id=sender_id,
                    correlation_id=req_id,
                ))

            except FollowupRequired as followup:
                if consumer_task and not consumer_task.done():
                    _notify_q.put(_SENTINEL)
                    try:
                        await asyncio.wait_for(consumer_task, timeout=2.0)
                    except Exception:
                        consumer_task.cancel()
                if self._agent:
                    self._agent.plan_callback = None
                    self._agent.step_callback = None
                duration_ms = (time.monotonic() - t0) * 1000
                self._tasks_completed += 1
                followup_payload: dict = {
                    "question":    followup.question,
                    "question_id": str(uuid.uuid4()),
                    "answer_format": "text",
                }
                if followup.field_name:
                    followup_payload["field"] = followup.field_name
                if followup.agent_capability:
                    followup_payload["agent_capability"] = followup.agent_capability
                    followup_payload["agent_task"] = followup.agent_task
                if followup.intent:
                    followup_payload["intent"] = followup.intent
                await self._ws_send(ws, self._msg(
                    "task_response",
                    {
                        "success": True,
                        "output_data": {"followup_request": followup_payload},
                        "duration_ms": round(duration_ms, 1),
                    },
                    recipient_id=sender_id,
                    correlation_id=req_id,
                ))

            except BudgetExhausted as pause:
                # Agent proactively wrapped up before hitting a hard limit.
                if consumer_task and not consumer_task.done():
                    _notify_q.put(_SENTINEL)
                    try:
                        await asyncio.wait_for(consumer_task, timeout=2.0)
                    except Exception:
                        consumer_task.cancel()
                if self._agent:
                    self._agent.plan_callback = None
                    self._agent.step_callback = None
                duration_ms = (time.monotonic() - t0) * 1000
                self._tasks_completed += 1
                logger.info("browse_web graceful pause (%s) after %.0f ms", pause.reason, duration_ms)

                # Notify user through their originating channel
                if reply_context:
                    pause_msg = (
                        f"⏸️ *Browser task paused ({pause.reason}):*\n\n"
                        f"{pause.summary}\n\n"
                        f"_Reply to this thread with your instructions to continue._"
                    )
                    await self._notify_user(reply_context, pause_msg)

                # Return a followup_request — the planner will route the user's
                # reply back here as followup_answers, resuming from saved history.
                await self._ws_send(ws, self._msg(
                    "task_response",
                    {
                        "success":     True,
                        "output_data": {
                            "followup_request": {
                                "question":      pause.summary,
                                "question_id":   str(uuid.uuid4()),
                                "answer_format": "text",
                                "intent":        "graceful_pause",
                            }
                        },
                        "duration_ms": round(duration_ms, 1),
                    },
                    recipient_id=sender_id,
                    correlation_id=req_id,
                ))

            except asyncio.TimeoutError:
                if consumer_task and not consumer_task.done():
                    _notify_q.put(_SENTINEL)
                    consumer_task.cancel()
                if self._agent:
                    self._agent.plan_callback = None
                    self._agent.step_callback = None
                duration_ms = (time.monotonic() - t0) * 1000
                self._tasks_completed += 1
                logger.warning("browse_web timed out after %.0f ms — pausing for resume", duration_ms)

                # Preserve completed context so the next retry resumes from this point.
                progress_summary = (
                    f"Task '{task_text[:80]}' paused — timed out after {timeout_ms:.0f} ms. "
                    "Reply to continue from where I left off."
                )
                if self._agent is not None:
                    self._agent.inject_resume_placeholder(
                        f"timed out after {timeout_ms:.0f} ms"
                    )

                if reply_context:
                    await self._notify_user(
                        reply_context,
                        f"⏸️ *Browser task timed out after {timeout_ms:.0f} ms.*\n\n"
                        f"{progress_summary}\n\n_Reply to resume._",
                    )
                await self._ws_send(ws, self._msg(
                    "task_response",
                    {
                        "success": True,
                        "output_data": {
                            "followup_request": {
                                "question":      progress_summary,
                                "question_id":   str(uuid.uuid4()),
                                "answer_format": "text",
                                "intent":        "timeout_resume",
                            },
                        },
                        "duration_ms": round(duration_ms, 1),
                    },
                    recipient_id=sender_id,
                    correlation_id=req_id,
                ))

            except asyncio.CancelledError:
                if consumer_task and not consumer_task.done():
                    _notify_q.put(_SENTINEL)
                    consumer_task.cancel()
                if self._agent:
                    self._agent.plan_callback = None
                    self._agent.step_callback = None
                # Stop the Playwright browser to interrupt the thread — fire-and-forget
                # via the single-threaded executor so a subsequent task can restart it.
                if self._browser is not None:
                    asyncio.get_running_loop().run_in_executor(
                        self._browser_thread, self._browser.stop
                    )
                    self._browser = None
                logger.info("browse_web task cancelled (req_id=%s)", req_id)
                raise

            except Exception as exc:
                if consumer_task and not consumer_task.done():
                    _notify_q.put(_SENTINEL)
                    consumer_task.cancel()
                if self._agent:
                    self._agent.plan_callback = None
                    self._agent.step_callback = None
                duration_ms = (time.monotonic() - t0) * 1000
                self._tasks_failed += 1

                from providers import LlmRetryExhausted as _LlmRetryExhausted
                if isinstance(exc, _LlmRetryExhausted):
                    logger.error(
                        "browse_web LLM retries exhausted after %d attempt(s): %s",
                        exc.attempts, exc.last_error,
                    )
                    progress_parts = [f"Task '{task_text[:80]}' paused — LLM unavailable after {exc.attempts} attempt(s)."]
                    if exc.completed_steps:
                        progress_parts.append(f"Last steps: {'; '.join(exc.completed_steps[-3:])}")
                    if exc.current_url:
                        progress_parts.append(f"Current page: {exc.current_url}")
                    progress_summary = " ".join(progress_parts)

                    # Preserve completed context so the next retry resumes from here.
                    if self._agent is not None:
                        self._agent.inject_resume_placeholder(
                            f"llm_unavailable after {exc.attempts} attempt(s): {exc.last_error[:80]}"
                        )

                    notify_msg = (
                        f"⚠️ *LLM unavailable after {exc.attempts} attempt(s).*\n"
                        f"Last error: `{exc.last_error}`\n"
                        f"Current page: {exc.current_url or 'unknown'}\n"
                        "Task paused — will resume from this point on retry."
                    )
                    if reply_context:
                        await self._notify_user(reply_context, notify_msg)
                    await self._ws_send(ws, self._msg(
                        "task_response",
                        {
                            "success": True,
                            "output_data": {
                                "followup_request": {
                                    "question":      progress_summary,
                                    "question_id":   str(uuid.uuid4()),
                                    "answer_format": "text",
                                    "intent":        "llm_error_resume",
                                },
                            },
                            "duration_ms": round(duration_ms, 1),
                        },
                        recipient_id=sender_id,
                        correlation_id=req_id,
                    ))
                else:
                    logger.exception("browse_web task raised an unhandled exception")
                    if reply_context:
                        await self._notify_user(reply_context, f"❌ Browser task failed: {exc}")
                    await self._ws_send(ws, self._msg(
                        "task_response",
                        {
                            "success": False,
                            "error": str(exc),
                            "duration_ms": round(duration_ms, 1),
                        },
                        recipient_id=sender_id,
                        correlation_id=req_id,
                    ))

            finally:
                self._active_tasks -= 1
                self._status = "draining" if self._shutting_down else "available"
                await self._send_status_update(ws)
                if req_id:
                    self._running_tasks.pop(req_id, None)

    # ── Status update ──────────────────────────────────────────────────────

    async def _send_status_update(self, ws) -> None:
        await self._ws_send(ws, self._msg(
            "status_update",
            {
                "status": self._status,
                "current_load": float(min(self._active_tasks, 1)),
                "active_tasks": self._active_tasks,
                "metrics": self._metrics(),
            },
        ))

    # ── User notification helper ───────────────────────────────────────────

    async def _notify_user(self, reply_context: dict, message: str) -> None:
        """Send a status message back to the originating user channel."""
        if not reply_context or not message:
            return
        try:
            await self._http.post(
                f"{self._base}/api/v1/notify",
                json={**reply_context, "message": message, "sender_agent_id": self._agent_id},
                timeout=10.0,
            )
        except Exception as exc:
            logger.warning("_notify_user failed: %s", exc)

    # ── Graceful shutdown ──────────────────────────────────────────────────

    async def _graceful_shutdown(self) -> None:
        if self._shutting_down:
            return
        self._shutting_down = True
        logger.info("Shutdown signal received — entering draining state …")
        self._status = "draining"

        # Wait for the in-flight task to finish (browser is single-tab)
        deadline = time.monotonic() + DRAIN_TIMEOUT_S
        while self._active_tasks > 0 and time.monotonic() < deadline:
            await asyncio.sleep(0.5)

        if self._active_tasks:
            logger.warning("Drain timeout reached — %d task(s) still active", self._active_tasks)

        # Deregister from the orchestrator
        if self._agent_id:
            try:
                await self._http.delete(f"{self._base}/api/v1/agents/{self._agent_id}")
                logger.info("Deregistered from orchestrator.")
            except Exception as exc:
                logger.warning("Failed to deregister: %s", exc)

        # Stop backend resources
        if self._backend == "chrome-mcp":
            if self._chrome is not None:
                try:
                    await self._chrome.stop()
                except Exception:
                    pass
        else:
            # Stop Playwright browser on its own thread
            if self._browser:
                try:
                    loop = asyncio.get_running_loop()
                    await loop.run_in_executor(self._browser_thread, self._browser.stop)
                except Exception:
                    pass
        self._browser_thread.shutdown(wait=False)

        await self._http.aclose()
        logger.info("Shutdown complete.")

    # ── Helpers ────────────────────────────────────────────────────────────

    async def _ws_send(self, ws, msg_str: str) -> None:
        """Send a WebSocket frame and log it to the console."""
        msg = json.loads(msg_str)
        mtype = msg.get("type", "?")
        payload_preview = json.dumps(msg.get("payload", {}))[:200]
        log = logger.debug if mtype in ("heartbeat", "status_update") else logger.info
        log("→ [%s] to=%s  %s", mtype, msg.get("recipient_id") or "orchestrator", payload_preview)
        await ws.send(msg_str)

    def _msg(
        self,
        msg_type: str,
        payload: dict,
        recipient_id: str | None = None,
        correlation_id: str | None = None,
    ) -> str:
        return _envelope(self._agent_id, msg_type, payload, recipient_id, correlation_id)

    def _metrics(self) -> dict:
        n = self._tasks_completed + self._tasks_failed
        metrics: dict = {
            "tasks_completed": self._tasks_completed,
            "tasks_failed": self._tasks_failed,
            "avg_response_time_ms": (
                round(self._total_duration_ms / n, 1) if n else 0.0
            ),
            "uptime_seconds": round(time.monotonic() - self._start_time, 1),
        }
        if self._backend == "chrome-mcp":
            metrics["chrome_extension_connected"] = (
                self._chrome is not None and self._chrome.is_connected
            )
        return metrics


# Entry point is main.py. Run directly only for quick debugging:
#   python orchestrator_client.py [--orchestrator-url http://localhost:8000]
