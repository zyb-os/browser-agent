"""
chrome_controller.py

WebSocket server that the Chrome extension connects to.
Provides an async command interface to drive Chrome via the extension.

The agent runs the server; the Chrome extension connects as a client.
Each command is sent with a unique ID; the extension replies with the same ID.
"""
from __future__ import annotations

import asyncio
import json
import logging
import uuid
from typing import Any

import websockets
from websockets.server import WebSocketServerProtocol

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT = 30.0  # seconds per command


class ChromeNotConnectedError(Exception):
    """Raised when no Chrome extension is connected."""


class ChromeCommandError(Exception):
    """Raised when the extension returns an error or the command times out."""


class ChromeController:
    """
    Runs a local WebSocket server on `host:port`.
    The Chrome extension connects to this server and handles browser commands.
    """

    def __init__(self, host: str = "localhost", port: int = 8765) -> None:
        self._host = host
        self._port = port
        self._ws: WebSocketServerProtocol | None = None
        self._pending: dict[str, asyncio.Future] = {}
        self._server = None
        self._lock = asyncio.Lock()

    # ── Server lifecycle ──────────────────────────────────────────────────

    async def start(self) -> None:
        self._server = await websockets.serve(self._handle_connection, self._host, self._port)
        logger.info("Chrome controller listening on ws://%s:%d", self._host, self._port)

    async def stop(self) -> None:
        if self._server:
            self._server.close()
            await self._server.wait_closed()
        if self._ws:
            await self._ws.close()

    async def _handle_connection(self, ws: WebSocketServerProtocol) -> None:
        logger.info("Chrome extension connected from %s", ws.remote_address)
        async with self._lock:
            if self._ws is not None:
                logger.warning("Replacing existing extension connection")
                try:
                    await self._ws.close()
                except Exception:
                    pass
            self._ws = ws

        try:
            async for raw in ws:
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    logger.warning("Invalid JSON from extension: %r", str(raw)[:200])
                    continue

                msg_id = msg.get("id")
                if msg_id and msg_id in self._pending:
                    fut = self._pending.pop(msg_id)
                    if not fut.done():
                        if "error" in msg:
                            fut.set_exception(ChromeCommandError(msg["error"]))
                        else:
                            fut.set_result(msg.get("result", {}))
        except websockets.exceptions.ConnectionClosed:
            logger.info("Chrome extension disconnected")
        finally:
            async with self._lock:
                if self._ws is ws:
                    self._ws = None
            # Fail any still-pending commands
            for fut in list(self._pending.values()):
                if not fut.done():
                    fut.set_exception(ChromeNotConnectedError("Extension disconnected mid-command"))
            self._pending.clear()

    # ── Command dispatch ──────────────────────────────────────────────────

    async def execute(
        self,
        command: str,
        params: dict[str, Any] | None = None,
        timeout: float = _DEFAULT_TIMEOUT,
    ) -> Any:
        if self._ws is None:
            raise ChromeNotConnectedError(
                "Chrome extension is not connected. "
                "Load the extension in Chrome and make sure it shows 'Connected'."
            )

        msg_id = str(uuid.uuid4())
        loop = asyncio.get_event_loop()
        fut: asyncio.Future = loop.create_future()
        self._pending[msg_id] = fut

        payload = json.dumps({"id": msg_id, "command": command, "params": params or {}})
        try:
            await self._ws.send(payload)
        except Exception as exc:
            self._pending.pop(msg_id, None)
            raise ChromeNotConnectedError(f"Failed to send command to extension: {exc}") from exc

        try:
            return await asyncio.wait_for(asyncio.shield(fut), timeout=timeout)
        except asyncio.TimeoutError:
            self._pending.pop(msg_id, None)
            raise ChromeCommandError(f"Command '{command}' timed out after {timeout:.0f}s")

    @property
    def is_connected(self) -> bool:
        return self._ws is not None

    # ── High-level browser API ────────────────────────────────────────────

    async def navigate(self, url: str) -> dict:
        return await self.execute("navigate", {"url": url}, timeout=30.0)

    async def screenshot(self) -> str:
        """Returns base64-encoded PNG (without the data:image/... prefix)."""
        result = await self.execute("screenshot", {}, timeout=15.0)
        data_url: str = result.get("dataUrl", "")
        # Strip "data:image/png;base64," prefix if present
        if "," in data_url:
            return data_url.split(",", 1)[1]
        return data_url

    async def click(self, x: int, y: int) -> dict:
        return await self.execute("click", {"x": x, "y": y})

    async def type_text(self, text: str, clear_first: bool = False) -> dict:
        return await self.execute("typeText", {"text": text, "clearFirst": clear_first})

    async def key_press(self, key: str) -> dict:
        return await self.execute("keyPress", {"key": key})

    async def scroll(self, direction: str, amount: int = 3) -> dict:
        delta = amount * 120
        if direction == "up":
            delta = -delta
        return await self.execute("scroll", {"deltaY": delta})

    async def hover(self, x: int, y: int) -> dict:
        return await self.execute("hover", {"x": x, "y": y})

    async def wait(self, seconds: float) -> dict:
        await asyncio.sleep(seconds)
        return {"waited": seconds}

    async def go_back(self) -> dict:
        return await self.execute("goBack", {})

    async def get_url(self) -> str:
        result = await self.execute("getUrl", {})
        return result.get("url", "")

    async def get_title(self) -> str:
        result = await self.execute("getTitle", {})
        return result.get("title", "")

    async def get_page_text(self) -> str:
        result = await self.execute("getPageText", {})
        return result.get("text", "")

    async def evaluate_js(self, code: str) -> Any:
        result = await self.execute("evaluateJs", {"code": code}, timeout=15.0)
        return result.get("value")

    async def find_element(self, selector: str) -> dict:
        return await self.execute("findElement", {"selector": selector})

    async def get_viewport_size(self) -> dict:
        result = await self.execute("getViewportSize", {})
        return result  # {width, height}
