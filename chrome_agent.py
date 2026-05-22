"""
chrome_agent.py

Async LLM agent loop for the Chrome MCP backend of browser-agent.
Used when browser_backend = "chrome-mcp" (the default).

Drives a real Chrome browser via the chrome_controller WebSocket server
instead of using Playwright's headless Chromium.
"""
from __future__ import annotations

import json
import logging
import time
from typing import Any

from chrome_controller import ChromeController, ChromeNotConnectedError, ChromeCommandError
from profile import load_profile, save_profile, update_profile_field

logger = logging.getLogger(__name__)

MAX_ITERATIONS = 60
MAX_HISTORY_MESSAGES = 24
MAX_SCREENSHOTS_IN_CONTEXT = 1

SYSTEM_PROMPT = """You are an AI agent controlling a real Chrome browser via a set of tools.
Your goal is to complete the user's task by interacting with web pages.

## Guidelines

1. **Always start with a screenshot** to see the current page state before deciding what to do.
2. **Use coordinates precisely.** Pixel coordinates come from screenshots — top-left is (0,0).
3. **After every click**, take a screenshot to confirm what changed.
4. **For forms**: click the input field first to focus it, then call type_text.
5. **For searches**: type the query, then press Enter (or click the search button).
6. **Use find_element** to get exact center coordinates for known CSS selectors.
7. **Use get_page_text** to read long articles or extract structured text without images.
8. **Use evaluate_js** for advanced interactions (e.g. reading hidden data, triggering events).
9. **Be efficient** — avoid redundant screenshots or unnecessary waits.
10. **Call task_complete** with a clear summary once the task is done.
11. **Call ask_user** only when truly blocked with no way to proceed.

## Important
- The viewport coordinates are in pixels relative to the visible Chrome window.
- Dynamic pages (SPAs, React, Vue) may take a moment to update after clicks — use wait() if needed.
- If a page seems unchanged after clicking, try waiting 1-2 seconds and take another screenshot.
"""

# ── Tool definitions ──────────────────────────────────────────────────────────

TOOL_DEFINITIONS: list[dict] = [
    {
        "name": "navigate",
        "description": "Navigate the browser to a URL and wait for the page to load.",
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "Full URL to navigate to, including https:// (e.g. https://google.com)",
                }
            },
            "required": ["url"],
        },
    },
    {
        "name": "screenshot",
        "description": (
            "Take a screenshot of the current browser viewport. "
            "Always take a screenshot first to see what is on screen before deciding what to do next."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "click",
        "description": (
            "Click at specific pixel coordinates in the browser viewport. "
            "Use screenshot first to identify element coordinates. "
            "After clicking, a new screenshot is automatically taken."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "x": {"type": "integer", "description": "X coordinate (pixels from left edge)"},
                "y": {"type": "integer", "description": "Y coordinate (pixels from top edge)"},
            },
            "required": ["x", "y"],
        },
    },
    {
        "name": "type_text",
        "description": (
            "Type text into the currently focused element. "
            "Always click the input field first to focus it, then call type_text."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Text to type"},
                "clear_first": {
                    "type": "boolean",
                    "description": "Clear existing text before typing (default: false)",
                    "default": False,
                },
            },
            "required": ["text"],
        },
    },
    {
        "name": "key_press",
        "description": (
            "Press a keyboard key. Common values: Enter, Tab, Escape, "
            "ArrowDown, ArrowUp, ArrowLeft, ArrowRight, Backspace, Delete, Space."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "key": {"type": "string", "description": "Key name (e.g. Enter, Tab, Escape, ArrowDown)"}
            },
            "required": ["key"],
        },
    },
    {
        "name": "scroll",
        "description": "Scroll the page up or down.",
        "input_schema": {
            "type": "object",
            "properties": {
                "direction": {
                    "type": "string",
                    "enum": ["up", "down"],
                    "description": "Direction to scroll",
                },
                "amount": {
                    "type": "integer",
                    "description": "Scroll units (1–10). Each unit is ~120px. Default: 3.",
                    "default": 3,
                },
            },
            "required": ["direction"],
        },
    },
    {
        "name": "hover",
        "description": "Move the mouse cursor to specific coordinates to reveal tooltips or dropdown menus.",
        "input_schema": {
            "type": "object",
            "properties": {
                "x": {"type": "integer", "description": "X coordinate"},
                "y": {"type": "integer", "description": "Y coordinate"},
            },
            "required": ["x", "y"],
        },
    },
    {
        "name": "wait",
        "description": "Pause execution for a number of seconds. Use after navigation or clicking to wait for content to load.",
        "input_schema": {
            "type": "object",
            "properties": {
                "seconds": {
                    "type": "number",
                    "description": "Seconds to wait (0.5–15). Default: 2.",
                    "default": 2,
                }
            },
            "required": ["seconds"],
        },
    },
    {
        "name": "go_back",
        "description": "Navigate back to the previous page in browser history.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "get_page_text",
        "description": (
            "Get the full visible text content of the current page as plain text. "
            "Useful for reading long-form content or extracting data without image analysis."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "evaluate_js",
        "description": (
            "Execute JavaScript in the page context and return the result. "
            "Use for complex DOM queries, extracting data, or interactions that simple clicks cannot achieve."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "JavaScript expression to evaluate. Must return a serialisable value.",
                }
            },
            "required": ["code"],
        },
    },
    {
        "name": "find_element",
        "description": (
            "Find an element by CSS selector and return its text, center coordinates, and attributes. "
            "Use the returned centerX/centerY to click the element precisely."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "selector": {
                    "type": "string",
                    "description": "CSS selector (e.g. '#submit-btn', 'input[name=q]', '.product-title')",
                }
            },
            "required": ["selector"],
        },
    },
    {
        "name": "get_user_profile",
        "description": (
            "Retrieve the full user preference profile. "
            "Call this at the start of a task to understand the user's budget, "
            "brand preferences, use cases, and past purchases."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "update_user_profile",
        "description": (
            "Save a new preference or piece of information learned about the user. "
            "Use this whenever the user mentions their budget, preferred brands, "
            "use cases, or any other personal preference. "
            "Valid keys: budget_range, preferred_brands, disliked_brands, use_cases, "
            "priorities, location, currency, notes, purchase_history."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "key": {
                    "type": "string",
                    "description": "Profile field to update.",
                },
                "value": {
                    "description": (
                        "New value. For list fields (preferred_brands, use_cases, etc.) "
                        "pass a string or list of strings to append. "
                        "For scalar fields (budget_range, location, currency) pass a string."
                    ),
                },
            },
            "required": ["key", "value"],
        },
    },
    {
        "name": "task_complete",
        "description": (
            "Signal that the task is fully complete. Call this when you have finished all required steps. "
            "Provide a clear, well-formatted summary of findings and your recommendation."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "Final summary of findings and recommendation for the user.",
                }
            },
            "required": ["summary"],
        },
    },
    {
        "name": "ask_user",
        "description": (
            "Pause the task and request information you cannot obtain from the page. "
            "Before calling this tool, think: could another agent already connected to "
            "the orchestrator supply this automatically? For example, if you need a "
            "verification code from email, set agent_capability='read_gmail' and "
            "agent_task='Find the most recent verification email and return the code'. "
            "The planner will try that agent first and only fall back to asking the "
            "human if the agent is unavailable or fails. "
            "Use this for: OTP / verification codes (→ email/SMS agents), "
            "credentials the user stored somewhere, decisions between options, "
            "captcha responses, or anything else only a human can provide."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "The question or prompt to show the user (fallback if no agent can answer).",
                },
                "field_name": {
                    "type": "string",
                    "description": (
                        "Short snake_case key for this answer, e.g. 'verification_code', "
                        "'email_address'. Used to inject the answer back into the task context."
                    ),
                },
                "agent_capability": {
                    "type": "string",
                    "description": (
                        "Orchestrator capability name of an agent that might answer this "
                        "automatically, e.g. 'read_gmail', 'read_sms', 'read_calendar'. "
                        "The planner will discover and dispatch to this agent first."
                    ),
                },
                "agent_task": {
                    "type": "string",
                    "description": (
                        "Task description to send to the agent. Be specific: "
                        "'Find the most recent email from noreply@example.com and extract "
                        "the 6-digit verification code'. Defaults to the question text."
                    ),
                },
                "intent": {
                    "type": "string",
                    "description": "Semantic hint: email_code | otp | credential | preference | captcha | other",
                    "enum": ["email_code", "otp", "credential", "preference", "captcha", "other"],
                },
            },
            "required": ["question"],
        },
    },
]


# ── Sentinel exceptions (internal to agent loop) ──────────────────────────────

class _TaskComplete(Exception):
    def __init__(self, summary: str) -> None:
        self.summary = summary


class _FollowupRequired(Exception):
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
        self.agent_task = agent_task or question
        self.intent = intent


# ── Tool dispatcher ───────────────────────────────────────────────────────────

async def _execute_tool(
    tool_name: str,
    tool_input: dict[str, Any],
    chrome: ChromeController,
) -> tuple[str, str | None]:
    """
    Execute a browser tool and return (result_text, screenshot_b64_or_None).
    Raises _TaskComplete or _FollowupRequired for terminal tools.
    """
    try:
        if tool_name == "navigate":
            await chrome.navigate(tool_input["url"])
            import asyncio
            await asyncio.sleep(1.0)
            url = await chrome.get_url()
            title = await chrome.get_title()
            return f"Navigated to: {url}\nPage title: {title}", None

        elif tool_name == "screenshot":
            b64 = await chrome.screenshot()
            return "Screenshot taken.", b64

        elif tool_name == "click":
            x, y = int(tool_input["x"]), int(tool_input["y"])
            await chrome.click(x, y)
            import asyncio
            await asyncio.sleep(0.8)
            b64 = await chrome.screenshot()
            return f"Clicked at ({x}, {y}).", b64

        elif tool_name == "type_text":
            text = tool_input["text"]
            clear_first = bool(tool_input.get("clear_first", False))
            await chrome.type_text(text, clear_first=clear_first)
            return f"Typed text ({len(text)} chars).", None

        elif tool_name == "key_press":
            key = tool_input["key"]
            await chrome.key_press(key)
            import asyncio
            await asyncio.sleep(0.5)
            return f"Pressed key: {key}", None

        elif tool_name == "scroll":
            direction = tool_input["direction"]
            amount = int(tool_input.get("amount", 3))
            await chrome.scroll(direction, amount)
            import asyncio
            await asyncio.sleep(0.4)
            b64 = await chrome.screenshot()
            return f"Scrolled {direction} ({amount} units).", b64

        elif tool_name == "hover":
            x, y = int(tool_input["x"]), int(tool_input["y"])
            await chrome.hover(x, y)
            import asyncio
            await asyncio.sleep(0.3)
            return f"Hovered at ({x}, {y}).", None

        elif tool_name == "wait":
            seconds = float(tool_input.get("seconds", 2))
            seconds = max(0.1, min(seconds, 30.0))
            await chrome.wait(seconds)
            return f"Waited {seconds:.1f}s.", None

        elif tool_name == "go_back":
            await chrome.go_back()
            import asyncio
            await asyncio.sleep(1.2)
            url = await chrome.get_url()
            return f"Navigated back. Now at: {url}", None

        elif tool_name == "get_page_text":
            text = await chrome.get_page_text()
            if len(text) > 8000:
                text = text[:8000] + "\n\n[...truncated, page has more content]"
            return f"Page text content:\n{text}", None

        elif tool_name == "evaluate_js":
            code = tool_input["code"]
            result = await chrome.evaluate_js(code)
            return f"JavaScript result: {result}", None

        elif tool_name == "find_element":
            selector = tool_input["selector"]
            result = await chrome.find_element(selector)
            if result.get("found"):
                return (
                    f"Element found — tag: {result.get('tag')}, "
                    f"text: {result.get('text', '')!r}, "
                    f"value: {result.get('value', '')!r}, "
                    f"type: {result.get('type', '')}, "
                    f"center: ({result.get('centerX')}, {result.get('centerY')}), "
                    f"rect: {result.get('rect')}"
                ), None
            return f"Element not found for selector: {selector!r}", None

        elif tool_name == "get_user_profile":
            profile = load_profile()
            return json.dumps(profile, indent=2), None

        elif tool_name == "update_user_profile":
            profile = load_profile()
            key = tool_input["key"]
            value = tool_input["value"]
            profile = update_profile_field(profile, key, value)
            save_profile(profile)
            return f"Profile updated: {key} = {value!r}", None

        elif tool_name == "task_complete":
            raise _TaskComplete(tool_input["summary"])

        elif tool_name == "ask_user":
            raise _FollowupRequired(
                question=tool_input["question"],
                field_name=tool_input.get("field_name"),
                agent_capability=tool_input.get("agent_capability"),
                agent_task=tool_input.get("agent_task"),
                intent=tool_input.get("intent"),
            )

        else:
            return f"Unknown tool: {tool_name!r}", None

    except (_TaskComplete, _FollowupRequired):
        raise
    except ChromeNotConnectedError as exc:
        return (
            f"Error: Chrome extension is not connected. {exc}\n"
            "Make sure the Chrome extension is loaded and shows 'Connected' status."
        ), None
    except ChromeCommandError as exc:
        return f"Chrome command error in {tool_name!r}: {exc}", None
    except Exception as exc:
        return f"Unexpected error in {tool_name!r}: {exc}", None


# ── Screenshot history pruning ────────────────────────────────────────────────

def _drop_old_screenshots(messages: list[dict]) -> None:
    """Replace all but the most recent screenshot with a text placeholder."""
    positions: list[tuple] = []
    for i, msg in enumerate(messages):
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for j, block in enumerate(content):
            if not isinstance(block, dict):
                continue
            if block.get("type") == "image":
                positions.append((i, j, None))
            elif block.get("type") == "tool_result":
                for k, sub in enumerate(block.get("content") or []):
                    if isinstance(sub, dict) and sub.get("type") == "image":
                        positions.append((i, j, k))

    placeholder = {"type": "text", "text": "[screenshot removed to save context]"}
    for i, j, k in positions[:-1]:
        if k is None:
            messages[i]["content"][j] = placeholder
        else:
            messages[i]["content"][j]["content"][k] = placeholder


# ── Agent ─────────────────────────────────────────────────────────────────────

class ChromeMcpAgent:
    """
    Async LLM agent that drives Chrome via ChromeController.
    run_task_async() always returns a result dict — never raises (except asyncio.TimeoutError
    from the caller's wait_for, which is intentional).
    """

    def __init__(self, chrome: ChromeController, provider: Any) -> None:
        self._chrome = chrome
        self._provider = provider

    async def run_task_async(
        self,
        task: str,
        followup_answers: dict[str, str] | None = None,
        timeout_s: float = 120.0,
    ) -> dict[str, Any]:
        """
        Run a browsing task and return a result dict:
          {"success": True,  "summary": "..."}
          {"success": True,  "followup_required": True, "question": "...", ...}
          {"success": False, "error": "..."}
        """
        import asyncio

        deadline = time.monotonic() + timeout_s

        # Build initial user message
        user_content = task
        if followup_answers:
            user_content += "\n\nAdditional context from user:\n"
            for q, a in followup_answers.items():
                user_content += f"Q: {q}\nA: {a}\n"

        messages: list[dict] = [{"role": "user", "content": user_content}]
        screenshot_count = 0

        for iteration in range(MAX_ITERATIONS):
            if time.monotonic() >= deadline:
                return {
                    "success": False,
                    "error": f"Task timed out after {timeout_s:.0f}s (reached deadline before iteration {iteration})",
                }

            # LLM call — provider.complete is synchronous, run in executor
            try:
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(
                    None,
                    lambda msgs=messages: self._provider.complete(
                        messages=msgs,
                        tools=TOOL_DEFINITIONS,
                        system=SYSTEM_PROMPT,
                        max_tokens=4096,
                    ),
                )
            except Exception as exc:
                logger.warning("LLM call failed: %s", exc)
                return {"success": False, "error": f"LLM call failed: {exc}"}

            content, stop_reason = response

            messages.append({"role": "assistant", "content": content})

            # Natural completion (no tool calls)
            if stop_reason == "end_turn" or not any(
                c.get("type") == "tool_use" for c in content
            ):
                text = " ".join(
                    c.get("text", "") for c in content if c.get("type") == "text"
                ).strip()
                return {"success": True, "summary": text or "Task completed successfully."}

            # Execute all tool calls in this turn
            tool_results: list[dict] = []
            for block in content:
                if block.get("type") != "tool_use":
                    continue

                tool_name: str = block["name"]
                tool_input: dict = block.get("input", {})
                tool_use_id: str = block["id"]

                try:
                    result_text, screenshot_b64 = await _execute_tool(
                        tool_name, tool_input, self._chrome
                    )
                except _TaskComplete as tc:
                    return {"success": True, "summary": tc.summary}
                except _FollowupRequired as fr:
                    return {
                        "success": True,
                        "followup_required": True,
                        "question": fr.question,
                        "field_name": fr.field_name,
                        "agent_capability": fr.agent_capability,
                        "agent_task": fr.agent_task,
                        "intent": fr.intent,
                    }

                result_content: list[dict] = [{"type": "text", "text": result_text}]
                if screenshot_b64:
                    screenshot_count += 1
                    if screenshot_count > MAX_SCREENSHOTS_IN_CONTEXT:
                        _drop_old_screenshots(messages)
                    result_content.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": screenshot_b64,
                        },
                    })

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_use_id,
                    "content": result_content,
                })

            messages.append({"role": "user", "content": tool_results})

            # Trim history to avoid unbounded context growth
            if len(messages) > MAX_HISTORY_MESSAGES * 2:
                messages = messages[:1] + messages[-(MAX_HISTORY_MESSAGES * 2 - 1):]

        return {
            "success": False,
            "error": f"Reached max iterations ({MAX_ITERATIONS}) without completing the task.",
        }
