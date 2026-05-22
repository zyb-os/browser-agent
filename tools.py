"""Tool definitions for the Claude API and a dispatcher to execute them."""

import json
import logging
from typing import Any

from browser import BrowserController
from profile import load_profile, save_profile, update_profile_field

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ observe modes
# "tree"       — primary mode: accessibility tree snapshot, ref-based clicking
# "screenshot" — fallback/legacy mode: full screenshot + coordinate clicking

OBSERVE_MODE_TREE       = "tree"
OBSERVE_MODE_SCREENSHOT = "screenshot"

# ------------------------------------------------------------------ definitions

TOOL_DEFINITIONS = [
    {
        "name": "navigate",
        "description": (
            "Navigate the browser to a URL. Use this to open websites, "
            "search result pages, or product pages. Always include the full URL."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The full URL to navigate to (e.g. https://amazon.com).",
                }
            },
            "required": ["url"],
        },
    },
    {
        "name": "screenshot",
        "description": (
            "Take a screenshot of the current browser viewport. "
            "The image will be returned in your next turn so you can see the current page state. "
            "Use this frequently to understand what's on screen before clicking."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "click",
        "description": (
            "Click at a specific (x, y) coordinate on the browser viewport. "
            "Use the screenshot to identify coordinates. "
            "The viewport is 1280x800 pixels."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "x": {"type": "integer", "description": "Horizontal pixel coordinate (0-1280)."},
                "y": {"type": "integer", "description": "Vertical pixel coordinate (0-800)."},
            },
            "required": ["x", "y"],
        },
    },
    {
        "name": "type_text",
        "description": (
            "Type text into the currently focused element (e.g. a search box). "
            "Always click the input field first, then use this tool to type."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "The text to type.",
                }
            },
            "required": ["text"],
        },
    },
    {
        "name": "key_press",
        "description": (
            "Press a keyboard key. Common keys: Enter, Tab, Escape, ArrowDown, ArrowUp, "
            "Backspace, Delete, F5. Use 'Enter' to submit forms or confirm searches."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "key": {
                    "type": "string",
                    "description": "Key name as a Playwright key string (e.g. 'Enter', 'Tab', 'Escape').",
                }
            },
            "required": ["key"],
        },
    },
    {
        "name": "scroll",
        "description": "Scroll the page up or down to reveal more content.",
        "input_schema": {
            "type": "object",
            "properties": {
                "direction": {
                    "type": "string",
                    "enum": ["down", "up"],
                    "description": "Direction to scroll.",
                },
                "amount": {
                    "type": "integer",
                    "description": "Number of pixels to scroll. Default is 300.",
                },
            },
            "required": ["direction"],
        },
    },
    {
        "name": "hover",
        "description": (
            "Move the mouse over a coordinate without clicking. "
            "Useful for revealing dropdown menus, tooltips, or hover-activated elements."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "x": {"type": "integer", "description": "Horizontal pixel coordinate."},
                "y": {"type": "integer", "description": "Vertical pixel coordinate."},
            },
            "required": ["x", "y"],
        },
    },
    {
        "name": "wait",
        "description": (
            "Wait for a number of seconds. Use after navigating or clicking when "
            "the page needs extra time to load dynamic content."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "seconds": {
                    "type": "number",
                    "description": "Number of seconds to wait (0.5 to 5).",
                }
            },
            "required": ["seconds"],
        },
    },
    {
        "name": "go_back",
        "description": "Navigate back to the previous page in browser history.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "get_user_profile",
        "description": (
            "Retrieve the full user preference profile. "
            "Call this at the start of a task to understand the user's budget, "
            "brand preferences, use cases, and past purchases."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
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
            "Signal that the task is finished. Provide a clear, well-formatted summary "
            "of findings and your recommendation. This ends the agent loop."
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

# ------------------------------------------------------------------ tree-mode tool definitions
# These replace / supplement TOOL_DEFINITIONS when observe_mode == "tree".

_TREE_ONLY_TOOLS = [
    {
        "name": "observe",
        "description": (
            "Observe the current page state. Returns a compact accessibility tree "
            "listing every interactive element (links, buttons, inputs, headings) "
            "with a ref number (e.g. #3 button \"Add to cart\"). "
            "Use ref numbers with click_element / hover_element instead of coordinates. "
            "Automatically falls back to a screenshot when the tree is empty "
            "(canvas, visual-only, or custom-component pages)."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "click_element",
        "description": (
            "Click an element by its ref number from the last observe call "
            "(e.g. ref=3 clicks the element labelled #3 in the tree). "
            "Always call observe first on a new page to get fresh ref numbers. "
            "Fall back to click(x, y) only for canvas elements or when observe returns no refs."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ref": {
                    "type": "integer",
                    "description": "The ref number shown in the accessibility tree (e.g. 3 for #3).",
                }
            },
            "required": ["ref"],
        },
    },
    {
        "name": "hover_element",
        "description": (
            "Hover the mouse over an element by its ref number to reveal "
            "dropdown menus, tooltips, or hover-activated content. "
            "Use after observe to get the current ref numbers."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ref": {
                    "type": "integer",
                    "description": "The ref number from the accessibility tree.",
                }
            },
            "required": ["ref"],
        },
    },
]

# screenshot tool — kept in tree mode as an explicit fallback
_SCREENSHOT_TOOL = {
    "name": "screenshot",
    "description": (
        "Take a full-viewport screenshot. Use this only when observe returns an "
        "empty tree (canvas, heavily custom UI, or visual-only pages where the "
        "accessibility tree doesn't reflect what you need to see)."
    ),
    "input_schema": {"type": "object", "properties": {}, "required": []},
}

# The original screenshot tool definition (unchanged description for screenshot mode)
_SCREENSHOT_TOOL_PRIMARY = next(t for t in TOOL_DEFINITIONS if t["name"] == "screenshot")


def get_tool_definitions(observe_mode: str = OBSERVE_MODE_TREE) -> list[dict]:
    """
    Return the tool list appropriate for the given observe mode.

    "tree"       — replaces the screenshot tool with observe/click_element/hover_element;
                   screenshot is kept as an explicit fallback with a narrower description.
    "screenshot" — original TOOL_DEFINITIONS unchanged.
    """
    if observe_mode == OBSERVE_MODE_SCREENSHOT:
        return TOOL_DEFINITIONS

    # Tree mode: swap out the primary screenshot tool, inject tree tools
    base = [t for t in TOOL_DEFINITIONS if t["name"] != "screenshot"]
    return _TREE_ONLY_TOOLS + base + [_SCREENSHOT_TOOL]


# ------------------------------------------------------------------ dispatcher

class ToolExecutionError(Exception):
    pass


def execute_tool(
    tool_name: str,
    tool_input: dict[str, Any],
    browser: BrowserController,
    observe_mode: str = OBSERVE_MODE_TREE,
) -> tuple[str, str | None]:
    """
    Execute a tool and return (result_text, screenshot_b64_or_None).

    The second element is a base64 PNG string when the screenshot tool is called,
    so the caller can inject it into the next Claude message as an image block.
    """
    try:
        if tool_name == "observe":
            if observe_mode == OBSERVE_MODE_SCREENSHOT:
                # In screenshot mode, observe behaves identically to screenshot
                b64 = browser.screenshot()
                return "Screenshot taken. See the image in this message.", b64
            text, elements = browser.get_accessibility_snapshot()
            if not elements:
                logger.info("Accessibility tree empty — falling back to screenshot.")
                b64 = browser.screenshot()
                return (
                    "Accessibility tree was empty (canvas or visual-only page). "
                    "Falling back to screenshot — see image.",
                    b64,
                )
            return text, None

        elif tool_name == "click_element":
            result = browser.click_element(int(tool_input["ref"]))
            return result, None

        elif tool_name == "hover_element":
            result = browser.hover_element(int(tool_input["ref"]))
            return result, None

        elif tool_name == "navigate":
            result = browser.navigate(tool_input["url"])
            return result, None

        elif tool_name == "screenshot":
            b64 = browser.screenshot()
            return "Screenshot taken. See the image in this message.", b64

        elif tool_name == "click":
            result = browser.click(tool_input["x"], tool_input["y"])
            return result, None

        elif tool_name == "type_text":
            result = browser.type_text(tool_input["text"])
            return result, None

        elif tool_name == "key_press":
            result = browser.key_press(tool_input["key"])
            return result, None

        elif tool_name == "scroll":
            amount = tool_input.get("amount", 300)
            result = browser.scroll(tool_input["direction"], amount)
            return result, None

        elif tool_name == "hover":
            result = browser.hover(tool_input["x"], tool_input["y"])
            return result, None

        elif tool_name == "wait":
            result = browser.wait(tool_input["seconds"])
            return result, None

        elif tool_name == "go_back":
            result = browser.go_back()
            return result, None

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
            return tool_input["summary"], None

        else:
            raise ToolExecutionError(f"Unknown tool: {tool_name!r}")

    except ToolExecutionError:
        raise
    except Exception as exc:
        logger.exception("Tool %r failed.", tool_name)
        raise ToolExecutionError(f"Tool {tool_name!r} raised an error: {exc}") from exc
