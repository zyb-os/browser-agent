"""Core agent loop — vision + browser tool execution (provider-agnostic)."""

import logging
import time
import uuid
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

from browser import BrowserController
from profile import load_profile, profile_summary
from providers import AnthropicProvider, LlmRetryExhausted, get_provider
from site_knowledge import SiteKnowledge
from tools import OBSERVE_MODE_TREE, OBSERVE_MODE_SCREENSHOT, execute_tool, get_tool_definitions

logger = logging.getLogger(__name__)


class BudgetExhausted(Exception):
    """
    Raised by BrowserAgent when the time or context-token budget is nearly
    exhausted.  The agent has already made a graceful wrap-up LLM call and
    injected the synthetic ask_user / [Awaiting user answer] placeholders into
    self.messages so that the next run_task call can resume seamlessly.
    """
    def __init__(self, summary: str, reason: str = "budget") -> None:
        self.summary = summary
        self.reason  = reason
        super().__init__(summary)


_WRAP_UP_PROMPT = (
    "[SYSTEM: Your time/context budget for this task is nearly exhausted. "
    "Do NOT call any more tools. Instead write a concise response with three sections:\n"
    "1. **Progress** – what you have accomplished so far\n"
    "2. **Remaining** – what still needs to be done to fully complete the original task\n"
    "3. **Next step** – one specific, actionable instruction the user can send to resume\n"
    "Keep it brief and actionable.]"
)

# Default context-token budget; trigger wrap-up when messages consume this many tokens.
# Anthropic Sonnet context is 200 K; we use 100 K as a conservative default so
# there is always headroom for the wrap-up call.
DEFAULT_MAX_CONTEXT_TOKENS = 100_000

# Trigger graceful wrap-up at this fraction of the budget (time or tokens).
_BUDGET_FRACTION = 0.80


def _estimate_tokens(messages: list[dict]) -> int:
    """Rough token estimate: ~4 chars per token across all message content."""
    total = 0
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            total += len(content)
        elif isinstance(content, list):
            for block in content:
                if not isinstance(block, dict):
                    continue
                for key in ("text", "input"):
                    val = block.get(key, "")
                    if isinstance(val, str):
                        total += len(val)
                    elif isinstance(val, dict):
                        total += len(str(val))
    return total // 4


MAX_SCREENSHOTS_IN_CONTEXT = 1  # Keep only the most recent screenshot to save tokens
MAX_HISTORY_MESSAGES = 20       # Max messages to carry across tasks (older turns dropped)
_MAX_TOOL_TEXT_CHARS   = 2000   # Trim long tool-result text blocks beyond this length

# ── Token-reduction constants ─────────────────────────────────────────────────
# (A) System-prompt compression
_COMPRESS_CTX_MARKER = "<compressed_context>"
_COMPRESS_CTX_END    = "</compressed_context>"
# Injected once into the system prompt on the very first LLM call of a session.
_COMPRESS_CTX_INSTRUCTION = (
    "\n\n[FIRST-TURN ONLY — strip this block from your response before the user sees it]\n"
    "Before your normal plan/response, emit exactly:\n"
    "<compressed_context>\n"
    "≤120 words capturing: your single most important operating rule, the available "
    "tools, and the current task goal. Future turns will use ONLY this block as the "
    "system prompt instead of the full instructions above.\n"
    "</compressed_context>\n"
    "Then continue with your normal response."
)

# (B) History rolling summary
_SUMMARIZE_EVERY_N_TURNS = 6   # compress after this many completed tool-turn pairs
_KEEP_RECENT_PAIRS       = 4   # always keep these many recent pairs verbatim
_SUMMARIZE_SYSTEM = (
    "You are a concise summariser for a browser-automation agent. "
    "Condense the conversation turns below into ≤200 words: URLs visited, "
    "actions taken, key findings, and any errors. Short bullet points only. "
    "Omit raw HTML, base64 data, and screenshot descriptions."
)

_PROMPT_FILE = Path(__file__).parent / "prompts" / "system_prompt.md"

_FALLBACK_PROMPT_TEMPLATE = """\
You are an expert browser agent controlling a real Chromium browser on behalf of the user.
You can see screenshots of the browser viewport (1280x800 px) and interact with it using tools.

{user_profile}

## How to operate

1. **Start every task** by calling `get_user_profile` to recall user preferences, then form a clear plan.
2. **Use screenshots when needed** — call `screenshot` to see the current state before acting. You may see messages prefixed with `[Fast path]` — these are cached steps from prior sessions that were executed without a screenshot; assume they succeeded and continue from the current page state.
3. **Locate elements by their pixel coordinates** in the screenshot. The viewport is 1280x800.
4. **Navigate like a human**: search for products, click results, read specs, compare options.
5. **Learn the user** — whenever the user reveals a preference (budget, brand, use case), call `update_user_profile` immediately to remember it for future sessions.
6. **Handle pagination and filtering**: scroll down to see more results, use filters when available.
7. **When you have enough information**, call `task_complete` with a clear, structured summary including your recommendation and why.

## Navigation tips
- After `navigate` or `click`, call `screenshot` to see the new state.
- If a page is slow to load, call `wait` (1-2 seconds) then `screenshot`.
- To type in a search box: click the box, then call `type_text`, then `key_press` with "Enter".
- For dropdowns: try `hover` first, then `click`.
- If you get stuck on a page, use `go_back` and try a different approach.

## Profile learning
Extract and save to the user profile anything the user mentions: budget, preferred brands,
use cases (travel, camping, office), feature priorities (fast charging, weight, capacity),
disliked products, past purchases. Always update the profile before ending a task.
"""


def _load_prompt_template() -> str:
    try:
        return _PROMPT_FILE.read_text(encoding="utf-8").strip()
    except Exception:
        return _FALLBACK_PROMPT_TEMPLATE.strip()


SYSTEM_PROMPT_TEMPLATE: str = _load_prompt_template()


def update_system_prompt_template(content: str) -> None:
    """Hot-reload the system prompt template (called on prompt_push from orchestrator)."""
    global SYSTEM_PROMPT_TEMPLATE
    if content and content.strip():
        SYSTEM_PROMPT_TEMPLATE = content.strip()
        logger.info("Browser system prompt updated (%d chars)", len(SYSTEM_PROMPT_TEMPLATE))


def _build_system_prompt() -> str:
    profile = load_profile()
    return SYSTEM_PROMPT_TEMPLATE.format(user_profile=profile_summary(profile))


def _image_block(b64_png: str) -> dict:
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": "image/png",
            "data": b64_png,
        },
    }


def _is_pure_tool_result_message(msg: dict) -> bool:
    """Return True if this user message contains only tool_result blocks."""
    if msg.get("role") != "user":
        return False
    content = msg.get("content", [])
    return (
        isinstance(content, list)
        and bool(content)
        and all(isinstance(b, dict) and b.get("type") == "tool_result" for b in content)
    )


def _repair_message_history(messages: list[dict]) -> list[dict]:
    """
    Validate that every assistant tool_use block has a corresponding tool_result
    in the immediately following user message, and that every user tool_result
    message has a preceding assistant message with matching tool_use blocks.

    Any incomplete or orphaned turns (e.g. from a previously interrupted run_task
    or aggressive history trimming) are silently removed so the API never sees a
    tool_use without its matching tool_result, or vice-versa.
    """
    repaired: list[dict] = []
    i = 0
    while i < len(messages):
        msg = messages[i]
        if msg.get("role") == "assistant":
            content = msg.get("content", [])
            tool_use_ids = [
                b["id"]
                for b in (content if isinstance(content, list) else [])
                if isinstance(b, dict) and b.get("type") == "tool_use" and "id" in b
            ]
            if tool_use_ids:
                next_idx = i + 1
                if next_idx < len(messages):
                    nxt = messages[next_idx]
                    nxt_content = nxt.get("content", [])
                    if isinstance(nxt_content, list):
                        result_ids = {
                            b.get("tool_use_id")
                            for b in nxt_content
                            if isinstance(b, dict) and b.get("type") == "tool_result"
                        }
                        if nxt.get("role") == "user" and all(
                            tid in result_ids for tid in tool_use_ids
                        ):
                            # Valid pair — keep both and advance past them
                            repaired.append(msg)
                            repaired.append(nxt)
                            i += 2
                            continue
                # No valid following tool_result message — drop this assistant turn.
                # Also drop the immediately following user message if it is a pure
                # tool_result message; it is now orphaned and would cause a 400 from
                # the API ("tool_result block without matching tool_use").
                logger.warning(
                    "Dropping incomplete assistant turn with %d dangling tool_use(s).",
                    len(tool_use_ids),
                )
                i += 1  # skip this assistant turn
                if i < len(messages) and _is_pure_tool_result_message(messages[i]):
                    logger.warning("Dropping orphaned tool_result user message.")
                    i += 1  # skip the now-orphaned user turn too
                continue

        # Guard against pure tool_result user messages that appear without a
        # preceding assistant tool_use turn (can happen after history trimming
        # cuts the conversation right before an assistant turn).
        if _is_pure_tool_result_message(msg):
            last_repaired = repaired[-1] if repaired else None
            if last_repaired is None or last_repaired.get("role") != "assistant":
                logger.warning("Dropping orphaned tool_result user message at history boundary.")
                i += 1
                continue

        repaired.append(msg)
        i += 1
    return repaired


def _trim_old_screenshots(messages: list[dict]) -> list[dict]:
    """
    Walk through messages and remove old base64 image blocks, keeping only the
    most recent MAX_SCREENSHOTS_IN_CONTEXT screenshots. Also trims oversized
    tool-result text blocks in older turns to a capped length.

    Screenshots appear in two places:
    - Directly as {"type": "image", ...} in a message's content list  (fast-path)
    - Nested inside {"type": "tool_result", "content": [...]} blocks  (normal path)

    Both locations are scanned so the limit is correctly enforced.
    Each position is stored as (msg_idx, outer_content_idx, inner_idx_or_None).
    """
    # Collect all image block positions
    image_positions: list[tuple[int, int, int | None]] = []
    for mi, msg in enumerate(messages):
        content = msg.get("content", [])
        if not isinstance(content, list):
            continue
        for ci, block in enumerate(content):
            if not isinstance(block, dict):
                continue
            if block.get("type") == "image":
                image_positions.append((mi, ci, None))
            elif block.get("type") == "tool_result":
                inner = block.get("content", [])
                if isinstance(inner, list):
                    for ni, nblock in enumerate(inner):
                        if isinstance(nblock, dict) and nblock.get("type") == "image":
                            image_positions.append((mi, ci, ni))

    to_remove = image_positions[:-MAX_SCREENSHOTS_IN_CONTEXT] if MAX_SCREENSHOTS_IN_CONTEXT else image_positions
    for mi, ci, ni in to_remove:
        placeholder = {"type": "text", "text": "[screenshot removed to save context]"}
        if ni is None:
            messages[mi]["content"][ci] = placeholder
        else:
            messages[mi]["content"][ci]["content"][ni] = placeholder

    # Trim oversized text in old tool-result blocks (e.g. extract_text dumps).
    # Only apply to messages that are NOT the latest tool-result turn.
    cutoff = len(messages) - 2  # keep the last assistant+user pair intact
    for mi, msg in enumerate(messages):
        if mi >= cutoff:
            break
        content = msg.get("content", [])
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict) or block.get("type") != "tool_result":
                continue
            inner = block.get("content", [])
            if isinstance(inner, list):
                for ib in inner:
                    if isinstance(ib, dict) and ib.get("type") == "text":
                        text = ib.get("text", "")
                        if len(text) > _MAX_TOOL_TEXT_CHARS:
                            ib["text"] = text[:_MAX_TOOL_TEXT_CHARS] + "…[trimmed]"
            elif isinstance(inner, str) and len(inner) > _MAX_TOOL_TEXT_CHARS:
                block["content"] = inner[:_MAX_TOOL_TEXT_CHARS] + "…[trimmed]"

    return messages


def _trim_history(messages: list[dict]) -> list[dict]:
    """
    Drop the oldest message pairs when history exceeds MAX_HISTORY_MESSAGES.
    Always cuts at a user-role boundary so tool_use/tool_result pairs stay intact.
    """
    if len(messages) <= MAX_HISTORY_MESSAGES:
        return messages

    # Find a safe cut point: a user message that is NOT a pure tool_result block
    # (i.e. a real user turn we can use as the new start of history)
    excess = len(messages) - MAX_HISTORY_MESSAGES
    cut_at = 0
    for i in range(excess, len(messages)):
        msg = messages[i]
        if msg.get("role") != "user":
            continue
        content = msg.get("content", "")
        # Skip tool_result-only messages — they need the preceding assistant turn
        if isinstance(content, list) and content and all(
            isinstance(b, dict) and b.get("type") == "tool_result" for b in content
        ):
            continue
        cut_at = i
        break

    if cut_at:
        logger.debug("Trimming history: dropping %d messages (total was %d)", cut_at, len(messages))
        return messages[cut_at:]
    return messages


def _extract_compressed_context(content: list[dict]) -> tuple[str, list[dict]]:
    """
    Scan LLM response content blocks for a <compressed_context>…</compressed_context>
    block.  Returns (compressed_text, cleaned_content) where the block has been
    stripped from cleaned_content so it never appears in the conversation history.
    Returns ("", content) unchanged if no block is found.
    """
    compressed = ""
    cleaned: list[dict] = []
    for block in content:
        if block.get("type") != "text":
            cleaned.append(block)
            continue
        text = block.get("text", "")
        start = text.find(_COMPRESS_CTX_MARKER)
        end   = text.find(_COMPRESS_CTX_END)
        if start != -1 and end != -1:
            compressed = text[start + len(_COMPRESS_CTX_MARKER):end].strip()
            remainder  = (text[:start] + text[end + len(_COMPRESS_CTX_END):]).strip()
            if remainder:
                cleaned.append({"type": "text", "text": remainder})
        else:
            cleaned.append(block)
    return compressed, cleaned


def _messages_to_plain_text(messages: list[dict]) -> str:
    """
    Convert a message list to a compact plain-text representation suitable for
    the summarisation LLM call.  Images and large base64 blobs are omitted.
    """
    lines: list[str] = []
    for msg in messages:
        role    = msg.get("role", "?")
        content = msg.get("content", "")
        if isinstance(content, str):
            lines.append(f"{role}: {content[:400]}")
            continue
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            btype = block.get("type")
            if btype == "text":
                lines.append(f"{role}: {block.get('text','')[:400]}")
            elif btype == "tool_use":
                lines.append(
                    f"  [{role} tool_use] {block.get('name','')} "
                    f"{str(block.get('input',''))[:120]}"
                )
            elif btype == "tool_result":
                inner = block.get("content", "")
                if isinstance(inner, str):
                    lines.append(f"  [tool_result] {inner[:300]}")
                elif isinstance(inner, list):
                    for ib in inner:
                        if isinstance(ib, dict) and ib.get("type") == "text":
                            lines.append(f"  [tool_result] {ib.get('text','')[:300]}")
            # skip image blocks entirely
    return "\n".join(lines)


def _default_human_input(question: str, **kwargs) -> str:
    """Default pause handler — prompts via the console."""
    print(f"\n\033[1;33m[Agent asks]\033[0m {question}")
    try:
        return input("\033[1mYour answer:\033[0m ").strip()
    except (EOFError, KeyboardInterrupt):
        return ""


def _print_tool_call(name: str, tool_input: dict) -> None:
    icons = {
        "navigate": "🌐",
        "screenshot": "📸",
        "click": "🖱️",
        "type_text": "⌨️",
        "key_press": "⌨️",
        "scroll": "↕️",
        "hover": "🔍",
        "wait": "⏳",
        "go_back": "⬅️",
        "get_user_profile": "👤",
        "update_user_profile": "💾",
        "task_complete": "✅",
        "ask_user": "❓",
    }
    icon = icons.get(name, "🔧")
    short_input = str(tool_input)[:80]
    print(f"  {icon} {name}({short_input}{'...' if len(str(tool_input)) > 80 else ''})")


_STEP_LABELS: dict[str, str] = {
    "navigate":          "Navigating to",
    "click":             "Clicking at",
    "type_text":         "Typing",
    "key_press":         "Pressing key",
    "scroll":            "Scrolling",
    "hover":             "Hovering over",
    "go_back":           "Going back",
    "get_user_profile":  "Loading user profile",
    "update_user_profile": "Updating user profile",
    "extract_text":      "Extracting text from page",
    "task_complete":     "Task complete",
    "ask_user":          "Asking user",
}

# Tool names to skip for step notifications (low signal)
_SILENT_TOOLS = frozenset({"screenshot", "wait"})


def _step_summary(tool_name: str, tool_input: dict) -> str:
    """One-line human-readable description of a tool call."""
    label = _STEP_LABELS.get(tool_name, f"Running {tool_name}")
    if tool_name == "navigate":
        return f"{label} {tool_input.get('url', '')}"
    if tool_name == "type_text":
        t = tool_input.get("text", "")
        return f"{label} '{t[:40]}{'…' if len(t) > 40 else ''}'"
    if tool_name == "key_press":
        return f"{label} {tool_input.get('key', '')}"
    if tool_name in ("click", "hover"):
        x, y = tool_input.get("x", "?"), tool_input.get("y", "?")
        return f"{label} ({x}, {y})"
    if tool_name == "scroll":
        direction = tool_input.get("direction", "")
        return f"{label} {direction}"
    if tool_name == "task_complete":
        return label
    if tool_name == "ask_user":
        q = tool_input.get("question", "")
        return f"{label}: {q[:60]}{'…' if len(q) > 60 else ''}"
    return label


class BrowserAgent:
    """Stateful agent that maintains conversation history across tasks."""

    def __init__(
        self,
        browser: BrowserController,
        provider=None,
        human_input_fn: Callable[[str], str] | None = None,
    ) -> None:
        self.browser         = browser
        self.provider        = provider or AnthropicProvider()
        self.human_input_fn  = human_input_fn or _default_human_input
        self.messages: list[dict] = []
        self.knowledge = SiteKnowledge()
        self._replayed_urls: set[str] = set()
        # Optional callbacks set by the orchestrator client for user notifications
        self.plan_callback: Callable[[str], None] | None = None
        self.step_callback: Callable[[str, str], None] | None = None
        # Token-reduction state (A) compressed system prompt, (B) rolling summary
        self._compressed_system: str | None = None   # set after first LLM call
        self._history_summary: str = ""              # rolling plain-text summary
        self._turns_since_summary: int = 0           # tool-turn pairs since last summary
        # Observation mode toggle
        self._observe_mode: str = OBSERVE_MODE_TREE  # "tree" or "screenshot"

    def clear_history(self) -> None:
        self.messages = []
        self._compressed_system = None
        self._history_summary = ""
        self._turns_since_summary = 0
        # Note: _observe_mode is intentionally NOT reset — it's a persistent setting.

    def set_observe_mode(self, mode: str) -> None:
        """Switch between 'tree' (accessibility snapshot) and 'screenshot' modes."""
        if mode not in (OBSERVE_MODE_TREE, OBSERVE_MODE_SCREENSHOT):
            logger.warning("Unknown observe mode %r — ignoring (valid: tree, screenshot)", mode)
            return
        if mode != self._observe_mode:
            self._observe_mode = mode
            logger.info("Observe mode set to %r", mode)

    # ── Token-reduction helpers ───────────────────────────────────────────────

    def _effective_system(self, full_system: str, *, first_call: bool) -> str:
        """
        Return the system string to pass to the LLM for this call.

        First call of the session → full system + compression instruction.
        Subsequent calls → compressed context (if available) or full system,
        with the rolling history summary appended when non-empty.
        """
        if first_call:
            return full_system + _COMPRESS_CTX_INSTRUCTION

        base = self._compressed_system if self._compressed_system else full_system
        if self._history_summary:
            return (
                base
                + "\n\n## Conversation history summary (older turns)\n"
                + self._history_summary
            )
        return base

    def _summarize_old_history(self) -> None:
        """
        (B) Compress old message turns into self._history_summary and drop them
        from self.messages, keeping only the most recent _KEEP_RECENT_PAIRS pairs.

        Runs synchronously via the existing provider so no extra HTTP client is needed.
        Silently skips if there is nothing old enough to summarise or on any error.
        """
        keep = _KEEP_RECENT_PAIRS * 2          # messages to preserve verbatim
        cutoff = len(self.messages) - keep
        if cutoff <= 0:
            return                              # nothing to summarise yet

        old_turns = self.messages[:cutoff]
        history_text = _messages_to_plain_text(old_turns)
        if self._history_summary:
            history_text = (
                f"[Prior summary]\n{self._history_summary}\n\n"
                f"[New turns]\n{history_text}"
            )

        try:
            content, _ = self.provider.complete(
                _SUMMARIZE_SYSTEM,
                [{"role": "user", "content": history_text}],
                [],   # no tools
            )
            summary = "\n".join(
                b.get("text", "") for b in content
                if isinstance(b, dict) and b.get("type") == "text"
            ).strip()
        except Exception as exc:
            logger.warning("History summarisation failed — keeping full history: %s", exc)
            return

        if summary:
            self._history_summary = summary
            self.messages = self.messages[cutoff:]
            self._turns_since_summary = 0
            logger.debug(
                "History summarised: %d messages → %d words",
                len(old_turns), len(summary.split()),
            )

    # ── Graceful wrap-up ──────────────────────────────────────────────────────

    def _do_graceful_wrap_up(self, user_task: str, system: str, reason: str) -> None:
        """
        Called when the time or context budget is nearly exhausted.

        1. Makes one final LLM call (no tools) asking for a progress summary and
           suggested next step.
        2. Injects a synthetic ask_user / [Awaiting user answer] pair into
           self.messages so the existing followup_answers resume path works.
        3. Raises BudgetExhausted with the summary text.

        This method always raises — it never returns.
        """
        logger.info("Budget limit reached (%s) — running graceful wrap-up", reason)

        # One-shot LLM call without tools to get a summary
        wrap_messages = self.messages + [
            {"role": "user", "content": _WRAP_UP_PROMPT}
        ]
        try:
            wrap_content, _ = self.provider.complete(system, wrap_messages, [])
            summary = "\n".join(
                b.get("text", "") for b in wrap_content
                if isinstance(b, dict) and b.get("type") == "text"
            ).strip()
        except Exception as exc:
            logger.warning("Wrap-up LLM call failed: %s", exc)
            wrap_content = []
            summary = ""

        if not summary:
            summary = (
                f"Task paused ({reason}). "
                f"I was working on: '{user_task}'. "
                "Please ask me to continue and I will resume from where I left off."
            )

        # Inject synthetic ask_user + placeholder so the resume path works:
        # The next run_task(followup_answers={"continue": "..."}) will fill in
        # the [Awaiting user answer] placeholder and the agent resumes.
        tool_use_id = f"wrap_{uuid.uuid4().hex[:8]}"
        assistant_content: list[dict] = list(wrap_content)
        assistant_content.append({
            "type":  "tool_use",
            "id":    tool_use_id,
            "name":  "ask_user",
            "input": {"question": "[Paused — awaiting user decision to continue]"},
        })
        self.messages.append({"role": "assistant", "content": assistant_content})
        self.messages.append({
            "role": "user",
            "content": [{
                "type":        "tool_result",
                "tool_use_id": tool_use_id,
                "content":     "[Awaiting user answer]",
            }],
        })
        self.knowledge.commit()

        raise BudgetExhausted(summary=summary, reason=reason)

    # ------------------------------------------------------------------ fast path

    def _try_fast_steps(self) -> bool:
        """
        Check whether the current page has high-confidence cached steps and, if
        so, execute them without calling the LLM.  Synthetic assistant/user turns
        are injected into self.messages so the conversation history stays coherent.

        Returns True  → cached steps executed; caller should skip the LLM call.
        Returns False → no cache hit or execution failed; caller should use LLM.
        """
        try:
            current_url = self.browser.get_url()
        except Exception:
            return False

        # Each (domain+path) is replayed at most once per task to avoid loops
        p = urlparse(current_url)
        url_key = f"{p.netloc}{p.path}"
        if url_key in self._replayed_urls:
            return False

        steps = self.knowledge.get_steps(current_url)
        if not steps:
            return False

        print(
            f"\n  ⚡ [Fast path] {len(steps)} cached step(s) for {url_key}"
            " — skipping screenshot"
        )

        synthetic_tool_uses: list[dict] = []
        synthetic_results: list[dict] = []
        failed_at: int | None = None

        for i, step in enumerate(steps):
            tool_name = step["tool"]
            tool_input = step["input"]
            tool_id = f"fast_{i}_{tool_name}"

            _print_tool_call(tool_name, tool_input)

            try:
                result_text, screenshot_b64 = execute_tool(
                    tool_name, tool_input, self.browser, self._observe_mode
                )
            except Exception as exc:
                logger.warning("Fast path step %d (%s) failed: %s", i, tool_name, exc)
                failed_at = i
                break

            synthetic_tool_uses.append(
                {"type": "tool_use", "id": tool_id, "name": tool_name, "input": tool_input}
            )
            result_content: list[dict] | str = result_text
            if screenshot_b64:
                result_content = [
                    {"type": "text", "text": result_text},
                    _image_block(screenshot_b64),
                ]
            synthetic_results.append(
                {"type": "tool_result", "tool_use_id": tool_id, "content": result_content}
            )

        if synthetic_tool_uses:
            # Inject whatever succeeded so the LLM has accurate context
            self.messages.append({
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": "[Fast path] Replaying cached steps — screenshot skipped.",
                    },
                    *synthetic_tool_uses,
                ],
            })
            self.messages.append({
                "role": "user",
                "content": synthetic_results,
            })
            self._replayed_urls.add(url_key)

        if failed_at is not None:
            self.knowledge.record_replay_miss(current_url)
            logger.info("Fast path failed at step %d — switching to LLM.", failed_at)
            return False  # LLM will screenshot and recover from current state

        return bool(synthetic_tool_uses)

    def run_task(
        self,
        user_task: str,
        followup_answers: dict | None = None,
        soft_deadline: float | None = None,
        max_context_tokens: int | None = None,
    ) -> str:
        """
        Run a single task. Appends to conversation history so context is preserved
        across multiple tasks in the same session.

        If followup_answers is provided the agent resumes from a previous pause
        (ask_user or BudgetExhausted).

        soft_deadline: monotonic timestamp. When reached (at _BUDGET_FRACTION of
            the available window) the agent wraps up and raises BudgetExhausted.
        max_context_tokens: estimated token budget for conversation history.
            When _BUDGET_FRACTION of it is consumed the agent wraps up.

        Raises BudgetExhausted before hard limits are hit so the caller can
        notify the user and set up a resumable followup_request.
        """
        print(f"\n\033[1mAgent starting task:\033[0m {user_task}")

        # Trim old turns, then remove dangling tool_use turns from interrupted tasks
        self.messages = _trim_history(self.messages)
        self.messages = _repair_message_history(self.messages)

        # Reset per-task fast-path state
        self._replayed_urls = set()
        self.knowledge.reset_recording()

        if followup_answers:
            # Inject the answer into the placeholder tool_result written when
            # ask_user was first called.  This avoids appending a new user message
            # (which would create two consecutive user turns — invalid for the API).
            answer_text = "\n".join(f"{k}: {v}" for k, v in followup_answers.items())
            injected = False
            for msg in reversed(self.messages):
                if msg.get("role") != "user":
                    continue
                content = msg.get("content", [])
                if not isinstance(content, list):
                    continue
                for block in content:
                    if (
                        isinstance(block, dict)
                        and block.get("type") == "tool_result"
                        and block.get("content") == "[Awaiting user answer]"
                    ):
                        block["content"] = f"User answered: {answer_text}"
                        injected = True
                        break
                if injected:
                    break
            if not injected:
                # History was lost (e.g. trimmed); fall back to a fresh user message
                logger.warning("followup_answers provided but no placeholder found — starting fresh")
                self.messages.append({"role": "user", "content": user_task})
            # else: placeholder was injected; no new user message needed
        else:
            # Append the user's task as a new message
            self.messages.append({
                "role": "user",
                "content": user_task,
            })

        final_summary = ""
        task_done = False
        _plan_sent   = False   # fire plan_callback only on the first LLM response
        _first_call  = True    # skip budget check before the first LLM call
        _session_first_call = self._compressed_system is None  # first call of the whole session
        _token_budget = max_context_tokens if max_context_tokens is not None else DEFAULT_MAX_CONTEXT_TOKENS
        _accumulated_tokens = 0
        system = _build_system_prompt()  # build once per task, not every loop iteration

        while not task_done:
            # Trim old screenshots before each API call
            self.messages = _trim_old_screenshots(self.messages)

            # ── Budget check (skip on the very first call) ───────────────────
            if not _first_call:
                _time_exhausted = (
                    soft_deadline is not None
                    and time.monotonic() >= soft_deadline
                )
                _tokens_exhausted = (
                    _accumulated_tokens >= int(_token_budget * _BUDGET_FRACTION)
                )
                if _time_exhausted or _tokens_exhausted:
                    reason = "time limit" if _time_exhausted else "context limit"
                    self._do_graceful_wrap_up(user_task, system, reason)
                    # _do_graceful_wrap_up always raises BudgetExhausted — unreachable:
                    break
            _first_call = False

            # ── Fast path: replay cached steps without calling the LLM ──────
            if self._try_fast_steps():
                continue  # skip LLM call; loop again from new page state

            # ── (A) Build effective system prompt ─────────────────────────────
            effective_system = self._effective_system(system, first_call=_session_first_call)

            # Call the active provider (streams text to console internally)
            try:
                collected_content, stop_reason = self.provider.complete(
                    effective_system, self.messages, get_tool_definitions(self._observe_mode)
                )
            except LlmRetryExhausted as exc:
                # Attach live browser context so orchestrator can build replan payload
                try:
                    exc.current_url     = self.browser.get_url()
                    exc.page_title      = self.browser.get_title()
                except Exception:
                    pass
                exc.completed_steps = self._extract_completed_steps()
                raise
            print()  # newline after streamed text

            # ── (A) Extract and store the compressed context block ────────────
            if _session_first_call:
                _session_first_call = False
                compressed, collected_content = _extract_compressed_context(collected_content)
                if compressed:
                    self._compressed_system = compressed
                    logger.debug(
                        "System prompt compressed: %d → %d chars",
                        len(system), len(compressed),
                    )
                else:
                    logger.debug("No compressed_context block in first response — keeping full system prompt")

            if collected_content:
                self.messages.append({"role": "assistant", "content": collected_content})

            # Accumulate token usage for budget tracking
            _accumulated_tokens += (
                getattr(self.provider, "last_input_tokens",  0) +
                getattr(self.provider, "last_output_tokens", 0)
            )

            # Fire plan_callback on the first LLM response (it's the agent's plan)
            if not _plan_sent and self.plan_callback:
                text_parts = [
                    b.get("text", "") for b in collected_content
                    if isinstance(b, dict) and b.get("type") == "text"
                ]
                plan_text = "\n".join(p for p in text_parts if p).strip()
                if plan_text:
                    try:
                        self.plan_callback(plan_text)
                    except Exception:
                        pass
                _plan_sent = True

            current_tool_uses = [
                b for b in collected_content
                if isinstance(b, dict) and b.get("type") == "tool_use"
            ]

            if not current_tool_uses:
                break

            # Execute tool calls
            tool_results: list[dict] = []

            for idx, tool_use in enumerate(current_tool_uses):
                tool_name = tool_use["name"]
                tool_input = tool_use["input"]
                tool_use_id = tool_use["id"]

                _print_tool_call(tool_name, tool_input)

                # Capture URL before execution so we can record this step
                try:
                    url_before = self.browser.get_url()
                except Exception:
                    url_before = ""

                if tool_name == "ask_user":
                    # Pause execution and collect the user's response.
                    # Extra fields (agent_capability, agent_task, etc.) are forwarded
                    # to the orchestrator so the planner can try to resolve the answer
                    # via another agent before falling back to asking the human.
                    question = tool_input.get("question", "")
                    try:
                        result_text = self.human_input_fn(
                            question,
                            field_name=tool_input.get("field_name"),
                            agent_capability=tool_input.get("agent_capability"),
                            agent_task=tool_input.get("agent_task"),
                            intent=tool_input.get("intent"),
                        )
                        screenshot_b64 = None
                    except Exception:
                        # Save a placeholder tool_result so the conversation history
                        # stays valid (no dangling tool_use turn).  _repair_message_history
                        # would otherwise drop the incomplete assistant turn and the agent
                        # would restart from scratch on re-dispatch.
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": tool_use_id,
                            "content": "[Awaiting user answer]",
                        })
                        # If the LLM batched other tools after ask_user in the same
                        # response, add stubs for them so every tool_use_id has a
                        # matching tool_result.  All providers (Anthropic, OpenAI,
                        # Gemini) reject messages where a tool_use has no response,
                        # and _repair_message_history would silently drop the turn.
                        for remaining in current_tool_uses[idx + 1:]:
                            tool_results.append({
                                "type": "tool_result",
                                "tool_use_id": remaining["id"],
                                "content": "[Not executed — ask_user interrupted batch]",
                            })
                        self.messages.append({"role": "user", "content": tool_results})
                        self.knowledge.commit()
                        raise
                else:
                    try:
                        result_text, screenshot_b64 = execute_tool(
                            tool_name, tool_input, self.browser, self._observe_mode
                        )
                        # Record successful LLM-driven action for future fast-path replay
                        if url_before:
                            self.knowledge.observe(url_before, tool_name, tool_input)
                    except Exception as exc:
                        result_text = f"Error: {exc}"
                        screenshot_b64 = None

                    # Fire step_callback for meaningful actions (skip silent tools)
                    if tool_name not in _SILENT_TOOLS and self.step_callback:
                        try:
                            self.step_callback(tool_name, _step_summary(tool_name, tool_input))
                        except Exception:
                            pass

                if tool_name == "task_complete":
                    final_summary = result_text
                    task_done = True

                tool_result_content: list[dict] | str
                if screenshot_b64:
                    # Embed screenshot image directly in the tool result
                    tool_result_content = [
                        {"type": "text", "text": result_text},
                        _image_block(screenshot_b64),
                    ]
                else:
                    tool_result_content = result_text

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_use_id,
                    "content": tool_result_content,
                })

            # Append tool results as a user message
            self.messages.append({
                "role": "user",
                "content": tool_results,
            })

            # ── (B) Rolling history summarisation ────────────────────────────
            self._turns_since_summary += 1
            if (
                self._turns_since_summary >= _SUMMARIZE_EVERY_N_TURNS
                and len(self.messages) > _KEEP_RECENT_PAIRS * 2 + 2
            ):
                self._summarize_old_history()

        # Persist what the LLM did this task so future runs can use the fast path
        self.knowledge.commit()

        print()  # final newline
        return final_summary or "Task completed."

    def _extract_completed_steps(self) -> list[str]:
        """
        Walk self.messages and return a compact list of tool calls that succeeded.
        Used to populate LlmRetryExhausted.completed_steps for replanning context.
        """
        steps: list[str] = []
        for msg in self.messages:
            content = msg.get("content", [])
            if not isinstance(content, list):
                continue
            # tool_result blocks from user messages describe what each tool returned
            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "tool_result":
                    tool_content = block.get("content", "")
                    if isinstance(tool_content, str) and tool_content.strip():
                        steps.append(tool_content[:200])
                elif block.get("type") == "tool_use":
                    name  = block.get("name", "")
                    inp   = block.get("input", {})
                    brief = f"{name}({', '.join(f'{k}={str(v)[:60]}' for k, v in inp.items())})"
                    steps.append(brief)
        return steps[-20:]  # last 20 actions is enough for replan context
