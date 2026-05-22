"""
Multi-provider LLM abstraction for the browser agent.

All providers accept and return messages in Anthropic's canonical format so
the rest of the codebase never needs to know which backend is in use.

Supported providers:
  anthropic  — Claude models via Anthropic SDK  (ANTHROPIC_API_KEY)
  openai     — GPT models via OpenAI SDK        (OPENAI_API_KEY)
  gemini     — Gemini models via google-genai   (GOOGLE_API_KEY)
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

MAX_TOKENS = 4096

PROVIDER_DEFAULTS: dict[str, str] = {
    "anthropic": "claude-sonnet-4-6",
    "openai":    "gpt-4o",
    "gemini":    "gemini-2.5-flash",
}


# ── Tool definition translation ──────────────────────────────────────────────

def _tools_to_openai(tools: list[dict]) -> list[dict]:
    """Anthropic tool defs → OpenAI function-call format."""
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t.get("description", ""),
                "parameters": t.get("input_schema", {"type": "object", "properties": {}}),
            },
        }
        for t in tools
    ]




# ── Message translation ───────────────────────────────────────────────────────

def _messages_to_openai(messages: list[dict]) -> list[dict]:
    """Translate Anthropic-format messages → OpenAI chat messages."""
    result: list[dict] = []

    for msg in messages:
        role    = msg["role"]
        content = msg.get("content", "")

        # Plain string content
        if isinstance(content, str):
            result.append({"role": role, "content": content})
            continue

        # user turn
        if role == "user":
            # Pure tool_result messages → one "tool" role message per result
            if content and all(
                isinstance(b, dict) and b.get("type") == "tool_result"
                for b in content
            ):
                for block in content:
                    tc = block.get("content", "")
                    if isinstance(tc, list):
                        tc = " ".join(
                            b.get("text", "") for b in tc
                            if isinstance(b, dict) and b.get("type") == "text"
                        )
                    result.append({
                        "role": "tool",
                        "tool_call_id": block["tool_use_id"],
                        "content": str(tc),
                    })
                continue

            # Regular user message (text + optional images)
            parts: list[dict] = []
            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "text":
                    parts.append({"type": "text", "text": block["text"]})
                elif block.get("type") == "image":
                    src = block.get("source", {})
                    if src.get("type") == "base64":
                        mt   = src.get("media_type", "image/png")
                        data = src.get("data", "")
                        parts.append({
                            "type": "image_url",
                            "image_url": {"url": f"data:{mt};base64,{data}"},
                        })
            result.append({"role": "user", "content": parts or ""})

        # assistant turn
        elif role == "assistant":
            text       = ""
            tool_calls: list[dict] = []
            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "text":
                    text += block.get("text", "")
                elif block.get("type") == "tool_use":
                    tool_calls.append({
                        "id":   block["id"],
                        "type": "function",
                        "function": {
                            "name":      block["name"],
                            "arguments": json.dumps(block.get("input", {})),
                        },
                    })
                # "thinking" blocks are silently skipped for non-Anthropic

            out: dict[str, Any] = {"role": "assistant", "content": text or None}
            if tool_calls:
                out["tool_calls"] = tool_calls
            result.append(out)

    return result


def _messages_to_gemini(messages: list[dict]) -> list[dict]:
    """
    Translate Anthropic-format messages → Gemini contents list.
    Builds an id→name map so tool_result blocks can emit correct function_response names.
    """
    # Pre-pass: map tool_use id → function name
    id_to_name: dict[str, str] = {}
    for msg in messages:
        if msg["role"] == "assistant":
            for block in (msg.get("content") or []):
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    id_to_name[block["id"]] = block["name"]

    contents: list[dict] = []
    for msg in messages:
        role        = msg["role"]
        gemini_role = "model" if role == "assistant" else "user"
        content     = msg.get("content", "")
        parts: list[dict] = []

        if isinstance(content, str):
            parts.append({"text": content})
        else:
            for block in content:
                if not isinstance(block, dict):
                    continue
                btype = block.get("type")
                if btype == "text":
                    parts.append({"text": block.get("text", "")})
                elif btype == "image":
                    src = block.get("source", {})
                    if src.get("type") == "base64":
                        parts.append({
                            "inline_data": {
                                "mime_type": src.get("media_type", "image/png"),
                                "data":      src.get("data", ""),
                            }
                        })
                elif btype == "tool_use":
                    parts.append({
                        "function_call": {
                            "name": block["name"],
                            "args": block.get("input", {}),
                        }
                    })
                elif btype == "tool_result":
                    tc = block.get("content", "")
                    if isinstance(tc, list):
                        tc = " ".join(
                            b.get("text", "") for b in tc
                            if isinstance(b, dict) and b.get("type") == "text"
                        )
                    name = id_to_name.get(block.get("tool_use_id", ""), "unknown")
                    parts.append({
                        "function_response": {
                            "name":     name,
                            "response": {"result": str(tc)},
                        }
                    })
                # thinking blocks skipped

        if parts:
            contents.append({"role": gemini_role, "parts": parts})

    return contents


# ── Providers ─────────────────────────────────────────────────────────────────

class AnthropicProvider:
    """Uses Anthropic streaming with adaptive thinking."""

    def __init__(self, model: str = PROVIDER_DEFAULTS["anthropic"]) -> None:
        import anthropic
        self.client = anthropic.Anthropic()
        self.model  = model
        self.last_input_tokens:  int = 0
        self.last_output_tokens: int = 0

    def complete(
        self,
        system:   str,
        messages: list[dict],
        tools:    list[dict],
    ) -> tuple[list[dict], str]:
        collected: list[dict]  = []
        current:   dict | None = None

        with self.client.messages.stream(
            model=self.model,
            max_tokens=MAX_TOKENS,
            system=system,
            tools=tools,
            messages=messages,
        ) as stream:
            for event in stream:
                etype = event.type

                if etype == "content_block_start":
                    b = event.content_block
                    current = {"type": b.type}
                    if b.type == "text":
                        current["text"] = ""
                    elif b.type == "tool_use":
                        current["id"]    = b.id
                        current["name"]  = b.name
                        current["input"] = ""

                elif etype == "content_block_delta":
                    delta = event.delta
                    if current is None:
                        continue
                    if delta.type == "text_delta":
                        current["text"] = current.get("text", "") + delta.text
                        print(delta.text, end="", flush=True)
                    elif delta.type == "input_json_delta":
                        current["input"] = current.get("input", "") + delta.partial_json

                elif etype == "content_block_stop":
                    if current:
                        if current["type"] == "tool_use":
                            raw = current.get("input", "{}")
                            try:
                                current["input"] = json.loads(raw) if isinstance(raw, str) else raw
                            except Exception:
                                current["input"] = {}
                        collected.append(current)
                        current = None

        final       = stream.get_final_message()
        stop_reason = final.stop_reason
        if hasattr(final, "usage") and final.usage:
            self.last_input_tokens  = getattr(final.usage, "input_tokens",  0) or 0
            self.last_output_tokens = getattr(final.usage, "output_tokens", 0) or 0
        return collected, stop_reason


class OpenAIProvider:
    """Uses OpenAI streaming; translates messages/tools to OpenAI format."""

    def __init__(self, model: str = PROVIDER_DEFAULTS["openai"]) -> None:
        import openai
        self.client = openai.OpenAI()
        self.model  = model
        self.last_input_tokens:  int = 0
        self.last_output_tokens: int = 0

    def complete(
        self,
        system:   str,
        messages: list[dict],
        tools:    list[dict],
    ) -> tuple[list[dict], str]:
        oai_messages = [{"role": "system", "content": system}] + _messages_to_openai(messages)
        oai_tools    = _tools_to_openai(tools)

        text_buf:       str                    = ""
        tool_calls_buf: dict[int, dict]        = {}  # stream index → accumulated fields

        stream = self.client.chat.completions.create(
            model=self.model,
            messages=oai_messages,
            tools=oai_tools,
            max_tokens=MAX_TOKENS,
            stream=True,
        )

        for chunk in stream:
            choice = chunk.choices[0] if chunk.choices else None
            if choice is None:
                continue
            delta = choice.delta
            if delta.content:
                text_buf += delta.content
                print(delta.content, end="", flush=True)
            if delta.tool_calls:
                for tc in delta.tool_calls:
                    idx = tc.index
                    if idx not in tool_calls_buf:
                        tool_calls_buf[idx] = {"id": "", "name": "", "arguments": ""}
                    if tc.id:
                        tool_calls_buf[idx]["id"] += tc.id
                    if tc.function:
                        if tc.function.name:
                            tool_calls_buf[idx]["name"] += tc.function.name
                        if tc.function.arguments:
                            tool_calls_buf[idx]["arguments"] += tc.function.arguments

        # Build Anthropic-format content blocks
        collected: list[dict] = []
        if text_buf:
            collected.append({"type": "text", "text": text_buf})
        for _, tc in sorted(tool_calls_buf.items()):
            try:
                input_dict = json.loads(tc["arguments"]) if tc["arguments"] else {}
            except Exception:
                input_dict = {}
            collected.append({
                "type":  "tool_use",
                "id":    tc["id"],
                "name":  tc["name"],
                "input": input_dict,
            })

        # Estimate tokens from buffer sizes (OpenAI streaming doesn't expose usage easily)
        self.last_input_tokens  = sum(len(str(m)) for m in oai_messages) // 4
        self.last_output_tokens = (len(text_buf) + sum(len(str(tc)) for tc in tool_calls_buf.values())) // 4
        stop_reason = "tool_use" if tool_calls_buf else "end_turn"
        return collected, stop_reason


class GeminiProvider:
    """
    Uses google-genai SDK (pip install google-genai).
    Import: from google import genai
    No IPython dependency — safe alongside the project's profile.py.
    """

    def __init__(self, model: str = PROVIDER_DEFAULTS["gemini"]) -> None:
        from google import genai
        self.client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY", ""))
        self.model  = model
        self.last_input_tokens:  int = 0
        self.last_output_tokens: int = 0

    def _build_tools(self, tools: list[dict]) -> list[dict]:
        """Anthropic tool defs → Gemini function_declarations dict format."""
        declarations = [
            {
                "name":        t["name"],
                "description": t.get("description", ""),
                "parameters":  t.get("input_schema", {"type": "object", "properties": {}}),
            }
            for t in tools
        ]
        return [{"function_declarations": declarations}]

    def complete(
        self,
        system:   str,
        messages: list[dict],
        tools:    list[dict],
    ) -> tuple[list[dict], str]:
        from google.genai import types

        contents     = _messages_to_gemini(messages)
        config       = types.GenerateContentConfig(
            system_instruction=system,
            tools=self._build_tools(tools),
            max_output_tokens=MAX_TOKENS,
        )

        text_buf:   str      = ""
        tool_calls: list     = []
        fc_seen:    set[str] = set()

        for chunk in self.client.models.generate_content_stream(
            model=self.model,
            contents=contents,
            config=config,
        ):
            candidate = chunk.candidates[0] if chunk.candidates else None
            if not candidate or not candidate.content or not candidate.content.parts:
                continue
            for part in candidate.content.parts:
                if part.text:
                    text_buf += part.text
                    print(part.text, end="", flush=True)
                elif hasattr(part, "function_call") and part.function_call:
                    fc  = part.function_call
                    key = fc.name + json.dumps(dict(fc.args or {}), sort_keys=True)
                    if key not in fc_seen:
                        fc_seen.add(key)
                        tool_calls.append({
                            "type":  "tool_use",
                            "id":    f"gemini_{len(tool_calls)}_{fc.name}",
                            "name":  fc.name,
                            "input": dict(fc.args or {}),
                        })

        collected: list[dict] = []
        if text_buf:
            collected.append({"type": "text", "text": text_buf})
        collected.extend(tool_calls)

        self.last_input_tokens  = sum(len(str(c)) for c in contents) // 4
        self.last_output_tokens = (len(text_buf) + sum(len(str(tc)) for tc in tool_calls)) // 4
        stop_reason = "tool_use" if tool_calls else "end_turn"
        return collected, stop_reason


# ── Proxy Provider ────────────────────────────────────────────────────────────

import httpx as _httpx
import time as _time

_proxy_logger = logging.getLogger(__name__)


class LlmRetryExhausted(RuntimeError):
    """
    Raised by ProxyProvider after all retry attempts are exhausted.
    Carries metadata the browser agent uses to assemble a replan context.
    """
    def __init__(self, message: str, attempts: int, last_error: str) -> None:
        super().__init__(message)
        self.attempts   = attempts
        self.last_error = last_error
        # Populated by BrowserAgent before re-raising so orchestrator_client can read it
        self.current_url:      str       = ""
        self.page_title:       str       = ""
        self.completed_steps:  list[str] = []


def _provider_from_model(model: str) -> str:
    """Infer provider name from the model string prefix."""
    m = model.lower()
    if m.startswith("claude"):
        return "anthropic"
    if m.startswith("gpt") or m.startswith("o1") or m.startswith("o3") or m.startswith("o4"):
        return "openai"
    if m.startswith("gemini"):
        return "google"
    if m.startswith("grok"):
        return "xai"
    if m.startswith("deepseek"):
        return "deepseek"
    if m.startswith("mistral") or m.startswith("mixtral") or m.startswith("codestral"):
        return "mistral"
    if m.startswith("command") or m.startswith("c4ai"):
        return "cohere"
    return "anthropic"  # safe default


class ProxyProvider:
    """Sync provider that routes all calls through the orchestrator LLM proxy."""

    def __init__(
        self,
        proxy_url:      str,
        agent_id:       str,
        model:          str,
        retry_count:    int   = 3,
        retry_delay_s:  float = 5.0,
    ) -> None:
        self._proxy_url      = proxy_url
        self._agent_id       = agent_id
        self.model           = model
        self._provider       = _provider_from_model(model)
        self._retry_count    = max(0, retry_count)
        self._retry_delay_s  = max(0.0, retry_delay_s)
        self._client         = _httpx.Client(timeout=180)
        self.last_input_tokens:  int = 0
        self.last_output_tokens: int = 0

    def update_retry_settings(self, retry_count: int, retry_delay_s: float) -> None:
        self._retry_count   = max(0, retry_count)
        self._retry_delay_s = max(0.0, retry_delay_s)
        _proxy_logger.info(
            "ProxyProvider retry settings updated: count=%d  delay=%.1fs",
            self._retry_count, self._retry_delay_s,
        )

    def __del__(self) -> None:
        try:
            self._client.close()
        except Exception:
            pass

    def create_message(
        self,
        system: str,
        messages: list[dict],
        max_tokens: int = 8096,
        thinking=None,
        tools: list[dict] | None = None,
        tool_choice: dict | None = None,
    ) -> dict:
        # Use prompt_segments for Anthropic to enable caching of static system/tools.
        # Non-Anthropic providers fall back to the flat payload format.
        if self._provider == "anthropic" and not thinking:
            segments: list[dict] = []
            if system:
                segments.append({
                    "name": "system_prompt",
                    "type": "system",
                    "content": system,
                    "cacheable": True,
                })
            if tools:
                segments.append({
                    "name": "tools",
                    "type": "tools",
                    "content": tools,
                    "cacheable": True,
                })
            segments.append({
                "name": "history",
                "type": "messages",
                "content": messages,
                "cacheable": False,
            })
            payload: dict = {
                "provider": self._provider,
                "model": self.model,
                "max_tokens": max_tokens,
                "prompt_segments": segments,
            }
            if tool_choice and tools:
                payload["tool_choice"] = tool_choice or {"type": "auto"}
        else:
            payload = {
                "provider": self._provider,
                "model": self.model,
                "messages": messages,
                "system": system,
                "max_tokens": max_tokens,
            }
            if thinking:
                payload["thinking"] = thinking
            if tools:
                payload["tools"] = tools
                payload["tool_choice"] = tool_choice or {"type": "auto"}
        max_attempts = self._retry_count + 1   # 1 original + N retries
        last_error   = ""
        for attempt in range(max_attempts):
            try:
                r = self._client.post(
                    self._proxy_url,
                    headers={"X-Agent-Id": self._agent_id},
                    json=payload,
                )
                if r.status_code == 404:
                    # Route missing — retrying won't help; fail fast
                    raise RuntimeError(
                        f"LLM proxy endpoint not found ({self._proxy_url}). "
                        "The orchestrator may need to be restarted."
                    )
                if 400 <= r.status_code < 500:
                    # Client errors (auth, rate-limit, etc.) — no point retrying
                    r.raise_for_status()
                r.raise_for_status()
                return r.json()

            except (_httpx.ConnectError, _httpx.TimeoutException) as exc:
                last_error = f"Connection/timeout error: {exc}"
            except _httpx.HTTPStatusError as exc:
                if 400 <= exc.response.status_code < 500:
                    raise  # client errors are not retried
                last_error = f"HTTP {exc.response.status_code}: {exc}"
            except RuntimeError:
                raise  # propagate RuntimeErrors (like 404 above) immediately

            if attempt < max_attempts - 1:
                _proxy_logger.warning(
                    "LLM proxy call failed (attempt %d/%d) — retrying in %.0fs: %s",
                    attempt + 1, max_attempts, self._retry_delay_s, last_error,
                )
                _time.sleep(self._retry_delay_s)

        raise LlmRetryExhausted(
            f"LLM call failed after {max_attempts} attempt(s)",
            attempts=max_attempts,
            last_error=last_error,
        )

    def complete(
        self,
        system: str,
        messages: list[dict],
        tools: list[dict],
        max_tokens: int = MAX_TOKENS,
    ) -> tuple[list[dict], str]:
        """Adapter so ProxyProvider works as a drop-in for AnthropicProvider."""
        result = self.create_message(
            system=system,
            messages=messages,
            max_tokens=max_tokens,
            tools=tools or [],
            tool_choice={"type": "auto"} if tools else None,
        )
        content = result.get("content", [])
        stop_reason = result.get("stop_reason", "end_turn")
        usage = result.get("usage", {})
        self.last_input_tokens  = usage.get("input_tokens",  0) or 0
        self.last_output_tokens = usage.get("output_tokens", 0) or 0
        return content, stop_reason


# ── Factory ───────────────────────────────────────────────────────────────────

def get_provider(name: str, model: str | None = None, proxy_url: str = "", agent_id: str = "") -> AnthropicProvider | OpenAIProvider | GeminiProvider | ProxyProvider:
    """
    Instantiate a provider by name.

    Args:
        name:      "anthropic" | "openai" | "gemini" | "proxy"
        model:     override the default model for that provider
        proxy_url: required when name="proxy"
        agent_id:  required when name="proxy"

    Required environment variables:
        anthropic → ANTHROPIC_API_KEY
        openai    → OPENAI_API_KEY
        gemini    → GOOGLE_API_KEY
    """
    if name == "proxy":
        if not proxy_url or not agent_id:
            raise ValueError("proxy_url and agent_id are required for ProxyProvider")
        return ProxyProvider(proxy_url=proxy_url, agent_id=agent_id, model=model or PROVIDER_DEFAULTS.get("anthropic", "claude-sonnet-4-6"))
    model = model or PROVIDER_DEFAULTS.get(name, "")
    if name == "anthropic":
        return AnthropicProvider(model)
    if name == "openai":
        return OpenAIProvider(model)
    if name == "gemini":
        return GeminiProvider(model)
    raise ValueError(
        f"Unknown provider {name!r}. Choose from: {list(PROVIDER_DEFAULTS) + ['proxy']}"
    )
