#!/usr/bin/env python3
"""
Browser Agent entry point.

Default mode  — registers with the agent orchestrator and waits for tasks.
Interactive   — pass -i / --interactive to open the REPL instead.
"""

import argparse
import asyncio
import json
import logging
import os
import sys

from browser import BrowserController
from agent import BrowserAgent
from profile import load_profile
from providers import get_provider, PROVIDER_DEFAULTS

# ------------------------------------------------------------------ logging

logging.basicConfig(
    level=logging.getLevelName(os.environ.get("LOG_LEVEL", "INFO").upper()),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logging.getLogger("playwright").setLevel(logging.ERROR)
logging.getLogger("websockets").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

# ------------------------------------------------------------------ banner

BANNER = """\033[1m
╔══════════════════════════════════════════╗
║          🤖  Browser Agent  🤖           ║
║  Chrome MCP (default) · Playwright       ║
╚══════════════════════════════════════════╝\033[0m

Commands:
  /profile  — View your learned preferences
  /clear    — Clear conversation history
  /quit     — Exit (closes browser)

Just type your task to get started.
"""

HELP_TEXT = """\
Examples:
  find me the best 10,000mAh powerbank under $50
  search amazon for noise-cancelling headphones and compare the top 3
  look up the latest MacBook Air reviews
"""


def print_profile() -> None:
    profile = load_profile()
    print("\n\033[1m=== Your Profile ===\033[0m")
    print(json.dumps(profile, indent=2, ensure_ascii=False))
    print()


def run_interactive(provider_name: str, model: str | None) -> None:
    """REPL mode — browse directly from the terminal."""
    print(BANNER)
    print(HELP_TEXT)

    provider = get_provider(provider_name, model)
    print(f"Provider: \033[1m{provider_name}\033[0m  model: \033[1m{provider.model}\033[0m\n")

    browser = BrowserController()
    agent: BrowserAgent | None = None

    try:
        while True:
            try:
                user_input = input("\033[1mYou:\033[0m ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nExiting...")
                break

            if not user_input:
                continue

            # ---- built-in commands ----
            if user_input.lower() in ("/quit", "/exit", "quit", "exit"):
                print("Goodbye!")
                break

            if user_input.lower() == "/profile":
                print_profile()
                continue

            if user_input.lower() == "/clear":
                if agent:
                    agent.clear_history()
                print("Conversation history cleared.")
                continue

            if user_input.lower() in ("/help", "help"):
                print(HELP_TEXT)
                continue

            # ---- agent task ----
            if not browser.is_running():
                print("\nStarting browser...")
                browser.start()

            if agent is None:
                agent = BrowserAgent(browser, provider)

            try:
                summary = agent.run_task(user_input)
                if summary and summary != "Task completed.":
                    print(f"\n\033[1m=== Result ===\033[0m\n{summary}\n")
            except KeyboardInterrupt:
                print("\n[Task interrupted by user]")
            except Exception as exc:
                print(f"\n\033[31mAgent error: {exc}\033[0m")
                logging.getLogger(__name__).exception("Agent raised an unhandled exception.")

    finally:
        if browser.is_running():
            print("Closing browser...")
            browser.stop()


def run_orchestrator(orchestrator_url: str, provider_name: str, model: str | None, backend: str) -> None:
    """Orchestrator mode — register and wait for tasks over WebSocket."""
    from orchestrator_client import OrchestratorClient

    async def _start() -> None:
        client = OrchestratorClient(orchestrator_url, provider_name, model, backend=backend)
        await client.start()

    print(f"Connecting to orchestrator at {orchestrator_url} … (backend: {backend})")
    asyncio.run(_start())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Browser Agent — orchestrator mode by default, REPL with -i"
    )
    parser.add_argument(
        "-i", "--interactive",
        action="store_true",
        help="Start in interactive (REPL) mode instead of connecting to the orchestrator",
    )
    parser.add_argument(
        "--provider",
        default=os.environ.get("LLM_PROVIDER", "anthropic"),
        choices=list(PROVIDER_DEFAULTS),
        help="LLM provider to use (default: anthropic)",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("LLM_MODEL"),
        help="Model name override (uses provider default if omitted)",
    )
    parser.add_argument(
        "--orchestrator-url",
        default=os.environ.get("ORCHESTRATOR_URL", "http://localhost:8000"),
        help="Orchestrator base URL (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--backend",
        default=os.environ.get("BROWSER_BACKEND", "chrome-mcp"),
        choices=["chrome-mcp", "playwright"],
        help=(
            "Browser backend to use (default: chrome-mcp). "
            "chrome-mcp: control a real Chrome browser via the Chrome MCP extension. "
            "playwright: launch a headless/visible Chromium instance (started on first task)."
        ),
    )
    args = parser.parse_args()

    if args.interactive:
        run_interactive(args.provider, args.model)
    else:
        run_orchestrator(args.orchestrator_url, args.provider, args.model, args.backend)
