"""
Per-domain UI action cache.

Observes LLM-driven tool calls, groups them by (domain, url-path), and
replays them on future visits without sending screenshots to the LLM —
cutting both image-token cost and inference round-trips for known pages.

data/site_knowledge.json schema
────────────────────────────────
{
  "<domain>": {
    "<url-path>": {
      "steps":  [{"tool": "<name>", "input": {…}}, …],
      "hits":   <int>,   // times replayed successfully
      "misses": <int>    // times replay failed
    }
  }
}
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

KNOWLEDGE_FILE = Path("data/site_knowledge.json")

# Tools whose behaviour is position-based and task-independent: safe to replay.
# navigate  → excluded; the LLM decides which URL to visit.
# type_text → excluded; content is task-specific (see RECORDING_STOPPERS).
# screenshot / task_complete / profile tools → excluded; not browser interactions.
CACHEABLE_TOOLS: frozenset[str] = frozenset(
    {"click", "key_press", "scroll", "hover", "wait", "go_back"}
)

# When one of these tools is called on a page, stop recording for that page.
# Any steps that follow depend on the task-specific content (e.g. what was
# typed) and are therefore unsafe to replay out of context.
RECORDING_STOPPERS: frozenset[str] = frozenset({"type_text", "task_complete"})

MIN_HITS = 1            # successes required before enabling auto-replay
MIN_CONFIDENCE = 0.70   # minimum hit / (hit + miss) ratio


def _split_url(url: str) -> tuple[str, str]:
    """Return (domain_without_www, path) for a URL string."""
    try:
        p = urlparse(url)
        return p.netloc.removeprefix("www."), (p.path or "/")
    except Exception:
        return "", "/"


class SiteKnowledge:
    """Records and replays per-page action sequences to reduce LLM token usage."""

    def __init__(self) -> None:
        self._db: dict = {}
        self._recording: list[dict] = []
        # Pages for which recording has been sealed (a RECORDING_STOPPER was seen)
        self._stopped_pages: set[tuple[str, str]] = set()
        self._load()

    # ── persistence ──────────────────────────────────────────────────────────

    def _load(self) -> None:
        if KNOWLEDGE_FILE.exists():
            try:
                self._db = json.loads(KNOWLEDGE_FILE.read_text(encoding="utf-8"))
            except Exception as exc:
                logger.warning("site_knowledge: failed to load — %s", exc)
                self._db = {}

    def _save(self) -> None:
        KNOWLEDGE_FILE.parent.mkdir(parents=True, exist_ok=True)
        KNOWLEDGE_FILE.write_text(
            json.dumps(self._db, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    # ── recording ─────────────────────────────────────────────────────────────

    def reset_recording(self) -> None:
        """Call at the start of each task to clear any stale recording state."""
        self._recording = []
        self._stopped_pages = set()

    def observe(self, url: str, tool_name: str, tool_input: dict) -> None:
        """
        Record one successful LLM-driven tool call made while on *url*.

        Recording for a (domain, path) pair is automatically sealed when a
        RECORDING_STOPPER tool is seen — subsequent steps on that page are
        skipped because they may depend on the task-specific content that
        triggered the stop.

        Non-cacheable tools (navigate, screenshot, profile tools) are silently
        ignored; they neither record nor seal the page.
        """
        domain, path = _split_url(url)
        if not domain:
            return

        page_key = (domain, path)

        if page_key in self._stopped_pages:
            return  # sequence for this page is already sealed

        if tool_name in RECORDING_STOPPERS:
            self._stopped_pages.add(page_key)
            return

        if tool_name in CACHEABLE_TOOLS:
            self._recording.append(
                {"domain": domain, "path": path, "tool": tool_name, "input": dict(tool_input)}
            )

    def commit(self) -> None:
        """
        Persist the current recording as a successful observation.
        Groups steps by (domain, path), overwrites any prior sequence for that
        page, and increments hit counters.  No-op when nothing was recorded.
        """
        if not self._recording:
            self._stopped_pages = set()
            return

        # Group consecutive steps by page, preserving order
        seen: dict[tuple[str, str], list[dict]] = {}
        for s in self._recording:
            key = (s["domain"], s["path"])
            seen.setdefault(key, []).append({"tool": s["tool"], "input": s["input"]})

        for (domain, path), steps in seen.items():
            self._db.setdefault(domain, {}).setdefault(
                path, {"steps": [], "hits": 0, "misses": 0}
            )
            entry = self._db[domain][path]
            entry["steps"] = steps
            entry["hits"] = entry.get("hits", 0) + 1

        self._recording = []
        self._stopped_pages = set()
        self._save()
        logger.info("site_knowledge: committed %d page-group(s).", len(seen))

    def discard(self) -> None:
        """Throw away the current recording (task interrupted or failed)."""
        self._recording = []
        self._stopped_pages = set()

    # ── replay ────────────────────────────────────────────────────────────────

    def get_steps(self, url: str) -> list[dict] | None:
        """
        Return the cached step sequence for *url* when confidence is sufficient.
        Returns None when the URL is unknown, has too few samples, or the
        success rate is below MIN_CONFIDENCE.
        """
        domain, path = _split_url(url)
        if not domain:
            return None

        entry = self._db.get(domain, {}).get(path)
        if not entry or not entry.get("steps"):
            return None

        hits = entry.get("hits", 0)
        misses = entry.get("misses", 0)

        if hits < MIN_HITS:
            return None

        total = hits + misses
        if total and (hits / total) < MIN_CONFIDENCE:
            return None

        return list(entry["steps"])

    def record_replay_miss(self, url: str) -> None:
        """Increment the miss counter when a cached replay fails for *url*."""
        domain, path = _split_url(url)
        entry = self._db.get(domain, {}).get(path)
        if entry is not None:
            entry["misses"] = entry.get("misses", 0) + 1
            self._save()
