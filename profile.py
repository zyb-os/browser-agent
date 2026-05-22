"""User profile manager — persists learned preferences to data/profile.json."""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

PROFILE_PATH = Path(__file__).parent / "data" / "profile.json"

DEFAULT_PROFILE = {
    "budget_range": None,
    "preferred_brands": [],
    "disliked_brands": [],
    "use_cases": [],
    "priorities": [],
    "location": None,
    "currency": "USD",
    "notes": [],
    "purchase_history": [],
}


def load_profile() -> dict:
    """Load the user profile from disk, creating a default one if absent."""
    if not PROFILE_PATH.exists():
        PROFILE_PATH.parent.mkdir(parents=True, exist_ok=True)
        save_profile(DEFAULT_PROFILE.copy())
        return DEFAULT_PROFILE.copy()

    try:
        with open(PROFILE_PATH, "r", encoding="utf-8") as f:
            profile = json.load(f)
        # Merge any missing keys from default
        for key, value in DEFAULT_PROFILE.items():
            if key not in profile:
                profile[key] = value
        return profile
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Failed to load profile (%s), using defaults.", e)
        return DEFAULT_PROFILE.copy()


def save_profile(profile: dict) -> None:
    """Write the profile back to disk."""
    PROFILE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(PROFILE_PATH, "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2, ensure_ascii=False)
    logger.debug("Profile saved to %s", PROFILE_PATH)


def profile_summary(profile: dict) -> str:
    """Format the profile as a concise human-readable string for Claude's system prompt."""
    lines = ["=== USER PROFILE ==="]

    if profile.get("budget_range"):
        lines.append(f"Budget: {profile['budget_range']}")

    if profile.get("currency"):
        lines.append(f"Currency: {profile['currency']}")

    if profile.get("location"):
        lines.append(f"Location: {profile['location']}")

    if profile.get("preferred_brands"):
        lines.append(f"Preferred brands: {', '.join(profile['preferred_brands'])}")

    if profile.get("disliked_brands"):
        lines.append(f"Disliked brands: {', '.join(profile['disliked_brands'])}")

    if profile.get("use_cases"):
        lines.append(f"Use cases: {', '.join(profile['use_cases'])}")

    if profile.get("priorities"):
        lines.append(f"Priorities: {', '.join(profile['priorities'])}")

    if profile.get("notes"):
        lines.append("Notes:")
        for note in profile["notes"]:
            lines.append(f"  - {note}")

    if profile.get("purchase_history"):
        lines.append("Past purchases:")
        for item in profile["purchase_history"]:
            lines.append(f"  - {item}")

    if len(lines) == 1:
        lines.append("(No preferences recorded yet — will learn from this session)")

    lines.append("===================")
    return "\n".join(lines)


def update_profile_field(profile: dict, key: str, value) -> dict:
    """Update a single profile field, handling list-type fields by appending."""
    if key not in DEFAULT_PROFILE:
        # Store unknown keys under notes
        if "notes" not in profile:
            profile["notes"] = []
        profile["notes"].append(f"{key}: {value}")
        return profile

    if isinstance(DEFAULT_PROFILE[key], list):
        if not isinstance(profile.get(key), list):
            profile[key] = []
        if isinstance(value, list):
            for item in value:
                if item not in profile[key]:
                    profile[key].append(item)
        elif value not in profile[key]:
            profile[key].append(value)
    else:
        profile[key] = value

    return profile
