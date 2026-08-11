"""Changed-sessions payload loader."""

from __future__ import annotations

import json
import os
from typing import Optional


def load_changed_sessions(path: Optional[str] = None) -> Optional[dict]:
    path = path or os.environ.get("CHANGED_SESSIONS_FILE", "changed_sessions.json")
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, ValueError, json.JSONDecodeError):
        return None

    if isinstance(payload, list):
        return {"full_rebuild": False, "session_ids": [str(item) for item in payload]}
    if isinstance(payload, dict):
        session_ids = payload.get("session_ids")
        if not isinstance(session_ids, list):
            return None
        return {
            "full_rebuild": bool(payload.get("full_rebuild", False)),
            "session_ids": [str(item) for item in session_ids],
        }
    return None
