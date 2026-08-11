"""Build ordered unique-page URI sequences from device NDJSON logs."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set


def build_session_uris_from_ndjson(
    ndjson_path: Path | str,
    session_ids: Optional[Iterable[str]] = None,
) -> Dict[str, List[str]]:
    """Return session_id → [uri…] (first URI per sess.c, ordered by timestamp)."""
    wanted: Optional[Set[str]] = None
    if session_ids is not None:
        wanted = {str(sid) for sid in session_ids}

    pages_raw: Dict[str, List[tuple]] = defaultdict(list)
    path = Path(ndjson_path)
    if not path.is_file():
        return {}

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            sess = record.get("sess") or {}
            session_id = sess.get("a")
            if not session_id:
                continue
            session_id = str(session_id)
            if wanted is not None and session_id not in wanted:
                continue

            sess_c = sess.get("c")
            uri = ((record.get("uri") or {}).get("v")) or ""
            timestamp = record.get("ts")
            pages_raw[session_id].append((timestamp, sess_c, uri))

    session_uris: Dict[str, List[str]] = {}
    for session_id, events in pages_raw.items():
        events = sorted(events, key=lambda item: (item[0] or 0))
        seen: Set = set()
        uris: List[str] = []
        for _ts, sess_c, uri in events:
            if sess_c in seen:
                continue
            seen.add(sess_c)
            uris.append(uri)
        session_uris[session_id] = uris
    return session_uris
