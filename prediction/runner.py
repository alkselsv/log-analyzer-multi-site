"""Run bot UMAP + LightGBM backends and write predict_results.csv."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import pandas as pd

from prediction.backends.lgbm import score_lgbm
from prediction.backends.umap import score_bot_umap
from prediction.changed import load_changed_sessions
from prediction.merge import read_predict_frame, write_predict_frame
from prediction.schema import (
    COL_PROBABILITY_BOT_UMAP,
    COL_PROBABILITY_LGBM,
    COL_SESSION_ID,
    PREDICT_COLUMNS,
)


def resolve_predict_results_path(device: str) -> Path:
    explicit = os.environ.get("PREDICT_RESULTS_PATH")
    if explicit:
        return Path(explicit)
    if device == "mobile":
        return Path(os.environ.get("MOBILE_PREDICT_RESULTS", "predict_results.csv"))
    return Path(os.environ.get("DESKTOP_PREDICT_RESULTS", "predict_results.csv"))


def _empty_score_row(sid: str) -> dict:
    return {
        COL_SESSION_ID: sid,
        COL_PROBABILITY_BOT_UMAP: "",
        COL_PROBABILITY_LGBM: "",
    }


def _needs_bot_umap_rebuild(existing) -> bool:
    """True when predict frame has no usable bot UMAP scores (e.g. after schema migration)."""
    if existing is None or existing.empty:
        return True
    if COL_PROBABILITY_BOT_UMAP not in existing.columns:
        return True
    values = existing[COL_PROBABILITY_BOT_UMAP].fillna("").astype(str).str.strip()
    return bool((values == "").all())


def run_predictions(device: str, *, changed_sessions_file: Optional[str] = None) -> Path:
    device = device.lower()
    output_path = resolve_predict_results_path(device)
    changed_sessions_file = changed_sessions_file or os.environ.get("CHANGED_SESSIONS_FILE")

    existing = read_predict_frame(output_path)
    changed = load_changed_sessions(changed_sessions_file)
    force_env = os.environ.get("FORCE_FULL_REBUILD", "").strip() == "1"
    full_rebuild = (
        changed is None
        or bool(changed.get("full_rebuild", False))
        or existing.empty
        or force_env
        or _needs_bot_umap_rebuild(existing)
    )
    if full_rebuild and not (
        changed is None or bool(changed.get("full_rebuild", False)) or existing.empty
    ):
        reason = "FORCE_FULL_REBUILD" if force_env else "пустой probability_bot_umap"
        print(f"[{device}] полный пересчёт прогнозов ({reason})")

    if changed is not None and changed.get("session_ids") == [] and not full_rebuild:
        print(f"[{device}] нет changed sessions — сохраняем предыдущий predict_results.csv")
        write_predict_frame(output_path, existing)
        return output_path

    # Backends also read CHANGED_SESSIONS_FILE; for forced rebuild pass None
    # so they score the full features CSV.
    backend_changed = None if full_rebuild else changed_sessions_file
    bot_umap_scores = score_bot_umap(device, changed_sessions_file=backend_changed)
    lgbm_scores = score_lgbm(device, changed_sessions_file=backend_changed)

    if full_rebuild:
        all_ids = sorted(set(bot_umap_scores) | set(lgbm_scores))
        rows = {
            sid: {
                COL_SESSION_ID: sid,
                COL_PROBABILITY_BOT_UMAP: bot_umap_scores.get(sid, ""),
                COL_PROBABILITY_LGBM: lgbm_scores.get(sid, ""),
            }
            for sid in all_ids
        }
    else:
        rows = {
            str(row[COL_SESSION_ID]): {
                COL_SESSION_ID: str(row[COL_SESSION_ID]),
                COL_PROBABILITY_BOT_UMAP: str(row.get(COL_PROBABILITY_BOT_UMAP, "")),
                COL_PROBABILITY_LGBM: str(row[COL_PROBABILITY_LGBM]),
            }
            for _, row in existing.iterrows()
        }
        for sid, value in bot_umap_scores.items():
            entry = rows.setdefault(sid, _empty_score_row(sid))
            entry[COL_PROBABILITY_BOT_UMAP] = value
        for sid, value in lgbm_scores.items():
            entry = rows.setdefault(sid, _empty_score_row(sid))
            entry[COL_PROBABILITY_LGBM] = value

    out = pd.DataFrame(
        [rows[sid] for sid in sorted(rows.keys())],
        columns=PREDICT_COLUMNS,
    )
    write_predict_frame(output_path, out)
    print(
        f"[{device}] predict_results: {output_path} "
        f"(bot_umap={len(bot_umap_scores)}, lgbm={len(lgbm_scores)}, total={len(out)})"
    )
    return output_path
