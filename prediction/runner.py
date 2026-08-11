"""Run UMAP + LightGBM backends and write a single predict_results.csv."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import pandas as pd

from prediction.backends.lgbm import score_lgbm
from prediction.backends.umap import score_umap
from prediction.changed import load_changed_sessions
from prediction.merge import read_predict_frame, write_predict_frame
from prediction.schema import (
    COL_PROBABILITY_LGBM,
    COL_PROBABILITY_UMAP,
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


def run_predictions(device: str, *, changed_sessions_file: Optional[str] = None) -> Path:
    device = device.lower()
    output_path = resolve_predict_results_path(device)
    changed_sessions_file = changed_sessions_file or os.environ.get("CHANGED_SESSIONS_FILE")

    existing = read_predict_frame(output_path)
    changed = load_changed_sessions(changed_sessions_file)
    full_rebuild = (
        changed is None
        or bool(changed.get("full_rebuild", False))
        or existing.empty
    )

    if changed is not None and changed.get("session_ids") == [] and not full_rebuild:
        print(f"[{device}] нет changed sessions — сохраняем предыдущий predict_results.csv")
        write_predict_frame(output_path, existing)
        return output_path

    umap_scores = score_umap(device, changed_sessions_file=changed_sessions_file)
    lgbm_scores = score_lgbm(device, changed_sessions_file=changed_sessions_file)

    if full_rebuild:
        all_ids = sorted(set(umap_scores) | set(lgbm_scores))
        rows = {
            sid: {
                COL_SESSION_ID: sid,
                COL_PROBABILITY_UMAP: umap_scores.get(sid, ""),
                COL_PROBABILITY_LGBM: lgbm_scores.get(sid, ""),
            }
            for sid in all_ids
        }
    else:
        rows = {
            str(row[COL_SESSION_ID]): {
                COL_SESSION_ID: str(row[COL_SESSION_ID]),
                COL_PROBABILITY_UMAP: str(row[COL_PROBABILITY_UMAP]),
                COL_PROBABILITY_LGBM: str(row[COL_PROBABILITY_LGBM]),
            }
            for _, row in existing.iterrows()
        }
        for sid, value in umap_scores.items():
            entry = rows.setdefault(
                sid,
                {
                    COL_SESSION_ID: sid,
                    COL_PROBABILITY_UMAP: "",
                    COL_PROBABILITY_LGBM: "",
                },
            )
            entry[COL_PROBABILITY_UMAP] = value
        for sid, value in lgbm_scores.items():
            entry = rows.setdefault(
                sid,
                {
                    COL_SESSION_ID: sid,
                    COL_PROBABILITY_UMAP: "",
                    COL_PROBABILITY_LGBM: "",
                },
            )
            entry[COL_PROBABILITY_LGBM] = value

    out = pd.DataFrame(
        [rows[sid] for sid in sorted(rows.keys())],
        columns=PREDICT_COLUMNS,
    )
    write_predict_frame(output_path, out)
    print(
        f"[{device}] predict_results: {output_path} "
        f"(umap={len(umap_scores)}, lgbm={len(lgbm_scores)}, total={len(out)})"
    )
    return output_path
