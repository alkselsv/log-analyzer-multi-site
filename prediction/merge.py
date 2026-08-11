"""Load/write predict_results.csv with dual score columns."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from prediction.schema import (
    COL_PROBABILITY_LGBM,
    COL_PROBABILITY_UMAP,
    COL_SESSION_ID,
    PREDICT_COLUMNS,
)


def _clean_score_series(series: pd.Series) -> pd.Series:
    out = series.fillna("").astype(str)
    return out.replace({"nan": "", "None": "", "<NA>": ""})


def empty_predict_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=PREDICT_COLUMNS)


def normalize_predict_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if COL_SESSION_ID not in out.columns:
        raise ValueError("В predict frame нет session_id")
    out[COL_SESSION_ID] = out[COL_SESSION_ID].astype(str)

    # Migrate legacy column name.
    if COL_PROBABILITY_UMAP not in out.columns and "probability" in out.columns:
        out[COL_PROBABILITY_UMAP] = out["probability"]

    if COL_PROBABILITY_UMAP not in out.columns:
        out[COL_PROBABILITY_UMAP] = ""
    if COL_PROBABILITY_LGBM not in out.columns:
        out[COL_PROBABILITY_LGBM] = ""

    out[COL_PROBABILITY_UMAP] = _clean_score_series(out[COL_PROBABILITY_UMAP])
    out[COL_PROBABILITY_LGBM] = _clean_score_series(out[COL_PROBABILITY_LGBM])
    return out


def read_predict_frame(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return empty_predict_frame()
    return normalize_predict_frame(pd.read_csv(path))


def write_predict_frame(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = normalize_predict_frame(df)
    out[PREDICT_COLUMNS].sort_values(COL_SESSION_ID).to_csv(path, index=False)


def format_score(value: float) -> str:
    return f"{float(value):.10g}"
