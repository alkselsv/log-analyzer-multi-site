"""Shared prediction schema: dual scores probability_umap + probability_lgbm."""

from __future__ import annotations

COL_SESSION_ID = "session_id"
COL_PROBABILITY_UMAP = "probability_umap"
COL_PROBABILITY_LGBM = "probability_lgbm"

PREDICT_COLUMNS = [
    COL_SESSION_ID,
    COL_PROBABILITY_UMAP,
    COL_PROBABILITY_LGBM,
]

DEVICE_OUT_FIELDNAMES = [
    "date",
    COL_SESSION_ID,
    COL_PROBABILITY_UMAP,
    COL_PROBABILITY_LGBM,
    "ip",
    "user_agent",
]

COMBINED_OUT_FIELDNAMES = [
    "date",
    "device_type",
    COL_SESSION_ID,
    COL_PROBABILITY_UMAP,
    COL_PROBABILITY_LGBM,
    "ip",
    "user_agent",
]

COMBINED_PREDICT_FIELDNAMES = [
    "device_type",
    COL_SESSION_ID,
    COL_PROBABILITY_UMAP,
    COL_PROBABILITY_LGBM,
]
