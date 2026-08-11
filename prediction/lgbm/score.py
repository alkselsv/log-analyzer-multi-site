"""Feature matrix helpers for buyer LightGBM (used by prediction.backends.lgbm)."""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from prediction.lgbm.embed import zero_embedding


def embedding_column_names(dim: int) -> List[str]:
    return [f"transformer_embedding_{i}" for i in range(dim)]


def build_feature_matrix(
    session_ids: Sequence[str],
    embeddings: Dict[str, np.ndarray],
    features_df: pd.DataFrame,
    artifact: dict,
) -> Tuple[np.ndarray, List[str]]:
    """Return X aligned to artifact['feature_columns'] and ordered session_ids used."""
    nobot_cols: List[str] = list(artifact["nobot_feature_columns"])
    scaler = artifact["feature_scaler"]
    embedding_dim = int(artifact.get("embedding_dim", 64))
    feature_columns: List[str] = list(artifact["feature_columns"])

    if "session_id" in features_df.columns:
        feat = features_df.copy()
        feat["session_id"] = feat["session_id"].astype(str)
        feat = feat.set_index("session_id")
    else:
        feat = features_df.copy()
        feat.index = feat.index.astype(str)

    missing_feats = [col for col in nobot_cols if col not in feat.columns]
    if missing_feats:
        raise ValueError(f"В features CSV нет колонок: {missing_feats}")

    used_ids: List[str] = []
    rows: List[np.ndarray] = []
    for sid in session_ids:
        sid = str(sid)
        if sid not in feat.index:
            continue
        emb = embeddings.get(sid)
        if emb is None:
            emb = zero_embedding(embedding_dim)
        emb = np.asarray(emb, dtype=np.float32).reshape(-1)
        if emb.shape[0] != embedding_dim:
            raise ValueError(
                f"embedding dim {emb.shape[0]} != expected {embedding_dim} for {sid}"
            )

        raw = feat.loc[sid, nobot_cols]
        if hasattr(raw, "to_frame"):
            raw_df = raw.to_frame().T
        else:
            raw_df = pd.DataFrame([raw], columns=nobot_cols)
        raw_df = raw_df.fillna(0.0)
        scaled = scaler.transform(raw_df).astype(np.float32).reshape(-1)

        vector_map = {
            **{
                name: float(emb[i])
                for i, name in enumerate(embedding_column_names(embedding_dim))
            },
            **{name: float(scaled[i]) for i, name in enumerate(nobot_cols)},
        }
        rows.append(
            np.asarray([vector_map[col] for col in feature_columns], dtype=np.float32)
        )
        used_ids.append(sid)

    if not rows:
        return np.zeros((0, len(feature_columns)), dtype=np.float32), []
    return np.vstack(rows), used_ids


def score_matrix(model, X: np.ndarray) -> np.ndarray:
    if X.size == 0:
        return np.asarray([], dtype=np.float64)
    return model.predict_proba(X)[:, 1].astype(np.float64)
