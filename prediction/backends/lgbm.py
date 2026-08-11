"""LightGBM buyer score → probability_lgbm."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import pandas as pd

from prediction.lgbm.artifact import load_buyer_lgbm_artifact
from prediction.lgbm.embed import encode_sessions, load_transformer_bundle
from prediction.lgbm.score import build_feature_matrix, score_matrix
from prediction.lgbm.tokenize import build_token_features
from prediction.lgbm.uris import build_session_uris_from_ndjson
from prediction.changed import load_changed_sessions
from prediction.merge import format_score


def resolve_lgbm_paths(device: str) -> tuple[Path, Path, Path, Path]:
    device = device.lower()
    if device == "mobile":
        artifact = Path(os.environ.get("MOBILE_BUYER_LGBM_PATH", ""))
        transformer = Path(os.environ.get("MOBILE_TRANSFORMER_PATH", ""))
        features = Path(
            os.environ.get(
                "MOBILE_FEATURES_PATH",
                "tmv_session_features_enhanced_mean_only.csv",
            )
        )
        ndjson = Path(
            os.environ.get(
                "OUTPUT_MOBILE",
                os.environ.get("INPUT_FILE", "merged.cloud.mobile.ndjson"),
            )
        )
    elif device == "desktop":
        artifact = Path(os.environ.get("DESKTOP_BUYER_LGBM_PATH", ""))
        transformer = Path(os.environ.get("DESKTOP_TRANSFORMER_PATH", ""))
        features = Path(
            os.environ.get(
                "DESKTOP_FEATURES_PATH",
                "mmv_session_features_enhanced_mean_only.csv",
            )
        )
        ndjson = Path(
            os.environ.get(
                "JSON_PATH",
                os.environ.get("OUTPUT_DESKTOP", "merged.cloud.desktop.ndjson"),
            )
        )
    else:
        raise ValueError(f"Unknown device: {device}")
    return artifact, transformer, features, ndjson


def score_lgbm(
    device: str,
    *,
    changed_sessions_file: Optional[str] = None,
    existing_session_ids: Optional[Sequence[str]] = None,
) -> Dict[str, str]:
    """Return session_id → probability_lgbm. Empty dict if artifacts missing."""
    artifact_path, transformer_path, features_csv, ndjson_path = resolve_lgbm_paths(device)

    if not artifact_path.as_posix() or not transformer_path.as_posix():
        print(f"[{device} LGBM] артефакты не заданы в env — пропускаем")
        return {}
    if not artifact_path.is_file() or not transformer_path.is_file():
        print(f"[{device} LGBM] файлы не найдены — пропускаем")
        return {}
    if not features_csv.is_file():
        print(f"[{device} LGBM] нет features CSV '{features_csv}' — пропускаем")
        return {}

    artifact = load_buyer_lgbm_artifact(artifact_path)
    features_df = pd.read_csv(features_csv)
    changed = load_changed_sessions(changed_sessions_file)

    full_rebuild = changed is None or bool(changed.get("full_rebuild", False))
    if changed is not None and changed.get("session_ids") == [] and not full_rebuild:
        print(f"[{device} LGBM] нет changed session_id")
        return {}

    if full_rebuild:
        if "session_id" in features_df.columns:
            target_ids = features_df["session_id"].astype(str).tolist()
        else:
            target_ids = features_df.index.astype(str).tolist()
    else:
        target_ids = list(changed["session_ids"])

    session_uris = build_session_uris_from_ndjson(ndjson_path, session_ids=target_ids)
    token_features = build_token_features(session_uris, session_ids=target_ids)
    bundle = load_transformer_bundle(transformer_path)
    embeddings = encode_sessions(token_features, bundle, session_ids=target_ids)

    X, used_ids = build_feature_matrix(target_ids, embeddings, features_df, artifact)
    scores = score_matrix(artifact["model"], X)
    print(f"[{device} LGBM] scored={len(used_ids)} / targets={len(target_ids)}")
    _ = existing_session_ids
    return {sid: format_score(score) for sid, score in zip(used_ids, scores)}
