"""UMAP + bot-centroid distance → probability_bot_umap."""

from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path
from typing import Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

from prediction.order_probability import (
    distances_to_probability,
    resolve_distance_p95,
    resolve_probability_mode,
)
from prediction.changed import load_changed_sessions
from prediction.merge import format_score


warnings.filterwarnings(
    "ignore",
    message=".*Trying to unpickle estimator.*",
    module="sklearn.base",
)
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="sklearn.base",
)


def resolve_bot_umap_paths(device: str) -> Tuple[Path, Path, Path]:
    device = device.lower()
    if device == "mobile":
        features = Path(
            os.environ.get(
                "MOBILE_NORMALIZED_FEATURES_PATH",
                os.environ.get(
                    "NORMALIZED_FEATURES_PATH",
                    "tmv_session_features_enhanced_mean_only_normalized.csv",
                ),
            )
        )
        umap = Path(
            os.environ.get(
                "MOBILE_BOT_UMAP_MODEL_PATH",
                os.environ.get("BOT_UMAP_MODEL_PATH", "bot_umap_model.joblib"),
            )
        )
        centroid = Path(
            os.environ.get(
                "MOBILE_BOT_CENTROID_PATH",
                os.environ.get("BOT_CENTROID_PATH", "bot_cluster_centroid.json"),
            )
        )
    elif device == "desktop":
        features = Path(
            os.environ.get(
                "DESKTOP_NORMALIZED_FEATURES_PATH",
                os.environ.get(
                    "NORMALIZED_FEATURES_PATH",
                    "mmv_session_features_enhanced_mean_only_normalized.csv",
                ),
            )
        )
        umap = Path(
            os.environ.get(
                "DESKTOP_BOT_UMAP_MODEL_PATH",
                os.environ.get("BOT_UMAP_MODEL_PATH", "bot_umap_model.joblib"),
            )
        )
        centroid = Path(
            os.environ.get(
                "DESKTOP_BOT_CENTROID_PATH",
                os.environ.get("BOT_CENTROID_PATH", "bot_cluster_centroid.json"),
            )
        )
    else:
        raise ValueError(f"Unknown device: {device}")
    return features, umap, centroid


def load_centroid_config(centroid_path: Path) -> dict:
    if not centroid_path.is_file():
        print(f"[FAIL] Не найден файл центроида: '{centroid_path}'")
        print("Сначала обучите и опубликуйте артефакты из log-analyzer-model-training")
        sys.exit(1)

    with centroid_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    missing = [key for key in ("centroid_x", "centroid_y") if key not in config]
    if missing:
        print(f"[FAIL] В '{centroid_path}' отсутствуют поля: {missing}")
        sys.exit(1)
    try:
        resolve_distance_p95(config)
    except KeyError as exc:
        print(f"[FAIL] В '{centroid_path}': {exc}")
        sys.exit(1)
    return config


def select_feature_matrix(df: pd.DataFrame, centroid_config: dict):
    if "session_id" in df.columns:
        session_ids = df["session_id"].astype(str)
        feature_df = df.drop(columns=["session_id"])
    else:
        session_ids = df.index.astype(str)
        feature_df = df.copy()

    feature_columns = centroid_config.get("feature_columns")
    if feature_columns:
        missing = [col for col in feature_columns if col not in feature_df.columns]
        if missing:
            print("[FAIL] В данных отсутствуют признаки, на которых обучены UMAP/scaler:")
            print(", ".join(missing[:10]))
            if len(missing) > 10:
                print(f"... и ещё {len(missing) - 10}")
            sys.exit(1)
        feature_df = feature_df[feature_columns]

    feature_df = feature_df.fillna(0)
    X = feature_df.values.astype(np.float64)
    X = np.nan_to_num(X, nan=0.0, posinf=1e10, neginf=-1e10)
    return session_ids, X


def score_bot_umap(
    device: str,
    *,
    changed_sessions_file: Optional[str] = None,
) -> Dict[str, str]:
    """Return session_id → probability_bot_umap string for sessions that need scoring."""
    features_path, umap_path, centroid_path = resolve_bot_umap_paths(device)

    if not features_path.is_file():
        print(f"[FAIL] Не найден файл с признаками '{features_path}'.")
        sys.exit(1)
    if not umap_path.is_file():
        print(f"[FAIL] Не найдена UMAP-модель: '{umap_path}'")
        sys.exit(1)

    centroid_config = load_centroid_config(centroid_path)
    centroid = np.array(
        [centroid_config["centroid_x"], centroid_config["centroid_y"]],
        dtype=np.float64,
    )
    probability_mode = resolve_probability_mode(centroid_config)

    print("\n=== UMAP (bot): загрузка признаков ===")
    df = pd.read_csv(features_path)
    changed_payload = load_changed_sessions(changed_sessions_file)

    full_rebuild = changed_payload is None or bool(
        changed_payload.get("full_rebuild", False)
    )

    if changed_payload is not None and changed_payload.get("session_ids") == [] and not full_rebuild:
        print("UMAP (bot): нет changed session_id, пропускаем пересчёт.")
        return {}

    if not full_rebuild:
        changed_set = set(changed_payload["session_ids"])
        if "session_id" in df.columns:
            df["session_id"] = df["session_id"].astype(str)
            df = df[df["session_id"].isin(changed_set)].copy()
        else:
            df.index = df.index.astype(str)
            df = df[df.index.isin(changed_set)].copy()
        print(f"UMAP (bot): инкрементальный прогноз для {len(df)} session_id.")
        if df.empty:
            return {}
    else:
        print("UMAP (bot): полный пересчёт прогнозов.")

    session_ids, X = select_feature_matrix(df, centroid_config)
    print(f"UMAP (bot): сессий={len(X)}, признаков={X.shape[1]}")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        umap_model = joblib.load(umap_path)

    expected_features = getattr(umap_model, "n_features_in_", None)
    if expected_features is not None and X.shape[1] != expected_features:
        print(
            f"[FAIL] Число признаков ({X.shape[1]}) не совпадает с UMAP-моделью ({expected_features})."
        )
        sys.exit(1)

    X_umap = umap_model.transform(X)
    X_umap = np.nan_to_num(X_umap, nan=0.0, posinf=1e10, neginf=-1e10)
    distances = np.linalg.norm(X_umap - centroid, axis=1)
    probabilities = distances_to_probability(distances, centroid_config, probability_mode)

    print(f"UMAP (bot): режим={probability_mode}")
    print(
        "UMAP (bot) вероятности: "
        f"min={probabilities.min():.6f}, median={np.median(probabilities):.6f}, "
        f"max={probabilities.max():.6f}"
    )

    ids = session_ids.astype(str).tolist()
    return {sid: format_score(prob) for sid, prob in zip(ids, probabilities)}
