"""
Прогноз вероятности заказа по расстоянию до центра кластера заказов в UMAP-пространстве.

Для каждой mobile-сессии:
  1. Берёт нормализованные признаки (scaler уже применён в preprocessor)
  2. Понижает размерность обученным UMAP
  3. Считает евклидово расстояние до центра масс кластера заказов
  4. Преобразует расстояние в вероятность через обратную линейную нормализацию
     относительно 95-го перцентиля расстояний обучающих заказов
"""

import json
import os
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from order_probability import (
    PROBABILITY_FLOOR,
    distances_to_probability,
    resolve_probability_mode,
)

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

FEATURES_PATH = os.environ.get(
    "MOBILE_NORMALIZED_FEATURES_PATH",
    os.environ.get("NORMALIZED_FEATURES_PATH", "tmv_session_features_enhanced_mean_only_normalized.csv"),
)
UMAP_MODEL_PATH = os.environ.get(
    "MOBILE_UMAP_MODEL_PATH",
    os.environ.get("UMAP_MODEL_PATH", "order_umap_model.joblib"),
)
CENTROID_PATH = os.environ.get(
    "MOBILE_CENTROID_PATH",
    os.environ.get("CENTROID_PATH", "order_cluster_centroid.json"),
)
OUTPUT_PATH = os.environ.get(
    "PREDICT_RESULTS_PATH",
    os.environ.get("MOBILE_PREDICT_RESULTS", "predict_results.csv"),
)
CHANGED_SESSIONS_FILE = os.environ.get("CHANGED_SESSIONS_FILE", "changed_sessions.json")
PROBABILITY_FLOOR = 0.01


def load_changed_sessions():
    if not os.path.exists(CHANGED_SESSIONS_FILE):
        return None

    try:
        with open(CHANGED_SESSIONS_FILE, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except (ValueError, OSError, json.JSONDecodeError):
        return None

    if isinstance(payload, list):
        return {
            "full_rebuild": False,
            "session_ids": [str(item) for item in payload],
        }

    if isinstance(payload, dict):
        session_ids = payload.get("session_ids")
        if not isinstance(session_ids, list):
            return None
        return {
            "full_rebuild": bool(payload.get("full_rebuild", False)),
            "session_ids": [str(item) for item in session_ids],
        }

    return None


def load_centroid_config():
    if not os.path.exists(CENTROID_PATH):
        print(f"[FAIL] Не найден файл центроида: '{CENTROID_PATH}'")
        print("Сначала выполните scripts/train_base_models.py или scripts/train_site_models.py")
        sys.exit(1)

    with open(CENTROID_PATH, "r", encoding="utf-8") as f:
        config = json.load(f)

    required = ("centroid_x", "centroid_y", "order_distance_p95")
    missing = [key for key in required if key not in config]
    if missing:
        print(f"[FAIL] В '{CENTROID_PATH}' отсутствуют поля: {missing}")
        sys.exit(1)

    return config


def select_feature_matrix(df, centroid_config):
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


def run_order_prediction():
    if not os.path.exists(FEATURES_PATH):
        print(
            f"[FAIL] Не найден файл с признаками '{FEATURES_PATH}'. "
            "Убедитесь, что preprocessor_tmv_enhanced_fixed.py выполнен успешно."
        )
        sys.exit(1)

    centroid_config = load_centroid_config()
    centroid = np.array([
        centroid_config["centroid_x"],
        centroid_config["centroid_y"],
    ], dtype=np.float64)
    probability_mode = resolve_probability_mode(centroid_config)

    print("\n=== Загрузка признаков сессий ===")
    df = pd.read_csv(FEATURES_PATH)
    changed_payload = load_changed_sessions()
    changed_sessions = None if changed_payload is None else changed_payload["session_ids"]
    existing_predictions = None

    if os.path.exists(OUTPUT_PATH):
        existing_predictions = pd.read_csv(OUTPUT_PATH)
        if "session_id" in existing_predictions.columns:
            existing_predictions["session_id"] = existing_predictions["session_id"].astype(str)

    full_rebuild = (
        changed_payload is None
        or bool(changed_payload.get("full_rebuild", False))
        or existing_predictions is None
    )

    if changed_sessions == [] and existing_predictions is not None:
        print("Новых session_id для прогноза нет, сохраняем существующие результаты.")
        existing_predictions.to_csv(OUTPUT_PATH, index=False)
        return

    if not full_rebuild:
        changed_set = set(changed_sessions)
        if "session_id" in df.columns:
            df["session_id"] = df["session_id"].astype(str)
            df = df[df["session_id"].isin(changed_set)].copy()
        else:
            df.index = df.index.astype(str)
            df = df[df.index.isin(changed_set)].copy()
        print(f"Режим: инкрементальный прогноз для {len(df)} session_id.")
        if df.empty:
            print("В normalized CSV нет строк для changed session_id, сохраняем предыдущий predict_results.csv.")
            existing_predictions.to_csv(OUTPUT_PATH, index=False)
            return
    else:
        print("Режим: полный пересчет прогнозов.")

    session_ids, X = select_feature_matrix(df, centroid_config)
    print(f"Число сессий: {len(X)}, число признаков: {X.shape[1]}")

    if not os.path.exists(UMAP_MODEL_PATH):
        print(f"[FAIL] Не найдена UMAP-модель: '{UMAP_MODEL_PATH}'")
        sys.exit(1)

    print("\n=== UMAP и расчёт расстояния до центра кластера заказов ===")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        umap_model = joblib.load(UMAP_MODEL_PATH)

    expected_features = getattr(umap_model, "n_features_in_", None)
    if expected_features is not None and X.shape[1] != expected_features:
        print(
            f"[FAIL] Число признаков ({X.shape[1]}) не совпадает с UMAP-моделью ({expected_features}). "
            "Переобучите order_umap_model.joblib и order_cluster_centroid.json на текущем наборе признаков."
        )
        sys.exit(1)

    X_umap = umap_model.transform(X)
    X_umap = np.nan_to_num(X_umap, nan=0.0, posinf=1e10, neginf=-1e10)

    distances = np.linalg.norm(X_umap - centroid, axis=1)
    probabilities = distances_to_probability(distances, centroid_config, probability_mode)

    print(f"Центроид: x={centroid[0]:.6f}, y={centroid[1]:.6f}")
    print(f"Режим probability: {probability_mode}")
    if probability_mode == "decay":
        print(
            "Нормализация: d<=p95 -> max(1-d/p95,0.01); "
            f"d>p95 -> 0.01*exp(-(d-p95)/{centroid_config.get('decay_sigma', 'sigma')})"
        )
        print(
            f"p95={float(centroid_config['order_distance_p95']):.6f}, "
            f"p99_all={float(centroid_config['all_distance_p99']):.6f}"
        )
    else:
        print(
            f"Нормализация: max(clip(1 - distance / p95, 0, 1), {PROBABILITY_FLOOR}), "
            f"p95={float(centroid_config['order_distance_p95']):.6f}"
        )
    print(
        "Расстояния: "
        f"min={distances.min():.6f}, median={np.median(distances):.6f}, max={distances.max():.6f}"
    )
    print(
        "Вероятности: "
        f"min={probabilities.min():.6f}, median={np.median(probabilities):.6f}, max={probabilities.max():.6f}"
    )

    print(f"\n=== Сохранение результатов прогноза в '{OUTPUT_PATH}' ===")
    out_df = pd.DataFrame(
        {
            "session_id": session_ids.values,
            "probability": probabilities,
            "distance_to_order_centroid": distances,
        }
    )

    if not full_rebuild and existing_predictions is not None:
        keep_columns = ["session_id", "probability"]
        if "distance_to_order_centroid" in existing_predictions.columns:
            keep_columns.append("distance_to_order_centroid")
        unchanged_df = existing_predictions[
            ~existing_predictions["session_id"].isin(out_df["session_id"])
        ][keep_columns].copy()
        if not unchanged_df.empty:
            out_df = pd.concat([unchanged_df, out_df], ignore_index=True)
        out_df = out_df.drop_duplicates(subset=["session_id"], keep="last")
        out_df = out_df.sort_values("session_id").reset_index(drop=True)

    out_df[["session_id", "probability"]].to_csv(OUTPUT_PATH, index=False)
    print(f"[OK] Результаты сохранены в '{OUTPUT_PATH}'")


if __name__ == "__main__":
    run_order_prediction()
