"""Core UMAP + HDBSCAN training logic."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Optional, Set

import hdbscan
import joblib
import numpy as np
import pandas as pd
import umap.umap_ as umap

from model_training.config import (
    CENTROID_NAME,
    EXCLUDED_COLUMNS,
    HDBSCAN_MIN_CLUSTER_SIZE,
    HDBSCAN_MIN_SAMPLES,
    UMAP_MODEL_NAME,
    UMAP_RANDOM_STATE,
    DEVICE_CONFIGS,
)
from model_training.paths import TrainingPaths
from order_probability import (
    PROBABILITY_FLOOR,
    PROBABILITY_TAIL_FLOOR,
    compute_decay_sigma,
)


def ensure_can_write(output_dir: Path, is_base: bool, force: bool) -> None:
    umap_path = output_dir / UMAP_MODEL_NAME
    centroid_path = output_dir / CENTROID_NAME
    if is_base and (umap_path.exists() or centroid_path.exists()) and not force:
        print("[ERROR] Базовые модели уже существуют в", output_dir)
        print("Для перезаписи добавьте --force или укажите --site-id / --output-dir.")
        sys.exit(1)


def load_unique_order_sessions(path: Path) -> Set[str]:
    if not path.is_file():
        print(f"[ERROR] Не найден файл сессий с заказами: {path}")
        sys.exit(1)

    if path.suffix == ".pkl":
        sessions = joblib.load(path)
    elif path.suffix == ".json":
        with path.open("r", encoding="utf-8") as handle:
            sessions = set(json.load(handle))
    else:
        with path.open("r", encoding="utf-8") as handle:
            sessions = {line.strip() for line in handle if line.strip()}

    if not isinstance(sessions, set):
        sessions = set(sessions)

    print(f"Загружено {len(sessions)} сессий с заказами из {path}")
    return sessions


def find_order_cluster_centroid(embeddings, labels, is_order):
    cluster_stats = {}
    for cluster_id in np.unique(labels):
        if cluster_id == -1:
            continue

        mask = labels == cluster_id
        order_count = int(is_order[mask].sum())
        cluster_stats[int(cluster_id)] = {
            "order_count": order_count,
            "cluster_size": int(mask.sum()),
        }

    if not cluster_stats:
        print("[ERROR] HDBSCAN не нашёл ни одного кластера (все точки — шум).")
        sys.exit(1)

    best_cluster_id = max(cluster_stats, key=lambda cid: cluster_stats[cid]["order_count"])
    best_stats = cluster_stats[best_cluster_id]

    cluster_mask = labels == best_cluster_id
    centroid = embeddings[cluster_mask].mean(axis=0)

    order_mask = cluster_mask & (is_order == 1)
    if order_mask.any():
        order_distances = np.linalg.norm(embeddings[order_mask] - centroid, axis=1)
        order_distance_p95 = float(np.percentile(order_distances, 95))
        order_distance_median = float(np.median(order_distances))
        order_distance_max = float(np.max(order_distances))
    else:
        all_distances = np.linalg.norm(embeddings[cluster_mask] - centroid, axis=1)
        order_distance_p95 = float(np.percentile(all_distances, 95))
        order_distance_median = float(np.median(all_distances))
        order_distance_max = float(np.max(all_distances))

    if order_distance_p95 <= 0:
        order_distance_p95 = max(order_distance_max, 1e-6)

    all_distances_to_centroid = np.linalg.norm(embeddings - centroid, axis=1)
    all_distance_p99 = float(np.percentile(all_distances_to_centroid, 99))
    decay_sigma = compute_decay_sigma(
        order_distance_p95,
        all_distance_p99,
        PROBABILITY_FLOOR,
        PROBABILITY_TAIL_FLOOR,
    )

    return {
        "cluster_id": best_cluster_id,
        "centroid_x": float(centroid[0]),
        "centroid_y": float(centroid[1]),
        "order_count": best_stats["order_count"],
        "cluster_size": best_stats["cluster_size"],
        "order_ratio": best_stats["order_count"] / best_stats["cluster_size"],
        "order_distance_p95": order_distance_p95,
        "order_distance_median": order_distance_median,
        "order_distance_max": order_distance_max,
        "all_distance_p99": all_distance_p99,
        "decay_sigma": decay_sigma,
        "probability_tail_floor": PROBABILITY_TAIL_FLOOR,
        "normalization": (
            "linear: max(clip(1-d/p95,0,1),0.01); "
            "decay: d<=p95 same, d>p95: 0.01*exp(-(d-p95)/sigma)"
        ),
        "cluster_stats": cluster_stats,
    }


def copy_base_scaler(device: str, output_dir: Path) -> Path:
    config = DEVICE_CONFIGS[device]
    source = config.models_dir / config.scaler_name
    destination = output_dir / config.scaler_name
    if not source.is_file():
        print(f"[ERROR] Не найден базовый scaler: {source}")
        sys.exit(1)

    shutil.copy2(source, destination)
    print(f"Scaler скопирован из базовой модели: {destination}")
    return destination


def train_order_cluster(
    device: str,
    paths: TrainingPaths,
    *,
    force: bool = False,
    copy_scaler: Optional[bool] = None,
) -> Path:
    ensure_can_write(paths.output_dir, paths.is_base, force)

    if not paths.features_path.is_file():
        print(f"[ERROR] Не найден файл признаков: {paths.features_path}")
        if paths.is_base:
            config = DEVICE_CONFIGS[device]
            print(
                f"Положите CSV в {config.training_dir}/ "
                f"или укажите --features."
            )
        sys.exit(1)

    paths.output_dir.mkdir(parents=True, exist_ok=True)
    umap_path = paths.output_dir / UMAP_MODEL_NAME
    centroid_path = paths.output_dir / CENTROID_NAME

    print(f"\n=== Обучение модели ({device}) ===")
    print(f"Признаки: {paths.features_path}")
    print(f"Заказы:   {paths.orders_path}")
    print(f"Выход:    {paths.output_dir}")

    print("\n=== Загрузка признаков ===")
    df_normalized = pd.read_csv(paths.features_path)
    print(f"Загружено сессий: {len(df_normalized)}")

    cols_to_drop = [col for col in EXCLUDED_COLUMNS if col in df_normalized.columns]
    feature_matrix = df_normalized.drop(columns=cols_to_drop)
    print(f"Признаков для UMAP: {feature_matrix.shape[1]}")

    unique_order_sess_a = load_unique_order_sessions(paths.orders_path)

    if "session_id" in df_normalized.columns:
        session_ids = df_normalized["session_id"].astype(str)
    else:
        session_ids = df_normalized.index.astype(str)

    is_order = session_ids.isin(unique_order_sess_a).astype(int).values
    print(f"Сессий с заказами в датафрейме: {is_order.sum()}")

    print("\n=== Обучение UMAP ===")
    umap_model = umap.UMAP(random_state=UMAP_RANDOM_STATE)
    embeddings = umap_model.fit_transform(feature_matrix)
    joblib.dump(umap_model, umap_path)
    print(f"UMAP-модель сохранена в {umap_path}")

    print("\n=== Кластеризация HDBSCAN ===")
    clusterer = hdbscan.HDBSCAN(
        min_samples=HDBSCAN_MIN_SAMPLES,
        min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE,
    )
    labels = clusterer.fit_predict(embeddings)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int((labels == -1).sum())
    print(f"Кластеров: {n_clusters}, шумовых точек: {n_noise}")

    print("\n=== Поиск кластера с наибольшим числом заказов ===")
    result = find_order_cluster_centroid(embeddings, labels, is_order)
    result["feature_columns"] = list(feature_matrix.columns)
    result["hdbscan_min_samples"] = HDBSCAN_MIN_SAMPLES
    result["hdbscan_min_cluster_size"] = HDBSCAN_MIN_CLUSTER_SIZE
    result["umap_random_state"] = UMAP_RANDOM_STATE
    result["features_path"] = str(paths.features_path)
    result["orders_path"] = str(paths.orders_path)
    result["model_target"] = "base" if paths.is_base else "custom"
    if paths.site_id:
        result["site_id"] = paths.site_id

    with centroid_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)

    print(f"Лучший кластер: id={result['cluster_id']}")
    print(
        f"  заказов: {result['order_count']} / {result['cluster_size']} "
        f"({result['order_ratio']:.2%})"
    )
    print(f"  центр масс: x={result['centroid_x']:.6f}, y={result['centroid_y']:.6f}")
    print(f"  p95 расстояния заказов: {result['order_distance_p95']:.6f}")
    print(f"Координаты сохранены в {centroid_path}")

    should_copy_scaler = not paths.is_base if copy_scaler is None else copy_scaler
    if should_copy_scaler:
        copy_base_scaler(device, paths.output_dir)

    return paths.output_dir
