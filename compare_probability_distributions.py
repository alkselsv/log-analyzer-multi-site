#!/usr/bin/env python3
"""Сравнение linear и decay распределений probability на текущих артефактах."""

import json
import os
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)

from order_probability import (
    distances_to_probability_decay,
    distances_to_probability_linear,
)

DEVICES = (
    {
        "name": "mobile",
        "models_dir": ROOT / "mobile" / "models",
        "features": ROOT / "mobile" / "training" / "tmv_session_features_enhanced_mean_only_normalized.csv",
    },
    {
        "name": "desktop",
        "models_dir": ROOT / "desktop" / "models",
        "features": ROOT / "desktop" / "training" / "mmv_session_features_enhanced_mean_only_normalized.csv",
    },
)


def load_matrix(features_path, centroid_config):
    df = pd.read_csv(features_path)
    feature_columns = centroid_config["feature_columns"]
    missing = [col for col in feature_columns if col not in df.columns]
    if missing:
        raise ValueError(f"{features_path}: missing features {missing[:5]}")
    X = df[feature_columns].fillna(0).values.astype(np.float64)
    X = np.nan_to_num(X, nan=0.0, posinf=1e10, neginf=-1e10)
    return X


def summarize(name, distances, linear, decay):
    p95 = float(np.percentile(distances, 95))
    above_p95 = distances > p95

    def stats(prob):
        return {
            "min": float(np.min(prob)),
            "p25": float(np.percentile(prob, 25)),
            "median": float(np.median(prob)),
            "p75": float(np.percentile(prob, 75)),
            "max": float(np.max(prob)),
            "at_floor_linear": int(np.sum(prob <= 0.0100001)),
            "below_0.005": int(np.sum(prob < 0.005)),
        }

    print(f"\n=== {name} ({len(distances)} sessions) ===")
    print(f"distance: min={distances.min():.3f}, median={np.median(distances):.3f}, "
          f"max={distances.max():.3f}, >p95={above_p95.sum()} ({100*above_p95.mean():.1f}%)")

    s_linear = stats(linear)
    s_decay = stats(decay)
    print("\nlinear:")
    for key, value in s_linear.items():
        print(f"  {key}: {value}")
    print("\ndecay:")
    for key, value in s_decay.items():
        print(f"  {key}: {value}")

    diff = decay - linear
    print("\ndelta (decay - linear):")
    print(f"  mean={diff.mean():+.6f}, median={np.median(diff):+.6f}, "
          f"max={diff.max():+.6f}, min={diff.min():+.6f}")
    print(f"  changed sessions (|delta|>1e-6): {int(np.sum(np.abs(diff) > 1e-6))}")

    buckets = [0.01, 0.005, 0.002, 0.001]
    print("\nbuckets (decay only, d > p95_order):")
    far = above_p95
    if far.any():
        for threshold in buckets:
            count = int(np.sum(decay[far] <= threshold + 1e-12))
            print(f"  <= {threshold:.3f}: {count} ({100*count/far.sum():.1f}% of far)")


def main():
    for device in DEVICES:
        models_dir = device["models_dir"]
        features_path = device["features"]

        with open(models_dir / "order_cluster_centroid.json", "r", encoding="utf-8") as f:
            centroid_config = json.load(f)

        p95 = float(centroid_config["order_distance_p95"])
        p99_all = centroid_config.get("all_distance_p99")
        if p99_all is None:
            print(f"[WARN] {device['name']}: all_distance_p99 missing, computing from features...")
        centroid = np.array([
            centroid_config["centroid_x"],
            centroid_config["centroid_y"],
        ], dtype=np.float64)

        X = load_matrix(features_path, centroid_config)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            umap_model = joblib.load(models_dir / "order_umap_model.joblib")
        X_umap = umap_model.transform(X)
        X_umap = np.nan_to_num(X_umap, nan=0.0, posinf=1e10, neginf=-1e10)
        distances = np.linalg.norm(X_umap - centroid, axis=1)

        if p99_all is None:
            p99_all = float(np.percentile(distances, 99))
            centroid_config["all_distance_p99"] = p99_all

        linear = distances_to_probability_linear(distances, p95)
        decay = distances_to_probability_decay(distances, p95, float(p99_all))
        summarize(device["name"], distances, linear, decay)
        print(
            f"\n  calibration: p95_order={p95:.3f}, p99_all={float(p99_all):.3f}, "
            f"decay_sigma={centroid_config.get('decay_sigma', 'n/a')}"
        )


if __name__ == "__main__":
    main()
