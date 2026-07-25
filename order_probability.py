"""Преобразование расстояния до центроида заказов в probability."""

import numpy as np

PROBABILITY_FLOOR = 0.01
PROBABILITY_TAIL_FLOOR = 0.001


def compute_decay_sigma(
    p95,
    p99_all,
    floor=PROBABILITY_FLOOR,
    tail_floor=PROBABILITY_TAIL_FLOOR,
):
    if p99_all <= p95:
        return max(float(p95), 1e-6)
    ratio = floor / tail_floor
    if ratio <= 1:
        return max(float(p99_all - p95), 1e-6)
    return float(p99_all - p95) / np.log(ratio)


def distances_to_probability_linear(distances, p95, floor=PROBABILITY_FLOOR):
    if p95 <= 0:
        return np.zeros_like(distances, dtype=np.float64)
    prob = 1.0 - (np.asarray(distances, dtype=np.float64) / p95)
    prob = np.clip(prob, 0.0, 1.0)
    return np.maximum(prob, floor)


def distances_to_probability_decay(
    distances,
    p95,
    p99_all,
    floor=PROBABILITY_FLOOR,
    tail_floor=PROBABILITY_TAIL_FLOOR,
):
    if p95 <= 0:
        return np.zeros_like(distances, dtype=np.float64)

    distances = np.asarray(distances, dtype=np.float64)
    sigma = compute_decay_sigma(p95, p99_all, floor, tail_floor)
    prob = np.empty_like(distances)

    inner = distances <= p95
    prob[inner] = np.maximum(1.0 - distances[inner] / p95, floor)

    excess = distances[~inner] - p95
    prob[~inner] = floor * np.exp(-excess / sigma)
    return np.maximum(prob, tail_floor)


def resolve_probability_mode(centroid_config):
    import os

    mode = os.environ.get("PROBABILITY_MODE", "decay").strip().lower()
    if mode not in {"linear", "decay"}:
        mode = "decay"
    if mode == "decay" and centroid_config.get("all_distance_p99") is None:
        mode = "linear"
    return mode


def distances_to_probability(distances, centroid_config, mode=None):
    p95 = float(centroid_config["order_distance_p95"])
    if mode is None:
        mode = resolve_probability_mode(centroid_config)

    if mode == "decay":
        return distances_to_probability_decay(
            distances,
            p95,
            float(centroid_config["all_distance_p99"]),
        )
    return distances_to_probability_linear(distances, p95)
