"""Resolve input and output paths for model training."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from model_training.config import (
    ORDER_SESS_FILENAMES,
    DeviceConfig,
    DEVICE_CONFIGS,
)
from site_registry import SITES_DIR


@dataclass(frozen=True)
class TrainingPaths:
    features_path: Path
    orders_path: Path
    output_dir: Path
    is_base: bool
    site_id: Optional[str]


def resolve_training_input(config: DeviceConfig, filename: str) -> Path:
    candidate = config.training_dir / filename
    if candidate.is_file():
        return candidate.resolve()
    return candidate.resolve()


def resolve_orders_path(
    config: DeviceConfig,
    orders: Optional[Path],
    site_id: Optional[str],
    is_base: bool,
) -> Path:
    if orders is not None:
        return orders.resolve()

    search_dirs: List[Path] = []
    if site_id:
        site_dir = SITES_DIR / site_id
        search_dirs.extend([site_dir / "workspace", site_dir / "training"])
    if is_base:
        search_dirs.append(config.training_dir)

    for directory in search_dirs:
        for filename in ORDER_SESS_FILENAMES:
            candidate = directory / filename
            if candidate.is_file():
                return candidate.resolve()

    locations = ", ".join(str(path) for path in search_dirs) or str(config.training_dir)
    print("[ERROR] Не найден список сессий с заказами.")
    print("Укажите --orders или положите unique_order_sess_a.* в один из каталогов:")
    print(f"  {locations}")
    sys.exit(1)


def resolve_features_path(
    config: DeviceConfig,
    features: Optional[Path],
    site_id: Optional[str],
    is_base: bool,
) -> Path:
    if features is not None:
        return features.resolve()

    if site_id:
        return (
            SITES_DIR / site_id / "workspace" / config.default_features_name
        ).resolve()

    if is_base:
        return resolve_training_input(config, config.default_features_name)

    print("[ERROR] Не указан --features.")
    sys.exit(1)


def resolve_output_dir(
    config: DeviceConfig,
    *,
    site_id: Optional[str],
    output_dir: Optional[Path],
    target: Optional[str],
) -> Tuple[Optional[Path], bool]:
    if output_dir is not None:
        resolved = output_dir.resolve()
        is_base = resolved == config.models_dir.resolve()
        return resolved, is_base

    if site_id:
        return (SITES_DIR / site_id / "models" / config.device).resolve(), False

    if target == "base":
        return config.models_dir.resolve(), True

    return None, False


def resolve_training_paths(
    device: str,
    *,
    site_id: Optional[str] = None,
    output_dir: Optional[Path] = None,
    features: Optional[Path] = None,
    orders: Optional[Path] = None,
    target: Optional[str] = None,
) -> TrainingPaths:
    config = DEVICE_CONFIGS[device]
    resolved_output, is_base = resolve_output_dir(
        config,
        site_id=site_id,
        output_dir=output_dir,
        target=target,
    )
    if resolved_output is None:
        print("[ERROR] Укажите --site-id, --output-dir или --target base.")
        print("Примеры:")
        print(
            f"  uv run python scripts/train_site_models.py 1_200\n"
            f"  uv run python scripts/train_base_models.py --device {device} --force"
        )
        sys.exit(1)

    features_path = resolve_features_path(config, features, site_id, is_base)
    orders_path = resolve_orders_path(config, orders, site_id, is_base)
    return TrainingPaths(
        features_path=features_path,
        orders_path=orders_path,
        output_dir=resolved_output,
        is_base=is_base,
        site_id=site_id,
    )
