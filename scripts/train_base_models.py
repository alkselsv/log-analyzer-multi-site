"""Train base (shared) mobile and/or desktop models."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model_training import DEVICE_CONFIGS, resolve_training_paths, train_order_cluster  # noqa: E402


def train_base_device(device: str, *, force: bool, features: Path | None, orders: Path | None) -> None:
    config = DEVICE_CONFIGS[device]
    config.training_dir.mkdir(parents=True, exist_ok=True)

    paths = resolve_training_paths(
        device,
        features=features,
        orders=orders,
        target="base",
    )
    train_order_cluster(device, paths, force=force)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Переобучить базовую модель (UMAP + centroid) для mobile и/или desktop. "
            "Обучающие данные берутся из mobile/training/ и desktop/training/."
        )
    )
    parser.add_argument(
        "--device",
        choices=["mobile", "desktop", "both"],
        default="both",
        help="Какой device обучить (по умолчанию: both)",
    )
    parser.add_argument(
        "--features",
        type=Path,
        help="CSV признаков (иначе — из {device}/training/)",
    )
    parser.add_argument(
        "--orders",
        type=Path,
        help="Файл session_id с заказами (иначе — из {device}/training/)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Перезаписать существующие базовые модели",
    )
    args = parser.parse_args()

    devices = ["mobile", "desktop"] if args.device == "both" else [args.device]
    for device in devices:
        print(f"\n{'=' * 60}")
        print(f"Базовая модель: {device}")
        print(f"{'=' * 60}")
        train_base_device(
            device,
            force=args.force,
            features=args.features,
            orders=args.orders,
        )


if __name__ == "__main__":
    main()
