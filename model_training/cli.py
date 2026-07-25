"""CLI for order cluster model training."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

from model_training.config import DEVICE_CONFIGS
from model_training.paths import resolve_training_paths
from model_training.train import train_order_cluster


def add_training_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--site-id", help="Сайт для custom-модели, например 1_200")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Каталог для order_umap_model.joblib и order_cluster_centroid.json",
    )
    parser.add_argument("--features", type=Path, help="CSV нормализованных признаков")
    parser.add_argument(
        "--orders",
        type=Path,
        help="Файл session_id с заказами (.pkl / .json / .txt)",
    )
    parser.add_argument(
        "--target",
        choices=["base", "custom"],
        help="base — mobile/ или desktop/; custom — sites/{site_id}/models/",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Разрешить перезапись существующих базовых моделей",
    )
    parser.add_argument(
        "--no-copy-scaler",
        action="store_true",
        help="Не копировать базовый scaler в output-dir (только UMAP + centroid)",
    )


def build_parser(device: str) -> argparse.ArgumentParser:
    config = DEVICE_CONFIGS[device]
    parser = argparse.ArgumentParser(
        description=(
            f"Обучение UMAP и центроида кластера заказов ({device}). "
            "Custom: --site-id. Base: --target base --force. "
            f"Данные base: {config.training_dir}/ → {config.models_dir}/"
        )
    )
    add_training_arguments(parser)
    return parser


def _run_from_args(device: str, args: argparse.Namespace) -> None:
    if args.site_id and args.target == "base":
        print("[ERROR] Нельзя одновременно указывать --site-id и --target base.")
        sys.exit(1)

    paths = resolve_training_paths(
        device,
        site_id=args.site_id,
        output_dir=args.output_dir,
        features=args.features,
        orders=args.orders,
        target=args.target,
    )
    train_order_cluster(
        device,
        paths,
        force=args.force,
        copy_scaler=not args.no_copy_scaler,
    )


def main(device: str, argv: Optional[list[str]] = None) -> None:
    if device not in DEVICE_CONFIGS:
        print(f"[ERROR] Неизвестный device: {device}")
        sys.exit(1)

    parser = build_parser(device)
    args = parser.parse_args(argv)
    _run_from_args(device, args)


def main_unified(argv: Optional[list[str]] = None) -> None:
    """Entry point: train_order_cluster.py mobile|desktop [options]."""
    parser = argparse.ArgumentParser(
        description="Обучение UMAP и центроида (mobile или desktop)."
    )
    subparsers = parser.add_subparsers(dest="device", required=True)

    for device in sorted(DEVICE_CONFIGS):
        config = DEVICE_CONFIGS[device]
        sub = subparsers.add_parser(
            device,
            help=f"Обучить {device}-модель (данные base: {config.training_dir}/)",
        )
        add_training_arguments(sub)

    args = parser.parse_args(argv)
    _run_from_args(args.device, args)
