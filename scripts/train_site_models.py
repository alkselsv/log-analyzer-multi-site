"""Train custom mobile + desktop models for a single site."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from site_registry import (  # noqa: E402
    DESKTOP_MODEL_FILES,
    MOBILE_MODEL_FILES,
    SITES_DIR,
    get_site,
)
from model_training import (  # noqa: E402
    DEVICE_CONFIGS,
    ORDER_SESS_FILENAMES,
    resolve_training_paths,
    train_order_cluster,
)


def resolve_orders_for_site(site_id: str, orders: Path | None) -> Path | None:
    if orders is not None:
        return orders.resolve()

    for directory in (SITES_DIR / site_id / "workspace", SITES_DIR / site_id / "training"):
        for filename in ORDER_SESS_FILENAMES:
            candidate = directory / filename
            if candidate.is_file():
                return candidate.resolve()
    return None


def verify_model_set(model_dir: Path, filenames: tuple[str, ...]) -> bool:
    return all((model_dir / name).is_file() for name in filenames)


def train_site_device(
    site_id: str,
    device: str,
    *,
    orders: Path | None,
    force: bool,
) -> Path:
    shared_orders = resolve_orders_for_site(site_id, orders)
    if shared_orders is None:
        print(
            f"[ERROR] Не найден unique_order_sess_a.* для сайта {site_id}.\n"
            f"Положите файл в sites/{site_id}/workspace/ или sites/{site_id}/training/, "
            "либо передайте --orders."
        )
        sys.exit(1)

    paths = resolve_training_paths(
        device,
        site_id=site_id,
        orders=shared_orders,
    )
    return train_order_cluster(device, paths, force=force)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Обучить custom-модели mobile и desktop для одного сайта. "
            "Результат сохраняется в sites/{site_id}/models/; базовые модели не трогаются."
        )
    )
    parser.add_argument("site_id", help="Идентификатор сайта, например 1_200")
    parser.add_argument(
        "--orders",
        type=Path,
        help="Файл session_id с заказами (.pkl / .json / .txt)",
    )
    parser.add_argument(
        "--mobile-only",
        action="store_true",
        help="Обучить только mobile-модель",
    )
    parser.add_argument(
        "--desktop-only",
        action="store_true",
        help="Обучить только desktop-модель",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Перезаписать существующие custom-модели сайта",
    )
    args = parser.parse_args()

    if args.mobile_only and args.desktop_only:
        print("[ERROR] Нельзя одновременно указывать --mobile-only и --desktop-only.")
        sys.exit(1)

    workspace = SITES_DIR / args.site_id / "workspace"
    if not workspace.is_dir():
        print(
            f"[ERROR] Нет workspace для сайта {args.site_id}: {workspace}\n"
            f"Сначала выполните: uv run python run_site_pipeline.py {args.site_id} --force"
        )
        sys.exit(1)

    devices = []
    if args.mobile_only:
        devices = ["mobile"]
    elif args.desktop_only:
        devices = ["desktop"]
    else:
        devices = ["mobile", "desktop"]

    missing_features = []
    for device in devices:
        config = DEVICE_CONFIGS[device]
        features_path = workspace / config.default_features_name
        if not features_path.is_file():
            missing_features.append(str(features_path))

    if missing_features:
        print("[ERROR] Не найдены CSV признаков в workspace:")
        for path in missing_features:
            print(f"  - {path}")
        print(
            f"\nСначала выполните: uv run python run_site_pipeline.py {args.site_id} --force"
        )
        sys.exit(1)

    for device in devices:
        print(f"\n{'=' * 60}")
        print(f"Сайт {args.site_id}: обучение {device}")
        print(f"{'=' * 60}")
        train_site_device(
            args.site_id,
            device,
            orders=args.orders,
            force=args.force,
        )

    site = get_site(args.site_id)
    if site is None:
        print(
            f"\n[WARN] Сайт {args.site_id} не найден в discover_sites() "
            f"(нет лога в MLOG_DIR или неполные модели)."
        )
        return

    print("\n=== Проверка ===")
    print(f"mobile:  {site.mobile_models.source} ({site.mobile_models.model_dir})")
    print(f"desktop: {site.desktop_models.source} ({site.desktop_models.model_dir})")

    mobile_ok = verify_model_set(site.mobile_models.model_dir, MOBILE_MODEL_FILES)
    desktop_ok = verify_model_set(site.desktop_models.model_dir, DESKTOP_MODEL_FILES)
    if "mobile" in devices and not mobile_ok:
        print("[WARN] Набор mobile-модели неполный.")
    if "desktop" in devices and not desktop_ok:
        print("[WARN] Набор desktop-модели неполный.")


if __name__ == "__main__":
    main()
