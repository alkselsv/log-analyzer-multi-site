"""Run pipelines for multiple sites with bounded parallelism."""

from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv

from run_site_pipeline import run_site_pipeline
from site_registry import discover_sites, has_log_changed

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")


def default_max_workers() -> int:
    configured = os.environ.get("MAX_PARALLEL_SITES", "").strip()
    if configured.isdigit():
        return max(1, int(configured))
    return 1


def select_sites(rescan_only: bool, force: bool):
    sites = discover_sites()
    if force:
        return sites
    if rescan_only:
        return [site for site in sites if has_log_changed(site)]
    return [site for site in sites if has_log_changed(site)]


def run_all_sites(rescan_only: bool = False, force: bool = False) -> int:
    sites = select_sites(rescan_only=rescan_only, force=force)
    if not sites:
        print("Нет сайтов для обработки")
        return 0

    max_workers = min(default_max_workers(), len(sites))
    print(f"Сайтов к обработке: {len(sites)}, workers={max_workers}")

    failures = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(run_site_pipeline, site.site_id, force): site.site_id
            for site in sites
        }
        for future in as_completed(futures):
            site_id = futures[future]
            try:
                status = future.result()
                print(f"{site_id}: {status}")
            except Exception as exc:
                failures += 1
                print(f"{site_id}: error: {exc}", file=sys.stderr)

    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description="Параллельный запуск пайплайнов по сайтам")
    parser.add_argument(
        "--rescan-only",
        action="store_true",
        help="Обработать только сайты с изменившимся логом",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Обработать все обнаруженные сайты",
    )
    args = parser.parse_args()
    failures = run_all_sites(rescan_only=args.rescan_only, force=args.force)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
