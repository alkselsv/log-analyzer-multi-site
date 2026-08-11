"""Run prediction pipeline for a single site with flock and logging."""

from __future__ import annotations

import argparse
import fcntl
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from site_paths import build_site_env
from site_registry import (
    PROJECT_ROOT,
    get_site,
    has_log_changed,
    is_site_allowed,
    read_log_stat,
    save_trigger_state,
)

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")


def resolve_python() -> str:
    configured = os.environ.get("PYTHON", "").strip()
    if configured:
        return configured

    candidate = PROJECT_ROOT / ".venv" / "bin" / "python"
    if candidate.is_file():
        return str(candidate)

    return sys.executable


class SiteLockBusy(Exception):
    pass


def acquire_site_lock(lock_path: Path):
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("w", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        handle.close()
        raise SiteLockBusy(str(lock_path)) from exc
    handle.write(f"pid={os.getpid()}\n")
    handle.flush()
    return handle


def append_site_log(site_dir: Path, message: str) -> None:
    log_path = site_dir / "pipeline.log"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"[{timestamp}] {message}\n")


def run_site_pipeline(site_id: str, force: bool = False) -> str:
    if not is_site_allowed(site_id):
        raise ValueError(
            f"Сайт {site_id} не входит в ALLOWED_SITE_IDS: "
            f"{os.environ.get('ALLOWED_SITE_IDS', '')}"
        )

    site = get_site(site_id)
    if site is None:
        raise ValueError(f"Сайт не найден или модели недоступны: {site_id}")

    if not force and not has_log_changed(site):
        append_site_log(site.site_dir, "skip: лог не изменился")
        return "skipped_unchanged"

    lock_handle = None
    try:
        lock_handle = acquire_site_lock(site.site_dir / ".lock")
    except SiteLockBusy:
        append_site_log(site.site_dir, "skip: сайт уже обрабатывается")
        return "skipped_locked"

    try:
        env = os.environ.copy()
        env.update(build_site_env(site))
        if force:
            # Пересчёт всех сессий (не только delta по логу).
            env["FORCE_FULL_REBUILD"] = "1"
        append_site_log(
            site.site_dir,
            f"start: mobile_model={site.mobile_models.source} "
            f"desktop_model={site.desktop_models.source}"
            f"{' force_full_rebuild=1' if force else ''}",
        )

        python_bin = resolve_python()
        result = subprocess.run(
            [python_bin, str(PROJECT_ROOT / "run_pipeline.py")],
            cwd=PROJECT_ROOT,
            env=env,
            capture_output=True,
            text=True,
        )

        log_path = site.site_dir / "pipeline.log"
        if result.stdout:
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(result.stdout)
                if not result.stdout.endswith("\n"):
                    handle.write("\n")
        if result.stderr:
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(result.stderr)
                if not result.stderr.endswith("\n"):
                    handle.write("\n")

        if result.returncode != 0:
            append_site_log(site.site_dir, f"fail: exit code {result.returncode}")
            raise RuntimeError(
                f"Пайплайн {site_id} завершился с кодом {result.returncode}"
            )

        save_trigger_state(site.site_dir, read_log_stat(site.json_path))
        append_site_log(site.site_dir, "ok")
        return "ok"
    finally:
        if lock_handle is not None:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
            lock_handle.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Запуск пайплайна для одного сайта")
    parser.add_argument("site_id", help="Идентификатор сайта, например 1_466")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Запустить даже если лог не менялся; полный пересчёт всех сессий",
    )
    args = parser.parse_args()

    status = run_site_pipeline(args.site_id, force=args.force)
    print(status)


if __name__ == "__main__":
    main()
