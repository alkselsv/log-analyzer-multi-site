"""Hybrid watcher: inotify debounce + startup/hourly rescan."""

from __future__ import annotations

import logging
import os
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Set

from dotenv import load_dotenv
from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from run_site_pipeline import run_site_pipeline
from site_registry import MLOG_DIR, discover_sites, has_log_changed, is_site_allowed

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

LOG_FILE = ROOT / "watcher.log"
DEBOUNCE_SECONDS = int(os.environ.get("DEBOUNCE_SECONDS", "45"))
MAX_DEBOUNCE_SECONDS = int(os.environ.get("MAX_DEBOUNCE_SECONDS", "120"))
RESCAN_INTERVAL_SECONDS = int(os.environ.get("RESCAN_INTERVAL_SECONDS", "3600"))
SITE_FILE_RE = re.compile(r"^1_\d+\.json$")


@dataclass
class _PendingRun:
    timer: threading.Timer
    first_at: float

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("site_watcher")


class SiteScheduler:
    def __init__(self) -> None:
        self._pending: Dict[str, _PendingRun] = {}
        self._dirty: Set[str] = set()
        self._running: Set[str] = set()
        self._lock = threading.Lock()

    def schedule(self, site_id: str) -> None:
        run_now = False
        with self._lock:
            if site_id in self._running:
                self._dirty.add(site_id)
                logger.info("site %s dirty while running", site_id)
                return

            pending = self._pending.get(site_id)
            if pending is not None:
                elapsed = time.monotonic() - pending.first_at
                if elapsed >= MAX_DEBOUNCE_SECONDS:
                    pending.timer.cancel()
                    del self._pending[site_id]
                    logger.info(
                        "site %s max debounce (%.0fs), running now",
                        site_id,
                        elapsed,
                    )
                    run_now = True
                return

            timer = threading.Timer(DEBOUNCE_SECONDS, self._on_debounce, args=(site_id,))
            self._pending[site_id] = _PendingRun(timer=timer, first_at=time.monotonic())
            timer.daemon = True
            timer.start()
            logger.info(
                "site %s scheduled in %ss (max %ss)",
                site_id,
                DEBOUNCE_SECONDS,
                MAX_DEBOUNCE_SECONDS,
            )

        if run_now:
            threading.Thread(target=self._run_site, args=(site_id,), daemon=True).start()

    def _on_debounce(self, site_id: str) -> None:
        with self._lock:
            self._pending.pop(site_id, None)
        self._run_site(site_id)

    def _run_site(self, site_id: str) -> None:
        with self._lock:
            self._pending.pop(site_id, None)
            if site_id in self._running:
                return
            self._running.add(site_id)

        try:
            status = run_site_pipeline(site_id, force=False)
            logger.info("site %s finished: %s", site_id, status)
        except Exception:
            logger.exception("site %s failed", site_id)
        finally:
            with self._lock:
                self._running.discard(site_id)
                needs_rerun = site_id in self._dirty
                self._dirty.discard(site_id)
            if needs_rerun:
                logger.info("site %s rerun after dirty flag", site_id)
                self.schedule(site_id)

    def scan_changed(self) -> None:
        for site in discover_sites():
            if has_log_changed(site):
                self.schedule(site.site_id)


class MlogHandler(FileSystemEventHandler):
    def __init__(self, scheduler: SiteScheduler) -> None:
        self.scheduler = scheduler

    def on_created(self, event: FileSystemEvent) -> None:
        self._handle(event)

    def on_modified(self, event: FileSystemEvent) -> None:
        self._handle(event)

    def on_moved(self, event: FileSystemEvent) -> None:
        self._handle(event)

    def _handle(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            return
        name = Path(event.src_path).name
        if not SITE_FILE_RE.match(name):
            return
        site_id = Path(name).stem
        if not is_site_allowed(site_id):
            return
        self.scheduler.schedule(site_id)


def rescan_loop(scheduler: SiteScheduler) -> None:
    while True:
        time.sleep(RESCAN_INTERVAL_SECONDS)
        logger.info("hourly rescan")
        scheduler.scan_changed()


def main() -> None:
    if not MLOG_DIR.is_dir():
        raise FileNotFoundError(f"MLOG_DIR не найден: {MLOG_DIR}")

    scheduler = SiteScheduler()
    logger.info("startup scan")
    scheduler.scan_changed()

    observer = Observer()
    observer.schedule(MlogHandler(scheduler), str(MLOG_DIR), recursive=False)
    observer.start()

    rescan_thread = threading.Thread(target=rescan_loop, args=(scheduler,), daemon=True)
    rescan_thread.start()

    logger.info("watching %s", MLOG_DIR)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("stopping")
    finally:
        observer.stop()
        observer.join()


if __name__ == "__main__":
    main()
