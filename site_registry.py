"""Discovery of site logs in MLOG_DIR and model resolution."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional, Tuple

from dotenv import load_dotenv

from prediction.lgbm.paths import BuyerLgbmPaths, resolve_buyer_lgbm_paths

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

MLOG_DIR = Path(os.environ.get("MLOG_DIR", "/var/www/mlog"))
PROJECT_ROOT = ROOT
SITES_DIR = Path(os.environ.get("SITES_DIR", str(ROOT / "sites")))

SITE_LOG_PATTERN = re.compile(r"^1_\d+\.json$")


def _normalize_site_id(token: str) -> str:
    value = token.strip()
    if value.endswith(".json"):
        value = Path(value).stem
    return value


def get_allowed_site_ids() -> Optional[FrozenSet[str]]:
    """Return configured allowlist, or None when all sites are allowed."""
    raw = os.environ.get("ALLOWED_SITE_IDS", "").strip()
    if not raw:
        return None

    ids = {
        site_id
        for part in raw.split(",")
        if (site_id := _normalize_site_id(part))
    }
    return frozenset(ids) if ids else None


def is_site_allowed(site_id: str) -> bool:
    allowed = get_allowed_site_ids()
    if allowed is None:
        return True
    return site_id in allowed


MOBILE_MODEL_FILES = (
    "bot_umap_model.joblib",
    "bot_cluster_centroid.json",
    "minmax_scaler.pkl",
)
DESKTOP_MODEL_FILES = (
    "bot_umap_model.joblib",
    "bot_cluster_centroid.json",
    "minmax_scaler.pkl",
)


@dataclass
class DeviceModelPaths:
    model_dir: Path
    bot_umap: Path
    bot_centroid: Path
    scaler: Path
    source: str  # "custom" or "base"
    buyer_lgbm: Optional[BuyerLgbmPaths] = None


@dataclass
class SiteConfig:
    site_id: str
    json_path: Path
    out_log_path: Path
    site_dir: Path
    mobile_models: DeviceModelPaths
    desktop_models: DeviceModelPaths


@dataclass
class LogTriggerState:
    size: int
    mtime_ns: int
    inode: Optional[int]


def _model_set_complete(model_dir: Path, filenames: Tuple[str, ...]) -> bool:
    return all((model_dir / name).is_file() for name in filenames)


def base_model_dir(device: str) -> Path:
    return PROJECT_ROOT / "models" / device


def resolve_device_models(
    site_dir: Path,
    device: str,
    filenames: Tuple[str, ...],
    scaler_name: str,
) -> DeviceModelPaths:
    custom_dir = site_dir / "models" / device
    base_dir = base_model_dir(device)
    legacy_base_dir = PROJECT_ROOT / device

    if _model_set_complete(custom_dir, filenames):
        model_dir = custom_dir
        source = "custom"
    elif _model_set_complete(base_dir, filenames):
        model_dir = base_dir
        source = "base"
    elif _model_set_complete(legacy_base_dir, filenames):
        model_dir = legacy_base_dir
        source = "base"
    else:
        raise FileNotFoundError(
            f"Неполный набор моделей для {device}: нет полного custom в {custom_dir} "
            f"и нет полного base в {base_dir}"
        )

    buyer_lgbm = resolve_buyer_lgbm_paths(
        project_root=PROJECT_ROOT,
        site_dir=site_dir,
        device=device,
    )
    return DeviceModelPaths(
        model_dir=model_dir,
        bot_umap=model_dir / "bot_umap_model.joblib",
        bot_centroid=model_dir / "bot_cluster_centroid.json",
        scaler=model_dir / scaler_name,
        source=source,
        buyer_lgbm=buyer_lgbm,
    )


def discover_sites() -> List[SiteConfig]:
    if not MLOG_DIR.is_dir():
        raise FileNotFoundError(f"MLOG_DIR не найден: {MLOG_DIR}")

    configs: List[SiteConfig] = []
    for path in sorted(MLOG_DIR.glob("1_*.json")):
        if not SITE_LOG_PATTERN.match(path.name):
            continue

        site_id = path.stem
        if not is_site_allowed(site_id):
            continue

        site_dir = SITES_DIR / site_id
        try:
            mobile = resolve_device_models(
                site_dir, "mobile", MOBILE_MODEL_FILES, "minmax_scaler.pkl"
            )
            desktop = resolve_device_models(
                site_dir, "desktop", DESKTOP_MODEL_FILES, "minmax_scaler.pkl"
            )
        except FileNotFoundError:
            continue

        configs.append(
            SiteConfig(
                site_id=site_id,
                json_path=path,
                out_log_path=MLOG_DIR / f"{site_id}.out.log",
                site_dir=site_dir,
                mobile_models=mobile,
                desktop_models=desktop,
            )
        )

    return configs


def get_site(site_id: str) -> Optional[SiteConfig]:
    for site in discover_sites():
        if site.site_id == site_id:
            return site
    return None


def read_log_stat(path: Path) -> LogTriggerState:
    stat = path.stat()
    inode = getattr(stat, "st_ino", None)
    mtime_ns = getattr(stat, "st_mtime_ns", int(stat.st_mtime * 1_000_000_000))
    return LogTriggerState(size=stat.st_size, mtime_ns=mtime_ns, inode=inode)


def load_trigger_state(site_dir: Path) -> Optional[LogTriggerState]:
    state_path = site_dir / ".log_state.json"
    if not state_path.exists():
        return None
    with state_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return LogTriggerState(
        size=int(payload.get("size", 0)),
        mtime_ns=int(payload.get("mtime_ns", 0)),
        inode=payload.get("inode"),
    )


def save_trigger_state(site_dir: Path, state: LogTriggerState) -> None:
    site_dir.mkdir(parents=True, exist_ok=True)
    state_path = site_dir / ".log_state.json"
    payload = {
        "size": state.size,
        "mtime_ns": state.mtime_ns,
        "inode": state.inode,
    }
    with state_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def has_log_changed(site: SiteConfig) -> bool:
    if not site.json_path.exists():
        return False

    current = read_log_stat(site.json_path)
    previous = load_trigger_state(site.site_dir)
    if previous is None:
        return True

    return (
        current.size != previous.size
        or current.mtime_ns != previous.mtime_ns
        or current.inode != previous.inode
    )


def ensure_site_layout(site: SiteConfig) -> None:
    for sub in (
        "checkpoints",
        "workspace",
        "outputs",
        "outputs/mobile",
        "outputs/desktop",
        "outputs/predictions_history",
    ):
        (site.site_dir / sub).mkdir(parents=True, exist_ok=True)
