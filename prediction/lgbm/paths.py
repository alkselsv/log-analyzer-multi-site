"""Optional resolution of buyer LGBM + transformer artifacts (custom → base)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from prediction.lgbm.artifact import ARTIFACT_FILENAME, TRANSFORMER_FILENAME


@dataclass
class BuyerLgbmPaths:
    model_dir: Path
    artifact: Path
    transformer: Path
    source: str  # "custom" | "base" | "env" | "missing"


def _complete(model_dir: Path) -> bool:
    return (model_dir / ARTIFACT_FILENAME).is_file() and (
        model_dir / TRANSFORMER_FILENAME
    ).is_file()


def resolve_buyer_lgbm_paths(
    *,
    project_root: Path,
    site_dir: Optional[Path],
    device: str,
    artifact_env: Optional[str] = None,
    transformer_env: Optional[str] = None,
) -> BuyerLgbmPaths:
    """Resolve optional LGBM artifacts. Never raises if missing — source='missing'."""
    if artifact_env and transformer_env:
        artifact = Path(artifact_env)
        transformer = Path(transformer_env)
        if artifact.is_file() and transformer.is_file():
            return BuyerLgbmPaths(
                model_dir=artifact.parent,
                artifact=artifact,
                transformer=transformer,
                source="env",
            )

    candidates: list[tuple[Path, str]] = []
    if site_dir is not None:
        candidates.append((site_dir / "models" / device, "custom"))
    candidates.append((project_root / "models" / device, "base"))

    for model_dir, source in candidates:
        if _complete(model_dir):
            return BuyerLgbmPaths(
                model_dir=model_dir,
                artifact=model_dir / ARTIFACT_FILENAME,
                transformer=model_dir / TRANSFORMER_FILENAME,
                source=source,
            )

    fallback_dir = (
        (site_dir / "models" / device)
        if site_dir is not None
        else (project_root / "models" / device)
    )
    return BuyerLgbmPaths(
        model_dir=fallback_dir,
        artifact=fallback_dir / ARTIFACT_FILENAME,
        transformer=fallback_dir / TRANSFORMER_FILENAME,
        source="missing",
    )
