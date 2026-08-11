"""Production buyer_lgbm.pkl artifact format."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

import lightgbm as lgb
import numpy as np

ARTIFACT_FILENAME = "buyer_lgbm.pkl"
TRANSFORMER_FILENAME = "transformer_autoencoder.pt"

ARTIFACT_VERSION = 1


class BoosterProbabilityModel:
    """Thin wrapper so score code can call predict_proba like sklearn."""

    def __init__(self, booster: lgb.Booster):
        self.booster = booster

    def predict_proba(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        pos = self.booster.predict(X)
        pos = np.asarray(pos, dtype=np.float64).reshape(-1)
        neg = 1.0 - pos
        return np.column_stack([neg, pos])


def save_buyer_lgbm_artifact(path: Path | str, artifact: Dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(artifact)
    payload.setdefault("version", ARTIFACT_VERSION)
    with path.open("wb") as handle:
        pickle.dump(payload, handle)
    return path


def _resolve_model(artifact: Dict[str, Any]):
    if artifact.get("model") is not None:
        return artifact["model"]
    booster_str = artifact.get("booster_str")
    if not booster_str:
        raise ValueError("В артефакте нет model и booster_str")
    booster = lgb.Booster(model_str=booster_str)
    return BoosterProbabilityModel(booster)


def load_buyer_lgbm_artifact(path: Path | str) -> Dict[str, Any]:
    with Path(path).open("rb") as handle:
        artifact = pickle.load(handle)

    required = ("feature_columns", "nobot_feature_columns", "feature_scaler")
    missing = [key for key in required if key not in artifact]
    if missing:
        raise ValueError(f"В '{path}' отсутствуют поля: {missing}")
    if artifact.get("model") is None and not artifact.get("booster_str"):
        raise ValueError(f"В '{path}' нет model/booster_str")

    artifact = dict(artifact)
    artifact["model"] = _resolve_model(artifact)
    return artifact


def build_artifact(
    *,
    device: str,
    feature_columns: List[str],
    nobot_feature_columns: List[str],
    feature_scaler: Any,
    embedding_dim: int = 64,
    metrics: Optional[Dict[str, Any]] = None,
    best_threshold: Optional[float] = None,
    use_features: bool = True,
    model: Any = None,
    booster_str: Optional[str] = None,
) -> Dict[str, Any]:
    if model is None and not booster_str:
        raise ValueError("Нужен model или booster_str")
    payload: Dict[str, Any] = {
        "version": ARTIFACT_VERSION,
        "device": device,
        "use_features": use_features,
        "embedding_dim": int(embedding_dim),
        "feature_columns": list(feature_columns),
        "nobot_feature_columns": list(nobot_feature_columns),
        "feature_scaler": feature_scaler,
        "metrics": metrics or {},
        "best_threshold": best_threshold,
        "model": None,
        "booster_str": None,
    }
    if booster_str:
        payload["booster_str"] = booster_str
    elif hasattr(model, "booster_"):
        payload["booster_str"] = model.booster_.model_to_string()
    else:
        payload["model"] = model
    return payload
