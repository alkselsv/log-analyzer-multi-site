"""Device-specific paths and filenames for model training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from site_registry import PROJECT_ROOT, base_model_dir

UMAP_MODEL_NAME = "order_umap_model.joblib"
CENTROID_NAME = "order_cluster_centroid.json"
EXCLUDED_COLUMNS = ["session_id"]
HDBSCAN_MIN_SAMPLES = 10
HDBSCAN_MIN_CLUSTER_SIZE = 100
UMAP_RANDOM_STATE = 42
ORDER_SESS_FILENAMES = (
    "unique_order_sess_a.pkl",
    "unique_order_sess_a.json",
    "unique_order_sess_a.txt",
)


@dataclass(frozen=True)
class DeviceConfig:
    device: str
    device_dir: Path
    models_dir: Path
    training_dir: Path
    default_features_name: str
    scaler_name: str


DEVICE_CONFIGS = {
    "mobile": DeviceConfig(
        device="mobile",
        device_dir=PROJECT_ROOT / "mobile",
        models_dir=base_model_dir("mobile"),
        training_dir=PROJECT_ROOT / "mobile" / "training",
        default_features_name="tmv_session_features_enhanced_mean_only_normalized.csv",
        scaler_name="minmax_scaler_mobile.pkl",
    ),
    "desktop": DeviceConfig(
        device="desktop",
        device_dir=PROJECT_ROOT / "desktop",
        models_dir=base_model_dir("desktop"),
        training_dir=PROJECT_ROOT / "desktop" / "training",
        default_features_name="mmv_session_features_enhanced_mean_only_normalized.csv",
        scaler_name="minmax_scaler.pkl",
    ),
}
