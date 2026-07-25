"""Backward-compatible re-exports. Prefer: from model_training import ..."""

from model_training.cli import build_parser, main, main_unified
from model_training.config import DEVICE_CONFIGS, ORDER_SESS_FILENAMES
from model_training.paths import TrainingPaths, resolve_training_paths
from model_training.train import train_order_cluster

__all__ = [
    "DEVICE_CONFIGS",
    "ORDER_SESS_FILENAMES",
    "TrainingPaths",
    "build_parser",
    "main",
    "main_unified",
    "resolve_training_paths",
    "train_order_cluster",
]
