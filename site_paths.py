"""Build per-site environment for the prediction pipeline."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict

from site_registry import PROJECT_ROOT, SiteConfig, ensure_site_layout

CPU_THREAD_LIMITS = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}


def build_site_env(site: SiteConfig) -> Dict[str, str]:
    ensure_site_layout(site)

    workspace = site.site_dir / "workspace"
    checkpoints = site.site_dir / "checkpoints"
    outputs = site.site_dir / "outputs"
    mobile_out = outputs / "mobile"
    desktop_out = outputs / "desktop"
    history_dir = outputs / "predictions_history"

    max_tmv = os.environ.get("MAX_TMV_RECORDS", "10")
    max_mmv = os.environ.get("MAX_MMV_RECORDS", "5")

    split_mobile = workspace / "merged.cloud.mobile.ndjson"
    split_desktop = workspace / "merged.cloud.desktop.ndjson"

    tmv_stats = workspace / f"mobile_tmv_clk_statistics_max{max_tmv}.json"
    mmv_stats = workspace / "desktop_mmv_clk_statistics.json"

    mobile_features = workspace / "tmv_session_features_enhanced_mean_only.csv"
    mobile_features_norm = workspace / "tmv_session_features_enhanced_mean_only_normalized.csv"
    mobile_summary = workspace / "tmv_bot_detection_summary.json"

    desktop_features = workspace / "mmv_session_features_enhanced_mean_only.csv"
    desktop_features_norm = workspace / "mmv_session_features_enhanced_mean_only_normalized.csv"
    desktop_summary = workspace / "mmv_bot_detection_summary.json"

    env: Dict[str, str] = {
        **CPU_THREAD_LIMITS,
        "SITE_ID": site.site_id,
        "SITE_WORK_DIR": str(workspace),
        "JSON_PATH": str(site.json_path),
        "OUT_LOG_PATH": str(site.out_log_path),
        "PREDICTIONS_HISTORY_DIR": str(history_dir),
        "SPLIT_CHECKPOINT_FILE": str(checkpoints / "split_records_checkpoint.json"),
        "OUTPUT_MOBILE": str(split_mobile),
        "OUTPUT_DESKTOP": str(split_desktop),
        "COMBINED_PREDICT_RESULTS": str(outputs / "predict_results.csv"),
        "MOBILE_PREDICT_RESULTS": str(mobile_out / "predict_results.csv"),
        "DESKTOP_PREDICT_RESULTS": str(desktop_out / "predict_results.csv"),
        "MOBILE_OUT_LOG_PATH": str(mobile_out / f"{site.site_id}.out.log"),
        "DESKTOP_OUT_LOG_PATH": str(desktop_out / f"{site.site_id}.out.log"),
        "MOBILE_OUT_DELTA_FILE": str(mobile_out / ".out_delta.csv"),
        "DESKTOP_OUT_DELTA_FILE": str(desktop_out / ".out_delta.csv"),
        "TMV_CHECKPOINT_FILE": str(checkpoints / "tmv_process_checkpoint.json"),
        "MMV_CHECKPOINT_FILE": str(checkpoints / "mmv_process_checkpoint.json"),
        "TMV_STATISTICS_JSON": str(tmv_stats),
        "MMV_STATISTICS_JSON": str(mmv_stats),
        "MOBILE_FEATURES_PATH": str(mobile_features),
        "MOBILE_NORMALIZED_FEATURES_PATH": str(mobile_features_norm),
        "MOBILE_BOT_SUMMARY_PATH": str(mobile_summary),
        "DESKTOP_FEATURES_PATH": str(desktop_features),
        "DESKTOP_NORMALIZED_FEATURES_PATH": str(desktop_features_norm),
        "DESKTOP_BOT_SUMMARY_PATH": str(desktop_summary),
        "MOBILE_UMAP_MODEL_PATH": str(site.mobile_models.umap),
        "MOBILE_CENTROID_PATH": str(site.mobile_models.centroid),
        "MOBILE_SCALER_PATH": str(site.mobile_models.scaler),
        "DESKTOP_UMAP_MODEL_PATH": str(site.desktop_models.umap),
        "DESKTOP_CENTROID_PATH": str(site.desktop_models.centroid),
        "DESKTOP_SCALER_PATH": str(site.desktop_models.scaler),
        "MAX_TMV_RECORDS": max_tmv,
        "MAX_MMV_RECORDS": max_mmv,
    }

    if site.mobile_models.source == "custom":
        env["MOBILE_MODEL_SOURCE"] = "custom"
    if site.desktop_models.source == "custom":
        env["DESKTOP_MODEL_SOURCE"] = "custom"

    return env
