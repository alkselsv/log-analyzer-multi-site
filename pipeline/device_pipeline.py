"""Shared mobile/desktop pipeline: process → features → dual predict → postprocess."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
import uuid
from datetime import timedelta
from pathlib import Path

from prediction.runner import run_predictions
from pipeline.postprocess import run_postprocess

ROOT = Path(__file__).resolve().parents[1]


def run_step(name: str, cmd: list[str], *, cwd: Path, env: dict) -> None:
    print(f"\n=== {name} ===")
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            cwd=str(cwd),
            env=env,
        )
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr)
        print(f"[OK] {name}")
    except subprocess.CalledProcessError as exc:
        print(f"[FAIL] {name}")
        print(exc.stdout)
        print(exc.stderr)
        sys.exit(1)


def resolve_work_dir(device: str) -> Path:
    explicit = os.environ.get("SITE_WORK_DIR")
    if explicit:
        work_dir = Path(explicit)
    else:
        work_dir = ROOT / "outputs" / device
    work_dir.mkdir(parents=True, exist_ok=True)
    return work_dir


def build_pipeline_env(work_dir: Path) -> dict:
    env = os.environ.copy()
    run_id = uuid.uuid4().hex
    env["PIPELINE_RUN_ID"] = run_id
    env["CHANGED_SESSIONS_FILE"] = str(work_dir / f".changed_sessions_{run_id}.json")
    out_delta = os.environ.get("OUT_DELTA_FILE")
    env["OUT_DELTA_FILE"] = out_delta or str(work_dir / ".out_delta.csv")
    return env


def cleanup_pipeline_env(env: dict) -> None:
    changed_sessions_path = env.get("CHANGED_SESSIONS_FILE")
    if changed_sessions_path and os.path.exists(changed_sessions_path):
        os.unlink(changed_sessions_path)


def prepare_for_postprocessor(env: dict) -> None:
    target_file = env.get("PREDICT_RESULTS_PATH", "predict_results.csv")
    old_file = env.get("PREDICT_RESULTS_OLD_PATH", "predict_results_old.csv")
    if os.path.exists(target_file):
        print(f"Сохраняем предыдущий {target_file} в {old_file}...")
        shutil.copy2(target_file, old_file)
        print("[OK] Предыдущие результаты сохранены")


def run_mobile_external_steps(env: dict, work_dir: Path, features_dir: Path) -> None:
    if env.get("USE_PRE_SPLIT_RECORDS") == "1":
        print("\n=== Разделение записей по устройству (split_records.js) ===")
        print("Используем готовые результаты split_records из общего orchestrator.")
        print("[OK] Разделение записей по устройству (split_records.js)")
    else:
        run_step(
            "Разделение записей по устройству (split_records.js)",
            ["node", str(ROOT / "split_records.js")],
            cwd=work_dir,
            env=env,
        )

    max_tmv_records = os.environ.get("MAX_TMV_RECORDS", "10")
    env["MODE"] = "incremental"
    env["INPUT_FILE"] = env.get(
        "OUTPUT_MOBILE",
        os.environ.get("OUTPUT_MOBILE", "merged.cloud.mobile.ndjson"),
    )
    env["MAX_TMV_RECORDS"] = max_tmv_records
    if "TMV_CHECKPOINT_FILE" not in env:
        env["TMV_CHECKPOINT_FILE"] = "tmv_process_checkpoint.json"
    tmv_stats = env.get(
        "TMV_STATISTICS_JSON",
        os.environ.get(
            "TMV_STATISTICS_JSON",
            f"mobile_tmv_clk_statistics_max{max_tmv_records}.json",
        ),
    )
    env["TMV_STATISTICS_JSON"] = tmv_stats
    run_step(
        "Обработка логов TMV/CLK/SCL (features/mobile/process.js)",
        ["node", str(features_dir / "process.js")],
        cwd=work_dir,
        env=env,
    )
    run_step(
        "Построение признаков сессий (features/mobile/preprocessor.py)",
        [sys.executable, str(features_dir / "preprocessor.py"), tmv_stats],
        cwd=work_dir,
        env=env,
    )


def run_desktop_external_steps(env: dict, work_dir: Path, features_dir: Path) -> None:
    env["MODE"] = "incremental"
    if env.get("JSON_PATH") and not env.get("INPUT_FILE"):
        env["INPUT_FILE"] = env["JSON_PATH"]
    mmv_stats = os.environ.get("MMV_STATISTICS_JSON", "desktop_mmv_clk_statistics.json")
    env["OUTPUT_FILE"] = mmv_stats
    env["MMV_STATISTICS_JSON"] = mmv_stats
    env["MMV_CHECKPOINT_FILE"] = os.environ.get(
        "MMV_CHECKPOINT_FILE", "mmv_process_checkpoint.json"
    )
    run_step(
        "Обработка логов MMV/CLK/SCL (features/desktop/process.js)",
        ["node", str(features_dir / "process.js")],
        cwd=work_dir,
        env=env,
    )
    run_step(
        "Построение признаков сессий (features/desktop/preprocessor.py)",
        [sys.executable, str(features_dir / "preprocessor.py")],
        cwd=work_dir,
        env=env,
    )


def run_device_pipeline(device: str) -> None:
    device = device.lower()
    if device not in {"mobile", "desktop"}:
        raise ValueError(f"Unknown device: {device}")

    features_dir = ROOT / "features" / device
    work_dir = resolve_work_dir(device)
    os.chdir(work_dir)

    print(
        f"Запуск {device}-пайплайна: признаки → probability_bot_umap + probability_lgbm..."
    )
    pipeline_start = time.perf_counter()
    pipeline_env = build_pipeline_env(work_dir)
    pipeline_env["DEVICE_TYPE"] = device

    try:
        if device == "mobile":
            run_mobile_external_steps(pipeline_env, work_dir, features_dir)
        else:
            run_desktop_external_steps(pipeline_env, work_dir, features_dir)

        # Prediction/postprocess use env paths; ensure child env is current.
        os.environ.update(
            {
                key: value
                for key, value in pipeline_env.items()
                if isinstance(value, str)
            }
        )

        prepare_for_postprocessor(pipeline_env)
        print("\n=== Dual prediction (UMAP + LightGBM) ===")
        run_predictions(
            device,
            changed_sessions_file=pipeline_env.get("CHANGED_SESSIONS_FILE"),
        )
        print("\n=== Постпроцессинг ===")
        run_postprocess(device)

        elapsed = time.perf_counter() - pipeline_start
        print("\nПайплайн успешно завершён!")
        print(
            f"Время работы пайплайна: {timedelta(seconds=elapsed)} "
            f"({elapsed:.2f} с)"
        )
    finally:
        cleanup_pipeline_env(pipeline_env)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run mobile or desktop device pipeline")
    parser.add_argument(
        "--device",
        choices=["mobile", "desktop"],
        required=True,
        help="Device pipeline to run",
    )
    args = parser.parse_args(argv)
    run_device_pipeline(args.device)


if __name__ == "__main__":
    main()
