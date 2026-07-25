import os
import sys
import subprocess
import shutil
import time
import uuid
from datetime import timedelta

# Все относительные пути (входные NDJSON, mobile_tmv_clk_statistics_*.json) — от каталога проекта
_ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(_ROOT)


def run_step(name, cmd, env=None):
    print(f"\n=== {name} ===")
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            env=env,
        )
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr)
        print(f"[OK] {name}")
    except subprocess.CalledProcessError as e:
        print(f"[FAIL] {name}")
        print(e.stdout)
        print(e.stderr)
        sys.exit(1)


def build_pipeline_env():
    env = os.environ.copy()
    run_id = uuid.uuid4().hex
    env["PIPELINE_RUN_ID"] = run_id
    env["CHANGED_SESSIONS_FILE"] = os.path.join(
        _ROOT, f".changed_sessions_{run_id}.json"
    )
    out_delta = os.environ.get("OUT_DELTA_FILE")
    env["OUT_DELTA_FILE"] = out_delta or os.path.join(_ROOT, ".out_delta.csv")
    return env


def cleanup_pipeline_env(env):
    changed_sessions_path = env.get("CHANGED_SESSIONS_FILE")
    if changed_sessions_path and os.path.exists(changed_sessions_path):
        os.unlink(changed_sessions_path)


def run_external_steps(env):
    """Запуск внешних скриптов обработки логов и построения признаков."""
    if env.get("USE_PRE_SPLIT_RECORDS") == "1":
        print("\n=== Разделение записей по устройству (split_records.js) ===")
        print("Используем готовые результаты split_records из общего orchestrator.")
        print("[OK] Разделение записей по устройству (split_records.js)")
    else:
        run_step(
            "Разделение записей по устройству (split_records.js)",
            ["node", "../split_records.js"],
            env=env,
        )

    max_tmv_records = os.environ.get("MAX_TMV_RECORDS", "10")
    env["MODE"] = "incremental"
    env["INPUT_FILE"] = env.get("OUTPUT_MOBILE", os.environ.get("OUTPUT_MOBILE", "merged.cloud.mobile.ndjson"))
    env["MAX_TMV_RECORDS"] = max_tmv_records
    if "CHANGED_SESSIONS_FILE" not in env:
        env["CHANGED_SESSIONS_FILE"] = "changed_sessions.json"
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
        "Обработка логов TMV/CLK/SCL (process_tmv_clk_enhanced.js)",
        ["node", "src/process_tmv_clk_enhanced.js"],
        env=env,
    )

    tmv_stats = env.get(
        "TMV_STATISTICS_JSON",
        os.environ.get(
            "TMV_STATISTICS_JSON",
            f"mobile_tmv_clk_statistics_max{max_tmv_records}.json",
        ),
    )
    run_step(
        "Построение признаков сессий (preprocessor_tmv_enhanced_fixed.py)",
        [sys.executable, "src/preprocessor_tmv_enhanced_fixed.py", tmv_stats],
        env=env,
    )


def run_order_prediction(env=None):
    run_step(
        "UMAP и прогноз по расстоянию до кластера заказов (predict_tmv_order.py)",
        [sys.executable, "src/predict_tmv_order.py"],
        env=env,
    )


def prepare_for_postprocessor():
    target_file = os.environ.get("PREDICT_RESULTS_PATH", "predict_results.csv")
    old_file = os.environ.get(
        "PREDICT_RESULTS_OLD_PATH",
        "predict_results_old.csv",
    )

    if os.path.exists(target_file):
        print(f"Сохраняем предыдущий {target_file} в {old_file}...")
        shutil.copy2(target_file, old_file)
        print(f"[OK] Предыдущие результаты сохранены")


def run_postprocessor(env=None):
    run_step(
        "Постпроцессинг результатов (postprocessor.py)",
        [sys.executable, "src/postprocessor.py"],
        env=env,
    )


if __name__ == "__main__":
    print("Запуск mobile-пайплайна обработки логов с UMAP и прогнозом по кластеру заказов...")
    pipeline_start = time.perf_counter()
    pipeline_env = build_pipeline_env()
    try:
        run_external_steps(pipeline_env)
        prepare_for_postprocessor()
        run_order_prediction(pipeline_env)
        run_postprocessor(pipeline_env)
        elapsed = time.perf_counter() - pipeline_start
        print("\nПайплайн успешно завершён!")
        print(
            f"Время работы пайплайна: {timedelta(seconds=elapsed)} "
            f"({elapsed:.2f} с)"
        )
    finally:
        cleanup_pipeline_env(pipeline_env)
