import os
import sys
import subprocess
import shutil
import time
import uuid

# Относительные пути к входным данным и артефактам — от каталога проекта
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
        "Обработка логов MMV/CLK/SCL (process_mmv_clk_enhanced_new.js)",
        ["node", "src/process_mmv_clk_enhanced_new.js"],
        env=env,
    )

    run_step(
        "Построение признаков сессий (preprocessor_mmv_enhanced_fixed_new.py)",
        [sys.executable, "src/preprocessor_mmv_enhanced_fixed_new.py"],
        env=env,
    )


def run_order_prediction(env=None):
    run_step(
        "UMAP и прогноз по расстоянию до кластера заказов (predict_mmv_order.py)",
        [sys.executable, "src/predict_mmv_order.py"],
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
    print("Запуск desktop-пайплайна обработки логов с UMAP и прогнозом по кластеру заказов...")
    pipeline_start = time.perf_counter()
    pipeline_env = build_pipeline_env()
    try:
        run_external_steps(pipeline_env)
        prepare_for_postprocessor()
        run_order_prediction(pipeline_env)
        run_postprocessor(pipeline_env)
        elapsed = time.perf_counter() - pipeline_start
        print("\nПайплайн успешно завершён!")
        print(f"Время работы пайплайна: {elapsed:.2f} с ({elapsed / 60.0:.2f} мин)")
    finally:
        cleanup_pipeline_env(pipeline_env)
