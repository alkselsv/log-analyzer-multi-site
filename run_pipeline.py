import csv
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set, Tuple

from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parent
MOBILE_DIR = ROOT / "mobile"
DESKTOP_DIR = ROOT / "desktop"

from out_log_daily import append_out_log, ensure_out_log_file

COMBINED_OUT_FIELDNAMES = [
    "date",
    "device_type",
    "session_id",
    "probability",
    "ip",
    "user_agent",
]
DEFAULT_SITE_ID = "1_466"
DEFAULT_JSON_PATH = Path("/var/www/mlog/1_466.json")


class PipelineConfig:
    def __init__(
        self,
        device_type: str,
        working_directory: Path,
        script_path: Path,
        predict_results_path: Path,
        out_log_path: Path,
        out_delta_path: Path,
        env_overrides: Dict[str, str],
    ) -> None:
        self.device_type = device_type
        self.working_directory = working_directory
        self.script_path = script_path
        self.predict_results_path = predict_results_path
        self.out_log_path = out_log_path
        self.out_delta_path = out_delta_path
        self.env_overrides = env_overrides


class SitePaths:
    def __init__(self, env: Dict[str, str]) -> None:
        site_id = env.get("SITE_ID", DEFAULT_SITE_ID)
        site_work_dir = Path(
            env.get("SITE_WORK_DIR", str(ROOT))
        )
        if not site_work_dir.is_absolute():
            site_work_dir = (ROOT / site_work_dir).resolve()

        self.site_id = site_id
        self.site_work_dir = site_work_dir
        self.json_path = Path(
            env.get("JSON_PATH", str(DEFAULT_JSON_PATH))
        )
        self.out_log_path = Path(
            env.get("OUT_LOG_PATH", str(ROOT / f"{site_id}.out.log"))
        )
        self.combined_predict_results = Path(
            env.get("COMBINED_PREDICT_RESULTS", str(ROOT / "predict_results.csv"))
        )
        self.combined_history_dir = Path(
            env.get("PREDICTIONS_HISTORY_DIR", str(ROOT / "predictions_history"))
        )
        self.split_output_mobile = Path(
            env.get("OUTPUT_MOBILE", str(ROOT / "merged.cloud.mobile.ndjson"))
        )
        self.split_output_desktop = Path(
            env.get("OUTPUT_DESKTOP", str(ROOT / "merged.cloud.desktop.ndjson"))
        )
        self.split_checkpoint = Path(
            env.get(
                "SPLIT_CHECKPOINT_FILE",
                str(ROOT / "split_records_checkpoint.json"),
            )
        )
        self.mobile_predict_results = Path(
            env.get(
                "MOBILE_PREDICT_RESULTS",
                str(MOBILE_DIR / "predict_results.csv"),
            )
        )
        self.desktop_predict_results = Path(
            env.get(
                "DESKTOP_PREDICT_RESULTS",
                str(DESKTOP_DIR / "predict_results.csv"),
            )
        )
        self.mobile_out_log = Path(
            env.get(
                "MOBILE_OUT_LOG_PATH",
                str(MOBILE_DIR / f"{site_id}.out.log"),
            )
        )
        self.desktop_out_log = Path(
            env.get(
                "DESKTOP_OUT_LOG_PATH",
                str(DESKTOP_DIR / f"{site_id}.out.log"),
            )
        )
        self.mobile_out_delta = Path(
            env.get(
                "MOBILE_OUT_DELTA_FILE",
                str(MOBILE_DIR / ".out_delta.csv"),
            )
        )
        self.desktop_out_delta = Path(
            env.get(
                "DESKTOP_OUT_DELTA_FILE",
                str(DESKTOP_DIR / ".out_delta.csv"),
            )
        )


def require_directory(path: Path, label: str) -> None:
    if not path.is_dir():
        raise FileNotFoundError(f"Не найден каталог {label}: {path}")


def build_configs(site_paths: SitePaths) -> List[PipelineConfig]:
    require_directory(MOBILE_DIR, "mobile-пайплайна")
    require_directory(DESKTOP_DIR, "desktop-пайплайна")

    return [
        PipelineConfig(
            device_type="mobile",
            working_directory=ROOT,
            script_path=MOBILE_DIR / "run_pipeline.py",
            predict_results_path=site_paths.mobile_predict_results,
            out_log_path=site_paths.mobile_out_log,
            out_delta_path=site_paths.mobile_out_delta,
            env_overrides={
                "USE_PRE_SPLIT_RECORDS": "1",
                "OUTPUT_MOBILE": str(site_paths.split_output_mobile),
                "OUTPUT_DESKTOP": str(site_paths.split_output_desktop),
                "SPLIT_CHECKPOINT_FILE": str(site_paths.split_checkpoint),
                "OUT_LOG_PATH": str(site_paths.mobile_out_log),
                "OUT_DELTA_FILE": str(site_paths.mobile_out_delta),
                "PREDICT_RESULTS_PATH": str(site_paths.mobile_predict_results),
                "PREDICT_RESULTS_OLD_PATH": str(
                    site_paths.mobile_predict_results.with_name(
                        "predict_results_old.csv"
                    )
                ),
            },
        ),
        PipelineConfig(
            device_type="desktop",
            working_directory=ROOT,
            script_path=DESKTOP_DIR / "run_pipeline.py",
            predict_results_path=site_paths.desktop_predict_results,
            out_log_path=site_paths.desktop_out_log,
            out_delta_path=site_paths.desktop_out_delta,
            env_overrides={
                "JSON_PATH": str(site_paths.split_output_desktop),
                "OUTPUT_MOBILE": str(site_paths.split_output_mobile),
                "OUTPUT_DESKTOP": str(site_paths.split_output_desktop),
                "SPLIT_CHECKPOINT_FILE": str(site_paths.split_checkpoint),
                "OUT_LOG_PATH": str(site_paths.desktop_out_log),
                "OUT_DELTA_FILE": str(site_paths.desktop_out_delta),
                "PREDICT_RESULTS_PATH": str(site_paths.desktop_predict_results),
                "PREDICT_RESULTS_OLD_PATH": str(
                    site_paths.desktop_predict_results.with_name(
                        "predict_results_old.csv"
                    )
                ),
            },
        ),
    ]


def resolve_path(value: str, fallback: Path) -> Path:
    if not value:
        return fallback

    candidate = Path(value)
    if candidate.is_absolute():
        return candidate

    return (ROOT / candidate).resolve()


def normalize_input_path(env: Dict[str, str], fallback: Path) -> None:
    json_path = env.get("JSON_PATH")
    if not json_path:
        env["JSON_PATH"] = str(fallback)
        return

    candidate = Path(json_path)
    if not candidate.is_absolute():
        env["JSON_PATH"] = str((ROOT / candidate).resolve())


def prepare_root_env(site_paths: SitePaths) -> Dict[str, str]:
    env = os.environ.copy()
    normalize_input_path(env, site_paths.json_path)
    env["OUTPUT_MOBILE"] = str(site_paths.split_output_mobile)
    env["OUTPUT_DESKTOP"] = str(site_paths.split_output_desktop)
    env["SPLIT_CHECKPOINT_FILE"] = str(site_paths.split_checkpoint)
    return env


def run_split_records(env: Dict[str, str]) -> None:
    result = subprocess.run(
        ["node", str(ROOT / "split_records.js")],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
    )

    if result.stdout:
        print(f"\n=== split_records: stdout ===\n{result.stdout}")
    if result.stderr:
        print(f"\n=== split_records: stderr ===\n{result.stderr}")

    if result.returncode != 0:
        raise RuntimeError(f"split_records завершился с кодом {result.returncode}")


def run_single_pipeline(config: PipelineConfig, base_env: Dict[str, str]) -> None:
    env = base_env.copy()
    env.update(config.env_overrides)
    result = subprocess.run(
        [sys.executable, str(config.script_path)],
        cwd=config.working_directory,
        env=env,
        capture_output=True,
        text=True,
    )

    if result.stdout:
        print(f"\n=== {config.device_type}: stdout ===\n{result.stdout}")
    if result.stderr:
        print(f"\n=== {config.device_type}: stderr ===\n{result.stderr}")

    if result.returncode != 0:
        raise RuntimeError(
            f"Пайплайн {config.device_type} завершился с кодом {result.returncode}"
        )


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Не найден обязательный CSV-файл: {path}")

    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fieldnames: List[str], rows: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def merge_predict_results(
    configs: List[PipelineConfig],
    combined_predict_results: Path,
) -> None:
    rows = []
    for config in configs:
        for row in read_csv_rows(config.predict_results_path):
            rows.append(
                {
                    "device_type": config.device_type,
                    "session_id": str(row.get("session_id", "")),
                    "probability": str(row.get("probability", "")),
                }
            )

    rows.sort(key=lambda item: (item["device_type"], item["session_id"]))
    write_csv(
        combined_predict_results,
        ["device_type", "session_id", "probability"],
        rows,
    )


def collect_today_device_session_keys(
    day_str: str,
    split_output_mobile: Path,
    split_output_desktop: Path,
) -> Set[Tuple[str, str]]:
    keys: Set[Tuple[str, str]] = set()
    sources = [
        ("mobile", split_output_mobile),
        ("desktop", split_output_desktop),
    ]

    for device_type, path in sources:
        if not path.exists():
            continue

        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue

                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue

                record_date = obj.get("date")
                if not (isinstance(record_date, str) and record_date.startswith(day_str)):
                    continue

                sess = obj.get("sess")
                if not isinstance(sess, dict) or not sess.get("a"):
                    continue

                keys.add((device_type, str(sess["a"])))

    return keys


def merge_out_logs(
    configs: List[PipelineConfig],
    combined_out_log: Path,
    combined_history_dir: Path,
    split_output_mobile: Path,
    split_output_desktop: Path,
) -> str:
    delta_rows: List[Dict[str, str]] = []

    for config in configs:
        if not config.out_delta_path.exists():
            continue
        for row in read_csv_rows(config.out_delta_path):
            delta_rows.append(
                {
                    "date": str(row.get("date", "")),
                    "device_type": config.device_type,
                    "session_id": str(row.get("session_id", "")),
                    "probability": str(row.get("probability", "")),
                    "ip": str(row.get("ip", "")),
                    "user_agent": str(row.get("user_agent", "")),
                }
            )

    out_path, appended = append_out_log(
        combined_out_log,
        COMBINED_OUT_FIELDNAMES,
        delta_rows,
    )
    if appended:
        print(
            f"\n=== merge_out_logs: append {appended} строк в {out_path} ==="
        )
    else:
        ensure_out_log_file(combined_out_log, COMBINED_OUT_FIELDNAMES)
        print(f"\n=== merge_out_logs: delta пуст, out.log актуален: {combined_out_log} ===")

    day_str = datetime.now().strftime("%Y-%m-%d")
    today_keys = collect_today_device_session_keys(
        day_str,
        split_output_mobile,
        split_output_desktop,
    )
    history_path = combined_history_dir / f"predictions_{day_str}.csv"
    history_by_key: Dict[Tuple[str, str], Dict[str, str]] = {}
    row_key = lambda item: (item.get("device_type", ""), item.get("session_id", ""))

    if history_path.exists():
        for row in read_csv_rows(history_path):
            key = row_key(row)
            if key in today_keys:
                history_by_key[key] = {
                    name: str(row.get(name, "")) for name in COMBINED_OUT_FIELDNAMES
                }

    for row in delta_rows:
        key = row_key(row)
        if key in today_keys:
            history_by_key[key] = {name: str(row.get(name, "")) for name in COMBINED_OUT_FIELDNAMES}

    ordered_rows = [history_by_key[key] for key in sorted(history_by_key.keys())]
    write_csv(history_path, COMBINED_OUT_FIELDNAMES, ordered_rows)
    return day_str


def main() -> None:
    load_dotenv(ROOT / ".env")
    site_paths = SitePaths(os.environ)
    for path in (
        site_paths.site_work_dir,
        site_paths.split_output_mobile.parent,
        site_paths.combined_history_dir,
        site_paths.mobile_predict_results.parent,
        site_paths.desktop_predict_results.parent,
    ):
        path.mkdir(parents=True, exist_ok=True)

    configs = build_configs(site_paths)
    base_env = prepare_root_env(site_paths)
    combined_out_log = site_paths.out_log_path
    combined_history_dir = site_paths.combined_history_dir
    combined_predict_results = site_paths.combined_predict_results

    print(
        f"Запуск unified-пайплайна для site={site_paths.site_id} "
        f"(mobile и desktop)..."
    )
    run_split_records(base_env)

    with ThreadPoolExecutor(max_workers=len(configs)) as executor:
        futures = [executor.submit(run_single_pipeline, config, base_env) for config in configs]
        for future in futures:
            future.result()

    merge_predict_results(configs, combined_predict_results)
    day = merge_out_logs(
        configs,
        combined_out_log,
        combined_history_dir,
        site_paths.split_output_mobile,
        site_paths.split_output_desktop,
    )

    for config in configs:
        if config.out_delta_path.exists():
            config.out_delta_path.unlink()

    print("\nUnified-пайплайн успешно завершён.")
    print(f"Общий файл прогнозов: {combined_predict_results}")
    print(f"Общий лог изменений: {combined_out_log}")
    print(f"Общая дневная история: {combined_history_dir / f'predictions_{day}.csv'}")


if __name__ == "__main__":
    main()
