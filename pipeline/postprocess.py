"""Unified postprocessor for device predict_results → out.log + history."""

from __future__ import annotations

import csv
import glob
import json
import os
from datetime import datetime
from typing import Dict

from out_log_daily import append_out_log, write_out_delta
from prediction.schema import (
    COL_PROBABILITY_BOT_UMAP,
    COL_PROBABILITY_LGBM,
    COL_SESSION_ID,
    DEVICE_OUT_FIELDNAMES,
)


def load_history_metadata(history_dir: str) -> Dict[str, Dict[str, str]]:
    history_meta: Dict[str, Dict[str, str]] = {}
    history_pattern = os.path.join(history_dir, "predictions_*.csv")

    for path in sorted(glob.glob(history_pattern)):
        with open(path, "r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                sid_raw = row.get(COL_SESSION_ID)
                if not sid_raw:
                    continue
                sid = str(sid_raw)
                previous = history_meta.get(sid, {"ip": "", "user_agent": ""})
                ip = (row.get("ip") or "").strip()
                ua = (row.get("user_agent") or "").strip()
                history_meta[sid] = {
                    "ip": ip or previous["ip"],
                    "user_agent": ua or previous["user_agent"],
                }
    return history_meta


def resolve_session_meta(session_id, current_meta, current_user_agents, history_meta):
    previous_meta = history_meta.get(session_id, {})
    ip = current_meta.get(session_id) or previous_meta.get("ip", "")
    ua = current_user_agents.get(session_id) or previous_meta.get("user_agent", "")
    return ip, ua


def update_if_empty(mapping, key, value):
    if value is None:
        return
    value = str(value).strip()
    if not value:
        return
    current = (mapping.get(key) or "").strip()
    if not current:
        mapping[key] = value


def scores_changed(old_row, new_bot_umap, new_lgbm) -> bool:
    if old_row is None:
        return True
    return (
        str(old_row.get(COL_PROBABILITY_BOT_UMAP, "")) != str(new_bot_umap)
        or str(old_row.get(COL_PROBABILITY_LGBM, "")) != str(new_lgbm)
    )


def _score_from_row(row: dict, key: str) -> str:
    return row.get(key, "") or ""


def collect_meta_mobile(session_ids: set[str]) -> tuple[dict, dict]:
    meta: Dict[str, str] = {}
    user_agents: Dict[str, str] = {}
    stats_path = os.getenv("TMV_STATISTICS_JSON", "mobile_tmv_clk_statistics_max10.json")
    if not os.path.exists(stats_path):
        print(f"[WARNING] Файл статистики не найден: {stats_path}, ip и user_agent будут пустыми")
        return meta, user_agents

    with open(stats_path, "r", encoding="utf-8", errors="replace") as handle:
        try:
            records = json.load(handle)
        except json.JSONDecodeError:
            records = []
    for obj in records:
        sess_a_raw = obj.get("sessA")
        if not sess_a_raw:
            continue
        sess_a = str(sess_a_raw)
        if sess_a not in session_ids:
            continue
        if sess_a not in meta:
            ip = obj.get("ip")
            meta[sess_a] = str(ip) if ip else ""
        if sess_a not in user_agents:
            ua = obj.get("ua", "")
            user_agents[sess_a] = str(ua) if ua else ""
    return meta, user_agents


def collect_meta_desktop(session_ids: set[str]) -> tuple[dict, dict]:
    meta: Dict[str, str] = {}
    user_agents: Dict[str, str] = {}
    stats_path = os.getenv("MMV_STATISTICS_JSON", "desktop_mmv_clk_statistics.json")
    if os.path.exists(stats_path):
        with open(stats_path, "r", encoding="utf-8", errors="replace") as handle:
            try:
                records = json.load(handle)
            except json.JSONDecodeError:
                records = []
        for obj in records:
            sess_a_raw = obj.get("sessA")
            if not sess_a_raw:
                continue
            sess_a = str(sess_a_raw)
            if sess_a not in session_ids:
                continue
            update_if_empty(meta, sess_a, obj.get("ip"))
            update_if_empty(user_agents, sess_a, obj.get("ua", ""))
    else:
        print(f"[WARNING] Файл статистики не найден: {stats_path}, пробуем fallback по JSON_PATH")

    json_path = os.getenv("JSON_PATH", "/var/www/mlog/1_466.json")
    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                sess_a_raw = (obj.get("sess") or {}).get("a")
                if sess_a_raw is None:
                    continue
                sess_a = str(sess_a_raw)
                if sess_a not in session_ids:
                    continue
                if obj.get("act") == "qml.ready":
                    update_if_empty(
                        user_agents,
                        sess_a,
                        (obj.get("ua") or {}).get("v", ""),
                    )
                update_if_empty(meta, sess_a, obj.get("ip"))
    return meta, user_agents


def run_postprocess(device: str) -> None:
    device = device.lower()
    predict_results_path = os.getenv(
        "PREDICT_RESULTS_PATH",
        os.getenv(
            "MOBILE_PREDICT_RESULTS" if device == "mobile" else "DESKTOP_PREDICT_RESULTS",
            "predict_results.csv",
        ),
    )
    predict_results_old_path = os.getenv(
        "PREDICT_RESULTS_OLD_PATH",
        "predict_results_old.csv",
    )

    probs_bot_umap: Dict[str, str] = {}
    probs_lgbm: Dict[str, str] = {}
    with open(predict_results_path, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            sess = str(row[COL_SESSION_ID])
            probs_bot_umap[sess] = _score_from_row(row, COL_PROBABILITY_BOT_UMAP)
            probs_lgbm[sess] = _score_from_row(row, COL_PROBABILITY_LGBM)

    old_rows: Dict[str, dict] = {}
    if os.path.exists(predict_results_old_path):
        with open(predict_results_old_path, "r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                old_rows[str(row[COL_SESSION_ID])] = {
                    COL_PROBABILITY_BOT_UMAP: _score_from_row(row, COL_PROBABILITY_BOT_UMAP),
                    COL_PROBABILITY_LGBM: _score_from_row(row, COL_PROBABILITY_LGBM),
                }

    session_ids = set(probs_bot_umap)
    if device == "mobile":
        meta, user_agents = collect_meta_mobile(session_ids)
    else:
        meta, user_agents = collect_meta_desktop(session_ids)

    history_dir = os.getenv("PREDICTIONS_HISTORY_DIR", "predictions_history")
    os.makedirs(history_dir, exist_ok=True)
    history_meta = load_history_metadata(history_dir)
    run_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    rows_to_write = []
    for sess, bot_umap in probs_bot_umap.items():
        lgbm = probs_lgbm.get(sess, "")
        if scores_changed(old_rows.get(sess), bot_umap, lgbm):
            ip, ua = resolve_session_meta(sess, meta, user_agents, history_meta)
            rows_to_write.append(
                {
                    "date": run_date,
                    COL_SESSION_ID: sess,
                    COL_PROBABILITY_BOT_UMAP: bot_umap,
                    COL_PROBABILITY_LGBM: lgbm,
                    "ip": ip,
                    "user_agent": ua,
                }
            )

    print(f"Новых/изменённых сессий для записи: {len(rows_to_write)}")

    out_log_path = os.getenv("OUT_LOG_PATH", "1_466.out.log")
    out_path, appended = append_out_log(
        out_log_path,
        DEVICE_OUT_FIELDNAMES,
        rows_to_write,
    )

    delta_file = os.getenv("OUT_DELTA_FILE")
    if delta_file:
        write_out_delta(delta_file, DEVICE_OUT_FIELDNAMES, rows_to_write)

    day_str = datetime.now().strftime("%Y-%m-%d")
    history_path = os.path.join(history_dir, f"predictions_{day_str}.csv")

    history_by_session: Dict[str, dict] = {}
    if os.path.exists(history_path):
        with open(history_path, "r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if not row.get(COL_SESSION_ID):
                    continue
                sid = str(row[COL_SESSION_ID])
                history_by_session[sid] = {
                    name: (row.get(name) or "") for name in DEVICE_OUT_FIELDNAMES
                }

    for sess, bot_umap in probs_bot_umap.items():
        sid = str(sess)
        ip, ua = resolve_session_meta(sid, meta, user_agents, history_meta)
        history_by_session[sid] = {
            "date": run_date,
            COL_SESSION_ID: sid,
            COL_PROBABILITY_BOT_UMAP: str(bot_umap),
            COL_PROBABILITY_LGBM: str(probs_lgbm.get(sess, "")),
            "ip": ip,
            "user_agent": ua,
        }

    ordered_sessions = sorted(history_by_session.keys())
    tmp_path = os.path.join(
        history_dir, f".predictions_{day_str}_{os.getpid()}.tmp.csv"
    )
    try:
        with open(tmp_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=DEVICE_OUT_FIELDNAMES, extrasaction="ignore"
            )
            writer.writeheader()
            for sid in ordered_sessions:
                writer.writerow(history_by_session[sid])
        os.replace(tmp_path, history_path)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    print(
        f"Готово! out.log: {out_path} (append: {appended} строк), "
        f"дата прогноза: {run_date}; "
        f"дневная история: {history_path} ({len(history_by_session)} уникальных сессий за {day_str})"
    )


if __name__ == "__main__":
    run_postprocess(os.environ.get("DEVICE_TYPE", "mobile"))
