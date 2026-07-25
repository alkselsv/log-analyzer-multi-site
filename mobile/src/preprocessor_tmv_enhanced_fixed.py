"""
Preprocessor для анализа логов с поддержкой настраиваемой агрегации.

Функция preprocess_sessions поддерживает параметр use_mean_only:
- use_mean_only=True: использует только средние значения (mean) для агрегации
- use_mean_only=False: использует все статистики (min, max, mean, std, sum, mad)

Это позволяет выбирать между упрощенным (только mean) и полным (все статистики) режимами анализа.
"""

import argparse
import json
import os
import pickle
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def get_min(lst):
    arr = np.array(lst, dtype=np.float32)
    result = np.nanmin(np.abs(arr)) if arr.size > 0 else 0
    return float(result) if not np.isnan(result) else 0.0


def get_max(lst):
    arr = np.array(lst, dtype=np.float32)
    result = np.nanmax(np.abs(arr)) if arr.size > 0 else 0
    return float(result) if not np.isnan(result) else 0.0


def get_mean(lst):
    arr = np.array(lst, dtype=np.float32)
    result = np.nanmean(np.abs(arr)) if arr.size > 0 else 0
    return float(result) if not np.isnan(result) else 0.0


def get_mad(lst):
    arr = np.array(lst, dtype=np.float32)
    mean_val = np.nanmean(arr)
    if np.isnan(mean_val):
        return 0.0
    abs_deviation = np.abs(arr - mean_val)
    mad = np.nanmean(abs_deviation)
    return float(mad) if not np.isnan(mad) else 0.0


def get_sum(lst):
    arr = np.array(lst, dtype=np.float32)
    result = np.sum(np.abs(arr)) if arr.size > 0 else 0
    return float(result) if not np.isnan(result) else 0.0


def get_std(lst):
    arr = np.array(lst, dtype=np.float32)
    result = np.nanstd(arr) if arr.size > 0 else 0
    return float(result) if not np.isnan(result) else 0.0


def preprocess_sessions(records, use_mean_only=False):
    sessions = defaultdict(list)
    for rec in records:
        key = rec["sessA"]
        sessions[key].append(rec)

    dataset = []
    for session_id, session_records in sessions.items():
        res = {}
        res['session_id'] = session_id

        scroll_prm_data_yt = []
        tmv_b_n = []
        tmv_b_ms = []
        tmv_b_l = []

        bot_direction_entropy = []
        bot_jerk = []
        bot_linearity = []
        bot_jitter = []
        bot_speed_variability = []
        bot_curvature = []
        bot_repetition_patterns = []
        bot_time_irregularity = []
        bot_acceleration_variability = []
        bot_score = []
        bot_is_likely_bot = []

        for rec in session_records:
            if rec['type'] in ('tmv_clk', 'qml_ready'):
                scroll_data = rec.get('scrollData', [])
                scroll_prm_data_yt.extend(scroll_data)
                stat = rec['statistics']

                b = stat.get('_b', {})
                if b:
                    tmv_b_n.append(b.get('n', 0))
                    tmv_b_ms.append(b.get('ms', 0))
                    tmv_b_l.append(b.get('l', 0))
                else:
                    tmv_b_n.append(0)
                    tmv_b_ms.append(0)
                    tmv_b_l.append(0)

                bot_detection = stat.get('bot_detection', {})
                if bot_detection:
                    bot_direction_entropy.append(bot_detection.get('direction_entropy', 0))
                    bot_jerk.append(bot_detection.get('jerk', 0))
                    bot_linearity.append(bot_detection.get('linearity', 0))
                    bot_jitter.append(bot_detection.get('jitter', 0))
                    bot_speed_variability.append(bot_detection.get('speed_variability', 0))
                    bot_curvature.append(bot_detection.get('curvature', 0))
                    bot_repetition_patterns.append(bot_detection.get('repetition_patterns', 0))
                    bot_time_irregularity.append(bot_detection.get('time_irregularity', 0))
                    bot_acceleration_variability.append(bot_detection.get('acceleration_variability', 0))
                    bot_score.append(bot_detection.get('bot_score', 0))
                    bot_is_likely_bot.append(1 if bot_detection.get('is_likely_bot', False) else 0)
                else:
                    bot_direction_entropy.append(0)
                    bot_jerk.append(0)
                    bot_linearity.append(0)
                    bot_jitter.append(0)
                    bot_speed_variability.append(0)
                    bot_curvature.append(0)
                    bot_repetition_patterns.append(0)
                    bot_time_irregularity.append(0)
                    bot_acceleration_variability.append(0)
                    bot_score.append(0)
                    bot_is_likely_bot.append(0)

        scroll_count = len(scroll_prm_data_yt)
        mousemove_count = len(tmv_b_n)

        if use_mean_only:
            res['scroll_prm_data_mean'] = get_mean(scroll_prm_data_yt)
        else:
            res['scroll_prm_data_min'] = get_min(scroll_prm_data_yt)
            res['scroll_prm_data_max'] = get_max(scroll_prm_data_yt)
            res['scroll_prm_data_mean'] = get_mean(scroll_prm_data_yt)
            res['scroll_prm_data_sum'] = get_sum(scroll_prm_data_yt)
            res['scroll_prm_data_mad'] = get_mad(scroll_prm_data_yt)
            res['scroll_prm_data_std'] = get_std(scroll_prm_data_yt)

        if use_mean_only:
            res['mousemove_prm_data_b_n_mean'] = get_mean(tmv_b_n)
            res['mousemove_prm_data_b_ms_mean'] = get_mean(tmv_b_ms)
            res['mousemove_prm_data_b_l_mean'] = get_mean(tmv_b_l)
        else:
            res['mousemove_prm_data_b_n_min'] = get_min(tmv_b_n)
            res['mousemove_prm_data_b_n_max'] = get_max(tmv_b_n)
            res['mousemove_prm_data_b_n_mean'] = get_mean(tmv_b_n)
            res['mousemove_prm_data_b_n_std'] = get_std(tmv_b_n)
            res['mousemove_prm_data_b_ms_min'] = get_min(tmv_b_ms)
            res['mousemove_prm_data_b_ms_max'] = get_max(tmv_b_ms)
            res['mousemove_prm_data_b_ms_mean'] = get_mean(tmv_b_ms)
            res['mousemove_prm_data_b_ms_std'] = get_std(tmv_b_ms)
            res['mousemove_prm_data_b_l_min'] = get_min(tmv_b_l)
            res['mousemove_prm_data_b_l_max'] = get_max(tmv_b_l)
            res['mousemove_prm_data_b_l_mean'] = get_mean(tmv_b_l)
            res['mousemove_prm_data_b_l_std'] = get_std(tmv_b_l)

        if use_mean_only:
            res['bot_direction_entropy_mean'] = get_mean(bot_direction_entropy)
            res['bot_jerk_mean'] = get_mean(bot_jerk)
            res['bot_linearity_mean'] = get_mean(bot_linearity)
            res['bot_jitter_mean'] = get_mean(bot_jitter)
            res['bot_speed_variability_mean'] = get_mean(bot_speed_variability)
            res['bot_curvature_mean'] = get_mean(bot_curvature)
            res['bot_repetition_patterns_mean'] = get_mean(bot_repetition_patterns)
            res['bot_time_irregularity_mean'] = get_mean(bot_time_irregularity)
            res['bot_acceleration_variability_mean'] = get_mean(bot_acceleration_variability)
            res['bot_score_mean'] = get_mean(bot_score)
        else:
            res['bot_direction_entropy_min'] = get_min(bot_direction_entropy)
            res['bot_direction_entropy_max'] = get_max(bot_direction_entropy)
            res['bot_direction_entropy_mean'] = get_mean(bot_direction_entropy)
            res['bot_direction_entropy_std'] = get_std(bot_direction_entropy)
            res['bot_jerk_min'] = get_min(bot_jerk)
            res['bot_jerk_max'] = get_max(bot_jerk)
            res['bot_jerk_mean'] = get_mean(bot_jerk)
            res['bot_jerk_std'] = get_std(bot_jerk)
            res['bot_linearity_min'] = get_min(bot_linearity)
            res['bot_linearity_max'] = get_max(bot_linearity)
            res['bot_linearity_mean'] = get_mean(bot_linearity)
            res['bot_linearity_std'] = get_std(bot_linearity)
            res['bot_jitter_min'] = get_min(bot_jitter)
            res['bot_jitter_max'] = get_max(bot_jitter)
            res['bot_jitter_mean'] = get_mean(bot_jitter)
            res['bot_jitter_std'] = get_std(bot_jitter)
            res['bot_speed_variability_min'] = get_min(bot_speed_variability)
            res['bot_speed_variability_max'] = get_max(bot_speed_variability)
            res['bot_speed_variability_mean'] = get_mean(bot_speed_variability)
            res['bot_speed_variability_std'] = get_std(bot_speed_variability)
            res['bot_curvature_min'] = get_min(bot_curvature)
            res['bot_curvature_max'] = get_max(bot_curvature)
            res['bot_curvature_mean'] = get_mean(bot_curvature)
            res['bot_curvature_std'] = get_std(bot_curvature)
            res['bot_repetition_patterns_min'] = get_min(bot_repetition_patterns)
            res['bot_repetition_patterns_max'] = get_max(bot_repetition_patterns)
            res['bot_repetition_patterns_mean'] = get_mean(bot_repetition_patterns)
            res['bot_repetition_patterns_std'] = get_std(bot_repetition_patterns)
            res['bot_time_irregularity_min'] = get_min(bot_time_irregularity)
            res['bot_time_irregularity_max'] = get_max(bot_time_irregularity)
            res['bot_time_irregularity_mean'] = get_mean(bot_time_irregularity)
            res['bot_time_irregularity_std'] = get_std(bot_time_irregularity)
            res['bot_acceleration_variability_min'] = get_min(bot_acceleration_variability)
            res['bot_acceleration_variability_max'] = get_max(bot_acceleration_variability)
            res['bot_acceleration_variability_mean'] = get_mean(bot_acceleration_variability)
            res['bot_acceleration_variability_std'] = get_std(bot_acceleration_variability)
            res['bot_score_min'] = get_min(bot_score)
            res['bot_score_max'] = get_max(bot_score)
            res['bot_score_mean'] = get_mean(bot_score)
            res['bot_score_std'] = get_std(bot_score)

        res['bot_is_likely_bot_ratio'] = np.mean(bot_is_likely_bot) if bot_is_likely_bot else 0
        res['scroll_count'] = scroll_count
        res['mousemove_count'] = mousemove_count
        res['ratio_scroll_count_on_mousemove_count'] = res['scroll_count'] / res['mousemove_count'] if res['mousemove_count'] else 0
        res['bot_detection_features_count'] = len(bot_direction_entropy)
        res['bot_score_above_threshold_ratio'] = np.mean([1 if score > 0.5 else 0 for score in bot_score]) if bot_score else 0
        res['bot_score_above_high_threshold_ratio'] = np.mean([1 if score > 0.8 else 0 for score in bot_score]) if bot_score else 0

        dataset.append(res)

    if not dataset:
        empty_df = pd.DataFrame()
        empty_df.index.name = 'session_id'
        return empty_df

    return pd.DataFrame(dataset).set_index('session_id')


def analyze_bot_detection_features(df):
    print("=== Анализ признаков детекции ботов ===")
    bot_score_cols = [col for col in df.columns if 'bot_score' in col]
    if bot_score_cols:
        print(f"\nСтатистика по bot_score:")
        for col in bot_score_cols:
            print(f"{col}: {df[col].describe()}")

    bot_ratio_cols = [col for col in df.columns if 'bot_is_likely_bot' in col or 'bot_score_above' in col]
    if bot_ratio_cols:
        print(f"\nСтатистика по классификации ботов:")
        for col in bot_ratio_cols:
            print(f"{col}: {df[col].describe()}")

    bot_features = [col for col in df.columns if col.startswith('bot_')]
    if len(bot_features) > 1:
        print(f"\nКорреляции между признаками детекции ботов:")
        corr_matrix = df[bot_features].corr()
        print(corr_matrix)


def create_bot_detection_summary(df):
    summary = {}
    summary['total_sessions'] = int(len(df))
    if 'bot_detection_features_count' in df.columns:
        summary['sessions_with_bot_features'] = int(len(df[df['bot_detection_features_count'] > 0]))
    else:
        summary['sessions_with_bot_features'] = 0

    if 'bot_score_mean' in df.columns:
        summary['avg_bot_score'] = float(df['bot_score_mean'].mean())
        summary['sessions_above_0.5_threshold'] = int(len(df[df['bot_score_mean'] > 0.5]))
        summary['sessions_above_0.8_threshold'] = int(len(df[df['bot_score_mean'] > 0.8]))

    if 'bot_is_likely_bot_ratio' in df.columns:
        summary['avg_bot_ratio'] = float(df['bot_is_likely_bot_ratio'].mean())
        summary['sessions_classified_as_bot'] = int(len(df[df['bot_is_likely_bot_ratio'] > 0.5]))

    return summary


def load_changed_sessions(path):
    if not path or not os.path.exists(path):
        return None

    with open(path, 'r', encoding='utf-8') as f:
        try:
            payload = json.load(f)
        except json.JSONDecodeError:
            return None

    if isinstance(payload, list):
        return {
            "full_rebuild": False,
            "session_ids": [str(item) for item in payload],
        }

    if isinstance(payload, dict):
        session_ids = payload.get("session_ids")
        if not isinstance(session_ids, list):
            return None
        return {
            "full_rebuild": bool(payload.get("full_rebuild", False)),
            "session_ids": [str(item) for item in session_ids],
        }

    return None


def load_dataframe(path):
    if not os.path.exists(path):
        return pd.DataFrame()

    df = pd.read_csv(path, index_col='session_id')
    df.index = df.index.astype(str)
    return df


def merge_session_frames(existing_df, updated_df):
    if existing_df.empty:
        return updated_df.sort_index()
    if updated_df.empty:
        return existing_df.sort_index()
    base_df = existing_df.drop(index=updated_df.index, errors='ignore')
    merged_df = pd.concat([base_df, updated_df], axis=0)
    merged_df = merged_df[~merged_df.index.duplicated(keep='last')]
    return merged_df.sort_index()


def normalize_subset(df, scaler):
    feature_names = list(df.columns)
    saved_names = getattr(scaler, 'feature_names_in_', None)
    if saved_names is not None and list(saved_names) != feature_names:
        missing = [name for name in saved_names if name not in df.columns]
        if missing:
            raise ValueError(
                "В данных отсутствуют признаки, на которых обучен scaler: "
                + ", ".join(missing[:5])
                + ("..." if len(missing) > 5 else "")
            )
        df = df[list(saved_names)]

    normalized = scaler.transform(df)
    return pd.DataFrame(normalized, columns=df.columns, index=df.index)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Препроцессор логов TMV/CLK с агрегацией по сессиям')
    parser.add_argument(
        'input_file',
        nargs='?',
        default='mobile_tmv_clk_statistics.json',
        help='Входной JSON-файл (по умолчанию: mobile_tmv_clk_statistics.json)'
    )
    parser.add_argument(
        '-s', '--site-name',
        default='',
        help='Название сайта (добавляется в имена выходных файлов)'
    )
    args = parser.parse_args()
    input_file = args.input_file
    site_name = (args.site_name or '').strip().replace(' ', '_') or None
    out_prefix = f"{site_name}_" if site_name else ""
    changed_sessions_path = os.environ.get('CHANGED_SESSIONS_FILE', 'changed_sessions.json')
    changed_payload = load_changed_sessions(changed_sessions_path)
    changed_sessions = None if changed_payload is None else changed_payload["session_ids"]
    use_mean_only = True
    suffix = "_mean_only" if use_mean_only else "_full"
    csv_file = os.environ.get('MOBILE_FEATURES_PATH', f'{out_prefix}tmv_session_features_enhanced{suffix}.csv')
    csv_norm_file = os.environ.get(
        'MOBILE_NORMALIZED_FEATURES_PATH',
        f'{out_prefix}tmv_session_features_enhanced{suffix}_normalized.csv',
    )
    summary_file = os.environ.get(
        'MOBILE_BOT_SUMMARY_PATH',
        f'{out_prefix}tmv_bot_detection_summary.json',
    )
    scaler_path = os.environ.get(
        'MOBILE_SCALER_PATH',
        f'{out_prefix}minmax_scaler_mobile.pkl',
    )

    print(f"Загрузка данных из {input_file}...")
    with open(input_file, 'r') as f:
        records = json.load(f)

    print("Обработка сессий...")
    existing_df = load_dataframe(csv_file)
    existing_norm_df = load_dataframe(csv_norm_file)
    full_rebuild = (
        changed_payload is None
        or bool(changed_payload.get("full_rebuild", False))
        or existing_df.empty
    )

    if changed_sessions == [] and not existing_df.empty:
        print("Новых session_id для пересчета нет, используем существующие признаки.")
        df = existing_df.copy()
        updated_df = pd.DataFrame(columns=df.columns)
    else:
        if full_rebuild:
            print("Режим: полный пересчет признаков.")
            updated_records = records
        else:
            changed_set = set(changed_sessions)
            print(f"Режим: инкрементальный пересчет для {len(changed_set)} session_id.")
            updated_records = [rec for rec in records if str(rec.get('sessA')) in changed_set]

        updated_df = preprocess_sessions(updated_records, use_mean_only=use_mean_only)
        updated_df.fillna(0, inplace=True)
        df = updated_df if full_rebuild else merge_session_frames(existing_df, updated_df)

    df.fillna(0, inplace=True)

    print(f"Режим агрегации: {'только mean' if use_mean_only else 'все статистики'}")
    print(f"Обработано сессий: {len(df)}")
    print(f"Количество признаков: {len(df.columns)}")

    analyze_bot_detection_features(df)
    summary = create_bot_detection_summary(df)
    print(f"\n=== Сводка по детекции ботов ===")
    for key, value in summary.items():
        print(f"{key}: {value}")

    if not os.path.exists(scaler_path):
        print(f"[ERROR] Не найден scaler: '{scaler_path}'")
        print("Перенесите minmax_scaler_mobile.pkl в каталог mobile/models/.")
        sys.exit(1)

    print(f"Загружаем сохраненный scaler из '{scaler_path}'...")
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)

    try:
        if full_rebuild or existing_norm_df.empty or updated_df.empty:
            df_normalized = normalize_subset(df, scaler)
        else:
            updated_norm_df = normalize_subset(updated_df, scaler)
            df_normalized = merge_session_frames(existing_norm_df, updated_norm_df)
    except ValueError as exc:
        print(f"[ERROR] {exc}")
        sys.exit(1)

    print("Применены параметры масштабирования из сохраненного scaler'а")

    df.to_csv(csv_file, index_label='session_id')
    df_normalized.to_csv(csv_norm_file, index_label='session_id')

    print("Результаты сохранены:")
    print(f"- {csv_file} (исходные данные)")
    print(f"- {csv_norm_file} (нормализованные данные)")

    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"- {summary_file} (сводка по детекции ботов)")
