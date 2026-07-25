"""
Preprocessor для анализа логов с поддержкой настраиваемой агрегации.

Функция preprocess_sessions поддерживает параметр use_mean_only:
- use_mean_only=True: использует только средние значения (mean) для агрегации
- use_mean_only=False: использует все статистики (min, max, mean, std, sum, mad)

Это позволяет выбирать между упрощенным (только mean) и полным (все статистики) режимами анализа.
"""
import os
import json
import sys
import numpy as np
import pandas as pd
from collections import defaultdict
import pickle
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

# (ключ в bot_detection, префикс колонок в CSV)
NEW_BOT_DETECTION_FIELDS = [
    ('micro_jitter_2px', 'bot_micro_jitter_2px'),
    ('mmv_points_per_group', 'bot_mmv_points_per_group'),
    ('long_straight_run_share', 'bot_long_straight_run_share'),
    ('turn_rate_45_per_100', 'bot_turn_rate_45_per_100'),
    ('mean_abs_angle_deg', 'bot_mean_abs_angle_deg'),
    ('p90_abs_angle_deg', 'bot_p90_abs_angle_deg'),
    ('angle_entropy', 'bot_angle_entropy'),
    ('curvature_per_100px', 'bot_curvature_per_100px'),
]


def aggregate_scalar_features(res, prefix, values, use_mean_only):
    if use_mean_only:
        res[f'{prefix}_mean'] = get_mean(values)
        return
    res[f'{prefix}_min'] = get_min(values)
    res[f'{prefix}_max'] = get_max(values)
    res[f'{prefix}_mean'] = get_mean(values)
    res[f'{prefix}_std'] = get_std(values)


def scaler_matches_features(scaler, feature_names):
    saved_names = getattr(scaler, 'feature_names_in_', None)
    if saved_names is not None:
        return list(saved_names) == list(feature_names)
    return getattr(scaler, 'n_features_in_', None) == len(feature_names)


def load_or_fit_scaler(df, scaler_path):
    feature_names = list(df.columns)

    if os.path.exists(scaler_path):
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        if scaler_matches_features(scaler, feature_names):
            print(f"Загружаем сохраненный scaler из '{scaler_path}'...")
            df_normalized = pd.DataFrame(
                scaler.transform(df), columns=feature_names, index=df.index
            )
            print("Применены параметры масштабирования из сохраненного scaler'а")
            return df_normalized, scaler

        saved_count = len(getattr(scaler, 'feature_names_in_', [])) or getattr(
            scaler, 'n_features_in_', '?'
        )
        print(
            f"Сохраненный scaler не совпадает с текущими признаками "
            f"({saved_count} vs {len(feature_names)}). Пересоздаем scaler..."
        )

    print(f"Создаем новый scaler и сохраняем в '{scaler_path}'...")
    scaler = MinMaxScaler()
    df_normalized = pd.DataFrame(
        scaler.fit_transform(df), columns=feature_names, index=df.index
    )
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"Новый scaler сохранен в '{scaler_path}'")
    return df_normalized, scaler


def preprocess_sessions(records, use_mean_only=False):
    sessions = defaultdict(list)
    for rec in records:
        key = rec["sessA"]
        sessions[key].append(rec)

    dataset = []
    for session_id, session_records in sessions.items():
        res = {}
        res['session_id'] = session_id

        # --- scroll ---
        scroll_prm_data_yt = []
        
        # --- mousemove (только базовые признаки) ---
        mmv_b_n = []
        mmv_b_ms = []
        mmv_b_l = []

        # --- новые признаки для детекции ботов ---
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
        bot_new_features = {key: [] for key, _ in NEW_BOT_DETECTION_FIELDS}

        for rec in session_records:
            if rec['type'] in ('mmv_clk', 'qml_ready'):
                # Обрабатываем агрегированные скроллы из scrollData
                scroll_data = rec.get('scrollData', [])
                scroll_prm_data_yt.extend(scroll_data)
                stat = rec['statistics']
                
                # Обрабатываем базовые признаки из _b
                b = stat.get('_b', {})
                if b:
                    mmv_b_n.append(b.get('n', 0))
                    mmv_b_ms.append(b.get('ms', 0))
                    mmv_b_l.append(b.get('l', 0))
                else:
                    # Если _b отсутствует, добавляем нулевые значения
                    mmv_b_n.append(0)
                    mmv_b_ms.append(0)
                    mmv_b_l.append(0)

                # Обрабатываем новые признаки для детекции ботов
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
                    for key, _ in NEW_BOT_DETECTION_FIELDS:
                        bot_new_features[key].append(bot_detection.get(key, 0))
                else:
                    # Если bot_detection отсутствует, добавляем нулевые значения
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
                    for key, _ in NEW_BOT_DETECTION_FIELDS:
                        bot_new_features[key].append(0)

        scroll_count = len(scroll_prm_data_yt)
        mousemove_count = len(mmv_b_n)

        # scroll
        if use_mean_only:
            res['scroll_prm_data_mean'] = get_mean(scroll_prm_data_yt)
        else:
            res['scroll_prm_data_min'] = get_min(scroll_prm_data_yt)
            res['scroll_prm_data_max'] = get_max(scroll_prm_data_yt)
            res['scroll_prm_data_mean'] = get_mean(scroll_prm_data_yt)
            res['scroll_prm_data_sum'] = get_sum(scroll_prm_data_yt)
            res['scroll_prm_data_mad'] = get_mad(scroll_prm_data_yt)
            res['scroll_prm_data_std'] = get_std(scroll_prm_data_yt)

        # mousemove (только базовые признаки)
        if use_mean_only:
            res['mousemove_prm_data_b_n_mean'] = get_mean(mmv_b_n)
            res['mousemove_prm_data_b_ms_mean'] = get_mean(mmv_b_ms)
            res['mousemove_prm_data_b_l_mean'] = get_mean(mmv_b_l)
        else:
            res['mousemove_prm_data_b_n_min'] = get_min(mmv_b_n)
            res['mousemove_prm_data_b_n_max'] = get_max(mmv_b_n)
            res['mousemove_prm_data_b_n_mean'] = get_mean(mmv_b_n)
            res['mousemove_prm_data_b_n_std'] = get_std(mmv_b_n)

            res['mousemove_prm_data_b_ms_min'] = get_min(mmv_b_ms)
            res['mousemove_prm_data_b_ms_max'] = get_max(mmv_b_ms)
            res['mousemove_prm_data_b_ms_mean'] = get_mean(mmv_b_ms)
            res['mousemove_prm_data_b_ms_std'] = get_std(mmv_b_ms)

            res['mousemove_prm_data_b_l_min'] = get_min(mmv_b_l)
            res['mousemove_prm_data_b_l_max'] = get_max(mmv_b_l)
            res['mousemove_prm_data_b_l_mean'] = get_mean(mmv_b_l)
            res['mousemove_prm_data_b_l_std'] = get_std(mmv_b_l)

        # --- Новые признаки для детекции ботов ---
        
        if use_mean_only:
            # Энтропия направлений движения
            res['bot_direction_entropy_mean'] = get_mean(bot_direction_entropy)
            # Джерк (третья производная)
            res['bot_jerk_mean'] = get_mean(bot_jerk)
            # Линейность траектории
            res['bot_linearity_mean'] = get_mean(bot_linearity)
            # Микродвижения (дрожание)
            res['bot_jitter_mean'] = get_mean(bot_jitter)
            # Вариативность скорости
            res['bot_speed_variability_mean'] = get_mean(bot_speed_variability)
            # Кривизна траектории
            res['bot_curvature_mean'] = get_mean(bot_curvature)
            # Паттерны повторяемости
            res['bot_repetition_patterns_mean'] = get_mean(bot_repetition_patterns)
            # Нерегулярность временных интервалов
            res['bot_time_irregularity_mean'] = get_mean(bot_time_irregularity)
            # Вариативность ускорения
            res['bot_acceleration_variability_mean'] = get_mean(bot_acceleration_variability)
            # Общий индекс бота
            res['bot_score_mean'] = get_mean(bot_score)
        else:
            # Энтропия направлений движения
            res['bot_direction_entropy_min'] = get_min(bot_direction_entropy)
            res['bot_direction_entropy_max'] = get_max(bot_direction_entropy)
            res['bot_direction_entropy_mean'] = get_mean(bot_direction_entropy)
            res['bot_direction_entropy_std'] = get_std(bot_direction_entropy)

            # Джерк (третья производная)
            res['bot_jerk_min'] = get_min(bot_jerk)
            res['bot_jerk_max'] = get_max(bot_jerk)
            res['bot_jerk_mean'] = get_mean(bot_jerk)
            res['bot_jerk_std'] = get_std(bot_jerk)

            # Линейность траектории
            res['bot_linearity_min'] = get_min(bot_linearity)
            res['bot_linearity_max'] = get_max(bot_linearity)
            res['bot_linearity_mean'] = get_mean(bot_linearity)
            res['bot_linearity_std'] = get_std(bot_linearity)

            # Микродвижения (дрожание)
            res['bot_jitter_min'] = get_min(bot_jitter)
            res['bot_jitter_max'] = get_max(bot_jitter)
            res['bot_jitter_mean'] = get_mean(bot_jitter)
            res['bot_jitter_std'] = get_std(bot_jitter)

            # Вариативность скорости
            res['bot_speed_variability_min'] = get_min(bot_speed_variability)
            res['bot_speed_variability_max'] = get_max(bot_speed_variability)
            res['bot_speed_variability_mean'] = get_mean(bot_speed_variability)
            res['bot_speed_variability_std'] = get_std(bot_speed_variability)

            # Кривизна траектории
            res['bot_curvature_min'] = get_min(bot_curvature)
            res['bot_curvature_max'] = get_max(bot_curvature)
            res['bot_curvature_mean'] = get_mean(bot_curvature)
            res['bot_curvature_std'] = get_std(bot_curvature)

            # Паттерны повторяемости
            res['bot_repetition_patterns_min'] = get_min(bot_repetition_patterns)
            res['bot_repetition_patterns_max'] = get_max(bot_repetition_patterns)
            res['bot_repetition_patterns_mean'] = get_mean(bot_repetition_patterns)
            res['bot_repetition_patterns_std'] = get_std(bot_repetition_patterns)

            # Нерегулярность временных интервалов
            res['bot_time_irregularity_min'] = get_min(bot_time_irregularity)
            res['bot_time_irregularity_max'] = get_max(bot_time_irregularity)
            res['bot_time_irregularity_mean'] = get_mean(bot_time_irregularity)
            res['bot_time_irregularity_std'] = get_std(bot_time_irregularity)

            # Вариативность ускорения
            res['bot_acceleration_variability_min'] = get_min(bot_acceleration_variability)
            res['bot_acceleration_variability_max'] = get_max(bot_acceleration_variability)
            res['bot_acceleration_variability_mean'] = get_mean(bot_acceleration_variability)
            res['bot_acceleration_variability_std'] = get_std(bot_acceleration_variability)

            # Общий индекс бота
            res['bot_score_min'] = get_min(bot_score)
            res['bot_score_max'] = get_max(bot_score)
            res['bot_score_mean'] = get_mean(bot_score)
            res['bot_score_std'] = get_std(bot_score)

        for key, prefix in NEW_BOT_DETECTION_FIELDS:
            aggregate_scalar_features(res, prefix, bot_new_features[key], use_mean_only)

        # Доля записей, классифицированных как бот
        res['bot_is_likely_bot_ratio'] = np.mean(bot_is_likely_bot) if bot_is_likely_bot else 0

        # counts
        res['scroll_count'] = scroll_count
        res['mousemove_count'] = mousemove_count
        res['ratio_scroll_count_on_mousemove_count'] = res['scroll_count'] / res['mousemove_count'] if res['mousemove_count'] else 0

        # Дополнительные метрики для анализа
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
    """Анализ признаков детекции ботов"""
    print("=== Анализ признаков детекции ботов ===")
    
    # Статистика по bot_score
    bot_score_cols = [col for col in df.columns if 'bot_score' in col]
    if bot_score_cols:
        print(f"\nСтатистика по bot_score:")
        for col in bot_score_cols:
            print(f"{col}: {df[col].describe()}")
    
    # Статистика по is_likely_bot
    bot_ratio_cols = [col for col in df.columns if 'bot_is_likely_bot' in col or 'bot_score_above' in col]
    if bot_ratio_cols:
        print(f"\nСтатистика по классификации ботов:")
        for col in bot_ratio_cols:
            print(f"{col}: {df[col].describe()}")
    
    # Корреляции между признаками детекции ботов
    bot_features = [col for col in df.columns if col.startswith('bot_')]
    if len(bot_features) > 1:
        print(f"\nКорреляции между признаками детекции ботов:")
        corr_matrix = df[bot_features].corr()
        print(corr_matrix)

def create_bot_detection_summary(df):
    """Создание сводки по детекции ботов"""
    summary = {}
    
    # Общая статистика
    summary['total_sessions'] = int(len(df))
    if 'bot_detection_features_count' in df.columns:
        summary['sessions_with_bot_features'] = int(len(df[df['bot_detection_features_count'] > 0]))
    else:
        summary['sessions_with_bot_features'] = 0
    
    # Статистика по bot_score
    if 'bot_score_mean' in df.columns:
        summary['avg_bot_score'] = float(df['bot_score_mean'].mean())
        summary['sessions_above_0.5_threshold'] = int(len(df[df['bot_score_mean'] > 0.5]))
        summary['sessions_above_0.8_threshold'] = int(len(df[df['bot_score_mean'] > 0.8]))
    
    # Статистика по is_likely_bot
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
    stats_path = os.environ.get('MMV_STATISTICS_JSON', 'desktop_mmv_clk_statistics.json')
    print(f"Загрузка данных из {stats_path}...")
    with open(stats_path, 'r', encoding='utf-8') as f:
        records = json.load(f)

    print("Обработка сессий...")
    changed_sessions_path = os.environ.get('CHANGED_SESSIONS_FILE', 'changed_sessions.json')
    changed_payload = load_changed_sessions(changed_sessions_path)
    changed_sessions = None if changed_payload is None else changed_payload["session_ids"]
    use_mean_only = True
    suffix = "_mean_only" if use_mean_only else "_full"
    csv_file = os.environ.get(
        'DESKTOP_FEATURES_PATH',
        f'mmv_session_features_enhanced{suffix}.csv',
    )
    csv_norm_file = os.environ.get(
        'DESKTOP_NORMALIZED_FEATURES_PATH',
        f'mmv_session_features_enhanced{suffix}_normalized.csv',
    )
    summary_file = os.environ.get(
        'DESKTOP_BOT_SUMMARY_PATH',
        'mmv_bot_detection_summary.json',
    )
    scaler_path = os.environ.get('DESKTOP_SCALER_PATH', 'minmax_scaler.pkl')

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

        df_updated = preprocess_sessions(updated_records, use_mean_only=use_mean_only)
        df_updated.fillna(0, inplace=True)
        updated_df = df_updated
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
        print("Перенесите minmax_scaler.pkl в каталог desktop/models/.")
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

    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    print(f"- {summary_file} (сводка по детекции ботов)")
