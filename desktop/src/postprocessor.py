import csv
import glob
import json
import os
import sys
from dotenv import load_dotenv
from datetime import datetime
from pathlib import Path

load_dotenv()

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from out_log_daily import append_out_log, write_out_delta

HISTORY_FIELDNAMES = ['date', 'session_id', 'probability', 'ip', 'user_agent']


def load_history_metadata(history_dir):
    """Последние непустые ip/user_agent по session_id из накопленной истории."""
    history_meta = {}
    history_pattern = os.path.join(history_dir, 'predictions_*.csv')

    for path in sorted(glob.glob(history_pattern)):
        with open(path, 'r', encoding='utf-8', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                sid_raw = row.get('session_id')
                if not sid_raw:
                    continue

                sid = str(sid_raw)
                previous = history_meta.get(sid, {'ip': '', 'user_agent': ''})
                ip = (row.get('ip') or '').strip()
                ua = (row.get('user_agent') or '').strip()

                history_meta[sid] = {
                    'ip': ip or previous['ip'],
                    'user_agent': ua or previous['user_agent'],
                }

    return history_meta


def resolve_session_meta(session_id, current_meta, current_user_agents, history_meta):
    previous_meta = history_meta.get(session_id, {})
    ip = current_meta.get(session_id) or previous_meta.get('ip', '')
    ua = current_user_agents.get(session_id) or previous_meta.get('user_agent', '')
    return ip, ua

def update_if_empty(mapping, key, value):
    """Записывает value только если оно непустое и текущее значение пустое/отсутствует."""
    if value is None:
        return
    value = str(value).strip()
    if not value:
        return
    current = (mapping.get(key) or '').strip()
    if not current:
        mapping[key] = value

predict_results_path = os.getenv(
    'PREDICT_RESULTS_PATH',
    os.getenv('DESKTOP_PREDICT_RESULTS', 'predict_results.csv'),
)
predict_results_old_path = os.getenv(
    'PREDICT_RESULTS_OLD_PATH',
    'predict_results_old.csv',
)

probs = {}
with open(predict_results_path, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        sess = str(row['session_id'])
        prob = row['probability']
        probs[sess] = prob

old_probs = {}
if os.path.exists(predict_results_old_path):
    with open(predict_results_old_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            old_probs[str(row['session_id'])] = row['probability']

meta = {}
user_agents = {}
stats_path = os.getenv('MMV_STATISTICS_JSON', 'desktop_mmv_clk_statistics.json')
if os.path.exists(stats_path):
    with open(stats_path, 'r', encoding='utf-8', errors='replace') as f:
        try:
            records = json.load(f)
        except json.JSONDecodeError:
            records = []
    for obj in records:
        sessA_raw = obj.get('sessA')
        if not sessA_raw:
            continue
        sessA = str(sessA_raw)
        if sessA not in probs:
            continue
        update_if_empty(meta, sessA, obj.get('ip'))
        update_if_empty(user_agents, sessA, obj.get('ua', ''))
else:
    print(f'[WARNING] Файл статистики не найден: {stats_path}, пробуем fallback по JSON_PATH')

json_path = os.getenv('JSON_PATH', '/var/www/mlog/1_466.json')
if os.path.exists(json_path):
    with open(json_path, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            try:
                obj = json.loads(line)
            except Exception:
                continue
            sessA_raw = obj.get('sess', {}).get('a')
            if sessA_raw is None:
                continue
            sessA = str(sessA_raw)
            if sessA not in probs:
                continue
            if obj.get('act') == 'qml.ready':
                update_if_empty(user_agents, sessA, obj.get('ua', {}).get('v', ''))
            update_if_empty(meta, sessA, obj.get('ip'))

history_dir = os.getenv('PREDICTIONS_HISTORY_DIR', 'predictions_history')
os.makedirs(history_dir, exist_ok=True)
history_meta = load_history_metadata(history_dir)

run_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

rows_to_write = []
for sess, prob in probs.items():
    old_prob = old_probs.get(sess)
    if old_prob is None or str(old_prob) != str(prob):
        ip, ua = resolve_session_meta(sess, meta, user_agents, history_meta)
        rows_to_write.append({
            'date': run_date,
            'session_id': sess,
            'probability': prob,
            'ip': ip,
            'user_agent': ua,
        })

print(f'Новых/изменённых сессий для записи: {len(rows_to_write)}')

out_log_path = os.getenv('OUT_LOG_PATH', '1_466.out.log')
out_path, appended = append_out_log(
    out_log_path,
    HISTORY_FIELDNAMES,
    rows_to_write,
)

delta_file = os.getenv('OUT_DELTA_FILE')
if delta_file:
    write_out_delta(delta_file, HISTORY_FIELDNAMES, rows_to_write)

day_str = datetime.now().strftime('%Y-%m-%d')
history_path = os.path.join(history_dir, f'predictions_{day_str}.csv')

history_by_session = {}
if os.path.exists(history_path):
    with open(history_path, 'r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row.get('session_id'):
                continue
            sid = str(row['session_id'])
            history_by_session[sid] = {k: (row.get(k) or '') for k in HISTORY_FIELDNAMES}

for sess, prob in probs.items():
    sid = str(sess)
    ip, ua = resolve_session_meta(sid, meta, user_agents, history_meta)
    history_by_session[sid] = {
        'date': run_date,
        'session_id': sid,
        'probability': str(prob),
        'ip': ip,
        'user_agent': ua,
    }

ordered_sessions = sorted(history_by_session.keys())
tmp_path = os.path.join(history_dir, f'.predictions_{day_str}_{os.getpid()}.tmp.csv')
try:
    with open(tmp_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=HISTORY_FIELDNAMES, extrasaction='ignore')
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
    f'Готово! out.log: {out_path} (append: {appended} строк), '
    f'дата прогноза: {run_date}; '
    f'дневная история: {history_path} ({len(history_by_session)} уникальных сессий за {day_str})'
)
