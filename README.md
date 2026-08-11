# log-analyzer-multi-site

Мультисайтовый пайплайн **prediction** вероятности заказа по mobile- и desktop-сессиям.

Обучение моделей и выпуск артефактов — в соседнем проекте
[`log-analyzer-model-training`](../log-analyzer-model-training).

## Назначение

В `/var/www/mlog` лежат NDJSON-логи вида `1_466.json`, `1_200.json` — по одному файлу на сайт. Проект:

1. **обнаруживает** логи автоматически;
2. **запускает** инкрементальный пайплайн mobile + desktop для каждого сайта;
3. **изолирует** checkpoint'ы и промежуточные файлы в `sites/{site_id}/`;
4. **пишет** итоговый журнал в `/var/www/mlog/{site_id}.out.log`.

Алгоритм: UMAP + расстояние до центра кластера заказов → `probability_umap`;
опционально LightGBM (Transformer-эмбеддинг + nobot-признаки) → `probability_lgbm`
(см. раздел «Алгоритм»).

## Структура проекта

```
log-analyzer-multi-site/
├── run_pipeline.py              # unified orchestrator (split + mobile || desktop)
├── run_site_pipeline.py         # запуск одного сайта
├── run_all_sites.py             # параллельный запуск
├── site_watcher.py              # inotify-демон
├── site_registry.py             # discovery сайтов и моделей
├── site_paths.py                # env для site-scoped путей
├── out_log_daily.py             # дневной out.log
├── prediction/                  # весь scoring
│   ├── schema.py
│   ├── runner.py
│   ├── order_probability.py     # decay/linear UMAP probability
│   ├── backends/                # umap.py, lgbm.py
│   └── lgbm/                    # LightGBM artifacts / encode / tokenize
├── pipeline/                    # device pipeline + postprocess
│   ├── device_pipeline.py
│   └── postprocess.py
├── features/                    # извлечение признаков (mobile ≠ desktop)
│   ├── mobile/                  # process.js + preprocessor.py
│   └── desktop/
├── models/                      # base runtime-артефакты (из training-проекта)
│   ├── mobile/
│   └── desktop/
├── split_records.js
├── sites/{site_id}/             # per-site workspace и outputs
└── scripts/
    ├── site-watcher.service
    ├── cron_rescan_sites.sh
    └── reset_out_log_daily.py
```

## Архитектура

```
/var/www/mlog/1_*.json
        │
        ▼
site_watcher.py (inotify)  +  cron_rescan (страховка)
        │
        ▼
run_site_pipeline.py  →  flock  →  run_pipeline.py
        │
        ├── split_records.js
        ├── python -m pipeline.device_pipeline --device mobile  ─┐ parallel
        └── python -m pipeline.device_pipeline --device desktop ─┘
        │
        ▼
/var/www/mlog/{site_id}.out.log
sites/{site_id}/outputs/
```

## Установка

```bash
cd /var/www/app/log-analyzer-multi-site
uv sync
```

Требуется `node` для JS-скриптов (загрузка `.env` встроена в `js/load-env.js`).

## Запуск

```bash
# один сайт
uv run python run_site_pipeline.py 1_466 --force

# все изменившиеся
uv run python run_all_sites.py --rescan-only

# watcher
uv run python site_watcher.py
```

## Модели

### Dual output

В `predict_results.csv` и `{site_id}.out.log` две колонки скора (**breaking**: раньше UMAP писался в `probability`):

| Колонка | Источник |
|---------|----------|
| `probability_umap` | UMAP + расстояние до центроида заказов (обязательный) |
| `probability_lgbm` | LightGBM `predict_proba` (опциональный; пусто, если нет артефактов) |

Схема combined `out.log`:

```text
date,device_type,session_id,probability_umap,probability_lgbm,ip,user_agent
```

### Выбор модели при запуске

`site_registry.py` для каждого сайта и device (mobile / desktop) проверяет наличие **полного** набора артефактов:

| Device | Файлы |
|--------|-------|
| mobile | `order_umap_model.joblib`, `order_cluster_centroid.json`, `minmax_scaler.pkl` |
| desktop | `order_umap_model.joblib`, `order_cluster_centroid.json`, `minmax_scaler.pkl` |

**Порядок выбора:**

1. Полный набор в `sites/{site_id}/models/{mobile,desktop}/` → **кастомная** модель сайта.
2. Иначе полный набор в `models/{mobile,desktop}/` → **базовая** модель.
3. Если набор неполный — сайт использует base fallback для этого device; если нет и base — сайт пропускается при discovery.

Онлайн-пайплайн **не вызывает** утилиты обучения — только читает готовые файлы моделей.

### LightGBM buyer-модель (опционально)

Полный набор на device (custom → base):

| Файл | Назначение |
|------|------------|
| `buyer_lgbm.pkl` | LightGBM + MinMaxScaler nobot-признаков + `feature_columns` |
| `transformer_autoencoder.pt` | Transformer для эмбеддинга URL-токенов сессии |

Если LGBM-набора нет, пайплайн всё равно пишет `probability_umap`; `probability_lgbm` остаётся пустым.

### Обучение и публикация артефактов

Код обучения вынесен в [`log-analyzer-model-training`](../log-analyzer-model-training).  
Кратко:

```bash
cd ../log-analyzer-model-training

# base UMAP + LGBM → artifacts/, затем в models/
uv run python scripts/train_base_models.py --force --publish-to ../log-analyzer-multi-site
uv run python scripts/train_buyer_lgbm.py --target base --force --publish-to ../log-analyzer-multi-site

# custom для сайта (сначала прогоните run_site_pipeline.py здесь)
uv run python scripts/train_site_models.py 1_200 \
  --predict-root ../log-analyzer-multi-site \
  --publish-to ../log-analyzer-multi-site
```

Либо отдельно: `uv run python scripts/publish_artifacts.py --publish-to ../log-analyzer-multi-site`.

Вернуть сайт на базовую модель: удалить `sites/{id}/models/` (или только один device).

## Переменные окружения

| Переменная | По умолчанию | Описание |
|------------|--------------|----------|
| `MLOG_DIR` | `/var/www/mlog` | Каталог входных логов |
| `SITES_DIR` | `./sites` | Per-site workspace |
| `MAX_PARALLEL_SITES` | `1` | Параллелизм между сайтами |
| `DEBOUNCE_SECONDS` | `45` | Debounce inotify |
| `RESCAN_INTERVAL_SECONDS` | `3600` | Hourly rescan в watcher |
| `ALLOWED_SITE_IDS` | *(пусто)* | Список site_id через запятую; если задан — обрабатываются только эти логи (`1_466`, `1_200.json` и т.п.) |

### Ограничение списка логов

По умолчанию пайплайн обрабатывает все файлы `1_*.json` в `MLOG_DIR`. Чтобы ограничить набор сайтов, задайте в `.env`:

```bash
ALLOWED_SITE_IDS=1_466,1_200
```

Можно указывать идентификаторы с расширением или без: `1_466.json` и `1_466` эквивалентны. Watcher, `run_all_sites.py`, cron и ручной `run_site_pipeline.py` учитывают этот фильтр.

## Деплой

### Systemd

```bash
cp scripts/site-watcher.service /etc/systemd/system/
systemctl daemon-reload && systemctl enable --now site-watcher.service
```

### Cron

```cron
*/10 * * * * /var/www/app/log-analyzer-multi-site/scripts/cron_rescan_sites.sh
0 0 * * * /var/www/app/log-analyzer-multi-site/scripts/reset_out_log_daily.sh >> /var/www/app/log-analyzer-multi-site/watcher.log 2>&1
```

## Алгоритм

1. Node извлекает TMV/MMV + CLK + SCL по сессиям.
2. Preprocessor строит mean-only признаки и применяет MinMaxScaler (только transform).
3. `prediction.runner`: UMAP → `probability_umap`; LightGBM (если есть артефакты) → `probability_lgbm`; один CSV.
4. Postprocessor дописывает delta в out.log и history (если изменился любой из двух скоров).

Режим UMAP probability: `PROBABILITY_MODE=decay` (по умолчанию) или `linear`.

## Миграция со старого single-site проекта

1. Остановите cron/systemd на `log-analyzer-zwilling-combined-clusterbased`.
2. Запустите watcher и cron из этого проекта.
3. Опционально перенесите checkpoint'ы в `sites/1_466/checkpoints/`:
   - `split_records_checkpoint.json`
   - `tmv_process_checkpoint.json`
   - `mmv_process_checkpoint.json`

Старый проект после переключения **не нужен**.

## Диагностика

```bash
# список сайтов
uv run python -c "from site_registry import discover_sites; print([s.site_id for s in discover_sites()])"

# лог сайта
tail -100 sites/1_466/pipeline.log
```
