# log-analyzer-multi-site

Автономный мультисайтовый пайплайн прогнозирования вероятности заказа по mobile- и desktop-сессиям.

Проект **не зависит** от других репозиториев: весь код обработки, модели, утилиты переобучения и оркестратор находятся в этом каталоге.

## Назначение

В `/var/www/mlog` лежат NDJSON-логи вида `1_466.json`, `1_200.json` — по одному файлу на сайт. Проект:

1. **обнаруживает** логи автоматически;
2. **запускает** инкрементальный пайплайн mobile + desktop для каждого сайта;
3. **изолирует** checkpoint'ы и промежуточные файлы в `sites/{site_id}/`;
4. **пишет** итоговый журнал в `/var/www/mlog/{site_id}.out.log`.

Алгоритм: UMAP + расстояние до центра кластера заказов (см. раздел «Алгоритм»).

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
├── order_probability.py         # decay/linear probability
├── split_records.js             # split mobile/desktop
├── compare_probability_distributions.py
├── model_training/              # общая логика обучения UMAP + centroid
│   ├── config.py
│   ├── paths.py
│   ├── train.py
│   └── cli.py
├── mobile/                      # TMV pipeline
│   ├── run_pipeline.py
│   ├── models/                  # базовые runtime-артефакты
│   ├── training/                # обучающие данные base-модели
│   └── src/                     # process, preprocessor, predict, postprocessor
├── desktop/                     # MMV pipeline
│   ├── run_pipeline.py
│   ├── models/
│   ├── training/
│   └── src/
├── sites/{site_id}/             # per-site workspace и outputs
└── scripts/
    ├── train_site_models.py     # custom-модели для одного сайта
    ├── train_base_models.py     # переобучение базовых моделей
    ├── train_order_cluster.py   # unified CLI (--device mobile|desktop)
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
        ├── mobile/run_pipeline.py  ─┐ parallel
        └── desktop/run_pipeline.py ─┘
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

### Выбор модели при запуске

`site_registry.py` для каждого сайта и device (mobile / desktop) проверяет наличие **полного** набора артефактов:

| Device | Файлы |
|--------|-------|
| mobile | `order_umap_model.joblib`, `order_cluster_centroid.json`, `minmax_scaler_mobile.pkl` |
| desktop | `order_umap_model.joblib`, `order_cluster_centroid.json`, `minmax_scaler.pkl` |

**Порядок выбора:**

1. Полный набор в `sites/{site_id}/models/{mobile,desktop}/` → **кастомная** модель сайта.
2. Иначе полный набор в `{mobile,desktop}/models/` → **базовая** модель.
3. Если набор неполный — сайт использует base fallback для этого device; если нет и base — сайт пропускается при discovery.

Онлайн-пайплайн **не вызывает** утилиты обучения — только читает готовые файлы моделей.

### Базовая модель (одна на все сайты)

Обучающие данные кладутся в `{mobile,desktop}/training/` (не в `src/utils/`):

```
mobile/training/tmv_session_features_enhanced_mean_only_normalized.csv
mobile/training/unique_order_sess_a.pkl
desktop/training/mmv_session_features_enhanced_mean_only_normalized.csv
desktop/training/unique_order_sess_a.pkl
```

Переобучение UMAP и центроида:

```bash
cd /var/www/app/log-analyzer-multi-site
uv run python scripts/train_base_models.py --force
```

Только один device:

```bash
uv run python scripts/train_base_models.py --device mobile --force
```

Альтернатива — unified CLI:

```bash
uv run python scripts/train_order_cluster.py mobile --target base --force
```

Результат сохраняется в `mobile/models/` и `desktop/models/` (UMAP + centroid; scaler для base уже должен быть в `models/`).

Без `--force` базовые модели **не перезаписываются** — скрипт завершится с ошибкой, если файлы уже существуют.

**Важно:** обучение выполняется только через `scripts/train_*.py`. Scaler для base-модели утилита **не создаёт** — файл `minmax_scaler*.pkl` нужно иметь в `{device}/models/` отдельно.

Каталог `sites/*/models/` при этом не трогать — все сайты продолжат использовать базовую модель.

### Кастомная модель для конкретного сайта

Пример для сайта `1_200`.

#### 1. Получить признаки сайта

Сначала прогоните пайплайн, чтобы в workspace появились CSV:

```bash
cd /var/www/app/log-analyzer-multi-site
uv run python run_site_pipeline.py 1_200 --force
```

Нужные файлы:

```
sites/1_200/workspace/tmv_session_features_enhanced_mean_only_normalized.csv   # mobile
sites/1_200/workspace/mmv_session_features_enhanced_mean_only_normalized.csv   # desktop
```

#### 2. Подготовить список сессий с заказами

Положите `unique_order_sess_a.pkl` (или `.json` / `.txt`) в каталог сайта:

```
sites/1_200/workspace/unique_order_sess_a.pkl
# или sites/1_200/training/unique_order_sess_a.pkl
```

#### 3. Обучить custom-модели (базовые не трогаются)

```bash
cd /var/www/app/log-analyzer-multi-site
uv run python scripts/train_site_models.py 1_200
```

Скрипт сохраняет полный набор артефактов сразу в:

```
sites/1_200/models/mobile/
sites/1_200/models/desktop/
```

Для каждого device создаются: `order_umap_model.joblib`, `order_cluster_centroid.json` и копия базового `minmax_scaler*.pkl`.

#### Альтернатива: обучить один device через unified CLI

```bash
cd /var/www/app/log-analyzer-multi-site
uv run python scripts/train_order_cluster.py mobile \
  --site-id 1_200 \
  --orders sites/1_200/workspace/unique_order_sess_a.pkl
```

Базовые файлы в `{device}/models/` при этом **не перезаписываются**.

#### 4. Проверить

```bash
uv run python -c "
from site_registry import get_site
s = get_site('1_200')
print('mobile:', s.mobile_models.source)
print('desktop:', s.desktop_models.source)
"
# Ожидается: mobile: custom, desktop: custom
```

После этого watcher и `run_site_pipeline.py` для `1_200` автоматически используют модели из `sites/1_200/models/`.

### Шпаргалка

| Цель | Действие |
|------|----------|
| Одна модель на всех | данные в `{device}/training/` → `scripts/train_base_models.py --force` → `{device}/models/` |
| Своя модель для сайта | `uv run python scripts/train_site_models.py {id}` |
| Вернуть сайт на базовую | удалить `sites/{id}/models/` (или только один device) |
| Сравнить режимы probability | `uv run python compare_probability_distributions.py` |

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
3. Predict: UMAP transform → расстояние до центроида → probability (`order_probability.py`).
4. Postprocessor дописывает delta в out.log и history.

Режим probability: `PROBABILITY_MODE=decay` (по умолчанию) или `linear`.

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
