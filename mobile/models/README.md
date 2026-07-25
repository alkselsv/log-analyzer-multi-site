# Базовая mobile-модель (runtime)

Артефакты inference для всех сайтов без custom-модели:

- `order_umap_model.joblib`
- `order_cluster_centroid.json`
- `minmax_scaler_mobile.pkl`

Переобучение: `uv run python scripts/train_base_models.py --device mobile --force`

Обучающие данные лежат в `mobile/training/`, не здесь.
