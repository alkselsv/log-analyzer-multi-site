# Базовая desktop-модель (runtime)

Артефакты inference для всех сайтов без custom-модели:

- `order_umap_model.joblib`
- `order_cluster_centroid.json`
- `minmax_scaler.pkl`

Переобучение: `uv run python scripts/train_base_models.py --device desktop --force`

Обучающие данные лежат в `desktop/training/`, не здесь.
