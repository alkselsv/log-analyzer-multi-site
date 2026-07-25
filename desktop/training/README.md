# Обучающие данные базовой desktop-модели

Положите сюда:

- `mmv_session_features_enhanced_mean_only_normalized.csv`
- `unique_order_sess_a.pkl` (или `.json` / `.txt`)

Переобучение:

```bash
cd /var/www/app/log-analyzer-multi-site
uv run python scripts/train_base_models.py --device desktop --force
```

Результат сохраняется в `desktop/models/` (UMAP + centroid).
