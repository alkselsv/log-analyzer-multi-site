# Обучающие данные базовой mobile-модели

Положите сюда:

- `tmv_session_features_enhanced_mean_only_normalized.csv`
- `unique_order_sess_a.pkl` (или `.json` / `.txt`)

Переобучение:

```bash
cd /var/www/app/log-analyzer-multi-site
uv run python scripts/train_base_models.py --device mobile --force
```

Результат сохраняется в `mobile/models/` (UMAP + centroid).
