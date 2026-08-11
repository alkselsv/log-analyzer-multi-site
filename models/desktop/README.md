Базовые runtime-артефакты desktop:

UMAP bot-cluster (обязательные):
- `bot_umap_model.joblib`
- `bot_cluster_centroid.json`
- `minmax_scaler.pkl`

LightGBM buyer (опциональные):
- `buyer_lgbm.pkl`
- `transformer_autoencoder.pt`

Файлы в этом каталоге не коммитятся. Обучение и публикация — в соседнем проекте:

```bash
cd ../log-analyzer-model-training
uv run python scripts/train_base_models.py --force --publish-to ../log-analyzer-multi-site
uv run python scripts/train_buyer_lgbm.py --device desktop --target base --force \
  --publish-to ../log-analyzer-multi-site
```
