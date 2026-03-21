# sae-muc

Пакет [`sae_muc/`](sae_muc/) — интервенция в латентном пространстве предобученного SAE (SAELens, релиз `mistral-7b-res-wg`), совместимая по путям данных с [verbal_uncertainty_feature_calibration](https://github.com/facebookresearch/verbal_uncertainty_feature_calibration).

```bash
pip install -r sae_muc/requirements.txt
cd /path/to/project   # рядом лежат datasets/ и detection/
python -m sae_muc.build_intervention_config --help
python -m sae_muc.run_muc --help
```

Детали: [sae_muc/README.md](sae_muc/README.md).
