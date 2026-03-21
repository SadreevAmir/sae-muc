# sae-muc

Пакет [`sae_muc/`](sae_muc/) — интервенция в латентном пространстве предобученного SAE (SAELens, релиз `mistral-7b-res-wg`), совместимая по путям данных с [verbal_uncertainty_feature_calibration](https://github.com/facebookresearch/verbal_uncertainty_feature_calibration).

**Colab (в корне репозитория):**

- [`colab_sae_playground.ipynb`](colab_sae_playground.ipynb) — быстрые сравнения до/после интервенции; первый шаг клонирует этот же репозиторий в `/content/sae-muc`.
- [`colab_sae_muc.ipynb`](colab_sae_muc.ipynb) — полный пайплайн (Drive-бэкап, `build_intervention_config`, `run_muc`); данные VUF (`datasets/`, `detection/`, `calibration/`) нужно положить в корень клона или задать `--repo_root`.

```bash
pip install -r sae_muc/requirements.txt
cd /path/to/project   # рядом лежат datasets/ и detection/
python -m sae_muc.build_intervention_config --help
python -m sae_muc.run_muc --help
```

Детали: [sae_muc/README.md](sae_muc/README.md).
