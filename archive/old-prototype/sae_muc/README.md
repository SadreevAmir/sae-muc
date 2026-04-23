# SAE latent MUC (отдельный вариант)

Код **самодостаточный внутри папки `sae_muc/`**: не импортирует `calibration/`, `src/`, `verbal_uncertainty/`, `sem_uncertainty/`. Вместо сырого VUF-вектора — **SAE encode → сдвиг латентов → decode + error** (`hooks.py`).

## Зависимости (только SAE)

```bash
cd /path/to/parent_of_sae_muc   # каталог, в котором лежит папка sae_muc/
pip install -r sae_muc/requirements.txt
```

Либо из корня репозитория: `pip install -r requirements.txt` (включает все зависимости).

**Данные для `run_muc.py`** лежат **рядом** с `sae_muc/` (не внутри неё): каталоги `datasets/` и `detection/` в том же стиле, что у Meta (см. `--repo_root`).

## 1. Собрать интервенцию из `Hs_hedge_universal.pt`

Для Mistral и релиза `mistral-7b-res-wg` (слои SAE → HF 7, 15, 23):

```bash
cd /path/to/parent_of_sae_muc
python -m sae_muc.build_intervention_config \
  --hedge_path calibration/outputs/merged/Mistral-7B-Instruct-v0.3/uncertainty/Hs_hedge_universal.pt \
  --out_path sae_muc/artifacts/mistral_intervention.pt \
  --release mistral-7b-res-wg \
  --top_k 64
```

Для **Llama 3.1 8B** (тот же `d_model`, что у Instruct) и релиза `llama_scope_lxr_32x` нужен `Hs_hedge` формы `[32, d_model]` под эту модель; затем `--release llama_scope_lxr_32x`, `--out_path .../llama_intervention.pt`.

## 2. Запуск MUC (iti_method=2 как в `semantic_control.py`)

Из **родителя** каталога `sae_muc` (там же должны быть `datasets/`, `detection/`):

```bash
python -m sae_muc.run_muc \
  --repo_root . \
  --dataset nq_open \
  --split test \
  --model_name Mistral-7B-Instruct-v0.3 \
  --prompt_type uncertainty \
  --str_process_layers 'range(15,32)' \
  --intervention_path sae_muc/artifacts/mistral_intervention.pt \
  --max_alpha 1.0
```

Если `datasets/` не рядом с `sae_muc`, укажите `--repo_root /path/to/project`.

## Размеченные признаки (Neuronpedia)

Индексы ненулевых координат в `delta` — это **номера латентов SAE**. Чтобы открыть страницы с **autointerp** и примерами, используйте [Neuronpedia](https://www.neuronpedia.org): URL вида `.../[MODEL_ID]/[SAE_ID]/[index]`.  
Для **`llama_scope_lxr_32x`** slug’и Neuronpedia подставляются из `layer_map.neuronpedia_residual_slug` (модель `llama3.1-8b`, SAE вида `15-llamascope-res-131k`). Для других релизов `SAE_ID` в URL может отличаться от id в SAELens — копируйте из адресной строки на сайте.

CLI для этого нет: импортируйте из `sae_muc.inspect_delta` функции `print_sparse_delta_report`, `neuronpedia_feature_urls`, `print_top_explanations` (см. `colab_sae_playground.ipynb`, секция 4b).

Результаты: `sae_muc/outputs/{dataset}/{model}/{prompt}/{split}/with_sae_muc_*.jsonl`.

## Colab

- SAE MUC: [`colab_sae_muc.ipynb`](../notebooks/colab_sae_muc.ipynb) (чекпоинты, Drive-бэкап).
- Песочница: [`colab_sae_playground.ipynb`](../notebooks/colab_sae_playground.ipynb).

Все ноутбуки находятся в папке [`notebooks/`](../notebooks/).

## Согласованность с `colab_pipeline.ipynb`

| Что в пайплайне Colab (фаза 7) | `sae_muc` |
|--------------------------------|-----------|
| `semantic_control.py`, `iti_method=2`, `α = clip(SU/MAX_SE − VU, 0, max_α)` | То же в `run_muc.py` (`MAX_SE`, формула, порог детектора `dr==0` → без генерации) |
| Детектор: `detection/LR_outputs/{ds}/{model}/{split}_verbal_uncertainty_sentence_semantic_entropy.json` | Тот же путь и поле `y_pred` |
| CSV: `verbal_uncertainty`, `sentence_semantic_entropy`, `question` | Те же колонки |
| Промпт MUC: `--prompt_type uncertainty` | Задаётся аргументом; **как в ячейке фазы 10** — `uncertainty` |
| Слои: `range(15,32)` | Дефолт тот же; с релизом **Mistral** SAE на **15 и 23** (слой 7 в range не входит). С **Llama Scope** в playground — те же HF-слои **15 и 23** |
| `generate_all_responses` из `calibration/causal.py` | Импортируется без изменений |
| Имя jsonl: `with_vufi_{iti}_{str_process_layers}_{max_alpha}.jsonl` | **Такое же имя** — можно подставить файл в `calibration/outputs/.../test/` и гонять **те же** `calibration/eval/*.py`, что в фазе 10.3 |
| `Hs_hedge_universal.pt` для MUC | Для SAE берите **`.../merged/{model}/uncertainty/Hs_hedge_universal.pt`** (не `sentence`), чтобы соответствовать промпту `uncertainty` в фазе 7 |

Опции **`--use_predicted`** и другие `iti_method` из `semantic_control.py` в `run_muc` пока **не** реализованы (только `iti_method=2`).

Писать сразу в каталог eval (перезапишет вывод vanilla MUC, если он там же):

```bash
python sae_muc/run_muc.py ... \
  --output_dir calibration/outputs/nq_open/Mistral-7B-Instruct-v0.3/uncertainty/test
```

## Ограничения

- SAE **Llama Scope** обучены на **базовой** Llama 3.1 8B; с **Instruct** возможен сдвиг распределения активаций (обычная практика для таких SAE).
- Релиз **Mistral**: SAE под базовый Mistral-7B, с Instruct — тот же риск.
- Интервенция ставится только на HF-слоях, для которых есть запись в `intervention.pt` и которые попали в `--str_process_layers`.
