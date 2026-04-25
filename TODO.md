# TODO


## ✅ P1 — done

### Post-intervention metrics (paper Tab.3 "after")
- [x] После `intervene` перепрогонять `judge` + `accuracy_judge` +
      `semantic_entropy` на каждом `intervention/alpha_*/generations.parquet`
      (или на `intervention/adaptive/generations.parquet`). Все три
      стадии теперь принимают путь к generations параметром.
- [x] `evaluate` выдаёт `metrics_before.json` (topmost `metrics.json`)
      + per-variant `intervention/<variant>/metrics.json` + сводную
      `metrics_comparison.parquet` (Tab.3 строка «before/after»).

### Server bootstrap (шаг 11 исходного плана)
- [x] `scripts/server_setup.md`: CUDA / torch install, uv sync, .env,
      `huggingface-cli login`, первый прогон под tmux, shared `$HF_HOME`.
- [x] `scripts/sync_artifacts.sh`: rsync parquet/json на локалку,
      `--heavy` включает safetensors.
- [x] `scripts/remote_run.sh` — ssh + tmux + sae-muc run с локалки.

### Composable configs (`configs/model/*`, `dataset/*`, `judge/*`, `nli/*`)
- [x] YAML-лоадер: `extends:` как список + ссылочные поля
      (`model: ../model/mistral7b.yaml`). Канонические файлы под
      Mistral-7B, Llama-3.1-8B, Qwen2.5-7B / 0.5B / fake, под
      TriviaQA/NQ-Open/PopQA/fake, под OpenRouter/CherryIn/fake судей,
      DeBERTa-v3-base/v2-xxlarge/fake NLI. Пример paper-scale
      конфига — `configs/experiment/qwen25_7b_triviaqa.yaml`.

---

## ✅ P2 — done

### Реальный SAE + `sae_emd` + `sae_clamp`
- [x] `SAELensBackend`: lazy sae-lens loader по `(release, sae_id)`.
      Auto-device (CUDA/MPS/CPU). sae-lens остаётся в optional
      `[sae]` extra.
- [x] `SAEConfig` в `ExperimentConfig.sae` + `ctx.sae` в
      `PipelineContext`; все SAE-хуки получают SAE через DI.
- [x] Стадия `sae_features`: pools hidden states → SAE encode →
      Cohen's d per latent → top-k positive как "uncertainty", top-k
      negative как "certainty". Артефакт
      `sae_features/stats.parquet`. Early-return если
      `intervene.method` не SAE (чтобы не падать на dim mismatch и не
      тратить forward впустую).
- [x] `sae_emd`: `f' = f + α·δ`, `h' = decode(f') + err`, где `δ`
      cohen-d-weighted + L2-norm (или multihot за флагом
      `cfg.stages.intervene.sae_emd_delta`). Закрыто review-фиксом S1.
- [x] `sae_clamp`: paper / old-prototype soft-push к
      `mean_uncertain[idx]` для uncertainty + soft-suppression
      `(1-α)` для certainty, α clipped в `[0,1]`. Закрыто review-фиксом S2.
- [x] `vuf` сохраняет `vuf/splits.parquet` чтобы `sae_features` не
      пересчитывала разбиение на uncertain/certain.

---

## ✅ Paper fidelity (review batch — done)

После критического ревью (см. `REVIEW.md`) закрыты:

- [x] **C1** — `SU_norm = SE/ln(N)` (paper App G.1) вместо min-max-нормализации.
- [x] **C2** — multi-layer intervention: `cfg.stages.intervene.layer` принимает
      `int | list[int] | "auto" | "paper_range"`. App E.1 ranges в
      `pipeline/paper_layer_ranges.py` (Llama 15-31, Mistral 15-31, Qwen 16-27).
- [x] **C3** — paper §4.1 hidden-state hallucination detector + intervention
      gating: `gate_by_detector=True` пропускает интервенцию для безопасных
      сэмплов и переиспользует baseline.
- [x] **C4 + S4** — `VUFStage.selection: top_n | vu_threshold` (default
      `vu_threshold` для mitigation, paper App G.1). `sae_features`
      автоматически наследует через `vuf/splits.parquet`.
- [x] **C5** — пометка в docstring config / detect / evaluate, что
      `refusal_vu_threshold=0.85` — наша калибровка, не paper.
- [x] **C6 + I2 + N1** — `EvaluateStage` с Kossen-style threshold modes
      (`vu/su_threshold_mode: kossen | fixed | median`). `evaluate_post`
      переиспользует пороги baseline. Docstring evaluate.py починен.
- [x] **C7** — `HiddenStatesStage.storage: full | question_only | last_k_tokens`,
      `last_k`. Default `full`. Закрывает scale-item ниже.
- [x] **C8** — `cfg.seed` теперь прокидывается в `LLMBackend.generate(seed=...)` и
      `generate_with_hook(...)`; HF backend делает `torch.manual_seed(seed)`
      перед `do_sample=True .generate()`. Remote API игнорят seed.
- [x] **S1** — δ для `sae_emd` весом cohen_d + L2-нормировка
      (`cfg.stages.intervene.sae_emd_delta`).
- [x] **S2** — `_build_sae_clamp_hook` переписан под soft-push
      (per-feature `mean_uncertain[idx]`) + soft-suppression. `sae_clamp_target`
      из конфига убран.
- [x] **S3** — `cfg.stages.sae_features.selection_mode: topk | consensus`
      (bootstrap + BH-FDR + |d|, как в old-prototype).
- [x] **P1** (review) — `cfg.stages.intervene.apply_during_generation` +
      skip-for-seq_len-1 в хуке.
- [x] **N2** — лог числа SAE-фич per-layer в начале `intervene.run`.
- [x] **N3** — `_pool` / `_resolve_layer(s)` вынесены в `pipeline/_utils.py`.
- [x] **N5** — комментарий о `model.model.layers[i]` пути в HF backend.
- [x] **N6** — `~fillna(True)` → `(is_correct == False)` в evaluate.py.
- [x] **I3** — assert `ctx.sae.d_in == d_model` перед `sae_features.encode`.

---

## P3 — paper fidelity (нужно для воспроизведения цифр Tab.1–3)

### Paper dataset splits — **review-flagged P3**
- [ ] Использовать фиксированные train/val/test из Appendix B
      (10k/1k/1k на каждый датасет), а не `shuffle(seed).select(range(n))`.
      Иначе выборка вопросов не совпадает с paper и цифры не сравнимы.
      **Без этого Tab.1/Tab.3 не воспроизводимы.**

### VUF layer auto-selection (paper Fig.4) — **review-flagged P4**
- [ ] Считать cosine-similarity VUF'ов между датасетами и выбирать слой
      с максимальной согласованностью. Сейчас либо все слои
      (`layers: auto`), либо ручное указание в конфиге, либо
      `paper_range` по App E.1. После C2 multi-layer auto должен
      возвращать paper-диапазон / cross-dataset-stable layer set,
      а не середину диапазона.

### popqa loader — проверка на живом датасете (review-flagged P5)
- [ ] `_load_popqa` в `data/loaders.py:102-122` использует
      `possible_answers` или `obj`. Не проверял на живом датасете.
      **Это не код, а проверка перед первым popqa-ран'ом**: прогнать
      `prepare.run` на `dataset: popqa` с `n_samples=10`, посмотреть
      `gold_answers` глазами и сверить с
      [akariasai/PopQA](https://huggingface.co/datasets/akariasai/PopQA).
      Если поля другие — обновить `_load_popqa`.

### Hedging prompt vs accuracy judge (review-flagged P6) — **paper-scale важно**
- [ ] `generate.py:27` и `hidden_states.py:48` используют
      `eliciting=True` prompt («precisely hedging»). Это правильно для
      VU-измерения (App A.1). Но `accuracy_judge` потом сравнивает
      hedged-ответ с golden как «семантически эквивалентен»; если
      модель оборачивает корректный ответ в «I'm not sure, but…», LLM-as-judge
      может его отвергать. На paper-scale эффект может смазать
      `correct_rate`. Идея: подать в `accuracy_judge` голую модель без
      hedging-prompt, либо явно сказать судье принимать hedged-форму.
      Замерять delta до/после на 50 sample subset.

---

## P4 — scale (важно, когда пойдём на 1k+ вопросов)

### Parallelism in judge / accuracy_judge — **review-flagged N4** (приоритет)
- [ ] `concurrent.futures.ThreadPoolExecutor` на 8–16 воркеров.
      1000 вопросов × 11 генераций = 11k вызовов, сейчас последовательно
      ~20–40 мин; параллельно 2–5 мин, при соблюдении rate limits.
      **Узкое место всех paper-scale ранов** — без него experiment
      iteration loop неудобен.

### Mid-stage checkpointing
- [ ] Промежуточный `save_parquet` в `judge` / `accuracy_judge` каждые
      ~50 успешных вызовов. Падение SSH в середине 20-минутной judge-стадии
      сейчас теряет все в-буфере результаты — пер-prompt isolation
      покрывает только 1 зафэйленный вызов, не краш процесса.

### HF generation speed (нужен для локального LLM на сервере)
- [ ] FlashAttention-2 (`attn_implementation="flash_attention_2"`).
- [ ] `num_return_sequences=N` в одном `model.generate` вместо N
      последовательных вызовов в adaptive-ветке `intervene`.
- [ ] **Multi-question batching with left-padding (review-flagged P2 — приоритет)**.
      Сейчас в intervene adaptive один промпт за раз, т.к. у каждого свой α.
      На paper-scale (1k вопросов × 11 генераций × 2 cycle [before/after])
      это десятки минут → часы. Группировать вопросы по совпадающему α
      (или дискретизировать α на bins), как `get_answers_muc` в old prototype
      ([archive/old-prototype/sae_muc/run_muc.py:278-332](archive/old-prototype/sae_muc/run_muc.py#L278-L332)).
- [ ] `bitsandbytes` 4/8-bit за флагом конфига — только при нехватке VRAM.

### Scale artefacts
- [x] `hidden_states.storage: last_k_tokens` для k=16. При 10k сэмплов
      полная последовательность × 32 слоя × d_model × fp16 = 130+ ГБ на
      прогон, last_k_tokens ужмёт в ~3×. **Закрыто review-фиксом C7**:
      доступны `full | question_only | last_k_tokens`.
- [ ] Streaming-loader для датасетов (сейчас читаем всё в память).

### Pre-download weights to shared `$HF_HOME`
- [ ] Документировать в QUICKSTART: при нескольких разработчиках на одном
      сервере использовать shared `$HF_HOME=/opt/hf-cache` чтобы не качать
      Mistral-7B / DeBERTa по второму разу.

---

## P5 — ergonomics (качество жизни; делать по потребности)

- [ ] `sae-muc join <run_id>` CLI: слить все parquet'ы run'а в одну
      широкую таблицу (samples ⋈ generations ⋈ judge ⋈ se ⋈ accuracy)
      для ad-hoc анализа в ноутбуке.
- [ ] Optional Weights & Biases integration gated by `WANDB_API_KEY`.
