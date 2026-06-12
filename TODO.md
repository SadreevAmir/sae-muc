# TODO


## ⬜⬜ HIGH — share the method-independent upstream across runs

Runs that differ only in `intervene.method` (e.g. linear_vuf vs sae_emd on the
same model/dataset/layers) recompute an IDENTICAL upstream — `prepare → generate
→ judge(VU) → accuracy_judge → semantic_entropy → hidden_states → vuf → detect`
(~3.5 h at N=2500: generate ~70 min + VU-judge pre ~73 min + SE ~1 h + …). Only
`sae_features → intervene → evaluate → *_post → diagnostics` actually differ.
Today we pay that upstream once per method.

Why it can't be done by a plain `--run-id` resume now:
- **Manifests are existence-only, no config hash** (`artifacts/manifest.py`): the
  runner can't tell which stages a `method` change invalidates, so it would skip
  the wrong things (also the root cause of the alpha_max-resume caveat).
- **`sae_features` runs as a no-op stub on a linear run**, so resuming sae into a
  linear run dir would SKIP it and never compute the real consensus features.

Fix options (pick one):
1. **Config-hash-aware per-stage manifests** — each stage records a hash of its
   relevant config slice + its input manifests; re-runs iff that hash changed.
   General fix: changing `method` correctly invalidates `sae_features`/`intervene`
   onward while keeping `generate…detect` valid. Also fixes alpha_max resume.
2. **Explicit upstream import** — `upstream_from: <run_id>` config field (or
   `--reuse-upstream <run_id>`) that symlinks/imports the shared upstream
   artefacts so only the method-specific tail runs.

Payoff scales with N and with the number of method / alpha_max variants (a sweep
re-pays the whole upstream per variant today). NOTE: when the two runs execute in
PARALLEL on separate GPUs the duplicated upstream is free in wall-clock — this
optimisation mainly helps sequential / single-card and multi-variant sweeps.


## ⬜ Multi-GPU (deferred — single-card assumption baked in)

- [ ] `models/hf_local.py` pins the model to `cuda:0` (single visible card via
      Docker `--gpus device=N`), NOT `device_map="auto"`. To run a model sharded
      across several cards: (1) revert to `device_map="auto"`, (2) make SAE
      (`models/sae.py`) and NLI (`models/nli.py`) placement device-aware — they
      hardcode `.cuda()` == cuda:0, which would force cross-device hops in the
      intervene hook, (3) drop the single-card `--gpus device=N` restriction in
      `scripts/docker/run.sh` / `shell.sh`.


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

## ✅ Per-category VUF: ABSTAIN vs HEDGE disentanglement (done)

Supervisor pointed out that the paper's `uncertain = {VU ≥ 0.9}` bucket
silently mixes "I don't know" (ABSTAIN) and "I think X" (HEDGE) — both
have decisiveness ≈ 0, so the linear VUF averages two distinct circuits.
This is the most likely cause of paper Tab.3's Correctness Rate drop:
the abstain-half of the direction makes the model refuse correct answers
alongside the desired Confident Hallucination Rate decrease.

- [x] New stage `categorize_uncertainty` between `hidden_states` and `vuf`.
      Opt-in via `cfg.stages.categorize.enabled` (default False ⇒
      byte-identical back-compat). LLM-as-judge with a 4-shot
      ABSTAIN/HEDGE/CONFIDENT prompt at
      `data/prompts.py:CATEGORIZE_PROMPT`. Reuses `ctx.judge` — no
      new backend.
- [x] Per-generation classification → per-question majority vote
      (≥60% threshold, else MIXED). CONFIDENT-tagged generations are
      kept for inspection but excluded from the vote (judge said this
      gen doesn't actually look uncertain). Vote counts persisted for
      future re-thresholding.
- [x] **Certain-bucket questions are NOT categorised** — ~50% saving on
      judge calls. Labels there are meaningless and would just add noise.
- [x] `vuf.run` gained `per_category` flag. When on + labels available,
      builds `r_abstain^(l)` and `r_hedge^(l)` against the **shared**
      certain pool (cosine isolates abstain-vs-hedge axis from the
      generic certainty axis). Skips a category with <2 members.
- [x] Per-category direction metadata in a **separate** parquet
      `vuf/category_meta.parquet` so existing `vuf/meta.parquet`
      consumers (`intervene`, `sae_features`, `diagnostics`) see no
      duplicate layer rows. Main direction filename unchanged →
      byte-identical checksums when per_category is off.
- [x] `diagnostics` gained `category_directions.parquet` — per-layer
      cosines `cosine_abstain_hedge`, `cosine_abstain_main`,
      `cosine_hedge_main` + pool sizes. The disentanglement question
      gets a scalar answer per layer.
- [x] Floating-point fix in `_split_ids_by_threshold`: `groupby.mean()`
      of identical values can shift by a ULP (mean([0.05, 0.05, 0.05]) ==
      0.05000000000000001), so user threshold `vu_certain_max=0.05`
      silently dropped boundary rows. Added 1e-9 epsilon. **Pre-existing
      bug** caught by category tests — fixed.
- [x] 20 unit-tests + 244/244 full suite green.
- [x] Docs: README pipeline-diagram, QUICKSTART "Per-category VUF" with
      ready-to-paste Python snippet, configs/README entries for
      `stages.categorize` + `vuf.per_category`.

Iteration 2 (done): cross direction + per-category SAE features:
- [x] `r_abstain_vs_hedge = normalize(r_abstain − r_hedge)` persisted as
      `vuf/direction_abstain_vs_hedge_layer_{L}.safetensors`. This is the
      direct steering axis for flipping uncertain *type* (abstain ↔ hedge)
      without changing overall uncertainty level. Algebraically derived
      from the two per-category directions, so no extra hidden-state pass.
- [x] `diagnostics/category_directions.parquet` gained `cosine_cross_main`
      column — measures whether paper's lumped VUF is accidentally
      encoding the type-flip axis (high) or only the
      uncertain-vs-certain axis (low).
- [x] `sae_features.run` extended: when `vuf.per_category` is on and
      `vuf/splits.parquet` carries category labels, runs additional
      Cohen's d per category against the shared certain pool. Output:
      `sae_features/category_stats.parquet` (separate parquet, legacy
      `stats.parquet` schema preserved for `sae_emd`/`sae_clamp` hooks).
      Selection mode locked to `topk` here (consensus filter on small
      per-category buckets is too noisy).
- [x] `diagnostics/category_sae_jaccards.parquet` — per-layer Jaccard
      between top-K feature sets across main / abstain / hedge variants.
      Two-view triangulation: if VUF cosines say "same axis" but SAE
      Jaccards say "disjoint features", that's the publish-worthy
      result ("SAE picks up structure linear probes miss").
- [x] 29 unit-tests + 253/253 full suite green.

Deferred extensions (next iteration if cosines/jaccards show disentanglement):
- [ ] Per-category intervention in `intervene.run` — `α_abstain·r_abstain
      + α_hedge·r_hedge` as a single hook, or steering on the cross
      direction `α·r_abstain_vs_hedge`. Config shape reserved as
      `intervene.directions: list[str]` + `intervene.alphas: list[float]`
      but not implemented.
- [ ] Per-category SAE intervention — `sae_emd` / `sae_clamp` reading
      from `category_stats.parquet` instead of (or alongside) `stats.parquet`.
- [ ] Additional categories: RANGE (numeric imprecision), CONDITIONAL
      ("it depends"), UNIVERSAL ("experts disagree"). MVP only needs
      ABSTAIN vs HEDGE because they're the two dominant types in
      closed-book short-form QA.
- [ ] LLM-judge agreement audit: sample ~50 generations, hand-label them,
      compare with the categorize judge. Calibrate the prompt if
      precision/recall on either category drops below ~80%.

---

## ✅ Diagnostics: intervention side-effect on general LM capability (done)

Paper Tab.3 меряет только task-specific damage (Correctness Rate / Refusal
Rate on the same QA dataset). Авторы не меряли perplexity / KL drift — only
empirical `max_α` tuning per-model (App G.1). Научрук попросила проверить,
не плывёт ли perplexity от наших интервенций. Стандарт steering-литературы
(CAA, ITI, RepE, ReFT) — WikiText-2 perplexity + KL divergence.

- [x] Стадия `diagnostics` в конце пайплайна (после `evaluate_post`). Sidecar
      артефакты под `diagnostics/...`; не трогает существующие. Skippable via
      `stages.diagnostics.enabled=false` или для backend-ов без logits API
      (openrouter).
- [x] `forward_nll_with_hook` + `forward_kl_with_hook` на `HFLocalBackend`:
      stride-512 NLL с overlap-маской (-100, HF perplexity recipe); KL в
      float32, softmax/log в fp32 даже под bf16 logits; `padding_side=right`
      для KL-форварда; KL(p_base || p_int) + top-1 disagreement + top-5
      mass shift.
- [x] Пер-вариантная reconstruction хуков: чистая `intervene.build_per_layer_hooks`
      с явными аргументами; читает variant row из `intervention/meta.parquet`,
      а не `ctx.cfg.stages.intervene` (которая могла поменяться на re-run).
- [x] Адаптивные варианты: оцениваются один раз на `mean_alpha` (per-sample
      diagnostics — N форвардов на вариант — deferred, см. ниже).
- [x] FakeBackend стабы: probe-pattern (`ones(1,1,D)` через хук), identity
      hook ⇒ KL == 0 ровно; SAE round-trip ⇒ KL ≈ 0 atol=1e-3.
- [x] Артефакты: `diagnostics/perplexity.parquet` (one row per variant +
      baseline), `diagnostics/kl.parquet`, `diagnostics/summary.json`.
- [x] 16 unit-тестов на FakeBackend; полный test suite зелёный (215/215).

Post-supervisor-feedback extension (done):
- [x] **MMLU / HellaSwag / GSM8K (No-CoT)** scorer'ы — `pipeline/diagnostics_datasets.py`.
      MC scoring через NLL argmin (paper-faithful approach из lm-eval-harness);
      GSM8K brief greedy generation + numeric parsing. Subset 200 questions
      default (paper-style steering side-effect probes использует 200-500).
- [x] **Multi-method × α sweep** в одном run — `compare_methods + alpha_sweep`
      конфиг-флаги. Стадия строит хуки для каждой (method, α) пары из тех же
      VUF directions / SAE feature stats и пишет
      `diagnostics/method_alpha_sweep.parquet` long format. Cross-run workflow
      (один method per run) тоже остался — это default.
- [x] **`sae_features` un-gating**: запускается также если diagnostics sweep
      требует SAE-методы (не только если основная intervene.method — SAE).
- [x] Документация: README pipeline diagram, QUICKSTART new section
      "Diagnostics artefacts" с готовыми Python-сниппетами для cross-run
      merge + sweep pivot, configs/README со всеми knobs.

Deferred extensions (отдельной итерацией, по запросу научрука):
- [ ] **GSM8K с CoT 8-shot** — paper-faithful generation (Arditi 2024 style).
      Дороже на ~10× (~30-60 min/variant vs ~5 min), но absolute accuracy
      выше и α-дельта чище. No-CoT NLL уже даёт ppl-vs-α signal, CoT нужен
      когда захотим публиковать absolute accuracy numbers.
- [ ] **TruthfulQA + ARC + HumanEval** — Arditi 2024 полный набор. Дальше
      MMLU/HellaSwag/GSM8K добавляются дёшево (тот же MC scoring шаблон),
      HumanEval требует sandbox для execution.
- [ ] Generation health probes на не-QA промптах: refusal-rate / repetition /
      token-entropy (30-prompt curated set).
- [ ] Второй корпус (C4 small slice) как cross-check против WikiText
      memorization (известный риск для моделей, обученных на Common Crawl).
- [ ] Per-sample адаптивная диагностика — сейчас один forward на `mean_alpha`.
      Per-sample N forward-проходов даст распределение damage по α-квантилям,
      нужно если адаптивные runs покажут heavy tails ppl-drift.

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

## ✅ Per-layer SAE registry (done — commit `ccd4319`)

Multi-layer SAE intervention (paper App E.1: Llama 15-31, Mistral 15-31,
Qwen 16-27) requires a **separate SAE per residual-stream layer** —
Gemma-Scope / Llama-Scope train one SAE per layer, and reusing one across
layers is OOD noise. Closed:

- [x] `SAEConfig` hybrid schema: `sae_id` (legacy single layer) +
      `sae_id_template` (e.g. `"layer_{layer}/width_16k/canonical"`) +
      `sae_id_overrides: dict[int, str]` (sparse releases like
      `mistral-7b-res-wg` covering only {8, 16, 24}). Resolution priority
      via `models.sae.resolve_sae_id_for_layer`: overrides > template > legacy.
- [x] `ctx.saes: dict[int, SAEBackend]` built eagerly in
      `pipeline.context.build_context` over candidate layers [0, 96); sae-lens
      weights still load lazily on first encode (wrappers are cheap).
- [x] `intervene._build_per_layer_hooks` and `sae_features.run` take the SAE
      per layer from `ctx.saes[layer]` (never a global `ctx.sae`).
- [x] `assert_sae_layers_available(cfg, target_layers)` validator called
      at the top of `intervene.run` and `sae_features.run` after
      `_resolve_layers` and before any forward. Also wired into
      `diagnostics.run` for the in-run sweep when `compare_methods`
      includes SAE methods.
- [x] FakeSAE per-layer registry test coverage; `mistral7b_sae_sparse_smoke.yaml`
      + `gemma2_2b_sae_multilayer_smoke.yaml` validated against the live
      sae-lens registry. End-to-end CUDA smoke on Mistral sparse — see
      P5 «Live smoke на Mistral-7B + sparse SAE overrides».

---

## P3 — paper fidelity (нужно для воспроизведения цифр Tab.1–3)

### SAE activation normalization — investigation (может влиять на результаты)
- [ ] Разобраться, что `cfg.normalize_activations` делает в каждом релизе и
      как это взаимодействует с нашим `sae_emd` / `sae_clamp` / `sae_projected`.
      sae-lens StandardSAE имеет 3 ветки в `sae_lens/saes/sae.py:319-356`:
      `"constant_norm_rescale"` (per-token L2-rescale к √d_in), `"layer_norm"`
      (классический LN с mu/std), и no-op (identity) — у разных релизов
      встречаются все три.

      Важные точки разобрать:
      1. **Когда state SAE-объекта (x_norm_coeff / ln_mu / ln_std) корректен,
         а когда стейл.** Сейчас в commit'е 1c21e9f мы пропатчили
         `run_time_activation_norm_fn_out` — убрали `del self.x_norm_coeff`,
         чтобы второй `decode()` в одном hook не падал. Это безопасно если
         `f` и `f_new` имеют одинаковую shape; если когда-то будем менять
         batch/seq между парами encode/decode — патч сломается тихо.
      2. **Какой режим у каждого pretrained релиза, который мы используем.**
         `gemma-scope-2b-pt-res-canonical` — no-op (нормализация запечена в
         веса DeepMind'ом при тренировке). `mistral-7b-res-wg` —
         `constant_norm_rescale`. Llama-Scope — TBD, проверить перед первым
         Llama-смок'ом. Qwen-andyrdt, Pythia — TBD.
      3. **Корректность `sae_emd` δ под constant_norm_rescale.** Мы строим
         δ из cohen_d на латентах, потом `f' = f + α·δ`. Но если у нас
         нормализатор на входе (`x' = x · c`), то `f` обусловлено
         нормированным входом, и `f'` живёт в той же нормированной системе.
         Когда `decode` денормирует через `/c`, эффект нашего сдвига
         в residual stream получает фактор `1/c`. Не уверен — надо
         посчитать на бумаге, что мы реально подмешиваем в residual для
         двух разных режимов normalize_activations и эквивалентны ли они.
      4. **Cohen's d в `sae_features` под нормализацией.** Encode'им
         pooled hidden states одной партией — `x_norm_coeff` пересчитывается
         на каждом encode. Нормализация per-batch, и Cohen's d считается
         между двумя группами с разными средними норм — может дать
         артефакты, не отражающие реальное различие в латентах.
      5. **Old prototype** (archive/old-prototype) — как они работали с
         нормализацией? Возможно использовали другой режим, и наши
         результаты на Mistral расходятся именно из-за этого.

      Никакого блокера; просто здоровая паранойя — paper-сравнения через
      MUC с реальными SAE могут смещаться от выбора `normalize_activations`,
      и нужно понимать когда наш sae_emd-сдвиг семантически тот же, что
      paper'ная linear_vuf-интервенция, а когда нет.

### Paper dataset splits — **review-flagged P3**
- [ ] Использовать фиксированные train/val/test из Appendix B
      (10k/1k/1k на каждый датасет), а не `shuffle(seed).select(range(n))`.
      Иначе выборка вопросов не совпадает с paper и цифры не сравнимы.
      **Без этого Tab.1/Tab.3 не воспроизводимы.**

### VUF layer auto-selection (paper Fig.4) — **review-flagged P4**
- [~] Cross-dataset cosine-similarity VUF'ов теперь считается:
      `combine_vuf` пишет `vuf/cross_dataset_cosine.parquet` (per-layer mean
      pairwise cosine источников) — количественная половина §3.2. Осталось:
      авто-выбор слоёв ПО этому сигналу (сейчас читается глазами; band берётся
      из `paper_range`/App E.1). PCA-separability — качественная визуализация
      в статье, не алгоритм.

### popqa loader — проверка на живом датасете (review-flagged P5)
- [ ] `_load_popqa` в `data/loaders.py:102-122` использует
      `possible_answers` или `obj`. Не проверял на живом датасете.
      **Это не код, а проверка перед первым popqa-ран'ом**: прогнать
      `prepare.run` на `dataset: popqa` с `n_samples=10`, посмотреть
      `gold_answers` глазами и сверить с
      [akariasai/PopQA](https://huggingface.co/datasets/akariasai/PopQA).
      Если поля другие — обновить `_load_popqa`.

### Hedging prompt vs accuracy judge (review-flagged P6) — **paper-scale важно**
- [x] РЕШЕНО через `generate.prompt_regime="split"` (default): plain-промпт
      (App A.1 box 1) генерирует ответы для SU/accuracy/most-likely, eliciting
      (box 2) — только для VU/VUF. `accuracy_judge` / `semantic_entropy` теперь
      читают plain-set, `hidden_states` остаётся на eliciting (§3.1). Steered
      MUC-генерация — нейтральный промпт (§2.15). `eliciting_only` оставлен для
      дешёвых smoke-прогонов. См. `_utils.select_prompt_kind`.

### EigenScore baseline (Table 2) — **DEFERRED, внешний**
- [ ] Статья цитирует EigenScore как бейзлайн Table 2, но формулы не даёт —
      это INSIDE (Chen et al. 2024, ICLR). Считается по ковариации эмбеддингов
      K *сэмплированных ответов* (EigenScore = (1/K)·logdet(Σ+αI)), а пайплайн
      хранит hidden только для вопроса (+greedy), не для 10 сэмплов. Нужен
      отдельный forward-проход по текстам сэмплов из `generations.parquet` —
      добавляется в любой момент без потери данных. На предлагаемый детектор
      не влияет; SEP (paper-описанный) реализован, EigenScore — нет.

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

## P5 — ergonomics & follow-ups (качество жизни; делать по потребности)

### GPU index mapping на caniculus — открытый вопрос
- [ ] `scripts/server_setup.md:121-125` утверждает что `nvidia-smi` index ===
      Docker `--gpus device=N` (verified 2026-04-27). Это **работает**:
      все наши последние смоки запускались `scripts/docker/run.sh 4 ...`
      и Mistral/Gemma реально клались на `cuda:0` внутри контейнера, а
      `nvidia-smi` снаружи показывал GPU 4 занятой.

      Открытый момент: была ранняя гипотеза о перестановке индексов
      между nvtop и nvidia-smi (зафиксирована в memory как
      `0→1, 1→3, 2→4, 3→5, 4→6, 5→2, 6→0`). Источник этой таблицы — какой-то
      ранний эксперимент Амира?, направление стрелки и применимость к
      текущему стейту так и не верифицировали. Наша эмпирика 2026-05
      указывает на 1-к-1, но без понимания, **откуда взялся тот mapping**,
      непонятно: это было артефактом старого драйвера, конкретного
      `nvtop`'а, переноса hardware между PCIE-слотами, или просто
      неправильно запомнено?

      Что сделать:
      1. Прогнать сейчас простой эксперимент: на каждом из 7 индексов
         запустить mini-run, в логе записать `torch.cuda.get_device_name(0)`
         и сравнить с `nvidia-smi --query-gpu=index,name --format=csv`.
         Сейчас один-в-один, или порядок отличается?
      2. Если сейчас 1-к-1 — обновить `project_server_gpu_mapping`
         memory ("проверено YYYY-MM, mapping 1-к-1") и удалить таблицу.
      3. Если порядок другой — задокументировать таблицу с правильным
         направлением стрелки и добавить assertion в `scripts/docker/run.sh`.

### Из ревью-сессии — проверки и hardening
- [ ] **Live smoke на Qwen2.5-0.5B перед первым серверным ран'ом.**
      На FakeBackend проверены multi-layer hooks (C2), `last_k_tokens`
      slicing (C7), `torch.manual_seed` (C8); реальное поведение на CUDA
      / на реальной архитектуре HF-модели не запускалось.
- [ ] **Live smoke на Mistral-7B + sparse SAE overrides** (per-layer SAE
      registry, sparse-coverage кейс). `mistral-7b-res-wg` покрывает только
      слои 8/16/24; resolver/validator/registry проверены unit-тестами и
      validator пройден против живого реестра sae-lens, но end-to-end на
      CUDA с реальной загрузкой 2-3 SAE не запускался. План: собрать
      `configs/experiment/mistral7b_sae_sparse_smoke.yaml` с
      `sae_id_overrides: {16: ..., 24: ...}` и `intervene.layer: [16, 24]`,
      прогнать на сервере, убедиться что в логах intervene видно загрузку
      разных sae_id на разных слоях, а `sae_features/stats.parquet`
      содержит строки для обоих слоёв.
- [ ] **Migration script для legacy `intervention/meta.parquet`.**
      После C2 колонка `layer: int` → `layers: str`. Старые ран-папки
      несовместимы при дозапуске `evaluate_post`. Делать **только если
      есть legacy данные**, которые нельзя перепрогнать.
- [ ] **Mandatory pre-check на `detection.parquet`** в начале
      `intervene.run` при `gate_by_detector=True`. Сейчас падает с
      `FileNotFoundError` в `load_parquet`, можно дать дружелюбную
      ошибку «detect-стадия не была запущена».
- [ ] **Гибкость поведения при `detect skipped + gate_by_detector=True`.**
      Сейчас `is_at_risk = False` для всех (default «не вмешиваемся при
      неуверенности»). Может потребоваться флаг
      `if_detect_skipped: gate_off | apply_to_all` если хочется иначе.
- [ ] **Granular `paper_layer_ranges.py` substring matching.** Сейчас
      Llama-8B и Llama-70B попадают в одну ветку (15-31). App E.1 даёт
      разные диапазоны для разных размеров — добавить точные подстроки.
- [ ] **`last_k_tokens` edge case** при `pooling=last_token_q` если
      вопрос обрезан раньше окна: текущий fallback берёт первый токен
      ответа. Поведение задокументировано; добавить assert/warning,
      когда `last_k < question_len`, или менять pooling автоматически.

### Раннее существовавшее
- [ ] `sae-muc join <run_id>` CLI: слить все parquet'ы run'а в одну
      широкую таблицу (samples ⋈ generations ⋈ judge ⋈ se ⋈ accuracy)
      для ad-hoc анализа в ноутбуке.
- [ ] Optional Weights & Biases integration gated by `WANDB_API_KEY`.
