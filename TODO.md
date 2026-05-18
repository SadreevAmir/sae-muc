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

## P3 — paper fidelity (нужно для воспроизведения цифр Tab.1–3)

### Per-layer SAE registry — **HIGH PRIORITY** (блокер multi-layer SAE)
- [ ] `SAEConfig` хранит одну `(release, sae_id)`, `ctx.sae` — один объект
      на весь рантайм. SAE-методы (`sae_emd`, `sae_clamp`, `sae_projected`)
      и стадия `sae_features` на multi-layer (`intervene.layer: list[int]`)
      сейчас вызывают **тот же `ctx.sae` на всех слоях**, но Gemma-Scope /
      Llama-Scope SAE — **per-layer** (каждый `layer_X/...` обучен на
      residual'е именно слоя X). На слоях ≠ trained_layer encode/decode
      даёт OOD-шум; multi-layer SAE-интервенция paper App E.1
      (Llama 15-31 / Qwen 16-27) сейчас **невозможна корректно**.

      Нужно:
      1. Расширить `SAEConfig`: либо `layers: dict[int, {release, sae_id}]`,
         либо общий `release` + `sae_id_template: "layer_{layer}/width_16k/canonical"`.
      2. `ctx.saes: dict[int, SAEBackend]` lazy-load по слою (не грузить
         всё сразу — каждая SAE ~50-200MB).
      3. Поправить `intervene._build_per_layer_hooks` и `sae_features.run`,
         чтобы брали `ctx.saes[layer]` вместо `ctx.sae`.
      4. Validation: каждый `ctx.saes[layer].d_in == d_model` и `layer`
         совпадает со слоем хука.
      5. Тесты (FakeSAE per-layer mock на 2-3 слоях).

      Workaround до фикса: SAE-методы на одном слое, multi-layer только
      `linear_vuf` (см. `configs/experiment/gemma2_2b_multilayer_smoke.yaml`).

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
