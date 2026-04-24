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
      мульти-hot (+1 на uncertainty, -1 на certainty).
- [x] `sae_clamp`: uncertainty features → α·target (configurable),
      certainty features → 0.
- [x] `vuf` сохраняет `vuf/splits.parquet` чтобы `sae_features` не
      пересчитывала разбиение на uncertain/certain.

---

## P3 — paper fidelity (нужно для воспроизведения цифр Tab.1–3)

### Kossen-style пороги для метрик
- [ ] Заменить `vu_threshold=0.5` в `evaluate` на порог, подобранный
      минимизацией суммы квадратов расстояний от VU до порога
      (Kossen et al. 2024). То же для SU-порога (сейчас медиана).

### Paper dataset splits
- [ ] Использовать фиксированные train/val/test из Appendix B
      (10k/1k/1k на каждый датасет), а не `shuffle(seed).select(range(n))`.
      Иначе выборка вопросов не совпадает с paper и цифры не сравнимы.

### VUF layer auto-selection (paper Fig.4)
- [ ] Считать cosine-similarity VUF'ов между датасетами и выбирать слой
      с максимальной согласованностью. Сейчас либо все слои
      (`layers: auto`), либо ручное указание в конфиге.

---

## P4 — scale (важно, когда пойдём на 1k+ вопросов)

### Parallelism in judge / accuracy_judge
- [ ] `concurrent.futures.ThreadPoolExecutor` на 8–16 воркеров.
      1000 вопросов × 11 генераций = 11k вызовов, сейчас последовательно
      ~20–40 мин; параллельно 2–5 мин, при соблюдении rate limits.

### Mid-stage checkpointing
- [ ] Промежуточный `save_parquet` в `judge` / `accuracy_judge` каждые
      ~50 успешных вызовов. Падение SSH в середине 20-минутной judge-стадии
      сейчас теряет все в-буфере результаты — пер-prompt isolation
      покрывает только 1 зафэйленный вызов, не краш процесса.

### HF generation speed (нужен для локального LLM на сервере)
- [ ] FlashAttention-2 (`attn_implementation="flash_attention_2"`).
- [ ] `num_return_sequences=N` в одном `model.generate` вместо N
      последовательных вызовов в adaptive-ветке `intervene`.
- [ ] Multi-question batching с left-padding (сейчас в intervene adaptive
      один промпт за раз, т.к. у каждого свой α).
- [ ] `bitsandbytes` 4/8-bit за флагом конфига — только при нехватке VRAM.

### Scale artefacts
- [ ] `hidden_states.storage: last_k_tokens` для k=16. При 10k сэмплов
      полная последовательность × 32 слоя × d_model × fp16 = 130+ ГБ на
      прогон, last_k_tokens ужмёт в ~3×.
- [ ] Streaming-loader для датасетов (сейчас читаем всё в память).

### Torch RNG seeding (всплыло при smoke-анализе)
- [ ] `cfg.seed` пока только seed'ит numpy/sklearn/HF shuffle; для
      `model.generate(do_sample=True)` RNG не фиксирован → второй
      проход (post-intervention) даёт шум даже при α=0. Поставить
      `torch.manual_seed(cfg.seed)` в начале каждого generate-вызова.
      На больших выборках шум усредняется, но на smoke сравнение
      before/after с α=0 визуально "что-то поменялось" — неправда.

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
