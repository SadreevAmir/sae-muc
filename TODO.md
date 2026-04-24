# TODO


## P1 — блокирует основной результат

### Post-intervention metrics (paper Tab.3 "after")
Без этого MUC неизмеряем: мы строим интервентные генерации, но не
мерим на них hallucination_rate / correlation / disagreement.

- [ ] После `intervene` перепрогонять `judge` + `accuracy_judge` +
      `semantic_entropy` на каждом `intervention/alpha_*/generations.parquet`
      (или на `intervention/adaptive/generations.parquet`). Нужен
      рефактор: все три стадии должны принимать путь к generations
      параметром (сейчас прибиты к `generations.parquet`).
- [ ] `evaluate` выдаёт `metrics_before.json` + `metrics_after_*.json`
      + сводную `metrics_comparison.parquet` (Tab.3 строка «before/after»).

### Server bootstrap (шаг 11 исходного плана)
Без этого коллега не поднимет прогон на сервере.

- [ ] `scripts/server_setup.md`: CUDA / torch install, uv sync, .env,
      `huggingface-cli login`, первый прогон под tmux. Поставить `$HF_HOME`
      на shared-путь чтобы не скачивать веса дважды.
- [ ] `scripts/sync_artifacts.sh`: rsync `data/runs/<id>/{metrics.json,*.parquet}`
      обратно на локалку.
- [ ] Optional: `scripts/remote_run.sh` — ssh + tmux + sae-muc run из
      одной команды с локалки.

### Composable configs (`configs/model/*`, `dataset/*`, `judge/*`)
Сейчас каждый experiment-yaml дублирует model / dataset / judge inline.
При 3 моделях × 3 датасетах × 2 судьях это 18 почти одинаковых файлов.

- [ ] Расширить YAML-лоадер: `extends:` как список + ссылочные поля
      (`model: model/mistral7b.yaml`). Создать канонические файлы под
      Mistral-7B, Llama-3.1-8B, Qwen2.5-7B (как в статье), под
      TriviaQA/NQ-Open/PopQA, под OpenRouter/CherryIn судей.

---

## P2 — SAE-ветвь (наше собственное расширение)

### Реальный SAE + `sae_emd` + `sae_clamp`
Сейчас `sae_projected` работает на FakeSAE (случайная проекция) —
полезно как proof-of-life хука, но не как научный результат.

- [ ] `HFLocalSAEBackend`: sae-lens loader по `(release, sae_id)`.
      Переводит `sae-lens` из `[project.optional-dependencies.sae]` в
      required deps.
- [ ] Стадия `sae_features`: прогнать hidden_states через SAE, посчитать
      per-feature Cohen's d / t-test / bootstrap-stability (reference —
      `archive/old-prototype/sae_muc/build_intervention_config_v2.py`),
      отобрать consensus uncertainty-features и certainty-features.
      Артефакт: `sae_features/meta.parquet`.
- [ ] `sae_emd`: `f' = f + α·δ`, `h' = decode(f') + err`, где `δ` — это
      one-hot по отобранным фичам.
- [ ] `sae_clamp`: uncertainty features → высокое значение,
      certainty features → 0.

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
