# TODO

Items intentionally deferred out of the MVP. Revisit after the baseline
pipeline passes end-to-end.

## Paper fidelity — unimplemented / simplified vs Ji et al. 2025

### Post-intervention metrics (paper Tab.3)
- [ ] After `intervene`, re-run `judge` / `accuracy_judge` / `semantic_entropy`
      on the intervened generations (one batch per α in fixed mode, one
      batch in adaptive mode), then compute metrics on each.
- [ ] Extend `evaluate` to emit `metrics_before.json` + one
      `metrics_after_alpha_{a}.json` (or `metrics_after_adaptive.json`) per
      intervention variant and a combined comparison table.
- [ ] Without this, the pipeline produces intervened generations but no
      measurable hallucination reduction — the whole MUC claim is
      observable only via eyeballing the output texts.

### SAE branch (our own extension)
- [ ] SAE feature-analysis stage: SAE-encode hidden states, compute Cohen's
      d / t-test / bootstrap-stable feature indices, write
      `sae_features/meta.parquet`. See
      `archive/old-prototype/sae_muc/build_intervention_config_v2.py` for
      reference code.
- [ ] `sae_emd` intervention: `f' = f + α · δ`, `h' = decode(f') + err`.
      Needs the feature-analysis output.
- [ ] `sae_clamp` intervention: raise uncertainty features to a target,
      suppress certainty features.
- [ ] Real SAE loading via `sae-lens` in `HFLocalSAEBackend` (currently
      `NotImplementedError`). Adds `sae-lens>=6.0` as a required dep (today
      it is in the optional `sae` extra).

### Metric thresholds (paper §4.2)
- [ ] Confident-hallucination VU threshold: paper fits it by minimising
      the sum of squared distances from VU values to the threshold
      (Kossen et al. 2024 style). We hard-code `vu_threshold=0.5`.
- [ ] SU threshold: paper uses the same Kossen method. We use the median
      of `semantic_entropy` in the current run.

### Dataset protocol (paper Appendix B)
- [ ] Use paper's fixed train/val/test splits for TriviaQA (10k/1k/1k),
      NQ-Open (10k/1k/1k), PopQA (10k/1k/1k). Currently we just
      `shuffle(seed).select(range(n))` which picks different questions.

## Ergonomics and speed

### Parallelism
- [ ] Judge / accuracy_judge call concurrency. At 1k questions × 11 gens
      = 11k sequential API calls is 20–40 min; a `concurrent.futures`
      thread pool (8–16 workers, respecting OpenRouter rate limits) would
      cut that dramatically.

### Retry stacking
- [ ] openai SDK default `max_retries=2` stacks with our
      `_create_with_retry(max_retries=3)` → up to 9 attempts per call.
      On a stuck provider this wastes 5–10 minutes. Consider either
      disabling SDK retries (per-backend) or shrinking our loop.

### Checkpoint during judge stage
- [ ] Partial `judge_scores.parquet` flush every N calls so a crash does
      not lose in-flight progress. Per-prompt isolation (already done)
      catches single-call failures, but a process kill still loses
      everything not yet written.

## Correctness / generality

### Seeding torch RNG
- [ ] `cfg.seed` is applied to numpy / sklearn / HF dataset shuffling but
      NOT to torch's global RNG. Generation at T=1.0 is therefore not
      repeatable. Add `torch.manual_seed(cfg.seed)` at the start of each
      generate call (local scope, to avoid leaking to unrelated code).

### Layer path for non-Llama models
- [ ] `HFLocalBackend.generate_with_hook` uses
      `self._model.model.layers[i]` — only works for Llama / Mistral /
      Qwen2 family. For GPT-2 / T5 / Bloom we'd need a layer-map (see
      `archive/old-prototype/sae_muc/layer_map.py` for reference).

### VUF layer auto-selection
- [ ] Paper Fig.4 selects the best layer per model by measuring
      cosine-similarity of VUFs extracted from different datasets.
      We currently either iterate all layers (`layers: auto`) or expect
      the user to pick. Automating the "pick the layer where VUFs are
      most consistent" is a straightforward addition.

## LLM speed and memory (existing list)


## LLM speed and memory

- [ ] FlashAttention-2: `attn_implementation="flash_attention_2"` on
      `HFLocalBackend` load. Needs `flash-attn` added to deps.
- [ ] bf16 by default on modern GPUs (A100/H100/Ada); keep fp16 fallback.
- [ ] Batching: 10 sampled answers via `num_return_sequences=10` in one
      `model.generate` call, not 10 sequential calls.
- [ ] Multi-question batching with left-padding so the last-token position
      aligns across the batch; batch size tuned per GPU.
- [ ] `bitsandbytes` 8-bit / 4-bit quantisation behind a config flag.
      Only enable when VRAM forces it; measure quality regression.
- [ ] Pre-download weights to a shared `$HF_HOME` on the server; document
      in `QUICKSTART.md`.
- [ ] vLLM / SGLang as an optional drop-in for generation-only stages
      (prepare→generate, not hidden_states / intervene). Requires a thin
      adapter in `models/`.
- [ ] Determinism: fix seeds for `torch`, `numpy`, `random`; document that
      GPU cublas/cudnn are not bit-exact.
- [ ] Profile end-to-end with `torch.profiler` once baseline works;
      re-prioritise the items above based on the real bottleneck.

## Pipeline ergonomics

- [ ] Locally-triggered SSH runs (`sae-muc remote run ...`): rsync configs
      to server, `ssh … tmux … python -m sae_muc run …`, rsync results back.
- [ ] `sae-muc join <run_id>` CLI: build a single wide table from all stage
      parquets for ad-hoc analysis.
- [ ] `sae-muc status <run_id>` CLI: list which stages are done, stale,
      or missing.

## Experiment tracking

- [ ] Optional Weights & Biases integration gated by `WANDB_API_KEY`.
      Not needed for MVP.

## Data / scale

- [ ] `storage: last_k_tokens` mode for hidden-state artefact when scaling
      beyond ~1k samples per run.
- [ ] Streaming loaders for large splits (currently assume full load into
      memory).

## Testing

- [ ] GPU smoke test with a tiny open model (Qwen2.5-0.5B or distilgpt2) on
      the server — end-to-end pipeline in ≤ 2 minutes on 5 samples.
