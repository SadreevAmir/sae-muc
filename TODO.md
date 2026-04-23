# TODO

Items intentionally deferred out of the MVP. Revisit after the baseline
pipeline passes end-to-end.

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
