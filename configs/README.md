# Configs

Composable YAML configs for sae-muc experiments. Resolved via
[`load_experiment_config`](../src/sae_muc/config.py) into a frozen
[`ExperimentConfig`](../src/sae_muc/config.py) Pydantic model
(`extra="forbid"` — unknown keys fail loudly).

## Layout

```
configs/
  base.yaml              # shared defaults (seed, NLI default, stage defaults)
  model/                 # ModelConfig fragments — one per generator
  dataset/               # DatasetConfig fragments
  judge/                 # JudgeConfig fragments
  nli/                   # NLIConfig fragments
  experiment/            # full ExperimentConfig — these are what you pass to --config
```

Run with:

```bash
sae-muc run --config configs/experiment/<name>.yaml
```

## How composition works

There are two mechanisms that merge YAML files:

### 1. `extends:` — list-or-string of parents to deep-merge first

```yaml
extends: ../base.yaml             # single parent (string)
# OR
extends:                          # multiple parents, later wins
  - ../base.yaml
  - ./shared/some_overrides.yaml
```

Resolution: each parent is loaded recursively (their own `extends:` are
resolved first), parents are merged left-to-right, and finally **the
current file's keys override the merged parents**. Dicts merge deeply;
scalars and lists overwrite.

### 2. Reference fields — string value points at a fragment YAML

For four sections (`model`, `dataset`, `judge`, `nli`), a string value is
treated as a relative path to a fragment file:

```yaml
model: ../model/qwen2_5_7b.yaml          # loaded as the model section
judge: ../judge/openrouter_llama31_70b.yaml
```

Inline dicts work too (use them when a one-off override is shorter than
copying a fragment):

```yaml
dataset:
  name: nq_open
  split: validation
  n_samples: 10
  seed: 0
```

## Sections (top-level fields of `ExperimentConfig`)

| Field | Required | Type | Notes |
|---|---|---|---|
| `model` | yes | [`ModelConfig`](../src/sae_muc/config.py) | provider ∈ {hf_local, openrouter, fake} |
| `dataset` | yes | [`DatasetConfig`](../src/sae_muc/config.py) | name ∈ {triviaqa, nq_open, popqa, fake} |
| `judge` | yes | [`JudgeConfig`](../src/sae_muc/config.py) | provider ∈ {openrouter, cherryin, fake} |
| `nli` | no  | [`NLIConfig`](../src/sae_muc/config.py) | default DeBERTa-v2-xxlarge |
| `sae` | no  | [`SAEConfig`](../src/sae_muc/config.py) | default `provider: fake` (linear_vuf doesn't need it) |
| `stages` | no | [`StagesConfig`](../src/sae_muc/config.py) | per-stage knobs; defaults in `base.yaml` |
| `seed` | no | int (default 42) | propagated to dataset shuffle, RNG, generate(seed=…) |
| `data_root` | no | path (default `data`) | run dir = `<data_root>/runs/<run_id>/` |

## Stage flags reference

The full schema lives in [`config.py`](../src/sae_muc/config.py). Highlights:

### `stages.generate` — `GenerateStage`
- `n_samples: int = 10` — paper default, count of T=1.0 samples for SE.
- `temperature_low / temperature_high: float` — for greedy / sampled.
- `max_new_tokens: int = 100`.

### `stages.hidden_states` — `HiddenStatesStage`
- `storage: "full" | "question_only" | "last_k_tokens"` — what to keep
  per sample. `last_k_tokens` saves disk on long contexts.
- `last_k: int = 8` (only when `storage="last_k_tokens"`).

### `stages.vuf` — `VUFStage`
- `layers: list[int] | "auto"` — layers to extract a VUF for. `auto` =
  every layer present in `hidden_states/`.
- `selection: "top_n" | "vu_threshold"` — how to split uncertain/certain.
  `top_n` (paper §3.1, validation) vs `vu_threshold` (paper App G.1, mitigation).
  Default `vu_threshold` (matches the mitigation pipeline).
- `n_top / n_bot: int` — for `top_n`.
- `vu_uncertain_min / vu_certain_max: float` — for `vu_threshold`.
- `pooling: "last_token_q" | "last_token_a" | "mean_q" | "mean_a"` — how
  to pool the per-token hidden state. Paper default `last_token_q`.

### `stages.intervene` — `InterveneStage`
- `method: "linear_vuf" | "sae_emd" | "sae_clamp" | "sae_projected"` —
  hook style. `linear_vuf` doesn't need an SAE; the three SAE methods do.
- `mode: "fixed" | "adaptive"` — α policy. `fixed` sweeps `alpha_grid`;
  `adaptive` is paper Eq.5-6 MUC.
- `alpha_grid: list[float]` — fixed-mode sweep.
- `alpha_max: float` — adaptive-mode cap (paper App G.1: Llama 1.0,
  Mistral 0.4, Qwen 3.0, Llama-70B 4.0).
- `layer: int | list[int] | "auto" | "paper_range"` — where the hook
  fires. `paper_range` looks up App E.1 by `model.name` substring
  (Llama / Mistral 15-31, Qwen 16-27).
- `gate_by_detector: bool = False` — paper §4.2 only intervenes on
  detected hallucinations. When True, the adaptive run reads
  `detection.parquet` and reuses baseline for safe samples.
- `apply_during_generation: bool = True` — paper-faithful (steers every
  token). Set False for back-compat with the old prototype's prefill-only
  hooks.
- `sae_emd_delta: "cohen_d" | "multihot"` — δ-vector shape for sae_emd.

### `stages.detect` — `DetectStage`
- `refusal_vu_threshold: float = 0.85` — **NOT from the paper** (our
  calibration; paper doesn't pin a number).
- `detector_method: "lr_vu_se" | "lr_hidden" | "combined"` — paper Tab.1
  reports `lr_hidden` is best.
- `detector_layer: int | "auto"` — for the hidden-state probe.

### `stages.sae_features` — `SAEFeaturesStage`
- `k_top: int = 50` — top-k features per direction.
- `selection_mode: "topk" | "consensus"` — `consensus` is the old
  prototype's bootstrap + BH-FDR + |d| filter (more stable, slower).

### `stages.evaluate` — `EvaluateStage`
- `vu_threshold_mode / su_threshold_mode: "fixed" | "kossen" | "median"` —
  threshold for "Confident Hallucination Rate" in Tab.3. `kossen` is the
  paper-faithful default (mean of the baseline distribution).

### `stages.diagnostics` — `DiagnosticsStage`

Sidecar stage: intervention side-effect probes — perplexity, MMLU,
HellaSwag, GSM8K (No-CoT), KL — paper doesn't run any of these. The
standard steering-literature triad (Arditi 2024, CAA, ITI, RepE).
See [QUICKSTART#diagnostics-artefacts](../QUICKSTART.md#diagnostics-artefacts).

- `enabled: bool = true` — set `false` to skip the stage entirely (CI,
  remote-only backends, etc.).
- `corpora: list[Literal["wikitext","mmlu","hellaswag","gsm8k"]]` —
  which benchmarks to score. Default: all four. Remove any to save time.
- `corpus_n_chars: int = 300_000` — WikiText-2-raw-v1 validation cap.
  `0` ⇒ synthetic test corpus (no HF download).
- `n_mmlu / n_hellaswag / n_gsm8k: int = 200` — subset size per
  benchmark. 200 ≈ ±2-3% absolute-accuracy noise, enough for relative
  damage measurement; bump to 500 for more stable numbers, 1000 for
  publication-grade.
- `gsm8k_max_new_tokens: int = 32` — brief greedy generation budget for
  the No-CoT GSM8K scorer.
- `kl_max_prompts: int = 100` — KL is computed on the first N QA prompts
  from `samples.parquet`.
- `compare_methods: list[Literal["linear_vuf","sae_emd","sae_clamp","sae_projected"]]`
  — when non-empty, the stage also runs an in-run sweep over these
  methods. Empty (default) ⇒ per-variant from `intervention/meta.parquet`
  only (the cross-run workflow).
- `alpha_sweep: list[float]` — α values for the in-run sweep. Empty ⇒
  sweep disabled regardless of `compare_methods`.

### `sae` — `SAEConfig`

Per-layer SAE selection (Gemma-Scope and Llama-Scope train one SAE per
residual layer; reusing a layer-15 SAE on layer 20 is OOD noise). Three
fields, resolved with priority **overrides > template > legacy `sae_id`**:

- `sae_id: str | None` — single SAE for the whole run; legacy back-compat
  for one-layer configs.
- `sae_id_template: str | None` — string with `{layer}` placeholder,
  expanded per requested layer.
- `sae_id_overrides: dict[int, str]` — per-layer exceptions (sparse releases).
- `release: str | None` — sae-lens release name (required for `sae_lens`
  provider).
- `provider: "fake" | "sae_lens"` — `fake` is in-process random projection
  for tests; `sae_lens` loads pretrained.
- `d_in / d_latent / seed` — only relevant for `provider: fake`.

Validation: at the top of `intervene.run` and `sae_features.run`, the
helper [`assert_sae_layers_available`](../src/sae_muc/models/sae.py)
checks every requested layer exists in the sae-lens registry for
`release`. Paper-range that drifts past the release coverage **fails
loud**, no silent narrowing. See [`project_sae_registry.md`](https://example.invalid)
in auto-memory for the per-release map (Gemma-Scope, Llama-Scope,
mistral-7b-res-wg, llama-3-8b-it-res-jh).

---

## Cookbook — recipes

### Recipe 1: fake-only smoke (no network, no GPU)

```yaml
extends: ../base.yaml

model: ../model/fake.yaml
dataset: ../dataset/fake.yaml
judge: ../judge/fake.yaml
nli: ../nli/fake.yaml

stages:
  generate: {n_samples: 3, max_new_tokens: 16}
  vuf: {n_top: 2, n_bot: 2, pooling: last_token_q}
  intervene: {method: linear_vuf, mode: fixed, alpha_grid: [0.0, 1.0]}

seed: 42
```

Used by [`fake_smoke.yaml`](experiment/fake_smoke.yaml) and the integration test.

### Recipe 2: real model, single-layer SAE (Gemma-Scope smoke)

```yaml
extends: ../base.yaml

model: ../model/gemma2_2b_base.yaml
judge: ../judge/openrouter_qwen35_flash.yaml
nli: ../nli/deberta_v3_base.yaml

dataset: {name: nq_open, split: validation, n_samples: 10, seed: 0}

sae:
  provider: sae_lens
  release: gemma-scope-2b-pt-res-canonical
  sae_id: layer_20/width_16k/canonical          # legacy single-layer

stages:
  generate: {n_samples: 4, max_new_tokens: 40}
  vuf: {layers: [20], selection: top_n, n_top: 4, n_bot: 4}
  sae_features: {k_top: 8}
  intervene:
    method: sae_emd
    mode: fixed
    alpha_grid: [0.0, 1.0]
    layer: 20

seed: 42
```

See [`gemma2_2b_sae_smoke.yaml`](experiment/gemma2_2b_sae_smoke.yaml).

### Recipe 3: multi-layer SAE via template (Gemma-Scope or Llama-Scope)

```yaml
sae:
  provider: sae_lens
  release: gemma-scope-2b-pt-res-canonical
  sae_id_template: "layer_{layer}/width_16k/canonical"

stages:
  vuf: {layers: [18, 19, 20, 21, 22]}
  intervene:
    method: sae_emd
    layer: [18, 19, 20, 21, 22]
```

The template is expanded once per requested layer
(`layer_18/width_16k/canonical`, …, `layer_22/...`). For Llama-Scope use
`release: llama_scope_lxr_32x` with `sae_id_template: "l{layer}r_32x"`.

See [`gemma2_2b_sae_multilayer_smoke.yaml`](experiment/gemma2_2b_sae_multilayer_smoke.yaml).

### Recipe 4: sparse-coverage release via overrides (Mistral)

`mistral-7b-res-wg` ships only layers 8/16/24 with non-templatable
sae_ids (`blocks.N.hook_resid_pre`). Use explicit overrides and an
explicit layer list:

```yaml
sae:
  provider: sae_lens
  release: mistral-7b-res-wg
  sae_id_overrides:
    8:  "blocks.8.hook_resid_pre"
    16: "blocks.16.hook_resid_pre"
    24: "blocks.24.hook_resid_pre"

stages:
  vuf: {layers: [8, 16, 24]}
  intervene:
    method: sae_emd
    mode: adaptive
    alpha_max: 0.4                              # paper App G.1 for Mistral-7B
    layer: [8, 16, 24]
```

See [`mistral7b_sae_sparse_smoke.yaml`](experiment/mistral7b_sae_sparse_smoke.yaml)
for the full paper-shape (n=50, adaptive MUC, combined detector).

Don't pass `intervene.layer: paper_range` here — the resolver will fail
loud on the layers without an override (15, 17, 18, …), which is the
intended behaviour: paper_range with sparse SAE coverage is genuinely
incomplete.

### Recipe 5: paper-scale `linear_vuf` on `paper_range`

```yaml
extends: ../base.yaml

model: ../model/llama3_1_8b.yaml
judge: ../judge/openrouter_llama31_70b.yaml
nli: ../nli/deberta_v2_xxlarge.yaml

dataset: {name: triviaqa, split: validation, n_samples: 1000, seed: 42}

stages:
  generate: {n_samples: 10, max_new_tokens: 100}
  vuf: {layers: auto, n_top: 250, n_bot: 250, pooling: last_token_q}
  intervene:
    method: linear_vuf
    mode: adaptive                               # paper Eq.5-6 MUC
    alpha_max: 1.0                               # paper App G.1: Llama-3.1-8B
    layer: paper_range                           # 15-31 for Llama, 16-27 for Qwen

seed: 42
```

`linear_vuf` doesn't need an SAE, so `sae:` is omitted (defaults to
`provider: fake`, untouched). `mode: adaptive` is the MUC headline result.

---

## Validation, errors, and debugging

- **Frozen + extra=forbid**: typos in field names raise at config load.
  `pydantic.ValidationError: 1 validation error for ExperimentConfig`.
- **`assert_sae_layers_available`** runs at the start of `intervene.run`
  / `sae_features.run` for SAE methods and lists exactly which layers
  were missing — no silent narrowing.
- **`paper_range` mismatch**: if the resolved range doesn't intersect
  `available` VUF layers, `_resolve_layers` raises with the available
  list in the message ([`pipeline/_utils.py`](../src/sae_muc/pipeline/_utils.py)).
- **Resolved-config artefact**: every run dumps `data/runs/<run_id>/config.resolved.json`
  with the merged + validated config — handy when an `extends:` chain
  is hard to read by hand.
- **Hash-stable run_id**: the SHA-256 of the resolved config feeds the
  run_id suffix; identical configs reuse the same run dir
  (manifest-based stage skipping kicks in).

## Adding a new fragment

1. Drop a new YAML in `model/` / `dataset/` / `judge/` / `nli/` with
   only that section's fields (no `extends:`, no other top-level keys).
2. Reference it from an experiment file:
   ```yaml
   model: ../model/your_model.yaml
   ```
3. If the fragment introduces a new provider/name not yet recognised by
   the schema, `extra="forbid"` will reject it — extend the `Literal[…]`
   in [`config.py`](../src/sae_muc/config.py) and the corresponding
   [`models/registry.py`](../src/sae_muc/models/registry.py) /
   loader first.

## Known gotchas

- **`auto` for layer is a single layer**, not a range. It picks the
  middle of the available VUF layers, which is rarely what paper-scale
  experiments want — prefer `paper_range` or an explicit list.
- **`vuf.selection` default is `vu_threshold`** (mitigation), not
  `top_n`. On small smokes the threshold-mode split is often empty and
  falls back to `top_n` with a warning — set `selection: top_n`
  explicitly to keep smokes deterministic.
- **`intervene.gate_by_detector` is honoured only in `mode: adaptive`**.
  Setting it under `mode: fixed` logs a warning and is ignored (the
  α-grid sweep is meant to perturb every question uniformly).
- **`stages.detect.refusal_vu_threshold = 0.85`** is our calibration,
  not paper's. Override per experiment.
- **Server runs are Docker-only**. Even with the right config,
  `uv run sae-muc run` on the host is forbidden — see
  [`scripts/server_setup.md`](../scripts/server_setup.md).
- **Base models hedge ~everything**. `gemma-2-2b` (base) on the paper's
  completion-style prompt produces "I'm not sure" / continues a fake
  FAQ for ≥90% of NQ-Open questions, and the refusal filter eats them →
  `n_refusal=9/10`, the detector can't fit, metrics are noise. For
  meaningful Tab.3-style numbers use the `-it` variants (`gemma-2-2b-it`,
  `Mistral-7B-Instruct-v0.3`, `Llama-3.1-8B-Instruct`). Caveat: SAEs
  trained on the base model (Gemma-Scope, Llama-Scope) used on `-it`
  are slightly OOD; the reconstruction degrades but still works in
  practice.
- **n=10 is for plumbing, not statistics**. Detector AUROC is
  computed on the 80/20 split of the trainable subset; with refusals
  excluded, n=10 leaves ≤2 test samples → CI ≈ ±50%. Use n≥50 for any
  comparison you'd quote.
