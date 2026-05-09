# SAE Steering v2: Three Methods for VU Calibration

## Overview

Extension of the original SAE MUC pipeline with three principled steering methods, all based on consensus feature selection from statistical analysis of SAE latent activations.

**Goal**: increase Verbal Uncertainty (VU) when Semantic Uncertainty (SU) is high, using SAE features instead of raw VUF vectors. This provides interpretability (we know *which* features we change) and precision (we only touch relevant monosemantic features).

### Methods

| Method | `--steering` flag | How it works |
|--------|------------------|-------------|
| **EMD** (Encode-Modify-Decode) | `sae_emd` | `f' = f + α·δ`, `h' = decode(f') + (h - decode(f))`. δ weighted by Cohen's d on consensus features. |
| **Projected VUF** | `sae_projected_vuf` | Same formula, but δ = SAE-projected VUF vector masked to significant features only. "Cleaned MUC". |
| **Feature Clamping** | `sae_clamp` | Raise uncertainty features toward target activation, suppress certainty features toward zero. |

All methods preserve the SAE reconstruction error term `(h - decode(f))`, ensuring zero side effects when α=0.

## Pipeline

```
┌──────────────────────────┐     ┌──────────────────────────────┐     ┌─────────────────┐
│ build_intervention_      │     │ run_muc.py                   │     │ eval metrics     │
│ config_v2.py             │────▶│ --steering sae_emd           │────▶│ (same as paper)  │
│                          │     │ --steering sae_projected_vuf │     │                  │
│ Analyzes hidden states   │     │ --steering sae_clamp         │     │                  │
│ via SAE encode, computes │     │                              │     │                  │
│ consensus features       │     │ Loads model + SAE + config,  │     │                  │
│                          │     │ generates steered responses   │     │                  │
└──────────────────────────┘     └──────────────────────────────┘     └─────────────────┘
```

### Step 1: Build intervention config

```bash
python -m sae_muc.build_intervention_config_v2 \
  --repo_root /path/to/sae \
  --split both \
  --release mistral-7b-res-wg \
  --hedge_path vuf_checkpoint/calibration/outputs/merged/Mistral-7B-Instruct-v0.3/uncertainty/Hs_hedge_universal.pt \
  --out_path sae_muc/artifacts/intervention_v2.pt \
  --top_k 64 \
  --n_bootstrap 200 \
  --fdr_alpha 0.05 \
  --cohens_d_threshold 0.3
```

This script:
1. Loads hidden state `.pt` files for certain (VU ≤ 0.05) and uncertain (VU ≥ 0.90) samples
2. Encodes them through the SAE for each layer
3. Computes per-feature statistics: mean-diff, Welch t-test + BH-FDR, Cohen's d, bootstrap stability
4. Selects **consensus features**: bootstrap-stable AND (FDR-significant OR |Cohen's d| > threshold)
5. Builds intervention configs for all 3 methods in a single `.pt` file

### Step 2: Run steering

```bash
# Method 1: EMD with consensus features
python -m sae_muc.run_muc \
  --repo_root . \
  --model_name Mistral-7B-Instruct-v0.3 \
  --prompt_type uncertainty \
  --steering sae_emd \
  --intervention_path sae_muc/artifacts/intervention_v2.pt \
  --max_alpha 1.0

# Method 2: SAE-projected VUF
python -m sae_muc.run_muc \
  --repo_root . \
  --model_name Mistral-7B-Instruct-v0.3 \
  --prompt_type uncertainty \
  --steering sae_projected_vuf \
  --intervention_path sae_muc/artifacts/intervention_v2.pt \
  --max_alpha 1.0

# Method 3: Feature clamping
python -m sae_muc.run_muc \
  --repo_root . \
  --model_name Mistral-7B-Instruct-v0.3 \
  --prompt_type uncertainty \
  --steering sae_clamp \
  --intervention_path sae_muc/artifacts/intervention_v2.pt \
  --max_alpha 1.0
```

## File structure

```
sae_muc/
├── build_intervention_config_v2.py   # NEW: builds v2 config for all 3 methods
├── build_intervention_config.py      # legacy v1 (hedge projection only)
├── hooks.py                          # UPDATED: supports emd, projected_vuf, clamp
├── run_muc.py                        # UPDATED: --steering sae_emd/sae_projected_vuf/sae_clamp
├── vuf_hooks.py                      # raw VUF residual steering (baseline)
├── generation.py                     # text generation (unchanged)
├── layer_map.py                      # SAE <-> HF layer mapping
├── inspect_delta.py                  # delta analysis + Neuronpedia links
├── artifacts/
│   ├── mistral_intervention.pt       # v1 config (legacy)
│   └── intervention_v2.pt           # v2 config (all 3 methods)
└── README_intervention_v2.md         # this file
```

## Intervention config format (v2)

The `.pt` file saved by `build_intervention_config_v2.py` contains:

```python
{
    "release": "mistral-7b-res-wg",
    "meta": {
        "split": "both",
        "vu_certain_max": 0.05,
        "vu_uncertain_min": 0.90,
        "n_certain": 615,
        "n_uncertain": 35,
        "top_k": 64,
        "fdr_alpha": 0.05,
        "n_bootstrap": 200,
        "bootstrap_freq_threshold": 0.8,
        "cohens_d_threshold": 0.3,
    },
    "layers": {
        15: {
            "layer": 15,
            "sae_id": "blocks.16.hook_resid_pre",
            "method_emd": {
                "delta": Tensor[65536],       # L2-normalized, Cohen's d weighted
                "feature_indices": [int, ...], # which features are nonzero
                "weights": [float, ...],       # original Cohen's d weights
            },
            "method_projected_vuf": {
                "delta": Tensor[65536],        # L2-normalized, masked hedge scores
                "feature_mask": Tensor[65536], # binary mask
                "n_features_kept": int,
            },
            "method_clamp": {
                "uncertainty_features": [int, ...],
                "certainty_features": [int, ...],
                "target_uncertain_values": {idx: float, ...},  # mean activation on uncertain samples
                "mean_certain_values": {idx: float, ...},
                "freq_uncertain": {idx: float, ...},
                "freq_certain": {idx: float, ...},
            },
            "stats": { ... },  # summary statistics
        },
        23: { ... },
    },
}
```

## Method details

### Method 1: EMD (Encode-Modify-Decode + Error Term)

```python
f = SAE.encode(h)
recon = SAE.decode(f)
error = h - recon                    # preserve what SAE doesn't capture
f_steered = f + alpha * delta        # shift in feature space
h' = SAE.decode(f_steered) + error   # reconstruct + error
```

**Feature selection**: consensus = bootstrap-stable(≥80%) ∩ (FDR-significant ∨ |Cohen's d| > 0.3)

**Weights (δ)**: Cohen's d per feature. Positive for uncertainty-up features, negative (inverted) for certainty-up features. L2-normalized.

**When α=0**: h' = h exactly (identity).

### Method 2: Projected VUF

Same EMD formula, but δ is computed differently:

1. Project the raw VUF vector (Hs_hedge) into SAE feature space via `SAE.encode(r_VUF)`
2. Zero out all features not in the consensus set (or not FDR-significant among top hedge projections)
3. L2-normalize the resulting sparse vector

This is a "cleaned" version of the original MUC: the VUF direction is filtered through SAE to retain only interpretable, statistically validated components.

**Requires** `--hedge_path` when building the config.

### Method 3: Feature Clamping

```python
f = SAE.encode(h)
error = h - SAE.decode(f)

# For uncertainty features: push up toward target
f'[i] = f[i] + α * max(0, target[i] - f[i])

# For certainty features: suppress
f'[i] = f[i] * (1 - α)

h' = SAE.decode(f') + error
```

Where `target[i]` is the mean activation of feature `i` on uncertain samples.

**α is clamped to [0, 1]** in this method (it acts as an interpolation coefficient).

Most interpretable: we literally "turn on uncertainty features and turn off certainty features".

## Comparison with baselines

| Method | `--steering` | Interpretable? | Preserves error? | Config needed |
|--------|-------------|---------------|-----------------|---------------|
| Raw VUF (paper) | `residual` | No | N/A (no SAE) | `Hs_hedge.pt` |
| v1 hedge projection | `sae` | Partial | Yes | v1 `.pt` |
| **v2 EMD** | `sae_emd` | Yes (consensus) | Yes | v2 `.pt` |
| **v2 Projected VUF** | `sae_projected_vuf` | Yes (masked VUF) | Yes | v2 `.pt` + hedge |
| **v2 Clamp** | `sae_clamp` | Maximum | Yes | v2 `.pt` |

## Alpha semantics

For all methods, α is computed the same way as in the paper:

```
α = clip(SU_norm - VU, 0, max_α)
```

where `SU_norm = SE / ln(N)` (normalized semantic entropy).

- For EMD and Projected VUF: α scales the magnitude of the feature-space shift
- For Clamp: α ∈ [0,1] controls how aggressively features are pushed to targets

## Evaluation

Output `.jsonl` files are compatible with the same evaluation scripts as the original paper:
- `eval_vu.py` — verbal uncertainty
- `eval_acc.py` — correctness
- `eval_refusal.py` — refusal rate
- `compute_semantic_entropy.py` — semantic entropy

Metrics to track (from Table 3 in Ji et al.):
- Confident Hallucination Rate ↓
- VU/SU Disagreement Rate ↓
- Correlation ↑
- VU for Incorrect ↑
- VU for Correct (should stay stable)
