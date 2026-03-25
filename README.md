# SAE-MUC: Sparse Autoencoder Steering for Hallucination Reduction in LLMs

Extension of the **Model Uncertainty Calibration (MUC)** pipeline from
[Ji et al., 2025](https://arxiv.org/abs/2501.09825)
*"Calibrating Verbal Uncertainty as a Linear Feature to Reduce Hallucinations"*
with **Sparse Autoencoder (SAE)** interpretability-based interventions.

**Model:** Mistral-7B-Instruct-v0.3 &nbsp;|&nbsp; **Dataset:** NQ-Open &nbsp;|&nbsp; **SAE:** [SAELens](https://github.com/jbloomAus/SAELens) `mistral-7b-res-wg`

## Key Idea

The original MUC method shifts model activations along a learned **Verbal Uncertainty Feature (VUF)** direction to reduce hallucinations. We decompose this direction through a pre-trained SAE and steer individual **monosemantic latent features** instead, gaining interpretability: we know exactly *which* features are changed and can inspect them on [Neuronpedia](https://www.neuronpedia.org).

Three SAE-based steering methods are implemented:

| Method | Flag | Description |
|--------|------|-------------|
| **EMD** | `sae_emd` | Encode → shift consensus features weighted by Cohen's d → Decode + error |
| **Projected VUF** | `sae_projected_vuf` | SAE-projected VUF masked to significant features only |
| **Feature Clamping** | `sae_clamp` | Raise uncertainty features, suppress certainty features |

## Repository Structure

```
├── sae_muc/                    Core SAE steering library
│   ├── run_muc.py              Main generation pipeline
│   ├── hooks.py                SAE intervention hooks (EMD / projected VUF / clamp)
│   ├── build_intervention_config.py      Build v1 config from VUF vectors
│   ├── build_intervention_config_v2.py   Build v2 config (consensus features)
│   ├── generation.py           Text generation utilities
│   ├── layer_map.py            SAE ↔ HF layer mapping
│   ├── inspect_delta.py        Analyse interventions + Neuronpedia links
│   └── vuf_hooks.py            Raw VUF residual steering (baseline)
│
├── probes_experiment/          Hallucination detection probes (reproduces Tables 2, 4)
│   ├── exp1_probes.py          Linear probes SE/VU → LR
│   ├── exp2_direct.py          Direct LR / PCA+LR on hidden states
│   ├── exp3_vuf_projection.py  VUF-projection + MLP detectors
│   └── sae_feature_analysis.py SAE feature ranking & interpretability
│
├── muc_metrics/                Table 3 metrics: before/after MUC intervention
├── eval_bundle/                Evaluation scripts (SE, VU, accuracy, refusal)
├── metrics_finder/             LLM-judge evaluation notebooks
│
├── notebooks/                  Colab notebooks (see list below)
│
├── datasets__nq_open/          NQ-Open CSVs with VU / SE columns (not in git)
├── nq_open/                    Generated responses & SE/accuracy JSONs (not in git)
└── verbal_uncertainty__outputs/  VU-judge outputs (not in git)
```

## Setup

```bash
pip install -r requirements.txt
```

For the full pipeline you also need data files (CSVs, hidden states, VUF vectors).
See [sae_muc/README.md](sae_muc/README.md) for data layout and CLI usage.

## Quick Start

```bash
# Build intervention config from VUF vectors
python -m sae_muc.build_intervention_config_v2 \
  --repo_root . \
  --release mistral-7b-res-wg \
  --hedge_path path/to/Hs_hedge_universal.pt \
  --out_path sae_muc/artifacts/intervention_v2.pt

# Run SAE-steered generation
python -m sae_muc.run_muc \
  --repo_root . \
  --model_name Mistral-7B-Instruct-v0.3 \
  --steering sae_emd \
  --intervention_path sae_muc/artifacts/intervention_v2.pt \
  --max_alpha 1.0
```

## Colab Notebooks

All notebooks are in the [`notebooks/`](notebooks/) folder:

| Notebook | Purpose |
|----------|---------|
| `colab_sae_playground.ipynb` | Quick before/after comparison of interventions |
| `colab_sae_muc.ipynb` | Full MUC pipeline with Drive backup |
| `colab_sae_steering.ipynb` | SAE EMD / projected VUF steering |
| `colab_sae_clamp_alpha1.ipynb` | Feature clamping method |
| `colab_sae_single_feature.ipynb` | Single-feature interpretability |
| `colab_intervention_layers_15_23.ipynb` | Intervention on layers 15 & 23 |
| `colab_run_eval_metrics.ipynb` | Evaluation metrics computation |
| `colab_uncertainty_prompt_t01.ipynb` | Uncertainty prompt generation |

## Results Summary

**Hallucination detection** (probes, Table 2 reproduction):

| Approach | AUROC | Requires |
|----------|-------|----------|
| Raw SE+VU → LR | **0.777** | 10 samples + LLM-judge |
| Probe SE+VU → LR | **0.734** | only prefill |
| PCA(32)+LR on hs | 0.724 | only prefill |
| VUF proj+resid L17 | 0.700 | only prefill, interpretable |

**MUC intervention** (Table 3 reproduction):

| Metric | Before | After | Δ |
|--------|--------|-------|---|
| Hallucination Rate ↓ | 0.507 | **0.326** | −0.181 |
| Correctness Rate ↑ | 0.469 | 0.389 | −0.080 |
| Refusal Rate | 0.029 | 0.377 | +0.348 |

See [probes_experiment/README.md](probes_experiment/README.md) and
[muc_metrics/README.md](muc_metrics/README.md) for full results.

## References

- Ji et al. (2025). *Calibrating Verbal Uncertainty as a Linear Feature to Reduce Hallucinations.*
  [arXiv:2501.09825](https://arxiv.org/abs/2501.09825)
- [SAELens](https://github.com/jbloomAus/SAELens) — Sparse Autoencoder library
- [Neuronpedia](https://www.neuronpedia.org) — SAE feature explorer
