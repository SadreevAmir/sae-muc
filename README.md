# sae-muc

Calibrate verbal uncertainty in LLMs using linear features (VUF) and sparse
autoencoder (SAE)-based interventions. Reproduces and extends
*Calibrating Verbal Uncertainty as a Linear Feature to Reduce Hallucinations*
(Ji et al., Meta FAIR, 2025 — [arXiv:2503.14477](https://arxiv.org/abs/2503.14477)).

**Status.** Migrated from a Colab-era prototype (see
[`archive/old-prototype/`](archive/old-prototype/)) to a reproducible
server-based pipeline. Pipeline is end-to-end on real models + SAEs;
research follow-ups tracked in [TODO.md](TODO.md).

## Layout

- `src/sae_muc/` — active pipeline code (config, models, data, pipeline
  stages, artefacts, analysis).
- `configs/` — composable YAML configs (`model/`, `dataset/`, `judge/`,
  `experiment/`); see [configs/README.md](configs/README.md) for the
  reference + cookbook.
- `scripts/` — orchestration helpers (rsync, server setup).
- `tests/` — unit and integration tests; integration runs end-to-end on
  fake backends, no GPU required.
- `archive/` — prior-art reference, not on the active development path.

## Pipeline (current shape)

`prepare → generate → judge (OpenRouter) → accuracy_judge → semantic_entropy
(NLI) → hidden_states → vuf → sae_features → detect → intervene
(linear | SAE) → evaluate → judge_post → accuracy_judge_post →
semantic_entropy_post → evaluate_post → diagnostics`.

The final `diagnostics` stage is a sidecar paper extension — measures
intervention side-effects via:

- **Perplexity** drift on WikiText-2 (vanilla LM probe).
- **Accuracy + NLL** on MMLU / HellaSwag / GSM8K No-CoT (the
  steering-side-effect benchmark trio used by Arditi 2024, CAA, ITI, RepE).
- **KL divergence** of next-token distributions on cached QA prompts.
- Optional **multi-method × α sweep** in a single run (`compare_methods`
  + `alpha_sweep`) so `linear_vuf` vs SAE-methods comparison doesn't need
  4 separate runs.

Skip via `stages.diagnostics.enabled=false` or `--stage <other>`. See
[diagnostics/ artefacts in QUICKSTART](QUICKSTART.md#diagnostics-artefacts).

Server runs are Docker-only. Artefacts and per-run logs (`run.log`) land
under `/mnt/ssd/sae-muc/runs/<run_id>/` on the shared GPU box, readable
and resumable by any teammate. See [QUICKSTART.md](QUICKSTART.md) and
[`scripts/server_setup.md`](scripts/server_setup.md).

## Getting started

See [QUICKSTART.md](QUICKSTART.md) for local setup and server onboarding.

## Deferred work

See [TODO.md](TODO.md) for optimisations and items out of the MVP scope.
