# sae-muc

Calibrate verbal uncertainty in LLMs using linear features (VUF) and sparse
autoencoder (SAE)-based interventions. Reproduces and extends
*Calibrating Verbal Uncertainty as a Linear Feature to Reduce Hallucinations*
(Ji et al., Meta FAIR, 2025 — [arXiv:2503.14477](https://arxiv.org/abs/2503.14477)).

**Status.** Migrating from a Colab-era prototype (see
[`archive/old-prototype/`](archive/old-prototype/)) into a reproducible
server-based pipeline. Work in progress on `feature/server-pipeline`.

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
semantic_entropy_post → evaluate_post`.

Server runs are Docker-only. Artefacts and per-run logs (`run.log`) land
under `/mnt/ssd/sae-muc/runs/<run_id>/` on the shared GPU box, readable
and resumable by any teammate. See [QUICKSTART.md](QUICKSTART.md) and
[`scripts/server_setup.md`](scripts/server_setup.md).

## Getting started

See [QUICKSTART.md](QUICKSTART.md) for local setup and server onboarding.

## Deferred work

See [TODO.md](TODO.md) for optimisations and items out of the MVP scope.
