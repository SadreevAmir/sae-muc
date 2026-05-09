# Quickstart

Two environments, separate paths:

- **Local** = editor + unit / integration tests on fake backends. No GPU,
  no real LLM weights, no API keys.
- **Server** = real GPU runs. **Docker-only** (no system Python, no host
  venv); see [`scripts/server_setup.md`](scripts/server_setup.md) for the
  detailed walkthrough.

## Prerequisites

- **Local:** Python 3.11+ (3.12 preferred to match the image), git, [uv](https://docs.astral.sh/uv/)
  (`curl -LsSf https://astral.sh/uv/install.sh | sh`).
- **Server:** Docker with `nvidia-container-toolkit`, member of the
  `docker` and (on caniculus) `ipadocker` group, NVIDIA GPU with CUDA
  12.2+ driver. HF account with licenses accepted for any gated models
  you want (Gemma-2, Llama-3, Mistral). OpenRouter API key for the judge.

## Local setup (editing + tests)

```bash
git clone https://github.com/SadreevAmir/sae-muc.git && cd sae-muc
git checkout feature/server-pipeline
uv sync --all-extras          # runtime + dev deps
cp .env.example .env           # tokens NOT needed for local tests
uv run pytest -q               # ~200 unit + integration tests on fakes (~20s)
```

Validate a real-shape config without touching the network or GPU:

```bash
uv run python -c "
from sae_muc.config import load_experiment_config
cfg = load_experiment_config('configs/experiment/gemma2_2b_sae_smoke.yaml')
print(cfg.model.name, cfg.stages.intervene.method)
"
```

For an SAE-availability check against the live sae-lens registry (still
no GPU needed):

```bash
uv run python -c "
from sae_muc.config import load_experiment_config
from sae_muc.models.sae import assert_sae_layers_available
cfg = load_experiment_config('configs/experiment/mistral7b_sae_sparse_smoke.yaml')
assert_sae_layers_available(cfg.sae, [8, 16, 24])
print('OK')
"
```

## Server runs (real GPU)

Full walkthrough at [`scripts/server_setup.md`](scripts/server_setup.md).
TL;DR — first time on the box:

```bash
ssh user@caniculus
git clone https://github.com/SadreevAmir/sae-muc.git ~/sae-muc && cd ~/sae-muc
git checkout feature/server-pipeline
cp .env.example .env && $EDITOR .env     # HF_TOKEN, OPENROUTER_API_KEY
                                          # (do NOT run huggingface-cli login —
                                          #  it would write the token to shared cache)
scripts/docker/setup_shared.sh           # one-time, idempotent
scripts/docker/build.sh                  # ~5 min first time, per-user image tag
```

Then for every run:

```bash
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader
                                          # pick a free L40 / L40S
tmux new -s smoke
scripts/docker/run.sh <gpu> run --config configs/experiment/gemma2_2b_sae_smoke.yaml
                                          # detach: Ctrl-b d
                                          # reattach: tmux attach -t smoke
```

The pipeline writes its full log to `/mnt/ssd/sae-muc/runs/<run_id>/run.log`
**alongside the artefacts**. From any other ssh session:

```bash
tail -f /mnt/ssd/sae-muc/runs/<run_id>/run.log
```

Resume a partially-completed run (your own or a teammate's):

```bash
scripts/docker/run.sh <gpu> run --config <yaml> --run-id <existing_run_id>
# stages with valid manifests are skipped; only what's missing runs
```

## Pull artefacts back to your laptop

```bash
export SAE_MUC_RUNS_REMOTE=user@caniculus:/mnt/ssd/sae-muc/runs
./scripts/sync_artifacts.sh <run_id>            # parquet + json + run.log (small)
./scripts/sync_artifacts.sh --heavy <run_id>    # also safetensors (GBs)
```

(The legacy `SAE_MUC_SSH=user@server:/path/to/sae-muc` env still works
and points at `<repo>/data/runs/`; use it only for ranов that landed in
per-user storage rather than shared.)

## Common operations

- **Switch experiment** — change `--config <yaml>`. Each config + seed
  combination resolves to its own `<run_id>`. Schema reference and
  cookbook recipes live in [`configs/README.md`](configs/README.md).
- **Read a parquet artefact** (locally, after sync, or directly on the
  server through `scripts/docker/shell.sh`):
  ```python
  import pandas as pd
  pd.read_parquet("data/runs/<run_id>/generations.parquet")
  ```
- **Compare before/after intervention** — every run produces
  `metrics_comparison.parquet` with one row per intervention variant.
- **Read a teammate's run** — same path under `/mnt/ssd/sae-muc/runs/`,
  group `ipadocker` makes it readable. Run-id prefix shows the owner.

## Secrets

`.env` lives per-user inside the repo, gitignored, **never in shared
storage**. The Docker image excludes `.env` from the build context too,
so it's never baked in — it's mounted at runtime via `--env-file`.

Never run `huggingface-cli login` inside the container; it would write
the token to `~/.cache/huggingface/token` which on caniculus resolves to
the shared cache directory, readable by ~50 people.
