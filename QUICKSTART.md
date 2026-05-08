# Quickstart

## Prerequisites

- **Everywhere:** Python 3.11+, git, access to this GitHub repo, [uv](https://docs.astral.sh/uv/)
  (`curl -LsSf https://astral.sh/uv/install.sh | sh`).
- **Server only:** NVIDIA GPU + CUDA 12+, HuggingFace account (for gated
  models like Mistral-7B-Instruct), OpenRouter API key.
- **Recommended on server:** `tmux` or `screen` so long runs survive SSH
  disconnects.

## Local setup (development)

Local machine = editor + tests on fake backends. No GPU, no real LLM locally.

```bash
git clone <repo-url> sae-muc && cd sae-muc
git checkout feature/server-pipeline
uv sync --all-extras          # runtime + dev deps
cp .env.example .env           # tokens are only needed on the server
uv run pytest -q               # unit + integration tests on fakes
```

Run the pipeline on mocked models (no GPU, no network):

```bash
uv run sae-muc --help          # stub today; real stages land as they are built
```

## Server setup (real runs)

Full walkthrough with GPU / CUDA / tmux specifics lives at
[`scripts/server_setup.md`](scripts/server_setup.md). TL;DR:

```bash
ssh user@server
git clone https://github.com/SadreevAmir/sae-muc.git ~/sae-muc && cd ~/sae-muc
git checkout feature/server-pipeline
uv sync --all-extras
cp .env.example .env && $EDITOR .env     # fill HF_TOKEN + OPENROUTER_API_KEY
uv run huggingface-cli login             # for gated models (Mistral, Llama)

tmux new -s run
uv run sae-muc run --config configs/experiment/qwen05b_smoke.yaml
# detach: Ctrl-b d         reattach: tmux attach -t run
```

## Launch a server run from your laptop

```bash
export SAE_MUC_SSH_HOST=user@server
./scripts/remote_run.sh configs/experiment/qwen25_7b_triviaqa.yaml
```

The helper ssh's in, pulls the branch, `uv sync`s, and starts the run in
a fresh `tmux` session whose name it prints. Attach later with
`ssh $SAE_MUC_SSH_HOST tmux attach -t <session>`.

## Pull artefacts back

```bash
export SAE_MUC_SSH=user@server:/home/you/sae-muc
./scripts/sync_artifacts.sh <run_id>            # parquet / json only
./scripts/sync_artifacts.sh --heavy <run_id>    # also safetensors (bigger)
```

## Common operations

- **Switch config** — change `--config` argument; each config resolves into
  one `data/runs/<run_id>/`. Schema reference, composition rules
  (`extends:`, fragment refs), and ready-to-edit recipes live in
  [`configs/README.md`](configs/README.md).
- **Read a parquet artefact:**
  ```python
  import pandas as pd
  pd.read_parquet("data/runs/<run_id>/generations.parquet")
  ```
- **Compare before/after intervention:**
  ```python
  pd.read_parquet("data/runs/<run_id>/metrics_comparison.parquet")
  ```

## Secrets

All secrets live in `.env` on the server. `.env` is gitignored; the template
is `.env.example`. Never commit tokens.
