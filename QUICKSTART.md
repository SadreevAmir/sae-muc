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

```bash
ssh user@server
git clone <repo-url> ~/sae-muc && cd ~/sae-muc
git checkout feature/server-pipeline
uv sync --all-extras
cp .env.example .env && $EDITOR .env   # fill HF_TOKEN and OPENROUTER_API_KEY
uv run huggingface-cli login           # for gated models (Mistral, Llama)

tmux new -s run
# inside tmux — the run survives if SSH drops:
uv run sae-muc run all --config configs/experiment/mistral_nq.yaml
# detach: Ctrl-b d         reattach: tmux attach -t run
```

## Common operations

- **Switch config** — change `--config` argument; each config resolves into
  one `data/runs/<run_id>/`.
- **Pull artefacts back to your laptop:**
  ```bash
  rsync -az server:~/sae-muc/data/runs/<run_id>/{metrics.json,*.parquet} \
        ./local_copy/
  ```
- **Read a parquet artefact:**
  ```python
  import pandas as pd
  pd.read_parquet("data/runs/<run_id>/generations.parquet")
  ```

## Secrets

All secrets live in `.env` on the server. `.env` is gitignored; the template
is `.env.example`. Never commit tokens.
