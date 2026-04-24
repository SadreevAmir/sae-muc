# Server setup

Step-by-step to get a fresh server ready to run `sae-muc` with real LLMs.
Assumes Linux + NVIDIA GPU + CUDA ≥ 12.

## 1. Pre-flight

```bash
nvidia-smi                 # confirm GPU and driver are visible
python --version           # expect 3.11+
git --version
```

If CUDA / driver is missing, fix that first — PyTorch picks up the
driver at install time; bootstrapping later is messier than doing it now.

## 2. Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
exec $SHELL                # reload PATH
uv --version
```

## 3. Clone and install

```bash
git clone https://github.com/SadreevAmir/sae-muc.git ~/sae-muc
cd ~/sae-muc
git checkout feature/server-pipeline

uv sync --all-extras       # runtime + dev deps; pulls torch w/ CUDA
```

First `uv sync` takes a few minutes because of torch. Subsequent syncs
are cached.

## 4. Secrets

```bash
cp .env.example .env
$EDITOR .env               # fill HF_TOKEN and OPENROUTER_API_KEY / CHERRYIN_API_KEY
```

`.env` is gitignored. Do not commit tokens.

```bash
uv run huggingface-cli login   # paste HF_TOKEN — needed for Mistral / Llama
```

## 5. Shared HF cache (optional but recommended)

If more than one user / checkout is on this machine, pin `$HF_HOME` to a
shared path so Qwen / Mistral / DeBERTa aren't downloaded twice:

```bash
echo 'export HF_HOME=/opt/hf-cache' | sudo tee -a /etc/profile.d/hf_home.sh
sudo mkdir -p /opt/hf-cache && sudo chmod 777 /opt/hf-cache
exec $SHELL
```

(Skip the sudo bits and pick `~/.cache/huggingface` if you're the only
user on the box.)

## 6. Smoke test

Under `tmux` (or `screen`), so the run survives SSH drops:

```bash
tmux new -s sae-muc

# inside tmux:
cd ~/sae-muc
uv run sae-muc run --config configs/experiment/qwen05b_smoke.yaml
```

Detach with `Ctrl-b d`. Reattach with `tmux attach -t sae-muc`.

Expected shape on GPU:
- Generate (Qwen2.5-0.5B) loads weights once, then < 1 s / question.
- Judge + accuracy_judge via OpenRouter: ~1 s each, ~1 min for 60 calls.
- NLI (DeBERTa-v3-base): loads once (~180 MB), then fast.
- Full pipeline on 10 questions: **2–4 minutes**.

## 7. Paper-scale run

```bash
tmux new -s paper-run
uv run sae-muc run --config configs/experiment/qwen25_7b_triviaqa.yaml
```

Rough cost/time on A100-80GB with `qwen/qwen-2.5-72b-instruct` judge:
- 1 000 questions × 11 generations ≈ 11 000 judge calls ≈ 10–15 min if
  sequential (today). See `TODO.md` P4 for planned concurrency.
- Total run ~30–60 min including intervene + all post-stages.

## 8. Pull artefacts back to your laptop

From your local machine (not the server), use the rsync helper:

```bash
# scripts/sync_artifacts.sh expects SAE_MUC_SSH=user@server:/path/to/sae-muc
export SAE_MUC_SSH=user@server:/home/you/sae-muc
./scripts/sync_artifacts.sh <run_id>
```

It pulls only small artefacts (parquet / json / manifests). Hidden-state
and SAE safetensors are deliberately excluded — they are big and stay on
the server. Pull them explicitly if you need to analyse offline.

## Troubleshooting

- **Out of VRAM on 7B**: try `dtype: float16` in the model YAML first;
  last resort, enable bnb quantisation (TODO.md P4).
- **HF download hangs**: check outbound internet; `HF_HUB_ENABLE_HF_TRANSFER=1`
  for faster multi-connection downloads.
- **Judge 402 "Insufficient credits"**: OpenRouter keeps a reserve buffer
  on paid models; top up $5+ or switch to a `*:free` route. See
  `scripts/check_openrouter.py --balance`.
- **SSH dropped mid-run**: reattach with `tmux attach`; the run keeps
  going. Stages that completed wrote manifests, so re-running with
  `--run-id <existing>` resumes from where it left off.
