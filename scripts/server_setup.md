# Server setup

How to get `sae-muc` running on a shared GPU server. **Everything goes
through Docker** — no system Python, no host venv. The team rule is:
non-root containers, narrow GPU visibility, predictable file ownership.

Tested on `caniculus` (driver 535.154.05, CUDA 12.2, Docker 25.0.3 with
nvidia-container-toolkit, 7× mixed H100 / L40 / L40S).

## 1. Pre-flight

```bash
nvidia-smi | head -3                     # driver + CUDA visible?
docker run --rm hello-world              # docker daemon reachable?
docker run --rm --gpus all \
    nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi -L
                                         # GPU passthrough works?
id                                       # note your UID/GID; image bakes them
```

You need to be in the `docker` group (`groups | grep -w docker`). If not,
ask the admin — do not `sudo docker`.

## 2. Clone the repo

```bash
git clone https://github.com/SadreevAmir/sae-muc.git ~/sae-muc
cd ~/sae-muc
git checkout feature/server-pipeline
```

No `uv sync` on the host — uv lives inside the image.

## 3. Secrets

```bash
cp .env.example .env
$EDITOR .env       # HF_TOKEN, OPENROUTER_API_KEY / CHERRYIN_API_KEY
```

`.env` is gitignored and is also excluded from the docker build context
(`.dockerignore`), so it is never baked into the image. It is mounted
at runtime via `--env-file`.

## 4. Shared storage (one-time, any teammate)

The team writes HF model weights, run artefacts, and the uv cache to
`/mnt/ssd/sae-muc/`. `/mnt/ssd` is `0777` on caniculus (no sudo needed).

```bash
scripts/docker/setup_shared.sh
```

This is idempotent — re-running it just re-asserts the perms. Result:

```
/mnt/ssd/sae-muc/
├── hf-cache/      (mounted as $HF_HOME inside the container)
├── uv-cache/      (uv wheel cache)
└── runs/          (mounted as data/runs/ inside the container)
```

Group `ipadocker` + setgid bit + container-side `umask 002` mean files
written by any teammate are group-writable, so anyone can resume or
extend a teammate's run via `--run-id <foreign>`.

**Caveat.** `ipadocker` has ~300 members on caniculus — anyone with a
server account technically can write to `/mnt/ssd/sae-muc/`. Run-ids
are prefixed with the creator's username (`k.frolov__nq_open__…`), so
ownership is at least visible in `ls runs/`. Don't put secrets here;
`.env` stays per-user inside the repo (gitignored).

To run with personal storage instead (e.g., for noisy ablations you
don't want polluting shared `runs/`):

```bash
SAE_MUC_SHARED= scripts/docker/run.sh <gpu> run --config <yaml>
```

## 5. Build the image

```bash
scripts/docker/build.sh
```

This passes `--build-arg USER_UID=$(id -u) USER_GID=$(id -g)` so the
container's `appuser` matches your host UID. Files written into
`data/runs/` will be owned by you. First build takes ~5 minutes (torch
+ transformers + sae-lens), subsequent builds reuse the uv cache layer.

Default tag is `sae-muc:latest`. Multi-tenant tip: each developer can
tag their own image to avoid stomping each other's `latest`:

```bash
IMAGE=sae-muc:k.frolov scripts/docker/build.sh
```

## 6. Pick a free GPU

```bash
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader
```

Look for a card with `memory.used` near 0 and `utilization.gpu = 0%`.
The index column is what `scripts/docker/run.sh` expects — Docker's
`--gpus device=N` matches `nvidia-smi` index N exactly (verified on
caniculus 2026-04-27).

**Etiquette**: take one card unless you have agreement in chat.
Reserve H100s (idx 0, 2 on caniculus) for paper-scale runs; smoke
tests fit on an L40 (idx 1, 4, 5) or L40S (idx 3, 6).

## 7. Smoke test

Under `tmux` so the run survives an SSH drop:

```bash
tmux new -s sae-muc

# inside tmux:
cd ~/sae-muc
scripts/docker/run.sh 4 run --config configs/experiment/qwen05b_smoke.yaml
```

Detach with `Ctrl-b d`, reattach with `tmux attach -t sae-muc`.

Expected on a free L40:
- Generate (Qwen2.5-0.5B) loads weights once, then < 1 s / question.
- Judge + accuracy_judge via OpenRouter: ~1 s each.
- NLI (DeBERTa-v3-base): loads once (~180 MB), then fast.
- Full pipeline on 10 questions: **2–4 minutes**.

Model weights are cached at `~/.cache/huggingface/` on the host (mounted
into the container), so subsequent runs skip the download.

## 8. Paper-scale run

Same shape, bigger config and a beefier GPU:

```bash
tmux new -s paper-run
scripts/docker/run.sh 0 run --config configs/experiment/qwen25_7b_triviaqa.yaml
```

## 9. Interactive debugging

```bash
scripts/docker/shell.sh 4
# inside container:
python -c "import torch; print(torch.cuda.get_device_name(0))"
sae-muc --help
pytest -q
```

`appuser`'s HOME is `/home/appuser`, the venv is at `/opt/venv` (already
on PATH), the repo is bind-mounted at `/app`.

## 10. Pull artefacts back to your laptop

From your local machine (not the server):

```bash
export SAE_MUC_SSH=user@server:/home/user/sae-muc
./scripts/sync_artifacts.sh <run_id>           # parquet + json only
./scripts/sync_artifacts.sh --heavy <run_id>   # also safetensors
```

## Troubleshooting

- **`docker: Got permission denied … unix:///var/run/docker.sock`** —
  not in the docker group; ask the admin.
- **`could not select device driver "" with capabilities: [[gpu]]`** —
  nvidia-container-toolkit not installed / not configured.
- **Out of VRAM on 7B** — try `dtype: float16` in the model YAML; last
  resort, enable bnb quantisation (TODO.md P4).
- **HF download hangs** — set `HF_HUB_ENABLE_HF_TRANSFER=1` in `.env`.
- **Judge 402 "Insufficient credits"** — top up OpenRouter or switch to
  a `*:free` route. See `scripts/check_openrouter.py --balance`.
- **SSH dropped mid-run** — reattach with `tmux attach`. Stages with
  manifests are skipped on resume; pass `--run-id <existing>` to keep
  the same artefact directory.
- **Want to inspect the image build context size** —
  `tar --exclude-from=.dockerignore -cf - . | wc -c` (should be a few MB,
  not GB; if GB, something heavy slipped past `.dockerignore`).

## Image hygiene

- `docker image ls sae-muc` to see local tags.
- `docker system prune` to drop dangling layers (will not touch named
  images or running containers).
- The image bakes only metadata + `src/`; configs / scripts / tests are
  bind-mounted at runtime, so most edits do not require a rebuild.
  **Rebuild needed** when `pyproject.toml`, `uv.lock`, or `Dockerfile`
  changes.
