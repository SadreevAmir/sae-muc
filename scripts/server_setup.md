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

`scripts/docker/run.sh` and `shell.sh` automatically pass
`--group-add ipadocker` to docker run, so the container's process gets
the group as a supplementary GID and can write through the group bit
even when the parent dir is owned by another teammate. Without this,
`mkdir` in `data/runs/` fails with PermissionError on the very first
non-owner run.

**Caveat.** `ipadocker` has ~50 effective members on caniculus (verified
2026-05-07: 49 of 54 active /home users). The alternative `docker(999)`
is the same size (~52), so neither group offers real isolation — anyone
with server access technically can read `/mnt/ssd/sae-muc/`. Run-ids
are prefixed with the creator's username (`k.frolov__nq_open__…`), so
ownership is at least visible in `ls runs/`.

**Do not put secrets in shared storage.** Specifically:

- `.env` stays per-user inside the repo (gitignored, also excluded from
  the docker build context).
- **Never run `huggingface-cli login` inside the container.** It writes
  the token to `~/.cache/huggingface/token`, which inside the container
  resolves to `/mnt/ssd/sae-muc/hf-cache/token` and would be readable by
  ~50 people. Pass `HF_TOKEN` via `.env` instead — the HF library reads
  the env var and skips the on-disk token entirely.
- Same logic for any other provider login: keep credentials in `.env`,
  never persist them under `~/.cache/`, `~/.config/`, or other paths
  that overlap shared cache mounts.

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

**Default tag is per-user** (`sae-muc:k_frolov`, `sae-muc:d_koblov`, …)
to prevent four teammates from stomping each other's baked-in UID on
the shared `:latest`. `run.sh`/`shell.sh`/`remote_run.sh` derive the
same tag from `id -un` (or the SSH username). Override only if you
need a shared image:

```bash
IMAGE=sae-muc:shared scripts/docker/build.sh
IMAGE=sae-muc:shared scripts/docker/run.sh 4 ...
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
tmux new -s sae-smoke

# inside tmux:
cd ~/sae-muc
scripts/docker/run.sh 4 run --config configs/experiment/gemma2_2b_sae_smoke.yaml
```

Detach with `Ctrl-b d`, reattach with `tmux attach -t sae-smoke`.

Expected on a free L40 with weights already in shared cache:
- Generate (Gemma-2-2B): ~15 s for 10 questions × 5 generations.
- Judge + accuracy_judge via OpenRouter: ~30 s + ~5 s.
- NLI (DeBERTa-v3-base): loads once, then ~3 s.
- sae_features (Gemma-Scope SAE layer 20): ~5 s.
- intervene (sae_emd, 2 alphas): ~7 s.
- judge_post (re-scoring 2 variants): ~60 s.
- Full pipeline on 10 questions: **~3–5 minutes**.

Cold first run on a teammate's account that hasn't touched these models
yet: add ~3 min to download Gemma-2-2B (~5 GB) and ~5 s for the SAE.
Subsequent runs reuse the shared cache.

## 8. Paper-shape run (n=50, adaptive MUC)

Same shape, bigger config:

```bash
tmux new -s paper-run
scripts/docker/run.sh 4 run --config configs/experiment/mistral7b_sae_sparse_smoke.yaml
```

Mistral-7B-Instruct + sparse Gemma-Scope-style SAE on layers 8/16/24,
adaptive MUC (paper §4.2 Eq.5–6), n=50. Wall time on a free L40:
**~25 minutes**, dominated by `intervene` (~11 min) and `judge_post`
(~5 min, OpenRouter latency).

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

## 10. Watching logs

The pipeline writes its own log (no ANSI codes, plain text) to the run
directory **alongside the artefacts**:

```
/mnt/ssd/sae-muc/runs/<run_id>/run.log
```

So from any ssh session (yours or a teammate's), without tmux, without
script(1), and without re-attaching:

```bash
RUN=$(ls -t /mnt/ssd/sae-muc/runs | head -1)   # most recent run
tail -f /mnt/ssd/sae-muc/runs/$RUN/run.log
```

The terminal in tmux still shows the cyan `==> stage` / green `[ok] …`
banners with timestamps; the file gets the same lines without colour
codes. Resumes via `--run-id` append to the same file, so the resume
history is in one place.

## 11. Pull artefacts back to your laptop

From your local machine (not the server):

```bash
export SAE_MUC_RUNS_REMOTE=user@caniculus:/mnt/ssd/sae-muc/runs
./scripts/sync_artifacts.sh <run_id>           # parquet + json + run.log
./scripts/sync_artifacts.sh --heavy <run_id>   # also safetensors (GBs)
```

`SAE_MUC_RUNS_REMOTE` points at the runs root in shared storage. The
legacy `SAE_MUC_SSH=user@server:/abs/repo/path` env still works and
points at `<repo>/data/runs/`; only useful if you ran with
`SAE_MUC_SHARED=` (personal storage).

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
  the same artefact directory. Or skip tmux entirely and `tail -f` the
  run.log from a fresh ssh session.
- **`PermissionError: ... data/runs/<run_id>/`** on a teammate's first
  run — the supplementary group `ipadocker` wasn't propagated into the
  container. `git pull` for the latest `scripts/docker/run.sh` (since
  commit `dc0111c` it passes `--group-add` automatically).
- **`setup_shared.sh: chgrp: Operation not permitted`** on second-and-later
  runs — harmless, fixed in commit `6663353`. The script now skips
  already-correct entries and tolerates files owned by other teammates.
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
