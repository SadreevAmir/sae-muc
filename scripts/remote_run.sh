#!/usr/bin/env bash
# remote_run.sh — kick off a sae-muc run on the server from your laptop.
#
# Usage:
#   export SAE_MUC_SSH_HOST=user@server                # just host, no path
#   export SAE_MUC_REPO_PATH=/home/user/sae-muc        # default shown
#   export SAE_MUC_GPU=4                               # nvtop index
#   ./scripts/remote_run.sh configs/experiment/qwen05b_smoke.yaml
#
# What it does:
#   1. ssh to the server
#   2. git fetch + checkout feature/server-pipeline + pull
#   3. (re)build the docker image if needed
#   4. launch `scripts/docker/run.sh <gpu> run --config <cfg>` inside tmux
#   5. print the session name so you can attach with `ssh … tmux attach -t …`
set -euo pipefail

SSH_HOST=${SAE_MUC_SSH_HOST:?"set SAE_MUC_SSH_HOST (e.g. user@server)"}
REPO_PATH=${SAE_MUC_REPO_PATH:-"~/sae-muc"}
BRANCH=${SAE_MUC_BRANCH:-"main"}
GPU=${SAE_MUC_GPU:?"set SAE_MUC_GPU (nvtop index of the GPU to use, e.g. 4)"}

# Default image tag is derived from the SSH user (the server-side username)
# so two teammates on the same host don't share a `:latest` tag with each
# other's UID baked in. Falls back to local `id -un` if SSH_HOST has no user@.
REMOTE_USER="${SSH_HOST%%@*}"
[[ "$REMOTE_USER" == "$SSH_HOST" ]] && REMOTE_USER="$(id -un)"
USER_TAG="$(echo "$REMOTE_USER" | tr '.A-Z' '_a-z')"
IMAGE=${SAE_MUC_IMAGE:-"sae-muc:${USER_TAG}"}
CONFIG=${1:?"usage: $0 <config-yaml relative to repo root>"}

SESSION="sae-muc-$(date +%Y%m%d-%H%M%S)"

ssh "$SSH_HOST" bash -s <<EOF
set -euo pipefail
cd "$REPO_PATH"
git fetch origin
git checkout "$BRANCH"
git pull --ff-only

# Rebuild only if pyproject / uv.lock / Dockerfile changed since last build.
NEED_BUILD=0
if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    NEED_BUILD=1
else
    IMAGE_TS=\$(docker image inspect "$IMAGE" --format '{{.Created}}' | xargs -I{} date -d {} +%s)
    for f in pyproject.toml uv.lock Dockerfile; do
        if [[ -f "\$f" && \$(stat -c %Y "\$f") -gt \$IMAGE_TS ]]; then
            NEED_BUILD=1
            break
        fi
    done
fi
if [[ \$NEED_BUILD -eq 1 ]]; then
    echo "==> rebuilding $IMAGE"
    IMAGE="$IMAGE" scripts/docker/build.sh
fi

mkdir -p data
tmux new-session -d -s "$SESSION" \\
    "cd '$REPO_PATH' && IMAGE='$IMAGE' scripts/docker/run.sh '$GPU' run --config '$CONFIG' 2>&1 | tee -a data/run.log"

echo "tmux session launched: $SESSION"
echo "attach: ssh $SSH_HOST tmux attach -t $SESSION"
EOF
