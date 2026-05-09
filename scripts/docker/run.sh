#!/usr/bin/env bash
# Run sae-muc inside the docker image on a single GPU.
#
# Usage:
#   scripts/docker/run.sh <gpu_id> [sae-muc args...]
#
# Examples:
#   scripts/docker/run.sh 0 run --config configs/experiment/qwen05b_smoke.yaml
#   scripts/docker/run.sh 4 run --config configs/experiment/qwen25_7b_triviaqa.yaml
#
# <gpu_id> is the nvtop / nvidia-smi index of the card (verified to match
# Docker's --gpus device=N — see project_server_gpu_mapping memory note).
#
# Shared storage:
#   Default: SAE_MUC_SHARED=/mnt/ssd/sae-muc — all 4 teammates write here.
#   To use personal caches, set SAE_MUC_SHARED= (empty).
set -euo pipefail

if [[ $# -lt 1 ]]; then
    cat >&2 <<EOF
usage: $0 <gpu_id> [sae-muc args...]
  gpu_id: nvtop / nvidia-smi index of the GPU to expose (single card).
          Check free cards: nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv
EOF
    exit 2
fi

GPU_ID="$1"
shift

REPO_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
USER_TAG="$(id -un | tr '.A-Z' '_a-z')"
IMAGE="${IMAGE:-sae-muc:${USER_TAG}}"

# Default to shared storage on caniculus; explicitly empty disables it.
SHARED="${SAE_MUC_SHARED-/mnt/ssd/sae-muc}"

if [[ -n "${SHARED}" && -d "${SHARED}" ]]; then
    HF_CACHE="${SHARED}/hf-cache"
    UV_CACHE="${SHARED}/uv-cache"
    RUNS_HOST="${SHARED}/runs"
    echo "shared storage: ${SHARED}" >&2
else
    if [[ -n "${SHARED}" ]]; then
        echo "warning: SAE_MUC_SHARED=${SHARED} does not exist; falling back to personal caches" >&2
        echo "         run scripts/docker/setup_shared.sh once to create it" >&2
    fi
    HF_CACHE="${HF_HOME:-$HOME/.cache/huggingface}"
    UV_CACHE="${UV_CACHE_DIR:-$HOME/.cache/uv}"
    RUNS_HOST=""
fi

mkdir -p "${HF_CACHE}" "${UV_CACHE}"
[[ -n "${RUNS_HOST}" ]] && mkdir -p "${RUNS_HOST}"

ENV_FILE_ARG=()
if [[ -f "${REPO_DIR}/.env" ]]; then
    ENV_FILE_ARG=(--env-file "${REPO_DIR}/.env")
fi

# -t only when both stdin and stdout are TTYs (so tmux + nohup-detached runs work)
TTY_ARG=(-i)
if [[ -t 0 && -t 1 ]]; then
    TTY_ARG+=(-t)
fi

RUNS_MOUNT=()
if [[ -n "${RUNS_HOST}" ]]; then
    RUNS_MOUNT=(-v "${RUNS_HOST}:/app/data/runs")
fi

# Supplementary group for shared-storage write access. The shared dir's
# group bit (drwxrws---) requires membership; --user only forwards the
# primary group, so without --group-add a non-owner teammate gets
# PermissionError on the very first mkdir under data/runs/. Resolve the
# GID of $SAE_MUC_GROUP (default ipadocker) from /etc/group; works even
# if the group doesn't exist inside the container — Docker only needs
# the numeric GID. Skip if the host doesn't have the group at all
# (private dev box etc).
GROUP_NAME="${SAE_MUC_GROUP:-ipadocker}"
GROUP_ADD_ARG=()
if SHARED_GID="$(getent group "${GROUP_NAME}" | cut -d: -f3)" && [[ -n "${SHARED_GID}" ]]; then
    GROUP_ADD_ARG=(--group-add "${SHARED_GID}")
fi

exec docker run --rm "${TTY_ARG[@]}" \
    --gpus "\"device=${GPU_ID}\"" \
    --user "$(id -u):$(id -g)" \
    "${GROUP_ADD_ARG[@]}" \
    --shm-size=2g \
    -v "${REPO_DIR}:/app" \
    -v "${HF_CACHE}:/home/appuser/.cache/huggingface" \
    -v "${UV_CACHE}:/home/appuser/.cache/uv" \
    "${RUNS_MOUNT[@]}" \
    -e HOME=/home/appuser \
    -e HF_HOME=/home/appuser/.cache/huggingface \
    -e USER="$(id -un)" \
    -e UMASK=002 \
    "${ENV_FILE_ARG[@]}" \
    "${IMAGE}" \
    "$@"
