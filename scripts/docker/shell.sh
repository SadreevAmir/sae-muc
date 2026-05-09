#!/usr/bin/env bash
# Drop into bash inside the sae-muc container with one GPU attached.
# Useful for ad-hoc python/pytest/inspect, model warm-up, etc.
#
# Usage:
#   scripts/docker/shell.sh <gpu_id>
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "usage: $0 <gpu_id>" >&2
    exit 2
fi

GPU_ID="$1"

REPO_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
USER_TAG="$(id -un | tr '.A-Z' '_a-z')"
IMAGE="${IMAGE:-sae-muc:${USER_TAG}}"

SHARED="${SAE_MUC_SHARED-/mnt/ssd/sae-muc}"

if [[ -n "${SHARED}" && -d "${SHARED}" ]]; then
    HF_CACHE="${SHARED}/hf-cache"
    UV_CACHE="${SHARED}/uv-cache"
    RUNS_HOST="${SHARED}/runs"
else
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

RUNS_MOUNT=()
if [[ -n "${RUNS_HOST}" ]]; then
    RUNS_MOUNT=(-v "${RUNS_HOST}:/app/data/runs")
fi

# Same supplementary-group fix as run.sh — a teammate who isn't owner of
# the shared dirs needs ipadocker GID inside the container to write.
GROUP_NAME="${SAE_MUC_GROUP:-ipadocker}"
GROUP_ADD_ARG=()
if SHARED_GID="$(getent group "${GROUP_NAME}" | cut -d: -f3)" && [[ -n "${SHARED_GID}" ]]; then
    GROUP_ADD_ARG=(--group-add "${SHARED_GID}")
fi

# `with-umask bash` sets umask 002 then execs interactive bash.
exec docker run --rm -it \
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
    --entrypoint /usr/local/bin/with-umask \
    "${IMAGE}" \
    bash
