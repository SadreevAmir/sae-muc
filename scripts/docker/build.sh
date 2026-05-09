#!/usr/bin/env bash
# Build the sae-muc image, baking in the invoker's UID/GID so that files
# created inside the container are owned by you on the host.
#
# Usage:
#   scripts/docker/build.sh                  # tag = sae-muc:<your-username>
#   IMAGE=sae-muc:custom scripts/docker/build.sh
#
# Default tag is per-user (sae-muc:k_frolov, sae-muc:d_koblov, ...) so
# four teammates building on the same host don't stomp each other's
# baked-in UID. Override IMAGE if you need a shared name.
set -euo pipefail

cd "$(dirname "$0")/../.."

USER_TAG="$(id -un | tr '.A-Z' '_a-z')"
IMAGE="${IMAGE:-sae-muc:${USER_TAG}}"

DOCKER_BUILDKIT=1 docker build \
    --build-arg USER_UID="$(id -u)" \
    --build-arg USER_GID="$(id -g)" \
    --tag "${IMAGE}" \
    .

echo
echo "built ${IMAGE} (UID=$(id -u) GID=$(id -g))"
docker image inspect "${IMAGE}" --format 'size: {{.Size}} bytes'
