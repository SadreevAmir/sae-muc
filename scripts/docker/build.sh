#!/usr/bin/env bash
# Build the sae-muc image, baking in the invoker's UID/GID so that files
# created inside the container are owned by you on the host.
#
# Usage:
#   scripts/docker/build.sh                  # tag = sae-muc:latest
#   IMAGE=sae-muc:k.frolov scripts/docker/build.sh
set -euo pipefail

cd "$(dirname "$0")/../.."

IMAGE="${IMAGE:-sae-muc:latest}"

DOCKER_BUILDKIT=1 docker build \
    --build-arg USER_UID="$(id -u)" \
    --build-arg USER_GID="$(id -g)" \
    --tag "${IMAGE}" \
    .

echo
echo "built ${IMAGE} (UID=$(id -u) GID=$(id -g))"
docker image inspect "${IMAGE}" --format 'size: {{.Size}} bytes'
