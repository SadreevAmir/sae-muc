#!/usr/bin/env bash
# One-time setup of the team's shared storage on caniculus's /mnt/ssd.
#
# /mnt/ssd is mode 0777 (world-writable, no sudo needed). We carve out
# /mnt/ssd/sae-muc/ owned by group ipadocker with setgid bit, so new
# files inherit the group and team members can read/write each other's
# artefacts.
#
# Run this once from any teammate's account; subsequent re-runs are
# idempotent. Verify with `ls -ld /mnt/ssd/sae-muc/*`.
#
# Caveat: ipadocker has ~300 members on caniculus — this group is
# functionally "everyone with server access", not a team-private group.
# Files within /mnt/ssd/sae-muc/ are world-readable too. Don't put
# secrets here. .env stays in $REPO_DIR per user.
set -euo pipefail

ROOT="${SAE_MUC_SHARED:-/mnt/ssd/sae-muc}"
GROUP="${SAE_MUC_GROUP:-ipadocker}"

if ! getent group "${GROUP}" >/dev/null; then
    echo "ERR: group '${GROUP}' does not exist on this host" >&2
    exit 1
fi

if [[ ! -w "$(dirname "${ROOT}")" ]]; then
    echo "ERR: parent of ${ROOT} is not writable for $(id -un)" >&2
    echo "     check with: ls -ld $(dirname "${ROOT}")" >&2
    exit 1
fi

mkdir -p "${ROOT}/hf-cache" "${ROOT}/uv-cache" "${ROOT}/runs"

# setgid (2) + group rwx (7) + world rx (5) → new files inherit group ipadocker
chgrp -R "${GROUP}" "${ROOT}"
chmod 2775 "${ROOT}" "${ROOT}/hf-cache" "${ROOT}/uv-cache" "${ROOT}/runs"

# Repair drift: if the tree was already populated (e.g., a teammate ran
# under default umask 022), older subdirs/files may not be group-writable
# and won't carry setgid. Re-asserting fixes it idempotently.
find "${ROOT}" -type d -exec chmod g+ws {} +
find "${ROOT}" -type f -exec chmod g+w {} +

echo "ready: ${ROOT}"
echo
ls -ld "${ROOT}" "${ROOT}"/*
echo
echo "next: scripts/docker/run.sh <gpu_id> run --config <yaml>"
echo "      (SAE_MUC_SHARED defaults to ${ROOT})"
