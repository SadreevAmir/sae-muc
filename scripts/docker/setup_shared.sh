#!/usr/bin/env bash
# One-time setup of the team's shared storage on caniculus's /mnt/ssd.
#
# /mnt/ssd is mode 0777 (world-writable, no sudo needed). We carve out
# /mnt/ssd/sae-muc/ owned by group ipadocker with setgid bit, so new
# files inherit the group and team members can read/write each other's
# artefacts.
#
# Run this from any teammate's account; subsequent re-runs are
# idempotent. Verify with `ls -ld /mnt/ssd/sae-muc/*`.
#
# Permission semantics: chgrp and chmod can only be done by the file's
# owner (or root). On a re-run by user B, files created by user A in a
# previous run can't be modified by B. We handle this by:
#   1. Only attempting chgrp/chmod on entries that are actually drifted
#      (find ! -group / ! -perm) — no-op for already-correct files.
#   2. Tolerating failures (2>/dev/null || true) because chgrp on files
#      owned by another teammate is expected and harmless if those files
#      were already set correctly by the original owner's run.
#   3. Reporting any uncorrectable drift at the end so the team knows
#      to ask the original owner to re-run.
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

# Fix group on entries not already in $GROUP. Suppress permission errors:
# chgrp on another teammate's files is expected and harmless (the file is
# already in $GROUP from the original owner's run).
find "${ROOT}" ! -group "${GROUP}" -exec chgrp "${GROUP}" {} + 2>/dev/null || true

# setgid on directories that lack it (g+s on dirs makes new entries
# inherit the parent's group, the whole point of this script).
find "${ROOT}" -type d ! -perm -2000 -exec chmod g+s {} + 2>/dev/null || true

# Group write on entries that lack it.
find "${ROOT}" ! -perm -g+w -exec chmod g+w {} + 2>/dev/null || true

# Top-level: try to assert 2775 on the root and the four cache dirs.
# Silently no-op if we don't own them — they'll stay at whatever the
# original creator made them, which on a healthy tree is already 2775.
chmod 2775 "${ROOT}" "${ROOT}/hf-cache" "${ROOT}/uv-cache" "${ROOT}/runs" 2>/dev/null || true

echo "ready: ${ROOT}"
echo
ls -ld "${ROOT}" "${ROOT}"/*
echo

# Report drift we couldn't repair so the team knows what to chase down.
bad_group=$(find "${ROOT}" ! -group "${GROUP}" 2>/dev/null | wc -l)
bad_dir_setgid=$(find "${ROOT}" -type d ! -perm -2000 2>/dev/null | wc -l)
bad_gw=$(find "${ROOT}" ! -perm -g+w 2>/dev/null | wc -l)
if (( bad_group + bad_dir_setgid + bad_gw > 0 )); then
    echo "warning: drift remains (likely files owned by other teammates):"
    echo "  ${bad_group} entries not in group ${GROUP}"
    echo "  ${bad_dir_setgid} directories without setgid"
    echo "  ${bad_gw} entries without g+w"
    echo "  ask the original owner(s) to re-run scripts/docker/setup_shared.sh"
    echo
fi

echo "next: scripts/docker/run.sh <gpu_id> run --config <yaml>"
echo "      (SAE_MUC_SHARED defaults to ${ROOT})"
