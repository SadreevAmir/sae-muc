#!/usr/bin/env bash
# sync_artifacts.sh — rsync a run directory from the server back to local.
#
# Usage:
#   export SAE_MUC_SSH=user@server:/absolute/path/to/sae-muc
#   ./scripts/sync_artifacts.sh <run_id>
#
# Run dir resolution:
#   $SAE_MUC_RUNS_REMOTE/<run_id>/  if SAE_MUC_RUNS_REMOTE is set
#   else $SAE_MUC_SSH/data/runs/<run_id>/  (legacy: per-user repo-local runs)
#
# On caniculus the team writes to shared storage at /mnt/ssd/sae-muc/runs.
# Drop this in your shell rc once and forget it:
#   export SAE_MUC_RUNS_REMOTE=user@caniculus:/mnt/ssd/sae-muc/runs
#
# By default only small artefacts (parquet / json / manifests) are pulled.
# Hidden-state tensors and intervention/**/safetensors are excluded — they
# are big and typically only needed for deep analysis. Pass --heavy to
# also fetch safetensors.
set -euo pipefail

HEAVY=0
RUN_ID=""
for arg in "$@"; do
    case "$arg" in
        --heavy) HEAVY=1 ;;
        *) RUN_ID="$arg" ;;
    esac
done

if [[ -z "$RUN_ID" ]]; then
    echo "usage: $0 [--heavy] <run_id>" >&2
    exit 1
fi

if [[ -n "${SAE_MUC_RUNS_REMOTE:-}" ]]; then
    SRC="${SAE_MUC_RUNS_REMOTE}/${RUN_ID}/"
elif [[ -n "${SAE_MUC_SSH:-}" ]]; then
    SRC="${SAE_MUC_SSH}/data/runs/${RUN_ID}/"
else
    echo "ERR: set SAE_MUC_RUNS_REMOTE=user@server:/abs/runs/path" >&2
    echo "  or SAE_MUC_SSH=user@server:/abs/repo/path (legacy)" >&2
    exit 1
fi

DST="data/runs/$RUN_ID/"
mkdir -p "$DST"

EXCLUDES=(
    --exclude='hidden_states/*.safetensors'
    --exclude='hidden_states/embedding.safetensors'
    --exclude='vuf/*.safetensors'
)
if [[ $HEAVY -eq 0 ]]; then
    EXCLUDES+=(--exclude='*.safetensors')
fi

rsync -avzP "${EXCLUDES[@]}" "$SRC" "$DST"
echo ""
echo "pulled $RUN_ID into $DST"
du -sh "$DST"
