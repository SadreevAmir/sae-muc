#!/usr/bin/env bash
# remote_run.sh — kick off a pipeline run on the server from your laptop.
#
# Usage:
#   export SAE_MUC_SSH_HOST=user@server                # just host, no path
#   export SAE_MUC_REPO_PATH=/home/user/sae-muc        # default shown
#   ./scripts/remote_run.sh configs/experiment/qwen05b_smoke.yaml
#
# What it does:
#   1. ssh to the server
#   2. git fetch + checkout feature/server-pipeline + pull
#   3. uv sync (idempotent; cached after first run)
#   4. launch `sae-muc run --config <cfg>` inside a new `tmux` session
#   5. print the session name so you can attach with `ssh … tmux attach -t …`
set -euo pipefail

SSH_HOST=${SAE_MUC_SSH_HOST:?"set SAE_MUC_SSH_HOST (e.g. user@server)"}
REPO_PATH=${SAE_MUC_REPO_PATH:-"~/sae-muc"}
BRANCH=${SAE_MUC_BRANCH:-"feature/server-pipeline"}
CONFIG=${1:?"usage: $0 <config-yaml relative to repo root>"}

SESSION="sae-muc-$(date +%Y%m%d-%H%M%S)"

ssh "$SSH_HOST" bash -s <<EOF
set -euo pipefail
cd "$REPO_PATH"
git fetch origin
git checkout "$BRANCH"
git pull --ff-only
uv sync --all-extras

tmux new-session -d -s "$SESSION" \\
    "cd '$REPO_PATH' && uv run sae-muc run --config '$CONFIG' 2>&1 | tee -a data/run.log"

echo "tmux session launched: $SESSION"
echo "attach: ssh $SSH_HOST tmux attach -t $SESSION"
EOF
