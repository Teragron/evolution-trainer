#!/usr/bin/env bash
# One-shot grab of just the winner (a few KB). Run on your LOCAL machine.
#   ./deploy/pull_best.sh user@1.2.3.4 '~/evolution_game/runs/arch' runs/arch
set -euo pipefail
HOST="${1:?usage: pull_best.sh user@host remote_run_dir [local_dir]}"
REMOTE="${2:?}"
LOCAL="${3:-runs/pulled}"
mkdir -p "$LOCAL"
for f in config.json best.npz history.csv; do
    scp -q "$HOST:$REMOTE/$f" "$LOCAL/$f" && echo "got $f"
done
echo
echo "python main.py watch --run $LOCAL"
