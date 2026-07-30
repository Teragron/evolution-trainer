#!/usr/bin/env bash
# Space container entrypoint: train in the background, push checkpoints to
# a HF repo in the background, serve the run dir in the foreground so the
# Space has a live process to health-check (and something to poll directly).
set -euo pipefail
cd /app

RUN="${RUN:-/app/runs/arch}"
PRESET="${PRESET:-quadruped}"
POPULATION="${POPULATION:-200}"
GENERATIONS="${GENERATIONS:-5000}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-20}"
PUSH_INTERVAL="${PUSH_INTERVAL:-180}"
PORT="${PORT:-7860}"

mkdir -p "$RUN"

echo "[space] training: preset=$PRESET population=$POPULATION generations=$GENERATIONS"
nohup python main.py train --preset "$PRESET" --run "$RUN" \
    --population "$POPULATION" --generations "$GENERATIONS" \
    --checkpoint-every "$CHECKPOINT_EVERY" --workers 0 --resume \
    > "$RUN/train.log" 2>&1 &
echo "[space] train pid $!"

if [[ -n "${PUSH_REPO:-}" ]]; then
    echo "[space] pushing checkpoints to hf://$PUSH_REPO every ${PUSH_INTERVAL}s"
    nohup python main.py push --run "$RUN" --repo "$PUSH_REPO" \
        --repo-type "${PUSH_REPO_TYPE:-dataset}" --follow \
        --interval "$PUSH_INTERVAL" \
        > "$RUN/push.log" 2>&1 &
    echo "[space] push pid $!"
else
    echo "[space] PUSH_REPO not set -- checkpoints stay in this container only"
fi

echo "[space] serving $RUN on :$PORT"
exec python main.py serve --run "$RUN" --port "$PORT" --host 0.0.0.0
