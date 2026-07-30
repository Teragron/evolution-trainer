#!/usr/bin/env bash
# Start an island-model run in a detached tmux session, so it survives your
# laptop closing / ssh dropping. Run ON the training box, from the project root.
#
#   ./deploy/launch.sh
#   ISLANDS=12 GENERATIONS=1500 PRESET=biped ./deploy/launch.sh
#   CREATURE=my_creature.json RUN=runs/mine ./deploy/launch.sh
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$PROJECT_DIR"

CORES=$(nproc)
# 4 workers per island is a good default: enough cores each to keep the pool
# busy, enough islands to keep the search diverse.
ISLANDS="${ISLANDS:-$(( CORES / 4 > 1 ? CORES / 4 : 1 ))}"
POPULATION="${POPULATION:-150}"
GENERATIONS="${GENERATIONS:-800}"
DURATION="${DURATION:-10}"
MIGRATE_EVERY="${MIGRATE_EVERY:-10}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-10}"
PRESET="${PRESET:-quadruped}"
CREATURE="${CREATURE:-}"
RUN="${RUN:-runs/arch}"
SESSION="${SESSION:-evolution}"

if [[ -n "$CREATURE" ]]; then
    DESIGN=(--creature "$CREATURE")
else
    DESIGN=(--preset "$PRESET")
fi

PY=python3
[[ -x .venv/bin/python ]] && PY=.venv/bin/python

CMD=("$PY" main.py island "${DESIGN[@]}" --run "$RUN"
     --islands "$ISLANDS" --population "$POPULATION"
     --generations "$GENERATIONS" --duration "$DURATION"
     --migrate-every "$MIGRATE_EVERY" --checkpoint-every "$CHECKPOINT_EVERY")

if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "session '$SESSION' already exists -- attach with: tmux attach -t $SESSION"
    exit 1
fi

echo "cores        $CORES"
echo "islands      $ISLANDS x population $POPULATION"
echo "generations  $GENERATIONS   sim $DURATION s"
echo "run dir      $PROJECT_DIR/$RUN"
echo "command      ${CMD[*]}"
echo

mkdir -p "$RUN"
tmux new-session -d -s "$SESSION" "${CMD[*]} 2>&1 | tee -a $RUN/launch.log"
echo "started in tmux session '$SESSION'"
echo
echo "From your laptop:"
echo "  python main.py sync --from $(whoami)@<vm-ip>:$PROJECT_DIR/$RUN --run runs/arch"
echo "  python main.py watch --run runs/arch --live"
