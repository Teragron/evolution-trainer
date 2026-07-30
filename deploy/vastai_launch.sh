#!/usr/bin/env bash
# Start an island run with nohup (tmux is often missing in rental containers).
# Survives the Jupyter tab closing; dies if the instance is destroyed, which is
# why you also want `main.py push --follow` relaying to HF.
#
#   bash deploy/vastai_launch.sh
#   ISLANDS=4 POPULATION=200 GENERATIONS=2000 bash deploy/vastai_launch.sh
set -euo pipefail
cd "$(dirname "$0")/.."

CORES=$(python3 -c 'import sys; sys.path.insert(0,"."); from evolution.trainer import available_cores; print(available_cores())')
ISLANDS="${ISLANDS:-$(( CORES / 4 > 1 ? CORES / 4 : 1 ))}"
POPULATION="${POPULATION:-150}"
GENERATIONS="${GENERATIONS:-1000}"
DURATION="${DURATION:-10}"
PRESET="${PRESET:-quadruped}"
CREATURE="${CREATURE:-}"
RUN="${RUN:-runs/arch}"

if [[ -n "$CREATURE" ]]; then DESIGN=(--creature "$CREATURE"); else DESIGN=(--preset "$PRESET"); fi

mkdir -p "$RUN"
echo "usable cores  $CORES"
echo "islands       $ISLANDS x population $POPULATION"
echo "generations   $GENERATIONS   sim ${DURATION}s"
echo "run dir       $(pwd)/$RUN"
echo

nohup python3 main.py island "${DESIGN[@]}" --run "$RUN" \
    --islands "$ISLANDS" --population "$POPULATION" \
    --generations "$GENERATIONS" --duration "$DURATION" \
    --checkpoint-every 10 > "$RUN/train.log" 2>&1 &

echo "$!" > "$RUN/train.pid"
echo "started pid $(cat "$RUN/train.pid")"
echo
echo "  watch it:   tail -f $RUN/train.log"
echo "  stop it:    kill \$(cat $RUN/train.pid)"
echo "  progress:   column -s, -t $RUN/history.csv | tail"
