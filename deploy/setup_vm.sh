#!/usr/bin/env bash
# Run this ON the training box, once, from the project root.
#   ssh user@vm 'bash -s' < deploy/setup_vm.sh      # or just scp it up and run it
#
# Installs only the headless dependencies -- no pygame, no SDL, no display.
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/evolution_game}"

echo "==> project dir: $PROJECT_DIR"
cd "$PROJECT_DIR"

if command -v apt-get >/dev/null 2>&1; then
    echo "==> installing system packages"
    sudo apt-get update -qq
    sudo apt-get install -y -qq python3 python3-venv python3-dev build-essential tmux
elif command -v dnf >/dev/null 2>&1; then
    sudo dnf install -y python3 python3-devel gcc gcc-c++ tmux
fi

echo "==> creating virtualenv"
python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
pip install --quiet --upgrade pip wheel
pip install --quiet -r requirements.txt

CORES=$(nproc)
echo
echo "==> ready: $CORES cores"
python3 - <<'PY'
import pymunk, numpy, platform
print(f"    python {platform.python_version()}  numpy {numpy.__version__}  pymunk {pymunk.version}")
print("    ^ pin these same versions locally or replays will drift slightly")
PY

echo
echo "==> throughput check (a few seconds)"
python3 main.py bench --preset quadruped --duration 10 --population 120 --workers-list 1 0

cat <<TXT

Next:
  ./deploy/launch.sh                 # start training in a detached tmux session
  tmux attach -t evolution           # look at it
  ctrl-b d                           # detach again, training keeps going
TXT
