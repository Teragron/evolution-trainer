#!/usr/bin/env bash
# Run this in the Vast.ai Jupyter TERMINAL, from wherever you put the code.
#
#   bash deploy/vastai_setup.sh
#
# Differences from setup_vm.sh: no sudo (containers already run as root and
# often have no sudo installed), no venv (the container's python is yours
# alone), and no tmux dependency for launching.
set -euo pipefail

cd "$(dirname "$0")/.."
echo "==> project: $(pwd)"

echo "==> installing python deps (wheels only, no compiler needed)"
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt
pip install --quiet huggingface_hub          # the results relay

python3 - <<'PY'
import os, platform, numpy, pymunk
import sys
sys.path.insert(0, ".")
from evolution.trainer import available_cores
usable, host = available_cores(), os.cpu_count()
print()
print(f"    python {platform.python_version()}  numpy {numpy.__version__}  pymunk {pymunk.version}")
print(f"    ^ pin these same versions on your laptop or replays will drift")
print()
print(f"    usable cores: {usable}" + (f"  (the host has {host}, but this"
      f" container is limited to {usable})" if usable != host else ""))
if usable < 8:
    print("    NOTE: this is a small CPU allocation. Physics is CPU-bound and")
    print("          cannot use the GPU you are renting -- consider an instance")
    print("          filtered on high vCPU count instead.")
PY

echo
echo "==> throughput check"
python3 main.py bench --preset quadruped --duration 10 --population 60 --workers-list 1 0

cat <<'TXT'

Next:
  1. log in so results can be pushed back:
       hf auth login                  # or: export HF_TOKEN=hf_xxx
  2. start training:
       bash deploy/vastai_launch.sh
  3. relay results to your HF repo (leave running):
       nohup python3 main.py push --run runs/arch --repo YOU/evo-run \
             --follow --interval 180 > push.log 2>&1 &
TXT
