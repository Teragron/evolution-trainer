# Running on Vast.ai (or any container you can't ssh into)

Vast.ai gives you a Jupyter tab, not an ssh account, and usually no inbound
ports. So the plan is: **code in via a Hugging Face repo, results out via the
same repo.** Nothing needs to accept an inbound connection.

## What to put in the HF repo

Everything except your local-only bits. From `evolution_game/`, upload:

```
main.py                  <- the CLI
requirements.txt         <- numpy + pymunk, pinned
evolution/               <- the whole package (all 12 .py files)
deploy/vastai_setup.sh   <- installer for the container
deploy/vastai_launch.sh  <- starts training with nohup
my_creature.json         <- only if you designed your own
```

Skip: `runs/` (old results), `__pycache__/`, `requirements-gui.txt` and
`viewer_*.png`. The GUI files inside `evolution/` (`editor.py`, `viewer.py`,
`render.py`) are harmless to include — they're only imported when you actually
open a window, so a box with no pygame never touches them. Uploading the whole
`evolution/` folder is simpler than cherry-picking, and the total is well under
200 KB.

Make it a **model** repo (the default). Create it at
huggingface.co/new, then drag the files into the web uploader — or from your
laptop:

```bash
pip install huggingface_hub
hf auth login                 # older versions: huggingface-cli login
python deploy/upload_to_hf.py --repo YOU/evo-run --dry-run   # check the list
python deploy/upload_to_hf.py --repo YOU/evo-run             # do it
```

`upload_to_hf.py` picks exactly the right files, creates the repo private by
default, and prints the next commands with your repo name filled in. It reads
your login from the hf CLI's own credential store, so no token goes into a
script or a chat window.

If the repo is private, the training box needs a token with read access; if you
also push results back, it needs write.

## On the Vast.ai instance

Open the Jupyter tab, then **File → New → Terminal**, and:

```bash
pip install -q huggingface_hub
hf download YOU/evo-run --local-dir ~/evo    # add --token hf_xxx if private
cd ~/evo
bash deploy/vastai_setup.sh                  # installs deps, prints core count
```

`vastai_setup.sh` finishes with a throughput benchmark. Read it before you
commit to a long run — see the warning below.

```bash
bash deploy/vastai_launch.sh                 # nohup, survives closing the tab
export HF_TOKEN=hf_xxx                       # write token
nohup python3 main.py push --run runs/arch --repo YOU/evo-run \
      --follow --interval 180 > push.log 2>&1 &
```

That second command is the return path: every 3 minutes it commits
`config.json`, `history.csv` and `best.npz` to the repo. That's a few KB, so the
commit history becomes a readable training log — each message says
`best 14.82 m at generation 240`.

## Back on your laptop

```bash
python main.py sync  --from hf://YOU/evo-run --run runs/arch --interval 60
python main.py watch --run runs/arch --live
```

Two terminals: the first pulls, the second notices new bests and swaps the brain
in mid-flight. Set `--interval 60` rather than the default 15 — there's no point
polling HF faster than the box pushes.

Grab everything including the population archive at the end with
`python main.py sync --from hf://YOU/evo-run --run runs/arch --once --full`.

## An honest warning about Vast.ai for this workload

Vast.ai rents **GPUs**. This simulation is rigid-body physics: serial, branchy,
cache-bound, and it cannot use a GPU at all. Renting a 4090 to run it means
paying for silicon that will sit at 0% for the entire run.

Worse, GPU instances tend to allocate vCPUs in proportion to GPUs, so a
single-GPU box often gives you only 4–16 usable cores — possibly fewer than your
own desktop. **Check `nproc` and the benchmark before you commit.**

If you're set on Vast.ai, filter the search by **CPU cores** rather than GPU
model and look for the cheapest instance with a high core count. Otherwise a
compute-optimised VM elsewhere (Hetzner CCX, AWS `c7i`, GCP `c3-highcpu`) is
both cheaper and several times faster for this specific job — see the main
README.

One more container gotcha, already handled: `os.cpu_count()` inside a container
reports the *host's* cores, not your cgroup quota. The trainer reads the quota
and the affinity mask instead, so `--workers 0` won't spawn 128 processes on an
8-core allocation. `main.py bench` prints both numbers when they disagree.

## If you do have a mapped port

Vast.ai can expose extra ports at instance-creation time. If you have one, you
can skip HF for the download direction:

```bash
# on the box
python3 main.py serve --run runs/arch --port 8080 --token something-secret
# on your laptop
python main.py sync --from http://INSTANCE_IP:PORT --token something-secret --run runs/arch
```

It's read-only and serves nothing but that directory, but it's plain HTTP — use
the token and shut it down when the run ends.
