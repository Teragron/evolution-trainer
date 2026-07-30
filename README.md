# Evolution (Python)

A Python re-implementation of [Keiwan's Evolution](https://keiwan.itch.io/evolution)
([source](https://github.com/keiwando/evolution)): draw a creature out of joints,
bones and muscles, then let a genetic algorithm evolve a neural network that
makes it run. Unlike the original — which simulates one creature at a time on the
main thread — this version trains **headlessly across every CPU core**, so a
generation that takes a minute in the browser takes a second or two here.

## Install

```bash
pip install -r requirements-gui.txt    # local: training + editor + viewer
pip install -r requirements.txt        # training box: headless, no SDL needed
```

Versions are pinned on purpose. pymunk's solver *is* the fitness landscape — a
different pymunk on the training box means the gait you evolved there will drift
when you replay it locally. Every run records the versions it was trained with
and the viewer warns you if they don't match.

## Quick start

```bash
# 1. see a built-in design flail around with an untrained brain
python main.py preview --preset quadruped

# 2. evolve it (uses all cores; ctrl-C is safe, progress is checkpointed)
python main.py train --preset quadruped --run runs/quad --generations 300

# 3. watch the winner
python main.py watch --run runs/quad
```

Got a lot of cores? Use the island model instead of one big population:

```bash
python main.py island --preset quadruped --run runs/arch --islands 8 \
    --population 150 --generations 800
python main.py watch --run runs/arch
```

To design your own:

```bash
python main.py editor --out my_creature.json
python main.py train --creature my_creature.json --run runs/mine --generations 400
python main.py watch --run runs/mine
```

## The editor

| key | action |
|---|---|
| `1` | **joint** mode — click empty space to place a joint |
| `2` | **bone** mode — drag joint → joint (drag into empty space makes a new joint too) |
| `3` | **muscle** mode — drag bone → bone |
| `4` | **move** mode — drag a joint to reposition it |
| `5` | **delete** mode — click a joint, bone or muscle |
| `tab` | cycle modes |
| `ctrl+S` / `ctrl+Z` | save / undo |
| `G` | toggle 0.1-unit grid snap |
| `D` | drop the creature onto the ground |
| `C` | clear |
| wheel / shift+drag | zoom / pan |

Rules of thumb: a creature needs at least two bones sharing a joint and one
muscle spanning them, and the whole skeleton must be connected. The HUD tells
you when it is trainable.

## The viewer

`space` pause · `R` restart · `+` / `-` playback speed · `F` camera follow ·
`M` muscle overlay · `←` / `→` step through checkpointed generations ·
`B` jump to the all-time best · `esc` quit. Muscles are tinted red when
contracting and blue when expanding; the graph in the corner is best/mean
fitness per generation, read live from `history.csv`.

## How it works

**Bodies.** Every bone is a rigid pymunk capsule whose mass is proportional to
its length. Bones that share a joint are tied together with a `PivotJoint`, so
the skeleton behaves like a real linkage. Creature parts never collide with each
other (shared collision filter group) but they do collide with the ground.

**Muscles.** Each muscle is a damped spring between two bone centres. The brain
sets its rest length anywhere between 55% and 145% of the length it was drawn
at; contracting one pulls its two bones together, which rotates them about their
shared joint. Spring stiffness and the force clamp are scaled by total body
mass (`--stiffness-per-kg`, `--force-per-kg`), which is what stops evolution
from discovering "vibrate at 30 Hz and teleport" instead of a gait.

**Brains.** A plain feed-forward network (`--hidden 24 16` for two layers),
`tanh` hidden units, logistic outputs — one output per muscle. Inputs, all
normalised:

- height above ground of every joint
- `sin`/`cos` of every bone's angle, plus its x and y velocity
- a ground-contact flag per bone
- the muscle activations from the previous tick (this feedback is what lets
  static networks produce rhythmic gaits)
- a constant bias

Physics runs at 120 Hz and the brain at 30 Hz by default.

**Evolution.** Genomes are flat float32 weight vectors, so the operators are
pure array maths: tournament selection (or rank-proportional roulette with
`--tournament 0`), single-point crossover, per-gene Gaussian mutation plus a
small chance of full re-randomisation, and elitism that carries the top 8%
through untouched. Fitness for the running task is simply how far the centre of
mass moved in `+x` during the simulation.

**Parallelism.** `multiprocessing` with the `spawn` start method (so it behaves
identically on Windows), one worker per core, each holding a pre-built copy of
the creature and config. The population is split into ~4 chunks per worker per
generation, which keeps every core busy even when some creatures fall over early.
BLAS threading is pinned to 1 per worker so NumPy doesn't fight the pool.

Measure your machine:

```bash
python main.py bench --preset quadruped --workers-list 1 2 4 8 0
```

## Tuning

| flag | default | notes |
|---|---|---|
| `--population` | 200 | 100–500 is a sensible range; bigger explores more per generation |
| `--generations` | 300 | running gaits usually show up within 50–150 |
| `--duration` | 10.0 | simulated seconds per creature — the single biggest cost knob |
| `--hidden` | 16 | e.g. `--hidden 24 16`; more weights = slower convergence |
| `--physics-hz` | 120 | drop to 60 to roughly double throughput, at some accuracy cost |
| `--mutation-rate` | 0.06 | raise to escape plateaus, lower to refine |
| `--force-per-kg` | 45 | ≈4.5× body weight; raise for cartoon athletics |
| `--workers` | 0 (all cores) | `train` only |
| `--islands` | 8 | `island` only; aim for ~4 cores each |
| `--migrate-every` | 10 | 0 disables migration (fully independent runs) |
| `--resume` | off | continue from the newest checkpoint in `--run` |

## Run directory layout

```
runs/quad/
  config.json      creature + brain shape + sim/GA settings + version stamps
  history.csv      per-generation best / mean / median / worst / speed
  best.npz         all-time best genome  (~a few KB -- this is the whole model)
  gen/gen_00050.npz  full population + fitness, every --checkpoint-every gens

runs/arch/         an island run adds:
  island_00/       ...a complete run directory per island
  migrants/        ...the migration mailboxes
  config.json, best.npz, history.csv, gen/   ...written by `aggregate`
```

## Training remotely, testing locally

The nice thing about this problem: **the trained artifact is tiny.** A brain is
a few hundred float32 weights, so `best.npz` is a couple of kilobytes. There is
no model checkpoint to shuttle around, no GPU memory to think about. The remote
box does CPU-seconds; your laptop does OpenGL. All that crosses the wire is a
handful of numbers.

### Pick the right machine

Rigid-body physics is a serial, branchy, cache-bound workload. It does **not**
vectorise onto a GPU — renting an A100 for this would be lighting money on fire.
What you want is **many fast cores and nothing else**:

| Instance family | Good because |
|---|---|
| AWS `c7i` / `c7a`, GCP `c3-highcpu`, Azure `Fsv2` | compute-optimised, high clocks, 32–192 vCPU |
| Hetzner `CCX` dedicated vCPU | by far the cheapest per core if you don't need a hyperscaler |
| Anything `*.metal` | no hypervisor jitter, if you care about tight timings |

Avoid burstable instances (`t3`, `e2-micro`): they throttle exactly when you
start using all cores. Prefer spot/preemptible — every run is checkpointed, so
losing the box costs you at most `--checkpoint-every` generations, and `--resume`
picks up where it stopped.

Rough sizing: one core does ~10–25 evaluations/second at the default 10-second,
120 Hz simulation. A 64-core box is therefore ~1000 evals/s, which is a
1200-creature generation every second or so. Measure yours:

```bash
python main.py bench --preset quadruped --workers-list 1 8 32 0
```

### The workflow

```bash
# 1. ship the code (from your laptop; PowerShell and bash both have scp)
ssh user@VM 'mkdir -p ~/evolution_game'
scp -r main.py requirements.txt evolution deploy user@VM:~/evolution_game/
scp my_creature.json user@VM:~/evolution_game/          # if not using a preset

# 2. one-time setup on the box
ssh user@VM 'cd ~/evolution_game && bash deploy/setup_vm.sh'

# 3. launch, detached, so it survives your ssh dropping
ssh user@VM 'cd ~/evolution_game && ISLANDS=16 GENERATIONS=1500 ./deploy/launch.sh'

# 4. back on your laptop: pull results on a loop, in one terminal...
python main.py sync --from user@VM:~/evolution_game/runs/arch --run runs/arch

# 5. ...and watch evolution happen live in another
python main.py watch --run runs/arch --live
```

`watch --live` re-reads the run directory every couple of seconds. When a better
creature appears it swaps the brain in and restarts the simulation, so you get a
running highlight reel of the best-so-far while the box keeps grinding. Press
`L` to toggle it, `O` to toggle auto-replay.

`sync` skips the `gen/` population dumps by default — only `config.json`,
`history.csv` and `best.npz` come down, which is a few KB per poll. Add `--full`
when you want the whole archive at the end. It uses `rsync` when available and
falls back to `ssh` + `scp`, so it works from a stock Windows PowerShell (which
has OpenSSH but no rsync).

When you're done, `./deploy/pull_best.sh user@VM '~/evolution_game/runs/arch'`
grabs just the winner, and you can destroy the box.

### The island model

`main.py island` runs N independent populations instead of one big one. Every
`--migrate-every` generations each island writes its top `--migrants` genomes to
`<run>/migrants/island_XX.npz` and reads whatever its neighbours left there
(`--topology full` reads everyone, `ring` reads one neighbour). Immigrants
replace the worst individuals of the receiving island, never its elites.

Why bother:

1. **Diversity.** One population locks onto the first workable gait and spends
   the next 500 generations polishing it. Eight populations find eight different
   gaits, and migration lets the best one spread.
2. **Scaling.** Coordination is a rename on a shared filesystem, so nothing
   waits on anything. On one VM the islands are just processes. On a cluster,
   submit a job array and pass `--island-id $SLURM_ARRAY_TASK_ID` — same
   protocol, no MPI, no head node.

What migration actually bought, measured on a tiny controlled A/B (4 islands ×
population 12, 12 generations, identical starting populations, `--migrate-every 4`
vs `0`):

| seed | isolated islands | with migration |
|---|---|---|
| 100 | best 3.66 m, islands 1.79–3.66 | best 3.72 m, islands 2.89–3.72 |
| 200 | best 4.56 m, islands 2.82–4.56 | best 5.17 m, islands 3.59–5.17 |

The consistent effect is on the *floor*, not the ceiling — migration drags the
unlucky islands up to the leader instead of letting them waste CPU on a dead
end. The ceiling gain was +2% and +13% on two seeds, which is suggestive but
nowhere near enough runs to call it. Treat the ceiling number as unproven and
the floor effect as the real reason to use it. If you want the honest baseline
for your own creature, run once with `--migrate-every 0` and compare.

`aggregate` rolls the islands up into a normal run directory (global `best.npz`,
merged `history.csv`, and `gen/` checkpoints with every island's population
stacked together) so the viewer can open the archipelago as if it were a single
run. It runs automatically every 30 s while training, and once more at the end.

```bash
python main.py island --preset biped --run runs/arch --islands 16 \
    --population 150 --generations 1500 --migrate-every 10 --migrants 4
python main.py aggregate --run runs/arch --follow     # if running islands by hand
```

Each island is *also* a complete run on its own, so
`python main.py watch --run runs/arch/island_03` works if you want to see what
one island got up to.

### No ssh? (Vast.ai, Runpod, any rented container)

Relay through a Hugging Face repo — code in, results out, nothing needs to
accept an inbound connection:

```bash
# on the box
python main.py push --run runs/arch --repo YOU/evo-run --follow --interval 180
# on your laptop
python main.py sync --from hf://YOU/evo-run --run runs/arch --interval 60
python main.py watch --run runs/arch --live
```

Or, if you have a mapped port, `main.py serve --run runs/arch --port 8080
--token secret` on the box and `sync --from http://host:8080 --token secret`
locally. Full walkthrough in [VASTAI.md](VASTAI.md).

### Cluster note

Nothing above is cloud-specific. For SLURM, the whole job script is:

```bash
#SBATCH --array=0-15
#SBATCH --cpus-per-task=8
python main.py island --island-id $SLURM_ARRAY_TASK_ID --islands 16 \
    --workers-per-island 8 --run $SHARED_FS/runs/arch \
    --preset quadruped --population 150 --generations 1500
```

as long as `$SHARED_FS` is visible to every node.

## Project layout

```
main.py                CLI: editor / train / island / sync / push / serve / watch / ...
evolution/creature.py  joints, bones, muscles, JSON load/save, validation
evolution/brain.py     genome <-> weight matrices, forward pass
evolution/physics.py   pymunk world, sensors, muscle actuation, fitness
evolution/ga.py        selection, crossover, mutation, elitism
evolution/trainer.py   process pool, checkpointing, history
evolution/editor.py    pygame creature editor
evolution/viewer.py    pygame playback + fitness graph
evolution/render.py    shared drawing helpers
evolution/island.py    island model: migration mailboxes, aggregation, launcher
evolution/sync.py      rsync/scp pull loop for remote runs
evolution/presets.py   quadruped, biped, worm, tripod
deploy/setup_vm.sh     one-time dependency install + throughput check
deploy/launch.sh       detached tmux island run, sized from nproc
deploy/pull_best.sh    grab just the winner
deploy/vastai_*.sh     sudo-free setup + nohup launch for rental containers
deploy/upload_to_hf.py ship the trainer to a HF repo using your own hf login
evolution/serve.py     read-only http server for a run dir
evolution/hf.py        Hugging Face repo as the transport in and out
```

## What's not here yet

Only the **running** task is implemented. Jumping, obstacle jump and climbing
are additions to `Simulation.fitness()` (plus, for the last two, some extra
static geometry in `_build_ground`) — the rest of the pipeline is task-agnostic.
