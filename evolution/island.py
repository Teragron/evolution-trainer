"""Island-model training: many independent populations that trade migrants.

Each island is an ordinary `Trainer` writing to `<run>/island_XX/`, so every
island is itself a complete, viewable run. Islands never talk to each other
directly -- every `migrate_every` generations an island drops its best genomes
into `<run>/migrants/island_XX.npz` and picks up whatever its neighbours have
left there. The only coordination primitive is the filesystem, which means the
same code runs as N processes on one fat VM or as N nodes of a cluster sharing
a network mount, with no MPI and no head node.

Why bother instead of one huge population? Because a single population
converges on the first gait it stumbles into and then polishes it forever.
Eight populations of 150 find eight different gaits and migration lets the best
one take over -- the same total CPU, but far less of it spent inbreeding.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from typing import Callable, List, Optional, Sequence

import numpy as np

from .creature import Creature
from .ga import GAConfig, Population
from .physics import SimConfig
from .trainer import GenStats, Trainer

MIGRANT_DIR = "migrants"


@dataclass
class IslandConfig:
    islands: int = 8
    migrate_every: int = 10          # generations between exchanges
    migrants: int = 4                # genomes offered per exchange
    topology: str = "full"           # "full" = read everyone, "ring" = read left neighbour
    seed_base: int = 1000

    def to_dict(self) -> dict:
        return asdict(self)


def island_dir(run_dir: str, index: int) -> str:
    return os.path.join(run_dir, f"island_{index:02d}")


# --------------------------------------------------------------------- migration
class Migrator:
    """Filesystem mailbox for one island."""

    def __init__(self, run_dir: str, index: int, cfg: IslandConfig):
        self.dir = os.path.join(run_dir, MIGRANT_DIR)
        self.index = index
        self.cfg = cfg
        os.makedirs(self.dir, exist_ok=True)
        self.sent = 0
        self.received = 0

    def _path(self, index: int) -> str:
        return os.path.join(self.dir, f"island_{index:02d}.npz")

    def _peers(self) -> List[int]:
        n = self.cfg.islands
        if n <= 1:
            return []
        if self.cfg.topology == "ring":
            return [(self.index - 1) % n]
        return [i for i in range(n) if i != self.index]

    def offer(self, genomes: np.ndarray, fitness: np.ndarray, generation: int) -> None:
        """Publish our best genomes. Written to a temp file then renamed, so a
        peer can never read a half-written array."""
        k = min(self.cfg.migrants, len(genomes))
        best = np.argsort(fitness)[::-1][:k]
        # note the ".npz" suffix on the temp name: numpy silently appends
        # ".npz" to any path that lacks it, which would break the rename.
        tmp = self._path(self.index).replace(".npz", ".tmp.npz")
        np.savez_compressed(tmp, genomes=genomes[best],
                            fitness=fitness[best],
                            generation=np.int64(generation))
        os.replace(tmp, self._path(self.index))
        self.sent += k

    def collect(self) -> List[np.ndarray]:
        """Read peers' mailboxes, best genome first."""
        pool: List[tuple] = []
        for peer in self._peers():
            path = self._path(peer)
            if not os.path.exists(path):
                continue
            try:
                data = np.load(path)
                for g, f in zip(data["genomes"], data["fitness"]):
                    pool.append((float(f), g))
            except (OSError, ValueError, KeyError):
                continue  # peer was mid-write on a filesystem without atomic rename
        pool.sort(key=lambda t: -t[0])
        self.received += len(pool)
        return [g for _, g in pool]


def inject(pop: Population, migrants: Sequence[np.ndarray], keep_elites: int) -> int:
    """Overwrite the worst slots of a fresh population with immigrants.

    The population has already been rebuilt by `evolve`, so slots [0:keep_elites]
    hold this island's own elites -- we replace from the tail instead and never
    touch them.
    """
    if not migrants:
        return 0
    room = max(0, len(pop.genomes) - keep_elites)
    n = min(len(migrants), room)
    for i in range(n):
        pop.genomes[-1 - i] = np.asarray(migrants[i], dtype=np.float32)
    return n


# ------------------------------------------------------------------ single island
def run_island(creature: Creature, run_dir: str, index: int,
               generations: int,
               icfg: IslandConfig,
               hidden: Sequence[int] = (16,),
               sim_cfg: Optional[SimConfig] = None,
               ga_cfg: Optional[GAConfig] = None,
               workers: Optional[int] = None,
               checkpoint_every: int = 10,
               resume: bool = False,
               quiet: bool = False,
               on_generation: Optional[Callable] = None) -> Trainer:
    """Train one island, exchanging migrants on schedule."""
    idir = island_dir(run_dir, index)
    trainer = Trainer(creature, idir, hidden=hidden, sim_cfg=sim_cfg,
                      ga_cfg=ga_cfg, workers=workers,
                      seed=icfg.seed_base + index)
    start_gen = trainer.load_latest_generation() if resume else 0
    mig = Migrator(run_dir, index, icfg)
    n_elite = max(1, int(round(trainer.ga_cfg.elite_frac * trainer.ga_cfg.population)))
    state = {"pending": None}

    def _before(stats: GenStats, fitness: np.ndarray, genomes: np.ndarray) -> None:
        if icfg.islands > 1 and icfg.migrate_every > 0 \
                and stats.generation % icfg.migrate_every == 0:
            mig.offer(genomes, fitness, stats.generation)
            state["pending"] = mig.collect()
        if not quiet:
            print(f"island {index:02d} gen {stats.generation:5d}  "
                  f"best {stats.best:7.2f}m  mean {stats.mean:7.2f}m  "
                  f"all-time {trainer.best_fitness:7.2f}m  "
                  f"{stats.evals_per_sec:6.1f} ev/s  "
                  f"in {mig.received} out {mig.sent}", flush=True)
        if on_generation:
            on_generation(index, stats)

    def _after(pop: Population, generation: int) -> None:
        pending = state.pop("pending", None)
        state["pending"] = None
        if pending:
            inject(pop, pending, n_elite)

    if not quiet:
        print(f"island {index:02d} starting at generation {start_gen} "
              f"({trainer.workers} workers, population {trainer.ga_cfg.population})",
              flush=True)
    trainer.run(generations, checkpoint_every=checkpoint_every,
                on_generation=_before, after_evolve=_after)
    return trainer


# --------------------------------------------------------------------- aggregate
def aggregate(run_dir: str, verbose: bool = True) -> dict:
    """Roll the islands up into a run directory the viewer can open directly.

    Writes `<run>/config.json`, `<run>/best.npz`, `<run>/history.csv` and
    `<run>/gen/*.npz` (all islands' populations for that generation, stacked).
    Safe to call while training is still going.
    """
    dirs = sorted(d for d in os.listdir(run_dir)
                  if d.startswith("island_") and
                  os.path.isdir(os.path.join(run_dir, d)))
    if not dirs:
        raise FileNotFoundError(f"No island_* directories in {run_dir}")

    # ---- config comes from the first island that has one
    meta = None
    for d in dirs:
        p = os.path.join(run_dir, d, "config.json")
        if os.path.exists(p):
            with open(p, encoding="utf-8") as fh:
                meta = json.load(fh)
            break
    if meta is None:
        raise FileNotFoundError("No island has written config.json yet.")
    meta["islands"] = len(dirs)
    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)

    # ---- global best
    best = None
    for d in dirs:
        p = os.path.join(run_dir, d, "best.npz")
        if not os.path.exists(p):
            continue
        try:
            data = np.load(p)
        except (OSError, ValueError):
            continue
        fit = float(data["fitness"])
        if best is None or fit > best[0]:
            best = (fit, data["genome"], int(data["generation"]), d)
    if best is not None:
        np.savez_compressed(os.path.join(run_dir, "best.npz"),
                            genome=best[1], fitness=np.float64(best[0]),
                            generation=np.int64(best[2]))

    # ---- history: per generation, best across islands and mean of means
    per_gen: dict = {}
    for d in dirs:
        p = os.path.join(run_dir, d, "history.csv")
        if not os.path.exists(p):
            continue
        import csv as _csv
        with open(p, newline="", encoding="utf-8") as fh:
            for row in _csv.DictReader(fh):
                g = int(row["generation"])
                slot = per_gen.setdefault(g, {"best": [], "mean": [], "median": [],
                                              "worst": [], "eps": []})
                slot["best"].append(float(row["best"]))
                slot["mean"].append(float(row["mean"]))
                slot["median"].append(float(row["median"]))
                slot["worst"].append(float(row["worst"]))
                slot["eps"].append(float(row["evals_per_sec"]))
    if per_gen:
        import csv as _csv
        with open(os.path.join(run_dir, "history.csv"), "w", newline="",
                  encoding="utf-8") as fh:
            w = _csv.writer(fh)
            w.writerow(["generation", "best", "mean", "median", "worst",
                        "seconds", "evals_per_sec"])
            for g in sorted(per_gen):
                s = per_gen[g]
                w.writerow([g, f"{max(s['best']):.5f}",
                            f"{float(np.mean(s['mean'])):.5f}",
                            f"{float(np.median(s['median'])):.5f}",
                            f"{min(s['worst']):.5f}", "0.000",
                            f"{sum(s['eps']):.1f}"])

    # ---- stacked generation checkpoints
    gout = os.path.join(run_dir, "gen")
    os.makedirs(gout, exist_ok=True)
    names: dict = {}
    for d in dirs:
        gdir = os.path.join(run_dir, d, "gen")
        if not os.path.isdir(gdir):
            continue
        for f in os.listdir(gdir):
            if f.endswith(".npz"):
                names.setdefault(f, []).append(os.path.join(gdir, f))
    stacked = 0
    for name, paths in sorted(names.items()):
        target = os.path.join(gout, name)
        if os.path.exists(target) and \
                os.path.getmtime(target) >= max(os.path.getmtime(p) for p in paths):
            continue
        gs, fs = [], []
        for p in paths:
            try:
                data = np.load(p)
            except (OSError, ValueError):
                continue
            gs.append(data["genomes"]); fs.append(data["fitness"])
        if not gs:
            continue
        np.savez_compressed(target, genomes=np.concatenate(gs),
                            fitness=np.concatenate(fs))
        stacked += 1

    summary = {
        "islands": len(dirs),
        "best_fitness": None if best is None else best[0],
        "best_generation": None if best is None else best[2],
        "best_island": None if best is None else best[3],
        "generations_recorded": len(per_gen),
        "checkpoints_stacked": stacked,
    }
    if verbose:
        if best is None:
            print(f"aggregate: {len(dirs)} islands, no best.npz yet")
        else:
            print(f"aggregate: {len(dirs)} islands | best {best[0]:.2f} m "
                  f"from {best[3]} gen {best[2]} | "
                  f"{len(per_gen)} generations | +{stacked} checkpoints")
    return summary


# ------------------------------------------------------------------- local launch
def launch_local(argv_builder: Callable[[int], List[str]], islands: int,
                 run_dir: str, aggregate_every: float = 30.0) -> int:
    """Spawn N island processes on this machine and aggregate while they run.

    Used by `main.py island` without `--island-id`. On a cluster you would
    instead submit N tasks that each call `--island-id $TASK_ID`; the migration
    protocol is identical either way.
    """
    os.makedirs(run_dir, exist_ok=True)
    procs = []
    for i in range(islands):
        idir = island_dir(run_dir, i)
        os.makedirs(idir, exist_ok=True)
        log = open(os.path.join(idir, "train.log"), "a", buffering=1, encoding="utf-8")
        p = subprocess.Popen(argv_builder(i), stdout=log, stderr=subprocess.STDOUT)
        procs.append((i, p, log))
        print(f"  island {i:02d}  pid {p.pid}  -> {os.path.join(idir, 'train.log')}")
    print()

    last = 0.0
    try:
        while any(p.poll() is None for _, p, _ in procs):
            time.sleep(2.0)
            if time.time() - last > aggregate_every:
                last = time.time()
                try:
                    aggregate(run_dir)
                except (FileNotFoundError, OSError) as exc:
                    print(f"aggregate: waiting ({exc})")
    except KeyboardInterrupt:
        print("\ninterrupted - stopping islands...")
        for _, p, _ in procs:
            if p.poll() is None:
                p.terminate()
        for _, p, _ in procs:
            try:
                p.wait(timeout=20)
            except subprocess.TimeoutExpired:
                p.kill()
    finally:
        for _, _, log in procs:
            log.close()

    codes = [(i, p.returncode) for i, p, _ in procs]
    bad = [f"island {i} exited {c}" for i, c in codes if c not in (0, None, -15)]
    if bad:
        print("warnings: " + "; ".join(bad))
    try:
        aggregate(run_dir)
    except (FileNotFoundError, OSError) as exc:
        print(f"final aggregate failed: {exc}")
    return 0 if not bad else 1
