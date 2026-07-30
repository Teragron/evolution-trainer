"""Parallel training loop: one process per CPU core, chunked genome batches."""
from __future__ import annotations

import csv
import json
import multiprocessing as mp
import os
import platform
import time
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .brain import Brain, BrainSpec
from .creature import Creature
from .ga import GAConfig, Population
from .physics import SimConfig, Simulation, brain_spec_for

# ---------------------------------------------------------------- worker side
_W: dict = {}


def _worker_init(creature_d: dict, spec_d: dict, cfg_d: dict) -> None:
    # Keep BLAS single-threaded: we already use every core with processes.
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS"):
        os.environ[var] = "1"
    _W["creature"] = Creature.from_dict(creature_d)
    _W["spec"] = BrainSpec.from_dict(spec_d)
    _W["cfg"] = SimConfig.from_dict(cfg_d)


def _eval_chunk(genomes: np.ndarray) -> List[float]:
    creature, spec, cfg = _W["creature"], _W["spec"], _W["cfg"]
    out = []
    for g in genomes:
        sim = Simulation(creature, cfg)
        out.append(sim.run(Brain(spec, g)))
    return out


def evaluate_serial(creature: Creature, spec: BrainSpec, cfg: SimConfig,
                    genomes: np.ndarray) -> np.ndarray:
    return np.array([Simulation(creature, cfg).run(Brain(spec, g))
                     for g in genomes], dtype=np.float64)


def available_cores() -> int:
    """How many cores we may ACTUALLY use.

    `os.cpu_count()` reports the host's cores, not the container's share -- on
    a rented container (Vast.ai, Runpod, any Kubernetes pod) that can be 128
    when your cgroup quota is 8, and spawning 128 workers there is slower than
    spawning 8. Check the scheduler affinity mask and the cgroup CPU quota too,
    and believe the smallest number.
    """
    n = os.cpu_count() or 1
    try:
        n = min(n, len(os.sched_getaffinity(0)))       # taskset / cpuset
    except AttributeError:
        pass                                           # not on Linux
    try:                                               # cgroup v2
        with open("/sys/fs/cgroup/cpu.max", encoding="utf-8") as fh:
            quota, period = fh.read().split()[:2]
        if quota != "max":
            n = min(n, max(1, int(float(quota) / float(period))))
    except (OSError, ValueError, IndexError):
        pass
    try:                                               # cgroup v1
        with open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us", encoding="utf-8") as fh:
            quota = int(fh.read())
        with open("/sys/fs/cgroup/cpu/cpu.cfs_period_us", encoding="utf-8") as fh:
            period = int(fh.read())
        if quota > 0 and period > 0:
            n = min(n, max(1, quota // period))
    except (OSError, ValueError):
        pass
    return max(1, n)


def runtime_versions() -> dict:
    """Stamped into every run so a local replay can warn about drift."""
    import numpy as _np
    import pymunk as _pm
    return {"python": platform.python_version(),
            "numpy": _np.__version__,
            "pymunk": _pm.version}


def version_warnings(meta: dict) -> List[str]:
    """Compare a run's recorded versions against what is installed here."""
    recorded = meta.get("versions") or {}
    if not recorded:
        return ["This run predates version stamping; physics may differ slightly."]
    here = runtime_versions()
    out = []
    for key in ("pymunk", "numpy"):
        if key in recorded and recorded[key] != here[key]:
            note = " (physics may differ)" if key == "pymunk" else ""
            out.append(f"{key} {recorded[key]} on the trainer vs "
                       f"{here[key]} here{note}")
    return out


# ----------------------------------------------------------------- main side
@dataclass
class GenStats:
    generation: int
    best: float
    mean: float
    median: float
    worst: float
    seconds: float
    evals_per_sec: float


class Trainer:
    def __init__(self, creature: Creature, run_dir: str,
                 hidden: Sequence[int] = (16,),
                 sim_cfg: Optional[SimConfig] = None,
                 ga_cfg: Optional[GAConfig] = None,
                 workers: Optional[int] = None,
                 seed: Optional[int] = None):
        problems = creature.problems()
        if problems:
            raise ValueError("Creature is not simulatable:\n  - " + "\n  - ".join(problems))

        self.creature = creature
        self.run_dir = run_dir
        self.sim_cfg = sim_cfg or SimConfig()
        self.ga_cfg = ga_cfg or GAConfig()
        self.spec = brain_spec_for(creature, tuple(hidden))
        self.pop = Population(self.spec.genome_size, self.ga_cfg, seed=seed)
        self.workers = workers if workers and workers > 0 else available_cores()
        self.history: List[GenStats] = []
        self.best_genome: Optional[np.ndarray] = None
        self.best_fitness = -np.inf
        self.best_generation = -1
        self._pool: Optional[mp.pool.Pool] = None

        os.makedirs(os.path.join(run_dir, "gen"), exist_ok=True)
        self._write_config(seed)

    # --------------------------------------------------------------- config
    def _write_config(self, seed) -> None:
        meta = {
            "versions": runtime_versions(),
            "creature": self.creature.to_dict(),
            "brain": self.spec.to_dict(),
            "sim": self.sim_cfg.to_dict(),
            "ga": self.ga_cfg.to_dict(),
            "seed": seed,
            "hidden": list(self.spec.layer_sizes[1:-1]),
        }
        with open(os.path.join(self.run_dir, "config.json"), "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2)

    @staticmethod
    def load_config(run_dir: str) -> dict:
        with open(os.path.join(run_dir, "config.json"), "r", encoding="utf-8") as fh:
            return json.load(fh)

    # ----------------------------------------------------------------- pool
    def _ensure_pool(self):
        if self.workers <= 1:
            return None
        if self._pool is None:
            ctx = mp.get_context("spawn")
            self._pool = ctx.Pool(
                processes=self.workers,
                initializer=_worker_init,
                initargs=(self.creature.to_dict(), self.spec.to_dict(),
                          self.sim_cfg.to_dict()),
            )
        return self._pool

    def close(self) -> None:
        if self._pool is not None:
            self._pool.close()
            self._pool.join()
            self._pool = None

    # ----------------------------------------------------------- evaluation
    def evaluate(self, genomes: np.ndarray) -> np.ndarray:
        pool = self._ensure_pool()
        if pool is None:
            return evaluate_serial(self.creature, self.spec, self.sim_cfg, genomes)
        n = len(genomes)
        # a few chunks per worker keeps cores busy even with uneven costs
        n_chunks = min(n, self.workers * 4)
        chunks = np.array_split(genomes, n_chunks)
        results = pool.map(_eval_chunk, chunks)
        return np.array([f for part in results for f in part], dtype=np.float64)

    # ------------------------------------------------------------- training
    def run(self, generations: int, checkpoint_every: int = 10,
            on_generation=None, after_evolve=None) -> List[GenStats]:
        try:
            for _ in range(generations):
                t0 = time.perf_counter()
                genomes = self.pop.genomes
                fitness = self.evaluate(genomes)
                elapsed = time.perf_counter() - t0

                gen = self.pop.generation
                best_i = int(np.argmax(fitness))
                if fitness[best_i] > self.best_fitness:
                    self.best_fitness = float(fitness[best_i])
                    self.best_genome = genomes[best_i].copy()
                    self.best_generation = gen
                    self._save_best()

                stats = GenStats(
                    generation=gen,
                    best=float(fitness.max()),
                    mean=float(fitness.mean()),
                    median=float(np.median(fitness)),
                    worst=float(fitness.min()),
                    seconds=elapsed,
                    evals_per_sec=len(genomes) / max(elapsed, 1e-9),
                )
                self.history.append(stats)
                self._append_history(stats)
                if on_generation:
                    on_generation(stats, fitness, genomes)
                if checkpoint_every and gen % checkpoint_every == 0:
                    self._save_generation(gen, genomes, fitness)

                self.pop.evolve(fitness)
                if after_evolve is not None:
                    after_evolve(self.pop, gen)
        finally:
            self.close()
        return self.history

    # -------------------------------------------------------------- outputs
    def _save_best(self) -> None:
        np.savez_compressed(
            os.path.join(self.run_dir, "best.npz"),
            genome=self.best_genome,
            fitness=np.float64(self.best_fitness),
            generation=np.int64(self.best_generation),
        )

    def _save_generation(self, gen: int, genomes: np.ndarray,
                         fitness: np.ndarray) -> None:
        np.savez_compressed(
            os.path.join(self.run_dir, "gen", f"gen_{gen:05d}.npz"),
            genomes=genomes, fitness=fitness)

    def _append_history(self, s: GenStats) -> None:
        path = os.path.join(self.run_dir, "history.csv")
        new = not os.path.exists(path)
        with open(path, "a", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            if new:
                w.writerow(["generation", "best", "mean", "median", "worst",
                            "seconds", "evals_per_sec"])
            w.writerow([s.generation, f"{s.best:.5f}", f"{s.mean:.5f}",
                        f"{s.median:.5f}", f"{s.worst:.5f}",
                        f"{s.seconds:.3f}", f"{s.evals_per_sec:.1f}"])

    # --------------------------------------------------------------- resume
    def load_latest_generation(self) -> int:
        gdir = os.path.join(self.run_dir, "gen")
        files = sorted(f for f in os.listdir(gdir) if f.endswith(".npz")) \
            if os.path.isdir(gdir) else []
        if not files:
            return 0
        data = np.load(os.path.join(gdir, files[-1]))
        genomes = data["genomes"]
        if genomes.shape[1] != self.spec.genome_size:
            raise ValueError("Checkpoint genome size does not match this brain shape.")
        self.pop.genomes = genomes.astype(np.float32)
        self.pop.generation = int(files[-1][4:-4])
        bpath = os.path.join(self.run_dir, "best.npz")
        if os.path.exists(bpath):
            b = np.load(bpath)
            self.best_genome = b["genome"]
            self.best_fitness = float(b["fitness"])
            self.best_generation = int(b["generation"])
        return self.pop.generation


def load_best(run_dir: str) -> Tuple[Creature, BrainSpec, SimConfig, np.ndarray, float, int]:
    """Everything needed to replay the best creature of a run."""
    meta = Trainer.load_config(run_dir)
    creature = Creature.from_dict(meta["creature"])
    spec = BrainSpec.from_dict(meta["brain"])
    cfg = SimConfig.from_dict(meta["sim"])
    data = np.load(os.path.join(run_dir, "best.npz"))
    return (creature, spec, cfg, data["genome"],
            float(data["fitness"]), int(data["generation"]))


def load_generation_best(run_dir: str, generation: int):
    """Best genome from a specific checkpointed generation."""
    meta = Trainer.load_config(run_dir)
    creature = Creature.from_dict(meta["creature"])
    spec = BrainSpec.from_dict(meta["brain"])
    cfg = SimConfig.from_dict(meta["sim"])
    path = os.path.join(run_dir, "gen", f"gen_{generation:05d}.npz")
    data = np.load(path)
    i = int(np.argmax(data["fitness"]))
    return creature, spec, cfg, data["genomes"][i], float(data["fitness"][i]), generation
