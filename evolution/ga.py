"""Genetic algorithm over flat weight genomes."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional, Tuple

import numpy as np


@dataclass
class GAConfig:
    population: int = 200
    elite_frac: float = 0.08        # fraction copied unchanged into next gen
    tournament_size: int = 4        # 0 => rank-proportional roulette instead
    crossover_rate: float = 0.75
    mutation_rate: float = 0.06     # per-gene probability
    mutation_sigma: float = 0.35    # gaussian step size
    reset_rate: float = 0.005       # per-gene chance of full re-randomisation
    init_scale: float = 0.9
    weight_clip: float = 6.0

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "GAConfig":
        known = {f: d[f] for f in GAConfig.__dataclass_fields__ if f in d}
        return GAConfig(**known)


class Population:
    def __init__(self, genome_size: int, cfg: Optional[GAConfig] = None,
                 seed: Optional[int] = None,
                 genomes: Optional[np.ndarray] = None):
        self.cfg = cfg or GAConfig()
        self.genome_size = genome_size
        self.rng = np.random.default_rng(seed)
        if genomes is not None:
            self.genomes = np.asarray(genomes, dtype=np.float32)
            self.cfg.population = self.genomes.shape[0]
        else:
            self.genomes = self.rng.normal(
                0.0, self.cfg.init_scale,
                (self.cfg.population, genome_size)).astype(np.float32)
        self.generation = 0

    # ------------------------------------------------------------ selection
    def _tournament(self, fitness: np.ndarray) -> int:
        k = self.cfg.tournament_size
        idx = self.rng.integers(0, len(fitness), k)
        return int(idx[np.argmax(fitness[idx])])

    def _roulette_table(self, fitness: np.ndarray) -> np.ndarray:
        # rank-based: worst gets weight 1, best gets weight N. Robust to
        # negative fitness and to a few runaway outliers.
        order = np.argsort(fitness)
        weights = np.empty(len(fitness), dtype=np.float64)
        weights[order] = np.arange(1, len(fitness) + 1, dtype=np.float64)
        return weights / weights.sum()

    def _pick(self, fitness: np.ndarray, table: np.ndarray) -> int:
        if self.cfg.tournament_size >= 2:
            return self._tournament(fitness)
        return int(self.rng.choice(len(fitness), p=table))

    # ------------------------------------------------------------ operators
    def _crossover(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        if self.rng.random() > self.cfg.crossover_rate:
            return a.copy()
        cut = int(self.rng.integers(1, self.genome_size))
        child = a.copy()
        child[cut:] = b[cut:]
        return child

    def _mutate(self, g: np.ndarray) -> np.ndarray:
        cfg = self.cfg
        mask = self.rng.random(self.genome_size) < cfg.mutation_rate
        n = int(mask.sum())
        if n:
            g[mask] += self.rng.normal(0.0, cfg.mutation_sigma, n).astype(np.float32)
        if cfg.reset_rate > 0:
            rmask = self.rng.random(self.genome_size) < cfg.reset_rate
            n = int(rmask.sum())
            if n:
                g[rmask] = self.rng.normal(0.0, cfg.init_scale, n).astype(np.float32)
        np.clip(g, -cfg.weight_clip, cfg.weight_clip, out=g)
        return g

    # ----------------------------------------------------------- next round
    def evolve(self, fitness: np.ndarray) -> Tuple[int, float]:
        """Replace the population using the supplied fitness scores.

        Returns (index of best individual in the OLD population, its fitness).
        """
        fitness = np.asarray(fitness, dtype=np.float64)
        n = len(self.genomes)
        best = int(np.argmax(fitness))
        n_elite = max(1, int(round(self.cfg.elite_frac * n)))
        elite_idx = np.argsort(fitness)[::-1][:n_elite]

        table = self._roulette_table(fitness)
        children = np.empty_like(self.genomes)
        children[:n_elite] = self.genomes[elite_idx]
        for i in range(n_elite, n):
            pa = self.genomes[self._pick(fitness, table)]
            pb = self.genomes[self._pick(fitness, table)]
            children[i] = self._mutate(self._crossover(pa, pb))
        self.genomes = children
        self.generation += 1
        return best, float(fitness[best])
