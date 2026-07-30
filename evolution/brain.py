"""Feed-forward neural network brains, stored as flat NumPy genomes.

A genome is a 1-D float32 array. The same architecture is shared by the whole
population, so a genome can be reshaped into weight matrices on demand -- which
keeps crossover and mutation trivial (they are just array ops).
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np


class BrainSpec:
    """Architecture description + genome (de)serialisation."""

    def __init__(self, layer_sizes: Sequence[int]):
        if len(layer_sizes) < 2:
            raise ValueError("Need at least an input and an output layer.")
        self.layer_sizes: Tuple[int, ...] = tuple(int(s) for s in layer_sizes)

    # ------------------------------------------------------------------ size
    @property
    def n_inputs(self) -> int:
        return self.layer_sizes[0]

    @property
    def n_outputs(self) -> int:
        return self.layer_sizes[-1]

    @property
    def genome_size(self) -> int:
        return sum(a * b + b for a, b in zip(self.layer_sizes[:-1],
                                             self.layer_sizes[1:]))

    # ------------------------------------------------------------- genomes
    def random_genome(self, rng: np.random.Generator, scale: float = 0.9) -> np.ndarray:
        return rng.normal(0.0, scale, self.genome_size).astype(np.float32)

    def unpack(self, genome: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        layers, i = [], 0
        for a, b in zip(self.layer_sizes[:-1], self.layer_sizes[1:]):
            w = genome[i:i + a * b].reshape(a, b); i += a * b
            bias = genome[i:i + b]; i += b
            layers.append((w, bias))
        return layers

    def to_dict(self) -> dict:
        return {"layer_sizes": list(self.layer_sizes)}

    @staticmethod
    def from_dict(d: dict) -> "BrainSpec":
        return BrainSpec(d["layer_sizes"])

    def __repr__(self) -> str:
        return f"BrainSpec({'-'.join(map(str, self.layer_sizes))}, {self.genome_size} weights)"


class Brain:
    """A concrete brain: architecture + weights, ready to run."""

    __slots__ = ("spec", "layers")

    def __init__(self, spec: BrainSpec, genome: np.ndarray):
        self.spec = spec
        self.layers = spec.unpack(np.asarray(genome, dtype=np.float32))

    def forward(self, x: np.ndarray) -> np.ndarray:
        """x: (n_inputs,) -> (n_outputs,) in [0, 1]."""
        h = x
        last = len(self.layers) - 1
        for i, (w, b) in enumerate(self.layers):
            h = h @ w + b
            if i == last:
                # logistic squash -> muscle activation in [0, 1]
                h = 1.0 / (1.0 + np.exp(-np.clip(h, -30.0, 30.0)))
            else:
                h = np.tanh(h)
        return h
