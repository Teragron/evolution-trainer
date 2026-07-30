"""A Python re-implementation of Keiwan's Evolution, built for fast CPU training."""

from .creature import Bone, Creature, Joint, Muscle
from .brain import Brain, BrainSpec
from .physics import SimConfig, Simulation, brain_spec_for, evaluate
from .ga import GAConfig, Population
from .trainer import Trainer, load_best, load_generation_best

__all__ = [
    "Bone", "Creature", "Joint", "Muscle",
    "Brain", "BrainSpec",
    "SimConfig", "Simulation", "brain_spec_for", "evaluate",
    "GAConfig", "Population",
    "Trainer", "load_best", "load_generation_best",
]
__version__ = "1.0.0"
