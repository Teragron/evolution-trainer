"""A few ready-made creature designs so you can train without opening the editor."""
from __future__ import annotations

from .creature import Bone, Creature, Joint, Muscle


def quadruped() -> Creature:
    c = Creature(
        joints=[Joint(-0.75, 1.05), Joint(0.75, 1.05),
                Joint(-0.80, 0.55), Joint(-0.72, 0.06),
                Joint(0.80, 0.55), Joint(0.72, 0.06)],
        bones=[Bone(0, 1),          # spine
               Bone(0, 2), Bone(2, 3),   # rear leg
               Bone(1, 4), Bone(4, 5)],  # front leg
        muscles=[Muscle(0, 1), Muscle(1, 2),
                 Muscle(0, 3), Muscle(3, 4),
                 Muscle(0, 2), Muscle(0, 4)],
        name="quadruped",
    )
    return c.drop_to_ground()


def biped() -> Creature:
    c = Creature(
        joints=[Joint(0.0, 1.5),    # head / torso top
                Joint(0.0, 0.95),   # hip
                Joint(-0.28, 0.5), Joint(-0.22, 0.06),   # left leg
                Joint(0.28, 0.5), Joint(0.22, 0.06)],    # right leg
        bones=[Bone(0, 1),
               Bone(1, 2), Bone(2, 3),
               Bone(1, 4), Bone(4, 5)],
        muscles=[Muscle(0, 1), Muscle(1, 2),
                 Muscle(0, 3), Muscle(3, 4),
                 Muscle(0, 2), Muscle(0, 4), Muscle(1, 3)],
        name="biped",
    )
    return c.drop_to_ground()


def worm() -> Creature:
    # deliberately zig-zagged: a perfectly straight chain is degenerate, its
    # muscles can only pull along one line and it can never undulate.
    joints = [Joint(-1.1 + 0.55 * i, 0.20 if i % 2 == 0 else 0.50)
              for i in range(5)]
    bones = [Bone(i, i + 1) for i in range(4)]
    muscles = [Muscle(0, 1), Muscle(1, 2), Muscle(2, 3),
               Muscle(0, 2), Muscle(1, 3), Muscle(0, 3)]
    return Creature(joints=joints, bones=bones, muscles=muscles,
                    name="worm").drop_to_ground()


def tripod() -> Creature:
    c = Creature(
        joints=[Joint(0.0, 1.2),
                Joint(-0.7, 0.06), Joint(0.7, 0.06), Joint(0.0, 0.55),
                Joint(0.0, 0.06)],
        bones=[Bone(0, 3), Bone(3, 4),
               Bone(0, 1), Bone(0, 2)],
        muscles=[Muscle(0, 1), Muscle(0, 2), Muscle(0, 3),
                 Muscle(1, 2), Muscle(1, 3), Muscle(2, 3)],
        name="tripod",
    )
    return c.drop_to_ground()


PRESETS = {
    "quadruped": quadruped,
    "biped": biped,
    "worm": worm,
    "tripod": tripod,
}


def get(name: str) -> Creature:
    if name not in PRESETS:
        raise KeyError(f"Unknown preset '{name}'. Options: {', '.join(PRESETS)}")
    return PRESETS[name]()
