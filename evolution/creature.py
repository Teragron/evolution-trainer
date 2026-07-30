"""Creature data model: joints, bones, muscles.

Coordinate system: world units (roughly metres), y-axis points UP, ground is y = 0.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Tuple

# ---------------------------------------------------------------- constants
BONE_DENSITY = 1.0          # mass per unit length
BONE_RADIUS = 0.055         # visual + collision thickness of a bone
JOINT_RADIUS = 0.09         # visual radius of a joint marker
MUSCLE_MIN_SCALE = 0.55     # fully contracted = rest_length * this
MUSCLE_MAX_SCALE = 1.45     # fully expanded   = rest_length * this


@dataclass
class Joint:
    x: float
    y: float

    @property
    def pos(self) -> Tuple[float, float]:
        return (self.x, self.y)


@dataclass
class Bone:
    joint_a: int
    joint_b: int


@dataclass
class Muscle:
    bone_a: int
    bone_b: int
    strength: float = 1.0


@dataclass
class Creature:
    """A creature *design* (morphology only). Brains are separate genomes."""

    joints: List[Joint] = field(default_factory=list)
    bones: List[Bone] = field(default_factory=list)
    muscles: List[Muscle] = field(default_factory=list)
    name: str = "creature"

    # ------------------------------------------------------------ topology
    def bones_at_joint(self, ji: int) -> List[int]:
        return [i for i, b in enumerate(self.bones)
                if b.joint_a == ji or b.joint_b == ji]

    def bone_length(self, bi: int) -> float:
        a = self.joints[self.bones[bi].joint_a]
        b = self.joints[self.bones[bi].joint_b]
        return math.hypot(b.x - a.x, b.y - a.y)

    def bone_midpoint(self, bi: int) -> Tuple[float, float]:
        a = self.joints[self.bones[bi].joint_a]
        b = self.joints[self.bones[bi].joint_b]
        return ((a.x + b.x) / 2.0, (a.y + b.y) / 2.0)

    def bone_mass(self, bi: int) -> float:
        return max(0.05, self.bone_length(bi) * BONE_DENSITY)

    def muscle_rest_length(self, mi: int) -> float:
        m = self.muscles[mi]
        ax, ay = self.bone_midpoint(m.bone_a)
        bx, by = self.bone_midpoint(m.bone_b)
        return max(0.05, math.hypot(bx - ax, by - ay))

    # ------------------------------------------------------------ validity
    def problems(self) -> List[str]:
        out: List[str] = []
        if not self.bones:
            out.append("Creature has no bones.")
        if not self.muscles:
            out.append("Creature has no muscles - it can never move.")
        for i, b in enumerate(self.bones):
            if b.joint_a == b.joint_b:
                out.append(f"Bone {i} connects a joint to itself.")
            elif self.bone_length(i) < 1e-6:
                out.append(f"Bone {i} has zero length.")
        for i, m in enumerate(self.muscles):
            if m.bone_a == m.bone_b:
                out.append(f"Muscle {i} connects a bone to itself.")
            elif self.muscle_rest_length(i) < 1e-6:
                out.append(f"Muscle {i} has zero length.")
        if self.bones:
            adj = {i: set() for i in range(len(self.bones))}
            for ji in range(len(self.joints)):
                bs = self.bones_at_joint(ji)
                for a in bs:
                    for b in bs:
                        if a != b:
                            adj[a].add(b)
            seen, stack = {0}, [0]
            while stack:
                cur = stack.pop()
                for nb in adj[cur]:
                    if nb not in seen:
                        seen.add(nb)
                        stack.append(nb)
            if len(seen) != len(self.bones):
                out.append("Skeleton is not fully connected (creature is in pieces).")
        return out

    @property
    def is_valid(self) -> bool:
        return not self.problems()

    # ------------------------------------------------------------- geometry
    def bounds(self) -> Tuple[float, float, float, float]:
        xs = [j.x for j in self.joints] or [0.0]
        ys = [j.y for j in self.joints] or [0.0]
        return min(xs), min(ys), max(xs), max(ys)

    def drop_to_ground(self, clearance: float = 0.15) -> "Creature":
        """Translate so the lowest point sits just above y = 0, centred on x = 0."""
        if not self.joints:
            return self
        x0, y0, x1, _ = self.bounds()
        dx = -(x0 + x1) / 2.0
        dy = -y0 + clearance + BONE_RADIUS
        for j in self.joints:
            j.x += dx
            j.y += dy
        return self

    # -------------------------------------------------------- (de)serialise
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "joints": [asdict(j) for j in self.joints],
            "bones": [asdict(b) for b in self.bones],
            "muscles": [asdict(m) for m in self.muscles],
        }

    @staticmethod
    def from_dict(d: dict) -> "Creature":
        return Creature(
            joints=[Joint(**j) for j in d.get("joints", [])],
            bones=[Bone(**b) for b in d.get("bones", [])],
            muscles=[Muscle(**m) for m in d.get("muscles", [])],
            name=d.get("name", "creature"),
        )

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2)

    @staticmethod
    def load(path: str) -> "Creature":
        with open(path, "r", encoding="utf-8") as fh:
            return Creature.from_dict(json.load(fh))

    # ------------------------------------------------------------- editing
    def add_joint(self, x: float, y: float) -> int:
        self.joints.append(Joint(x, y))
        return len(self.joints) - 1

    def add_bone(self, a: int, b: int) -> Optional[int]:
        if a == b:
            return None
        for bone in self.bones:
            if {bone.joint_a, bone.joint_b} == {a, b}:
                return None
        self.bones.append(Bone(a, b))
        return len(self.bones) - 1

    def add_muscle(self, a: int, b: int) -> Optional[int]:
        if a == b:
            return None
        for m in self.muscles:
            if {m.bone_a, m.bone_b} == {a, b}:
                return None
        self.muscles.append(Muscle(a, b))
        return len(self.muscles) - 1

    def delete_joint(self, ji: int) -> None:
        for bi in sorted(self.bones_at_joint(ji), reverse=True):
            self.delete_bone(bi)
        self.joints.pop(ji)
        for b in self.bones:
            if b.joint_a > ji:
                b.joint_a -= 1
            if b.joint_b > ji:
                b.joint_b -= 1

    def delete_bone(self, bi: int) -> None:
        self.muscles = [m for m in self.muscles if bi not in (m.bone_a, m.bone_b)]
        self.bones.pop(bi)
        for m in self.muscles:
            if m.bone_a > bi:
                m.bone_a -= 1
            if m.bone_b > bi:
                m.bone_b -= 1

    def delete_muscle(self, mi: int) -> None:
        self.muscles.pop(mi)
