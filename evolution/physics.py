"""pymunk-backed creature simulation.

Bones are rigid bodies (capsule segments). Where two bones share a joint they
are tied together with a PivotJoint. Muscles are damped springs between two
bone centres whose rest length the brain modulates -- contracting a muscle
pulls the two bones together, which rotates them about their shared joint.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple

import numpy as np
import pymunk

from .brain import Brain, BrainSpec
from .creature import (BONE_RADIUS, MUSCLE_MAX_SCALE, MUSCLE_MIN_SCALE,
                       Creature)

CREATURE_GROUP = 1  # shared filter group => creature parts never self-collide


@dataclass
class SimConfig:
    duration: float = 10.0       # simulated seconds per evaluation
    physics_hz: int = 120        # physics steps per second
    control_hz: int = 30         # brain updates per second
    gravity: float = -9.81
    ground_friction: float = 1.0
    bone_friction: float = 0.9
    # Muscle actuation is scaled by creature mass so that big and small
    # creatures are equally (un)athletic, and so no creature can simply
    # out-force gravity and vibrate its way across the map.
    stiffness_per_kg: float = 260.0     # spring constant per kg of body mass
    damping_per_kg: float = 9.0
    force_per_kg: float = 45.0          # force clamp per kg (~4.5x body weight)
    muscle_stiffness: Optional[float] = None    # absolute override
    muscle_damping: Optional[float] = None
    max_muscle_force: Optional[float] = None
    joint_error_bias: float = 0.0       # 0 = rigid joints

    @property
    def dt(self) -> float:
        return 1.0 / self.physics_hz

    @property
    def steps_per_control(self) -> int:
        return max(1, round(self.physics_hz / self.control_hz))

    @property
    def total_steps(self) -> int:
        return int(round(self.duration * self.physics_hz))

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "SimConfig":
        known = {f: d[f] for f in SimConfig.__dataclass_fields__ if f in d}
        return SimConfig(**known)


def sensor_count(creature: Creature) -> int:
    """Brain input width for a given morphology."""
    return (len(creature.joints)          # height above ground
            + 4 * len(creature.bones)     # sin/cos angle + vx/vy
            + len(creature.bones)         # ground contact flag
            + len(creature.muscles)       # current activation (feedback)
            + 1)                          # bias


def brain_spec_for(creature: Creature, hidden: Tuple[int, ...] = (16,)) -> BrainSpec:
    return BrainSpec([sensor_count(creature), *hidden, len(creature.muscles)])


class Simulation:
    """One creature in one world. Cheap to build, so we build one per genome."""

    def __init__(self, creature: Creature, cfg: Optional[SimConfig] = None,
                 record: bool = False):
        self.creature = creature
        self.cfg = cfg or SimConfig()
        self.record = record
        self.frames: List[np.ndarray] = []

        self.space = pymunk.Space()
        self.space.gravity = (0.0, self.cfg.gravity)
        self.space.damping = 1.0
        self.space.iterations = 20

        self._build_ground()
        self._build_bones()
        self._build_joints()
        self._build_muscles()

        self.n_muscles = len(creature.muscles)
        self.activation = np.full(self.n_muscles, 0.5, dtype=np.float32)
        self._sensors = np.zeros(sensor_count(creature), dtype=np.float32)
        self._sensors[-1] = 1.0  # bias
        self.start_com_x = self.com()[0]
        self.steps_done = 0
        self.exploded = False

    # ------------------------------------------------------------- building
    def _build_ground(self) -> None:
        ground = pymunk.Segment(self.space.static_body, (-1e4, 0.0), (1e4, 0.0), 0.05)
        ground.friction = self.cfg.ground_friction
        ground.elasticity = 0.0
        self.space.add(ground)
        self.ground = ground

    def _build_bones(self) -> None:
        c = self.creature
        self.bodies: List[pymunk.Body] = []
        self.shapes: List[pymunk.Segment] = []
        self.bone_half: List[pymunk.Vec2d] = []
        for bi, bone in enumerate(c.bones):
            a = pymunk.Vec2d(*c.joints[bone.joint_a].pos)
            b = pymunk.Vec2d(*c.joints[bone.joint_b].pos)
            mid = (a + b) / 2
            half = (b - a) / 2
            mass = c.bone_mass(bi)
            local_a, local_b = -half, half
            moment = pymunk.moment_for_segment(mass, local_a, local_b, BONE_RADIUS)
            body = pymunk.Body(mass, moment)
            body.position = mid
            shape = pymunk.Segment(body, local_a, local_b, BONE_RADIUS)
            shape.friction = self.cfg.bone_friction
            shape.elasticity = 0.0
            shape.filter = pymunk.ShapeFilter(group=CREATURE_GROUP)
            self.space.add(body, shape)
            self.bodies.append(body)
            self.shapes.append(shape)
            self.bone_half.append(half)
        self.total_mass = sum(b.mass for b in self.bodies) or 1.0

    def _build_joints(self) -> None:
        c = self.creature
        self.pivots: List[pymunk.PivotJoint] = []
        # joint index -> (bone index, local offset) so we can read joint heights
        self.joint_probe: List[Tuple[int, pymunk.Vec2d]] = []
        for ji, joint in enumerate(c.joints):
            attached = c.bones_at_joint(ji)
            if not attached:
                self.joint_probe.append((0, pymunk.Vec2d(0, 0)))
                continue
            world = pymunk.Vec2d(*joint.pos)
            first = attached[0]
            self.joint_probe.append(
                (first, world - self.bodies[first].position))
            for prev, cur in zip(attached, attached[1:]):
                pj = pymunk.PivotJoint(self.bodies[prev], self.bodies[cur], world)
                pj.error_bias = pow(1.0 - 0.9999, 60.0) if self.cfg.joint_error_bias == 0 \
                    else self.cfg.joint_error_bias
                pj.max_force = 1e7
                self.space.add(pj)
                self.pivots.append(pj)

    def _build_muscles(self) -> None:
        c = self.creature
        cfg = self.cfg
        m_tot = self.total_mass
        stiffness = cfg.muscle_stiffness if cfg.muscle_stiffness is not None \
            else cfg.stiffness_per_kg * m_tot
        damping = cfg.muscle_damping if cfg.muscle_damping is not None \
            else cfg.damping_per_kg * m_tot
        max_force = cfg.max_muscle_force if cfg.max_muscle_force is not None \
            else cfg.force_per_kg * m_tot
        self.muscle_stiffness = stiffness
        self.muscle_max_force = max_force
        self.springs: List[pymunk.DampedSpring] = []
        self.rest_lengths: List[float] = []
        for mi, m in enumerate(c.muscles):
            rest = c.muscle_rest_length(mi)
            spring = pymunk.DampedSpring(
                self.bodies[m.bone_a], self.bodies[m.bone_b],
                (0, 0), (0, 0), rest,
                stiffness * m.strength,
                damping,
            )
            spring.max_force = max_force * m.strength
            self.space.add(spring)
            self.springs.append(spring)
            self.rest_lengths.append(rest)

    # -------------------------------------------------------------- sensing
    def com(self) -> Tuple[float, float]:
        mx = my = m_tot = 0.0
        for body in self.bodies:
            mx += body.position.x * body.mass
            my += body.position.y * body.mass
            m_tot += body.mass
        if m_tot == 0:
            return (0.0, 0.0)
        return (mx / m_tot, my / m_tot)

    def read_sensors(self) -> np.ndarray:
        s = self._sensors
        c = self.creature
        nj, nb, nm = len(c.joints), len(c.bones), self.n_muscles
        i = 0
        for ji in range(nj):
            bi, local = self.joint_probe[ji]
            body = self.bodies[bi]
            y = (body.position + local.rotated(body.angle)).y
            s[i] = min(max(y / 2.0, -1.0), 3.0)
            i += 1
        for bi in range(nb):
            body = self.bodies[bi]
            ang = body.angle
            s[i] = math.sin(ang); i += 1
            s[i] = math.cos(ang); i += 1
            s[i] = min(max(body.velocity.x / 8.0, -1.0), 1.0); i += 1
            s[i] = min(max(body.velocity.y / 8.0, -1.0), 1.0); i += 1
        thresh = BONE_RADIUS * 1.8
        for bi in range(nb):
            body = self.bodies[bi]
            half = self.bone_half[bi].rotated(body.angle)
            ya = (body.position + half).y
            yb = (body.position - half).y
            s[i] = 1.0 if min(ya, yb) < thresh else 0.0
            i += 1
        s[i:i + nm] = self.activation
        return s

    # ------------------------------------------------------------ actuation
    def apply_activation(self, act: np.ndarray) -> None:
        np.clip(act, 0.0, 1.0, out=act)
        self.activation = act.astype(np.float32, copy=False)
        for mi, spring in enumerate(self.springs):
            rest = self.rest_lengths[mi]
            scale = MUSCLE_MIN_SCALE + (MUSCLE_MAX_SCALE - MUSCLE_MIN_SCALE) * self.activation[mi]
            spring.rest_length = rest * scale

    # ---------------------------------------------------------------- loops
    def step(self) -> None:
        self.space.step(self.cfg.dt)
        self.steps_done += 1
        if self.record:
            self.frames.append(self.snapshot())

    def tick(self, brain: Brain) -> None:
        """One physics step, updating the brain on the control schedule."""
        if self.steps_done % self.cfg.steps_per_control == 0:
            out = brain.forward(self.read_sensors())
            self.apply_activation(np.asarray(out, dtype=np.float32).copy())
        self.step()
        if not self.sanity():
            self.exploded = True

    @property
    def done(self) -> bool:
        return self.exploded or self.steps_done >= self.cfg.total_steps

    @property
    def elapsed(self) -> float:
        return self.steps_done * self.cfg.dt

    def snapshot(self) -> np.ndarray:
        """(n_bones, 3) array of x, y, angle -- for replay / rendering."""
        return np.array([[b.position.x, b.position.y, b.angle] for b in self.bodies],
                        dtype=np.float32)

    def sanity(self) -> bool:
        for body in self.bodies:
            p = body.position
            if not (math.isfinite(p.x) and math.isfinite(p.y)) or abs(p.y) > 1e4:
                return False
        return True

    def run(self, brain: Brain) -> float:
        """Simulate to completion and return running fitness (metres travelled)."""
        cfg = self.cfg
        spc = cfg.steps_per_control
        for step in range(cfg.total_steps):
            if step % spc == 0:
                out = brain.forward(self.read_sensors())
                self.apply_activation(np.asarray(out, dtype=np.float32).copy())
            self.step()
            if step % 60 == 0 and not self.sanity():
                self.exploded = True
                return 0.0
        if not self.sanity():
            self.exploded = True
            return 0.0
        return self.fitness()

    # -------------------------------------------------------------- fitness
    def fitness(self) -> float:
        cx, cy = self.com()
        if not (math.isfinite(cx) and math.isfinite(cy)):
            return 0.0
        return float(cx - self.start_com_x)


def evaluate(creature: Creature, spec: BrainSpec, genome: np.ndarray,
             cfg: Optional[SimConfig] = None) -> float:
    sim = Simulation(creature, cfg)
    return sim.run(Brain(spec, genome))
