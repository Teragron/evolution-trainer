"""Shared pygame drawing helpers (world units -> screen pixels)."""
from __future__ import annotations

import math
from typing import Iterable, Optional, Sequence, Tuple

import pygame

from .creature import BONE_RADIUS, JOINT_RADIUS, Creature

BG = (18, 20, 26)
GRID = (30, 34, 44)
GROUND = (58, 64, 80)
GROUND_FILL = (26, 29, 37)
BONE_COL = (226, 230, 240)
BONE_EDGE = (140, 148, 165)
JOINT_COL = (108, 178, 255)
MUSCLE_CONTRACT = (255, 92, 92)
MUSCLE_EXPAND = (92, 168, 255)
MUSCLE_IDLE = (150, 150, 160)
TEXT = (216, 220, 230)
DIM = (130, 136, 150)
ACCENT = (255, 196, 84)


class Camera:
    """Maps world coordinates (y up, ground at 0) to screen pixels."""

    def __init__(self, width: int, height: int, ppu: float = 110.0,
                 ground_frac: float = 0.78):
        self.width = width
        self.height = height
        self.ppu = ppu
        self.ground_frac = ground_frac
        self.cx = 0.0
        self.cy = 0.9

    def to_screen(self, x: float, y: float) -> Tuple[int, int]:
        sx = (x - self.cx) * self.ppu + self.width * 0.5
        sy = self.height * self.ground_frac - (y - self.cy_offset()) * self.ppu
        return int(sx), int(sy)

    def cy_offset(self) -> float:
        return 0.0  # ground stays pinned at ground_frac; vertical pan unused

    def to_world(self, sx: float, sy: float) -> Tuple[float, float]:
        x = (sx - self.width * 0.5) / self.ppu + self.cx
        y = (self.height * self.ground_frac - sy) / self.ppu
        return x, y

    def px(self, world_len: float) -> int:
        return max(1, int(world_len * self.ppu))

    def resize(self, width: int, height: int) -> None:
        self.width, self.height = width, height


# ------------------------------------------------------------------ world art
def draw_world(surf: pygame.Surface, cam: Camera, show_grid: bool = True) -> None:
    surf.fill(BG)
    gy = cam.to_screen(0, 0)[1]
    if show_grid:
        step = 1.0
        x0 = math.floor(cam.to_world(0, 0)[0] / step) * step
        x = x0
        while True:
            sx = cam.to_screen(x, 0)[0]
            if sx > cam.width + 2:
                break
            if sx >= -2:
                pygame.draw.line(surf, GRID, (sx, 0), (sx, gy), 1)
                if abs(x) % 5 < 1e-6:
                    pygame.draw.line(surf, GROUND, (sx, gy - 14), (sx, gy), 2)
            x += step
        y = 1.0
        while True:
            sy = cam.to_screen(0, y)[1]
            if sy < -2:
                break
            pygame.draw.line(surf, GRID, (0, sy), (cam.width, sy), 1)
            y += 1.0
    pygame.draw.rect(surf, GROUND_FILL, pygame.Rect(0, gy, cam.width, cam.height - gy))
    pygame.draw.line(surf, GROUND, (0, gy), (cam.width, gy), 3)


def _capsule(surf: pygame.Surface, a: Tuple[int, int], b: Tuple[int, int],
             r: int, colour, edge=None) -> None:
    if edge:
        pygame.draw.line(surf, edge, a, b, r * 2 + 2)
        pygame.draw.circle(surf, edge, a, r + 1)
        pygame.draw.circle(surf, edge, b, r + 1)
    pygame.draw.line(surf, colour, a, b, r * 2)
    pygame.draw.circle(surf, colour, a, r)
    pygame.draw.circle(surf, colour, b, r)


def draw_design(surf: pygame.Surface, cam: Camera, creature: Creature,
                highlight_joint: Optional[int] = None,
                highlight_bone: Optional[int] = None,
                show_muscles: bool = True) -> None:
    """Draw a static creature design (editor view)."""
    r = cam.px(BONE_RADIUS)
    if show_muscles:
        for mi in range(len(creature.muscles)):
            m = creature.muscles[mi]
            a = cam.to_screen(*creature.bone_midpoint(m.bone_a))
            b = cam.to_screen(*creature.bone_midpoint(m.bone_b))
            _dashed(surf, MUSCLE_IDLE, a, b, 3)
    for bi, bone in enumerate(creature.bones):
        a = cam.to_screen(*creature.joints[bone.joint_a].pos)
        b = cam.to_screen(*creature.joints[bone.joint_b].pos)
        col = ACCENT if bi == highlight_bone else BONE_COL
        _capsule(surf, a, b, r, col, BONE_EDGE)
    jr = cam.px(JOINT_RADIUS)
    for ji, j in enumerate(creature.joints):
        p = cam.to_screen(*j.pos)
        pygame.draw.circle(surf, (20, 22, 28), p, jr + 2)
        pygame.draw.circle(surf, ACCENT if ji == highlight_joint else JOINT_COL, p, jr)


def draw_sim(surf: pygame.Surface, cam: Camera, sim, show_muscles: bool = True) -> None:
    """Draw a live Simulation."""
    import pymunk
    r = cam.px(BONE_RADIUS)
    if show_muscles:
        for mi, spring in enumerate(sim.springs):
            a = cam.to_screen(*spring.a.position)
            b = cam.to_screen(*spring.b.position)
            act = float(sim.activation[mi])
            col = _lerp_col(MUSCLE_CONTRACT, MUSCLE_EXPAND, act)
            _dashed(surf, col, a, b, 3)
    for bi, body in enumerate(sim.bodies):
        half = sim.bone_half[bi].rotated(body.angle)
        a = cam.to_screen(*(body.position + half))
        b = cam.to_screen(*(body.position - half))
        _capsule(surf, a, b, r, BONE_COL, BONE_EDGE)
    jr = cam.px(JOINT_RADIUS * 0.85)
    for ji in range(len(sim.creature.joints)):
        bi, local = sim.joint_probe[ji]
        body = sim.bodies[bi]
        p = cam.to_screen(*(body.position + local.rotated(body.angle)))
        pygame.draw.circle(surf, (20, 22, 28), p, jr + 2)
        pygame.draw.circle(surf, JOINT_COL, p, jr)


def _lerp_col(c0, c1, t: float):
    t = max(0.0, min(1.0, t))
    return tuple(int(a + (b - a) * t) for a, b in zip(c0, c1))


def _dashed(surf, colour, a, b, width: int, dash: int = 8, gap: int = 6) -> None:
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    length = math.hypot(dx, dy)
    if length < 1e-6:
        return
    ux, uy = dx / length, dy / length
    pos = 0.0
    while pos < length:
        end = min(pos + dash, length)
        pygame.draw.line(surf, colour,
                         (ax + ux * pos, ay + uy * pos),
                         (ax + ux * end, ay + uy * end), width)
        pos = end + gap


# ---------------------------------------------------------------------- HUD
def text_block(surf: pygame.Surface, font: pygame.font.Font,
               lines: Iterable[Sequence], x: int, y: int,
               line_h: int = 20) -> int:
    for line in lines:
        if isinstance(line, str):
            txt, col = line, TEXT
        else:
            txt, col = line
        surf.blit(font.render(txt, True, col), (x, y))
        y += line_h
    return y


def panel(surf: pygame.Surface, rect: pygame.Rect, alpha: int = 190) -> None:
    s = pygame.Surface(rect.size, pygame.SRCALPHA)
    s.fill((12, 14, 20, alpha))
    surf.blit(s, rect.topleft)
    pygame.draw.rect(surf, (46, 52, 66), rect, 1)
