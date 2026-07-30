"""Pygame creature editor: draw joints, bones and muscles, then save to JSON."""
from __future__ import annotations

import copy
import math
import os
from typing import List, Optional, Tuple

import pygame

from . import render
from .creature import BONE_RADIUS, JOINT_RADIUS, Creature
from .render import Camera

MODES = ["joint", "bone", "muscle", "move", "delete"]
MODE_HELP = {
    "joint":  "click empty space to add a joint",
    "bone":   "drag from one joint to another",
    "muscle": "drag from one bone to another",
    "move":   "drag a joint to reposition it",
    "delete": "click a joint / bone / muscle to remove it",
}


def _dist_point_segment(p, a, b) -> float:
    px, py = p; ax, ay = a; bx, by = b
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy
    if L2 < 1e-12:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / L2))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


class Editor:
    def __init__(self, creature: Optional[Creature] = None,
                 out_path: str = "creature.json",
                 size: Tuple[int, int] = (1180, 720)):
        pygame.init()
        pygame.display.set_caption("Evolution - Creature Editor")
        self.screen = pygame.display.set_mode(size, pygame.RESIZABLE)
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("consolas,menlo,monospace", 15)
        self.font_big = pygame.font.SysFont("consolas,menlo,monospace", 19, bold=True)

        self.creature = creature or Creature(name="creature")
        self.out_path = out_path
        self.cam = Camera(*size, ppu=130.0, ground_frac=0.80)
        self.mode = "joint"
        self.drag_from: Optional[int] = None      # joint or bone index
        self.moving_joint: Optional[int] = None
        self.undo_stack: List[dict] = []
        self.status = f"New creature - will save to {os.path.basename(out_path)}"
        self.snap = True
        self.running = True
        self.saved = False
        self._panning = None

    # ------------------------------------------------------------ undo/redo
    def push_undo(self) -> None:
        self.undo_stack.append(copy.deepcopy(self.creature.to_dict()))
        if len(self.undo_stack) > 100:
            self.undo_stack.pop(0)

    def undo(self) -> None:
        if self.undo_stack:
            self.creature = Creature.from_dict(self.undo_stack.pop())
            self.status = "Undone"

    # --------------------------------------------------------------- picking
    def joint_at(self, mouse) -> Optional[int]:
        best, best_d = None, 1e9
        for i, j in enumerate(self.creature.joints):
            sp = self.cam.to_screen(*j.pos)
            d = math.hypot(sp[0] - mouse[0], sp[1] - mouse[1])
            if d < max(12, self.cam.px(JOINT_RADIUS) + 5) and d < best_d:
                best, best_d = i, d
        return best

    def bone_at(self, mouse) -> Optional[int]:
        best, best_d = None, 1e9
        for i, bone in enumerate(self.creature.bones):
            a = self.cam.to_screen(*self.creature.joints[bone.joint_a].pos)
            b = self.cam.to_screen(*self.creature.joints[bone.joint_b].pos)
            d = _dist_point_segment(mouse, a, b)
            if d < max(10, self.cam.px(BONE_RADIUS) + 4) and d < best_d:
                best, best_d = i, d
        return best

    def muscle_at(self, mouse) -> Optional[int]:
        best, best_d = None, 1e9
        for i, m in enumerate(self.creature.muscles):
            a = self.cam.to_screen(*self.creature.bone_midpoint(m.bone_a))
            b = self.cam.to_screen(*self.creature.bone_midpoint(m.bone_b))
            d = _dist_point_segment(mouse, a, b)
            if d < 8 and d < best_d:
                best, best_d = i, d
        return best

    def world_at(self, mouse) -> Tuple[float, float]:
        x, y = self.cam.to_world(*mouse)
        if self.snap:
            x = round(x * 10) / 10
            y = round(y * 10) / 10
        return x, max(y, 0.02)

    # ---------------------------------------------------------------- events
    def handle(self, ev) -> None:
        if ev.type == pygame.QUIT:
            self.running = False
        elif ev.type == pygame.VIDEORESIZE:
            self.screen = pygame.display.set_mode((ev.w, ev.h), pygame.RESIZABLE)
            self.cam.resize(ev.w, ev.h)
        elif ev.type == pygame.KEYDOWN:
            self.on_key(ev)
        elif ev.type == pygame.MOUSEBUTTONDOWN:
            self.on_mouse_down(ev)
        elif ev.type == pygame.MOUSEBUTTONUP:
            self.on_mouse_up(ev)
        elif ev.type == pygame.MOUSEWHEEL:
            self.cam.ppu = max(35.0, min(420.0, self.cam.ppu * (1.1 ** ev.y)))

    def on_key(self, ev) -> None:
        mods = pygame.key.get_mods()
        ctrl = mods & pygame.KMOD_CTRL or mods & pygame.KMOD_META
        if ev.key in (pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4, pygame.K_5):
            self.mode = MODES[ev.key - pygame.K_1]
            self.drag_from = None
        elif ev.key == pygame.K_TAB:
            self.mode = MODES[(MODES.index(self.mode) + 1) % len(MODES)]
        elif ctrl and ev.key == pygame.K_z:
            self.undo()
        elif ctrl and ev.key == pygame.K_s:
            self.save()
        elif ev.key == pygame.K_g:
            self.snap = not self.snap
            self.status = f"Grid snap {'on' if self.snap else 'off'}"
        elif ev.key == pygame.K_d:
            self.push_undo()
            self.creature.drop_to_ground()
            self.status = "Dropped to ground"
        elif ev.key == pygame.K_c:
            self.push_undo()
            self.creature = Creature(name=self.creature.name)
            self.status = "Cleared"
        elif ev.key == pygame.K_ESCAPE:
            if self.drag_from is not None:
                self.drag_from = None
            else:
                self.running = False
        elif ev.key in (pygame.K_LEFT, pygame.K_a):
            self.cam.cx -= 0.25
        elif ev.key in (pygame.K_RIGHT,):
            self.cam.cx += 0.25

    def on_mouse_down(self, ev) -> None:
        if ev.button == 2 or (ev.button == 1 and pygame.key.get_mods() & pygame.KMOD_SHIFT):
            self._panning = ev.pos
            return
        if ev.button != 1:
            return
        m = ev.pos
        if self.mode == "joint":
            if self.joint_at(m) is None:
                self.push_undo()
                self.creature.add_joint(*self.world_at(m))
                self.status = f"{len(self.creature.joints)} joints"
        elif self.mode == "bone":
            j = self.joint_at(m)
            if j is not None:
                self.drag_from = j
        elif self.mode == "muscle":
            b = self.bone_at(m)
            if b is not None:
                self.drag_from = b
        elif self.mode == "move":
            self.moving_joint = self.joint_at(m)
            if self.moving_joint is not None:
                self.push_undo()
        elif self.mode == "delete":
            self.push_undo()
            mi = self.muscle_at(m)
            ji = self.joint_at(m)
            bi = self.bone_at(m)
            if ji is not None:
                self.creature.delete_joint(ji); self.status = "Joint deleted"
            elif mi is not None:
                self.creature.delete_muscle(mi); self.status = "Muscle deleted"
            elif bi is not None:
                self.creature.delete_bone(bi); self.status = "Bone deleted"
            else:
                self.undo_stack.pop()

    def on_mouse_up(self, ev) -> None:
        if ev.button != 1:
            return
        m = ev.pos
        if self.mode == "bone" and self.drag_from is not None:
            target = self.joint_at(m)
            if target is None:
                # dragging into empty space creates a new joint + bone
                self.push_undo()
                target = self.creature.add_joint(*self.world_at(m))
                if self.creature.add_bone(self.drag_from, target) is None:
                    self.creature.delete_joint(target)
            else:
                self.push_undo()
                if self.creature.add_bone(self.drag_from, target) is None:
                    self.undo_stack.pop()
            self.status = f"{len(self.creature.bones)} bones"
            self.drag_from = None
        elif self.mode == "muscle" and self.drag_from is not None:
            target = self.bone_at(m)
            if target is not None:
                self.push_undo()
                if self.creature.add_muscle(self.drag_from, target) is None:
                    self.undo_stack.pop()
                    self.status = "Muscle already exists"
                else:
                    self.status = f"{len(self.creature.muscles)} muscles"
            self.drag_from = None
        elif self.mode == "move":
            self.moving_joint = None

    # ------------------------------------------------------------------ save
    def save(self) -> None:
        problems = self.creature.problems()
        self.creature.save(self.out_path)
        self.saved = True
        if problems:
            self.status = f"Saved (with warnings): {problems[0]}"
        else:
            self.status = f"Saved to {self.out_path}"

    # ------------------------------------------------------------------ draw
    def draw(self) -> None:
        cam, surf = self.cam, self.screen
        render.draw_world(surf, cam)
        mouse = pygame.mouse.get_pos()
        hj = self.joint_at(mouse) if self.mode in ("bone", "move", "delete", "joint") else None
        hb = self.bone_at(mouse) if self.mode in ("muscle", "delete") else None
        render.draw_design(surf, cam, self.creature, highlight_joint=hj,
                           highlight_bone=hb)

        # in-progress drag
        if self.drag_from is not None:
            if self.mode == "bone":
                a = cam.to_screen(*self.creature.joints[self.drag_from].pos)
            else:
                a = cam.to_screen(*self.creature.bone_midpoint(self.drag_from))
            pygame.draw.line(surf, render.ACCENT, a, mouse, 2)
        if self.moving_joint is not None:
            self.creature.joints[self.moving_joint].x, \
                self.creature.joints[self.moving_joint].y = self.world_at(mouse)

        self.draw_hud()
        pygame.display.flip()

    def draw_hud(self) -> None:
        surf = self.screen
        rect = pygame.Rect(12, 12, 330, 214)
        render.panel(surf, rect)
        surf.blit(self.font_big.render("CREATURE EDITOR", True, render.ACCENT), (26, 22))
        lines = []
        for i, m in enumerate(MODES):
            mark = ">" if m == self.mode else " "
            col = render.ACCENT if m == self.mode else render.DIM
            lines.append((f" {mark} [{i+1}] {m:<7} {MODE_HELP[m]}", col))
        lines.append("")
        lines.append((f"joints {len(self.creature.joints)}   bones {len(self.creature.bones)}"
                      f"   muscles {len(self.creature.muscles)}", render.TEXT))
        problems = self.creature.problems()
        lines.append((("ready to train" if not problems else problems[0])[:44],
                      render.DIM if not problems else (255, 140, 110)))
        render.text_block(surf, self.font, lines, 26, 50, 19)

        foot = ("ctrl+S save   ctrl+Z undo   G snap   D drop   C clear   "
                "wheel zoom   shift+drag pan   esc quit")
        surf.blit(self.font.render(foot, True, render.DIM), (16, self.cam.height - 46))
        surf.blit(self.font.render(self.status, True, render.TEXT),
                  (16, self.cam.height - 26))

    # ------------------------------------------------------------------ loop
    def run(self) -> Creature:
        while self.running:
            for ev in pygame.event.get():
                self.handle(ev)
            pressed = pygame.mouse.get_pressed()
            if self._panning is not None and (pressed[0] or pressed[1]):
                mx = pygame.mouse.get_pos()[0]
                self.cam.cx -= (mx - self._panning[0]) / self.cam.ppu
                self._panning = pygame.mouse.get_pos()
            elif not any(pressed):
                self._panning = None
            self.draw()
            self.clock.tick(60)
        pygame.quit()
        return self.creature


def run_editor(path: Optional[str] = None, out: str = "creature.json") -> Creature:
    creature = Creature.load(path) if path and os.path.exists(path) else None
    ed = Editor(creature, out_path=out)
    if path:
        ed.status = f"Loaded {os.path.basename(path)}"
    return ed.run()
