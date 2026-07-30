"""Watch a trained creature run, with generation stepping and a fitness graph."""
from __future__ import annotations

import csv
import os
import time
from typing import List, Optional, Tuple

import numpy as np
import pygame

from . import render
from .brain import Brain, BrainSpec
from .creature import Creature
from .physics import SimConfig, Simulation
from .render import Camera
from .trainer import (Trainer, load_best, load_generation_best,
                      version_warnings)


def _checkpoint_gens(run_dir: str) -> List[int]:
    gdir = os.path.join(run_dir, "gen")
    if not os.path.isdir(gdir):
        return []
    out = []
    for f in sorted(os.listdir(gdir)):
        if f.startswith("gen_") and f.endswith(".npz"):
            try:
                out.append(int(f[4:-4]))
            except ValueError:
                pass
    return out


def _history(run_dir: str) -> List[Tuple[int, float, float]]:
    path = os.path.join(run_dir, "history.csv")
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            rows.append((int(r["generation"]), float(r["best"]), float(r["mean"])))
    return rows


class Viewer:
    def __init__(self, creature: Creature, spec: BrainSpec, cfg: SimConfig,
                 genome: np.ndarray, label: str = "",
                 run_dir: Optional[str] = None,
                 size: Tuple[int, int] = (1280, 720),
                 live: bool = False,
                 warnings: Optional[List[str]] = None):
        pygame.init()
        pygame.display.set_caption("Evolution - Viewer")
        self.screen = pygame.display.set_mode(size, pygame.RESIZABLE)
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("consolas,menlo,monospace", 15)
        self.font_big = pygame.font.SysFont("consolas,menlo,monospace", 19, bold=True)

        self.creature = creature
        self.spec = spec
        self.cfg = cfg
        self.genome = genome
        self.label = label
        self.run_dir = run_dir
        self.gens = _checkpoint_gens(run_dir) if run_dir else []
        self.gen_cursor = len(self.gens) - 1
        self.history = _history(run_dir) if run_dir else []

        self.cam = Camera(*size, ppu=115.0, ground_frac=0.80)
        self.speed = 1.0
        self.paused = False
        self.show_muscles = True
        self.follow = True
        self.trail: List[Tuple[float, float]] = []
        self.running = True

        # live-follow: re-read the run directory while it is still being
        # written (locally, or by a sync loop pulling from a remote trainer)
        self.live = bool(live and run_dir)
        self.loop = self.live
        self.warnings = list(warnings or [])
        self._poll_interval = 2.0
        self._last_poll = 0.0
        self._signature = self._run_signature()
        self._last_update: Optional[float] = None
        self._pending_note = ""
        self.reset()

    # ---------------------------------------------------------------- state
    def reset(self) -> None:
        if not hasattr(self, "_best_seen"):
            self._best_seen = -float("inf")
        self.sim = Simulation(self.creature, self.cfg)
        self.brain = Brain(self.spec, self.genome)
        self.trail.clear()
        self.cam.cx = 0.0

    def load_gen(self, delta: int) -> None:
        if not self.gens:
            return
        self.gen_cursor = max(0, min(len(self.gens) - 1, self.gen_cursor + delta))
        gen = self.gens[self.gen_cursor]
        _, _, _, genome, fit, _ = load_generation_best(self.run_dir, gen)
        self.genome = genome
        self.label = f"generation {gen} best (fitness {fit:.2f} m)"
        self.reset()

    def load_overall_best(self) -> None:
        _, _, _, genome, fit, gen = load_best(self.run_dir)
        self.genome = genome
        self.label = f"all-time best from gen {gen} (fitness {fit:.2f} m)"
        self.reset()

    # ----------------------------------------------------------------- live
    def _run_signature(self):
        """Cheap fingerprint of the run dir: catches new bests and new history."""
        if not self.run_dir:
            return None
        sig = []
        for name in ("best.npz", "history.csv"):
            p = os.path.join(self.run_dir, name)
            try:
                st = os.stat(p)
                sig.append((name, st.st_mtime_ns, st.st_size))
            except OSError:
                sig.append((name, 0, 0))
        return tuple(sig)

    def poll_run(self, force: bool = False) -> None:
        """Reload history / best genome if the run directory changed."""
        if not self.run_dir:
            return
        now = time.time()
        if not force and now - self._last_poll < self._poll_interval:
            return
        self._last_poll = now
        sig = self._run_signature()
        if sig == self._signature and not force:
            return
        self._signature = sig
        self.history = _history(self.run_dir)
        gens = _checkpoint_gens(self.run_dir)
        if gens != self.gens:
            at_end = self.gen_cursor >= len(self.gens) - 1
            self.gens = gens
            self.gen_cursor = len(gens) - 1 if at_end else min(self.gen_cursor,
                                                              len(gens) - 1)
        try:
            _, _, _, genome, fit, gen = load_best(self.run_dir)
        except (OSError, ValueError, KeyError, FileNotFoundError):
            return
        if fit > self._best_seen + 1e-9:
            self._best_seen = fit
            self.genome = genome
            self.label = f"all-time best from gen {gen} (fitness {fit:.2f} m)"
            self._pending_note = f"new best {fit:.2f} m from gen {gen}"
            self._last_update = now
            self.reset()

    # --------------------------------------------------------------- events
    def handle(self, ev) -> None:
        if ev.type == pygame.QUIT:
            self.running = False
        elif ev.type == pygame.VIDEORESIZE:
            self.screen = pygame.display.set_mode((ev.w, ev.h), pygame.RESIZABLE)
            self.cam.resize(ev.w, ev.h)
        elif ev.type == pygame.MOUSEWHEEL:
            self.cam.ppu = max(20.0, min(400.0, self.cam.ppu * (1.1 ** ev.y)))
        elif ev.type == pygame.KEYDOWN:
            if ev.key in (pygame.K_ESCAPE, pygame.K_q):
                self.running = False
            elif ev.key == pygame.K_SPACE:
                self.paused = not self.paused
            elif ev.key == pygame.K_r:
                self.reset()
            elif ev.key == pygame.K_m:
                self.show_muscles = not self.show_muscles
            elif ev.key == pygame.K_f:
                self.follow = not self.follow
            elif ev.key in (pygame.K_PLUS, pygame.K_EQUALS):
                self.speed = min(16.0, self.speed * 2)
            elif ev.key == pygame.K_MINUS:
                self.speed = max(0.125, self.speed / 2)
            elif ev.key == pygame.K_RIGHT:
                self.load_gen(+1)
            elif ev.key == pygame.K_LEFT:
                self.load_gen(-1)
            elif ev.key == pygame.K_b and self.run_dir:
                self.load_overall_best()
            elif ev.key == pygame.K_l and self.run_dir:
                self.live = not self.live
                if self.live:
                    self.poll_run(force=True)
            elif ev.key == pygame.K_o:
                self.loop = not self.loop

    # ----------------------------------------------------------------- loop
    def update(self) -> None:
        if self.live:
            self.poll_run()
        if self.paused:
            return
        if self.sim.done:
            if self.loop:
                self.reset()
            return
        steps = max(1, int(round(self.cfg.physics_hz / 60.0 * self.speed)))
        for _ in range(steps):
            if self.sim.done:
                break
            self.sim.tick(self.brain)
        com = self.sim.com()
        self.trail.append(com)
        if len(self.trail) > 4000:
            self.trail.pop(0)
        if self.follow:
            self.cam.cx += (com[0] - self.cam.cx) * 0.12

    def draw(self) -> None:
        surf, cam = self.screen, self.cam
        render.draw_world(surf, cam)
        if len(self.trail) > 2:
            pts = [cam.to_screen(x, y) for x, y in self.trail[::2]]
            pygame.draw.lines(surf, (60, 90, 130), False, pts, 2)
        render.draw_sim(surf, cam, self.sim, self.show_muscles)
        self.draw_hud()
        pygame.display.flip()

    def draw_hud(self) -> None:
        surf = self.screen
        sim = self.sim
        dist = sim.com()[0] - sim.start_com_x
        extra = 19 * (1 + len(self.warnings)) if (self.live or self.warnings) else 0
        rect = pygame.Rect(12, 12, 360, 150 + extra)
        render.panel(surf, rect)
        surf.blit(self.font_big.render("VIEWER", True, render.ACCENT), (26, 22))
        if self.live:
            dot = (120, 230, 140) if (time.time() % 2) < 1.4 else (60, 110, 70)
            pygame.draw.circle(surf, dot, (rect.right - 62, 30), 5)
            surf.blit(self.font.render("LIVE", True, (120, 230, 140)),
                      (rect.right - 50, 23))
        lines = [
            (self.label[:42], render.TEXT),
            "",
            (f"distance   {dist:+7.2f} m", render.TEXT),
            (f"time       {sim.elapsed:5.2f} / {self.cfg.duration:.1f} s", render.DIM),
            (f"speed      x{self.speed:g}"
             + ("   [PAUSED]" if self.paused else "")
             + ("   [DONE]" if sim.done else ""), render.DIM),
        ]
        if self.live:
            if self._pending_note:
                age = time.time() - (self._last_update or 0)
                lines.append((self._pending_note if age < 12 else
                              "watching for new generations...", (120, 230, 140)))
            else:
                lines.append(("watching for new generations...", render.DIM))
        for w in self.warnings:
            lines.append((w[:44], (255, 190, 120)))
        render.text_block(surf, self.font, lines, 26, 50, 19)

        if self.history:
            self.draw_graph(pygame.Rect(self.cam.width - 372, 12, 360, 150))

        keys = ("space pause   R restart   +/- speed   F follow   M muscles   "
                + ("left/right generation   B best   " if self.gens else "")
                + ("L live   O loop   " if self.run_dir else "")
                + "esc quit")
        surf.blit(self.font.render(keys, True, render.DIM), (16, self.cam.height - 26))

    def draw_graph(self, rect: pygame.Rect) -> None:
        surf = self.screen
        render.panel(surf, rect)
        surf.blit(self.font.render("fitness by generation", True, render.DIM),
                  (rect.x + 10, rect.y + 6))
        gens = [h[0] for h in self.history]
        best = [h[1] for h in self.history]
        mean = [h[2] for h in self.history]
        lo = min(min(best), min(mean))
        hi = max(max(best), max(mean))
        if hi - lo < 1e-6:
            hi = lo + 1.0
        inner = rect.inflate(-24, -46).move(0, 10)

        def pt(i, v):
            x = inner.x + (inner.w * i / max(1, len(gens) - 1))
            y = inner.bottom - inner.h * (v - lo) / (hi - lo)
            return (x, y)

        if lo < 0 < hi:
            zy = pt(0, 0.0)[1]
            pygame.draw.line(surf, (52, 58, 72), (inner.x, zy), (inner.right, zy), 1)
        if len(gens) > 1:
            pygame.draw.lines(surf, (90, 110, 150), False,
                              [pt(i, v) for i, v in enumerate(mean)], 1)
            pygame.draw.lines(surf, render.ACCENT, False,
                              [pt(i, v) for i, v in enumerate(best)], 2)
        surf.blit(self.font.render(f"{hi:.1f} m", True, render.DIM),
                  (rect.x + 10, rect.y + 24))
        surf.blit(self.font.render(f"gen {gens[-1]}", True, render.DIM),
                  (rect.right - 78, rect.bottom - 22))

    def run(self) -> None:
        while self.running:
            for ev in pygame.event.get():
                self.handle(ev)
            self.update()
            self.draw()
            self.clock.tick(60)
        pygame.quit()


def watch_run(run_dir: str, generation: Optional[int] = None,
              live: bool = False) -> None:
    warns = []
    try:
        warns = version_warnings(Trainer.load_config(run_dir))
    except (OSError, ValueError):
        pass
    for w in warns:
        print("warning:", w)
    if generation is None:
        creature, spec, cfg, genome, fit, gen = load_best(run_dir)
        label = f"all-time best from gen {gen} (fitness {fit:.2f} m)"
    else:
        creature, spec, cfg, genome, fit, gen = load_generation_best(run_dir, generation)
        label = f"generation {gen} best (fitness {fit:.2f} m)"
    v = Viewer(creature, spec, cfg, genome, label=label, run_dir=run_dir,
               live=live, warnings=warns)
    v._best_seen = fit
    if generation is None and v.gens:
        v.gen_cursor = len(v.gens) - 1
    v.run()


def watch_random(creature: Creature, hidden=(16,), cfg: Optional[SimConfig] = None,
                 seed: Optional[int] = None) -> None:
    """Sanity-check a morphology with an untrained brain."""
    from .physics import brain_spec_for
    spec = brain_spec_for(creature, tuple(hidden))
    rng = np.random.default_rng(seed)
    v = Viewer(creature, spec, cfg or SimConfig(), spec.random_genome(rng),
               label="untrained random brain")
    v.run()
