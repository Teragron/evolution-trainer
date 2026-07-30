#!/usr/bin/env python3
"""Evolution - command line entry point.

Local:
  python main.py editor --out my_creature.json
  python main.py train  --preset quadruped --run runs/quad --generations 300
  python main.py watch  --run runs/quad

Remote box, many cores:
  python main.py island --preset quadruped --run runs/arch --islands 8 \
      --generations 600 --population 150

Rented container with no ssh (Vast.ai etc) - relay through a HF repo:
  on the box       python main.py push --run runs/arch --repo you/evo-run --follow
  on your laptop   python main.py sync --from hf://you/evo-run --run runs/arch

Back on your laptop, while that is still running:
  python main.py sync  --from user@vm:~/evolution_game/runs/arch --run runs/arch
  python main.py watch --run runs/arch --live
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

from evolution import presets
from evolution.creature import Creature
from evolution.ga import GAConfig
from evolution.physics import SimConfig
from evolution.trainer import Trainer, available_cores


# --------------------------------------------------------------------- helpers
def resolve_creature(args) -> Creature:
    if getattr(args, "creature", None):
        return Creature.load(args.creature)
    return presets.get(args.preset)


def bar(frac: float, width: int = 22) -> str:
    n = int(max(0.0, min(1.0, frac)) * width)
    return "#" * n + "." * (width - n)


def sim_config(args) -> SimConfig:
    return SimConfig(duration=args.duration, physics_hz=args.physics_hz,
                     control_hz=args.control_hz,
                     stiffness_per_kg=args.stiffness_per_kg,
                     force_per_kg=args.force_per_kg)


def ga_config(args) -> GAConfig:
    return GAConfig(population=args.population, elite_frac=args.elite_frac,
                    mutation_rate=args.mutation_rate,
                    mutation_sigma=args.mutation_sigma,
                    crossover_rate=args.crossover_rate,
                    tournament_size=args.tournament)


def check_creature(creature: Creature) -> None:
    problems = creature.problems()
    if problems:
        print("Cannot train this creature:")
        for p in problems:
            print("  -", p)
        sys.exit(1)


# ---------------------------------------------------------------------- editor
def cmd_editor(args) -> None:
    from evolution.editor import run_editor
    src = args.load
    if args.preset and not src:
        presets.get(args.preset).save(args.out)
        src = args.out
    creature = run_editor(src, out=args.out)
    if creature.joints:
        creature.save(args.out)
        print(f"Saved design to {args.out}")
        problems = creature.problems()
        if problems:
            print("Warnings:")
            for p in problems:
                print("  -", p)
        else:
            stem = os.path.splitext(os.path.basename(args.out))[0]
            print(f"Ready to train:  python main.py train --creature {args.out} "
                  f"--run runs/{stem}")


# ----------------------------------------------------------------------- train
def cmd_train(args) -> None:
    creature = resolve_creature(args)
    check_creature(creature)
    sim_cfg, ga_cfg = sim_config(args), ga_config(args)

    trainer = Trainer(creature, args.run, hidden=args.hidden, sim_cfg=sim_cfg,
                      ga_cfg=ga_cfg, workers=args.workers, seed=args.seed)
    start_gen = trainer.load_latest_generation() if args.resume else 0

    print(f"creature   {creature.name}: {len(creature.joints)} joints, "
          f"{len(creature.bones)} bones, {len(creature.muscles)} muscles")
    print(f"brain      {trainer.spec}")
    print(f"population {ga_cfg.population}   generations {args.generations}"
          f"   workers {trainer.workers}")
    print(f"sim        {sim_cfg.duration}s @ {sim_cfg.physics_hz}Hz "
          f"(brain {sim_cfg.control_hz}Hz)")
    if start_gen:
        print(f"resumed at generation {start_gen}")
    print(f"run dir    {os.path.abspath(args.run)}\n")

    t_start = time.time()
    total = args.generations

    def report(stats, fitness, genomes):
        done = stats.generation - start_gen + 1
        eta = (time.time() - t_start) / max(done, 1) * (total - done)
        print(f"\rgen {stats.generation:5d} [{bar(done / total)}] "
              f"best {stats.best:7.2f}m  mean {stats.mean:7.2f}m  "
              f"all-time {trainer.best_fitness:7.2f}m  "
              f"{stats.evals_per_sec:6.1f} ev/s  eta {eta/60:5.1f}m",
              end="", flush=True)
        if stats.generation % 25 == 0:
            print()

    trainer.run(total, checkpoint_every=args.checkpoint_every, on_generation=report)
    print(f"\n\nbest fitness {trainer.best_fitness:.2f} m "
          f"(generation {trainer.best_generation})")
    print(f"watch it:  python main.py watch --run {args.run}")


# ---------------------------------------------------------------------- island
def cmd_island(args) -> None:
    from evolution.island import IslandConfig, aggregate, launch_local, run_island

    creature = resolve_creature(args)
    check_creature(creature)
    icfg = IslandConfig(islands=args.islands, migrate_every=args.migrate_every,
                        migrants=args.migrants, topology=args.topology,
                        seed_base=args.seed if args.seed is not None else 1000)

    # ---- one island only: used by the launcher below, and by a cluster's
    # ---- job array (--island-id $SLURM_ARRAY_TASK_ID etc.)
    if args.island_id is not None:
        workers = args.workers_per_island or 0
        run_island(creature, args.run, args.island_id, args.generations, icfg,
                   hidden=args.hidden, sim_cfg=sim_config(args),
                   ga_cfg=ga_config(args), workers=workers,
                   checkpoint_every=args.checkpoint_every, resume=args.resume)
        return

    cores = available_cores()
    per = args.workers_per_island or max(1, cores // args.islands)
    print(f"creature   {creature.name}")
    print(f"archipelago {args.islands} islands x population {args.population}"
          f"  ({args.islands * args.population} creatures per generation)")
    print(f"migration  every {icfg.migrate_every} gens, {icfg.migrants} genomes, "
          f"{icfg.topology} topology")
    print(f"hardware   {cores} usable cores -> {per} workers per island"
          f" ({per * args.islands} busy)")
    print(f"run dir    {os.path.abspath(args.run)}\n")
    if per * args.islands > cores:
        print("note: oversubscribed - islands will time-slice.\n")

    here = os.path.abspath(__file__)

    def argv_for(i: int):
        argv = [sys.executable, here, "island", "--island-id", str(i),
                "--run", args.run, "--islands", str(args.islands),
                "--generations", str(args.generations),
                "--population", str(args.population),
                "--duration", str(args.duration),
                "--physics-hz", str(args.physics_hz),
                "--control-hz", str(args.control_hz),
                "--stiffness-per-kg", str(args.stiffness_per_kg),
                "--force-per-kg", str(args.force_per_kg),
                "--elite-frac", str(args.elite_frac),
                "--mutation-rate", str(args.mutation_rate),
                "--mutation-sigma", str(args.mutation_sigma),
                "--crossover-rate", str(args.crossover_rate),
                "--tournament", str(args.tournament),
                "--migrate-every", str(args.migrate_every),
                "--migrants", str(args.migrants),
                "--topology", args.topology,
                "--checkpoint-every", str(args.checkpoint_every),
                "--workers-per-island", str(per),
                "--hidden", *[str(h) for h in args.hidden]]
        if args.creature:
            argv += ["--creature", args.creature]
        else:
            argv += ["--preset", args.preset]
        if args.seed is not None:
            argv += ["--seed", str(args.seed)]
        if args.resume:
            argv += ["--resume"]
        return argv

    code = launch_local(argv_for, args.islands, args.run,
                        aggregate_every=args.aggregate_every)
    summary = aggregate(args.run, verbose=False)
    if summary["best_fitness"] is not None:
        print(f"\nbest fitness {summary['best_fitness']:.2f} m "
              f"({summary['best_island']}, generation {summary['best_generation']})")
    print(f"watch it:  python main.py watch --run {args.run}")
    sys.exit(code)


def cmd_aggregate(args) -> None:
    from evolution.island import aggregate
    if not os.path.isdir(args.run):
        print(f"error: no such run directory: {args.run}")
        sys.exit(2)
    if args.follow:
        try:
            while True:
                try:
                    aggregate(args.run)
                except (FileNotFoundError, OSError) as exc:
                    print(f"waiting: {exc}")
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\nstopped")
        return
    try:
        aggregate(args.run)
    except (FileNotFoundError, OSError) as exc:
        print(f"error: {exc}")
        sys.exit(1)


# ------------------------------------------------------------------------ sync
def cmd_sync(args) -> None:
    from evolution.sync import follow, make_target
    try:
        remote = make_target(getattr(args, "from"), port=args.port,
                             identity=args.identity, token=args.token,
                             path_in_repo=args.path, repo_type=args.repo_type)
    except ValueError as exc:
        print(f"error: {exc}")
        sys.exit(2)
    try:
        follow(remote, args.run, interval=args.interval, full=args.full,
               once=args.once)
    except KeyboardInterrupt:
        print("\nstopped")
    except RuntimeError as exc:
        print(f"error: {exc}")
        sys.exit(1)


def cmd_push(args) -> None:
    """Upload a run to a Hugging Face repo -- the return path from a box with
    no ssh and no open ports."""
    from evolution.hf import HfRemote, parse_repo, push_loop
    if not os.path.isdir(args.run):
        print(f"error: no such run directory: {args.run}")
        sys.exit(2)
    try:
        repo = parse_repo(args.repo)
    except ValueError as exc:
        print(f"error: {exc}")
        sys.exit(2)
    remote = HfRemote(repo_id=repo, path_in_repo=args.path,
                      repo_type=args.repo_type, token=args.token)
    try:
        push_loop(remote, args.run, interval=args.interval, full=args.full,
                  once=not args.follow)
    except KeyboardInterrupt:
        print("\nstopped")
    except RuntimeError as exc:
        print(f"error: {exc}")
        sys.exit(1)


def cmd_serve(args) -> None:
    from evolution.serve import serve
    if not os.path.isdir(args.run):
        print(f"error: no such run directory: {args.run}")
        sys.exit(2)
    serve(args.run, port=args.port, host=args.host, token=args.token)


# ----------------------------------------------------------------------- watch
def cmd_watch(args) -> None:
    from evolution.viewer import watch_run
    watch_run(args.run, args.generation, live=args.live)


def cmd_preview(args) -> None:
    from evolution.viewer import watch_random
    watch_random(resolve_creature(args), hidden=args.hidden,
                 cfg=SimConfig(duration=args.duration), seed=args.seed)


# --------------------------------------------------------------------- presets
def cmd_preset(args) -> None:
    if args.list or not args.name:
        print("presets:", ", ".join(presets.PRESETS))
        return
    out = args.out or f"{args.name}.json"
    presets.get(args.name).save(out)
    print(f"wrote {out}")


# ----------------------------------------------------------------------- bench
def cmd_bench(args) -> None:
    from evolution.physics import brain_spec_for
    creature = resolve_creature(args)
    sim_cfg = SimConfig(duration=args.duration)
    spec = brain_spec_for(creature, tuple(args.hidden))
    print(f"{spec}  |  {sim_cfg.duration}s @ {sim_cfg.physics_hz}Hz  |  "
          f"{available_cores()} usable cores"
          + (f" (os.cpu_count reports {os.cpu_count()})"
             if available_cores() != os.cpu_count() else ""))
    run_dir = os.path.join(args.run or ".bench", "tmp")
    for w in args.workers_list:
        tr = Trainer(creature, run_dir, hidden=args.hidden, sim_cfg=sim_cfg,
                     ga_cfg=GAConfig(population=args.population), workers=w, seed=0)
        t = time.perf_counter()
        tr.evaluate(tr.pop.genomes)
        el = time.perf_counter() - t
        tr.close()
        label = "all" if w == 0 else str(w)
        print(f"  workers {label:>3}: {el:6.2f}s for {args.population} evals "
              f"-> {args.population/el:7.1f} evals/s"
              f"  ({args.population/el*3600/1000:.0f}k evals/hour)")


# ------------------------------------------------------------------------- CLI
def add_creature_flags(sp, default_preset="quadruped"):
    g = sp.add_mutually_exclusive_group()
    g.add_argument("--creature", help="path to a creature JSON file")
    g.add_argument("--preset", default=default_preset,
                   choices=sorted(presets.PRESETS), help="built-in design")


def add_tuning_flags(sp):
    sp.add_argument("--population", type=int, default=200)
    sp.add_argument("--hidden", type=int, nargs="*", default=[16],
                    help="hidden layer sizes, e.g. --hidden 24 16")
    sp.add_argument("--duration", type=float, default=10.0, help="sim seconds")
    sp.add_argument("--physics-hz", dest="physics_hz", type=int, default=120)
    sp.add_argument("--control-hz", dest="control_hz", type=int, default=30)
    sp.add_argument("--stiffness-per-kg", dest="stiffness_per_kg", type=float,
                    default=260.0, help="muscle spring constant per kg of body mass")
    sp.add_argument("--force-per-kg", dest="force_per_kg", type=float, default=45.0,
                    help="muscle force clamp per kg (45 = ~4.5x body weight)")
    sp.add_argument("--elite-frac", dest="elite_frac", type=float, default=0.08)
    sp.add_argument("--mutation-rate", dest="mutation_rate", type=float, default=0.06)
    sp.add_argument("--mutation-sigma", dest="mutation_sigma", type=float, default=0.35)
    sp.add_argument("--crossover-rate", dest="crossover_rate", type=float, default=0.75)
    sp.add_argument("--tournament", type=int, default=4,
                    help="tournament size (0 = rank roulette)")
    sp.add_argument("--checkpoint-every", dest="checkpoint_every", type=int, default=10)
    sp.add_argument("--resume", action="store_true")
    sp.add_argument("--seed", type=int, default=None)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="evolution", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("editor", help="draw a creature")
    e.add_argument("--load", help="open an existing creature JSON")
    e.add_argument("--preset", choices=sorted(presets.PRESETS),
                   help="start from a built-in design")
    e.add_argument("--out", default="creature.json", help="where to save")
    e.set_defaults(func=cmd_editor)

    t = sub.add_parser("train", help="evolve one population on this machine")
    add_creature_flags(t)
    add_tuning_flags(t)
    t.add_argument("--run", default="runs/run1", help="output directory")
    t.add_argument("--generations", type=int, default=300)
    t.add_argument("--workers", type=int, default=0, help="0 = all CPU cores")
    t.set_defaults(func=cmd_train)

    i = sub.add_parser("island", help="evolve many populations that trade migrants")
    add_creature_flags(i)
    add_tuning_flags(i)
    i.add_argument("--run", default="runs/arch", help="output directory")
    i.add_argument("--generations", type=int, default=400)
    i.add_argument("--islands", type=int, default=8)
    i.add_argument("--migrate-every", dest="migrate_every", type=int, default=10,
                   help="generations between migrant exchanges (0 = never)")
    i.add_argument("--migrants", type=int, default=4,
                   help="genomes offered per exchange")
    i.add_argument("--topology", choices=["full", "ring"], default="full")
    i.add_argument("--workers-per-island", dest="workers_per_island", type=int,
                   default=0, help="0 = cores / islands")
    i.add_argument("--island-id", dest="island_id", type=int, default=None,
                   help="run ONLY this island (for job arrays / manual placement)")
    i.add_argument("--aggregate-every", dest="aggregate_every", type=float,
                   default=30.0, help="seconds between roll-ups while running")
    i.set_defaults(func=cmd_island)

    a = sub.add_parser("aggregate", help="roll islands up into a viewable run")
    a.add_argument("--run", required=True)
    a.add_argument("--follow", action="store_true", help="keep aggregating")
    a.add_argument("--interval", type=float, default=30.0)
    a.set_defaults(func=cmd_aggregate)

    s = sub.add_parser("sync", help="pull a remote run directory to this machine")
    s.add_argument("--from", required=True, metavar="user@host:/remote/run",
                   help="remote run directory")
    s.add_argument("--run", required=True, help="local directory to write into")
    s.add_argument("--interval", type=float, default=15.0, help="seconds between polls")
    s.add_argument("--once", action="store_true", help="single pull, then exit")
    s.add_argument("--full", action="store_true",
                   help="also pull gen/ population checkpoints (much bigger)")
    s.add_argument("--port", type=int, default=22)
    s.add_argument("--identity", help="ssh private key path")
    s.add_argument("--token", help="HF token, or shared secret for http")
    s.add_argument("--path", default="run",
                   help="subdirectory inside a HF repo (default: run)")
    s.add_argument("--repo-type", dest="repo_type", default="model",
                   choices=["model", "dataset"], help="HF repo type")
    s.set_defaults(func=cmd_sync)

    ps = sub.add_parser("push", help="upload a run to a Hugging Face repo")
    ps.add_argument("--run", required=True)
    ps.add_argument("--repo", required=True, metavar="user/repo",
                    help="hf://user/repo or user/repo")
    ps.add_argument("--path", default="run",
                    help="subdirectory inside the repo (default: run)")
    ps.add_argument("--repo-type", dest="repo_type", default="model",
                    choices=["model", "dataset"])
    ps.add_argument("--token", help="HF write token (or set HF_TOKEN)")
    ps.add_argument("--follow", action="store_true",
                    help="keep pushing while training continues")
    ps.add_argument("--interval", type=float, default=180.0,
                    help="seconds between pushes when following")
    ps.add_argument("--full", action="store_true",
                    help="also upload gen/ checkpoints (much bigger)")
    ps.set_defaults(func=cmd_push)

    sv = sub.add_parser("serve", help="expose a run dir over http (needs a port)")
    sv.add_argument("--run", required=True)
    sv.add_argument("--port", type=int, default=8080)
    sv.add_argument("--host", default="0.0.0.0")
    sv.add_argument("--token", help="require ?t=TOKEN on every request")
    sv.set_defaults(func=cmd_serve)

    w = sub.add_parser("watch", help="replay a trained creature")
    w.add_argument("--run", required=True)
    w.add_argument("--generation", type=int, default=None,
                   help="specific checkpointed generation (default: all-time best)")
    w.add_argument("--live", action="store_true",
                   help="reload new bests as they appear (pair with sync)")
    w.set_defaults(func=cmd_watch)

    v = sub.add_parser("preview", help="see a design move with an untrained brain")
    add_creature_flags(v)
    v.add_argument("--hidden", type=int, nargs="*", default=[16])
    v.add_argument("--duration", type=float, default=20.0)
    v.add_argument("--seed", type=int, default=None)
    v.set_defaults(func=cmd_preview)

    pr = sub.add_parser("preset", help="export a built-in design to JSON")
    pr.add_argument("--name", nargs="?", choices=sorted(presets.PRESETS))
    pr.add_argument("--out")
    pr.add_argument("--list", action="store_true")
    pr.set_defaults(func=cmd_preset)

    b = sub.add_parser("bench", help="measure evaluations per second")
    add_creature_flags(b)
    b.add_argument("--population", type=int, default=200)
    b.add_argument("--duration", type=float, default=10.0)
    b.add_argument("--hidden", type=int, nargs="*", default=[16])
    b.add_argument("--workers-list", dest="workers_list", type=int, nargs="*",
                   default=[1, 0])
    b.add_argument("--run", default=None)
    b.set_defaults(func=cmd_bench)

    return p


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
