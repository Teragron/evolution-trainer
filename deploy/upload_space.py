#!/usr/bin/env python3
"""Deploy the trainer as a Hugging Face Space (works on the free CPU tier).

The Space's container trains in the background, pushes checkpoints to a
separate repo every few minutes, and serves the run directory in the
foreground -- which doubles as the Space's health check.

    pip install huggingface_hub
    python deploy/upload_space.py --space YOU/evolution-trainer \\
        --push-repo YOU/evo-run --hf-token hf_xxx

--hf-token needs write access. It is stored as a *Space secret* (HF_TOKEN),
never committed to the repo -- that's what lets the container push
checkpoints back out on its own.
"""
from __future__ import annotations

import argparse
import os
import sys


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--space", required=True, metavar="user/space-name")
    ap.add_argument("--push-repo", required=True, metavar="user/repo",
                    help="repo the container pushes checkpoints to (created if missing)")
    ap.add_argument("--push-repo-type", default="dataset", choices=["model", "dataset"])
    ap.add_argument("--hf-token", help="write token, stored as the Space secret "
                    "HF_TOKEN (or set HF_TOKEN in your shell)")
    ap.add_argument("--preset", default="quadruped")
    ap.add_argument("--population", type=int, default=200)
    ap.add_argument("--generations", type=int, default=5000)
    ap.add_argument("--checkpoint-every", dest="checkpoint_every", type=int, default=20)
    ap.add_argument("--push-interval", dest="push_interval", type=int, default=180)
    ap.add_argument("--public", action="store_true",
                    help="make the Space and push-repo public (default: private)")
    args = ap.parse_args()

    token = args.hf_token or os.environ.get("HF_TOKEN")
    if not token:
        print("error: pass --hf-token or set HF_TOKEN")
        sys.exit(1)

    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("error: pip install huggingface_hub")
        sys.exit(1)

    api = HfApi(token=token)
    try:
        me = api.whoami()
        print(f"logged in as: {me.get('name', '?')}")
    except Exception:                                    # noqa: BLE001
        print("error: token rejected. Check --hf-token / HF_TOKEN")
        sys.exit(1)

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    space = args.space.strip("/")
    push_repo = args.push_repo.strip("/")
    private = not args.public

    print(f"creating space   https://huggingface.co/spaces/{space}")
    api.create_repo(space, repo_type="space", space_sdk="docker",
                    exist_ok=True, private=private)
    print(f"creating repo    https://huggingface.co/datasets/{push_repo}"
          if args.push_repo_type == "dataset" else
          f"creating repo    https://huggingface.co/{push_repo}")
    api.create_repo(push_repo, repo_type=args.push_repo_type, exist_ok=True,
                    private=True)

    print("uploading trainer code ...")
    api.upload_folder(folder_path=root, repo_id=space, repo_type="space",
                      allow_patterns=["main.py", "requirements.txt", "evolution/*.py"],
                      commit_message="trainer code")
    print("uploading space app (Dockerfile, entrypoint, card) ...")
    api.upload_folder(folder_path=os.path.join(root, "deploy", "space_app"),
                      repo_id=space, repo_type="space",
                      commit_message="space app")

    print("setting secret + config variables ...")
    api.add_space_secret(space, "HF_TOKEN", token)
    for key, value in {
        "PUSH_REPO": push_repo,
        "PUSH_REPO_TYPE": args.push_repo_type,
        "PRESET": args.preset,
        "POPULATION": str(args.population),
        "GENERATIONS": str(args.generations),
        "CHECKPOINT_EVERY": str(args.checkpoint_every),
        "PUSH_INTERVAL": str(args.push_interval),
    }.items():
        api.add_space_variable(space, key, value)

    print(f"\ndone: https://huggingface.co/spaces/{space}")
    print("It builds and starts automatically -- watch the Logs tab there.")
    print(f"\nOnce it's training, from your laptop:")
    print(f"  python main.py sync --from hf://{push_repo} --run runs/arch --interval 60")
    print(f"  python main.py watch --run runs/arch --live")


if __name__ == "__main__":
    main()
