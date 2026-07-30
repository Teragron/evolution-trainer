#!/usr/bin/env python3
"""Push the trainer to a Hugging Face repo, using YOUR existing hf login.

Run this on your own machine, from the project root:

    pip install huggingface_hub
    hf auth login                       # or: huggingface-cli login
    python deploy/upload_to_hf.py --repo YOU/evo-run

No token is ever typed into a chat or a script -- the hf CLI stores it in your
own keyring and this reads it from there. Add --public if you want the repo
world-readable (the default is private).
"""
from __future__ import annotations

import argparse
import fnmatch
import os
import sys

# Exactly what the training box needs, and nothing else.
INCLUDE = [
    "main.py",
    "requirements.txt",
    "evolution/*.py",
    "deploy/vastai_setup.sh",
    "deploy/vastai_launch.sh",
    "VASTAI.md",
    "README.md",
    "*.json",              # your creature designs, if any live at the root
]
EXCLUDE = [
    "**/__pycache__/**", "*.pyc",
    "runs/**",             # old results - don't ship them to the box
    "requirements-gui.txt",
    "*.png",
]


def matches(rel: str, patterns) -> bool:
    return any(fnmatch.fnmatch(rel, p) or fnmatch.fnmatch("/" + rel, "*/" + p.lstrip("*/"))
               for p in patterns)


def collect(root: str):
    picked = []
    for pattern in INCLUDE:
        if "/" in pattern:
            d, pat = pattern.rsplit("/", 1)
            base = os.path.join(root, d)
            if not os.path.isdir(base):
                continue
            for name in sorted(os.listdir(base)):
                rel = f"{d}/{name}"
                if fnmatch.fnmatch(name, pat) and not matches(rel, EXCLUDE) \
                        and os.path.isfile(os.path.join(root, rel)):
                    picked.append(rel)
        else:
            for name in sorted(os.listdir(root)):
                if fnmatch.fnmatch(name, pattern) and not matches(name, EXCLUDE) \
                        and os.path.isfile(os.path.join(root, name)):
                    picked.append(name)
    seen, out = set(), []
    for rel in picked:
        if rel not in seen:
            seen.add(rel)
            out.append(rel)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", required=True, metavar="user/repo",
                    help="target repo, e.g. can/evolution-trainer")
    ap.add_argument("--repo-type", default="model", choices=["model", "dataset"])
    ap.add_argument("--public", action="store_true", help="default is private")
    ap.add_argument("--dry-run", action="store_true",
                    help="list what would be uploaded and stop")
    args = ap.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    files = collect(root)
    if not files:
        print(f"error: nothing to upload -- is {root} the project root?")
        sys.exit(1)

    total = sum(os.path.getsize(os.path.join(root, f)) for f in files)
    print(f"project root: {root}")
    print(f"{len(files)} files, {total/1024:.0f} KB total:\n")
    for f in files:
        print(f"  {f:<32} {os.path.getsize(os.path.join(root, f))/1024:6.1f} KB")
    print()

    if args.dry_run:
        print("dry run - nothing uploaded")
        return

    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("error: pip install huggingface_hub")
        sys.exit(1)

    api = HfApi()
    try:
        me = api.whoami()
        print(f"logged in as: {me.get('name', '?')}")
    except Exception:                                    # noqa: BLE001
        print("error: not logged in. Run:  hf auth login")
        print("       (older huggingface_hub: huggingface-cli login)")
        sys.exit(1)

    repo = args.repo.replace("hf://", "").strip("/")
    if repo.count("/") != 1:
        print("error: --repo must be user/repo")
        sys.exit(2)

    api.create_repo(repo, repo_type=args.repo_type, exist_ok=True,
                    private=not args.public)
    print(f"uploading to https://huggingface.co/{repo} ...")
    api.upload_folder(
        folder_path=root, repo_id=repo, repo_type=args.repo_type,
        allow_patterns=files,
        commit_message="evolution trainer",
    )
    user = repo.split("/")[0]
    print(f"\ndone: https://huggingface.co/{repo}\n")
    print("Now, in the Vast.ai Jupyter terminal:")
    print(f"  pip install -q huggingface_hub")
    print(f"  hf download {repo} --local-dir ~/evo"
          + ("" if args.public else " --token hf_YOUR_READ_TOKEN"))
    print(f"  cd ~/evo && bash deploy/vastai_setup.sh")
    print(f"  bash deploy/vastai_launch.sh")
    print(f"\nAnd back here, to watch it:")
    print(f"  python main.py sync --from hf://{repo} --run runs/arch --interval 60")
    print(f"  python main.py watch --run runs/arch --live")
    del user


if __name__ == "__main__":
    main()
