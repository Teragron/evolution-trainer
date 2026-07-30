"""Use a Hugging Face repo as the transport to and from a box you can't ssh into.

Vast.ai hands you a Jupyter tab, not an ssh account, and often no open ports
either. A HF repo solves both directions with no networking to arrange:

    on the rented box   python main.py push --run runs/arch --repo you/evo-run --follow
    on your laptop      python main.py sync --from hf://you/evo-run --run runs/arch
                        python main.py watch --run runs/arch --live

The payload is tiny -- `best.npz` is a couple of KB -- so pushing every few
minutes costs nothing. Keep `--full` off unless you want the whole `gen/`
archive; each push is a commit, and there is no reason to version megabytes of
intermediate populations.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

LIGHT_PATTERNS = ("config.json", "history.csv", "best.npz")


def _api(token: Optional[str] = None):
    try:
        from huggingface_hub import HfApi
    except ImportError:
        raise RuntimeError(
            "huggingface_hub is not installed. Run:\n"
            "    pip install huggingface_hub") from None
    return HfApi(token=token or os.environ.get("HF_TOKEN"))


def parse_repo(spec: str) -> str:
    """'hf://user/repo' or 'user/repo' -> 'user/repo'."""
    repo = spec[5:] if spec.startswith("hf://") else spec
    repo = repo.strip("/")
    if repo.count("/") != 1 or not all(repo.split("/")):
        raise ValueError("Expected hf://<user>/<repo>")
    return repo


@dataclass
class HfRemote:
    """A run directory living inside a Hugging Face repo."""

    repo_id: str
    path_in_repo: str = "run"
    repo_type: str = "model"
    token: Optional[str] = None
    revision: str = "main"

    @property
    def host(self) -> str:
        return f"hf://{self.repo_id}"

    @property
    def path(self) -> str:
        return self.path_in_repo or "(repo root)"

    # ---------------------------------------------------------------- listing
    def listing(self, full: bool) -> List[Tuple[str, int]]:
        """[(path relative to the run dir, size)]."""
        api = _api(self.token)
        try:
            info = api.repo_info(self.repo_id, repo_type=self.repo_type,
                                 revision=self.revision, files_metadata=True)
        except Exception as exc:                       # noqa: BLE001 - hub raises many
            raise RuntimeError(f"cannot read {self.host}: {exc}") from None
        prefix = (self.path_in_repo.strip("/") + "/") if self.path_in_repo else ""
        out = []
        for sib in info.siblings or []:
            name = sib.rfilename
            if prefix and not name.startswith(prefix):
                continue
            rel = name[len(prefix):] if prefix else name
            if not rel or rel.startswith(".gitattributes"):
                continue
            if not full and (rel.startswith("gen/") or rel.endswith(".log")):
                continue
            out.append((rel, sib.size if sib.size is not None else -1))
        return out

    # ------------------------------------------------------------------ pull
    def pull(self, local_dir: str, full: bool = False) -> int:
        from huggingface_hub import hf_hub_download
        os.makedirs(local_dir, exist_ok=True)
        prefix = (self.path_in_repo.strip("/") + "/") if self.path_in_repo else ""
        fetched = 0
        for rel, size in self.listing(full):
            local = os.path.join(local_dir, *rel.split("/"))
            if size >= 0 and os.path.exists(local) and os.path.getsize(local) == size:
                continue
            try:
                cached = hf_hub_download(
                    self.repo_id, prefix + rel, repo_type=self.repo_type,
                    revision=self.revision,
                    token=self.token or os.environ.get("HF_TOKEN"))
            except Exception as exc:                   # noqa: BLE001
                raise RuntimeError(f"download of {rel} failed: {exc}") from None
            os.makedirs(os.path.dirname(local) or ".", exist_ok=True)
            tmp = local + ".part"
            with open(cached, "rb") as src, open(tmp, "wb") as dst:
                dst.write(src.read())
            os.replace(tmp, local)      # the viewer never sees a half-written file
            fetched += 1
        return fetched

    # ------------------------------------------------------------------ push
    def push(self, local_dir: str, full: bool = False,
             message: Optional[str] = None) -> int:
        api = _api(self.token)
        api.create_repo(self.repo_id, repo_type=self.repo_type, exist_ok=True,
                        private=True)
        patterns = None if full else ["*.json", "*.csv", "best.npz"]
        ignore = None if full else ["gen/*", "*/gen/*", "*.log", "*.tmp.npz"]
        n = _count(local_dir, full)
        api.upload_folder(folder_path=local_dir, repo_id=self.repo_id,
                          repo_type=self.repo_type, path_in_repo=self.path_in_repo,
                          allow_patterns=patterns, ignore_patterns=ignore,
                          commit_message=message or _default_message(local_dir))
        return n


def _count(local_dir: str, full: bool) -> int:
    n = 0
    for dirpath, dirnames, files in os.walk(local_dir):
        dirnames[:] = [d for d in dirnames if d != "__pycache__"]
        rel_dir = os.path.relpath(dirpath, local_dir).replace(os.sep, "/")
        if not full and (rel_dir == "gen" or rel_dir.endswith("/gen")):
            continue
        for f in files:
            if not full and (f.endswith((".log", ".tmp.npz"))):
                continue
            n += 1
    return n


def _default_message(local_dir: str) -> str:
    """Commit message that says where the run got to, so the repo history is
    itself a training log."""
    import json
    try:
        import numpy as np
        data = np.load(os.path.join(local_dir, "best.npz"))
        return (f"best {float(data['fitness']):.2f} m "
                f"at generation {int(data['generation'])}")
    except Exception:                                  # noqa: BLE001
        return "training progress"


def push_loop(remote: HfRemote, local_dir: str, interval: float = 180.0,
              full: bool = False, once: bool = False) -> None:
    """Keep the repo in step with a run that is still going."""
    print(f"pushing {os.path.abspath(local_dir)}")
    print(f"     -> {remote.host} ({remote.path}, {remote.repo_type} repo)")
    if not once:
        print(f"  every {interval:g}s, ctrl-C to stop"
              f"{'' if full else ', excluding gen/ checkpoints'}\n")
    while True:
        t0 = time.time()
        try:
            n = remote.push(local_dir, full=full)
            print(f"[{time.strftime('%H:%M:%S')}] pushed {n} file(s) "
                  f"in {time.time()-t0:4.1f}s", flush=True)
        except RuntimeError as exc:
            print(f"[{time.strftime('%H:%M:%S')}] {exc}", flush=True)
        if once:
            return
        time.sleep(max(30.0, interval))
