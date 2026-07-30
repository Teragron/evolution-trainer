"""Pull a remote run directory to this machine, once or on a loop.

Three transports, because rented hardware varies in what it lets you reach:

* **ssh** (`user@host:/path`) -- rsync when available, else plain ssh + scp,
  since Windows ships an OpenSSH client but no rsync.
* **hf** (`hf://user/repo`) -- a Hugging Face repo as the middleman. Needs no
  inbound network on the training box at all, which is what makes it work on
  Vast.ai and similar container rentals.
* **http** (`http://host:port`) -- point at a box running `main.py serve`, for
  when you do have a mapped port.

Only the small files matter for watching progress: config.json, history.csv and
best.npz are a few kilobytes between them, so a 15 second poll costs nothing.
The big per-generation population dumps in `gen/` are skipped unless you ask
for `--full`.
"""
from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

from .hf import HfRemote, parse_repo

LIGHT_NAMES = ("config.json", "history.csv", "best.npz")


@dataclass
class Remote:
    host: str                    # user@1.2.3.4
    path: str                    # ~/evolution_game/runs/arch
    port: int = 22
    identity: Optional[str] = None

    def ssh_cmd(self) -> List[str]:
        cmd = ["ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new"]
        if self.port != 22:
            cmd += ["-p", str(self.port)]
        if self.identity:
            cmd += ["-i", self.identity]
        return cmd + [self.host]

    def scp_cmd(self) -> List[str]:
        cmd = ["scp", "-q", "-o", "BatchMode=yes",
               "-o", "StrictHostKeyChecking=accept-new"]
        if self.port != 22:
            cmd += ["-P", str(self.port)]
        if self.identity:
            cmd += ["-i", self.identity]
        return cmd

    def rsync_ssh(self) -> str:
        parts = ["ssh", "-o", "BatchMode=yes"]
        if self.port != 22:
            parts += ["-p", str(self.port)]
        if self.identity:
            parts += ["-i", self.identity]
        return " ".join(shlex.quote(p) for p in parts)


@dataclass
class HttpRemote:
    """A box running `main.py serve` -- no shell access required."""

    base_url: str                 # http://1.2.3.4:8080
    token: Optional[str] = None
    timeout: float = 30.0

    @property
    def host(self) -> str:
        return self.base_url

    @property
    def path(self) -> str:
        return "(served run directory)"

    def url(self, rel: str, **params) -> str:
        base = self.base_url.rstrip("/") + "/" + rel.lstrip("/")
        if self.token:
            params["t"] = self.token
        if params:
            base += "?" + urllib.parse.urlencode(params)
        return base

    def get(self, rel: str, **params) -> bytes:
        req = urllib.request.Request(self.url(rel, **params),
                                     headers={"User-Agent": "evolution-sync"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return resp.read()
        except urllib.error.HTTPError as exc:
            if exc.code == 403:
                raise RuntimeError("server rejected the token (--token)") from None
            raise RuntimeError(f"HTTP {exc.code} fetching {rel}") from None
        except (urllib.error.URLError, OSError) as exc:
            raise RuntimeError(f"cannot reach {self.base_url}: "
                               f"{getattr(exc, 'reason', exc)}") from None


Target = Union["Remote", HttpRemote, HfRemote]


def have_rsync() -> bool:
    return shutil.which("rsync") is not None


# ------------------------------------------------------------------------ rsync
def _pull_rsync(remote: Remote, local_dir: str, full: bool) -> int:
    cmd = ["rsync", "-rtz", "--partial", "-i", "-e", remote.rsync_ssh()]
    if not full:
        cmd += ["--exclude", "gen/", "--exclude", "*.log"]
    cmd += [f"{remote.host}:{remote.path.rstrip('/')}/", local_dir + os.sep]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"rsync failed: {res.stderr.strip()[:400]}")
    return sum(1 for line in res.stdout.splitlines()
               if line and not line.startswith("cd") and "/" in line or
               (line and line[0] in "<>c" and not line.startswith("cd")))


# -------------------------------------------------------------------- ssh + scp
def _remote_listing(remote: Remote, full: bool) -> List[Tuple[str, int]]:
    """[(relative path, size)] for the files we care about."""
    prune = "" if full else r" -not -path './gen/*' -not -name '*.log'"
    script = (f"cd {shlex.quote(remote.path)} 2>/dev/null || exit 3; "
              f"find . -type f{prune} -printf '%s %P\\n' 2>/dev/null || "
              f"find . -type f{prune} -exec stat -c '%s %n' {{}} \\;")
    res = subprocess.run(remote.ssh_cmd() + [script], capture_output=True, text=True)
    if res.returncode == 3:
        raise RuntimeError(f"remote path not found: {remote.path}")
    if res.returncode != 0:
        raise RuntimeError(f"ssh failed: {res.stderr.strip()[:400]}")
    out = []
    for line in res.stdout.splitlines():
        parts = line.strip().split(" ", 1)
        if len(parts) != 2:
            continue
        try:
            size = int(parts[0])
        except ValueError:
            continue
        rel = parts[1].lstrip("./")
        if rel:
            out.append((rel, size))
    return out


def _pull_scp(remote: Remote, local_dir: str, full: bool) -> int:
    listing = _remote_listing(remote, full)
    todo = []
    for rel, size in listing:
        local = os.path.join(local_dir, *rel.split("/"))
        if not os.path.exists(local) or os.path.getsize(local) != size:
            todo.append(rel)
    for rel in todo:
        local = os.path.join(local_dir, *rel.split("/"))
        os.makedirs(os.path.dirname(local) or ".", exist_ok=True)
        src = f"{remote.host}:{shlex.quote(remote.path.rstrip('/') + '/' + rel)}"
        res = subprocess.run(remote.scp_cmd() + [src, local],
                             capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"scp {rel} failed: {res.stderr.strip()[:200]}")
    return len(todo)


# ------------------------------------------------------------------------ http
def _pull_http(remote: HttpRemote, local_dir: str, full: bool) -> int:
    raw = remote.get("manifest.json", full="1" if full else "0")
    try:
        listing = json.loads(raw.decode())["files"]
    except (ValueError, KeyError):
        raise RuntimeError("that URL does not look like an evolution server "
                           "(no manifest.json). Is `main.py serve` running?") from None
    fetched = 0
    for entry in listing:
        rel = entry["path"].lstrip("/")
        if ".." in rel.split("/"):
            continue                      # never write outside the target dir
        local = os.path.join(local_dir, *rel.split("/"))
        if os.path.exists(local) and os.path.getsize(local) == entry["size"]:
            continue
        os.makedirs(os.path.dirname(local) or ".", exist_ok=True)
        body = remote.get(rel)
        tmp = local + ".part"
        with open(tmp, "wb") as fh:
            fh.write(body)
        os.replace(tmp, local)            # never let the viewer read a half file
        fetched += 1
    return fetched


# ------------------------------------------------------------------------- API
def pull_once(remote: Target, local_dir: str, full: bool = False,
              prefer_rsync: bool = True) -> int:
    """Copy down anything new. Returns the number of files updated."""
    os.makedirs(local_dir, exist_ok=True)
    if isinstance(remote, HfRemote):
        return remote.pull(local_dir, full=full)
    if isinstance(remote, HttpRemote):
        return _pull_http(remote, local_dir, full)
    if prefer_rsync and have_rsync():
        return _pull_rsync(remote, local_dir, full)
    return _pull_scp(remote, local_dir, full)


def follow(remote: Target, local_dir: str, interval: float = 15.0,
           full: bool = False, once: bool = False) -> None:
    """Poll forever (ctrl-C to stop), printing progress as it lands."""
    if isinstance(remote, HfRemote):
        tool = "huggingface hub"
    elif isinstance(remote, HttpRemote):
        tool = "http"
    else:
        tool = "rsync" if have_rsync() else "ssh+scp"
    print(f"syncing {remote.host}"
          + ("" if isinstance(remote, (HfRemote, HttpRemote)) else f":{remote.path}"))
    print(f"     -> {os.path.abspath(local_dir)}")
    print(f"  via {tool}, every {interval:g}s, "
          f"{'including' if full else 'excluding'} gen/ checkpoints\n")
    while True:
        t0 = time.time()
        try:
            n = pull_once(remote, local_dir, full=full)
            stamp = time.strftime("%H:%M:%S")
            summary = _local_summary(local_dir)
            print(f"[{stamp}] {n:3d} file(s) in {time.time()-t0:4.1f}s   {summary}",
                  flush=True)
        except RuntimeError as exc:
            print(f"[{time.strftime('%H:%M:%S')}] {exc}", flush=True)
        if once:
            return
        time.sleep(max(1.0, interval))


def _local_summary(local_dir: str) -> str:
    """One-line 'where is the run at' readout from what we've pulled."""
    import csv
    import numpy as np
    bits = []
    hist = os.path.join(local_dir, "history.csv")
    if os.path.exists(hist):
        try:
            with open(hist, newline="", encoding="utf-8") as fh:
                rows = list(csv.DictReader(fh))
            if rows:
                bits.append(f"gen {rows[-1]['generation']}")
        except (OSError, KeyError):
            pass
    bp = os.path.join(local_dir, "best.npz")
    if os.path.exists(bp):
        try:
            data = np.load(bp)
            bits.append(f"best {float(data['fitness']):.2f}m "
                        f"(gen {int(data['generation'])})")
        except (OSError, ValueError, KeyError):
            pass
    return " | ".join(bits) if bits else "nothing to report yet"


def parse_target(target: str) -> Tuple[str, str]:
    """'user@host:~/path/runs/arch' -> ('user@host', '~/path/runs/arch')."""
    if ":" not in target:
        raise ValueError("Expected user@host:/remote/path")
    host, path = target.split(":", 1)
    if not host or not path:
        raise ValueError("Expected user@host:/remote/path")
    return host, path


def make_target(spec: str, port: int = 22, identity: Optional[str] = None,
                token: Optional[str] = None, path_in_repo: str = "run",
                repo_type: str = "model") -> Target:
    """Build the right transport from a --from value.

    'hf://user/repo'            -> HfRemote     (no inbound network needed)
    'http://1.2.3.4:8080'       -> HttpRemote   (needs a mapped port)
    'user@1.2.3.4:~/runs/arch'  -> Remote       (needs ssh)
    """
    if spec.startswith("hf://"):
        return HfRemote(repo_id=parse_repo(spec), path_in_repo=path_in_repo,
                        repo_type=repo_type, token=token)
    if spec.startswith(("http://", "https://")):
        return HttpRemote(base_url=spec, token=token)
    host, path = parse_target(spec)
    return Remote(host=host, path=path, port=port, identity=identity)
