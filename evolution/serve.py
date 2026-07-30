"""Expose a run directory over plain HTTP, for boxes you cannot ssh into.

Vast.ai (and most container-rental services) hand you a Jupyter tab and a few
mapped ports, not an ssh account -- so `scp` is off the table. Running

    python main.py serve --run runs/arch --port 8080 --token mysecret

turns the run directory into something `main.py sync --from http://...` can pull
from, which restores the live-follow workflow without any shell access.

It serves exactly one directory and only ever reads. What's inside is neural
network weights and fitness numbers, but the connection is unencrypted, so use
`--token` and don't leave it running longer than the run.
"""
from __future__ import annotations

import json
import os
import posixpath
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from typing import List, Optional
from urllib.parse import parse_qs, urlparse

MANIFEST_PATH = "/manifest.json"


def build_manifest(root: str, full: bool = True) -> List[dict]:
    """[{path, size, mtime}] for everything under root, newest info each call."""
    out = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d != "__pycache__"]
        for name in filenames:
            if name.endswith((".tmp", ".tmp.npz")):
                continue
            abs_path = os.path.join(dirpath, name)
            rel = os.path.relpath(abs_path, root).replace(os.sep, "/")
            if not full and (rel.startswith("gen/") or "/gen/" in rel
                             or rel.endswith(".log")):
                continue
            try:
                st = os.stat(abs_path)
            except OSError:
                continue
            out.append({"path": rel, "size": st.st_size, "mtime": st.st_mtime})
    out.sort(key=lambda d: d["path"])
    return out


class RunHandler(SimpleHTTPRequestHandler):
    """Read-only file server + a manifest endpoint, optionally token-gated."""

    token: Optional[str] = None
    root: str = "."

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=self.root, **kwargs)

    # ------------------------------------------------------------ plumbing
    def log_message(self, fmt, *args):  # quieter than the default
        pass

    def _authorised(self, query: dict) -> bool:
        if not self.token:
            return True
        supplied = (query.get("t", [None])[0]
                    or self.headers.get("X-Auth-Token"))
        return supplied == self.token

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        if not self._authorised(query):
            self.send_error(403, "bad or missing token")
            return
        route = posixpath.normpath(parsed.path)
        if route == MANIFEST_PATH:
            full = query.get("full", ["1"])[0] not in ("0", "false", "no")
            body = json.dumps({"files": build_manifest(self.root, full)}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
            return
        if route in ("/", "/index.html"):
            body = _landing(self.root).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.path = parsed.path  # strip the query before hitting the file server
        super().do_GET()

    def do_POST(self):  # noqa: N802
        self.send_error(405, "read-only")

    do_PUT = do_DELETE = do_POST


def _landing(root: str) -> str:
    files = build_manifest(root, full=False)
    rows = "".join(
        f"<tr><td><a href='{f['path']}'>{f['path']}</a></td>"
        f"<td style='text-align:right'>{f['size']:,} B</td></tr>"
        for f in files)
    return f"""<!doctype html><meta charset=utf-8>
<title>evolution run</title>
<style>body{{font:14px/1.5 ui-monospace,monospace;background:#12141a;color:#dde;
padding:2rem}}a{{color:#6cb2ff}}td{{padding:2px 14px 2px 0}}h1{{font-size:1rem;
color:#ffc454}}</style>
<h1>evolution run: {os.path.basename(os.path.abspath(root))}</h1>
<p>Pull this from your laptop with:</p>
<pre>python main.py sync --from http://HOST:PORT --run runs/arch
python main.py watch --run runs/arch --live</pre>
<table>{rows}</table>
<p><a href="manifest.json">manifest.json</a></p>"""


def serve(run_dir: str, port: int = 8080, host: str = "0.0.0.0",
          token: Optional[str] = None, background: bool = False):
    """Start the server. Returns the server object (call shutdown() to stop)."""
    run_dir = os.path.abspath(run_dir)
    os.makedirs(run_dir, exist_ok=True)
    handler = type("BoundRunHandler", (RunHandler,),
                   {"root": run_dir, "token": token})
    httpd = ThreadingHTTPServer((host, port), handler)
    httpd.daemon_threads = True
    if background:
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        return httpd
    suffix = f"?t={token}" if token else ""
    print(f"serving {run_dir}")
    print(f"     on http://{host}:{port}/{suffix}")
    print(f"  pull with: python main.py sync --from http://<host>:{port} --run runs/arch"
          + (f" --token {token}" if token else ""))
    print("  ctrl-C to stop\n")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        httpd.shutdown()
    return httpd
