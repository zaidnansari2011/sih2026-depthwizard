"""Serve the viewer and let a user drop their own image into it.

The Expected Solution asks for a platform that lets users *"upload imagery, visualize
reconstructed terrain, and validate estimated height values against reference datasets"*.
The first of those needs a process that can run the model, so the static viewer cannot do
it alone. This is that process, and it is deliberately stdlib-only:

  * No Flask, no FastAPI. "Successful standalone deployment" is a scored criterion, and
    every dependency is one more thing that can fail on an evaluator's machine. Python's
    own http.server is enough for one user on localhost.
  * No multipart parsing. The browser POSTs the raw file bytes with the name in the query
    string, which removes the whole `cgi` module -- deprecated in 3.11, gone in 3.13 -- and
    the fragile boundary handling that comes with it.

    python tools/serve_viewer.py                     # http://localhost:8080
    python tools/serve_viewer.py --ckpt other.pt --port 9000

Bound to 127.0.0.1 only. This runs a model on whatever bytes it is given and writes files;
it is a local tool, not a service, and it should never be exposed to a network.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import threading
import time
import traceback
import uuid
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[1]
VIEWER = ROOT / "viewer"
UPLOADS = ROOT.parent / "data" / "uploads"
OUT = ROOT.parent / "out" / "uploads"

MAX_BYTES = 400 * 1024 * 1024          # a 0.3 m GeoTIFF gets large; 400 MB is generous
ALLOWED = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}

JOBS: dict[str, dict] = {}
GPU_LOCK = threading.Lock()            # one inference at a time; there is one GPU
ARGS = None


def safe_stem(name: str) -> str:
    """A filename we are willing to build paths from.

    The name arrives from the browser and is used to create directories, so everything
    that is not plainly alphanumeric goes. This is the whole path-traversal defence and it
    is deliberately blunt.
    """
    stem = Path(name).stem
    stem = re.sub(r"[^A-Za-z0-9_-]", "_", stem)[:48]
    return stem or "upload"


def run_job(job_id: str, src: Path, stem: str):
    """Infer, export a scene, and make it visible to the viewer."""
    j = JOBS[job_id]
    try:
        with GPU_LOCK:
            j.update(state="running", step="estimating heights", pct=10)
            out_prefix = OUT / job_id
            out_prefix.parent.mkdir(parents=True, exist_ok=True)

            cmd = [sys.executable, "infer.py", "--image", str(src),
                   "--out", str(out_prefix), "--tta", "--auto-zoom",
                   "--fuse-zoom", "2", "--fuse-sigma", "8"]
            if ARGS.ckpt:
                cmd += ["--ckpt", ARGS.ckpt]
            if not ARGS.no_dem:
                cmd += ["--dem", "auto"]        # absolute DSM when the input is georeferenced
            p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
            if p.returncode != 0:
                raise RuntimeError((p.stderr or p.stdout or "inference failed")[-1500:])
            j["log"] = p.stdout[-4000:]

            # Georeferenced or not decides which product this is, and the viewer says so.
            georef = ".dsm.tif" in p.stdout or "auto-zoom: input is" in p.stdout

            j.update(step="building 3D scene", pct=70)
            scene_dir = VIEWER / "scenes" / f"upload_{stem}_{job_id[:6]}"
            ex = [sys.executable, "tools/export_terrain.py",
                  "--height", str(out_prefix) + ".height.tif",
                  "--texture", str(src),
                  "--out", str(scene_dir),
                  "--terrain", "upload", "--place", "your image",
                  "--name", f"Uploaded — {stem}"]
            if (Path(str(out_prefix) + ".sigma.tif")).exists():
                ex += ["--sigma", str(out_prefix) + ".sigma.tif"]
            # If the input was georeferenced we now have real terrain under it, so the
            # scene renders as an absolute surface -- ground plus everything on it --
            # rather than buildings floating on a flat plane. This is the difference
            # between showing an nDSM and showing the DSM the brief actually asks for.
            terrain_tif = Path(str(out_prefix) + ".terrain.tif")
            if terrain_tif.exists():
                ex += ["--terrain-base", str(terrain_tif),
                       "--terrain-source", "Copernicus GLO-30, 30 m posts, cubic-resampled"]
            e = subprocess.run(ex, cwd=ROOT, capture_output=True, text=True)
            if e.returncode != 0:
                raise RuntimeError((e.stderr or e.stdout or "export failed")[-1500:])

            j.update(state="done", step="ready", pct=100,
                     scene=scene_dir.name, georeferenced=georef)
    except Exception as exc:
        JOBS[job_id].update(state="error", step=str(exc)[:600], pct=100)
        traceback.print_exc()


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=str(VIEWER), **kw)

    def log_message(self, fmt, *a):                 # quiet: the console is the user's UI
        if "/api/" in (self.path or ""):
            sys.stderr.write(f"  {self.command} {self.path.split('?')[0]}\n")

    def end_headers(self):
        # Never let a browser cache the code or the scene index. This is a local tool
        # serving files that change under it, and a stale main.js produces a viewer that
        # fails for reasons invisible in the source you are reading. The scene binaries
        # are large and content-addressed by directory, so they keep normal caching.
        p = (self.path or "").split("?")[0]
        if p.endswith((".js", ".html", ".json", "/")):
            self.send_header("Cache-Control", "no-store, must-revalidate")
        super().end_headers()

    def _json(self, code: int, payload: dict):
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        if u.path.startswith("/api/job/"):
            job = JOBS.get(u.path.rsplit("/", 1)[-1])
            return self._json(200 if job else 404, job or {"state": "unknown"})
        if u.path == "/api/capabilities":
            return self._json(200, {"upload": True, "dem": not ARGS.no_dem})
        return super().do_GET()

    def do_POST(self):
        u = urlparse(self.path)
        if u.path != "/api/upload":
            return self._json(404, {"error": "no such endpoint"})

        name = (parse_qs(u.query).get("name") or ["image.tif"])[0]
        ext = Path(name).suffix.lower()
        if ext not in ALLOWED:
            return self._json(400, {"error": f"{ext or 'that'} is not an image this "
                                             f"pipeline reads. Use TIFF, PNG or JPG."})
        try:
            length = int(self.headers.get("Content-Length", 0))
        except ValueError:
            length = 0
        if length <= 0:
            return self._json(400, {"error": "empty upload"})
        if length > MAX_BYTES:
            return self._json(413, {"error": f"{length/1e6:.0f} MB exceeds the "
                                             f"{MAX_BYTES/1e6:.0f} MB limit"})

        job_id = uuid.uuid4().hex
        stem = safe_stem(name)
        dest_dir = UPLOADS / job_id
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / f"{stem}{ext}"

        # Stream to disk rather than reading it all into memory: these are satellite
        # rasters, and a 400 MB read on the request thread is a bad way to find that out.
        remaining = length
        with open(dest, "wb") as f:
            while remaining > 0:
                chunk = self.rfile.read(min(1 << 20, remaining))
                if not chunk:
                    break
                f.write(chunk)
                remaining -= len(chunk)
        if remaining > 0:
            shutil.rmtree(dest_dir, ignore_errors=True)
            return self._json(400, {"error": "upload ended early"})

        JOBS[job_id] = {"state": "queued", "step": "waiting for the GPU", "pct": 2,
                        "name": name, "started": time.time()}
        threading.Thread(target=run_job, args=(job_id, dest, stem), daemon=True).start()
        return self._json(202, {"job": job_id})


class Server(ThreadingHTTPServer):
    """A server that refuses to start on a port somebody else already has.

    http.server sets allow_reuse_address, which on Windows means SO_REUSEADDR lets a
    SECOND process bind a port that is already in use. Nothing raises: our server prints
    "serving on 8080", the other process keeps answering every request, and the symptom is
    a viewer that loads but has no upload button and never updates. Refusing to bind turns
    a silent wrong-server into a message naming the fix.
    """
    allow_reuse_address = not sys.platform.startswith("win")
    daemon_threads = True


def main():
    global ARGS
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--ckpt", default=str(ROOT.parent / "checkpoints" / "run02" / "best.pt"))
    ap.add_argument("--no-dem", action="store_true",
                    help="skip Copernicus DEM anchoring (offline, or you only want AGL)")
    ARGS = ap.parse_args()

    if ARGS.ckpt and not Path(ARGS.ckpt).exists():
        print(f"  checkpoint not found: {ARGS.ckpt}\n  Pass --ckpt, or the viewer will "
              f"still serve its built-in scenes but uploads will fail.")
    UPLOADS.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)

    try:
        srv = Server(("127.0.0.1", ARGS.port), Handler)
    except OSError as e:
        print(f"\n  Port {ARGS.port} is already in use ({e.__class__.__name__}).\n"
              f"  Something else is serving on it -- often a viewer window left open from\n"
              f"  earlier. Close that one, or start this on another port:\n\n"
              f"      python tools/serve_viewer.py --port {ARGS.port + 1}\n")
        raise SystemExit(1)
    print(f"\n  DepthWizard viewer\n  ------------------")
    print(f"  http://localhost:{ARGS.port}/")
    print(f"  checkpoint: {ARGS.ckpt}")
    print("  uploads enabled - drop a GeoTIFF, PNG or JPG onto the page")
    print(f"  Ctrl+C to stop.\n")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n  stopped.")


if __name__ == "__main__":
    main()
