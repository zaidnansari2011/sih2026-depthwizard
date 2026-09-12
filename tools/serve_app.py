"""Serve DepthWizard publicly -- the same product as serve_viewer.py, hardened for a host.

`serve_viewer.py` is a local tool and says so in its own docstring: it binds 127.0.0.1,
accepts 400 MB uploads, and trusts whoever is at the keyboard because that is the person
who started it. None of those hold on a public URL, so this is a separate entry point
rather than a flag on that one -- the local tool stays exactly as convenient as it was,
and nothing about the deployed surface can silently change it.

What actually differs, and why each one matters once anyone can reach it:

  * **Binds 0.0.0.0**, port from $PORT. Azure Container Apps terminates TLS at the
    ingress and forwards plain HTTP to whatever port it advertised, so binding loopback
    would make the container unreachable rather than safe.
  * **Fast mode by default.** Measured on CPU with the GPU hidden, a 1024x1024 tile costs
    5.0 s single-pass against 57.2 s with x8 D4 TTA and zoom-2 fusion. A visitor who
    waits a minute assumes it is broken, so the default is the fast path and the accurate
    one is opt-in -- and the viewer is told which it got, because a number whose settings
    are unstated is not evidence.
  * **Bounded queue.** One request costs CPU-minutes, so an unbounded queue is a way to
    take the site down by being mildly popular. Beyond MAX_QUEUE the answer is 429 with
    the reason, not a job that will never run.
  * **Per-IP spacing**, so one client cannot hold the whole queue.
  * **50 MB uploads**, not 400. A 0.3 m GeoTIFF gets large, but not on a free tier.
  * **Scene reaping.** Every upload writes a scene of 13-50 MB into an ephemeral
    container filesystem. Without a cap that fills the disk and the process dies of
    something unrelated hours later; SCENE_BUDGET keeps the newest and deletes the rest.
  * **DEM anchoring off by default.** `--dem auto` reaches Copernicus at request time.
    That is a network dependency inside a request that already takes a minute, so it is
    opt-in per deployment rather than a surprise timeout.
  * **/healthz**, because the platform restarts what it cannot probe.

    python tools/serve_app.py                    # 0.0.0.0:8080
    python tools/serve_app.py --dem              # enable Copernicus anchoring
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import traceback
import uuid
from collections import deque
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[1]
VIEWER = ROOT / "viewer"
SCENES = VIEWER / "scenes"

# Uploads and raw model output are scratch: they are read once by export_terrain and never
# again. Keeping them outside the served tree means a path bug cannot turn into a way to
# download other people's inputs.
WORK = Path(os.environ.get("DW_WORK", "/tmp/depthwizard"))
UPLOADS = WORK / "uploads"
OUT = WORK / "out"

MAX_BYTES = int(os.environ.get("DW_MAX_UPLOAD_MB", "50")) * 1024 * 1024
MAX_QUEUE = int(os.environ.get("DW_MAX_QUEUE", "3"))       # incl. the one running
IP_SPACING = float(os.environ.get("DW_IP_SPACING", "20"))  # seconds between one IP's jobs
SCENE_BUDGET = int(os.environ.get("DW_SCENE_BUDGET", "12"))
JOB_TTL = float(os.environ.get("DW_JOB_TTL", "3600"))
ALLOWED = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}

# Cost is driven by PIXELS, not bytes, and the two are barely related for satellite
# imagery: a Maxar visual strip is 17408x17408 (303 Mpx) in 12 MB, because it is sparse
# and JPEG-compressed inside the TIFF. It sails through a 50 MB byte cap and then needs
# about three hours, so the job times out at 600 s having told the user nothing until
# then. These constants turn that into an instant, specific refusal.
#
# Measured on this B2 plan, 6 Sep: a 1024x1024 (1.05 Mpx) upload completes in 60 s wall.
# Roughly 25 s of that is fixed -- loading the 99 MB checkpoint, exporting the scene,
# writing four arrays -- and the remainder scales with area.
FIXED_S = float(os.environ.get("DW_FIXED_S", "25"))
SEC_PER_MPX = float(os.environ.get("DW_SEC_PER_MPX", "33"))
# 57.2 s / 5.02 s, the two inference timings measured with and without x8 D4 TTA and
# zoom-2 fusion. Accurate mode is an order of magnitude dearer and its ceiling must move
# with it, or "accurate" becomes a reliable way to hit the timeout.
TTA_FACTOR = float(os.environ.get("DW_TTA_FACTOR", "11.4"))
# Refuse anything projected past this share of the timeout. Leaves room for a cold cache,
# a noisy neighbour, or an image that tiles less evenly than the one that was measured.
BUDGET_FRAC = float(os.environ.get("DW_BUDGET_FRAC", "0.8"))

JOBS: dict[str, dict] = {}
JOB_ORDER: deque[str] = deque()
LAST_BY_IP: dict[str, float] = {}
STATE_LOCK = threading.Lock()      # guards JOBS, JOB_ORDER, LAST_BY_IP
RUN_LOCK = threading.Lock()        # one inference at a time; the box has a few cores
ARGS = None

# Scene directories that shipped in the image. These are the evidence the site is built
# on and must never be reaped to make room for a visitor's upload.
BAKED: set[str] = set()


def safe_stem(name: str) -> str:
    """A filename we are willing to build paths from.

    The name arrives from the browser and becomes a directory, so anything not plainly
    alphanumeric goes. This is the whole path-traversal defence and it is deliberately
    blunt.
    """
    stem = re.sub(r"[^A-Za-z0-9_-]", "_", Path(name).stem)[:48]
    return stem or "upload"


def image_shape(path: Path) -> tuple[int, int]:
    """(width, height) from the file's header, without decoding the pixels.

    Both readers are lazy about data but strict about structure, which is the second
    reason to do this: a truncated upload fails here, in the millisecond after it lands
    and with a sentence the user can act on, instead of forty seconds into inference with
    a TIFFReadEncodedTile traceback. Imports are deferred so the parent process does not
    carry GDAL for the sake of serving static files.
    """
    if path.suffix.lower() in (".tif", ".tiff"):
        import rasterio                       # GeoTIFF: rasterio reads BigTIFF, PIL may not
        with rasterio.open(path) as d:
            return d.width, d.height
    from PIL import Image
    with Image.open(path) as im:
        return im.width, im.height


def verify_decodes(path: Path) -> None:
    """Force a full decode, and raise ValueError if the pixels are not all there.

    The header check above is not enough, and the comment that used to say it was is now
    corrected. Measured 12 Sep 2026: a PNG truncated to a third of its bytes opens
    perfectly through GDAL, which pads the missing **26% of rows with black** and emits
    only a warning -- and the pipeline then produced a confident 512x512 height map over
    invented pixels. Nothing downstream can tell fabricated black from a dark field, so a
    judge whose download was cut off would get a plausible-looking result and no hint.

    PIL is strict exactly where GDAL is forgiving, so the check is PIL's. Where PIL cannot
    open the file at all -- BigTIFF, exotic compressions, the formats rasterio is here for
    -- it stays silent rather than refusing on GDAL's behalf, because a false refusal of a
    valid GeoTIFF would be the worse error.
    """
    from PIL import Image
    try:
        with Image.open(path) as im:
            im.load()                          # decodes; Image.open alone only reads a header
    except OSError as exc:
        text = str(exc).lower()
        if "truncated" in text or "broken" in text or "incomplete" in text:
            raise ValueError(
                "that file ends part-way through -- the download or upload did not "
                "finish. About a quarter of an image can be missing and still open, so "
                "send the whole file rather than trusting the preview.") from exc
        # PIL simply does not read this format. Leave it to the GeoTIFF reader.
    except Exception:
        pass                                   # same reasoning: not PIL's to judge


def estimate_seconds(mpx: float, quality: str) -> float:
    """Projected wall time for an upload of this size, from measured constants."""
    per_mpx = SEC_PER_MPX * (TTA_FACTOR if quality == "accurate" else 1.0)
    return FIXED_S + mpx * per_mpx


def max_megapixels(quality: str, timeout: float) -> float:
    """The largest image whose projected time fits the budget."""
    per_mpx = SEC_PER_MPX * (TTA_FACTOR if quality == "accurate" else 1.0)
    return max(0.1, (timeout * BUDGET_FRAC - FIXED_S) / per_mpx)


def human_time(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:.0f} seconds"
    if seconds < 5400:
        return f"{seconds/60:.0f} minutes"
    return f"{seconds/3600:.1f} hours"


def prune_index(scenes_root: Path) -> None:
    """Drop index entries whose directory is gone.

    export_terrain.py does this too, but only when it next runs. If a scene is reaped and
    nobody uploads afterwards, the stale entry sits in the index and the viewer fetches a
    manifest that is not there -- which presents as the whole scene picker being broken,
    not as one missing scene.
    """
    index = scenes_root / "index.json"
    if not index.exists():
        return
    try:
        known = json.loads(index.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return
    live = [k for k in known if (scenes_root / k.get("dir", "")).is_dir()]
    if len(live) != len(known):
        index.write_text(json.dumps(live, indent=2), encoding="utf-8")


def reap_scenes() -> None:
    """Keep the newest SCENE_BUDGET uploaded scenes; delete the rest.

    Ephemeral container storage is small and each scene is 13-50 MB, so this is the
    difference between a site that stays up and one that dies of a full disk during the
    demo. Baked scenes are exempt.
    """
    try:
        uploaded = [d for d in SCENES.iterdir() if d.is_dir() and d.name not in BAKED]
    except OSError:
        return
    uploaded.sort(key=lambda d: d.stat().st_mtime, reverse=True)
    for d in uploaded[SCENE_BUDGET:]:
        shutil.rmtree(d, ignore_errors=True)
    prune_index(SCENES)


def reap_jobs(now: float) -> None:
    """Forget finished jobs and their scratch files once nobody is polling them."""
    for jid in list(JOB_ORDER):
        j = JOBS.get(jid)
        if not j or j.get("state") in ("queued", "running"):
            continue
        if now - j.get("started", now) < JOB_TTL:
            continue
        JOBS.pop(jid, None)
        try:
            JOB_ORDER.remove(jid)
        except ValueError:
            pass
        shutil.rmtree(UPLOADS / jid, ignore_errors=True)
        for f in OUT.glob(f"{jid}.*"):
            f.unlink(missing_ok=True)


class JobFailed(Exception):
    """A failure we can describe in a sentence, carrying the raw text for the log."""

    def __init__(self, message: str, raw: str = ""):
        super().__init__(message)
        self.message = message
        self.raw = raw


# Ordered, because the first match wins and the specific ones have to come before the
# general. Each entry is (marker seen in the tool's output, what to tell the visitor).
FAILURE_SIGNS: list[tuple[str, str]] = [
    ("out of memory",
     "the server ran out of memory on that image. Try a smaller one, or the fast setting."),
    ("CUDA error",
     "the graphics card reported an error part-way through. Try again in a moment."),
    ("not a multiple of the patch size",
     "that image is an awkward size for the model and it could not pad around it."),
    ("cannot identify image file",
     "that file could not be read as an image. It may be truncated, or not the format its "
     "name claims."),
    ("not recognized as a supported file format",
     "that file could not be read as an image. It may be truncated, or not the format its "
     "name claims."),
    ("TIFFReadDirectory",
     "that TIFF is damaged or incomplete -- its directory could not be read."),
    ("TIFFReadEncodedTile",
     "that TIFF is damaged or incomplete -- it ends part-way through the image data."),
    ("truncated",
     "that file ends part-way through. The upload may not have finished."),
    ("RasterioIOError",
     "that file could not be opened as a raster."),
    ("No such file or directory",
     "the uploaded file went missing before it could be processed. Try again."),
    ("MemoryError",
     "that image is too large to hold in memory. Try a smaller one."),
]


def plain_failure(raw: str) -> str:
    """Turn a subprocess's dying words into a sentence a visitor can act on.

    What used to happen: run_job raised RuntimeError with the last 1500 characters of
    stderr, the handler stored str(exc)[:600] as the job's `step`, and the viewer wrote
    `step` straight into the progress label. Production showed a judge
    `recent call last):\\n  File "/tmp/8df0c0ea6632463/infer.py", line 548...` -- a
    traceback starting mid-word, truncated before the actual error, carrying server paths.
    A stability criterion can see nothing worse.

    The trace is not discarded, it just goes where traces belong: the server log.
    """
    low = (raw or "").lower()
    for marker, sentence in FAILURE_SIGNS:
        if marker.lower() in low:
            return sentence
    return ("that image could not be processed. The details are in the server log; try "
            "another file, or a smaller crop.")


def active_jobs() -> int:
    return sum(1 for j in JOBS.values() if j.get("state") in ("queued", "running"))


def run_job(job_id: str, src: Path, stem: str, quality: str):
    """Infer, export a scene, and make it visible to the viewer.

    Deliberately still a subprocess, exactly as serve_viewer.py does it. Loading the
    checkpoint costs ~8 s of every request and holding the model in-process would remove
    that, but it means reimplementing the auto-zoom, fusion and DEM logic that lives in
    infer.py's main(). Duplicating the inference path is how a served result quietly stops
    matching the measured one, so the 8 s stays until infer.py exposes a reusable entry
    point.
    """
    j = JOBS[job_id]
    try:
        with RUN_LOCK:
            j.update(state="running", step="estimating heights", pct=10)
            out_prefix = OUT / job_id
            out_prefix.parent.mkdir(parents=True, exist_ok=True)

            cmd = [sys.executable, "infer.py", "--image", str(src),
                   "--out", str(out_prefix), "--auto-zoom"]
            if quality == "accurate":
                # The settings every published per-terrain number was measured with.
                cmd += ["--tta", "--fuse-zoom", "2", "--fuse-sigma", "8"]
            if ARGS.ckpt:
                cmd += ["--ckpt", ARGS.ckpt]
            if ARGS.dem:
                cmd += ["--dem", "auto"]

            p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                               timeout=ARGS.timeout)
            if p.returncode != 0:
                out = p.stderr or p.stdout or ""
                raise JobFailed(plain_failure(out), out)
            georef = ".dsm.tif" in p.stdout or "auto-zoom: input is" in p.stdout

            j.update(step="building 3D scene", pct=70)
            scene_dir = SCENES / f"upload_{stem}_{job_id[:6]}"
            ex = [sys.executable, "tools/export_terrain.py",
                  "--height", f"{out_prefix}.height.tif",
                  "--texture", str(src),
                  "--out", str(scene_dir),
                  "--terrain", "upload", "--place", "your image",
                  "--name", f"Uploaded — {stem}"]
            if Path(f"{out_prefix}.sigma.tif").exists():
                ex += ["--sigma", f"{out_prefix}.sigma.tif"]
            terrain_tif = Path(f"{out_prefix}.terrain.tif")
            if terrain_tif.exists():
                ex += ["--terrain-base", str(terrain_tif),
                       "--terrain-source", "Copernicus GLO-30, 30 m posts, cubic-resampled"]
            e = subprocess.run(ex, cwd=ROOT, capture_output=True, text=True,
                               timeout=ARGS.timeout)
            if e.returncode != 0:
                out = e.stderr or e.stdout or ""
                raise JobFailed(plain_failure(out), out)

            reap_scenes()
            j.update(state="done", step="ready", pct=100,
                     scene=scene_dir.name, georeferenced=georef, quality=quality)
    except subprocess.TimeoutExpired:
        JOBS[job_id].update(state="error", pct=100,
                            step=f"gave up after {ARGS.timeout:.0f}s. Try a smaller image.")
    except JobFailed as f:
        # The sentence goes to the browser; the trace goes to the log, in full.
        print(f"job {job_id} failed: {f.message}\n{f.raw}", file=sys.stderr, flush=True)
        JOBS[job_id].update(state="error", step=f.message, pct=100)
    except Exception:
        # A bug in here, rather than in what it runs. Same rule: the visitor gets a
        # sentence, the operator gets the trace.
        traceback.print_exc()
        JOBS[job_id].update(state="error", pct=100,
                            step="something went wrong on the server while handling that "
                                 "image. It has been logged.")
    finally:
        shutil.rmtree(src.parent, ignore_errors=True)


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=str(VIEWER), **kw)

    def log_message(self, fmt, *a):
        if "/api/" in (self.path or ""):
            sys.stderr.write(f"  {self.command} {self.path.split('?')[0]}\n")

    def client_ip(self) -> str:
        """The visitor's address as the ingress saw it.

        Behind Container Apps every connection arrives from the platform's proxy, so
        self.client_address is the same value for everyone and per-IP spacing would
        throttle the whole world as one client. X-Forwarded-For's first hop is the
        original caller. It is client-supplied and therefore spoofable -- which is fine
        here, because this only paces polite users apart and the queue cap is what
        actually bounds the work.
        """
        xff = self.headers.get("X-Forwarded-For", "")
        if xff:
            return xff.split(",")[0].strip()
        return self.client_address[0]

    def end_headers(self):
        p = (self.path or "").split("?")[0]
        if p.endswith((".js", ".html", ".json", "/")):
            self.send_header("Cache-Control", "no-store, must-revalidate")
        elif p.endswith((".bin", ".jpg", ".png")):
            # Scene binaries are the bulk of the bytes and are content-addressed by
            # directory, so let the CDN and the browser keep them.
            self.send_header("Cache-Control", "public, max-age=86400")
        self.send_header("X-Content-Type-Options", "nosniff")
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
        if u.path == "/healthz":
            return self._json(200, {"ok": True, "jobs": active_jobs()})
        if u.path.startswith("/api/job/"):
            job = JOBS.get(u.path.rsplit("/", 1)[-1])
            return self._json(200 if job else 404, job or {"state": "unknown"})
        if u.path == "/api/capabilities":
            cap = max_megapixels(ARGS.quality, ARGS.timeout)
            return self._json(200, {"upload": True, "dem": bool(ARGS.dem),
                                    "quality": ARGS.quality,
                                    "max_upload_mb": MAX_BYTES // (1024 * 1024),
                                    # Pixels, not bytes, are what actually bound a job.
                                    # The client can warn before spending the upload.
                                    "max_megapixels": round(cap, 1),
                                    "max_side_px": int((cap * 1e6) ** 0.5),
                                    "sec_per_megapixel": SEC_PER_MPX})
        return super().do_GET()

    def do_POST(self):
        u = urlparse(self.path)
        if u.path != "/api/upload":
            return self._json(404, {"error": "no such endpoint"})

        q = parse_qs(u.query)
        name = (q.get("name") or ["image.tif"])[0]
        quality = (q.get("quality") or [ARGS.quality])[0]
        if quality not in ("fast", "accurate"):
            quality = ARGS.quality

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

        now = time.time()
        ip = self.client_ip()
        with STATE_LOCK:
            reap_jobs(now)
            if active_jobs() >= MAX_QUEUE:
                return self._json(429, {"error": "The queue is full -- this runs on a "
                                                 "small CPU box and takes a minute per "
                                                 "image. Try again shortly."})
            wait = IP_SPACING - (now - LAST_BY_IP.get(ip, 0.0))
            if wait > 0:
                return self._json(429, {"error": f"One image at a time, please. "
                                                 f"{wait:.0f}s to go."})
            LAST_BY_IP[ip] = now
            job_id = uuid.uuid4().hex
            JOBS[job_id] = {"state": "queued", "step": "waiting for a worker", "pct": 2,
                            "name": name, "started": now, "quality": quality}
            JOB_ORDER.append(job_id)

        stem = safe_stem(name)
        dest_dir = UPLOADS / job_id
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / f"{stem}{ext}"

        # Stream to disk: these are satellite rasters, and a 50 MB read on the request
        # thread is a bad way to discover that.
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
            with STATE_LOCK:
                JOBS.pop(job_id, None)
            return self._json(400, {"error": "upload ended early"})

        # Only now can the real cost be known: bytes were never the constraint, pixels
        # are. Refuse here, before a worker is committed, so the answer arrives in the
        # same second as the upload rather than at the timeout.
        def reject(code: int, msg: str):
            shutil.rmtree(dest_dir, ignore_errors=True)
            with STATE_LOCK:
                JOBS.pop(job_id, None)
                try:
                    JOB_ORDER.remove(job_id)
                except ValueError:
                    pass
                LAST_BY_IP.pop(ip, None)     # a refusal must not cost them their turn
            return self._json(code, {"error": msg})

        try:
            iw, ih = image_shape(dest)
        except Exception:
            # The exception class name used to be shown. It told the visitor nothing and
            # named an internal, which is the same leak class as a traceback.
            return reject(400, "That file could not be read as an image. If it is a "
                               "GeoTIFF it may be damaged, or still downloading.")

        try:
            verify_decodes(dest)
        except ValueError as exc:
            return reject(400, str(exc))

        mpx = (iw * ih) / 1e6
        est = estimate_seconds(mpx, quality)
        cap = max_megapixels(quality, ARGS.timeout)
        if mpx > cap:
            side = int((cap * 1e6) ** 0.5)
            return reject(413,
                          f"{iw}x{ih} is {mpx:.0f} megapixels, which would take about "
                          f"{human_time(est)} on this CPU box -- past the "
                          f"{human_time(ARGS.timeout)} limit. Crop it to about "
                          f"{side}x{side} ({cap:.0f} Mpx) or smaller and try again. "
                          f"Satellite strips are often huge even when the file is small.")

        with STATE_LOCK:
            if job_id in JOBS:
                JOBS[job_id].update(width=iw, height=ih, megapixels=round(mpx, 2),
                                    eta_seconds=round(est))

        threading.Thread(target=run_job, args=(job_id, dest, stem, quality),
                         daemon=True).start()
        return self._json(202, {"job": job_id, "width": iw, "height": ih,
                                "megapixels": round(mpx, 2), "eta_seconds": round(est)})


class Server(ThreadingHTTPServer):
    allow_reuse_address = True     # a container restarts into its own port; TIME_WAIT here is noise
    daemon_threads = True


def main():
    global ARGS
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8080")))
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--ckpt", default=os.environ.get("DW_CKPT", str(ROOT / "weights" / "best.pt")))
    ap.add_argument("--dem", action="store_true",
                    default=os.environ.get("DW_DEM", "") == "1",
                    help="anchor to Copernicus GLO-30 at request time (needs egress)")
    ap.add_argument("--quality", default=os.environ.get("DW_QUALITY", "fast"),
                    choices=["fast", "accurate"],
                    help="default when the client does not ask; 5 s vs 57 s per tile on CPU")
    ap.add_argument("--timeout", type=float,
                    default=float(os.environ.get("DW_TIMEOUT", "600")),
                    help="seconds before a job is abandoned")
    ARGS = ap.parse_args()

    # Anchor the model cache to wherever this file actually is, never to an absolute path
    # in configuration. App Service builds with Oryx and then runs the app from an
    # extracted copy under /tmp/<hash>/ rather than from /home/site/wwwroot, and that hash
    # changes between deployments -- so DW_CKPT=/home/site/wwwroot/... pointed at nothing
    # and every upload died on a missing checkpoint while the site itself looked healthy.
    # ROOT is derived from __file__, so it is right in every host, container and checkout.
    baked_hf = ROOT / "hf"
    if baked_hf.is_dir():
        os.environ["HF_HOME"] = str(baked_hf)
        os.environ.setdefault("HF_HUB_OFFLINE", "1")   # the cache is complete or we want to know

    if SCENES.is_dir():
        BAKED.update(d.name for d in SCENES.iterdir() if d.is_dir())
    UPLOADS.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)

    have_ckpt = Path(ARGS.ckpt).exists() if ARGS.ckpt else False
    if not have_ckpt:
        # Say it loudly at boot rather than at the first upload. A viewer that serves its
        # baked scenes perfectly and fails every upload looks like a bug in the upload
        # code, not a missing file.
        print(f"  !! checkpoint not found: {ARGS.ckpt}\n"
              f"  !! baked scenes will serve; every upload will fail.", flush=True)

    srv = Server((ARGS.host, ARGS.port), Handler)
    print(f"\n  DepthWizard — public\n  --------------------")
    print(f"  listening on {ARGS.host}:{ARGS.port}")
    print(f"  checkpoint : {ARGS.ckpt}{'' if have_ckpt else '   (MISSING)'}")
    print(f"  quality    : {ARGS.quality}   dem: {'on' if ARGS.dem else 'off'}")
    print(f"  limits     : {MAX_BYTES//1024//1024} MB upload, {MAX_QUEUE} in flight, "
          f"{SCENE_BUDGET} scenes kept")
    print(f"  baked      : {len(BAKED)} scenes\n", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n  stopped.")


if __name__ == "__main__":
    main()
