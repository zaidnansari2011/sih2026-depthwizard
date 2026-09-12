"""Upload two images to a real server and check what a judge gets back.

    python tools/verify_upload.py --ckpt ../checkpoints/run07/best.pt

Why this exists
---------------
The problem statement asks the platform to "output a high-fidelity DSM in a standard
geospatial format", and the hosted demo is the path a judge is most likely to take. Until
A6 the rasters were written and then reachable by no route at all, so that path produced a
picture and nothing to open in QGIS. This drives the whole thing the way a browser does --
start the server, POST a file, poll the job, follow the download links -- because each of
those pieces worked in isolation before and the route between them did not exist.

Four things it is really checking, none of which a unit test would catch:

  * the georeference claim. The viewer says "heights are above sea level" on the strength
    of one flag, and that flag used to be set by sniffing stdout for a string that prints
    for any georeferenced *input* -- whether or not a DSM was ever anchored. So an upload
    with no DEM anchor was told its heights were above sea level. The flag must now follow
    the file on disk.
  * the downloads exist, are named in the problem statement's vocabulary, and arrive as
    attachments. The above-ground raster is an nDSM when there is an absolute DSM beside
    it and the PS's rDSM when there is not, and the filename has to say which.
  * nothing served leaks a server path. The run summary records the input image and the
    checkpoint by absolute path, which on App Service means /tmp/<job>/... -- the same
    leak as a traceback, just quieter.
  * an upload does not join the shared scene picker. viewer/scenes/index.json is served to
    everyone, so a scene listed there appears for every later visitor.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
# A drive letter or a unix system path, but not a URL scheme: "https://" would
# otherwise match [A-Za-z]:[/] and flag the repository link in the readme.
LEAK = re.compile(r"(?<!http)(?<!https)[A-Za-z]:[\\/]|/tmp/|/home/|Traceback"
                  r"|File \"[^\"]+\", line \d+")


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def make_fixtures(d: Path) -> list[tuple[str, Path, bool]]:
    """(label, path, is_georeferenced)."""
    from PIL import Image
    import rasterio
    from rasterio.transform import from_origin

    rng = np.random.default_rng(11)
    a = np.clip(rng.normal(130, 20, (560, 560, 3)), 0, 255).astype(np.uint8)
    for _ in range(30):
        y, x = rng.integers(20, 500, 2)
        a[y:y + 34, x:x + 34] = rng.integers(80, 210, 3)

    plain = d / "plain_crop.png"
    Image.fromarray(a).save(plain)

    geo = d / "geo_crop.tif"
    with rasterio.open(geo, "w", driver="GTiff", width=560, height=560, count=3,
                       dtype="uint8", crs="EPSG:32645",
                       transform=from_origin(633000, 3005000, 0.3, 0.3)) as w:
        for i in range(3):
            w.write(a[:, :, i], i + 1)
    return [("plain PNG, no CRS", plain, False), ("GeoTIFF with a CRS", geo, True)]


def post(url: str, blob: bytes) -> dict:
    """POST, waiting out the per-IP spacing rather than tripping over it.

    The server allows one job per IP per DW_IP_SPACING seconds (20 by default), which is
    a real defence and not something to switch off for a test -- so the test waits, and
    in doing so also proves the limiter answers with a sentence rather than a stack.
    """
    for attempt in range(14):
        req = urllib.request.Request(url, data=blob, method="POST")
        req.add_header("Content-Type", "application/octet-stream")
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            # Reading the error body can itself abort: the server closes the connection
            # as soon as it has written a short refusal, and a large request body is
            # still in flight towards it.
            try:
                body = json.loads(e.read() or b"{}")
            except Exception:
                body = {}
            if e.code == 429 and attempt < 13:
                time.sleep(6)
                continue
            return {"error": body.get("error", f"HTTP {e.code}"), "code": e.code}
        except (urllib.error.URLError, ConnectionError, OSError) as e:
            if attempt < 13:
                time.sleep(6)
                continue
            return {"error": f"connection failed: {type(e).__name__}"}
    return {"error": "gave up waiting for a slot"}


def get_json(url: str) -> dict:
    try:
        with urllib.request.urlopen(url, timeout=60) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        return json.loads(e.read())


def get_file(url: str) -> tuple[int, bytes, str]:
    try:
        with urllib.request.urlopen(url, timeout=120) as r:
            return r.status, r.read(), r.headers.get("Content-Disposition", "")
    except urllib.error.HTTPError as e:
        return e.code, e.read(), ""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--timeout", type=int, default=900)
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

    work = Path(tempfile.mkdtemp(prefix="dwz-upload-"))
    port = free_port()
    env = dict(os.environ, DW_WORK=str(work / "server"))
    cmd = [sys.executable, "tools/serve_app.py", "--port", str(port), "--quality", "fast"]
    if args.ckpt:
        cmd += ["--ckpt", args.ckpt]

    # The scene directory is shared with the checked-in demo scenes, so remember what is
    # in the index before any upload and put it back afterwards.
    index = ROOT / "viewer" / "scenes" / "index.json"
    index_before = index.read_bytes() if index.exists() else None
    made: list[Path] = []

    print(f"\nupload check  http://127.0.0.1:{port}   (work dir {work})")
    proc = subprocess.Popen(cmd, cwd=ROOT, env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    checks: list[tuple[bool, str]] = []
    try:
        base = f"http://127.0.0.1:{port}"
        for _ in range(120):
            try:
                if get_json(f"{base}/healthz").get("ok"):
                    break
            except Exception:
                time.sleep(0.5)
        else:
            raise SystemExit("server never became healthy")

        for label, path, want_geo in make_fixtures(work):
            print(f"\n  {label}")
            up = post(f"{base}/api/upload?name={path.name}&quality=fast", path.read_bytes())
            job = up.get("job")
            if not job:
                checks.append((False, f"{label}: upload refused: {up}"))
                continue

            state = {}
            deadline = time.time() + args.timeout
            while time.time() < deadline:
                state = get_json(f"{base}/api/job/{job}")
                if state.get("state") in ("done", "error", "unknown"):
                    break
                time.sleep(1.0)
            if state.get("state") != "done":
                checks.append((False, f"{label}: job ended {state.get('state')!r}: "
                                      f"{state.get('step')!r}"))
                continue
            made.append(ROOT / "viewer" / "scenes" / state["scene"])
            checks.append((True, f"{label}: job completed, scene {state['scene']!r}"))

            # 1. a georeferenced input should now come back with the problem
            # statement's absolute DSM, because infer.py anchors by default. "Should",
            # not "must": the anchor is fetched from Copernicus over the network, and a
            # host with no egress is expected to fall back to the relative products --
            # so the test accepts that only when the summary says so in as many words.
            dsm_offered = "dsm" in (state.get("results") or [])
            if want_geo and not dsm_offered:
                code, body, _ = get_file(f"{base}/api/result/{job}/summary")
                said = json.loads(body or b"{}").get("absolute_dsm", {})
                excused = said.get("anchored") is False and bool(said.get("reason"))
                checks.append((excused,
                               "GeoTIFF: no absolute DSM, and the summary "
                               + (f"explains why: {said.get('reason')!r}" if excused
                                  else "does not say why -- a georeferenced input should "
                                       "anchor")))
            elif want_geo:
                checks.append((True, "GeoTIFF: absolute DSM produced, as the PS asks for "
                                     "georeferenced input"))

            # 2. and the "above sea level" claim must follow the file, not the input's CRS
            got_geo = bool(state.get("georeferenced"))
            checks.append((got_geo == dsm_offered,
                           f"{label}: 'above sea level' is claimed only when an absolute DSM "
                           f"exists (claim {got_geo}, DSM offered {dsm_offered})"))

            # 2. the downloads
            kinds = state.get("results") or []
            checks.append(("ndsm" in kinds and "readme" in kinds,
                           f"{label}: offers {kinds}"))
            for kind in kinds:
                code, body, disp = get_file(f"{base}/api/result/{job}/{kind}")
                name = re.search(r'filename="([^"]+)"', disp)
                ok = code == 200 and len(body) > 0 and "attachment" in disp and name
                checks.append((bool(ok),
                               f"{label}: {kind} -> {code}, {len(body)} bytes, "
                               f"{name.group(1) if name else 'NO FILENAME'}"))
                if kind == "ndsm" and name:
                    want = "_ndsm.tif" if want_geo and dsm_offered else "_rdsm.tif"
                    checks.append((name.group(1).endswith(want),
                                   f"{label}: the above-ground raster is named "
                                   f"{name.group(1)!r} (expected to end {want!r})"))
                if kind in ("summary", "readme"):
                    text = body.decode("utf-8", "replace")
                    hit = LEAK.search(text)
                    checks.append((hit is None,
                                   f"{label}: {kind} carries no server path"
                                   + ("" if hit is None else f" -- LEAKS {hit.group(0)!r}")))

            # 3. a refused kind must not be a way to name a file
            code, _, _ = get_file(f"{base}/api/result/{job}/../../../etc/passwd")
            checks.append((code in (400, 404), f"{label}: path traversal refused ({code})"))

        # 4. no upload may join the shared picker
        after = json.loads(index.read_text(encoding="utf-8")) if index.exists() else []
        strays = [k.get("dir") for k in after if str(k.get("dir", "")).startswith("upload_")]
        checks.append((not strays,
                       "uploads stay out of viewer/scenes/index.json"
                       + ("" if not strays else f" -- but found {strays}")))
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()
        for d in made:
            shutil.rmtree(d, ignore_errors=True)
        if index_before is not None:
            index.write_bytes(index_before)
        shutil.rmtree(work, ignore_errors=True)

    ok = True
    print()
    for passed, msg in checks:
        print(f"  {'PASS' if passed else 'FAIL'}  {msg}")
        ok &= passed
    print(f"\n{'UPLOAD OK' if ok else 'UPLOAD BROKEN'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
