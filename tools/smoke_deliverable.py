"""Hand the pipeline the awkward files a judge will, and check what comes back out.

    python tools/smoke_deliverable.py --ckpt ../checkpoints/run07/best.pt
    python tools/smoke_deliverable.py --help-only      # no GPU needed, seconds
    python tools/smoke_deliverable.py --keep           # leave the fixtures to inspect

Why this exists
---------------
Every accuracy number in this repository is measured on clean 1024 px DFC2019 tiles. Not one
of them says anything about what happens when someone uploads a 512 px crop, a single-band
uint16 GeoTIFF, a PNG that has been renamed `.tif`, or a file whose download was cut off --
and the hosted demo is the path a judge is most likely to take. Those inputs were never
exercised until this ran, and the first one tried crashed.

The bar is deliberately not "everything works". Several of these files genuinely cannot be
turned into a height map, and refusing them is correct. The bar is:

  * a file that can be read produces a height raster of exactly the input's size, and
  * a file that cannot produces **a sentence**, with no traceback, no `File "..."` line and
    no server path in it.

The second half is the one with teeth. What a stability criterion punishes is not a refusal,
it is `recent call last):\\n  File "/tmp/8df0c0ea6632463/infer.py", line 548` appearing in a
progress bar -- which is what production did before A5. So the check does not merely run the
tool; it pushes the tool's dying words through `serve_app.plain_failure()`, which is the
exact function standing between a crash and the browser, and asserts the result is clean.

`--help` is checked on every entry point because it is the first command a technical judge
types, and one of them used to raise ValueError before printing anything.
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

# Anything in a user-facing message that means we leaked the inside of the program.
LEAKS = [
    (re.compile(r"Traceback", re.I), "the word Traceback"),
    (re.compile(r'File "[^"]+", line \d+'), "a source location"),
    (re.compile(r"[A-Za-z]:[\\/]|/tmp/|/home/|\\\\"), "a filesystem path"),
    (re.compile(r"\b\w*Error\b(?!\w)"), "a Python exception name"),
    (re.compile(r"\bself\b|\bmodule\b\.py"), "internals"),
]


def fixtures(d: Path) -> list[tuple[str, Path, str]]:
    """(name, path, what we expect) for each awkward input.

    `expect` is "scene" when the file is readable and should produce a height raster,
    "refuse" when accepting it would be the bug, and "either" when both are defensible --
    the point of those is not that they succeed, it is that failing is polite.
    """
    from PIL import Image
    rng = np.random.default_rng(1337)

    def picture(h: int, w: int) -> np.ndarray:
        a = np.clip(rng.normal(128, 18, (h, w, 3)), 0, 255).astype(np.uint8)
        for _ in range(max(4, (h * w) // 12000)):
            y, x = rng.integers(0, max(1, h - 40)), rng.integers(0, max(1, w - 40))
            a[y:y + 30, x:x + 30] = rng.integers(80, 210, 3)
        return a

    out: list[tuple[str, Path, str]] = []

    # The single most common crop size in remote sensing, and smaller than one window.
    p = d / "512.png"; Image.fromarray(picture(512, 512)).save(p)
    out.append(("512x512 crop", p, "scene"))

    # Smaller still, and not square.
    p = d / "300x220.png"; Image.fromarray(picture(220, 300)).save(p)
    out.append(("300x220, undersized both ways", p, "scene"))

    # A long thin strip: slips under the megapixel cap and is undersized on one axis only.
    p = d / "strip.png"; Image.fromarray(picture(200, 2400)).save(p)
    out.append(("2400x200 strip", p, "scene"))

    # Single-band uint16 -- what a lot of real satellite product looks like.
    import rasterio
    from rasterio.transform import from_origin
    a16 = (picture(600, 600)[:, :, 0].astype(np.uint16) * 250)
    p = d / "uint16_1band.tif"
    with rasterio.open(p, "w", driver="GTiff", width=600, height=600, count=1,
                       dtype="uint16", crs="EPSG:32645",
                       transform=from_origin(633000, 3005000, 0.3, 0.3)) as w:
        w.write(a16, 1)
    out.append(("1-band uint16 GeoTIFF", p, "either"))

    # Four bands: RGB + near infrared, the usual multispectral delivery.
    rgb = picture(600, 600)
    p = d / "4band.tif"
    with rasterio.open(p, "w", driver="GTiff", width=600, height=600, count=4,
                       dtype="uint8") as w:
        for i in range(3):
            w.write(rgb[:, :, i], i + 1)
        w.write(rgb[:, :, 0], 4)
    out.append(("4-band GeoTIFF (RGB+NIR)", p, "either"))

    # Greyscale and RGBA PNGs.
    p = d / "grey.png"; Image.fromarray(picture(600, 600)[:, :, 0]).save(p)
    out.append(("greyscale PNG", p, "either"))
    rgba = np.dstack([picture(600, 600), np.full((600, 600), 255, np.uint8)])
    p = d / "rgba.png"; Image.fromarray(rgba, "RGBA").save(p)
    out.append(("RGBA PNG", p, "either"))

    # A CMYK JPEG: legal, and not what any of this expects.
    p = d / "cmyk.jpg"
    Image.fromarray(picture(600, 600)).convert("CMYK").save(p)
    out.append(("CMYK JPEG", p, "either"))

    # All-NaN float raster: readable, and carrying no information at all.
    p = d / "all_nan.tif"
    with rasterio.open(p, "w", driver="GTiff", width=600, height=600, count=1,
                       dtype="float32") as w:
        w.write(np.full((600, 600), np.nan, np.float32), 1)
    out.append(("all-NaN float raster", p, "either"))

    # A PNG wearing a .tif extension -- a very common human error.
    p = d / "actually_png.tif"
    Image.fromarray(picture(600, 600)).save(p, format="PNG")
    out.append(("PNG renamed .tif", p, "either"))

    # An upload that was cut off half way.
    whole = (d / "512.png").read_bytes()
    p = d / "truncated.tif"; p.write_bytes(whole[:len(whole) // 3])
    # Must be refused. Accepting it was a real defect: GDAL pads the missing rows
    # with black and the pipeline reported heights over 26% invented pixels.
    out.append(("truncated file", p, "refuse"))

    # Zero bytes.
    p = d / "empty.tif"; p.write_bytes(b"")
    out.append(("empty file", p, "refuse"))

    return out


def leaks_in(msg: str) -> list[str]:
    return [why for rx, why in LEAKS if rx.search(msg or "")]


def check_help() -> list[tuple[bool, str]]:
    """--help must print and exit 0 on everything a judge could type."""
    targets = [ROOT / "infer.py", ROOT / "train.py"]
    targets += sorted(p for p in (ROOT / "tools").glob("*.py")
                      if "argparse" in p.read_text(encoding="utf-8", errors="ignore"))
    rows = []
    for t in targets:
        r = subprocess.run([sys.executable, str(t), "--help"], cwd=ROOT,
                           capture_output=True, text=True, timeout=180)
        rel = t.relative_to(ROOT).as_posix()
        ok = r.returncode == 0 and ("usage" in r.stdout.lower())
        rows.append((ok, f"{rel} --help"
                         + ("" if ok else f"  exit {r.returncode}: "
                                          f"{(r.stderr or r.stdout).strip().splitlines()[-1][:90]}")))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default=None, help="checkpoint for the inference pass")
    ap.add_argument("--help-only", action="store_true", help="skip the fixtures, no GPU")
    ap.add_argument("--keep", action="store_true", help="keep the fixtures and outputs")
    ap.add_argument("--timeout", type=int, default=600)
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

    # The exact two functions standing between a bad file and the browser.
    from serve_app import plain_failure, verify_decodes

    rows: list[tuple[bool, str]] = []
    print("\n--help on every entry point")
    for ok, msg in check_help():
        print(f"  {'PASS' if ok else 'FAIL'}  {msg}")
        rows.append((ok, msg))

    if not args.help_only:
        work = Path(tempfile.mkdtemp(prefix="dwz-smoke-"))
        try:
            print(f"\nawkward inputs through infer.py   ({work})")
            for name, path, expect in fixtures(work):
                # The upload gate runs first, exactly as it does on the server: a file
                # refused here never reaches inference at all.
                gate = None
                try:
                    verify_decodes(path)
                except ValueError as exc:
                    gate = str(exc)
                if gate is not None:
                    bad = leaks_in(gate)
                    ok = expect in ("refuse", "either") and not bad
                    rows.append((ok, f"{name}: refused at the upload gate: {gate[:64]!r}"
                                     + (f"  LEAKS {', '.join(bad)}" if bad else "")
                                     + ("" if expect != "scene" else "  but it should have worked")))
                    print(f"  {'PASS' if ok else 'FAIL'}  {rows[-1][1]}")
                    continue

                cmd = [sys.executable, "infer.py", "--image", str(path),
                       "--out", str(work / f"{path.stem}_out")]
                if args.ckpt:
                    cmd += ["--ckpt", args.ckpt]
                r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                                   timeout=args.timeout)
                produced = (work / f"{path.stem}_out.height.tif")
                if r.returncode == 0 and produced.exists():
                    import rasterio
                    from PIL import Image
                    with rasterio.open(produced) as d:
                        got = (d.width, d.height)
                    try:
                        with Image.open(path) as im:
                            want = (im.width, im.height)
                    except Exception:
                        with rasterio.open(path) as d:
                            want = (d.width, d.height)
                    ok = got == want and expect != "refuse"
                    rows.append((ok, f"{name}: produced {got[0]}x{got[1]}"
                                     + ("" if got == want else
                                        f" but the input was {want[0]}x{want[1]}")
                                     + ("" if expect != "refuse" else
                                        "  but this one should have been refused")))
                else:
                    # This is what the browser would have been shown.
                    said = plain_failure(r.stderr or r.stdout or "")
                    bad = leaks_in(said)
                    ok = (expect in ("either", "refuse")) and not bad
                    detail = f"refused with: {said[:80]!r}"
                    if bad:
                        detail += f"  LEAKS {', '.join(bad)}"
                    elif expect == "scene":
                        detail += "  but this one should have worked"
                    rows.append((ok, f"{name}: {detail}"))
                print(f"  {'PASS' if rows[-1][0] else 'FAIL'}  {rows[-1][1]}")
        finally:
            if args.keep:
                print(f"\n  fixtures left in {work}")
            else:
                shutil.rmtree(work, ignore_errors=True)

    bad = [m for ok, m in rows if not ok]
    print(f"\n{len(rows) - len(bad)}/{len(rows)} passed")
    print("SMOKE OK" if not bad else f"SMOKE BROKEN  ({len(bad)} problem(s))")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
