"""Check the nodata detector in depthwizard/rgb.py against real imagery.

    python tools/verify_nodata.py

Three parts, each a claim the detector has to earn:

  A. No regression. On real images with no border, read_rgb returns exactly the pixels the
     previous reader returned (loaded from git HEAD~ for rgb.py, or --baseline), and the
     mask flags nothing. A shadow punched out of a real scene is worse than the slab.
  B. Real borders are found. Footprint edges cut from raw Maxar Open Data tiles, whose own
     per-dataset mask is the truth, written every way a visitor might upload them.
  C. Controls. A black square inside a scene and a short black notch on its edge stay
     valid.

Needs the raw Maxar tiles under D:/sih2026/data/maxar (part B) and the DFC2019 tiles
(part A); parts without their data are reported as skipped, never as passed.
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import random
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from depthwizard.rgb import read_rgb_valid  # noqa: E402

warnings.filterwarnings("ignore")
DATA = Path("D:/sih2026/data")
OUT = ROOT.parent / "out" / "nodata_fixtures"
FAILS: list[str] = []


def check(ok: bool, msg: str) -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {msg}")
    if not ok:
        FAILS.append(msg)


def load_baseline(rev: str):
    """The previous rgb.py, straight from git, as a module."""
    src = subprocess.run(["git", "show", f"{rev}:depthwizard/rgb.py"], cwd=ROOT,
                         capture_output=True, text=True, encoding="utf-8", check=True).stdout
    tmp = Path(tempfile.mkdtemp()) / "rgb_baseline.py"
    tmp.write_text(src, encoding="utf-8")
    spec = importlib.util.spec_from_file_location("rgb_baseline", tmp)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def part_a(baseline, n_dfc: int) -> None:
    print("\nA. real images without a border: same pixels as before, nothing masked")
    random.seed(0)
    dfc = glob.glob(str(DATA / "extracted/**/*_RGB*.tif"), recursive=True)
    files = (glob.glob("D:/sih2026/testkit/*.tif") + glob.glob("D:/sih2026/demo_uploads/*.tif")
             + glob.glob("D:/sih2026/demo_uploads/*.png")
             + glob.glob(str(ROOT / "viewer/samples/*.tif"))
             + glob.glob(str(ROOT / "viewer/samples/*.png"))
             + [f for f in glob.glob(str(ROOT / "viewer/samples/*.jpg")) if "thumb" not in f]
             + random.sample(dfc, min(n_dfc, len(dfc))))
    if not files:
        print("  SKIP  no images found")
        return
    changed, masked = [], []
    for f in files:
        rgb, valid = read_rgb_valid(f)
        if not np.array_equal(rgb, baseline.read_rgb(f)):
            changed.append(Path(f).name)
        if not valid.all():
            masked.append(f"{Path(f).name} ({int((~valid).sum())} px)")
    check(not changed, f"{len(files)} images read identically to the previous reader"
          + (f"; changed: {changed[:5]}" if changed else ""))
    check(not masked, f"no pixel masked on any of them"
          + (f"; masked: {masked[:5]}" if masked else ""))


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    u = (a | b).sum()
    return 1.0 if u == 0 else float((a & b).sum() / u)


def part_b() -> None:
    print("\nB. real Maxar footprint edges, truth = the tile's own mask")
    import rasterio
    from PIL import Image
    from rasterio.windows import Window
    tiles = ["45_120220031230_10300100CF621C00", "44_120220011002_103001003B39FB00"]
    wins = {tiles[0]: (5376, 1280), tiles[1]: (3328, 14592)}
    OUT.mkdir(parents=True, exist_ok=True)
    found = False
    for t in tiles:
        p = glob.glob(str(DATA / f"maxar/*/{t}-visual.tif"))
        if not p:
            continue
        found = True
        x, y = wins[t]
        w = Window(x, y, 1024, 1024)
        with rasterio.open(p[0]) as s:
            a = s.read([1, 2, 3], window=w)                  # 3 x H x W, uint8
            truth = s.dataset_mask(window=w) == 0            # True = invalid
        tag = t[:15]
        prof = dict(driver="GTiff", width=1024, height=1024, count=3, dtype="uint8",
                    compress="deflate")

        def tif(name, arr, **extra):
            f = OUT / f"{tag}_{name}.tif"
            with rasterio.open(f, "w", **{**prof, **extra, "count": arr.shape[0],
                                           "dtype": str(arr.dtype)}) as d:
                d.write(arr)
            return f

        cases = {}
        cases["GeoTIFF, no tags (lossless)"] = tif("plain", a)
        f = OUT / f"{tag}_mask.tif"
        with rasterio.open(f, "w", **prof) as d:
            d.write(a)
            d.write_mask(np.where(truth, 0, 255).astype(np.uint8))
        cases["GeoTIFF, internal mask"] = f
        cases["GeoTIFF, nodata=0 declared"] = tif("nodata0", a, nodata=0)
        a16 = (a.astype(np.uint16) * 16)
        cases["GeoTIFF, 16-bit, no tags"] = tif("u16", a16)
        hwc = np.transpose(a, (1, 2, 0))
        rgba = np.dstack([hwc, np.where(truth, 0, 255).astype(np.uint8)])
        f = OUT / f"{tag}_alpha.png"
        Image.fromarray(rgba, "RGBA").save(f)
        cases["PNG with alpha"] = f
        f = OUT / f"{tag}_plain.png"
        Image.fromarray(hwc).save(f)
        cases["PNG, no alpha"] = f
        f = OUT / f"{tag}_q90.jpg"
        Image.fromarray(hwc).save(f, quality=90)
        cases["JPEG q90"] = f

        print(f"  {t}  window ({x},{y}) 1024 px, {truth.mean()*100:.1f}% truly invalid")
        for name, f in cases.items():
            _, valid = read_rgb_valid(f)
            det = ~valid
            iou = _iou(det, truth)
            # Masking real imagery is the costly error: count it separately.
            over = int((det & ~truth).sum())
            exact = name in ("GeoTIFF, internal mask", "PNG with alpha")
            ok = (iou == 1.0) if exact else (iou >= 0.98 and over <= 0.002 * (~truth).sum())
            check(ok, f"{name:<28} IoU {iou:.4f}, {over} valid px wrongly masked"
                  + ("  (must be exact)" if exact else ""))
    if not found:
        print("  SKIP  raw Maxar tiles not on this machine")


def part_c() -> None:
    print("\nC. controls: black inside the scene, and a short notch on the edge, stay valid")
    from PIL import Image
    src = ROOT / "viewer/samples/terraced_village.jpg"
    if not src.exists():
        print("  SKIP  control image missing")
        return
    rgb = np.array(Image.open(src).convert("RGB"))[:1024, :1024].copy()
    H, W = rgb.shape[:2]
    OUT.mkdir(parents=True, exist_ok=True)
    a = rgb.copy(); a[400:600, 400:600] = 0                     # black roof, interior
    f = OUT / "ctl_interior.png"; Image.fromarray(a).save(f)
    check(read_rgb_valid(f)[1].all(), "200x200 black square inside the scene: nothing masked")
    b = rgb.copy(); b[0:40, 300:360] = 0                        # shadow clipping the edge
    f = OUT / "ctl_notch.png"; Image.fromarray(b).save(f)
    check(read_rgb_valid(f)[1].all(), "60 px black notch on the top edge: nothing masked")
    c = rgb.copy(); c[:, :int(W * 0.2)] = 0                     # a real border, for contrast
    f = OUT / "ctl_border.png"; Image.fromarray(c).save(f)
    v = read_rgb_valid(f)[1]
    check((~v).sum() == H * int(W * 0.2), "a 20% left border: exactly that strip masked")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baseline", default="HEAD",
                    help="git revision holding the reader to compare against")
    ap.add_argument("--dfc", type=int, default=300, help="DFC2019 tiles to sample")
    args = ap.parse_args()
    print(f"nodata detector check  (baseline reader: {args.baseline}:depthwizard/rgb.py)")
    part_a(load_baseline(args.baseline), args.dfc)
    part_b()
    part_c()
    print(f"\n{'NODATA OK' if not FAILS else f'{len(FAILS)} FAILED'}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
