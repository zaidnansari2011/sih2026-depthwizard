"""Can shadow geometry recover the tall buildings the network cannot? (v2)

Every lever inside the model is now measured and spent: bigger backbone (void run), head
architecture (regression / bins / bins+HTC all land at slope 0.43-0.49), decode (argmax sits
+1.86 m from the expectation, there is no tall mode), ensembling (error correlation 0.925),
and loss reweighting (pixels above 20 m already carry 78.78% of the squared error, so LDS's
premise does not hold). The model assigns 91% of its probability mass above 20 m on a tall
building and still cannot separate 26 m from 76 m. A nadir view of a rooftop does not carry
that information.

Shadow is a different, physical cue:

    height = shadow_length * tan(sun_elevation)

and its error profile is the inverse of the network's. At 0.31 m GSD with the sun at 24 deg,
a 38 m building casts ~270 px of shadow, where a couple of pixels of edge error is tens of
centimetres of height. The same geometry on a 4 m building at high sun gives three pixels
and is useless -- which is fine, because the network is already at 1.45 m there.

v1 measured the cue and failed to extract it (overall r 0.028). Two causes, both fixed here:

  1. A p22 luminance cut landed INSIDE the shadow. Measured on OMA_288_012: shadow sits at
     luminance 93-113, lit ground at 187-255, and p22 was 111. Rays died halfway down their
     own shadow and every building came back ~2 m tall. Otsu finds the valley instead.

  2. Independent per-pixel rays are brittle. One ray meets a dark roof, a car, a road, or a
     gap in the shadow and terminates, and the median over rays inherits that. v2 marches a
     CORRIDOR: for each distance along the sun axis it asks what FRACTION of the building's
     width is still in shadow, and takes the run while that fraction holds. A few stray
     pixels no longer end the measurement, and a dark road crossing the corridor does not
     extend it either, because the road does not span the building's width at every step.

Buildings come from the truth AGL, deliberately. The question is whether the shadow CUE
carries height, not whether we can segment footprints; mixing the two would measure the
wrong thing. Occlusion is detected the same way -- a shadow falling on another structure is
flagged rather than silently truncating the answer.

    python tools/shadow_probe.py --tiles 24
"""
from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path("D:/sih2026")
sys.path.insert(0, str(ROOT / "depthwizard"))
sys.path.insert(0, str(ROOT / "depthwizard" / "tools"))

from parse_imd import parse_imd  # noqa: E402

RGB_DIR = ROOT / "data" / "extracted" / "Track1-RGB"
AGL_DIR = ROOT / "data" / "extracted" / "Track1-Truth"
METADATA = ROOT / "data" / "extracted" / "Track3-Metadata" / "Track3-Metadata"
TILE_RE = re.compile(r"^([A-Z]+)_(\d+)_(\d+)$")

GSD = 0.31
TALL = 20.0
BANDS = [(0, 3), (3, 6), (6, 10), (10, 20), (20, 40), (40, 1e9)]
OCCUPANCY = 0.45      # fraction of the corridor that must still be shadow to continue
MIN_WIDTH = 8         # corridor narrower than this cannot be measured reliably


def band_label(lo, hi):
    return f"{lo:g}-{hi:g} m" if hi < 1e8 else f"{lo:g} m+"


def load_scenes() -> dict:
    """(site, scene index) -> parsed .IMD. The join tools/view_angle.py verified."""
    scenes = {}
    for p in sorted(METADATA.rglob("*.IMD")):
        try:
            idx = int(p.stem)
        except ValueError:
            continue
        scenes[(p.parent.name.upper(), idx)] = parse_imd(p.read_text(errors="replace"))
    return scenes


def shadow_mask(rgb: np.ndarray) -> np.ndarray:
    """Otsu on luminance: find the valley between the shadow and lit modes.

    A fixed percentile cannot work across sun elevations from 23 to 74 degrees, and v1's
    p22 landed inside the shadow. Otsu makes no assumption about what fraction of a tile is
    shaded, which varies with both sun angle and how built-up the scene is.
    """
    lum = rgb.astype(np.float32).mean(axis=2)
    try:
        from skimage.filters import threshold_otsu
        thr = float(threshold_otsu(lum))
    except ImportError:
        thr = float(np.percentile(lum, 45))
    return lum < thr, thr


def measure_corridor(mask_b, shade, other_bld, drow, dcol, max_steps):
    """Shadow run for one building, marched PER COLUMN across its width.

    The bug this fixes, found 28 Aug: v2 used a single `s_far = s.max()` -- one far corner
    of the building -- as the start line for the whole corridor. For anything not square-on
    to the sun, the corridor one step past that corner is mostly beside the building rather
    than behind it, occupancy fails on step 1, and the building is dropped. It silently
    discarded exactly the large, irregular footprints that tall buildings have: a run over
    the ten tiles with the most tall stock in the dataset measured ZERO buildings above
    10 m.

    Each column across the building's width now starts at its OWN far edge, which is what
    "the shadow begins at the wall" actually means.
    """
    rows, cols = np.nonzero(mask_b)
    if len(rows) < 60:
        return None
    s = rows * drow + cols * dcol
    p = -rows * dcol + cols * drow
    p_int = np.round(p).astype(int)
    # Far edge of the building per column across the sun axis.
    order = np.argsort(p_int)
    p_sorted, s_sorted = p_int[order], s[order]
    uniq, starts = np.unique(p_sorted, return_index=True)
    s_far = np.maximum.reduceat(s_sorted, starts)
    # Drop thin slivers at the extremes: one stray pixel is not a column.
    counts = np.diff(np.append(starts, len(p_sorted)))
    keep = counts >= 3
    uniq, s_far = uniq[keep], s_far[keep]
    if len(uniq) < MIN_WIDTH:
        return None

    H, W = mask_b.shape
    occluded, length = False, 0
    for step in range(1, max_steps + 1):
        ss = s_far + step
        rr = np.round(ss * drow - uniq * dcol).astype(int)
        cc = np.round(ss * dcol + uniq * drow).astype(int)
        inside = (rr >= 0) & (rr < H) & (cc >= 0) & (cc < W)
        if inside.sum() < MIN_WIDTH:
            break
        rr, cc = rr[inside], cc[inside]
        # Ignore samples that landed back on this building (concave footprints).
        own = mask_b[rr, cc]
        rr, cc = rr[~own], cc[~own]
        if len(rr) < MIN_WIDTH:
            continue
        if other_bld[rr, cc].mean() > 0.5:
            occluded = True
            break
        if shade[rr, cc].mean() < OCCUPANCY:
            break
        length = step
    return (length, occluded) if length else None
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles", type=int, default=24)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--prefer-tall", action="store_true", default=True,
                    help="prioritise tiles containing tall buildings; they are rare and "
                         "are the only place this cue has to prove itself")
    args = ap.parse_args()

    try:
        import rasterio
        from scipy import ndimage
    except ImportError as e:
        raise SystemExit(f"needs rasterio and scipy: {e}")

    scenes = load_scenes()
    names = sorted(p.stem[:-4] for p in RGB_DIR.glob("*_RGB.tif"))

    # Rank tiles by how much tall stock they hold, so the sample can say something about
    # the band that matters instead of spending itself on 4 m sheds.
    ranked = []
    for t in names:
        m = TILE_RE.match(t)
        if not m or not (AGL_DIR / f"{t}_AGL.tif").exists():
            continue
        if (m.group(1).upper(), int(m.group(3))) not in scenes:
            continue
        ranked.append(t)
    rng = np.random.default_rng(args.seed)
    rng.shuffle(ranked)

    OUT, TRU, SUN, OCC = [], [], [], []
    used = 0
    for t in ranked:
        if used >= args.tiles:
            break
        m = TILE_RE.match(t)
        sc = scenes[(m.group(1).upper(), int(m.group(3)))]
        if "meanSunEl" not in sc or "meanSunAz" not in sc:
            continue
        with rasterio.open(AGL_DIR / f"{t}_AGL.tif") as ds:
            agl = ds.read(1).astype(np.float32)
        if args.prefer_tall and np.nanmax(agl) < TALL:
            continue
        with rasterio.open(RGB_DIR / f"{t}_RGB.tif") as ds:
            rgb = np.transpose(ds.read()[:3], (1, 2, 0))

        el, az = float(sc["meanSunEl"]), float(sc["meanSunAz"])
        a = math.radians(az)
        drow, dcol = math.cos(a), -math.sin(a)      # anti-sun, verified empirically in v1
        shade, thr = shadow_mask(rgb)
        k = math.tan(math.radians(el))
        max_steps = int(130.0 / (k * GSD)) + 10

        blds, n = ndimage.label(np.isfinite(agl) & (agl > 2.0))
        if n == 0:
            continue
        objs = ndimage.find_objects(blds)
        got = 0
        for lab in range(1, n + 1):
            sl = objs[lab - 1]
            if sl is None:
                continue
            mb = blds == lab
            truth = float(np.nanmedian(agl[mb]))
            if not np.isfinite(truth):
                continue
            res = measure_corridor(mb, shade, (blds > 0) & ~mb, drow, dcol, max_steps)
            if res is None:
                continue
            run, occ = res
            OUT.append(run * GSD * k)
            TRU.append(truth)
            SUN.append(el)
            OCC.append(occ)
            got += 1
        used += 1
        print(f"  {t}: sun el {el:.1f} az {az:.1f}  otsu {thr:.0f}  -> {got} buildings")

    if not OUT:
        raise SystemExit("no buildings measured")
    o, t_, s, occ = (np.array(OUT), np.array(TRU), np.array(SUN), np.array(OCC, bool))
    print(f"\n{len(o):,} buildings measured, {occ.sum()} flagged occluded "
          f"({100*occ.mean():.1f}%), sun elevation {s.min():.1f}-{s.max():.1f} deg")

    for label, keep in (("ALL", np.ones_like(occ)), ("UNOCCLUDED ONLY", ~occ)):
        if keep.sum() < 10:
            continue
        oo, tt = o[keep], t_[keep]
        print(f"\n== {label}: {int(keep.sum()):,} buildings ==")
        print(f"  {'band':>10} {'n':>6} {'truth':>8} {'shadow':>9} {'RMSE':>9} {'r':>7}")
        for lo, hi in BANDS:
            m = (tt >= lo) & (tt < hi)
            if m.sum() < 3:
                continue
            rmse = float(np.sqrt(np.mean((oo[m] - tt[m]) ** 2)))
            r = float(np.corrcoef(oo[m], tt[m])[0, 1])
            print(f"  {band_label(lo, hi):>10} {int(m.sum()):>6} {tt[m].mean():>7.1f}m "
                  f"{oo[m].mean():>8.1f}m {rmse:>8.2f}m {r:>7.3f}")
        print(f"  overall r {np.corrcoef(oo, tt)[0, 1]:.3f}  "
              f"slope {np.polyfit(tt, oo, 1)[0]:.3f}")
        tall = tt >= TALL
        if tall.sum() > 2:
            print(f"  ABOVE {TALL:g} m (n={int(tall.sum())}): "
                  f"r {np.corrcoef(oo[tall], tt[tall])[0, 1]:.3f}, "
                  f"shadow {oo[tall].mean():.1f} m vs truth {tt[tall].mean():.1f} m "
                  f"| the NETWORK predicts ~26 m here")


if __name__ == "__main__":
    main()
