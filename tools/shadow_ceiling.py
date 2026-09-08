"""The upper bound on shadow: what would PERFECT shadow segmentation recover?

Before spending a day on shadow detection, this settles whether it is worth spending at
all. Instead of detecting shadows from pixels, it computes them geometrically from the
truth DSM: march from each pixel toward the sun and ask whether anything along that ray
rises above the solar ray. That is the shadow an oracle detector would return.

Feed those oracle shadows through the same length -> height conversion the real probe uses.
The result is a CEILING:

  * If oracle shadow recovers tall building heights, the cue is sound and everything
    between here and there is a segmentation problem worth solving.
  * If it does not, no amount of work on shadow detection will help, because the geometry
    itself does not deliver the height at this resolution and these sun angles. Then the
    honest move is to stop and say so.

This also produces, for free, a labelled shadow mask that any future detector can be scored
against -- which is the right way to choose a threshold, rather than by whether the answer
comes out flattering.

    python tools/shadow_ceiling.py --tiles 8
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
OCCUPANCY = 0.45
MIN_WIDTH = 8


def band_label(lo, hi):
    return f"{lo:g}-{hi:g} m" if hi < 1e8 else f"{lo:g} m+"


def geometric_shadow(agl: np.ndarray, el_deg: float, az_deg: float) -> np.ndarray:
    """Ray-cast the truth surface toward the sun. True where the sun is blocked.

    Standard horizon shadowing: stepping distance d metres toward the sun, the solar ray
    from a pixel has risen d*tan(elevation). The pixel is shadowed if the surface anywhere
    along that ray is higher than the ray.
    """
    H, W = agl.shape
    a = math.radians(az_deg)
    # Toward the sun: north is -row, east is +col, azimuth clockwise from north.
    dr, dc = -math.cos(a), math.sin(a)
    tan_el = math.tan(math.radians(el_deg))
    surf = np.nan_to_num(agl, nan=0.0)
    shadow = np.zeros((H, W), bool)
    rr, cc = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    # Stop once the ray is above anything in the scene: no occluder can reach it.
    max_d = float(np.nanmax(surf)) / max(tan_el, 1e-6) / GSD
    for step in range(1, int(min(max_d, 900)) + 1):
        r2 = np.round(rr + dr * step).astype(np.int32)
        c2 = np.round(cc + dc * step).astype(np.int32)
        ok = (r2 >= 0) & (r2 < H) & (c2 >= 0) & (c2 < W)
        if not ok.any():
            break
        ray_h = surf + step * GSD * tan_el
        blocked = np.zeros((H, W), bool)
        blocked[ok] = surf[r2[ok], c2[ok]] > ray_h[ok]
        shadow |= blocked
    return shadow


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
    ap.add_argument("--tiles", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--rank-tall", action="store_true", default=True)
    args = ap.parse_args()

    import rasterio
    from scipy import ndimage

    scenes = {}
    for p in sorted(METADATA.rglob("*.IMD")):
        try:
            scenes[(p.parent.name.upper(), int(p.stem))] = parse_imd(
                p.read_text(errors="replace"))
        except ValueError:
            continue

    names = [t for t in sorted(p.stem[:-4] for p in RGB_DIR.glob("*_RGB.tif"))
             if TILE_RE.match(t) and (AGL_DIR / f"{t}_AGL.tif").exists()]
    # Rank by tall content rather than sampling at random. Buildings above 20 m are 1.4% of
    # building pixels, so a random draw of six tiles returned NINE of them and could say
    # nothing about the only band this cue has to justify itself in.
    if args.rank_tall:
        scored = []
        for t in names:
            try:
                with rasterio.open(AGL_DIR / f"{t}_AGL.tif") as ds:
                    a_ = ds.read(1, out_shape=(1, 512, 512)).astype(np.float32)
            except Exception:
                continue
            scored.append((int(np.nansum(a_ > TALL)), t))
        scored.sort(reverse=True)
        names = [t for n_, t in scored if n_ > 0]
        print(f"ranked {len(names)} tiles by tall-building content; "
              f"top tile has {scored[0][0]:,} tall px (subsampled)")
    else:
        rng = np.random.default_rng(args.seed)
        rng.shuffle(names)

    OUT, TRU, OCC, SUN = [], [], [], []
    det_tp = det_fp = det_fn = det_tn = 0
    used = 0
    for t in names:
        if used >= args.tiles:
            break
        m = TILE_RE.match(t)
        sc = scenes.get((m.group(1).upper(), int(m.group(3))))
        if not sc or "meanSunEl" not in sc:
            continue
        with rasterio.open(AGL_DIR / f"{t}_AGL.tif") as ds:
            agl = ds.read(1).astype(np.float32)
        if np.nanmax(agl) < TALL:
            continue
        with rasterio.open(RGB_DIR / f"{t}_RGB.tif") as ds:
            rgb = np.transpose(ds.read()[:3], (1, 2, 0))
        el, az = float(sc["meanSunEl"]), float(sc["meanSunAz"])

        oracle = geometric_shadow(agl, el, az)
        # Score the luminance detector against the oracle, so a future threshold choice is
        # made on evidence rather than on whichever value flatters the height numbers.
        lum = rgb.astype(np.float32).mean(axis=2)
        try:
            from skimage.filters import threshold_otsu
            thr = float(threshold_otsu(lum))
        except ImportError:
            thr = float(np.percentile(lum, 45))
        det = lum < thr
        det_tp += int((det & oracle).sum());  det_fp += int((det & ~oracle).sum())
        det_fn += int((~det & oracle).sum()); det_tn += int((~det & ~oracle).sum())

        a = math.radians(az)
        drow, dcol = math.cos(a), -math.sin(a)
        k = math.tan(math.radians(el))
        max_steps = int(130.0 / (k * GSD)) + 10
        blds, n = ndimage.label(np.isfinite(agl) & (agl > 2.0))
        got = 0
        for lab in range(1, n + 1):
            mb = blds == lab
            truth = float(np.nanmedian(agl[mb]))
            if not np.isfinite(truth):
                continue
            res = measure_corridor(mb, oracle, (blds > 0) & ~mb, drow, dcol, max_steps)
            if res is None:
                continue
            run, occ = res
            OUT.append(run * GSD * k); TRU.append(truth)
            OCC.append(occ); SUN.append(el)
            got += 1
        used += 1
        print(f"  {t}: sun el {el:.1f}  oracle shadow {100*oracle.mean():.1f}% of tile "
              f"-> {got} buildings")

    o, t_, occ = np.array(OUT), np.array(TRU), np.array(OCC, bool)
    prec = det_tp / max(det_tp + det_fp, 1)
    rec = det_tp / max(det_tp + det_fn, 1)
    print(f"\nLUMINANCE+OTSU DETECTOR vs oracle: precision {prec:.3f}  recall {rec:.3f}  "
          f"(IoU {det_tp/max(det_tp+det_fp+det_fn,1):.3f})")
    print(f"{len(o):,} buildings via ORACLE shadow, {occ.sum()} occluded "
          f"({100*occ.mean():.1f}%)")

    for label, keep in (("ORACLE, all", np.ones_like(occ)), ("ORACLE, unoccluded", ~occ)):
        if keep.sum() < 10:
            continue
        oo, tt = o[keep], t_[keep]
        print(f"\n== {label}: {int(keep.sum()):,} buildings ==")
        print(f"  {'band':>10} {'n':>6} {'truth':>8} {'shadow':>9} {'RMSE':>9} {'r':>7}")
        for lo, hi in BANDS:
            mm = (tt >= lo) & (tt < hi)
            if mm.sum() < 3:
                continue
            print(f"  {band_label(lo, hi):>10} {int(mm.sum()):>6} {tt[mm].mean():>7.1f}m "
                  f"{oo[mm].mean():>8.1f}m "
                  f"{float(np.sqrt(np.mean((oo[mm]-tt[mm])**2))):>8.2f}m "
                  f"{float(np.corrcoef(oo[mm], tt[mm])[0,1]):>7.3f}")
        print(f"  overall r {np.corrcoef(oo, tt)[0,1]:.3f}  "
              f"slope {np.polyfit(tt, oo, 1)[0]:.3f}  "
              f"RMSE {float(np.sqrt(np.mean((oo-tt)**2))):.2f} m")


if __name__ == "__main__":
    main()
