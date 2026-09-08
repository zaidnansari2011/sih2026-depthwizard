"""Sharpen a predicted height map against the image it came from, and score the result.

The model's output is smooth at object boundaries: with a direct-regression head the
loss-minimising value at an ambiguous roof-edge pixel is the average of ground and roof,
so a step in the world becomes a ramp in the prediction. Buildings therefore render as
rounded mounds rather than the flat-topped blocks an LOD1 product should show.

Guided filtering (He, Sun & Tang, ECCV 2010) is the standard fix in depth refinement: the
filter smooths the height map WITHIN regions but stops at edges present in a guide image.
Feed it the satellite image as the guide and roof boundaries -- which are strong in the
imagery, because roofs differ from ground in brightness and cast shadows -- get transferred
into the height map. No retraining, seconds per tile.

The honest risk, which is why this script scores rather than just writes a picture: the
guide's edges are NOT all height edges. A painted road marking, a dark roof on dark ground,
or a tree shadow can all pull a step into a surface that should be flat. Whether the trade
is worth it is a measurement, not an opinion.

    python tools/guided_refine.py --tiles JAX_203_010 OMA_288_042 --radius 8
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import rasterio
from scipy.ndimage import uniform_filter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from depthwizard.metrics import building_instances, building_wise_metrics  # noqa: E402

ROOT = Path("D:/sih2026")
TRUTH = ROOT / "data" / "extracted" / "Track1-Truth"
RGB = ROOT / "data" / "extracted" / "Track1-RGB"


def guided_filter(guide: np.ndarray, src: np.ndarray, radius: int, eps: float) -> np.ndarray:
    """He et al. guided filter, grayscale guide. Both inputs float32 in comparable units.

    The linear model q = a*I + b is fitted in every window, so where the guide has an edge
    and the source does not, `a` is large and the edge is carried across.
    """
    size = 2 * radius + 1
    box = lambda x: uniform_filter(x, size=size, mode="nearest")   # noqa: E731
    mean_i, mean_p = box(guide), box(src)
    cov = box(guide * src) - mean_i * mean_p
    var = box(guide * guide) - mean_i * mean_i
    a = cov / (var + eps)
    b = mean_p - a * mean_i
    return box(a) * guide + box(b)


def score(pred, agl, cls, label):
    m = np.isfinite(pred) & np.isfinite(agl)
    px = float(np.sqrt(np.mean((pred[m] - agl[m]) ** 2)))
    b = (cls == 6) & m
    bpx = float(np.sqrt(np.mean((pred[b] - agl[b]) ** 2))) if b.any() else float("nan")
    ours, truth = building_instances(pred, agl, cls)
    bw = building_wise_metrics(ours, truth)
    print(f"  {label:22s} px {px:6.3f}   building-px {bpx:6.3f}   "
          f"per-building {bw['rmse']:6.3f} (bias {bw['bias']:+.2f}, n={bw['n_buildings']})")
    return px, bpx, bw["rmse"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles", nargs="+", required=True)
    ap.add_argument("--pred-dir", default=str(ROOT / "out"))
    ap.add_argument("--pattern", default="scene_{tile}.height.tif")
    ap.add_argument("--radius", type=int, nargs="+", default=[4, 8, 16])
    ap.add_argument("--eps", type=float, nargs="+", default=[1e-4, 1e-3, 1e-2])
    ap.add_argument("--write", help="directory to write the best refined tiles into")
    a = ap.parse_args()

    grid = [(r, e) for r in a.radius for e in a.eps]
    totals = {k: [] for k in [("base", 0.0)] + grid}

    for tile in a.tiles:
        p = Path(a.pred_dir) / a.pattern.format(tile=tile)
        if not p.exists():
            print(f"skip {tile}: no {p.name}")
            continue
        with rasterio.open(p) as s:
            pred = s.read(1).astype("float32")
        with rasterio.open(TRUTH / f"{tile}_AGL.tif") as s:
            agl = s.read(1).astype("float32")
        with rasterio.open(TRUTH / f"{tile}_CLS.tif") as s:
            cls = s.read(1)
        with rasterio.open(RGB / f"{tile}_RGB.tif") as s:
            rgb = s.read().astype("float32")

        # Luminance, scaled to metres-ish so eps means something comparable across tiles.
        g = (0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2])
        g = (g - g.mean()) / (g.std() + 1e-6)

        print(f"\n{tile}")
        totals[("base", 0.0)].append(score(pred, agl, cls, "unfiltered"))
        for (r, e) in grid:
            q = guided_filter(g, pred, r, e)
            totals[(r, e)].append(score(q, agl, cls, f"guided r={r} eps={e:g}"))

    print("\n" + "=" * 78)
    print(f"{'setting':22s} {'per-pixel':>10s} {'building-px':>12s} {'per-building':>13s}")
    base = None
    for k, rows in totals.items():
        if not rows:
            continue
        px, bpx, bw = (float(np.mean([r[i] for r in rows])) for i in range(3))
        if base is None:
            base = (px, bpx, bw)
            tag = "unfiltered"
            delta = ""
        else:
            tag = f"guided r={k[0]} eps={k[1]:g}"
            delta = (f"   [{px-base[0]:+.3f} {bpx-base[1]:+.3f} {bw-base[2]:+.3f}]")
        print(f"{tag:22s} {px:10.3f} {bpx:12.3f} {bw:13.3f}{delta}")
    print("\nNegative deltas are improvements. per-building is the figure the field reports.")


if __name__ == "__main__":
    main()
