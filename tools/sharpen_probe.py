"""Can any post-process turn our ramps back into steps? Measured, not guessed.

Probe 03 showed guided filtering cannot: it PRESERVES edges that exist and ours do not
exist, so it only smoothed further. The operators here are different in kind -- each one
can manufacture a step from a ramp.

  unsharp    h + a*(h - blur(h)). Steepens an existing gradient. Classic, but it
             overshoots: a halo of too-high pixels appears just outside every roof, which
             on a height map is a moat of fake elevation around each building.

  toggle     Toggle contrast (Kramer & Bruckner 1975). Each pixel snaps to whichever of
             its local max or local min it is already nearer. A ramp becomes a step at the
             midpoint, with no overshoot at all -- it cannot invent a value that was not
             already present in the neighbourhood. This is the operator guided filtering
             should have been.

  toggle+    Toggle applied only where the guide image says there is an edge, so flat
             roofs and open ground are left alone and only real boundaries are squared up.

Scored on per-pixel, building-pixel and per-building RMSE. Building-pixel is the one that
should move: it is dominated by roof edges, which is exactly what is being sharpened.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import rasterio
from scipy.ndimage import gaussian_filter, grey_dilation, grey_erosion, uniform_filter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from depthwizard.metrics import building_instances, building_wise_metrics  # noqa: E402

ROOT = Path("D:/sih2026")
TRUTH = ROOT / "data" / "extracted" / "Track1-Truth"
RGB = ROOT / "data" / "extracted" / "Track1-RGB"


def unsharp(h, sigma, amount):
    return h + amount * (h - gaussian_filter(h, sigma))


def toggle(h, size):
    """Snap each pixel to the nearer of its local max and local min."""
    d = grey_dilation(h, size=size)
    e = grey_erosion(h, size=size)
    return np.where((d - h) < (h - e), d, e)


def toggle_masked(h, size, edge, thresh):
    """Toggle only where the height field itself has a meaningful local range.

    Restricting it keeps flat roofs and open ground untouched -- toggling a noisy flat
    surface quantises it into blotches, which looks worse than the blur it replaced.
    """
    out = toggle(h, size)
    return np.where(edge > thresh, out, h)


def score(pred, agl, cls, label, rows):
    m = np.isfinite(pred) & np.isfinite(agl)
    px = float(np.sqrt(np.mean((pred[m] - agl[m]) ** 2)))
    b = (cls == 6) & m
    bpx = float(np.sqrt(np.mean((pred[b] - agl[b]) ** 2))) if b.any() else float("nan")
    ours, truth = building_instances(pred, agl, cls)
    bw = building_wise_metrics(ours, truth)
    rows.setdefault(label, []).append((px, bpx, bw["rmse"], bw["bias"]))
    return px, bpx, bw["rmse"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles", nargs="+", required=True)
    ap.add_argument("--pred-dir", default=str(ROOT / "out"))
    ap.add_argument("--pattern", default="scene_{tile}.height.tif")
    a = ap.parse_args()

    rows: dict[str, list] = {}
    for tile in a.tiles:
        p = Path(a.pred_dir) / a.pattern.format(tile=tile)
        if not p.exists():
            continue
        with rasterio.open(p) as s:
            h = s.read(1).astype("float32")
        with rasterio.open(TRUTH / f"{tile}_AGL.tif") as s:
            agl = s.read(1).astype("float32")
        with rasterio.open(TRUTH / f"{tile}_CLS.tif") as s:
            cls = s.read(1)

        # Local height range: where the surface is actually transitioning.
        rng = grey_dilation(h, size=5) - grey_erosion(h, size=5)

        print(f"\n{tile}")
        for label, fn in [
            ("unfiltered", lambda x: x),
            ("unsharp s1 a0.5", lambda x: unsharp(x, 1.0, 0.5)),
            ("unsharp s2 a1.0", lambda x: unsharp(x, 2.0, 1.0)),
            ("toggle 3", lambda x: toggle(x, 3)),
            ("toggle 5", lambda x: toggle(x, 5)),
            ("toggle 7", lambda x: toggle(x, 7)),
            ("toggle 5 @rng>1", lambda x: toggle_masked(x, 5, rng, 1.0)),
            ("toggle 5 @rng>2", lambda x: toggle_masked(x, 5, rng, 2.0)),
            ("toggle 7 @rng>2", lambda x: toggle_masked(x, 7, rng, 2.0)),
        ]:
            px, bpx, bw = score(fn(h), agl, cls, label, rows)
            print(f"  {label:18s} px {px:6.3f}  building-px {bpx:7.3f}  per-building {bw:6.3f}")

    print("\n" + "=" * 80)
    print(f"{'operator':18s} {'per-pixel':>10s} {'building-px':>12s} {'per-building':>13s} "
          f"{'bias':>7s}")
    base = None
    for label, r in rows.items():
        px, bpx, bw, bias = (float(np.mean([x[i] for x in r])) for i in range(4))
        if base is None:
            base = (px, bpx, bw)
            print(f"{label:18s} {px:10.3f} {bpx:12.3f} {bw:13.3f} {bias:+7.2f}")
        else:
            print(f"{label:18s} {px:10.3f} {bpx:12.3f} {bw:13.3f} {bias:+7.2f}"
                  f"   [{px-base[0]:+.3f} {bpx-base[1]:+.3f} {bw-base[2]:+.3f}]")
    print("\nNegative deltas are improvements.")


if __name__ == "__main__":
    main()
