"""Would LDS have room to work here? Measure before writing any training code.

Yang et al., "Delving into Deep Imbalanced Regression", ICML 2021. Label Distribution
Smoothing convolves the empirical label density with a symmetric kernel to get an
*effective* density, then reweights each sample by the inverse of it. It is a loss-side
change, not an architecture change -- which matters, because four runs have ruled out head
design (regression / bins / bins+HTC all land at slope 0.43-0.49) and none of them touched
how much the loss cares about a tall pixel in the first place.

The question this answers, before spending a 2-hour training run: **is the training signal
actually dominated by short pixels, and by how much?** If tall pixels already carry a fair
share of the gradient, LDS has nothing to fix and the run is wasted. If they carry almost
none, LDS is aimed at a real target.

Reports, over the training shards:
  - the empirical height distribution, which is the thing said to be long-tailed
  - each band's share of pixels and of squared error against a constant predictor, the
    latter standing in for "how much does the loss currently care about this band"
  - the LDS weight each band would receive, and the share of loss it would then carry

A constant predictor is the right reference here on purpose: regression to the mean IS the
model collapsing toward a constant, so measuring error against the median is measuring the
gradient that has to overcome that collapse.

    python tools/lds_probe.py
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np

ROOT = Path("D:/sih2026")
BANDS = [(0, 3), (3, 6), (6, 10), (10, 20), (20, 40), (40, 1e9)]
BIN_W = 1.0          # metres per histogram bin
MAX_H = 120.0


def band_label(lo, hi):
    return f"{lo:g}-{hi:g} m" if hi < 1e8 else f"{lo:g} m+"


def gaussian_kernel(sigma: float, half: int) -> np.ndarray:
    x = np.arange(-half, half + 1, dtype=np.float64)
    k = np.exp(-(x ** 2) / (2 * sigma ** 2))
    return k / k.sum()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", type=int, default=8, help="train shards to sample")
    ap.add_argument("--stride", type=int, default=7, help="pixel subsample stride")
    ap.add_argument("--sigma", type=float, default=2.0, help="LDS kernel sigma, in bins")
    ap.add_argument("--alpha", type=float, default=0.5,
                    help="weight = 1/density**alpha; 1.0 is full inverse, 0.5 is the "
                         "square-root damping usually used to avoid extreme weights")
    args = ap.parse_args()

    files = sorted(glob.glob(str(ROOT / "data" / "shards" / "train_*.npz")))[:args.shards]
    if not files:
        raise SystemExit("no train shards found")

    vals = []
    for f in files:
        with np.load(f) as z:
            a = z["agl"][:, ::args.stride, ::args.stride].astype(np.float32).ravel()
        vals.append(a[np.isfinite(a)])
    h = np.concatenate(vals)
    h = h[(h >= 0) & (h <= MAX_H)]
    print(f"{len(h):,} pixels from {len(files)} train shards "
          f"(stride {args.stride})")
    print(f"  median {np.median(h):.2f} m   mean {h.mean():.2f} m   "
          f"p99 {np.percentile(h, 99):.1f} m   max {h.max():.1f} m\n")

    edges = np.arange(0, MAX_H + BIN_W, BIN_W)
    counts, _ = np.histogram(h, bins=edges)
    centres = 0.5 * (edges[:-1] + edges[1:])

    # LDS: smooth the empirical density, then weight by its inverse.
    k = gaussian_kernel(args.sigma, half=int(3 * args.sigma))
    smooth = np.convolve(counts.astype(np.float64), k, mode="same")
    smooth = np.maximum(smooth, 1.0)                 # empty bins must not blow up
    w = 1.0 / smooth ** args.alpha
    w = w / w.max()                                  # relative, tallest bin = 1.0

    # Squared error against a constant predictor at the median: the collapse LDS must beat.
    const = float(np.median(h))
    sq = (h - const) ** 2
    tot_px, tot_sq = len(h), sq.sum()

    print(f"  {'band':>10} {'% pixels':>9} {'% sq.err':>9} {'LDS weight':>11} "
          f"{'% sq.err after':>15}")
    rows = []
    for lo, hi in BANDS:
        m = (h >= lo) & (h < hi)
        n = int(m.sum())
        if n == 0:
            continue
        bm = (centres >= lo) & (centres < hi)
        wb = float(w[bm].mean()) if bm.any() else 0.0
        rows.append((band_label(lo, hi), n, float(sq[m].sum()), wb))
    tot_after = sum(s * wb for _, _, s, wb in rows)
    for lab, n, s, wb in rows:
        print(f"  {lab:>10} {100*n/tot_px:>8.2f}% {100*s/tot_sq:>8.2f}% {wb:>11.4f} "
              f"{100*s*wb/tot_after:>14.2f}%")

    lo_w = float(w[(centres >= 3) & (centres < 6)].mean())
    hi_w = float(w[(centres >= 40)].mean())
    print(f"\n  a 40 m+ pixel is upweighted {hi_w/lo_w:.1f}x relative to a 3-6 m pixel "
          f"(alpha={args.alpha}, sigma={args.sigma} bins)")
    tall = h >= 20
    print(f"  pixels above 20 m are {100*tall.mean():.3f}% of the training signal and carry "
          f"{100*sq[tall].sum()/tot_sq:.2f}% of the squared error against a constant")


if __name__ == "__main__":
    main()
