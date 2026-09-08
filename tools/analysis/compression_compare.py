"""Compare runs on the metric that actually shows the tail: the compression slope.

Probe 06 measured `ours = 0.473 * truth + 2.66 m` on run02 -- every height pulled toward
the ~4 m median of the training distribution. run03's binned head exists to stop exactly
that, because a head that picks a bin cannot average two answers into a wrong middle one.

run03 was dropped on whole-tile RMSE (6.950 vs 6.456) before probe 06 existed. That was
the wrong instrument: the compression lives in 1.9% of buildings carrying 79% of the
squared error, and an aggregate RMSE is dominated by the 98% that were already fine. So a
binned head can fix the thing it was built to fix and still lose on the number it was
judged by.

This reads the per_building.npz that evaluate.py writes and reports, per run:

  - the linear fit `ours = a * truth + b`; a slope near 1.0 is the goal, 0.473 is run02
  - median(ours)/median(truth) per height band, which is probe 06's table
  - per-building RMSE per band, so a slope win that costs accuracy is visible

    python tools/analysis/compression_compare.py run02 run03 run04
    python tools/analysis/compression_compare.py --dir D:/sih2026/out
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

ROOT = Path("D:/sih2026")

# Probe 06's bands, kept identical so the numbers can be read side by side with
# docs/probe-06-compression.md rather than re-derived.
BANDS = [(0, 3), (3, 6), (6, 10), (10, 20), (20, 40), (40, 1e9)]


def band_label(lo: float, hi: float) -> str:
    return f"{lo:g}-{hi:g} m" if hi < 1e8 else f"{lo:g} m+"


def load(path: Path) -> tuple[np.ndarray, np.ndarray]:
    z = np.load(path)
    ours, truth = np.asarray(z["ours"], float), np.asarray(z["truth"], float)
    # Guard against the odd non-finite that a diverged run can leave behind; a single
    # NaN would otherwise poison polyfit and report nothing at all.
    m = np.isfinite(ours) & np.isfinite(truth)
    return ours[m], truth[m]


def describe(name: str, ours: np.ndarray, truth: np.ndarray) -> dict:
    slope, intercept = np.polyfit(truth, ours, 1)
    rmse = float(np.sqrt(np.mean((ours - truth) ** 2)))
    r = float(np.corrcoef(ours, truth)[0, 1])
    print(f"\n=== {name} — {len(ours):,} buildings ===")
    print(f"  ours = {slope:.4f} * truth {intercept:+.3f} m     (slope 1.0 = no compression)")
    print(f"  per-building RMSE {rmse:.3f} m     r {r:.3f}")
    print(f"  {'band':>10}  {'n':>6}  {'med ratio':>9}  {'RMSE':>8}  {'bias':>8}")
    rows = []
    for lo, hi in BANDS:
        m = (truth >= lo) & (truth < hi)
        n = int(m.sum())
        if n == 0:
            continue
        # Median ratio, not mean: probe 06 used medians and the tail is skewed enough
        # that a mean ratio would be moved by a handful of buildings.
        ratio = float(np.median(ours[m]) / np.median(truth[m]))
        b_rmse = float(np.sqrt(np.mean((ours[m] - truth[m]) ** 2)))
        bias = float(np.mean(ours[m] - truth[m]))
        rows.append((band_label(lo, hi), n, ratio, b_rmse, bias))
        print(f"  {band_label(lo, hi):>10}  {n:>6}  {ratio:>9.2f}  {b_rmse:>7.2f}m  {bias:>+7.2f}m")
    return {"name": name, "slope": float(slope), "intercept": float(intercept),
            "rmse": rmse, "r": r, "n": len(ours), "bands": rows}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="*", default=[],
                    help="run names; resolved as <dir>/eval_*<name>*/per_building.npz")
    ap.add_argument("--dir", default=str(ROOT / "out"))
    args = ap.parse_args()

    out_dir = Path(args.dir)
    found: list[tuple[str, Path]] = []
    if args.runs:
        for name in args.runs:
            hits = sorted(p for p in out_dir.glob(f"eval*{name}*/per_building.npz"))
            if not hits:
                print(f"  (no per_building.npz for {name} — run tools/evaluate.py first)")
            found += [(f"{name} [{p.parent.name}]", p) for p in hits]
    else:
        found = [(p.parent.name, p) for p in sorted(out_dir.glob("eval*/per_building.npz"))]

    if not found:
        raise SystemExit("nothing to compare")

    results = [describe(n, *load(p)) for n, p in found]

    print("\n" + "=" * 72)
    print(f"{'run':<34} {'slope':>7} {'intercept':>10} {'RMSE':>9} {'r':>7}")
    for r in sorted(results, key=lambda d: -d["slope"]):
        print(f"{r['name']:<34} {r['slope']:>7.3f} {r['intercept']:>9.2f}m "
              f"{r['rmse']:>8.3f}m {r['r']:>7.3f}")
    print("\nslope closer to 1.0 is less compression. run02 baseline is 0.473.")


if __name__ == "__main__":
    main()
