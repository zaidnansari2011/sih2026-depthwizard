"""How many Ground Control Points does a user need before heights are right?

The PS names this as its own milestone: "convert relative depth to absolute height using
scene-level statistics, low-resolution DEMs, semantic priors, or minimal Ground Control
Points." Our defect is a scale error -- `ours = 0.473 * truth + 2.66` -- and four training
runs failed to move it from inside the network. This measures the fix the PS actually asks
for, in the place the PS puts it.

Not the same experiment as `tools/decompress_probe.py`, and the difference is the point.
That one fitted ONE global correction on the train split and applied it to val, which
failed because train holds 7 buildings above 20 m against val's 60: it applied 1.24x where
2.1x was needed. A GCP is placed in ONE scene by a user looking at that scene, so the fit
must be per-tile and tested only on buildings that were NOT used as control points.

Honest about what calibration can and cannot do. Within ONE tile a positive affine map
cannot change Pearson correlation, so calibration adds no information -- it only moves
error around. Report `r` WITHIN tiles and take the median; an earlier version pooled `r`
across tiles and watched it fall 0.654 -> 0.130, which measured nothing but the fact that
each tile had been given a different map.

And it does not move RMSE and MAE together. Measured on the 13 tall tiles, the best
setting (5 spread control points) takes RMSE 9.362 -> 7.366 m, -21%, while MAE goes
4.933 -> 5.843 m, +18%. Calibration buys the big errors on tall buildings by spending
accuracy on the many short ones. Of the three metrics the PS names it improves one,
worsens one, and cannot touch the third.

Two ways of choosing control points, because they answer different questions:
  random  -- a user who clicks arbitrary buildings; the conservative number.
  spread  -- a user who deliberately picks a short and a tall structure. This is what the
             instructions for a real GCP workflow would say, and with k=2 an affine fit
             through two near-identical heights is near-singular, so the choice matters.

    python tools/gcp_calibration_probe.py --eval out/eval_run02_tta_gcp
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

ROOT = Path("D:/sih2026")
N_DRAWS = 200          # random GCP draws per tile per k, to average out which ones landed

# MEASURED 28 Aug, and it dictates the thresholds below. Requiring 25 buildings per tile
# excluded 100% of the tall stock: the 36 tiles with >=25 buildings hold ZERO buildings
# above 20 m (RMSE 1.381 m, slope 0.723), while all 60 tall buildings sit in the 34 tiles
# with fewer (RMSE 8.487 m, slope 0.438). Dense suburban tiles have many small buildings;
# downtown tiles have few large ones. So a threshold tuned for statistical comfort tests
# calibration exactly where there is no problem to fix. Keep the bar low and stratify.


def fit_apply(g_pred: np.ndarray, g_true: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Calibrate x using the control points (g_pred -> g_true).

    k == 1 fits scale only. One point cannot determine both slope and intercept, and
    forcing an intercept through a single point would just translate every height by that
    building's error.
    """
    if len(g_pred) == 1:
        denom = g_pred[0]
        if abs(denom) < 1e-3:
            return x
        return x * (g_true[0] / denom)
    if np.ptp(g_pred) < 1e-6:          # degenerate: all control points identical
        return x
    slope, intercept = np.polyfit(g_pred, g_true, 1)
    return x * slope + intercept


def metrics(pred: np.ndarray, true: np.ndarray) -> tuple[float, float]:
    return (float(np.sqrt(np.mean((pred - true) ** 2))), float(np.mean(np.abs(pred - true))))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", default="out/eval_run02_tta_gcp",
                    help="an eval dir whose per_building.npz carries tile_idx")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--min-test", type=int, default=5,
                    help="held-out buildings a tile must retain to be scored")
    ap.add_argument("--k", type=int, nargs="+", default=[1, 2, 3, 5])
    args = ap.parse_args()
    K_VALUES, MIN_TEST = tuple(args.k), args.min_test

    z = np.load(ROOT / args.eval / "per_building.npz")
    if "tile_idx" not in z.files:
        raise SystemExit(f"{args.eval}/per_building.npz predates tile tagging; "
                         "re-run tools/evaluate.py")
    ours, truth, tidx = z["ours"], z["truth"], z["tile_idx"]
    rng = np.random.default_rng(args.seed)

    need = MIN_TEST + max(K_VALUES)
    tiles = [t for t in np.unique(tidx) if (tidx == t).sum() >= need]
    tall_tiles = [t for t in tiles if (truth[tidx == t] > 20).any()]
    print(f"{len(ours):,} buildings over {len(np.unique(tidx))} tiles; "
          f"{len(tiles)} tiles have >= {need} buildings and are usable, "
          f"of which {len(tall_tiles)} contain a building above 20 m")
    n_used = sum(int((tidx == t).sum()) for t in tiles)
    print(f"{n_used:,} buildings in usable tiles\n")

    # Uncalibrated baseline, over exactly the tiles the calibrated numbers will use, so the
    # comparison is like for like rather than against the whole-set figure.
    sel = np.isin(tidx, tiles)
    b_rmse, b_mae = metrics(ours[sel], truth[sel])
    b_r = float(np.corrcoef(ours[sel], truth[sel])[0, 1])
    b_slope = float(np.polyfit(truth[sel], ours[sel], 1)[0])
    print(f"  uncalibrated over those tiles: RMSE {b_rmse:.3f} m  MAE {b_mae:.3f} m  "
          f"r {b_r:.3f}  slope {b_slope:.3f}\n")

    run(ours, truth, tidx, tiles, rng, K_VALUES, MIN_TEST, "ALL usable tiles")
    if tall_tiles:
        run(ours, truth, tidx, tall_tiles, rng, K_VALUES, MIN_TEST,
            "TILES WITH A BUILDING > 20 m")
    print(
        "\n  'vs base' is change in RMSE against that stratum's uncalibrated number;"
        " negative is better."
        "\n  'worse' counts tiles where calibration HURT."
        "\n  r is the MEDIAN WITHIN-TILE correlation. Pooling r across tiles was wrong:"
        " each tile gets its"
        "\n  own affine map, so the pooled statistic is not an affine image of anything."
        " It moved 0.654 ->"
        "\n  0.130 purely from cross-tile inconsistency, not from any real change.")


def run(ours, truth, tidx, tiles, rng, K_VALUES, MIN_TEST, title):
    sel = np.isin(tidx, tiles)
    b_rmse, b_mae = metrics(ours[sel], truth[sel])
    b_r = float(np.median([np.corrcoef(ours[tidx == t], truth[tidx == t])[0, 1]
                           for t in tiles if np.ptp(truth[tidx == t]) > 1e-6]))
    print(f"\n== {title}: {len(tiles)} tiles, {int(sel.sum()):,} buildings ==")
    print(f"  uncalibrated: RMSE {b_rmse:.3f} m  MAE {b_mae:.3f} m  "
          f"median within-tile r {b_r:.3f}")
    print(f"  {'strategy':>8} {'k':>3} {'RMSE':>9} {'MAE':>9} {'r':>7} "
          f"{'vs base':>9} {'worse':>10}")
    for strategy in ("random", "spread"):
        for k in K_VALUES:
            if strategy == "spread" and k == 1:
                continue                    # identical to random with one point
            rmses, maes, rs, worse = [], [], [], 0
            for t in tiles:
                m = np.where(tidx == t)[0]
                op, tp = ours[m], truth[m]
                order = np.argsort(tp)
                acc_r, acc_m, acc_corr = [], [], []
                for _ in range(N_DRAWS):
                    if strategy == "random":
                        gi = rng.choice(len(m), k, replace=False)
                    else:
                        # Span the tile's height range: k quantiles of true height, which
                        # is what "pick a short one and a tall one" means operationally.
                        q = np.linspace(0, len(m) - 1, k).round().astype(int)
                        jitter = rng.integers(-1, 2, k)
                        gi = order[np.clip(q + jitter, 0, len(m) - 1)]
                        gi = np.unique(gi)
                    test = np.setdiff1d(np.arange(len(m)), gi)
                    if len(test) < MIN_TEST:
                        continue
                    cal = fit_apply(op[gi], tp[gi], op[test])
                    r_, m_ = metrics(cal, tp[test])
                    acc_r.append(r_)
                    acc_m.append(m_)
                    if np.ptp(tp[test]) > 1e-6 and np.ptp(cal) > 1e-6:
                        acc_corr.append(float(np.corrcoef(cal, tp[test])[0, 1]))
                if not acc_r:
                    continue
                base_r, _ = metrics(op, tp)
                rmses.append(np.median(acc_r))
                maes.append(np.median(acc_m))
                if acc_corr:
                    rs.append(np.median(acc_corr))
                worse += int(np.median(acc_r) > base_r)
            if not rmses:
                continue
            r = float(np.median(rs)) if rs else float("nan")
            rm, ma = float(np.mean(rmses)), float(np.mean(maes))
            print(f"  {strategy:>8} {k:>3} {rm:>8.3f}m {ma:>8.3f}m {r:>7.3f} "
                  f"{100*(rm-b_rmse)/b_rmse:>+8.1f}% {worse:>4}/{len(rmses):<5}")

    # Caveat on the r column under 'spread': the strategy removes the tallest and shortest
    # buildings from the tile to use as control points, so the held-out set it is scored on
    # is narrower than 'random' leaves behind. A lower r there is a change of test set, not
    # a change of correlation.


if __name__ == "__main__":
    main()
