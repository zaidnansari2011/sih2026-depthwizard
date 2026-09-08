"""Which functional form should a GCP calibration use? Affine is measurably the wrong one.

Probe 06 measured the error as a RATIO that varies with height: short structures come out
1.20x too tall at 0-3 m and 1.14x at 3-6 m, while tall ones come out at 0.52x above 40 m.
A straight line cannot be above 1 at one end and below 1 at the other, so an affine
calibration fitted from control points that are mostly short tilts the wrong way for the
tall end -- measured end to end on OMA_288_012 as scale 0.830, taking tile RMSE from
23.62 to 25.09 m.

A power law can do what a line cannot:

    truth = a * pred**b        <=>     log truth = log a + b * log pred

With b > 1 it expands the tall end while contracting the short end, which is the shape the
error actually has. It also has the same two free parameters as an affine fit, so it needs
no more control points.

This compares the two on the SAME protocol as tools/gcp_calibration_probe.py: fit per tile
on k control points, score only on buildings NOT used as control points, many draws, and
report the tiles where calibration made things worse. Nothing here is fitted on the
buildings it is scored on.

    python tools/analysis/calibration_form.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

ROOT = Path("D:/sih2026")
N_DRAWS = 200
EPS = 0.05          # metres; guards log() on near-zero heights


def fit_affine(gp, gt, x):
    if np.ptp(gp) < 1e-6:
        return x
    a, b = np.polyfit(gp, gt, 1)
    return x * a + b


def fit_power(gp, gt, x):
    """truth = a * pred**b, fitted in log space.

    Heights near zero are clipped rather than dropped: a 0.1 m building is real data, and
    log() of it is merely large-negative, not invalid. Clipping at 5 cm keeps the fit from
    being dominated by the logarithm's behaviour at the origin.
    """
    gp_, gt_ = np.maximum(gp, EPS), np.maximum(gt, EPS)
    if np.ptp(np.log(gp_)) < 1e-6:
        return x
    b, loga = np.polyfit(np.log(gp_), np.log(gt_), 1)
    if not (0.2 <= b <= 4.0):
        return x
    return np.exp(loga) * np.maximum(x, EPS) ** b


FORMS = {"affine": fit_affine, "power": fit_power}


def metrics(p, t):
    return float(np.sqrt(np.mean((p - t) ** 2))), float(np.mean(np.abs(p - t)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", default="out/eval_run02_tta_gcp")
    ap.add_argument("--min-test", type=int, default=5)
    ap.add_argument("--k", type=int, nargs="+", default=[3, 5, 8])
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    z = np.load(ROOT / args.eval / "per_building.npz")
    ours, truth, tidx = z["ours"], z["truth"], z["tile_idx"]
    rng = np.random.default_rng(args.seed)
    need = args.min_test + max(args.k)
    tiles = [t for t in np.unique(tidx) if (tidx == t).sum() >= need]
    tall_tiles = [t for t in tiles if (truth[tidx == t] > 20).any()]

    for title, group in (("ALL usable tiles", tiles),
                         ("TILES WITH A BUILDING > 20 m", tall_tiles)):
        if not group:
            continue
        sel = np.isin(tidx, group)
        b_rmse, b_mae = metrics(ours[sel], truth[sel])
        print(f"\n== {title}: {len(group)} tiles, {int(sel.sum()):,} buildings ==")
        print(f"  uncalibrated: RMSE {b_rmse:.3f} m  MAE {b_mae:.3f} m")
        print(f"  {'form':>7} {'k':>3} {'RMSE':>9} {'MAE':>9} {'vs base':>9} {'worse':>9}")
        for form, fn in FORMS.items():
            for k in args.k:
                rmses, maes, worse = [], [], 0
                for t in group:
                    m = np.where(tidx == t)[0]
                    op, tp = ours[m], truth[m]
                    order = np.argsort(tp)
                    acc_r, acc_m = [], []
                    for _ in range(N_DRAWS):
                        # Spread across the tile's height range: the strategy that won in
                        # SS 9e, and the one a GCP instruction sheet would describe.
                        q = np.linspace(0, len(m) - 1, k).round().astype(int)
                        gi = np.unique(order[np.clip(q + rng.integers(-1, 2, k),
                                                     0, len(m) - 1)])
                        test = np.setdiff1d(np.arange(len(m)), gi)
                        if len(test) < args.min_test:
                            continue
                        cal = fn(op[gi], tp[gi], op[test])
                        r_, m_ = metrics(cal, tp[test])
                        acc_r.append(r_); acc_m.append(m_)
                    if not acc_r:
                        continue
                    base_r, _ = metrics(op, tp)
                    rmses.append(np.median(acc_r)); maes.append(np.median(acc_m))
                    worse += int(np.median(acc_r) > base_r)
                if not rmses:
                    continue
                rm, ma = float(np.mean(rmses)), float(np.mean(maes))
                print(f"  {form:>7} {k:>3} {rm:>8.3f}m {ma:>8.3f}m "
                      f"{100*(rm-b_rmse)/b_rmse:>+8.1f}% {worse:>4}/{len(rmses):<4}")


if __name__ == "__main__":
    main()
