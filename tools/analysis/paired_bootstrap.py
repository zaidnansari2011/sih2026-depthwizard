"""Did the model actually change, or did the sample? Paired bootstrap over two eval runs.

    python tools/analysis/paired_bootstrap.py out/eval_run02_ship out/eval_run07_ship

`gamus-integration.md` §6b quotes confidence intervals on the run02 -> run07 difference that
no committed tool could reproduce. This is that tool. The evidence pack's standing rule is
that every figure regenerates from a metrics file; an interval that exists only in prose is
the one kind of number a jury can ask for and not be shown.

Why paired, and why it matters here
-----------------------------------
Both runs are scored on the same 3,090 buildings in the same order, so the runs differ only
in the model. Resampling *buildings* (not predictions) and taking the difference inside each
resample cancels the building-to-building variation that both runs share, which is most of
the variance. An unpaired comparison of two RMSEs would be far wider and would hide a real
effect -- the >20 m band has 60 buildings and cannot afford the waste.

The band breakdown is the point. A single overall delta averages a 2,305-building band
against a 60-building one and says nothing about either. `metrics.json` reports per-band
point estimates but no intervals, so a band that moved the wrong way looks identical to a
band that moved by luck.

Reading the output
------------------
`delta` is run B minus run A, so for RMSE and |bias| **negative is better**; for a bias that
is negative to begin with (we under-call tall buildings) a *positive* bias delta moves it
toward zero. A 95% CI that excludes zero means the sample can resolve the change; one that
includes zero means the change may be real but this validation set cannot tell -- which for
small bands is a statement about the instrument, not about the model.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

# Matches evaluate.py's building_wise_by_height bands, plus >30 m, which §6b reports
# separately because it is where the under-call is worst and the sample is thinnest.
BANDS: list[tuple[str, float, float]] = [
    ("0-3 m", 0.0, 3.0),
    ("3-6 m", 3.0, 6.0),
    ("6-10 m", 6.0, 10.0),
    ("10-20 m", 10.0, 20.0),
    (">20 m", 20.0, np.inf),
    (">30 m", 30.0, np.inf),
    ("all", 0.0, np.inf),
]


def load(d: Path) -> tuple[np.ndarray, np.ndarray]:
    z = np.load(d / "per_building.npz", allow_pickle=True)
    return np.asarray(z["ours"], float), np.asarray(z["truth"], float)


def rmse(e: np.ndarray) -> float:
    return float(np.sqrt(np.mean(e ** 2)))


def boot(err_a: np.ndarray, err_b: np.ndarray, n: int, rng: np.random.Generator):
    """Paired bootstrap of the B-minus-A difference in bias and in RMSE.

    Returns (delta, lo, hi, p) for each, where p is the two-sided bootstrap p-value:
    twice the smaller tail mass on either side of zero.
    """
    k = len(err_a)
    idx = rng.integers(0, k, size=(n, k))
    a, b = err_a[idx], err_b[idx]
    d_bias = b.mean(1) - a.mean(1)
    d_rmse = np.sqrt((b ** 2).mean(1)) - np.sqrt((a ** 2).mean(1))

    def summarise(d: np.ndarray, point: float):
        lo, hi = np.percentile(d, [2.5, 97.5])
        p = 2.0 * min((d <= 0).mean(), (d >= 0).mean())
        return point, float(lo), float(hi), float(min(p, 1.0))

    return (summarise(d_bias, float(err_b.mean() - err_a.mean())),
            summarise(d_rmse, rmse(err_b) - rmse(err_a)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_a", help="baseline eval dir (holds per_building.npz)")
    ap.add_argument("run_b", help="candidate eval dir")
    ap.add_argument("--labels", nargs=2, default=None)
    ap.add_argument("--resamples", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=1337, help="project-standard seed")
    ap.add_argument("--markdown", action="store_true")
    args = ap.parse_args()

    da, db = Path(args.run_a), Path(args.run_b)
    la, lb = args.labels or (da.name, db.name)
    ours_a, truth_a = load(da)
    ours_b, truth_b = load(db)

    # The pairing is the whole method. If the two runs did not score the same buildings in
    # the same order, every interval below would be meaningless, so refuse rather than warn.
    if ours_a.shape != ours_b.shape:
        raise SystemExit(f"different building counts: {ours_a.shape} vs {ours_b.shape} "
                         "-- these runs did not score the same set, pairing is invalid")
    if not np.allclose(truth_a, truth_b, atol=1e-9):
        raise SystemExit("truth arrays differ between runs -- pairing is invalid; the two "
                         "evals used different ground truth or a different building order")

    err_a, err_b = ours_a - truth_a, ours_b - truth_b
    rng = np.random.default_rng(args.seed)

    print(f"\npaired bootstrap, {args.resamples:,} resamples, seed {args.seed}")
    print(f"  A = {la}  ({da})")
    print(f"  B = {lb}  ({db})")
    print(f"  {len(err_a):,} buildings, identical truth on both sides\n")

    rows = []
    hdr = (f"{'band':<9}{'n':>6}{'  bias A':>10}{'bias B':>9}{'delta':>8}"
           f"{'95% CI':>18}{'p':>8}   {'rmse A':>7}{'rmse B':>8}{'delta':>8}"
           f"{'95% CI':>18}{'p':>8}")
    print(hdr)
    print("-" * len(hdr))
    for name, lo_h, hi_h in BANDS:
        m = (truth_a >= lo_h) & (truth_a < hi_h)
        n = int(m.sum())
        if n < 2:
            print(f"{name:<9}{n:>6}   (too few buildings to resample)")
            continue
        (bd, bl, bh, bp), (rd, rl, rh, rp) = boot(err_a[m], err_b[m], args.resamples, rng)
        print(f"{name:<9}{n:>6}{err_a[m].mean():>+10.3f}{err_b[m].mean():>+9.3f}{bd:>+8.3f}"
              f"{f'[{bl:+.3f}, {bh:+.3f}]':>18}{bp:>8.4f}   "
              f"{rmse(err_a[m]):>7.3f}{rmse(err_b[m]):>8.3f}{rd:>+8.3f}"
              f"{f'[{rl:+.3f}, {rh:+.3f}]':>18}{rp:>8.4f}")
        rows.append((name, n, err_a[m].mean(), err_b[m].mean(), bd, bl, bh, bp,
                     rmse(err_a[m]), rmse(err_b[m]), rd, rl, rh, rp))

    print("\nreading: bias delta positive = under-call reduced (toward zero). RMSE delta")
    print("negative = more accurate. A CI spanning zero on a thin band is an instrument")
    print("limit, not a verdict -- say so rather than reporting the point estimate alone.")

    if args.markdown:
        print(f"\n\n| band | n | {la} bias | {lb} bias | delta | 95% CI | p | "
              f"{la} RMSE | {lb} RMSE | delta | 95% CI | p |")
        print("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for r in rows:
            print(f"| **{r[0]}** | {r[1]} | {r[2]:+.3f} | {r[3]:+.3f} | **{r[4]:+.3f}** | "
                  f"[{r[5]:+.3f}, {r[6]:+.3f}] | {r[7]:.4f} | {r[8]:.3f} | {r[9]:.3f} | "
                  f"**{r[10]:+.3f}** | [{r[11]:+.3f}, {r[12]:+.3f}] | {r[13]:.4f} |")


if __name__ == "__main__":
    main()
