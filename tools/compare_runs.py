"""Compare training runs side by side. The ablation table, generated rather than typed.

    python tools/compare_runs.py D:/sih2026/checkpoints/run01 D:/sih2026/checkpoints/run02
    python tools/compare_runs.py --label "beta=0" run01 --label "beta=0.5" run02

Reads each run's train_log.jsonl -- which is append-only, so a killed run still compares --
and prints the numbers that decide whether a change earned its place.

Why per-class is the headline and not RMSE
------------------------------------------
On DFC2019 a global RMSE is close to useless on its own. Ground is 74% of pixels and easy;
buildings are 13% and carry over 90% of the squared error. A change can improve global
RMSE by getting slightly better at ground while leaving the actual failure untouched, so
this tool always shows where the error *lives*, not just how much of it there is.

It also reports calibration, because a change that improves the mean by breaking the
uncertainty is a trade, not a win -- and beta-NLL in particular is expected to push in
exactly that direction.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_run(d: Path) -> dict:
    log = d / "train_log.jsonl"
    if not log.exists():
        return {}
    rows = [json.loads(l) for l in log.read_text(encoding="utf-8").splitlines() if l.strip()]
    val = [r for r in rows if r.get("t") == "val"]
    train = [r for r in rows if r.get("t") == "train"]
    if not val:
        return {"name": d.name, "val": [], "train": train}
    best = min(val, key=lambda r: r["rmse"])
    return {"name": d.name, "val": val, "train": train, "best": best, "last": val[-1]}


def arrow(new: float, old: float, lower_is_better: bool = True) -> str:
    if old is None or new is None:
        return ""
    d = new - old
    if abs(d) < 1e-9:
        return "  ="
    good = (d < 0) if lower_is_better else (d > 0)
    return f"  {'+' if d > 0 else ''}{d:.3f} {'better' if good else 'WORSE'}"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="checkpoint directories")
    ap.add_argument("--labels", nargs="*", default=None, help="display names, in order")
    ap.add_argument("--markdown", action="store_true", help="emit a markdown table")
    args = ap.parse_args()

    runs = [load_run(Path(r)) for r in args.runs]
    runs = [r for r in runs if r]
    if not runs:
        raise SystemExit("no runs with a train_log.jsonl found")
    labels = args.labels or [r["name"] for r in runs]
    labels += [r["name"] for r in runs[len(labels):]]

    print(f"\n{'='*84}\nRUN COMPARISON — best validation epoch of each\n{'='*84}")
    hdr = f"{'run':<16} {'ep':>3} {'RMSE':>8} {'MAE':>8} {'med|e|':>8} {'r':>7} {'bias':>8} {'ECE':>7} {'rank-r':>7}"
    print(hdr)
    print("-" * len(hdr))
    ref = None
    for lab, r in zip(labels, runs):
        if not r.get("best"):
            print(f"{lab:<16}  (no validation epoch completed; {len(r['train'])} train logs)")
            continue
        b = r["best"]
        print(f"{lab:<16} {b['epoch']:>3} {b['rmse']:8.3f} {b['mae']:8.3f} "
              f"{b.get('median_ae', float('nan')):8.3f} {b['corr']:7.3f} {b['bias']:+8.3f} "
              f"{b.get('ece', float('nan')):7.4f} {b.get('sigma_rank_corr', float('nan')):+7.3f}")
        if ref is None:
            ref = b

    # ---------------------------------------------------------------- per class
    print(f"\n{'-'*84}\nWHERE THE ERROR LIVES  (best epoch, per semantic class)\n{'-'*84}")
    classes = ["ground", "vegetation", "building", "water", "bridge"]
    print(f"{'class':<12}" + "".join(f"{lab:>22}" for lab in labels))
    for c in classes:
        cells = []
        for r in runs:
            m = (r.get("best") or {}).get("per_class", {}).get(c)
            cells.append(f"{m['rmse']:8.2f} /{m['bias']:+7.2f}" if m else " " * 16)
        print(f"{c:<12}" + "".join(f"{x:>22}" for x in cells))
    print(f"{'':12}" + "".join(f"{'RMSE / bias, m':>22}" for _ in labels))

    # Share of squared error, which is what a single RMSE actually reports.
    print(f"\n{'-'*84}\nSHARE OF TOTAL SQUARED ERROR\n{'-'*84}")
    print(f"{'class':<12}" + "".join(f"{lab:>22}" for lab in labels))
    for c in classes:
        cells = []
        for r in runs:
            pc = (r.get("best") or {}).get("per_class", {})
            if not pc or c not in pc:
                cells.append("")
                continue
            tot = sum(m["rmse"] ** 2 * m["n"] for m in pc.values())
            cells.append(f"{100 * pc[c]['rmse']**2 * pc[c]['n'] / tot:6.1f}%")
        print(f"{c:<12}" + "".join(f"{x:>22}" for x in cells))

    # ---------------------------------------------------------------- verdict
    if len(runs) >= 2 and all(r.get("best") for r in runs[:2]):
        a, b = runs[0]["best"], runs[1]["best"]
        print(f"\n{'-'*84}\nVERDICT: {labels[1]} vs {labels[0]}\n{'-'*84}")
        print(f"  RMSE      {a['rmse']:7.3f} -> {b['rmse']:7.3f}{arrow(b['rmse'], a['rmse'])}")
        print(f"  MAE       {a['mae']:7.3f} -> {b['mae']:7.3f}{arrow(b['mae'], a['mae'])}")
        ab = a.get("per_class", {}).get("building")
        bb = b.get("per_class", {}).get("building")
        if ab and bb:
            print(f"  building RMSE {ab['rmse']:7.3f} -> {bb['rmse']:7.3f}"
                  f"{arrow(bb['rmse'], ab['rmse'])}")
            print(f"  building bias {ab['bias']:+7.3f} -> {bb['bias']:+7.3f}"
                  f"   (toward zero is the goal; this is the whole point of the change)")
        if "ece" in a and "ece" in b:
            print(f"  ECE       {a['ece']:7.4f} -> {b['ece']:7.4f}{arrow(b['ece'], a['ece'])}")
            print(f"  sigma rank-r {a.get('sigma_rank_corr',0):+7.3f} -> "
                  f"{b.get('sigma_rank_corr',0):+7.3f}"
                  f"{arrow(b.get('sigma_rank_corr',0), a.get('sigma_rank_corr',0), lower_is_better=False)}")
            print("\n  Read ECE and rank-r together with RMSE. A change that improves the mean by")
            print("  degrading the uncertainty is a trade to be reported, not a win to be claimed.")

    # ---------------------------------------------------------------- trajectory
    print(f"\n{'-'*84}\nVALIDATION RMSE BY EPOCH  (flat means the objective, not the budget, is the limit)\n{'-'*84}")
    for lab, r in zip(labels, runs):
        if r.get("val"):
            print(f"  {lab:<14} " + "  ".join(f"{v['rmse']:.2f}" for v in r["val"]))

    if args.markdown:
        print(f"\n\n## Ablation\n")
        print("| Run | Epoch | RMSE | MAE | building RMSE | building bias | ECE | σ rank-r |")
        print("|---|---|---|---|---|---|---|---|")
        for lab, r in zip(labels, runs):
            b = r.get("best")
            if not b:
                continue
            bd = b.get("per_class", {}).get("building", {})
            print(f"| {lab} | {b['epoch']} | {b['rmse']:.3f} m | {b['mae']:.3f} m | "
                  f"{bd.get('rmse', float('nan')):.2f} m | {bd.get('bias', float('nan')):+.2f} m | "
                  f"{b.get('ece', float('nan')):.4f} | {b.get('sigma_rank_corr', float('nan')):+.3f} |")


if __name__ == "__main__":
    main()
