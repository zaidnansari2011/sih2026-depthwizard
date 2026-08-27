"""Score a checkpoint on a region-disjoint split and write a report.

Defaults to --split val. The test split is held out on day one and is not a
development metric: score it only for a final, reported number, because every
look at it leaks a little of its independence (prepare_data.py, standing rule 5).

    python tools/evaluate.py --ckpt D:/sih2026/checkpoints/run01/best.pt --tiles 80

Evaluates on **whole 1024x1024 tiles**, not on training crops. That distinction is not
pedantic: crop-level numbers are measured on the same geometry the model was optimised
for, whereas a deployed system is handed a whole scene and has to stitch. Sliding-window
inference with cosine blending is part of what we are scoring, so it belongs inside the
evaluation.

Produces
  report.md    a table that can go straight into the submission
  metrics.json every number, for plots and for the record
  calibration.png / error_hist.png   the figures that make the uncertainty claim legible

The comparison is always against the zero-shot global-affine baseline from
tools/zero_shot_baseline.py. A model that does not beat that has not earned its place.
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from depthwizard.metrics import (  # noqa: E402
    height_metrics, calibration_curve, expected_calibration_error,
    uncertainty_error_correlation, terrain_category_with_relief,
    building_instances, building_wise_metrics,
)
from depthwizard.model import from_checkpoint  # noqa: E402
from infer import infer_scene, load_image  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RGB_DIR = ROOT / "data" / "extracted" / "Track1-RGB"
TRUTH_DIR = ROOT / "data" / "extracted" / "Track1-Truth"
SHARDS = ROOT / "data" / "shards"


def read_tif(p: Path):
    import rasterio
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with rasterio.open(p) as src:
            return src.read(1)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tiles", type=int, default=80)
    ap.add_argument("--tile-size", type=int, default=518)
    ap.add_argument("--overlap", type=int, default=140)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--precision", default="auto")
    ap.add_argument("--tta", action="store_true",
                    help="D4 test-time augmentation; ~8x slower")
    ap.add_argument("--baseline", default=None,
                    help="zero-shot json to compare against; defaults to the one "
                         "matching --split")
    ap.add_argument("--out", default=str(ROOT / "out" / "eval"))
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--split", default="val", choices=["val", "test"],
                    help="region split to score. val for development; test only for a "
                         "final reported number (see module docstring)")
    args = ap.parse_args()
    if args.baseline is None:
        cand = ROOT / "out" / f"zero_shot_baseline_{args.split}.json"
        if not cand.exists() and args.split == "test":
            cand = ROOT / "out" / "zero_shot_baseline.json"   # pre-split-flag filename
        args.baseline = str(cand)
    if args.split == "test":
        print("!! scoring the HELD-OUT TEST split.")
        print("!! Use it for a reported number only, never to pick between checkpoints.")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    assign = json.loads((SHARDS / "split.json").read_text())
    eval_regions = {r for r, v in assign.items() if v == args.split}
    tiles = [p.stem[:-4] for p in sorted(RGB_DIR.glob("*_RGB.tif"))
             if p.stem[:-4].rsplit("_", 1)[0] in eval_regions
             and (TRUTH_DIR / f"{p.stem[:-4]}_AGL.tif").exists()]
    rng = np.random.default_rng(args.seed)
    tiles = list(rng.choice(tiles, min(args.tiles, len(tiles)), replace=False))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    from train import pick_precision
    amp_dtype, _, prec = pick_precision(args.precision)

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = from_checkpoint(ck).to(device).eval()
    print(f"checkpoint {args.ckpt}  epoch {ck.get('epoch')}  |  {prec}  |  {device}")
    print(f"evaluating on {len(tiles)} whole {args.split} tiles "
          f"from {len(eval_regions)} region-disjoint {args.split} regions")

    P, T, S, C = [], [], [], []
    per_terrain = defaultdict(lambda: {"p": [], "t": []})
    per_tile = []
    # One height per building, accumulated at full resolution before subsampling --
    # connected components cannot be recovered from a thinned array.
    BP, BT = [], []

    for i, t in enumerate(tiles):
        rgb = load_image(RGB_DIR / f"{t}_RGB.tif")
        pred, sigma = infer_scene(model, rgb, args.tile_size, args.overlap, device,
                                  amp_dtype, args.batch, verbose=False, tta=args.tta)
        truth = read_tif(TRUTH_DIR / f"{t}_AGL.tif").astype(np.float32)
        cls_p = TRUTH_DIR / f"{t}_CLS.tif"
        cls = read_tif(cls_p) if cls_p.exists() else None
        mask = np.isfinite(truth) & np.isfinite(pred)

        step = 7                      # thin for pooling; 80 Mpx does not fit comfortably
        P.append(pred[mask][::step])
        T.append(truth[mask][::step])
        if sigma is not None:
            S.append(sigma[mask][::step])
        if cls is not None:
            C.append(cls[mask][::step])
            cat = terrain_category_with_relief(cls, truth)
            per_terrain[cat]["p"].append(pred[mask][::step])
            per_terrain[cat]["t"].append(truth[mask][::step])
            bp, bt = building_instances(pred, truth, cls)
            if bp.size:
                BP.append(bp)
                BT.append(bt)

        e = pred[mask] - truth[mask]
        per_tile.append({"tile": t, "rmse": float(np.sqrt((e ** 2).mean())),
                         "mae": float(np.abs(e).mean()),
                         "sigma_mean": float(sigma[mask].mean()) if sigma is not None else None})
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(tiles)}")

    p, t = np.concatenate(P), np.concatenate(T)
    s = np.concatenate(S) if S else None
    c = np.concatenate(C) if C else None
    m = height_metrics(p, t, cls=c)

    results = {"overall": m.to_dict(), "n_tiles": len(tiles), "checkpoint": str(args.ckpt),
               "epoch": ck.get("epoch"), "tta": args.tta,
               "split": args.split}
    if s is not None:
        results["ece"] = expected_calibration_error(p, t, s)
        results["sigma_rank_corr"] = uncertainty_error_correlation(p, t, s)
        results["sigma_mean"] = float(s.mean())
        results["calibration"] = [{"k": k, "expected": e, "observed": o}
                                  for k, e, o in calibration_curve(p, t, s)]

    terrain = {}
    for cat, d in sorted(per_terrain.items()):
        tm = height_metrics(np.concatenate(d["p"]), np.concatenate(d["t"]))
        terrain[cat] = tm.to_dict()
    results["per_terrain"] = terrain
    results["per_tile"] = per_tile

    bw = building_wise_metrics(np.concatenate(BP), np.concatenate(BT)) if BP else {
        "n_buildings": 0}
    results["building_wise"] = bw

    # Stratify by true height. A single per-building RMSE hides whether the model is good
    # on the 4 m sheds that dominate an American suburb and useless on the tall stock that
    # dominates an Asian city -- which is exactly the comparison we want to make.
    bw_strata = []
    if BP:
        bo, bt_ = np.concatenate(BP), np.concatenate(BT)
        np.savez_compressed(out / "per_building.npz", ours=bo, truth=bt_)
        for lo, hi in ((0, 3), (3, 6), (6, 10), (10, 20), (20, 1e9)):
            sel = (bt_ >= lo) & (bt_ < hi)
            if sel.sum() < 20:
                continue
            st = building_wise_metrics(bo[sel], bt_[sel])
            st["band"] = f"{lo}-{hi:g} m" if hi < 1e9 else f">{lo} m"
            bw_strata.append(st)
    results["building_wise_by_height"] = bw_strata

    base = None
    if Path(args.baseline).exists():
        base = json.loads(Path(args.baseline).read_text())

    # ------------------------------------------------------------------ report
    L = []
    L.append(f"# DepthWizard — {args.split} split results\n")
    L.append(f"Checkpoint `{Path(args.ckpt).name}` (epoch {ck.get('epoch')}), "
             f"**{len(tiles)} whole 1024x1024 tiles** from {len(eval_regions)} regions "
             f"that appear in no training split.\n")
    L.append("Inference is sliding-window with cosine blending, the same path a deployed "
             "system would take — not crop-level scoring.\n")

    L.append("\n## Accuracy\n")
    L.append("| Model | RMSE | MAE | r | bias | p90&#124;e&#124; |")
    L.append("|---|---|---|---|---|---|")
    if base:
        b = base["results"]["global_affine"]
        L.append(f"| Zero-shot DA-V2 + global affine *(baseline)* | {b['rmse']:.3f} m | "
                 f"{b['mae']:.3f} m | {b['corr']:.3f} | {b['bias']:+.3f} m | {b['p90_ae']:.3f} m |")
        o = base["results"]["oracle_affine"]
        L.append(f"| Zero-shot + *oracle* affine *(not deployable)* | {o['rmse']:.3f} m | "
                 f"{o['mae']:.3f} m | {o['corr']:.3f} | {o['bias']:+.3f} m | {o['p90_ae']:.3f} m |")
    L.append(f"| **DepthWizard (fine-tuned{' + TTA' if args.tta else ''})** | **{m.rmse:.3f} m** | **{m.mae:.3f} m** | "
             f"**{m.corr:.3f}** | {m.bias:+.3f} m | {m.p90_ae:.3f} m |")

    if base:
        b = base["results"]["global_affine"]
        d = 100 * (b["rmse"] - m.rmse) / b["rmse"]
        verdict = f"**{d:+.1f}% RMSE vs the deployable baseline.**"
        if m.rmse >= b["rmse"]:
            verdict += (" That is worse than a two-parameter linear correction — "
                        "the fine-tuning is not earning its place yet.")
        L.append(f"\n{verdict}\n")

    if bw.get("n_buildings"):
        L.append("\n## Per building — the metric the literature reports\n")
        L.append("One median height per building instance, on both sides, for buildings of "
                 "at least 25 m2. Our per-pixel building RMSE is **not** comparable to "
                 "published figures: a pixel score is dominated by roof edges and lets a "
                 "single large building outvote a whole neighbourhood.\n")
        L.append(f"| | value |")
        L.append("|---|---|")
        L.append(f"| buildings scored | {bw['n_buildings']:,} |")
        L.append(f"| **RMSE** | **{bw['rmse']:.3f} m** |")
        L.append(f"| MAE | {bw['mae']:.3f} m |")
        L.append(f"| bias | {bw['bias']:+.3f} m |")
        L.append(f"| median abs error | {bw['median_ae']:.3f} m |")
        L.append(f"| r | {bw['corr']:+.3f} |")
        L.append(f"| true building height (median / p90) | "
                 f"{bw['truth_median_h']:.1f} m / {bw['truth_p90_h']:.1f} m |")
        L.append("\n**Reference points.** GlobalBuildingAtlas (ESSD 2025) reports "
                 "per-building height RMSE of **5.9 m over Asia** and 5.5 m globally, "
                 "using HTC-DC Net on 3 m PlanetScope imagery — the closest published peer, "
                 "and a generalisation number like ours. HTC-DC Net's own DFC2019 "
                 "building-wise figure is 2.3-2.8 m, but on a **randomly split** crop "
                 "protocol where a test crop's neighbour is in training; ours is "
                 "region-disjoint and the two are not the same measurement.\n")
        if bw_strata:
            L.append("\n**By true building height.** A single figure hides whether we are "
                     "good on the low-rise stock that dominates an American suburb and "
                     "useless on the tall stock that dominates an Asian city — which is "
                     "exactly what a comparison against an Asia-wide number turns on.\n")
            L.append("| true height | buildings | RMSE | bias | RMSE / median height |")
            L.append("|---|---|---|---|---|")
            for st in bw_strata:
                ratio = st["rmse"] / st["truth_median_h"] if st["truth_median_h"] else float("nan")
                L.append(f"| {st['band']} | {st['n_buildings']:,} | {st['rmse']:.2f} m | "
                         f"{st['bias']:+.2f} m | {ratio:.2f}x |")

        if bw["rmse"] > 0 and bw["truth_median_h"] > 0:
            ratio = bw["rmse"] / bw["truth_median_h"]
            L.append(f"Our per-building RMSE is **{ratio:.2f}x the median building "
                     f"height** in this split. Below 1.0 the model is doing better than "
                     f"guessing a constant; well below is where it becomes useful.\n")

    L.append("\n## Per semantic class\n")
    L.append("| Class | RMSE | MAE | bias | pixels |")
    L.append("|---|---|---|---|---|")
    for name, cm in m.per_class.items():
        L.append(f"| {name} | {cm['rmse']:.3f} m | {cm['mae']:.3f} m | "
                 f"{cm['bias']:+.3f} m | {cm['n']:,} |")

    if terrain:
        L.append("\n## Per terrain category\n")
        L.append("ISRO's problem statement asks for stability across terrain types, so a "
                 "single global RMSE is the wrong number to look at alone.\n")
        L.append("| Terrain | RMSE | MAE | bias | pixels |")
        L.append("|---|---|---|---|---|")
        for cat, tm in terrain.items():
            L.append(f"| {cat} | {tm['rmse']:.3f} m | {tm['mae']:.3f} m | "
                     f"{tm['bias']:+.3f} m | {tm['n_valid']:,} |")
        sp = max(t2["rmse"] for t2 in terrain.values()) - min(t2["rmse"] for t2 in terrain.values())
        L.append(f"\nSpread across terrain types: **{sp:.3f} m**. Lower is the claim.\n")

    if s is not None:
        L.append("\n## Uncertainty calibration (differentiator 6.1)\n")
        L.append("If we claim 68% of errors fall inside 1σ, that had better be true.\n")
        L.append("| k | expected coverage | observed | Δ |")
        L.append("|---|---|---|---|")
        for row in results["calibration"]:
            L.append(f"| {row['k']:.1f} | {100*row['expected']:.1f}% | "
                     f"{100*row['observed']:.1f}% | {100*(row['observed']-row['expected']):+.1f} pp |")
        L.append(f"\n- **Expected calibration error: {results['ece']:.4f}** (lower is better)")
        L.append(f"- **σ vs |error| rank correlation: {results['sigma_rank_corr']:+.3f}** — "
                 "does σ actually track where the model is wrong?")
        L.append(f"- Mean predicted σ: {results['sigma_mean']:.3f} m\n")
        L.append("> The rank correlation has a ceiling well below 1.0 even for a perfect "
                 "model: |error| is a single noisy draw from the predicted distribution, "
                 "so a perfectly calibrated σ still only reaches ≈0.44. Read it against "
                 "that ceiling, not against 1.0.\n")

    r = np.array([x["rmse"] for x in per_tile])
    L.append(f"\n## Spread across tiles\n")
    L.append(f"RMSE ranges **{r.min():.2f} – {r.max():.2f} m** (median {np.median(r):.2f} m). ")
    worst = sorted(per_tile, key=lambda d: -d["rmse"])[:5]
    L.append("Worst: " + ", ".join(f"`{w['tile']}` ({w['rmse']:.2f} m)" for w in worst) + ".\n")

    (out / "report.md").write_text("\n".join(L), encoding="utf-8")
    (out / "metrics.json").write_text(json.dumps(results, indent=2))

    # ------------------------------------------------------------------ figures
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if s is not None:
            fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
            ks = [r0["k"] for r0 in results["calibration"]]
            exp = [100 * r0["expected"] for r0 in results["calibration"]]
            obs = [100 * r0["observed"] for r0 in results["calibration"]]
            ax[0].plot(exp, exp, "--", color="#888", label="perfect calibration")
            ax[0].plot(exp, obs, "o-", color="#4da3ff", label="DepthWizard")
            for k, e, o in zip(ks, exp, obs):
                ax[0].annotate(f"{k:g}σ", (e, o), textcoords="offset points", xytext=(6, -10),
                               fontsize=8, color="#555")
            ax[0].set_xlabel("expected coverage (%)")
            ax[0].set_ylabel("observed coverage (%)")
            ax[0].set_title(f"Uncertainty calibration  (ECE {results['ece']:.4f})")
            ax[0].legend()
            ax[0].grid(alpha=.3)

            # Does error actually grow with predicted sigma? Binned, because a scatter of
            # 10M points is unreadable and hides the trend.
            qs = np.quantile(s, np.linspace(0, 1, 11))
            xs, ys = [], []
            for lo, hi in zip(qs[:-1], qs[1:]):
                sel = (s >= lo) & (s < hi)
                if sel.sum() > 100:
                    xs.append(float(s[sel].mean()))
                    ys.append(float(np.abs(p[sel] - t[sel]).mean()))
            ax[1].plot(xs, ys, "o-", color="#ffb454")
            lim = max(max(xs, default=1), max(ys, default=1))
            ax[1].plot([0, lim], [0, lim], "--", color="#888", label="y = x")
            ax[1].set_xlabel("predicted σ (m), decile mean")
            ax[1].set_ylabel("mean |error| (m)")
            ax[1].set_title(f"σ tracks error  (rank r {results['sigma_rank_corr']:+.3f})")
            ax[1].legend()
            ax[1].grid(alpha=.3)
            fig.tight_layout()
            fig.savefig(out / "calibration.png", dpi=140)
            plt.close(fig)

        fig, ax = plt.subplots(figsize=(6.5, 4))
        err = p - t
        ax.hist(err, bins=200, range=(-25, 25), color="#4da3ff", alpha=.85)
        ax.axvline(0, color="#888", ls="--")
        ax.set_xlabel("error (m)   predicted − true")
        ax.set_ylabel("pixels")
        ax.set_title(f"Error distribution  (RMSE {m.rmse:.2f} m, bias {m.bias:+.2f} m)")
        ax.grid(alpha=.3)
        fig.tight_layout()
        fig.savefig(out / "error_hist.png", dpi=140)
        plt.close(fig)
        print(f"figures -> {out}")
    except ImportError:
        print("matplotlib not installed; skipped figures")

    print("\n" + "\n".join(L[:40]))
    print(f"\nwrote {out/'report.md'} and {out/'metrics.json'}")


if __name__ == "__main__":
    main()
