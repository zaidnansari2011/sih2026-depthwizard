"""How far does stock Depth Anything V2 get us, with no fine-tuning at all?

    python tools/zero_shot_baseline.py --fit-tiles 40 --eval-tiles 60

This is the control experiment the whole project is measured against. PLAN.md section 5
rests on the claim that DA-V2 needs adapting for nadir imagery; that claim is worth
nothing as an assertion and everything as a measured number, so we measure it.

Three numbers, and the gap between them is the argument
------------------------------------------------------
  raw            DA-V2 output compared directly to metres AGL. Meaningless in absolute
                 terms -- relative depth has arbitrary units -- but its CORRELATION with
                 truth tells us whether the pretrained features see height at all.

  global affine  one scale and shift, fitted on TRAIN tiles, applied unchanged to TEST
                 tiles. This is the honest zero-shot baseline: it is what you could
                 actually deploy, because it needs no ground truth for the scene you
                 are predicting.

  oracle affine  scale and shift refitted per test tile using that tile's own truth.
                 Not deployable -- it needs the answer to compute the answer -- but it
                 bounds what any purely linear correction could ever achieve. Fine-tuning
                 has to beat the global number, and should be judged against the oracle.

Reporting the oracle as if it were a result would be the easiest way to fool ourselves
and, at a demo, the easiest way to be caught.
"""
from __future__ import annotations

import argparse
import json
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from depthwizard.metrics import height_metrics, terrain_category_with_relief, CLS_NAMES  # noqa: E402
from depthwizard.model import DEFAULT_MODEL, build  # noqa: E402
from infer import infer_scene, load_image  # noqa: E402

EXTRACTED = ROOT / "data" / "extracted"
SHARDS = ROOT / "data" / "shards"
RGB_DIR = EXTRACTED / "Track1-RGB"
TRUTH_DIR = EXTRACTED / "Track1-Truth"


def read_tif(p: Path) -> np.ndarray:
    import rasterio
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with rasterio.open(p) as src:
            return src.read(1)


def tiles_for(split: str, assign: dict) -> list[str]:
    regions = {r for r, v in assign.items() if v == split}
    out = []
    for p in sorted(RGB_DIR.glob("*_RGB.tif")):
        tile = p.stem[:-4]                       # strip "_RGB"
        if tile.rsplit("_", 1)[0] in regions and (TRUTH_DIR / f"{tile}_AGL.tif").exists():
            out.append(tile)
    return out


def predict(model, tile: str, device, amp_dtype, args):
    rgb = load_image(RGB_DIR / f"{tile}_RGB.tif")
    pred, sigma = infer_scene(model, rgb, args.tile, args.overlap, device, amp_dtype,
                              args.batch, verbose=False)
    truth = read_tif(TRUTH_DIR / f"{tile}_AGL.tif").astype(np.float32)
    cls_path = TRUTH_DIR / f"{tile}_CLS.tif"
    cls = read_tif(cls_path) if cls_path.exists() else None
    mask = np.isfinite(truth) & np.isfinite(pred)
    return pred, sigma, truth, cls, mask


def fit_affine(preds, truths):
    """Least squares scale and shift mapping prediction onto metres."""
    p = np.concatenate(preds)
    t = np.concatenate(truths)
    A = np.stack([p, np.ones_like(p)], 1)
    coef, *_ = np.linalg.lstsq(A, t, rcond=None)
    return float(coef[0]), float(coef[1])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-id", default=DEFAULT_MODEL)
    ap.add_argument("--fit-tiles", type=int, default=40, help="train tiles used to fit the affine")
    ap.add_argument("--eval-tiles", type=int, default=60, help="test tiles to score")
    ap.add_argument("--tile", type=int, default=518)
    ap.add_argument("--overlap", type=int, default=140)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--precision", default="auto")
    ap.add_argument("--out", default=str(ROOT / "out" / "zero_shot_baseline.json"))
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    split_path = SHARDS / "split.json"
    if not split_path.exists():
        raise SystemExit(f"{split_path} missing. Run: python tools/prepare_data.py shard")
    assign = json.loads(split_path.read_text())

    rng = np.random.default_rng(args.seed)
    fit_pool, eval_pool = tiles_for("train", assign), tiles_for("test", assign)
    fit_tiles = list(rng.choice(fit_pool, min(args.fit_tiles, len(fit_pool)), replace=False))
    eval_tiles = list(rng.choice(eval_pool, min(args.eval_tiles, len(eval_pool)), replace=False))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    from train import pick_precision
    amp_dtype, _, prec = pick_precision(args.precision)
    model = build(model_id=args.model_id, height_scale=1.0).to(device).eval()

    print(f"zero-shot baseline  |  {args.model_id}  |  {prec}  |  {device}")
    print(f"  fit on {len(fit_tiles)} train tiles, evaluate on {len(eval_tiles)} test tiles")
    print(f"  train regions {sum(v=='train' for v in assign.values())}, "
          f"test regions {sum(v=='test' for v in assign.values())} -- no region appears in both")

    # ---------------------------------------------------------------- fit
    fp, ft = [], []
    for i, t in enumerate(fit_tiles):
        pred, _, truth, _, mask = predict(model, t, device, amp_dtype, args)
        # Thin before pooling: 40 tiles of a million pixels is 40M floats we do not need
        # for a two-parameter fit.
        fp.append(pred[mask][::53])
        ft.append(truth[mask][::53])
        if (i + 1) % 10 == 0:
            print(f"    fit {i+1}/{len(fit_tiles)}")
    scale, shift = fit_affine(fp, ft)
    print(f"\n  global affine fitted on TRAIN: height_m = {scale:.4f} * dav2 {shift:+.3f}")

    # ---------------------------------------------------------------- evaluate
    raw_p, glob_p, orac_p, all_t, all_c = [], [], [], [], []
    per_terrain = defaultdict(lambda: {"p": [], "t": []})
    per_tile = []

    for i, t in enumerate(eval_tiles):
        pred, _, truth, cls, mask = predict(model, t, device, amp_dtype, args)
        p, gt = pred[mask], truth[mask]
        g = p * scale + shift
        A = np.stack([p, np.ones_like(p)], 1)
        c, *_ = np.linalg.lstsq(A, gt, rcond=None)
        o = p * c[0] + c[1]

        step = 7
        raw_p.append(p[::step]); glob_p.append(g[::step]); orac_p.append(o[::step])
        all_t.append(gt[::step])
        if cls is not None:
            all_c.append(cls[mask][::step])
            cat = terrain_category_with_relief(cls, truth)
            per_terrain[cat]["p"].append(g[::step])
            per_terrain[cat]["t"].append(gt[::step])

        per_tile.append({
            "tile": t,
            "rmse_global": float(np.sqrt(((g - gt) ** 2).mean())),
            "rmse_oracle": float(np.sqrt(((o - gt) ** 2).mean())),
            "corr": float(np.corrcoef(p, gt)[0, 1]) if p.std() > 0 else float("nan"),
        })
        if (i + 1) % 10 == 0:
            print(f"    eval {i+1}/{len(eval_tiles)}")

    T = np.concatenate(all_t)
    C = np.concatenate(all_c) if all_c else None
    results = {
        "raw": height_metrics(np.concatenate(raw_p), T, cls=C).to_dict(),
        "global_affine": height_metrics(np.concatenate(glob_p), T, cls=C).to_dict(),
        "oracle_affine": height_metrics(np.concatenate(orac_p), T, cls=C).to_dict(),
    }

    def line(name, m):
        return (f"  {name:16s} RMSE {m['rmse']:7.3f}  MAE {m['mae']:7.3f}  "
                f"r {m['corr']:6.3f}  bias {m['bias']:+7.3f}  p90|e| {m['p90_ae']:7.3f}")

    print(f"\n{'='*78}\nZERO-SHOT DA-V2 ON {len(eval_tiles)} HELD-OUT TEST TILES  (metres)\n{'='*78}")
    print(line("raw", results["raw"]) + "   <- units are arbitrary; read r, not RMSE")
    print(line("global affine", results["global_affine"]) + "   <- the deployable baseline")
    print(line("oracle affine", results["oracle_affine"]) + "   <- upper bound, needs truth")

    print("\nper semantic class, global affine:")
    for name, m in results["global_affine"].get("per_class", {}).items():
        print(f"  {name:14s} RMSE {m['rmse']:7.3f}  MAE {m['mae']:7.3f}  "
              f"bias {m['bias']:+7.3f}  n={m['n']:,}")

    if per_terrain:
        print("\nper terrain category, global affine (the ISRO stability requirement):")
        terrain = {}
        for cat, d in sorted(per_terrain.items()):
            m = height_metrics(np.concatenate(d["p"]), np.concatenate(d["t"]))
            terrain[cat] = m.to_dict()
            print(f"  {cat:14s} RMSE {m['rmse']:7.3f}  MAE {m['mae']:7.3f}  n={m['n_valid']:,}")
        results["per_terrain"] = terrain

    rmses = np.array([t["rmse_global"] for t in per_tile])
    corrs = np.array([t["corr"] for t in per_tile])
    print(f"\nspread across tiles: RMSE {rmses.min():.2f} to {rmses.max():.2f} m "
          f"(median {np.median(rmses):.2f}), correlation median {np.median(corrs):.3f}")
    worst = sorted(per_tile, key=lambda d: -d["rmse_global"])[:3]
    print("worst tiles: " + ", ".join(f"{w['tile']} ({w['rmse_global']:.1f} m)" for w in worst))

    print(f"\n  >>> the bar for fine-tuning: beat {results['global_affine']['rmse']:.3f} m RMSE")
    print(f"  >>> and it should approach or beat the oracle's "
          f"{results['oracle_affine']['rmse']:.3f} m")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "model_id": args.model_id,
        "affine": {"scale": scale, "shift": shift},
        "n_fit_tiles": len(fit_tiles), "n_eval_tiles": len(eval_tiles),
        "fit_tiles": fit_tiles, "eval_tiles": eval_tiles,
        "results": results, "per_tile": per_tile,
    }, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
