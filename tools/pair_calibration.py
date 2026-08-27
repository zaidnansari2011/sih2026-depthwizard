"""Are the error bars on a height DIFFERENCE honest? Measured, by separation.

The viewer's measurement tool (PLAN 6.3) reports `dh +/- hypot(sigma_a, sigma_b)`. That
quadrature is correct only if the two pixels' errors are independent. They are not:
neighbouring pixels come from the same patch of the same forward pass and their errors are
strongly correlated, which makes the quoted bar too WIDE at short range. At long range the
assumption is reasonable.

Nothing had checked this. Our calibration evidence -- ECE, the sigma/error rank
correlation -- is all PER PIXEL, while the number a user actually reads off the screen is a
difference. Those are different claims, and the second is the one the demo makes.

Method: for each of several separation buckets, sample many random point pairs at that
separation, and compare the predicted difference against the true one in units of the
quoted bar. If the bar is honest, |z| <= k should hold for a Gaussian fraction of pairs.
The measured error correlation in each bucket is reported alongside, because it is the
mechanism: coverage above the Gaussian expectation and correlation above zero are the same
fact seen twice.

    python tools/pair_calibration.py --ckpt checkpoints/run02/best.pt --tiles 20
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "depthwizard"))

from depthwizard.model import from_checkpoint  # noqa: E402
from infer import infer_scene, load_image  # noqa: E402

RGB_DIR = ROOT / "data" / "extracted" / "Track1-RGB"
TRUTH_DIR = ROOT / "data" / "extracted" / "Track1-Truth"
SHARDS = ROOT / "data" / "shards"

# Separation buckets in pixels. At 0.3 m GSD these are roughly 1-3 m, 3-15 m, 15-60 m and
# 60-300 m: a rooftop, a building, a block, a neighbourhood.
BUCKETS = [(2, 10), (10, 50), (50, 200), (200, 1000)]
KS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]


def read_tif(p: Path) -> np.ndarray:
    import rasterio
    with rasterio.open(p) as src:
        return src.read(1)


def gaussian_coverage(k: float) -> float:
    return math.erf(k / math.sqrt(2.0))


def sample_pairs(valid: np.ndarray, rmin: float, rmax: float, n: int, rng):
    """Random pairs of valid pixels separated by rmin..rmax pixels.

    Sampling an offset rather than two independent points is what makes the separation
    controlled; uniform pairs over a 1024x1024 tile would nearly all be far apart and the
    short-range behaviour -- the interesting part -- would never be seen.
    """
    ys, xs = np.nonzero(valid)
    if ys.size < 2:
        return None
    H, W = valid.shape
    i = rng.integers(0, ys.size, n)
    ay, ax = ys[i], xs[i]
    r = rng.uniform(rmin, rmax, n)
    th = rng.uniform(0.0, 2.0 * math.pi, n)
    by = np.rint(ay + r * np.sin(th)).astype(np.int64)
    bx = np.rint(ax + r * np.cos(th)).astype(np.int64)
    ok = (by >= 0) & (by < H) & (bx >= 0) & (bx < W)
    ay, ax, by, bx = ay[ok], ax[ok], by[ok], bx[ok]
    if ay.size == 0:
        return None
    ok2 = valid[by, bx]
    return ay[ok2], ax[ok2], by[ok2], bx[ok2]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--tiles", type=int, default=20)
    ap.add_argument("--pairs", type=int, default=200_000, help="attempts per bucket per tile")
    ap.add_argument("--tile-size", type=int, default=518)
    ap.add_argument("--overlap", type=int, default=140)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--precision", default="auto")
    ap.add_argument("--gsd", type=float, default=0.3, help="metres per pixel, for labels")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--out", default=str(ROOT / "out" / "pair_calibration"))
    args = ap.parse_args()

    assign = json.loads((SHARDS / "split.json").read_text())
    regions = {r for r, v in assign.items() if v == args.split}
    tiles = [p.stem[:-4] for p in sorted(RGB_DIR.glob("*_RGB.tif"))
             if p.stem[:-4].rsplit("_", 1)[0] in regions
             and (TRUTH_DIR / f"{p.stem[:-4]}_AGL.tif").exists()]
    rng = np.random.default_rng(args.seed)
    tiles = list(rng.choice(tiles, min(args.tiles, len(tiles)), replace=False))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    from train import pick_precision
    amp_dtype, _, prec = pick_precision(args.precision)

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = from_checkpoint(ck).to(device).eval()
    print(f"checkpoint {args.ckpt} (epoch {ck.get('epoch')})  |  {prec}  |  {device}")
    print(f"{len(tiles)} {args.split} tiles, {len(BUCKETS)} separation buckets")

    acc = {b: [] for b in BUCKETS}          # z scores
    errs = {b: ([], []) for b in BUCKETS}   # paired raw errors, for the correlation
    single = []

    for n, t in enumerate(tiles):
        rgb = load_image(RGB_DIR / f"{t}_RGB.tif")
        pred, sigma = infer_scene(model, rgb, args.tile_size, args.overlap, device,
                                  amp_dtype, args.batch, verbose=False)
        truth = read_tif(TRUTH_DIR / f"{t}_AGL.tif").astype(np.float32)
        if sigma is None:
            raise SystemExit("checkpoint has no uncertainty head; nothing to calibrate")
        valid = np.isfinite(truth) & np.isfinite(pred) & np.isfinite(sigma) & (sigma > 0)
        err = pred - truth
        single.append((err[valid] / sigma[valid]).astype(np.float32))

        for b in BUCKETS:
            got = sample_pairs(valid, b[0], b[1], args.pairs, rng)
            if got is None:
                continue
            ay, ax, by, bx = got
            d_pred = pred[by, bx] - pred[ay, ax]
            d_true = truth[by, bx] - truth[ay, ax]
            s_del = np.hypot(sigma[ay, ax], sigma[by, bx])
            acc[b].append(((d_pred - d_true) / s_del).astype(np.float32))
            errs[b][0].append(err[ay, ax].astype(np.float32))
            errs[b][1].append(err[by, bx].astype(np.float32))
        print(f"  [{n+1}/{len(tiles)}] {t}", flush=True)

    single = np.concatenate(single)
    rows = []
    for b in BUCKETS:
        if not acc[b]:
            continue
        z = np.concatenate(acc[b])
        ea, eb = np.concatenate(errs[b][0]), np.concatenate(errs[b][1])
        rho = float(np.corrcoef(ea, eb)[0, 1]) if ea.size > 2 else float("nan")
        rows.append({
            "bucket_px": list(b),
            "bucket_m": [b[0] * args.gsd, b[1] * args.gsd],
            "n_pairs": int(z.size),
            "error_corr": rho,
            "coverage": {str(k): float(np.mean(np.abs(z) <= k)) for k in KS},
            "z_std": float(z.std()),
        })

    result = {
        "checkpoint": str(args.ckpt), "epoch": ck.get("epoch"), "split": args.split,
        "n_tiles": len(tiles), "gsd_m": args.gsd,
        "expected_coverage": {str(k): gaussian_coverage(k) for k in KS},
        "single_pixel": {
            "n": int(single.size), "z_std": float(single.std()),
            "coverage": {str(k): float(np.mean(np.abs(single) <= k)) for k in KS},
        },
        "by_separation": rows,
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "pair_calibration.json").write_text(json.dumps(result, indent=2))

    L = ["# Calibration of the error bar on a height difference\n",
         f"`{len(tiles)}` whole `{args.split}` tiles, checkpoint "
         f"`{Path(args.ckpt).name}` (epoch {ck.get('epoch')}).\n",
         "The viewer reports `dh +/- hypot(sigma_a, sigma_b)`. That is right only if the two",
         "pixels' errors are independent. This measures whether they are.\n",
         "## Coverage, by how far apart the two points are\n",
         "| separation | pairs | error corr | " + " | ".join(f"+/-{k}s" for k in KS) + " |",
         "|---" * (3 + len(KS)) + "|",
         "| *Gaussian expectation* | - | 0.000 | "
         + " | ".join(f"{gaussian_coverage(k)*100:.1f}%" for k in KS) + " |",
         "| **single pixel** | "
         + f"{result['single_pixel']['n']:,} | - | "
         + " | ".join(f"{result['single_pixel']['coverage'][str(k)]*100:.1f}%" for k in KS) + " |"]
    for r in rows:
        lab = f"{r['bucket_m'][0]:.0f}-{r['bucket_m'][1]:.0f} m"
        L.append(f"| {lab} | {r['n_pairs']:,} | {r['error_corr']:+.3f} | "
                 + " | ".join(f"{r['coverage'][str(k)]*100:.1f}%" for k in KS) + " |")
    L += ["",
          "## Reading it\n",
          "Coverage **above** the Gaussian row means the quoted bar is too wide -- the tool is",
          "claiming more doubt than it has. Coverage **below** it means the bar is too narrow,",
          "which is the dangerous direction for an instrument someone trusts.\n",
          "The error-correlation column is the mechanism. Where it is strongly positive the",
          "two errors cancel in the difference, so the true spread of `dh` is narrower than",
          "quadrature predicts and coverage runs high. It should fall toward zero as the",
          "points separate, and quadrature should become correct.\n"]
    (out / "pair_calibration.md").write_text("\n".join(L))

    print()
    print(f"{'separation':>14}  {'corr':>7}  " + "  ".join(f"+/-{k}s" for k in KS))
    print(f"{'expected':>14}  {0.0:>7.3f}  "
          + "  ".join(f"{gaussian_coverage(k)*100:5.1f}%" for k in KS))
    print(f"{'single px':>14}  {'-':>7}  "
          + "  ".join(f"{result['single_pixel']['coverage'][str(k)]*100:5.1f}%" for k in KS))
    for r in rows:
        lab = f"{r['bucket_m'][0]:.0f}-{r['bucket_m'][1]:.0f} m"
        print(f"{lab:>14}  {r['error_corr']:>+7.3f}  "
              + "  ".join(f"{r['coverage'][str(k)]*100:5.1f}%" for k in KS))
    print(f"\nwrote {out/'pair_calibration.json'} and {out/'pair_calibration.md'}")


if __name__ == "__main__":
    main()
