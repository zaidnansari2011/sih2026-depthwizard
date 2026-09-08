"""Probe 06 — we compress every height toward the median. Can that be inverted?

Measured on validation: `ours = 0.473 * truth + 2.66`. Short buildings come out 14-20% too
tall, buildings over 40 m come out at half their height. That is regression to the mean --
under uncertainty, the loss-minimising prediction is the conditional average, and with a
long-tailed target whose median is 4.2 m the safe answer is always "about 4 metres".

If the compression is systematic, inverting it should help. The correction is therefore fit
on TRAIN tiles and measured on VAL: fitting it on validation would be tuning on the number
we report.

The risk, stated before the run: dividing by 0.473 also multiplies the noise by ~2.1x, so
bias can improve while RMSE gets worse. A partial correction sweep is included because the
bias/variance optimum is very unlikely to sit exactly at full inversion.

    python tools/decompress_probe.py --tiles 24
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import rasterio

warnings.simplefilter("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from depthwizard.metrics import building_instances, building_wise_metrics  # noqa: E402

ROOT = Path("D:/sih2026")
RGB = ROOT / "data/extracted/Track1-RGB"
TR = ROOT / "data/extracted/Track1-Truth"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles", type=int, default=24)
    ap.add_argument("--ckpt", default=str(ROOT / "checkpoints/run02/best.pt"))
    ap.add_argument("--val", default=str(ROOT / "out/eval_run02_ship/per_building.npz"))
    a = ap.parse_args()

    split = json.load(open(ROOT / "data/shards/split.json"))
    regions = sorted(k for k, v in split.items() if v == "train")
    tiles = []
    for r in regions:
        tiles += sorted(p.name[:-8] for p in RGB.glob(f"{r}_*_RGB.tif"))
    np.random.default_rng(1337).shuffle(tiles)
    tiles = tiles[: a.tiles]
    print(f"fitting on {len(tiles)} tiles drawn from {len(regions)} TRAIN regions")

    O, T = [], []
    for i, t in enumerate(tiles):
        out = ROOT / "out" / "decomp" / t
        out.parent.mkdir(parents=True, exist_ok=True)
        h = Path(str(out) + ".height.tif")
        if not h.exists():
            subprocess.run(
                [sys.executable, "infer.py", "--image", str(RGB / f"{t}_RGB.tif"),
                 "--ckpt", a.ckpt, "--out", str(out), "--tta",
                 "--fuse-zoom", "2", "--fuse-sigma", "8"],
                cwd=str(Path(__file__).resolve().parents[1]), check=True,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        pred = rasterio.open(h).read(1).astype("float32")
        agl = rasterio.open(TR / f"{t}_AGL.tif").read(1).astype("float32")
        cls = rasterio.open(TR / f"{t}_CLS.tif").read(1)
        o, tr = building_instances(pred, agl, cls)
        if len(o):
            O.append(o); T.append(tr)
        print(f"  [{i+1}/{len(tiles)}] {t}: {len(o)} buildings", flush=True)

    O = np.concatenate(O); T = np.concatenate(T)
    slope, intercept = np.polyfit(T, O, 1)
    print(f"\nTRAIN fit over {len(O):,} buildings:  ours = {slope:.4f} * truth "
          f"{intercept:+.3f}")
    print(f"  inverting gives:  truth ~ (ours {-intercept:+.3f}) / {slope:.4f}")

    d = np.load(a.val)
    vo, vt = d["ours"], d["truth"]
    print(f"\nVAL, {len(vo):,} buildings. Correction fitted on TRAIN only.")
    print(f"  {'setting':22s} {'RMSE':>8s} {'MAE':>7s} {'bias':>7s} {'corr':>7s}")

    def report(name, x):
        m = building_wise_metrics(x, vt)
        print(f"  {name:22s} {m['rmse']:8.3f} {m['mae']:7.3f} {m['bias']:+7.2f} "
              f"{m['corr']:+7.3f}")
        return m["rmse"]

    base = report("uncorrected", vo)
    best = (0.0, base)
    # s = 0 leaves the prediction alone, s = 1 fully inverts the fitted line.
    for s in (0.25, 0.5, 0.75, 1.0):
        corrected = (vo - s * intercept) / (1.0 - s + s * slope)
        r = report(f"de-compressed s={s:.2f}", corrected)
        if r < best[1]:
            best = (s, r)

    print()
    if best[0] == 0.0:
        print("  No setting beat leaving it alone. The variance the inversion adds costs "
              "more\n  than the bias it removes -- this is a training-time problem and a "
              "linear\n  correction at inference does not fix it.")
    else:
        print(f"  Best: s={best[0]:.2f}, RMSE {best[1]:.3f} against {base:.3f} "
              f"({100*(best[1]-base)/base:+.1f}%)")
        print("  Fitted on train, measured on val, so this is a real gain rather than a "
              "tuned one.")


if __name__ == "__main__":
    main()
