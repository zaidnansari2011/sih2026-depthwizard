"""Probe 05b: does --auto-zoom still hold at ISRO's 0.6 m on the shipped model?

Pre-registered in docs/probe-05b-gsd-run07.md before this ran -- read that first; the
thresholds live there, not here. ISRO's FAQ fixed final evaluation at Cartosat-2S 0.6 m,
and probe 05's "within 3.6 %" was run02, three hand-picked tiles, single pass, averaged per
tile. This re-asks the question the way the headline number is measured:

  * the same 80 val tiles as tools/evaluate.py (same split file, seed and sampling);
  * the shipped configuration: 8-way TTA plus zoom-2 Laplacian fusion at sigma 8;
  * buildings pooled across tiles, not per-tile RMSEs averaged.

Three conditions per tile, each scored against the untouched 1024 px LiDAR truth:
  A  native 0.3 m, zoom 1                        -- must reproduce 3.464 m (sanity gate)
  B  Lanczos-downsampled to 0.6 m, zoom 1        -- untreated
  C  the same 0.6 m image, zoom 2 (detail pass 4) -- what --auto-zoom does

    python tools/gsd_recheck.py --ckpt ../checkpoints/run07/best.pt
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import gaussian_filter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from depthwizard.metrics import (building_instances, building_wise_metrics,  # noqa: E402
                                 height_metrics, terrain_category_with_relief)
from depthwizard.model import from_checkpoint  # noqa: E402
from infer import infer_scene, load_image  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RGB_DIR = ROOT / "data" / "extracted" / "Track1-RGB"
TRUTH_DIR = ROOT / "data" / "extracted" / "Track1-Truth"
SPLIT = ROOT / "data" / "shards" / "split.json"

# name -> (input GSD in metres, base zoom). The fusion detail pass runs at base * 2, exactly
# as infer.py does when --auto-zoom and --fuse-zoom are combined.
CONDITIONS = {"A_native_0.3m": (0.3, 1), "B_0.6m_untreated": (0.6, 1),
              "C_0.6m_autozoom": (0.6, 2)}


def gpu_temp() -> int | None:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu",
                              "--format=csv,noheader,nounits"], capture_output=True,
                             text=True, timeout=10).stdout
        return int(out.strip().splitlines()[0])
    except Exception:
        return None


def cool_down(hot=84, ok=75) -> None:
    """The 3060 self-throttles at 91 C. Inference is lighter than training, but a long
    TTA sweep is still sustained load, so pause rather than let it cook."""
    t = gpu_temp()
    if t is None or t < hot:
        return
    print(f"    GPU {t} C: pausing until {ok} C", flush=True)
    while (t := gpu_temp()) is not None and t > ok:
        time.sleep(10)


def read_tif(p: Path) -> np.ndarray:
    import rasterio
    with rasterio.open(p) as src:
        return src.read(1)


def predict(model, rgb, gsd, zoom, device, amp, batch):
    """Shipped path on an image degraded to `gsd`, returned at the input's own grid."""
    H, W = rgb.shape[:2]
    if gsd != 0.3:
        f = 0.3 / gsd
        rgb = np.asarray(Image.fromarray(rgb).resize((round(W * f), round(H * f)),
                                                     Image.LANCZOS))
    h, w = rgb.shape[:2]
    # A 518 px window cannot sit on a 512 px image at zoom 1; shrink it to the largest
    # multiple of 14 that fits, as probe 05 did. At zoom >= 2 the window's ground footprint
    # (518 / zoom) already fits, so the shipped 518 stays.
    tile = 518 if zoom >= 2 or min(h, w) >= 518 else (min(h, w) // 14) * 14
    overlap = 140
    base, _ = infer_scene(model, rgb, tile, overlap, device, amp, batch, verbose=False,
                          tta=True, zoom=zoom)
    detail, _ = infer_scene(model, rgb, tile, overlap, device, amp, batch, verbose=False,
                            tta=False, zoom=zoom * 2)
    pred = (gaussian_filter(base, 8.0) + (detail - gaussian_filter(detail, 8.0))
            ).astype(np.float32)
    if (h, w) != (H, W):
        pred = np.asarray(Image.fromarray(pred).resize((W, H), Image.BILINEAR),
                          dtype=np.float32)
    return pred, tile


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tiles", type=int, default=80)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--out", default=str(ROOT / "out" / "gsd_recheck_run07"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    # Tile selection copied from tools/evaluate.py line for line, so condition A is the
    # headline measurement and the sanity gate means something.
    assign = json.loads(SPLIT.read_text())
    regions = {r for r, v in assign.items() if v == "val"}
    tiles = [p.stem[:-4] for p in sorted(RGB_DIR.glob("*_RGB.tif"))
             if p.stem[:-4].rsplit("_", 1)[0] in regions
             and (TRUTH_DIR / f"{p.stem[:-4]}_AGL.tif").exists()]
    rng = np.random.default_rng(a.seed)
    tiles = list(rng.choice(tiles, min(a.tiles, len(tiles)), replace=False))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    from train import pick_precision
    amp, _, prec = pick_precision("auto")
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    model = from_checkpoint(ck).to(device).eval()
    print(f"checkpoint {a.ckpt} | {prec} | {device} | {len(tiles)} val tiles", flush=True)

    acc = {k: {"BP": [], "BT": [], "P": [], "T": [], "terr": defaultdict(lambda: [[], []])}
           for k in CONDITIONS}
    t0 = time.time()
    for i, t in enumerate(tiles):
        rgb = load_image(RGB_DIR / f"{t}_RGB.tif")
        truth = read_tif(TRUTH_DIR / f"{t}_AGL.tif").astype(np.float32)
        cls = read_tif(TRUTH_DIR / f"{t}_CLS.tif")
        cat = terrain_category_with_relief(cls, truth)
        for name, (gsd, zoom) in CONDITIONS.items():
            cool_down()
            pred, _ = predict(model, rgb, gsd, zoom, device, amp, a.batch)
            m = np.isfinite(truth) & np.isfinite(pred)
            d = acc[name]
            d["P"].append(pred[m][::7])
            d["T"].append(truth[m][::7])
            d["terr"][cat][0].append(pred[m][::7])
            d["terr"][cat][1].append(truth[m][::7])
            bp, bt = building_instances(pred, truth, cls)
            if bp.size:
                d["BP"].append(bp)
                d["BT"].append(bt)
        if (i + 1) % 5 == 0:
            el = time.time() - t0
            print(f"  {i + 1}/{len(tiles)}  {el / 60:.1f} min  "
                  f"(~{el / (i + 1) * (len(tiles) - i - 1) / 60:.0f} min left)  "
                  f"GPU {gpu_temp()} C", flush=True)

    res = {}
    for name, d in acc.items():
        bo, bt = np.concatenate(d["BP"]), np.concatenate(d["BT"])
        bw = building_wise_metrics(bo, bt)
        low = bt <= 20
        bw20 = building_wise_metrics(bo[low], bt[low])
        whole = height_metrics(np.concatenate(d["P"]), np.concatenate(d["T"])).to_dict()
        terr = {c: float(np.sqrt(np.mean((np.concatenate(p) - np.concatenate(q)) ** 2)))
                for c, (p, q) in sorted(d["terr"].items())}
        res[name] = {"per_building_rmse": bw["rmse"], "per_building_bias": bw["bias"],
                     "per_building_corr": bw["corr"], "n_buildings": bw["n_buildings"],
                     "per_building_rmse_le20m": bw20["rmse"],
                     "n_buildings_le20m": bw20["n_buildings"],
                     "whole_tile_rmse": whole["rmse"], "per_terrain_rmse": terr}
    res["_meta"] = {"checkpoint": a.ckpt, "tiles": [str(x) for x in tiles],
                    "minutes": (time.time() - t0) / 60, "config": "TTA x8 + fuse zoom-2 s8"}
    (out / "results.json").write_text(json.dumps(res, indent=2))

    A, B, C = (res[k] for k in CONDITIONS)
    print("\ncondition            per-bldg   bias    <=20m   whole-tile")
    for k in CONDITIONS:
        r = res[k]
        print(f"{k:<20} {r['per_building_rmse']:8.3f} {r['per_building_bias']:+6.2f} "
              f"{r['per_building_rmse_le20m']:8.3f} {r['whole_tile_rmse']:10.3f}")
    gate = abs(A["per_building_rmse"] - 3.464) <= 0.01
    prim = C["per_building_rmse"] <= 1.10 * A["per_building_rmse"]
    sec = C["per_building_rmse"] < B["per_building_rmse"]
    cost = 100 * (C["per_building_rmse"] / A["per_building_rmse"] - 1)
    print(f"\nsanity gate  A = {A['per_building_rmse']:.3f} vs 3.464 +/- 0.01  -> "
          f"{'PASS' if gate else 'FAIL (run void)'}")
    print(f"primary      C <= 1.10 x A: {C['per_building_rmse']:.3f} vs "
          f"{1.10 * A['per_building_rmse']:.3f}  -> {'HOLDS' if prim else 'FALSIFIED'}"
          f"   (0.6 m costs {cost:+.1f} %)")
    print(f"secondary    C < B: {C['per_building_rmse']:.3f} vs {B['per_building_rmse']:.3f}"
          f"  -> {'HOLDS' if sec else 'FALSIFIED'}")
    print(f"\n  -> {out / 'results.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
