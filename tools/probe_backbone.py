"""Probe 02: is a larger backbone a better prior for satellite height?

We run Depth-Anything-V2-Small (24.9 M) because V2's Base and Large are CC-BY-NC and this
has to ship. But V1 is Apache-2.0 at every size, and DPT/ZoeDepth are MIT, so capacity was
never actually blocked -- I had assumed it was. Meanwhile the models we benchmark against
(HTC-DC Net, GlobalBuildingAtlas) use EfficientNet-B5/B7, several times larger.

The question this answers, before spending seven hours on a run: does a bigger encoder
carry visibly more height signal on overhead imagery, or is 24.9 M already enough for a
domain this far from the natural images it was pretrained on?

Method. Zero-shot, no fine-tuning. Relative depth is in arbitrary units, so RMSE is
meaningless here -- read the CORRELATION with true AGL. That is the honest measure of how
much height information the representation contains before we teach it anything. We report
it three ways: over all pixels, over building pixels only, and per building instance, since
buildings are where our error actually lives.

    python tools/probe_backbone.py --tiles 12
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import rasterio
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from depthwizard.dataset import IMAGENET_MEAN, IMAGENET_STD  # noqa: E402
from depthwizard.metrics import building_instances, CLS_BUILDING  # noqa: E402

ROOT = Path("D:/sih2026")
RGB_DIR = ROOT / "data" / "extracted" / "Track1-RGB"
TRUTH_DIR = ROOT / "data" / "extracted" / "Track1-Truth"

CANDIDATES = [
    ("depth-anything/Depth-Anything-V2-Small-hf", "V2-Small  (current)", "Apache-2.0"),
    ("LiheYoung/depth-anything-base-hf", "V1-Base", "Apache-2.0"),
    ("LiheYoung/depth-anything-large-hf", "V1-Large", "Apache-2.0"),
]


def read(p, band=1):
    with rasterio.open(p) as s:
        return s.read(band)


@torch.no_grad()
def depth_of(model, rgb_u8, device, tile=518):
    """Zero-shot relative depth over one tile, centre-cropped to the model's input size."""
    H, W = rgb_u8.shape[:2]
    y, x = (H - tile) // 2, (W - tile) // 2
    crop = rgb_u8[y:y + tile, x:x + tile].astype(np.float32) / 255.0
    t = torch.from_numpy(
        np.ascontiguousarray(((crop - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1))
    )[None].to(device)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
        out = model(pixel_values=t).predicted_depth
    d = torch.nn.functional.interpolate(
        out[:, None] if out.dim() == 3 else out, size=(tile, tile),
        mode="bicubic", align_corners=False)[0, 0].float().cpu().numpy()
    return d, (y, x, tile)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles", type=int, default=12)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--out", default=str(ROOT / "out" / "probe_backbone.json"))
    a = ap.parse_args()

    from transformers import AutoModelForDepthEstimation

    assign = json.loads((ROOT / "data" / "shards" / "split.json").read_text())
    val = {r for r, v in assign.items() if v == "val"}
    tiles = [p.stem[:-4] for p in sorted(RGB_DIR.glob("*_RGB.tif"))
             if p.stem[:-4].rsplit("_", 1)[0] in val
             and (TRUTH_DIR / f"{p.stem[:-4]}_AGL.tif").exists()]
    rng = np.random.default_rng(a.seed)
    tiles = list(rng.choice(tiles, min(a.tiles, len(tiles)), replace=False))
    print(f"{len(tiles)} val tiles, region-disjoint\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = {}
    for mid, label, lic in CANDIDATES:
        print(f"=== {label}  [{lic}]")
        t0 = time.time()
        try:
            model = AutoModelForDepthEstimation.from_pretrained(mid).to(device).eval()
        except Exception as e:
            print(f"    could not load: {type(e).__name__}: {str(e)[:160]}\n")
            continue
        n_par = sum(p.numel() for p in model.parameters())
        print(f"    {n_par/1e6:.1f} M params, loaded in {time.time()-t0:.0f}s")

        all_d, all_h, bld_d, bld_h, inst_d, inst_h = [], [], [], [], [], []
        for t in tiles:
            rgb = np.transpose(read_rgb(RGB_DIR / f"{t}_RGB.tif"), (0, 1, 2))
            d, (y, x, s) = depth_of(model, rgb, device)
            agl = read(TRUTH_DIR / f"{t}_AGL.tif").astype("float32")[y:y+s, x:x+s]
            cls_p = TRUTH_DIR / f"{t}_CLS.tif"
            cls = read(cls_p)[y:y+s, x:x+s] if cls_p.exists() else None
            m = np.isfinite(agl) & np.isfinite(d)
            all_d.append(d[m][::9]); all_h.append(agl[m][::9])
            if cls is not None:
                bm = m & (cls == CLS_BUILDING)
                if bm.sum() > 500:
                    bld_d.append(d[bm][::5]); bld_h.append(agl[bm][::5])
                od, ot = building_instances(d, agl, cls)
                if od.size:
                    inst_d.append(od); inst_h.append(ot)
        del model
        torch.cuda.empty_cache()

        def corr(a_, b_):
            if not a_:
                return float("nan")
            x_, y_ = np.concatenate(a_), np.concatenate(b_)
            if x_.size < 3 or x_.std() == 0 or y_.std() == 0:
                return float("nan")
            return float(np.corrcoef(x_, y_)[0, 1])

        r_all, r_bld, r_inst = corr(all_d, all_h), corr(bld_d, bld_h), corr(inst_d, inst_h)
        n_inst = sum(v.size for v in inst_d)
        print(f"    r(all px)      {r_all:+.3f}")
        print(f"    r(building px) {r_bld:+.3f}")
        print(f"    r(per building) {r_inst:+.3f}   n={n_inst:,}\n")
        results[label] = {"model_id": mid, "licence": lic, "params_m": n_par / 1e6,
                          "r_all": r_all, "r_building_px": r_bld,
                          "r_per_building": r_inst, "n_buildings": int(n_inst)}

    Path(a.out).write_text(json.dumps(results, indent=2))
    print("=" * 60)
    base = results.get("V2-Small  (current)", {}).get("r_per_building")
    for label, d in results.items():
        delta = ""
        if base and np.isfinite(d["r_per_building"]) and label != "V2-Small  (current)":
            delta = f"   ({d['r_per_building'] - base:+.3f} vs current)"
        print(f"{label:22s} {d['params_m']:6.1f} M   "
              f"r/building {d['r_per_building']:+.3f}{delta}")
    print(f"\nwrote {a.out}")


def read_rgb(p):
    with rasterio.open(p) as s:
        return np.transpose(s.read()[:3], (1, 2, 0))


if __name__ == "__main__":
    main()
