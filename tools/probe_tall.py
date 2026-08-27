"""Probe 01: can tall-building supervision move the tail, and does it generalise?

See docs/probe-01-tall-buildings.md for the predictions, which were written before this
was run. In short: 79% of run02's squared per-building error comes from the 1.9% of
buildings above 20 m, where it under-calls by 17 m. run03 and run04 both attacked that
with architecture and both moved it under a metre. Before a third 3.5-hour run, find out
in thirty minutes whether the model can represent a tall building at all.

Two measurements, and keeping them apart is the whole point:

  FIT      held-out tall buildings from the same train regions we fine-tune on.
           Answers "can the model represent a tall building?"
  TRANSFER tall buildings in val regions, never trained on here.
           Answers "does tall supervision generalise?"

"Cannot fit" and "fits but does not transfer" call for opposite next moves -- debugging
the objective versus going and getting more data -- and one number cannot separate them.

    python tools/probe_tall.py --steps 400
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import rasterio
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from depthwizard.dataset import IMAGENET_MEAN, IMAGENET_STD  # noqa: E402
from depthwizard.metrics import building_instances, building_wise_metrics  # noqa: E402
from depthwizard.model import from_checkpoint  # noqa: E402

ROOT = Path("D:/sih2026")
RGB_DIR = ROOT / "data" / "extracted" / "Track1-RGB"
TRUTH_DIR = ROOT / "data" / "extracted" / "Track1-Truth"
CROP = 518
TALL = 20.0


def read(p):
    with rasterio.open(p) as s:
        return s.read(1)


def find_tall_crops(regions, min_h=TALL, per_tile_cap=6):
    """Crop centres sitting on buildings at least `min_h` tall."""
    from scipy import ndimage
    out = []
    for p in sorted(TRUTH_DIR.glob("*_AGL.tif")):
        tile = p.stem[:-4]
        if tile.rsplit("_", 1)[0] not in regions:
            continue
        cls_p = TRUTH_DIR / f"{tile}_CLS.tif"
        if not cls_p.exists():
            continue
        agl, cls = read(p).astype("float32"), read(cls_p)
        mask = (cls == 6) & np.isfinite(agl)
        if not mask.any():
            continue
        lab, n = ndimage.label(mask, structure=np.ones((3, 3)))
        if n == 0:
            continue
        idx = np.arange(1, n + 1)
        med = ndimage.median(agl, lab, idx)
        cnt = np.bincount(lab.ravel(), minlength=n + 1)[1:]
        sel = np.where((med >= min_h) & (cnt * 0.09 >= 25.0))[0]
        if sel.size == 0:
            continue
        cents = ndimage.center_of_mass(mask, lab, idx[sel])
        H, W = agl.shape
        for (cy, cx), h in list(zip(cents, med[sel]))[:per_tile_cap]:
            y = int(np.clip(cy - CROP // 2, 0, max(0, H - CROP)))
            x = int(np.clip(cx - CROP // 2, 0, max(0, W - CROP)))
            if H < CROP or W < CROP:
                continue
            out.append({"tile": tile, "y": y, "x": x, "h": float(h)})
    return out


def load_crop(c):
    rgb = np.transpose(read_rgb(RGB_DIR / f"{c['tile']}_RGB.tif",
                                c["y"], c["x"]), (2, 0, 1))
    agl = read_win(TRUTH_DIR / f"{c['tile']}_AGL.tif", c["y"], c["x"])
    cls = read_win(TRUTH_DIR / f"{c['tile']}_CLS.tif", c["y"], c["x"])
    return rgb, agl, cls


def read_rgb(p, y, x):
    with rasterio.open(p) as s:
        a = s.read(window=((y, y + CROP), (x, x + CROP)))
    a = np.transpose(a[:3], (1, 2, 0)).astype(np.float32) / 255.0
    return (a - IMAGENET_MEAN) / IMAGENET_STD


def read_win(p, y, x):
    with rasterio.open(p) as s:
        return s.read(1, window=((y, y + CROP), (x, x + CROP)))


@torch.no_grad()
def score(model, crops, device, label):
    """Per-building metrics restricted to buildings at least TALL metres."""
    model.eval()
    ours, truth = [], []
    for i in range(0, len(crops), 4):
        batch = crops[i:i + 4]
        xs, agls, clss = [], [], []
        for c in batch:
            r, a, cl = load_crop(c)
            xs.append(r)
            agls.append(a.astype("float32"))
            clss.append(cl)
        t = torch.from_numpy(np.stack(xs)).to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            mu = model(t)[0].float().cpu().numpy()[:, 0]
        for k in range(len(batch)):
            o, tr = building_instances(mu[k], agls[k], clss[k])
            if o.size:
                keep = tr >= TALL
                if keep.any():
                    ours.append(o[keep])
                    truth.append(tr[keep])
    if not ours:
        print(f"  {label}: no tall buildings found")
        return None
    m = building_wise_metrics(np.concatenate(ours), np.concatenate(truth))
    print(f"  {label:22s} n={m['n_buildings']:4d}  RMSE {m['rmse']:6.2f} m  "
          f"bias {m['bias']:+6.2f} m  (true median {m['truth_median_h']:.1f} m)")
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(ROOT / "checkpoints" / "run02" / "best.pt"))
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--lr-head", type=float, default=2.5e-4)
    ap.add_argument("--max-temp", type=float, default=87.0)
    ap.add_argument("--out", default=str(ROOT / "out" / "probe_tall.json"))
    a = ap.parse_args()

    assign = json.loads((ROOT / "data" / "shards" / "split.json").read_text())
    train_regions = {r for r, v in assign.items() if v == "train"}
    val_regions = {r for r, v in assign.items() if v == "val"}

    print("locating tall buildings...")
    tr_crops = find_tall_crops(train_regions)
    va_crops = find_tall_crops(val_regions)
    rng = np.random.default_rng(1337)
    rng.shuffle(tr_crops)
    n_hold = max(20, len(tr_crops) // 5)
    fit_crops, hold_crops = tr_crops[n_hold:], tr_crops[:n_hold]
    print(f"  train tall crops {len(tr_crops)}  -> fit {len(fit_crops)}, "
          f"held-out {len(hold_crops)}")
    print(f"  val tall crops   {len(va_crops)}")
    if len(fit_crops) < 8:
        raise SystemExit("not enough tall crops to probe")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    model = from_checkpoint(ck).to(device)
    print(f"\nloaded {a.ckpt} (epoch {ck.get('epoch')})\n")

    print("BEFORE fine-tuning")
    before = {"fit_heldout": score(model, hold_crops, device, "FIT (held-out train)"),
              "transfer_val": score(model, va_crops, device, "TRANSFER (val)")}

    head = [p for n, p in model.named_parameters() if not n.startswith("backbone")]
    back = [p for n, p in model.named_parameters() if n.startswith("backbone")]
    opt = torch.optim.AdamW([{"params": back, "lr": a.lr},
                             {"params": head, "lr": a.lr_head}], weight_decay=0.01)

    print(f"\nfine-tuning {a.steps} steps on tall crops only "
          f"(batch {a.batch}, lr {a.lr}/{a.lr_head})")
    model.train()
    order = rng.permutation(len(fit_crops))
    ptr = 0
    import subprocess
    for step in range(a.steps):
        if ptr + a.batch > len(order):
            order = rng.permutation(len(fit_crops))
            ptr = 0
        batch = [fit_crops[i] for i in order[ptr:ptr + a.batch]]
        ptr += a.batch
        xs, ys = [], []
        for c in batch:
            r, agl, _ = load_crop(c)
            xs.append(r)
            ys.append(agl.astype("float32"))
        x = torch.from_numpy(np.stack(xs)).to(device)
        y = torch.from_numpy(np.stack(ys)[:, None]).to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            mu = model(x)[0]
        # Plain L1 on height. The probe asks whether the tail is reachable at all, so it
        # deliberately avoids the uncertainty and gradient terms -- if a bare regression
        # loss on tall-only crops cannot move the bias, no weighting scheme built on top
        # of it will either.
        loss = (mu.float() - y).abs().mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        if (step + 1) % 50 == 0:
            t = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu",
                                "--format=csv,noheader"], capture_output=True, text=True)
            temp = int(t.stdout.strip().split("\n")[0]) if t.returncode == 0 else 0
            print(f"  step {step+1:4d}/{a.steps}  L1 {loss.item():6.3f} m  {temp} C")
            while temp >= a.max_temp:
                import time
                time.sleep(3)
                t = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu",
                                    "--format=csv,noheader"], capture_output=True, text=True)
                temp = int(t.stdout.strip().split("\n")[0])
                print(f"      [thermal] cooling to {temp} C")

    print("\nAFTER fine-tuning")
    after = {"fit_heldout": score(model, hold_crops, device, "FIT (held-out train)"),
             "transfer_val": score(model, va_crops, device, "TRANSFER (val)")}

    print("\n" + "=" * 62)
    for key, name in (("fit_heldout", "FIT"), ("transfer_val", "TRANSFER")):
        b, af = before[key], after[key]
        if not b or not af:
            continue
        print(f"{name:9s} bias {b['bias']:+7.2f} -> {af['bias']:+7.2f} m   "
              f"RMSE {b['rmse']:6.2f} -> {af['rmse']:6.2f} m")
    t_after = after["transfer_val"]["bias"] if after["transfer_val"] else None
    if t_after is not None:
        print()
        if t_after > -10.0:
            print("VERDICT: tall supervision GENERALISES. run05 is justified.")
        elif t_after > -13.0:
            print("VERDICT: partial transfer. Real but weak -- fold into a data run, "
                  "not worth 3.5 h alone.")
        else:
            f_after = after["fit_heldout"]["bias"] if after["fit_heldout"] else 0
            if f_after > -8.0:
                print("VERDICT: DATA SCARCITY. The model can represent tall buildings but "
                      "cannot generalise from 166 of them. Get tall-building data; a "
                      "DFC2019 re-split will not fix this.")
            else:
                print("VERDICT: STRUCTURAL. It cannot even fit tall buildings it trained "
                      "on. Debug the objective before any further training.")
    Path(a.out).write_text(json.dumps({"before": before, "after": after,
                                       "steps": a.steps, "ckpt": a.ckpt}, indent=2))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
