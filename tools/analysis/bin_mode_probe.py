"""Does the binned head KNOW a building is tall and then average the knowledge away?

Every head we have tried compresses by about half: run02 regression 0.471, run03 bins
0.429, run04 bins+HTC 0.491. No architecture moved it. Before spending another training
run, this settles which of two very different worlds we are in.

Even with the head-tail cut, `model.py` decodes height as the EXPECTATION over bins:

    mu = (probs * centres).sum(dim=1)

HTC separates roof from ground, but inside the roof branch the height is still an average
over 128 bins. So on a pixel whose truth is 40 m, there are two possibilities:

  A. The distribution has a mode near 40 m, and the expectation drags it down toward the
     4 m median because probability mass also sits low. The model KNOWS. The fix is the
     decode -- argmax or a sharpening constraint -- and costs no training at all.

  B. The distribution has no tall mode. The model does not know, the information is not
     in the image, and no head or loss will conjure it. Stop tuning heads and go to a new
     cue (shadow, PLAN SS 6.2).

The discriminator is `argmax height` vs `expectation` per true-height band. If A, the
argmax sits far above the expectation on tall pixels. If B, both sit near the median.

    python tools/analysis/bin_mode_probe.py --ckpt D:/sih2026/checkpoints/run04/best.pt
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path("D:/sih2026")
sys.path.insert(0, str(ROOT / "depthwizard"))

from depthwizard.dataset import IMAGENET_MEAN, IMAGENET_STD  # noqa: E402
from depthwizard.metrics import CLS_BUILDING  # noqa: E402
from depthwizard.model import from_checkpoint  # noqa: E402

BANDS = [(0, 3), (3, 6), (6, 10), (10, 20), (20, 40), (40, 1e9)]
TALL = 20.0  # the height above which run02/run03/run04 all collapse


def band_label(lo: float, hi: float) -> str:
    return f"{lo:g}-{hi:g} m" if hi < 1e8 else f"{lo:g} m+"


def pick_crops(shards: list[str], want: int, rng) -> list[tuple[str, int]]:
    """Prefer crops that actually contain tall buildings.

    Tall pixels are ~1.9% of buildings, so a uniform sample of crops would spend almost
    all its GPU time on the bands that already work. We scan the AGL arrays -- cheap, they
    are float16 on disk -- and take every crop with a tall building first.
    """
    tall, plain = [], []
    for f in shards:
        with np.load(f) as z:
            agl, cls = z["agl"], z["cls"]
            for i in range(agl.shape[0]):
                b = cls[i] == CLS_BUILDING
                if not b.any():
                    continue
                (tall if np.nanmax(np.where(b, agl[i], 0)) > TALL else plain).append((f, i))
    rng.shuffle(plain)
    n_plain = max(0, want - len(tall))
    print(f"  {len(tall)} crops contain a building above {TALL:g} m; "
          f"adding {min(n_plain, len(plain))} ordinary crops")
    return tall[:want] + plain[:n_plain]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(ROOT / "checkpoints" / "run04" / "best.pt"))
    ap.add_argument("--crops", type=int, default=64)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = from_checkpoint(ck).to(device).eval()
    if not getattr(model, "bins", 0):
        raise SystemExit(f"{args.ckpt} has no binned head; this probe needs one")
    print(f"checkpoint {args.ckpt}  epoch {ck.get('epoch')}  bins={model.bins}  {device}")

    rng = np.random.default_rng(args.seed)
    shards = sorted(glob.glob(str(ROOT / "data" / "shards" / "val_*.npz")))
    picks = pick_crops(shards, args.crops, rng)

    cols = {k: [] for k in ("truth", "exp", "amax", "mass")}
    by_shard: dict[str, list[int]] = {}
    for f, i in picks:
        by_shard.setdefault(f, []).append(i)

    for f, idxs in by_shard.items():
        with np.load(f) as z:
            for s in range(0, len(idxs), args.batch):
                sel = idxs[s:s + args.batch]
                rgb = z["rgb"][sel].astype(np.float32) / 255.0
                agl = z["agl"][sel].astype(np.float32)
                cls = z["cls"][sel]
                x = (rgb - IMAGENET_MEAN) / IMAGENET_STD
                x = torch.from_numpy(x.transpose(0, 3, 1, 2)).to(device)

                with torch.no_grad():
                    out = model(x, return_bins=True)
                mu, aux = out[0], out[-1]
                probs = aux["probs"]                       # (B, N, H, W)
                centres = aux["centres"]                   # (B, N)

                # Decode two ways from the SAME distribution: the expectation the model
                # ships with, and the single most likely bin.
                amax = centres.gather(1, probs.argmax(dim=1).flatten(1)).view(mu.shape)
                mass = (probs * (centres[:, :, None, None] > TALL)).sum(dim=1, keepdim=True)

                # Building pixels only, and only where truth is finite.
                keep = torch.from_numpy((cls == CLS_BUILDING) & np.isfinite(agl)).to(device)
                keep = keep.unsqueeze(1)
                if not keep.any():
                    continue
                t = torch.from_numpy(agl).to(device).unsqueeze(1)[keep]
                cols["truth"].append(t.cpu().numpy())
                cols["exp"].append(mu[keep].float().cpu().numpy())
                cols["amax"].append(amax[keep].float().cpu().numpy())
                cols["mass"].append(mass[keep].float().cpu().numpy())

    d = {k: np.concatenate(v) for k, v in cols.items()}
    print(f"\n{len(d['truth']):,} building pixels\n")
    print(f"  {'band':>10} {'n':>9} {'truth':>8} {'expectation':>12} {'argmax bin':>11} "
          f"{'P(h>20m)':>9}")
    for lo, hi in BANDS:
        m = (d["truth"] >= lo) & (d["truth"] < hi)
        if not m.any():
            continue
        print(f"  {band_label(lo, hi):>10} {int(m.sum()):>9,} {d['truth'][m].mean():>7.1f}m "
              f"{d['exp'][m].mean():>11.1f}m {d['amax'][m].mean():>10.1f}m "
              f"{d['mass'][m].mean():>9.3f}")

    tall = d["truth"] >= TALL
    if tall.any():
        gap = d["amax"][tall].mean() - d["exp"][tall].mean()
        print(f"\n  On pixels truly above {TALL:g} m: argmax sits {gap:+.2f} m from the "
              f"expectation,\n  mean mass above {TALL:g} m is {d['mass'][tall].mean():.3f}, "
              f"and truth averages {d['truth'][tall].mean():.1f} m.")
        print("\n  Read it as: a large positive gap and real mass up top means the model "
              "knows\n  and the decode is throwing it away (fix the decode). A gap near "
              "zero with\n  negligible mass means the tall mode was never there (go to "
              "shadow).")


if __name__ == "__main__":
    main()
