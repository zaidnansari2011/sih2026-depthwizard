"""Run DepthWizard over a full scene and write a height map (and its uncertainty).

    # zero-shot, no checkpoint -- the week-1 vertical slice
    python infer.py --image data/extracted/Track1-RGB/JAX_004_007_RGB.tif --out out/JAX_004_007

    # with a fine-tuned checkpoint
    python infer.py --image scene.tif --ckpt checkpoints/best.pt --out out/scene

Sliding-window, because scenes are larger than the model's input and a single resize
would throw away the resolution that makes small structures visible. Windows overlap and
are blended with a cosine taper: seams are the most obvious artefact in a 3D flythrough,
far more visible than they ever are in a 2D error metric, and a jury looking at a
rendered surface will spot a grid before they read a number.

Uncertainty is combined properly across overlaps. Averaging sigma directly would be
wrong; we average variances, weighted the same way as the means.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from depthwizard.dataset import IMAGENET_MEAN, IMAGENET_STD
from depthwizard.model import DEFAULT_MODEL, build, PATCH


def load_image(path: Path) -> np.ndarray:
    """Return HWC uint8 RGB."""
    if path.suffix.lower() in (".tif", ".tiff"):
        import rasterio
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with rasterio.open(path) as src:
                a = src.read()
        a = a[0] if a.shape[0] == 1 else np.transpose(a, (1, 2, 0))
    else:
        from PIL import Image
        a = np.array(Image.open(path).convert("RGB"))
    if a.ndim == 2:
        a = np.stack([a] * 3, -1)
    a = a[..., :3]
    if a.dtype != np.uint8:
        lo, hi = np.percentile(a[np.isfinite(a)], [2, 98])
        a = (np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1) * 255).astype(np.uint8)
    return a


def cosine_window(n: int, overlap: int) -> np.ndarray:
    """1-D taper that is flat in the middle and falls to ~0 across the overlap band.

    Two tapers that meet sum to 1, so blending is seamless with no brightness dip in
    the join -- the thing you would otherwise see as a faint grid in the render.
    """
    w = np.ones(n, np.float32)
    if overlap > 0:
        r = np.arange(overlap, dtype=np.float32)
        ramp = 0.5 - 0.5 * np.cos(np.pi * (r + 0.5) / overlap)
        w[:overlap] = ramp
        w[-overlap:] = ramp[::-1]
    return w


def _d4(x: torch.Tensor, k: int, flip: bool) -> torch.Tensor:
    if flip:
        x = torch.flip(x, dims=[3])
    return torch.rot90(x, k, dims=(2, 3))


def _d4_inv(y: torch.Tensor, k: int, flip: bool) -> torch.Tensor:
    y = torch.rot90(y, -k, dims=(2, 3))
    if flip:
        y = torch.flip(y, dims=[3])
    return y


@torch.no_grad()
def _predict_window(model, t: torch.Tensor, amp_dtype, device: str, tta: bool):
    """One batch of windows through the model, optionally averaged over the D4 group.

    D4 (four rotations x optional mirror) is exactly valid for nadir height: rotating
    the scene rotates the height map identically and height is invariant to reflection.
    It would NOT be valid once a shadow prior is in play, since shadow direction is tied
    to sun azimuth.

    The uncertainty combination is the interesting part. Averaging the eight sigmas would
    be wrong twice over. By the law of total variance the correct total is

        Var[h] = E[sigma^2]  +  Var[mu]
                 \\_ aleatoric _/    \\_ disagreement between the eight views _/

    so TTA does not just sharpen the mean, it *adds* a genuine epistemic term the single
    forward pass cannot see: where the eight views disagree, the model is unsure in a way
    its own sigma head never expressed.
    """
    variants = [(0, False)] if not tta else [(k, f) for f in (False, True) for k in range(4)]
    mus, varis = [], []
    for k, f in variants:
        with torch.autocast("cuda", dtype=amp_dtype,
                            enabled=(device == "cuda" and amp_dtype != torch.float32)):
            mu, log_var = model(_d4(t, k, f))
        mus.append(_d4_inv(mu.float(), k, f))
        if log_var is not None:
            varis.append(_d4_inv(torch.exp(log_var.float().clamp(-20, 20)), k, f))

    mu_stack = torch.stack(mus)
    mu_mean = mu_stack.mean(0)
    if not varis:
        return mu_mean, None
    total = torch.stack(varis).mean(0)
    if len(mus) > 1:
        total = total + mu_stack.var(0, unbiased=False)
    return mu_mean, total


@torch.no_grad()
def infer_scene(model, rgb: np.ndarray, tile: int, overlap: int, device: str,
                amp_dtype, batch: int = 4, verbose: bool = True, tta: bool = False):
    H, W = rgb.shape[:2]
    stride = tile - overlap
    ys = list(range(0, max(1, H - tile + 1), stride))
    xs = list(range(0, max(1, W - tile + 1), stride))
    if ys[-1] + tile < H:
        ys.append(H - tile)
    if xs[-1] + tile < W:
        xs.append(W - tile)

    acc_mu = np.zeros((H, W), np.float64)
    acc_var = np.zeros((H, W), np.float64)
    acc_w = np.zeros((H, W), np.float64)
    taper = np.outer(cosine_window(tile, overlap // 2), cosine_window(tile, overlap // 2))

    coords = [(y, x) for y in ys for x in xs]
    if verbose:
        print(f"  {len(coords)} windows of {tile}px, stride {stride}px"
              + ("  x8 D4 test-time augmentation" if tta else ""))

    for i in range(0, len(coords), batch):
        chunk = coords[i: i + batch]
        crops = np.stack([rgb[y:y + tile, x:x + tile] for y, x in chunk]).astype(np.float32) / 255.0
        crops = (crops - IMAGENET_MEAN) / IMAGENET_STD
        t = torch.from_numpy(crops.transpose(0, 3, 1, 2)).to(device)

        mu_t, var_t = _predict_window(model, t, amp_dtype, device, tta)
        mu = mu_t.cpu().numpy()[:, 0]
        var = var_t.cpu().numpy()[:, 0] if var_t is not None else None

        for j, (y, x) in enumerate(chunk):
            acc_mu[y:y + tile, x:x + tile] += mu[j] * taper
            acc_w[y:y + tile, x:x + tile] += taper
            if var is not None:
                # Variances add under a weighted mean; sigmas do not. Averaging sigma
                # here would quietly understate uncertainty in every overlap band.
                acc_var[y:y + tile, x:x + tile] += var[j] * (taper ** 2)

    w = np.maximum(acc_w, 1e-8)
    height = acc_mu / w
    sigma = np.sqrt(acc_var / (w ** 2)) if acc_var.any() else None
    return height.astype(np.float32), (sigma.astype(np.float32) if sigma is not None else None)


def write_tif(path: Path, arr: np.ndarray, like: Path | None = None):
    import rasterio
    import warnings
    profile = {
        "driver": "GTiff", "height": arr.shape[0], "width": arr.shape[1],
        "count": 1, "dtype": "float32", "compress": "deflate",
    }
    if like is not None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with rasterio.open(like) as src:
                if src.crs:
                    profile["crs"] = src.crs
                    profile["transform"] = src.transform
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with rasterio.open(path, "w", **profile) as dst:
            dst.write(arr, 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", required=True, help="output prefix, e.g. out/JAX_004_007")
    ap.add_argument("--ckpt", default=None, help="omit for zero-shot DA-V2")
    ap.add_argument("--model-id", default=DEFAULT_MODEL)
    ap.add_argument("--tile", type=int, default=518, help="must be a multiple of 14")
    ap.add_argument("--overlap", type=int, default=140)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--tta", action="store_true",
                    help="average over the 8 D4 transforms. ~8x slower, and it adds a "
                         "genuine epistemic term to sigma via the law of total variance.")
    ap.add_argument("--height-scale", type=float, default=None)
    ap.add_argument("--precision", default="auto", choices=["auto", "bf16", "fp16", "fp32"])
    ap.add_argument("--truth", default=None, help="AGL GeoTIFF; if given, score the result")
    args = ap.parse_args()

    if args.tile % PATCH:
        raise SystemExit(f"--tile must be a multiple of {PATCH}; {args.tile} is not")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    from train import pick_precision
    amp_dtype, _, prec = pick_precision(args.precision)

    height_scale = args.height_scale or 30.0
    ckpt = None
    if args.ckpt:
        ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        height_scale = args.height_scale or ckpt.get("height_scale", height_scale)

    model = build(model_id=(ckpt or {}).get("model_id", args.model_id),
                  height_scale=height_scale).to(device).eval()
    if ckpt:
        model.load_state_dict(ckpt["model"])
        ep, rmse = ckpt.get("epoch", "?"), (ckpt.get("val") or {}).get("rmse")
        print(f"loaded {args.ckpt}  (epoch {ep}"
              + (f", val RMSE {rmse:.3f} m" if rmse else "") + ")")
    else:
        # Worth stating plainly. PLAN.md section 5 is built on the finding that DA-V2
        # degrades at nadir: it was trained on ground-level and oblique imagery full of
        # horizon and perspective cues that simply are not present looking straight
        # down. Zero-shot output is a relative surface, not metres, and this run exists
        # to prove the pipeline end to end -- not to be believed.
        print("no checkpoint: ZERO-SHOT DA-V2. Output is relative, not calibrated metres.")

    rgb = load_image(Path(args.image))
    print(f"{Path(args.image).name}  {rgb.shape[1]} x {rgb.shape[0]}  |  {prec}  |  {device}")

    t0 = time.time()
    height, sigma = infer_scene(model, rgb, args.tile, args.overlap, device, amp_dtype,
                                args.batch, tta=args.tta)
    dt = time.time() - t0

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_tif(out.with_suffix(".height.tif"), height, Path(args.image))
    if sigma is not None:
        write_tif(out.with_suffix(".sigma.tif"), sigma, Path(args.image))

    px = rgb.shape[0] * rgb.shape[1]
    print(f"  {dt:.2f} s  ({px/dt/1e6:.2f} Mpx/s)")
    print(f"  height {height.min():.2f} .. {height.max():.2f}  mean {height.mean():.2f}")
    if sigma is not None:
        print(f"  sigma  {sigma.min():.2f} .. {sigma.max():.2f}  mean {sigma.mean():.2f}")

    summary = {
        "image": str(args.image), "ckpt": args.ckpt, "zero_shot": args.ckpt is None, "tta": args.tta,
        "seconds": round(dt, 3), "mpx_per_s": round(px / dt / 1e6, 3),
        "height_min": float(height.min()), "height_max": float(height.max()),
        "height_mean": float(height.mean()),
    }

    if args.truth:
        from depthwizard.metrics import height_metrics, report
        import warnings
        import rasterio
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with rasterio.open(args.truth) as src:
                truth = src.read(1).astype(np.float32)
        cls_path = Path(args.truth).with_name(Path(args.truth).name.replace("_AGL", "_CLS"))
        cls = None
        if cls_path.exists():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with rasterio.open(cls_path) as src:
                    cls = src.read(1)
        mask = np.isfinite(truth)
        m = height_metrics(height, truth, mask, cls)
        print("\nagainst truth:")
        print("  " + str(m).replace("\n", "\n  "))
        if args.ckpt is None:
            # A zero-shot model predicts relative depth in arbitrary units, so raw RMSE
            # against metres is meaningless. The correlation is the number that actually
            # says whether the pretrained features carry height information at nadir --
            # and fitting the best possible affine map gives the fairest upper bound on
            # what a linear rescale alone could achieve.
            p, t = height[mask], truth[mask]
            A = np.stack([p, np.ones_like(p)], 1)
            coef, *_ = np.linalg.lstsq(A, t, rcond=None)
            fitted = height * coef[0] + coef[1]
            mf = height_metrics(fitted, truth, mask, cls)
            print(f"\n  after best-fit affine rescale (scale {coef[0]:.4f}, shift {coef[1]:+.3f}):")
            print("  " + str(mf).replace("\n", "\n  "))
            print("  ^ this is the ceiling for zero-shot + a linear fix. Fine-tuning has")
            print("    to beat it or differentiator 6.1 has nothing to stand on.")
            summary["zero_shot_affine"] = {"scale": float(coef[0]), "shift": float(coef[1]),
                                           **mf.to_dict()}
            write_tif(out.with_suffix(".height_affine.tif"), fitted.astype(np.float32),
                      Path(args.image))
        summary["metrics"] = m.to_dict()

    out.with_suffix(".json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {out.with_suffix('.height.tif')}")
    if sigma is not None:
        print(f"      {out.with_suffix('.sigma.tif')}")
    print(f"      {out.with_suffix('.json')}")


if __name__ == "__main__":
    main()
