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
import math
import time
from pathlib import Path

import numpy as np
import torch

from depthwizard.dataset import IMAGENET_MEAN, IMAGENET_STD
from depthwizard.model import DEFAULT_MODEL, build, from_checkpoint, PATCH


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
                amp_dtype, batch: int = 4, verbose: bool = True, tta: bool = False,
                zoom: int = 1):
    """Sliding-window inference.

    `zoom` trades ground coverage per window for angular resolution. The backbone has
    patch size 14, so a 518-px window is 37x37 tokens and each token covers 14 px -- 4.2 m
    at DFC2019's 0.3 m GSD, which puts a 12 m building at under three tokens across. Fine
    structure is simply below the model's native resolution, and no post-process recovers
    it (docs/probe-03-guided-filter.md).

    At zoom=2 the window covers 259 px of ground upsampled to 518, so a token spans 2.1 m
    and the same building gets six tokens. The cost is 4x the windows, and the risk is that
    objects now appear at twice the size the head was fine-tuned on -- which is a
    measurement, not a reason not to try.
    """
    H, W = rgb.shape[:2]
    # Guard the zoom before it becomes nonsense. auto-zoom derives this from the input's
    # GSD, so a Sentinel-2 scene at 10 m would ask for zoom 33 and a 15 px window -- barely
    # one patch, upsampled 33x into pure invention. Cap so a window is at least 5 patches
    # and say so, rather than returning a confident-looking hallucination.
    max_zoom = max(1, tile // (5 * PATCH))
    if zoom > max_zoom:
        print(f"  zoom x{zoom} would leave a {tile // zoom}px window, which is too little "
              f"real signal to upsample from. Capping at x{max_zoom}. The input is far "
              f"coarser than the {tile}px/0.3 m scale this checkpoint was trained at.")
        zoom = max_zoom
    src = tile // zoom                 # ground pixels per window
    src_overlap = overlap // zoom
    stride = src - src_overlap
    ys = list(range(0, max(1, H - src + 1), stride))
    xs = list(range(0, max(1, W - src + 1), stride))
    if ys[-1] + src < H:
        ys.append(H - src)
    if xs[-1] + src < W:
        xs.append(W - src)

    acc_mu = np.zeros((H, W), np.float64)
    acc_var = np.zeros((H, W), np.float64)
    acc_w = np.zeros((H, W), np.float64)
    taper = np.outer(cosine_window(src, src_overlap // 2), cosine_window(src, src_overlap // 2))

    coords = [(y, x) for y in ys for x in xs]
    if verbose:
        print(f"  {len(coords)} windows of {src}px ground -> {tile}px input"
              + (f"  zoom x{zoom}" if zoom > 1 else "")
              + ("  x8 D4 test-time augmentation" if tta else ""))

    for i in range(0, len(coords), batch):
        chunk = coords[i: i + batch]
        crops = np.stack([rgb[y:y + src, x:x + src] for y, x in chunk]).astype(np.float32) / 255.0
        crops = (crops - IMAGENET_MEAN) / IMAGENET_STD
        t = torch.from_numpy(crops.transpose(0, 3, 1, 2)).to(device)
        if zoom > 1:
            # Bicubic up, so the backbone sees a plausible image rather than a blocky one.
            t = torch.nn.functional.interpolate(t, size=(tile, tile), mode="bicubic",
                                                align_corners=False)

        mu_t, var_t = _predict_window(model, t, amp_dtype, device, tta)
        if zoom > 1:
            # Back to ground resolution. Area averaging, because the prediction is a
            # quantity per pixel and bilinear would bias the overlap accumulation.
            mu_t = torch.nn.functional.interpolate(mu_t, size=(src, src), mode="area")
            if var_t is not None:
                var_t = torch.nn.functional.interpolate(var_t, size=(src, src), mode="area")
        mu = mu_t.cpu().numpy()[:, 0]
        var = var_t.cpu().numpy()[:, 0] if var_t is not None else None

        for j, (y, x) in enumerate(chunk):
            acc_mu[y:y + src, x:x + src] += mu[j] * taper
            acc_w[y:y + src, x:x + src] += taper
            if var is not None:
                # Variances add under a weighted mean; sigmas do not. Averaging sigma
                # here would quietly understate uncertainty in every overlap band.
                acc_var[y:y + src, x:x + src] += var[j] * (taper ** 2)

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
    ap.add_argument("--dem", default=None, metavar="auto|PATH",
                    help="anchor above-ground heights to absolute elevation. 'auto' fetches "
                         "Copernicus GLO-30 for the scene footprint (open, no account); or "
                         "give a path to your own DEM. Writes <out>.dsm.tif.")
    ap.add_argument("--dem-bare", action="store_true",
                    help="approximate bare earth from the DEM before adding our heights. "
                         "The 30 m anchor is a SURFACE model, so without this dense urban "
                         "areas double-count part of the building height.")
    ap.add_argument("--gcp", default=None, metavar="JSON",
                    help="JSON list of [x, y, elevation] ground control points in the "
                         "input CRS. Fits an offset by default; add --gcp-scale to also "
                         "fit a scale.")
    ap.add_argument("--gcp-form", default="power", choices=["power", "affine"],
                    help="functional form for --gcp-scale. 'power' (elevation = a*AGL**b) "
                         "is the default because the error is a height-dependent RATIO, "
                         "not an offset: measured -24.9%% RMSE on tall tiles vs affine's "
                         "-15.1%%, with MAE unchanged where affine costs +11.6%%.")
    ap.add_argument("--gcp-scale", action="store_true",
                    help="fit scale AND offset from the control points, correcting the "
                         "measured height compression (ours = 0.483*truth + 2.49 m). Needs "
                         "5+ points spanning 5+ m. A TRADE, measured: -21.3%% RMSE on tall "
                         "structures, +18%% MAE on short ones. Off by default so the "
                         "shipped default stays the conservative one.")
    ap.add_argument("--auto-zoom", action="store_true",
                    help="read the ground sample distance from a georeferenced input and "
                         "upsample so the backbone sees the scale it was trained at. "
                         "Recovers most of the accuracy lost on coarser imagery (probe 05).")
    ap.add_argument("--native-gsd", type=float, default=0.3,
                    help="metres per pixel the checkpoint was fine-tuned at")
    ap.add_argument("--fuse-zoom", type=int, default=1,
                    help="fuse a zoom-1 pass with a zoomed pass: accuracy from the first, "
                         "edge detail from the second. 2 is measured; see probe-04.")
    ap.add_argument("--fuse-sigma", type=float, default=8.0,
                    help="crossover for the fusion, in pixels. Larger takes more detail "
                         "from the zoomed pass: 8 gives +29%% sharpness for +1%% RMSE.")
    ap.add_argument("--zoom", type=int, default=1,
                    help="ground pixels per window = tile/zoom, upsampled to tile before "
                         "the backbone. zoom 2 halves the ground area a token covers "
                         "(4.2 m -> 2.1 m at 0.3 m GSD) at 4x the windows.")
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

    if ckpt:
        # One place knows how to rebuild a model from a checkpoint, so adding a head
        # option cannot silently break inference.
        model = from_checkpoint(ckpt, height_scale=height_scale).to(device).eval()
    else:
        model = build(model_id=args.model_id,
                      height_scale=height_scale).to(device).eval()

    if ckpt:
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

    # ---- resolution matching -------------------------------------------------------
    # The model was fine-tuned at 0.3 m. Probe 05 measured what happens when it is not
    # given that: at 0.6 m per-building RMSE goes 1.350 -> 2.260 and bias -0.16 -> -1.67,
    # because a token covers twice the ground and buildings get averaged with their
    # surroundings. Upsampling the input back to the trained scale recovers almost all of
    # it (2.260 -> 1.399, bias -0.29) at no training cost.
    #
    # This matters for evaluation, not just tidiness: ISRO will score on their own
    # imagery, which is nearer 0.6-1.0 m than 0.3 m.
    if args.auto_zoom:
        gsd = None
        if Path(args.image).suffix.lower() in (".tif", ".tiff"):
            import rasterio, warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with rasterio.open(args.image) as src:
                    if src.crs:
                        if src.crs.is_geographic:
                            # Degrees, not metres. linear_units_factor RAISES on a
                            # geographic CRS rather than returning anything, and lat/lon
                            # GeoTIFFs are a normal delivery format -- so convert instead
                            # of declining. A degree of latitude is ~111.32 km; a degree of
                            # longitude shrinks by cos(latitude), so use the scene centre.
                            lat = src.bounds.bottom + (src.bounds.top - src.bounds.bottom) / 2
                            gsd = abs(src.transform.a) * 111320.0 * math.cos(math.radians(lat))
                        else:
                            try:
                                if src.crs.linear_units_factor[1] == 1.0:
                                    gsd = abs(src.transform.a)
                            except Exception:
                                gsd = None
        if gsd is None:
            print("  auto-zoom: no metric transform on the input, so its ground sample "
                  "distance is unknown. Leaving zoom at 1; pass --zoom to force it.")
        else:
            z = max(1, int(round(gsd / args.native_gsd)))
            print(f"  auto-zoom: input is {gsd:.2f} m/px against the {args.native_gsd:.2f} m "
                  f"the model was trained at -> zoom x{z}")
            args.zoom = z
            
    t0 = time.time()
    if args.fuse_zoom > 1:
        # Laplacian fusion (docs/probe-04-resolution.md). The zoom-1 pass carries the
        # absolute calibration and supplies the low frequencies; the zoomed pass resolves
        # roof edges the 4.2 m token footprint cannot see, and supplies the high ones.
        #
        # No TTA on the zoomed pass, deliberately: averaging eight D4 views is a smoothing
        # operation, and smoothing is the opposite of what that pass is there to provide.
        from scipy.ndimage import gaussian_filter
        base, sigma = infer_scene(model, rgb, args.tile, args.overlap, device, amp_dtype,
                                  args.batch, tta=args.tta, zoom=args.zoom)
        detail, _ = infer_scene(model, rgb, args.tile, args.overlap, device, amp_dtype,
                                args.batch, tta=False, zoom=args.zoom * args.fuse_zoom)
        g = float(args.fuse_sigma)
        height = (gaussian_filter(base, g) + (detail - gaussian_filter(detail, g))
                  ).astype(np.float32)
        # Uncertainty stays the zoom-1 head's: it was calibrated at that scale, and the
        # zoomed pass contributes no absolute level for it to describe.
    else:
        height, sigma = infer_scene(model, rgb, args.tile, args.overlap, device, amp_dtype,
                                    args.batch, tta=args.tta, zoom=args.zoom)
    dt = time.time() - t0

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_tif(out.with_suffix(".height.tif"), height, Path(args.image))
    if sigma is not None:
        write_tif(out.with_suffix(".sigma.tif"), sigma, Path(args.image))

    # ---- absolute DSM ---------------------------------------------------------------
    # The model predicts height ABOVE GROUND, which is metric but is not a DSM. The brief
    # asks georeferenced input to yield "an Absolute Digital Surface Model with metric
    # height values", so anchor to a coarse DEM exactly as it suggests:
    #     DSM = terrain elevation + our above-ground height
    dsm, summary_dsm = None, None
    if args.dem or args.gcp:
        import rasterio, warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with rasterio.open(args.image) as src:
                crs, tr, bnds = src.crs, src.transform, src.bounds
        if crs is None:
            print("  absolute DSM: the input has no coordinate system, so there is nothing "
                  "to anchor to. This is the relative-DSM path; the height map above is the "
                  "product.")
        else:
            from depthwizard.dem import (elevation_on_grid, to_bare_earth,
                                         fit_gcp_offset, fit_gcp_affine, fit_gcp_power)
            terrain, datum_note = None, ""
            if args.gcp:
                pts = json.loads(Path(args.gcp).read_text())
                print(f"  absolute DSM: anchoring on {len(pts)} ground control points")
                fit = None
                if args.gcp_scale:
                    # Power law by default: measured -24.9% RMSE on tall tiles against
                    # affine's -15.1%, and it leaves MAE where it found it while affine
                    # costs +11.6%. See tools/analysis/calibration_form.py.
                    if args.gcp_form == "power":
                        pw = fit_gcp_power(height, tr, pts)
                        if pw is not None:
                            a_, b_, n, rms = pw
                            dsm = a_ * np.maximum(height, 0.05) ** b_
                            datum_note = (f"elevation = {a_:.3f}*AGL**{b_:.3f} from {n} "
                                          f"control points (residual RMS {rms:.2f} m)")
                            fit = pw
                    else:
                        af = fit_gcp_affine(height, tr, pts)
                        if af is not None:
                            scale, off, n, rms = af
                            dsm = height * scale + off
                            datum_note = (f"elevation = {scale:.3f}*AGL {off:+.2f} m from "
                                          f"{n} control points (residual RMS {rms:.2f} m)")
                            fit = af
                if fit is None:
                    off, n, spread = fit_gcp_offset(height, tr, pts)
                    if off is not None:
                        dsm = height + off
                        datum_note = (f"offset {off:+.2f} m from {n} control points "
                                      f"(68% spread +/-{spread:.2f} m)")
            elif args.dem == "auto":
                print("  absolute DSM: fetching Copernicus GLO-30 for the scene footprint")
                terrain, used = elevation_on_grid(tr, crs, height.shape[1], height.shape[0],
                                                  bnds, crs)
                datum_note = f"Copernicus GLO-30 ({', '.join(used)})" if used else ""
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    with rasterio.open(args.dem) as d:
                        from rasterio.warp import Resampling, reproject
                        terrain = np.full(height.shape, np.nan, np.float32)
                        reproject(rasterio.band(d, 1), terrain, src_transform=d.transform,
                                  src_crs=d.crs, dst_transform=tr, dst_crs=crs,
                                  resampling=Resampling.cubic)
                datum_note = f"user DEM {Path(args.dem).name}"

            if terrain is not None:
                gsd_m = abs(tr.a) if not crs.is_geographic else abs(tr.a) * 111320.0
                if args.dem_bare:
                    terrain = to_bare_earth(terrain, gsd_m)
                    datum_note += ", low-percentile bare-earth approximation"
                dsm = (terrain + height).astype(np.float32)
                # Write the bare terrain too. The viewer needs the ground and the
                # above-ground height as SEPARATE layers, not their sum: it stands our
                # heights on the terrain to render a true surface, while still colouring
                # by above-ground metres. Handing it only the summed DSM would make the
                # height ramp silently become an elevation map.
                write_tif(out.with_suffix(".terrain.tif"), terrain.astype(np.float32),
                          Path(args.image))

            if dsm is not None:
                write_tif(out.with_suffix(".dsm.tif"), dsm.astype(np.float32),
                          Path(args.image))
                summary_dsm = {
                    "source": datum_note,
                    "min_m": float(np.nanmin(dsm)), "max_m": float(np.nanmax(dsm)),
                    "vertical_datum": "EGM2008 geoid (Copernicus GLO-30)" if args.dem == "auto"
                                      else "as supplied",
                    "caveat": ("the 30 m anchor is itself a surface model, so dense urban "
                               "areas double-count part of the building height"),
                }
                print(f"  absolute DSM: {dsm.min():.1f} .. {dsm.max():.1f} m  [{datum_note}]")
            else:
                print("  absolute DSM: could not be anchored; height map is relative to ground")

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

    if summary_dsm is not None:
        summary["absolute_dsm"] = summary_dsm
    out.with_suffix(".json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {out.with_suffix('.height.tif')}")
    if sigma is not None:
        print(f"      {out.with_suffix('.sigma.tif')}")
    if dsm is not None:
        print(f"      {out.with_suffix('.dsm.tif')}   <- absolute DSM, metres above the "
              f"geoid")
    print(f"      {out.with_suffix('.json')}")


if __name__ == "__main__":
    main()
