"""Score a DepthWizard prediction against Google Open Buildings 2.5D over India.

This is the only quantitative check we have on Indian ground. DFC2019 is Jacksonville and
Omaha; Bhoonidhi cannot referee buildings at all. Open Buildings publishes AGL building
height over South Asia, so it can -- with three caveats that shape how this script works.

**It is a model, not truth.** Open Buildings heights are inferred from Sentinel-2 at 10 m
and published at 4 m effective resolution. Agreement across many buildings is evidence our
absolute scale transfers to India. A single-building disagreement is evidence of nothing.
Every number this prints is a level of agreement between two models, and it is labelled
that way on purpose.

**Buildings, not pixels.** Their footprint edges are Sentinel-2 blurry and ours are not, so
a per-pixel score at 0.5 m mostly measures their blur. We compare per building: their
median height inside a component against our robust height inside the same component. That
is the comparison a jury cares about anyway -- "how tall is that building", not "what does
pixel 4,891,203 say".

**The image leans.** Our Sikkim imagery is 26 degrees off-nadir, so a roof is displaced
from its own footprint by height * tan(26 deg) -- about 5 m for a 10 m building, sixteen
pixels. Open Buildings footprints come from near-nadir Sentinel-2. Comparing without
correcting that would charge us for a co-registration offset and call it height error. So
we recover a single global shift by FFT phase correlation first, report it, and check it
against the lean the view geometry predicts. If the recovered shift disagrees with the
predicted lean, that is a finding, not something to quietly absorb.

Confidence intervals resample **buildings**, never pixels. Neighbouring pixels in a roof
are almost perfectly correlated, so a pixel bootstrap would report a fake precision. This
project has already been burnt once by resampling the correlated unit instead of the
independent one.

    python tools/compare_open_buildings.py \\
        --pred out/sikkim_1232.height.tif \\
        --ob data/open_buildings/45_120220211232_..._obheight_2022.tif \\
        --off-nadir 26.2 --sat-azimuth 24.0
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[2]


def load_aligned(pred_path: Path, ob_path: Path):
    """Both rasters on the Open Buildings grid. Ours is averaged down, not sampled.

    Averaging matters: our 0.305 m prediction carries detail theirs cannot, and nearest
    sampling would let one lucky pixel speak for a 0.5 m cell.
    """
    with rasterio.open(ob_path) as ob:
        prof, ob_arr = ob.profile, ob.read(1)
        ob_arr = np.where(ob_arr == ob.nodata, np.nan, ob_arr)
    with rasterio.open(pred_path) as p:
        if p.crs != prof["crs"]:
            raise SystemExit(f"CRS mismatch: pred {p.crs} vs ob {prof['crs']}")
        with WarpedVRT(p, crs=prof["crs"], transform=prof["transform"],
                       width=prof["width"], height=prof["height"],
                       resampling=Resampling.average) as v:
            pred = v.read(1).astype("float32")
            if p.nodata is not None:
                pred = np.where(pred == p.nodata, np.nan, pred)
    return pred, ob_arr.astype("float32"), prof


def phase_shift(a: np.ndarray, b: np.ndarray, max_px: int, coarse: int = 4):
    """Global (dy, dx) that best aligns a onto b, by FFT phase correlation.

    Run on a coarsened copy: the offset we are hunting is metres, not centimetres, and a
    full-resolution FFT of a 10k square buys nothing but time.
    """
    def prep(x):
        x = np.nan_to_num(x, nan=0.0)
        x = x[: x.shape[0] // coarse * coarse, : x.shape[1] // coarse * coarse]
        x = x.reshape(x.shape[0] // coarse, coarse, x.shape[1] // coarse, coarse).mean((1, 3))
        x = x - x.mean()
        # Hann window: without it the FFT sees the tile border as the strongest edge
        # in the scene and locks onto it instead of onto the buildings.
        wy = np.hanning(x.shape[0])[:, None]
        wx = np.hanning(x.shape[1])[None, :]
        return x * wy * wx

    A, B = np.fft.rfft2(prep(a)), np.fft.rfft2(prep(b))
    R = A * np.conj(B)
    R /= np.abs(R) + 1e-9
    c = np.fft.irfft2(R)
    lim = max(1, max_px // coarse)
    # Peak must lie in the plausible band; wrap-around means both ends of each axis.
    idx = [(dy, dx) for dy in range(-lim, lim + 1) for dx in range(-lim, lim + 1)]
    vals = [c[dy % c.shape[0], dx % c.shape[1]] for dy, dx in idx]
    dy, dx = idx[int(np.argmax(vals))]
    return dy * coarse, dx * coarse, float(max(vals))


def shift_arr(a: np.ndarray, dy: int, dx: int):
    out = np.full_like(a, np.nan)
    ys, ye = max(0, dy), min(a.shape[0], a.shape[0] + dy)
    xs, xe = max(0, dx), min(a.shape[1], a.shape[1] + dx)
    out[ys - dy:ye - dy, xs - dx:xe - dx] = a[ys:ye, xs:xe]
    return out


def per_building(pred, ob, res, min_area_m2, our_pct):
    """One row per Open Buildings component: their median height against our robust one."""
    mask = np.isfinite(ob) & (ob > 0.5)
    lab, n = ndimage.label(mask, structure=np.ones((3, 3)))
    if n == 0:
        raise SystemExit("no building components in the Open Buildings raster")
    idx = np.arange(1, n + 1)
    # bincount, not ndimage.sum over an ones array: the latter allocates a second copy
    # of a 113-megapixel raster to count pixels it already has labels for.
    counts = np.bincount(lab.ravel(), minlength=n + 1)[1:].astype("float64")
    keep = counts * res * res >= min_area_m2

    ob_med = ndimage.median(ob, lab, idx)
    ours_ok = np.isfinite(pred)
    cov = ndimage.mean(ours_ok.astype("float32"), lab, idx)
    safe = np.where(ours_ok, pred, 0.0)

    # A percentile, not a mean: a roof component includes its own shadowed eaves, and at
    # 26 degrees off-nadir it also clips a little ground on the lit side.
    ours = _fast_pct(safe, lab, idx, ours_ok, keep, cov, our_pct)

    good = keep & (cov > 0.5) & np.isfinite(ours) & np.isfinite(ob_med)
    return {
        "n_components": int(n),
        "n_scored": int(good.sum()),
        "area_m2": counts[good] * res * res,
        "ob": ob_med[good],
        "ours": ours[good],
        # kept so the caller can re-summarise footprints at another percentile without
        # paying for the labelling and the sort a second time
        "lab": lab, "idx": idx, "keep": keep, "cov": cov, "good": good,
    }


def _fast_pct(safe, lab, idx, ok, keep, cov, pct):
    """Per-component percentile in one pass.

    The obvious `safe[lab == i]` per component is O(components x pixels): on a 10k tile
    with a few thousand buildings that is hours of masking a 113-megapixel array. Sort the
    labelled pixels once instead, then each component is a contiguous slice.
    """
    flat_l, flat_v, flat_ok = lab.ravel(), safe.ravel(), ok.ravel()
    sel = (flat_l > 0) & flat_ok
    order = np.argsort(flat_l[sel], kind="stable")
    ls, vs = flat_l[sel][order], flat_v[sel][order]
    bounds = np.searchsorted(ls, idx)
    ends = np.searchsorted(ls, idx, side="right")
    out = np.full(idx.size, np.nan)
    for k in range(idx.size):
        if not (keep[k] and cov[k] > 0.5) or ends[k] <= bounds[k]:
            continue
        out[k] = np.percentile(vs[bounds[k]:ends[k]], pct)
    return out


def bootstrap(ours, ob, n_boot, rng):
    """Resample BUILDINGS. Pixels inside one roof are not independent observations."""
    n = ours.size
    stats = {"bias": [], "mae": [], "rmse": [], "r": []}
    for _ in range(n_boot):
        s = rng.integers(0, n, n)
        d = ours[s] - ob[s]
        stats["bias"].append(d.mean())
        stats["mae"].append(np.abs(d).mean())
        stats["rmse"].append(math.sqrt((d ** 2).mean()))
        stats["r"].append(np.corrcoef(ours[s], ob[s])[0, 1])
    return {k: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
            for k, v in stats.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", required=True, help="height GeoTIFF from infer.py")
    ap.add_argument("--ob", required=True, help="height GeoTIFF from open_buildings.py")
    ap.add_argument("--min-area", type=float, default=25.0, help="m2; drop specks")
    ap.add_argument("--our-pct", type=float, default=50.0,
                    help="percentile of our height inside a footprint. Defaults to 50 so "
                         "it is like-for-like against their median: a self-test with a "
                         "planted +0.8 m bias reported +2.1 m at p75, because a higher "
                         "percentile of a noisy field is higher for free.")
    ap.add_argument("--max-shift", type=float, default=40.0, help="metres to search")
    ap.add_argument("--no-shift", action="store_true", help="skip co-registration")
    ap.add_argument("--off-nadir", type=float, help="degrees, for the lean prediction")
    ap.add_argument("--sat-azimuth", type=float, help="degrees clockwise from north")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--out")
    a = ap.parse_args()

    pred, ob, prof = load_aligned(Path(a.pred), Path(a.ob))
    res = abs(prof["transform"].a)
    print(f"grid {prof['width']} x {prof['height']} at {res} m  ({prof['crs']})")
    print(f"our coverage {100*np.isfinite(pred).mean():.1f}%   "
          f"their buildings {100*np.nanmean(ob > 0.5):.2f}% of pixels")

    dy = dx = 0
    if not a.no_shift:
        dy, dx, peak = phase_shift(pred, np.nan_to_num(ob), int(a.max_shift / res))
        print(f"\nco-registration: shift our raster by "
              f"dy {dy*res:+.1f} m, dx {dx*res:+.1f} m  (peak {peak:.4f})")
        if a.off_nadir is not None:
            lean = math.tan(math.radians(a.off_nadir))
            print(f"  view geometry predicts a lean of {lean:.2f} m per metre of height, "
                  f"so ~{lean*8:.1f} m for a typical 8 m building")
            if a.sat_azimuth is not None:
                az = math.radians(a.sat_azimuth)
                print(f"  predicted lean direction: dx {math.sin(az):+.2f}, "
                      f"dy {-math.cos(az):+.2f} (unit); recovered "
                      f"dx {dx/max(abs(dx)+abs(dy),1):+.2f}, "
                      f"dy {dy/max(abs(dx)+abs(dy),1):+.2f}")
        pred = shift_arr(pred, dy, dx)

    b = per_building(pred, ob, res, a.min_area, a.our_pct)
    ours, obh = b["ours"], b["ob"]
    d = ours - obh
    print(f"\n{b['n_scored']:,} buildings scored of {b['n_components']:,} components "
          f"(>= {a.min_area:g} m2, our coverage > 50%)")
    print(f"  their height  median {np.median(obh):5.1f} m   p90 {np.percentile(obh,90):5.1f} m")
    print(f"  our height    median {np.median(ours):5.1f} m   p90 {np.percentile(ours,90):5.1f} m")

    rng = np.random.default_rng(a.seed)
    ci = bootstrap(ours, obh, a.boot, rng)
    rmse = math.sqrt((d ** 2).mean())
    r = float(np.corrcoef(ours, obh)[0, 1])
    print(f"\nagreement with Open Buildings (two models, not truth):")
    print(f"  bias  {d.mean():+6.2f} m   95% CI [{ci['bias'][0]:+.2f}, {ci['bias'][1]:+.2f}]")
    print(f"  MAE   {np.abs(d).mean():6.2f} m   95% CI [{ci['mae'][0]:.2f}, {ci['mae'][1]:.2f}]")
    print(f"  RMSE  {rmse:6.2f} m   95% CI [{ci['rmse'][0]:.2f}, {ci['rmse'][1]:.2f}]")
    print(f"  r     {r:+6.3f}   95% CI [{ci['r'][0]:+.3f}, {ci['r'][1]:+.3f}]")

    # How much of the bias is the summary statistic rather than the model? A percentile
    # sweep answers it in one line, and stops anyone (us included) picking the flattering
    # one after the fact.
    print("\n  sensitivity to how we summarise a footprint:")
    ok_m = np.isfinite(pred)
    safe_m = np.where(ok_m, pred, 0.0)
    for p in (25.0, 50.0, 75.0, 90.0):
        alt = _fast_pct(safe_m, b["lab"], b["idx"], ok_m, b["keep"], b["cov"], p)[b["good"]]
        print(f"    our p{p:<4.0f} vs their median: bias {(alt - obh).mean():+6.2f} m, "
              f"MAE {np.abs(alt - obh).mean():5.2f} m")

    big = b["area_m2"] >= 100.0
    if big.sum() > 30:
        db = d[big]
        print(f"\n  buildings >= 100 m2 only ({big.sum():,}): bias {db.mean():+.2f} m, "
              f"MAE {np.abs(db).mean():.2f} m, r "
              f"{np.corrcoef(ours[big], obh[big])[0,1]:+.3f}")

    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps({
            "pred": a.pred, "ob": a.ob, "shift_m": [dy * res, dx * res],
            "n_scored": b["n_scored"], "bias": float(d.mean()),
            "mae": float(np.abs(d).mean()), "rmse": rmse, "r": r, "ci": ci,
            "note": "agreement between two models; Open Buildings is Sentinel-2 derived, "
                    "not ground truth",
        }, indent=2))
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
