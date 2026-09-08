"""Coarse DEM anchoring: turn above-ground heights into absolute elevation.

The problem statement asks for two different products:

    non-georeferenced (PNG/JPG)  ->  relative DSM
    georeferenced (GeoTIFF)      ->  ABSOLUTE DSM, metric height values

Our model predicts height **above ground** (AGL / nDSM), which is neither. It is already
metric -- that part needs no calibration, because the head was fine-tuned to regress metres
directly rather than to produce a scale-free depth. But a DSM is the elevation of the
surface above a vertical datum, so AGL has to be added to bare earth:

    DSM = terrain elevation + our above-ground height

The problem statement names the anchor it expects: *"a lower-resolution DEM source such as
SRTM 30 m"*. We use Copernicus GLO-30 instead, for three reasons: it is open with no
account or key, it is 30 m like SRTM, and it covers latitudes above 60 degrees N where SRTM
simply has no data. CartoDEM would be the obvious Indian choice and is deliberately not
used -- it is free only to Indian Government Entities, and as students we are not one.

**An honesty note that belongs in the report.** GLO-30 is itself a *surface* model derived
from TanDEM-X, not bare earth, so in dense urban areas it already contains part of the
building height we are about to add. At 30 m posting a single building contributes only a
fraction of a cell, but the double-count is real and is largest exactly where buildings are
tallest and densest. SRTM has the same property. Removing it properly needs a true bare-
earth product (FABDEM), whose licence is non-commercial. `--dem-bare` applies a coarse
low-percentile filter as an approximation, and says what it is rather than pretending the
composite is clean.
"""
from __future__ import annotations

import math

import numpy as np

BUCKET = "https://copernicus-dem-30m.s3.amazonaws.com"


def _tile_name(lat: int, lon: int) -> str:
    ns = "N" if lat >= 0 else "S"
    ew = "E" if lon >= 0 else "W"
    return (f"Copernicus_DSM_COG_10_{ns}{abs(lat):02d}_00_"
            f"{ew}{abs(lon):03d}_00_DEM")


def tiles_for_bounds(west: float, south: float, east: float, north: float) -> list[str]:
    """Every 1x1 degree GLO-30 tile touching the box, as /vsicurl/ URLs.

    A scene straddling a tile edge is normal, not exotic -- assuming one tile is how a
    coastal or border scene ends up with half its elevation missing and no error raised.
    """
    urls = []
    for lat in range(int(math.floor(south)), int(math.floor(north)) + 1):
        for lon in range(int(math.floor(west)), int(math.floor(east)) + 1):
            t = _tile_name(lat, lon)
            urls.append(f"/vsicurl/{BUCKET}/{t}/{t}.tif")
    return urls


def elevation_on_grid(dst_transform, dst_crs, width: int, height: int,
                      bounds, src_crs, verbose: bool = True):
    """Copernicus GLO-30 elevation resampled onto an arbitrary raster grid.

    Returns (array, list_of_tiles_used) or (None, []) if nothing could be fetched.
    """
    import rasterio
    from rasterio.warp import Resampling, reproject, transform_bounds

    ll = transform_bounds(src_crs, "EPSG:4326", *bounds)
    urls = tiles_for_bounds(*ll)

    out = np.full((height, width), np.nan, np.float32)
    used = []
    for url in urls:
        try:
            with rasterio.open(url) as src:
                buf = np.full((height, width), np.nan, np.float32)
                # Cubic, not nearest: we are upsampling 30 m posts to sub-metre pixels and
                # nearest gives 100-pixel stair steps that read as a rendering fault.
                reproject(rasterio.band(src, 1), buf,
                          src_transform=src.transform, src_crs=src.crs,
                          dst_transform=dst_transform, dst_crs=dst_crs,
                          src_nodata=src.nodata, dst_nodata=np.nan,
                          resampling=Resampling.cubic)
                fill = np.isnan(out) & np.isfinite(buf)
                if fill.any():
                    out[fill] = buf[fill]
                    used.append(url.rsplit("/", 1)[-1])
        except Exception as e:                       # missing tile = ocean, or no network
            if verbose:
                print(f"    {url.rsplit('/', 1)[-1]}: {type(e).__name__}")
    if not used:
        return None, []
    if np.isnan(out).any():
        # Partial coverage is usually ocean inside the box. Fill flat rather than leaving
        # holes that would punch through the mesh.
        out[np.isnan(out)] = float(np.nanmedian(out))
    return out, used


def to_bare_earth(elev: np.ndarray, gsd_m: float, window_m: float = 180.0,
                  percentile: float = 10.0) -> np.ndarray:
    """Crude bare-earth approximation: a low percentile over a wide window.

    GLO-30 is a surface model, so adding our above-ground heights to it double-counts
    buildings and canopy. Taking a low percentile over a window much wider than a building
    keeps the terrain trend and drops what sits on it.

    This is an approximation and it has a real cost: over genuinely steep ground the low
    percentile also shaves the terrain itself, biasing elevation downward on slopes. It is
    off by default for that reason.
    """
    from scipy.ndimage import percentile_filter, zoom
    # Work coarse. A 180 m window at 0.3 m/px is a 600x600 kernel over millions of pixels,
    # which takes minutes and pins a core. The terrain estimate is smooth by construction,
    # so compute it at ~10 m posts and interpolate back up: same answer, seconds not
    # minutes. (The first version of this ran at full resolution and appeared to hang.)
    step = max(1, int(round(10.0 / max(gsd_m, 1e-6))))
    coarse = elev[::step, ::step]
    size = max(3, int(round(window_m / (gsd_m * step))))
    filt = percentile_filter(coarse, percentile, size=size, mode="nearest")
    if step == 1:
        return filt.astype(np.float32)
    out = zoom(filt, (elev.shape[0] / filt.shape[0], elev.shape[1] / filt.shape[1]), order=1)
    # zoom's rounding can land a pixel short; pad or crop to match exactly.
    fixed = np.empty_like(elev)
    h = min(out.shape[0], elev.shape[0]); w = min(out.shape[1], elev.shape[1])
    fixed[:h, :w] = out[:h, :w]
    if h < elev.shape[0]:
        fixed[h:, :w] = fixed[h - 1, :w]
    if w < elev.shape[1]:
        fixed[:, w:] = fixed[:, w - 1:w]
    return fixed.astype(np.float32)


def _structure_median(pred_agl: np.ndarray, r: int, c: int,
                      max_px: int = 20000, tol: float = 0.30) -> float | None:
    """Median predicted height of the STRUCTURE a control point sits on, not a fixed patch.

    A control point marks a building, and the gain this calibration was measured to give
    (-24.9% RMSE on tall tiles) was measured with per-BUILDING height estimates. Reading a
    fixed 5x5 patch instead is a different quantity and a much noisier one: measured on
    OMA_288_012, point samples on small structures returned 0.59 m where the truth was
    5.6 m, because a small building is not reliably resolved at a single location. That made
    the ratio FALL with height (4.86 at 3 m down to 1.98 at 63 m), inverted the fitted
    exponent to 0.567, and the guard -- correctly -- refused a fit that had been posed
    wrongly.

    So grow a region from the point over connected pixels of similar predicted height, and
    take its median. That is the same quantity the measurement used.
    """
    H, W = pred_agl.shape
    seed = pred_agl[r, c]
    if not np.isfinite(seed):
        return None
    # Seed from a small neighbourhood so a single noisy pixel does not set the band.
    r0, r1 = max(0, r - 2), min(H, r + 3)
    c0, c1 = max(0, c - 2), min(W, c + 3)
    local = pred_agl[r0:r1, c0:c1]
    local = local[np.isfinite(local)]
    if not local.size:
        return None
    seed = float(np.median(local))
    band = max(1.0, abs(seed) * tol)         # a metre of slack minimum, else 30%
    lo, hi = seed - band, seed + band

    seen = np.zeros((H, W), bool)
    stack = [(r, c)]
    seen[r, c] = True
    vals = []
    while stack and len(vals) < max_px:
        i, j = stack.pop()
        v = pred_agl[i, j]
        if not np.isfinite(v) or not (lo <= v <= hi):
            continue
        vals.append(float(v))
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            a, b = i + di, j + dj
            if 0 <= a < H and 0 <= b < W and not seen[a, b]:
                seen[a, b] = True
                stack.append((a, b))
    if len(vals) < 9:                        # too small to trust; fall back to the patch
        return float(np.median(local))
    return float(np.median(vals))


def fit_gcp_power(pred_agl: np.ndarray, transform, gcps, min_points: int = 5,
                  min_span: float = 5.0, verbose: bool = True):
    """Fit elevation = a * AGL**b from control points. The right form, measured.

    An affine calibration cannot work here and we have the measurement to say why. The
    model's error is a RATIO that varies with height -- short structures come out 1.20x too
    tall at 0-3 m and 1.14x at 3-6 m, tall ones at 0.52x above 40 m (probe 06). A straight
    line cannot be above 1 at one end and below 1 at the other, so an affine fit from
    control points that are mostly short tilts the wrong way for the tall end: measured end
    to end on OMA_288_012 as scale 0.830, taking tile RMSE from 23.62 to 25.09 m.

    A power law can. With b > 1 it expands the tall end while contracting the short end,
    which is the shape the error actually has, and it costs the same two parameters.

    Measured head to head on held-out buildings, fitting per tile and scoring only on
    buildings NOT used as control points (`tools/analysis/calibration_form.py`), over the
    10 validation tiles containing a building above 20 m:

        form            k   RMSE      vs uncalibrated   MAE      tiles worse
        uncalibrated    -   7.248 m         --          4.013 m       --
        affine          8   6.154 m      -15.1%         4.477 m      8 of 10
        power           5   5.730 m      -20.9%         4.202 m      4 of 10
        power           8   5.442 m      -24.9%         4.038 m      5 of 10

    Affine buys RMSE by spending MAE (+11.6%). The power law takes more RMSE and leaves MAE
    where it found it (+0.6%). Over all 57 usable tiles it is the only form that improves
    on doing nothing at all (-4.4% at k=8, against affine's +0.1%).
    """
    inv = ~transform
    xs, zs = [], []
    for x, y, z in gcps:
        c, r = inv * (x, y)
        c, r = int(round(c)), int(round(r))
        if not (0 <= r < pred_agl.shape[0] and 0 <= c < pred_agl.shape[1]):
            continue
        v = _structure_median(pred_agl, r, c)
        if v is not None:
            xs.append(v)
            zs.append(float(z))
    n = len(xs)
    if n < min_points:
        if verbose:
            print(f"    {n} usable control points; a scale fit needs {min_points}. "
                  f"Falling back to offset only.")
        return None
    xs_a, zs_a = np.asarray(xs), np.asarray(zs)
    span = float(xs_a.max() - xs_a.min())
    if span < min_span:
        if verbose:
            print(f"    control points span only {span:.1f} m of predicted height "
                  f"(need {min_span:.0f} m). Falling back to offset only.")
        return None
    # At least two points must constrain the tall end, or the fit is a lever with no
    # support -- the failure that produced scale 0.830 on OMA_288_012.
    #
    # The threshold is calibrated, not chosen. Requiring two points in the upper HALF of
    # the span would refuse 24.6% of the k=8 draws that produced the measured -24.9% gain,
    # and 38.8% at k=5 -- it blocks a quarter to a third of the cases it is supposed to
    # allow. Measured over the same tall tiles and draws:
    #     upper 50% of span, >=2 points  ->  refuses 22.7%
    #     upper 60% of span, >=2 points  ->  refuses  6.3%
    #     upper 70% of span, >=2 points  ->  refuses  0.3%   (barely a guard)
    # 60% keeps 94% of the beneficial draws while still demanding real support at height.
    # The exponent check below is what actually catches the "compresses further" failure.
    if int((xs_a > xs_a.min() + 0.4 * span).sum()) < 2:
        if verbose:
            print(f"    fewer than 2 of {n} control points reach the upper half of their "
                  f"own height range; a fit with no support at height tilts the wrong way. "
                  f"Falling back to offset only.")
        return None

    eps = 0.05          # metres: log() of a 0.1 m building is large-negative, not invalid
    lp, lt = np.log(np.maximum(xs_a, eps)), np.log(np.maximum(zs_a, eps))
    if np.ptp(lp) < 1e-6:
        return None
    b, log_a = np.polyfit(lp, lt, 1)
    if not (0.95 <= b <= 4.0):
        if verbose:
            print(f"    rejected exponent {b:.3f}: outside 0.95-4.0. Below 1 would compress "
                  f"further, and our measured error is UNDER-call at height. "
                  f"Falling back to offset only.")
        return None
    a = float(np.exp(log_a))
    resid = zs_a - a * np.maximum(xs_a, eps) ** b
    rms = float(np.sqrt(np.mean(resid ** 2)))
    if verbose:
        print(f"    {n} control points spanning {span:.1f} m -> "
              f"elevation = {a:.3f} * AGL**{b:.3f}  (residual RMS {rms:.2f} m)")
    return a, float(b), n, rms


def fit_gcp_affine(pred_agl: np.ndarray, transform, gcps, min_points: int = 5,
                   min_span: float = 5.0, verbose: bool = True):
    """Fit elevation = scale * AGL + offset from control points, not offset alone.

    `fit_gcp_offset` declines to fit a scale on the grounds that "our heights are already
    metric". Measured 28 Aug, they are not: over 3,090 region-disjoint validation buildings
    the model sits at `ours = 0.483 * truth + 2.49 m`. Height is compressed toward the ~4 m
    median of the training distribution, so a 76 m building renders at about 26 m. An
    offset cannot correct a slope, and the PS names exactly this as its own milestone --
    "convert relative depth to absolute height using ... minimal Ground Control Points".

    The old docstring's worry was still right, though, and the thresholds here come from
    measuring it (`tools/gcp_calibration_probe.py`, per-tile fits scored only on buildings
    NOT used as control points, 200 draws each):

        control points   RMSE on tall tiles      tiles made worse
        2, spread        11.017 m   (+17.7%)     10 of 13
        3, spread         8.263 m   (-11.7%)     10 of 13
        5, random         7.915 m   (-15.5%)     11 of 13
        5, spread         7.366 m   (-21.3%)      7 of 13
        none              9.362 m        --        --

    So: five or more points, and they must span a real height range -- with two points an
    affine fit through near-identical heights is near-singular and actively harmful. Below
    those thresholds this returns None and the caller falls back to an offset.

    It is a trade, not a free win, and the caller must say so: the same measurement puts
    MAE UP from 4.933 to 5.843 m. Scaling buys the large errors on tall buildings by
    spending accuracy on the many short ones. Of the three metrics the PS names it improves
    RMSE, worsens MAE, and cannot change correlation.
    """
    inv = ~transform
    xs, zs = [], []
    half = 2                                    # 5x5 window
    for x, y, z in gcps:
        c, r = inv * (x, y)
        c, r = int(round(c)), int(round(r))
        if not (0 <= r < pred_agl.shape[0] and 0 <= c < pred_agl.shape[1]):
            continue
        # Median of a small window, not a single pixel. A control point marks a STRUCTURE,
        # and one pixel of a noisy height field is a poor estimate of it -- measured on
        # OMA_288_012, single-pixel sampling read a 38.2 m building as 1.54 m because the
        # point landed on the tile border, where outer-pixel RMSE is about double the
        # interior. The window costs nothing and removes that failure mode.
        r0, r1 = max(0, r - half), min(pred_agl.shape[0], r + half + 1)
        c0, c1 = max(0, c - half), min(pred_agl.shape[1], c + half + 1)
        patch = pred_agl[r0:r1, c0:c1]
        patch = patch[np.isfinite(patch)]
        if patch.size:
            xs.append(float(np.median(patch)))
            zs.append(float(z))
    n = len(xs)
    if n < min_points:
        if verbose:
            print(f"    {n} usable control points; a scale fit needs {min_points}. "
                  f"Falling back to offset only.")
        return None
    xs_a, zs_a = np.asarray(xs), np.asarray(zs)
    span = float(xs_a.max() - xs_a.min())
    if span < min_span:
        if verbose:
            print(f"    control points span only {span:.1f} m of predicted height "
                  f"(need {min_span:.0f} m). A scale fitted through them would be "
                  f"near-singular. Falling back to offset only.")
        return None
    # The compression is NOT affine, and this is where that bites. Measured over 3,090
    # buildings: short structures are over-called (median ratio 1.20 at 0-3 m, 1.14 at
    # 3-6 m) while tall ones are under-called (0.52 above 40 m). A least-squares line
    # through control points that are mostly SHORT is dominated by the over-called end and
    # tilts the wrong way for the tall end.
    #
    # Measured end to end on OMA_288_012 with 7 points at 3.6-30.7 m, six of them below
    # 11 m: the fit returned scale 0.830 -- compressing FURTHER -- and made the tile worse,
    # RMSE 23.62 -> 25.09 m and tall-structure RMSE 37.30 -> 39.71 m. The arithmetic was
    # right; the control points did not constrain the end that needed constraining.
    #
    # So refuse unless at least two points sit in the upper part of the range they span.
    # Two, not one, because a single tall point is a lever with no support: any error in it
    # rotates the whole fit.
    span_mid = xs_a.min() + 0.5 * span
    n_high = int((xs_a > span_mid).sum())
    if n_high < 2:
        if verbose:
            print(f"    only {n_high} of {n} control points sit in the upper half of their "
                  f"own height range. A line fitted through mostly-low points tilts the "
                  f"wrong way for tall structures (measured: scale 0.830, tile RMSE "
                  f"23.62 -> 25.09 m). Falling back to offset only.")
        return None
    scale, offset = np.polyfit(xs_a, zs_a, 1)
    if scale < 1.0:
        # Our error is under-call at height, so a corrective scale should be >= 1. Below 1
        # means the control points are describing the over-called short end, and applying
        # it would push tall structures further down.
        if verbose:
            print(f"    rejected scale {scale:.3f}: below 1.0, which would compress heights "
                  f"further. Our measured error is UNDER-call at height, so a corrective "
                  f"scale is >= 1. Falling back to offset only.")
        return None
    if not (0.2 <= scale <= 5.0):
        if verbose:
            print(f"    rejected scale {scale:.3f}: outside the plausible 0.2-5.0 range, "
                  f"which means the control points disagree rather than define a line.")
        return None
    resid = zs_a - (scale * xs_a + offset)
    rms = float(np.sqrt(np.mean(resid ** 2)))
    if verbose:
        print(f"    {n} control points spanning {span:.1f} m -> "
              f"elevation = {scale:.3f} * AGL {offset:+.2f} m  (residual RMS {rms:.2f} m)")
        print(f"    NOTE: scaling reduces RMSE on tall structures and INCREASES MAE on "
              f"short ones. Measured -21.3% RMSE / +18% MAE on tall tiles.")
    return float(scale), float(offset), n, rms


def fit_gcp_offset(pred_agl: np.ndarray, transform, gcps, verbose: bool = True):
    """Minimal ground-control anchoring: solve elevation = AGL + offset from known points.

    `gcps` is a list of (x, y, elevation) in the raster's own CRS. Only an offset is fitted,
    not a scale: our heights are already metric, so a scale term would be fitting noise, and
    with a handful of points it would fit it enthusiastically.
    """
    import rasterio
    inv = ~transform
    res, kept = [], []
    for x, y, z in gcps:
        c, r = inv * (x, y)
        c, r = int(round(c)), int(round(r))
        if 0 <= r < pred_agl.shape[0] and 0 <= c < pred_agl.shape[1]:
            v = pred_agl[r, c]
            if np.isfinite(v):
                res.append(z - float(v))
                kept.append((x, y, z))
    if not res:
        return None, 0, float("nan")
    off = float(np.median(res))                 # median, not mean: one bad point is normal
    spread = float(np.percentile(np.abs(np.array(res) - off), 68)) if len(res) > 1 else 0.0
    if verbose:
        print(f"    {len(res)}/{len(gcps)} control points inside the raster, "
              f"offset {off:+.2f} m, 68% spread +/-{spread:.2f} m")
        # A single offset can only be right if the ground is flat. On sloped terrain the
        # control points disagree by the relief itself, and the spread says so loudly --
        # measured at +/-44 m on a Sikkim hillside with 248 m of relief. Shipping that
        # silently would be a confidently wrong DSM, which is worse than none.
        if spread > 3.0:
            print(f"    WARNING: the control points disagree by +/-{spread:.1f} m. A single "
                  f"offset cannot represent terrain relief, so this scene almost certainly "
                  f"has slope. Use --dem instead; GCP anchoring suits flat scenes only.")
    return off, len(res), spread
