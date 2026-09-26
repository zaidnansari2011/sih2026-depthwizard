"""Read any uploaded image as 8-bit RGB -- the one reader the model and the viewer share.

infer.py fed the model through PIL while export_terrain.py read the texture through
rasterio, so the two could disagree about the same file. Found 25 Sep, uploading odd
inputs to a local copy of the live server:

  * a palette PNG (what TinyPNG / pngquant produce) came through rasterio as colour-table
    indices: heights fine, draped texture grey noise;
  * a grey+alpha PNG had two bands, `[..., :3]` kept both, and the export crashed;
  * a 4-band B,G,R,NIR GeoTIFF -- the band order of Cartosat MX, PlanetScope and most
    multispectral stacks -- had its first three bands read as R,G,B, swapping red and blue
    in both what the model saw and what the viewer showed.

Kept free of torch so export_terrain.py can import it without paying for the model.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


BAND_ORDERS = ("auto", "rgb", "bgr")


def band_plan(src, order: str = "auto") -> tuple[list[int], str]:
    """(zero-based red, green, blue band indices, how that was decided).

    `how` is "tagged" (the file names its colour bands), "grey" (one or two bands),
    "requested" (the caller overrode it), or "assumed" -- the file does not say, and a
    convention was applied that some files break. Only "assumed" is worth offering the
    user a re-run for; see serve_app.py and the viewer's upload panel.
    """
    from rasterio.enums import ColorInterp

    if order not in BAND_ORDERS:
        raise ValueError(f"band order must be one of {BAND_ORDERS}, not {order!r}")
    n = src.count
    if n == 1 or n == 2:                      # grey, or grey + alpha: nothing to order
        return [0, 0, 0], "grey"
    if order == "rgb":
        return [0, 1, 2], "requested"
    if order == "bgr":
        return [2, 1, 0], "requested"
    ci = list(src.colorinterp or [])
    if all(c in ci for c in (ColorInterp.red, ColorInterp.green, ColorInterp.blue)):
        return ([ci.index(ColorInterp.red), ci.index(ColorInterp.green),
                 ci.index(ColorInterp.blue)], "tagged")
    if n >= 4 and src.dtypes[0] != "uint8":
        # Multispectral with no colour tags: blue, green, red, NIR is the convention for
        # Cartosat MX, PlanetScope, Sentinel-2 stacks and 4-band Maxar products.
        return [2, 1, 0], "assumed"
    return [0, 1, 2], "assumed"               # RGB, or 8-bit RGBA without tags


def _rgb_band_indices(src) -> list[int]:
    """Zero-based indices of the red, green and blue bands of an open rasterio dataset."""
    return band_plan(src)[0]


def describe_bands(path, order: str = "auto") -> dict:
    """What read_rgb_valid(path, order) reads as red, green and blue, and why.

    {"order": "R,G,B" | "B,G,R" | "grey", "how": tagged|grey|requested|assumed|format}
    """
    path = Path(path)
    if path.suffix.lower() not in (".tif", ".tiff"):
        # PNG and JPG define their channel order; there is nothing to guess.
        return {"order": "R,G,B", "how": "format"}
    import warnings
    import rasterio
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with rasterio.open(path) as src:
            idx, how = band_plan(src, order)
    label = ("grey" if how == "grey" else
             "B,G,R" if idx == [2, 1, 0] else
             "R,G,B" if idx == [0, 1, 2] else
             "bands " + ",".join(str(i + 1) for i in idx))
    return {"order": label, "how": how}


def _stretch(a: np.ndarray) -> np.ndarray:
    """Percentile stretch of a non-8-bit image to uint8; NaN (nodata) becomes 0."""
    a = a.astype(np.float32)
    ok = np.isfinite(a)
    if not ok.any():
        return np.zeros(a.shape, np.uint8)
    # Satellite imagery routinely has a long bright tail (specular roofs, cloud edges)
    # that a min/max stretch would let flatten everything else.
    lo, hi = np.percentile(a[ok], [2, 98])
    if hi <= lo:
        lo, hi = float(a[ok].min()), float(a[ok].max()) or 1.0
    out = np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1) * 255
    return np.where(ok, out, 0).astype(np.uint8)


def read_rgb(path, band_order: str = "auto") -> np.ndarray:
    """HxWx3 uint8 RGB from a GeoTIFF, PNG or JPG."""
    return read_rgb_valid(path, band_order)[0]


# What separates a footprint border from a shadow that happens to touch the edge, measured
# 26 Sep on 419 real border-free images (DFC2019, Maxar Sikkim/Nepal crops, samples) and 8
# real footprint edges cut from raw Maxar tiles with their masks dropped:
#   shadows:  at most 0.71 % of the perimeter in contact, 0.06 % of the area
#   borders:  47-68 % of the perimeter, 45-86 % of the area
# "Black and touching the edge" alone is NOT enough: a building shadow in OMA_281_005 is
# exactly (0,0,0) and touches the edge. The thresholds sit 4x above the worst shadow; a
# 150 px corner clip of a 1024 px tile (7 % contact, 2 % area) still qualifies.
_MIN_CONTACT = 0.03          # fraction of the image perimeter the region runs along
_MIN_AREA = 0.0025           # fraction of the image it covers
# JPEG compression -- in a .jpg, or inside a Maxar visual GeoTIFF -- leaves the border's
# boundary at 1-6 rather than 0. Grow a confirmed border this far into near-black.
_FRINGE_PX, _FRINGE_TOL = 4, 8


def _edge_connected(dark: np.ndarray, near_black: np.ndarray | None = None) -> np.ndarray:
    """The footprint border within `dark`: components running along the image edge.

    `near_black` (optional) is where the confirmed border may grow by a few pixels, to
    take in the compression fringe along its boundary."""
    if not dark.any():
        return dark
    from scipy.ndimage import binary_dilation, label
    lab, n = label(dark)
    H, W = lab.shape
    ring = np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]])
    contact = np.bincount(ring, minlength=n + 1)
    area = np.bincount(lab.ravel(), minlength=n + 1)
    keep = (contact >= _MIN_CONTACT * 2 * (H + W)) & (area >= _MIN_AREA * H * W)
    keep[0] = False
    border = keep[lab]
    if near_black is not None and border.any():
        border |= binary_dilation(border, iterations=_FRINGE_PX) & near_black
    return border


def read_rgb_valid(path, band_order: str = "auto") -> tuple[np.ndarray, np.ndarray]:
    """(HxWx3 uint8 RGB, HxW bool valid) from a GeoTIFF, PNG or JPG.

    A pixel is invalid when the file says so -- declared nodata, a zero alpha, an internal
    mask, NaN -- or when it belongs to a pure-black region touching the image edge. The
    second case is the footprint border of a clipped satellite scene, which often carries
    no nodata tag at all. `band_order` ("auto", "rgb", "bgr") overrides which GeoTIFF
    bands are read as red, green and blue; PNG and JPG define their own order. Without this mask the model estimated heights for that border
    and the viewer drew it as a flat black shelf around the scene.

    "Pure black" is judged on the RAW values, never after the percentile stretch: the
    stretch clips the darkest 2% of real pixels (deep shadow) to 0, and those must stay
    valid. Only regions connected to the edge count, so a black roof or a shadow inside
    the scene is never masked.
    """
    path = Path(path)
    if path.suffix.lower() in (".tif", ".tiff"):
        import warnings
        import rasterio
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with rasterio.open(path) as src:
                idx = band_plan(src, band_order)[0]
                a = src.read([i + 1 for i in idx])            # 3 x H x W
                nodata = src.nodata
                # Declared nodata, alpha bands and internal masks, as GDAL resolves them.
                valid = src.dataset_mask() > 0
                if (src.count == 4 and src.dtypes[0] == "uint8" and idx == [0, 1, 2]
                        and not any(c.name == "alpha" for c in src.colorinterp)):
                    # Untagged 8-bit 4-band is read as RGBA (see _rgb_band_indices), so
                    # its fourth band is the alpha GDAL did not know to apply.
                    valid &= src.read(4) > 0
        a = np.transpose(a, (1, 2, 0))
        if a.dtype.kind == "f":
            valid &= np.isfinite(a).all(-1)
        # Raw zero in every band. For 8-bit this is pure black; for 16-bit or float it
        # is a value real sensor data does not produce across all three bands at once.
        valid &= ~_edge_connected((a == 0).all(-1) & valid,
                                  (a.max(-1) <= _FRINGE_TOL) if a.dtype == np.uint8 else None)
        if a.dtype == np.uint8:
            return np.ascontiguousarray(a), valid
        a = a.astype(np.float32)
        if nodata is not None:
            # A finite nodata value (-9999, 0 on an unsigned border) would otherwise set
            # the low percentile and wash the whole image out.
            a[a == nodata] = np.nan
        a[~valid] = np.nan              # keep masked pixels out of the stretch percentiles
        return _stretch(a), valid

    from PIL import Image, ImageOps
    im = ImageOps.exif_transpose(Image.open(path))
    if im.mode in ("I;16", "I;16B", "I;16L", "I", "F"):
        # 16-bit / float greyscale: convert("RGB") would clip everything above 255 to white.
        raw = np.array(im)
        valid = np.isfinite(raw) if raw.dtype.kind == "f" else np.ones(raw.shape, bool)
        valid &= ~_edge_connected((raw == 0) & valid)
        g = _stretch(np.where(valid, raw, np.nan))
        return np.stack([g] * 3, -1), valid
    if im.mode in ("RGBA", "LA", "PA") or "transparency" in im.info:
        valid = np.array(im.convert("RGBA"))[..., 3] > 0
    else:
        valid = np.ones((im.height, im.width), bool)
    rgb = np.array(im.convert("RGB"))
    # PNG is lossless, so a border is exactly 0. JPEG can lift a flat black block to 1-3;
    # the contact and area thresholds, not this tolerance, are what keep shadows valid.
    core = 3 if path.suffix.lower() in (".jpg", ".jpeg") else 0
    mx = rgb.max(-1)
    valid &= ~_edge_connected((mx <= core) & valid, mx <= _FRINGE_TOL)
    return rgb, valid
