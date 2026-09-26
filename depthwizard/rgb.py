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


def _rgb_band_indices(src) -> list[int]:
    """Zero-based indices of the red, green and blue bands of an open rasterio dataset."""
    from rasterio.enums import ColorInterp

    ci = list(src.colorinterp or [])
    if all(c in ci for c in (ColorInterp.red, ColorInterp.green, ColorInterp.blue)):
        return [ci.index(ColorInterp.red), ci.index(ColorInterp.green), ci.index(ColorInterp.blue)]
    n = src.count
    if n == 1 or n == 2:                      # grey, or grey + alpha
        return [0, 0, 0]
    if n >= 4 and src.dtypes[0] != "uint8":
        # Multispectral with no colour tags: blue, green, red, NIR is the convention for
        # Cartosat MX, PlanetScope, Sentinel-2 stacks and 4-band Maxar products.
        return [2, 1, 0]
    return [0, 1, 2]                          # RGB, or 8-bit RGBA without tags


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


def read_rgb(path) -> np.ndarray:
    """HxWx3 uint8 RGB from a GeoTIFF, PNG or JPG."""
    path = Path(path)
    if path.suffix.lower() in (".tif", ".tiff"):
        import warnings
        import rasterio
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with rasterio.open(path) as src:
                idx = _rgb_band_indices(src)
                a = src.read([i + 1 for i in idx])            # 3 x H x W
                nodata = src.nodata
        a = np.transpose(a, (1, 2, 0))
        if a.dtype == np.uint8:
            return np.ascontiguousarray(a)
        a = a.astype(np.float32)
        if nodata is not None:
            # A finite nodata value (-9999, 0 on an unsigned border) would otherwise set
            # the low percentile and wash the whole image out.
            a[a == nodata] = np.nan
        return _stretch(a)

    from PIL import Image, ImageOps
    im = ImageOps.exif_transpose(Image.open(path))
    if im.mode in ("I;16", "I;16B", "I;16L", "I", "F"):
        # 16-bit / float greyscale: convert("RGB") would clip everything above 255 to white.
        g = _stretch(np.array(im))
        return np.stack([g] * 3, -1)
    return np.array(im.convert("RGB"))
