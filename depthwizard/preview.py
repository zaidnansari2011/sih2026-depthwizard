"""A colour quick look of the height and uncertainty rasters, for people without GIS software.

The downloads are float32 GeoTIFFs in metres, which is right for QGIS and wrong for every
ordinary image viewer. Windows Photos reads a float image as 0 = black .. 1 = white, so a
height raster -- everything above a metre -- opens as a white sheet with grey roads. Seen
25 Sep on a real download. The rasters were correct; the only way to check them looked
broken. This picture is the check anyone can open.

Pillow and numpy only: the public server carries no matplotlib.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

# Control points of matplotlib's viridis and magma: perceptually even, and they survive
# greyscale printing.
VIRIDIS = [(0, (68, 1, 84)), (.25, (59, 82, 139)), (.5, (33, 145, 140)),
           (.75, (94, 201, 98)), (1, (253, 231, 37))]
MAGMA = [(0, (0, 0, 4)), (.25, (81, 18, 124)), (.5, (183, 55, 121)),
         (.75, (252, 137, 97)), (1, (252, 253, 191))]

PANEL = 520          # longest side of each map, px
PAD = 18
BAR_W = 16


def _lut(stops) -> np.ndarray:
    xs = np.linspace(0, 1, 256)
    return np.stack([np.interp(xs, [p for p, _ in stops], [c[i] for _, c in stops])
                     for i in range(3)], -1).astype(np.uint8)


def _font(size: int):
    from PIL import ImageFont
    try:
        return ImageFont.load_default(size=size)      # Pillow >= 10.1
    except TypeError:
        return ImageFont.load_default()


def _read(path: Path) -> np.ndarray:
    import rasterio
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with rasterio.open(path) as src:
            return src.read(1).astype(np.float32)


def _panel(a: np.ndarray, stops, title: str, unit: str):
    """One colourised map with its own colour bar and 0..p99.5 range."""
    from PIL import Image, ImageDraw
    ok = np.isfinite(a)
    hi = float(np.percentile(a[ok], 99.5)) if ok.any() else 1.0
    lo = min(0.0, float(np.nanmin(a))) if ok.any() else 0.0
    hi = hi if hi > lo else lo + 1.0
    idx = np.clip((np.nan_to_num(a, nan=lo) - lo) / (hi - lo), 0, 1) * 255
    rgb = _lut(stops)[idx.astype(np.uint8)]
    rgb[~ok] = (200, 200, 200)
    im = Image.fromarray(rgb)
    s = PANEL / max(im.size)
    im = im.resize((max(1, round(im.width * s)), max(1, round(im.height * s))), Image.BILINEAR)

    W = im.width + BAR_W + 70
    H = im.height + 44
    out = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(out)
    d.text((0, 4), title, fill=(26, 32, 39), font=_font(17))
    out.paste(im, (0, 34))
    # Colour bar with its end values and midpoint, in the raster's own units.
    bar = Image.fromarray(np.repeat(_lut(stops)[::-1][:, None, :], BAR_W, 1)).resize(
        (BAR_W, im.height))
    bx = im.width + 10
    out.paste(bar, (bx, 34))
    f = _font(13)
    for frac in (0, .5, 1):
        v = hi - (hi - lo) * frac
        y = 34 + round(frac * (im.height - 1))
        d.line([(bx + BAR_W, y), (bx + BAR_W + 4, y)], fill=(90, 100, 110))
        d.text((bx + BAR_W + 7, y - 8), f"{v:.1f} {unit}", fill=(60, 70, 80), font=f)
    return out


def make_preview(height_tif, sigma_tif, out_png, *, heading: str) -> Path:
    """Write the quick look: height map, uncertainty map (if any), a heading and a note."""
    from PIL import Image, ImageDraw
    panels = [_panel(_read(Path(height_tif)), VIRIDIS, "Height above ground", "m")]
    if sigma_tif and Path(sigma_tif).exists():
        panels.append(_panel(_read(Path(sigma_tif)), MAGMA, "Uncertainty (1 sigma)", "m"))

    W = PAD * 2 + sum(p.width for p in panels) + PAD * (len(panels) - 1)
    H = PAD * 2 + 34 + max(p.height for p in panels) + 40
    out = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(out)
    d.text((PAD, PAD), heading, fill=(26, 32, 39), font=_font(20))
    x = PAD
    for p in panels:
        out.paste(p, (x, PAD + 34))
        x += p.width + PAD
    d.text((PAD, H - PAD - 16),
           "Quick look only. The .tif downloads hold the real values in metres; open them in "
           "QGIS or any GIS tool.", fill=(91, 102, 114), font=_font(13))
    out_png = Path(out_png)
    out.save(out_png, optimize=True)
    return out_png
