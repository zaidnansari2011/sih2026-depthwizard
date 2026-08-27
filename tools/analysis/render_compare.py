"""Side-by-side: Maxar RGB, our height, Open Buildings height, on one shared colour scale.

A shared scale is the whole point. Two height maps on independent auto-scales always look
like they agree.
"""
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from PIL import Image, ImageDraw

TAG = "sikkim_town_500m"
RGB = f"D:/sih2026/data/maxar/crops/{TAG}.tif"
PRED = "D:/sih2026/out/sikkim_town_run02.height.tif"
OB = f"D:/sih2026/data/open_buildings/{TAG}_obheight_2022.tif"
OUT = "D:/sih2026/out/sikkim_compare_500m.png"
N, VMAX = 760, 18.0


def turbo(x):
    """Compact perceptual ramp: dark blue -> cyan -> yellow -> red."""
    x = np.clip(x, 0, 1)
    stops = np.array([[0.19, 0.07, 0.23], [0.13, 0.57, 0.85], [0.20, 0.85, 0.55],
                      [0.95, 0.87, 0.20], [0.83, 0.20, 0.13]])
    pos = x * (len(stops) - 1)
    i = np.clip(pos.astype(int), 0, len(stops) - 2)
    f = (pos - i)[..., None]
    return (stops[i] * (1 - f) + stops[i + 1] * f)


with rasterio.open(OB) as s:
    prof = s.profile
    ob = s.read(1, out_shape=(N, N), resampling=Resampling.bilinear)
    ob = np.where(ob == s.nodata, 0.0, ob)
with rasterio.open(RGB) as s:
    rgb = np.transpose(s.read(out_shape=(3, N, N), resampling=Resampling.average), (1, 2, 0))
with rasterio.open(PRED) as p:
    with WarpedVRT(p, crs=prof["crs"], transform=prof["transform"],
                   width=prof["width"], height=prof["height"],
                   resampling=Resampling.average) as v:
        pred = v.read(1, out_shape=(N, N), resampling=Resampling.bilinear)

panels = [rgb.astype("uint8"),
          (turbo(pred / VMAX) * 255).astype("uint8"),
          (turbo(ob / VMAX) * 255).astype("uint8")]
labels = ["Maxar 0.31 m RGB  (Sikkim, 26 deg off-nadir)",
          f"DepthWizard run02  (median {np.median(pred[ob > 0.5]):.1f} m on their footprints)",
          f"Google Open Buildings 2.5D  (median {np.median(ob[ob > 0.5]):.1f} m)"]

PAD, BAR = 8, 26
W = N * 3 + PAD * 4
H = N + BAR + PAD * 2
canvas = Image.new("RGB", (W, H), (18, 18, 20))
for k, (p_img, lab) in enumerate(zip(panels, labels)):
    x = PAD + k * (N + PAD)
    canvas.paste(Image.fromarray(p_img), (x, PAD))
    ImageDraw.Draw(canvas).text((x + 4, PAD + N + 6), lab, fill=(215, 215, 220))

# Shared colour key, so the two height panels are provably on one scale.
d = ImageDraw.Draw(canvas)
kx, ky, kw = PAD + 2 * (N + PAD) + N - 210, PAD + 8, 200
for i in range(kw):
    c = tuple((turbo(np.array(i / kw)) * 255).astype(int))
    d.rectangle([kx + i, ky, kx + i + 1, ky + 9], fill=c)
d.text((kx, ky + 11), f"0 m", fill=(235, 235, 240))
d.text((kx + kw - 30, ky + 11), f"{VMAX:.0f} m", fill=(235, 235, 240))
canvas.save(OUT)
print("wrote", OUT, canvas.size)
