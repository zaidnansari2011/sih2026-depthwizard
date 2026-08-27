"""Is our AGL tracking the hillside? Correlate the low-frequency prediction with GLO-30.

AGL is terrain-removed by definition, so a correct model reads the same on a rooftop at
1200 m altitude as on one at 400 m. If the smoothed prediction instead tracks Copernicus
GLO-30 elevation, we are leaking terrain into height -- the failure hilly terrain exists
to expose, and one our flat Jacksonville/Omaha training set could never have shown us.

GLO-30 is a surface model, not a bare-earth one, so over forest it carries canopy as well
as ground. That weakens it as a terrain reference but not as a slope reference: at 30 m
posting, the elevation gradient across a Himalayan valley is terrain, not trees.
"""
import os
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from scipy import ndimage

os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".tif")

PRED = "D:/sih2026/out/sikkim_2000m_run02.height.tif"
DEM = ("/vsicurl/https://copernicus-dem-30m.s3.amazonaws.com/"
       "Copernicus_DSM_COG_10_N27_00_E088_00_DEM/"
       "Copernicus_DSM_COG_10_N27_00_E088_00_DEM.tif")
COARSE = 30.0   # match GLO-30's posting; finer is pretending

with rasterio.open(PRED) as s:
    res, prof = abs(s.transform.a), s.profile
    k = int(round(COARSE / res))
    H, W = s.height // k, s.width // k
    h = s.read(1)[: H * k, : W * k].reshape(H, k, W, k).mean((1, 3))
    tr = s.transform * s.transform.scale(k, k)
    crs = s.crs

with rasterio.open(DEM) as d:
    with WarpedVRT(d, crs=crs, transform=tr, width=W, height=H,
                   resampling=Resampling.bilinear) as v:
        z = v.read(1).astype("float32")

print(f"grid {W} x {H} at {COARSE:.0f} m")
print(f"GLO-30 elevation {z.min():.0f} .. {z.max():.0f} m  (relief {np.ptp(z):.0f} m)")
gy, gx = np.gradient(z, COARSE)
slope = np.degrees(np.arctan(np.hypot(gy, gx)))
print(f"slope median {np.median(slope):.1f} deg, p90 {np.percentile(slope, 90):.1f} deg")
print(f"our AGL  {h.min():.2f} .. {h.max():.2f} m  mean {h.mean():.2f} m")


def rep(name, a, b):
    m = np.isfinite(a) & np.isfinite(b)
    r = float(np.corrcoef(a[m], b[m])[0, 1])
    # slope of b on a, in metres of predicted AGL per unit of the driver
    k_ = float(np.polyfit(a[m], b[m], 1)[0])
    print(f"  {name:38s} r {r:+.3f}   {k_:+.4f} m AGL per unit")
    return r


print("\nlow-frequency AGL against terrain:")
for scale in (150.0, 400.0):
    lo = ndimage.gaussian_filter(h, scale / COARSE / 3.0, mode="nearest")
    rep(f"AGL@{scale:.0f}m  vs elevation (per m)", z, lo)
    rep(f"AGL@{scale:.0f}m  vs slope (per deg)", slope, lo)

print("\nreading: a strong positive elevation correlation means altitude is being")
print("read as height. A slope correlation instead means the lean of the ground is.")
