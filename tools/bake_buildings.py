"""Bake per-building elevation statistics into a scene, for the inundation tool.

Why this runs offline rather than in the browser
------------------------------------------------
The viewer needs to answer "how many buildings are under water at level L" every time a
slider moves. Doing that from a label raster in JavaScript would mean connected components
over four million cells per drag, and shipping a uint16 label band would add ~8 MB to a
standalone file that is already 15 MB.

Neither is necessary. A building's relationship to a water level depends only on its own
elevation, so the whole question collapses to a list of numbers computed once here. The
viewer then loops over a few hundred entries, which is free.

Where the footprints come from
------------------------------
Google Open Buildings 2.5D Temporal, already cached in data/open_buildings as height
rasters (not vectors) -- the `building_height` band, metres above terrain. We use it only
as a **footprint source**: which pixels are a building, and which building they belong to.
The heights we report are OUR heights, sampled inside those footprints.

That distinction matters and must survive into the UI. Open Buildings is a Sentinel-2
derived model output, not ground truth; `docs/evidence-pack.md` says so where it compares
against it, and the inundation readout has to say "footprints: Open Buildings" for the same
reason. Borrowing someone else's outlines is defensible; quietly borrowing their heights
and calling the result ours is not.

    python tools/bake_buildings.py                      # every scene that can support it
    python tools/bake_buildings.py --scene hilly_sikkim_valley
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_bounds
from rasterio.warp import Resampling, reproject
from scipy import ndimage

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
SCENES = ROOT / "viewer" / "scenes"
OB_DIR = Path("D:/sih2026/data/open_buildings")

MIN_AREA_M2 = 25.0          # same floor as depthwizard.metrics.building_instances
OB_CANDIDATES = ["sikkim_town_2000m_obheight_2022.tif",
                 "sikkim_town_500m_obheight_2022.tif"]


def load_bin(p: Path, n: int) -> np.ndarray:
    a = np.fromfile(p, dtype="<f4")
    if a.size != n:
        raise SystemExit(f"{p.name}: {a.size} floats, expected {n}")
    return a


def ob_on_grid(dst_transform, dst_crs, width: int, height: int) -> np.ndarray | None:
    """Open Buildings height resampled onto the scene grid, or None if nothing covers it."""
    west, north = dst_transform * (0, 0)
    east, south = dst_transform * (width, height)
    best = None
    for name in OB_CANDIDATES:
        p = OB_DIR / name
        if not p.exists():
            continue
        with rasterio.open(p) as src:
            b = src.bounds
            # Require full cover. A footprint source that stops half way across the scene
            # would silently report zero buildings on the uncovered half.
            if b.left <= west and b.right >= east and b.bottom <= south and b.top >= north:
                out = np.zeros((height, width), np.float32)
                reproject(
                    source=rasterio.band(src, 1), destination=out,
                    src_transform=src.transform, src_crs=src.crs,
                    dst_transform=dst_transform, dst_crs=dst_crs,
                    # Nearest, not bilinear: interpolating across a footprint edge invents
                    # half-buildings in the gap between two real ones.
                    resampling=Resampling.nearest)
                best = out
                break
    return best


def bake(scene_dir: Path) -> dict | None:
    man = json.loads((scene_dir / "manifest.json").read_text(encoding="utf-8"))
    geo = man.get("geo") or {}
    crs, bounds = geo.get("crs"), geo.get("bounds")
    datum = man.get("terrain_datum_m")

    if not crs or datum is None or "terrain" not in man.get("files", {}):
        # Flooding needs absolute elevation. The DFC2019 scenes have crs null and no
        # terrain band, so there is nothing honest to compute; say so and move on.
        print(f"  {scene_dir.name}: skipped (no CRS or no terrain band)")
        return None

    W, H = man["width"], man["height"]
    west, south, east, north = bounds
    tr = from_bounds(west, south, east, north, W, H)

    ob = ob_on_grid(tr, crs, W, H)
    if ob is None:
        print(f"  {scene_dir.name}: skipped (no Open Buildings raster covers it)")
        return None

    n = W * H
    height_m = load_bin(scene_dir / man["files"]["height"], n).reshape(H, W)
    terrain_m = load_bin(scene_dir / man["files"]["terrain"], n).reshape(H, W)
    sigma_p = man["files"].get("sigma")
    sigma_m = load_bin(scene_dir / sigma_p, n).reshape(H, W) if sigma_p else None

    # Absolute surface elevation, the only quantity a water level can be compared against.
    elev = datum + terrain_m + height_m

    px_area = abs(tr.a * tr.e)
    mask = np.isfinite(ob) & (ob > 0)
    lab, count = ndimage.label(mask, structure=np.ones((3, 3)))
    if count == 0:
        print(f"  {scene_dir.name}: no footprints found")
        return None

    counts = np.bincount(lab.ravel(), minlength=count + 1)[1:]
    keep = np.nonzero(counts * px_area >= MIN_AREA_M2)[0] + 1
    if not keep.size:
        print(f"  {scene_dir.name}: no footprint above {MIN_AREA_M2:g} m2")
        return None

    # Centroids, normalised to [0,1] on the scene grid. Without a position the viewer can
    # only ask "is this building below the water level", not "is it connected to water" --
    # so a building sitting in an enclosed hollow would be counted as flooded while the
    # ground rendered around it stayed dry. Storing where it is keeps the two consistent.
    cys, cxs = (np.array(c) for c in zip(*ndimage.center_of_mass(mask, lab, keep)))

    med_elev = ndimage.labeled_comprehension(elev, lab, keep, np.median, np.float64, np.nan)
    med_sig = (ndimage.labeled_comprehension(sigma_m, lab, keep, np.median, np.float64, np.nan)
               if sigma_m is not None else np.full(keep.size, np.nan))
    med_agl = ndimage.labeled_comprehension(height_m, lab, keep, np.median, np.float64, np.nan)
    areas = counts[keep - 1] * px_area

    ok = np.isfinite(med_elev)
    buildings = [
        # Rounded at write time: a millimetre of a metre-scale estimate is noise, and the
        # rounding keeps the JSON small enough to inline in the standalone build.
        {"e": round(float(e), 2), "s": round(float(s), 2) if np.isfinite(s) else None,
         "h": round(float(h), 2), "a": round(float(a), 1),
         "x": round(float(cx) / (W - 1), 4), "y": round(float(cy) / (H - 1), 4)}
        for e, s, h, a, cx, cy in zip(med_elev[ok], med_sig[ok], med_agl[ok], areas[ok],
                                      cxs[ok], cys[ok])
    ]

    out = {
        "source": "Google Open Buildings 2.5D Temporal (footprints only; heights are ours)",
        "min_area_m2": MIN_AREA_M2,
        "n": len(buildings),
        "elev_min_m": round(float(np.nanmin(med_elev[ok])), 2),
        "elev_max_m": round(float(np.nanmax(med_elev[ok])), 2),
        "buildings": buildings,
    }
    (scene_dir / "buildings.json").write_text(json.dumps(out), encoding="utf-8")

    man.setdefault("files", {})["buildings"] = "buildings.json"
    man["terrain_elev_min_m"] = round(float(np.nanmin(datum + terrain_m)), 2)
    man["terrain_elev_max_m"] = round(float(np.nanmax(datum + terrain_m)), 2)
    man["surface_elev_max_m"] = round(float(np.nanmax(elev)), 2)
    (scene_dir / "manifest.json").write_text(json.dumps(man, indent=2), encoding="utf-8")

    print(f"  {scene_dir.name}: {len(buildings)} buildings, "
          f"elevation {out['elev_min_m']:.0f}-{out['elev_max_m']:.0f} m, "
          f"terrain {man['terrain_elev_min_m']:.0f}-{man['terrain_elev_max_m']:.0f} m")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", default=None, help="one scene directory name")
    a = ap.parse_args()

    dirs = ([SCENES / a.scene] if a.scene
            else sorted(d for d in SCENES.iterdir() if (d / "manifest.json").exists()))
    done = 0
    for d in dirs:
        if bake(d):
            done += 1
    print(f"\nbaked {done} scene(s)")


if __name__ == "__main__":
    main()
