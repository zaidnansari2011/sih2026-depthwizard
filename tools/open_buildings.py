"""Google Open Buildings 2.5D Temporal: independent building heights over India.

Why this exists. Bhoonidhi has no sub-metre optical and its only height product is
CartoDEM at 30 m with 8 m LE90 -- worse than our ground RMSE, and absolute elevation
rather than AGL, so it cannot referee buildings at all. This dataset can. Its
`building_height` band is defined as "building height relative to the terrain in range
[0m, 100m]", which is the same quantity DepthWizard predicts, on the same UTM grid as
the Maxar imagery, over South Asia including India.

What it is NOT. These heights are themselves a model output, inferred from Sentinel-2 at
10 m and published at 4 m effective resolution. It is an independent second opinion, not
ground truth. Agreement over many buildings is evidence our absolute scale transfers to
India; a per-building disagreement proves nothing about which of the two is wrong. Say it
that way, because a jury that knows the dataset will know it is Sentinel-2 derived.

Storage layout, worked out from the bucket rather than the docs, which do not give it:

    v1/manifests/<s2token>_EPSG_<code>_<YYYY>_06_30.json
    v1/geotiffs/<s2token+suffix>_<YYYY>_06_30/tile_<id>.tif

Each manifest carries the affine transform and dimensions of every tile it owns, so the
tile covering a footprint can be found without downloading anything. Tiles are 25000 x
25000 at 0.5 m (12.5 km square), tiled COGs with overviews, so a footprint is a windowed
range read of a few MB rather than the 7 GB the full tile would be.

    python tools/open_buildings.py years --epsg 32645
    python tools/open_buildings.py fetch --ref data/maxar/<event>/<tile>-visual.tif --year 2022

Pick the year from the imagery date, not the event name: the Maxar India-Floods-Oct-2023
release contains acquisitions from March 2022.
"""
from __future__ import annotations

import argparse
import json
import os
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BUCKET = "open-buildings-temporal-data"
API = f"https://storage.googleapis.com/storage/v1/b/{BUCKET}/o"
MEDIA = f"https://storage.googleapis.com/{BUCKET}/"
NODATA = -99.0
# 1-based source bands: fractional_count, building_height, building_presence.
# We keep height and presence. Presence is their own model confidence, and it gives a
# tighter footprint than thresholding height does -- worth having when the alternative is
# scoring ourselves inside a blob that spans the gap between two houses.
HEIGHT_BAND = 2
PRESENCE_BAND = 3


def list_manifests() -> list[str]:
    names, tok = [], None
    while True:
        u = f"{API}?prefix=v1/manifests/&maxResults=1000" + (f"&pageToken={tok}" if tok else "")
        d = json.load(urllib.request.urlopen(u, timeout=90))
        names += [i["name"] for i in d.get("items", []) if i["name"].endswith(".json")]
        tok = d.get("nextPageToken")
        if not tok:
            return names


def tiles_for(epsg: int, year: int):
    """Every (url, bounds) the given zone and year owns. Several S2 cells can share a zone."""
    tag = f"_EPSG_{epsg}_{year}_"
    out = []
    for name in list_manifests():
        if tag not in name:
            continue
        m = json.load(urllib.request.urlopen(MEDIA + name, timeout=90))
        pref = m["uriPrefix"].replace(f"gs://{BUCKET}/", "")
        for ts in m.get("tilesets", []):
            for src in ts.get("sources", []):
                a, dim = src["affineTransform"], src["dimensions"]
                x0, y0 = a["translateX"], a["translateY"]
                x1 = x0 + a["scaleX"] * dim["width"]
                y1 = y0 + a["scaleY"] * dim["height"]
                out.append({
                    "url": MEDIA + pref + src["uris"][0],
                    "bounds": (x0, min(y0, y1), x1, max(y0, y1)),
                    "res": abs(a["scaleX"]),
                    "manifest": name,
                })
    return out


def cmd_years(args):
    tag = f"_EPSG_{args.epsg}_"
    mans = [n for n in list_manifests() if tag in n]
    yrs = sorted({n.split("_")[-3] for n in mans})
    cells = sorted({Path(n).name.split("_")[0] for n in mans})
    print(f"EPSG:{args.epsg}  S2 cells {cells}  years {yrs}")


def cmd_fetch(args):
    import numpy as np
    import rasterio
    from rasterio.windows import from_bounds
    os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
    os.environ.setdefault("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".tif")

    ref = Path(args.ref)
    with rasterio.open(ref) as r:
        epsg, rb = r.crs.to_epsg(), tuple(r.bounds)
    print(f"reference {ref.name}  EPSG:{epsg}  bounds {[round(v) for v in rb]}")

    cand = tiles_for(epsg, args.year)
    hit = [t for t in cand if not (t["bounds"][2] <= rb[0] or t["bounds"][0] >= rb[2]
                                   or t["bounds"][3] <= rb[1] or t["bounds"][1] >= rb[3])]
    if not hit:
        raise SystemExit(f"no Open Buildings tile covers that footprint in {args.year}. "
                         f"{len(cand)} tiles exist for EPSG:{epsg}.")
    # One S2 cell can duplicate another's ground; keep whichever covers more of us.
    by_cell: dict[str, list] = {}
    for t in hit:
        by_cell.setdefault(Path(t["manifest"]).name.split("_")[0], []).append(t)
    cell = max(by_cell, key=lambda c: len(by_cell[c]))
    hit = by_cell[cell]
    res = hit[0]["res"]
    print(f"{len(hit)} tile(s) from S2 cell {cell} at {res} m")

    # Output grid: the OB grid itself, snapped outward to cover the reference.
    ox, oy = hit[0]["bounds"][0], hit[0]["bounds"][3]
    c0 = int(np.floor((rb[0] - ox) / res)); c1 = int(np.ceil((rb[2] - ox) / res))
    r0 = int(np.floor((oy - rb[3]) / res)); r1 = int(np.ceil((oy - rb[1]) / res))
    W, H = c1 - c0, r1 - r0
    left, top = ox + c0 * res, oy - r0 * res
    transform = rasterio.transform.from_origin(left, top, res, res)
    out = np.full((2, H, W), NODATA, dtype="float32")
    print(f"output grid {W} x {H} at {res} m  ({W*res/1000:.1f} x {H*res/1000:.1f} km)")

    for t in hit:
        with rasterio.open("/vsicurl/" + t["url"]) as s:
            ix = (max(rb[0], t["bounds"][0]), max(rb[1], t["bounds"][1]),
                  min(rb[2], t["bounds"][2]), min(rb[3], t["bounds"][3]))
            w = from_bounds(*ix, transform=s.transform).round_offsets().round_lengths()
            if w.width < 1 or w.height < 1:
                continue
            a = s.read([HEIGHT_BAND, PRESENCE_BAND], window=w)
            wt = s.window_transform(w)
            dc, dr = int(round((wt.c - left) / res)), int(round((top - wt.f) / res))
            hh = min(a.shape[1], H - dr)
            ww = min(a.shape[2], W - dc)
            if hh < 1 or ww < 1:
                continue
            dst = out[:, dr:dr + hh, dc:dc + ww]
            src = a[:, :hh, :ww]
            np.copyto(dst, src, where=(src != NODATA) & (dst == NODATA))
            print(f"  {Path(t['url']).name}  read {a.shape} -> paste at ({dr}, {dc})")

    outp = Path(args.out) if args.out else (
        ROOT / "data" / "open_buildings" / f"{ref.stem}_obheight_{args.year}.tif")
    outp.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(outp, "w", driver="GTiff", height=H, width=W, count=2,
                       dtype="float32", crs=f"EPSG:{epsg}", transform=transform,
                       nodata=NODATA, compress="deflate", tiled=True) as d:
        d.write(out)
        d.set_band_description(1, "building_height_m_agl")
        d.set_band_description(2, "building_presence")
        d.update_tags(source="Google Open Buildings 2.5D Temporal v1",
                      band1="building_height (AGL, metres)",
                      band2="building_presence (uncalibrated confidence)",
                      year=str(args.year),
                      note="model output from Sentinel-2, 4 m effective resolution; "
                           "an independent second opinion, not ground truth")

    hgt, pres = out[0], out[1]
    v = hgt[hgt > 0]
    print(f"\nwrote {outp}")
    print(f"  building pixels {v.size:,} ({100.0*v.size/hgt.size:.2f}% of footprint)")
    if v.size:
        print(f"  height median {np.median(v):.1f} m  p95 {np.percentile(v, 95):.1f} m  "
              f"max {v.max():.1f} m")
    ok = pres != NODATA
    if ok.any():
        for thr in (0.5, 0.7):
            print(f"  presence > {thr}: {100.0*(pres[ok] > thr).mean():.2f}% of pixels")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    y = sub.add_parser("years", help="which years exist for a UTM zone")
    y.add_argument("--epsg", type=int, required=True)
    y.set_defaults(func=cmd_years)

    f = sub.add_parser("fetch", help="pull heights co-registered to a reference raster")
    f.add_argument("--ref", required=True, help="the image we will run the model on")
    f.add_argument("--year", type=int, required=True, help="match the imagery date")
    f.add_argument("--out")
    f.set_defaults(func=cmd_fetch)

    a = ap.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
