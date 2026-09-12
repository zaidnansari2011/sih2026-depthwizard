"""Turn a predicted height map into a bundle the browser viewer can load.

    python tools/export_terrain.py --height out/JAX_004.agl.tif \
                                   --texture out/JAX_004.rgb.tif \
                                   --sigma  out/JAX_004.sigma.tif \
                                   --out viewer/scenes/JAX_004

Writes:

    height.bin     float32, row-major, HEIGHT*WIDTH, metres AGL
    sigma.bin      float32, same shape, metres (optional)
    texture.jpg    the orthoimage draped over the surface
    manifest.json  everything needed to turn array indices back into metres

Why raw float32 rather than a PNG
---------------------------------
The obvious move is a 16-bit PNG, and it is a trap: browsers decode images through a
canvas that is 8 bits per channel, so a 16-bit PNG silently loses precision on load.
The usual workaround packs height across R/G/B (the Mapbox terrain-RGB trick), which
works but is lossy at the edges and awkward to debug. Our scenes are single-image
scale -- a 1024x1024 map is 4 MB raw -- so we just ship the floats and read them with
fetch(). Exact metres in the browser is what makes the measurement tool (differentiator
6.3) trustworthy rather than decorative.

The manifest carries the georeferencing so a measurement in the viewer is a measurement
on the ground, not in arbitrary units.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np


def _load_raster(path: Path):
    """Return (array, meta). Supports GeoTIFF via rasterio, or .npy for quick tests."""
    if path.suffix.lower() == ".npy":
        return np.load(path), {}
    try:
        import rasterio
    except ImportError:
        raise SystemExit("rasterio needed for GeoTIFF input:  pip install -r requirements.txt")
    with rasterio.open(path) as src:
        arr = src.read()
        arr = arr[0] if arr.shape[0] == 1 else np.transpose(arr, (1, 2, 0))
        meta = {
            "crs": str(src.crs) if src.crs else None,
            "transform": list(src.transform)[:6] if src.transform else None,
            "bounds": list(src.bounds) if src.transform else None,
        }
        # Ground sample distance in metres, taken from the affine transform. If the CRS
        # is geographic (degrees) this is NOT metres -- flag it rather than silently
        # reporting nonsense, because every measurement downstream depends on it.
        if src.transform:
            meta["px_size_x"] = abs(src.transform.a)
            meta["px_size_y"] = abs(src.transform.e)
            if src.crs is None:
                # No CRS at all. rasterio still hands back an identity transform, so the
                # old test -- "not geographic, therefore metres" -- called a plain PNG
                # 1 m/px and the viewer reported a 700 px image as 700 x 700 metres.
                # An unreferenced raster is measured in pixels; say so.
                meta["px_units"] = "pixels"
            elif src.crs.is_geographic:
                meta["px_units"] = "degrees"
            else:
                meta["px_units"] = "metres"

        # Enough for the viewer to say where on Earth it is. An EPSG code alone is not an
        # answer to that question for anyone who does not keep the register in their head,
        # and a geospatial agency reads a 3D view with no CRS, no datum and no north as a
        # toy. The corner longitudes and latitudes let the viewer report geographic
        # coordinates without shipping a projection library: measured over the 2 km Sikkim
        # tile, bilinear interpolation between these four corners sits within 4 mm of the
        # exact inverse projection, which is 244x finer than one pixel.
        if src.crs is not None:
            meta.update(_geodetic_identity(src))
    return arr, meta


def _geodetic_identity(src) -> dict:
    """Human-readable CRS, datum, and the tile's corners in WGS 84 lon/lat."""
    from rasterio.crs import CRS
    from rasterio.warp import transform as warp_transform

    out: dict = {}
    try:
        wkt = src.crs.wkt or ""
        # PROJCS["WGS 84 / UTM zone 45N", ... DATUM["WGS_1984", ...
        for key, field in (("PROJCS[\"", "crs_name"), ("GEOGCS[\"", "geographic_crs")):
            i = wkt.find(key)
            if i >= 0:
                j = wkt.find('"', i + len(key))
                out[field] = wkt[i + len(key):j]
        i = wkt.find('DATUM["')
        if i >= 0:
            out["datum"] = wkt[i + 7:wkt.find('"', i + 7)].replace("_", " ")
        out["units"] = src.crs.linear_units or None
    except (AttributeError, ValueError):
        pass

    try:
        w, h = src.width, src.height
        cols, rows = [0, w, 0, w], [0, 0, h, h]          # TL, TR, BL, BR in pixel order
        xs, ys = zip(*(src.transform * (c, r) for c, r in zip(cols, rows)))
        lon, lat = warp_transform(src.crs, CRS.from_epsg(4326), list(xs), list(ys))
        out["corners_lonlat"] = [[round(a, 8), round(b, 8)] for a, b in zip(lon, lat)]
        # Eastings and northings at the same four corners. The viewer reads position by
        # interpolating between corners rather than by applying `transform`, because the
        # standalone build decimates a 2048 px tile to 512 and rewrites width/height while
        # the transform still describes the original raster -- so the transform and the
        # grid disagree there. Corner values are in normalised tile coordinates, which
        # survive that, and for an affine transform the interpolation is exact rather
        # than approximate.
        out["corners_en"] = [[round(a, 4), round(b, 4)] for a, b in zip(xs, ys)]
        out["corners_order"] = "TL, TR, BL, BR in pixel space"
    except Exception as e:                                # projection can legitimately fail
        out["corners_lonlat_error"] = f"{type(e).__name__}: {e}"
    return out


def _to_uint8_rgb(tex: np.ndarray) -> np.ndarray:
    if tex.ndim == 2:
        tex = np.stack([tex] * 3, -1)
    tex = tex[..., :3]
    if tex.dtype == np.uint8:
        return tex
    # Percentile stretch: satellite imagery routinely has a long bright tail (specular
    # roofs, cloud edges) that a naive min/max stretch would let flatten everything else.
    lo, hi = np.percentile(tex[np.isfinite(tex)], [2, 98])
    if hi <= lo:
        lo, hi = float(np.nanmin(tex)), float(np.nanmax(tex)) or 1.0
    return (np.clip((tex - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--height", required=True, help="height map: .tif or .npy, metres AGL")
    ap.add_argument("--texture", help="orthoimage to drape: .tif or .npy")
    ap.add_argument("--sigma", help="per-pixel uncertainty, metres (differentiator 6.1)")
    ap.add_argument("--truth", help="LiDAR reference height on the same grid, metres AGL. "
                                    "Ships inside the scene so the viewer can drag-compare "
                                    "and difference the two surfaces.")
    ap.add_argument("--cls", help="semantic class raster, for building/ground error splits")
    ap.add_argument("--terrain-base", help="bare-earth elevation raster on the same grid. "
                                           "Our model predicts height ABOVE GROUND, so a "
                                           "mountain scene renders flat without this. The "
                                           "viewer stacks our heights on top of it.")
    ap.add_argument("--terrain-source", default="",
                    help="provenance string for the terrain base, shown in the viewer")
    ap.add_argument("--terrain", help="landscape label for the scene picker: urban, sparse, "
                                      "hilly, forested, mixed. ISRO names the first four.")
    ap.add_argument("--place", help="human place name, e.g. Omaha or Sikkim")
    ap.add_argument("--model", default="",
                    help="which model produced this height map, e.g. 'run02 + TTA'. Recorded "
                         "in the manifest and shown in the viewer. A scene that cannot say "
                         "where its heights came from is not evidence of anything.")
    ap.add_argument("--no-index", action="store_true",
                    help="do not list this scene in viewer/scenes/index.json. "
                         "For uploads: that index is served to every visitor.")
    ap.add_argument("--default-scene", action="store_true",
                    help="open the viewer on this scene. The picker's ORDER still mirrors "
                         "the problem statement, but the landing scene should be a "
                         "representative one rather than whichever landscape they name "
                         "first. Setting this clears the flag on every other scene.")
    ap.add_argument("--out", required=True, help="output scene directory")
    ap.add_argument("--name", help="display name (default: height filename stem)")
    ap.add_argument("--max-size", type=int, default=2048,
                    help="downsample if larger than this on the long edge")
    ap.add_argument("--vertical-exaggeration", type=float, default=1.0,
                    help="viewer default only; the metres in height.bin stay true")
    ap.add_argument("--gsd", type=float, default=None,
                    help="ground sample distance, metres per pixel. REQUIRED for DFC2019: "
                         "those tiles were rewritten by tifffile and carry no CRS or "
                         "transform, so there is nothing to read it from. US3D/DFC2019 "
                         "Track 1 is 0.3 m/px -- pass --gsd 0.3.")
    args = ap.parse_args()

    hpath = Path(args.height)
    height, hmeta = _load_raster(hpath)
    height = np.asarray(height, np.float32)
    if height.ndim != 2:
        raise SystemExit(f"height must be 2-D, got shape {height.shape}")

    # Void pixels would blow the range and punch holes in the mesh. Replace them with
    # the local median so the surface stays continuous, and record how many we touched
    # -- an honest number here is worth more than a clean-looking render.
    bad = ~np.isfinite(height)
    n_bad = int(bad.sum())
    if n_bad:
        fill = float(np.median(height[~bad])) if (~bad).any() else 0.0
        height = np.where(bad, fill, height)

    # Per-building scoring needs the native grid. Connected components cannot be recovered
    # from a thinned array -- decimating merges neighbouring roofs and splits long ones --
    # so keep the full-resolution surface even when the shipped mesh is downsampled.
    height_full = height
    native_gsd = float(args.gsd) if args.gsd else (
        hmeta.get("px_size_x") if hmeta.get("px_units") == "metres" else None)

    H, W = height.shape
    scale = 1.0
    if max(H, W) > args.max_size:
        scale = args.max_size / max(H, W)
        nh, nw = max(1, int(H * scale)), max(1, int(W * scale))
        ys = (np.linspace(0, H - 1, nh)).astype(np.int32)
        xs = (np.linspace(0, W - 1, nw)).astype(np.int32)
        height = height[np.ix_(ys, xs)]
        H, W = height.shape

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    height.astype("<f4").tofile(out / "height.bin")

    manifest = {
        "name": args.name or hpath.stem,
        "width": W,
        "height": H,
        "height_min_m": float(height.min()),
        "height_max_m": float(height.max()),
        "void_pixels_filled": n_bad,
        "downsampled_by": round(1.0 / scale, 3) if scale != 1.0 else 1.0,
        "vertical_exaggeration": args.vertical_exaggeration,
        "files": {"height": "height.bin"},
        "geo": hmeta,
    }
    if args.model:
        manifest["model"] = args.model

    # Ground sample distance, corrected for any downsampling we just did. Everything the
    # measurement tool reports is derived from this one number, so it is worth being
    # explicit that a missing transform means "pixels", not "metres".
    gsd = hmeta.get("px_size_x")
    if args.gsd:
        # An explicit GSD wins. The DFC2019 tiles have no transform to read, so this is
        # the normal path for them rather than an override.
        manifest["gsd_m"] = float(args.gsd) / scale
        manifest["gsd_source"] = "--gsd flag"
    elif gsd and hmeta.get("px_units") == "metres":
        manifest["gsd_m"] = float(gsd) / scale
        manifest["gsd_source"] = "GeoTIFF transform"
    else:
        manifest["gsd_m"] = None
        manifest["gsd_note"] = (
            "no metric transform on the input and no --gsd given; the viewer will report "
            "distances in pixels. DFC2019 tiles always land here -- pass --gsd 0.3."
        )
    if manifest["gsd_m"]:
        manifest["extent_m"] = [W * manifest["gsd_m"], H * manifest["gsd_m"]]

    if args.sigma:
        sig, _ = _load_raster(Path(args.sigma))
        sig = np.asarray(sig, np.float32)
        if sig.shape != (H, W):
            ys = np.linspace(0, sig.shape[0] - 1, H).astype(np.int32)
            xs = np.linspace(0, sig.shape[1] - 1, W).astype(np.int32)
            sig = sig[np.ix_(ys, xs)]
        sig = np.where(np.isfinite(sig), sig, 0.0)
        sig.astype("<f4").tofile(out / "sigma.bin")
        manifest["files"]["sigma"] = "sigma.bin"
        manifest["sigma_min_m"] = float(sig.min())
        manifest["sigma_max_m"] = float(sig.max())
        manifest["sigma_mean_m"] = float(sig.mean())

    if args.terrain:
        manifest["terrain"] = args.terrain
    if args.place:
        manifest["place"] = args.place

    if args.terrain_base:
        # Bare-earth elevation to stand our heights on. Without this a Himalayan scene is a
        # flat plane with 18 m buildings, because AGL is exactly the quantity that has the
        # mountain removed -- and "stability across hilly terrain" is one of the four
        # landscapes ISRO names, so a flat hill scene answers nothing.
        ter, _ = _load_raster(Path(args.terrain_base))
        ter = np.asarray(ter, np.float32)
        if ter.shape != (H, W):
            ys = np.linspace(0, ter.shape[0] - 1, H).astype(np.int32)
            xs = np.linspace(0, ter.shape[1] - 1, W).astype(np.int32)
            ter = ter[np.ix_(ys, xs)]
        ter = np.where(np.isfinite(ter), ter, np.nanmedian(ter))

        # Stored relative to the scene's lowest point, with the datum kept in the manifest.
        # Absolute metres above sea level would put the mesh 1.4 km from the origin and
        # every camera default -- all of which are derived from scene extent -- would aim
        # at empty sky.
        datum = float(ter.min())
        (ter - datum).astype("<f4").tofile(out / "terrain.bin")
        manifest["files"]["terrain"] = "terrain.bin"
        manifest["terrain_datum_m"] = datum
        manifest["terrain_relief_m"] = float(ter.max() - datum)
        manifest["terrain_source"] = args.terrain_source or "unspecified"
        gy, gx = np.gradient(ter, manifest.get("gsd_m") or 1.0)
        manifest["terrain_mean_slope_deg"] = float(
            np.degrees(np.arctan(np.hypot(gx, gy))).mean())

    if args.truth:
        # LiDAR reference on the same grid, so the viewer can put our surface and the real
        # one side by side and subtract them. Shipping truth INSIDE the scene rather than as
        # a separate scene is deliberate: the drag-to-compare divider and the error map both
        # need the surfaces registered pixel-for-pixel, and two independently exported
        # scenes give no guarantee of that.
        tru, _ = _load_raster(Path(args.truth))
        tru = np.asarray(tru, np.float32)
        tru_full = tru
        if tru_full.shape != height_full.shape:
            fy = np.linspace(0, tru_full.shape[0] - 1, height_full.shape[0]).astype(np.int32)
            fx = np.linspace(0, tru_full.shape[1] - 1, height_full.shape[1]).astype(np.int32)
            tru_full = tru_full[np.ix_(fy, fx)]
        if tru.shape != (H, W):
            ys = np.linspace(0, tru.shape[0] - 1, H).astype(np.int32)
            xs = np.linspace(0, tru.shape[1] - 1, W).astype(np.int32)
            tru = tru[np.ix_(ys, xs)]
        # Voids are real in DFC2019 AGL and they are NOT zero height. Filling keeps the
        # truth mesh watertight, but the error map must grey them out rather than report a
        # fabricated error, so the validity mask ships alongside.
        tvalid = np.isfinite(tru)
        n_tbad = int((~tvalid).sum())
        if n_tbad:
            fill_t = float(np.median(tru[tvalid])) if tvalid.any() else 0.0
            tru = np.where(tvalid, tru, fill_t)
        tru.astype("<f4").tofile(out / "truth.bin")
        manifest["files"]["truth"] = "truth.bin"
        manifest["truth_min_m"] = float(tru.min())
        manifest["truth_max_m"] = float(tru.max())
        manifest["truth_void_pixels"] = n_tbad
        if n_tbad:
            tvalid.astype(np.uint8).tofile(out / "truth_valid.bin")
            manifest["files"]["truth_valid"] = "truth_valid.bin"

        # Error stats over valid truth only. These are per-PIXEL and therefore NOT the
        # numbers we quote as headline: per-pixel building error is dominated by roof edges
        # where the prediction crosses from ground to roof (docs/evaluation-protocol.md).
        # They exist so the viewer can label its own error map honestly.
        err = (height - tru)[tvalid]
        if err.size:
            manifest["error_px_rmse_m"] = float(np.sqrt(np.mean(err ** 2)))
            manifest["error_px_mae_m"] = float(np.mean(np.abs(err)))
            manifest["error_px_bias_m"] = float(np.mean(err))
            manifest["error_px_p95_abs_m"] = float(np.percentile(np.abs(err), 95))
            manifest["error_note"] = ("per-pixel, whole tile. Roof edges dominate; "
                                      "per-building error is the comparable figure.")

        if args.cls:
            cls, _ = _load_raster(Path(args.cls))
            cls = np.asarray(cls)
            if cls.shape != (H, W):
                ys = np.linspace(0, cls.shape[0] - 1, H).astype(np.int32)
                xs = np.linspace(0, cls.shape[1] - 1, W).astype(np.int32)
                cls = cls[np.ix_(ys, xs)]
            bmask = (cls == 6) & tvalid          # CLS_BUILDING = 6
            gmask = (cls == 2) & tvalid          # ground
            if bmask.any():
                be = (height - tru)[bmask]
                manifest["error_building_px_rmse_m"] = float(np.sqrt(np.mean(be ** 2)))
                manifest["building_pixels"] = int(bmask.sum())
            if gmask.any():
                ge = (height - tru)[gmask]
                manifest["error_ground_px_rmse_m"] = float(np.sqrt(np.mean(ge ** 2)))

            # The comparable number. One height per building on both sides, median of each
            # component -- the same function `tools/evaluate.py` uses for the headline
            # figure, so the viewer and the benchmark table cannot drift apart. Computed on
            # the native grid for the reason given at `height_full`.
            if native_gsd:
                import sys
                sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
                from depthwizard.metrics import building_instances, building_wise_metrics

                cls_full = np.asarray(_load_raster(Path(args.cls))[0])
                if cls_full.shape != height_full.shape:
                    fy = np.linspace(0, cls_full.shape[0] - 1,
                                     height_full.shape[0]).astype(np.int32)
                    fx = np.linspace(0, cls_full.shape[1] - 1,
                                     height_full.shape[1]).astype(np.int32)
                    cls_full = cls_full[np.ix_(fy, fx)]
                bo, bt = building_instances(height_full, tru_full, cls_full, gsd=native_gsd)
                bw = building_wise_metrics(bo, bt)
                if bw.get("n_buildings"):
                    manifest["building_wise"] = bw
                    manifest["building_wise_note"] = (
                        "one height per building, median on both sides, components under "
                        "25 m2 dropped. This is the figure HTC-DC Net and "
                        "GlobalBuildingAtlas report, and the only one comparable to them."
                    )

    if args.texture:
        from PIL import Image
        tex, _ = _load_raster(Path(args.texture))
        rgb = _to_uint8_rgb(np.asarray(tex))
        im = Image.fromarray(rgb)
        if im.size != (W, H):
            im = im.resize((W, H), Image.LANCZOS)
        im.save(out / "texture.jpg", quality=92)
        manifest["files"]["texture"] = "texture.jpg"

    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))

    # Keep an index of scenes so the viewer can offer a picker without a server API.
    scenes_root = out.parent
    if getattr(args, "no_index", False):
        # An uploaded scene must not join the shared picker: this directory is served
        # to everyone, so one visitor's image would otherwise appear in the scene
        # picker of every visitor after them. serve_app passes this for uploads and
        # the viewer lists the scene client-side, for that session only.
        return
    index = scenes_root / "index.json"
    known = []
    if index.exists():
        try:
            # Explicit UTF-8: this file is WRITTEN as UTF-8 below, but read_text() defaults
            # to the locale encoding, which is cp1252 on Windows. Any scene name carrying
            # an em-dash therefore crashed the next export that touched the index.
            known = json.loads(index.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            known = []
    # Drop this scene's old entry, and any entry whose directory has since been deleted.
    # Without the second check a removed scene stays in the index forever and the viewer
    # tries to fetch a manifest that is not there.
    known = [k for k in known
             if k.get("dir") != out.name and (scenes_root / k.get("dir", "")).is_dir()]
    # Exactly one landing scene, or the viewer's choice depends on dict order.
    if args.default_scene:
        for k in known:
            k.pop("default", None)
    known.append({"dir": out.name, "name": manifest["name"],
                  "width": W, "height": H, "gsd_m": manifest["gsd_m"],
                  "terrain": manifest.get("terrain"), "place": manifest.get("place"),
                  "has_truth": "truth" in manifest["files"],
                  "has_sigma": "sigma" in manifest["files"],
                  **({"default": True} if args.default_scene else {})})
    # Order by landscape, in the order the problem statement names them -- "stability
    # across urban, sparse, hilly and forested landscapes" -- so the picker answers their
    # criterion on sight. Sorting by directory name instead put Forested first and, worse,
    # every upload re-sorted the list and destroyed the curated order.
    ORDER = ["urban", "sparse", "hilly", "forested", "mixed", "detail", "upload"]
    known.sort(key=lambda k: (ORDER.index(k.get("terrain")) if k.get("terrain") in ORDER
                              else len(ORDER), k.get("name") or k["dir"]))
    index.write_text(json.dumps(known, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"scene '{manifest['name']}' -> {out}")
    print(f"  {W} x {H}   height {manifest['height_min_m']:.1f} .. {manifest['height_max_m']:.1f} m")
    if manifest["gsd_m"]:
        print(f"  GSD {manifest['gsd_m']:.3f} m/px   extent "
              f"{manifest['extent_m'][0]:.0f} x {manifest['extent_m'][1]:.0f} m")
    else:
        print("  GSD unknown -- viewer will report distances in pixels")
    if n_bad:
        print(f"  filled {n_bad:,} void pixels ({100*n_bad/(H*W):.2f}%)")
    total = sum(f.stat().st_size for f in out.iterdir() if f.is_file())
    print(f"  bundle {total/1e6:.1f} MB")


if __name__ == "__main__":
    main()
