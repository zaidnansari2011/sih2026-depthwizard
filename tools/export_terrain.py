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
            meta["px_units"] = "degrees" if (src.crs and src.crs.is_geographic) else "metres"
    return arr, meta


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
    index = scenes_root / "index.json"
    known = []
    if index.exists():
        try:
            known = json.loads(index.read_text())
        except json.JSONDecodeError:
            known = []
    known = [k for k in known if k.get("dir") != out.name]
    known.append({"dir": out.name, "name": manifest["name"],
                  "width": W, "height": H, "gsd_m": manifest["gsd_m"]})
    index.write_text(json.dumps(sorted(known, key=lambda k: k["dir"]), indent=2))

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
