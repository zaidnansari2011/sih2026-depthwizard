"""DFC2019 Track 1 preparation: extract -> probe -> shard.

Three subcommands, run in order:

    python tools/prepare_data.py extract          # unzip archives into data/extracted
    python tools/prepare_data.py probe            # inspect real files, report facts
    python tools/prepare_data.py shard            # cut crops -> data/shards

`probe` exists because the DFC2019 docs do NOT specify tile dimensions, height units
beyond "metres AGL", or the void/NoData value. We measure them instead of assuming,
and every later stage reads the probe report rather than a hardcoded guess.

Splits are by GEOGRAPHIC REGION, never by crop. Crops from one region landing in both
train and test would leak, and the resulting RMSE would be a lie.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]          # D:\sih2026
RAW = ROOT / "data" / "raw"
EXTRACTED = ROOT / "data" / "extracted"
SHARDS = ROOT / "data" / "shards"
REPORT = SHARDS / "probe.json"

# JAX_004_007_RGB.tif -> region "JAX_004", tile "JAX_004_007"
TILE_RE = re.compile(r"^(?P<region>[A-Za-z]+_\d+)_(?P<sub>\d+)_(?P<kind>RGB|AGL|CLS)$", re.I)

# DFC2019 semantic classes (Track 1 CLS), used for per-terrain error breakdown.
CLS_NAMES = {2: "ground", 5: "vegetation", 6: "building", 9: "water", 17: "bridge"}


def _read_tif(path: Path) -> np.ndarray:
    try:
        import rasterio
    except ImportError:
        sys.exit("rasterio not installed:  pip install -r requirements.txt")
    with rasterio.open(path) as src:
        arr = src.read()
    return arr[0] if arr.shape[0] == 1 else np.transpose(arr, (1, 2, 0))


# --------------------------------------------------------------------------- extract

def _looks_like_dfc(zf: zipfile.ZipFile) -> int:
    """Count members that match our tile naming. Zero means this is somebody else's zip."""
    return sum(1 for n in zf.namelist() if TILE_RE.match(Path(n).stem))


def cmd_extract(args):
    if args.archives:
        archives = [Path(a) for a in args.archives]
        missing = [a for a in archives if not a.exists()]
        if missing:
            sys.exit(f"not found: {missing}")
    else:
        search = [RAW, Path(args.also)] if args.also else [RAW]
        archives = []
        for d in search:
            if d.exists():
                archives += sorted(d.glob("*.zip"))
        if not archives:
            sys.exit(f"No .zip found in {[str(d) for d in search]}")

    EXTRACTED.mkdir(parents=True, exist_ok=True)
    total = 0
    for z in archives:
        gb = z.stat().st_size / 1e9
        with zipfile.ZipFile(z) as zf:
            members = zf.namelist()
            n_match = _looks_like_dfc(zf)
            # A downloads folder holds all sorts of things. Only unpack archives that
            # actually contain DFC tiles, or a stray fonts.zip ends up in data/extracted
            # and every later stage has to reason about it.
            if not n_match and not args.force:
                print(f"\n{z.name}  ({gb:.2f} GB)  -- skipped, no DFC tiles inside")
                continue
            print(f"\n{z.name}  ({gb:.2f} GB)")
            print(f"  {len(members)} members, {n_match} recognised tiles; e.g. {members[1:3]}")
            if args.dry_run:
                continue
            zf.extractall(EXTRACTED)
            total += n_match
        print(f"  -> {EXTRACTED}")

    if not args.dry_run:
        tifs = list(EXTRACTED.rglob("*.tif"))
        print(f"\nextracted {len(tifs)} .tif files ({total} recognised this run)")


# ----------------------------------------------------------------------------- probe

def _index() -> dict[str, dict[str, Path]]:
    """Map tile id -> {kind: path}. Tolerant of whatever directory layout the zips use."""
    tiles: dict[str, dict[str, Path]] = defaultdict(dict)
    for p in EXTRACTED.rglob("*.tif"):
        m = TILE_RE.match(p.stem)
        if m:
            tiles[f"{m['region']}_{m['sub']}"][m["kind"].upper()] = p
    return tiles


def cmd_probe(args):
    tiles = _index()
    if not tiles:
        sys.exit(f"No recognisable *_RGB/_AGL/_CLS tifs under {EXTRACTED}. Run `extract` first.")

    paired = {t: k for t, k in tiles.items() if "RGB" in k and "AGL" in k}
    regions = defaultdict(list)
    for t in paired:
        regions[t.rsplit("_", 1)[0]].append(t)

    print(f"tiles total      : {len(tiles)}")
    print(f"tiles RGB+AGL    : {len(paired)}")
    print(f"tiles with CLS   : {sum('CLS' in k for k in paired.values())}")
    print(f"regions          : {len(regions)}")
    for r in sorted(regions)[:12]:
        print(f"    {r:12s} {len(regions[r]):4d} tiles")
    if len(regions) > 12:
        print(f"    ... and {len(regions) - 12} more")

    # Measure, don't assume. Sample across the whole set rather than the first N, or we
    # would characterise one city and call it the dataset.
    ordered = sorted(paired)
    step = max(1, len(ordered) // max(1, args.sample))
    sample = ordered[::step][: args.sample]
    shapes, dtypes, mins, maxs, void_candidates = set(), set(), [], [], defaultdict(int)
    gsds, crs_set, pooled, cls_heights = set(), set(), [], defaultdict(list)
    for t in sample:
        rgb = _read_tif(paired[t]["RGB"])
        agl = _read_tif(paired[t]["AGL"]).astype(np.float64)
        shapes.add((rgb.shape, agl.shape))
        dtypes.add((str(rgb.dtype), str(_read_tif(paired[t]["AGL"]).dtype)))
        finite = agl[np.isfinite(agl)]
        if finite.size:
            mins.append(float(finite.min()))
            maxs.append(float(finite.max()))
            pooled.append(finite[::37])          # thin, so 24 tiles stay cheap to pool
        # Void sentinels show up as repeated extreme values or NaN.
        if np.isnan(agl).any():
            void_candidates["nan"] += 1
        for sentinel in (-9999, -10000, -32768, 0):
            frac = float((agl == sentinel).mean())
            if frac > 0.01:
                void_candidates[str(sentinel)] += 1
        # Georeferencing: the ground sample distance decides whether any measurement in
        # the viewer is in metres or in pixels, so read it rather than trusting "30 cm".
        try:
            import rasterio
            with rasterio.open(paired[t]["AGL"]) as src:
                if src.transform:
                    gsds.add(round(abs(src.transform.a), 4))
                crs_set.add(str(src.crs))
        except Exception:
            pass
        if "CLS" in paired[t]:
            cls = _read_tif(paired[t]["CLS"])
            for code, name in CLS_NAMES.items():
                sel = (cls == code) & np.isfinite(agl)
                if sel.sum() > 500:
                    cls_heights[name].append(agl[sel][::17])

    print(f"\nsampled {len(sample)} tiles (every {step}th, spread across the set)")
    print(f"  shapes (rgb, agl) : {shapes}")
    print(f"  dtypes (rgb, agl) : {dtypes}")
    print(f"  CRS               : {crs_set or 'none'}")
    print(f"  GSD (m/px)        : {sorted(gsds) or 'no transform'}")
    if mins:
        print(f"  AGL min  : {min(mins):.2f} m   (per-tile min, range {min(mins):.1f}..{max(mins):.1f})")
        print(f"  AGL max  : {max(maxs):.2f} m   (per-tile max, range {min(maxs):.1f}..{max(maxs):.1f})")

    pcts = {}
    if pooled:
        allh = np.concatenate(pooled)
        for q in (1, 5, 25, 50, 75, 90, 95, 99, 99.9):
            pcts[f"p{q}"] = float(np.percentile(allh, q))
        print(f"\n  pooled height distribution over {allh.size:,} pixels:")
        print("    " + "  ".join(f"p{q}={pcts[f'p{q}']:.1f}" for q in (1, 25, 50, 75, 95, 99, 99.9)))
        print(f"    mean {allh.mean():.2f} m   std {allh.std():.2f} m")

    if cls_heights:
        print("\n  height by semantic class:")
        for name, chunks in cls_heights.items():
            v = np.concatenate(chunks)
            print(f"    {name:12s} median {np.median(v):7.2f} m   p95 {np.percentile(v, 95):7.2f} m   n={v.size:,}")

    print(f"\n  void candidates (>1% of pixels, or NaN present): {dict(void_candidates) or 'none detected'}")

    void = args.void
    if void is None:
        if "nan" in void_candidates:
            void = float("nan")
        else:
            numeric = [k for k in void_candidates if k not in ("nan", "0")]
            void = float(numeric[0]) if numeric else None
    print(f"\n  -> void value taken as: {void}")
    print("     (override with --void if this looks wrong; 0 is ambiguous, it is a legal height)")

    is_nan_void = bool(void is not None and isinstance(void, float) and np.isnan(void))
    SHARDS.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps({
        "n_tiles": len(paired),
        "n_regions": len(regions),
        "regions": {r: sorted(v) for r, v in regions.items()},
        "shapes": [list(map(list, s)) for s in shapes],
        "crs": sorted(crs_set),
        "gsd_m": sorted(gsds),
        "agl_min": min(mins) if mins else None,
        "agl_max": max(maxs) if maxs else None,
        # train.py reads agl_p95 to set the output scale. Using the max instead would
        # tie the scale to a single tall structure and squash everything else.
        "agl_p95": pcts.get("p95"),
        "agl_percentiles": pcts,
        "void": None if void is None or is_nan_void else float(void),
        "void_is_nan": is_nan_void,
    }, indent=2))
    print(f"\nwrote {REPORT}")


# ----------------------------------------------------------------------------- shard

def _split_regions(regions: list[str], seed: int) -> dict[str, str]:
    """Region-level split. Test is held out on day one and never tuned against."""
    rng = np.random.default_rng(seed)
    order = sorted(regions)
    rng.shuffle(order)
    n = len(order)
    n_test = max(1, round(0.15 * n))
    n_val = max(1, round(0.15 * n))
    assign = {}
    for i, r in enumerate(order):
        assign[r] = "test" if i < n_test else "val" if i < n_test + n_val else "train"
    return assign


def cmd_shard(args):
    if not REPORT.exists():
        sys.exit("No probe.json. Run `probe` first.")
    meta = json.loads(REPORT.read_text())
    void, void_is_nan = meta.get("void"), meta.get("void_is_nan", False)

    tiles = _index()
    paired = {t: k for t, k in tiles.items() if "RGB" in k and "AGL" in k}
    regions = sorted({t.rsplit("_", 1)[0] for t in paired})
    assign = _split_regions(regions, args.seed)

    counts = defaultdict(int)
    for r, s in assign.items():
        counts[s] += 1
    print(f"region split: {dict(counts)}  (seed {args.seed})")

    buckets: dict[str, list[str]] = defaultdict(list)
    for t in sorted(paired):
        buckets[assign[t.rsplit("_", 1)[0]]].append(t)

    # Shuffle tile order before cutting. Shards are the unit of shuffling at train time
    # (see depthwizard/dataset.py), so a shard filled from consecutive tiles would be
    # one neighbourhood of one city -- batches drawn from it would be near-duplicates
    # and the gradient estimate correspondingly poor. Shuffling here means each shard
    # spans ~32 tiles sampled from across the split.
    rng = np.random.default_rng(args.seed)
    for ids in buckets.values():
        rng.shuffle(ids)

    SHARDS.mkdir(parents=True, exist_ok=True)
    (SHARDS / "split.json").write_text(json.dumps(assign, indent=2))

    C, S = args.crop, args.stride
    for split, tile_ids in buckets.items():
        if args.max_tiles:
            tile_ids = tile_ids[: args.max_tiles]
        rgb_buf, agl_buf, cls_buf, ids = [], [], [], []
        shard_i, written = 0, 0

        def flush():
            nonlocal shard_i, rgb_buf, agl_buf, cls_buf, ids, written
            if not rgb_buf:
                return
            out = SHARDS / f"{split}_{shard_i:04d}.npz"
            perm = rng.permutation(len(rgb_buf))                  # decorrelate within shard too
            payload = {
                "rgb": np.stack(rgb_buf)[perm],                  # uint8  N,C,C,3
                "agl": np.stack(agl_buf).astype(np.float16)[perm],  # float16 N,C,C
                "tile": np.array(ids)[perm],
            }
            if cls_buf:
                payload["cls"] = np.stack(cls_buf)[perm]         # uint8 N,C,C
            np.savez_compressed(out, **payload)
            written += len(rgb_buf)
            print(f"  {out.name}  {len(rgb_buf)} crops  {out.stat().st_size/1e6:.1f} MB")
            shard_i += 1
            rgb_buf, agl_buf, cls_buf, ids = [], [], [], []

        print(f"\n[{split}] {len(tile_ids)} tiles")
        for t in tile_ids:
            rgb = _read_tif(paired[t]["RGB"])
            agl = _read_tif(paired[t]["AGL"]).astype(np.float32)
            cls = _read_tif(paired[t]["CLS"]).astype(np.uint8) if "CLS" in paired[t] else None
            if rgb.ndim == 3 and rgb.shape[2] > 3:
                rgb = rgb[:, :, :3]

            # Non-finite is always invalid, whatever the probe concluded about sentinels.
            # Measured on DFC2019: only ~3% of tiles contain any NaN and then under
            # 0.005% of their pixels, so this costs us essentially no data -- but a
            # single NaN reaching an MSE would take the whole run's gradient with it.
            invalid = ~np.isfinite(agl)
            if void is not None:
                invalid |= (agl == void)
            # Implausible heights are void in disguise. Measured range is -3 to 204 m.
            invalid |= (agl < -10.0) | (agl > 400.0)
            h, w = agl.shape
            for y in range(0, h - C + 1, S):
                for x in range(0, w - C + 1, S):
                    a = agl[y:y + C, x:x + C]
                    bad = invalid[y:y + C, x:x + C]
                    # Drop crops that are mostly void -- they teach the model nothing
                    # and quietly poison the loss.
                    if bad.mean() > args.max_void:
                        continue
                    a = np.where(bad, 0.0, a)
                    rgb_buf.append(rgb[y:y + C, x:x + C])
                    agl_buf.append(a)
                    if cls is not None:
                        cls_buf.append(cls[y:y + C, x:x + C])
                    ids.append(t)
                    if len(rgb_buf) >= args.per_shard:
                        flush()
        flush()
        print(f"[{split}] {written} crops total")

    total = sum(f.stat().st_size for f in SHARDS.glob("*.npz"))
    print(f"\nshards: {total/1e9:.2f} GB in {SHARDS}")
    print("Upload once as a Kaggle Dataset; every notebook session then mounts it instantly.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("extract", help="unzip archives into data/extracted")
    e.add_argument("archives", nargs="*", help="explicit .zip paths (preferred)")
    e.add_argument("--also", help="extra directory to scan for zips, e.g. D:\\Downloads")
    e.add_argument("--force", action="store_true", help="extract even if no DFC tiles are recognised")
    e.add_argument("--dry-run", action="store_true", help="list archive contents without extracting")
    e.set_defaults(func=cmd_extract)

    p = sub.add_parser("probe", help="measure shapes, ranges and the void value")
    p.add_argument("--sample", type=int, default=8, help="tiles to inspect")
    p.add_argument("--void", type=float, default=None, help="override detected void value")
    p.set_defaults(func=cmd_probe)

    s = sub.add_parser("shard", help="cut crops into npz shards")
    # 518 is DA-V2's native training resolution AND a multiple of its patch size 14.
    # 256 is neither: the head computes patch_h = H // 14, so a 256 input returns 252
    # with no warning. Cropping at 518 also means the ground truth is never resampled,
    # so no interpolation error is baked into the labels.
    # 1024 = 506 + 518, so stride 506 tiles a 1024 tile exactly 2x2 with 12 px overlap.
    s.add_argument("--crop", type=int, default=518)
    s.add_argument("--stride", type=int, default=506, help="506 tiles a 1024 tile 2x2")
    # 128 crops at 518x518 is ~200 MB uncompressed -- small enough that the training
    # loader can hold two in RAM for shuffling without the process ballooning.
    s.add_argument("--per-shard", type=int, default=128)
    s.add_argument("--max-void", type=float, default=0.30, help="drop crops with more void than this")
    s.add_argument("--max-tiles", type=int, default=0, help="cap tiles per split (0 = all); use a small value for a week-1 smoke run")
    s.add_argument("--seed", type=int, default=1337)
    s.set_defaults(func=cmd_shard)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
