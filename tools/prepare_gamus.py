"""GAMUS preparation: list -> probe -> shard, streaming, with no raw data kept.

    python tools/prepare_gamus.py probe                 # measure, verify class mapping
    python tools/prepare_gamus.py shard --pilot 500     # cheap pilot ingest
    python tools/prepare_gamus.py shard                 # full PHL+DC train + NYC holdout

Why this is not just `prepare_data.py` with a different reader
--------------------------------------------------------------
Four things about GAMUS differ from DFC2019 and every one of them fails *silently*:

1. **Class 3 is BUILDING and class 6 is TREE** -- the reverse of the obvious guess, and
   the reverse of DFC2019 where building is 6. Measured three ways (rectangularity,
   flat-roof height variance, and looking at a rendered Philadelphia tile). Taking the
   high index as "building" would train and score the per-building metric on canopy.
   `_verify_class_map` re-checks this on real data every run and aborts if it flips.

2. **Void is a -5.00 sentinel, not NaN.** GAMUS contains zero NaN, so DFC2019's
   `~np.isfinite` void test catches nothing at all. Water is written as exactly -5.00.

3. **GAMUS's own train/val/test splits leak spatially.** DC tiles are named on a grid
   (`DC_row_col`) and adjacent tiles land in different splits 900 times (test/train),
   872 (train/val) and 213 (test/val). We discard their splits and cut by CITY, which
   makes the leakage irrelevant by construction rather than by filtering.

4. **GSD is 0.33 m, not 0.31 m.** Recorded in the probe report; never hardcoded downstream.

5. **GAMUS ships no RGB for NYC.** Heights and classes exist for all 8,724 tiles, imagery
   for only 6,557 -- every one of the 2,167 NYC tiles is height-only. A pipeline whose
   input is a photograph cannot use them at all, so the usable corpus is PHL + DC and the
   holdout has to be carved from the DC grid instead of from a whole city.

Disk
----
Raw `.h5` is never kept: each tile is downloaded, cropped, and deleted before the next.
Transient usage stays near a few MB, and `--min-free-gb` aborts before filling the volume
rather than after. The run is resumable -- a manifest records every tile already folded
into a flushed shard, so an interrupted ingest restarts where it stopped.

Output contract is byte-identical to prepare_data.py, so depthwizard/dataset.py reads
these shards with no change: rgb uint8 (N,C,C,3), agl float16 (N,C,C), cls uint8, tile str.
Shards are named `train_g0000.npz` so they can be hardlinked alongside the DFC2019 shards
into a union directory without colliding.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]              # D:\sih2026
OUT = ROOT / "data" / "shards_gamus"
REPORT = OUT / "probe.json"
MANIFEST = OUT / "manifest.json"
TMP = ROOT / "data" / "_gamus_tmp"

HF_API = "https://huggingface.co/api/datasets/earthflow/GAMUS"
HF_FILE = "https://huggingface.co/datasets/earthflow/GAMUS/resolve/main/{}"

GAMUS_GSD_M = 0.33
VOID_SENTINEL = -5.0

# GAMUS class -> DFC2019 class, so metrics.py's CLS_BUILDING/CLS_VEG/... work unchanged.
# GAMUS: 0 clutter, 1 ground, 2 impervious/pavement, 3 BUILDING, 4 water, 5 road, 6 TREE.
# DFC2019: 2 ground, 5 vegetation, 6 building, 9 water, 17 bridge.
CLS_MAP = {0: 2, 1: 2, 2: 2, 3: 6, 4: 9, 5: 2, 6: 5}
GAMUS_BUILDING, GAMUS_TREE, GAMUS_WATER = 3, 6, 4

DFC_BUILDING, DFC_VEG, DFC_GROUND, DFC_WATER = 6, 5, 2, 9


# ------------------------------------------------------------------ listing / download

def _list_tiles() -> dict[str, list[str]]:
    """Every USABLE tile, grouped by city. Ignores GAMUS's own (leaking) split dirs.

    Listed from `images/`, not `heights/`, because the two do not match: GAMUS ships
    heights and classes for all 8,724 tiles but RGB for only 6,557. **All 2,167 NYC tiles
    are missing their imagery**, so NYC cannot be used for a model that takes a photograph
    as input. Listing from the height side and assuming a matching RGB exists is a 404
    two thousand tiles into a run.
    """
    with urllib.request.urlopen(HF_API, timeout=90) as r:
        meta = json.load(r)
    heights = {n.split("/")[-1].replace("_AGL.h5", "")
               for n in (s["rfilename"] for s in meta["siblings"])
               if n.startswith("heights/")}
    by_city: dict[str, list[str]] = defaultdict(list)
    for s in meta["siblings"]:
        name = s["rfilename"]
        if not name.startswith("images/") or not name.endswith("_RGB.h5"):
            continue
        split_dir = name.split("/")[1]                       # their split -- kept only to
        tile = name.split("/")[-1].replace("_RGB.h5", "")    # rebuild the download path
        if tile in heights:
            by_city[tile.split("_")[0]].append(f"{split_dir}/{tile}")
    return {c: sorted(v) for c, v in by_city.items()}


def _dc_block(path: str) -> int | None:
    """Grid column of a DC tile, or None if the name is not on the grid.

    DC tiles are named `DC_row_col` over a 77 x 66 grid. That grid is the only geography
    any GAMUS filename exposes, and it is what makes a spatially honest holdout possible
    at all -- PHL ids are sequential with no recoverable position.
    """
    m = re.match(r"DC_(\d+)_(\d+)$", path.split("/")[-1])
    return int(m.group(2)) if m else None


def _fetch(kind: str, split_dir: str, tile: str, dest: Path, tries: int = 4) -> Path:
    """kind is images|heights|classes. Returns the local path.

    Retries with backoff. At 12 concurrent streams roughly 5% of reads time out, and over
    6,204 tiles that would silently discard ~300 of them -- a hole in the training set
    nobody would notice until the corpus came up short. A transient timeout is not a
    missing file, so it must not be treated as one.
    """
    suffix = {"images": "RGB", "heights": "AGL", "classes": "CLS"}[kind]
    url = HF_FILE.format(f"{kind}/{split_dir}/{tile}_{suffix}.h5")
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=120) as r, open(dest, "wb") as f:
                shutil.copyfileobj(r, f)
            return dest
        except urllib.error.HTTPError:
            raise                                   # 404 is real; retrying cannot fix it
        except Exception:
            dest.unlink(missing_ok=True)
            if attempt == tries - 1:
                raise
            time.sleep(2 ** attempt)
    return dest


def _read_h5(path: Path) -> np.ndarray:
    import h5py
    with h5py.File(path, "r") as f:
        return f["image"][...]


def _load_tile(split_dir: str, tile: str, want_cls: bool = True):
    """Download one triplet, read it, delete the files. Nothing raw survives this call.

    `want_cls=False` skips the class raster entirely, which is 4.2 MB of the 11.5 MB per
    tile. Training never reads `cls` -- dataset.py only requests it for the *validation*
    set (train.py:351), and our validation set stays pure DFC2019. Downloading it for
    6,204 training tiles would cost 26 GB of transfer to fill an array nothing opens.
    """
    TMP.mkdir(parents=True, exist_ok=True)
    kinds = ("images", "heights", "classes") if want_cls else ("images", "heights")
    paths = {}
    try:
        for kind in kinds:
            paths[kind] = _fetch(kind, split_dir, tile, TMP / f"{tile}_{kind}.h5")
        rgb = _read_h5(paths["images"])
        agl = _read_h5(paths["heights"]).astype(np.float32)
        cls = _read_h5(paths["classes"]) if want_cls else None
    finally:
        for p in paths.values():
            p.unlink(missing_ok=True)
    return rgb, agl, cls


def _prefetch(paths: list[str], workers: int, want_cls: bool):
    """Yield (path, loaded-or-exception) in order, with `workers` downloads in flight.

    A single HTTPS stream from HuggingFace measured ~0.9 MB/s here, which is 22 hours for
    the full corpus -- the bottleneck is per-connection latency, not bandwidth, so the fix
    is concurrency rather than a bigger buffer. Order is preserved so the manifest stays a
    truthful resume point: a tile is only ever recorded as done after everything before it
    has been folded into a flushed shard.
    """
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=workers) as ex:
        pending: list = []
        it = iter(paths)
        for p in it:
            pending.append((p, ex.submit(_load_tile, *p.split("/"), want_cls)))
            if len(pending) >= workers * 2:
                break
        while pending:
            path, fut = pending.pop(0)
            try:
                yield path, fut.result()
            except Exception as e:
                yield path, e
            nxt = next(it, None)
            if nxt is not None:
                pending.append((nxt, ex.submit(_load_tile, *nxt.split("/"), want_cls)))


# ------------------------------------------------------------------------ verification

def _verify_class_map(samples: list[tuple[np.ndarray, np.ndarray]]) -> dict:
    """Abort unless class 3 really is building and class 6 really is tree.

    Two independent signals, because either alone can be argued with:
      * height   -- both are elevated, so this only proves they are the two raised classes
      * flatness -- roofs are flat, canopy is not. This is what actually separates them.

    A dataset revision that renumbered the classes would sail past a height-only check.
    """
    b_sd, t_sd, b_h, t_h, g_h = [], [], [], [], []
    from scipy import ndimage
    for agl, cls in samples:
        for code, sd_acc, h_acc in ((GAMUS_BUILDING, b_sd, b_h), (GAMUS_TREE, t_sd, t_h)):
            m = cls == code
            if m.sum() < 2000:
                continue
            h_acc.append(float(np.median(agl[m])))
            lab, n = ndimage.label(m, structure=np.ones((3, 3)))
            if n:
                cnt = np.bincount(lab.ravel(), minlength=n + 1)[1:]
                big = np.arange(1, n + 1)[cnt >= 278]
                if len(big):
                    sd_acc.append(float(np.median(ndimage.standard_deviation(agl, lab, big))))
        gm = np.isin(cls, [0, 1, 2, 5])
        if gm.sum() > 2000:
            g_h.append(float(np.median(agl[gm])))

    if not (b_sd and t_sd and b_h and t_h):
        sys.exit("class verification: not enough building/tree pixels sampled")

    stats = {
        "building_median_h": float(np.median(b_h)), "tree_median_h": float(np.median(t_h)),
        "ground_median_h": float(np.median(g_h)) if g_h else None,
        "building_roof_sd": float(np.median(b_sd)), "tree_roof_sd": float(np.median(t_sd)),
    }
    if stats["building_roof_sd"] >= stats["tree_roof_sd"]:
        sys.exit(
            "ABORT: class 3 is rougher than class 6 "
            f"({stats['building_roof_sd']:.2f} m vs {stats['tree_roof_sd']:.2f} m). "
            "Buildings have flatter tops than canopy, so the class mapping has changed. "
            "Re-derive CLS_MAP before sharding -- do not 'fix' this by swapping blindly."
        )
    if stats["ground_median_h"] is not None and \
            stats["building_median_h"] - stats["ground_median_h"] < 2.0:
        sys.exit("ABORT: class 3 is not raised above ground; mapping is wrong.")
    return stats


def _guard_disk(min_free_gb: float):
    free = shutil.disk_usage(str(ROOT)).free / 1e9
    if free < min_free_gb:
        sys.exit(f"ABORT: {free:.1f} GB free, below --min-free-gb {min_free_gb}. "
                 "Shards written so far are intact; rerun to resume after freeing space.")
    return free


# ------------------------------------------------------------------------------- probe

def cmd_probe(args):
    rng = np.random.default_rng(args.seed)
    cities = _list_tiles()
    print("GAMUS tiles by city:", {c: len(v) for c, v in cities.items()})

    picks = []
    for c, v in cities.items():
        picks += [(c, p) for p in rng.choice(v, size=min(args.sample, len(v)), replace=False)]

    samples, heights = [], []
    for c, path in picks:
        split_dir, tile = path.split("/")
        _, agl, cls = _load_tile(split_dir, tile)
        samples.append((agl, cls))
        heights.append(agl.ravel())
        print(f"  {tile:16s} {c}  p95 {np.percentile(agl, 95):6.2f} m  max {agl.max():7.2f} m")

    stats = _verify_class_map(samples)
    print("\nclass check PASSED:")
    print(f"  building (3): median {stats['building_median_h']:.2f} m, roof SD {stats['building_roof_sd']:.2f} m")
    print(f"  tree     (6): median {stats['tree_median_h']:.2f} m, roof SD {stats['tree_roof_sd']:.2f} m")

    v = np.concatenate(heights)
    report = {
        "n_tiles_total": sum(len(x) for x in cities.values()),
        "tiles_by_city": {c: len(x) for c, x in cities.items()},
        "gsd_m": GAMUS_GSD_M,
        "void_sentinel": VOID_SENTINEL,
        "void_is_nan": False,
        "nan_fraction": float(np.isnan(v).mean()),
        "sentinel_fraction": float((v == VOID_SENTINEL).mean()),
        "height_p50": float(np.percentile(v, 50)),
        "height_p95": float(np.percentile(v, 95)),
        "height_p99": float(np.percentile(v, 99)),
        "height_max": float(v.max()),
        "class_map_gamus_to_dfc": {str(k): val for k, val in CLS_MAP.items()},
        "class_check": stats,
        "sampled_tiles": len(picks),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {REPORT}")
    print(f"height p50 {report['height_p50']:.2f}  p95 {report['height_p95']:.2f}  max {report['height_max']:.1f}")


# ------------------------------------------------------------------------------- shard

def cmd_shard(args):
    if not REPORT.exists():
        sys.exit("No probe.json for GAMUS. Run `probe` first -- it verifies the class mapping.")

    cities = _list_tiles()
    print("usable GAMUS tiles (RGB present):", {c: len(v) for c, v in cities.items()})

    # Spatial holdout, carved from the DC grid.
    #
    # The original plan held out NYC entirely, which is impossible -- GAMUS ships no RGB
    # for NYC. The next-best honest split uses the only geography a filename exposes: DC's
    # grid. Columns at or beyond --holdout-col become the holdout, the single column before
    # it is DISCARDED as a buffer, and everything else trains. One column is 1024 px at
    # 0.33 m = 338 m of separation, so no holdout tile touches a training tile.
    #
    # Holding out all of DC instead would be cleaner still, and is wrong: DC is the tall
    # city (p95 39.7 m) and its tall buildings are the entire reason for this ingest.
    # Only training tiles are sharded. The held-out block is downloaded as whole GeoTIFFs
    # by `holdout`, because whole-tile RMSE is the deployable metric and reassembling
    # 1024-px tiles from overlapping 518-px crops would be reconstructing data we can keep.
    buckets: dict[str, list[str]] = {"train_g": []}
    n_hold = n_buffer = 0
    for c, paths in cities.items():
        for p in paths:
            col = _dc_block(p) if c == "DC" else None
            if col is None:
                buckets["train_g"].append(p)          # all of PHL
            elif col >= args.holdout_col:
                n_hold += 1                            # -> cmd_holdout, never trained on
            elif col == args.holdout_col - 1:
                n_buffer += 1                          # dropped, deliberately
            else:
                buckets["train_g"].append(p)

    rng = np.random.default_rng(args.seed)
    rng.shuffle(buckets["train_g"])
    if args.pilot:
        buckets["train_g"] = buckets["train_g"][: args.pilot]
    print(f"holdout {n_hold} tiles (DC cols >= {args.holdout_col}) excluded from training; "
          f"buffer column {args.holdout_col - 1} drops {n_buffer} more")

    OUT.mkdir(parents=True, exist_ok=True)
    done = set(json.loads(MANIFEST.read_text())["tiles"]) if MANIFEST.exists() else set()
    if done:
        print(f"resuming: {len(done)} tiles already sharded")

    print(f"train_g: {len(buckets['train_g'])} tiles to shard")

    C, S = args.crop, args.stride
    for split, paths in buckets.items():
        todo = [p for p in paths if p not in done]
        if not todo:
            continue
        existing = len(list(OUT.glob(f"{split}_*.npz")))
        rgb_buf, agl_buf, cls_buf, ids = [], [], [], []
        shard_i, written, kept_tiles = existing, 0, []

        def flush():
            nonlocal shard_i, rgb_buf, agl_buf, cls_buf, ids, written, kept_tiles
            if not rgb_buf:
                return
            out = OUT / f"{split}_{shard_i:04d}.npz"
            perm = rng.permutation(len(rgb_buf))
            payload = {
                "rgb": np.stack(rgb_buf)[perm],
                "agl": np.stack(agl_buf).astype(np.float16)[perm],
                "tile": np.array(ids)[perm],
            }
            if cls_buf:                       # absent for training shards, by design
                payload["cls"] = np.stack(cls_buf)[perm]
            # Write then rename. A 130 MB savez_compressed takes seconds, and this ingest
            # is meant to run for hours in the background while other things read the same
            # directory -- a reader that opens the final name mid-write gets a truncated
            # archive. Rename on the same volume is atomic, so a shard is either absent or
            # complete, never partial.
            # The temp name must still end in .npz (numpy appends the extension otherwise)
            # while NOT matching the `{split}_*.npz` glob that dataset.py discovers shards
            # with -- hence the prefix rather than a `.part` suffix.
            tmp = out.parent / f"_tmp_{out.name}"
            np.savez_compressed(tmp, **payload)
            tmp.replace(out)
            written += len(rgb_buf)
            print(f"  {out.name}  {len(rgb_buf)} crops  {out.stat().st_size/1e6:.1f} MB  "
                  f"[{len(done)}/{len(paths)} tiles]")
            shard_i += 1
            rgb_buf, agl_buf, cls_buf, ids = [], [], [], []
            done.update(kept_tiles)
            kept_tiles = []
            MANIFEST.write_text(json.dumps({"tiles": sorted(done)}))

        print(f"\n[{split}] {len(todo)} tiles to do, {args.workers} downloads in flight")
        for i, (path, loaded) in enumerate(_prefetch(todo, args.workers, args.with_cls)):
            if i % 25 == 0:
                _guard_disk(args.min_free_gb)
            tile = path.split("/")[1]
            if isinstance(loaded, Exception):            # a single bad tile must not end a
                print(f"  !! {tile}: {type(loaded).__name__} {loaded} -- skipped")  # long run
                continue
            rgb, agl, cls = loaded
            if rgb.ndim == 3 and rgb.shape[2] > 3:
                rgb = rgb[:, :, :3]

            # GAMUS has no NaN at all, so DFC2019's isfinite test would mark nothing.
            # The sentinel is what actually encodes "no data" here.
            invalid = ~np.isfinite(agl)
            invalid |= np.isclose(agl, VOID_SENTINEL, atol=1e-3)
            invalid |= (agl < -10.0) | (agl > 400.0)

            cls_mapped = None
            if cls is not None:
                cls_mapped = np.zeros_like(cls, dtype=np.uint8)
                for g, d in CLS_MAP.items():
                    cls_mapped[cls == g] = d

            h, w = agl.shape
            for y in range(0, h - C + 1, S):
                for x in range(0, w - C + 1, S):
                    bad = invalid[y:y + C, x:x + C]
                    if bad.mean() > args.max_void:
                        continue
                    # NaN, not 0.0. dataset.py builds its validity mask with
                    # `np.isfinite(agl)`, and 0.0 is finite -- writing zeros here hands
                    # every void pixel to the loss as a genuine "0 m of ground", biasing
                    # the model toward under-calling height, which is the exact defect
                    # this corpus was ingested to fix. `cmd_holdout` already converts the
                    # sentinel to NaN; the two paths must agree.
                    #
                    # Measured 6 Sep 2026 across 15 tiles spanning DC and PHL: 0.000% of
                    # pixels are invalid (no NaN, no -5 sentinel, nothing out of range),
                    # so this was latent rather than live. The shards already written are
                    # therefore unaffected and were deliberately NOT re-ingested.
                    a = np.where(bad, np.nan, agl[y:y + C, x:x + C])
                    rgb_buf.append(rgb[y:y + C, x:x + C])
                    agl_buf.append(a)
                    if cls_mapped is not None:
                        cls_buf.append(cls_mapped[y:y + C, x:x + C])
                    ids.append(tile)
            kept_tiles.append(path)
            # Flush only on a tile boundary so no tile is split across two shards --
            # that is what makes the manifest a truthful resume point.
            if len(rgb_buf) >= args.per_shard:
                flush()
        flush()
        print(f"[{split}] {written} crops written")

    total = sum(f.stat().st_size for f in OUT.glob("*.npz"))
    print(f"\nGAMUS shards: {total/1e9:.2f} GB in {OUT}")
    if TMP.exists():
        shutil.rmtree(TMP, ignore_errors=True)


# ----------------------------------------------------------------------------- holdout

def _write_tif(path: Path, arr: np.ndarray):
    """GeoTIFF with no CRS and no transform -- exactly what DFC2019's tiles carry.

    Writing a fabricated transform here would be worse than writing none: downstream code
    already refuses to invent a GSD and labels axes "pixels" when it cannot find one.
    """
    import rasterio
    path.parent.mkdir(parents=True, exist_ok=True)
    if arr.ndim == 2:
        arr = arr[None]
    else:
        arr = np.transpose(arr, (2, 0, 1))
    with rasterio.open(path, "w", driver="GTiff", height=arr.shape[1], width=arr.shape[2],
                       count=arr.shape[0], dtype=arr.dtype) as dst:
        dst.write(arr)


def cmd_holdout(args):
    """Materialise the held-out DC block as whole GeoTIFF tiles.

    The holdout is NOT sharded. Shards are 518-px crops, and the deployable metric is
    whole-tile RMSE -- reassembling 1024-px tiles from overlapping crops to score them
    would be a reconstruction of data we could simply keep. 317 tiles is 3.6 GB, which is
    the cheapest honest option, and it lets tools/evaluate.py --corpus gamus run the
    identical protocol it runs on DFC2019.
    """
    cities = _list_tiles()
    hold = [p for p in cities.get("DC", [])
            if (_dc_block(p) or -1) >= args.holdout_col]
    if args.holdout_tiles:
        hold = sorted(hold)[: args.holdout_tiles]
    print(f"holdout: {len(hold)} DC tiles, columns >= {args.holdout_col}")

    g = ROOT / "data" / "extracted_gamus"
    done = 0
    for i, path in enumerate(hold):
        if i % 25 == 0:
            _guard_disk(args.min_free_gb)
        split_dir, tile = path.split("/")
        if (g / "RGB" / f"{tile}_RGB.tif").exists():
            done += 1
            continue
        try:
            rgb, agl, cls = _load_tile(split_dir, tile)
        except Exception as e:
            print(f"  !! {tile}: {type(e).__name__} {e} -- skipped")
            continue
        if rgb.ndim == 3 and rgb.shape[2] > 3:
            rgb = rgb[:, :, :3]
        agl = np.where(np.isclose(agl, VOID_SENTINEL, atol=1e-3), np.nan, agl)
        cls_mapped = np.zeros_like(cls, dtype=np.uint8)
        for gcode, d in CLS_MAP.items():
            cls_mapped[cls == gcode] = d
        _write_tif(g / "RGB" / f"{tile}_RGB.tif", rgb.astype(np.uint8))
        _write_tif(g / "Truth" / f"{tile}_AGL.tif", agl.astype(np.float32))
        _write_tif(g / "Truth" / f"{tile}_CLS.tif", cls_mapped)
        done += 1
        if done % 25 == 0:
            print(f"  {done}/{len(hold)} tiles")

    # evaluate.py filters tiles by region, where region = tile id minus its last field.
    # Every tile in this directory is holdout by construction, so every region present
    # maps to 'gdc' and the filter is a no-op rather than a second, redundant split rule.
    regions = sorted({p.split("/")[-1].rsplit("_", 1)[0] for p in hold})
    (g / "split.json").write_text(json.dumps({r: "gdc" for r in regions}, indent=2))
    print(f"wrote {done} tiles and split.json ({len(regions)} regions) to {g}")
    if TMP.exists():
        shutil.rmtree(TMP, ignore_errors=True)


# -------------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("probe", help="measure GAMUS and verify the class mapping")
    p.add_argument("--sample", type=int, default=6, help="tiles per city")
    p.add_argument("--seed", type=int, default=1337)
    p.set_defaults(func=cmd_probe)

    s = sub.add_parser("shard", help="stream download -> crop -> delete")
    s.add_argument("--crop", type=int, default=518)
    s.add_argument("--stride", type=int, default=506)
    s.add_argument("--per-shard", type=int, default=128)
    s.add_argument("--max-void", type=float, default=0.30)
    s.add_argument("--holdout-col", type=int, default=50,
                   help="DC grid columns >= this are held out; col-1 is dropped as a buffer")
    s.add_argument("--holdout-tiles", type=int, default=0,
                   help="cap on held-out tiles (0 = all); the frozen DFC2019 val set is 80")
    s.add_argument("--pilot", type=int, default=0,
                   help="only this many training tiles -- the cheap distribution-shift test")
    s.add_argument("--min-free-gb", type=float, default=6.0)
    s.add_argument("--workers", type=int, default=16,
                   help="concurrent downloads. Measured: 1 stream ~0.9 MB/s, 12 workers "
                        "12 tiles/min, 32 workers 21.5 tiles/min. 32 is not the default "
                        "because this machine has 15.9 GB of RAM with ~1 GB typically "
                        "free, and 64 tiles in flight is a lot of transient array")
    s.add_argument("--with-cls", action="store_true",
                   help="also fetch class rasters. Off by default: training never reads "
                        "them, and they are 36%% of the bytes per tile")
    s.add_argument("--seed", type=int, default=1337)
    s.set_defaults(func=cmd_shard)

    h = sub.add_parser("holdout", help="download the held-out DC block as whole GeoTIFFs")
    h.add_argument("--holdout-col", type=int, default=50)
    h.add_argument("--holdout-tiles", type=int, default=0)
    h.add_argument("--min-free-gb", type=float, default=6.0)
    h.set_defaults(func=cmd_holdout)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
