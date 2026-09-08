"""Build data/shards_union as hardlinks to the DFC2019 and GAMUS shards.

    python tools/link_union.py            # create/refresh the union directory
    python tools/link_union.py --check    # report what it would do, change nothing

Why hardlinks rather than a --shards flag that takes two directories
--------------------------------------------------------------------
train.py takes one `--shards` directory and discovers splits by globbing
`{split}_*.npz`. Teaching it about multiple corpora would mean touching the one script
whose behaviour every existing checkpoint depends on. A union *directory* gets the same
result with no code change at all, and because hardlinks share the underlying blocks it
costs **zero additional bytes** on the same volume.

That the GAMUS shards are named `train_g0000.npz` is what makes this work: `train_*.npz`
matches both corpora, so the union trains on everything, while `val_*.npz` matches only
DFC2019 -- which is exactly the intent. **The validation set stays frozen and pure
DFC2019**, so every number in the evidence pack remains comparable to the union run.

The same trick is already used for data/kaggle_stage; see the note in gamus-integration.md
about it inflating naive disk audits, because a recursive size sum counts shared blocks
once per link.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DFC = ROOT / "data" / "shards"
GAMUS = ROOT / "data" / "shards_gamus"
UNION = ROOT / "data" / "shards_union"


def _sources() -> list[Path]:
    src: list[Path] = []
    # DFC2019: train/val/test plus the metadata evaluate.py and dataset.py read.
    for pat in ("train_*.npz", "val_*.npz", "test_*.npz",
                "probe.json", "split.json", "scene_geometry.json"):
        src += sorted(DFC.glob(pat))
    # GAMUS: training shards only. The held-out DC block is never sharded -- it lives as
    # whole GeoTIFFs in data/extracted_gamus so it can be scored with the whole-tile
    # protocol. Linking it here would quietly put the holdout into training.
    src += sorted(GAMUS.glob("train_g*.npz"))
    return src


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="report only, change nothing")
    ap.add_argument("--out", default=None,
                    help="target directory (default data/shards_union). Use a separate "
                         "one to freeze a snapshot while an ingest is still running.")
    args = ap.parse_args()

    global UNION
    if args.out:
        UNION = Path(args.out)

    if not DFC.exists():
        raise SystemExit(f"missing {DFC}")
    src = _sources()
    if not any(p.name.startswith("train_g") for p in src):
        print("!! no GAMUS train_g*.npz found -- the union would be DFC2019 only.")

    UNION.mkdir(parents=True, exist_ok=True)
    made = skipped = 0
    for p in src:
        dst = UNION / p.name
        if dst.exists():
            skipped += 1
            continue
        if args.check:
            made += 1
            continue
        try:
            os.link(p, dst)          # needs no privilege on the same volume, unlike symlink
        except OSError as e:
            raise SystemExit(f"hardlink failed for {p.name}: {e}\n"
                             "Both directories must be on the same volume.")
        made += 1

    n_dfc = len([p for p in src if p.name.startswith(("train_0", "train_1", "train_2"))])
    n_gam = len([p for p in src if p.name.startswith("train_g")])
    verb = "would link" if args.check else "linked"
    print(f"{verb} {made} files ({skipped} already present) -> {UNION}")
    print(f"  train shards: {n_dfc} DFC2019 + {n_gam} GAMUS")
    print(f"  val shards:   {len(list(UNION.glob('val_*.npz')))} (DFC2019 only, frozen)")
    if not args.check:
        total = sum(f.stat().st_size for f in UNION.glob("*.npz"))
        print(f"  apparent size {total/1e9:.2f} GB, actual additional bytes on disk: 0")
        print(f"\nTrain the union with:  python train.py --shards {UNION}")


if __name__ == "__main__":
    main()
