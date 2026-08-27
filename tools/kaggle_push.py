"""Push the training shards to Kaggle as a private dataset.

Only train and val go up. The test split stays on the local machine -- standing rule 5 says
never tune against it, and the surest way to keep that promise is for it not to exist on
the box where training happens.

Kaggle limits that shape how this is used: a notebook session is capped at 12 hours and the
weekly GPU budget is 30 hours, so the training script must checkpoint and resume rather
than assume it can run to completion in one sitting.

    python tools/kaggle_push.py --create      # first time: makes the dataset
    python tools/kaggle_push.py --version     # later: pushes an updated copy
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path("D:/sih2026")
SHARDS = ROOT / "data" / "shards"
STAGE = ROOT / "data" / "kaggle_stage"
SLUG = "depthwizard-dfc2019-shards"
TITLE = "DepthWizard DFC2019 shards (train+val)"


def stage(force: bool = False):
    """Hardlink the shards we want into a clean directory for upload.

    Hardlinks, not copies: 9 GB duplicated on a drive with 30 GB free is a bad trade, and
    the Kaggle client only needs to read the files.
    """
    STAGE.mkdir(parents=True, exist_ok=True)
    wanted = sorted(SHARDS.glob("train_*.npz")) + sorted(SHARDS.glob("val_*.npz"))
    wanted += [SHARDS / n for n in ("split.json", "scene_geometry.json", "probe.json")
               if (SHARDS / n).exists()]
    n_link, n_skip, total = 0, 0, 0
    for src in wanted:
        dst = STAGE / src.name
        total += src.stat().st_size
        if dst.exists():
            n_skip += 1
            continue
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)
        n_link += 1
    leaked = [p.name for p in STAGE.glob("test_*")]
    if leaked:
        raise SystemExit(f"test shards found in the staging dir: {leaked}. "
                         "They must never be uploaded (standing rule 5).")
    print(f"staged {len(wanted)} files ({total/1e9:.2f} GB) -> {STAGE}")
    print(f"  {n_link} linked, {n_skip} already present, 0 test shards")
    return total


CODE_STAGE = ROOT / "data" / "kaggle_code"
CODE_SLUG = "depthwizard-code"


def stage_code():
    """Copy the source the notebook needs. Small, so a plain copy is fine.

    Deliberately excludes checkpoints, data, logs and .git -- a code dataset that carries a
    900 MB checkpoint is a code dataset nobody re-uploads when the code changes.
    """
    src = Path(__file__).resolve().parents[1]
    if CODE_STAGE.exists():
        shutil.rmtree(CODE_STAGE)
    CODE_STAGE.mkdir(parents=True)

    # One archive, not a tree. The Kaggle client silently SKIPS subdirectories unless
    # --dir-mode is set ("Skipping folder: tools"), which would have shipped train.py
    # without the depthwizard package it imports and failed on Kaggle rather than here.
    import zipfile
    zpath = CODE_STAGE / "depthwizard_src.zip"
    n = 0
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for f in ("train.py", "infer.py"):
            z.write(src / f, f)
            n += 1
        for pkg in ("depthwizard", "tools"):
            for p in sorted((src / pkg).rglob("*")):
                if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc":
                    z.write(p, str(p.relative_to(src)))
                    n += 1
    print(f"zipped {n} source files -> {zpath.name} ({zpath.stat().st_size/1e6:.2f} MB)")
    (CODE_STAGE / "dataset-metadata.json").write_text(json.dumps({
        "title": "DepthWizard code",
        "id": f"{USER}/{CODE_SLUG}",
        "licenses": [{"name": "other"}],
        "description": "Training and evaluation source for DepthWizard (SIH 2026). "
                       "Attach alongside depthwizard-dfc2019-shards.",
    }, indent=2))
    print(f"staged -> {CODE_STAGE}")


def write_metadata():
    meta = {
        "title": TITLE,
        "id": f"{USER}/{SLUG}",
        "licenses": [{"name": "other"}],
        "description": (
            "Preprocessed 518x518 crops from the IEEE GRSS DFC2019 Track 1 (US3D) dataset, "
            "Jacksonville and Omaha. Each .npz holds 128 crops: rgb uint8, agl float16 "
            "(above-ground height in metres), cls uint8 (semantic class), tile id.\n\n"
            "Region-disjoint split, seed 1337. The held-out test split is deliberately "
            "NOT included. Private dataset for a Smart India Hackathon 2026 entry; "
            "underlying data is IEEE GRSS DFC2019, redistributed under its own terms."
        ),
    }
    (STAGE / "dataset-metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {STAGE / 'dataset-metadata.json'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--code", action="store_true",
                    help="push the source tree instead of the shards, as a second small "
                         "dataset the notebook imports from")
    ap.add_argument("--create", action="store_true", help="first upload")
    ap.add_argument("--version", action="store_true", help="update an existing dataset")
    ap.add_argument("--notes", default="update")
    ap.add_argument("--stage-only", action="store_true")
    a = ap.parse_args()

    os.environ.setdefault("KAGGLE_CONFIG_DIR", r"C:\Users\Zaid\.kaggle")
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    global USER
    USER = api.config_values.get("username")
    print(f"authenticated as {USER}")

    if a.code:
        stage_code()
    else:
        stage()
        write_metadata()
    if a.stage_only:
        return

    target = CODE_STAGE if a.code else STAGE
    slug = CODE_SLUG if a.code else SLUG
    if a.create:
        if not a.code:
            print("creating dataset and uploading -- this is 9 GB, expect a long wait")
        api.dataset_create_new(str(target), public=False, dir_mode="skip",
                               quiet=False)
    elif a.version:
        api.dataset_create_version(str(target), version_notes=a.notes,
                                   dir_mode="skip", quiet=False)
    else:
        print("nothing to do; pass --create or --version")
        return
    print(f"\ndone -> https://www.kaggle.com/datasets/{USER}/{slug}")


USER = ""
if __name__ == "__main__":
    main()
