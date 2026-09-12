"""Rebuild the whole viewer scene set, reproducibly.

    python tools/build_scenes.py            # all six
    python tools/build_scenes.py --only mixed_jax_020_020

`viewer/scenes/` is gitignored -- the bundles are large binaries and the .gitignore says
"regenerate with export_terrain". That instruction was not enough on its own: the display
names, the model provenance string, which scene the viewer opens on, and the two Sikkim
reference notes were all supplied by hand on the command line and existed nowhere in the
repo. A regeneration silently produced a different, worse scene set. This file is that
missing information.

Two things it handles that a shell loop does not:

* **The reference notes.** The Sikkim scenes have no LiDAR, so their honesty depends on a
  paragraph of prose comparing us to Google Open Buildings. `export_terrain.py` has no
  flag for it -- it was pasted into the manifests afterwards -- so it is carried here and
  re-applied after every export.
* **Names with an em-dash.** Passed through subprocess with a list argv, which Windows
  hands to the child as UTF-16. Do not move these into a shell script.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO.parent                      # D:/sih2026 -- where out/ and data/ live
PY = sys.executable
SCENES = REPO / "viewer" / "scenes"
EXPORT = REPO / "tools" / "export_terrain.py"
BAKE = REPO / "tools" / "bake_buildings.py"

# The model behind every scene here. Change these in ONE place when the shipping model
# changes, and re-run, or the viewer will confidently name the wrong checkpoint.
#
# RUN also selects the rasters, because the two must never disagree: the run02 scenes were
# built from untagged `out/scene_<tile>.height.tif` paths, so pointing MODEL at a new
# checkpoint without regenerating would have relabelled the old surfaces rather than
# replaced them. Tagging the path makes that failure impossible instead of merely unlikely.
RUN = "run07"
MODEL = "run07 + TTA + zoom-2 fusion"

DFC_GSD = "0.3"

# DFC2019 scenes: scored against LiDAR, so they carry per-building metrics.
#   dir, tile, display name, terrain label, place, is landing scene
DFC = [
    ("urban_oma_288_042", "OMA_288_042", "Urban \u2014 Omaha", "urban", "Omaha", False),
    ("sparse_oma_258_005", "OMA_258_005", "Sparse \u2014 Omaha", "sparse", "Omaha", False),
    ("forested_jax_203_010", "JAX_203_010", "Forested \u2014 Jacksonville",
     "forested", "Jacksonville", False),
    # The landing scene. 54 buildings at 2.16 m per-building RMSE, and "mixed" is the
    # least cherry-picked label available. The picker's ORDER still opens with urban --
    # it mirrors the problem statement's "urban, sparse, hilly and forested" -- but urban
    # here is downtown Omaha: seven buildings, one of them 93 m, against training data
    # that stops at 83 m. It is our worst case and it should not be the first impression.
    ("mixed_jax_020_020", "JAX_020_020", "Mixed \u2014 Jacksonville",
     "mixed", "Jacksonville", True),
]

# Sikkim: real Indian terrain, no LiDAR anywhere near it. Nothing is scored; the manifest
# note says so in as many words.
GLO30 = "Copernicus GLO-30, 30 m posts, cubic-resampled"
OB = "Google Open Buildings 2.5D Temporal (2022)"
SIKKIM = [
    {
        "dir": "hilly_sikkim_valley",
        "height": f"out/sikkim_2000m_{RUN}.height.tif",
        "sigma": f"out/sikkim_2000m_{RUN}.sigma.tif",
        "terrain_base": "out/sikkim_2000m_terrain.tif",
        "texture": "data/maxar/crops/sikkim_town_2000m.tif",
        "name": "Hilly \u2014 Sikkim, India",
        "terrain": "hilly",
        "reference_note": (
            "No laser survey here. Against Google Open Buildings \u2014 a satellite "
            "estimate, not ground truth \u2014 we read 6.3 m lower on the 168 buildings "
            "it is most confident about. We under-call tall buildings, and this is that "
            "same weakness showing up over India."
        ),
    },
    {
        "dir": "hilly_sikkim_town",
        "height": f"out/sikkim_town_{RUN}.height.tif",
        "sigma": f"out/sikkim_town_{RUN}.sigma.tif",
        "terrain_base": "out/sikkim_town_terrain.tif",
        "texture": "data/maxar/crops/sikkim_town_500m.tif",
        "name": "Detail \u2014 Sikkim town",
        "terrain": "detail",
        "reference_note": (
            "No laser survey here. Across 92 buildings we read 1.2 m lower than Google "
            "Open Buildings and agree within 3.6 m \u2014 but its outlines are coarse and "
            "it is itself a satellite estimate, so this is agreement between two models, "
            "not accuracy."
        ),
    },
]


def run(cmd: list[str]) -> None:
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                       errors="replace")
    if r.returncode != 0:
        print(r.stdout)
        print(r.stderr, file=sys.stderr)
        raise SystemExit(f"export failed: {' '.join(cmd[-2:])}")


def patch(manifest_path: Path, fields: dict) -> None:
    """Merge fields into a manifest without truncating it if anything goes wrong.

    Written through a temp file deliberately: `open(p, "w")` truncates before the value
    is evaluated, and an exception in the argument leaves a zero-byte file behind.
    """
    m = json.loads(manifest_path.read_text(encoding="utf-8"))
    m.update(fields)
    tmp = manifest_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(m, indent=2, ensure_ascii=False), encoding="utf-8")
    if tmp.stat().st_size == 0:
        tmp.unlink()
        raise SystemExit(f"refusing to write an empty manifest over {manifest_path}")
    tmp.replace(manifest_path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", help="rebuild just this scene directory")
    args = ap.parse_args()

    built = []

    for dirname, tile, name, terrain, place, is_default in DFC:
        if args.only and args.only != dirname:
            continue
        cmd = [
            PY, str(EXPORT),
            "--height", str(ROOT / f"out/scene_{RUN}_{tile}.height.tif"),
            "--sigma", str(ROOT / f"out/scene_{RUN}_{tile}.sigma.tif"),
            "--truth", str(ROOT / f"data/extracted/Track1-Truth/{tile}_AGL.tif"),
            "--cls", str(ROOT / f"data/extracted/Track1-Truth/{tile}_CLS.tif"),
            "--texture", str(ROOT / f"data/extracted/Track1-RGB/{tile}_RGB.tif"),
            "--gsd", DFC_GSD, "--terrain", terrain, "--place", place,
            "--name", name, "--model", MODEL,
            "--out", str(SCENES / dirname),
        ]
        if is_default:
            cmd.append("--default-scene")
        run(cmd)
        built.append(dirname)

    for s in SIKKIM:
        if args.only and args.only != s["dir"]:
            continue
        run([
            PY, str(EXPORT),
            "--height", str(ROOT / s["height"]),
            "--sigma", str(ROOT / s["sigma"]),
            "--terrain-base", str(ROOT / s["terrain_base"]),
            "--terrain-source", GLO30,
            "--texture", str(ROOT / s["texture"]),
            "--terrain", s["terrain"], "--place", "Sikkim",
            "--name", s["name"], "--model", MODEL,
            "--out", str(SCENES / s["dir"]),
        ])
        patch(SCENES / s["dir"] / "manifest.json",
              {"reference_source": OB, "reference_note": s["reference_note"]})
        built.append(s["dir"])

    # Re-bake the inundation tool's per-building elevations, always. export_terrain.py
    # rewrites manifest["files"] from scratch, which DROPS the "buildings" entry and
    # silently disables the tool -- and even where the entry survived, the elevations
    # would be the previous checkpoint's. Found on the run07 re-bake, where
    # buildings.json kept an 8 Sep timestamp underneath a 12 Sep surface.
    if built:
        print("\nbaking building elevations")
        run([PY, str(BAKE)] + (["--scene", args.only] if args.only else []))

    for dirname in built:
        m = json.loads((SCENES / dirname / "manifest.json").read_text(encoding="utf-8"))
        bw = m.get("building_wise")
        score = (f"{bw['n_buildings']} buildings @ {bw['rmse']:.2f} m" if bw
                 else "no LiDAR, not scored")
        flood = "flood" if "buildings" in (m.get("files") or {}) else "     "
        print(f"  {dirname:24} {score:28} {flood}  model={m.get('model')}")
    print(f"\n{len(built)} scene(s) rebuilt into {SCENES}")
    print("Now: python tools/build_standalone.py && python tools/verify_viewer.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
