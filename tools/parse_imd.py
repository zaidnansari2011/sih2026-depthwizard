"""Parse WorldView .IMD metadata into JSON -- solar and viewing geometry.

Feeds differentiator 6.2 (shadow prior) and the domain-gap analysis in 7.1.

    python tools/parse_imd.py D:\\Downloads\\Track3-Metadata.zip
    python tools/parse_imd.py D:\\sih2026\\data\\extracted\\Track3-Metadata

Why each field matters:

  meanSunEl   sun elevation, degrees. Shadow length for height h is h / tan(sunEl).
              Low sun -> long, measurable shadows. High sun -> shadows vanish and the
              prior carries almost no information. In DFC2019 this ranges 23-75 deg,
              so the prior's reliability is scene-dependent and we must weight it.
  meanSunAz   sun azimuth, degrees clockwise from north. Gives the direction to search
              for a shadow from each structure.
  meanSatEl / meanOffNadirViewAngle
              viewing geometry. Off-nadir look introduces building lean and occlusion,
              and is the closest proxy we have for Cartosat-vs-aerial domain shift.
"""
from __future__ import annotations

import json
import math
import re
import sys
import zipfile
from pathlib import Path

FIELDS = {
    "satId": str, "CatId": str, "firstLineTime": str,
    "meanSunAz": float, "meanSunEl": float,
    "meanSatAz": float, "meanSatEl": float,
    "meanInTrackViewAngle": float, "meanCrossTrackViewAngle": float,
    "meanOffNadirViewAngle": float,
}


def parse_imd(text: str) -> dict:
    out = {}
    for key, cast in FIELDS.items():
        m = re.search(rf'\b{key}\s*=\s*"?([^";\n]+)"?\s*;', text)
        if m:
            raw = m.group(1).strip()
            try:
                out[key] = cast(raw)
            except ValueError:
                out[key] = raw
    if "meanSunEl" in out:
        el = math.radians(out["meanSunEl"])
        # Metres of shadow cast per metre of height. Large = strong signal.
        out["shadow_len_per_m"] = round(1.0 / math.tan(el), 4) if el > 0 else None
        # Heuristic usefulness band for the shadow prior.
        out["shadow_quality"] = (
            "strong" if out["meanSunEl"] < 35 else
            "usable" if out["meanSunEl"] < 55 else
            "weak"
        )
    return out


def collect(src: Path) -> dict[str, dict]:
    scenes = {}
    if src.suffix.lower() == ".zip":
        with zipfile.ZipFile(src) as zf:
            for n in zf.namelist():
                if n.upper().endswith(".IMD"):
                    scenes[n.split("/", 1)[-1]] = parse_imd(zf.read(n).decode("utf-8", "replace"))
    else:
        for p in sorted(src.rglob("*.IMD")):
            scenes[str(p.relative_to(src))] = parse_imd(p.read_text(errors="replace"))
    return scenes


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    src = Path(sys.argv[1])
    if not src.exists():
        sys.exit(f"not found: {src}")

    scenes = collect(src)
    if not scenes:
        sys.exit("no .IMD files found")

    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("data/shards/scene_geometry.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(scenes, indent=2, sort_keys=True))

    els = sorted(s["meanSunEl"] for s in scenes.values() if "meanSunEl" in s)
    bands = {"strong": 0, "usable": 0, "weak": 0}
    for s in scenes.values():
        if q := s.get("shadow_quality"):
            bands[q] += 1

    print(f"{len(scenes)} scenes -> {out}")
    if els:
        print(f"sun elevation : {els[0]:.1f} to {els[-1]:.1f} deg  (median {els[len(els)//2]:.1f})")
        print(f"shadow signal : {bands['strong']} strong (<35 deg), "
              f"{bands['usable']} usable (35-55), {bands['weak']} weak (>55)")
        print("\nA 20 m building casts:")
        for el in (els[0], els[len(els)//2], els[-1]):
            print(f"  {20/math.tan(math.radians(el)):6.1f} m of shadow at sun elevation {el:.1f} deg")


if __name__ == "__main__":
    main()
