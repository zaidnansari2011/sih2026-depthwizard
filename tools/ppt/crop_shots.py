"""Crop the viewer captures into a consistent band for the deck.

The raw captures are 3200x2000 with a control panel on the left and a stats panel on the
right. Both panels are worth showing once, at readable size, but three full captures side
by side would render the text as illegible grey. So each is cropped to the 3D viewport at
a common 2.1:1 aspect, and the panels are shown separately only where there is room.

Crop boxes are in source pixels and were chosen by looking at the captures.

    python tools/ppt/crop_shots.py
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image

SHOTS = Path("D:/sih2026/depthwizard/tools/ppt/shots")
DOCS = Path("D:/sih2026/depthwizard/docs")
ASPECT = 2.1

# Hand-composed captures that are not 3200x2000 viewer screenshots. These are cropped to
# ASPECT at the vertical offset that leaves the least empty sky, rather than by a fixed
# box, because their framing is chosen by eye and will not be reproducible.
# name -> (source, anchor). "top" keeps the top of the frame and trims the bottom;
# "auto" picks the offset with the least sky. Auto was the first choice for the Sikkim
# capture and was wrong: minimising sky trimmed the ridgeline, which is the whole point
# of the shot. Sky against a white slide reads as sky; a decapitated ridge reads as a
# mistake.
SUPPLIED = {"sikkim_hilly": (DOCS / "sikkimhilly.png", "top")}

# name -> (left, top, right, bottom) in the 3200x2000 capture, clear of both HUD panels
BOXES = {
    # The valley renders as a diagonal ridge with empty grey below it, so this takes the
    # ridge band only. Left edges clear the control panel, whose right edge is at x 585;
    # right edges stop short of the stats panel, which starts at x 2688.
    "sikkim_valley": (1080, 70, 2650, 818),
    "sikkim_town": (620, 60, 2660, 1560),
    "urban_height": (640, 380, 2680, 1780),
    "urban_sigma": (640, 380, 2680, 1780),
}


def to_aspect(box, aspect=ASPECT):
    """Grow or shrink the box vertically about its centre to hit the target aspect."""
    l, t, r, b = box
    w = r - l
    h = w / aspect
    cy = (t + b) / 2
    return (l, int(cy - h / 2), r, int(cy + h / 2))


def fit_supplied(src, dest, anchor="top", aspect=ASPECT, step=4):
    """Crop a supplied image to `aspect`. See SUPPLIED for what `anchor` means."""
    import numpy as np
    im = Image.open(src).convert("RGB")
    a = np.asarray(im)
    h = int(round(im.width / aspect))
    if h >= im.height:
        raise ValueError(f"{src.name} is {im.width}x{im.height}; too short for {aspect}:1")
    if anchor == "top":
        top = 0
    else:
        top = min(range(0, im.height - h + 1, step),
                  key=lambda t: float((a[t:t + h].min(axis=2) > 235).mean()))
    white = float((a[top:top + h].min(axis=2) > 235).mean())
    im.crop((0, top, im.width, top + h)).save(dest, optimize=True)
    return top, white


def main() -> None:
    for name, box in BOXES.items():
        src = SHOTS / f"{name}.png"
        if not src.exists():
            print(f"  missing {src.name}; run capture_viewer.py first")
            continue
        im = Image.open(src).convert("RGB")
        l, t, r, b = to_aspect(box)
        # Never crop outside the capture.
        t, b = max(0, t), min(im.height, b)
        r = min(im.width, r)
        out = im.crop((l, t, r, b))
        dest = SHOTS / f"{name}_crop.png"
        out.save(dest, optimize=True)
        print(f"  {dest.name:<26} {out.width}x{out.height} "
              f"({out.width / out.height:.2f}:1)  {dest.stat().st_size / 1024:.0f} KB")

    for name, (src, anchor) in SUPPLIED.items():
        if not src.exists():
            print(f"  missing supplied image {src}")
            continue
        dest = SHOTS / f"{name}_crop.png"
        top, white = fit_supplied(src, dest, anchor)
        im = Image.open(dest)
        print(f"  {dest.name:<26} {im.width}x{im.height} "
              f"({im.width / im.height:.2f}:1)  {dest.stat().st_size / 1024:.0f} KB  "
              f"[supplied, top={top}, sky {white * 100:.1f}%]")


if __name__ == "__main__":
    main()
