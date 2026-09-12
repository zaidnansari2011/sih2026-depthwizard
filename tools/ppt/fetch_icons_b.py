"""Icon artwork for deck B, which needs colours deck A never did.

Deck A sets dark icons on white. Deck B sets them on filled navy badges, so the same
Lucide glyphs are re-rendered in white into icons/white/. It also uses a comparison
table, which needs tick / cross / caution marks in green, red and amber; those go into
icons/marks/.

Same source and licence as fetch_icons.py (Lucide, ISC), same PyMuPDF rasteriser, and
the same reason for it: cairo's native DLL is not installed on this machine.

    python tools/ppt/fetch_icons_b.py
"""
from __future__ import annotations

from pathlib import Path

import requests

from fetch_icons import LUCIDE, recolour, render

OUT = Path("D:/sih2026/depthwizard/tools/ppt/icons")
WHITE_DIR = OUT / "white"
MARK_DIR = OUT / "marks"

WHITE = "#FFFFFF"

# Every glyph deck B sets on a filled badge.
WHITE_ICONS = [
    # slide 2, the solution grid
    "image-up", "shield-check", "box", "ruler", "map-pinned", "wifi-off",
    # slide 3, the pipeline spine and the validation block
    "cpu", "layers", "workflow", "monitor", "database", "gauge", "flask-conical",
    # slide 4, risks
    "triangle-alert", "circle-check-big", "trending-up",
    # slide 5, who it serves
    "satellite", "building-2", "siren", "radio-tower", "leaf", "users", "globe",
    "indian-rupee", "mountain-snow",
    # slide 6
    "book-open", "file-text", "link", "search", "scale",
]

# Comparison-table marks. Amber is a real verdict here, not decoration: it means the
# capability exists but is partial or conditional, and several rows genuinely are.
MARKS = {"check": "#1E8E5A", "x": "#C0392B", "minus": "#C08A2E"}


def fetch(name: str, colour: str, dest: Path) -> bool:
    if dest.exists():
        return True
    r = requests.get(LUCIDE.format(name), timeout=20)
    if r.status_code != 200:
        print(f"    {name}: HTTP {r.status_code}")
        return False
    # Marks sit small in table cells, so they need a heavier stroke than a card icon.
    svg = recolour(r.text, colour)
    if dest.parent == MARK_DIR:
        svg = svg.replace('stroke-width="2.25"', 'stroke-width="3.2"')
    return render(svg, dest)


def main() -> None:
    ok = bad = 0
    for folder, jobs in ((WHITE_DIR, [(n, WHITE) for n in WHITE_ICONS]),
                         (MARK_DIR, list(MARKS.items()))):
        folder.mkdir(parents=True, exist_ok=True)
        for name, colour in jobs:
            if fetch(name, colour, folder / f"{name}.png"):
                ok += 1
            else:
                bad += 1
                print(f"  FAILED {name}")
    print(f"  {ok} icons ready ({WHITE_DIR}, {MARK_DIR})" + (f", {bad} failed" if bad else ""))


if __name__ == "__main__":
    main()
