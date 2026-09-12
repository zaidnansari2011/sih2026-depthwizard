"""Download real icon artwork and render it to PNG for the SIH deck.

Two sources, both permissively licensed and both usable in a submitted deck:
  Lucide       (ISC)  line icons, stroke-based, recoloured per call
  Simple Icons (CC0)  brand marks, filled paths, recoloured to each brand's hex

Rendering goes through svglib rather than cairosvg because cairo's native DLL is not
installed on this machine and svglib is pure Python.

    python tools/ppt/fetch_icons.py
"""
from __future__ import annotations

import re
from pathlib import Path

import requests
import pymupdf

OUT = Path("D:/sih2026/depthwizard/tools/ppt/icons")
LUCIDE = "https://cdn.jsdelivr.net/npm/lucide-static@latest/icons/{}.svg"
SIMPLE = "https://cdn.jsdelivr.net/npm/simple-icons@v13/icons/{}.svg"
PX = 256          # render size; the deck places them at ~0.3 in, so this is ~850 dpi

# Palette. Navy for structure, blue for flow, cyan for data, amber for our own results.
NAVY, BLUE, CYAN, AMBER, GREEN, RED = (
    "#0F2C52", "#1B6CA8", "#0E9AA7", "#E8710A", "#1E8E5A", "#C0392B")

LINE_ICONS = {
    # slide 2 - proposed solution
    "satellite": BLUE, "mountain-snow": BLUE, "box": BLUE, "gauge": BLUE,
    "ruler": BLUE, "wifi-off": BLUE, "image-up": BLUE, "layers": BLUE,
    # slide 3 - technical approach
    "cpu": NAVY, "workflow": NAVY, "database": NAVY, "monitor": NAVY,
    # slide 4 - feasibility
    "circle-check-big": GREEN, "triangle-alert": AMBER, "shield-check": GREEN,
    "trending-up": GREEN, "flask-conical": BLUE,
    # slide 5 - impact
    "building-2": BLUE, "globe": BLUE, "indian-rupee": GREEN, "users": BLUE,
    "leaf": GREEN, "siren": RED, "radio-tower": BLUE, "map-pinned": BLUE,
    # slide 6 - references
    "book-open": NAVY, "file-text": NAVY, "link": NAVY,
}

# Brand marks, each in its own official colour.
BRAND_ICONS = {
    "pytorch": "#EE4C2C", "onnx": "#005CED", "python": "#3776AB",
    "threedotjs": "#000000", "react": "#61DAFB", "numpy": "#4DABCF",
    "huggingface": "#FFD21E", "fastapi": "#009688", "docker": "#2496ED",
    "opencv": "#5C3EE8", "scipy": "#8CAAE6", "gdal": None,   # no brand mark
}


def recolour(svg: str, hex_colour: str) -> str:
    """Force a single colour. Lucide paints with currentColor on stroke; Simple Icons
    fill a black path. Handle both rather than guessing which we were given."""
    svg = svg.replace('stroke="currentColor"', f'stroke="{hex_colour}"')
    svg = re.sub(r'fill="(?!none)[^"]*"', f'fill="{hex_colour}"', svg)
    if "stroke=" not in svg and "fill=" not in svg:
        svg = svg.replace("<svg", f'<svg fill="{hex_colour}"', 1)
    # Lucide's default 24px stroke reads thin once scaled down into a card.
    return svg.replace('stroke-width="2"', 'stroke-width="2.25"')


def render(svg: str, path: Path) -> bool:
    """SVG -> transparent PNG.

    reportlab's renderPM backend wants rlPyCairo, and cairo's native DLL is not on this
    machine, so rasterising goes through PyMuPDF, which reads SVG directly and keeps an
    alpha channel. Scale is set by matrix rather than dpi because these icons declare a
    24-unit viewBox and a dpi figure would silently produce 200 px art.
    """
    try:
        doc = pymupdf.open(stream=svg.encode("utf8"), filetype="svg")
        page = doc[0]
        z = PX / max(page.rect.width, page.rect.height)
        page.get_pixmap(matrix=pymupdf.Matrix(z, z), alpha=True).save(str(path))
        return True
    except Exception as exc:
        print(f"    render failed: {type(exc).__name__}: {str(exc)[:70]}")
        return False


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ok, bad = [], []
    jobs = ([(n, c, LUCIDE) for n, c in LINE_ICONS.items()]
            + [(n, c, SIMPLE) for n, c in BRAND_ICONS.items() if c])
    for name, colour, url in jobs:
        dest = OUT / f"{name}.png"
        if dest.exists():
            ok.append(name)
            continue
        try:
            r = requests.get(url.format(name), timeout=20)
            if r.status_code != 200:
                bad.append(f"{name} (HTTP {r.status_code})")
                continue
            (ok if render(recolour(r.text, colour), dest) else bad).append(name)
        except Exception as exc:
            bad.append(f"{name} ({type(exc).__name__})")
    print(f"  rendered {len(ok)} icons -> {OUT}")
    if bad:
        print(f"  FAILED {len(bad)}: {', '.join(bad)}")


if __name__ == "__main__":
    main()
