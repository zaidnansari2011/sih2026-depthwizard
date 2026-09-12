"""Export the deck to PDF, which is the only format the SIH portal accepts.

Uses the installed PowerPoint through COM rather than a converter, so what is submitted
is exactly what PowerPoint renders -- no font substitution surprises between our preview
and the judges' copy. Also rasterises each page so the layout can be reviewed.

    python tools/ppt/export_pdf.py          # deck A, the editorial deck
    python tools/ppt/export_pdf.py B        # deck B, the poster variant
"""
from __future__ import annotations

import sys
from pathlib import Path

import pymupdf
import win32com.client

DOCS = Path("D:/sih2026/depthwizard/docs")
STEM = "SIH2026-DevUp-SIH26175-DepthWizard"
PREVIEW_ROOT = Path("D:/sih2026/depthwizard/tools/ppt")

PP_SAVE_AS_PDF = 32


def main() -> int:
    # Deck B is a second variant, not a replacement, so it exports to its own file and
    # its own preview folder; neither run can overwrite the other's output.
    variant = (sys.argv[1].upper() if len(sys.argv) > 1 else "")
    suffix = f"-{variant}" if variant else ""
    PPTX = DOCS / f"{STEM}{suffix}.pptx"
    PDF = PPTX.with_suffix(".pdf")
    PREVIEW = PREVIEW_ROOT / (f"preview{suffix.lower()}" if variant else "preview")

    if not PPTX.exists():
        print(f"  no deck at {PPTX}; run make_deck{suffix.lower()}.py first")
        return 1

    app = win32com.client.Dispatch("PowerPoint.Application")
    deck = None
    try:
        # WithWindow=False keeps the export from stealing focus mid-session.
        deck = app.Presentations.Open(str(PPTX), WithWindow=False)
        deck.SaveAs(str(PDF), PP_SAVE_AS_PDF)
    finally:
        if deck is not None:
            deck.Close()
        app.Quit()

    doc = pymupdf.open(str(PDF))
    PREVIEW.mkdir(parents=True, exist_ok=True)
    for i, page in enumerate(doc):
        page.get_pixmap(dpi=110).save(str(PREVIEW / f"slide{i + 1}.png"))
    print(f"  PDF   {PDF}  ({PDF.stat().st_size / 1024:.0f} KB, {len(doc)} pages)")
    print(f"  PNGs  {PREVIEW}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
