"""Which display faces does PowerPoint's PDF export actually embed?

Deck B was written in Bahnschrift because it is a technical grotesque that ships with
Windows and looks nothing like deck A's Calibri. The .pptx asked for it, the preview
looked plausible, and the exported PDF contained no Bahnschrift at all: every run had
been silently substituted with Calibri. Bahnschrift is a variable font, and the export
path will not embed one.

Nothing about that is visible in the render, so it needs measuring rather than eyeballing:
this sets one line in each candidate face, exports through the same COM path the deck
uses, and reads back the font each span was actually drawn in.

    python tools/ppt/probe_fonts.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pymupdf
import win32com.client
from pptx import Presentation
from pptx.util import Inches, Pt

TMP = Path("D:/sih2026/depthwizard/tools/ppt/_fontprobe")
PP_SAVE_AS_PDF = 32

CANDIDATES = [
    "Bahnschrift SemiBold",             # what deck B asked for
    "Bahnschrift SemiBold Condensed",
    "Bahnschrift Condensed",
    "Franklin Gothic Demi Cond",
    "Franklin Gothic Demi",
    "Franklin Gothic Medium Cond",
    "Segoe UI Black",
    "Segoe UI Semibold",
    "Corbel",
    "Candara",
    "Trebuchet MS",
]


def main() -> int:
    TMP.mkdir(parents=True, exist_ok=True)
    pptx, pdf = TMP / "fonts.pptx", TMP / "fonts.pdf"

    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    for i, name in enumerate(CANDIDATES):
        tb = slide.shapes.add_textbox(Inches(0.4), Inches(0.3 + i * 0.62),
                                      Inches(12.5), Inches(0.5))
        r = tb.text_frame.paragraphs[0].add_run()
        # A marker per line, so each span can be tied back to what it asked for.
        r.text = f"L{i:02d} ARCHITECTURE 3.62 m"
        r.font.name = name
        r.font.size = Pt(20)
    prs.save(str(pptx))

    app = win32com.client.Dispatch("PowerPoint.Application")
    deck = None
    try:
        deck = app.Presentations.Open(str(pptx), WithWindow=False)
        deck.SaveAs(str(pdf), PP_SAVE_AS_PDF)
    finally:
        if deck is not None:
            deck.Close()
        app.Quit()

    got = {}
    for b in pymupdf.open(str(pdf))[0].get_text("dict")["blocks"]:
        for line in b.get("lines", []):
            for sp in line["spans"]:
                t = sp["text"].strip()
                if t.startswith("L") and t[1:3].isdigit():
                    got[int(t[1:3])] = sp["font"]

    ok = 0
    for i, name in enumerate(CANDIDATES):
        actual = got.get(i, "(not found)")
        # PDF names drop spaces and add a subset prefix, so compare on letters only.
        kept = actual.split("+")[-1].replace("-", "").lower().startswith(
            name.replace(" ", "").lower()[:10])
        ok += kept
        print(f"  {'KEPT     ' if kept else 'SUBSTITUTED'} {name:<32} -> {actual}")
    print(f"\n  {ok}/{len(CANDIDATES)} faces survive the export")
    return 0


if __name__ == "__main__":
    sys.exit(main())
