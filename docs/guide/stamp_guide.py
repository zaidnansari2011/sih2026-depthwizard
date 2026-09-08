# -*- coding: utf-8 -*-
"""Chrome cannot template its own footer from the CLI, so the running foot is
stamped on afterwards. Page 1 is the cover and gets nothing."""
import fitz, pathlib

SRC = pathlib.Path(__file__).parent / "_raw.pdf"
DST = pathlib.Path(r"D:\sih2026\depthwizard\docs\DepthWizard-Deck-Guide-and-Defence-Notes.pdf")

INK3 = (0.43, 0.49, 0.53)
RULE = (0.84, 0.87, 0.89)

doc = fitz.open(SRC)
n = doc.page_count
for i, page in enumerate(doc, 1):
    if i == 1:
        continue
    w, h = page.rect.width, page.rect.height
    y = h - 26
    page.draw_line(fitz.Point(42, y - 11), fitz.Point(w - 42, y - 11),
                   color=RULE, width=0.5)
    page.insert_text(fitz.Point(42, y), "DepthWizard  ·  SIH26175  ·  Team Dev Up (ID 47)",
                     fontname="helv", fontsize=7.5, color=INK3)
    label = "%d / %d" % (i, n)
    tw = fitz.get_text_length(label, fontname="hebo", fontsize=7.5)
    page.insert_text(fitz.Point(w - 42 - tw, y), label,
                     fontname="hebo", fontsize=7.5, color=INK3)

doc.set_metadata({
    "title": "DepthWizard - Deck Guide and Defence Notes",
    "author": "Team Dev Up (Team ID 47)",
    "subject": "SIH 2026 - SIH26175 - slide-by-slide deck explanation and 43 judge questions",
    "keywords": "SIH2026, SIH26175, DepthWizard, ISRO, DSM, monocular depth",
})
doc.save(DST, deflate=True, garbage=3)
print("wrote", DST, DST.stat().st_size if DST.exists() else 0, "bytes,", n, "pages")
