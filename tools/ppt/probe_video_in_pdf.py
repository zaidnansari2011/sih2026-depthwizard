"""Does a video embedded in a .pptx survive PowerPoint's PDF export?

The SIH portal accepts PDF only, so the answer decides whether recording a clip for the
deck is worth doing. Rather than repeat what the internet says, this builds a one-slide
deck with a real video in it, exports it through the installed PowerPoint exactly as
export_pdf.py does, and inspects the resulting PDF for the annotation types a playable
video would need (/RichMedia, /Screen, /Movie).

    python tools/ppt/probe_video_in_pdf.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pymupdf
import win32com.client
from pptx import Presentation
from pptx.util import Inches

TMP = Path("D:/sih2026/depthwizard/tools/ppt/_videoprobe")
PP_SAVE_AS_PDF = 32


def make_clip(path: Path, seconds=2, fps=12, w=640, h=360) -> None:
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for i in range(seconds * fps):
        frame = np.full((h, w, 3), 30, np.uint8)
        cv2.circle(frame, (int(w * i / (seconds * fps)), h // 2), 40, (0, 120, 255), -1)
        vw.write(frame)
    vw.release()


def main() -> int:
    TMP.mkdir(parents=True, exist_ok=True)
    clip, pptx, pdf = TMP / "clip.mp4", TMP / "probe.pptx", TMP / "probe.pdf"

    make_clip(clip)
    if not clip.exists() or clip.stat().st_size < 1000:
        print("  could not write a test clip; cv2 has no mp4 encoder here")
        return 1
    print(f"  clip {clip.stat().st_size / 1024:.0f} KB")

    prs = Presentation()
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    slide.shapes.add_movie(str(clip), Inches(1), Inches(1), Inches(5), Inches(2.8),
                           mime_type="video/mp4")
    prs.save(str(pptx))
    print(f"  pptx {pptx.stat().st_size / 1024:.0f} KB, "
          f"{len(slide.shapes)} shapes (movie + its poster)")

    app = win32com.client.Dispatch("PowerPoint.Application")
    deck = None
    try:
        deck = app.Presentations.Open(str(pptx), WithWindow=False)
        deck.SaveAs(str(pdf), PP_SAVE_AS_PDF)
    finally:
        if deck is not None:
            deck.Close()
        app.Quit()

    doc = pymupdf.open(str(pdf))
    page = doc[0]
    kinds = [a.type[1] for a in page.annots()] if page.annots() else []
    raw = doc.xref_object(1, compressed=False) if doc.xref_length() > 1 else ""
    blob = b"".join(doc.xref_stream_raw(i) or b"" for i in range(1, min(doc.xref_length(), 60)))
    markers = {k: (k.encode() in blob or k in raw)
               for k in ("/RichMedia", "/Screen", "/Movie")}

    print(f"\n  PDF {pdf.stat().st_size / 1024:.0f} KB, {len(doc)} page(s)")
    print(f"  annotations on page 1: {kinds or 'none'}")
    for k, v in markers.items():
        print(f"  {k:<12} present: {v}")
    playable = bool(kinds) or any(markers.values())
    print(f"\n  VERDICT: video is {'PLAYABLE' if playable else 'NOT playable'} "
          f"in the exported PDF.")
    if not playable:
        print("  PowerPoint flattens it to the poster frame. Link to a hosted clip "
              "instead -- PDF hyperlinks do work.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
