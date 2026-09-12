"""Flag deck-B shapes that leave the safe area, before anyone has to squint at a render.

The template owns three strips a slide must not intrude on: the blue footer bar,
slimmed by trim_footer so content may now run to y 7.16, the SIH mark at x 10.70-13.16 above y 1.16, and the page margins. Text overflow
cannot be caught this way -- a text frame's declared height says nothing about how many
lines it grew to -- so this checks geometry only, and the rendered PDF still gets looked
at. It catches the whole class of mistake that put a figure under the footer bar.

    python tools/ppt/check_layout_b.py          # deck B
    python tools/ppt/check_layout_b.py C        # deck C
"""
from __future__ import annotations

import sys
from pathlib import Path

from pptx import Presentation

DOCS = Path("D:/sih2026/depthwizard/docs")
STEM = "SIH2026-DevUp-SIH26175-DepthWizard"

E = 914400.0
FOOTER_TOP = 7.16      # build_deck.trim_footer slims the bar to 0.28 in
SLIDE_W, SLIDE_H = 13.333, 7.5
LOGO = (10.70, 0.0, 13.17, 1.17)          # left, top, right, bottom
TOL = 0.02


def main() -> int:
    variant = sys.argv[1].upper() if len(sys.argv) > 1 else "B"
    DST = DOCS / f"{STEM}-{variant}.pptx"
    if not Path(DST).exists():
        print(f"  no deck at {DST}; run make_deck_{variant.lower()}.py first")
        return 1
    prs = Presentation(str(DST))
    bad = 0
    for i, s in enumerate(prs.slides, 1):
        for sh in s.shapes:
            if sh.left is None or sh.width is None:
                continue
            l, t = sh.left / E, sh.top / E
            r, b = l + sh.width / E, t + sh.height / E
            # The template's own furniture lives in these strips by design: the title
            # placeholders are deliberately taller than their text and sit partly off
            # the top edge, and the footer bar is the footer bar.
            if sh.name.startswith(("Rectangle", "Slide Number", "Footer", "Picture",
                                   "Freeform", "Title", "Subtitle", "Oval")):
                continue
            why = []
            if b > FOOTER_TOP + TOL:
                why.append(f"bottom {b:.2f} is under the footer bar ({FOOTER_TOP})")
            if r > SLIDE_W + TOL or l < -TOL:
                why.append(f"x spans {l:.2f}..{r:.2f}, slide is 0..{SLIDE_W}")
            if b > SLIDE_H + TOL or t < -TOL:
                why.append(f"y spans {t:.2f}..{b:.2f}, slide is 0..{SLIDE_H}")
            if (l < LOGO[2] and r > LOGO[0] and t < LOGO[3] and b > LOGO[1]
                    and t < 1.16):
                why.append("overlaps the SIH mark at top right")
            if why:
                bad += 1
                print(f"  slide {i}  {sh.name:<28} {l:5.2f},{t:5.2f} "
                      f"{r - l:5.2f}x{b - t:5.2f}  -- {'; '.join(why)}")
    print(f"\n  {'CLEAN' if not bad else f'{bad} shape(s) out of bounds'}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
