"""Layout primitives for the SIH deck.

Design notes, because the first version looked generated and was rebuilt:

The tell was uniformity. Every element was a rounded rectangle with a pale blue fill and a
thin border, laid on an even grid, in a five-colour palette, repeating an
icon-left / bold-title / grey-description pattern down the page. Nothing was ever just
text, and nothing had a different weight from anything else.

So this version:
  - draws **rules, not boxes**. A hairline above a block separates it as well as a border
    does and adds no shape. Boxes survive only where something genuinely is a container.
  - keeps **square corners**. Rounded corners on everything is half the signature.
  - runs **three colours**: ink, one accent, one slate for secondary text. Charts add a
    single steel blue. No teal, no green, no amber-plus-orange.
  - sets **wide typographic contrast**: 28 pt light numerals against 7.5 pt caps labels,
    rather than everything between 8 and 11 pt.
  - uses **ordinals (01, 02, 03)** as the repeating device instead of icon chips.
  - **breaks the grid on purpose**: asymmetric columns, and figures that bleed past the
    text column.
  - embeds the **real matplotlib figures** from docs/figures. A slightly utilitarian plot
    of measured data is the strongest signal that the work behind it is real.
"""
from __future__ import annotations

from pathlib import Path

from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Emu, Inches, Pt

DOCS = Path("D:/sih2026/depthwizard/docs")
FIGS = DOCS / "figures"
DECK_FIGS = Path("D:/sih2026/depthwizard/tools/ppt/figures")
SHOTS = Path("D:/sih2026/depthwizard/tools/ppt/shots")
ICONS = Path("D:/sih2026/depthwizard/tools/ppt/icons")
SRC = DOCS / "SIH2026-IDEA-Presentation-Format.pptx"
DST = DOCS / "SIH2026-DevUp-SIH26175-DepthWizard.pptx"

TEAM = "Dev Up"
FONT = "Calibri"
MONO = "Consolas"

# ------------------------------------------------------------------------- palette
INK = RGBColor(0x14, 0x18, 0x1D)      # headings and body
SLATE = RGBColor(0x5B, 0x66, 0x70)    # secondary text
ACCENT = RGBColor(0xC1, 0x44, 0x0E)   # the single accent, used sparingly
STEEL = RGBColor(0x2C, 0x4A, 0x63)    # data, and only data
RULE = RGBColor(0xD3, 0xD8, 0xDD)     # hairlines
FAINT = RGBColor(0xF2, 0xF4, 0xF6)    # the one permitted fill
WHITE = RGBColor(0xFF, 0xFF, 0xFF)

X0, X1 = 0.42, 12.92
Y0, Y1 = 1.22, 6.86


# --------------------------------------------------------------------------- basics
def drop(shape) -> None:
    shape._element.getparent().remove(shape._element)


def by_name(slide, *fragments):
    return [s for s in slide.shapes
            if any(f.lower() in s.name.lower() for f in fragments)]


def textbox(slide, x, y, w, h):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    return tb, tf


def write(tf, spans, size=10, colour=INK, bold=False, align=PP_ALIGN.LEFT,
          space_after=0, line=0.98, first=False, font=None, italic=False):
    p = tf.paragraphs[0] if (first and not tf.paragraphs[0].runs) else tf.add_paragraph()
    p.alignment = align
    p.space_after = Pt(space_after)
    p.line_spacing = line
    for text, over in ([(spans, {})] if isinstance(spans, str) else spans):
        r = p.add_run()
        r.text = text
        f = r.font
        f.name = over.get("font", font or FONT)
        f.size = Pt(over.get("size", size))
        f.bold = over.get("bold", bold)
        f.italic = over.get("italic", italic)
        f.color.rgb = over.get("colour", colour)
    return p


def rule(slide, x, y, w, colour=RULE, weight=0.75):
    """A hairline. This deck's main separator -- it divides without adding a shape."""
    ln = slide.shapes.add_connector(1, Inches(x), Inches(y), Inches(x + w), Inches(y))
    ln.line.color.rgb = colour
    ln.line.width = Pt(weight)
    return ln


def vrule(slide, x, y, h, colour=RULE, weight=0.75):
    ln = slide.shapes.add_connector(1, Inches(x), Inches(y), Inches(x), Inches(y + h))
    ln.line.color.rgb = colour
    ln.line.width = Pt(weight)
    return ln


def block(slide, x, y, w, h, fill=FAINT, edge=None, weight=0.75):
    """A square-cornered container. Used only where something really is a container."""
    sh = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y),
                                Inches(w), Inches(h))
    if fill is None:
        sh.fill.background()
    else:
        sh.fill.solid()
        sh.fill.fore_color.rgb = fill
    if edge is None:
        sh.line.fill.background()
    else:
        sh.line.color.rgb = edge
        sh.line.width = Pt(weight)
    sh.shadow.inherit = False
    sh.text_frame.text = ""
    return sh


def icon(slide, name, x, y, size=0.26):
    f = ICONS / f"{name}.png"
    if not f.exists():
        raise FileNotFoundError(f"icon {name!r} missing; run fetch_icons.py")
    return slide.shapes.add_picture(str(f), Inches(x), Inches(y),
                                    width=Inches(size), height=Inches(size))


# ------------------------------------------------------------------- composed parts
def label(slide, x, y, w, text, colour=SLATE, size=7.5, ruled=True):
    """A small-caps section label over a hairline. Replaces the old heading + bar."""
    _, tf = textbox(slide, x, y, w, 0.18)
    write(tf, text.upper(), size=size, colour=colour, bold=True, first=True)
    if ruled:
        rule(slide, x, y + 0.20, w)
    return y + 0.30


def entry(slide, x, y, w, n, title, body, num_colour=ACCENT, gap=0.40, size=9.6):
    """An ordinal, a bold title, a grey line or two. No box, no icon chip."""
    _, tf = textbox(slide, x, y - 0.045, gap - 0.06, 0.30)
    write(tf, n, size=13, colour=num_colour, bold=True, first=True, line=0.9)
    _, tf = textbox(slide, x + gap, y, w - gap, 0.70)
    write(tf, title, size=size, colour=INK, bold=True, first=True, line=0.94,
          space_after=1)
    if body:
        write(tf, body, size=8.3, colour=SLATE, line=0.98)


def stat(slide, x, y, w, value, unit, caption, colour=ACCENT, size=27):
    """One number, large and light, over a caption. No tile, no border."""
    _, tf = textbox(slide, x, y, w, 0.46)
    write(tf, [(value, {"size": size, "colour": colour}),
               (unit, {"size": size * 0.42, "colour": colour})],
          first=True, line=0.86, bold=False)
    _, tf = textbox(slide, x, y + 0.44, w, 0.42)
    write(tf, caption, size=7.6, colour=SLATE, first=True, line=1.0)


def kv(tf, k, v, first=False, size=8.4, space_after=4):
    write(tf, [(f"{k}   ", {"colour": INK, "bold": True}), (v, {"colour": SLATE})],
          size=size, first=first, space_after=space_after, line=1.0)


def figure(slide, name, x, y, w, h=None):
    # Prefer the deck-styled re-plots from make_figures.py; fall back to the evidence
    # pack's originals so a missing re-plot degrades rather than crashes.
    for base in (DECK_FIGS, FIGS):
        f = base / name
        if f.exists():
            break
    else:
        raise FileNotFoundError(f"figure {name} in neither {DECK_FIGS} nor {FIGS}; "
                                f"run tools/ppt/make_figures.py")
    kw = {"width": Inches(w)} if h is None else {"width": Inches(w), "height": Inches(h)}
    return slide.shapes.add_picture(str(f), Inches(x), Inches(y), **kw)


def shot(slide, name, x, y, w, h):
    """Place a viewer capture. Cropped to a fixed aspect upstream, so w/h is honoured
    exactly and a mismatched box shows as distortion rather than passing silently."""
    f = SHOTS / name
    if not f.exists():
        raise FileNotFoundError(f"capture {name} missing; run tools/ppt/capture_viewer.py "
                                f"then crop_shots.py")
    pic = slide.shapes.add_picture(str(f), Inches(x), Inches(y),
                                   width=Inches(w), height=Inches(h))
    pic.line.color.rgb = RULE
    pic.line.width = Pt(0.75)
    return pic


def flow(slide, x, y, w, steps, h=0.62):
    """A pipeline drawn as labelled stops on a single rule, not a row of boxes."""
    n = len(steps)
    seg = w / n
    rule(slide, x, y, w, colour=RULE, weight=1.0)
    for i, (title, sub) in enumerate(steps):
        cx = x + i * seg
        last = i == n - 1
        dot = slide.shapes.add_shape(MSO_SHAPE.OVAL, Inches(cx - 0.035),
                                     Inches(y - 0.035), Inches(0.07), Inches(0.07))
        dot.fill.solid()
        dot.fill.fore_color.rgb = ACCENT if last else INK
        dot.line.fill.background()
        dot.shadow.inherit = False
        _, tf = textbox(slide, cx - 0.04, y + 0.13, seg - 0.10, h)
        write(tf, title, size=8.6, colour=ACCENT if last else INK, bold=True,
              first=True, line=0.94, space_after=1)
        write(tf, sub, size=7.3, colour=SLATE, line=0.96)


def trim_footer(slide, bar_h=0.28, slide_h=7.5):
    """Drop the template's "@SIH Idea submission- Template" line and slim its blue bar.

    The bar ships 0.55 in tall with a caption nobody reads, and it sets the floor for
    every slide: content had to stop at 6.90. Slimming it to 0.28 and deleting the
    caption gives 0.26 in back on all six pages, which is what pays for a larger type
    scale. The page number stays -- the template numbers its slides and a judge flipping
    a printout needs it.

    Slide 1 carries no bar, and its big white Rectangle sits at top 0, so the y guard
    below leaves it alone.
    """
    top = slide_h - bar_h
    for sh in list(slide.shapes):
        name = sh.name
        if name.startswith("Footer Placeholder"):
            drop(sh)
        elif sh.top is None:
            continue
        elif name.startswith("Rectangle") and sh.top / 914400 > 6.5:
            sh.top, sh.height = Emu(int(top * 914400)), Emu(int(bar_h * 914400))
        elif name.startswith("Slide Number"):
            # This placeholder inherits its geometry from the layout. Setting only `top`
            # makes python-pptx write a complete a:xfrm, and the un-set left and width
            # come out as zero -- which put the page number in the bottom-left corner.
            # So all four are written explicitly.
            left, width = sh.left, sh.width
            sh.top, sh.height = Emu(int(top * 914400)), Emu(int(bar_h * 914400))
            sh.left, sh.width = left, width
            sh.text_frame.vertical_anchor = MSO_ANCHOR.MIDDLE
            for p in sh.text_frame.paragraphs:
                for r in p.runs:
                    r.font.size = Pt(10)
