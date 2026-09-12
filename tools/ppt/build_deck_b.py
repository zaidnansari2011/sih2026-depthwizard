"""Layout primitives for deck B -- the poster variant.

Deck A (build_deck.py) is editorial: hairlines, square corners, three colours, a lot of
white. It reads well when someone sits and reads it. Judges flick. Held at arm's length
next to a Canva-built deck, deck A reads as sparse and deck B has to answer that without
becoming the same deck as everyone else's.

What is worth taking from the reference deck:
  - every zone is a **bounded panel**, so a slide reads as one composed object rather
    than as text floating on white;
  - section headings are **filled pills**, not labels, so the eye lands on structure
    first and prose second;
  - **badged icons** give each item a fixed visual anchor at a consistent size;
  - a **comparison table** with tick / caution / cross, which is the single most
    persuasive object a judge can scan in two seconds.

What is deliberately not taken:
  - the lavender-to-blue gradient on every surface, which is the Canva house style and
    reads as template-picked;
  - five different container shapes per slide;
  - the visible Canva watermark on the reference deck's own slide 4, which is what an
    unlicensed element looks like once it is exported.

The palette is the differentiator. Rather than pick a nice blue, it is a **hypsometric
ramp** -- the colour scale a digital surface model is actually drawn in: deep water navy
through steel and sky to sand and, at the top of the range, ember. It is the product's
own subject used as its identity, so it cannot read as picked off a template shelf.

Type is Franklin Gothic Demi (and its condensed cut) for structure, Segoe UI for prose.
Deck A is Calibri throughout, so the two decks do not look like one deck recoloured.

That pairing was chosen by measurement, not taste. The first version used Bahnschrift,
which ships with Windows and looked right in the preview -- but the exported PDF
contained no Bahnschrift at all: it is a variable font, PowerPoint's export will not
embed one, and every run had been silently replaced with Calibri, the exact face deck B
exists to avoid. tools/ppt/probe_fonts.py sets each candidate through the real export
path and reads back what actually got drawn; Franklin Gothic and Segoe UI survive it.
"""
from __future__ import annotations

from pathlib import Path

from pptx.dml.color import RGBColor
from pptx.enum.dml import MSO_LINE_DASH_STYLE, MSO_THEME_COLOR
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

from build_deck import DOCS, by_name, drop, textbox

DECK_FIGS = Path("D:/sih2026/depthwizard/tools/ppt/figures")
FIGS = DOCS / "figures"
SHOTS = Path("D:/sih2026/depthwizard/tools/ppt/shots")
ICONS = Path("D:/sih2026/depthwizard/tools/ppt/icons")
WHITE_ICONS = ICONS / "white"
MARKS = ICONS / "marks"
SRC = DOCS / "SIH2026-IDEA-Presentation-Format.pptx"
DST = DOCS / "SIH2026-DevUp-SIH26175-DepthWizard-B.pptx"

TEAM = "Dev Up"
DISPLAY = "Franklin Gothic Demi"        # wordmark and numerals
COND = "Franklin Gothic Demi Cond"      # pills and small-caps labels
SANS = "Segoe UI"                       # prose
SANS_B = "Segoe UI Semibold"

# ------------------------------------------------------- palette: a hypsometric ramp
NAVY = RGBColor(0x0E, 0x2A, 0x3F)     # the bottom of the ramp; panel headers
DEEP = RGBColor(0x16, 0x3F, 0x5C)
STEEL = RGBColor(0x2E, 0x6E, 0x92)
SKY = RGBColor(0x7F, 0xB0, 0xCB)
SAND = RGBColor(0xD8, 0x9E, 0x3C)
EMBER = RGBColor(0xC1, 0x44, 0x0E)    # the top of the ramp, and deck A's accent
RAMP = (NAVY, DEEP, STEEL, SKY, SAND, EMBER)

INK = RGBColor(0x14, 0x1B, 0x22)
BODY = RGBColor(0x2A, 0x35, 0x40)     # prose. Near-ink: a mid grey reads as faded here
SLATE = RGBColor(0x55, 0x63, 0x6E)    # genuinely secondary text only
MIST = RGBColor(0xEC, 0xF1, 0xF5)     # card fill
PAPER = RGBColor(0xF7, 0xFA, 0xFC)    # panel fill
LINE = RGBColor(0xC7, 0xD5, 0xDF)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
GREEN = RGBColor(0x1E, 0x8E, 0x5A)
RED = RGBColor(0xC0, 0x39, 0x2B)

X0, X1 = 0.40, 12.93
W = X1 - X0
# 7.16, not 6.90: build_deck.trim_footer slims the template's blue bar from 0.55 in to
# 0.28 and deletes its caption, which hands 0.26 in back to every slide.
BODY_TOP, BODY_BOT = 1.22, 7.16

# ------------------------------------------------------------------------ type scale
# Set against the reference decks rather than by eye. Normalised onto a common page
# size, Debug Dynasty's body copy sits at 8.7 pt and BlueRadar -- a real SIH deck -- runs
# an 11.5 pt median with nothing at all below 8 pt. The first cut of this deck ran a
# 7.4 pt median because it carried 10,447 characters to the reference's 6,966.
#
# This scale is the second raise. It is paid for twice over: by cutting copy, and by
# reclaiming the 0.26 in the template's footer bar was spending on a caption nobody
# reads. Type went up first and every block was then laid out to fit it.
T_PILL = 12.5        # section heading in a filled bar
T_PILL_S = 11.0      # secondary heading
T_TITLE = 10.4       # card / entry title
T_BODY = 9.4         # card prose -- the size that matters most
T_LABEL = 8.8        # small caps on a dark fill
T_MICRO = 8.6        # captions, legends, figure notes


# --------------------------------------------------------------------------- text
def w_(tf, spans, size=T_BODY, colour=BODY, bold=False, align=PP_ALIGN.LEFT, space_after=0,
       line=1.0, first=False, font=SANS, italic=False, caps=False):
    """Write a paragraph. Same shape as build_deck.write but defaults to deck B's face."""
    p = tf.paragraphs[0] if (first and not tf.paragraphs[0].runs) else tf.add_paragraph()
    p.alignment = align
    p.space_after = Pt(space_after)
    p.line_spacing = line
    for text, over in ([(spans, {})] if isinstance(spans, str) else spans):
        r = p.add_run()
        r.text = text.upper() if caps else text
        f = r.font
        f.name = over.get("font", font)
        f.size = Pt(over.get("size", size))
        f.bold = over.get("bold", bold)
        f.italic = over.get("italic", italic)
        f.color.rgb = over.get("colour", colour)
    return p


def para(slide, x, y, w, h, spans, **kw):
    _, tf = textbox(slide, x, y, w, h)
    w_(tf, spans, first=True, **kw)
    return tf


# ------------------------------------------------------------------------- shapes
def _shape(slide, kind, x, y, w, h, fill, edge, weight=0.75):
    sh = slide.shapes.add_shape(kind, Inches(x), Inches(y), Inches(w), Inches(h))
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
    tf = sh.text_frame
    tf.text = ""
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = Inches(0.06)
    tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    return sh


def rounded(slide, x, y, w, h, fill=WHITE, edge=LINE, radius=0.08, weight=0.75):
    """A rounded rectangle with the corner radius set in inches rather than as a ratio.

    python-pptx exposes the adjustment as a fraction of the shorter side, which means an
    identical adjustment gives a 0.05 in corner on a tall card and a 0.30 in corner on a
    short bar. Fixing the radius instead keeps every container in the deck consistent.
    """
    sh = _shape(slide, MSO_SHAPE.ROUNDED_RECTANGLE, x, y, w, h, fill, edge, weight)
    sh.adjustments[0] = min(0.5, radius / max(0.01, min(w, h)))
    return sh


def rect(slide, x, y, w, h, fill=MIST, edge=None, weight=0.75):
    return _shape(slide, MSO_SHAPE.RECTANGLE, x, y, w, h, fill, edge, weight)


def panel(slide, x, y, w, h, fill=PAPER, edge=LINE):
    """The zone container. One per region, and nothing nests more than one deep."""
    return rounded(slide, x, y, w, h, fill=fill, edge=edge, radius=0.10)


def pill(slide, x, y, w, text, h=0.34, fill=NAVY, colour=WHITE, size=T_PILL,
         align=PP_ALIGN.CENTER, font=COND):
    """A section heading as a filled bar. Deck B's main structural device."""
    sh = rounded(slide, x, y, w, h, fill=fill, edge=None, radius=h / 2)
    w_(sh.text_frame, text, size=size, colour=colour, bold=False, align=align,
       first=True, font=font, caps=True, line=0.95)
    return y + h + 0.10


def tag(slide, x, y, w, h, text, fill=STEEL, colour=WHITE, size=T_LABEL, radius=0.05):
    sh = rounded(slide, x, y, w, h, fill=fill, edge=None, radius=radius)
    w_(sh.text_frame, text, size=size, colour=colour, align=PP_ALIGN.CENTER, first=True,
       font=COND, caps=True, line=0.95)
    return sh


def card(slide, x, y, w, h, fill=WHITE, edge=LINE, radius=0.07):
    return rounded(slide, x, y, w, h, fill=fill, edge=edge, radius=radius)


def dotted(slide, x1, y1, x2, y2, colour=SKY, weight=1.0):
    ln = slide.shapes.add_connector(1, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    ln.line.color.rgb = colour
    ln.line.width = Pt(weight)
    ln.line.dash_style = MSO_LINE_DASH_STYLE.ROUND_DOT
    return ln


# -------------------------------------------------------------------------- icons
def badge(slide, name, cx, cy, d=0.40, fill=STEEL, shape=MSO_SHAPE.OVAL, pad=0.26):
    """A filled disc (or hexagon) with a white glyph centred on it."""
    sh = _shape(slide, shape, cx - d / 2, cy - d / 2, d, d, fill, None)
    f = WHITE_ICONS / f"{name}.png"
    if not f.exists():
        raise FileNotFoundError(f"white icon {name!r} missing; run fetch_icons_b.py")
    g = d * pad * 2
    slide.shapes.add_picture(str(f), Inches(cx - g / 2), Inches(cy - g / 2),
                             width=Inches(g), height=Inches(g))
    return sh


def mark(slide, kind, cx, cy, d=0.17):
    """A comparison-table verdict: check, minus (partial) or x."""
    f = MARKS / f"{kind}.png"
    if not f.exists():
        raise FileNotFoundError(f"mark {kind!r} missing; run fetch_icons_b.py")
    return slide.shapes.add_picture(str(f), Inches(cx - d / 2), Inches(cy - d / 2),
                                    width=Inches(d), height=Inches(d))


def brand(slide, name, cx, cy, d=0.30):
    """A brand mark from deck A's icon set, which is already in each brand's colour."""
    f = ICONS / f"{name}.png"
    if not f.exists():
        raise FileNotFoundError(f"icon {name!r} missing; run fetch_icons.py")
    return slide.shapes.add_picture(str(f), Inches(cx - d / 2), Inches(cy - d / 2),
                                    width=Inches(d), height=Inches(d))


# ----------------------------------------------------------------- composed parts
def bullet_card(slide, x, y, w, h, title, body, ico=None, fill=WHITE, accent=STEEL,
                tsize=T_TITLE, bsize=T_BODY, edge=LINE):
    """Icon badge, bold title, grey body. The repeating unit of slides 2 and 5."""
    card(slide, x, y, w, h, fill=fill, edge=edge)
    tx = x + 0.12
    if ico:
        badge(slide, ico, x + 0.30, y + h / 2 if body is None else y + 0.30, 0.38,
              fill=accent)
        tx = x + 0.56
    tf = para(slide, tx, y + 0.11, w - (tx - x) - 0.12, h - 0.18, title, size=tsize,
              colour=INK, bold=True, font=SANS_B, line=0.96, space_after=1.5)
    if body:
        w_(tf, body, size=bsize, colour=SLATE, line=0.98)


def spine_step(slide, x, y, w, h, n, title, body, fill=NAVY, tsize=T_TITLE, bsize=T_BODY):
    """One stage of the pipeline: numbered square, title, description, on a tinted row."""
    rounded(slide, x, y, w, h, fill=MIST, edge=None, radius=0.05)
    sq = rounded(slide, x + 0.07, y + (h - 0.30) / 2, 0.30, 0.30, fill=fill, edge=None,
                 radius=0.05)
    sq.text_frame.word_wrap = False
    w_(sq.text_frame, n, size=10.5, colour=WHITE, align=PP_ALIGN.CENTER, first=True,
       font=DISPLAY, line=0.95)
    tf = para(slide, x + 0.46, y + 0.07, w - 0.56, h - 0.10, title, size=tsize,
              colour=INK, bold=True, font=SANS_B, line=0.94, space_after=1,
              caps=True)
    w_(tf, body, size=bsize, colour=SLATE, line=0.98)


def chevron(slide, x, y, w, h, text, ico, fill=DEEP, badge_fill=EMBER, size=T_BODY + 0.5):
    """A benefit as a right-pointing bar. The reference deck's most legible device."""
    sh = _shape(slide, MSO_SHAPE.PENTAGON, x, y, w, h, fill, None)
    sh.adjustments[0] = min(0.5, (h * 0.70) / w)
    w_(sh.text_frame, text, size=size, colour=WHITE, font=SANS, line=0.98, first=True)
    sh.text_frame.margin_left = Inches(0.54)
    sh.text_frame.margin_right = Inches(0.34)
    badge(slide, ico, x + 0.30, y + h / 2, h * 0.60, fill=badge_fill)
    return sh


def figure(slide, name, x, y, w, h=None):
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
    f = SHOTS / name
    if not f.exists():
        raise FileNotFoundError(f"capture {name} missing; run capture_viewer.py "
                                f"then crop_shots.py")
    pic = slide.shapes.add_picture(str(f), Inches(x), Inches(y), width=Inches(w),
                                   height=Inches(h))
    pic.line.color.rgb = LINE
    pic.line.width = Pt(0.75)
    return pic


CHIP_CAPTION = RGBColor(0xC4, 0xD3, 0xDE)     # legible on NAVY, not on MIST


def stat_chip(slide, x, y, w, h, value, unit, caption, fill=NAVY, vcolour=SAND,
              ccolour=None):
    rounded(slide, x, y, w, h, fill=fill, edge=None, radius=0.07)
    para(slide, x + 0.16, y + 0.09, w - 0.28, 0.36,
         [(value, {"size": 24, "colour": vcolour, "font": DISPLAY}),
          (unit, {"size": 11, "colour": vcolour, "font": DISPLAY})], line=0.9)
    para(slide, x + 0.16, y + 0.46, w - 0.28, h - 0.50, caption, size=T_MICRO,
         colour=ccolour or (CHIP_CAPTION if fill == NAVY else SLATE), line=0.98)


def compare_table(slide, x, y, w, cols, rows, param_w=None, head_h=0.34, row_h=0.36):
    """PARAMETER | alternative | alternative | ours, with tick / partial / cross marks.

    The last column is ours and is the only one filled navy, so the eye reads down it.
    Rows where an alternative genuinely wins are left winning; a table that gives us a
    clean sweep is the kind a panel stops trusting.
    """
    n = len(cols)
    pw = param_w if param_w is not None else w * 0.42
    cw = (w - pw) / n
    head = rounded(slide, x, y, pw - 0.04, head_h, fill=NAVY, edge=None, radius=0.05)
    w_(head.text_frame, "PARAMETER", size=9.0, colour=WHITE, align=PP_ALIGN.CENTER,
       first=True, font=DISPLAY)
    for i, c in enumerate(cols):
        last = i == n - 1
        sh = rounded(slide, x + pw + i * cw, y, cw - 0.04, head_h,
                     fill=EMBER if last else DEEP, edge=None, radius=0.05)
        # These cells are under 0.8 in wide. The default 0.06 in side margins were enough
        # to break "DEPTHWIZARD" across two lines as "DEPTHWIZAR / D".
        sh.text_frame.margin_left = sh.text_frame.margin_right = Inches(0.01)
        w_(sh.text_frame, c, size=8.0 if not last else 8.2, colour=WHITE,
           align=PP_ALIGN.CENTER, first=True, font=COND, caps=True, line=0.92)
    yy = y + head_h + 0.05
    for r, (name, verdicts) in enumerate(rows):
        band = MIST if r % 2 == 0 else WHITE
        rounded(slide, x, yy, pw - 0.04, row_h, fill=band, edge=None, radius=0.04)
        para(slide, x + 0.10, yy + 0.05, pw - 0.24, row_h - 0.08, name, size=8.6,
             colour=INK, font=SANS_B, line=0.96)
        for i, v in enumerate(verdicts):
            last = i == n - 1
            rounded(slide, x + pw + i * cw, yy, cw - 0.04, row_h,
                    fill=RGBColor(0xE7, 0xEF, 0xF4) if last else band, edge=None,
                    radius=0.04)
            mark(slide, v, x + pw + i * cw + (cw - 0.04) / 2, yy + row_h / 2,
                 d=0.19 if last else 0.16)
        yy += row_h + 0.04
    return yy


# --------------------------------------------------------- deck C additions
# Appended for deck C. Nothing above is changed, so deck B still builds byte-identically.
def arch_layer(slide, x, y, w, label, items, band=0.34, head=0.20, fill=NAVY,
               icon_d=0.24):
    """One band of a layered architecture diagram: a filled label bar over its parts.

    The reference deck draws its architecture this way and it reads as engineering; a
    numbered list of the same seven stages reads as bullets. Returns the band's bottom.
    """
    bar = rounded(slide, x, y, w, head, fill=fill, edge=None, radius=0.04)
    w_(bar.text_frame, label, size=T_MICRO, colour=WHITE, align=PP_ALIGN.CENTER,
       first=True, font=COND, caps=True, line=0.95)
    rounded(slide, x, y + head, w, band, fill=MIST, edge=None, radius=0.04)
    cw = w / len(items)
    for i, (ico, text) in enumerate(items):
        cx = x + i * cw
        if i:
            rect(slide, cx, y + head + 0.06, 0.01, band - 0.12, fill=LINE)
        badge(slide, ico, cx + 0.20, y + head + band / 2, icon_d, fill=STEEL)
        para(slide, cx + 0.36, y + head + 0.05, cw - 0.42, band - 0.08, text,
             size=T_MICRO, colour=BODY, line=0.96)
    return y + head + band


def arrow_down(slide, cx, y, h=0.10, w=0.16, fill=SKY):
    sh = _shape(slide, MSO_SHAPE.DOWN_ARROW, cx - w / 2, y, w, h, fill, None)
    return y + h


def tagstrip(slide, x, y, w, tags, h=0.24, gap=0.07, size=T_MICRO, fill=MIST,
             colour=INK, char=0.058, pad=0.20):
    """Technology names as inline chips instead of a logo grid.

    The grid this replaces occupied a whole column of slide 3 to say what one line of
    text says. Chips keep the template's "technologies to be used" pointer answered and
    give the column back. Widths are estimated from glyph count, so the row wraps rather
    than overflowing.
    """
    cx, cy = x, y
    for t in tags:
        tw = len(t) * char + pad
        if cx + tw > x + w:
            cx, cy = x, cy + h + 0.06
        chip = rounded(slide, cx, cy, tw, h, fill=fill, edge=LINE, radius=0.04)
        chip.text_frame.margin_left = chip.text_frame.margin_right = Inches(0.02)
        w_(chip.text_frame, t, size=size, colour=colour, align=PP_ALIGN.CENTER,
           first=True, font=SANS_B)
        cx += tw + gap
    return cy + h


def qr_png(url: str, dest: Path, dark="0E2A3F") -> Path:
    """Render a QR to PNG. A video cannot survive PowerPoint's PDF export -- proven by
    tools/ppt/probe_video_in_pdf.py -- so a scannable code is the only way to put a demo
    clip in a submitted PDF."""
    import segno
    dest.parent.mkdir(parents=True, exist_ok=True)
    segno.make(url, error="h").save(str(dest), scale=16, border=2, dark=dark,
                                    light="ffffff")
    return dest


def brandstrip(slide, x, y, w, items, d=0.32, size=T_MICRO - 1.0, gap=0.06):
    """The technology stack as a single row of brand marks with labels beneath.

    Replaces the text chips this started as. Deck B spends a whole column on a 2x5 logo
    grid; one row says the same thing in a quarter of the height, and a logo is read
    faster than its name. Marks come from deck A's icon set, each already in its own
    brand colour.
    """
    cw = w / len(items)
    for i, (ico, name) in enumerate(items):
        cx = x + i * cw
        brand(slide, ico, cx + cw / 2, y + d / 2, d)
        para(slide, cx, y + d + gap, cw, 0.26, name, size=size, colour=INK,
             align=PP_ALIGN.CENTER, font=SANS_B, line=0.94)
    return y + d + gap + 0.26
