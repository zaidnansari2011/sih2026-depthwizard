"""Deck D: deck C, restructured against the jury's five-slide evaluation rubric.

The rubric the panel scores against names, per slide, what it expects to find:

  1 Problem       PS id and title, the problem, affected users, current challenges, why it
                  matters -- with real data and a simple illustration.
  2 Solution      concept, features, how it solves it, the existing gap, the USP -- with a
                  before-vs-after comparison and a solution diagram.
  3 Technical     architecture, workflow, technologies, algorithms, hardware, APIs,
                  database, data flow -- with prototype evidence.
  4 Feasibility   technical, operational and economic feasibility, risks and mitigation,
                  a deployment plan -- with working modules, a cost estimate and a testing
                  plan.
  5 Impact        beneficiaries, measurable benefits, social / economic / environmental
                  impact, scalability -- quantified, with a before/after comparison.

The SIH template caps the deck at six slides including the title, so "Problem" has no
slide of its own: it shares slide 2 with the solution, as it did in deck C. What changed:

  slide 1  the three headline numbers were run02's (3.67 m, 38 %, 509 ms) and
           contradicted slide 6's 3.464 m. Now the shipped figures, plus live and code
           links.
  slide 2  the problem is grounded in ISRO's own wording and names who feels it; the
           image strip is a before -> after of ONE tile (the evidence-pack error map's
           satellite panel is OMA_288_042, the viewer's urban scene); a USP line states
           the gap.
  slide 3  the deployment column moves to slide 4. Its place answers "hardware, APIs,
           database": the GPU it trained on, the HTTP API, and that outputs are files. A
           live-demo QR is the prototype evidence.
  slide 4  both charts redrawn from the shipped metrics file (deck C's were run02: urban
           13.86 m against 13.084 m measured). Adds the deployment plan, a cost estimate
           and a testing plan; the four Indian-domain mitigations fold into risk 1.
  slide 5  leads with disaster -- the portal theme -- and shows the inundation tool, which
           is built and runs on the Indian scene but appeared nowhere in deck C. Benefits
           become a today / with-DepthWizard table tagged social, economic and
           environmental; the stat tiles become impact numbers.
  slide 6  deck C's last paragraph ran into the footer bar; it is shortened, the
           tall-building data ceiling moves into probe 01, and live / code links and the
           Maxar CC-BY credit are added.

Every figure traces to docs/evidence-pack.md, docs/deployment.md, docs/gamus-integration.md
or the shipped metrics file. Two are derived, and say so where they appear:
~1 min per km^2 (5.0 s per 1024 px tile, single pass, desktop CPU; a tile is 0.094 km^2) and
41 % (5.9 against 3.464 m).

    python tools/ppt/prep_assets_d.py
    python tools/ppt/make_deck_d.py
    python tools/ppt/check_layout_b.py D
    python tools/ppt/export_pdf.py D
"""
from __future__ import annotations

import math
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.dml import MSO_LINE_DASH_STYLE
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt

from build_deck import by_name, drop, textbox, trim_footer
from build_deck_b import (BODY, BODY_TOP, COND, DEEP, DISPLAY, EMBER, GREEN, INK, LINE,
                          MIST, NAVY, RAMP, RED, SAND, SANS, SANS_B, SKY, SLATE, SRC,
                          STEEL, T_BODY, T_LABEL, T_MICRO, T_PILL_S, T_TITLE, TEAM, W,
                          WHITE, X0, X1, _shape, arch_layer, arrow_down, badge,
                          brandstrip, card, compare_table, mark, panel, para, pill,
                          qr_png, rect, rounded, stat_chip, tag, w_)
from make_deck_b import PS_ID, PS_THEME, PS_TITLE, TEAM_ID

DST = Path("D:/sih2026/depthwizard/docs/SIH2026-DevUp-SIH26175-DepthWizard-D.pptx")
FIG = Path("D:/sih2026/depthwizard/tools/ppt/figures/d")

LIVE_URL = "https://project5.zaidansari.tech"
CODE_URL = "https://github.com/zaidnansari2011/sih2026-depthwizard"
DOCS_URL = "https://project5.zaidansari.tech/documentation/"

# Shown text -> target. Text alone is not a link: the first build of this deck printed
# these addresses and PowerPoint exported them to PDF as plain, unclickable words.
# link_runs() attaches the target to every run whose text is exactly one of these.
LINKS = {"project5.zaidansari.tech": LIVE_URL,
         "project5.zaidansari.tech/documentation": DOCS_URL,
         "github.com/zaidnansari2011/sih2026-depthwizard": CODE_URL}
# The hosted pitch clip. Set to None and slide 6 draws the reserved slot instead.
PITCH_VIDEO_URL = "https://youtu.be/4NpR74MS5oE"

TINT = RGBColor(0xE7, 0xEF, 0xF4)          # our column in every comparison

# Read off the sikkim_flood capture's HUD (tools/ppt/shots/sikkim_flood.png). If that
# capture is retaken these must be re-read from it: they describe that one frame.
FLOOD_READOUT = [("WATER LEVEL", "1,143 m"), ("GROUND UNDER WATER", "20.5 %"),
                 ("BUILDINGS FLOODED", "44")]


def img(s, name, x, y, w, h, edge=True):
    f = FIG / name
    if not f.exists():
        raise FileNotFoundError(f"{f} missing; run tools/ppt/prep_assets_d.py")
    pic = s.shapes.add_picture(str(f), Inches(x), Inches(y), width=Inches(w),
                               height=Inches(h))
    if edge:
        pic.line.color.rgb = LINE
        pic.line.width = Pt(0.75)
    return pic


def label(s, x, y, text, fill=NAVY, h=0.26, size=T_LABEL):
    """A caps tag pinned to an image's top-left corner, sized to its text."""
    w = 0.22 + len(text) * 0.066
    lab = rounded(s, x, y, w, h, fill=fill, edge=None, radius=0.04)
    w_(lab.text_frame, text, size=size, colour=WHITE, first=True, font=COND, caps=True)
    lab.text_frame.margin_left = Inches(0.10)
    return lab


def qr(s, url, name, x, y, d, caption):
    p = qr_png(url, FIG / f"{name}.png")
    pic = s.shapes.add_picture(str(p), Inches(x), Inches(y), width=Inches(d),
                               height=Inches(d))
    pic.click_action.hyperlink.address = url        # clickable in the PDF, not only scannable
    para(s, x - 0.20, y + d + 0.01, d + 0.40, 0.16, caption, size=7.4, colour=SLATE,
         align=PP_ALIGN.CENTER, font=SANS_B)


# ------------------------------------------------------------------------- slide 1
def slide_title(s) -> None:
    for sh in by_name(s, "TextBox 9"):
        drop(sh)
    for sh in by_name(s, "Subtitle 3"):
        drop(sh)

    PX, PW = X0, 5.62
    panel(s, PX, 1.30, PW, 3.56)
    rounded(s, PX, 1.30, PW, 0.68, fill=NAVY, edge=None, radius=0.10)
    para(s, PX + 0.20, 1.42, PW - 0.40, 0.42,
         [("DEPTH", {"colour": WHITE}), ("WIZARD", {"colour": SAND})],
         size=23, font=DISPLAY, line=0.95)
    para(s, PX + 0.20, 2.12, PW - 0.40, 0.28,
         "One satellite image in, a measurable 3D surface out.", size=11.5,
         colour=BODY, font=SANS)

    fields = [("Problem Statement ID", PS_ID), ("Problem Statement Title", PS_TITLE),
              ("Theme", PS_THEME), ("PS Category", "Software"),
              ("Team ID", TEAM_ID), ("Team Name (on portal)", TEAM)]
    TAG_W, BOX_W, CHAR_W = 1.95, PW - 2.49, 0.098
    y = 2.56
    for k, v in fields:
        tg = tag(s, PX + 0.20, y, TAG_W, 0.24, k, fill=STEEL, size=8.0)
        tg.text_frame.margin_left = tg.text_frame.margin_right = 0
        lines = max(1, math.ceil(len(v) * CHAR_W / BOX_W))
        para(s, PX + 0.20 + TAG_W + 0.18, y - 0.03, BOX_W, 0.24 + 0.19 * (lines - 1),
             v, size=12.5, colour=INK, font=SANS_B, line=0.98)
        y += 0.35 + 0.19 * (lines - 1)

    pill(s, PX, 5.00, PW, "Built and measured, not proposed", h=0.34, fill=EMBER,
         size=T_PILL_S)
    # Deck B/C carried 3.67 m, 38 % and 509 ms here -- run02's figures -- while
    # slide 6 quoted the shipped 3.464 m. These are the shipped numbers.
    stats = [("3.464", " m", "per-building RMSE\nagainst airborne LiDAR"),
             ("41", " %", "lower error than the 5.9 m\nGlobalBuildingAtlas bar"),
             ("536", " ms", "per tile on CPU alone,\nno GPU in the loop")]
    cw = (PW - 0.24) / 3
    for i, (v, u, cap) in enumerate(stats):
        stat_chip(s, PX + i * (cw + 0.12), 5.44, cw, 0.88, v, u, cap)

    for i, c in enumerate(RAMP):
        rect(s, PX + i * (PW / 6), 6.44, PW / 6, 0.12, fill=c)
    para(s, PX, 6.59, PW, 0.20,
         "The palette is the model's own output scale: 0 m at the left, 155 m at the "
         "right.", size=T_MICRO - 0.4, colour=SLATE)
    tf = para(s, PX, 6.82, PW, 0.32,
              [("LIVE  ", {"colour": EMBER, "font": COND, "size": 9.4}),
               ("project5.zaidansari.tech", {"colour": STEEL, "font": SANS_B}),
               ("      DOCS  ", {"colour": EMBER, "font": COND, "size": 9.4}),
               ("project5.zaidansari.tech/documentation",
                {"colour": STEEL, "font": SANS_B})], size=8.4, line=1.0)
    w_(tf, [("CODE  ", {"colour": EMBER, "font": COND, "size": 9.4}),
            ("github.com/zaidnansari2011/sih2026-depthwizard",
             {"colour": STEEL, "font": SANS_B})], size=8.4, line=1.0)


# ------------------------------------------------------------------------- slide 2
def slide_problem_solution(s) -> None:
    for sh in by_name(s, "TextBox 8"):
        drop(sh)
    for sh in by_name(s, "Title 1"):
        drop(sh)

    BX_, BW = 1.85, 8.70
    panel(s, BX_, 0.28, BW, 0.84, fill=MIST)
    para(s, BX_ + 0.22, 0.44, 2.90, 0.46,
         [("DEPTH", {"colour": NAVY}), ("WIZARD", {"colour": EMBER})],
         size=25, font=DISPLAY, line=0.95)
    rect(s, BX_ + 3.10, 0.44, 0.02, 0.54, fill=LINE)
    para(s, BX_ + 3.32, 0.42, BW - 3.56, 0.62,
         [("One satellite image in, a measurable 3D surface out. ", {"colour": INK}),
          ("Every submission promises accuracy. This one reports it.",
           {"colour": EMBER, "bold": True})], size=12.5, line=1.04)

    LW, RX = 4.30, 5.00
    RW = X1 - RX
    TOP, BOT = 1.64, 4.62
    pill(s, X0, 1.22, LW, "The problem", fill=NAVY)
    pill(s, RX, 1.22, RW, "Our solution", fill=EMBER)

    # ----------------------------------------------------------------- the problem
    panel(s, X0, TOP, LW, BOT - TOP)
    # "Why it matters", in the sponsor's words rather than ours: this sentence is the
    # problem statement's own background paragraph, quoted verbatim.
    rect(s, X0 + 0.14, 1.74, 0.05, 0.50, fill=SAND)
    para(s, X0 + 0.28, 1.70, LW - 0.40, 0.58,
         # The source reads "These approaches can be ..."; the bracket names what they are.
         [("“[Stereo, LiDAR and InSAR] can be cost-prohibitive, dependent on specific "
           "sensor availability, and computationally intensive.”",
           {"colour": INK, "italic": True}),
          ("\n— ISRO, problem statement SIH26175",
           {"colour": SLATE, "font": SANS_B, "size": 8.0, "italic": False})],
         size=9.2, line=1.02)
    problems = [
        ("Archives with no height",
         "Stereo and LiDAR cannot be flown into the past. Decades of Cartosat archive "
         "carry no height."),
        ("Relative depth, not metres",
         "Depth models learn from ground-level photos. From orbit they give shape, but "
         "no metric scale."),
        ("No confidence, no product",
         "A bare height hides where it is wrong, and a checkpoint is not a scene an "
         "analyst can open."),
    ]
    for i, (t, b) in enumerate(problems):
        y = 2.32 + i * 0.67
        card(s, X0 + 0.10, y, LW - 0.20, 0.62, fill=WHITE)
        n = rounded(s, X0 + 0.20, y + 0.10, 0.36, 0.30, fill=EMBER, edge=None,
                    radius=0.05)
        n.text_frame.word_wrap = False
        w_(n.text_frame, f"0{i + 1}", size=11.0, colour=WHITE, align=PP_ALIGN.CENTER,
           first=True, font=DISPLAY)
        para(s, X0 + 0.66, y + 0.05, LW - 0.82, 0.22, t, size=T_TITLE - 0.4, colour=INK,
             font=SANS_B, line=0.96)
        para(s, X0 + 0.66, y + 0.26, LW - 0.82, 0.34, b, size=T_BODY - 0.5,
             colour=BODY, line=1.0)
    # Who feels it -- the rubric's "affected users", named on the problem side.
    tg = tag(s, X0 + 0.14, 4.35, 1.02, 0.20, "Who feels it", fill=NAVY, size=7.6)
    tg.text_frame.margin_left = tg.text_frame.margin_right = 0
    para(s, X0 + 1.22, 4.32, LW - 1.32, 0.24,
         "SAC analysts · disaster responders · planners · foresters", size=8.4,
         colour=INK, font=SANS_B)

    # ---------------------------------------------------------------- the solution
    panel(s, RX, TOP, RW, BOT - TOP)
    items = [
        ("image-up", "One image to a metric DSM",
         "Depth Anything V2 fine-tuned on DFC2019 + GAMUS LiDAR. GeoTIFF out, metadata "
         "or not."),
        ("shield-check", "Confidence, not just height",
         "A per-pixel sigma beside every height. Where it says unsure, it is wrong — "
         "monotonically."),
        ("box", "A navigable 3D scene",
         "three.js flythrough with measure, flood and drag-to-compare-against-LiDAR "
         "tools."),
        ("ruler", "Validation inside the product",
         "RMSE, MAE and correlation against LiDAR — per terrain class, not one "
         "flattering average."),
        ("map-pinned", "Absolute heights, two ways",
         "Copernicus GLO-30, or ground control points via a power-law fit."),
        ("wifi-off", "Deployable, not demo-ware",
         "No GPU: 536 ms per tile. The viewer is one 15 MB HTML file that opens from "
         "disk."),
    ]
    cw = (RW - 0.20 - 0.24) / 3
    for i, (ico, t, b) in enumerate(items):
        x = RX + 0.10 + (i % 3) * (cw + 0.12)
        y = 1.72 + (i // 3) * 1.22
        card(s, x, y, cw, 1.14, fill=WHITE)
        badge(s, ico, x + 0.32, y + 0.30, 0.44, fill=STEEL if i % 2 == 0 else DEEP)
        para(s, x + 0.62, y + 0.12, cw - 0.72, 0.36, t, size=T_TITLE, colour=INK,
             font=SANS_B, line=0.96)
        para(s, x + 0.14, y + 0.58, cw - 0.28, 0.52, b, size=T_BODY - 0.2,
             colour=BODY, line=1.0)
    # The gap and the USP, in one line, pointing at the table that proves it.
    rounded(s, RX + 0.10, 4.17, RW - 0.20, 0.38, fill=MIST, edge=None, radius=0.05)
    tg = tag(s, RX + 0.18, 4.24, 1.40, 0.24, "The gap we close", fill=EMBER, size=8.2)
    tg.text_frame.margin_left = tg.text_frame.margin_right = 0
    para(s, RX + 1.68, 4.19, RW - 1.86, 0.34,
         [("Published single-view work stops at weights and a score. ",
           {"colour": BODY}),
          ("We ship metric height, calibrated confidence and a no-GPU 3D viewer",
           {"colour": INK, "font": SANS_B}),
          (" — compared row by row on slide 6.", {"colour": BODY})],
         size=8.8, line=1.0)

    # ------------------------------------- before -> after, one tile, then India
    pill(s, X0, 4.72, W, "Before → after: one archived image in, a measurable surface "
                         "out — live viewer captures, not mock-ups", h=0.30, fill=DEEP,
         size=T_PILL_S)
    IY, IH = 5.10, 1.74
    img(s, "omaha_input.png", X0, IY, IH, IH)
    label(s, X0, IY, "Before · input", fill=SLATE)
    ar = _shape(s, MSO_SHAPE.RIGHT_ARROW, X0 + IH + 0.07, IY + IH / 2 - 0.17, 0.30,
                0.34, EMBER, None)
    AW_ = 3.30
    ax = X0 + IH + 0.44
    panels = [("omaha_height_d.png", "After · height, in 3D",
               "Same tile: roofs, tree crowns and kerbs at 0.3 m.", NAVY),
              ("omaha_sigma_d.png", "After · confidence",
               "Where the estimate is weak, the sigma says so.", NAVY)]
    for i, (fn, cap, sub, fill) in enumerate(panels):
        x = ax + i * (AW_ + 0.14)
        img(s, fn, x, IY, AW_, IH)
        label(s, x, IY, cap, fill=fill)
        para(s, x, 6.88, AW_, 0.20, sub, size=T_MICRO, colour=SLATE, line=0.98)
    para(s, X0, 6.88, IH + 0.3, 0.20, "One Omaha tile, as flown.", size=T_MICRO,
         colour=SLATE, line=0.98)
    sx = X1 - AW_
    rect(s, sx - 0.13, IY + 0.10, 0.02, IH - 0.20, fill=LINE)
    img(s, "sikkim_hilly_d.png", sx, IY, AW_, IH)
    label(s, sx, IY, "Sikkim, India · hilly", fill=DEEP)
    para(s, sx, 6.88, AW_, 0.20,
         [("967 m of relief across 2 km, draped.", {"colour": SLATE}),
          ("  Imagery © Maxar, CC BY-NC 4.0", {"colour": SLATE, "size": 7.0})],
         size=T_MICRO, line=0.98)


# ------------------------------------------------------------------------- slide 3
def slide_technical(s) -> None:
    for sh in by_name(s, "TextBox 8"):
        drop(sh)

    AW, BX, BW, CX = 6.00, 6.70, 2.95, 9.95
    CW = X1 - CX

    pill(s, X0, BODY_TOP, AW, "Architecture — one image to a checked surface")
    panel(s, X0, 1.66, AW, 4.36)
    layers = [
        ("Input layer", [("file-text", "GeoTIFF with\ngeo metadata"),
                         ("image-up", "Plain PNG / JPG,\nno metadata"),
                         ("satellite", "Cartosat single-view\narchive")]),
        ("Normalisation", [("ruler", "Ground sample distance\nread from metadata"),
                           ("layers", "Auto-zoom and resample\nto a 0.3 m scale")]),
        ("Model core — Depth Anything V2", [("cpu", "ViT encoder"),
                                            ("workflow", "DPT neck"),
                                            ("database", "Fine-tuned on\nDFC2019 + GAMUS")]),
        ("Dual head", [("mountain-snow", "Height mu,\nper pixel"),
                       ("shield-check", "Log-variance sigma,\nper pixel")]),
        ("Post-process", [("box", "518 px tiling,\noverlap blend, TTA"),
                          ("map-pinned", "Scale anchor: GLO-30\nor ground control")]),
        ("Output layer", [("file-text", "GeoTIFF DSM"),
                          ("gauge", "Sigma map"),
                          ("monitor", "three.js scene,\nlive error map")]),
    ]
    y = 1.76
    for i, (lbl, items) in enumerate(layers):
        y = arch_layer(s, X0 + 0.12, y, AW - 0.24, lbl, items, head=0.22, band=0.38,
                       fill=EMBER if i == len(layers) - 1 else NAVY)
        if i < len(layers) - 1:
            y = arrow_down(s, X0 + AW / 2, y + 0.02, h=0.09) + 0.02

    # ------------------------- hardware, API and storage: what the rubric asks after
    pill(s, BX, BODY_TOP, BW, "How it runs — hardware, API, data")
    panel(s, BX, 1.66, BW, 4.36)
    stages = [
        ("Trained on one consumer GPU",
         "RTX 3060, 12 GB. The shipped model's last stage took 75.5 min. Backbone: "
         "Depth Anything V2 Small, Apache 2.0."),
        ("Exported for CPU",
         "ONNX int8, 36.8 MB. 536 ms per 518 px tile, with no GPU in the loop."),
        ("Served over HTTP",
         "POST /api/upload takes TIF, PNG or JPG; /api/result returns the DSM. Live on "
         "a two-core CPU host."),
        ("Files, not a database",
         "GeoTIFF DSM and sigma, float32 scene tiles. The viewer is one 15 MB HTML file "
         "that opens from disk."),
    ]
    for i, (t, b) in enumerate(stages):
        y = 1.78 + i * 1.06
        n = rounded(s, BX + 0.12, y, 0.32, 0.32, fill=STEEL, edge=None, radius=0.05)
        n.text_frame.word_wrap = False
        w_(n.text_frame, f"{i + 1}", size=11.5, colour=WHITE, align=PP_ALIGN.CENTER,
           first=True, font=DISPLAY)
        if i < len(stages) - 1:
            rect(s, BX + 0.27, y + 0.34, 0.02, 0.70, fill=LINE)
        para(s, BX + 0.56, y + 0.02, BW - 0.70, 0.26, t, size=T_TITLE - 0.4, colour=INK,
             font=SANS_B, line=0.96)
        para(s, BX + 0.56, y + 0.30, BW - 0.70, 0.70, b, size=T_BODY - 0.4, colour=BODY,
             line=1.0)

    pill(s, CX, BODY_TOP, CW, "Decisions, and why")
    panel(s, CX, 1.66, CW, 4.36)
    decisions = [
        ("gauge", "The head was chosen by measurement",
         "Regression, binned and head-tail-cut heads were all scored. All three land at "
         "a 0.43–0.49 slope, so the simplest ships."),
        ("shield-check", "Uncertainty is calibrated",
         "ECE 0.063, sigma-vs-error rank correlation +0.866 on held-out tiles."),
        ("cpu", "Export is checked, not assumed",
         "ONNX agrees with PyTorch to 0.0007 cm over 804,972 inputs; int8 costs 0.030 m."),
        ("layers", "Resolution is handled explicitly",
         "Auto-zoom resamples by GSD, recovering all but 3.6% of the coarse-input "
         "penalty."),
    ]
    for i, (ico, t, b) in enumerate(decisions):
        y = 1.78 + i * 1.06
        card(s, CX + 0.10, y, CW - 0.20, 0.98, fill=WHITE)
        badge(s, ico, CX + 0.40, y + 0.26, 0.40, fill=DEEP)
        para(s, CX + 0.68, y + 0.11, CW - 0.82, 0.30, t, size=T_TITLE - 0.4, colour=INK,
             font=SANS_B, line=0.96)
        para(s, CX + 0.20, y + 0.46, CW - 0.40, 0.46, b, size=T_BODY - 0.3, colour=BODY,
             line=1.0)

    pill(s, X0, 6.06, AW, "How it is validated", h=0.30, fill=STEEL, size=T_PILL_S)
    _, tf = textbox(s, X0 + 0.06, 6.42, AW - 0.12, 0.72)
    for i, (k, v) in enumerate([
        ("Split", "region-disjoint, not random — adjacent tiles share buildings."),
        ("Scored on", "80 whole tiles, 11,983,760 pixels, 3,090 buildings."),
        ("Reported per", "urban / sparse / forested / mixed, and per height band."),
        ("Held back", "the test split is untouched until the very end."),
    ]):
        w_(tf, [(f"{k}  ", {"colour": INK, "bold": True, "font": SANS_B}),
                (v, {"colour": BODY})], size=T_BODY - 0.3, first=(i == 0),
           space_after=2, line=1.0)

    # The stack gives up 1.1 in to a QR: the rubric's "prototype evidence" is best met by
    # a code a judge can scan and use, not by another still.
    QD = 0.86
    SW = X1 - BX - QD - 0.30
    pill(s, BX, 6.06, SW, "Technology stack — every dependency free and open",
         h=0.30, fill=STEEL, size=T_PILL_S - 0.6)
    brandstrip(s, BX, 6.44, SW,
               [("python", "Python 3.10"), ("pytorch", "PyTorch"),
                ("huggingface", "Hugging Face"), ("onnx", "ONNX Runtime"),
                ("numpy", "NumPy / SciPy"), ("database", "GDAL / rasterio"),
                ("threedotjs", "three.js"), ("docker", "Docker")])
    qr(s, LIVE_URL, "live_qr", X1 - QD - 0.07, 6.08, QD, "Scan: live demo")


# ------------------------------------------------------------------------- slide 4
def slide_feasibility(s) -> None:
    for sh in by_name(s, "TextBox 8"):
        drop(sh)

    chips = [("circle-check-big", "Technically proven", "3.464 m vs the 5.9 m bar, on LiDAR."),
             ("wifi-off", "Operationally simple", "One image, CPU only, works offline."),
             ("indian-rupee", "Economically viable", "₹0 licences, runs on existing PCs."),
             ("layers", "Already working", "All three ISRO milestones are built.")]
    cw = (W - 0.36) / 4
    for i, (ico, t, b) in enumerate(chips):
        x = X0 + i * (cw + 0.12)
        rounded(s, x, BODY_TOP, cw, 0.62, fill=NAVY, edge=None, radius=0.07)
        badge(s, ico, x + 0.34, BODY_TOP + 0.31, 0.40, fill=EMBER)
        para(s, x + 0.64, BODY_TOP + 0.08, cw - 0.74, 0.22, t, size=10.0, colour=WHITE,
             font=COND, line=0.96, caps=True)
        para(s, x + 0.64, BODY_TOP + 0.32, cw - 0.74, 0.24, b, size=T_MICRO, colour=SKY,
             line=0.98)

    LW, RX = 6.10, 6.86
    RW = X1 - RX

    # ----------------------------------------- accuracy, redrawn from metrics.json
    pill(s, X0, 1.96, LW, "Accuracy across the four landscapes ISRO names", h=0.32,
         size=T_PILL_S)
    panel(s, X0, 2.36, LW, 2.28)
    img(s, "terrain.png", X0 + 0.12, 2.48, 3.48, 3.48 / 2.093, edge=False)
    para(s, X0 + 3.74, 2.50, LW - 3.90, 2.06,
         [("Three of four classes sit at or under 3.3 m", {"colour": INK, "bold": True,
                                                           "font": SANS_B}),
          (", and correlation never drops below +0.77. The fourth is the honest "
           "problem, and it is one specific thing.\n", {"colour": BODY}),
          ("Hilly: ", {"colour": INK, "bold": True, "font": SANS_B}),
          ("DFC2019 has no hills, so Sikkim is scored as a cross-check — risk 1.",
           {"colour": BODY})],
         size=T_BODY - 0.2, line=1.04)

    pill(s, X0, 4.76, LW, "Urban 13.08 m, decomposed", h=0.32, fill=EMBER,
         size=T_PILL_S)
    panel(s, X0, 5.16, LW, 1.98)
    img(s, "error_by_height.png", X0 + 0.12, 5.36, 3.34, 3.34 / 2.510, edge=False)
    para(s, X0 + 3.60, 5.26, LW - 3.76, 1.76,
         [("77% of squared error comes from 1.9% of buildings.",
           {"colour": EMBER, "bold": True, "font": SANS_B}),
          ("  Under 10 m — 94% of the stock — we are at 1.4 m. Urban RMSE is 60 tall "
           "buildings dominating a squared metric.", {"colour": BODY})],
         size=T_BODY - 0.2, line=1.04)

    # ------------------------------------------------------------ risks, answered
    pill(s, RX, 1.96, RW, "Risks, and what answers each", h=0.32, size=T_PILL_S)
    risks = [
        ("No Indian ground truth exists yet",
         [("Sikkim vs Google Open Buildings: bias −6.28 m on the 168 most confident "
           "footprints — agreement, not accuracy. ", {"colour": BODY}),
          ("Mitigated now: ", {"colour": INK, "font": SANS_B}),
          ("auto-zoom recovers all but 3.6% at 0.6 m GSD; terrain leakage ruled out "
           "(r −0.044 on 31-degree slopes).", {"colour": BODY})]),
        ("Tall buildings compress toward the mean",
         [("A data problem: training tops out at 82.8 m, validation reaches 155.3 m. ",
           {"colour": BODY}),
          ("Answered by ", {"colour": INK, "font": SANS_B}),
          ("GAMUS (moved it, p = 0.0002), a control-point power-law fit (−24.9% "
           "tall-tile RMSE), and tall-building LiDAR.", {"colour": BODY})]),
    ]
    y = 2.36
    for t, b in risks:
        rounded(s, RX, y, RW, 0.30, fill=NAVY, edge=None, radius=0.05)
        badge(s, "triangle-alert", RX + 0.20, y + 0.15, 0.24, fill=SAND)
        para(s, RX + 0.40, y + 0.04, RW - 0.52, 0.24, t, size=T_TITLE - 0.6,
             colour=WHITE, font=SANS_B, line=0.96)
        card(s, RX + 0.20, y + 0.34, RW - 0.20, 0.64, fill=WHITE)
        rect(s, RX + 0.20, y + 0.38, 0.06, 0.56, fill=GREEN)
        para(s, RX + 0.36, y + 0.37, RW - 0.48, 0.60, b, size=T_BODY - 0.6, line=1.02)
        y += 1.04

    # --------------------------------------------------- deployment plan, four steps
    pill(s, RX, 4.46, RW, "Deployment plan", h=0.30, fill=DEEP, size=T_PILL_S)
    steps = [("Pilot", "Open Bhoonidhi CartoDEM and LISS-4 tiles; per-terrain metrics."),
             ("Validate", "SAC Cartosat stereo or DGPS turns the Sikkim check into "
                          "accuracy."),
             ("Integrate", "Container behind an internal API; the viewer opens from "
                           "disk."),
             ("Scale", "Batch the archive: every scene gains a height layer.")]
    sw = (RW - 3 * 0.10) / 4
    for i, (t, b) in enumerate(steps):
        x = RX + i * (sw + 0.10)
        card(s, x, 4.84, sw, 0.90, fill=WHITE)
        n = rounded(s, x + 0.08, 4.91, 0.26, 0.26, fill=STEEL, edge=None, radius=0.04)
        n.text_frame.word_wrap = False
        w_(n.text_frame, f"{i + 1}", size=10.0, colour=WHITE, align=PP_ALIGN.CENTER,
           first=True, font=DISPLAY)
        para(s, x + 0.40, 4.90, sw - 0.46, 0.26, t, size=T_TITLE - 0.4, colour=INK,
             font=SANS_B, line=0.96)
        para(s, x + 0.08, 5.20, sw - 0.14, 0.52, b, size=8.0, colour=BODY, line=0.98)
        if i < len(steps) - 1:
            _shape(s, MSO_SHAPE.RIGHT_ARROW, x + sw - 0.01, 5.23, 0.12, 0.14, SKY, None)

    # ------------------------------------------------------- cost, then testing plan
    HW = (RW - 0.12) / 2

    def keyed(x, title, fill, rows):
        pill(s, x, 5.84, HW, title, h=0.28, fill=fill, size=T_PILL_S - 1.0)
        _, tf = textbox(s, x + 0.04, 6.18, HW - 0.08, 0.96)
        for i, (k, v, kc) in enumerate(rows):
            w_(tf, [(f"{k}  ", {"colour": kc, "font": SANS_B}), (v, {"colour": BODY})],
               size=8.3, first=(i == 0), space_after=1.2, line=1.0)

    keyed(RX, "Cost estimate", NAVY, [
        ("Licences", "₹0 — every dependency open source", INK),
        ("Training", "one RTX 3060; last stage 75.5 min", INK),
        ("Compute", "5.0 s per tile: ~1 min per km², CPU", INK),
        ("Hosting", "one two-core CPU server, live today", INK),
        ("Imagery", "already flown and already paid for", INK),
    ])
    keyed(RX + HW + 0.12, "Testing plan", NAVY, [
        ("Done", "region-disjoint: 80 tiles, 3,090 buildings", GREEN),
        ("Done", "second city: DC holdout, 338 m buffer", GREEN),
        ("Done", "ONNX parity; viewer loaded headless", GREEN),
        ("Next", "held-out test split, scored once", STEEL),
        ("Next", "Indian truth: SAC stereo or DGPS", STEEL),
    ])


# ------------------------------------------------------------------------- slide 5
def slide_impact(s) -> None:
    for sh in by_name(s, "TextBox 8"):
        drop(sh)

    LW, RX = 6.10, 6.70
    RW = X1 - RX

    # ------------------------------------------ the theme's use case, built and shown
    pill(s, X0, BODY_TOP, LW, "Disaster use case — built, on Indian terrain", fill=EMBER)
    IH = LW / 2.70
    img(s, "sikkim_flood_d.png", X0, 1.62, LW, IH)
    label(s, X0 + LW - 2.10, 1.62, "Sikkim · live flood tool", fill=NAVY)
    # The viewer's own readout for this frame, redrawn at a size a judge can read.
    rx, ry, rw = X0 + 0.10, 1.62 + IH - 0.50, LW - 0.20
    rounded(s, rx, ry, rw, 0.42, fill=NAVY, edge=None, radius=0.06)
    cw = rw / len(FLOOD_READOUT)
    for i, (k, v) in enumerate(FLOOD_READOUT):
        if i:
            rect(s, rx + i * cw, ry + 0.08, 0.01, 0.26, fill=DEEP)
        para(s, rx + i * cw + 0.10, ry + 0.05, cw - 0.16, 0.34,
             [(v + "  ", {"colour": SAND, "font": DISPLAY, "size": 14}),
              (k, {"colour": WHITE, "font": COND, "size": 8.6})], line=1.0)
    para(s, X0, 1.62 + IH + 0.04, LW, 0.46,
         [("Raise the water and the viewer counts ground under water and buildings "
           "flooded", {"colour": INK, "font": SANS_B}),
          (", and flags the ones too close to call from its own sigma. Ground from "
           "Copernicus GLO-30; a flat-water model, labelled as one. Imagery © Maxar, "
           "CC-BY-4.0.", {"colour": SLATE})], size=T_MICRO, line=1.0)

    # ---------------------------------------------------------------- who it serves
    GY = 4.42
    pill(s, X0, GY, LW, "Who it serves", h=0.30)
    users = [
        ("siren", "Disaster response", "Post-event surfaces without waiting for a "
                                       "stereo pair."),
        ("satellite", "ISRO and SAC", "Height for the single-view Cartosat archive."),
        ("building-2", "Urban planning", "Building heights for FSI and shadow checks "
                                         "(AMRUT)."),
        ("leaf", "Forestry", "Canopy height proxies for biomass and encroachment."),
        ("radio-tower", "Telecom rollout", "Line-of-sight and tower siting without a "
                                           "LiDAR survey."),
    ]
    rows = [users[:3], users[3:]]
    for r, row in enumerate(rows):
        n = len(row)
        cw = (LW - 0.10 * (n - 1)) / n
        y = GY + 0.40 + r * 0.68
        for i, (ico, t, b) in enumerate(row):
            x = X0 + i * (cw + 0.10)
            card(s, x, y, cw, 0.62, fill=WHITE)
            badge(s, ico, x + 0.22, y + 0.20, 0.30, fill=EMBER if r + i == 0 else STEEL)
            para(s, x + 0.44, y + 0.08, cw - 0.52, 0.22, t, size=9.4, colour=INK,
                 font=COND, caps=True, line=0.96)
            para(s, x + 0.10, y + 0.33, cw - 0.18, 0.28, b, size=8.2, colour=BODY,
                 line=0.98)

    # --------------------------------------- before / after, tagged by impact type
    pill(s, RX, BODY_TOP, RW, "Before and after, for the people who use it")
    C1, C2 = 1.40, 1.96
    C3 = RW - C1 - C2 - 0.08
    x1, x2, x3 = RX, RX + C1 + 0.04, RX + C1 + C2 + 0.08
    hy = 1.64
    for x, w, t, f in [(x2, C2, "Today", DEEP), (x3, C3, "With DepthWizard", EMBER)]:
        h = rounded(s, x, hy, w, 0.28, fill=f, edge=None, radius=0.05)
        w_(h.text_frame, t, size=9.6, colour=WHITE, align=PP_ALIGN.CENTER, first=True,
           font=COND, caps=True)
    kinds = {"Social": STEEL, "Economic": SAND, "Environmental": GREEN}
    table = [
        ("Social", "Disaster response", "Wait for a stereo pair or a survey flight.",
         "A surface from one post-event image — about a minute per tile on a CPU web "
         "host."),
        ("Social", "Trust in the number", "One height, no error bar.",
         "Height plus per-pixel sigma; where it says unsure, it is wrong (rank r "
         "+0.866)."),
        ("Economic", "The archive", "Decades of single-view Cartosat with no height.",
         "Every archived scene can gain a height layer, with no new tasking."),
        ("Economic", "Municipal bodies", "Building heights need a survey budget.",
         "Free and open; 36.8 MB runs offline on a laptop CPU."),
        ("Environmental", "Survey flights", "New LiDAR or stereo sorties to fly.",
         "None: it reuses imagery already flown."),
        ("Environmental", "Forest cover", "Canopy height needs airborne LiDAR.",
         "A canopy proxy from optical imagery: 3.29 m RMSE on forested tiles."),
    ]
    RH = 0.64
    for i, (k, name, before, after) in enumerate(table):
        y = hy + 0.34 + i * (RH + 0.04)
        rounded(s, x1, y, C1, RH, fill=MIST, edge=None, radius=0.05)
        kt = tag(s, x1 + 0.08, y + 0.08, 0.10 + len(k) * 0.066, 0.18, k,
                 fill=kinds[k], size=7.2)
        kt.text_frame.margin_left = kt.text_frame.margin_right = 0
        para(s, x1 + 0.08, y + 0.30, C1 - 0.12, 0.36, name, size=9.0, colour=INK,
             font=SANS_B, line=0.96)
        rounded(s, x2, y, C2, RH, fill=WHITE, edge=LINE, radius=0.05)
        para(s, x2 + 0.08, y + 0.07, C2 - 0.16, RH - 0.10, before, size=8.8,
             colour=SLATE, line=1.0)
        rounded(s, x3, y, C3, RH, fill=TINT, edge=None, radius=0.05)
        para(s, x3 + 0.08, y + 0.07, C3 - 0.16, RH - 0.10, after, size=8.8, colour=INK,
             line=1.0)

    # ------------------------------------------------------- impact, in numbers
    stats = [("0", " new flights", "input is imagery already flown and paid for"),
             ("1.4", " m", "error on 94% of buildings — under half a storey"),
             ("1", " min / km²", "on a desktop CPU, single pass*"),
             ("0", " licence fees", "any state agency can adopt it")]
    cw = (W - 0.36) / 4
    for i, (v, u, cap) in enumerate(stats):
        stat_chip(s, X0 + i * (cw + 0.12), 6.22, cw, 0.72, v, u, cap, fill=MIST,
                  vcolour=EMBER)
    para(s, X0, 6.97, W, 0.18,
         [("Scales by batch, on CPUs agencies already own: the whole archive, not one "
           "scene.", {"colour": INK, "bold": True, "font": SANS_B}),
          ("   *Derived: 5.0 s per 1024 px tile (0.094 km²) measured on a desktop CPU.",
           {"colour": SLATE})], size=T_MICRO - 0.4, line=1.0)


# ------------------------------------------------------------------------- slide 6
def slide_references(s) -> None:
    for sh in by_name(s, "TextBox 8"):
        drop(sh)

    AW, BX, CX = 3.98, 4.60, 8.80
    BW = 3.98
    CW = X1 - CX

    def stack_col(x, w, top, title, items, fill=NAVY, row=0.42):
        y = pill(s, x, top, w, title, h=0.30, fill=fill, size=T_PILL_S)
        panel(s, x, y, w, 0.10 + len(items) * row)
        for i, (n, d) in enumerate(items):
            para(s, x + 0.12, y + 0.05 + i * row, w - 0.24, row - 0.04,
                 [(n + "  ", {"colour": INK, "bold": True, "font": SANS_B,
                              "size": T_TITLE - 0.9}),
                  (d, {"colour": BODY, "size": T_BODY - 0.4})], line=0.98)
        return y + 0.14 + len(items) * row

    y = stack_col(X0, AW, BODY_TOP, "Research that changed what we built", [
        ("Depth Anything V2", "Yang et al., NeurIPS 2024. The backbone we fine-tune."),
        ("HTC-DC Net", "Chen et al., TGRS 2023. Head-tail cut, distribution constraints."),
        ("Depth Any Canopy", "Rege Cambrin et al., ECCV-W 2024. The recipe for aerial height."),
        ("Beta-NLL", "Seitzer et al., ICLR 2022. Keeps the mean head learning."),
        ("IM2HEIGHT, TSE-Net", "Prior single-view height — our baseline for the field."),
    ])
    stack_col(X0, AW, y + 0.12, "Data and reference surfaces", [
        ("DFC2019 Track 1", "US3D at 0.3 m GSD with airborne LiDAR truth."),
        ("GAMUS", "ISRO's recommended set. 6,204 tiles, 0.33 m, DC and Philadelphia."),
        ("Copernicus GLO-30", "Free 30 m global DEM; the absolute metric anchor."),
        ("GlobalBuildingAtlas", "Published 5.9 m RMSE over Asia — the external bar."),
        ("Google Open Buildings", "A cross-check over India, never used as truth."),
        ("Maxar Open Data", "Sikkim imagery, CC BY-NC 4.0, © Maxar Technologies."),
    ])

    y = stack_col(BX, BW, BODY_TOP, "Tried, measured, rejected", [
        ("Ordinal / binned head", "All heads land at a 0.43–0.49 slope. No gain."),
        ("Shadow photogrammetry", "Oracle shadows reach r 0.503; the net reaches 0.787."),
        ("LDS tail reweighting", "Pixels above 20 m are 7.5% but carry 78.8% of error."),
        ("Two-model ensemble", "Error correlation 0.925 — averaging buys nothing."),
        ("Global de-compression", "A rescale moves error, it does not remove it."),
    ], fill=EMBER)
    stack_col(BX, BW, y + 0.12, "Probes we ran, and what each closed", [
        ("01 Tall buildings", "Data, not capacity. No open set at 0.3 m reaches 50 m."),
        ("03 Guided filtering", "Squaring off roofs made every metric worse."),
        ("04 Where detail went", "Detail follows the ground area one token covers."),
        ("05 Eval resolution", "Cartosat-class input costs 3.6% after auto-zoom."),
        ("06 Height compression", "Measured: 0.473 × truth + 2.66 m."),
    ], fill=STEEL)

    pill(s, CX, BODY_TOP, CW, "How we compare", h=0.30, size=T_PILL_S)
    rows = [
        ("One archived image", ("x", "check", "check")),
        ("Absolute metric height", ("check", "minus", "check")),
        ("Per-pixel uncertainty", ("x", "x", "check")),
        ("Scored per terrain", ("minus", "x", "check")),
        ("Runs with no GPU", ("minus", "x", "check")),
        ("3D viewer shipped", ("check", "x", "check")),
        ("Failure modes published", ("minus", "minus", "check")),
    ]
    end = compare_table(s, CX, 1.62, CW, ("Stereo or LiDAR survey",
                                          "Published single-view", "DepthWizard"),
                        rows, param_w=CW * 0.46, head_h=0.44, row_h=0.32)

    legend = [("check", "yes"), ("minus", "partial"), ("x", "no")]
    lx = CX + 0.04
    for k, t in legend:
        mark(s, k, lx + 0.09, end + 0.13, d=0.17)
        para(s, lx + 0.22, end + 0.03, 1.30, 0.22, t, size=T_MICRO, colour=SLATE)
        lx += 0.44 + len(t) * 0.052
    para(s, CX + 0.06, end + 0.28, CW - 0.12, 0.36,
         [("Two rows the alternatives win. ", {"colour": INK, "bold": True,
                                               "font": SANS_B}),
          ("A stereo survey gives absolute height directly and ships mature viewers — "
           "the table is only useful to a panel if the rows we lose are still in it.",
           {"colour": BODY})], size=T_MICRO - 0.2, line=1.0)

    QW = 0.92
    py = end + 0.80
    pill(s, CX, py, CW, "The bar we are measured against", h=0.30, fill=NAVY,
         size=T_PILL_S)
    para(s, CX + 0.06, py + 0.36, CW - QW - 0.24, 1.08,
         [("3.464 m", {"colour": EMBER, "bold": True, "font": DISPLAY, "size": 14}),
          ("  per building, against LiDAR, versus 5.9 m published for Asia. 3,090 "
           "buildings, 80 held-out tiles.\n", {"colour": BODY}),
          ("Where it fails: ", {"colour": INK, "bold": True, "font": SANS_B}),
          ("above 20 m we under-call by 15.6 m. We pre-registered −13 m and missed it; "
           "GAMUS moved it significantly (p = 0.0002).", {"colour": BODY})],
         size=T_BODY - 0.5, line=1.0)

    qx, qy = X1 - QW, py + 0.38
    if PITCH_VIDEO_URL:
        qr(s, PITCH_VIDEO_URL, "pitch_qr", qx, qy, QW, "Scan: pitch video")
    else:
        box = rounded(s, qx, qy, QW, QW, fill=MIST, edge=SLATE, radius=0.05)
        box.line.dash_style = MSO_LINE_DASH_STYLE.DASH
        box.line.width = Pt(1.0)
        para(s, qx, qy + 0.28, QW, 0.50, "PITCH\nVIDEO\nQR", size=8.0, colour=SLATE,
             align=PP_ALIGN.CENTER, font=COND, line=1.06)

    tf = None
    for k, v in [("Live  ", "project5.zaidansari.tech"),
                 ("Docs  ", "project5.zaidansari.tech/documentation"),
                 ("Code  ", "github.com/zaidnansari2011/sih2026-depthwizard")]:
        spans = [(k, {"colour": EMBER, "font": COND, "size": 9.0}),
                 (v, {"colour": STEEL, "font": SANS_B})]
        if tf is None:
            tf = para(s, CX, 6.77, CW, 0.38, spans, size=7.8, line=0.98)
        else:
            w_(tf, spans, size=7.8, line=0.98)


def link_runs(prs) -> int:
    """Turn every run whose text is a known address into a real hyperlink."""
    n = 0
    for s in prs.slides:
        for sh in s.shapes:
            if not sh.has_text_frame:
                continue
            for p in sh.text_frame.paragraphs:
                for r in p.runs:
                    url = LINKS.get(r.text.strip())
                    if url:
                        r.hyperlink.address = url
                        # A hyperlink takes the theme's link colour unless told otherwise.
                        r.font.color.rgb = STEEL
                        n += 1
    return n


# ------------------------------------------------------------------------------ main
def main() -> None:
    prs = Presentation(str(SRC))

    xml_slides = prs.slides._sldIdLst          # slide 7 is the template's own note sheet
    for sid in list(xml_slides)[6:]:
        prs.part.drop_rel(sid.rId)
        xml_slides.remove(sid)

    builders = [slide_title, slide_problem_solution, slide_technical,
                slide_feasibility, slide_impact, slide_references]
    for s, build in zip(prs.slides, builders):
        trim_footer(s)
        for oval in by_name(s, "Oval"):
            if oval.has_text_frame and oval.text_frame.paragraphs[0].runs:
                oval.text_frame.paragraphs[0].runs[0].text = TEAM
        build(s)

    n = link_runs(prs)
    assert n == 6, f"expected 6 linked addresses (3 on slide 1, 3 on slide 6), got {n}"
    prs.save(str(DST))
    print(f"  wrote {DST}")


if __name__ == "__main__":
    main()
