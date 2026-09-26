"""Deck E: deck D, restructured to what ISRO's own FAQ asks of the initial submission.

On 22 Sep ISRO answered, in the problem statement's own repository (docs/isro-faq.md):

    "For the initial submission, include only the problem understanding, implementation
     idea, proposed architecture, and any preliminary work you have done in one extra
     slide. However, after selection, more detailed technical documentation will be
     needed."

Deck D spread measured results across every slide. Deck E keeps the six template slides
about the idea and gathers the evidence onto a seventh:

  slide 1  identification only; the accuracy chips become what goes in and comes out.
  slide 2  unchanged in structure (problem, solution, the idea running), numbers removed.
  slide 3  "Decisions, and why" -- which quoted metrics -- becomes the design choices,
           including the one ISRO's FAQ now makes central: Cartosat-2S at 0.6 m.
  slide 4  the accuracy charts move to slide 7. Their place is the hard engineering
           problems (0.6 m, panchromatic input, trust, absolute height, no GPU) and how
           each is handled -- including the one not yet measured.
  slide 5  impact without accuracy figures; the flood use case stays.
  slide 6  references and positioning; the headline box moves to slide 7.
  slide 7  NEW. Preliminary work: built, live and measured -- every number that used to
           be spread across the deck, plus probe 05b (0.6 m on the shipped model), read
           straight from its results file so the slide cannot disagree with the run.

Deck D is untouched. Every figure still traces to docs/evidence-pack.md,
docs/deployment.md, the shipped metrics file or out/gsd_recheck_run07/results.json.

    python tools/ppt/prep_assets_d.py          # charts and crops (shared with deck D)
    python tools/gsd_recheck.py --ckpt ../checkpoints/run07/best.pt   # probe 05b
    python tools/ppt/make_deck_e.py
    python tools/ppt/check_layout_b.py E
    python tools/ppt/export_pdf.py E
"""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.dml import MSO_LINE_DASH_STYLE
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN
from pptx.opc.constants import RELATIONSHIP_TYPE as RT
from pptx.util import Inches, Pt

from build_deck import by_name, drop, textbox, trim_footer
from build_deck_b import (BODY, BODY_TOP, COND, DEEP, DISPLAY, EMBER, GREEN, INK, LINE,
                          MIST, NAVY, RAMP, SAND, SANS, SANS_B, SKY, SLATE, SRC, STEEL,
                          T_BODY, T_LABEL, T_MICRO, T_PILL_S, T_TITLE, TEAM, W, WHITE, X0,
                          X1, _shape, arch_layer, arrow_down, badge, brandstrip, card,
                          compare_table, mark, panel, para, pill, qr_png, rect, rounded,
                          stat_chip, tag, w_)
from make_deck_b import PS_ID, PS_THEME, PS_TITLE, TEAM_ID
from prep_assets_d import E_HEIGHT, E_TERRAIN

DST = Path("D:/sih2026/depthwizard/docs/SIH2026-DevUp-SIH26175-DepthWizard-E.pptx")
FIG = Path("D:/sih2026/depthwizard/tools/ppt/figures/d")
GSD_RESULTS = Path("D:/sih2026/out/gsd_recheck_run07/results.json")

LIVE_URL = "https://project5.zaidansari.tech"
CODE_URL = "https://github.com/zaidnansari2011/sih2026-depthwizard"
DOCS_URL = "https://project5.zaidansari.tech/documentation/"
EVIDENCE_URL = CODE_URL + "/blob/master/docs/evidence-pack.md"
# The full path wraps into the footer bar; the shown text is shortened, the link is not.
EVIDENCE_TEXT = "github.com/…/sih2026-depthwizard/docs/evidence-pack.md"

# Shown text -> target. PowerPoint's PDF export drops run hyperlinks unreliably, so
# tools/ppt/export_pdf.py also writes these into the PDF itself from this table.
LINKS = {"project5.zaidansari.tech": LIVE_URL,
         "project5.zaidansari.tech/documentation": DOCS_URL,
         EVIDENCE_TEXT: EVIDENCE_URL}
# The hosted pitch clip. Set to None and slide 6 draws the reserved slot instead.
PITCH_VIDEO_URL = "https://youtu.be/4NpR74MS5oE"

TINT = RGBColor(0xE7, 0xEF, 0xF4)          # our column in every comparison

# Read off the sikkim_flood capture's HUD (tools/ppt/shots/sikkim_flood.png). If that
# capture is retaken these must be re-read from it: they describe that one frame.
FLOOD_READOUT = [("WATER LEVEL", "1,143 m"), ("GROUND UNDER WATER", "20.5 %"),
                 ("BUILDINGS FLOODED", "44")]


def gsd_result() -> dict:
    """Probe 05b, read from its results file rather than typed in."""
    if not GSD_RESULTS.exists():
        raise FileNotFoundError(f"{GSD_RESULTS} missing; run tools/gsd_recheck.py first")
    r = json.loads(GSD_RESULTS.read_text())
    a, b, c = r["A_native_0.3m"], r["B_0.6m_untreated"], r["C_0.6m_autozoom"]
    # The sanity gate from docs/probe-05b-gsd-run07.md: a run that cannot reproduce the
    # shipped number measured something else, and must not reach a slide.
    assert abs(a["per_building_rmse"] - 3.464) <= 0.01, (
        f"probe 05b failed its sanity gate (A = {a['per_building_rmse']:.3f}); void")
    return {"A": a, "B": b, "C": c,
            "cost": 100 * (c["per_building_rmse"] / a["per_building_rmse"] - 1),
            "cost_b": 100 * (b["per_building_rmse"] / a["per_building_rmse"] - 1)}


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


def clone_slide(prs, src):
    """Append a copy of a template slide, chrome and all, before anything is drawn on it.

    python-pptx has no duplicate. The extra slide must carry the template's own furniture
    -- team oval, SIH mark, footer bar, page number -- or it reads as bolted on. Shapes are
    copied as XML; the SIH mark is a picture, so its image relationship is re-created on
    the new slide rather than pointing at an rId that only exists on the source.
    """
    dst = prs.slides.add_slide(src.slide_layout)
    for ph in list(dst.placeholders):
        ph._element.getparent().remove(ph._element)
    for sh in src.shapes:
        dst.shapes._spTree.insert_element_before(copy.deepcopy(sh._element), "p:extLst")
    A = "{http://schemas.openxmlformats.org/drawingml/2006/main}blip"
    R = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed"
    for blip in dst.shapes._spTree.iter(A):
        rid = blip.get(R)
        if rid:
            blip.set(R, dst.part.relate_to(src.part.related_part(rid), RT.IMAGE))
    return dst


def set_title(s, text: str) -> None:
    for sh in by_name(s, "Title"):
        runs = [r for p in sh.text_frame.paragraphs for r in p.runs]
        if runs:
            runs[0].text = text
            for r in runs[1:]:
                r.text = ""


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

    # What goes in and what comes out -- the idea in three numbers that are definitions,
    # not measurements. The measurements are on slide 7, where ISRO asked for them.
    pill(s, PX, 5.00, PW, "A working prototype — measured results on slide 7", h=0.34,
         fill=EMBER, size=T_PILL_S)
    stats = [("1", " image", "in: GeoTIFF, or plain\nPNG / JPG"),
             ("2", " layers", "out: metric height and\nper-pixel confidence"),
             ("0", " GPUs", "needed to run it or\nits 3D viewer")]
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
          ("Height, how sure we are, and a scene to check it in.",
           {"colour": EMBER, "bold": True})], size=12.5, line=1.04)

    LW, RX = 4.30, 5.00
    RW = X1 - RX
    TOP, BOT = 1.64, 4.62
    pill(s, X0, 1.22, LW, "The problem", fill=NAVY)
    pill(s, RX, 1.22, RW, "Our solution", fill=EMBER)

    # ----------------------------------------------------------------- the problem
    panel(s, X0, TOP, LW, BOT - TOP)
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
    tg = tag(s, X0 + 0.14, 4.35, 1.02, 0.20, "Who feels it", fill=NAVY, size=7.6)
    tg.text_frame.margin_left = tg.text_frame.margin_right = 0
    para(s, X0 + 1.22, 4.32, LW - 1.32, 0.24,
         "SAC analysts · disaster responders · planners · foresters", size=8.4,
         colour=INK, font=SANS_B)

    # ---------------------------------------------------------------- the solution
    panel(s, RX, TOP, RW, BOT - TOP)
    items = [
        ("image-up", "One image to a metric DSM",
         "Depth Anything V2, fine-tuned on airborne-LiDAR heights. GeoTIFF out, "
         "metadata or not."),
        ("shield-check", "Confidence, not just height",
         "A per-pixel sigma beside every height, so the model says where it is unsure."),
        ("box", "A navigable 3D scene",
         "three.js flythrough with measure, flood and drag-to-compare-against-LiDAR "
         "tools."),
        ("ruler", "Validation inside the product",
         "RMSE, MAE and correlation against LiDAR — per terrain class, not one "
         "flattering average."),
        ("map-pinned", "Absolute heights, two ways",
         "Copernicus GLO-30 for any GeoTIFF, or ground control points via a power-law "
         "fit."),
        ("wifi-off", "Deployable, not a demo",
         "No GPU needed. The viewer is one HTML file that opens from disk, offline."),
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

    # ------------------------------------------------ the idea, running, before -> after
    pill(s, X0, 4.72, W, "The idea, running: one archived image in, a measurable surface "
                         "out — captures of our prototype, not mock-ups", h=0.30,
         fill=DEEP, size=T_PILL_S)
    IY, IH = 5.10, 1.74
    img(s, "omaha_input.png", X0, IY, IH, IH)
    label(s, X0, IY, "Before · input", fill=SLATE)
    _shape(s, MSO_SHAPE.RIGHT_ARROW, X0 + IH + 0.07, IY + IH / 2 - 0.17, 0.30, 0.34,
           EMBER, None)
    AW_ = 3.30
    ax = X0 + IH + 0.44
    panels = [("omaha_height_d.png", "After · height, in 3D",
               "Same tile: roofs, tree crowns and kerbs at 0.3 m."),
              ("omaha_sigma_d.png", "After · confidence",
               "Where the estimate is weak, the sigma says so.")]
    for i, (fn, cap, sub) in enumerate(panels):
        x = ax + i * (AW_ + 0.14)
        img(s, fn, x, IY, AW_, IH)
        label(s, x, IY, cap, fill=NAVY)
        para(s, x, 6.88, AW_, 0.20, sub, size=T_MICRO, colour=SLATE, line=0.98)
    para(s, X0, 6.88, IH + 0.3, 0.20, "One Omaha tile, as flown.", size=T_MICRO,
         colour=SLATE, line=0.98)
    sx = X1 - AW_
    rect(s, sx - 0.13, IY + 0.10, 0.02, IH - 0.20, fill=LINE)
    img(s, "sikkim_hilly_d.png", sx, IY, AW_, IH)
    label(s, sx, IY, "Sikkim, India · hilly", fill=DEEP)
    para(s, sx, 6.88, AW_, 0.20,
         [("967 m of relief across 2 km, draped.", {"colour": SLATE}),
          ("  Imagery © Maxar, CC-BY-4.0", {"colour": SLATE, "size": 7.0})],
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
                         ("satellite", "Cartosat-2S,\n0.6 m")]),
        ("Normalisation", [("ruler", "Ground sample distance\nread from metadata"),
                           ("layers", "Auto-zoom and resample\nto a 0.3 m scale")]),
        ("Model core — Depth Anything V2", [("cpu", "ViT encoder"),
                                            ("workflow", "DPT neck"),
                                            ("database", "Fine-tuned on\nDFC2019 + GAMUS")]),
        ("Dual head", [("mountain-snow", "Height mu,\nper pixel"),
                       ("shield-check", "Log-variance sigma,\nper pixel")]),
        ("Post-process", [("box", "518 px tiling,\noverlap blend, TTA"),
                          ("map-pinned", "Scale anchor: GLO-30\nor ground control")]),
        ("Output layer", [("file-text", "Absolute DSM /\nrelative DSM"),
                          ("gauge", "Sigma map"),
                          ("monitor", "three.js scene,\nlive error map")]),
    ]
    y = 1.76
    for i, (lbl, items) in enumerate(layers):
        y = arch_layer(s, X0 + 0.12, y, AW - 0.24, lbl, items, head=0.22, band=0.38,
                       fill=EMBER if i == len(layers) - 1 else NAVY)
        if i < len(layers) - 1:
            y = arrow_down(s, X0 + AW / 2, y + 0.02, h=0.09) + 0.02

    pill(s, BX, BODY_TOP, BW, "How it runs — hardware, API, data")
    panel(s, BX, 1.66, BW, 4.36)
    stages = [
        ("Trained on one consumer GPU",
         "RTX 3060, 12 GB. Backbone: Depth Anything V2 Small, Apache 2.0, 24.8 M "
         "parameters."),
        ("Exported for CPU",
         "ONNX, int8-quantised to 36.8 MB. Inference needs no GPU at all."),
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

    # Design choices, argued rather than scored: the scores are on slide 7.
    pill(s, CX, BODY_TOP, CW, "Design choices, and why")
    panel(s, CX, 1.66, CW, 4.36)
    decisions = [
        ("gauge", "Metres learned from LiDAR",
         "Fine-tuning on airborne-LiDAR heights makes the output metric directly; a DEM "
         "supplies the ground under it."),
        ("shield-check", "Uncertainty as an output",
         "A twin head predicts sigma beside every height (heteroscedastic NLL with "
         "beta-NLL), so doubt is visible."),
        ("layers", "Built for ISRO's 0.6 m",
         "Auto-zoom reads a GeoTIFF's resolution and resamples Cartosat-2S to the scale "
         "the model learned."),
        ("cpu", "Chosen by measurement",
         "Regression, binned and head-tail-cut heads were all trained and scored; the "
         "simplest held its own, so it ships."),
    ]
    for i, (ico, t, b) in enumerate(decisions):
        y = 1.78 + i * 1.06
        card(s, CX + 0.10, y, CW - 0.20, 0.98, fill=WHITE)
        badge(s, ico, CX + 0.40, y + 0.26, 0.40, fill=DEEP)
        para(s, CX + 0.68, y + 0.11, CW - 0.82, 0.30, t, size=T_TITLE - 0.4, colour=INK,
             font=SANS_B, line=0.96)
        para(s, CX + 0.20, y + 0.46, CW - 0.40, 0.48, b, size=T_BODY - 0.5, colour=BODY,
             line=1.0)

    pill(s, X0, 6.06, AW, "How it is validated", h=0.30, fill=STEEL, size=T_PILL_S)
    _, tf = textbox(s, X0 + 0.06, 6.42, AW - 0.12, 0.72)
    for i, (k, v) in enumerate([
        ("Split", "region-disjoint, not random — adjacent tiles share buildings."),
        ("Scored on", "whole held-out tiles against airborne LiDAR, never on crops."),
        ("Reported per", "urban / sparse / forested / mixed, and per height band."),
        ("Held back", "the test split is untouched until the very end."),
    ]):
        w_(tf, [(f"{k}  ", {"colour": INK, "bold": True, "font": SANS_B}),
                (v, {"colour": BODY})], size=T_BODY - 0.3, first=(i == 0),
           space_after=2, line=1.0)

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

    chips = [("circle-check-big", "Technically proven", "Built and measured: slide 7."),
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

    # ------------------------------ the hard engineering problems, and the answers
    pill(s, X0, 1.96, LW, "The hard problems, and how each is handled", h=0.32,
         size=T_PILL_S)
    fit = [
        ("ISRO judges on 0.6 m Cartosat-2S; open LiDAR training imagery is 0.3 m",
         "Auto-zoom reads the GeoTIFF's pixel size and resamples to the scale the model "
         "learned. Its cost at 0.6 m is measured on the shipped model (slide 7); "
         "fine-tuning on 0.6 m-degraded tiles is next."),
        ("Cartosat-2S colour is not the aerial colour the model learned from",
         "Training jitters brightness, contrast and gamma to mimic sensor and haze; any "
         "band count loads. The cost on real Cartosat colour is not yet measured."),
        ("One image has no stereo parallax, so which heights can be trusted?",
         "Every pixel ships with an uncertainty (σ), calibrated against LiDAR. Unsure "
         "pixels are flagged for the analyst, not hidden."),
        ("An absolute DSM needs ground elevation a single image cannot give",
         "Copernicus GLO-30 terrain plus our above-ground heights, on by default. "
         "GLO-30's own error (< 4 m, LE90) is inherited, and stated."),
        ("Field offices have no GPU and often no reliable internet",
         "Inference runs on an ordinary desktop CPU, about a minute per km²; results "
         "open in a 3-D viewer that ships as one offline HTML file."),
    ]
    y = 2.36
    RH = 0.92
    for q, a in fit:
        card(s, X0, y, LW, RH - 0.06, fill=WHITE)
        rect(s, X0, y + 0.06, 0.06, RH - 0.18, fill=EMBER)
        para(s, X0 + 0.18, y + 0.07, LW - 0.30, 0.24,
             [("PROBLEM  ", {"colour": EMBER, "font": COND, "size": 9.4}),
              (q, {"colour": INK, "font": SANS_B})], size=9.4, line=0.96)
        para(s, X0 + 0.18, y + 0.33, LW - 0.30, 0.48,
             [("HANDLED  ", {"colour": GREEN, "font": COND, "size": 9.4}),
              (a, {"colour": BODY})], size=T_BODY - 0.5, line=1.02)
        y += RH

    # ------------------------------------------------------------ risks, answered
    pill(s, RX, 1.96, RW, "Open risks, and what answers each", h=0.32, size=T_PILL_S)
    risks = [
        ("No Indian ground truth exists yet",
         [("Over India we can only compare against another model, Google Open Buildings, "
           "so we report agreement, not accuracy. ", {"colour": BODY}),
          ("Answered by ", {"colour": INK, "font": SANS_B}),
          ("SAC stereo or DGPS control points in the pilot (below).", {"colour": BODY})]),
        ("Tall buildings compress toward the mean",
         [("Open training data stops near 83 m, so towers are under-called. ",
           {"colour": BODY}),
          ("Answered by ", {"colour": INK, "font": SANS_B}),
          ("ISRO's GAMUS (it moved the gap significantly), a control-point power-law "
           "fit, and tall-building LiDAR.", {"colour": BODY})]),
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

    HW = (RW - 0.12) / 2

    def keyed(x, title, fill, rows):
        pill(s, x, 5.84, HW, title, h=0.28, fill=fill, size=T_PILL_S - 1.0)
        _, tf = textbox(s, x + 0.04, 6.18, HW - 0.08, 0.96)
        for i, (k, v, kc) in enumerate(rows):
            w_(tf, [(f"{k}  ", {"colour": kc, "font": SANS_B}), (v, {"colour": BODY})],
               size=8.3, first=(i == 0), space_after=1.2, line=1.0)

    keyed(RX, "Cost estimate", NAVY, [
        ("Licences", "₹0 — every dependency open source", INK),
        ("Training", "one consumer RTX 3060 GPU", INK),
        ("Compute", "~1 min per km² on a desktop CPU*", INK),
        ("Hosting", "one two-core CPU server, live today", INK),
        ("Imagery", "already flown and already paid for", INK),
    ])
    keyed(RX + HW + 0.12, "Testing plan", NAVY, [
        ("Done", "region-disjoint LiDAR validation", GREEN),
        ("Done", "second city, with a 338 m buffer", GREEN),
        ("Done", "ISRO's 0.6 m, on the shipped model", GREEN),
        ("Next", "held-out test split, scored once", STEEL),
        ("Next", "Indian truth: SAC stereo or DGPS", STEEL),
    ])
    para(s, RX, 7.00, RW, 0.14,
         "*Derived: 5.0 s per 1024 px tile (0.094 km²), single pass, measured on a "
         "desktop CPU.", size=7.0, colour=SLATE, line=1.0)


# ------------------------------------------------------------------------- slide 5
def slide_impact(s) -> None:
    for sh in by_name(s, "TextBox 8"):
        drop(sh)

    LW, RX = 6.10, 6.70
    RW = X1 - RX

    pill(s, X0, BODY_TOP, LW, "Disaster use case — built, on Indian terrain", fill=EMBER)
    IH = LW / 2.70
    img(s, "sikkim_flood_d.png", X0, 1.62, LW, IH)
    label(s, X0 + LW - 2.10, 1.62, "Sikkim · live flood tool", fill=NAVY)
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
    for r, row in enumerate([users[:3], users[3:]]):
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
         "Height plus a per-pixel sigma: where it says unsure, it is wrong."),
        ("Economic", "The archive", "Decades of single-view Cartosat with no height.",
         "Every archived scene can gain a height layer, with no new tasking."),
        ("Economic", "Municipal bodies", "Building heights need a survey budget.",
         "Free and open; 36.8 MB runs offline on a laptop CPU."),
        ("Environmental", "Survey flights", "New LiDAR or stereo sorties to fly.",
         "None: it reuses imagery already flown."),
        ("Environmental", "Forest cover", "Canopy height needs airborne LiDAR.",
         "A canopy-height proxy from optical imagery alone."),
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

    stats = [("0", " new flights", "input is imagery already flown and paid for"),
             ("1", " image", "no stereo pair, no revisit wait"),
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
    g = gsd_result()

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
        ("Depth Any Canopy", "Ouaknine et al., 2024. The recipe for aerial height."),
        ("Beta-NLL", "Seitzer et al., ICLR 2022. Keeps the mean head learning."),
        ("IM2HEIGHT, TSE-Net", "Prior single-view height — our baseline for the field."),
    ])
    stack_col(X0, AW, y + 0.12, "Data and reference surfaces", [
        ("DFC2019 Track 1", "US3D at 0.3 m GSD with airborne LiDAR truth."),
        ("GAMUS", "ISRO's recommended set. 6,204 tiles, 0.33 m, DC and Philadelphia."),
        ("Copernicus GLO-30", "Free 30 m global DEM; the absolute metric anchor."),
        ("GlobalBuildingAtlas", "Published 5.9 m RMSE over Asia — the external bar."),
        ("Google Open Buildings", "A cross-check over India, never used as truth."),
        ("Maxar Open Data", "Sikkim imagery, CC-BY-4.0, © Maxar Technologies."),
    ])

    y = stack_col(BX, BW, BODY_TOP, "Tried, measured, rejected", [
        ("Ordinal / binned head", "All heads land at a 0.43–0.49 slope. No gain."),
        ("Shadow photogrammetry", "Oracle shadows reach r 0.503; the net reaches 0.787."),
        ("LDS tail reweighting", "Pixels above 20 m are 7.5% but carry 78.8% of error."),
        ("Two-model ensemble", "Error correlation 0.925 — averaging buys nothing."),
        ("Global de-compression", "A rescale moves error, it does not remove it."),
    ], fill=EMBER)
    stack_col(BX, BW, y + 0.12, "Probes we ran, and what each closed", [
        ("01 Tall buildings", "Data, not capacity. Open 0.3 m sets hold few towers."),
        ("03 Guided filtering", "Squaring off roofs made every metric worse."),
        ("04 Where detail went", "Detail follows the ground area one token covers."),
        ("05b ISRO's 0.6 m", f"Shipped model, 80 tiles: {g['cost']:+.0f}% per building "
                             f"after auto-zoom."),
        ("06 Height compression", "Measured: 0.473 × truth + 2.66\u00a0m."),
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
    pill(s, CX, py, CW, "Evidence behind every number", h=0.30, fill=NAVY,
         size=T_PILL_S)
    para(s, CX + 0.06, py + 0.36, CW - QW - 0.24, 1.08,
         [("Slide 7 is our preliminary work. ", {"colour": INK, "font": SANS_B}),
          ("Each figure on it regenerates from the evidence pack in the repository, and "
           "the documentation site walks through data, training, evaluation and "
           "failures in full.", {"colour": BODY})],
         size=T_BODY - 0.5, line=1.02)
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
                 ("Evidence  ", EVIDENCE_TEXT)]:
        spans = [(k, {"colour": EMBER, "font": COND, "size": 9.0}),
                 (v, {"colour": STEEL, "font": SANS_B})]
        if tf is None:
            tf = para(s, CX, 6.77, CW, 0.38, spans, size=7.8, line=0.98)
        else:
            w_(tf, spans, size=7.8, line=0.98)


# ------------------------------------------------------------------------- slide 7
def slide_preliminary(s) -> None:
    """The one extra slide ISRO's FAQ allows: what is already built, live and measured."""
    for sh in by_name(s, "TextBox 8"):
        drop(sh)
    set_title(s, "PRELIMINARY WORK")
    g = gsd_result()
    A, B, C = g["A"], g["B"], g["C"]

    # ------------------------------------------------------------- the headline row
    chips = [("3.464", " m", "per-building RMSE vs airborne LiDAR;\n5.9 m published "
                             "for Asia"),
             ("6.008", " m", "whole-tile RMSE on 80 held-out\ntiles from unseen regions"),
             ("0.063", "", "calibration error of the per-pixel\nuncertainty (ECE)"),
             (f"{g['cost']:+.0f}", " %", "per-building error at ISRO's 0.6 m,\nafter "
                                         "auto-zoom (80 tiles)")]
    cw = (W - 0.36) / 4
    for i, (v, u, cap) in enumerate(chips):
        stat_chip(s, X0 + i * (cw + 0.12), BODY_TOP, cw, 0.82, v, u, cap,
                  vcolour=EMBER if i == 3 else SAND)

    LW, RX = 6.10, 6.86
    RW = X1 - RX

    # ------------------------------------------------ accuracy, from metrics.json
    pill(s, X0, 2.14, LW, "Accuracy across the four landscapes ISRO names", h=0.30,
         size=T_PILL_S)
    panel(s, X0, 2.50, LW, 2.16)
    img(s, "terrain_e.png", X0 + 0.10, 2.57, *E_TERRAIN, edge=False)
    para(s, X0 + 3.56, 2.60, LW - 3.70, 1.98,
         [("Three of four classes at or under 3.3 m", {"colour": INK, "font": SANS_B}),
          (", correlation never below +0.77. Urban is one specific thing: the "
           "buildings above 20 m.\n", {"colour": BODY}),
          ("Hilly: ", {"colour": INK, "font": SANS_B}),
          ("DFC2019 has no hills, so Sikkim is a cross-check, not a score.",
           {"colour": BODY})],
         size=T_BODY - 0.4, line=1.03)

    pill(s, X0, 4.76, LW, "Where the error lives", h=0.30, fill=EMBER, size=T_PILL_S)
    panel(s, X0, 5.12, LW, 2.02)
    img(s, "error_by_height_e.png", X0 + 0.10, 5.19, *E_HEIGHT, edge=False)
    para(s, X0 + 3.60, 5.22, LW - 3.74, 1.86,
         [("77% of squared error comes from 1.9% of buildings.",
           {"colour": EMBER, "font": SANS_B}),
          (" Under 10 m, 94% of the stock, we are at 1.4 m. ", {"colour": BODY}),
          ("Above 20 m we under-call by 15.6 m; ", {"colour": INK, "font": SANS_B}),
          ("we pre-registered −13 m and missed it.", {"colour": BODY})],
         size=T_BODY - 0.4, line=1.03)

    # --------------------------------------------------- ISRO's 0.6 m, re-measured
    pill(s, RX, 2.14, RW, "ISRO's 0.6 m Cartosat-2S, simulated on the shipped model",
         h=0.30, size=T_PILL_S)
    heads = ["Input", "Per building", "Buildings ≤ 20 m", "Whole tile"]
    rows = [("0.3 m, native", A), ("0.6 m, untreated", B), ("0.6 m, auto-zoom", C)]
    cx = [RX, RX + 1.86, RX + 3.30, RX + 4.74]
    cws = [1.82, 1.40, 1.40, RW - 4.74]
    yy = 2.50
    for x, w, h in zip(cx, cws, heads):
        hb = rounded(s, x, yy, w - 0.04, 0.26, fill=DEEP, edge=None, radius=0.04)
        w_(hb.text_frame, h, size=8.6, colour=WHITE, align=PP_ALIGN.CENTER, first=True,
           font=COND, caps=True)
    yy += 0.30
    for r, (name, d) in enumerate(rows):
        ship = r == 2
        vals = [name, f"{d['per_building_rmse']:.3f} m",
                f"{d['per_building_rmse_le20m']:.3f} m", f"{d['whole_tile_rmse']:.3f} m"]
        for j, (x, w, v) in enumerate(zip(cx, cws, vals)):
            rounded(s, x, yy, w - 0.04, 0.26, fill=TINT if ship else MIST, edge=None,
                    radius=0.04)
            para(s, x + 0.08, yy + 0.03, w - 0.16, 0.22, v, size=8.8,
                 colour=INK, font=SANS_B if (ship or j == 0) else SANS,
                 align=PP_ALIGN.LEFT if j == 0 else PP_ALIGN.CENTER, line=1.0)
        yy += 0.30
    para(s, RX, yy + 0.02, RW, 0.36,
         [(f"Auto-zoom cuts the 0.6 m penalty from {g['cost_b']:+.0f}% to "
           f"{g['cost']:+.0f}%, all on tall buildings; ", {"colour": INK, "font": SANS_B}),
          ("under 20 m it does not help. We had pre-registered ≤ 10% and missed it. "
           "0.6 m is simulated by downsampling, so this is the resolution gap, not the "
           "sensor gap.", {"colour": SLATE})],
         size=8.0, line=1.0)

    # ------------------------------------------------------------ built and working
    pill(s, RX, 4.10, RW, "Built and working today", h=0.30, fill=DEEP, size=T_PILL_S)
    built = [("Elevation extraction", "fine-tuned DA-V2 with a sigma head"),
             ("Scale calibration", "GLO-30 absolute DSM; GCP power-law fit"),
             ("Visualisation", "three.js flythrough, measure, compare"),
             ("Live web app + API", "upload a GeoTIFF, download the DSM"),
             ("Flood tool", "water level over Indian terrain"),
             ("Offline + CPU", "one HTML viewer; ONNX int8, no GPU")]
    half = (RW - 0.10) / 2
    for i, (t, b) in enumerate(built):
        x = RX + (i % 2) * (half + 0.10)
        y = 4.46 + (i // 2) * 0.46
        card(s, x, y, half, 0.40, fill=WHITE)
        mark(s, "check", x + 0.17, y + 0.20, d=0.18)
        para(s, x + 0.32, y + 0.04, half - 0.38, 0.34,
             [(t + "\n", {"colour": INK, "font": SANS_B, "size": 8.6}),
              (b, {"colour": BODY, "size": 7.8})], line=0.95)

    # ------------------------------------------------------------ method and links
    pill(s, RX, 5.92, RW, "How these were measured", h=0.30, fill=STEEL, size=T_PILL_S)
    tf = para(s, RX, 6.28, RW - 1.06, 0.60,
              [("80 whole held-out tiles, 3,090 buildings, region-disjoint from training, "
                "against airborne LiDAR; the test split is still unscored. Full method: ",
                {"colour": BODY}),
               ("project5.zaidansari.tech/documentation", {"colour": STEEL,
                                                           "font": SANS_B})],
              size=8.4, line=1.02)
    qr(s, DOCS_URL, "docs_qr", X1 - 0.82, 6.26, 0.72, "Scan: documentation")


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
    # The extra slide is a copy of template slide 6, taken before slide 6 is drawn on.
    clone_slide(prs, prs.slides[5])

    builders = [slide_title, slide_problem_solution, slide_technical,
                slide_feasibility, slide_impact, slide_references, slide_preliminary]
    for s, build in zip(prs.slides, builders):
        trim_footer(s)
        for oval in by_name(s, "Oval"):
            if oval.has_text_frame and oval.text_frame.paragraphs[0].runs:
                oval.text_frame.paragraphs[0].runs[0].text = TEAM
        build(s)

    n = link_runs(prs)
    assert n == 6, f"expected 6 linked addresses (2 on 1, 3 on 6, 1 on 7), got {n}"
    prs.save(str(DST))
    print(f"  wrote {DST}  ({len(prs.slides)} slides)")


if __name__ == "__main__":
    main()
