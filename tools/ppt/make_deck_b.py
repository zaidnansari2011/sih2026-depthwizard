"""Deck B: the same six slides and the same measured content, in a poster register.

Deck A stays exactly as it is. This is the experiment beside it, written as a separate
pair of modules so neither can break the other. Every number here is the same number
deck A quotes and traces to docs/evidence-pack.md; nothing was rounded up to make a
panel look better.

Copy length is a design constraint here, not an afterthought. The first cut of this deck
set 10,447 characters across six slides against the reference deck's 6,966, and the type
had to shrink to 7.4 pt to fit -- against the reference's 8.7 pt body. Type went up to
build_deck_b's scale first, and every block below was then cut to fit it. Where a point
was already made on another slide it was deleted rather than shrunk: the "five negative
results" card came off slide 3 because slide 6 lists all five in full.

The one thing deck B adds that deck A does not have is the comparison table on slide 6.
It is the most persuasive object in a hackathon deck and it is also the easiest to rig,
so two of its rows are left won by the alternatives.

    python tools/ppt/make_deck_b.py
"""
from __future__ import annotations

import math

from pptx import Presentation
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches

from build_deck import by_name, drop, textbox, trim_footer
from build_deck_b import (BODY, BODY_BOT, BODY_TOP, COND, DEEP, DISPLAY, DST, EMBER,
                          GREEN, INK, LINE, MIST, NAVY, PAPER, RAMP, RED, SAND, SANS,
                          SANS_B, SKY, SLATE, SRC, STEEL, T_BODY, T_LABEL, T_MICRO,
                          T_PILL, T_PILL_S, T_TITLE, TEAM, W, WHITE, X0, X1, badge,
                          brand, card, chevron, compare_table, dotted, figure, mark,
                          panel, para, pill, rect, rounded, shot, spine_step, stat_chip,
                          tag, w_)

PS_ID = "SIH26175"
PS_TITLE = "DepthWizard - Single-View Height Estimation and 3D Flythrough"
PS_THEME = "Disaster Management"
TEAM_ID = "47"
YEAR = "2026"


# ------------------------------------------------------------------------- slide 1
def slide_title(s) -> None:
    for sh in by_name(s, "TextBox 9"):
        drop(sh)
    # The template's own subtitle still reads "TITLE PAGE" in deck A. It is a placeholder,
    # not a required heading, and the reference deck drops it too.
    for sh in by_name(s, "Subtitle 3"):
        drop(sh)
    heads = [p.text_frame.text for p in by_name(s, "Title 7")]
    assert any(YEAR in h for h in heads), (
        f"the source template's heading is {heads!r}, which does not mention {YEAR}. "
        f"Check build_deck_b.SRC points at the SIH {YEAR} format.")

    # The template's hexagon artwork starts around x 6.2, so the left column stops short
    # of it rather than sitting on top of it.
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

    pill(s, PX, 5.02, PW, "Built and measured, not proposed", h=0.34, fill=EMBER,
         size=T_PILL_S)
    stats = [("3.67", " m", "per-building RMSE\nagainst airborne LiDAR"),
             ("38", " %", "better than the 5.9 m\nGlobalBuildingAtlas bar"),
             ("509", " ms", "per tile on CPU alone,\nno GPU in the loop")]
    cw = (PW - 0.24) / 3
    for i, (v, u, cap) in enumerate(stats):
        stat_chip(s, PX + i * (cw + 0.12), 5.50, cw, 0.98, v, u, cap)

    # The palette is a hypsometric ramp -- the colour scale a surface model is drawn in.
    # Stating that on the title page makes the deck's own colours part of the argument
    # rather than decoration.
    for i, c in enumerate(RAMP):
        rect(s, PX + i * (PW / 6), 6.62, PW / 6, 0.15, fill=c)
    para(s, PX, 6.84, PW, 0.24,
         "The palette is the model's own output scale: 0 m at the left, 155 m at the "
         "right.", size=T_MICRO, colour=SLATE)


# ------------------------------------------------------------------------- slide 2
def slide_solution(s) -> None:
    for sh in by_name(s, "TextBox 8"):
        drop(sh)
    # The template's slide-2 heading is the placeholder "IDEA TITLE". The reference deck
    # replaces it with its own wordmark banner; so does this.
    for sh in by_name(s, "Title 1"):
        drop(sh)

    # Starts at 1.85 rather than the margin: the template's own team-name oval occupies
    # 0.36-1.73, and the first version of this banner sat on top of it.
    BX_, BW = 1.85, 8.70                        # stops clear of the SIH mark at x 10.70
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
    pill(s, X0, 1.22, LW, "The problem", fill=NAVY)
    pill(s, RX, 1.22, RW, "Our solution", fill=EMBER)

    panel(s, X0, 1.64, LW, 2.92)
    problems = [
        ("Archives with no height",
         "Decades of Cartosat single-view imagery exist. Stereo and LiDAR cannot be "
         "flown into the past."),
        ("Answers with no confidence",
         "Single-view height is ill-posed. One number implies a certainty the model "
         "does not have."),
        ("Checkpoints are not products",
         "The literature ships weights and an eval script. An analyst needs a scene "
         "they can open and check."),
    ]
    for i, (t, b) in enumerate(problems):
        y = 1.74 + i * 0.94
        card(s, X0 + 0.10, y, LW - 0.20, 0.86, fill=WHITE)
        n = rounded(s, X0 + 0.20, y + 0.12, 0.38, 0.34, fill=EMBER, edge=None,
                    radius=0.05)
        # Franklin Gothic Demi is wider than the Calibri these boxes were first sized
        # against, and "01" wrapped to two lines inside the square.
        n.text_frame.word_wrap = False
        w_(n.text_frame, f"0{i + 1}", size=11.5, colour=WHITE, align=PP_ALIGN.CENTER,
           first=True, font=DISPLAY)
        para(s, X0 + 0.68, y + 0.09, LW - 0.84, 0.24, t, size=T_TITLE, colour=INK,
             font=SANS_B, line=0.96)
        para(s, X0 + 0.68, y + 0.35, LW - 0.84, 0.44, b, size=T_BODY, colour=BODY,
             line=1.02)

    panel(s, RX, 1.64, RW, 2.92)
    items = [
        ("image-up", "One image to a metric DSM",
         "Depth Anything V2 fine-tuned on DFC2019 + GAMUS LiDAR. GeoTIFF out, metadata or not."),
        ("shield-check", "Confidence, not just height",
         "A per-pixel sigma beside every height. Where it says unsure, it is wrong — "
         "monotonically."),
        ("box", "A navigable 3D scene",
         "three.js flythrough, drag-to-compare against reference, and a live error map."),
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
        y = 1.74 + (i // 3) * 1.40
        card(s, x, y, cw, 1.30, fill=WHITE)
        badge(s, ico, x + 0.34, y + 0.32, 0.48, fill=STEEL if i % 2 == 0 else DEEP)
        para(s, x + 0.66, y + 0.16, cw - 0.78, 0.34, t, size=T_TITLE, colour=INK,
             font=SANS_B, line=0.96)
        para(s, x + 0.14, y + 0.66, cw - 0.28, 0.56, b, size=T_BODY, colour=BODY,
             line=1.02)

    # ------------------------------------------------- the product, captured running
    pill(s, X0, 4.64, W, "The product, running — captures of the live viewer, not "
                         "mock-ups", h=0.30, fill=DEEP, size=T_PILL_S)
    shots = [
        ("sikkim_hilly_crop.png", "Sikkim, India",
         "967 m of relief across 2 km, image draped on the surface."),
        ("urban_height_crop.png", "Omaha, height above ground",
         "Flat roofs, tree crowns and kerbs resolved at 0.3 m."),
        ("urban_sigma_crop.png", "Omaha, model confidence",
         "Per-pixel sigma: where the estimate is weak, it says so."),
    ]
    iw, ih = 3.99, 1.88
    for i, (fn, cap, sub) in enumerate(shots):
        x = X0 + i * (iw + 0.28)
        shot(s, fn, x, 5.02, iw, ih)
        lab = rounded(s, x, 5.02, 2.24, 0.30, fill=NAVY, edge=None, radius=0.04)
        w_(lab.text_frame, cap, size=T_LABEL, colour=WHITE, first=True, font=COND,
           caps=True)
        lab.text_frame.margin_left = Inches(0.12)
        para(s, x, 6.94, iw, 0.20, sub, size=T_MICRO, colour=SLATE, line=0.98)


# ------------------------------------------------------------------------- slide 3
def slide_technical(s) -> None:
    for sh in by_name(s, "TextBox 8"):
        drop(sh)

    AW, BX, BW, CX = 5.30, 6.00, 2.86, 9.14
    CW = X1 - CX

    # ------------------------------------------------------------------ the pipeline
    pill(s, X0, BODY_TOP, AW, "Architecture — one image to a checked surface")
    panel(s, X0, 1.66, AW, 4.14)
    steps = [
        ("Input", "GeoTIFF, or a plain PNG / JPG with no metadata at all"),
        ("GSD normalise", "auto-zoom reads ground sample distance, resamples to 0.3 m"),
        ("DA-V2 backbone", "ViT encoder and DPT neck, fine-tuned on DFC2019 + GAMUS"),
        ("Dual head", "height mu and log-variance sigma, emitted per pixel"),
        ("Tiled inference", "518 px sliding window, overlap blending, plus TTA"),
        ("Scale anchor", "Copernicus GLO-30, or ground control by power-law fit"),
        ("Outputs", "GeoTIFF DSM, sigma map, three.js scene with a live error map"),
    ]
    for i, (t, b) in enumerate(steps):
        spine_step(s, X0 + 0.10, 1.76 + i * 0.58, AW - 0.20, 0.52, f"{i + 1}", t, b,
                   fill=NAVY if i < 6 else EMBER)

    pill(s, X0, 5.92, AW, "How it is validated", h=0.30, fill=STEEL, size=T_PILL_S)
    _, tf = textbox(s, X0 + 0.06, 6.30, AW - 0.12, 0.86)
    for i, (k, v) in enumerate([
        ("Split", "region-disjoint, not random — adjacent tiles share buildings."),
        ("Scored on", "80 whole tiles, 11,983,760 pixels, 3,090 buildings."),
        ("Reported per", "urban / sparse / forested / mixed, and per height band."),
        ("Held back", "the test split is untouched until the very end."),
    ]):
        w_(tf, [(f"{k}  ", {"colour": INK, "bold": True, "font": SANS_B}),
                (v, {"colour": BODY})], size=T_BODY, first=(i == 0), space_after=3,
           line=1.0)

    # ------------------------------------------------------------------- tech stack
    pill(s, BX, BODY_TOP, BW, "Tech stack")
    panel(s, BX, 1.66, BW, 4.14)
    stack = [("python", "Python 3.11"), ("pytorch", "PyTorch"),
             ("huggingface", "Transformers"), ("onnx", "ONNX Runtime"),
             ("numpy", "NumPy / SciPy"), ("opencv", "OpenCV"),
             ("database", "GDAL / rasterio"), ("threedotjs", "three.js r169"),
             ("react", "React"), ("docker", "Docker")]
    cw = (BW - 0.24) / 2
    for i, (ico, name) in enumerate(stack):
        x = BX + 0.12 + (i % 2) * cw
        y = 1.80 + (i // 2) * 0.78
        brand(s, ico, x + cw / 2, y + 0.24, 0.38)
        para(s, x, y + 0.50, cw - 0.06, 0.24, name, size=T_MICRO, colour=INK,
             align=PP_ALIGN.CENTER, font=SANS_B, line=0.96)

    pill(s, BX, 5.92, BW, "What ships", h=0.30, fill=STEEL, size=T_PILL_S)
    para(s, BX + 0.06, 6.30, BW - 0.12, 0.86,
         [("An elevation module", {"colour": INK, "bold": True, "font": SANS_B}),
          (" emitting a GeoTIFF DSM and a sigma map, and ", {"colour": BODY}),
          ("a viewer", {"colour": INK, "bold": True, "font": SANS_B}),
          (" that opens it and checks a height against reference data. No GPU, no "
           "network.", {"colour": BODY})], size=T_BODY, line=1.02)

    # ------------------------------------------------------------------- decisions
    pill(s, CX, BODY_TOP, CW, "Decisions, and why")
    panel(s, CX, 1.66, CW, 5.14)
    # Four, not five. The fifth card said "five negative results are written up", which
    # slide 6 then lists in full -- so it was repetition paid for in type size.
    decisions = [
        ("gauge", "The head was chosen by measurement",
         "Regression, binned and head-tail-cut heads were all trained and scored. All "
         "three land at a 0.43–0.49 slope, so the simplest ships."),
        ("shield-check", "Uncertainty is calibrated, not decorative",
         "ECE 0.077, and a sigma-vs-error rank correlation of +0.829 on held-out tiles."),
        ("cpu", "Export is checked, not assumed",
         "ONNX agrees with PyTorch to 0.0010 cm over 804,972 inputs. int8 costs 0.161 m "
         "for a 63% smaller file."),
        ("layers", "Resolution is handled explicitly",
         "Auto-zoom reads GSD from metadata and resamples, recovering all but 3.6% of "
         "the coarse-input penalty."),
    ]
    for i, (ico, t, b) in enumerate(decisions):
        y = 1.78 + i * 1.24
        card(s, CX + 0.10, y, CW - 0.20, 1.16, fill=WHITE)
        badge(s, ico, CX + 0.44, y + 0.30, 0.46, fill=DEEP)
        para(s, CX + 0.76, y + 0.15, CW - 0.90, 0.32, t, size=T_TITLE, colour=INK,
             font=SANS_B, line=0.96)
        para(s, CX + 0.22, y + 0.60, CW - 0.44, 0.50, b, size=T_BODY, colour=BODY,
             line=1.02)


# ------------------------------------------------------------------------- slide 4
def slide_feasibility(s) -> None:
    for sh in by_name(s, "TextBox 8"):
        drop(sh)

    chips = [("circle-check-big", "Technically proven", "Built and measured, not "
                                                        "proposed."),
             ("wifi-off", "Operationally simple", "One image in. No GPU, no network."),
             ("indian-rupee", "Economically viable", "Every dependency free and open."),
             ("users", "Socially useful", "Planning, disaster and forestry use height.")]
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

    # ------------------------------------------------- the strength, then the weakness
    pill(s, X0, 1.96, LW, "Accuracy across the four landscapes ISRO names", h=0.32,
         size=T_PILL_S)
    panel(s, X0, 2.36, LW, 2.28)
    figure(s, "terrain.png", X0 + 0.12, 2.48, 3.48, 3.48 / 2.093)
    para(s, X0 + 3.74, 2.50, LW - 3.90, 1.96,
         [("Three of four classes sit under 3.4 m", {"colour": INK, "bold": True,
                                                     "font": SANS_B}),
          (", and correlation never drops below +0.58. The fourth is the honest "
           "problem, and it is one specific thing.", {"colour": BODY})],
         size=T_BODY, line=1.04)

    pill(s, X0, 4.76, LW, "Urban 13.86 m, decomposed", h=0.32, fill=EMBER,
         size=T_PILL_S)
    panel(s, X0, 5.16, LW, 1.98)
    # Height is given explicitly. Placed by width alone this figure is 2.51:1, which ran
    # off the bottom of the slide and under the template's footer bar.
    figure(s, "error_by_height.png", X0 + 0.12, 5.36, 3.34, 3.34 / 2.510)
    para(s, X0 + 3.60, 5.26, LW - 3.76, 1.76,
         [("79% of squared error comes from 1.9% of buildings.",
           {"colour": EMBER, "bold": True, "font": SANS_B}),
          ("  Under 10 m — 94% of the stock — we are at 1.4 m. Urban RMSE is 60 tall "
           "buildings dominating a squared metric.", {"colour": BODY})],
         size=T_BODY, line=1.04)

    # ---------------------------------------------------- risks, named and answered
    pill(s, RX, 1.96, RW, "The two real risks, and what answers them", h=0.32,
         size=T_PILL_S)
    risks = [
        ("No Indian ground truth exists to validate against",
         "Our only Indian check is Sikkim against Google Open Buildings: bias −6.94 m "
         "on 168 footprints. That is model-vs-model agreement, not accuracy, and we "
         "report it as such. No public Indian scene carries LiDAR truth."),
        ("Tall buildings compress toward the mean",
         "A data problem, not an architecture one: training tops out at 82.8 m while "
         "validation reaches 155.3 m. Answered by a power-law control-point fit that "
         "cuts tall-tile RMSE 24.9%, and structurally by adding tall-building LiDAR."),
    ]
    y = 2.42
    for t, b in risks:
        rounded(s, RX, y, RW, 0.40, fill=NAVY, edge=None, radius=0.05)
        badge(s, "triangle-alert", RX + 0.28, y + 0.20, 0.30, fill=SAND)
        para(s, RX + 0.54, y + 0.09, RW - 0.68, 0.28, t, size=T_TITLE, colour=WHITE,
             font=SANS_B, line=0.96)
        # The answer is marked by a green rule down the card's edge, not by a badge
        # straddling the header bar above it -- that read as a collision, not a link.
        card(s, RX + 0.26, y + 0.46, RW - 0.26, 0.98, fill=WHITE)
        rect(s, RX + 0.26, y + 0.52, 0.07, 0.86, fill=GREEN)
        para(s, RX + 0.50, y + 0.56, RW - 0.64, 0.82, b, size=T_BODY, colour=BODY,
             line=1.04)
        y += 1.56

    pill(s, RX, 5.62, RW, "What reduces the Indian-domain risk before we have the data",
         h=0.30, fill=STEEL, size=T_PILL_S)
    _, tf = textbox(s, RX + 0.06, 6.00, RW - 0.12, 1.16)
    for i, (k, v) in enumerate([
        ("Resolution", "auto-zoom normalises GSD; measured to recover all but 3.6%."),
        ("Look angle", "off-nadir spans 4.8 to 28.9 degrees in DFC, so degradation is "
                       "measurable on data we hold."),
        ("Terrain", "leakage ruled out — on 31-degree slopes, height-vs-slope "
                    "correlation is −0.044."),
        ("Licensing", "every dependency is free and open, so nothing blocks deployment "
                      "on ISRO imagery."),
    ]):
        w_(tf, [(f"{k}  ", {"colour": INK, "bold": True, "font": SANS_B}),
                (v, {"colour": BODY})], size=T_BODY, first=(i == 0), space_after=3,
           line=1.0)


# ------------------------------------------------------------------------- slide 5
def slide_impact(s) -> None:
    for sh in by_name(s, "TextBox 8"):
        drop(sh)

    LW, RX = 6.45, 7.10
    RW = X1 - RX

    # ------------------------------------------------------------- who it serves, radial
    pill(s, X0, BODY_TOP, LW, "Who it serves")
    panel(s, X0, 1.66, LW, 4.66)

    hub_cx, hub_cy, hub_d = X0 + LW / 2, 3.46, 1.06
    users = [
        ("satellite", "ISRO and SAC",
         "Height from single-view Cartosat archives, no new acquisition.", 0.30, 1.80),
        ("building-2", "Urban planning",
         "Building heights for FSI checks and shadow studies under AMRUT.", 3.37, 1.80),
        ("siren", "Disaster response",
         "Post-event surface models where a stereo pair cannot be waited for.",
         0.30, 4.10),
        ("leaf", "Forestry",
         "Canopy height proxies for biomass and encroachment monitoring.", 3.37, 4.10),
        ("radio-tower", "Telecom rollout",
         "Line-of-sight and tower siting without a costly LiDAR survey.", 1.83, 5.20),
    ]
    cw, ch = 2.78, 1.00
    for ico, t, b, dx, dy in users:
        dotted(s, hub_cx, hub_cy, X0 + dx + cw / 2, dy + ch / 2)
    for ico, t, b, dx, dy in users:
        x, y = X0 + dx, dy
        card(s, x, y, cw, ch, fill=WHITE)
        head = rounded(s, x, y, cw, 0.30, fill=NAVY, edge=None, radius=0.05)
        w_(head.text_frame, t, size=T_LABEL + 0.4, colour=WHITE, first=True, font=COND,
           caps=True)
        head.text_frame.margin_left = Inches(0.44)
        badge(s, ico, x + 0.20, y + 0.15, 0.34, fill=EMBER)
        para(s, x + 0.12, y + 0.40, cw - 0.24, 0.54, b, size=T_BODY, colour=BODY,
             line=1.02)

    # A plain ring, drawn as two ovals. The first version used badge(pad=0), which asked
    # PowerPoint for a zero-width picture and got the glyph at its native size instead --
    # a globe the width of the panel, painted over every card.
    rounded(s, hub_cx - (hub_d + 0.18) / 2, hub_cy - (hub_d + 0.18) / 2, hub_d + 0.18,
            hub_d + 0.18, fill=SKY, edge=None, radius=(hub_d + 0.18) / 2)
    rounded(s, hub_cx - hub_d / 2, hub_cy - hub_d / 2, hub_d, hub_d, fill=NAVY,
            edge=None, radius=hub_d / 2)
    para(s, hub_cx - hub_d / 2, hub_cy - 0.26, hub_d, 0.54,
         [("DEPTH\n", {"colour": WHITE}), ("WIZARD", {"colour": SAND})], size=12.5,
         font=DISPLAY, align=PP_ALIGN.CENTER, line=1.0)

    # ------------------------------------------------------------------- key benefits
    pill(s, RX, BODY_TOP, RW, "Key benefits")
    benefits = [
        ("image-up", "One image replaces a stereo pair — no revisit wait."),
        ("cpu", "No GPU procurement: 36.8 MB runs on a field laptop."),
        ("layers", "Decades of existing archive gain a new product."),
        ("building-2", "Municipal bodies get heights with no survey budget."),
        ("siren", "Disaster teams get a surface model in minutes."),
        ("indian-rupee", "Free and open — adoptable by any state agency."),
    ]
    for i, (ico, t) in enumerate(benefits):
        chevron(s, RX, 1.70 + i * 0.79, RW, 0.71, t, ico,
                fill=NAVY if i % 2 == 0 else DEEP)

    # -------------------------------------------------------- the numbers, full width
    # These sit under both columns rather than under the left one: at a quarter of the
    # left column each chip was 1.5 in wide and every caption ran to three lines.
    stats = [("1", " image", "input needed, not a stereo pair"),
             ("0", " GPUs", "536 ms per tile on a laptop CPU"),
             ("36.8", " MB", "the whole model, quantised, offline"),
             ("4", " terrains", "scored separately, never averaged")]
    cw = (W - 0.36) / 4
    for i, (v, u, cap) in enumerate(stats):
        stat_chip(s, X0 + i * (cw + 0.12), 6.40, cw, 0.76, v, u, cap, fill=MIST,
                  vcolour=EMBER)


# ------------------------------------------------------------------------- slide 6
def slide_references(s) -> None:
    for sh in by_name(s, "TextBox 8"):
        drop(sh)

    AW, BX, CX = 3.98, 4.60, 8.80
    BW = 3.98
    CW = X1 - CX

    def stack_col(x, w, top, title, items, fill=NAVY, row=0.48):
        y = pill(s, x, top, w, title, h=0.30, fill=fill, size=T_PILL_S)
        panel(s, x, y, w, 0.10 + len(items) * row)
        for i, (n, d) in enumerate(items):
            para(s, x + 0.12, y + 0.06 + i * row, w - 0.24, row - 0.04,
                 [(n + "  ", {"colour": INK, "bold": True, "font": SANS_B,
                              "size": T_TITLE - 0.8}),
                  (d, {"colour": BODY, "size": T_BODY - 0.3})], line=1.0)
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
        ("Copernicus GLO-30", "Free 30 m global DEM; the absolute metric anchor."),
        ("GlobalBuildingAtlas", "Published 5.9 m RMSE over Asia — the external bar."),
        ("ISRO Bhoonidhi", "Registered. CartoDEM 30 m and LISS-4 are the free tiers."),
        ("Google Open Buildings", "A cross-check over India, never used as truth."),
    ])

    y = stack_col(BX, BW, BODY_TOP, "Tried, measured, rejected", [
        ("Ordinal / binned head", "All heads land at a 0.43–0.49 slope. No gain."),
        ("Shadow photogrammetry", "Oracle shadows reach r 0.503; the net reaches 0.787."),
        ("LDS tail reweighting", "Pixels above 20 m are 7.5% but carry 78.8% of error."),
        ("Two-model ensemble", "Error correlation 0.925 — averaging buys nothing."),
        ("Global de-compression", "A rescale moves error, it does not remove it."),
    ], fill=EMBER)
    stack_col(BX, BW, y + 0.12, "Probes we ran, and what each closed", [
        ("01 Tall buildings", "More tall examples? No — training tops out at 82.8 m."),
        ("03 Guided filtering", "Squaring off roofs made every metric worse."),
        ("04 Where detail went", "Detail follows the ground area one token covers."),
        ("05 Eval resolution", "Cartosat-class input costs 3.6% after auto-zoom."),
        ("06 Height compression", "Measured: ours = 0.473 × truth + 2.66 m."),
    ], fill=STEEL)

    # ------------------------------------------------------------ the comparison table
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
                        rows, param_w=CW * 0.46, head_h=0.44, row_h=0.38)

    legend = [("check", "yes", GREEN), ("minus", "partial", SAND), ("x", "no", RED)]
    lx = CX + 0.04
    for k, t, c in legend:
        mark(s, k, lx + 0.09, end + 0.15, d=0.17)
        para(s, lx + 0.22, end + 0.05, 1.30, 0.22, t, size=T_MICRO, colour=SLATE)
        lx += 0.44 + len(t) * 0.052

    pill(s, CX, end + 0.36, CW, "Two rows the alternatives win", h=0.30, fill=SLATE,
         size=T_PILL_S)
    para(s, CX + 0.06, end + 0.76, CW - 0.12, 0.40,
         "A stereo survey gives absolute height directly and ships mature viewers. The "
         "table is only useful to a panel if the rows we lose are still in it.",
         size=T_BODY, colour=BODY, line=1.04)

    pill(s, CX, end + 1.20, CW, "The bar we are measured against", h=0.30, fill=NAVY,
         size=T_PILL_S)
    para(s, CX + 0.06, end + 1.60, CW - 0.12, 0.48,
         [("3.62 m", {"colour": EMBER, "bold": True, "font": DISPLAY, "size": 14}),
          ("  per-building RMSE against airborne LiDAR, versus 5.9 m published for "
           "Asia. Scored on 3,090 buildings across 80 held-out tiles.",
           {"colour": BODY})], size=T_BODY, line=1.04)


# ------------------------------------------------------------------------------ main
def main() -> None:
    prs = Presentation(str(SRC))

    xml_slides = prs.slides._sldIdLst          # slide 7 is the template's own note sheet
    for sid in list(xml_slides)[6:]:
        prs.part.drop_rel(sid.rId)
        xml_slides.remove(sid)

    builders = [slide_title, slide_solution, slide_technical,
                slide_feasibility, slide_impact, slide_references]
    for s, build in zip(prs.slides, builders):
        trim_footer(s)
        for oval in by_name(s, "Oval"):
            if oval.has_text_frame and oval.text_frame.paragraphs[0].runs:
                oval.text_frame.paragraphs[0].runs[0].text = TEAM
        build(s)

    prs.save(str(DST))
    print(f"  wrote {DST}")


if __name__ == "__main__":
    main()
