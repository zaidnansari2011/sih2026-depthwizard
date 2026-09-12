"""Deck C: deck B's theme, restructured against what the SIH decks actually do.

Decks A and B are untouched; each iteration is kept so they can be compared side by
side. C shares build_deck_b's palette, type scale and primitives -- the visual language
is settled. What changes is structure, and only on three slides.

Read across seven SIH decks (five from SlideShare, plus the two supplied locally), the
patterns that separate a substantive deck from a template fill are:

  - slide 3 draws a LAYERED architecture with arrows, not a list of stages;
  - the technology list earns one line, not a column of logos;
  - implementation methodology means a deployment path, not just a pipeline;
  - slide 6 carries a link to a demo, because no still can prove a thing runs.

So, versus deck B:
  slide 3  the numbered spine becomes a six-band architecture diagram; the tech-stack
           logo grid is cut to an inline chip strip, and the column it occupied now
           carries the deployment and adoption path.
  slide 5  an impact line that frames the numbers as cost avoided rather than accuracy.
  slide 6  a QR slot for the pitch video, beside the headline result.

No number changed. Nothing here is claimed that docs/evidence-pack.md does not support;
where a figure would have helped and we do not hold it, the slot is left for one rather
than filled with an estimate.

    python tools/ppt/make_deck_c.py
"""
from __future__ import annotations

import math
from pathlib import Path

from pptx import Presentation
from pptx.enum.dml import MSO_LINE_DASH_STYLE
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt

from build_deck import by_name, drop, textbox, trim_footer
from build_deck_b import (BODY, BODY_TOP, COND, DEEP, DISPLAY, EMBER, GREEN, INK, LINE,
                          MIST, NAVY, RAMP, RED, SAND, SANS_B, SKY, SLATE, SRC, STEEL,
                          T_BODY, T_LABEL, T_MICRO, T_PILL_S, T_TITLE, TEAM, W, WHITE,
                          X0, X1, arch_layer, arrow_down, badge, card, chevron,
                          compare_table, dotted, figure, mark, panel, para, pill,
                          qr_png, rect, rounded, shot, stat_chip, tag, brandstrip, w_)
from make_deck_b import (PS_ID, PS_THEME, PS_TITLE, TEAM_ID, YEAR, slide_feasibility,
                         slide_solution, slide_title)

DST = Path("D:/sih2026/depthwizard/docs/SIH2026-DevUp-SIH26175-DepthWizard-C.pptx")

# Set this once the pitch clip is hosted and the QR becomes real. Until then slide 6
# draws the slot so the layout is proven, rather than a code that scans to nothing.
PITCH_VIDEO_URL = None


# ------------------------------------------------------------------------- slide 3
def slide_technical(s) -> None:
    for sh in by_name(s, "TextBox 8"):
        drop(sh)

    AW, BX, BW, CX = 6.00, 6.70, 2.95, 9.95
    CW = X1 - CX

    # ------------------------------------------------ the architecture, drawn in bands
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
    for i, (label, items) in enumerate(layers):
        y = arch_layer(s, X0 + 0.12, y, AW - 0.24, label, items, head=0.22, band=0.38,
                       fill=EMBER if i == len(layers) - 1 else NAVY)
        if i < len(layers) - 1:
            y = arrow_down(s, X0 + AW / 2, y + 0.02, h=0.09) + 0.02

    # ------------------------------------------ the deployment path, in the freed column
    pill(s, BX, BODY_TOP, BW, "Deployment and adoption path")
    panel(s, BX, 1.66, BW, 4.36)
    stages = [
        ("Pilot on what is already open",
         "Run over Bhoonidhi CartoDEM and LISS-4 tiles we can access today, and publish "
         "per-terrain metrics rather than one average."),
        ("Validate on Indian truth",
         "With SAC-held Cartosat stereo or DGPS control points, the Sikkim "
         "model-vs-model check becomes a real accuracy number."),
        ("Integrate",
         "Ships as a container behind an internal API. The viewer needs no server and "
         "opens from disk on an analyst's own machine."),
        ("Scale across the archive",
         "Batch the back catalogue. Every new acquisition gains a height layer at no "
         "extra tasking cost."),
    ]
    for i, (t, b) in enumerate(stages):
        y = 1.78 + i * 1.06
        n = rounded(s, BX + 0.12, y, 0.32, 0.32, fill=STEEL, edge=None, radius=0.05)
        n.text_frame.word_wrap = False
        w_(n.text_frame, f"{i + 1}", size=11.5, colour=WHITE, align=PP_ALIGN.CENTER,
           first=True, font=DISPLAY)
        if i < len(stages) - 1:
            rect(s, BX + 0.27, y + 0.34, 0.02, 0.70, fill=LINE)   # the connecting rail
        para(s, BX + 0.56, y + 0.02, BW - 0.70, 0.26, t, size=T_TITLE - 0.4, colour=INK,
             font=SANS_B, line=0.96)
        para(s, BX + 0.56, y + 0.30, BW - 0.70, 0.70, b, size=T_BODY - 0.3, colour=BODY,
             line=1.0)

    # ------------------------------------------------------------------- decisions
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

    # ------------------------------------------------------ validation, then the stack
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

    # The logo grid this replaces held a whole column to say what one chip row says.
    pill(s, BX, 6.06, X1 - BX, "Technology stack — every dependency free and open",
         h=0.30, fill=STEEL, size=T_PILL_S)
    brandstrip(s, BX, 6.44, X1 - BX,
               [("python", "Python 3.10"), ("pytorch", "PyTorch"),
                ("huggingface", "Hugging Face"), ("onnx", "ONNX Runtime"),
                ("numpy", "NumPy / SciPy"),
                ("database", "GDAL / rasterio"), ("threedotjs", "three.js"),
                ("docker", "Docker")])


# ------------------------------------------------------------------------- slide 5
def slide_impact(s) -> None:
    for sh in by_name(s, "TextBox 8"):
        drop(sh)

    LW, RX = 6.45, 7.10
    RW = X1 - RX

    pill(s, X0, BODY_TOP, LW, "Who it serves")
    # 4.24 not 4.40: deck C adds a line of impact framing under the stat strip, and the
    # whole right-hand stack has to come up by that line's height to clear the footer.
    panel(s, X0, 1.66, LW, 4.52)

    hub_cx, hub_cy, hub_d = X0 + LW / 2, 3.40, 1.04
    users = [
        ("satellite", "ISRO and SAC",
         "Height from single-view Cartosat archives, no new acquisition.", 0.30, 1.78),
        ("building-2", "Urban planning",
         "Building heights for FSI checks and shadow studies under AMRUT.", 3.37, 1.78),
        ("siren", "Disaster response",
         "Post-event surface models where a stereo pair cannot be waited for.",
         0.30, 3.98),
        ("leaf", "Forestry",
         "Canopy height proxies for biomass and encroachment monitoring.", 3.37, 3.98),
        ("radio-tower", "Telecom rollout",
         "Line-of-sight and tower siting without a costly LiDAR survey.", 1.83, 5.04),
    ]
    cw, ch = 2.78, 0.98
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
        para(s, x + 0.12, y + 0.40, cw - 0.24, 0.52, b, size=T_BODY, colour=BODY,
             line=1.02)

    rounded(s, hub_cx - (hub_d + 0.18) / 2, hub_cy - (hub_d + 0.18) / 2, hub_d + 0.18,
            hub_d + 0.18, fill=SKY, edge=None, radius=(hub_d + 0.18) / 2)
    rounded(s, hub_cx - hub_d / 2, hub_cy - hub_d / 2, hub_d, hub_d, fill=NAVY,
            edge=None, radius=hub_d / 2)
    para(s, hub_cx - hub_d / 2, hub_cy - 0.26, hub_d, 0.54,
         [("DEPTH\n", {"colour": WHITE}), ("WIZARD", {"colour": SAND})], size=12.5,
         font=DISPLAY, align=PP_ALIGN.CENTER, line=1.0)

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
        chevron(s, RX, 1.70 + i * 0.76, RW, 0.68, t, ico,
                fill=NAVY if i % 2 == 0 else DEEP)

    # The four numbers are accuracy and cost-of-running figures. The sentence beside them
    # says what they are FOR -- deck B left a judge to make that leap unaided. A rupee
    # figure would be stronger still, and is deliberately absent: we have not measured
    # Cartosat tasking cost, and an invented one would be the only unmeasured claim here.
    stats = [("1", " image", "input needed, not a stereo pair"),
             ("0", " GPUs", "536 ms per tile on a laptop CPU"),
             ("36.8", " MB", "the whole model, quantised, offline"),
             ("4", " terrains", "scored separately, never averaged")]
    cw = (W - 0.36) / 4
    for i, (v, u, cap) in enumerate(stats):
        stat_chip(s, X0 + i * (cw + 0.12), 6.26, cw, 0.72, v, u, cap, fill=MIST,
                  vcolour=EMBER)
    para(s, X0, 7.00, W, 0.18,
         [("Every one of these runs on imagery that has already been flown and already "
           "been paid for.", {"colour": INK, "bold": True, "font": SANS_B}),
          ("  The cost avoided is the second acquisition, not the first.",
           {"colour": SLATE})], size=T_MICRO, line=1.0)


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
        ("GAMUS", "ISRO's recommended set. 6,204 tiles, 0.33 m, DC and Philadelphia."),
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
        ("01 Tall buildings", "Data, not capacity — and GAMUS then proved it."),
        ("03 Guided filtering", "Squaring off roofs made every metric worse."),
        ("04 Where detail went", "Detail follows the ground area one token covers."),
        ("05 Eval resolution", "Cartosat-class input costs 3.6% after auto-zoom."),
        ("06 Height compression", "Measured: ours = 0.473 × truth + 2.66 m."),
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
                        rows, param_w=CW * 0.46, head_h=0.44, row_h=0.34)

    legend = [("check", "yes", GREEN), ("minus", "partial", SAND), ("x", "no", RED)]
    lx = CX + 0.04
    for k, t, c in legend:
        mark(s, k, lx + 0.09, end + 0.15, d=0.17)
        para(s, lx + 0.22, end + 0.05, 1.30, 0.22, t, size=T_MICRO, colour=SLATE)
        lx += 0.44 + len(t) * 0.052

    # Deck B gave this its own pill. Here it is a plain line: the QR below needs the
    # 0.38 in that pill was spending, and the sentence carries itself.
    para(s, CX + 0.06, end + 0.33, CW - 0.12, 0.36,
         [("Two rows the alternatives win. ", {"colour": INK, "bold": True,
                                               "font": SANS_B}),
          ("A stereo survey gives absolute height directly and ships mature viewers — "
           "the table is only useful to a panel if the rows we lose are still in it.",
           {"colour": BODY})], size=T_MICRO, line=1.02)

    # ------------------------------------------------ the headline, and the demo link
    QW = 1.02
    pill(s, CX, end + 0.77, CW, "The bar we are measured against", h=0.30, fill=NAVY,
         size=T_PILL_S)
    para(s, CX + 0.06, end + 1.15, CW - QW - 0.30, 0.92,
         [("3.464 m", {"colour": EMBER, "bold": True, "font": DISPLAY, "size": 14}),
          ("  per-building RMSE against airborne LiDAR, versus 5.9 m published for "
           "Asia. Scored on 3,090 buildings across 80 held-out tiles.\n",
           {"colour": BODY}),
          ("Where it fails: ", {"colour": INK, "bold": True, "font": SANS_B}),
          ("buildings above 20 m are under-called by 15.6 m. We pre-registered −13 m "
           "and missed it. GAMUS improved it significantly (p = 0.0002); no open data "
           "at this resolution reaches 50 m.",
           {"colour": BODY})], size=T_BODY - 0.3, line=1.02)

    qx, qy = X1 - QW, end + 1.15
    if PITCH_VIDEO_URL:
        p = qr_png(PITCH_VIDEO_URL, Path("D:/sih2026/depthwizard/tools/ppt/figures/"
                                         "pitch_qr.png"))
        s.shapes.add_picture(str(p), Inches(qx), Inches(qy), width=Inches(QW),
                             height=Inches(QW))
        para(s, qx - 0.10, qy + QW + 0.02, QW + 0.20, 0.18, "Scan: pitch video",
             size=6.8, colour=SLATE, align=PP_ALIGN.CENTER)
    else:
        # Drawn, not filled: a code that scans to nothing is worse than an empty slot,
        # and this proves the space before the clip exists. One constant fills it.
        box = rounded(s, qx, qy, QW, QW, fill=MIST, edge=SLATE, radius=0.05)
        # Dashed, so it reads as a reserved slot rather than as a real object.
        box.line.dash_style = MSO_LINE_DASH_STYLE.DASH
        box.line.width = Pt(1.0)
        para(s, qx, qy + 0.34, QW, 0.60,
             "PITCH\nVIDEO\nQR", size=8.4, colour=SLATE, align=PP_ALIGN.CENTER,
             font=COND, line=1.08)


# ------------------------------------------------------------------------------ main
def main() -> None:
    prs = Presentation(str(SRC))

    xml_slides = prs.slides._sldIdLst          # slide 7 is the template's own note sheet
    for sid in list(xml_slides)[6:]:
        prs.part.drop_rel(sid.rId)
        xml_slides.remove(sid)

    # Slides 1, 2 and 4 are unchanged from deck B, so they are imported rather than
    # copied. A second copy of that content would be a second place to fix a number.
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
