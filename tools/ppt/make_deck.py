"""Compose the six SIH slides for team Dev Up and write the .pptx.

Layout primitives live in build_deck.py; this file is the content. Every figure quoted is
measured and traceable to docs/evidence-pack.md.

Two weaknesses are stated on the slides rather than left for a judge to find: no Indian
ground truth exists for validation, and urban RMSE is 13.86 m. Both are named with the
number attached and the decomposition beside them. A panel that finds a buried 13.86 m in
a deck otherwise advertising 3.62 m stops believing the 3.62 m too.

    python tools/ppt/make_deck.py
"""
from __future__ import annotations

import math

from pptx import Presentation
from pptx.enum.text import PP_ALIGN

from build_deck import (ACCENT, DST, FAINT, INK, RULE, SLATE, SRC, STEEL, TEAM, WHITE,
                        X0, block, by_name, drop, entry, figure, flow, icon, kv, label,
                        rule, shot, stat, textbox, trim_footer, vrule, write)

PS_ID = "SIH26175"
PS_TITLE = "DepthWizard - Single-View Height Estimation and 3D Flythrough"
PS_THEME = "Disaster Management"
TEAM_ID = "47"
YEAR = "2026"


# ------------------------------------------------------------------------- slide 1
def slide_title(s) -> None:
    for sh in by_name(s, "TextBox 9"):
        drop(sh)
    # Built on the official SIH 2026 template, so the year is correct in both the heading
    # text and the bulb artwork. This asserts rather than patches: silently rewriting the
    # year is how the 2025 template went unnoticed the first time.
    heads = [p.text_frame.text for p in by_name(s, "Title 7")]
    assert any(YEAR in h for h in heads), (
        f"the source template's heading is {heads!r}, which does not mention {YEAR}. "
        f"Check build_deck.SRC points at the SIH {YEAR} format.")

    fields = [("Problem Statement ID", PS_ID), ("Problem Statement Title", PS_TITLE),
              ("Theme", PS_THEME), ("PS Category", "Software"),
              ("Team ID", TEAM_ID), ("Team Name (Registered on portal)", TEAM)]
    # Row height follows whether the value actually wraps, rather than special-casing the
    # PS title: it wrapped on the 2025 template and fits on one line here, so a hardcoded
    # allowance left a visible hole. 12.5 pt Calibri bold averages ~0.086 in per glyph.
    BOX_W, CHAR_W = 5.85, 0.086
    y = 2.34
    for k, v in fields:
        _, tf = textbox(s, 0.46, y, BOX_W, 0.18)
        write(tf, k.upper(), size=7, colour=SLATE, bold=True, first=True)
        _, tf = textbox(s, 0.46, y + 0.17, BOX_W, 0.30)
        write(tf, v, size=12.5, colour=INK, bold=True, first=True, line=0.94)
        lines = max(1, math.ceil(len(v) * CHAR_W / BOX_W))
        y += 0.47 + 0.23 * (lines - 1)

    rule(s, 0.46, 5.44, 5.60, colour=RULE)
    _, tf = textbox(s, 0.46, 5.52, 5.60, 0.18)
    write(tf, "BUILT AND MEASURED, NOT PROPOSED", size=7, colour=SLATE, bold=True,
          first=True)
    for i, (val, unit, cap) in enumerate([
            ("3.62", " m", "per-building RMSE against\nairborne LiDAR, 3,090 buildings"),
            ("39", " %", "better than the 5.9 m\nGlobalBuildingAtlas Asia bar"),
            ("509", " ms", "per tile on CPU alone,\nno GPU required")]):
        stat(s, 0.46 + i * 1.92, 5.78, 1.80, val, unit, cap)


# ------------------------------------------------------------------------- slide 2
def slide_solution(s) -> None:
    for sh in by_name(s, "TextBox 8"):
        drop(sh)

    _, tf = textbox(s, X0, 1.20, 12.5, 0.32)
    write(tf, [("DepthWizard", {"colour": ACCENT, "bold": True, "size": 17}),
               ("   one satellite image in, a measurable 3D surface out",
                {"colour": INK, "size": 17})], first=True)
    _, tf = textbox(s, X0, 1.58, 12.5, 0.24)
    write(tf, [("Height from a single view. No stereo pair, no LiDAR, no second pass of "
                "the satellite. Every competing submission promises accuracy. ",
                {"colour": SLATE}),
               ("This one reports it.", {"colour": ACCENT, "bold": True})],
          size=10, first=True)

    LW, RX, RW = 4.55, 5.25, 7.67
    vrule(s, 4.94, 1.98, 2.52)

    y = label(s, X0, 1.94, LW, "The problem")
    for i, (t, b) in enumerate([
        ("Archives with no height",
         "India holds decades of Cartosat single-view imagery. Stereo and LiDAR give "
         "height, but cost more and cannot be flown into the past."),
        ("Answers with no confidence",
         "Single-view height is ill-posed. A model that emits one number implies a "
         "certainty it does not have, so an analyst cannot act on it."),
        ("Checkpoints are not products",
         "The literature ships weights and an eval script. An analyst needs to open a "
         "scene, fly through it, and check a height against a reference."),
    ]):
        entry(s, X0, y + i * 0.78, LW, f"0{i + 1}", t, b)


    y = label(s, RX, 1.94, RW, "Proposed solution")
    items = [
        ("image-up", "One image to a metric DSM",
         "Depth Anything V2 fine-tuned on DFC2019 + GAMUS LiDAR. GeoTIFF out, or a relative DSM "
         "when the input carries no geospatial metadata."),
        ("shield-check", "A confidence map, not just a height",
         "A heteroscedastic head emits per-pixel sigma beside every height. Where it says "
         "it is unsure, it is wrong, monotonically."),
        ("box", "A navigable 3D scene",
         "three.js flythrough with the image projected onto the surface, drag-to-compare "
         "against reference, and a live error map."),
        ("ruler", "Validation built into the product",
         "RMSE, MAE and correlation against LiDAR, reported per terrain class rather "
         "than as one flattering average."),
        ("map-pinned", "Absolute heights, two independent ways",
         "Copernicus GLO-30, or ground control points via a power-law fit that declines "
         "rather than degrade a scene."),
        ("wifi-off", "Deployable, not demo-ware",
         # Two separate facts. Run together they read as "the HTML file does inference",
         # which it does not: the standalone carries pre-computed scenes, and the upload
         # button is hidden there on purpose.
         "The model needs no GPU: 536 ms per tile on CPU. The viewer is one 15 MB HTML "
         "file, six scenes baked in, opening straight from disk."),
    ]
    # Two columns of three rather than one column of six: half the height, and it frees
    # the bottom of the slide for the viewer itself.
    sw = (RW - 0.30) / 2
    for i, (ico, t, b) in enumerate(items):
        cx = RX + (i % 2) * (sw + 0.30)
        ty = y + (i // 2) * 0.84
        if i // 2:
            rule(s, cx, ty - 0.11, sw)
        icon(s, ico, cx, ty + 0.02, 0.23)
        _, tf = textbox(s, cx + 0.33, ty, sw - 0.33, 0.76)
        write(tf, t, size=9.2, colour=INK, bold=True, first=True, line=0.94,
              space_after=1)
        write(tf, b, size=8.0, colour=SLATE, line=0.98)

    # ------------------------------------------------------- the product, on the slide
    # Half of SIH's score here is rendering quality and interface. A deck that only
    # quotes RMSE argues one half of the mark scheme. These are captures of the running
    # viewer, taken by tools/ppt/capture_viewer.py, not mock-ups.
    rule(s, X0, 4.56, 12.5)
    shots = [
        ("sikkim_hilly_crop.png", "Sikkim, India",
         "967 m of relief across 2 km. Satellite image draped on the recovered surface; "
         "terrain from Copernicus GLO-30, everything standing on it is ours."),
        ("urban_height_crop.png", "Omaha, height above ground",
         "The DSM itself. Flat roofs, tree crowns and kerb lines resolved at 0.3 m."),
        ("urban_sigma_crop.png", "Omaha, model confidence",
         "Per-pixel sigma. The viewer states where the estimate is weak instead of "
         "hiding it."),
    ]
    iw, ih = 3.78, 1.80
    for i, (fn, cap, sub) in enumerate(shots):
        x = X0 + i * (iw + 0.28)
        _, tf = textbox(s, x, 4.64, iw, 0.34)
        write(tf, cap, size=8.6, colour=INK, bold=True, first=True, line=0.94,
              space_after=1)
        write(tf, sub, size=7.2, colour=SLATE, line=0.96)
        shot(s, fn, x, 5.02, iw, ih)


# ------------------------------------------------------------------------- slide 3
def slide_technical(s) -> None:
    for sh in by_name(s, "TextBox 8"):
        drop(sh)

    _, tf = textbox(s, X0, 1.22, 12.5, 0.24)
    write(tf, "A monocular depth backbone re-purposed as a metric height regressor, "
              "with calibrated uncertainty and a browser renderer.",
          size=10, colour=SLATE, first=True)

    y = label(s, X0, 1.52, 12.5, "Pipeline")
    flow(s, X0 + 0.10, y + 0.22, 12.30, [
        ("Input", "GeoTIFF, or plain\nPNG / JPG"),
        ("GSD normalise", "auto-zoom to a\n0.3 m ground scale"),
        ("DA-V2 backbone", "ViT encoder, DPT\nneck, fine-tuned"),
        ("Dual head", "height mu and\nlog-variance sigma"),
        ("Tiled inference", "518 px sliding\nwindow, plus TTA"),
        ("Scale anchor", "GLO-30 DEM or\nground control"),
        ("DSM and viewer", "GeoTIFF, sigma map,\nthree.js scene"),
    ])

    LW, RX, RW = 7.10, 7.92, 5.00
    vrule(s, 7.58, 2.96, 3.86)

    y = label(s, X0, 2.92, LW, "Technology stack")
    stack = [("python", "Python 3.11"), ("pytorch", "PyTorch"),
             ("huggingface", "Transformers"), ("numpy", "NumPy / SciPy"),
             ("opencv", "OpenCV"), ("onnx", "ONNX Runtime"),
             ("threedotjs", "three.js r169"), ("react", "Web UI"),
             ("database", "GDAL / rasterio"), ("docker", "Docker")]
    cw = LW / 5
    for i, (ico, name) in enumerate(stack):
        x = X0 + (i % 5) * cw
        ty = y + (i // 5) * 0.72
        icon(s, ico, x, ty, 0.30)
        _, tf = textbox(s, x, ty + 0.34, cw - 0.10, 0.22)
        write(tf, name, size=7.8, colour=INK, first=True, line=0.94)

    y2 = y + 1.56
    rule(s, X0, y2 - 0.10, LW)
    _, tf = textbox(s, X0, y2, LW, 0.20)
    write(tf, "HOW IT IS VALIDATED", size=7, colour=SLATE, bold=True, first=True)
    _, tf = textbox(s, X0, y2 + 0.24, LW, 1.10)
    for i, (k, v) in enumerate([
        ("Split", "region-disjoint, not random. Adjacent tiles share buildings, so a "
                  "random split leaks and flatters the score."),
        ("Scored on", "80 whole tiles, 11,983,760 valid pixels, 3,090 buildings. Whole "
                      "tiles, not crops."),
        ("Reported per", "urban / sparse / forested / mixed, plus per-building and "
                         "per-height-band breakdowns."),
        ("Held back", "the test split is untouched until the very end, and holds one "
                      "building above 30 m, so it will flatter us. We will say so."),
    ]):
        kv(tf, k, v, first=(i == 0), size=8.3, space_after=4)

    y = label(s, RX, 2.92, RW, "Decisions, and why")
    for i, (t, b) in enumerate([
        ("The head was chosen by measurement",
         "Binned and head-tail-cut heads were both trained and scored. All three land at "
         "a 0.43-0.49 slope, so the simplest ships."),
        ("Uncertainty is calibrated, not decorative",
         "ECE 0.077, sigma-vs-error rank correlation +0.829 on held-out tiles."),
        ("Export is checked, not assumed",
         "ONNX agrees with PyTorch to 0.0010 cm over 804,972 inputs. int8 costs 0.161 m "
         "for a 63% smaller file."),
        ("Resolution is handled explicitly",
         "auto-zoom reads GSD from metadata and resamples, recovering all but 3.6% of "
         "the coarse-input penalty."),
    ]):
        entry(s, RX, y + i * 0.86, RW, f"0{i + 1}", t, b, num_colour=STEEL, gap=0.36)

    # A deployment-numbers strip lived here and collided with the footer bar; the same
    # figures now sit in slide 2's left column, so this closes with what the numbers are
    # actually for instead of repeating them.
    sy = 5.94
    rule(s, X0, sy, LW)
    _, tf = textbox(s, X0, sy + 0.10, LW, 0.20)
    write(tf, "WHAT SHIPS", size=7, colour=SLATE, bold=True, first=True)
    _, tf = textbox(s, X0, sy + 0.32, LW, 0.60)
    write(tf, [("An elevation module", {"colour": INK, "bold": True}),
               (" emitting a GeoTIFF DSM plus a per-pixel sigma map, and ",
                {"colour": SLATE}),
               ("a viewer", {"colour": INK, "bold": True}),
               (" that opens it, flies through it, and checks a height against reference "
                "data. Both run with no GPU and no network.", {"colour": SLATE})],
          size=8.6, first=True, line=1.0)


# ------------------------------------------------------------------------- slide 4
def slide_feasibility(s) -> None:
    for sh in by_name(s, "TextBox 8"):
        drop(sh)

    _, tf = textbox(s, X0, 1.22, 12.5, 0.24)
    write(tf, "The feasibility question is already answered: the system exists. What "
              "follows is where it is strong, and where it is not.",
          size=10, colour=SLATE, first=True)

    LW, RX, RW = 6.05, 6.92, 6.00
    vrule(s, 6.56, 1.66, 5.10)

    # -------------------------------------------------- strength, then the weakness
    y = label(s, X0, 1.54, LW, "Accuracy across the four landscapes ISRO names")
    figure(s, "terrain.png", X0, y - 0.02, 3.92)
    _, tf = textbox(s, X0, y + 1.90, LW, 0.40)
    write(tf, [("Three of four classes sit under 3.4 m", {"colour": INK, "bold": True}),
               (", and correlation never drops below +0.58. The fourth is the honest "
                "problem, and it is one specific thing.", {"colour": SLATE})],
          size=8.6, first=True, line=1.0)

    y = label(s, X0, y + 2.30, LW, "Urban 13.86 m, decomposed")
    _, tf = textbox(s, X0, y, LW, 0.42)
    write(tf, [("79% of all squared error comes from 1.9% of the buildings.",
                {"colour": ACCENT, "bold": True}),
               ("  Under 10 m, which is 94% of the stock, we are at 1.4 m. Urban RMSE is "
                "not a city problem: it is 60 tall buildings dominating a squared metric.",
                {"colour": SLATE})],
          size=8.4, first=True, line=1.0)
    figure(s, "error_by_height.png", X0, y + 0.40, 5.05)

    # ------------------------------------------------------ risks, named and answered
    y = label(s, RX, 1.54, RW, "The two real risks, and what answers them")
    for i, (t, b) in enumerate([
        ("No Indian ground truth exists to validate against",
         "Our only Indian check is Sikkim against Google Open Buildings: bias -6.94 m on "
         "168 confident footprints. That is model-vs-model agreement, not accuracy, and "
         "we report it as such. Cartosat at 0.25-2.5 m is a priced product; free LISS-4 "
         "at 5.8 m spans a building in 2-3 pixels. No public Indian scene carries LiDAR "
         "truth. This is the honest limit of what we can claim today."),
        ("Tall buildings compress toward the mean",
         "Proven to be a data problem, not an architecture one: training tops out at "
         "82.8 m while validation reaches 155.3 m, and fine-tuning on the few tall "
         "buildings we have moved transfer bias the wrong way (-18.22 to -19.10 m). "
         "Answered by a power-law control-point fit that cuts tall-tile RMSE 24.9%, and "
         "structurally by adding tall-building LiDAR."),
    ]):
        entry(s, RX, y + i * 1.52, RW, f"0{i + 1}", t, b, gap=0.36)

    y2 = y + 3.02
    rule(s, RX, y2 - 0.12, RW)
    _, tf = textbox(s, RX, y2, RW, 0.20)
    write(tf, "WHAT REDUCES THE INDIAN-DOMAIN RISK BEFORE WE HAVE THE DATA", size=7,
          colour=SLATE, bold=True, first=True)
    _, tf = textbox(s, RX, y2 + 0.24, RW, 0.90)
    for i, (k, v) in enumerate([
        ("Resolution", "auto-zoom normalises GSD; measured to recover all but 3.6%."),
        ("Look angle", "off-nadir spans 4.8-28.9 degrees in DFC, so degradation vs "
                       "view angle is measurable on data we already hold."),
        ("Terrain", "leakage ruled out. On 31-degree slopes, height-vs-slope "
                    "correlation is -0.044."),
        ("Licensing", "every dependency is free and open, so nothing blocks deployment "
                      "on ISRO's own imagery once access exists."),
    ]):
        kv(tf, k, v, first=(i == 0), size=8.2, space_after=3)


# ------------------------------------------------------------------------- slide 5
def slide_impact(s) -> None:
    for sh in by_name(s, "TextBox 8"):
        drop(sh)

    _, tf = textbox(s, X0, 1.22, 12.5, 0.24)
    write(tf, "Height is the missing dimension in India's satellite archive. Recovering "
              "it from imagery that already exists is the whole point.",
          size=10, colour=SLATE, first=True)

    y = label(s, X0, 1.54, 12.5, "Who it serves")
    users = [
        ("satellite", "ISRO and SAC", "Height from single-view Cartosat archives with "
                                      "no new acquisition."),
        ("building-2", "Urban planning", "Building heights for FSI checks, sky-view and "
                                         "shadow studies under AMRUT."),
        ("siren", "Disaster response", "Rapid post-event surface models where a stereo "
                                       "pair cannot be waited for."),
        ("radio-tower", "Telecom rollout", "Line-of-sight and tower siting from imagery "
                                           "rather than costly LiDAR survey."),
        ("leaf", "Forestry", "Canopy height proxies for biomass estimation and "
                             "encroachment monitoring."),
    ]
    cw = 12.5 / 5
    for i, (ico, t, b) in enumerate(users):
        x = X0 + i * cw
        if i:
            vrule(s, x - 0.16, y + 0.02, 0.98)
        icon(s, ico, x, y + 0.04, 0.28)
        _, tf = textbox(s, x, y + 0.40, cw - 0.34, 0.62)
        write(tf, t, size=9.4, colour=INK, bold=True, first=True, line=0.94,
              space_after=1)
        write(tf, b, size=7.9, colour=SLATE, line=0.98)

    y = label(s, X0, y + 1.24, 12.5, "Benefits")
    cols = [
        ("Economic", [
            "One image replaces a stereo pair: half the acquisition, none of the "
            "revisit wait.",
            "No GPU procurement. The quantised model runs on a field laptop.",
            "Existing archives gain a new product without re-flying anything."]),
        ("Social and civic", [
            "Planners and municipal bodies get building heights with no survey budget.",
            "Disaster teams get a surface model in minutes, with its uncertainty shown "
            "rather than hidden.",
            "Open, free dependencies keep it adoptable by any state agency."]),
        ("Environmental and strategic", [
            "Canopy and biomass proxies from imagery already being collected.",
            "No new flights means no additional survey emissions.",
            "Reduces dependence on foreign commercial elevation products."]),
    ]
    cw = 12.5 / 3
    for i, (title, points) in enumerate(cols):
        x = X0 + i * cw
        if i:
            vrule(s, x - 0.22, y + 0.02, 1.60)
        _, tf = textbox(s, x, y, cw - 0.44, 0.22)
        write(tf, title, size=9.4, colour=ACCENT, bold=True, first=True)
        _, tf = textbox(s, x, y + 0.30, cw - 0.44, 1.40)
        for j, p in enumerate(points):
            write(tf, p, size=8.3, colour=SLATE, first=(j == 0), space_after=6, line=1.0)

    y2 = y + 1.92
    rule(s, X0, y2, 12.5)
    for i, (val, unit, cap) in enumerate([
            ("1", " image", "input needed, where stereo\nphotogrammetry needs two"),
            ("0", " GPUs", "536 ms per tile on an\nordinary laptop CPU"),
            ("36.8", " MB", "the entire model, quantised\nand running offline"),
            ("4", " terrains", "scored separately rather than\naveraged into one number")]):
        stat(s, X0 + i * 3.15, y2 + 0.12, 3.00, val, unit, cap, size=23)


# ------------------------------------------------------------------------- slide 6
def slide_references(s) -> None:
    for sh in by_name(s, "TextBox 8"):
        drop(sh)

    _, tf = textbox(s, X0, 1.22, 12.5, 0.24)
    write(tf, "Each item below changed what we built. The note beside it says how.",
          size=10, colour=SLATE, first=True)

    cols = [
        ("Research", [
            ("Depth Anything V2", "Yang et al., NeurIPS 2024. The backbone we fine-tune. "
                                  "arxiv.org/abs/2406.09414"),
            ("HTC-DC Net", "Chen et al., IEEE TGRS 2023. Head-tail cut and distribution "
                           "constraints. arxiv.org/abs/2309.16486"),
            ("Depth Any Canopy", "Ouaknine et al., 2024. The published recipe for "
                                 "adapting Depth Anything to aerial height."),
            ("Beta-NLL", "Seitzer et al., ICLR 2022. Keeps the mean head learning under "
                         "a heteroscedastic loss. arxiv.org/abs/2203.09168"),
            ("IM2HEIGHT, TSE-Net", "Prior single-view height estimation. Our baseline "
                                   "for what the field already achieves."),
        ]),
        ("Data and reference surfaces", [
            ("IEEE GRSS DFC2019 Track 1", "US3D, Jacksonville and Omaha at 0.3 m GSD "
                                          "with airborne LiDAR truth. Training and "
                                          "validation."),
            ("Copernicus GLO-30", "Free 30 m global DEM, the absolute metric anchor. "
                                  "AWS Open Data."),
            ("GlobalBuildingAtlas", "Published 5.9 m building-height RMSE over Asia. The "
                                    "external bar this work is measured against."),
            ("ISRO Bhoonidhi", "bhoonidhi.nrsc.gov.in. Registered; CartoDEM 30 m and "
                               "LISS-4 are the free tiers."),
            ("Google Open Buildings 2.5D", "Sentinel-2 derived. Used only as a "
                                           "cross-check over India, never as truth."),
        ]),
        ("Tried, measured, rejected", [
            ("Ordinal / binned head", "Trained and scored. All three heads land at a "
                                      "0.43-0.49 slope. No gain over regression."),
            ("Shadow photogrammetry", "Oracle shadows ray-cast from truth reach only "
                                      "r 0.503. The network already reaches r 0.787."),
            ("LDS tail reweighting", "Premise fails here. Pixels above 20 m are 7.5% of "
                                     "building pixels but already carry 78.8% of error."),
            ("Two-model ensemble", "Error correlation 0.925. The models make the same "
                                   "mistakes, so averaging buys nothing."),
            ("Global de-compression", "A linear rescale cannot change correlation by "
                                      "construction. It moves error, it does not remove "
                                      "it."),
        ]),
    ]
    cw = 12.5 / 3
    y = 1.54
    for i, (title, items) in enumerate(cols):
        x = X0 + i * cw
        if i:
            vrule(s, x - 0.24, y, 2.86)
        _, tf = textbox(s, x, y, cw - 0.48, 0.20)
        write(tf, title.upper(), size=7.5,
              colour=ACCENT if i == 2 else SLATE, bold=True, first=True)
        rule(s, x, y + 0.22, cw - 0.48)
        _, tf = textbox(s, x, y + 0.36, cw - 0.48, 2.70)
        for j, (name, desc) in enumerate(items):
            write(tf, name, size=8.8, colour=INK, bold=True, first=(j == 0),
                  space_after=1, line=0.94)
            write(tf, desc, size=7.7, colour=SLATE, space_after=9, line=0.98)

    # Reading the field is table stakes. Our own written-up experiments are the part that
    # is hard to fake, and each was specified BEFORE it was run.
    py = label(s, X0, 4.62, 12.5, "Probes we ran, and what each one closed")
    probes = [
        ("01  Tall-building supervision",
         "Can more tall examples move the tail? No. Training holds 166 buildings above "
         "30 m and tops out at 82.8 m; fine-tuning on them moved transfer bias the wrong "
         "way, -18.22 to -19.10 m."),
        ("03  Guided filtering",
         "Can a post-process sharpen rounded roofs into blocks? No. Guided filtering and "
         "morphological toggle contrast each made every metric worse."),
        ("04  Where the detail went",
         "What governs detail is the ground area one backbone token covers, not the "
         "post-process. That reframed the whole resolution question."),
        ("05  Evaluation resolution",
         "Does the model survive Cartosat-class 0.6-1.0 m input? With auto-zoom, all but "
         "3.6% of the coarse-input penalty is recovered."),
        ("06  Height compression",
         "Found by eye in the viewer, then measured: ours = 0.473 x truth + 2.66 m, and "
         "it cannot be corrected at inference time."),
    ]
    pw = 12.5 / 5
    for i, (name, desc) in enumerate(probes):
        x = X0 + i * pw
        if i:
            vrule(s, x - 0.16, py + 0.02, 1.30)
        _, tf = textbox(s, x, py, pw - 0.34, 1.30)
        write(tf, name, size=8.4, colour=ACCENT, bold=True, first=True, space_after=2,
              line=0.94)
        write(tf, desc, size=7.5, colour=SLATE, line=0.98)


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
