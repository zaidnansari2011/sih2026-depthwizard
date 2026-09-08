# Feature plan — 8 September 2026

Four pieces of work, planned before any of them is started. Deadline is **30 September**,
so this exists to stop us building something that demos badly or cannot be defended.

Every feature below is justified against a **scored** line of the problem statement. The
rule for all of them: a feature we cannot explain to a Space Applications Centre specialist
in one sentence, without hedging, does not ship.

## How these map to what is actually scored

| Feature | PS hook | Currently |
|---|---|---|
| A. Inundation tool | **Theme: Disaster Management** | nothing in the project addresses it |
| B. GSD ladder to LISS-4 | *"final evaluation uses ISRO RGB-band optical satellite i…"* | validated to 1.0 m only |
| C. ISRO-imagery scene | same, plus jury affinity | Sikkim uses Copernicus terrain, not ISRO data |
| D. Scale bar + demo video | *"navigability of the 3D flythrough"*, 50 % of the score | both open in `viewer-design.md` |

---

# A. Inundation tool

**The pitch, in one sentence.** Drag a water level and the viewer shows which ground floods,
which buildings go under, and which are too close to call given our own uncertainty.

## Why the obvious implementation is wrong

Three reasons a specialist would reject a naive version on sight. All three shape the design.

**1. Our model outputs AGL, not elevation.** Thresholding height-above-ground means "every
building shorter than X floods", which inverts reality: a 3 m hut on a ridge would flood
before a 30 m block in a valley. Flooding is governed by **absolute elevation**, so the tool
must run on `terrain + AGL`, and terrain has to come from the DEM anchor.

**2. A bare threshold is not a flood.** It fills enclosed pits that no water can reach. The
standard first-order model is a *connected* planar fill from a water body or domain edge.
We implement the connectivity and we **call it a planar bathtub model in the UI**, because
that is what it is. It ignores flow, volume, time and infiltration.

**3. Buildings are instances, not pixels.** "47 buildings flooded" requires a building mask
and connected components. A pixel count would be dominated by large roofs and is not a
number anyone can act on.

## What each scene can actually support

This is the constraint that shapes the whole feature. Measured 8 Sep:

| Scene | terrain | CRS | truth | building mask | flood tool |
|---|---|---|---|---|---|
| Urban — Omaha | ✗ | **null** | ✓ | ✗ | **disabled** |
| Sparse — Omaha | ✗ | null | ✓ | ✗ | disabled |
| Forested / Mixed — Jacksonville | ✗ | null | ✓ | ✗ | disabled |
| **Hilly — Sikkim (valley, town)** | **✓** | **EPSG:32645** | ✗ | via Open Buildings | **enabled** |

The DFC2019 scenes carry no georeferencing at all — `crs: null`, `px_units: "pixels"` — so
absolute elevation for them cannot be recovered, and a DEM cannot be fetched for a tile
whose location is not encoded. **The tool disables itself there and says why**, exactly as
the compare and error tools already disable themselves on Sikkim. That pattern is
established in the viewer and reads as care, not absence.

Building instances on Sikkim come from **Google Open Buildings 2.5D** — 585 footprints, 168
above 0.85 presence, already cached in `data/open_buildings` with working tooling. It is a
model output, not ground truth, and the UI must say "footprints: Open Buildings" the same
way `evidence-pack.md` §India does.

## Why Sikkim is the right demo and not a consolation prize

It is the Indian scene, it is real mountain terrain, and mountain flooding is the live
hazard class there — the Teesta basin has a documented glacial-lake outburst history. An
ISRO jury looking at Indian terrain under an Indian hazard is a stronger five seconds than
the same slider over Omaha. **State the hazard class generically; do not claim our specific
tile was a disaster site**, which we have not verified.

## Definitions, fixed now

Let `E(x) = terrain(x) + AGL(x)` be surface elevation, `σ(x)` our per-pixel uncertainty,
and `L` the water level.

- **Flooded ground**: cells with `E < L` **connected** to the domain edge at or below `L`
  (4-connectivity BFS seeded from every boundary cell below `L`).
- **Building inundated (confident)**: footprint median `E + σ < L`.
- **Building uncertain**: `|median E − L| ≤ σ`. Reported separately, never folded into the
  confident count.
- **Building dry**: everything else.

The σ term is the point of the feature. Anyone can threshold a raster; **only a model with
calibrated per-pixel uncertainty can say which answers it does not trust**, and ours is
calibrated (ECE 0.076, σ/error rank correlation +0.87 on the pilot).

## Interface

Following `viewer-design.md`: plain words outside, precision inside, no new saturated colour
except the water itself.

- A **Water level** slider in the existing control panel, in metres, absolute elevation,
  with the scene's terrain range as its bounds. Default **off**, so nothing changes until
  asked.
- A translucent water plane at `L`, clipped to the connected flooded region.
- A readout, matching the `#readout` panel pattern:
  ```
  Water level          1 812.0 m
  Ground under water        18.4 %
  Buildings flooded             47
  Too close to call              6
  ```
- "Too close to call" is the plain-language rendering of the σ band. It must have a tooltip
  or sub-line: *"within our margin of error at this level"*.

## Acceptance criteria

Not done until all of these hold.

1. Flooding at a level below the terrain minimum floods **nothing**; above the maximum
   floods **everything connected**. Both checked.
2. An enclosed depression below `L` but unconnected to the edge stays **dry**, and this is
   demonstrated on a synthetic raster in a unit check.
3. Building counts reconcile: confident + uncertain + dry = total footprints in scene.
4. The tool is **absent, not broken**, on all four DFC scenes, with a one-line reason shown.
5. Moving the slider holds ≥ 30 fps on the 2048² Sikkim scene — recompute on
   `pointerup`/debounced, never per frame.
6. The standalone build and the hosted site both show it, verified after regeneration.
7. Every number carries a unit; nothing says "sigma", "nDSM" or "AGL".

## Effort and risk

Roughly a day. The real risks, in order:

- **Connectivity performance** on 2048² at interactive rates. Mitigation: compute at half
  resolution for the mask, upsample for display; the water edge is not a precision claim.
- **Open Buildings footprints are on a different grid** than the scene raster. They must be
  reprojected into EPSG:32645 and rasterised at bake time in `build_scenes.py`, not in the
  browser.
- **Scope creep into hydrology.** We are not modelling flow. If this starts acquiring
  rainfall or drainage parameters, it has failed.

---

# B. GSD ladder — extend probe 05 to the resolutions ISRO actually distributes

## Why this is the highest-value experiment left

Probe 05 validated 0.3 → 1.0 m and showed auto-zoom recovers 0.6 m to within **3.6 %** of
native. Genuinely good. But it stops at 1.0 m, and two facts sit past that edge:

- The PS sentence naming the evaluation imagery is **truncated mid-word** on the portal
  (*"ISRO RGB-band optical satellite i…"*), so which sensor is a **guess**.
- `tools/open_buildings.py` already records that **Bhoonidhi has no sub-metre optical.** Its
  RGB optical is ResourceSat **LISS-4 at 5.8 m** and LISS-3 at 23.5 m.

If evaluation uses Cartosat (0.65–1.13 m) we are covered. If it uses LISS-4, we have **no
evidence at all**, at 6× beyond anything tested. That is the largest unquantified risk in
the 50 % accuracy score, and it is answerable with inference alone — no training.

## Method

Reuse probe 05's harness exactly; only the ladder changes. Downsample RGB to the target
GSD, infer, upsample the prediction to the truth grid, score against the **same** LiDAR.
Truth never moves.

- **Ladder**: 0.3 (native), 0.6, 1.0, 2.0, 3.0, **5.8 (LISS-4)**, 10.0, **23.5 (LISS-3)**.
- **Both** auto-zoom off and on, since the auto-zoom recovery is the actual claim.
- **Report per tile, never the mean alone.** Probe 05's own lesson: the mean was dominated
  by one urban tile saturated by a 93.6 m building, and it hid a doubling of error on
  ordinary tiles. Same trap, same guard.
- Score per-building as well as per-pixel; per-pixel understates the damage to structures.

## Pre-registration, written before the run

We claim a **usable range**, defined as: the coarsest GSD at which per-building RMSE with
auto-zoom stays within **1.5×** its native value on non-saturated tiles. We commit to
publishing that number wherever it lands, including if it lands below 1.0 m and
embarrasses the auto-zoom claim.

Falsification: if auto-zoom does **not** beat no-zoom at ≥ 2.0 m, probe 05's token-footprint
account does not extend, and we say so.

## Acceptance

1. A table: GSD × {zoom off, zoom on} × {per-pixel, per-building, bias}, per tile and pooled.
2. One sentence stating the usable range, quoted with its protocol.
3. A figure suitable for the deck.
4. Written to `docs/probe-07-gsd-extended.md` with the pre-registration above **unedited**.

Half a day, inference only. Must not run while a training job holds the GPU.

---

# C. ISRO-imagery scene — gated on B

**Do not start this until B has a number.** Bhoonidhi's best optical is 5.8 m. If B shows
the model collapses there, an ISRO-imagery scene would ship a visibly bad reconstruction to
the exact jury we are trying to impress, and would be worse than not doing it.

- **If B says 5.8 m is usable**: pull LISS-4 MX over an Indian city, run the georeferenced
  path, anchor with Copernicus GLO-30, bake a scene, and the inundation tool works there
  too because it will have terrain and a CRS.
- **If B says it is not**: write the negative result up. *"We measured our usable range at
  X m; ISRO's openly distributed optical is 5.8 m; here is the gap and what would close
  it"* is a serious, defensible statement, and considerably better than silence.

**Licensing note that must not be quietly reversed.** `depthwizard/dem.py` excludes CartoDEM
as a shipped anchor because it is free only to Indian Government Entities and we are
students. Using it as a *validation reference* is a different act from redistributing it,
but that reasoning is on record and any change needs a deliberate decision, not a drift.

---

# D. Viewer completeness

Small, and both are open items in `viewer-design.md`.

**Scale bar.** Every serious DEM viewer has one and its absence is noticeable. Bottom-left,
adaptive to camera height, snapping to 1/2/5×10ⁿ metres. Must read in the standalone build
too. ~1 hour.

**Demo video.** The tour path exists. Record: satellite image → surface rising → drag-to-
compare against LiDAR → one measurement with its ± → **one honest look at where it fails**.
That last beat is what a specialist jury remembers, and it is the same instinct that makes
the σ band the centre of feature A. ~2 hours including a retake.

---

# Sequencing

Ordered by what unblocks what, and by what is safe to run against a busy GPU.

1. **D — scale bar** now. Pure browser work, no GPU, no dependencies.
2. **A — inundation tool.** Browser plus one bake step. Independent of run07.
3. **B — GSD ladder** once the GPU is free. Inference only, but it must not contend with
   training.
4. **C — ISRO scene**, only after B returns a number.
5. **D — demo video last**, so it records the finished viewer rather than needing a retake.

Regenerate **both** derived artefacts after any viewer change — `viewer_standalone.html`
via `tools/build_standalone.py` and the hosted copy via `deploy/stage.py`. `viewer/` is the
single source of truth; the 8 Sep index-pruning bug happened because those two paths had
drifted apart on stale scene entries.

# What we are deliberately not doing

- **Hydrological flow modelling.** Out of scope, and a bad planar model honestly labelled
  beats a bad flow model dressed up as a good one.
- **Retraining for coarse GSD.** Probe 05 showed the coarse-GSD gap is a scale mismatch that
  auto-zoom fixes at no training cost. Retraining would be spending our scarcest resource on
  a problem we already have a free answer to — unless B shows auto-zoom fails past 2 m, which
  is exactly why B is pre-registered.
- **A second uncertainty method.** The one we have is calibrated and is the differentiator.

---

# Audit findings — 9 September 2026

Produced by a 15-agent sweep: six independent lenses over the repo and the official
problem statement, deduplicated and ranked, then each survivor adversarially checked by
an agent instructed to **kill by default** and to prove the thing is not already built.
39 raw findings -> 8 ranked -> **6 confirmed, 2 killed**.

Everything below was verified against the repository, not inferred. Effort figures are
the *corrected* ones from the adversarial pass, which raised several of them.

Nothing here has been actioned yet.

## A1. There is no orbit-drag, no wheel zoom and no touch input — the only camera control is pointer-locked WASD

**Effort 7 h · impact high · lens `viewer-ux #15`**

**What:** Verified: `grep -c wheel viewer/main.js` = 0 (no zoom control of any kind, at any
input), `grep -c OrbitControls` = 0, and the two `touch` hits are both inside the word
"untouched" in comments. Camera motion comes only from `updateCamera()` (main.js ~786-826,
WASD/QE with damping) and a `mousemove` gated on `if (!locked) return;` — before the user
clicks to lock the pointer, moving the mouse does nothing at all. Pointer lock does not
exist on mobile browsers, so the hosted deployment is fully non-navigable on any tablet or
phone a juror opens it on. Work: add a default orbit/pan/zoom mode — left-drag orbits about
the surface point under the cursor, wheel dollies toward the cursor, middle/right-drag pans
— and demote the existing pointer-lock FPS to an explicit "Walk through" toggle rather than
the only mode. Route the same gestures through pointer events so one-finger drag and two-
finger pinch work. Hand-roll it inside main.js rather than vendoring OrbitControls, because
tools/build_standalone.py:98 strips exactly one `import * as THREE` line and a second module
would need that build changed too.

**Why it scores:** "Visualization - Rendering Quality and User Experience (50%): ...
navigability of the 3D flythrough" — a named sub-criterion. Every tool this jury uses (QGIS
3D, ArcGIS Pro, Google Earth, CesiumJS, Potree) is drag-to-orbit and scroll-to-zoom. A
viewer where scrolling does nothing reads as unfinished within five seconds of a specialist
touching it, and the phone they open the link on afterwards shows a frozen image.

**First step:** Add the wheel handler alone, before touching orbit: a ~20-line
`renderer.domElement.addEventListener('wheel', …)` in viewer/main.js next to the existing
click handler (~line 739) that raycasts the cursor against `mesh`, and moves
`camera.position` a fraction of the way toward the hit point (falling back to the camera-
forward axis on a miss), clamped against `extent`. Ship it as its own change: it works in
every mode including before pointer lock, is the single gesture whose absence reads worst,
and collides with none of the existing handlers — unlike left-drag orbit, which has to be
disambiguated from the measure click and the compare divider. Then build orbit/pan and the
Walk-through toggle on top, and regenerate viewer_standalone.html via
tools/build_standalone.py plus the hosted copy via deploy/stage.py, re-checking with
tools/verify_viewer.py.

## A2. The viewer will not survive a demo hall: panels overflow with no scrollbar, and a refused pointer-lock paints an unrecoverable error over a working scene

**Effort 7 h · impact high · lens `viewer-ux #13 (panel overflow, no responsive rules) + viewer-ux #14 (pointer-lock fatal, no WebGL context handling) + robustness #25 (no WebGL preflight, D: paths leaked)`**

**What:** Three stability defects in viewer/index.html and main.js. (1) `.panel` has no
`max-height` and no `overflow`, while index.html:27 sets `html, body { overflow: hidden }` —
measured in headless Chrome at 1366x768 (673 px usable), `#hud` is 783 px in the standalone
and 894 px with the upload block shown, so the colour legend and the upload button are
physically unreachable, not merely clipped. `grep -c '@media' viewer/index.html` = 0: there
is not one responsive rule in the project, and at a 500 px viewport the two fixed-width
panels (274 + 246 px) overlap. Fix: `max-height: calc(100vh - 28px)` with `overflow-y: auto`
and `overscroll-behavior: contain`, make `#stats` collapsible on short viewports, and add
the first `@media` rule so below ~700 px the panels become one collapsible sheet. (2)
main.js:742 calls `renderer.domElement.requestPointerLock()` bare with no `.catch`, and
main.js:43 registers `addEventListener('unhandledrejection', e => fatal(...))`; `fatal()` at
main.js:33-41 fills `#loading` (`position:absolute; inset:0; z-index:10`) with a stack trace
and "Press F12", with no close control. Chrome's `requestPointerLock()` returns a Promise
that rejects with NotAllowedError during the cooldown right after Esc — which is exactly the
gesture the `#help` bar teaches — so pressing Esc then clicking to look again buries the
running viewer with no way back but F5. Fix: `.catch()` and do nothing, narrow the
unhandledrejection handler to scene-loading failures only, and give `fatal()` Dismiss and
Reload buttons. (3) `grep -n "WebGLRenderer|getContext" viewer/main.js` returns exactly one
line — the top-level constructor at main.js:70 — so there is no preflight; on a machine
without WebGL the standalone shows "Error creating WebGL context. at new rc
(file:///D:/sih2026/depthwizard/viewer_standalone.html:327:438862)", leaking your local
paths. Probe `getContext('webgl2')||getContext('webgl')` first, show a plain panel plus a
baked PNG fallback, strip `file:///D:/...` from fatal()'s output, and add
`webglcontextlost`/`webglcontextrestored` handlers (`grep -c -i webglcontext` = 0) so a GPU
driver reset or a laptop switching to a projector's GPU rebuilds the scene instead of
leaving a black canvas.

**Why it scores:** "Visualization - Rendering Quality and User Experience (50%): ...
interface intuitiveness, software stability, and successful standalone deployment." A colour
legend below the bottom of a projector with no scrollbar is all three at once, and the
standalone HTML is the artefact that leaves your hands and gets opened on a machine you do
not control.

**First step:** Add three lines to `.panel` in viewer/index.html (it already has
position/background/border/padding at the `.panel` rule): `max-height: calc(100vh - 28px);
overflow-y: auto; overscroll-behavior: contain;`. This is width-agnostic and fixes the
entire overflow class at every resolution with no media query at all. Then verify: open the
Sikkim scene, click "Flood the valley", and at 1280x720 confirm Fly-through, Measure, Mesh
lines, Reset view and the colour legend are all reachable by scrolling the panel.
Immediately after, make the one-line companion fix -
`renderer.domElement.requestPointerLock()?.catch(() => {})` at main.js:742 - since it costs
a minute and removes an undismissable, mislabeled, F5-only overlay. Finally re-bake BOTH
derived artefacts (`python tools/build_standalone.py` and `deploy/stage.py`), which feature-
plan.md warns about by name after the 8 Sep drift bug. Descope the mobile sheet, the baked
PNG fallback, and the webglcontextrestored scene rebuild; keep the cheap WebGL preflight
with a plain message and stripped file paths.

## A3. The viewer never shows what it already knows — no coordinate, no EPSG, no datum, and none of the accuracy evidence

**Effort 9 h · impact high · lens `viewer-ux #16 (no coordinates/EPSG/datum/north) + differentiation #37 (accuracy evidence invisible in the viewer)`**

**What:** Two additions to the same panel, using data that already exists. (1) Projection:
`viewer/scenes/hilly_sikkim_valley/manifest.json` carries `geo.crs: "EPSG:32645"`, a full
affine `geo.transform` ([0.30517578125, 0, 632843.32, 0, -0.30517578125, 3005156.56]),
`geo.bounds` and `px_units: "metres"` — and `grep -c -i EPSG viewer/index.html
viewer/main.js` = 0/0, `compass`/`north` = 0 outside a comment at main.js:1337. `gridAt()`
(main.js ~899) already converts a world point to a raster index and then throws the
transform away. Add a cursor status line with projected easting/northing plus lat/lon, put
the EPSG code and pixel units in `#stats` next to GSD, name the vertical datum beside
"Height above sea" (infer.py:473 already writes "EGM2008 geoid (Copernicus GLO-30)" into the
summary and it never reaches the screen), and add a small north arrow that rotates with the
camera so the sun-direction slider's "315 degrees" has a visible referent once orbited. (2)
Evidence: `grep -rn "fig_|figures" viewer/index.html viewer/main.js
tools/build_standalone.py` = 0 hits, while `docs/figures/` holds fig_error_by_height.png,
fig_calibration.png, fig_terrain.png, fig_buildings_scatter.png and
fig_error_map_OMA_288_042.png, all unused. Add a "How accurate is this?" drawer off the
existing panel carrying those three figures, the headline table, and the plain-language
limitations (tall-building under-call, canopy penetration, tile borders); inline them as
data URIs in build_standalone.py the same way textures already are, so the single file
carries the evidence too.

**Why it scores:** "Visualization - Rendering Quality and User Experience (50%): Assess
projection accuracy" is the FIRST named sub-criterion and it is currently unaddressed — a
georeferenced product that never shows a coordinate, an EPSG code or a datum gives a SAC
specialist nothing to check, and it is the first thing they will look for. The evidence
drawer serves both halves at once: the artefact a jury opens is the viewer, and the artefact
carrying the 50% accuracy case is a set of markdown files they will never read. "The demo
shows you where it is wrong" is the beat that makes one submission out of five hundred
memorable to specialists.

**First step:** In viewer/main.js updateHover(), extend the existing #hover panel with an
easting/northing row computed from the manifest's geo.transform already in memory: E = t[2]
+ t[0]*g.x, N = t[5] + t[4]*g.y, using the g={x,y} that gridAt() already returns, and render
it only when (m.geo||{}).px_units === "metres" so the four crs:null DFC scenes show nothing
rather than a fabricated coordinate. Add the EPSG code and px_units as one row in #stats
next to "Each pixel", from m.geo.crs, with the same gate. Defer lat/lon, the north arrow,
the datum plumbing and the whole evidence drawer to separate commits -- this first slice is
~45 minutes, touches no Python, needs no re-bake, and makes the georeferenced branch of the
PS visible on the upload path.

## A4. Publish the repo a judge will clone — 31 commits unpushed, the Scale Calibration module untracked, no LICENCE, and a README that misstates the deadline

**Effort 10 h · impact high · lens `ps-compliance #1 (stale repo / missing Scale Calibration) + robustness #23 (31 commits unpushed) + submission #27 (12 days behind) + submission #29 (no LICENSE, unattributed CC-BY imagery) + submission #28 (stale README)`**

**What:** Verified: `git rev-list --left-right --count origin/master...HEAD` = 0/31, HEAD is
on branch `viewer-credibility-pass` with no upstream, `git ls-files | wc -l` = 45 against
~115 present, `git status --porcelain | wc -l` = 70, and `ls LICENSE* NOTICE* COPYING*`
returns nothing. Untracked-and-therefore-unpublished: `depthwizard/dem.py` (the DEM-anchor +
GCP module infer.py imports), `tools/serve_app.py`, `tools/build_standalone.py`,
`tools/prepare_gamus.py`, `tools/bake_buildings.py`, the whole `deploy/` tree,
`tests/test_gcp_affine.py`, `viewer_standalone.html`, and 11 of 14 docs including problem-
statement.md, evidence-pack.md, deployment.md and gamus-integration.md. Work, in one
sitting: (a) commit in coherent chunks and push the branch (`docs/figures/` is NOT
gitignored — it comes along free, which un-breaks the five `figures/fig_*.png` embeds in
evidence-pack.md); (b) add Apache-2.0 `LICENSE` (compatible with the DA-V2 backbone) plus
`NOTICE.md` covering GAMUS CC-BY-4.0, Maxar Open Data CC-BY-4.0 (the Sikkim `texture.jpg`
baked into the 15.2 MB standalone with no attribution — `grep -i maxar viewer/main.js
viewer/index.html` = 0 hits), Google Open Buildings, Copernicus GLO-30, three.js MIT; add an
`imagery_source` manifest field so the credit travels inside the standalone; (c)
`.gitignore` excludes `*.pt`, `*.tif`, `*.bin` and `viewer/scenes/*/`, so a fresh clone
opens an empty viewer — attach a GitHub Release with run02 weights, the int8 ONNX, one
sample GeoTIFF and one baked scene, linked from the README; (d) rewrite README.md, which is
dated 26 Aug: line 5 says "Idea submission closes 20 September 2026" while the team's own
verbatim scrape at docs/problem-statement.md:16 says 30 September, line 33 leads with "Zero-
shot baseline to beat 4.68 m RMSE", and `grep -ci` returns 0 for gamus and 0 for 3.667 — the
shipped per-building RMSE appears nowhere, there are zero images (`grep -c '!\[' README.md`
= 0), and the only internal link is PLAN.md, so evidence-pack.md and deployment.md are
unreachable.

**Why it scores:** Expected Solution: "Deliver a fully integrated software suite with
complete source code and technical documentation." Key Milestone 2 is "Scale Calibration:
Develop a module that converts relative depth to absolute height using scene-level
statistics, low-resolution DEMs, semantic priors, or minimal Ground Control Points" — that
module is depthwizard/dem.py and it is not in the repo a judge would clone. Code with no
licence grant is not usable source, and GitHub will render "no license". Everything else on
this list is invisible until this is done.

**First step:** Do NOT stage anything yet -- make the bulk commit safe first. Append to
.gitignore: `tools/_ortweb/` (120 MB of unreferenced onnxruntime-web vendor files),
`tools/ppt/` (44 MB of build scratch), `docs/SIH25039-*.pdf` and `docs/SIH2025-*.pptx` plus
the root-level SIH2025-IDEA-Presentation-Format.pptx (another team's document and stale 2025
templates). Then re-run `git add -An | wc -l` and confirm the count drops from 933 to
roughly 120-150 before a single `git add` runs for real; eyeball that list once, confirming
docs/figures/ (1.4 MB, five PNGs) is present so the evidence-pack.md embeds at lines
65/78/95/119/139 resolve, and that no kaggle.json or access_token appears. Only then start
the chunked commits, leading with the one that matters most: depthwizard/dem.py plus
tests/test_gcp_affine.py, which is Key Milestone 2.

## A5. The upload path a judge will actually use crashes on a 512x512 crop and shows them a raw Python traceback

**Effort 11 h · impact high · lens `robustness #21 (small-side crash) + robustness #22 (traceback leak, infinite poll) + robustness #24 (no test on inference/serving surface) + ps-compliance #6 and submission #30 (--help crash)`**

**What:** Three measured defects on one path, plus a one-line fix and a harness. (1)
`infer_scene()` (infer.py:150-200) computes `ys = range(0, max(1, H-src+1), stride)` then
appends `H-src`; I verified in isolation that for H=512, src=518 this yields ys=[0] and
`rgb[0:518]` returns 512 rows against a 518x518 `taper` — the broadcast dies at
infer.py:192, or the backbone rejects the non-multiple-of-14 tensor first. Fix: reflect-pad
the RGB up to `src` in each dimension before the window loop and crop `height`/`sigma` back
on the way out; slice the taper to the live extent as a guard. The area cap in
serve_app.py:110-120 does not catch this — a 20000x512 strip is 10.2 Mpx, under the 13.8 Mpx
ceiling, and still crashes. (2) `run_job` raises `RuntimeError((p.stderr or
p.stdout)[-1500:])` at tools/serve_app.py:239 and the handler stores `str(exc)[:600]` as the
job's `step`, which viewer/main.js:1299 (`if (s.state === 'error') { setP(s.step, 100); }`)
writes straight into the progress label — production returned `recent call last):\n  File
"/tmp/8df0c0ea6632463/infer.py", line 548...`, starting mid-word and truncated before the
actual error. Classify failures into plain sentences (too small / unreadable / out of memory
/ timed out), keep the trace in the server log. (3) The poll loop at
viewer/main.js:1287-1301 breaks only on `done` and `error`; `/api/job/<id>` answers 404
`{"state":"unknown"}` for any forgotten job — which happens on every container recycle — so
it spins at 1 req/900 ms forever with `$('pick').disabled` stuck true. `grep -n unknown
viewer/main.js` returns nothing. Treat `unknown` as terminal and re-enable the button. (4)
`infer.py --help` raises `ValueError: unsupported format character 'R' (0x52) at index 177`
— confirmed by running it; the cause is bare `%` at infer.py:254-255 (`-21.3% RMSE`, `+18%
MAE`) where lines 249-250 and 268 correctly use `%%`. Two characters. (5) Add
`tools/smoke_deliverable.py`: a fixture matrix (512px, 300x300, 20000x512 strip, 1-band
uint16, 4-band, CMYK JPEG, grayscale, RGBA, all-NaN float, PNG renamed .tif, truncated TIFF)
run through infer.py + export_terrain.py and then POSTed at a live serve_app, asserting each
either produces a scene or fails with a sentence containing no 'Traceback'; plus `--help` on
all 13 entry points.

**Why it scores:** "Visualization - Rendering Quality and User Experience (50%): ...
software stability" and "Interactive Visualization Platform: Provide a user-friendly 3D
flythrough experience that lets users upload imagery." 512x512 is the single most common
crop size in remote sensing, so it is the likeliest thing a judge hands the demo; `--help`
is the first command a technical judge types at the module the PS calls the Elevation
Estimation Module. A traceback carrying server paths is the most damaging single thing a
stability criterion can see.

**First step:** In infer.py, change the bare `%` to `%%` at lines 254-255 and add a reflect-
pad in infer_scene: after `src` is computed, if H < src or W < src, np.pad the RGB up to src
in each dimension with mode="reflect", run the existing window loop unchanged, then crop
height/sigma back to the original H,W before returning (guard the taper by slicing it to the
live extent). Gate the pad on H<src or W<src so it is a strict no-op for every existing
scene and no published RMSE number can move. Then verify by running `python infer.py --help`
(must print) and pushing a real 512x512 PNG end-to-end through infer.py +
tools/export_terrain.py to confirm it produces a scene.

## A6. The platform never hands back the DSM — the PS's named geospatial output is CLI-only, defaults to AGL, and the picker shows strangers' uploads

**Effort 11 h · impact high · lens `ps-compliance #2 + viewer-ux #17 + differentiation #35 (all: no DSM download) + ps-compliance #3 (AGL by default, no rDSM/nDSM naming) + robustness #24 (visitor uploads in the public picker)`**

**What:** Four changes in serve_app.py / infer.py / export_terrain.py / main.js. (1)
`do_GET` at tools/serve_app.py:318 routes exactly `/healthz`, `/api/job/<id>` and
`/api/capabilities`, then falls through to SimpleHTTPRequestHandler bound to
`directory=str(VIEWER)` — the `OUT` tree is on no served path, and `grep -c -i download
viewer/index.html viewer/main.js` = 0/0. `run_job` already leaves `{job}.height.tif`,
`.sigma.tif`, `.json` (and `.dsm.tif`/`.terrain.tif`) on disk and only
`shutil.rmtree(src.parent)` runs in `finally`, so the rasters survive. Add `GET
/api/result/<job>/<kind>` with Content-Disposition attachment, and a download row in
`#upload-block` (viewer/index.html:229-239, currently a pick button and a progress bar,
nothing else) once state is `done` — GeoTIFF, sigma, the summary JSON, and a one-page
readme.txt naming CRS, units, vertical datum and checkpoint. (2) `--dem` defaults to None
(infer.py:234) so `.dsm.tif` is written only inside `if dsm is not None:` at infer.py:467,
and serve_app appends `--dem auto` only `if ARGS.dem` — docs/deployment.md Known gaps
confirms DEM anchoring is off by default. Default to `--dem auto` when the input carries a
CRS, with a clean fallback that writes the AGL products and says so in the summary JSON if
the Copernicus fetch fails, add `--no-dem`, and turn it on in the deployment env. (3) Name
the artefacts after the PS's own words: `.rdsm.tif` on the non-georeferenced branch, alias
`.height.tif` to `.ndsm.tif`, keep `.dsm.tif` for the anchored product — `grep -rn rDSM
--include=*.py` currently hits only a docstring in dem.py:5. (4) `export_terrain.py` appends
every scene to `viewer/scenes/index.json` with no upload filter (the only exclusion is in
deploy/stage.py, which runs at build time); I fetched the live index and it returns 10
entries — the 6 curated scenes plus `upload_S2A_43RGN_..._L2A_RGB_9c9ac1` (a Sentinel-2 tile
at ~10 m GSD, far outside anything the model is validated on), a second S2 tile, and two
Grand Canyon uploads. Add `--no-index` for the upload path and have `loadScenes(s.scene)`
inject the returned scene client-side for that session only.

**Why it scores:** "Elevation Estimation Module: Accept single-view optical satellite
imagery in PNG, JPG, or TIFF format and output a high-fidelity DSM in a standard geospatial
format", and the Description's two branches verbatim — "Produce an Absolute Digital Surface
Model (DSM) with metric height values" for georeferenced input, "Produce a Relative Digital
Surface Model (rDSM)" otherwise. Both branches are implemented and neither is the default
artefact nor named after the thing asked for, and a judge evaluating through the hosted demo
— the likeliest path — receives no geospatial output at all. The first reflex of an ISRO/SAC
image-processing specialist is to open the DSM in QGIS and check CRS, units and histogram; a
submission where they can do that is in a different category. The picker fix matters because
it is the judge's first click and a stranger's 10 m Sentinel-2 reconstruction sitting beside
Omaha and Sikkim actively weakens the per-terrain stability argument.

**First step:** Fix the false claim before adding any feature: set DW_DEM=1 in the App
Service configuration, and change tools/serve_app.py:241 from the stdout sniff to georef =
Path(f"{out_prefix}.dsm.tif").exists(), so viewer/main.js:1295 says "heights are above sea
level" only when an absolute DSM was actually written and falls back to the relative wording
when the Copernicus fetch fails. Verify by uploading one georeferenced GeoTIFF to the live
host and confirming both the label and that OUT/<job>.dsm.tif exists - that same upload also
proves whether App Service egress reaches copernicus-dem-30m.s3.amazonaws.com, which the
rest of the work depends on. Then add GET /api/result/<job>/<kind> with Content-Disposition
attachment in do_GET after the /api/job branch.

## Killed by the adversarial pass

Recorded because a rejected finding is evidence too.

**The calibrated-uncertainty differentiator fails its own pre-registered ship gate out of domain, and the measurement proving it is cited by no document**

The headline comparison is not apples-to-apples, and correcting it is one command, not a 6 h
work item.  FATAL: 0.2201 was NOT measured on the shipping configuration.
`out/eval_run02_gamus_holdout/metrics.json` records `"tta": false`;
`out/eval_run02_ship/metrics.json` records `"tta": true`, and evidence-pack.md lines 6-7
plus tools/serve_app.py:231 both pin the ship path to `--tta --fuse-zoom 2 --fuse-sigma 8`.
TTA is not a neutral difference for sigma: infer.py:84-119 `_predict_window` returns
`E[sigma^2] + mu_stack.var(0)` — it ADDS an epistemic disagreement term the single pass
cannot see, and that term is largest exactly where the eight D4 views disagree, i.e. out of
domain. Ship sigma_mean 1.082 (TTA on, in domain) vs holdout 1.455 (TTA off, out of domain).
So an unknown and plausibly large share of the 0.077 -> 0.2201 gap is a missing variance
term, not domain shift. The gamus-integration.md runbook step 6 is itself the cause: it
prescribes `evaluate.py --split gdc --corpus gamus` with no `--tta`.  That inverts test 5.
evidence-pack.md line 3 reads "Everything here is measured on the shipping configuration,
not a favourable variant." Writing 0.2201 in beside 0.077 as "the cross-domain calibration
figure" would publish an unlabelled TTA ablation as a domain measurement — a NEW honesty
defect, not a fix. And fitting a `--sigma-temperature` scalar to close a gap that is partly
a measurement artifact is fitting a fudge factor to a config mistake.  TEST 2 FAILS:
docs/gamus-integration.md Phase 4 ("Fill the §6 matrix... the headline deliverable") and
Phase 5 ("Fold into the evidence pack and deck") already commit to exactly proposal (a).
§6's matrix has the DFC2019-only/GAMUS-held-out-city cell sitting blank on purpose,
alongside the ECE<=0.083 gate. The number is not a hidden landmine a jury will find; it is a
pre-registered empty cell whose other two rows are blocked on run07 — which is live right
now (logs/run07.log written this minute, step 10158/14000). Publishing one cell on a
checkpoint run07 may supersede, then rewriting it in two days, is churn.  TEST 3 IS WEAK for
the uncertainty half. The criterion is "DSM Estimation - Accuracy and Validation (50%):
Evaluate RMSE, MAE, and correlation against LiDAR or reference data, including performance
stability across urban, sparse, hilly, and forested landscapes" (docs/problem-
statement.md:71-73). ECE and AUSE appear nowhere in it. The project already owns four
uncertainty measurements — ECE, the k=0.5..3.0 coverage table, sigma/error rank correlation,
and the separation-resolved pair curve in out/pair_calibration_run02 — so AUSE is a fifth
view of an unscored property. The genuinely scored half (whole-tile 6.591 m, per-building
2.792 m, per-terrain spread 8.201 m) is Phase 4/5.  TWO FACTUAL ERRORS IN THE PROPOSAL. (i)
The "boilerplate printing DFC2019 language" claim is false: evaluate.py:240 computes
`{len(tiles)} whole ... from {len(eval_regions)} regions` from the actual split — the val
report prints "16 regions" and the gdc report prints "36", and GAMUS tiles are genuinely
1024 px (prepare_gamus.py:321,328,483). The only soft wording is "region-disjoint", which
gamus-integration.md:279 already discloses as block-level for DC. (ii) "no such flag exists
in infer.py" is correct, but tools/pair_calibration.py already fits scalar multipliers on
the sigma-derived bar, so the pattern is not new.  VERIFIED MISSING (the proposal's only
true novel claims): no `--sigma-temperature` anywhere; zero hits for sparsif/AUSE across the
repo; evidence-pack.md has zero occurrences of GAMUS, cross-domain, generalisation, or out-
of-domain. So the gap is real — it is just already assigned to Phase 4/5 and blocked on a
measurement that must be redone correctly first.  EFFORT: 6 h is not credible for the bundle
as written. Shipping a sigma multiplier means threading it through infer.py, the deploy copy
at deploy/_context/app/infer.py, serve_app.py, tools/export_onnx.py and the int8 artifacts,
the viewer error bars and the flood band, then re-measuring ECE, the coverage table and the
pair curve (fit on unscaled sigma, so it would silently double-count) — on two checkpoints,
since run07 lands first.

**Nothing tests what happens when ISRO's "RGB" is not our RGB — LISS-4 has no blue band, and 16-bit input takes an unvalidated code path**

It is genuinely missing -- I could not refute it on tests 1 or 2 -- but it fails 3, 4 and 5.
TEST 1 (implemented?) NO. `grep` for band order / channel permutation / NIR / false-colour
across the repo returns nothing outside `deploy/_context` mirrors. `depthwizard/dataset.py`
`Augment` is flip, rot90, brightness, contrast, gamma, noise -- there is no channel shuffle.
`tests/` holds only `test_gcp_affine.py` and `test_losses.py`; nothing touches dtype or
`load_image`. The `infer.py:49-51` claim is exact, and I measured the divergence: re-
encoding `JAX_004_006_RGB.tif` linearly to 11- and 16-bit and running both through
`load_image` gives channel means [100.8, 97.2, 113.9] against the uint8 path's [91.3, 88.8,
99.3] -- MAE 20.2 grey levels on identical content, with 2.01% of pixels clipped at each
end. `tools/view_angle.py:274` does say radiometry is "the Cartosat gap proper" and
unmeasured.  TEST 2 (covered by feature-plan?) NO. I read all 247 lines of `docs/feature-
plan.md`. B is a pure resolution ladder; A, C and D are unrelated. Minor correction to the
WHY: `tools/open_buildings.py:3` says only "Bhoonidhi has no sub-metre optical" -- the
LISS-4 identification is in `docs/feature-plan.md:148` and `PLAN.md:418`, not there.  TEST 3
(scored criterion?) WEAKLY, and less than claimed. There is no NIR band anywhere on disk:
`Track1-RGB` is 3-band uint8, and all 951 GAMUS tiles are 3-band uint8 (checked with
rasterio). So the "NIR proxy simulating the LISS-4 composite" -- the row that carries the
entire remote-sensing pitch -- would be synthesised out of RGB, measuring a sensor the
project has never seen. Separately, five of the nine families (gamma 0.6-1.6, per-band
gain/offset, haze veil, sensor noise, and the stretch-percentile choice, which is a
gain/offset in disguise) sit inside the training jitter envelope by construction --
`Augment` already trains against brightness +/-0.2, contrast +/-0.2, gamma exp +/-0.15,
noise 0.01 -- so their outcome is a predictable null. Blur overlaps probe 04 and 06. The one
axis jitter does not cover is band ORDER, and its operational answer needs no probe: a
false-colour composite is unmistakable to anyone looking, and the viewer already displays
the input texture beside the surface, so "surfaces as a bad number, not an error you can
debug" is overstated. The proposed mitigation cannot work either -- histogram matching does
not recover a blue band that was never captured.  TEST 4 (effort?) NO, 5 h is understated
roughly 2x, and the falsification is self-contradictory. The stated falsification -- does
the photometric jitter buy robustness "over a no-jitter counterfactual" -- cannot be
answered by inference; it needs a second training run, which the proposal itself rules out
one sentence earlier ("inference only, no training"). Drop it and the honesty framing
weakens; keep it and the cost is days. The rest: ~150-200 lines of harness against
gsd_probe.py's 74, ~30 conditions x 4 tiles = ~120 `infer.py` subprocesses each reloading
the checkpoint, then mitigation implementation, measurement and write-up. The GPU is at 97%
/ 8.2 GB / 80C on run07 right now, and feature-plan B already reserves the free-GPU slot
ahead of it ("must not run while a training job holds the GPU"), so this queues behind run07
AND behind B.  TEST 5 (honesty?) A LIVE RISK. A table headed with LISS-4, built from a NIR
channel invented from RGB, is exactly the kind of thing this project's own posture forbids
-- the same instinct behind probe 04's "n=1 is not a result" and evidence-pack's
"footprints: Open Buildings" labelling. It is fixable with labelling, but the proposal as
written does not label it.  VERDICT: kill the probe. It is competing against feature-plan B,
which the team has already written down as "the highest-value experiment left", is pre-
registered, unstarted, and gates C -- and against the demo video, which is unstarted and
sits under the 50% visualization half. Salvage the two cheap pieces instead (see first
step): the 16-bit divergence is real and worth a regression assertion, and the band-order
question deserves one documented input-contract sentence, not a probe.

