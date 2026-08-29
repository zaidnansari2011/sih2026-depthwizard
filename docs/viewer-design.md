# Viewer design notes

Half the marks. ISRO scores it as *"projection accuracy, visual fidelity, navigability of
the flythrough, interface intuitiveness, stability, and successful standalone deployment."*

## The two rules that outrank every feature below

**1. It has to be readable and usable without explanation.** Zaid's constraint, and it is
the right one: *"the main thing is that the viewer is easy to read and use, it shouldn't be
jargon."* A juror should understand what they are looking at in five seconds, without a
caption, a legend they have to study, or a term they have to already know. If a control
needs explaining, the label is wrong.

This does **not** mean hiding the rigour. It means the plain word goes on the button and the
precise number goes in the readout. "How sure we are, ±2.1 m" beats "sigma (m)" and carries
strictly more information.

**2. No decoration that carries no information.** This is what makes something look
generated rather than built. Purple gradients, glassmorphism cards, emoji headings, a hero
section, controls that exist to look busy. The jury are Space Applications Centre
image-processing specialists who have used QGIS for twenty years.

What reads as an instrument instead:

- The data is the only saturated thing on screen. UI stays near-neutral; all the colour
  belongs to the ramps. **Never a rainbow/jet colormap** — it is the classic amateur
  signal in remote sensing.
- Monospace for numbers. Every number carries a unit.
- One accent colour (`#0b5cab`), used only to show what is interactive.
- Restraint. A viewer that looks slightly austere reads as competent; one that looks
  *designed* reads as compensating.

**3. Light theme.** Zaid's call, and it holds up on grounds beyond taste: it survives a
projector in a demo hall, screenshots drop into the submission PDF without inverting, and
it puts us in the same visual register as QGIS and ArcGIS Pro rather than a game engine.
The register to aim at is a desktop GIS, not a dark-mode dashboard.

Two things a light background forces, both now done:

- **The canvas is `#eef1f4`, never pure white.** Against `#fff` the surface silhouette
  disappears and every edge glares.
- **The ramps run light → dark, not dark → bright**, or the low end dissolves into the
  page. All three are monotonic in lightness, so they survive greyscale printing:
  height is a sequential blue, uncertainty a sequential warm, error a diverging
  blue↔pale↔red centred on zero.
- Ambient light is much stronger than a dark scene needs (hemisphere 1.15), or the terrain
  reads as a dark blob cut out of the page.

## Label discipline — plain word outside, precision inside

| avoid | use |
|---|---|
| sigma / σ (m) | **How sure we are** — "±2.1 m" |
| Signed error vs ground truth | **Where we're wrong** — "blue = we said too low, red = too high" |
| Vertical exaggeration ×3 | **Height boost ×3** (say it is exaggerated, so nobody thinks the terrain is that steep) |
| nDSM / AGL | **Height above ground** |
| GSD 0.31 m | **Each pixel is 31 cm on the ground** |
| Hillshade azimuth | **Sun direction** |

## Status — 27 Aug 2026

Done this session:

- [x] **Four real scenes**, run02+TTA, replacing the week-one `zeroshot`/`truth` pair.
      Each carries our prediction, the LiDAR reference, per-pixel uncertainty and the
      satellite drape in one bundle, registered pixel-for-pixel.
- [x] **Light theme** throughout, ramps re-cut for a light background.
- [x] **Jargon removed** from every visible label (table above).
- [x] **Drag-to-compare** against LiDAR, with the clip plane rebuilt from the camera each
      frame so the divider keeps meaning the same thing while you orbit.
- [x] **"Where we're wrong"** error map, diverging and centred on zero, with LiDAR voids
      greyed rather than reported as zero error.
- [x] **Sun direction** slider, default 315° (north-west, the hillshade convention —
      lighting from the lower right inverts relief perception).
- [x] **`run_viewer.bat`**, because the viewer did not open from `file://` at all.
- [x] **Sikkim scene** — the *hilly* case, standing on real Copernicus terrain (below).
- [x] **`viewer_standalone.html`** — 12.7 MB, all five scenes, double-click, no install.

Still open, in order: hover readout · demo video · re-export from run05 if it wins.

### Sikkim needed the mountain putting back

Our model predicts height **above ground**, so the Sikkim output ranges 0–17.9 m: the
Himalaya is exactly the thing AGL removes. Rendered on its own it is a flat plane with
buildings on it, which answers nothing about *hilly* terrain.

Fixed by fetching Copernicus GLO-30 (open, no auth, `copernicus-dem-30m.s3.amazonaws.com`)
for the tile, cubic-resampling it onto the 0.305 m crop grid, and shipping it as
`terrain.bin`. The viewer stacks our heights on top: **248 m of relief across 500 m at a
mean slope of 24°**, with our buildings standing on it. That is a real DSM — ground plus
everything on it — and it is the actual deliverable the brief asks for.

Two rules that keep it honest:

- **Colour still comes from the above-ground value, never the mesh Y.** Otherwise the
  height ramp silently becomes an elevation map.
- **The stats panel names the provenance**: the mountain is Copernicus at 30 m posts, and
  everything standing on it is ours. Terrain is stored relative to a datum
  (`terrain_datum_m`), or the mesh sits 1.3 km from the origin and every camera default
  aims at empty sky.

Sikkim has no LiDAR, so the compare tool and error map disable themselves and the panel
says so, falling back to the Open Buildings cross-check — 89 buildings, 1.9 m low on
average, agreeing within 3.3 m, and labelled as a satellite estimate rather than truth.

### The standalone build

`tools/build_standalone.py` inlines everything into one file. Nothing in `main.js` changes:

- three.js is minified and ends in a single `export{a as Vector3,…}`; rewriting that to
  `const THREE={Vector3:a,…}` makes it plain inline script.
- `fetch` is shimmed against an embedded asset table. Because the shim owns the bytes
  `main.js` receives, heights are stored **uint16-quantised** and dequantised on the way
  out — half the file size for a few millimetres of precision, against a model whose error
  is metres.
- Textures go through `THREE.DefaultLoadingManager.setURLModifier`, which exists for this.

**`main.js` must be wrapped in its own scope.** Inlined, it shares scope with minified
three.js, which declares `$` — the bundle then fails to parse and the page is blank with
only a `SyntaxError` to show for it. Renaming the clash fixes today's and leaves the next
one waiting; a block removes the class.

### The deployment bug this caught

`index.html` loads as an ES module and fetches the scene binaries. **Both are blocked
under `file://`.** A judge who unzipped the submission and double-clicked `index.html`
would have seen an empty page, against a criterion that says "successful standalone
deployment" in as many words. `run_viewer.bat` covers it; the single-file build removes
the Python dependency entirely and is what should ship.

### Scene labels

Terrain first — `Urban — Omaha`, not `Omaha — urban`. "Urban" tells a juror what they are
about to look at and maps onto ISRO's stated criterion; the place name does neither.

DFC2019 gives us urban, sparse, forested and mixed. It cannot give us **hilly** —
Jacksonville and Omaha are both flat. Sikkim is the only thing that covers that word, and
it has no LiDAR, so its card must say so and fall back to the Open Buildings cross-check.

## What already exists

Four display modes (satellite drape, height ramp, uncertainty, slope), perceptually ordered
ramps, vertical exaggeration, a tour path, wireframe, reset, a legend, a stats panel, and the
click-two-points measurement with a **calibrated** error bar. The tool is not the problem.

## What is missing, in order of return

**0. Fresh scenes.** The viewer currently ships `zeroshot` and `truth` from week one — it is
demonstrating our *worst* model. Regenerate from run02+TTA (or run05). This changes more
than any new feature.

**1. Truth comparison, drag-to-compare.** DFC2019 ships LiDAR ground truth. A divider the
user drags between our surface and real LiDAR is the most convincing five seconds available
to us, and it is literally the "validation" half of the accuracy criterion. Nothing to
explain: the two pictures either match or they do not.

**2. "Where we're wrong" map.** Diverging ramp centred at zero against the truth. Specialists
read error maps instinctively, and it converts our known weakness into evidence of rigour —
the tall buildings glow, and we explain exactly why (see PLAN §9).

**3. Scene picker = the four terrain types ISRO named.** Urban, sparse, hilly, forested.
Answering their stated criterion in the interface itself. We already compute the per-terrain
breakdown in `evaluate.py`.

**4. A Sikkim scene.** Real Indian mountains at 0.31 m, for an ISRO jury. Even without truth
it says "this works on our terrain", and the Open Buildings cross-check backs it
(`docs/evaluation-protocol.md`).

**5. Sun direction slider (hillshade).** Every serious DEM viewer has one and its absence is
noticeable. It also makes the surface legible in the demo video.

**6. Hover readout** — position, height, and confidence, plus a real scale bar.

## Demo video

The tour path already exists. The video should show, in order: the satellite image, the
surface rising out of it, the drag-to-compare against LiDAR, one measurement with its ±, and
one honest look at where it fails. That last beat is what a specialist jury remembers.
