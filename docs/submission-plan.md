# Submission plan — SIH26175 DepthWizard

> ## Start here after a context reset
>
> **This file is the plan of record.** Read it before doing anything else.
>
> **Where things are.** The project is `D:/sih2026` — *not* `C:/Users/Zaid/Documents/sih2026`,
> which holds only a stale `ps2026.html`. The git repo is `D:/sih2026/depthwizard`. Python is
> `/d/sih2026/.venv/Scripts/python.exe`; there is no `pytest` installed, so run test files
> directly. Checkpoints are `D:/sih2026/checkpoints/<run>/best.pt`; eval output and prediction
> rasters are in `D:/sih2026/out/`.
>
> **The state, 12 September 2026.**
> - Deadline **30 Sep** (not the 20th the team brief says). Target submit **28 Sep**. The
>   portal shows **3/500** submissions, so there is no slot to race for.
> - **Team is registered.** The only deliverable left is the **DevUp PPT**.
> - **run07 ships**: per-building **3.464 m**, whole-tile 6.008 m, ECE 0.063. The
>   pre-registered >20 m criterion **FAILED** at −15.6 m against a −13 m bar, and is reported
>   as failed in the deck, the evidence pack and §6b. Do not let that be tidied into a pass.
> - **GAMUS is fully consumed** — there is no more of it to add. 8,724 tiles ship but only
>   6,557 carry RGB; all 2,167 NYC tiles are height-only and unusable for an image model.
> - Live at <https://project5.zaidansari.tech> serving run07; all six demo scenes re-baked on
>   run07; repo **public** at <https://github.com/zaidnansari2011/sih2026-depthwizard>.
> - **Tier 1 and Tier 2 complete** — demo-hall survivability, mouse controls, and
>   the geospatial identity (CRS, datum, coordinates, north arrow). That is the 50%.
>   **Tier 3 is next**: A6 (hand back the DSM; the scene picker also exposes other
>   visitors' uploads) then A5 (the upload path a judge will actually use).
>
> **Six things that cost real time — do not rediscover them.**
> 1. **The viewer is served at the site ROOT.** `/viewer/main.js` is a **404**; the script is
>    `/main.js`. Checking the wrong path looks exactly like a failed deploy.
> 2. **`deploy/dwz-app.zip` is a Docker build context — never zip-deploy it** (doing so caused
>    a two-day outage). Use `python deploy/make_appservice_zip.py`, which writes
>    `dwz-appservice.zip` and checks the layout before writing.
> 3. **`export_terrain.py` rewrites `manifest["files"]` from scratch**, silently dropping the
>    inundation tool's `buildings` entry. `build_scenes.py` now re-bakes it automatically — do
>    not bypass it, and check for the `flood` marker in its summary output.
> 4. **`az webapp config appsettings list` is sandbox-blocked** as credential material.
>    `--query "[].name" -o tsv` works, and is enough to confirm `DW_CKPT` and `HF_HOME` stay
>    unset (they must: Oryx runs the app from /tmp, so absolute paths point at nothing).
> 5. **Writing Python through a bash heredoc mangles backslash escapes.** An escape meant to
>    land literally in the target file collapses on the way through. Build those strings with
>    `chr(92)` and `chr(10)` instead, and prefer forward slashes in paths.
> 6. **Headless Chrome under `--virtual-time-budget` renders about six animation frames for
>    the whole session** (counted, 12 Sep). Anything gated on a frame counter therefore never
>    runs there: the scale bar needs 10 frames and the FPS readout needs 30, so both look
>    permanently broken in a DOM dump and are fine in a real browser. Do not "fix" either on
>    headless evidence; force the element visible in the probe and measure its geometry, which
>    is what `verify_layout.py` does.
>
> **How to re-verify anything, rather than trusting a claim:**
>
>     python tools/analysis/paired_bootstrap.py ../out/eval_run02_ship ../out/eval_run07_ship
>     python tools/verify_viewer.py          # headless render of the standalone
>     python tools/verify_layout.py          # panels fit four screen sizes, worst case open
>     python tools/verify_controls.py        # drag/wheel/pan move the camera, panels do not
>     python tools/verify_geo.py             # coordinates checked against rasterio
>     node tools/test_flood.mjs              # inundation tool, 7 checks
>     python tests/test_losses.py && python tests/test_gcp_affine.py
>     python tools/ppt/make_deck_c.py && python tools/ppt/export_pdf.py C
>
> A deployed model is verified by **what it serves**, not by what the deploy reports. Upload a
> tile and compare the returned scene's `height_max_m` and `sigma_mean_m` against a local
> single-pass run of each candidate checkpoint. On `OMA_288_042` that separates them cleanly:
> run07 gives 38.3 m and sigma 1.54, run02 gives 29.6 m and 0.87.

**Deadline: 30 September 2026.** **Target submission: 28 September**, leaving 29–30 Sep as
buffer. The deliverable is the DevUp PPT (`docs/SIH2026-DevUp-SIH26175-DepthWizard-C.pptx`,
exported to PDF); everything else in this repo exists to be evidence behind it or to survive
a judge following a link.

Verified from the portal 12 Sep 2026: SIH26175 reads **3/500 submissions**, deadline column
**30-09-2026**. There is no slot to race for, and no reason to submit on the last day into a
government portal under deadline load. This also closes PLAN §3's open check — the problem
statement is not swamped, so we stay on SIH26175 rather than hedging to SIH26143.

## Where we stand, 12 September

- [x] **Team registered.** Was the one blocker that could not be fixed by working harder.
- [x] **run07 ships.** Per-building 3.464 m; the pre-registered −13 m criterion failed at
      −15.6 m and is reported as failed. See `gamus-integration.md` §6b.
- [x] Evidence pack and deck both on run07, cross-domain matrix included.
- [x] Live site serving run07 (`depthwizard-sih2026.azurewebsites.net`, custom domain
      `project5.zaidansari.tech`), verified end to end by upload.
- [x] Six demo scenes re-baked on run07; inundation tool re-baked and passing 7/7.
- [x] Standalone viewer rebuilt and verified headless.

## What the submission is judged on

An *idea* submission: approach plus evidence, not a finished product. The evidence it wants
is a benchmark table, error maps, a calibration curve, and a demo video. Three of those four
exist and are current. **The demo video does not exist, and the deck has an empty dashed
placeholder where its QR code belongs.** That is the single largest hole.

Standing rule 1 still governs the rest: the viewer is 50% of the marks, so viewer work
outranks further model work, and the video must be recorded *after* the viewer is finished
or it needs a retake.

---

## Tier 1 — cheap, and a judge may hit it

- [x] **A4: publish the repo.** **Done 12 Sep.** Repo is **public**, `master` is the default
      branch and is current (fast-forwarded, no merge commit), everything pushed.
      Apache-2.0 `LICENSE` — GitHub reports `licenseInfo.key = apache-2.0`, so it renders a
      licence rather than "no license". `NOTICE.md` covers the model, three.js, GAMUS, Maxar,
      Google Open Buildings, Copernicus and DFC2019, and says which we do *not* redistribute.
      Maxar's CC-BY credit also travels **inside** the product as `imagery_source` and prints
      in the viewer. README now opens with the live URL, the 3.464 m headline, the
      tall-building failure, the error-map figure and links to all six docs.
- [x] **Track `tools/ppt/`.** **Done 12 Sep.** Narrowed to `tools/ppt/*` with a
      `!tools/ppt/*.py` negation, so the 14 generators (160 KB) are versioned and the 216 PNGs
      and built decks (44 MB) stay out of a repo a judge clones.

### Tier 1 verification, 12 Sep

- Anonymous `git clone` of the public URL returns **136 files**, including `depthwizard/dem.py`
  (Key Milestone 2, "Scale Calibration"), `tools/ppt/make_deck_c.py` and the evidence figures.
- Full-history secret scan: no `kaggle.json`, `.pem`, `.env` or `access_token` ever committed;
  no credential-shaped strings in any blob across all 135 files ever added. Auth modules read
  from paths outside the repo.
- Both test files **pass when run directly** (`python tests/test_losses.py`), which they did not
  before — they died on `ModuleNotFoundError` from a fresh clone, the same papercut class as
  the `--help` crash.
- Imagery credit confirmed by rendering `hilly_sikkim_valley` headless and reading the string
  back off the DOM, not by assuming the code path runs.
- README has zero broken local links.

## Tier 2 — the viewer, which is 50% of the marks

Do these in order. A2 first because it is the cheapest insurance on the biggest block.

- [x] **A2: survive a demo hall.** **Done 12 Sep.** Three failure modes, each measured before
      it was fixed and re-measured after.
      - *Panels off the bottom of a projector.* With the flood controls, legend and upload
        card open the HUD wants **889 px**; a browser on a 1366×768 hall laptop gives the page
        **673**. `html, body` carry `overflow: hidden`, so the rest did not scroll, it ceased
        to exist — and the HUD lay across the key hints. The hand-tuned absolute offsets are
        gone; there are now two full-height flex columns, the HUD and stats scroll internally,
        and the scale bar, key hints and readout are pinned to the foot of their column.
        Overlap is now structurally impossible rather than merely unlikely.
      - *A refused pointer lock painted an unrecoverable error over a working scene.* Chrome
        rejects the lock for ~1 s after Esc and while the window is unfocused; that rejection
        reached the global `unhandledrejection` handler, which showed the full-screen overlay
        reading "failed to load a scene" over a scene that was on screen and fine. Failures
        now route on whether anything is up yet: before the first scene, the overlay; after
        it, a dismissible note beside the scene.
      - *No GPU gave a minified three.js stack trace.* Preflighted, with an answer the person
        at the machine can act on. A `webglcontextlost` handler covers the same ground — a
        projector being plugged in used to freeze the image under a live frame counter.
- [x] **A1: orbit and wheel zoom.** **Done 12 Sep.** `grep -c wheel viewer/main.js` was 0:
      the wheel did nothing, a drag did nothing, and pointer-locked WASD was the only way to
      move — so the honest reading was that the viewer had no mouse controls. Drag orbits,
      wheel zooms, right-drag pans; fly mode is kept, on a tap, with a four-pixel threshold so
      an orbit neither captures the mouse nor drops a measurement pin. Both modes share `yaw`
      and `pitch`, so switching needs no handover, and the pivot is derived fresh from what
      the middle of the view rests on rather than stored — WASD moves the camera without
      touching it. The wheel is bound to the canvas, never the window, because the HUD scrolls
      its own overflow since A2.
      Verified by `tools/verify_controls.py`, which works on pixels because nothing in the DOM
      reflects the camera: each gesture is dispatched as a real event on the real canvas, the
      surface must move and the panels must not. It found two things — the scale bar's box had
      been stretching to the width of the key-hints line since A2 (now hugs its bar at a
      measured 140 px), and measurement needed covering, since "a drag is not a click" is
      exactly the rule that could break it. Two taps report 113.11 m, and that check was
      confirmed to **fail** when the click threshold is deliberately broken.
- [x] **A3: say where on Earth this is.** **Done 12 Sep.** The stats panel now carries a
      *Where this is* block — centre in longitude and latitude, grid centre in eastings and
      northings, the coordinate system with its human name, the datum — plus a north arrow
      that follows the camera and a position under the cursor in the hover readout. The
      confidence figure now says what it is worth: ECE 0.063 held out, 0.044 on a city never
      trained on.
      The half worth defending is the other one: **four of the six demo scenes are DFC2019
      tiles with no CRS at all**, and on those the viewer says exactly that, shows no
      position and draws no arrow. A rotated transform is refused the same way, because a
      compass that is quietly wrong is worse than none.
      No projection library ships to the browser — `export_terrain.py` bakes each tile's
      four corners in WGS 84 and in its own CRS and the viewer interpolates, which was
      **measured** at within 4 mm over the 2 km Sikkim tile, 244× finer than one pixel.

### Tier 2 verification, A2, 12 Sep

Each claim below was produced by a command, not by reading the diff.

- `tools/verify_layout.py` — new. Instruments a **copy** of the built page with a probe that
  reads `getBoundingClientRect()` and `scrollHeight` off every panel after the render loop has
  run, at four screen sizes, in two states: as a judge finds it, and with every block the HUD
  can reveal at once. **7 problems before the fix, 0 after.** The worst was the HUD 230–278 px
  below the bottom edge with no scrollbar.
- The corrected `verify_viewer.py` was itself checked against a page that really is broken —
  Chrome run with `--disable-webgl` — which also proves the preflight: **zero `<canvas>`** and
  the instruction on screen instead of a stack trace.
- A real rejected promise fired at a live page: `#loading` **stays hidden**, canvas and stats
  intact, reason in the corner. Same for a forced `WEBGL_lose_context`.
- 1024×768 screenshot read back by eye — it is what caught the toast covering the key hints,
  which the numeric check then adopted as a permanent pair.
- Unchanged and still green: `verify_viewer` 8/8, `test_flood.mjs` 7/7, `test_losses` 19/19,
  `test_gcp_affine` all pass.

**Found while deploying, unrelated to A2 but worth the sentence:** the deploy staging tree had
drifted from the repo, so the **Maxar CC-BY imagery credit was missing from the live site** in
both `main.js` and the two Sikkim manifests — the repo had it and the deployment did not.
`make_appservice_zip.py` now refreshes every mirrored file from the repo before packing, and
leaves alone the one file that is meant to differ (`scenes/index.json`, pruned by `stage.py`).

### Tier 2 verification, A1 and A3, 12 Sep

- `tools/verify_controls.py` — new. Works on pixels, because nothing in the DOM reflects the
  camera and a test hook has no business shipping in the product. Each gesture is a real
  event on the real canvas; the surface must move and the panels must not. **It found two
  things**: the scale bar's box had been stretching to the width of the key-hints line since
  A2, and measurement needed covering because "a drag is not a click" is exactly the rule
  that could break it. The measure check was then confirmed to **fail** against a copy with
  the click threshold deliberately broken.
- `tools/verify_geo.py` — new. Recomputes every claim from each scene's own transform through
  rasterio and compares it with what the page renders: centre agrees to 4.2 m (the width of
  4 decimal places), eastings and northings agree exactly. **It caught a real 330 m error**
  on first run — position was being read from `geo.transform`, which the standalone's
  decimation invalidates.
- The needle is verified by turning the camera a known amount and reading the angle back:
  −40.1° against a predicted −40.1°. Its *direction* is verified separately, from the data
  and the source, because a needle 180° out passes a rotation test.
- Screenshots were read by eye at 1024×768 and 1366×768, on both a flat tile and the Sikkim
  mountainside, which is how the stretched scale bar and a badly wrapped CRS row were found.

**One check reports SKIP rather than passing or failing.** The cursor-position readout is
throttled to one raycast per animation frame, and headless Chrome renders about a dozen
frames for a whole session, so the sample sometimes lands before any frame runs. Calling
that a pass would be a lie and calling it a failure would be a false alarm.

## Tier 3 — problem-statement compliance

- [ ] **A6: hand back the DSM.** The PS names a geospatial output and ours is CLI-only; the
      default is AGL with no rDSM/nDSM naming. Same item fixes the scene picker exposing other
      visitors' uploads.
- [ ] **A5: the upload path a judge will actually use.** A 512×512 crop crashes; tracebacks
      leak to the client; the job poll never terminates on error. `--help` was fixed 12 Sep.

## Tier 4 — evidence depth, first to be cut

- [ ] **B: the GSD ladder.** ISRO evaluates at roughly 0.6–1.0 m and we train at 0.3 m.
      Pre-registered in `feature-plan.md`. The existing probe-05 table already supports the
      claim but is the only run02 figure left in the evidence pack, so this refreshes it too.
- [ ] **C: an ISRO-imagery scene.** Gated on B returning a number.

## Then, and only then

- [ ] **Demo video.** Explicitly required. Record against the finished viewer so it does not
      need a retake. Fills the deck's dashed QR placeholder — set `PITCH_VIDEO_URL` in
      `tools/ppt/make_deck_c.py` and rebuild, which turns the placeholder into a real code.
- [ ] **Final deck rebuild + PDF export**, after the video URL is set.
- [ ] **Submit.**

---

## Suggested shape of the remaining 16 days

| dates | work |
|---|---|
| 13–14 Sep | Tier 1 (repo published, `tools/ppt` tracked) |
| 15–19 Sep | Tier 2 (A2 → A1 → A3) — the 50% |
| 20–23 Sep | Tier 3 (A6, A5) |
| 24–25 Sep | Tier 4 (B, and C only if B lands early) |
| 26–27 Sep | Demo video, then final deck rebuild and PDF |
| **28 Sep** | **Submit** |
| 29–30 Sep | Buffer. Do not plan work here. |

**If behind, cut in this order:** C, then B, then the accuracy-evidence half of A3. Say so in
the proposal as a stated next step — a named gap reads better than a thin experiment. Never
cut the demo video or Tier 2; they are the required artefact and the 50%.

---

## Progress log

Append one line per working session. Keep it factual — what moved, what was measured.

- **12 Sep (later still)** — **Tier 2 complete: A2, A1 and A3.** Viewer chrome rebuilt on two flex
  columns after measuring an 889 px HUD against a 673 px viewport; late failures no longer
  take the screen; WebGL is preflighted. New `tools/verify_layout.py`: 7 layout problems
  before, 0 after, across four screen sizes. `verify_viewer.py`'s fatal-panel check was
  found to be an unconditional pass and was fixed, then tested against a genuinely broken
  page. Deploy staging had drifted: the live site was missing the Maxar CC-BY credit, and
  the packager now refreshes from the repo. Deployed and verified by what it serves.
  Then A1: drag-orbit, wheel zoom, right-drag pan, tap to fly, checked in pixels by
  the new `tools/verify_controls.py`. Then A3: coordinate system, datum, centre
  position, cursor position and a north arrow, with the four unreferenced tiles saying
  plainly that they have none. `tools/verify_geo.py` checks the numbers against
  rasterio and caught a 330 m error from applying the source transform to the
  standalone's decimated grid.
- **12 Sep (later)** — **Tier 1 done.** Repo public with Apache-2.0 and NOTICE, master
  current, deck generators tracked, tests fixed to run from a clean clone, Maxar credit
  rendering inside the viewer. Verified by anonymous clone and a full-history secret scan.
- **12 Sep** — run07 shipped and deployed; evidence pack, deck, six demo scenes and the
  standalone all moved to run07. Four published numbers found unreproducible and corrected
  (§6b bootstrap protocol, the error-map building, forested ground error, the Sikkim
  co-registration). Deadline corrected from 20 → 30 Sep across README and PLAN. `infer.py
  --help` crash fixed. Portal checked: 3/500.
