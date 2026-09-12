# Submission plan — SIH26175 DepthWizard

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

- [ ] **A2: survive a demo hall.** `.panel` has no overflow rule and no responsive rules, so
      panels overflow off-screen at projector aspect ratios; a refused pointer-lock paints an
      unrecoverable error over a working scene; no WebGL preflight.
- [ ] **A1: orbit and wheel zoom.** `grep -c wheel viewer/main.js` = 0. Pointer-locked WASD is
      the only camera control, and a judge who picks up the mouse for thirty seconds is the
      scenario. (The 12 Sep grid-marching raycast made hover cheap enough that adding camera
      controls will not cost frame rate.)
- [ ] **A3: say where on Earth this is.** No coordinate readout, no EPSG, no datum, no north
      arrow, and none of the accuracy evidence is visible in the viewer. ISRO is a geospatial
      agency; a viewer with no CRS reads as a toy. Also surfaces our differentiator
      (calibrated uncertainty) where it is actually seen.

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

- **12 Sep (later)** — **Tier 1 done.** Repo public with Apache-2.0 and NOTICE, master
  current, deck generators tracked, tests fixed to run from a clean clone, Maxar credit
  rendering inside the viewer. Verified by anonymous clone and a full-history secret scan.
- **12 Sep** — run07 shipped and deployed; evidence pack, deck, six demo scenes and the
  standalone all moved to run07. Four published numbers found unreproducible and corrected
  (§6b bootstrap protocol, the error-map building, forested ground error, the Sikkim
  co-registration). Deadline corrected from 20 → 30 Sep across README and PLAN. `infer.py
  --help` crash fixed. Portal checked: 3/500.
