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

- [ ] **A4: publish the repo.** Over 40 commits unpushed to
      `github.com/zaidnansari2011/sih2026-depthwizard`; no LICENSE / NOTICE; CC-BY imagery
      unattributed. README's deadline is now correct.
- [ ] **Track `tools/ppt/`.** It is gitignored, so the deck *outputs* are version-controlled
      but the *generators* are not — every deck edit from 12 Sep exists only on this machine.
      One disk failure loses the ability to rebuild the submission artefact.

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

- **12 Sep** — run07 shipped and deployed; evidence pack, deck, six demo scenes and the
  standalone all moved to run07. Four published numbers found unreproducible and corrected
  (§6b bootstrap protocol, the error-map building, forested ground error, the Sikkim
  co-registration). Deadline corrected from 20 → 30 Sep across README and PLAN. `infer.py
  --help` crash fixed. Portal checked: 3/500.
