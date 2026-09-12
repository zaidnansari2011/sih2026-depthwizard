# GAMUS integration — plan and goal

**Written 6 September 2026.** 24 days to the 30 September idea-submission deadline.
Status: **complete, 12 September 2026.** Every phase has been run, run07 ships, and the
results and corrections are in §6b and Phase 5.

This is a plan of record for one decision: bringing GAMUS in as a primary corpus. It
follows the project convention — predictions and falsification conditions are written
*before* the runs, so that a result can disprove something.

---

## 1. Goal

**Convert our single-domain model into a demonstrably cross-domain one, and close the
tall-building gap that produces 79% of our per-building error.**

Two numbers define success. Both are pre-registered in §6.

---

## 2. Why GAMUS, stated properly

The weak argument is "ISRO recommended it, so use it." The strong argument is three-fold,
and only the first is really about winning.

### 2.1 It is the only route to a credible generalisation claim

The problem statement says final evaluation uses **ISRO RGB-band optical satellite
imagery** — a sensor and a geography we have never seen. Today every number we own comes
from DFC2019 (Jacksonville + Omaha). With one domain we cannot say *anything* measurable
about transfer; we can only assert it.

With GAMUS we can run a real cross-domain matrix — train on one corpus, evaluate on a
different corpus with different cities, a different sensor and a different LiDAR vendor.
**That number is the closest honest proxy we have for ISRO's held-out imagery, and no
competing team will have one.** For a criterion that explicitly rewards "performance
stability," this is worth more than shaving 0.3 m off an in-domain RMSE.

### 2.2 It is 36x our tall-building supervision, which is our one real defect

Probe 01 established, with a pre-registered falsification condition, that the tall-building
failure is **data scarcity, not architecture**: the model fits tall buildings in regions it
trained on (bias −4.61 m) and fails on regions it did not (−18.22 m), because the entire
training set holds **166 buildings above 30 m**. Probe 01 ruled out architecture,
oversampling, tall-enriched sampling and re-splitting, and named the only remaining move:
more real tall-building data.

Measured supply, using our own `building_instances` definition (3x3 components, 25 m²
floor, median height, GSD 0.33), from a 36-tile paired sample projected to all 8,724:

| | GAMUS | DFC2019 |
|---|---|---|
| Buildings > 20 m | ~12,100 | 341 |
| Buildings > 30 m | ~6,060 | 166 |
| Buildings > 50 m | **0** | present (val to 155.3 m) |

### 2.3 It is the sponsor's own recommendation

`IMG-PROCESS-SAC/SIH-DepthWizard-2026`, added 29 Aug 2026. Free credibility in the pitch,
and it costs one sentence. This is the weakest of the three reasons and should never be
the one we lead with.

---

## 3. Why not GAMUS *only*

"Use it fully" is right. "Use it instead" would destroy the thing that makes it valuable.

1. **A cross-domain claim needs two domains.** Dropping DFC2019 removes the second corpus
   and with it §2.1 entirely.
2. **DFC2019 is our entire evaluation history.** run01–run04, every evidence-pack figure,
   the 5.9 m GlobalBuildingAtlas comparison and the shipped 3.667 m headline are all on
   the 80-tile region-disjoint val set, seed 1337. Changing the validation set makes every
   prior number incomparable and forfeits three weeks of measurement.
3. **GAMUS tops out at 44.5 m.** DFC2019 val reaches 155.3 m. Only DFC2019 can score the
   extreme tail.
4. **Neither has hills.** DC/NYC/PHL are flat. Sikkim remains our only hilly evidence
   either way.

**Decision: GAMUS becomes the primary *training* corpus. DFC2019 val stays frozen as the
primary *evaluation* set.** Nothing about the existing protocol changes, so every number
in the evidence pack stays comparable and the new data can only be judged, never assumed.

---

## 4. What GAMUS actually is — measured, not from the card

| property | value |
|---|---|
| GSD | **0.33 m** (ours: 0.31 m) |
| Tile size | 1024 x 1024 (same as DFC2019) |
| Tiles shipped | 8,724 heights / 8,724 classes / **6,557 images** |
| Tiles **usable** | **6,557** — PHL 4,398 + DC 2,159. NYC is height-only (trap 5) |
| Cities | PHL · DC (· NYC, unusable) — all dense US urban, none hilly |
| Labels | nDSM (ground removed), LiDAR-derived — same convention as our output |
| Format | `.h5`, single key `image`, **no attrs, no CRS, no GSD** |
| Licence | CC-BY-4.0 |
| Height scale | p95 26.40 m (DFC2019: 13.2 m); max 128.5 m |

### Four traps, each of which fails silently

1. **Class 3 is BUILDING. Class 6 is TREE.** The opposite of the obvious guess. Verified
   three ways: class 3 is more rectangular (65.2% bbox fill vs 61.9%), flatter-topped
   (within-component height SD 1.79 m vs 2.30 m), and a rendered Philadelphia tile shows
   class 3 tracing rooftops exactly while class 6 covers tree clumps. Getting this backwards
   trains and scores the per-building metric on canopy.
2. **Void is a −5.00 sentinel, not NaN.** Zero NaN anywhere in GAMUS; 2.9% of pixels are
   negative and 12.1% are exactly 0. DFC2019 uses NaN. A shared loader without an explicit
   policy trains on −5 m water as ground truth.
3. **GAMUS's own splits are not spatially disjoint.** DC tiles are named on a grid
   (`DC_row_col`), and adjacent tiles land in different splits **900 times** across
   test/train, 872 across train/val, 213 across test/val. This is precisely the leakage the
   evidence pack criticises HTC-DC Net for. **Their splits must not be used**, and published
   GAMUS benchmark numbers are not comparable to ours.
4. **GSD is 0.33 m against our 0.31 m** — a 6% difference. probe-05 showed GSD matters;
   verify rather than assume it is negligible.
5. **GAMUS ships no RGB for NYC.** Found 6 Sep 2026 when the ingest 404'd on its first NYC
   tile. Heights and classes exist for all 8,724 tiles; imagery exists for only 6,557, and
   the 2,167 missing ones are *exactly* every NYC tile. A model whose input is a photograph
   cannot use a height-only tile, so **the usable corpus is 6,557 tiles, not 8,724**, and
   the planned "hold out NYC" split is impossible. `_list_tiles()` now enumerates from
   `images/` rather than `heights/` so this cannot recur silently.

---

## 5. Disk plan (blocking — do this first)

**Resolved 6 Sep 2026. Free space is now 39 GB** (was 30.7), and the ingest is sized to
fit inside that without touching anything outside the project.

Reclaimed: `checkpoints/tripwire_test/` — two 4 GB checkpoints from a 10-minute tripwire
self-test whose best RMSE was 693. Deleted; `train_log.jsonl`, which holds the actual
result, was preserved to `logs/tripwire_test_train_log.jsonl` first.

**Deliberately left alone:** `D:\gradle-home` (13.7 GB), `D:\hf-cache` (14.5 GB),
`D:\Downloads`, `D:\moved items`, `D:\SteamLibrary` (63.8 GB). Not needed — see the budget
below. If a later phase needs more room, gradle-home is the cheapest next target because it
regenerates on the next build.

**`data/kaggle_stage` is NOT reclaimable** — it is hardlinked to `data/shards` (verified
same inode), so it occupies 0 additional bytes despite reporting 9 GB. Any future disk
audit that recursively sums file sizes will double-count it.

### Budget

Shards cost **3.8 MB/tile** at our packing (RGB uint8, AGL float16, CLS uint8, npz).
Raw `.h5` is never stored: the ingest streams **download → shard → delete**, so transient
usage stays near 1 GB.

Measured on the first 200 tiles actually ingested: **4.2 MB/tile** of shards (RGB uint8 +
AGL float16, npz-compressed, no class raster — see below).

| what | tiles | size |
|---|---|---|
| Training shards — all PHL + DC grid columns 2–48 | 6,204 | **~26 GB** |
| Held-out GeoTIFFs — DC columns ≥ 50, all 3 modalities | 317 | ~3.6 GB |
| Buffer column 49 — discarded on purpose | 36 | 0 |
| **Total** | 6,521 | **~29.6 GB** |
| Free now | | **95.5 GB** |
| **Headroom** | | **~66 GB** |

**The disk constraint is gone.** Counter-Strike 2 was deleted on 6 Sep 2026 at the user's
instruction — 64 GB, the only game installed — taking free space from 33 GB to 95.5 GB.
Its `appmanifest_730.acf`, shader cache and workshop content went with it so Steam does not
still believe it is installed.

Consequently **nothing else needs reclaiming**: the page file can stay system-managed at
29.7 GB, and `gradle-home`, `hf-cache`, `Downloads` and `moved items` are all untouched.

**Class rasters are not downloaded for training tiles.** `dataset.py` only requests `cls`
for the *validation* set (train.py:351) and our validation set stays pure DFC2019, so
fetching them for 6,204 training tiles would move 26 GB of data into an array nothing
opens. They *are* kept for the held-out tiles, where the per-building metric needs them.

The holdout is stored as whole GeoTIFFs rather than shards because whole-tile RMSE is the
deployable metric; reassembling 1024-px tiles from overlapping 518-px crops would be
reconstructing data we could simply have kept.

**Every usable GAMUS tile is either trained on or held out.** Nothing is dropped for disk
reasons — the only discard is the 36-tile buffer column, and that is a correctness measure,
not an economy. "Use GAMUS fully" is satisfied literally.

---

## 6. Pre-registered success criteria

Written before any run. If these are not met, the union model is not shipped.

**Primary — the thing this is for.** On the unchanged DFC2019 val set, the >20 m building
bias must beat **−13 m** (probe 01's original pre-registered threshold; currently −17.26 m).
Anything worse means 36x the tall-building data did not transfer, and the problem is not
what probe 01 concluded.

**Guard 1 — the distribution-shift risk.** GAMUS buildings are taller than our val set
(GAMUS median 6.76 m / p90 10.10 m; DFC2019 val median 4.22 m / p90 7.53 m). The 3–6 m
per-building band, which carries most of the current 3.667 m headline, **must not degrade
by more than 0.3 m** (currently 1.45 m RMSE).

**Guard 2 — no overall regression.** Whole-tile val RMSE must not exceed **6.401 m**, and
per-building RMSE must not exceed **3.667 m**. Both are the shipped TTA+fused figures.

**Secondary — the cross-domain claim.** Report the full matrix. There is no threshold here
because no prior number exists; this establishes the baseline.

| train \ eval | DFC2019 val | GAMUS held-out city |
|---|---|---|
| DFC2019 only | 6.401 m (known) | — |
| GAMUS only | — | — |
| Union | — | — |

**Calibration must survive.** ECE ≤ 0.083 and sigma/error rank correlation ≥ +0.80. The
uncertainty head is our genuine differentiator (§6.1 of PLAN.md) and is not worth trading
for height accuracy.

---

---

## 6b. Results — run07, measured 9 September 2026

run07: 14,000 steps on the union corpus (61 DFC2019 + 192 GAMUS train shards, 76 % GAMUS),
warm-started from `run02/best.pt` with `height_scale` pinned to 13.223477220535276.
75.5 min on the RTX 3060, peak 83 C. Validation stayed pure DFC2019 throughout.

Training-time crop-wise validation, by epoch: 7.979 -> 8.223 -> 7.676 -> **7.644**. The
epoch-1 regression is the learning rate mid-schedule, not the corpus; the schedule anneals
to zero at step 14,000 and the last epoch is the best.

### Against the §6 criteria

Scored on the **shipped protocol** (whole-tile, TTA x8, zoom-2 fusion, 80 val tiles, the
same 3,090 buildings as every prior number), because that is the configuration §6 names.

| Criterion | Bar | run02 | run07 | |
|---|---|---|---|---|
| **Primary** — >20 m building bias | beat −13 m | −17.157 | **−15.642** | **NOT MET** |
| Guard 1 — 3–6 m band RMSE | ≤ +0.3 m degradation | 1.429 | **1.363** *(improved)* | met |
| Guard 2 — whole-tile RMSE | ≤ 6.401 m | 6.401 | **6.008** | met |
| Guard 2 — per-building RMSE | ≤ 3.667 m | 3.667 | **3.464** | met |
| Calibration — ECE | ≤ 0.083 | 0.077 | **0.063** | met |
| Calibration — sigma rank corr | ≥ +0.80 | 0.836 | **0.877** | met |

**The primary criterion failed and that is the headline.** We needed a 4.3 m shift in the
tall-building bias and got 1.5 m. §6 says plainly that an unmet criterion means the union
model is not shipped, and that sentence was written before any number existed.

### But the reason §6 gave for that bar is falsified

§6 justified the −13 m threshold as: *"Anything worse means 36x the tall-building data did
not transfer, and the problem is not what probe 01 concluded."* That inference does not
survive the measurement. Paired bootstrap, 20,000 resamples, run07 vs run02 over the same
buildings (no TTA, so the pilot's figures are directly comparable):

> **Reproduce with** `python tools/analysis/paired_bootstrap.py out/eval_run02 out/eval_run07`
> (add `--markdown`). Written 12 Sep 2026, because the intervals below were originally
> produced by a scratch script and the evidence pack's standing rule is that every figure
> regenerates. The committed tool returns these point estimates exactly; the CIs and p-values
> differ in the third decimal, which is resampling noise and not a correction.

| band | n | run02 bias | run07 bias | delta | 95 % CI | p |
|---|---|---|---|---|---|---|
| **>20 m** | 60 | −17.262 | −15.781 | **+1.481** | **[+0.202, +2.717]** | **0.025** |
| >30 m | 27 | −28.679 | −26.975 | +1.704 | [−0.696, +3.982] | 0.160 |
| 3–6 m | 2,305 | +0.213 | +0.054 | −0.160 | [−0.190, −0.130] | <0.0001 |
| all | 3,090 | −0.314 | −0.419 | −0.105 | [−0.147, −0.062] | <0.0001 |

Per-building RMSE improved −0.220 m overall (p = 0.017) and −1.636 m on the >20 m band
(p = 0.022). The tall-building interval excludes zero: **the data did transfer.** It
transferred less than the bar demanded, which is a different finding from the one the bar
was written to detect.

For contrast, the same test on the 1,500-step pilot gave +0.450 m, CI [−0.517, +1.393],
p = 0.357 — indistinguishable from zero. The pilot could not resolve the effect; the full
run can. That is the expected order of events, not a contradiction.

The >30 m band moves +1.704 m but at n = 27 cannot reach significance **on this pairing**.
DFC2019 val holds only 60 buildings above 20 m and 27 above 30 m, so the sample is thin
however it is scored.

### Corrected 12 Sep 2026 — the bootstrap above used the wrong protocol

The criteria table is scored on the shipping configuration, *"because that is the
configuration §6 names"* — but the bootstrap above was run on the no-TTA pairing. Mixing the
two understated the result. Re-run on the shipping protocol, over the same 3,090 buildings:

| band | n | run02 bias | run07 bias | delta | 95 % CI | p |
|---|---|---|---|---|---|---|
| 0–3 m | 166 | +0.491 | +0.045 | −0.446 | [−0.557, −0.341] | <0.0001 |
| 3–6 m | 2,305 | +0.294 | +0.010 | −0.284 | [−0.313, −0.255] | <0.0001 |
| 6–10 m | 443 | −0.775 | −0.971 | −0.196 | [−0.275, −0.119] | <0.0001 |
| 10–20 m | 116 | −1.029 | −1.098 | −0.070 | [−0.456, +0.345] | 0.72 |
| **>20 m** | 60 | −17.157 | −15.642 | **+1.515** | **[+0.661, +2.424]** | **0.0002** |
| **>30 m** | 27 | −28.122 | −26.227 | **+1.895** | **[+0.407, +3.489]** | **0.011** |
| all | 3,090 | −0.237 | −0.474 | −0.237 | [−0.271, −0.202] | <0.0001 |

Per-building RMSE: −0.202 m overall (p = 0.003), −1.595 m on >20 m (p = 0.0015), **−2.352 m
on >30 m (p = 0.002)**.

**So the >30 m band does resolve, and the sentence above was too pessimistic.** TTA and
zoom-2 fusion reduce prediction variance on both sides of the pairing, which narrows the
interval without touching the sample size — 27 buildings is thin, but it is not too thin once
the measurement itself is less noisy. The honest statement is that the tall-building
improvement is significant at every band above 20 m *on the configuration we deploy*, and
that "no amount of training will change it" was a claim about the instrument that the
instrument did not support.

**Two regressions, both significant, both small.** Overall per-building bias moves away from
zero (−0.237 → −0.474 m) and the 6–10 m band under-calls more (−0.775 → −0.971 m), each with
RMSE flat or better. That is a bias-for-variance trade: run07 buys significantly less scatter
at the cost of a slightly larger systematic under-call. It belongs in the limitations list,
not in a footnote.

**The 10–20 m band is the one that looked bad and is not.** Its RMSE rises +0.534 m, which
would be the worst single number in the comparison if it were real — CI [−0.094, +1.235],
p = 0.11. At n = 116 this split cannot resolve it. Reported here so that nobody later finds
it in `metrics.json` and believes it was hidden.

### Secondary — the cross-domain matrix

Filled at last. Whole-tile RMSE, and per-building RMSE in brackets.

| train \ eval | DFC2019 val (80 tiles) | GAMUS DC holdout (317 tiles) |
|---|---|---|
| **DFC2019 only** (run02) | 6.401 m (3.667 m) | 6.873 m (2.725 m), ECE 0.231 |
| **Union** (run07) | **6.008 m (3.464 m)** | **3.497 m (1.788 m)**, ECE 0.044 |

**State this one carefully or it becomes an overclaim.** run07 trained on GAMUS, so the DC
holdout is in-domain for it and out-of-domain for run02. The split is spatially disjoint —
training columns 2–48, holdout 50–67, column 49 discarded as a 338 m buffer — so there is
no leakage, but this is a **domain-adaptation result, not a generalisation win**. Saying
"we generalise better" would misdescribe it.

The defensible claim is the conjunction: run07 adapts to GAMUS **while giving nothing back
on DFC2019**, improving there too and significantly. A model that had simply chased the new
corpus would show the opposite.

The ECE column is the quietly important one. run02's uncertainty was badly miscalibrated
out of domain (0.231 against 0.077 in domain); run07 is at 0.044. Our calibration holds
where the training distribution covers the test domain and degrades where it does not.
That is worth reporting in both directions.

### Ship decision — taken 12 Sep 2026: run07 ships

**run07 is the shipping checkpoint. The primary criterion stands as FAILED and is reported
as failed.** Those two sentences are not in tension, and the distinction is the whole point:

Pre-registration exists to stop a threshold being moved so a result can be claimed as a
success. Nothing here is claimed. The bar stays at −13 m, the measurement stays at −15.6 m,
and the deck and this document both report a miss. What pre-registration does **not** oblige
is deploying the worse of two models out of literalism — §6's rule was written to protect the
*claim*, and refusing to ship would instead protect the claim by degrading the product.

The bar's own stated rationale is falsified. It was written to detect "36x the tall-building
data did not transfer"; the data transferred at p = 0.0002 on >20 m and p = 0.011 on >30 m.
The bar asked a yes/no question, got "yes, by less than hoped", and the rule attached to it
was written for "no".

Against that: run07 is better on per-building RMSE (p = 0.003), whole-tile RMSE, every band
below 6 m, every band above 20 m, ECE, σ rank correlation, and calibration coverage at every
k — and out of domain its ECE is 0.044 against run02's 0.231, which is our stated primary
differentiator holding where run02's collapsed. The two regressions are sub-0.25 m bias
shifts on bands whose RMSE did not worsen.

**What this decision does not license.** The −13 m bar is not rewritten, the primary criterion
is not reported as met, and limitation #1 is narrowed on the strength of *where the data runs
out* (no open dataset at this resolution holds buildings above 50 m), not on the strength of
having passed something. Anyone reading the deck should be able to find the failed criterion
without looking for it.

### Provenance

`out/eval_run07` (no TTA) · `out/eval_run07_ship` (TTA + zoom-2) ·
`out/eval_run07_gamus_full` and `out/eval_run02_gamus_full` (317-tile DC holdout) ·
checkpoint `checkpoints/run07/best.pt`.

Caveat on the holdout: `evaluate.py --corpus gamus` uses DFC's 0.30 m GSD, so GAMUS's
minimum building footprint is 30.3 m2 against DFC's 25 m2. The two building counts are
therefore not drawn on identical area floors.

## 7. Phases

### Phase 0 — Disk ⚠️ **partly done, 6 Sep 2026 — and now short again**
`tripwire_test` reclaimed (log preserved), taking free space 30.7 → **39 GB**.

**Then `pagefile.sys` grew from 17.0 to 31.9 GB during the first ingest run, taking free
space back down to 24 GB.** The page file is **system-managed** (`InitialSize`/
`MaximumSize` both 0), so Windows expands it under memory pressure and never shrinks it.
Measured immediately after: **29.7 GB allocated, peak usage 2.9 GB, current usage 2.7 GB**
— roughly 27 GB of allocated-but-untouched disk. This machine has 15.9 GB of RAM with
about 1 GB typically free, so the pressure is real but the allocation is far larger than
anything actually paged.

Timing (02:02) points at the 32-worker ingest as the trigger, but that is correlation:
peak page-file *usage* never exceeded 2.9 GB, so the ingest was not itself paging heavily.
The default worker count is now 16 rather than 32 regardless.

**Resolved by deleting Counter-Strike 2** (64 GB, the only game installed) on the user's
instruction. Free space is now **95.5 GB** and no further reclamation is needed — the page
file stays system-managed and `gradle-home` / `hf-cache` / `Downloads` are untouched.

The page-file behaviour is still worth knowing about for the next long job: it grows under
pressure and never shrinks, so **check free space after a heavy run, not just before**.
If it is ever needed, a fixed 8–12 GB page file would reclaim ~18–22 GB against a 2.9 GB
measured peak (System Properties → Advanced → Performance → Advanced → Virtual memory;
needs admin and a reboot).

### Phase 1 — Loader and a cheap pilot (~half a day)
`tools/prepare_gamus.py`, mirroring `prepare_data.py`'s output contract exactly so the
existing `dataset.py` consumes both corpora unchanged.

- Stream download → shard → delete raw, resumable, with a disk-space guard that aborts
  before filling the volume.
- Class remap: GAMUS 3 → our `CLS_BUILDING`; GAMUS 6 → vegetation. **Assert, don't assume**
  — fail loudly if building-class median height exceeds vegetation's by the wrong sign.
- Sentinel policy: −5.00 → void mask, matching DFC2019's NaN handling.
- Write `data/shards_gamus/probe.json` with measured stats, per the existing convention of
  reading facts rather than hardcoding them.

**Pilot on ~500 tiles (≈2 GB)** before committing to the full ingest. Purpose: test Guard 1
cheaply. If short buildings collapse on a 500-tile pilot, that is a sampler problem to
solve before spending 33 GB and a long run.

### Phase 2 — Full ingest and an honest split (~2–3 h, mostly unattended)
**Discard GAMUS's splits** (§4, trap 3) and re-split **by city**: no city appears in two
splits. City-level disjointness is stronger than our existing region-level protocol and is
trivially defensible to a jury.

**Revised 6 Sep 2026** — the original "hold out NYC" design is impossible (trap 5).

The holdout is carved from **DC's grid** instead, which is the only geography any GAMUS
filename exposes (PHL ids are sequential and positionless). Columns **≥ 50** are held out,
column **49 is discarded as a buffer**, and everything else trains. One column is
1024 px × 0.33 m = **338 m of separation**, so no held-out tile touches a training tile —
the same standard as our DFC2019 region split.

Holding out *all* of DC would be a cleaner split and is the wrong call: DC is the tall city
(p95 39.7 m) and its tall buildings are the entire reason for this ingest.

Because the split ignores GAMUS's own, trap 3's 900 leaking adjacent pairs become
irrelevant by construction rather than by filtering.

**This weakens the holdout from city-level to block-level, and that is acceptable** — the
cross-domain claim in §2.1 rests on GAMUS↔DFC2019, which is a genuine change of city,
sensor and LiDAR vendor. The DC holdout is a secondary in-corpus check, so losing NYC costs
a third evaluation set, not the argument.

### Phase 3 — Training (run07+)
Union sampler over DFC2019 train + GAMUS train. Start from run02's shipping checkpoint
rather than from scratch — it is the known-good configuration and the ablations behind it
are closed.

Watch the pre-registered numbers in §6 and nothing else. Per the project rule, score the
run and discuss before queueing another.

Thermals: Afterburner profile (fan 85%, Temp Limit 80, Power Limit 85%, Prioritize =
temperature) **applied before the run starts, never during** — changing it mid-run resets
the display driver and kills the CUDA context.

### Phase 4 — Cross-domain evaluation
Fill the §6 matrix. This is the headline deliverable of the whole exercise, and it should
appear in the deck as a table, not a claim.

### Phase 5 — Fold into the evidence pack and deck ✅ **done 12 Sep 2026**

`evidence-pack.md` now reports run07 throughout: headline, per class, per height band, per
terrain, calibration, the ablation, ONNX, and a new **"Another city, another sensor, another
LiDAR vendor"** section carrying the §6 matrix. All five figures regenerated from
`out/eval_run07_ship`. The deck reports 3.464 m with the failed criterion stated beside it.

Limitation #1 was narrowed as planned, but on a different basis than §6 anticipated. The
criterion was *not* met, so the narrowing does not rest on having passed anything — it rests
on where the data stops: the under-call above 20 m is reported as 15.6 m and significantly
improved, and the residual is attributed to no open dataset at this resolution holding
buildings above ~50 m. A second limitation was added for the bias-for-variance trade
(overall −0.24 → −0.47 m, 6–10 m band −0.78 → −0.97 m, both p < 0.0001).

**Four things were corrected rather than copied, and each is worth knowing:**

1. **The §6b bootstrap used the wrong protocol** — see the correction above. The tall-building
   result is stronger than first written, and >30 m does reach significance.
2. **The error map's "93.6 m building we call 31.5 m" was unreproducible.** Measured from the
   raster and the scored array: truth median 72.8 m, ours 14.1 m. The old figure appears to
   have been a tile-wide maximum. `make_figures.py` now takes `--pred` so the figure cannot
   silently show a different checkpoint than `--metrics`.
3. **"Forested ground error is 2.5 m" had no provenance in any metrics file** — `evaluate.py`
   built the per-terrain block without passing `cls=`, so every `per_terrain.per_class` was
   empty. Fixed; measured 2.47 m against sparse's 0.61 m.
4. **The Sikkim co-registration is not real.** Its phase-correlation peak is 0.006 and the
   shift it chooses moves from (−4,−4) to (−8,0) to (−26,−30) m depending only on which
   raster it is given and how far it may search. The India table is now `--no-shift` for both
   checkpoints on the same footprints, which is the first time those two columns have been
   comparable. GAMUS did not move the Indian result (−6.41 → −6.28 m on confident
   footprints), which is expected — GAMUS is dense US urban and adds nothing hilly.

**Deployment.** `deploy/stage.py --ckpt checkpoints/run07/best.pt` and a restage of
App Service. Note the App Service package is now `deploy/dwz-appservice.zip`, built by a
separate script from `deploy/dwz-app.zip`, which is a Docker build context and must never be
zip-deployed — doing so took the site down for two days on 8 Sep.

**The six baked viewer scenes were re-baked on run07** the same day, at the shipping
configuration. `tools/build_scenes.py` now derives its raster paths from the same `RUN`
constant that sets the model label, so the label and the surface cannot drift apart, and it
chains `bake_buildings.py` itself — the re-bake exposed that `export_terrain.py` rewrites
`manifest["files"]` from scratch and had silently dropped the inundation tool's data.

---

## 8. What this will NOT fix, stated now

So that no one claims it later:

- **Buildings above 50 m.** GAMUS contains none. Our val reaches 155.3 m. The extreme tail
  survives this work untouched.
- **The India / ISRO sensor gap.** GAMUS is US aerial imagery. The cross-domain matrix is a
  *proxy* for ISRO transfer, not a measurement of it.
- **Hilly terrain.** No hills in DC, NYC or PHL. Sikkim stays the only hilly evidence, and
  it has no LiDAR, so it stays labelled agreement rather than accuracy.
- **Forested ground error.** 2.50 m, and it is optics — a camera cannot see through canopy.
  More data does not change physics.

---

## 9. Runbook — what is built and how to run it

Everything below exists and compiles. **Nothing long-running has been started.**

| tool | state |
|---|---|
| `tools/prepare_gamus.py` | written; `probe` run and passing, 200 tiles ingested |
| `tools/link_union.py` | written, not yet run |
| `tools/evaluate.py` | `--corpus gamus` added; DFC2019 path unchanged and bit-identical |

```bash
# 0. free ~6 GB first -- see the page-file note in section 5

# 1. verify the corpus (already done; re-run after any dataset revision)
python tools/prepare_gamus.py probe --sample 6

# 2. ingest training tiles. Resumable: the manifest already holds 200 tiles,
#    and re-running picks up exactly where it stopped. ~5 h at 16 workers.
python tools/prepare_gamus.py shard --workers 16

# 3. download the held-out DC block as whole GeoTIFFs (~317 tiles, 3.6 GB)
python tools/prepare_gamus.py holdout

# 4. build the union directory -- hardlinks, zero extra bytes
python tools/link_union.py --check      # dry run first
python tools/link_union.py

# 5. the pilot: short fine-tune from run02 to test Guard 1 before the full run.
#    --init-from, NOT --resume, and a FRESH --out. See the warning below.
python train.py --shards D:/sih2026/data/shards_union \
    --init-from D:/sih2026/checkpoints/run02/best.pt \
    --out D:/sih2026/checkpoints/pilot_union \
    --height-scale 13.223477220535276 \
    --epochs 1 --max-steps 1500 --batch 8 --workers 2 --max-temp 83

# 6. score, on the frozen DFC2019 val set and then the GAMUS holdout.
#    Separate --out per run: both default to out/eval and would overwrite each other.
python tools/evaluate.py --ckpt <ckpt> --split val --out out/eval_union_val
python tools/evaluate.py --ckpt <ckpt> --split gdc --corpus gamus --out out/eval_union_gdc
```

**Do not run step 2 and a training run at the same time.** The machine has 15.9 GB of RAM
with roughly 1 GB free; the ingest is what pushed the page file up once already.

## 9b. The warm-start trap — found by adversarial audit, 6 Sep 2026

The runbook above originally said `--max-steps 400 ...` with no warm-start flag, and
Phase 3 says "start from run02's shipping checkpoint". The only mechanism that existed
was `--resume`, and **it silently trains nothing**:

- `train.py` restores `start_epoch = st["epoch"] + 1`. run02's `best.pt` is epoch 11, so
  `start_epoch = 12`, and `for epoch in range(12, args.epochs)` is **empty** for any
  `--epochs <= 12` — including the default of 3 and the 12 that matches run02.
- `--max-steps` does not rescue it: the stop check sits *inside* the epoch loop. So does
  the checkpoint write, so no checkpoint is produced.
- The run then prints `done in 0.0 min. best val RMSE 7.980 m` — **run02's restored
  score, presented as the union pilot's result** — and exits 0.
- Raising `--epochs` above 12 does not rescue it either. `sched.load_state_dict` restores
  run02's scheduler at step ~11532 against a pilot `total_steps` of 400, so
  `lr_lambda` returns `max(0.0, (400 - 11533)/380) = 0.0`: 400 steps at learning rate
  exactly zero, saving weights bit-identical to run02.

Both traps produce a plausible number with exit status 0. Fixed three ways:

1. **`--init-from CKPT`** loads model weights only and leaves optimiser, scheduler,
   epoch, step and best at their initial values. This is now the warm-start flag.
2. **`--resume` aborts** when `start_epoch >= args.epochs` instead of falling through to
   an empty loop.
3. **A height_scale guard** on `--init-from`: if the checkpoint's `height_scale` differs
   from the one resolved for this run, abort and name the multiplier. Loading weights
   calibrated for one scale into a model built with another is the run05 deadlock.

Verified 6 Sep 2026: the old invocation now exits 1 with a message naming both remedies;
`--init-from` trains real steps at a non-zero learning rate and writes a checkpoint; a
deliberate `--height-scale 20.0` aborts and reports the 1.512x jump.

## 10. Open questions

1. **Is a 6% GSD difference material?** Cheap to test with the existing `gsd_probe.py`.
2. **Does the union need a sampler weight**, or is naive concatenation enough? Decide from
   the Phase 1 pilot, not in advance.
3. **Do we shard GAMUS test at all?** 2,861 tiles is a third of the corpus and ~11 GB. If
   NYC is the held-out domain, much of it may be unnecessary.
