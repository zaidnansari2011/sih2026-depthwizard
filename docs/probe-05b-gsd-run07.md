# Probe 05b — does auto-zoom still hold at 0.6 m on the shipped model?

**Why now.** ISRO's FAQ (22 Sep, `docs/isro-faq.md`) settles the evaluation imagery:
*"Cartosat 2S imageries with resolution of 0.6m will be used for final evaluation."* The only
0.6 m measurement we have is probe 05's, and it is weak evidence for the model we ship:

- it is **run02**, not run07;
- it is **three hand-picked non-urban tiles**, averaged per tile rather than pooled;
- it was a manual `infer.py --zoom` sweep, **single pass**, not the shipped TTA + fusion.

So "0.6 m costs 3.6% after auto-zoom" is not yet a claim we can make about run07.

## Pre-registration — written 25 Sep 2026, before any run

**Method.** `tools/gsd_recheck.py` scores run07 on the **same 80 validation tiles**
`tools/evaluate.py` uses (seed 1337, region-disjoint val split), in-process, with the
**shipped configuration** (8-way TTA + zoom-2 Laplacian fusion, σ = 8). There are three
conditions per tile:

| | input | zoom (what `--auto-zoom` would pick) |
|---|---|---|
| **A** native | the 0.3 m tile as delivered | 1 |
| **B** untreated | Lanczos-downsampled to 0.6 m (512 px) | 1 |
| **C** auto-zoom | the same 0.6 m image | 2, with the fusion detail pass at 4 |

B and C predictions are bilinearly resampled back onto the 1024 px truth grid. Truth never
moves. Buildings are pooled across all 80 tiles, as in the headline metric.

**Sanity gate (must pass or the run is void).** Condition A reproduces the shipped
per-building RMSE of **3.464 m within ±0.01 m**. It is the same code path, so any larger gap
means the script is wrong, not the model.

**Primary prediction.** With auto-zoom, 0.6 m input costs **no more than 10 %** of native
pooled per-building RMSE: C ≤ 1.10 × A (≤ 3.81 m if A = 3.464). **Falsified if C > 1.10 × A.**

**Secondary prediction.** Auto-zoom helps: C < B on pooled per-building RMSE.
**Falsified if C ≥ B.**

**Also reported, not pre-judged:**
- per-building RMSE on buildings ≤ 20 m, where the tall-building tail cannot dominate;
- per-building bias;
- whole-tile RMSE;
- per-terrain whole-tile RMSE.

**What the deck will say, decided now.**
- If the primary holds: *"At ISRO's 0.6 m, measured on 80 tiles: +X % per-building error
  after auto-zoom"*, with X as measured.
- If it fails: the deck states the measured cost and the "within 3.6 %" line is removed
  everywhere it appears.

**Limitation stated in advance.** 0.6 m is *simulated* by downsampling 0.3 m WorldView-3.
Cartosat-2S differs in sensor, spectral response and look angle, so this measures the
resolution gap only, not the sensor gap.

## Result

*(filled in after the run; the section above is not edited)*

Run 25 Sep 2026, 7.5 min on the RTX 3060. The thermal guard paused four times at 84–86 °C.
Raw numbers are in `out/gsd_recheck_run07/results.json`.

| condition | per-building | bias | r | buildings ≤ 20 m | whole-tile |
|---|---|---|---|---|---|
| **A** native 0.3 m | **3.464 m** | −0.47 | +0.809 | 1.667 m | 6.008 m |
| **B** 0.6 m untreated | 4.185 m (+20.8 %) | −1.66 | +0.787 | 2.090 m | 6.738 m |
| **C** 0.6 m auto-zoom | 3.999 m (+15.4 %) | −1.35 | +0.761 | 2.095 m | 6.494 m |

Per-terrain whole-tile RMSE, A / B / C:

| terrain | A | B | C |
|---|---|---|---|
| sparse | 1.074 | 1.279 | 1.366 |
| mixed | 2.600 | 3.037 | 2.981 |
| forested | 3.286 | 3.961 | 3.784 |
| urban | 13.084 | 14.590 | 14.012 |

### Against the pre-registration

| | criterion | result |
|---|---|---|
| Sanity gate | A reproduces 3.464 m ± 0.01 | **PASS** (3.464) |
| Primary | C ≤ 1.10 × A (≤ 3.811 m) | **FALSIFIED** (3.999 m, +15.4 %) |
| Secondary | C < B | **holds** (3.999 < 4.185) |

### What it means

**"0.6 m costs 3.6 % after auto-zoom" does not hold for the shipped model.** On ISRO's
stated evaluation resolution, run07 loses 15.4 % of its per-building accuracy even with
auto-zoom, and 20.8 % without it. Probe 05's 3.6 % came from run02, single pass, three
hand-picked non-urban tiles; it should not be quoted again.

**Auto-zoom's gain is concentrated in the tall buildings.** On the 3,030 buildings at or
below 20 m, C and B are the same (2.095 vs 2.090 m); both cost about 25 % against native.
On sparse tiles auto-zoom is slightly *worse* than doing nothing (1.366 vs 1.279 m), and
correlation drops (+0.761 vs +0.787). So the pooled improvement from B to C is the tall
tail, which auto-zoom lets the model see at a larger scale.

This is recorded, not acted on. Candidate explanations, **none tested**:

- the fusion detail pass at zoom 4 over-sharpens a 0.6 m input;
- Lanczos downsampling is a poor stand-in for a real 0.6 m sensor;
- the model was never trained on upsampled imagery.

Any follow-up needs its own pre-registration.

**What changes now:** every place that says "within 3.6 %" is corrected to the measured
+15.4 %, and the deck reports this result as measured.
