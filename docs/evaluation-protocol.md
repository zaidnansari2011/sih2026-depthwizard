# Evaluation protocol - which number means what

Written 27 Aug 2026, after two numbers from the same checkpoint appeared to
contradict each other (global RMSE 8.22 vs 3.42, building bias -4.67 vs +2.36).

**There was no bug.** They were different splits. `train.py`'s in-training
validation scores the **val** regions; `tools/evaluate.py` scored the **test**
regions. Region-disjoint splits of different difficulty are not comparable, and
nothing in the tooling made that visible at the call site.

## Three protocols, all still in use

| # | Protocol | Data | Reported by |
|---|---|---|---|
| 1 | in-training val | 518x518 crops from `val_*` shards, valid pixels subsampled | `train.py` -> `train_log.jsonl` |
| 2 | whole-tile val | whole 1024x1024 val tiles, sliding window 518 / overlap 140 | `tools/evaluate.py --split val` |
| 3 | whole-tile test | same, on the held-out test regions | `tools/evaluate.py --split test` |

Protocols 1 and 2 differ by about **1.5 m of RMSE on the same checkpoint and the
same split** (run02 ep11: 7.980 crop-wise vs 6.454 whole-tile). Overlap averaging
in the sliding window plus a different pixel population account for it. Neither is
wrong; they are just not the same measurement. Quote which one you mean.

## Headline result (protocol 2, 80 identical val tiles, seed 1337)

All three rows scored on a bit-identical tile list - verified, not assumed.

| Method | RMSE | MAE | bias | bldg RMSE | bldg bias | grnd RMSE |
|---|---|---|---|---|---|---|
| Zero-shot DA-V2, raw | 9.998 | 3.856 | -2.047 | 25.38 | -13.72 | 1.85 |
| Zero-shot + global affine *(deployable baseline)* | 9.308 | 4.528 | +0.230 | 22.47 | -9.64 | 4.17 |
| Zero-shot + oracle affine *(not deployable)* | 7.285 | 3.147 | -0.000 | 17.33 | -2.64 | 2.99 |
| DepthWizard run01 (ep1) | 6.811 | 2.224 | -0.752 | 16.72 | -3.23 | 2.39 |
| **DepthWizard run02 (ep11)** | **6.454** | **1.840** | **-0.569** | **16.37** | -4.70 | **1.92** |

**Fine-tuning beats the deployable zero-shot baseline by 30.7%** (9.308 -> 6.454),
and beats the *oracle* affine variant, which is allowed to fit scale and shift from
each tile's own ground truth. run02 also carries the better uncertainty head:
ECE 0.083 and sigma/error rank correlation +0.836, against run01's 0.163 / +0.562.

Buildings remain the whole problem: 16.37 m RMSE and -4.70 m bias, still
underestimating, consistent with [literature.md](literature.md).

## Protocol 4 - India, against Google Open Buildings 2.5D

The first quantitative number DepthWizard has produced on Indian ground. Everything else
we report is Jacksonville and Omaha.

**Setup.** Maxar Open Data supplies 0.31 m RGB over the Teesta valley in Sikkim (event
`India-Floods-Oct-2023`, but the acquisitions in it are 2022-03-14, so the matching Open
Buildings year is 2022, not 2023). Google Open Buildings 2.5D Temporal supplies
`building_height`, defined as height above terrain in [0, 100] m -- the same quantity we
predict -- on the same UTM grid. `tools/open_buildings.py` pulls it as a windowed COG read;
`tools/compare_open_buildings.py` scores it.

**This is model versus model.** Open Buildings heights are inferred from Sentinel-2 at
10 m. Agreement across many buildings is evidence our absolute scale transfers to India.
A single-building disagreement is evidence of nothing, and we should never claim otherwise
in front of a jury that knows the dataset.

**Method choices that change the answer, so all of them get reported.** We compare per
building, not per pixel, because their footprint edges are Sentinel-2 blurry and a pixel
score would mostly measure their blur. Confidence intervals resample buildings, never
pixels. A single global shift is recovered by FFT phase correlation first, because at
26 degrees off-nadir a roof sits about 5 m from its own footprint while theirs come from
near-nadir Sentinel-2.

### First result: run02, 500 m town crop, 27.8% building coverage

| | value | 95% CI |
|---|---|---|
| buildings scored | 89 of 108 components | |
| bias (ours - theirs) | **-1.92 m** | [-2.50, -1.38] |
| MAE | 2.37 m | [1.94, 2.89] |
| RMSE | 3.29 m | [2.51, 4.20] |
| r | +0.530 | [+0.358, +0.696] |

**We under-call building heights on Indian data by about 2 m.** That is the same weakness
DFC2019 already shows, appearing independently on another continent against another
model -- which is worth more as corroboration than it costs us as a result.

**The diagnostic that matters is the percentile sweep.** Our median inside a footprint is
-1.92 m low, but our p90 inside the same footprint is only -0.28 m off. The model does
reach the right roof height somewhere in each building; what it fails to do is hold that
height flat across the roof. That is the soft-argmax over-smoothing signature, and it is
exactly what the binned head with the head-tail cut (run03, run04) was built to fix. So
this crop is the **before** measurement, and run04 gets scored against it unchanged.

### Two things not to repeat

**Erosion was my idea and it was wrong.** I expected shrinking their footprint to strip a
blur halo and move the bias toward zero. Measured: -1.92 m at 0 m erosion, -2.72 at 2 m,
-3.14 at 3 m -- the wrong direction, because erosion drops small components first and so
selects for the taller buildings we under-call most. Default is 0, and the sweep prints.

**The co-registration peak is weak on real data.** The synthetic self-test peaks at 0.40;
this crop peaks at **0.041**. The recovered shift still helps (r +0.436 without, +0.537
with) and its magnitude, 7.2 m, is the right order for a 26-degree lean on a 9 m building.
But a peak that weak is not a confident registration, and the honest reading is that some
of the residual disagreement is co-registration we have not removed. Report the with-shift
and no-shift numbers together, never the better one alone.

## Test-split discipline

The test split had already been scored twice before this note: once by
`zero_shot_baseline.py` (which used to hardcode `tiles_for("test", ...)`) and once
by `evaluate.py` on run01 ep1, saved as `out/eval_run01_e1/`. Both predate the
`--split` flag. Treat those two artifacts as spent, and do not use them to choose
anything.

Both tools now take `--split`, **defaulting to val**. Scoring test prints a warning.
Score it for a final reported number, not to pick between checkpoints.

Note the test split is materially **easier** than val - the same zero-shot global
affine baseline reads 4.68 m on test against 9.308 m on val. When the test number is
finally reported it will look better than val, and the write-up must say which split
each figure comes from rather than quietly quoting the flattering one.
