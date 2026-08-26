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
