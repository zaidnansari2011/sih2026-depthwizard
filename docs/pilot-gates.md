# Union pilot — pre-registered gates for the run07 decision

**Written 7 Sep 2026, 01:05, while the GAMUS ingest was still running (2,055 / 6,204
tiles) and before any union data existed.** Nothing below was chosen after seeing a
number.

## Provenance — read this before trusting the list

An equivalent gate set was agreed with Zaid in conversation on 6 Sep and was **never
written to disk**. That conversation was compacted away, so the list below is
**re-derived, not recovered verbatim**. It is pre-registered with respect to the pilot —
which had not run when this file was written — but it is *not* the original wording, and
it should not be described as such in the deck or the evidence pack.

The one place this matters: if a gate here is looser than the one agreed on 6 Sep, a run07
launched on it was authorised by a weaker bar than Zaid actually set. Every threshold below
is therefore justified in-line, so the reasoning can be checked rather than taken on trust.

## What the pilot is for

The pilot is ~1,500 optimiser steps warm-started from `run02/best.pt` on the union corpus.
It is **not** a converged model and must not be scored against the §6 shipping criteria in
`gamus-integration.md`. Its only job is to answer one question:

> Is there any reason *not* to spend a long GPU run on this corpus?

So these gates are catastrophe detectors, not quality bars. A pilot can pass every gate
here and still produce a union model that fails §6 — that is the expected order of events,
not a contradiction.

## Reference numbers (run02, the shipping checkpoint)

| Quantity | Value | Protocol |
|---|---|---|
| Val RMSE (crop-wise, training-time) | 7.980 m | 518 px crops, no TTA |
| Val RMSE (whole-tile, shipped) | 6.401 m | sliding window + TTA ×8 + zoom-2 fusion |
| Per-building RMSE | 3.667 m | whole-tile, TTA, fused |
| Per-building RMSE, 3–6 m band | 1.45 m | same |
| Per-building bias, >20 m band | −17.26 m | same |

**The pilot is evaluated without TTA and without zoom fusion** — TTA costs 8× inference for
a number that only has to detect catastrophe. Every pilot gate below is therefore stated
against a non-TTA reference, and the two must never be compared across protocols.

---

## The seven gates

All seven must hold. Any failure stops the chain and leaves the GPU idle.

### G1 — the pilot actually trained
`checkpoints/pilot_union/best.pt` exists with an mtime after the pilot started, the
training log shows **> 0 optimiser steps**, and the checkpoint records
`height_scale == 13.223477220535276`.

*Why:* this is the exact defect the 6 Sep adversarial audit confirmed — a `--resume`
warm start inherits run02's epoch counter, trains zero batches, and prints run02's own
`best val RMSE 7.980` as if it were the pilot's result, exit code 0. The `--init-from`
fix is in place and was verified by execution, but the gate stays: a silent no-op that
reports a plausible number is the most dangerous failure available to this project.

### G2 — validation is untouched DFC2019
The val shard list printed at runtime contains only `val_*.npz` and **zero** `train_g_*`
entries.

*Why:* every number in the deck rests on the val set never having seen GAMUS. This was
verified statically (the `val_*` glob cannot match `train_g_*`) and by the audit, but a
static argument about a glob is not evidence about the files a specific run actually
opened. Confirm it at runtime, once, from the run's own output.

### G3 — the loss is finite and descending
No NaN or Inf in the loss trace, and mean training loss over the final quartile of steps
is below the mean over the first quartile.

*Why:* catches an exploded learning rate, a broken NLL, or a corpus whose targets are on
the wrong scale. Cheap, and it fails loudly rather than producing a mediocre number that
invites debate.

### G4 — short buildings have not collapsed
Per-building RMSE in the **3–6 m band ≤ 1.95 m** (run02: 1.45 m).

*Why:* this is Guard 1 from §6, deliberately loosened from +0.3 m to +0.5 m. The union is
**76 % GAMUS** by crop count and GAMUS is the taller corpus (median 6.76 m vs DFC2019
val's 4.22 m), so some drift toward tall is expected and is not by itself a reason to
stop. 1,500 steps from a warm start is also the worst possible moment to judge a band that
run02 spent 12 epochs learning. What +0.5 m rules out is *collapse* — the model abandoning
the short stock that carries most of the 3.667 m headline. If G4 fails, the fix is the
50/50 sampler, not more steps.

### G5 — no whole-tile blow-up
Val whole-tile RMSE **≤ 9.0 m**, scored **without TTA** against run02's 7.980 m crop-wise
figure.

*Why:* a ~13 % allowance over run02 for a mid-flight checkpoint. Note the protocol
mismatch is real and deliberate: 9.0 m is compared to 7.980 m, **not** to the shipped
6.401 m, which is a TTA+fused number. Quoting the pilot against 6.401 m would manufacture
a regression that does not exist.

### G6 — the tall-building bias moved the right way
Per-building bias in the **>20 m band strictly better (less negative) than −17.26 m**.

*Why:* this is the whole premise of the ingest. 36× the tall-building instances should
show *some* movement within 1,500 steps even if nowhere near the −13 m shipping bar. If
the bias moves the **wrong** way, the premise behind probe 01 is wrong and a long run
would only spend GPU hours confirming it. This is the gate most likely to fail, and the
one it is most valuable to fail on.

### G7 — machine health
GPU peak ≤ **86 °C** across the pilot, **≥ 25 GB** free on D:, and no other CUDA process
holding memory.

*Why:* the 3060 self-throttles at 91 °C and the power limit needs admin rights we do not
have, so temperature is governed in software by `train.py --max-temp`. 86 °C leaves real
headroom below the throttle. The disk floor covers run07's checkpoints plus the holdout
GeoTIFFs. **GPU settings are never changed while a CUDA job is running** — if G7 fails,
the run stops and waits for Zaid.

---

## If all seven hold

run07 launches automatically under the conditional exception to the "never auto-queue a
training run" rule, with `--max-steps` sized from the pilot's measured throughput so it
**completes and is scored before Zaid is back at the PC** — a finished run with numbers
beats a longer run caught mid-epoch.

## If any gate fails

The chain stops, the GPU stays idle, and the failing gate with its measured value is the
first thing in the morning report. No retry, no threshold adjustment, no "it was close".

---

# Scorecard — measured 7 Sep 2026, 06:20

Pilot: 1,500/1,500 steps, 8.9 min train, warm-started from `run02/best.pt`.

**Baseline correction.** The gates above cite run02 at the wrong protocols. G5 names run02's
**7.980 m crop-wise training-time** figure, but `evaluate.py` reports **whole-tile**, so the
two are not comparable. The correct matched baseline already existed: `out/eval_run02`
(tta=False, split=val, 80 tiles, the same 3,090 buildings). Every comparison below uses it.
The G5 threshold of 9.0 m is unaffected and the pilot passes either way, but the reasoning
in G5 as originally written was wrong and is superseded here.

| | run02 (matched) | pilot | delta |
|---|---|---|---|
| whole-tile RMSE | 6.456 m | **6.237 m** | −0.219 m better |
| per-building RMSE | 3.771 m | **3.666 m** | −0.105 m better |
| 3–6 m band RMSE | 1.451 m | 1.464 m | +0.013 m |
| >20 m bias | −17.262 m | −16.812 m | +0.450 m better |
| ECE | 0.0835 | **0.0761** | better |
| sigma rank corr | 0.8356 | **0.8655** | better |

| Gate | Result | Evidence |
|---|---|---|
| G1 trained | **PASS** | `e0 s1500/1500`; epoch 0 not 11; height_scale exactly 13.223477220535276; weights differ from run02 (mean abs delta 3.4e-04, max 3.4e-02); best val 7.8025 vs 7.9797 |
| G2 val pure | **PASS** | 14 `val_*.npz` in `shards_union`, 0 matching `_g`. Union: 61 DFC + 192 GAMUS train shards = 75.9 % GAMUS |
| G3 loss sane | **PASS** | 27.01 → 1.855 over 60 logged points; no NaN/Inf (the only "inf" hit was the word "Inference") |
| G4 short buildings | **PASS** | 1.464 m vs 1.95 m bar |
| G5 no blow-up | **PASS** | 6.237 m vs 9.0 m bar, and better than the 6.456 m matched baseline |
| G6 tall bias | **PASS, but see below** | −16.812 m, strictly better than −17.262 m |
| G7 machine | **PASS** | GPU peak 82 °C (train.py's own reading; the 30 s sampler saw 80 and missed the peak) vs 86 bar; 61 GB free; GPU idle after |

## G6 passed on a number the data cannot resolve

A paired bootstrap (20,000 resamples, same 3,090 buildings, seed 1337):

| band | n | run02 bias | pilot bias | delta | 95 % CI | p |
|---|---|---|---|---|---|---|
| >20 m | 60 | −17.262 | −16.812 | +0.450 | [−0.517, +1.393] | **0.357** |
| 10–20 m | 116 | −1.174 | −1.391 | −0.217 | [−0.791, +0.355] | 0.454 |
| 3–6 m | 2,305 | +0.213 | −0.016 | −0.230 | [−0.256, −0.203] | <0.0001 |
| all | 3,090 | −0.314 | −0.576 | −0.262 | [−0.299, −0.224] | <0.0001 |

**The >20 m improvement is not distinguishable from zero.** The interval covers no change
and change in the wrong direction. G6's threshold was written as a point comparison, which
was a mistake — a 0.45 m shift on 60 buildings was never going to be resolvable.

This is a property of the **instrument, not the model**: DFC2019 val contains only 60
buildings above 20 m, so no amount of training will ever produce a significant tall-building
result on it. The GAMUS DC holdout (317 tiles, columns 50–67, the tall city) is the correct
instrument, and `out/eval_run02_gamus_holdout` was therefore measured before run07 started
so the comparison cell exists.

What *is* significant is the 3–6 m band: bias moved +0.213 → −0.016, i.e. **closer to zero**,
and overall RMSE improved. The pilot made the model generally better while leaving the
specific tall-building question unanswered.

## Decision

All seven gates pass, so run07 launched at 06:25 under the conditional exception:
14,000 steps, `--epochs 4`, warm-started from run02 per Phase 3, `height_scale` pinned.
14,000 rather than the ~20,000 the pilot's 168.5 steps/min would allow, because the pilot
ran 9 minutes from a cold 44 °C GPU and a multi-hour run throttles against `--max-temp 83`.
A completed and fully evaluated run is worth more than extra steps.
