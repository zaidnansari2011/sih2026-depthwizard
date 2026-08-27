# Probe 01 — can tall-building supervision move the tail?

**Written before the probe was run.** The point of writing it first is that a hypothesis
you can only evaluate after seeing the result is not a hypothesis.

## Why

run02's per-building error is concentrated almost entirely in the tail:

| true height | buildings | RMSE | bias |
|---|---|---|---|
| 0-3 m | 166 | 1.47 m | +0.26 |
| 3-6 m | 2,305 | 1.45 m | +0.21 |
| 6-10 m | 443 | 1.90 m | -0.75 |
| 10-20 m | 116 | 4.61 m | -1.17 |
| **> 20 m** | **60** | **24.04 m** | **-17.26** |

79% of the total squared error comes from 1.9% of the buildings. run03 and run04 both
attacked this with architecture (binned head, head-tail cut) and both moved the >20 m bias
by under a metre. The training split holds 166 buildings above 30 m and tops out at 82.8 m;
val goes to 155.3 m. So the working theory is **data scarcity, not architecture**.

Before spending another 3.5-hour run on that theory, falsify it in 30 minutes.

## Design

Fine-tune run02 for a few hundred steps on crops centred on tall buildings drawn only
from **train** regions, then measure the >20 m bias in two places:

- **Fit set** — held-out tall buildings from those same train regions. Answers *can the
  model represent a tall building at all?*
- **Transfer set** — tall buildings in the val regions (OMA_288, JAX_166, JAX_167), which
  the probe never trains on. Answers *does tall supervision generalise?*

Separating those two is the whole point. "It cannot fit" and "it fits but does not
transfer" call for opposite next moves, and a single number cannot tell them apart.

## Predictions, and what each outcome means

Baseline to beat: **>20 m bias -17.26 m**.

| outcome | reading | next move |
|---|---|---|
| Fit improves to better than -8 m, transfer stays worse than -13 m | **Data scarcity.** The model can represent tall buildings; it has not seen enough of them to generalise. | Get tall-building data (Indian metros + Open Buildings). A re-split of DFC2019 buys little — there are only 341 tall buildings in the whole dataset. |
| Neither improves | **Structural.** Something in the loss, the height scale, or the sampling prevents tall output regardless of supervision. | Stop training. Debug the objective on a single overfit tile before anything else. |
| Transfer improves past -10 m | **Trainable.** Tall supervision generalises and we were simply under-weighting it. | run05 on a tall-enriched sampler is justified. |
| Fit improves, transfer improves a little (-13 to -10 m) | Partial. Real but weak. | Not worth 3.5 hours alone; fold into the Indian-data run. |

**Falsification condition for "run05 will fix the tail":** if transfer bias does not beat
-13 m (a 25% improvement), a full run on the same data will not fix the tail and should
not be started.

## Result — hypothesis falsified, 27 Aug 2026

400 steps of fine-tuning run02 on tall crops drawn only from train regions.

| | n | true median | bias before -> after | RMSE before -> after | r before -> after |
|---|---|---|---|---|---|
| **FIT** (held-out, train regions) | 170 | 26.2 m | -4.61 -> **-4.22 m** | 10.22 -> **8.81 m** | +0.409 -> **+0.556** |
| **TRANSFER** (val regions) | 898 | 28.0 m | -18.22 -> **-19.10 m** | 27.12 -> **27.78 m** | +0.356 -> +0.349 |

**The falsification condition was met.** The pre-registered threshold was that transfer bias
must beat -13 m; it went from -18.22 to -19.10 m. Not merely short of the bar -- backwards.

**The model can represent tall buildings.** Before any fine-tuning it is off by 4.6 m on
tall buildings in regions it trained on and 18.2 m on tall buildings in regions it did not,
at the same 26-28 m median height. Four times the error, same building heights. That rules
out a capacity limit, a saturating head or a broken height scale.

**And it learns from tall supervision -- just nothing that transfers.** Fitting improved
across the board (RMSE -14%, r +0.15) while transfer degraded. That is the textbook
signature of overfitting a small sample, and the sample is 166 unique buildings above 30 m.

### What this rules out

- **run05 as "train harder on tall buildings" on DFC2019.** Directly measured, not assumed.
- **Naive oversampling or a tall-enriched sampler.** This probe is the extreme case of both
  -- 100% tall crops -- and it made transfer worse.
- **A DFC2019 re-split.** Rearranging 341 tall buildings does not create information.

### What this does NOT rule out

Honesty about the limits of the probe: training exclusively on tall crops is a severe
distribution shift with no low buildings to anchor the model, which is **not** the same
thing as LDS/FDS (Yang et al., ICML 2021). FDS in particular transfers feature statistics
from data-rich to data-poor parts of the target range, so what the model learns at 15 m
informs what it does at 40 m. That mechanism is untouched by this experiment. It remains
the one cheap idea still standing, and it should be judged on its own probe rather than
dismissed by this one.

The expensive idea also stands, and is the only one that adds real information: tall-building
imagery from Asian metros with Open Buildings labels.

### Cost

30 minutes, against the 3.5 hours a run would have cost to learn the same thing. This is
the pattern to keep: predict, write it down, probe cheaply, then decide.
