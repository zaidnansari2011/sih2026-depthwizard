# DepthWizard — Plan of Record

**SIH26175** · ISRO / Space Applications Centre · Software
**Idea submission closes 30 September 2026** · cap 500 submissions per problem statement
(corrected 12 Sep from 20 September, which came from the team brief; the portal says 30th —
`docs/problem-statement.md`. Dated entries below that say "20 Sep" are left as written.)
Repo: `zaidnansari2011/sih2026-depthwizard` (private) · Workspace: `D:\sih2026`

> This is the living reference. If a decision changes, change it *here* first.

---

## 1. What we are building

An end-to-end pipeline that takes **one** optical image and produces a **Digital Surface Model**, then projects that image onto a navigable 3D terrain mesh.

Two input paths are mandatory:

| Input | Output | Calibration |
|---|---|---|
| Plain PNG / JPG, no spatial metadata | **Relative** DSM | none — scale-free |
| Georeferenced GeoTIFF | **Absolute** DSM, real metric heights | coarse DEM (SRTM 30 m) or ground control points |

Deliverable is a deployable suite: an elevation module emitting a DSM in a standard geospatial format, plus an interactive viewer where a user uploads imagery, sees the reconstructed terrain, and **validates estimated heights against reference data**.

## 2. Scoring — the thing that drives every decision

| Weight | Criterion | What it actually means |
|---|---|---|
| **50%** | DSM accuracy & validation | RMSE, MAE, correlation vs LiDAR/reference. **Stability required across urban, sparse, hilly, forested.** |
| **50%** | Rendering quality & UX | Projection accuracy, visual fidelity, navigability, interface intuitiveness, stability, successful standalone deployment |

**Half the marks are the front end.** A pure-CV team pours 36 hours into the model and demos a clunky viewer, forfeiting most of the second half. A pure-web team cannot produce a defensible DSM at all. We can contest both halves, and they parallelise from hour one.

Corollary: *never* let the viewer become the last night's work.

## 3. Why this problem statement

226 statements, 172 software, roughly one winner each. Teams lose at selection, before any code. The crowd flows to whatever looks easiest — attendance apps, chatbots, dashboards — where hundreds of near-identical builds are judged on polish alone.

"Estimate height from a single image" *sounds* like original research, so most teams self-select out. It isn't research; see §5. That gap between perceived and actual difficulty is the entire opportunity.

**Honest caveat:** thin competition is not weak competition. Teams that do pick this will be strong CV teams. We must genuinely execute — we just aren't fighting 300 lookalikes.

**Open check:** every PS currently reads 0/500 because the window just opened. Crowding is inferred from sponsor patterns, not measured. **Re-check live counts around 10 September** and be willing to switch to SIH26143 (NTRO, oil-spill detection with AIS attribution) if this one turns out swamped.

## 4. Compute & environment

| Resource | Role |
|---|---|
| RTX 3060 12 GB (local) | dev loop, debugging, **all preprocessing**, viewer, demo machine |
| Kaggle — 30 GPU-h/week, 12 h sessions | experiment farm: benchmark sweeps, long unattended runs |
| Azure $100 credit | **hosting the standalone demo**, not training |

- 3060 beats a Kaggle P100 (Pascal, no tensor cores) and roughly matches a single T4. Only 2×T4 with DDP is faster, and only ~1.6×. Free cloud buys **parallelism**, not speed.
- **T4/P100 have no bf16.** Training code autodetects and falls back to fp16 + GradScaler.
- Preprocess locally (CPU/disk-bound, burns zero quota) → upload shards **once** as a Kaggle Dataset → every session mounts `/kaggle/input` instantly.
- Code lives in git and is cloned into notebooks. **Never write code inside a notebook.**
- Checkpoint every epoch — 12 h session cuts are routine, not exceptional.
- Verify Azure GPU quota in week 1: student subscriptions default N-series quota to 0 and cap vCPUs. If blocked, nothing is lost — training was always local.

### Measured throughput and thermals, 26 Aug 2026

| Batch | ms/step | crops/s | Peak VRAM |
|---|---|---|---|
| 4 | 174 | 23.0 | 2.21 GB |
| **8** | **327** | **24.5** | **6.01 GB** |
| 12 | 469 | 25.6 | 7.57 GB |

Throughput plateaus around **25 crops/s**, so batch 8 — the Depth Any Canopy setting — is
also the efficient one, and 12 GB is not the binding constraint. **Heat is.**

**The card runs at its thermal limit long before it runs out of memory.** Unregulated, the
3060 reached **91 °C in under ten minutes** and down-clocked itself from 1875 to
**1492 MHz** — so the "full speed" run was not faster, just hotter. `nvidia-smi -pl` is the
clean fix but needs an administrator shell (`Insufficient Permissions` otherwise), so
`train.py` carries a software `ThermalGovernor`: sample every N steps, pause proportional
to overshoot, hard-stop above a ceiling until it cools.

#### Solved 27 Aug 2026 — and the software governor was always a workaround

Duty-cycling in software was the wrong tree. MSI Afterburner has a **"Prioritize" switch
between power ⚡ and temperature 🌡**, and it defaults to *power* — which makes the Temp
Limit **advisory**. The card draws to its power limit and lets temperature land wherever
it lands. That is why a Temp Limit of 77 °C still produced 91 °C readings, and why raising
the fan from 62% to 85% bought only ~3 °C.

Setting **Prioritize = temperature** makes the limit binding in hardware:

| | Before | After |
|---|---|---|
| Temperature | 91 °C | **mean 80.1 °C, peak 85 °C** |
| Clocks | throttled to 1492 MHz | **1777–1867 MHz** |
| Throughput | 53 steps/min (governed) | **143 steps/min** |
| 12-epoch run | 3.6 h | **~1.6 h** |

Cooler *and* 2.7× faster, because the card down-clocks smoothly instead of being stopped
and started. Working config: **fan 85% (auto off), Temp Limit 80, Power Limit 85%,
Prioritize = temperature, ⇄ link off** (that toggle ties the two limits together; turn it
off to set them independently). The software `ThermalGovernor` now fires **zero** times and
remains only as a backstop at `83 / 87`.

> ⚠️ **Never change GPU settings while a CUDA job is running.** Applying Afterburner
> settings resets the display driver and invalidates live CUDA contexts. Doing it mid-run
> killed a training process with `torch.AcceleratorError: CUDA error: unknown error` on a
> `.to(device)` call, and silently reverted the power limit to 170 W. Pause first.

`tools/vitals.py` samples GPU, RAM and disk on a timer, logs to a CSV that outlives the
session, and prints only on breach. It exists because both hardware failures in this
project — ten minutes at 91 °C, and an OOM kill with no traceback — were sitting in
numbers nobody was reading.

> **User-side action worth taking:** MSI Afterburner is already installed. A more
> aggressive fan curve would let training run *both* faster and cooler than software
> pausing can, and running one shell as administrator would allow a proper 120 W power
> limit. Either beats duty-cycling.

## 5. Method

Three stages. Only the middle is genuinely novel work.

### Stage 01 — Relative depth · *CV pair*

Depth Anything V2 (DINOv2 encoder + DPT decoder) producing a scale-free depth map.

> **This is NOT zero-shot.** The team brief's claim that "the weights already exist, we train nothing" is wrong, and it is the plan's one load-bearing wrong assumption. DA-V2 was trained on natural photographs with a ground plane and a vanishing point; a nadir satellite tile has neither.

#### Measured, 26 Aug 2026 — and the earlier wording here was too harsh

This section used to claim that run zero-shot, DA-V2 "correlates with rooftop albedo, not height." That was an assertion, so it got tested: `tools/zero_shot_baseline.py`, DA-V2-Small over **80 held-out test tiles drawn from 16 regions that appear in no training split**.

| | RMSE | MAE | r | bias | Deployable? |
|---|---|---|---|---|---|
| Raw output vs metres | 5.50 m | 3.10 m | 0.43 | −1.56 m | units are arbitrary — read *r*, not RMSE |
| **Global affine** — one scale+shift fitted on *train* tiles | **4.68 m** | 3.32 m | 0.43 | +0.70 m | **yes — deployable, and the bar on the TEST split** |
| Oracle affine — refitted per test tile from its own truth | 3.94 m | 2.58 m | 0.63 | 0.00 m | no — needs the answer to compute the answer |

> **These are TEST-split numbers, and the test split is materially easier than val.**
> The same deployable global-affine baseline reads **9.308 m on val**. Development
> decisions use the val figure; test is reserved for one final reported number.
> Both tools now default to `--split val`. See
> [docs/evaluation-protocol.md](docs/evaluation-protocol.md).

**Verdict: the pretrained features do see height at nadir.** r ≈ 0.43 is not albedo, it is signal. What is broken is *calibration*, not perception — and the failure has a specific, diagnosable shape:

| Class | RMSE | Bias | |
|---|---|---|---|
| ground | 3.62 m | **+2.53 m** | flat ground pushed *up* |
| building | 7.60 m | **−3.30 m** | rooftops pulled *down* |
| vegetation | 5.93 m | −3.57 m | |
| water | 1.80 m | +0.60 m | |

The model **compresses dynamic range**: it lifts the ground and flattens the structures. That is what a relative-depth prior does when it has no absolute reference, and it is what fine-tuning on metric labels exists to fix. Per terrain: mixed 4.31 m, sparse 5.74 m (bias **+5.16 m**), urban 6.17 m. Per-tile RMSE spans 2.11–8.07 m, so cross-terrain stability is a measured problem, not a hypothetical one (§7.5).

**The fix is smaller than feared, but the need for it is confirmed.** Fine-tuning must beat the baseline **on the same split** -- 9.31 m on val, not the 4.68 m test figure above. Landing above it would mean we did worse than a two-parameter linear correction, and §6.1 would have nothing to stand on.

*The earlier "overestimates tree height" concern did not reproduce: vegetation bias is −3.57 m, an under-estimate. Range compression dominates it.*

**Fix — the [Depth Any Canopy](https://github.com/DarthReca/depth-any-canopy) recipe**, which did exactly this adaptation for canopy height and beat prior SOTA:

| Setting | Value |
|---|---|
| Loss | MSE on min-max normalised height (we replace this — see §6.1) |
| Crops | 256×256 |
| Optimiser | AdamW, lr 5e-6 |
| Schedule | 5% warmup, linear decay, 3 epochs |
| Batch | 8 |
| Cost | 1.5–2.6 h on an A6000 → **~5–9 h on our 3060** |

Backbones to benchmark: DA-V2 **Small** (Apache-2.0), **Base** / **Large** (CC-BY-NC-4.0), **Depth Anything 3**, and Marigold / Depth Pro as alternates.

*Keep a Small result current* — it is the only clean-licence fallback if ISRO's spec demands redistributable code.

### Stage 02 — Metric calibration · *CV pair + data*

Fit the relative map to absolute metres by regressing against coarse SRTM 30 m elevation. This is the real problem, and it is regression, not research.

> **Metric anchor is SRTM 30 m or Copernicus GLO-30 — not CartoDEM.** CartoDEM 30 m is free only to Indian Government Entities; as students we are an NGE and it is a priced product. SRTM is genuinely free and is named in the problem statement anyway.

### Stage 03 — 3D flythrough · *Full-stack pair*

Displace a mesh by the heightmap, texture with the original image, fly a camera through it in the browser. Three.js (Unity and Babylon.js also accepted). **Worth 50% of the marks.**

### Fine-tuned results, 27 Aug 2026

Three runs. All val figures are whole-tile `--split val`, the same 80 tiles for every
row; see [docs/evaluation-protocol.md](docs/evaluation-protocol.md) for why crop-wise and
whole-tile numbers differ by ~1.5 m and must not be mixed.

| Run | Head | Objective | val RMSE | bldg RMSE | bldg bias | ECE | sigma rank rho |
|---|---|---|---|---|---|---|---|
| baseline | zero-shot + global affine | - | 9.308 | 22.47 | -9.64 | - | - |
| baseline | zero-shot + oracle affine *(not deployable)* | - | 7.285 | 17.33 | -2.64 | - | - |
| run01 | direct regression | NLL + grad | 6.811 | 16.72 | -3.23 | 0.163 | +0.562 |
| run02 | direct regression | NLL + grad, beta=0.5 | **6.454** | 16.37 | -4.70 | 0.083 | +0.836 |
| run03 | binned soft-argmax, N=128 | + Chamfer + distribution CE | 6.950 | 17.81 | -5.50 | 0.106 | +0.784 |

**The differentiator is real:** run02 beats the deployable baseline by 30.7%, and beats
the *oracle* affine, which is allowed to fit scale and shift from each tile's own truth.
The uncertainty head works - ECE 0.083 and sigma/error rank correlation +0.836.

**The beta-NLL ablation is closed, negative.** run02 ran 12 full epochs against run01's 5
and moved building RMSE 21.63 -> 21.17 crop-wise, with the bias still at -6.42 m.
Reweighting a regression loss does not fix a long-tailed output space, which is what the
literature said would happen (docs/literature.md section 4).

**The binned head still deploys.** Checked before trusting it: ONNX export of a binned
checkpoint agrees with PyTorch to 0.0014 cm on height and 0.0006 cm on sigma against a
5 cm tolerance, at 101.0 MB versus 100.7 for the regression head, 1125 ms per tile on
CPU. Soft-argmax, cumsum bin edges and the pooled width predictor all survive the
export. Still fixed at 518x518, which is what sliding-window inference uses anyway.
Section 6.4 is not at risk from the architecture change.

**The binned head is a negative result.** run03 came in 7.7% worse overall and 8.8% worse
on buildings, with worse calibration. The one metric that improved tells the story: ground
RMSE fell to 1.75 m, the best of any run, while buildings degraded. Smoothing helps flat
terrain and hurts tall discontinuous structure, which is the over-smoothing failure of
soft-argmax that docs/literature.md section 5 predicted -- and the distribution
supervision added to prevent it did not. **run02 remains the model of record at 6.454 m.**

Crop-wise validation disagreed in direction on this, calling run03 the winner. Anyone
reading only train_log.jsonl would have shipped the worse model.

**Buildings remain the entire problem:** 12.8% of pixels carrying 89-92% of squared error,
and two architectures have now failed to move it.

**View angle is not what limits us.** DFC2019 images every AOI from many satellite
positions and the Track 3 metadata carries the geometry, so the domain-gap question in
section 7.1 is measurable today without Bhoonidhi. Joining per-tile error to
`meanOffNadirViewAngle` over 80 val tiles, 4.2 to 29.2 degrees:

| Measure | Pearson r | slope | 95% CI | p |
|---|---|---|---|---|
| Naive across tiles *(terrain-confounded)* | +0.000 | +0.000 m/deg | - | 0.998 |
| **Within-region** *(terrain controlled)* | +0.074 | +0.009 m/deg | [-0.021, +0.038] | 0.555 |

Every AOI appears at several angles, so subtracting each region's own mean removes the
terrain confound entirely and leaves only "this same ground, viewed more obliquely". The
95% interval bounds the cost at 0.038 m per degree, i.e. **at most ~1 m of RMSE across the
whole 25-degree span** against a 6.45 m headline. Sun elevation tells the same story and
shows why the control matters: naive r +0.212 collapses to -0.033 within-region, so the
apparent effect was which AOIs happened to be imaged at low sun.

This bounds sensitivity to *view geometry only*. A different sensor, GSD and radiometry is
the Cartosat gap proper, and this says nothing about it. `tools/view_angle.py`,
`out/view_angle_run02/`.

## 6. Our four differentiators

The research for this task exists — IM2HEIGHT, TSE-Net, THE Benchmark, a decade of monocular height estimation. **ISRO knows.** What does not exist is a *product*: those are checkpoints and eval scripts producing a number on a benchmark, not something an analyst can open and use. The 50/50 split is ISRO saying so out loud.

So uniqueness is **not algorithmic**. Attempting novel architecture in four weeks yields a half-finished novel thing instead of a complete solid one plus a sharp differentiator.

### 6.1 Per-pixel uncertainty — *the strongest, do this first*

Single-view height is mathematically ill-posed: no unique 3D solution exists for a 2D image. Every other team ships one DSM and one RMSE, implying certainty they don't have. We ship a DSM **plus a calibrated confidence map**.

- **Implementation:** variance head + Gaussian NLL loss instead of MSE. ~20 lines.
- **Must be in the architecture from the first training run**, not bolted on in week 3.
- **Why it wins:** a jury of remote-sensing scientists works in error budgets. Nobody operationalises a product without knowing where it fails. This is their dialect.
- **Scores in both halves:** shade the flythrough by confidence and judges *see* the model's doubt over water, shadow, dense canopy.
- Also defuses the "don't oversell accuracy" risk in §7.4.

**The error bar is honest only inside a window, and we now know which one.** The viewer
quotes `dh +/- hypot(sigma_a, sigma_b)`, but every calibration number we had was per pixel,
while the thing a user reads is a difference. Measured on run02 over 20 val tiles, with
bootstrap intervals over tiles:

| separation | error corr | +/-1 sigma coverage | 95% CI | median abs z |
|---|---|---|---|---|
| *Gaussian expectation* | 0.000 | 68.3% | - | 0.674 |
| single pixel | - | 64.5% | - | 0.62 |
| 1-3 m | +0.944 | 88.4% | [86.8, 90.3] | 0.19 |
| **3-15 m** | +0.802 | **71.4%** | **[67.9, 74.8]** | 0.47 |
| 15-60 m | +0.312 | 56.5% | [51.6, 61.3] | 0.78 |
| 60-300 m | +0.051 | 51.7% | [46.1, 57.8] | 0.94 |

68.3% sits inside the interval at 3-15 m and outside it everywhere else. Below that the
two errors are nearly the same error (corr +0.94), they cancel in the difference, and the
quoted bar is far too wide. Above it independence arrives but coverage keeps falling, so
the bar is too narrow -- the dangerous direction. Tails are heavy throughout: +/-3 sigma
covers 84.8% on single pixels, not 99.7%, so a 3-sigma bound from this model must never be
presented as one.

Two consequences. Measuring a single roof against adjacent ground is the *most* reliable
use of the tool and the bar there is conservative. And the bar wants an empirical
separation-dependent scale factor, roughly 0.28x at 1 m rising to 1.39x at 100 m, refit per
checkpoint on val. `tools/pair_calibration.py`, `out/pair_calibration_run02/`.

### 6.2 Shadow as a physical prior — *the novel-ish one*

`height = shadow_length × tan(sun_elevation)`. This is how photointerpreters measured building heights before computers. Sun azimuth and elevation ship in satellite metadata.

The clever version is not an input channel — it is a **self-supervised signal on unlabelled Cartosat data.** We cannot fine-tune on Indian imagery because we have no height labels there; shadow geometry yields free physics-derived pseudo-labels on exactly the imagery we otherwise cannot supervise. **This turns our single biggest risk (§7.1) into our best story.**

- **Gate: PASSED (26 Aug 2026).** `Track 3 / Metadata` holds 67 WorldView-3 `.IMD` files with full solar and viewing geometry: `meanSunAz`, `meanSunEl`, `meanSatAz`, `meanSatEl`, `meanOffNadirViewAngle`. Parse with `tools/parse_imd.py`.

**Sun elevation spans 23.4°–74.5°**, and that spread is the substance, not the mere presence of the field. A 20 m building casts 46 m of shadow at 23.4° and 5.6 m at 74.5°. So shadow reliability is scene-dependent and *we know the scene's geometry*, which means pseudo-label confidence can be weighted by solar elevation instead of trusted uniformly — feeding straight into the uncertainty head (§6.1).

That gives a real method rather than a trick:

1. Estimate heights from shadows on DFC2019, where ground-truth AGL exists.
2. Measure how shadow-derived error scales with sun elevation → a calibrated confidence model.
3. Apply to unlabelled Cartosat with that confidence attached.

**Bonus:** `meanOffNadirViewAngle` spans 4.8°–28.9°, so we can measure accuracy degradation vs look angle — an empirical handle on the domain gap (§7.1) that predicts Cartosat behaviour before we touch Cartosat.

**Open follow-up:** this metadata is Track *3*. Track 1 tiles come from the same WV-3 collection over JAX/OMA, but we must confirm each Track 1 tile traces back to a source scene (filename or TIFF tags). If it doesn't, the prior still works on Cartosat (Bhoonidhi ships sun angles) — we just lose DFC ground-truth validation of it.

### 6.3 A measurement instrument, not a flythrough

The PS explicitly asks that a user *"validates estimated heights against reference data."* Most teams will skim that on the way to making something pretty.

- Click two points → height delta **with an error bar** (from 6.1)
- Drag a line → terrain cross-section profile
- Load a reference DSM → instant difference map + per-terrain-class error breakdown
- Report against the four terrain classes ISRO named: urban, sparse, hilly, forested

A flythrough is a screensaver. This is an analyst's tool.

### 6.4 Deployability as a scored feature — *this is the "optimisation" answer*

ONNX export, quantised, **CPU-only inference**, air-gapped Docker image. "Runs in N seconds per tile on a laptop, no GPU required" is a deployment story, and government facilities are frequently air-gapped — a demo needing internet is a liability. "Successful standalone deployment" is written into their criteria.

## 7. Risks

### 7.1 Domain gap — the one that matters

DFC2019 is aerial. ISRO evaluates on Cartosat-class imagery with different GSD and look angles. A model tuned only on DFC degrades on their held-out data.

**Mitigation:** pull Cartosat samples from Bhoonidhi in week 1 and validate continuously. Hold out our own test set from day one. Pursue §6.2 as the structural fix.

### 7.2 The reference repo is empty

`IMG-PROCESS-SAC/SIH-DepthWizard-2026` holds only a README ("This repository contains dataset regarding the problem-statement of DepthWizard"). It gained that README after 22 Aug, so data is coming — timing unknown. **It has 5 stars: other teams are watching.**

**Mitigation:** our public data path is fully independent. Keep the output layer modular so a changed file format is a config edit, not a rewrite. Everyone watches the repo.

### 7.3 They are format-strict

ISRO's sibling hackathon repo mandated exact output directory structure, exact file naming, explicit band ordering, plus code, model weights and a technical report.

**Assume the same. Teams with better models lose to teams that follow the spec.**

### 7.4 Do not oversell accuracy

The jury are SAC image-processing specialists. *"Relative DSM is solid, absolute scaling is ±8% against SRTM, here is the error map"* reads as competent. A centimetre-accuracy claim gets dismantled in questions.

### 7.5 Build what you can explain

At the finale you answer live questions — *"why is your error higher in forested terrain?"* Nothing helps but having built it and understood it. **Any component nobody on the team can explain from memory is a liability**, however well it performs.

### 7.6 What actually beats us

Not a team with a better model. A team with a **mediocre model and a beautiful viewer**, because they collected the half we neglected. Or a team that followed the output spec while we didn't.

## 8. Data

### Training / benchmark — DFC2019 Track 1

[IEEE DataPort](https://ieee-dataport.org/open-access/data-fusion-contest-2019-dfc2019), free account. Single-view RGB paired with ground-truth AGL height — *literally this task*, with published baselines and metrics, so we report against an established benchmark instead of hand-waving.

| File | Size | Contents |
|---|---|---|
| `Track 1 / Training data / RGB images 1/1` | 6.59 GB | `*_RGB.tif` inputs |
| `Track 1 / Training data / Reference` | 8.17 GB | `*_AGL.tif` heights (m), `*_CLS.tif` classes |
| `Track 1 / Validation data` | 683 MB | imagery only, truth withheld |
| `Track 3 / Metadata` | 138 KB | **gate for §6.2** — check for sun angles |

Skip all MSI (30 GB) and all of Tracks 2/3/4. Cities: JAX (Jacksonville), OMA (Omaha).

Metrics + baselines: [pubgeo/dfc2019](https://github.com/pubgeo/dfc2019), MIT.
SOTA reference line: [TSE-Net](https://github.com/zhu-xlab/tse-net) reports on DFC2019.

#### Measured facts — downloaded, extracted and probed 26 Aug 2026

The DFC2019 documentation specifies none of the following, so `tools/prepare_data.py probe` measures them and every later stage reads the report rather than a hardcoded guess.

| Property | Measured value | Why it matters |
|---|---|---|
| Tiles | **2 783** RGB+AGL+CLS triples, **108 regions** | 108 regions is enough for a clean geographic split |
| Tile size | **1024 × 1024** | decides the crop geometry, below |
| dtypes | RGB `uint8`, AGL `float32` | |
| **Georeferencing** | **none — no CRS, no transform** | ⚠️ see below |
| GSD | **not in the files**; US3D/DFC2019 Track 1 is **0.3 m/px** | must be passed as `--gsd 0.3` |
| Height range | −2.98 m to 204.15 m | |
| Height distribution | p50 **0.0**, p75 4.6, p95 **13.2**, p99 21.1 | p95 sets the model's output scale |
| Void | **NaN only, and negligible** — 6/200 tiles contain any, then <0.005% of pixels | void handling is nearly a non-issue |
| Split | **76 train / 16 val / 16 test regions**, seed 1337 | region-level, so no tile neighbourhood spans two splits |

**⚠️ The tiles carry no georeferencing at all.** They were rewritten by `tifffile.py`, which stripped CRS and affine transform — `rasterio` returns the identity matrix and warns. Nothing in the files says a pixel is 30 cm. Every distance the viewer reports is therefore derived from an explicitly supplied `--gsd`, and `export_terrain.py` refuses to invent one: with no GSD it labels the axes *pixels* rather than printing confident nonsense. Sanity check: 1024 px × 0.3 m = 307 m per tile, which is the right order for a city block.

**Class distribution** (60 random tiles, 62.9 M px): ground 66.3%, building 15.8%, vegetation 13.2%, water 2.4%, bridge 1.2%, and **65 = "unlabeled" 1.1%** — an undocumented code. Its heights are finite and valid, so those pixels stay in the regression but are excluded from per-class breakdowns.

Median height by class: vegetation **7.97 m**, building **6.83 m**. *Vegetation is taller than buildings in this dataset* — Jacksonville tree canopy — which is worth remembering before reading too much into any single aggregate number.

#### Crop geometry — 518, not 256

The Depth Any Canopy recipe uses 256×256 crops. **We cannot.** DA-V2's patch size is 14 and its head computes `patch_h = H // 14`, then interpolates its output to `patch_h × 14`. 256 is not a multiple of 14, so a 256×256 input silently returns **252×252** — a size mismatch that reaches the loss as a confusing broadcast error, or worse, quietly works after an accidental resize.

So: **crop 518** (= 14 × 37, and DA-V2's native training resolution), **stride 506**. Since 1024 = 506 + 518, that tiles each tile exactly 2×2 with 12 px of overlap and full coverage. The ground truth is never resampled, so no interpolation error is baked into the labels. `model.check_input_size()` raises rather than let a bad size through.

Result: **~11 000 crops, ~9.5 GB of compressed shards.**

### Domain imagery — Bhoonidhi (Cartosat)

ISRO's own portal, free registration. The API is a **STAC** interface (use `pystac-client`), requested via `bhoonidhi@nrsc.gov.in` — **lead-time item, send in week 1.** The API is optional; the *account* is what matters, and browser download works without it.

**Access tiers, per Indian Space Policy 2023 (from the NRSC Bhoonidhi brochure):**

| Data | Tier |
|---|---|
| 5 m resolution **and coarser** | Free and open to everyone |
| **Finer than 5 m** | Free to Government Entities *on declaration*; **priced for Non-Government Entities** |
| 30 m Carto DSM | Open for GE, priced for NGE |

| Sensor | GSD | Us as an NGE |
|---|---|---|
| Cartosat-1 / 2 / 3 | 2.5 → 0.25 m | **priced** |
| Resourcesat LISS-IV | 5.8 m | **free** |
| LISS-III / AWiFS | 23.5 / 56 m | free |

LISS-IV is free and genuinely Indian, but at 5.8 m a building spans 2–3 pixels — adequate for terrain relief, weak for building height, and a DSM needs both.

**Worth pursuing:** GEs get finer-than-5 m data free on declaration. SIH is a Government of India initiative under the Ministry of Education, this is an ISRO problem statement, and many Indian engineering colleges are government institutions. Ask Bhoonidhi directly whether that qualifies — worst case is a no.

**Not a blocker either way.** `meanOffNadirViewAngle` spans 4.8°–28.9° across the 67 DFC scenes (§6.2), so we can already measure accuracy degradation vs look angle on data we hold. Cartosat strengthens the domain story; its absence does not sink it.

### Bhoonidhi, resolved 27 Aug 2026 -- and it is not what section 8 wanted

Access works, and needed no email. Despite the Applications page saying to contact NRSC for
access, ordinary portal credentials authenticate at `/auth/token`. `tools/bhoonidhi.py`
wraps auth, collections and search, with credentials kept outside the git tree.

**The ceiling is the problem.** All 64 live collections, checked against the API rather
than the docs: the only CartoSat entry is `CartoSat-1_PAN_CartoDEM_30m`, a 30 m **DEM**
rather than imagery. The best optical is ResourceSat LISS4-MX70 at **5.8 m**, then LISS3 at
23.5 m and AWiFS at 56 m. Everything else is SAR, ocean colour or NISAR. There is no
sub-metre optical imagery in this API at all.

Two consequences, both of which shrink what section 7.1 can claim:

- **LISS4 at 5.8 m against our 0.3 m training data is a 19x GSD gap.** That is a different
  problem, not a domain gap. Using it would mean retraining at that scale rather than
  testing transfer to it.
- **CartoDEM cannot validate building heights.** Its own metadata reports 30 m posting and
  a vertical accuracy of **8 m LE90 / 15 m CE90**. A building footprint is sub-pixel at
  30 m, and 8 m LE90 is *worse than our ground RMSE of 1.92 m*. It cannot referee the one
  thing we are weakest at.

What it is genuinely good for is a coarse **absolute-scale and bias anchor** over large
flat areas, taking the metric-anchor role from SRTM/Copernicus. For a SAC jury, checking
our DSM against ISRO's own CartoDEM reads better than checking it against an American DEM
-- provided we quote its 8 m LE90 rather than implying it arbitrates buildings.

Retrieval is verified, not assumed. `tools/bhoonidhi.py download` pulled a real tile:
42 MB zip holding a 3600x3600 float32 GeoTIFF, EPSG:4326, 30 m pixels, 100% valid,
elevation 497.8 to 1393.7 m over north Bengaluru.

**And CartoDEM measures a different quantity than we predict.** It is absolute elevation
above sea level. Our model outputs AGL, height above the local ground, which is zero on
the ground by construction. The two cannot be differenced directly, so CartoDEM cannot
validate our output even at its own coarse scale. What it can do is supply the terrain
surface that turns our AGL into an absolute DSM, which matters for output-format
compliance (7.3) if the deliverable is specified as absolute elevation rather than AGL.
That is a product feature, not a validation.

Practical notes: LISS4 scenes come back `Online: N`, so they need ordering rather than
direct download, and item `assets` expose only metadata and thumbnail, so the raster comes
from `/download` rather than from an asset href. Search wants full ISO timestamps and a
range of 366 days or less; both fail as HTTP 406 otherwise.

### Metric anchor — SRTM 30 m / Copernicus GLO-30

Public, free, named in the problem statement. *Not* CartoDEM (see §5 Stage 02).

## 9. State of play — 27 Aug 2026

24 days to the 20 Sep idea submission. Written after a long model day so the next session
does not have to reconstruct it.

### The model is in better shape than it felt, and our own metric was the problem

**Best model today: run02 + TTA.** Region-disjoint, 80 whole val tiles, 3,090 buildings.

| | run02 | run02 + TTA | run04 |
|---|---|---|---|
| whole-tile RMSE | 6.456 m | **6.369 m** | 7.031 m |
| **per-building RMSE** | 3.771 m | **3.616 m** | 3.991 m |
| ground RMSE | 1.92 m | — | 2.09 m |
| ECE | 0.084 | 0.078 | **0.048** |

TTA is a free 4% on the headline metric and it never cost a training run. run04 wins only
on calibration.

**Per-building is the metric the field reports and we were not computing it.** Our
per-pixel building RMSE of 16.37 m is dominated by roof edges, where the prediction crosses
from ground to roof and is briefly wrong by the whole building height. It is not comparable
to any published figure and it made the model look far worse than it is.

### Where the error actually lives, and why more training will not fix it

| true height | buildings | RMSE | bias |
|---|---|---|---|
| 0-3 m | 166 | 1.47 m | +0.26 |
| 3-6 m | 2,305 | **1.45 m** | +0.21 |
| 6-10 m | 443 | 1.90 m | -0.75 |
| 10-20 m | 116 | 4.61 m | -1.17 |
| **> 20 m** | **60** | **24.04 m** | **-17.26** |

**79% of the squared error comes from 1.9% of the buildings.** Under 10 m — 94% of the
stock — we are at 1.45 m, which is genuinely good.

The tail is a **data** problem, proven rather than assumed (`docs/probe-01-tall-buildings.md`):
train holds 166 buildings above 30 m and tops out at 82.8 m while val reaches 155.3 m, and
fine-tuning hard on those tall buildings moved transfer bias the *wrong* way, -18.22 to
-19.10 m, while fit improved. Classic overfitting of a tiny sample. This rules out run05 as
tall-enriched training and rules out re-splitting DFC2019 — rearranging 341 tall buildings
creates no information.

**Also note: the held-out test split contains ONE building above 30 m** (max 34.0 m). It
will flatter us relative to val. Say so when reporting it.

### There is no target number, and the criterion we under-weighted

ISRO specifies **no threshold**. The brief says: *"RMSE, MAE and correlation against LiDAR
or reference data — with stability required across urban, sparse, hilly and forested
landscapes."* Judging is comparative and validation-quality driven. The 5.9 m
GlobalBuildingAtlas figure is a **credibility anchor we chose**, not a requirement.

The scored thing we have been under-weighting is **stability across the four named terrain
types**. `evaluate.py` already breaks results down that way; Sikkim now covers *hilly* with
947 m of relief at 31 degrees median slope and slope leakage measured at r -0.044.

And the register the brief asks for, quoted: *"A centimetre-accuracy claim gets dismantled
in questions"*; *"Relative DSM is solid, absolute scaling is +/-8% against SRTM, here is the
error map"* reads as competent. Our honest tail story is therefore an asset, not a hole.

### Remaining work, in priority order

**Blocking, gated to Zaid**
- [ ] **Six names registered for SIH.** Overdue since week 1. Nothing else matters without it.

**Half the marks, at week-one state**
- [ ] **Viewer.** Currently three scenes and they are `zeroshot` and `truth` — it demos our
      *worst* model. Load a real run02+TTA scene, verify navigation, standalone deployment.
- [ ] **Demo video.** Explicitly required by the submission.

**Evidence pack — mostly measured, needs assembling**
- [ ] Benchmark table with the per-height breakdown and the honest tail explanation
- [ ] Error maps, calibration curve
- [ ] Per-terrain stability across urban / sparse / hilly / forested
- [ ] The India story: Sikkim + Open Buildings cross-check (`docs/evaluation-protocol.md`)

**Cheap model wins, no training needed**
- [ ] Apply TTA in the shipping path (already measured: 3.771 -> 3.616 m)
- [ ] Ensemble run02 + run04 (+run05) — different error profiles, ~30 min of inference
- [ ] ONNX export + int8, verified — differentiator 6.4, explicitly scored
- [ ] **Score the held-out test split ONCE**, at the very end, and report whatever it says

**In flight**
- [ ] **run05 — V1-Large (335 M, Apache-2.0) on Kaggle.** Only the backbone changes from
      run02's recipe, so the result is attributable. Probe 02 measured zero-shot per-building
      correlation +0.351 (V2-Small) -> +0.470 (V1-Base) -> +0.659 (V1-Large).
      `kaggle/run_train.py`, resumable, fp16.

**Deferred, and worth marks as stated next steps rather than gaps**
- FDS / LDS for the long tail (Yang et al., ICML 2021) — a 30-minute probe would test it
- Indian weak-supervision training (Chen et al. 2025) — Maxar gives 410 km2 of sub-0.5 m
  Indian imagery and Open Buildings labels it; four times our current training area, in the
  target domain

### Process rule earned the hard way

run03 and run04 both spent 3.5 hours testing a hypothesis that a 30-minute probe would have
killed. From here: **predict the outcome in writing, state what would falsify it, probe
cheaply, then decide.** If you cannot say what result would prove you wrong, do not start
the run.

## 9c. State of play — end of 28 Aug 2026

23 days to submission. Written at the end of the day so tomorrow starts from facts.

> **Recovered 28 Aug after an accidental truncation of this file** (a bad write in a
> tooling script emptied PLAN.md; everything up to the last commit came back from git,
> and this section was rebuilt from the session transcript). The wording may differ in
> small ways from what was originally typed; the numbers are the measured ones.

### The finding that reframes the model work

**We compress every height toward the median** (`docs/probe-06-compression.md`). Spotted by
Zaid in the viewer's drag-to-compare before it was measured:

    ours = 0.473 * truth + 2.66 m

Short buildings come out 14-20% **too tall**, buildings over 40 m at **half** height. This is
regression to the mean: with a long-tailed target whose median is 4.2 m, the loss-minimising
answer under uncertainty is always "about four metres".

**The per-height RMSE table was hiding it**, and I had been quoting that table. "94% of
buildings are under 10 m and we are at 1.4 m" is true in metres and conceals a systematic
15% over-call in ratio.

### Three things this rules out

1. **Inference-time correction - measured, not shipped.** Fitted on TRAIN, tested on VAL:
   RMSE 3.667 -> 3.543 (-3.4%) but MAE 1.498 -> 1.578 (+5.3%) and **correlation unchanged**,
   as it must be under a linear map. No information added, only error redistributed. It also
   cannot be calibrated: train holds 7 buildings above 20 m, val holds 60, so the fit applies
   1.24x where 2.1x is needed. **The data needed to calibrate it is the data we do not have.**
2. **More epochs.** run03 and run04 both plateau at epoch 4-5 of 12 and oscillate in a 0.15 m
   band for the rest. run04 spent 107 minutes and 819 thermal cool-downs learning nothing.
3. **More data of the same kind.** The compression follows the distribution's *shape*, not
   its size. Ten times more 4 m buildings gives the model the same reason to hedge.

### Still open, in priority order

- [ ] **Ordinal / binned head, revisited.** The best-founded experiment we have: a head that
      *picks a bin* cannot average two answers into a wrong middle one, which is the
      mechanism now measured. Code already exists from run03 (`--bins`, head-tail cut). run03
      was dropped on whole-tile RMSE 6.950 vs 6.456 - too quick a judgement, since its
      **ground RMSE improved (1.75 vs 1.92)** and crop-wise scoring put it ahead.
- [ ] **run05 (V1-Large) - still training on Kaggle.** Read it first. If a 13x bigger backbone
      also plateaus near 7.9, that confirms the ceiling is the loss, not capacity.
- [ ] **Tall-building data.** The correct long-term fix and a multi-day project: open city
      LiDAR plus open imagery. Belongs in "next steps" for the submission, not in the 23 days.
- [ ] **Demo video.** Everything it needs now exists.
- [ ] **Score the held-out test split ONCE**, at the very end.
- [ ] **BLOCKING: six names registered for SIH.** Outstanding since week 1.

### Deliberately not doing

- **Depth Anything 3.** DA3MONO-LARGE is Apache-2.0 at 0.35B and genuinely the right next
  backbone, but the package pulls ~90 transitive dependencies including xformers and a numpy
  downgrade, and our head is built on the transformers DPT interface. Architecture migration,
  not a probe. Write it up as identified future work.
- **Indian Open Buildings weak supervision.** Would help domain adaptation and **hurt** the
  height scale: OB under-calls tall buildings by -6.94 m, the same bias we are trying to fix.

### Shipped today

Six viewer scenes with real terrain, drag-to-compare, error map, upload
(`tools/serve_viewer.py`), absolute DSM via Copernicus GLO-30 (`--dem auto`), GCP anchoring,
`--auto-zoom` for evaluation resolution, single-file standalone build, evidence pack with
five figures, probes 03-06.

## 9d. Measured 28 Aug, later session — five hypotheses closed, one reopened

Every line here carries the measurement it rests on. Five hypotheses were tested and came
back negative. This section originally concluded from that "every lever internal to the
model is now spent". **That conclusion was too broad and is withdrawn** — see 9h. The five
results below stand; what does not stand is generalising them to levers never tested.

### run05 (V1-Large) is VOID — nine GPU-hours, zero information

It ran 543.6 min to exit 0 and printed "best val RMSE 28.057 m", which is misleading.
Crop-wise epoch 0 was **70.8 m at r -0.046** where healthy run04 epoch 0 is **8.24 m at
r 0.660**. `sigma` sat at exactly **33.115 m = exp(7/2)** in every logged step and `ECE` at
**0.1881** in all 12 epochs: `log_var` was railed against `log_var_max=7.0` (`losses.py`)
from the first step, and `clamp` zeroes the gradient there, so the variance head was dead on
arrival. Loss went NaN at `e6 s6066/11532` (289.6 min) and stayed NaN for the remaining
4.2 h — 218 of 455 logged steps. If ever rerun: raise `log_var_max`, raise the LR (peak was
2.77e-06), and add a NaN guard that aborts instead of burning half the run.

**Superseded 29 Aug: the cause was found, and it is a bug in `model.py`, not the backbone.**
See 9h. "Bigger backbone" was listed under "ruled out" on the strength of this void run,
which is precisely backwards — a run that produced zero information cannot rule anything
out. It has been removed from that list.

### No head design moves the compression

`tools/analysis/compression_compare.py`, per-building on the same region-disjoint val split:

| run | head | slope | per-building RMSE | r |
|---|---|---|---|---|
| run02 | regression | 0.471 | 3.771 m | 0.755 |
| run02+TTA | regression | 0.483 | **3.616 m** | **0.787** |
| run03 | bins + soft-argmax | 0.429 | 4.050 m | 0.717 |
| run04 | bins + **HTC** | 0.491 | 3.991 m | 0.711 |

**run04 IS the head-tail-cut run** — its checkpoint carries `conv_bins_bg` and `conv_htc`
weights with `no_htc=False, htc_weight=1.0`. (Reading `args['htc']` returns None because the
stored key is `no_htc`; that misread briefly put a redundant run06 on the schedule.) So the
field's prescribed fix has been tested. Every head lands at slope 0.43–0.49.

### The decode is not the fix either

`tools/analysis/bin_mode_probe.py`, 3.65 M building pixels from 48 val crops each containing
a building above 20 m:

| true band | n | truth | expectation | argmax bin | P(h>20 m) |
|---|---|---|---|---|---|
| 20-40 m | 733,129 | 25.6 m | 21.6 m | 23.8 m | 0.779 |
| 40 m+ | 561,653 | **75.6 m** | **24.9 m** | **26.4 m** | 0.907 |

On pixels truly above 20 m the argmax sits only **+1.86 m** from the expectation. There is no
tall mode being averaged away. HTC also works as advertised (91% of mass above 20 m), so
roof-vs-ground was never the failure. **The model knows a building is tall and cannot tell
26 m from 76 m.**

Secondary defect logged: the adaptive bins starve the tail — typically 32 bins over 0–10 m
and ~88 over 10–40 m, but only **3–6 bins for all of 40–110 m**, with adjacent-centre gaps of
17.8–24.2 m. Two of ten images collapse outright. run04 cut `--chamfer` 10x from run03
(0.1 -> 0.01), and Chamfer is what pulls bin edges onto the truth quantiles.

### LDS/FDS is ruled out — its premise does not hold

`tools/lds_probe.py`, 801,233 building pixels. LDS assumes the tail is *under*-weighted:

| band | % of building pixels | % of squared error |
|---|---|---|
| 3-6 m | 36.57% | 2.01% |
| 20-40 m | 6.10% | 33.82% |
| 40 m+ | 1.36% | **44.96%** |

**Pixels above 20 m are 7.47% of building pixels and already carry 78.78% of the squared
error.** Nearly four fifths of the building-pixel gradient already points at them. LDS at
alpha=0.5 would upweight 40 m+ by a further **78x**, past 95% of the loss, wrecking the
1.45 m short-building accuracy. (A first pass over ALL pixels was misleading — median AGL
0.23 m because 63.6% of pixels are ground. That imbalance is the one HTC already solves.)

### Ensembling is ruled out — zero inference spent

Both evals cover the same 3,090 buildings in the same order, so the mix was computed
directly. `corr(err_run02, err_run04) = 0.9248` — they make the same mistakes. No weighting
beats run02+TTA's 3.616 m (best mix 0.7/0.3 = 3.647 m); tall RMSE 23.32 -> 23.29 m, noise.
PLAN had costed this at "~30 min of inference".

### Shadow: taken to its ceiling, and it is below what we have

`tools/shadow_probe.py` (corridor occupancy) and `tools/shadow_ceiling.py` (oracle shadows
ray-cast from the truth DSM). The cue is real — marching anti-sun on OMA_288_012's tallest
building gives luminance 166 (roof) -> 93–113 held ~120 px -> 254 (lit ground); marching
sunward leaves the tile at once. Three findings closed it:

1. **Even perfect segmentation is not enough.** Oracle shadows give overall **r 0.503,
   slope 0.583, RMSE 2.61 m** (675 unoccluded buildings). The network already achieves
   **r 0.787** per building.
2. **Real detection is nowhere near the oracle.** Luminance+Otsu scores **IoU 0.187**
   (precision 0.204, recall 0.689); Otsu picks thresholds from 76 to 154 across tiles.
3. **The target population barely exists.** Across the ten tiles with the most tall stock in
   the dataset there are **10 building components above 20 m out of 766**, and their shadow
   paths are *unobstructed* (0%). Occlusion is not the limiter — sample size is. ~60 such
   buildings exist in the whole validation set.

Two real bugs were found and fixed on the way: a p22 luminance cut that landed INSIDE the
shadow (every building came back ~2 m), and a corridor starting from one far CORNER of a
building rather than each column's own far edge, which silently dropped every large
irregular footprint — i.e. exactly the tall ones. Fixing it raised the yield 616 -> 732.

## 9e. GCP calibration — the PS milestone, implemented and measured

`evaluate.py` now tags each building with its tile (`per_building.npz` gained `tile_idx`),
because a control point calibrates one scene and pooling across tiles flatters the result.
Sanity: the retagged eval reproduces per-building RMSE 3.616 m exactly.

**A selection trap worth remembering.** The first probe required 25 buildings per tile and
looked clean. It was worthless: those 36 tiles contain **zero** buildings above 20 m
(RMSE 1.381 m), while all 60 tall buildings sit in tiles with fewer (RMSE 8.487 m). Dense
suburbs have many small buildings, downtowns few large ones, so a threshold chosen for
sample size excluded 100% of the problem.

**Affine is the wrong functional form, and the right one is a power law.** The error is a
height-dependent RATIO (1.20x at 0–3 m, 0.52x above 40 m); no straight line can be above 1 at
one end and below it at the other. Measured head to head on held-out buildings, per tile,
over the 10 validation tiles containing a building above 20 m
(`tools/analysis/calibration_form.py`):

| form | k | RMSE | vs uncalibrated | MAE | tiles worse |
|---|---|---|---|---|---|
| uncalibrated | - | 7.248 m | - | 4.013 m | - |
| affine | 8 | 6.154 m | -15.1% | 4.477 m | 8 of 10 |
| power | 5 | 5.730 m | **-20.9%** | 4.202 m | **4 of 10** |
| power | 8 | **5.442 m** | **-24.9%** | **4.038 m** | 5 of 10 |

Affine buys RMSE by spending MAE (+11.6%); the power law takes more RMSE and leaves MAE where
it found it (+0.6%). Over all 57 usable tiles it is the only form that beats doing nothing
(-4.4% vs affine's +0.1%).

Shipped as `fit_gcp_power` in `dem.py` with `--gcp-scale` and `--gcp-form power` (default) in
`infer.py`. `fit_gcp_offset` fitted an offset only, justified by "our heights are already
metric" — which the compression measurement falsifies.

**Guards calibrated against the measurement, not chosen.** Requiring 2 control points in the
upper HALF of their span would refuse 24.6% of the k=8 draws that produced the gain; the
upper 60% refuses 6.3%, so that is the threshold. The exponent must be >= 0.95, which is what
catches "would compress further". Control values are read as a STRUCTURE median (region-grown
from the point), not a fixed patch, because the gain was measured with per-building estimates
and a 5x5 patch is a noisier quantity — on OMA_288_012 point samples returned 0.59 m where
truth was 5.6 m.

`tests/test_gcp_affine.py` — 9 tests, all passing, including exact recovery of a known ratio
error (exponent 1.3889 against a true 1.3889, residual 7.7e-07 m). Existing suite 19/0.

**Honest end-to-end status: on the real tiles tested the guards DECLINE and it falls back to
offset-only.** On OMA_288_012 the fitted exponent is 0.40–0.57 because that tile
under-predicts everything (offset-only RMSE 22.82 m against a 6.4 m whole-tile average); on
JAX_165_018 it is 0.505 because the model is already good there (tall bias -4.55 m). Both
refusals are correct. **It never degrades a result** — it helps only when control points
carry building-level heights and the scene genuinely spans a height range.

## 9f. Banked 28 Aug — deployment and evidence

- **ONNX export of the SHIPPING model.** Only run01 — our worst checkpoint, epoch 1 — had
  ever been exported. run02 now exports self-contained at 100.7 MB with max divergence
  **0.0010 cm** against PyTorch over 804,972 inputs (PASS at 5 cm), int8 at **36.8 MB, 63%
  smaller**, costing 0.161 m. CPU inference **509 ms per 518x518 tile**, so it runs with no
  GPU. Export is FIXED at 518x518 — dynamic shapes raise a runtime exception — fine for
  sliding-window inference but stated rather than implied.
- **Standalone verified by loading it.** Headless Chrome on `file://viewer_standalone.html`
  (15.2 MB): no external refs, 0 leftover ES imports, built from **run02** with zero
  `zeroshot` mentions. Proof it runs rather than parses: the source has **0** `<canvas>` tags
  and the rendered DOM has **1**, `data-engine="three.js r169"` at 764x429.
- **TTA was already shipped** — `serve_viewer.py:72` passes `--tta`. The old checkbox was
  stale.
- **Per-terrain stability** (run02+TTA), the PS criterion verbatim: sparse **1.819 m**, mixed
  **2.676 m**, forested **3.250 m**, urban **13.809 m**, r +0.575 to +0.829. Three of four
  terrains are strong and the entire height problem is the urban row. Already written up in
  `docs/evidence-pack.md`, which also states plainly that DFC2019 has no hills and carries
  Sikkim separately.

## 9g. Remaining work — do not divert

**Blocking, gated to Zaid**
- [x] Six names registered — Zaid Ansari, Hassaan Shaikh, Justin Fernandes, Gracian Lopes,
      Karan Patel, Riya Gholap. Done 28 Aug.
- [x] Renders reviewed by Zaid, 27 Aug.

**Open**
- [ ] **Demo video.** Explicitly required by the submission. Not started.
- [ ] **GCP calibration as a click-to-place viewer control** with visible before/after. The
      backend is done and tested; only the UI remains.
- [ ] **Raise `--chamfer` and revisit bin allocation** if any further training happens — the
      tail is starved of bins (3–6 for 40–110 m).
- [ ] **Score the held-out test split ONCE, at the very end.** It holds ONE building above
      30 m, so it will flatter us; say so when reporting.

**Ruled out with evidence — do not revisit**
Head architecture (slope 0.43–0.49 across three heads), decode
change (+1.86 m, no tall mode), LDS reweighting (tail already carries 78.78% of the error),
ensembling (error correlation 0.925), shadow (oracle ceiling r 0.503 below our 0.787),
inference-time global de-compression (correlation unchanged by construction).

## 9h. Measured 29 Aug — run05 did not fail because V1-Large is too big

Zaid, 28 Aug: *"I find it hard to believe that it isn't fixable."* He was right. The
capacity hypothesis was never tested; run05 was broken before it started, by a bug.

### The bug: the pretrained readout is not calibrated for a V1 backbone

`conv_mu` is `base.head.conv3`, reused from the checkpoint. It emits DISPARITY, and its
output scale is a property of that checkpoint's feature magnitudes. Measured on real train
crops with `tools/preflight.py`:

| backbone | median abs(truth - mu) at init | conv3 abs(w) |
|---|---|---|
| DA-V2-Small | **16.53 m** | 0.088 |
| DA-V1-Large | **666 m** | 0.100 |

The head weights are comparable; the necks are not. Nothing rescaled the readout when the
backbone family changed.

### Why a large initial residual is fatal, measured

`tools/nll_deadlock_probe.py` measures the gradients directly, at a fixed state:

| log_var | sigma | beta | gradient reaching mu |
|---|---|---|---|
| 0.0 | 1.000 m | 0.0 | 1.0000x |
| 3.0 | 4.482 m | 0.0 | 0.0498x |
| **7.0** | **33.115 m** | 0.0 | **0.0009x** |
| 7.0 | 33.115 m | 1.0 | 1.0000x |

And `d loss / d log_var` is negative — pushing log_var UP toward the clamp — whenever the
residual exceeds sigma: -0.0029 at 5 m, -0.0487 at 20 m, -0.4393 at 60 m. So a 666 m
residual drives log_var into `log_var_max=7.0`, where `clamp` zeroes its gradient while
suppressing mu's by ~1100x. Deadlock. That is run05's `sigma = 33.115 m` in every step.

Note the beta column: beta-NLL removes the sigma dependence from mu's gradient entirely.
run02, our best model, was trained with `--beta 0.5`; the default is 0.0.

**It is a race, not a certainty.** An 80-step probe with a fast LR ramp recovered from the
666 m start on its own (val RMSE 6.366 m, r 0.749) — the head lr pulled mu down before
log_var railed. run05 lost that race because its schedule warmed over 576 steps to a peak
of only 2.77e-06 while NLL activated at step 100. Whether the run dies is a function of the
LR schedule, which is why this was never reproducible from the hyperparameters alone.

### The fix, and what it does

`init_mu="constant"` zeroes `conv_mu.weight` and sets its bias to a constant height —
exactly the argument `conv_log_var` already made for itself. It discards a 32->1 linear
readout of disparity that has to be relearned as height regardless; backbone, neck, conv1
and conv2 are untouched. Default stays `"pretrained"`, so run01-run04 remain reproducible.

Measured, V1-Large, everything else held at run02's recipe:

| | pretrained head | constant head | run04 reference |
|---|---|---|---|
| residual at init | 666 m | **0.02 m** | 16.53 m |
| median grad-norm | 3.1e+06 (exploding) | **6.9e+03** | 1.3e+03 |
| train RMSE at step 60 | 164 m | **3.255 m** | 3.887 m |
| preflight verdict | DO NOT LAUNCH | **GO** | GO |

At step 60 V1-Large is already ahead of V2-Small. That is not a result yet, but it is the
first evidence the capacity question has an answer worth having.

Also measured: **V1-Large OOMs at batch 8** on the 12 GB 3060 (wants 16.77 GB). Batch 2
fits at 9.51 GB and runs 0.59-0.62 s/step. Batch 4 "fits" at 13.36 GB only by spilling to
shared memory and is 12x slower per step (7.28 s) — a silent trap.

### Nothing long-running is launched unwatched again

`tools/preflight.py` runs the REAL config — same model, loss, optimiser and shards — for
60 steps and returns GO or DO NOT LAUNCH against eight checks. Calibrated on the known-good
V2-Small config, where it passes and its ETA reproduces run04's actual 107 min.

Three tripwires now abort `train.py` mid-run: non-finite loss, sigma railed at the clamp
for `--rail-patience` steps, and an epoch-0 val gate (`--gate-rmse 25`, `--gate-corr 0.20`;
run04 reached 8.239 m / +0.660). `tools/tripwire_selftest.py` exercises all three plus a
control, and all four behave as specified — critically the control, a healthy run with
tripwires armed, is NOT aborted. Against run05's recorded numbers all four would have
fired; the epoch-0 gate alone caps the loss at roughly 45 min instead of 543.6 min.

### run06 — the experiment run05 was supposed to be

V1-Large, `--init-mu constant`, run02's recipe otherwise (`--beta 0.5 --lr 5e-6
--warmup-mse 200 --clip 1.0 --grad-weight 0.5`), effective batch 8 as `--batch 2 --accum 4`,
6 epochs (~3.9 h) because run03 and run04 both plateaued at epoch 4-5 of 12. Preflight GO.
Only the backbone and the head init differ from run02, so the result is attributable.

## 9b. Schedule

Build the baseline first: once it runs end to end we always have something demoable, and every later improvement becomes optional rather than critical-path.

### Week 1 — The ugly vertical slice

**Goal: a complete end-to-end path, however bad the output looks.**

- [x] Track 3 metadata checked for sun angles → §6.2 gate passed (26 Aug)
- [x] DFC2019 Track 1 extracted, probed, sharded; held-out test split carved out (26 Aug)
- [x] **Zero-shot DA-V2 → heightmap → GeoTIFF → Three.js flythrough, working end to end** (26 Aug)
- [x] Honest zero-shot baseline on 80 held-out tiles → **4.68 m RMSE** (test split; the val bar is 9.31 m) (§5)
- [x] Training loop verified end to end: resumable, bf16/fp16 autodetect, per-class + calibration eval
- [x] Loss suite under test — 19 tests, incl. recovery of known heteroscedastic σ
- [x] Email Bhoonidhi (§8) — **sent 26 Aug**
- [ ] Resolve SIH team registration (§11) ← *the only unrecoverable open item*
- [ ] Verify Azure GPU quota (expect it blocked; nothing lost if so)

*The output will be mediocre. Irrelevant. After this week there is always something to
demo, and nothing left that is critical-path.*

**Closed 26 Aug.** The slice runs: image → sliding-window inference with cosine blending
→ GeoTIFF → browser bundle → Three.js flythrough with texture / height / uncertainty /
slope surfaces and a two-point measurement tool. Ground truth is exported as a second
scene so the two can be flown side by side.

**Local compute turned out to be far less of a constraint than assumed.** Measured on the
3060: batch 12 fits in 7.6 GB and throughput plateaus near **25 crops/s** at batch 8, so
one full 3-epoch pass over ~7 800 training crops is roughly **15 minutes**, not the
5–9 hours §5 estimated from the A6000 figure. That estimate assumed a much larger crop
count. The consequence is strategic: we can afford many training runs and real ablations
locally, and Kaggle becomes a parallel-experiments resource rather than a dependency.

### Week 2 — Make the model good

- [x] DA-V2 Small fine-tuned on DFC2019 AGL, uncertainty head in from the first run (§6.1)
- [ ] Metric calibration against SRTM 30 m
- [x] **The honest table:** zero-shot vs fine-tuned vs TSE-Net published, same held-out tiles
- [ ] Benchmark remaining backbones on Kaggle (parallel sessions, 30 GPU-h/week)
- [x] Calibration curves — is the predicted sigma actually right?

*Slow down here and understand the calibration and the NLL properly (§10). These are what
the jury probes hardest.*

### Week 3 — Make the viewer good, and measure the domain

- [ ] Confidence shading on the mesh (§6.1) — judges *see* the model's doubt
- [ ] Point-to-point height measurement with error bar (§6.3)
- [ ] Error maps + per-terrain breakdown: urban / sparse / hilly / forested
- [x] Accuracy vs `meanOffNadirViewAngle` (4.8°–28.9°) — the domain-gap number we can get
      without Bhoonidhi (§7.1)
- [ ] Cartosat check if access came through

### Week 4 — Package and submit

- [ ] ONNX export, quantisation, CPU-only inference timing (§6.4)
- [ ] Air-gapped Docker image; deploy demo to Azure
- [ ] Output format compliance pass (§7.3)
- [ ] Technical report + benchmark write-up
- [ ] Idea submission written **against the evaluation criteria, in their language**
- [x] Re-check live submission counts on the portal (§3) — 3/500 on 12 Sep 2026

## 10. Execution — solo build

**Decided 26 Aug 2026: this is a solo build.** The six-person split is withdrawn. That
invalidates the original competitive premise — *"both halves build in parallel from hour
one, so nobody is idle"* — because one person cannot parallelise. Everything is serial now.

### The sequencing inverts

For a team: model baseline first, viewer second. **Solo that is the failure mode** — three
weeks perfecting a model, a rushed viewer, half the marks forfeited.

**Vertical slice first.** Week 1 produces a complete, ugly, end-to-end path: image in →
*some* heightmap (zero-shot is fine) → mesh → flythrough in the browser. It will look
mediocre. Irrelevant. Once it exists, every later improvement is optional rather than
critical-path, and there is no world where we hold an excellent model and nothing to show.

Improve the model *behind a viewer that already works*.

### Two timelines — this is what makes solo viable

| Date | Deliverable |
|---|---|
| **30 Sep 2026** | **Idea submission** — a proposal judged on approach. Needs *evidence*: benchmark table, error maps, calibration curve, demo video. Not a finished product. |
| Grand finale (later, if shortlisted) | 36 hours of building. The real runway. |

25 days from 26 Aug. Solo, that is enough for evidence and a demo. It is not enough for
four differentiators plus a polished system — hence the scope call below.

### Scope, revised for solo

| | Verdict |
|---|---|
| **6.1 Uncertainty** | **Committed.** Already in `losses.py`. Cheapest, best return. |
| **6.3 Measurement tooling** | **Committed, trimmed.** Point-to-point height + error bar for submission. Cross-sections and difference maps → finale phase. |
| **6.4 Deployability** | **Committed.** Mostly packaging, explicitly scored. |
| **6.2 Shadow prior** | **Relocated to finale phase.** Not cut — solo, shadow detection + geometry + validation is a week we do not have before 20 Sep. It is excellent *submission* material as a stated method with evidence behind it (54 of 67 scenes carry usable solar geometry). Full credit for the insight in the proposal; build it on the finale runway. |

### The explain-it-yourself risk is now concentrated

§7.5 applies with force: solo, with AI-assisted code, **every component that cannot be
explained from memory is a liability carried by one person.** Standing rule — readable over
clever, and the metric calibration and uncertainty head get walked through properly, because
those are what a SAC jury probes hardest.

## 11. Open decisions

- [x] ~~Who takes the two CV seats~~ — solo build, concept withdrawn (26 Aug)
- [x] Viewer framework → **Three.js** (26 Aug)
- [x] Does DFC2019 ship solar metadata → **yes**, gate passed (26 Aug), see §6.2
- [x] Differentiator scope → 6.1 / 6.3 / 6.4 committed, 6.2 to finale phase (26 Aug)
- [x] **Six names registered for SIH** — **done, confirmed 12 Sep 2026.** The one blocker
      that could not be fixed by working harder. The build stays solo; only the registration
      needed six names.
- [x] Commit to SIH26175 or hedge to SIH26143 — **resolved 12 Sep 2026: stay.** The portal
      shows SIH26175 at **3/500** submissions against a 30-09-2026 deadline, so §3's crowding
      worry does not materialise and there is no slot to race for.
- [x] Bhoonidhi access tier — **resolved 27 Aug**: works with portal credentials, but
      the ceiling is 5.8 m LISS4 and a 30 m CartoDEM. Cannot referee buildings.
      Replaced by Maxar Open Data + Google Open Buildings 2.5D (§9).
- [x] Backbone — **decided 27 Aug**: Depth-Anything-V1 is Apache-2.0 at every size,
      so capacity was never licence-blocked as previously assumed. run05 tests
      V1-Large (335 M) on Kaggle.
- [ ] **Ship run02+TTA or run05?** Decide on measured per-building RMSE, not
      crop-wise val, which has picked the wrong winner three times.

## 12. Standing rules

1. Never let the viewer be the last night's work — 50% of marks.
2. Report uncertainty, never bare precision (§7.4).
3. Follow their output spec exactly, even over a better model (§7.3).
4. Build what you can explain live (§7.5).
5. Hold out a test set from day one; never tune against it.
6. Preprocess locally, train on Kaggle, deploy on Azure.
7. No code inside notebooks — clone from git.
