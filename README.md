# DepthWizard — SIH26175

**One optical satellite image in. A navigable, measurable 3D surface out.**

Smart India Hackathon 2026, ISRO / Space Applications Centre. Idea submission closes **30 September 2026**
(the SIH portal prints 30 September in all 233 rows; the 20 September in the team brief is wrong — see
`docs/problem-statement.md`).

The plan of record — every design decision and why it was made — is [PLAN.md](PLAN.md).

## Scoring (this drives everything)

| Weight | Criterion |
|--------|-----------|
| 50% | DSM accuracy — RMSE / MAE / correlation, **stable across urban, sparse, hilly, forested** |
| 50% | Rendering quality & UX — projection accuracy, navigability, standalone deployment |

Half the marks are the front end.

---

## What makes this more than a depth model with a camera flying through it

Monocular height estimation is ill-posed — there is no unique 3D scene behind a 2D image. The model does not pretend otherwise:

1. **Every pixel carries an error bar.** The network predicts a distribution, not a number, trained with a heteroscedastic Gaussian NLL. It says where it is unsure — and the claim is falsifiable: if we assert 68% of errors fall inside 1σ, the calibration curve has to show it.
2. **The viewer is a measuring instrument.** Two clicks give ground distance, height difference and slope distance *in metres*, with uncertainty on the height difference propagated in quadrature.
3. **Heights survive the trip to the browser exactly.** Terrain ships as raw float32, not a 16-bit PNG — browsers decode images through an 8-bit canvas and would silently quantise the metres.
4. **It runs on a laptop GPU.** DA-V2-Small, Apache-2.0, 24.8 M parameters.

## Where it stands

| | |
|---|---|
| **Zero-shot baseline to beat** | **4.68 m RMSE** over 80 held-out test tiles |
| Oracle ceiling for a linear fix | 3.94 m — *not deployable; it needs the answer to compute the answer* |
| Data | 2 783 tiles / 108 regions, split **by region** so no neighbourhood spans two splits |
| Pipeline | end to end: image → heightmap → GeoTIFF → browser bundle → flythrough |
| Tests | 19, including recovery of a known heteroscedastic σ (true 0.50 / 3.00 → 0.497 / 2.99) |

**Stage 01 is not zero-shot** — but the original claim that DA-V2 "correlates with rooftop albedo, not height" at nadir was **measured and found too harsh**. It correlates r ≈ 0.43 with true AGL. What is broken is *calibration*, not perception: ground biased **+2.53 m** and buildings **−3.30 m**, i.e. the model compresses dynamic range. That is exactly what fine-tuning on metric labels fixes. Full numbers in [PLAN.md §5](PLAN.md).

## Differentiators

1. **Per-pixel uncertainty** — Gaussian NLL + variance head. Calibrated confidence, not false precision. Scores in *both* halves. **Committed.**
2. **Measurement tooling** — the problem statement asks users to *validate* heights. Point-to-point deltas with error bars. **Committed.**
3. **Deployability** — ONNX export, quantised, CPU-only inference. **Committed.**
4. **Shadow prior** — `height = shadow_length × tan(sun_elevation)`. Physics-derived pseudo-labels on unlabelled imagery. *Relocated to the finale phase.* Gate passed: 54 of 67 DFC scenes have usable sun elevation.

---

## Pipeline

```
DFC2019 archives
      │  tools/prepare_data.py   extract → probe → shard
      ▼
data/shards/*.npz               518×518 crops, region-level split
      │  train.py                DA-V2 + twin (μ, log σ²) head
      ▼
checkpoints/best.pt
      │  infer.py                sliding window, cosine blend
      ▼
*.height.tif  *.sigma.tif
      │  tools/export_terrain.py
      ▼
viewer/scenes/<name>/           → Three.js flythrough
```

## Running it

```bash
python -m venv .venv
.venv/Scripts/pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
.venv/Scripts/pip install -r requirements.txt

# 1. data — `probe` MEASURES what the DFC2019 docs do not specify
python tools/prepare_data.py extract path/to/Train-Track1-RGB.zip path/to/Train-Track1-Truth.zip
python tools/prepare_data.py probe
python tools/prepare_data.py shard

# 2. what does doing nothing cost? establish this before training anything
python tools/zero_shot_baseline.py --fit-tiles 40 --eval-tiles 80

# 3. train  (--max-temp governs GPU temperature — see PLAN.md §4)
python train.py --epochs 12 --batch 8 --workers 4 --max-temp 82

# 4. score on whole tiles, not crops
python tools/evaluate.py --ckpt D:/sih2026/checkpoints/run01/best.pt --tiles 80

# 5. predict, export, view
python infer.py --image scene.tif --ckpt D:/sih2026/checkpoints/run01/best.pt --out out/scene
python tools/export_terrain.py --height out/scene.height.tif --sigma out/scene.sigma.tif \
       --texture scene.tif --gsd 0.3 --out viewer/scenes/scene
python -m http.server -d viewer 8080          # → http://localhost:8080
```

**Viewer** — click to look, `WASD` to move, `Q`/`E` down/up, `Shift` to sprint, `Esc` to release. Surfaces: satellite drape, height ramp, uncertainty, slope. *Auto tour* flies a cinematic orbit; *Measure* takes two clicks.

---

## Three things that will bite you

Measured, not guessed. All three fail **silently**.

**DA-V2's patch size is 14 and its head computes `patch_h = H // 14`.** Feed it 256×256 — the size the Depth Any Canopy recipe uses — and you get **252×252** back with no warning. Crops are 518 (= 14 × 37, DA-V2's native resolution); stride 506 tiles a 1024 tile exactly 2×2. `model.check_input_size()` raises rather than let a bad size through.

**The DFC2019 tiles have no georeferencing at all.** They were rewritten by `tifffile`, which stripped CRS and transform — `rasterio` returns the identity matrix. Nothing in the files says a pixel is 30 cm, so GSD must be passed explicitly (`--gsd 0.3`). `export_terrain.py` refuses to invent one and labels distances "pixels" rather than printing confident nonsense.

**Never quote the oracle affine as a result.** It refits scale and shift using the test tile's own ground truth. It bounds what a purely linear correction could achieve; it is not deployable, because it needs the answer to compute the answer.

## Layout

Repo is `D:\sih2026\depthwizard`. Data, weights and the venv live *outside* it.

| Path | |
|---|---|
| `depthwizard/model.py` | DA-V2 backbone + twin mean / log-variance head |
| `depthwizard/losses.py` | Gaussian NLL, β-NLL, gradient matching |
| `depthwizard/metrics.py` | RMSE/MAE per class *and per terrain*, calibration, ECE |
| `depthwizard/dataset.py` | shard loader, two-stage shuffle, D4 + photometric augmentation |
| `train.py` | resumable, bf16/fp16 autodetect, thermal governor |
| `infer.py` | sliding-window inference; variances combine in quadrature |
| `tools/` | data prep, zero-shot baseline, evaluation, terrain export, IMD metadata |
| `viewer/` | Three.js flythrough, vendored for offline demos |
| `notebooks/kaggle_train.py` | Kaggle driver — resumable across session kills |

## Compute

| | Role |
|---|---|
| RTX 3060 12 GB (local) | dev loop, all preprocessing, training, viewer, demo machine |
| Kaggle (30 GPU-h/week) | parallel experiments — sweeps and ablations |
| Azure $100 credit | hosting the standalone demo, **not** training |

Throughput plateaus at **~25 crops/s** from batch 8, so 12 GB is not the binding constraint — **heat is**. Unregulated, the card hits 91 °C and self-throttles to 1492 MHz. `train.py` carries a software thermal governor because `nvidia-smi -pl` needs administrator rights. T4 and P100 have no bf16; training autodetects and falls back to fp16 + GradScaler.

## Licence notes

DA-V2 **Small is Apache-2.0** and is what we ship. Base / Large / Giant are CC-BY-NC-4.0 and cannot go into a deliverable. Three.js is MIT, vendored under `viewer/vendor/`. DFC2019 data is **not** redistributed here — get it from [IEEE DataPort](https://ieee-dataport.org/open-access/data-fusion-contest-2019-dfc2019) (free account). Baselines and metrics: [pubgeo/dfc2019](https://github.com/pubgeo/dfc2019) (MIT).
