# DepthWizard — SIH26175

Single-view height estimation and 3D flythrough.
ISRO / Space Applications Centre. Idea submission closes **20 September 2026**.

Turn one optical image into a Digital Surface Model, then project the image onto a
navigable 3D terrain mesh.

## Scoring (this drives everything)

| Weight | Criterion |
|--------|-----------|
| 50% | DSM accuracy — RMSE / MAE / correlation, stable across urban, sparse, hilly, forested |
| 50% | Rendering quality & UX — projection accuracy, navigability, standalone deployment |

Half the marks are the front end. Both halves are built in parallel from day one.

## Method

| Stage | What | Owner |
|-------|------|-------|
| 01 | Relative depth — Depth Anything V2 backbone, **fine-tuned** for nadir | CV pair |
| 02 | Metric calibration — regress to absolute metres against SRTM 30 m | CV pair + data |
| 03 | 3D flythrough — mesh displacement + texture projection in Three.js | Full-stack pair |

**Stage 01 is not zero-shot.** Depth Anything V2 degrades badly at nadir — no horizon
cues, and it overestimates tree height from straight down. We fine-tune it on DFC2019
Track 1 AGL labels following the [Depth Any Canopy](https://github.com/DarthReca/depth-any-canopy)
recipe (MSE, 256x256 crops, lr 5e-6, 3 epochs, ~2 GPU-hours).

## Differentiators

1. **Per-pixel uncertainty** — Gaussian NLL loss + variance head. Single-view height is
   ill-posed; we ship calibrated confidence, not false precision. Scores in both halves.
2. **Shadow prior** — `height = shadow_length * tan(sun_elevation)`. Free physics-derived
   pseudo-labels on unlabelled Cartosat imagery, which attacks the domain-gap risk directly.
3. **Measurement tooling** — the PS asks users to *validate* heights. Point-to-point deltas
   with error bars, cross-section profiles, difference maps against a reference DSM.
4. **Deployability** — ONNX export, quantised, CPU-only inference, air-gapped Docker image.

## Layout

Repo is `D:\sih2026\depthwizard`. Data, weights and the venv live *outside* it:

```
D:\sih2026\
├── .venv\              python 3.10 environment
├── data\raw\           DFC2019 zips from IEEE DataPort
│      \extracted\
│      \shards\         preprocessed crops -> uploaded once as a Kaggle Dataset
├── checkpoints\
├── docs\               problem statement, team brief
└── depthwizard\        <- this repo
```

## Compute

| | Role |
|---|---|
| RTX 3060 12GB (local) | dev loop, debugging, all preprocessing, viewer, demo machine |
| Kaggle (30 GPU-h/week) | experiment farm — benchmark sweeps, long unattended runs |
| Azure $100 credit | **hosting the standalone demo**, not training |

T4 and P100 have no bf16. Training code autodetects and falls back to fp16 + GradScaler.

## Data

DFC2019 Track 1 — [IEEE DataPort](https://ieee-dataport.org/open-access/data-fusion-contest-2019-dfc2019), free account.

- `Train-Track1-RGB-1.zip` (6.59 GB)
- `Train-Track1-Truth.zip` (8.17 GB) — AGL height ground truth
- `Validate-Track1.zip` (683 MB)

Skip the MSI files (30 GB, unused). Baselines and metrics: [pubgeo/dfc2019](https://github.com/pubgeo/dfc2019) (MIT).

## Setup

```bash
D:\sih2026\.venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
python tools/thermal_soak.py --minutes 20    # sanity-check GPU thermals
```
