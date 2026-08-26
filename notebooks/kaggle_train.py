"""Kaggle training driver. Paste into a Kaggle notebook cell, or run with %run.

Kept as a .py rather than a .ipynb so it stays reviewable in git -- notebook JSON diffs
are unreadable, and this file is the thing that actually has to be correct.

To use on Kaggle
----------------
1. Settings -> Accelerator -> GPU (T4 x2 or P100).
2. Add Data -> your uploaded `depthwizard-shards` dataset.
3. In a cell:

       !git clone https://github.com/zaidnansari2011/sih2026-depthwizard.git /kaggle/working/dw
       %cd /kaggle/working/dw
       !pip install -q transformers safetensors
       %run notebooks/kaggle_train.py

Why the session limit shapes this file
--------------------------------------
Kaggle gives ~9 GPU hours a week and kills a session at 12 hours, sometimes sooner. So
the run is designed to be resumed, not restarted: it writes checkpoints to
/kaggle/working (which persists as notebook output) and picks up from the last one. Each
session does as much as it can inside `--max-hours` and stops cleanly.

Precision: Kaggle's T4 is Turing and its P100 is Pascal, so neither supports bf16 and
train.py falls back to fp16 with a GradScaler. The P100 has no tensor cores at all, so
fp16 there buys memory rather than speed. Do not assume the local timings transfer.
"""
import os
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------- locate the shards
CANDIDATES = [
    Path("/kaggle/input/depthwizard-shards"),
    Path("/kaggle/input/depthwizard-shards/shards"),
    Path("D:/sih2026/data/shards"),                 # local fallback, same script
]
SHARDS = next((p for p in CANDIDATES if p.exists() and any(p.glob("train_*.npz"))), None)
if SHARDS is None:
    found = [str(p) for p in Path("/kaggle/input").glob("*")] if Path("/kaggle/input").exists() else []
    sys.exit(f"no shard directory with train_*.npz found. /kaggle/input holds: {found}")

ON_KAGGLE = Path("/kaggle/working").exists()
OUT = Path("/kaggle/working/checkpoints") if ON_KAGGLE else Path("D:/sih2026/checkpoints")
OUT.mkdir(parents=True, exist_ok=True)

# Keep the HF cache off the small root filesystem.
os.environ.setdefault("HF_HOME", str(Path("/kaggle/working/hf") if ON_KAGGLE else Path("D:/sih2026/.hf")))

print(f"shards      {SHARDS}  ({len(list(SHARDS.glob('train_*.npz')))} train shards)")
print(f"checkpoints {OUT}")

cmd = [
    sys.executable, "train.py",
    "--shards", str(SHARDS),
    "--out", str(OUT),
    "--epochs", "3",
    "--batch", "4",
    "--accum", "2",            # effective batch 8, the Depth Any Canopy setting
    "--workers", "2",
    "--lr", "5e-6",
    "--warmup-mse", "200",
    "--precision", "auto",
    "--resume", "auto",        # the whole point: survive a killed session
    "--max-hours", "8.5",      # stop and checkpoint before Kaggle reclaims the session
    "--val-batches", "60",
]

print("\n" + " ".join(cmd) + "\n", flush=True)
raise SystemExit(subprocess.call(cmd))
