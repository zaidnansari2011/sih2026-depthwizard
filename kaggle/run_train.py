"""Kaggle notebook body for run05 — V1-Large on DFC2019.

Paste this into a Kaggle notebook with GPU enabled and both datasets attached:

    depthwizard-dfc2019-shards   (train + val, 9.65 GB)
    depthwizard-code             (this repo)

Why the settings are what they are, so a resumed session does not quietly differ:

* **fp16, not bf16.** Kaggle hands out T4 (Turing) and P100 (Pascal); neither supports
  bf16. `pick_precision` detects this, but we pass it explicitly so the log is unambiguous.
  Validated locally through the MSE-to-NLL transition, which is where fp16 breaks.
* **batch 2.** V1-Large at 518 crops peaks at 8.87 GB. Batch 4 fits 16 GB but gradient
  checkpointing is 6.7x slower and buys nothing here.
* **`--max-temp 0`.** The thermal governor exists for a 3060 in a desk case that hits 91 C.
  A datacentre card does not need it and the pauses would be pure waste.
* **`--resume auto`.** Kaggle kills a GPU session at 12 hours whatever you do. `/kaggle/working`
  survives between sessions of the same notebook, so re-running this cell continues from
  `last.pt` rather than starting over. At ~82 steps/min a 12-epoch run is 9-10 hours, so
  one session should finish it — but the run must not depend on that being true.
* **HF_HOME on /kaggle/working.** Otherwise the 1.3 GB backbone re-downloads every session.
"""
import os
import subprocess
import sys
from pathlib import Path

CODE_ZIP = "/kaggle/input/depthwizard-code/depthwizard_src.zip"
SHARDS = "/kaggle/input/depthwizard-dfc2019-shards"
WORK = Path("/kaggle/working")
CODE = str(WORK / "code")
OUT = WORK / "checkpoints" / "run05"

os.environ["HF_HOME"] = str(WORK / "hf")          # cache the backbone across sessions
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
OUT.mkdir(parents=True, exist_ok=True)

# --- sanity: fail loudly and immediately rather than 20 minutes in -------------
import torch
print(f"torch {torch.__version__}  cuda {torch.cuda.is_available()}  "
      f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-'}")
print(f"bf16 supported: {torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False}"
      "   (expect False on T4/P100 -> fp16 + GradScaler)")
for p in (CODE_ZIP, SHARDS):
    if not Path(p).exists():
        raise SystemExit(f"missing dataset: {p}. Attach it in the notebook sidebar.")

# The source ships as one archive because the Kaggle client silently skips
# subdirectories on upload -- shipping a tree would have delivered train.py without the
# package it imports. Re-extract each session; /kaggle/input is read-only.
import zipfile
shutil_target = Path(CODE)
if shutil_target.exists():
    import shutil as _sh
    _sh.rmtree(shutil_target)
with zipfile.ZipFile(CODE_ZIP) as z:
    z.extractall(CODE)
print(f"extracted {len(list(Path(CODE).rglob('*.py')))} source files -> {CODE}")
n_train = len(list(Path(SHARDS).glob("train_*.npz")))
n_val = len(list(Path(SHARDS).glob("val_*.npz")))
n_test = len(list(Path(SHARDS).glob("test_*.npz")))
print(f"shards: {n_train} train, {n_val} val, {n_test} test")
if n_test:
    raise SystemExit("test shards are present on the training box. They must not be. "
                     "Standing rule 5.")
if (OUT / "last.pt").exists():
    print(f"resuming from {OUT/'last.pt'}")

cmd = [
    sys.executable, f"{CODE}/train.py",
    "--shards", SHARDS,
    "--out", str(OUT),
    "--model-id", "LiheYoung/depth-anything-large-hf",
    "--precision", "fp16",
    "--batch", "2",
    "--accum", "4",              # effective batch 8, matching run02's recipe
    "--epochs", "12",
    "--lr", "5e-6",
    "--grad-weight", "0.5",
    "--beta", "0.5",             # run02's setting; it is our best model
    "--warmup-mse", "200",
    "--workers", "2",
    "--val-batches", "60",
    "--log-every", "200",
    "--resume", "auto",
    "--max-temp", "0",           # datacentre cooling; governor is for the 3060
    "--seed", "1337",
]
print("\n" + " ".join(cmd) + "\n", flush=True)

env = dict(os.environ, PYTHONPATH=CODE)
proc = subprocess.run(cmd, env=env)
print(f"\nexit code {proc.returncode}")
print("checkpoints in", OUT)
for f in sorted(OUT.glob("*.pt")):
    print(f"   {f.name}  {f.stat().st_size/1e6:.0f} MB")
print("\nIf the session was cut short, re-run this cell -- it resumes from last.pt.")
print("Download best.pt before the session expires; /kaggle/working is not forever.")
