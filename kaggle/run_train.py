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

WORK = Path("/kaggle/working")
CODE = str(WORK / "code")
OUT = WORK / "checkpoints" / "run05"

# Find the inputs rather than hardcoding their mount points. Kaggle derives the directory
# from the dataset slug, and a slug that differs by one character would fail the run after
# the queue wait rather than now. Searching costs nothing and removes the whole class of
# "it didn't work" round-trips.
INPUT = Path("/kaggle/input")
_present = sorted(str(p.relative_to(INPUT)) for p in INPUT.rglob("*") if p.is_dir())[:20]

# Search RECURSIVELY. Kaggle does not guarantee the mount depth: the first attempt assumed
# /kaggle/input/<slug>/ and the datasets actually landed under /kaggle/input/datasets/...,
# which failed with "Present: ['datasets']". Depth is an implementation detail of theirs,
# so stop predicting it and just look.
#
# The source may arrive either way and both are normal: Kaggle unpacks an uploaded .zip
# into a real tree, but a dataset created another way can still hold the archive.
_trees = sorted({p.parent for p in INPUT.rglob("train.py")})
_zips = sorted(INPUT.rglob("depthwizard_src.zip"))
_shard_dirs = sorted({p.parent for p in INPUT.rglob("train_*.npz")})

if not _shard_dirs:
    raise SystemExit(f"no train_*.npz anywhere under {INPUT}. Attach "
                     f"'depthwizard-dfc2019-shards'. Directories present: {_present}")
SHARDS = str(_shard_dirs[0])

if _trees:
    CODE = str(_trees[0])            # already a usable tree; nothing to extract
    CODE_ZIP = None
elif _zips:
    CODE_ZIP = str(_zips[0])
else:
    raise SystemExit(f"no train.py and no depthwizard_src.zip anywhere under "
                     f"{INPUT}. Attach 'depthwizard-code'. Directories present: {_present}")
print(f"code   -> {CODE if CODE_ZIP is None else CODE_ZIP}")
print(f"shards -> {SHARDS}")

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
if CODE_ZIP is not None:
    # /kaggle/input is read-only, so unpack into working space each session.
    import shutil as _sh
    import zipfile
    if Path(CODE).exists():
        _sh.rmtree(CODE)
    with zipfile.ZipFile(CODE_ZIP) as z:
        z.extractall(CODE)
    print(f"extracted {len(list(Path(CODE).rglob('*.py')))} source files -> {CODE}")
else:
    print(f"using source tree in place ({len(list(Path(CODE).rglob('*.py')))} .py files)")

for need in ("train.py", "depthwizard/model.py", "depthwizard/dataset.py"):
    if not (Path(CODE) / need).exists():
        raise SystemExit(f"source is incomplete: {need} missing under {CODE}. "
                         f"Found: {sorted(p.name for p in Path(CODE).glob('*'))}")
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
