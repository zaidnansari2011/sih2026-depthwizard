"""Fine-tune DepthWizard on DFC2019 height labels.

    python train.py --shards D:/sih2026/data/shards --epochs 3 --batch 8

Follows the Depth Any Canopy recipe, which is the only published adaptation of
Depth Anything to aerial height that reports its hyperparameters: AdamW, backbone
lr 5e-6, 5% linear warmup then linear decay, a handful of epochs. We differ in two
ways that PLAN.md section 6.1 argues for -- a Gaussian NLL objective instead of MSE,
and a separate, larger learning rate for the head.

Built for interruption
----------------------
Kaggle sessions are capped at 12 hours and can be reclaimed sooner. Everything here
assumes the process may die at any moment: state is checkpointed every epoch and on a
wall-clock budget, `--resume` restores model, optimiser, scheduler and RNG, and the
metrics log is append-only JSONL so a partial run still plots.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from depthwizard.dataset import Augment, HeightShardDataset, ShardStream
from depthwizard.losses import CompositeLoss
from depthwizard.metrics import height_metrics, calibration_curve, \
    expected_calibration_error, uncertainty_error_correlation
from depthwizard.model import build, check_input_size, DEFAULT_MODEL


def pick_precision(requested: str):
    """bf16 where the hardware has it, fp16 where it does not, fp32 on CPU.

    This matters because the two training targets differ: the local 3060 is Ampere and
    has bf16, while Kaggle hands out T4 (Turing) and P100 (Pascal), which do not. bf16
    needs no loss scaling; fp16 does, and a P100 has no tensor cores at all so fp16 buys
    memory rather than speed. Getting this wrong shows up as silent NaNs, so we detect
    rather than assume.

    **Do not use `torch.cuda.is_bf16_supported()` here.** It counts *emulated* bf16 and so
    returns True on hardware that has none: run06's Kaggle log printed "bf16 supported:
    True" on a Tesla P100, which has no bf16 at all. On a T4 it would likewise pick bf16
    and get a slow software path instead of Turing's fp16 tensor cores -- the exact
    opposite of what this function's first line promises. Compute capability is the
    unambiguous test: bf16 is native from Ampere (8.0) onward, so 3060 yes, T4 (7.5) no,
    P100 (6.0) no.
    """
    if not torch.cuda.is_available():
        return torch.float32, False, "cpu/fp32"
    if requested == "fp32":
        return torch.float32, False, "fp32 (forced)"
    native_bf16 = torch.cuda.get_device_capability()[0] >= 8
    if requested in ("auto", "bf16") and native_bf16:
        return torch.bfloat16, False, "bf16"
    if requested == "bf16":
        print("  ! bf16 requested but unsupported on this GPU; falling back to fp16")
    return torch.float16, True, "fp16 + GradScaler"


class ThermalGovernor:
    """Keep the GPU below a temperature by inserting short pauses.

    Measured on this 3060: an unconstrained run reaches 91 C within ten minutes and the
    card down-clocks itself from 1875 to 1492 MHz. At that point it is already throttling,
    so the "full speed" run is not actually faster -- it is just hotter. nvidia-smi -pl
    would be the clean fix but setting a power limit needs administrator rights on
    Windows, so we regulate from user space instead.

    A proportional controller: sample every `check_every` optimiser steps, and if we are
    over target, sleep in proportion to the overshoot. Sleeping between steps lets the
    cooler catch up while costing only the sleep itself.

    Set --max-temp 0 to disable.
    """

    def __init__(self, max_temp: float = 80.0, check_every: int = 10, verbose: bool = True,
                 gain: float = 0.8, hard_ceiling: float | None = None):
        self.max_temp = float(max_temp)
        self.check_every = max(1, int(check_every))
        self.verbose = verbose
        self.gain = float(gain)
        self.hard_ceiling = float(hard_ceiling if hard_ceiling is not None else max_temp + 6)
        self.enabled = max_temp > 0 and self._read() is not None
        self.slept = 0.0
        self.peak = 0.0
        self.cooldowns = 0
        self.samples = []
        if max_temp > 0 and not self.enabled:
            print("  ! thermal governor: nvidia-smi unavailable, running unregulated")

    @staticmethod
    def _read():
        import subprocess
        try:
            o = subprocess.run(
                ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5,
            )
            return float(o.stdout.strip().splitlines()[0])
        except Exception:
            return None

    def step(self, i: int):
        if not self.enabled or i % self.check_every:
            return
        t = self._read()
        if t is None:
            return
        self.peak = max(self.peak, t)
        self.samples.append(t)
        over = t - self.max_temp
        if over <= 0:
            return

        # Hard ceiling. Proportional control alone stabilises a few degrees above target
        # -- the card heats faster than short pauses shed -- so past a threshold we stop
        # feeding it work until it has actually come down. This is the guard that makes
        # a 91 C excursion impossible rather than merely unlikely.
        if t >= self.hard_ceiling:
            t0 = time.time()
            while True:
                time.sleep(3.0)
                cur = self._read()
                if cur is None or cur <= self.max_temp or (time.time() - t0) > 120:
                    break
            waited = time.time() - t0
            self.slept += waited
            self.cooldowns += 1
            if self.verbose:
                print(f"      [thermal] {t:.0f} C hit the {self.hard_ceiling:.0f} C ceiling, "
                      f"held {waited:.0f}s until {cur if cur else '?'} C")
            return

        pause = min(15.0, over * self.gain)
        time.sleep(pause)
        self.slept += pause
        if self.verbose and over > 3:
            print(f"      [thermal] {t:.0f} C > {self.max_temp:.0f} C, paused {pause:.1f}s")

    def summary(self):
        if not self.enabled or not self.samples:
            return "thermal governor: off"
        a = np.array(self.samples)
        return (f"thermal: mean {a.mean():.1f} C, peak {self.peak:.0f} C, "
                f"{self.cooldowns} hard cool-downs, paused {self.slept/60:.1f} min total")


def _ram() -> str:
    """Report host memory. The first long run died to an OOM with no traceback, so the
    number that made it possible belongs in the run header."""
    try:
        import ctypes

        class S(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        st = S()
        st.dwLength = ctypes.sizeof(S)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))
        return f"{st.ullTotalPhys/2**30:.1f} GB total, {st.ullAvailPhys/2**30:.1f} GB free"
    except Exception:
        return "unknown"


def lr_lambda_factory(total_steps: int, warmup_frac: float = 0.05):
    warm = max(1, int(total_steps * warmup_frac))

    def fn(step):
        if step < warm:
            return step / warm
        return max(0.0, (total_steps - step) / max(1, total_steps - warm))

    return fn


@torch.no_grad()
def evaluate(model, loader, device, amp_dtype, max_batches=0, height_scale=1.0,
             max_pixels: int = 20_000_000):
    """Full evaluation: the accuracy numbers ISRO scores, plus calibration of sigma.

    Subsampling is not optional here. At 518x518, 80 batches of 8 is ~171 M valid pixels;
    holding those as float32 predictions, targets and sigmas plus an int64 class map is
    about 3.4 GB, and np.concatenate briefly doubles it. On a 16 GB machine already
    running four dataloader workers, that is an out-of-memory kill with no traceback --
    which is exactly how the first 12-epoch run died, silently, entering its first
    validation.

    So: thin each batch to fit a pixel budget, and keep classes as uint8 rather than the
    int64 the collate produces. Metrics over 20 M pixels are indistinguishable from
    metrics over 171 M, and they fit.
    """
    model.eval()
    P, T, S, C = [], [], [], []
    n_batches = max_batches if max_batches else len(getattr(loader, "dataset", []) or [1])
    per_batch = max(1, max_pixels // max(1, n_batches))
    for i, b in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        rgb = b["rgb"].to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype != torch.float32):
            mu, log_var = model(rgb)
        mu = mu.float().cpu()
        m = b["mask"].bool()
        n_valid = int(m.sum())
        stride = max(1, n_valid // per_batch)

        P.append(mu[m].numpy()[::stride].astype(np.float32))
        T.append(b["agl"][m].numpy()[::stride].astype(np.float32))
        if log_var is not None:
            S.append(torch.exp(0.5 * log_var.float().cpu())[m].numpy()[::stride].astype(np.float32))
        if "cls" in b:
            C.append(b["cls"][m].numpy()[::stride].astype(np.uint8))
        del mu, log_var, rgb
    model.train()
    if not P:
        return {}

    p, t = np.concatenate(P), np.concatenate(T)
    cls = np.concatenate(C) if C else None
    hm = height_metrics(p, t, cls=cls)
    out = hm.to_dict()
    if S:
        s = np.concatenate(S)
        out["ece"] = expected_calibration_error(p, t, s)
        out["sigma_rank_corr"] = uncertainty_error_correlation(p, t, s)
        out["sigma_mean"] = float(s.mean())
        out["calibration"] = [{"k": k, "expected": e, "observed": o}
                              for k, e, o in calibration_curve(p, t, s)]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shards", default="D:/sih2026/data/shards")
    ap.add_argument("--model-id", default=DEFAULT_MODEL,
                    help="backbone. Default is Depth-Anything-V2-Small (Apache-2.0, "
                         "24.8 M). V2's Base and Large are CC-BY-NC so they cannot ship; "
                         "V1 is Apache-2.0 at every size. Measured zero-shot correlation "
                         "with per-building height on 40 val tiles: V2-Small +0.351, "
                         "LiheYoung/depth-anything-base-hf +0.470, "
                         "LiheYoung/depth-anything-large-hf +0.659.")
    ap.add_argument("--out", default="D:/sih2026/checkpoints")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--accum", type=int, default=1, help="gradient accumulation steps")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--lr", type=float, default=5e-6, help="backbone lr (Depth Any Canopy)")
    ap.add_argument("--lr-head", type=float, default=None, help="default: 50x backbone")
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--height-scale", type=float, default=None,
                    help="metres per normalised unit; default reads probe.json")
    ap.add_argument("--init-sigma", type=float, default=5.0)
    ap.add_argument("--grad-weight", type=float, default=0.5, help="gradient-matching weight")
    ap.add_argument("--warmup-mse", type=int, default=200, help="steps of plain MSE first")
    ap.add_argument("--beta", type=float, default=0.0, help="beta-NLL (Seitzer et al.)")
    ap.add_argument("--log-var-max", type=float, default=7.0,
                    help="clamp on log predictive variance; sigma rails at exp(this/2)")
    ap.add_argument("--init-mu", default="pretrained", choices=["pretrained", "constant"],
                    help="'constant' re-inits the mean readout; required for DA-V1 "
                         "backbones, whose pretrained head starts ~666 m off (preflight)")
    ap.add_argument("--init-mu-m", type=float, default=0.0,
                    help="height the constant mean init predicts everywhere, in metres")
    # -------------------------------------------------------------------- tripwires
    ap.add_argument("--no-tripwires", action="store_true",
                    help="disable the health aborts. run05 had no tripwires and spent "
                         "543.6 min producing nothing.")
    ap.add_argument("--rail-patience", type=int, default=50,
                    help="steps of railed sigma tolerated before aborting")
    ap.add_argument("--gate-rmse", type=float, default=25.0,
                    help="abort if epoch-0 val RMSE exceeds this (run04: 8.239 m)")
    ap.add_argument("--gate-corr", type=float, default=0.20,
                    help="abort if epoch-0 val correlation is below this (run04: +0.660)")
    ap.add_argument("--precision", default="auto", choices=["auto", "bf16", "fp16", "fp32"])
    ap.add_argument("--clip", type=float, default=1.0, help="grad-norm clip; 0 disables")
    ap.add_argument("--buffer-shards", type=int, default=2)
    ap.add_argument("--max-steps", type=int, default=0, help="stop early; 0 = full run")
    ap.add_argument("--max-hours", type=float, default=0.0,
                    help="checkpoint and stop before a session limit; 0 = no budget")
    ap.add_argument("--val-batches", type=int, default=40, help="0 = whole val split")
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--resume", default=None, help="checkpoint path, or 'auto'. "
                    "Continues the SAME run: restores optimiser, scheduler, epoch and "
                    "step counters. To start a new run from trained weights, you want "
                    "--init-from instead.")
    ap.add_argument("--init-from", default=None, metavar="CKPT",
                    help="warm start a NEW run from a checkpoint's weights only. "
                         "Optimiser, scheduler, epoch counter, step counter and best "
                         "score all stay at their initial values.\n"
                         "This exists because --resume cannot do it. --resume restores "
                         "epoch+1 (12 for run02), so `range(start_epoch, args.epochs)` "
                         "is empty for any --epochs <= 12: the run trains nothing, "
                         "writes no checkpoint, prints the RESUMED checkpoint's best "
                         "RMSE as its own result, and exits 0. Raising --epochs does not "
                         "rescue it either, because the restored scheduler is already "
                         "past this run's total_steps and lr_lambda then returns exactly "
                         "0.0. Both traps found by adversarial audit, 6 Sep 2026.")
    ap.add_argument("--freeze-backbone", action="store_true")
    ap.add_argument("--no-uncertainty", action="store_true",
                    help="ablation: train a plain regressor, the 6.1 baseline")
    ap.add_argument("--checkpointing", action="store_true", help="gradient checkpointing")
    ap.add_argument("--max-temp", type=float, default=80.0,
                    help="pause briefly above this GPU temperature; 0 disables. "
                         "Unregulated, this 3060 hits 91 C and self-throttles.")
    ap.add_argument("--temp-check-every", type=int, default=10)
    ap.add_argument("--temp-gain", type=float, default=0.8,
                    help="seconds of pause per degree over target")
    ap.add_argument("--temp-ceiling", type=float, default=None,
                    help="hard stop-and-cool threshold; default max-temp + 6")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--bins", type=int, default=0,
                    help="0 = direct regression. >0 switches to the adaptive-bin "
                         "classification head with soft-argmax")
    ap.add_argument("--bin-min", type=float, default=-3.0)
    ap.add_argument("--bin-max", type=float, default=120.0,
                    help="bins span [bin-min, bin-max] and soft-argmax cannot predict "
                         "outside it. Measured: p99.9 is 44 m and the max is 204 m, so "
                         "120 m clears p99.9 by 2.7x without spending most of the "
                         "resolution on one skyscraper")
    ap.add_argument("--chamfer", type=float, default=0.01,
                    help="bi-directional Chamfer on bin EDGES. HTC-DC Net uses "
                         "mu1 = 0.01; run02-era code used AdaBins 0.1, ten times too "
                         "hot. Ignored without --bins")
    ap.add_argument("--dist-weight", type=float, default=1.0,
                    help="distribution-based constraint, KL against a reference whose "
                         "width is solved from the model own confidence (HTC-DC Net "
                         "mu3 = 1). Ignored without --bins")
    ap.add_argument("--htc-weight", type=float, default=1.0,
                    help="head-tail cut cross-entropy (HTC-DC Net mu2 = 1)")
    ap.add_argument("--no-htc", action="store_true",
                    help="single shared bin head, i.e. what run03 did")
    ap.add_argument("--fg-threshold", type=float, default=1.0,
                    help="metres; the head-tail split")
    ap.add_argument("--htc-warmup", type=int, default=500,
                    help="steps routing by ground truth before handing over to the "
                         "learned gate; stops one bin head starving at init")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    shard_dir = Path(args.shards)

    # ---------------------------------------------------------------- height scale
    height_scale = args.height_scale
    probe_path = shard_dir / "probe.json"
    if height_scale is None and probe_path.exists():
        pr = json.loads(probe_path.read_text())
        # The 95th percentile if the probe recorded one, else a fraction of the max.
        # Deliberately not the maximum: that is one tall building, and scaling to it
        # would compress every ordinary rooftop into the bottom few percent of range.
        height_scale = pr.get("agl_p95") or (pr.get("agl_max") or 60.0) * 0.5
    height_scale = float(height_scale or 30.0)

    # ---------------------------------------------------------------- data
    aug = Augment()
    train_ds = ShardStream(shard_dir, "train", augment=aug,
                           buffer_shards=args.buffer_shards, seed=args.seed)
    # cache_shards=1: validation sweeps sequentially, so a single cached shard already
    # gives a ~100% hit rate, and each 518x518 shard is ~200 MB decompressed. Caching two
    # buys nothing and competes for RAM with the training workers, which stay alive
    # through validation because persistent_workers is on.
    val_ds = HeightShardDataset(shard_dir, "val", augment=None, return_cls=True,
                                cache_shards=1)
    train_ld = DataLoader(train_ds, batch_size=args.batch, num_workers=args.workers,
                          pin_memory=(device == "cuda"), drop_last=True,
                          persistent_workers=args.workers > 0)
    val_ld = DataLoader(val_ds, batch_size=args.batch, num_workers=0, pin_memory=(device == "cuda"))

    # Row counts come from the shard headers, so the schedule is exact rather than an
    # estimate -- an IterableDataset has no __len__ and a wrong total would misshape
    # the warmup/decay curve.
    n_train = sum(
        int(np.load(f, allow_pickle=False)["agl"].shape[0]) for f in train_ds.index.files
    )
    steps_per_epoch = max(1, n_train // (args.batch * args.accum))
    total_steps = args.max_steps or steps_per_epoch * args.epochs

    # ---------------------------------------------------------------- model
    model = build(model_id=args.model_id,
                  height_scale=height_scale, init_sigma_m=args.init_sigma,
                  freeze_backbone=args.freeze_backbone,
                  predict_uncertainty=not args.no_uncertainty,
                  bins=args.bins, bin_min=args.bin_min,
                  bin_max=args.bin_max, htc=not args.no_htc,
                  init_mu=args.init_mu, init_mu_m=args.init_mu_m).to(device)
    if args.checkpointing:
        model.enable_gradient_checkpointing()

    amp_dtype, needs_scaler, prec_name = pick_precision(args.precision)
    scaler = torch.amp.GradScaler("cuda", enabled=needs_scaler)

    opt = torch.optim.AdamW(
        model.param_groups(args.lr, args.lr_head, args.weight_decay), betas=(0.9, 0.999)
    )
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda_factory(total_steps))
    head_desc = (f"binned soft-argmax, N={args.bins}, range "
                 f"[{args.bin_min:.0f}, {args.bin_max:.0f}] m"
                 + ("" if args.no_htc else f", head-tail cut at {args.fg_threshold:.0f} m")
                 if args.bins else "direct regression")
    loss_fn = CompositeLoss(grad_weight=args.grad_weight, beta=args.beta,
                            warmup_mse=args.warmup_mse,
                            log_var_max=args.log_var_max,
                            chamfer_weight=args.chamfer if args.bins else 0.0,
                            dist_weight=args.dist_weight if args.bins else 0.0,
                            htc_weight=(args.htc_weight if args.bins
                                        and not args.no_htc else 0.0),
                            fg_threshold=args.fg_threshold).to(device)

    start_epoch, gstep, best = 0, 0, float("inf")
    ckpt_last = out_dir / "last.pt"
    resume = ckpt_last if args.resume == "auto" and ckpt_last.exists() else \
        (Path(args.resume) if args.resume and args.resume != "auto" else None)
    if args.init_from and resume:
        raise SystemExit(
            "--init-from and --resume are mutually exclusive. --init-from starts a new "
            "run from trained weights; --resume continues an existing one. Note that "
            "--resume auto silently picks up out/last.pt if it exists, so pass a fresh "
            "--out directory alongside --init-from.")
    if args.init_from:
        st = torch.load(args.init_from, map_location=device, weights_only=False)
        # Weights ONLY. Optimiser, scheduler, epoch, gstep and best stay at their
        # initial values, which is the whole point: this is a new run, and inheriting
        # any of that state is what makes --resume unusable for a warm start.
        model.load_state_dict(st["model"])
        # The head emits mu = conv_mu(x) * height_scale. Loading weights calibrated for
        # one scale into a model built with another multiplies every prediction at step
        # 0, and a large initial residual rails log_var at its clamp and deadlocks the
        # mean head -- the mechanism that voided run05 (9 GPU-hours, zero information).
        prev = st.get("height_scale")
        if prev is not None and abs(float(prev) - height_scale) > 1e-6:
            raise SystemExit(
                f"height_scale mismatch: the checkpoint was trained with {prev!r} but "
                f"this run resolved {height_scale!r} from {probe_path}. Every prediction "
                f"would be scaled by {height_scale / float(prev):.3f}x at step 0. Pass "
                f"--height-scale {prev!r} to keep the head calibrated, or start from "
                f"scratch deliberately.")
        print(f"warm start from {args.init_from}: weights only "
              f"(epoch/step/optimiser/scheduler reset), height_scale {height_scale:.6f}")
    elif resume and resume.exists():
        st = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        if st.get("scaler") and needs_scaler:
            scaler.load_state_dict(st["scaler"])
        start_epoch, gstep, best = st["epoch"] + 1, st["gstep"], st.get("best", float("inf"))
        torch.set_rng_state(st["rng"].cpu() if torch.is_tensor(st["rng"]) else st["rng"])
        print(f"resumed {resume} at epoch {start_epoch}, step {gstep}, best RMSE {best:.3f}")
        # Without this, range(start_epoch, args.epochs) is empty, the epoch loop body
        # never executes, no checkpoint is written, and the run prints the RESUMED
        # checkpoint's best RMSE as its own result with exit status 0.
        if start_epoch >= args.epochs:
            raise SystemExit(
                f"nothing to do: {resume} is already at epoch {st['epoch']} and "
                f"--epochs is {args.epochs}, so the epoch range is empty and this run "
                f"would train zero batches, write no checkpoint, and still print "
                f"'best val RMSE {best:.3f}' from the checkpoint you resumed. Raise "
                f"--epochs above {start_epoch} to continue this run, or use "
                f"--init-from {resume} to start a NEW run from these weights.")

    tot, tr = model.n_params()
    print(f"""
DepthWizard training
  device        {device}  {torch.cuda.get_device_name(0) if device == 'cuda' else ''}
  precision     {prec_name}
  params        {tot/1e6:.1f} M total, {tr/1e6:.1f} M trainable
  height scale  {height_scale:.1f} m per normalised unit
  uncertainty   {'off (ablation)' if args.no_uncertainty else 'on'}
  train rows    {n_train:,}  ->  {steps_per_epoch:,} steps/epoch
  total steps   {total_steps:,}  ({args.epochs} epochs, accum {args.accum})
  head          {head_desc}
  htc warmup    {args.htc_warmup if args.bins and not args.no_htc else 0} steps routed by ground truth
  objective     NLL + {args.grad_weight} * gradient-matching, {args.warmup_mse} MSE warmup steps
  host RAM      {_ram()}
""".rstrip())

    log_path = out_dir / "train_log.jsonl"
    log = log_path.open("a", encoding="utf-8")

    def write_log(rec):
        log.write(json.dumps(rec) + "\n")
        log.flush()

    gov = ThermalGovernor(args.max_temp, args.temp_check_every,
                          gain=args.temp_gain, hard_ceiling=args.temp_ceiling)
    if gov.enabled:
        print(f"  thermal      governor on, target {args.max_temp:.0f} C, "
              f"hard ceiling {gov.hard_ceiling:.0f} C")

    t0 = time.time()
    stop = False
    model.train()

    # A "railed" sigma is the fingerprint of the NLL deadlock that killed run05: log_var
    # pinned at log_var_max, its own gradient zeroed by clamp() and mu's suppressed ~1100x.
    sigma_rail = math.exp(args.log_var_max / 2) * 0.99
    n_railed = 0
    if not args.no_tripwires:
        print(f"  tripwires    on: abort on non-finite loss, on sigma >= "
              f"{sigma_rail:.2f} m for {args.rail_patience} steps, and on epoch-"
              f"{start_epoch} val worse than {args.gate_rmse} m / r {args.gate_corr}")

    for epoch in range(start_epoch, args.epochs):
        train_ds.set_epoch(epoch)
        run, seen, t_ep = {}, 0, time.time()
        opt.zero_grad(set_to_none=True)

        for i, b in enumerate(train_ld):
            rgb = b["rgb"].to(device, non_blocking=True)
            agl = b["agl"].to(device, non_blocking=True)
            mask = b["mask"].to(device, non_blocking=True)
            check_input_size(rgb.shape[-2], rgb.shape[-1], model.patch_size)

            with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype != torch.float32):
                if args.bins:
                    route = ((agl > args.fg_threshold)
                             if (not args.no_htc and gstep < args.htc_warmup)
                             else None)
                    mu, log_var, aux = model(rgb, return_bins=True, route_mask=route)
                    aux = {k: (v.float() if torch.is_tensor(v) else v)
                           for k, v in aux.items()}
                else:
                    mu, log_var = model(rgb)
                    aux = None
                if log_var is None:                       # ablation path
                    log_var = torch.zeros_like(mu)
                loss, stats = loss_fn(mu.float(), log_var.float(), agl, mask, aux=aux)

            # ------------------------------------------------------------- tripwires
            # run05 went NaN at step 6066 of 11532 and kept going for another 4.2 hours,
            # because nothing was watching. These abort instead. Every threshold is
            # measured against logs/run04.log, the healthy 12-epoch reference.
            if not args.no_tripwires:
                why = None
                if not math.isfinite(stats["loss"]):
                    why = (f"loss is {stats['loss']} at step {gstep}. run05 hit this at "
                           f"6066/11532 and burned 4.2 h more producing nothing.")
                elif stats.get("sigma_mean", 0) >= sigma_rail:
                    # clamp() zeroes log_var's gradient at the rail and suppresses mu's by
                    # ~1100x (tools/nll_deadlock_probe.py). Nothing recovers from here.
                    n_railed += 1
                    if n_railed >= args.rail_patience:
                        why = (f"sigma has sat at the clamp ({stats['sigma_mean']:.3f} m "
                               f">= {sigma_rail:.3f}) for {n_railed} steps. The variance "
                               f"head is dead; this is exactly how run05 failed.")
                else:
                    n_railed = 0
                if why:
                    print(f"\n  TRIPWIRE: {why}\n  Aborting. Nothing after this point "
                          f"would have been usable.")
                    write_log({"t": "tripwire", "epoch": epoch, "step": gstep, "why": why})
                    return 2

            loss = loss / args.accum

            if needs_scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (i + 1) % args.accum == 0:
                if args.clip:
                    if needs_scaler:
                        scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                if needs_scaler:
                    scaler.step(opt)
                    scaler.update()
                else:
                    opt.step()
                opt.zero_grad(set_to_none=True)
                sched.step()
                gstep += 1
                gov.step(gstep)

            for k, v in stats.items():
                run[k] = run.get(k, 0.0) + v
            seen += 1

            if seen % args.log_every == 0:
                avg = {k: v / seen for k, v in run.items()}
                el = time.time() - t0
                print(f"  e{epoch} s{gstep:6d}/{total_steps}  "
                      f"loss {avg['loss']:8.4f}  rmse {avg['rmse']:7.3f} m  "
                      f"sigma {avg.get('sigma_mean', 0):6.3f} m  "
                      f"lr {sched.get_last_lr()[0]:.2e}  {el/60:.1f} min"
                      + (f"  {gov.peak:.0f}C" if gov.enabled else ""))
                write_log({"t": "train", "epoch": epoch, "step": gstep,
                           "elapsed_s": round(el, 1), **{k: round(v, 5) for k, v in avg.items()}})
                run, seen = {}, 0

            if args.max_steps and gstep >= args.max_steps:
                stop = True
            if args.max_hours and (time.time() - t0) > args.max_hours * 3600:
                print(f"  wall-clock budget of {args.max_hours} h reached -- checkpointing")
                stop = True
            if stop:
                break

        # ------------------------------------------------------------- validate
        val = evaluate(model, val_ld, device, amp_dtype, args.val_batches, height_scale)
        if val:
            line = (f"[epoch {epoch}] val RMSE {val['rmse']:.3f} m  MAE {val['mae']:.3f} m  "
                    f"r {val['corr']:.3f}  bias {val['bias']:+.3f} m")
            if "ece" in val:
                line += f"  ECE {val['ece']:.4f}  sigma-rank-r {val['sigma_rank_corr']:+.3f}"
            print(line)
            for name, m in val.get("per_class", {}).items():
                print(f"      {name:12s} RMSE {m['rmse']:7.3f}  MAE {m['mae']:7.3f}  n={m['n']:,}")
            write_log({"t": "val", "epoch": epoch, "step": gstep,
                       "elapsed_s": round(time.time() - t0, 1), **val})

            # The epoch-0 gate. run04, healthy, ended epoch 0 at RMSE 8.239 m / r 0.660.
            # run05 ended it at 70.8 m / r -0.046 and then ran eight more hours. One
            # epoch is enough to tell those apart, so decide here rather than at the end.
            if epoch == start_epoch and not args.no_tripwires:
                bad = []
                if val["rmse"] > args.gate_rmse:
                    bad.append(f"RMSE {val['rmse']:.3f} m > {args.gate_rmse} m "
                               f"(run04 reached 8.239 m; run05 sat at 70.8 m)")
                if val["corr"] < args.gate_corr:
                    bad.append(f"r {val['corr']:+.3f} < {args.gate_corr} "
                               f"(run04 reached +0.660; run05 was -0.046)")
                if bad:
                    print(f"\n  TRIPWIRE: epoch {epoch} is not on a healthy trajectory.")
                    for s in bad:
                        print(f"    - {s}")
                    print("  Aborting rather than spending the remaining "
                          f"{args.epochs - epoch - 1} epochs on it. Override with "
                          "--no-tripwires if this is expected.")
                    write_log({"t": "tripwire", "epoch": epoch, "step": gstep,
                               "why": "; ".join(bad)})
                    log.close()
                    return 2

        state = {
            "model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
            "scaler": scaler.state_dict() if needs_scaler else None,
            "epoch": epoch, "gstep": gstep, "best": best, "rng": torch.get_rng_state(),
            "args": vars(args), "height_scale": height_scale, "val": val,
            "model_id": model.model_id, "model_config": model.config(),
        }
        torch.save(state, ckpt_last)
        if val and val.get("rmse", float("inf")) < best:
            best = val["rmse"]
            state["best"] = best
            torch.save(state, out_dir / "best.pt")
            print(f"      new best: {best:.3f} m -> {out_dir/'best.pt'}")
        print(f"      epoch {epoch} took {(time.time()-t_ep)/60:.1f} min")

        if stop:
            break

    log.close()
    print(f"\ndone in {(time.time()-t0)/60:.1f} min. best val RMSE {best:.3f} m")
    print(f"  {gov.summary()}")
    print(f"checkpoints: {ckpt_last}" + (f" and {out_dir/'best.pt'}" if best < float('inf') else ""))


if __name__ == "__main__":
    raise SystemExit(main() or 0)
