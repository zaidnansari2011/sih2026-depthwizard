"""Refuse to start a long run that is already broken.

run05 spent 543.6 minutes producing nothing. Its log shows the failure was visible in the
FIRST logged line: sigma 33.115 m = exp(7/2), the variance head pinned against
log_var_max = 7.0. tools/nll_deadlock_probe.py measures what that costs -- the gradient
reaching the mean head drops to 0.0009x of healthy, and the clamp zeroes log_var's own
gradient, so neither head can recover. Nine hours of arithmetic on a dead model.

This runs the REAL config -- same model, same loss, same optimiser, same shards -- for a
couple of hundred steps and checks the invariants that run05 broke. It costs a few minutes
and it either says GO or names the reason it will not.

    python tools/preflight.py --model-id LiheYoung/depth-anything-large-hf --steps 60

Exit code 0 = safe to launch. Non-zero = do not launch; the reason is printed.
"""
from __future__ import annotations

import argparse
import math
import sys
import time

import numpy as np
import torch

# Same shim as tools/evaluate.py: run from a clean clone this file is executed as a
# script, so the repo root is not on the path and `import depthwizard` fails before
# argparse ever gets a chance to print --help. Found by tools/smoke_deliverable.py.
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from depthwizard.dataset import Augment, ShardStream
from depthwizard.losses import CompositeLoss
from depthwizard.model import DEFAULT_MODEL, build

SHARDS = "D:/sih2026/data/shards"

# run04 is the healthy reference: a 12-epoch run that trained cleanly to per-building
# RMSE 3.991 m. These are its measured first-100-step values, from logs/run04.log.
HEALTHY = {"sigma_s100": 5.000, "rmse_s100": 4.280, "loss_s100": 25.3743}


class Check:
    def __init__(self):
        self.rows = []

    def add(self, name, ok, detail, fatal=True):
        self.rows.append((name, bool(ok), detail, fatal))
        return ok

    def report(self):
        print("\n" + "=" * 78)
        width = max(len(r[0]) for r in self.rows)
        bad = 0
        for name, ok, detail, fatal in self.rows:
            tag = "PASS" if ok else ("FAIL" if fatal else "WARN")
            print(f"  [{tag}] {name:<{width}}  {detail}")
            bad += int(not ok and fatal)
        print("=" * 78)
        return bad


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default=DEFAULT_MODEL)
    ap.add_argument("--shards", default=SHARDS)
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--lr-head", type=float, default=None)
    ap.add_argument("--beta", type=float, default=0.0)
    ap.add_argument("--warmup-mse", type=int, default=200)
    ap.add_argument("--init-sigma", type=float, default=5.0)
    ap.add_argument("--init-mu", default="pretrained", choices=["pretrained", "constant"])
    ap.add_argument("--init-mu-m", type=float, default=0.0)
    ap.add_argument("--height-scale", type=float, default=13.2)
    ap.add_argument("--log-var-max", type=float, default=7.0)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--rows", type=int, default=7688, help="train rows, for the ETA")
    ap.add_argument("--bins", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    C = Check()
    print(f"PREFLIGHT  {args.model_id}\n  device {dev}  batch {args.batch} x accum "
          f"{args.accum}  lr {args.lr:.2e}  beta {args.beta}  warmup-mse {args.warmup_mse}")

    # ------------------------------------------------------------------ 1. build
    t0 = time.time()
    model = build(model_id=args.model_id, height_scale=args.height_scale,
                  init_sigma_m=args.init_sigma, bins=args.bins,
                  init_mu=args.init_mu, init_mu_m=args.init_mu_m).to(dev)
    n_par = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  built in {time.time() - t0:.0f}s, {n_par:.1f} M params")

    # ---------------------------------------------- 2. sigma at init, before any step
    # run05's very first logged sigma was already at the clamp. If that is true at init,
    # no amount of training fixes it, because clamp() zeroes the gradient there.
    lv0 = float(model.conv_log_var.bias.detach().cpu()) + 2.0 * math.log(args.height_scale)
    s0 = math.exp(lv0 / 2)
    margin = args.log_var_max - lv0
    C.add("sigma at init", margin > 1.5,
          f"log_var {lv0:+.3f}, sigma {s0:.3f} m, {margin:+.3f} below the "
          f"{args.log_var_max} clamp (need > 1.5)")

    # --------------------------------------------- 3. residual at init on REAL crops
    ds = ShardStream(args.shards, "train", augment=Augment(), buffer_shards=1, seed=args.seed)
    ld = torch.utils.data.DataLoader(ds, batch_size=args.batch, num_workers=0, drop_last=True)
    it = iter(ld)
    b = next(it)
    model.eval()
    with torch.no_grad():
        out = model(b["rgb"].to(dev))
        mu0 = out[0]
    m = b["mask"].to(dev) > 0
    res = float(((b["agl"].to(dev) - mu0)[m]).abs().median()) if m.any() else float("nan")
    # A pretrained head whose output scale suits the task starts within a few metres.
    # DA-V1 and DA-V2 emit disparity on different scales; a large residual here is the
    # thing that drives log_var to the clamp (nll_deadlock_probe: at a 60 m residual,
    # d loss / d log_var = -0.439, i.e. hard upward).
    C.add("residual at init", res < 40.0,
          f"median |truth - mu| = {res:.2f} m on real crops (large residuals drive "
          f"log_var up)", fatal=False)

    # ------------------------------------------------------------- 4. train for real
    model.train()
    head, back = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (head if ("conv_" in n or "neck" in n) else back).append(p)
    opt = torch.optim.AdamW([{"params": back, "lr": args.lr},
                             {"params": head, "lr": args.lr_head or args.lr * 50}])
    loss_fn = CompositeLoss(beta=args.beta, warmup_mse=args.warmup_mse).to(dev)
    amp = torch.bfloat16 if dev == "cuda" else torch.float32

    rail = math.exp(args.log_var_max / 2)
    sig, rmses, gnorm = [], [], []
    nan_at = None
    t0 = time.time()
    for i in range(args.steps):
        try:
            b = next(it)
        except StopIteration:
            it = iter(ld)
            b = next(it)
        rgb, agl = b["rgb"].to(dev), b["agl"].to(dev)
        mask = b["mask"].to(dev)
        with torch.autocast(dev, dtype=amp, enabled=(dev == "cuda")):
            mu, log_var = model(rgb)
            loss, st = loss_fn(mu.float(), log_var.float(), agl, mask)
        if not math.isfinite(float(loss)):
            nan_at = i
            break
        (loss / args.accum).backward()
        if (i + 1) % args.accum == 0:
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            gnorm.append(float(gn))
            opt.step()
            opt.zero_grad(set_to_none=True)
        sig.append(st["sigma_mean"])
        rmses.append(st["rmse"])

    done = len(sig)
    sec_step = (time.time() - t0) / max(done, 1)

    C.add("loss stays finite", nan_at is None,
          "no NaN/Inf" if nan_at is None else f"NaN at step {nan_at} of {args.steps}")
    if done:
        s_end = float(np.mean(sig[-10:]))
        C.add("sigma not railed", s_end < rail * 0.95,
              f"sigma {s_end:.3f} m after {done} steps; clamp sits at {rail:.3f} m")
        r_end = float(np.mean(rmses[-10:]))
        C.add("rmse in a sane band", r_end < 3 * HEALTHY["rmse_s100"],
              f"{r_end:.3f} m vs run04 {HEALTHY['rmse_s100']:.3f} m at s100 "
              f"(fails above {3 * HEALTHY['rmse_s100']:.1f} m)")
        C.add("rmse is falling", rmses[-1] < np.mean(rmses[:5]) * 1.5,
              f"first-5 mean {np.mean(rmses[:5]):.3f} -> last {rmses[-1]:.3f} m", fatal=False)
        if gnorm:
            g = float(np.median(gnorm))
            C.add("gradients alive", 1e-7 < g < 1e4,
                  f"median grad-norm {g:.3e} (dead below 1e-7, exploding above 1e4)")
    if dev == "cuda":
        peak = torch.cuda.max_memory_allocated() / 2 ** 30
        tot = torch.cuda.get_device_properties(0).total_memory / 2 ** 30
        C.add("VRAM headroom", peak < tot * 0.85, f"peak {peak:.2f} GB of {tot:.1f} GB")

    steps_ep = args.rows // (args.batch * args.accum)
    hours = sec_step * args.accum * steps_ep * args.epochs / 3600
    print(f"\n  {sec_step:.2f} s/step measured  ->  {steps_ep} steps/epoch x {args.epochs} "
          f"epochs ~= {hours:.1f} h (no thermal throttling in this estimate)")

    bad = C.report()
    if bad:
        print(f"\n  DO NOT LAUNCH: {bad} fatal check(s) failed. Fix these first.\n")
        return 1
    print(f"\n  GO: config is healthy at step {done}. Launch it.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
