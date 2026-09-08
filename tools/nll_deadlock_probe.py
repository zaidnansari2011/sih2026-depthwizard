"""Why run05 died, measured rather than argued.

run05 logged sigma = 33.115 m = exp(7/2) in EVERY step: log_var was pinned against
log_var_max = 7.0 from the first log line. This asks the only question that matters --
once log_var is pinned there, can the mean head still learn? -- by measuring the actual
gradient that reaches mu.

Run:  python tools/nll_deadlock_probe.py
"""
from __future__ import annotations

import torch

from depthwizard.losses import GaussianNLLLoss

torch.manual_seed(0)


def mu_grad(log_var_value: float, beta: float, err: float = 20.0) -> tuple[float, float]:
    """Gradient magnitude reaching mu, and the gradient pushing log_var, at a fixed state."""
    mu = torch.zeros(1, 1, 64, 64, requires_grad=True)
    lv = torch.full((1, 1, 64, 64), log_var_value, requires_grad=True)
    target = torch.full((1, 1, 64, 64), err)
    fn = GaussianNLLLoss(beta=beta, warmup_mse=0)
    fn.train()
    loss, _ = fn(mu, lv, target)
    loss.backward()
    return float(mu.grad.abs().mean()), float(lv.grad.mean())


print("Setting: residual 20 m (a tall building the model has not learned yet).")
print("log_var_max = 7.0 in losses.py, so log_var is clamped at +7 and the clamp")
print("zeroes its gradient there. Question: does mu still get a gradient?\n")

print(f"{'log_var':>9} {'sigma':>9} {'beta':>6} {'|grad mu|':>12} {'vs healthy':>12}")
ref = None
for lv in (0.0, 3.0, 7.0):
    for beta in (0.0, 1.0):
        g, _ = mu_grad(lv, beta)
        if ref is None:
            ref = g
        tag = f"{g / ref:8.4f}x"
        print(f"{lv:>9.1f} {torch.exp(torch.tensor(lv / 2)):>9.3f} {beta:>6.1f} "
              f"{g:>12.6f} {tag:>12}")

print("\nDoes log_var actually want to run to the clamp? (negative = pushed UP)")
print(f"{'residual':>10} {'d loss / d log_var':>20} {'direction':>12}")
for err in (1.0, 5.0, 20.0, 60.0):
    _, glv = mu_grad(0.0, 0.0, err=err)
    print(f"{err:>9.0f}m {glv:>20.6f} {'UP -> clamp' if glv < 0 else 'down':>12}")
