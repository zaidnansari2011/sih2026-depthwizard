"""Tests for the uncertainty losses.

The claim behind differentiator 6.1 is that our sigma means something. That claim is
only as good as the objective that produces it, so this file does two kinds of check:

  1. Analytic  -- the NLL is minimised where theory says it is, masking is exact,
                  gradients reach both heads.
  2. Recovery  -- given synthetic data with a KNOWN, spatially varying noise level,
                  optimising this loss must recover that noise level. That is the test
                  which actually earns the calibration curve on the slide.

Run:  python -m tests.test_losses
"""
from __future__ import annotations

import math

import torch

from depthwizard.losses import (
    GaussianNLLLoss,
    MaskedL1Loss,
    GradientMatchingLoss,
    CompositeLoss,
)

torch.manual_seed(0)
PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{'   ' + detail if detail else ''}")


# --------------------------------------------------------------- analytic properties

def test_nll_minimum_is_where_theory_says():
    """For a fixed residual r, NLL over log_var is minimised at log_var = log(r^2)."""
    loss = GaussianNLLLoss()
    loss.eval()
    target = torch.zeros(1, 1, 4, 4)
    mu = torch.full_like(target, 2.0)          # residual r = 2, so r^2 = 4
    best_lv, best_val = None, float("inf")
    for lv in torch.linspace(-2, 5, 701):
        v, _ = loss(mu, torch.full_like(target, float(lv)), target)
        if float(v) < best_val:
            best_val, best_lv = float(v), float(lv)
    check("NLL minimised at log_var = log(residual^2)",
          abs(best_lv - math.log(4.0)) < 0.02,
          f"argmin {best_lv:.3f} vs log(4)={math.log(4.0):.3f}")


def test_variance_cannot_be_bought():
    """Inflating variance without cause must increase the loss -- no free pass."""
    loss = GaussianNLLLoss()
    loss.eval()
    t = torch.zeros(1, 1, 8, 8)
    mu = torch.zeros_like(t)                    # perfect prediction
    honest, _ = loss(mu, torch.full_like(t, -4.0), t)
    inflated, _ = loss(mu, torch.full_like(t, 3.0), t)
    check("inflating sigma with no error raises loss", float(inflated) > float(honest),
          f"{float(honest):.3f} -> {float(inflated):.3f}")


def test_mask_is_exact():
    """Masked pixels must contribute exactly nothing, however extreme they are."""
    loss = GaussianNLLLoss()
    loss.eval()
    t = torch.zeros(1, 1, 4, 4)
    mu = torch.zeros_like(t)
    lv = torch.zeros_like(t)
    mask = torch.ones_like(t)
    ref, _ = loss(mu, lv, t, mask)

    mu_poisoned = mu.clone()
    mu_poisoned[0, 0, 0, 0] = 1e4               # a void pixel carrying an absurd value
    mask[0, 0, 0, 0] = 0
    got, _ = loss(mu_poisoned, lv, t, mask)
    check("masked pixels excluded exactly", torch.isclose(ref, got, atol=1e-5),
          f"{float(ref):.6f} vs {float(got):.6f}")


def test_gradients_reach_both_heads():
    loss = GaussianNLLLoss()
    loss.train()
    mu = torch.randn(2, 1, 8, 8, requires_grad=True)
    lv = torch.randn(2, 1, 8, 8, requires_grad=True)
    t = torch.randn(2, 1, 8, 8)
    v, _ = loss(mu, lv, t)
    v.backward()
    check("gradient reaches mu", mu.grad is not None and mu.grad.abs().sum() > 0)
    check("gradient reaches log_var", lv.grad is not None and lv.grad.abs().sum() > 0)


def test_extreme_log_var_is_finite():
    """Clamping must survive values that would otherwise overflow exp()."""
    loss = GaussianNLLLoss()
    loss.eval()
    t = torch.zeros(1, 1, 4, 4)
    for lv_val in (-500.0, 500.0):
        v, s = loss(torch.ones_like(t), torch.full_like(t, lv_val), t)
        if not (torch.isfinite(v) and math.isfinite(s["sigma_mean"])):
            check(f"finite loss at log_var={lv_val}", False)
            return
    check("finite loss at log_var = +/-500", True)


def test_warmup_is_plain_mse():
    loss = GaussianNLLLoss(warmup_mse=5)
    loss.train()
    t = torch.zeros(1, 1, 4, 4)
    mu = torch.full_like(t, 3.0)
    v, _ = loss(mu, torch.full_like(t, 9.0), t)   # log_var must be ignored during warmup
    check("warmup returns MSE", abs(float(v) - 9.0) < 1e-5, f"got {float(v):.4f} want 9.0")
    for _ in range(6):
        loss(mu, torch.full_like(t, 9.0), t)
    v_after, _ = loss(mu, torch.full_like(t, 9.0), t)
    check("warmup expires", abs(float(v_after) - 9.0) > 1e-3, f"got {float(v_after):.4f}")


def test_beta_nll_endpoints():
    """beta=0 is standard NLL; beta=1 makes the objective MSE-proportional.

    Seitzer et al. 2022 weight the WHOLE per-pixel NLL by stopgrad(sigma^2)^beta.
    At beta=1 that cancels the 1/sigma^2 in the residual term, so the gradient w.r.t.
    mu stops depending on sigma -- which is the entire point of beta-NLL: it prevents
    the mean head from abandoning noisy regions.
    """
    t = torch.zeros(1, 1, 4, 4)
    mu = torch.full_like(t, 2.0)
    lv = torch.full_like(t, 1.5)

    plain = GaussianNLLLoss(beta=0.0)
    plain.eval()
    ref, _ = plain(mu, lv, t)
    manual = 0.5 * (1.5 + 4.0 / math.exp(1.5))
    check("beta=0 equals textbook NLL", abs(float(ref) - manual) < 1e-5,
          f"{float(ref):.6f} vs {manual:.6f}")

    beta1 = GaussianNLLLoss(beta=1.0)
    beta1.eval()
    got, _ = beta1(mu, lv, t)
    want = math.exp(1.5) * manual
    check("beta=1 scales NLL by sigma^2", abs(float(got) - want) < 1e-4,
          f"{float(got):.6f} vs {want:.6f}")


def test_gradient_matching():
    g = GradientMatchingLoss()
    t = torch.randn(1, 1, 16, 16)
    check("grad-match zero on exact match", float(g(t, t)) < 1e-6)
    blurred = torch.nn.functional.avg_pool2d(t, 3, 1, 1)
    check("grad-match penalises blur", float(g(blurred, t)) > 0)
    mask = torch.ones_like(t)
    check("grad-match zero on exact match, masked", float(g(t, t, mask)) < 1e-6)


def test_l1_masked():
    l1 = MaskedL1Loss()
    t = torch.zeros(1, 1, 4, 4)
    p = torch.ones_like(t)
    mask = torch.zeros_like(t)
    mask[0, 0, :2, :] = 1
    check("masked L1 uses mask denominator", abs(float(l1(p, t, mask)) - 1.0) < 1e-6)


# ------------------------------------------------------------------ recovery of sigma

def test_recovers_known_heteroscedastic_noise():
    """The load-bearing test.

    Build data whose true noise sigma varies across the image in a way we choose:
    quiet on the left, loud on the right. Fit a mean and a log-variance per pixel and
    check the optimiser lands on the true sigmas. If this fails, every calibration
    curve we ever plot is decoration.
    """
    torch.manual_seed(7)
    N, H, W = 256, 8, 8
    sigma_lo, sigma_hi = 0.5, 3.0

    true_sigma = torch.full((1, 1, H, W), sigma_lo)
    true_sigma[..., W // 2:] = sigma_hi
    target = torch.randn(N, 1, H, W) * true_sigma          # true mean is 0

    mu = torch.zeros(1, 1, H, W, requires_grad=True)
    log_var = torch.zeros(1, 1, H, W, requires_grad=True)
    opt = torch.optim.Adam([mu, log_var], lr=0.05)
    loss_fn = GaussianNLLLoss()
    loss_fn.train()

    for _ in range(1200):
        opt.zero_grad()
        v, _ = loss_fn(mu.expand(N, -1, -1, -1), log_var.expand(N, -1, -1, -1), target)
        v.backward()
        opt.step()

    est = torch.exp(0.5 * log_var.detach())
    lo = float(est[..., : W // 2].mean())
    hi = float(est[..., W // 2:].mean())
    print(f"      true sigma  lo {sigma_lo:.2f}  hi {sigma_hi:.2f}")
    print(f"      recovered   lo {lo:.2f}  hi {hi:.2f}")
    check("recovers low-noise sigma", abs(lo - sigma_lo) / sigma_lo < 0.15, f"{lo:.3f}")
    check("recovers high-noise sigma", abs(hi - sigma_hi) / sigma_hi < 0.15, f"{hi:.3f}")
    check("separates the two regimes", hi > 3 * lo, f"ratio {hi / lo:.2f} (true 6.0)")


def test_composite_runs_and_reports():
    c = CompositeLoss(grad_weight=0.5)
    c.train()
    mu = torch.randn(2, 1, 32, 32, requires_grad=True)
    lv = torch.zeros(2, 1, 32, 32, requires_grad=True)
    t = torch.randn(2, 1, 32, 32)
    mask = torch.ones_like(t)
    v, stats = c(mu, lv, t, mask)
    v.backward()
    check("composite backprops", mu.grad.abs().sum() > 0 and lv.grad.abs().sum() > 0)
    check("composite reports grad + rmse + sigma",
          {"loss", "rmse", "sigma_mean", "grad"} <= set(stats))


def main():
    for fn in [
        test_nll_minimum_is_where_theory_says,
        test_variance_cannot_be_bought,
        test_mask_is_exact,
        test_gradients_reach_both_heads,
        test_extreme_log_var_is_finite,
        test_warmup_is_plain_mse,
        test_beta_nll_endpoints,
        test_gradient_matching,
        test_l1_masked,
        test_composite_runs_and_reports,
        test_recovers_known_heteroscedastic_noise,
    ]:
        print(f"\n{fn.__name__}")
        fn()
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        print("FAILED: " + ", ".join(FAIL))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
