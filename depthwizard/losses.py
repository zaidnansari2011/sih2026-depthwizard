"""Losses for height regression with per-pixel uncertainty.

Differentiator 6.1. Single-view height is ill-posed -- there is no unique 3D solution
for a 2D image -- so the model predicts a distribution per pixel, not a point estimate.
We train with Gaussian negative log-likelihood and let the model say where it is unsure.

The network emits two channels: mu (height) and log_var (log predictive variance).
Predicting log-variance rather than variance keeps it positive without a clamp and
keeps gradients well-scaled.

    L = 0.5 * ( log_var + (y - mu)^2 / exp(log_var) )

Read the two terms as a trade: the model can lower the residual term by admitting
uncertainty, but pays log_var for doing so. It cannot buy a free pass by predicting
huge variance everywhere.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class GaussianNLLLoss(nn.Module):
    """Heteroscedastic Gaussian NLL with an optional validity mask.

    Args:
        beta: detach-weight in the residual term, per Kendall & Gal. 0 = standard NLL.
            Small positive values stabilise very early training, when a randomly
            initialised variance head can dominate the loss.
        log_var_min/max: clamp range. Guards against exp() overflow and against the
            model escaping into "infinitely uncertain everywhere".
        warmup_mse: if set, the first N steps use plain MSE so mu becomes reasonable
            before the variance head starts shaping gradients.
    """

    def __init__(
        self,
        beta: float = 0.0,
        log_var_min: float = -7.0,
        log_var_max: float = 7.0,
        warmup_mse: int = 0,
    ):
        super().__init__()
        self.beta = beta
        self.log_var_min = log_var_min
        self.log_var_max = log_var_max
        self.warmup_mse = warmup_mse
        self.register_buffer("_step", torch.zeros((), dtype=torch.long))

    def forward(
        self,
        mu: torch.Tensor,
        log_var: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        log_var = log_var.clamp(self.log_var_min, self.log_var_max)
        sq_err = (target - mu) ** 2

        if self.training:
            self._step += 1
        if self.warmup_mse and int(self._step) <= self.warmup_mse:
            loss_map = sq_err
        else:
            inv_var = torch.exp(-log_var)
            loss_map = 0.5 * (log_var + sq_err * inv_var)
            if self.beta > 0:
                # beta-NLL, Seitzer et al. 2022: weight each pixel's NLL by
                # stopgrad(sigma^2)^beta. The weight covers the WHOLE term, not just
                # the residual. At beta=1 it cancels the 1/sigma^2 so the mean head's
                # gradient stops depending on sigma -- which is the point: plain NLL
                # lets mu abandon noisy regions, because down-weighting them is
                # cheaper than fitting them.
                loss_map = loss_map * torch.exp(log_var.detach() * self.beta)

        if mask is not None:
            n = mask.sum().clamp(min=1.0)
            loss = (loss_map * mask).sum() / n
            rmse = ((sq_err * mask).sum() / n).sqrt()
        else:
            loss = loss_map.mean()
            rmse = sq_err.mean().sqrt()

        stats = {
            "loss": float(loss.detach()),
            "rmse": float(rmse.detach()),
            "sigma_mean": float(torch.exp(0.5 * log_var).mean().detach()),
        }
        return loss, stats


class MaskedL1Loss(nn.Module):
    """Plain masked L1. Baseline comparator -- what a team without 6.1 would train."""

    def forward(self, pred, target, mask=None):
        err = (pred - target).abs()
        if mask is None:
            return err.mean()
        return (err * mask).sum() / mask.sum().clamp(min=1.0)


class GradientMatchingLoss(nn.Module):
    """Penalise disagreement in first differences.

    Height maps are piecewise smooth with hard edges at building footprints. Pure
    per-pixel losses produce blurred roof edges, which look bad in the flythrough
    (differentiator 6.3) even when RMSE is respectable. This sharpens them.
    """

    def forward(self, pred, target, mask=None):
        def diffs(t):
            return t[..., :, 1:] - t[..., :, :-1], t[..., 1:, :] - t[..., :-1, :]

        px, py = diffs(pred)
        tx, ty = diffs(target)
        lx, ly = (px - tx).abs(), (py - ty).abs()
        if mask is None:
            return lx.mean() + ly.mean()
        mx, my = diffs(mask)
        mx, my = (mx == 0).float() * mask[..., :, 1:], (my == 0).float() * mask[..., 1:, :]
        return ((lx * mx).sum() / mx.sum().clamp(min=1.0)
                + (ly * my).sum() / my.sum().clamp(min=1.0))


class ChamferBinLoss(nn.Module):
    """Bi-directional Chamfer between bin EDGES and the heights actually present.

    Without it the adaptive widths are free to park anywhere. This pulls them onto the
    quantiles of the scene height distribution, which is the point of making them
    adaptive at all.

    Edges rather than centres: AdaBins supervises centres, HTC-DC Net supervises edges
    (Eqn 13) so the intervals themselves land on the quantiles of the truth. run03
    followed AdaBins here. The weight follows HTC-DC Net too, mu1 = 0.01 rather than
    AdaBins' 0.1 -- run03 ran this term ten times too hot.
    """

    def __init__(self, max_points: int = 4096):
        super().__init__()
        self.max_points = int(max_points)

    def forward(self, edges, target, mask=None):
        total, n = edges.new_zeros(()), 0
        for b in range(edges.shape[0]):
            t = target[b][mask[b] > 0] if mask is not None else target[b].flatten()
            if t.numel() == 0:
                continue
            if t.numel() > self.max_points:
                t = t[torch.randint(0, t.numel(), (self.max_points,), device=t.device)]
            d = (t[:, None] - edges[b][None, :]).abs()
            total = total + d.min(dim=1).values.mean() + d.min(dim=0).values.mean()
            n += 1
        return total / max(n, 1)


class HeadTailCutLoss(nn.Module):
    """Cross-entropy on the head-tail gate (HTC-DC Net, TGRS 2023, Eqn 14).

    The gate decides which of the two bin distributions a pixel reads from, so it has to
    be trained on its own terms rather than indirectly through the height loss. Measured
    on our shards the split at 1 m is 35.6/64.4, close enough to balanced that plain BCE
    needs no reweighting.
    """

    def __init__(self, threshold: float = 1.0):
        super().__init__()
        self.threshold = float(threshold)

    def forward(self, logit, target, mask=None):
        tgt = (target > self.threshold).to(logit.dtype)
        loss = F.binary_cross_entropy_with_logits(logit, tgt, reduction="none")
        if mask is None:
            return loss.mean()
        m = mask.to(loss.dtype)
        return (loss * m).sum() / m.sum().clamp(min=1.0)


class DistributionConstraint(nn.Module):
    """HTC-DC Net's distribution-based constraint (TGRS 2023, Eqns 15-22).

    Soft-argmax returns the EXPECTATION of the predicted distribution. That coincides
    with the mode only when the distribution is symmetric and unimodal, and nothing makes
    it so unless you say so. This says so.

    The part that matters is the scale, and it is not a hyperparameter. The reference
    width is solved from the model's own probability on the bin holding the truth:
    assuming the truth sits at that bin's centre, Pm = erf(w / (2*sqrt(2)*sigma)), which
    inverts to sigma = w / (2*sqrt(2)*erfinv(Pm)). A confident pixel gets a sharp
    reference and a hesitant one a broad reference, so the term pulls the expectation
    toward the mode without dictating how sharp to be.

    run03 used one fixed width for every pixel instead. That fought the heteroscedastic
    sigma head rather than cooperating with it, and left building error untouched. This is
    the difference between the published method and the approximation of it that failed.

    Foreground takes the Gaussian. Background takes a uniform of the width the same mode
    probability implies, as the paper specifies: ground is near-constant, and forcing a
    sharp Gaussian onto it invents confidence that is not there.

    Scored on a random subsample of valid pixels -- a full (B, N, H, W) reference would be
    another half-gigabyte on a card already at 5.7 of 12 GB, and a few thousand pixels
    estimate this term perfectly well.
    """

    def __init__(self, n_points: int = 4096, fg_threshold: float = 1.0,
                 p_clamp: float = 1e-3):
        super().__init__()
        self.n_points = int(n_points)
        self.fg_threshold = float(fg_threshold)
        self.p_clamp = float(p_clamp)

    def forward(self, probs, edges, target, mask=None):
        N = probs.shape[1]
        p = probs.flatten(2)                              # (B, N, P)
        t = target.flatten(2).squeeze(1)                  # (B, P)
        m = (mask.flatten(2).squeeze(1) > 0) if mask is not None else None
        root2 = math.sqrt(2.0)
        total, n = p.new_zeros(()), 0

        for b in range(probs.shape[0]):
            idx = (m[b].nonzero(as_tuple=False).squeeze(1) if m is not None
                   else torch.arange(t.shape[1], device=t.device))
            if idx.numel() == 0:
                continue
            if idx.numel() > self.n_points:
                idx = idx[torch.randint(0, idx.numel(), (self.n_points,),
                                        device=idx.device)]
            tb = t[b][idx]                                # (K,)
            pb = p[b][:, idx].t()                         # (K, N)
            e = edges[b]                                  # (N + 1,)

            # Which bin holds the truth, and how wide that bin is.
            j = (torch.bucketize(tb.contiguous(), e.contiguous()) - 1).clamp(0, N - 1)
            w = (e[j + 1] - e[j]).clamp(min=1e-4)
            pm = pb.gather(1, j[:, None]).squeeze(1).clamp(self.p_clamp,
                                                           1.0 - self.p_clamp)

            # Eqn 20. Clamped because erfinv diverges as pm approaches 1.
            sigma = (w / (2.0 * root2 * torch.erfinv(pm))).clamp(min=1e-3, max=1e3)
            cdf = 0.5 * (1.0 + torch.erf((e[None, :] - tb[:, None])
                                         / (sigma[:, None] * root2)))
            ref_g = (cdf[:, 1:] - cdf[:, :-1]).clamp(min=0.0)          # Eqn 21

            # Uniform of the width this same mode probability implies.
            width = (w / pm).clamp(min=w)
            lo, hi = tb - 0.5 * width, tb + 0.5 * width
            a = torch.maximum(e[None, :-1], lo[:, None])
            c = torch.minimum(e[None, 1:], hi[:, None])
            ref_u = (c - a).clamp(min=0.0) / width[:, None]

            ref = torch.where((tb > self.fg_threshold)[:, None], ref_g, ref_u)
            ref = ref / ref.sum(dim=1, keepdim=True).clamp(min=1e-8)

            # KL(reference || predicted), Eqn 22.
            kl = (ref * (ref.clamp(min=1e-8).log() - pb.clamp(min=1e-8).log())).sum(1)
            total = total + kl.mean()
            n += 1
        return total / max(n, 1)


class CompositeLoss(nn.Module):
    """NLL + optional gradient matching. The default training objective."""

    def __init__(self, grad_weight: float = 0.5, chamfer_weight: float = 0.0,
                 dist_weight: float = 0.0, htc_weight: float = 0.0,
                 fg_threshold: float = 1.0, **nll_kwargs):
        super().__init__()
        self.nll = GaussianNLLLoss(**nll_kwargs)
        self.grad = GradientMatchingLoss()
        self.chamfer = ChamferBinLoss()
        self.dist = DistributionConstraint(fg_threshold=fg_threshold)
        self.htc = HeadTailCutLoss(threshold=fg_threshold)
        self.grad_weight = grad_weight
        self.chamfer_weight = chamfer_weight
        self.dist_weight = dist_weight
        self.htc_weight = htc_weight

    def forward(self, mu, log_var, target, mask=None, aux=None):
        loss, stats = self.nll(mu, log_var, target, mask)
        if self.grad_weight > 0:
            g = self.grad(mu, target, mask)
            loss = loss + self.grad_weight * g
            stats["grad"] = float(g.detach())
        if aux is not None:
            if self.chamfer_weight > 0:
                c = self.chamfer(aux["edges"], target, mask)
                loss = loss + self.chamfer_weight * c
                stats["chamfer"] = float(c.detach())
            if self.dist_weight > 0:
                d = self.dist(aux["probs"], aux["edges"], target, mask)
                loss = loss + self.dist_weight * d
                stats["dist"] = float(d.detach())
            if self.htc_weight > 0 and aux.get("htc_logit") is not None:
                h = self.htc(aux["htc_logit"], target, mask)
                loss = loss + self.htc_weight * h
                stats["htc"] = float(h.detach())
        # Outside the grad_weight branch. With --grad-weight 0 the logged loss used to
        # be whatever the NLL alone reported, which went stale the moment any other
        # term existed.
        stats["loss"] = float(loss.detach())
        return loss, stats
