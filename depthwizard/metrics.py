"""Evaluation metrics for DSM accuracy.

Two jobs:

1. The numbers ISRO names in the scoring criteria -- RMSE, MAE, correlation -- and
   critically, **broken down per terrain type**. The problem statement demands stability
   across urban, sparse, hilly and forested. A single global RMSE hides exactly the
   failure the jury will probe (section 7.5 of PLAN.md).

2. Calibration of the uncertainty head (differentiator 6.1). Predicting a sigma is
   worthless if the sigma is wrong. If we claim 68% of errors fall within 1 sigma, that
   had better be true, and we should be able to show the curve.

Pure numpy so it runs anywhere, including on eval outputs with no GPU present.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field

import numpy as np

# DFC2019 Track 1 semantic classes.
CLS_GROUND, CLS_VEG, CLS_BUILDING, CLS_WATER, CLS_BRIDGE = 2, 5, 6, 9, 17
CLS_NAMES = {
    CLS_GROUND: "ground",
    CLS_VEG: "vegetation",
    CLS_BUILDING: "building",
    CLS_WATER: "water",
    CLS_BRIDGE: "bridge",
}


@dataclass
class HeightMetrics:
    rmse: float
    mae: float
    corr: float
    bias: float               # mean signed error -- systematic over/under-estimation
    median_ae: float
    p90_ae: float             # tail behaviour; RMSE alone hides it
    n_valid: int
    per_class: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)

    def __str__(self):
        s = (f"RMSE {self.rmse:7.3f} m   MAE {self.mae:7.3f} m   r {self.corr:6.3f}   "
             f"bias {self.bias:+7.3f} m   p90|e| {self.p90_ae:7.3f} m   n={self.n_valid:,}")
        for name, m in self.per_class.items():
            s += f"\n    {name:12s} RMSE {m['rmse']:7.3f}  MAE {m['mae']:7.3f}  n={m['n']:,}"
        return s


def _flat_valid(pred, target, mask):
    pred, target = np.asarray(pred, np.float64), np.asarray(target, np.float64)
    valid = np.isfinite(pred) & np.isfinite(target)
    if mask is not None:
        valid &= np.asarray(mask).astype(bool)
    return pred[valid], target[valid]


def height_metrics(pred, target, mask=None, cls=None) -> HeightMetrics:
    """Core accuracy numbers, optionally broken out per semantic class."""
    p, t = _flat_valid(pred, target, mask)
    if p.size == 0:
        return HeightMetrics(np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 0)

    err = p - t
    ae = np.abs(err)
    corr = float(np.corrcoef(p, t)[0, 1]) if p.size > 1 and p.std() > 0 and t.std() > 0 else np.nan

    per_class = {}
    if cls is not None:
        cls = np.asarray(cls)
        valid = np.isfinite(np.asarray(pred, np.float64)) & np.isfinite(np.asarray(target, np.float64))
        if mask is not None:
            valid &= np.asarray(mask).astype(bool)
        cv = cls[valid]
        for code, name in CLS_NAMES.items():
            sel = cv == code
            if sel.sum() < 100:
                continue
            e = err[sel]
            per_class[name] = {
                "rmse": float(np.sqrt((e ** 2).mean())),
                "mae": float(np.abs(e).mean()),
                "bias": float(e.mean()),
                "n": int(sel.sum()),
            }

    return HeightMetrics(
        rmse=float(np.sqrt((err ** 2).mean())),
        mae=float(ae.mean()),
        corr=corr,
        bias=float(err.mean()),
        median_ae=float(np.median(ae)),
        p90_ae=float(np.percentile(ae, 90)),
        n_valid=int(p.size),
        per_class=per_class,
    )


def terrain_category(cls: np.ndarray) -> str:
    """Bucket a tile into the four terrain types ISRO names, from its class mix.

    DFC2019 ships semantic classes, not ISRO's terrain vocabulary, so we map:
      building fraction high            -> urban
      vegetation fraction high          -> forested
      little of either                  -> sparse
      otherwise                         -> mixed

    'hilly' is a relief property, not a land-cover one -- classify that from the
    ground-height range instead, via terrain_category_with_relief().
    """
    cls = np.asarray(cls)
    n = cls.size
    if n == 0:
        return "unknown"
    b = float((cls == CLS_BUILDING).sum()) / n
    v = float((cls == CLS_VEG).sum()) / n
    if b > 0.20:
        return "urban"
    if v > 0.40:
        return "forested"
    if b < 0.05 and v < 0.15:
        return "sparse"
    return "mixed"


def terrain_category_with_relief(cls: np.ndarray, agl: np.ndarray, relief_m: float = 30.0) -> str:
    """As above, but promote to 'hilly' when ground-class relief is large."""
    cat = terrain_category(cls)
    ground = np.asarray(agl)[np.asarray(cls) == CLS_GROUND]
    if ground.size > 100:
        rel = float(np.percentile(ground, 95) - np.percentile(ground, 5))
        if rel > relief_m:
            return "hilly"
    return cat


# ------------------------------------------------------------------ uncertainty (6.1)

def calibration_curve(pred, target, sigma, mask=None, ks=(0.5, 1.0, 1.5, 2.0, 2.5, 3.0)):
    """Empirical vs theoretical coverage at k sigma.

    For a correctly calibrated Gaussian, |error| < k*sigma should hold for
    erf(k/sqrt(2)) of pixels: 38.3% at 0.5, 68.3% at 1, 95.4% at 2, 99.7% at 3.

    Returns a list of (k, expected, observed). Plot it -- a diagonal is the claim
    "our confidence means something", and it is a slide the jury will remember.
    """
    from math import erf, sqrt

    p, t = _flat_valid(pred, target, mask)
    s = np.asarray(sigma, np.float64)
    valid = np.isfinite(np.asarray(pred, np.float64)) & np.isfinite(np.asarray(target, np.float64))
    if mask is not None:
        valid &= np.asarray(mask).astype(bool)
    s = s[valid]
    s = np.maximum(s, 1e-6)
    ae = np.abs(p - t)

    return [(float(k), erf(k / sqrt(2.0)), float((ae < k * s).mean())) for k in ks]


def expected_calibration_error(pred, target, sigma, mask=None) -> float:
    """Mean |expected - observed| coverage across k. Lower is better; report it."""
    curve = calibration_curve(pred, target, sigma, mask)
    return float(np.mean([abs(exp - obs) for _, exp, obs in curve]))


def uncertainty_error_correlation(pred, target, sigma, mask=None) -> float:
    """Does predicted sigma actually track realised error?

    Spearman rank correlation between sigma and |error|. Calibration can look fine on
    average while sigma is uninformative pixel to pixel; this catches that. A model
    that knows where it is wrong scores high here, and that is the whole claim of 6.1.
    """
    p, t = _flat_valid(pred, target, mask)
    s = np.asarray(sigma, np.float64)
    valid = np.isfinite(np.asarray(pred, np.float64)) & np.isfinite(np.asarray(target, np.float64))
    if mask is not None:
        valid &= np.asarray(mask).astype(bool)
    s = s[valid]
    if p.size < 2:
        return float("nan")

    def rank(x):
        order = x.argsort()
        r = np.empty_like(order, dtype=np.float64)
        r[order] = np.arange(len(x))
        return r

    rs, re = rank(s), rank(np.abs(p - t))
    if rs.std() == 0 or re.std() == 0:
        return float("nan")
    return float(np.corrcoef(rs, re)[0, 1])


# ------------------------------------------------------------------------ reporting

def report(pred, target, sigma=None, mask=None, cls=None) -> dict:
    """Everything at once -- the shape of the week-1 comparison table."""
    m = height_metrics(pred, target, mask, cls)
    out = m.to_dict()
    if sigma is not None:
        out["calibration"] = [
            {"k": k, "expected": e, "observed": o}
            for k, e, o in calibration_curve(pred, target, sigma, mask)
        ]
        out["ece"] = expected_calibration_error(pred, target, sigma, mask)
        out["sigma_error_rank_corr"] = uncertainty_error_correlation(pred, target, sigma, mask)
    return out
