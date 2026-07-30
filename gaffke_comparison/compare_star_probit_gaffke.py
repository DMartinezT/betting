#!/usr/bin/env python3
"""
Compare three fixed-horizon confidence intervals for the mean of [0,1]-valued data:

1. Product STaR-Bets (square-root / exponential planning feedback)
2. Regularized Efficient betting
3. Equal-tail Gaffke confidence interval

The default experiment design matches the paper's primary experiment:
    delta = 0.01
    n in {10, 50, 100, 500, 1000, 5000, 10000, 50000, 100000, 500000, 1000000}
    50 paths through n=10,000, 30 through n=100,000, 20 through n=1,000,000
    nine distributions:
        Beta(2,2), Beta(1,5), Bernoulli(0.5), Uniform(0,1),
        Beta(1/2,1/2), Bernoulli(0.1), Beta(50,50), Beta(20,80),
        Uniform(0.45,0.55)

Notes
-----
* Product STaR is directly thresholded using the equally weighted two-arm wealth,
  as in the draft's primary experiment. The optional terminal randomization from
  the original STaR code is not used.
* Efficient betting uses b_n = n^(2/3), one fixed U per arm for the whole inversion,
  and the same predictable second-moment regularization as product STaR.
* Gaffke endpoints are Dirichlet-average quantiles:
      lower = Q_{delta/2}(sum x_i D_i)
      upper = Q_{1-delta/2}(D_0 + sum x_i D_i)
  For Bernoulli data these are exact Clopper-Pearson beta quantiles.
  For continuous data this script uses the exact normalized B-spline
  representation up to --gaffke-exact-cutoff, and a fourth-order
  Cornish-Fisher approximation based on exact Dirichlet moments above it.
  The fallback is extremely accurate at the large n where it is used, but it is
  still an approximation. Set a larger cutoff if you want more exact spline work.

Dependencies
------------
numpy pandas scipy numba matplotlib

Example
-------
python compare_star_probit_gaffke.py --quick
python compare_star_probit_gaffke.py --output results_full
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from numba import njit, prange
from scipy.interpolate import BSpline
from scipy.optimize import brentq
from scipy.stats import beta as beta_dist
from scipy.stats import norm


NEG_INF = -1.0e300
LOG_2PI_HALF = 0.5 * math.log(2.0 * math.pi)


# ---------------------------------------------------------------------------
# Fast Gaussian utilities for Numba
# ---------------------------------------------------------------------------

@njit(cache=True)
def _norm_ppf(p: float) -> float:
    """Acklam's inverse-normal approximation."""
    a1 = -3.969683028665376e01
    a2 = 2.209460984245205e02
    a3 = -2.759285104469687e02
    a4 = 1.383577518672690e02
    a5 = -3.066479806614716e01
    a6 = 2.506628277459239e00

    b1 = -5.447609879822406e01
    b2 = 1.615858368580409e02
    b3 = -1.556989798598866e02
    b4 = 6.680131188771972e01
    b5 = -1.328068155288572e01

    c1 = -7.784894002430293e-03
    c2 = -3.223964580411365e-01
    c3 = -2.400758277161838e00
    c4 = -2.549732539343734e00
    c5 = 4.374664141464968e00
    c6 = 2.938163982698783e00

    d1 = 7.784695709041462e-03
    d2 = 3.224671290700398e-01
    d3 = 2.445134137142996e00
    d4 = 3.754408661907416e00

    plow = 0.02425
    phigh = 1.0 - plow

    if p <= 0.0:
        return -math.inf
    if p >= 1.0:
        return math.inf

    if p < plow:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c1*q + c2)*q + c3)*q + c4)*q + c5)*q + c6) / \
               ((((d1*q + d2)*q + d3)*q + d4)*q + 1.0)

    if p > phigh:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -(((((c1*q + c2)*q + c3)*q + c4)*q + c5)*q + c6) / \
                ((((d1*q + d2)*q + d3)*q + d4)*q + 1.0)

    q = p - 0.5
    r = q * q
    return (((((a1*r + a2)*r + a3)*r + a4)*r + a5)*r + a6) * q / \
           (((((b1*r + b2)*r + b3)*r + b4)*r + b5)*r + 1.0)


@njit(cache=True)
def _psi_from_logp(logp: float) -> float:
    """
    psi(p) = phi(Phi^{-1}(p))/p, evaluated from log(p).
    """
    # The wealth can become tiny. Clipping here only avoids floating-point
    # underflow; the corresponding fraction is still subsequently clipped by
    # the pathwise solvency bound.
    if logp >= -1.0e-14:
        return 0.0
    lp = max(logp, -700.0)
    p = math.exp(lp)
    z = _norm_ppf(p)
    logpsi = -0.5 * z * z - LOG_2PI_HALF - lp
    return math.exp(min(logpsi, 50.0))


@njit(cache=True)
def _logaddexp(a: float, b: float) -> float:
    if a <= NEG_INF / 2:
        return b
    if b <= NEG_INF / 2:
        return a
    if a >= b:
        return a + math.log1p(math.exp(b - a))
    return b + math.log1p(math.exp(a - b))


# ---------------------------------------------------------------------------
# Product STaR and Efficient betting
# ---------------------------------------------------------------------------

@njit(cache=True)
def _vhat(
    sum_sq: float,
    t_reference: float,
    candidate_for_regularizer: float,
    m: float,
    n: int,
    regularizer: float,
    eps: float,
) -> float:
    """
    Predictable second-moment estimate modeled on the reference STaR code.

    At the first rounds the n/t^2 term pushes the estimate to the Bernoulli
    upper bound m(1-m); later, the empirical second moment dominates.
    """
    raw = (
        sum_sq / t_reference
        + regularizer * (candidate_for_regularizer + eps) * n
        / (t_reference * t_reference)
    )
    cap = eps + m * (1.0 - m)
    out = min(raw, cap)
    return max(out, eps)


@njit(cache=True)
def _update_logwealth(logk: float, lam: float, y: float) -> float:
    if logk <= NEG_INF / 2:
        return NEG_INF
    factor = 1.0 + lam * y
    if factor <= 0.0:
        return NEG_INF
    return logk + math.log(factor)


@njit(cache=True)
def _star_score_scalar(
    x: np.ndarray,
    m_in: float,
    delta: float,
    regularizer: float,
    solvency_c: float,
    eps: float,
) -> float:
    """
    Positive score means rejection.

    Each arm plans against target 2/delta. The reported two-sided e-value is
    (K_plus + K_minus)/2 and is thresholded at 1/delta.
    """
    n = x.size
    m = min(max(m_in, eps), 1.0 - eps)
    log_target_arm = math.log(2.0 / delta)
    logk_plus = 0.0
    logk_minus = 0.0
    sum_sq = 0.0

    for j in range(n):
        # This matches the indexing convention in the public STaR reference
        # implementation: t is approximately j, not j+1.
        tref = j + 1.0e-4
        remaining = max(n - tref, 1.0e-8)

        y = x[j] - m

        if logk_plus < log_target_arm and logk_plus > NEG_INF / 2:
            vp = _vhat(sum_sq, tref, m, m, n, regularizer, eps)
            lam_p = math.sqrt(
                2.0 * max(log_target_arm - logk_plus, 0.0)
                / (remaining * vp)
            )
            lam_p = min(lam_p, solvency_c / max(m, eps))
            logk_plus = _update_logwealth(logk_plus, lam_p, y)

        if logk_minus < log_target_arm and logk_minus > NEG_INF / 2:
            vm = _vhat(sum_sq, tref, 1.0 - m, m, n, regularizer, eps)
            lam_m = math.sqrt(
                2.0 * max(log_target_arm - logk_minus, 0.0)
                / (remaining * vm)
            )
            lam_m = min(lam_m, solvency_c / max(1.0 - m, eps))
            logk_minus = _update_logwealth(logk_minus, lam_m, -y)

        sum_sq += y * y

    log_two_sided_wealth = _logaddexp(logk_plus, logk_minus) - math.log(2.0)
    return log_two_sided_wealth - math.log(1.0 / delta)


@njit(cache=True)
def _probit_score_scalar(
    x: np.ndarray,
    m_in: float,
    delta: float,
    u_plus: float,
    u_minus: float,
    regularizer: float,
    solvency_c: float,
    eps: float,
) -> float:
    """
    Positive score means that at least one randomized one-sided arm rejects.

    A single U per arm is held fixed for all candidate means.
    """
    n = x.size
    m = min(max(m_in, eps), 1.0 - eps)
    alpha = delta / 2.0
    log_alpha = math.log(alpha)
    log_target_arm = math.log(1.0 / alpha)
    log_u_plus = math.log(max(u_plus, 1.0e-300))
    log_u_minus = math.log(max(u_minus, 1.0e-300))
    bn = n ** (2.0 / 3.0)

    logk_plus = 0.0
    logk_minus = 0.0
    sum_sq = 0.0

    for j in range(n):
        tref = j + 1.0e-4
        remaining_buffered = n - j + bn
        y = x[j] - m

        if logk_plus < log_target_arm and logk_plus > NEG_INF / 2:
            vp = _vhat(sum_sq, tref, m, m, n, regularizer, eps)
            logp = log_alpha + logk_plus
            psi = _psi_from_logp(logp)
            lam_p = psi / math.sqrt(max(remaining_buffered * vp, eps))
            lam_p = min(lam_p, solvency_c / max(m, eps))
            logk_plus = _update_logwealth(logk_plus, lam_p, y)

        if logk_minus < log_target_arm and logk_minus > NEG_INF / 2:
            vm = _vhat(sum_sq, tref, 1.0 - m, m, n, regularizer, eps)
            logp = log_alpha + logk_minus
            psi = _psi_from_logp(logp)
            lam_m = psi / math.sqrt(max(remaining_buffered * vm, eps))
            lam_m = min(lam_m, solvency_c / max(1.0 - m, eps))
            logk_minus = _update_logwealth(logk_minus, lam_m, -y)

        sum_sq += y * y

    logp_plus = min(log_alpha + logk_plus, 0.0)
    logp_minus = min(log_alpha + logk_minus, 0.0)
    return max(logp_plus - log_u_plus, logp_minus - log_u_minus)


@njit(cache=True, parallel=True)
def _star_scores_grid(
    x: np.ndarray,
    ms: np.ndarray,
    delta: float,
    regularizer: float,
    solvency_c: float,
    eps: float,
) -> np.ndarray:
    out = np.empty(ms.size)
    for j in prange(ms.size):
        out[j] = _star_score_scalar(
            x, ms[j], delta, regularizer, solvency_c, eps
        )
    return out


@njit(cache=True, parallel=True)
def _probit_scores_grid(
    x: np.ndarray,
    ms: np.ndarray,
    delta: float,
    u_plus: float,
    u_minus: float,
    regularizer: float,
    solvency_c: float,
    eps: float,
) -> np.ndarray:
    out = np.empty(ms.size)
    for j in prange(ms.size):
        out[j] = _probit_score_scalar(
            x, ms[j], delta, u_plus, u_minus,
            regularizer, solvency_c, eps
        )
    return out


def _bisect_crossing(
    score_fn: Callable[[float], float],
    left: float,
    right: float,
    steps: int,
) -> float:
    fl = score_fn(left)
    fr = score_fn(right)
    if not (np.isfinite(fl) and np.isfinite(fr)):
        return 0.5 * (left + right)
    if fl == 0.0:
        return left
    if fr == 0.0:
        return right
    if fl * fr > 0:
        # This should not happen after the grid bracketing. Return the midpoint
        # rather than silently extrapolating.
        return 0.5 * (left + right)

    lo, hi = left, right
    flo = fl
    for _ in range(steps):
        mid = 0.5 * (lo + hi)
        fm = score_fn(mid)
        if flo * fm <= 0:
            hi = mid
        else:
            lo = mid
            flo = fm
    return 0.5 * (lo + hi)


def invert_center_component(
    x: np.ndarray,
    method: str,
    delta: float,
    grid_size: int,
    bisection_steps: int,
    regularizer: float,
    solvency_c: float,
    eps: float,
    u_plus: float | None = None,
    u_minus: float | None = None,
) -> tuple[float, float, bool]:
    """
    Return the accepted component containing the sample mean.

    The coarse scan covers [0,1] and is augmented by the sample mean. The two
    adjacent sign changes are then refined by bisection.
    """
    mean = float(np.mean(x))
    base = np.linspace(eps, 1.0 - eps, grid_size)
    ms = np.unique(np.concatenate((base, np.array([min(max(mean, eps), 1-eps)]))))

    if method == "star":
        scores = _star_scores_grid(
            x, ms, delta, regularizer, solvency_c, eps
        )
        scalar = lambda m: float(
            _star_score_scalar(x, m, delta, regularizer, solvency_c, eps)
        )
    elif method == "probit":
        assert u_plus is not None and u_minus is not None
        scores = _probit_scores_grid(
            x, ms, delta, u_plus, u_minus,
            regularizer, solvency_c, eps
        )
        scalar = lambda m: float(
            _probit_score_scalar(
                x, m, delta, u_plus, u_minus,
                regularizer, solvency_c, eps
            )
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    accepted = scores < 0.0
    center_idx = int(np.argmin(np.abs(ms - mean)))

    if not accepted[center_idx]:
        return mean, mean, True

    left_idx = center_idx
    while left_idx > 0 and accepted[left_idx - 1]:
        left_idx -= 1

    right_idx = center_idx
    while right_idx + 1 < ms.size and accepted[right_idx + 1]:
        right_idx += 1

    if left_idx == 0:
        lower = 0.0
    else:
        lower = _bisect_crossing(
            scalar, float(ms[left_idx - 1]), float(ms[left_idx]),
            bisection_steps
        )

    if right_idx == ms.size - 1:
        upper = 1.0
    else:
        upper = _bisect_crossing(
            scalar, float(ms[right_idx]), float(ms[right_idx + 1]),
            bisection_steps
        )

    lower = float(np.clip(lower, 0.0, 1.0))
    upper = float(np.clip(upper, lower, 1.0))
    return lower, upper, False


# ---------------------------------------------------------------------------
# Gaffke interval
# ---------------------------------------------------------------------------

def _is_binary(x: np.ndarray) -> bool:
    return bool(np.all((x == 0.0) | (x == 1.0)))


def _gaffke_binary_ci(x: np.ndarray, delta: float) -> tuple[float, float]:
    """Exact Clopper-Pearson form of the equal-tail Gaffke interval."""
    n = x.size
    k = int(np.sum(x))
    q = delta / 2.0

    lower = 0.0 if k == 0 else float(beta_dist.ppf(q, k, n + 1 - k))
    upper = 1.0 if k == n else float(beta_dist.ppf(1.0 - q, k + 1, n - k))
    return lower, upper


def _dirichlet_average_quantile_bspline(
    knots: np.ndarray,
    q: float,
) -> float:
    """
    Exact quantile of sum a_i D_i for D~Dirichlet(1,...,1), using the
    normalized B-spline density.
    """
    a = np.sort(np.asarray(knots, dtype=np.float64))
    lo = float(a[0])
    hi = float(a[-1])

    if hi - lo <= 1.0e-15:
        return lo

    # A constant sample creates repeated knots and has a simple closed form.
    # More generally, Bernoulli samples are handled before this function.
    if np.var(a) <= 1.0e-30:
        return lo

    spline = BSpline.basis_element(a, extrapolate=False)
    antiderivative = spline.antiderivative()
    degree_plus_one = a.size - 1
    normalizer = degree_plus_one / (hi - lo)
    base = float(antiderivative(lo))

    def cdf(t: float) -> float:
        if t <= lo:
            return 0.0
        if t >= hi:
            return 1.0
        val = normalizer * (float(antiderivative(t)) - base)
        return float(np.clip(val, 0.0, 1.0))

    return float(
        brentq(
            lambda t: cdf(t) - q,
            lo,
            hi,
            xtol=2.0e-13,
            rtol=2.0e-13,
            maxiter=100,
        )
    )


def _dirichlet_average_quantile_cf(
    knots: np.ndarray,
    q: float,
) -> float:
    """
    Fourth-order Cornish-Fisher approximation using exact centered moments
    of a uniform-Dirichlet average.

    If m is the number of knots and b_i = a_i - mean(a), then:
        Var(W) = sum b_i^2 / [m(m+1)]
        E[(W-EW)^3] = 2 sum b_i^3 / [m(m+1)(m+2)]
        E[(W-EW)^4] =
            [3(sum b_i^2)^2 + 6 sum b_i^4]
            / [m(m+1)(m+2)(m+3)]
    """
    a = np.asarray(knots, dtype=np.float64)
    m = float(a.size)
    mu = float(np.mean(a))
    b = a - mu

    p2 = float(np.dot(b, b))
    if p2 <= 1.0e-30:
        return mu

    p3 = float(np.sum(b * b * b))
    p4 = float(np.dot(b * b, b * b))

    var = p2 / (m * (m + 1.0))
    sd = math.sqrt(max(var, 0.0))
    cm3 = 2.0 * p3 / (m * (m + 1.0) * (m + 2.0))
    cm4 = (
        3.0 * p2 * p2 + 6.0 * p4
    ) / (m * (m + 1.0) * (m + 2.0) * (m + 3.0))

    skew = cm3 / (sd ** 3)
    excess = cm4 / (sd ** 4) - 3.0

    z = float(norm.ppf(q))
    zcf = (
        z
        + (skew / 6.0) * (z * z - 1.0)
        + (excess / 24.0) * (z ** 3 - 3.0 * z)
        - (skew * skew / 36.0) * (2.0 * z ** 3 - 5.0 * z)
    )
    return float(np.clip(mu + sd * zcf, np.min(a), np.max(a)))


def gaffke_ci(
    x: np.ndarray,
    delta: float,
    exact_cutoff: int,
    force_cf: bool = False,
) -> tuple[float, float, str]:
    """
    Equal-tail Gaffke confidence interval.
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    q = delta / 2.0

    if _is_binary(x):
        lower, upper = _gaffke_binary_ci(x, delta)
        return lower, upper, "beta-exact"

    # Exact degenerate-sample formulas from the paper.
    if float(np.max(x) - np.min(x)) <= 1.0e-15:
        mu = float(x[0])
        root = q ** (1.0 / n)
        lower = mu * root
        upper = 1.0 - (1.0 - mu) * root
        return lower, upper, "degenerate-exact"

    lower_knots = np.concatenate((np.array([0.0]), x))
    upper_knots = np.concatenate((x, np.array([1.0])))

    if (not force_cf) and n <= exact_cutoff:
        lower = _dirichlet_average_quantile_bspline(lower_knots, q)
        upper = _dirichlet_average_quantile_bspline(
            upper_knots, 1.0 - q
        )
        backend = "bspline-exact"
    else:
        lower = _dirichlet_average_quantile_cf(lower_knots, q)
        upper = _dirichlet_average_quantile_cf(
            upper_knots, 1.0 - q
        )
        backend = "cornish-fisher"

    return float(lower), float(upper), backend


# ---------------------------------------------------------------------------
# Experiment design
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DistributionSpec:
    name: str
    mean: float
    variance: float
    sampler: Callable[[np.random.Generator, int], np.ndarray]


DISTRIBUTIONS = [
    DistributionSpec(
        "Beta(2,2)", 0.5, 2*2 / ((2+2)**2 * (2+2+1)),
        lambda rng, n: rng.beta(2.0, 2.0, n),
    ),
    DistributionSpec(
        "Beta(1,5)", 1/6, 1*5 / ((1+5)**2 * (1+5+1)),
        lambda rng, n: rng.beta(1.0, 5.0, n),
    ),
    DistributionSpec(
        "Bernoulli(0.5)", 0.5, 0.25,
        lambda rng, n: rng.binomial(1, 0.5, n).astype(np.float64),
    ),
    DistributionSpec(
        "Uniform(0,1)", 0.5, 1/12,
        lambda rng, n: rng.random(n),
    ),
    DistributionSpec(
        "Beta(0.5,0.5)", 0.5,
        0.5*0.5 / ((0.5+0.5)**2 * (0.5+0.5+1)),
        lambda rng, n: rng.beta(0.5, 0.5, n),
    ),
    DistributionSpec(
        "Bernoulli(0.1)", 0.1, 0.09,
        lambda rng, n: rng.binomial(1, 0.1, n).astype(np.float64),
    ),
    DistributionSpec(
        "Beta(50,50)", 0.5, 1/404,
        lambda rng, n: rng.beta(50.0, 50.0, n),
    ),
    DistributionSpec(
        "Beta(20,80)", 0.2, 20*80 / ((20+80)**2 * (20+80+1)),
        lambda rng, n: rng.beta(20.0, 80.0, n),
    ),
    DistributionSpec(
        "Uniform(0.45,0.55)", 0.5, 0.1**2/12,
        lambda rng, n: rng.uniform(0.45, 0.55, n),
    ),
]


PAPER_SAMPLE_SIZES = [
    10, 50, 100, 500, 1_000, 5_000, 10_000,
    50_000, 100_000, 500_000, 1_000_000,
]


def paper_reps(n: int) -> int:
    if n <= 10_000:
        return 50
    if n <= 100_000:
        return 30
    return 20


def _path_max_n_for_rep(rep: int, sample_sizes: list[int], reps_by_n: dict[int, int]) -> int:
    eligible = [n for n in sample_sizes if rep < reps_by_n[n]]
    return max(eligible) if eligible else 0


def _make_seed(base_seed: int, dist_idx: int, rep: int) -> np.random.SeedSequence:
    return np.random.SeedSequence([base_seed, dist_idx, rep])


def run_experiment(args: argparse.Namespace) -> pd.DataFrame:
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    if args.quick:
        sample_sizes = [50, 100, 500, 1_000]
        reps_by_n = {n: args.quick_reps for n in sample_sizes}
    else:
        sample_sizes = PAPER_SAMPLE_SIZES
        reps_by_n = {n: paper_reps(n) for n in sample_sizes}

    max_reps = max(reps_by_n.values())
    rows: list[dict] = []
    start_all = time.time()

    # Trigger JIT compilation on a tiny sample before timing the real work.
    warm = np.array([0.2, 0.8], dtype=np.float64)
    _star_score_scalar(warm, 0.5, args.delta, args.regularizer, args.solvency_c, args.eps)
    _probit_score_scalar(
        warm, 0.5, args.delta, 0.3, 0.7,
        args.regularizer, args.solvency_c, args.eps
    )

    for dist_idx, dist in enumerate(DISTRIBUTIONS):
        print(f"\n=== {dist.name} ===", flush=True)

        for rep in range(max_reps):
            max_n = _path_max_n_for_rep(rep, sample_sizes, reps_by_n)
            if max_n == 0:
                continue

            ss = _make_seed(args.seed, dist_idx, rep)
            rng_data, rng_aux = [
                np.random.default_rng(s) for s in ss.spawn(2)
            ]
            path = np.asarray(dist.sampler(rng_data, max_n), dtype=np.float64)

            for n in sample_sizes:
                if rep >= reps_by_n[n]:
                    continue

                x = np.ascontiguousarray(path[:n], dtype=np.float64)
                # Fixed across candidate means for this dataset and arm.
                u_plus = float(rng_aux.random())
                u_minus = float(rng_aux.random())

                t0 = time.time()
                l_star, u_star, empty_star = invert_center_component(
                    x=x,
                    method="star",
                    delta=args.delta,
                    grid_size=args.grid_size,
                    bisection_steps=args.bisection_steps,
                    regularizer=args.regularizer,
                    solvency_c=args.solvency_c,
                    eps=args.eps,
                )
                star_sec = time.time() - t0

                t0 = time.time()
                l_probit, u_probit, empty_probit = invert_center_component(
                    x=x,
                    method="probit",
                    delta=args.delta,
                    grid_size=args.grid_size,
                    bisection_steps=args.bisection_steps,
                    regularizer=args.regularizer,
                    solvency_c=args.solvency_c,
                    eps=args.eps,
                    u_plus=u_plus,
                    u_minus=u_minus,
                )
                probit_sec = time.time() - t0

                t0 = time.time()
                l_gaffke, u_gaffke, g_backend = gaffke_ci(
                    x,
                    delta=args.delta,
                    exact_cutoff=args.gaffke_exact_cutoff,
                    force_cf=args.gaffke_force_cf,
                )
                gaffke_sec = time.time() - t0

                method_results = [
                    ("STaR", l_star, u_star, empty_star, "direct-two-arm", star_sec),
                    ("Efficient betting", l_probit, u_probit, empty_probit, "randomized", probit_sec),
                    ("Gaffke", l_gaffke, u_gaffke, False, g_backend, gaffke_sec),
                ]

                for method, lower, upper, empty, backend, seconds in method_results:
                    width = max(upper - lower, 0.0)
                    rows.append({
                        "distribution": dist.name,
                        "true_mean": dist.mean,
                        "true_variance": dist.variance,
                        "n": n,
                        "rep": rep,
                        "method": method,
                        "lower": lower,
                        "upper": upper,
                        "width": width,
                        "sqrt_n_width": math.sqrt(n) * width,
                        "normalized_halfwidth": (
                            math.sqrt(n) * width / (2.0 * math.sqrt(dist.variance))
                            if dist.variance > 0 else np.nan
                        ),
                        "covered": lower <= dist.mean <= upper,
                        "empty_center_component": empty,
                        "backend": backend,
                        "runtime_seconds": seconds,
                        "u_plus": u_plus if method == "Efficient betting" else np.nan,
                        "u_minus": u_minus if method == "Efficient betting" else np.nan,
                    })

                if args.progress_every > 0 and (rep + 1) % args.progress_every == 0:
                    elapsed = time.time() - start_all
                    print(
                        f"n={n:>8,d}, rep={rep+1:>3d}/{reps_by_n[n]}, "
                        f"elapsed={elapsed/60:.1f} min",
                        flush=True,
                    )

            # Save a checkpoint after every path.
            pd.DataFrame(rows).to_csv(output / "results_checkpoint.csv", index=False)

    df = pd.DataFrame(rows)
    df.to_csv(output / "results.csv", index=False)

    summary = (
        df.groupby(["distribution", "n", "method"], as_index=False)
        .agg(
            replications=("width", "size"),
            coverage=("covered", "mean"),
            mean_width=("width", "mean"),
            median_width=("width", "median"),
            q10_width=("width", lambda z: np.quantile(z, 0.10)),
            q90_width=("width", lambda z: np.quantile(z, 0.90)),
            mean_sqrt_n_width=("sqrt_n_width", "mean"),
            mean_normalized_halfwidth=("normalized_halfwidth", "mean"),
            mean_runtime_seconds=("runtime_seconds", "mean"),
            empty_rate=("empty_center_component", "mean"),
        )
    )
    summary.to_csv(output / "summary.csv", index=False)

    config = vars(args).copy()
    config["sample_sizes"] = sample_sizes
    config["reps_by_n"] = reps_by_n
    with open(output / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    make_plots(df, output, args.delta)
    print(f"\nSaved results to {output.resolve()}")
    return df


def _slug(text: str) -> str:
    return (
        text.lower()
        .replace("(", "_")
        .replace(")", "")
        .replace(",", "_")
        .replace(".", "p")
        .replace("/", "_")
        .replace(" ", "_")
    )


def make_plots(df: pd.DataFrame, output: Path, delta: float) -> None:
    plot_dir = output / "plots"
    plot_dir.mkdir(exist_ok=True)

    for dist_name, ddf in df.groupby("distribution"):
        grouped = (
            ddf.groupby(["n", "method"])["sqrt_n_width"]
            .agg(
                mean="mean",
                q10=lambda z: np.quantile(z, 0.10),
                q90=lambda z: np.quantile(z, 0.90),
            )
            .reset_index()
        )

        fig, ax = plt.subplots(figsize=(7.0, 4.5))
        for method, mdf in grouped.groupby("method"):
            mdf = mdf.sort_values("n")
            ax.plot(mdf["n"], mdf["mean"], marker="o", label=method)
            ax.fill_between(
                mdf["n"], mdf["q10"], mdf["q90"], alpha=0.15
            )
        ax.set_xscale("log")
        ax.set_xlabel("sample size n")
        ax.set_ylabel(r"$\sqrt{n}\,\mathrm{width}$")
        ax.set_title(f"{dist_name}: scaled CI width, confidence={1-delta:.3f}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(plot_dir / f"scaled_width_{_slug(dist_name)}.png", dpi=180)
        plt.close(fig)

        raw = (
            ddf.groupby(["n", "method"])["width"]
            .agg(
                mean="mean",
                q10=lambda z: np.quantile(z, 0.10),
                q90=lambda z: np.quantile(z, 0.90),
            )
            .reset_index()
        )
        fig, ax = plt.subplots(figsize=(7.0, 4.5))
        for method, mdf in raw.groupby("method"):
            mdf = mdf.sort_values("n")
            ax.plot(mdf["n"], mdf["mean"], marker="o", label=method)
            ax.fill_between(
                mdf["n"], mdf["q10"], mdf["q90"], alpha=0.15
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("sample size n")
        ax.set_ylabel("CI width")
        ax.set_title(f"{dist_name}: raw CI width, confidence={1-delta:.3f}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(plot_dir / f"raw_width_{_slug(dist_name)}.png", dpi=180)
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare STaR, Efficient betting, and Gaffke CIs."
    )
    parser.add_argument("--output", default="star_probit_gaffke_results")
    parser.add_argument("--delta", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=20260722)

    parser.add_argument(
        "--grid-size", type=int, default=129,
        help="Global candidate-mean scan size; the sample mean is added."
    )
    parser.add_argument("--bisection-steps", type=int, default=24)
    parser.add_argument("--regularizer", type=float, default=1.0)
    parser.add_argument("--solvency-c", type=float, default=1.0)
    parser.add_argument("--eps", type=float, default=1.0e-6)

    parser.add_argument(
        "--gaffke-exact-cutoff", type=int, default=5_000,
        help="Use exact B-spline Dirichlet quantiles up to this n."
    )
    parser.add_argument(
        "--gaffke-force-cf", action="store_true",
        help="Use Cornish-Fisher Gaffke quantiles for all nonbinary samples."
    )

    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--quick-reps", type=int, default=5)
    parser.add_argument("--progress-every", type=int, default=5)
    return parser.parse_args()


if __name__ == "__main__":
    run_experiment(parse_args())
