#!/usr/bin/env python3
"""
betting.py

Compares the Lebesgue measure of (1-delta)-confidence intervals from three
testing-by-betting procedures for bounded data in [0, 1]:

  M_{n,inf}  : product martingale (Waudby-Smith & Ramdas 2024, Sec. 2.3)
  M_bar_{n,2}: legacy stopped polynomial martingale (Construction 2)
  M_heat     : continuation-value hedge of the terminal Bentkus payoff

For three bounded distributions the script estimates, across num_sims datasets,
the expected CI width (Lebesgue measure approximated on a fine grid of m in [0,1]).
"""

import os
import math
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt
from tqdm import tqdm
from numba import njit


# ------------------------------------------------------------------
# 1.  Hyperparameters for M_bar_{n,2}
# ------------------------------------------------------------------

def I2(lam):
    """
    Closed-form E[(Z - lam)_+^2] for Z ~ N(0,1):
        I2(lam) = (1 + lam^2) * Phi(-lam) - lam * phi(lam)
    where Phi is the standard normal CDF and phi its density.
    """
    return (1.0 + lam**2) * norm.sf(lam) - lam * norm.pdf(lam)


def U2(lam, delta):
    """Threshold function U_{2,delta}(lam) = lam + sqrt(I2(lam) / delta)."""
    return lam + np.sqrt(max(I2(lam), 0.0) / delta)


def get_optimal_lambda(delta):
    """
    Numerically minimize U_{2,delta}(lam) over lam >= 0.
    Returns (lam_star, I2(lam_star)).
    """
    res = minimize_scalar(lambda l: U2(l, delta), bounds=(1e-6, 20.0), method="bounded")
    lam_star = float(res.x)
    return lam_star, float(I2(lam_star))


@njit
def _normal_cdf(x):
    """Standard normal CDF, in a form supported by numba."""
    return 0.5 * math.erfc(-x / np.sqrt(2.0))


@njit
def _normal_pdf(x):
    """Standard normal density, in a form supported by numba."""
    return np.exp(-0.5 * x * x) / np.sqrt(2.0 * np.pi)


@njit
def heat_payoff_value(x, remaining_variance, strike):
    r"""Gaussian continuation value E[(x + sqrt(v) Z - strike)_+^2]."""
    a = x - strike
    if remaining_variance <= 1e-15:
        return max(a, 0.0) ** 2
    sd = np.sqrt(remaining_variance)
    d = a / sd
    return (
        (a * a + remaining_variance) * _normal_cdf(d)
        + a * sd * _normal_pdf(d)
    )


@njit
def heat_payoff_delta(x, remaining_variance, strike):
    r"""Delta of the Gaussian continuation value.

    If

        u(v, x) = E[(x + sqrt(v) Z - strike)_+^2],

    this returns ``partial_x u(v, x)``.  Unlike ``2 (x-strike)_+``, this
    coefficient prices the uncertainty that is still to arrive.  Consequently
    the limiting stochastic integral starts at ``I2(strike)`` and ends at the
    nonnegative Bentkus payoff instead of consuming its capital at a random
    time before the experiment is over.
    """
    a = x - strike
    if remaining_variance <= 1e-15:
        return 2.0 * max(a, 0.0)
    sd = np.sqrt(remaining_variance)
    d = a / sd
    return 2.0 * (a * _normal_cdf(d) + sd * _normal_pdf(d))


# ------------------------------------------------------------------
# 2.  M_{n,inf}: product martingale (Waudby-Smith & Ramdas 2024)
# ------------------------------------------------------------------

@njit
def compute_M_inf(X, m, delta, c=0.5):
    """
    M_{n,inf}(m) = 0.5 * M^+(m) + 0.5 * M^-(m), where

      M^+(m) = prod_{i=1}^n [1 + lam_i^+(m) * (X_i - m)]
      M^-(m) = prod_{i=1}^n [1 - lam_i^-(m) * (X_i - m)]

      lam_i^+(m) = min(tilde_lam_i, c / m)
      lam_i^-(m) = min(tilde_lam_i, c / (1-m))
      tilde_lam_i = sqrt(2 * log(2/delta) / (n * sigma_hat_{i-1}^2))

      sigma_hat_{i-1}^2 = (1/4 + sum_{j<i}(X_j - mu_hat_{j-1})^2) / i
      mu_hat_{i-1}      = (1/2 + sum_{j<i} X_j) / i

    Under H_0: m = mu, both M^+ and M^- are nonneg martingales starting at 1,
    so M_{n,inf} is a nonneg martingale starting at 1.
    CI: {m : M_{n,inf}(m) < 1/delta}.
    """
    n = len(X)
    M_plus  = 1.0
    M_minus = 1.0
    sum_x = 0.0
    pred_sq = 0.0   # sum of (X_i - mu_hat_{i-1})^2
    log2d   = np.log(2.0 / delta)

    for i in range(n):
        # Predictable variance estimate sigma_hat_{i-1}^2
        mean_hat = (0.5 + sum_x) / (1.0 + i)
        var_hat = (0.25 + pred_sq) / (1.0 + i)
        tilde_lam = np.sqrt(2.0 * log2d / (n * var_hat))

        # Clip to keep each factor nonneg (worst case: X_i=0 for M^+, X_i=1 for M^-)
        lam_plus  = min(tilde_lam, c / (m       + 1e-14))
        lam_minus = min(tilde_lam, c / (1.0 - m + 1e-14))

        M_plus  *= 1.0 + lam_plus  * (X[i] - m)
        M_minus *= 1.0 - lam_minus * (X[i] - m)

        residual = X[i] - mean_hat
        sum_x += X[i]
        pred_sq += residual * residual

    return 0.5 * M_plus + 0.5 * M_minus


# ------------------------------------------------------------------
# 3.  M_{n,2}: Construction 2 (one fixed ordering)
# ------------------------------------------------------------------

@njit
def compute_Mt2_path(X, m, lam_per_step, I2_val, c=0.5):
    """
    Single-path M_{n,2}(m) = 0.5 * M^+(m) + 0.5 * M^-(m).

    Initialization: M_0^+ = M_0^- = I2_val  (= I2(lam^*))

    Recursive update for i = 0, ..., n-1  (step i+1 in 1-indexed notation):

      W^+  = (S^+ - Lambda^+)_+   [arm-specific positive part, computed before update]
      W^-  = (S^- - Lambda^-)_+

      tilde_gamma = 1 / (sqrt(n) * sigma_hat_{i-1})

      gamma^+ = min(tilde_gamma, c * M^+ / (2 * W^+ * m))       if W^+, m > 0
      gamma^- = min(tilde_gamma, c * M^- / (2 * W^- * (1-m)))   if W^-, (1-m) > 0

      M^+      += 2 * gamma^+ * (X[i] - m) * W^+
      M^-      += 2 * gamma^- * (m - X[i]) * W^-

      S^+      += gamma^+ * (X[i] - m)
      S^-      += gamma^- * (m - X[i])
      Lambda^+ += lam_per_step * (gamma^+ / tilde_gamma)   [rescaled per paper eq. 5.5]
      Lambda^- += lam_per_step * (gamma^- / tilde_gamma)

    Clipping ensures M^+, M^- >= 0 at every step (worst case X_i = 0 for M^+,
    X_i = 1 for M^-).  Lambda is rescaled arm-by-arm so the shift/scale ratio
    gamma/lambda stays constant when clipping is active.

    CI: {m : M_bar_{n,2}(m) < I2_val / delta}.
    """
    n            = len(X)
    sqrt_n       = np.sqrt(float(n))
    S_plus       = 0.0
    S_minus      = 0.0
    Lambda_plus  = 0.0
    Lambda_minus = 0.0
    M_plus       = I2_val
    M_minus      = I2_val
    sum_x         = 0.0
    pred_sq       = 0.0

    for i in range(n):
        # Predictable variance estimate sigma_hat_{i-1}^2
        mean_hat    = (0.5 + sum_x) / (1.0 + i)
        var_hat     = (0.25 + pred_sq) / (1.0 + i)
        tilde_gamma = 1.0 / (sqrt_n * np.sqrt(var_hat))

        Wp = max(S_plus  - Lambda_plus,  0.0)   # (S^+_{i-1} - Lambda^+_{i-1})_+
        Wm = max(S_minus - Lambda_minus, 0.0)   # (S^-_{i-1} - Lambda^-_{i-1})_+

        # Clip gamma^+ to keep M^+ nonneg
        if Wp > 1e-14 and m > 1e-14:
            gamma_plus = min(tilde_gamma, c * M_plus / (2.0 * Wp * m))
        else:
            gamma_plus = tilde_gamma

        # Clip gamma^- to keep M^- nonneg
        one_m = 1.0 - m
        if Wm > 1e-14 and one_m > 1e-14:
            gamma_minus = min(tilde_gamma, c * M_minus / (2.0 * Wm * one_m))
        else:
            gamma_minus = tilde_gamma

        xi = X[i] - m   # signed deviation

        M_plus  += 2.0 * gamma_plus  *  xi * Wp
        M_minus += 2.0 * gamma_minus * (-xi) * Wm

        S_plus       += gamma_plus  *  xi
        S_minus      += gamma_minus * (-xi)
        Lambda_plus  += lam_per_step * (gamma_plus  / tilde_gamma)
        Lambda_minus += lam_per_step * (gamma_minus / tilde_gamma)

        residual = X[i] - mean_hat
        sum_x += X[i]
        pred_sq += residual * residual

    return 0.5 * M_plus + 0.5 * M_minus


@njit
def compute_M_bar_n2(X, m, lam_per_step, I2_val, B):
    """
    M_bar_{n,2}(m) = (1/B) * sum_{b=1}^B M_{n,2}(m ; X_{pi_b})

    Rao-Blackwellization: averaging over B uniform random permutations
    integrates out path-dependent noise while preserving the e-value property:
        E_{H_0}[M_bar_{n,2}(m)] = I2_val  =>  CI at level delta via Markov.
    """
    total = 0.0
    for _ in range(B):
        total += compute_Mt2_path(np.random.permutation(X), m, lam_per_step, I2_val)
    return total / B


@njit
def compute_M_bar_n2_fixed(X_perms, m, lam_per_step, I2_val):
    """Legacy permutation average with common randomness across candidate m."""
    B = X_perms.shape[0]
    total = 0.0
    for b in range(B):
        total += compute_Mt2_path(X_perms[b], m, lam_per_step, I2_val)
    return total / B


# ------------------------------------------------------------------
# 3a. Corrected construction: hedge the terminal Bentkus payoff
# ------------------------------------------------------------------

@njit
def compute_M_heat_path(X, m, strike, initial_wealth, c=1.0):
    r"""Nonnegative delta hedge for the terminal squared-hinge payoff.

    The legacy construction bets with ``2 (S_t - Lambda_t)_+``, the delta of
    the payoff *at the current time*.  With the very small initial capital
    ``I2(strike)``, that martingale hits its solvency constraint with positive
    probability even in the Brownian limit.  Its effective clock then stops,
    invalidating the advertised full-horizon asymptotics.

    Here the predictable bet is instead the delta of

        u(v, S_t) = E[(S_t + sqrt(v) Z - strike)_+^2],

    where ``v`` is the variance still to arrive.  In the Brownian limit,
    Ito's formula gives

        I2(strike) + integral u_x(1-t, W_t) dW_t
            = (W_1 - strike)_+^2 >= 0.

    We retain an exact finite-sample solvency clip.  It clips the amount
    invested, not the standardized score, so a rare clip does not stop or
    time-change the statistical signal.
    """
    n = len(X)
    sqrt_n = np.sqrt(float(n))
    S_plus = 0.0
    S_minus = 0.0
    M_plus = initial_wealth
    M_minus = initial_wealth
    sum_x = 0.0
    pred_sq = 0.0

    for i in range(n):
        mean_hat = (0.5 + sum_x) / (1.0 + i)
        var_hat = (0.25 + pred_sq) / (1.0 + i)
        gamma = 1.0 / (sqrt_n * np.sqrt(var_hat))
        remaining_variance = 1.0 - i / float(n)

        beta_plus = gamma * heat_payoff_delta(
            S_plus, remaining_variance, strike
        )
        beta_minus = gamma * heat_payoff_delta(
            S_minus, remaining_variance, strike
        )

        # M + beta (X-m) is nonnegative for every X in [0,1].
        if m > 1e-14:
            beta_plus = min(beta_plus, c * M_plus / m)
        one_m = 1.0 - m
        if one_m > 1e-14:
            beta_minus = min(beta_minus, c * M_minus / one_m)

        xi = X[i] - m
        M_plus += beta_plus * xi
        M_minus -= beta_minus * xi

        # The score follows the full predictable scale even if capital clips.
        S_plus += gamma * xi
        S_minus -= gamma * xi

        residual = X[i] - mean_hat
        sum_x += X[i]
        pred_sq += residual * residual

    return 0.5 * (M_plus + M_minus)


@njit
def compute_M_bar_heat(X_perms, m, strike, initial_wealth):
    """Average the corrected construction over a fixed set of permutations."""
    B = X_perms.shape[0]
    total = 0.0
    for b in range(B):
        total += compute_M_heat_path(
            X_perms[b], m, strike, initial_wealth
        )
    return total / B


@njit
def heat_clip_fractions(X, m, strike, initial_wealth, c=1.0):
    """Clip fractions for the plus and minus heat-flow hedges."""
    n = len(X)
    sqrt_n = np.sqrt(float(n))
    S_plus = 0.0
    S_minus = 0.0
    M_plus = initial_wealth
    M_minus = initial_wealth
    sum_x = 0.0
    pred_sq = 0.0
    clipped_plus = 0
    clipped_minus = 0

    for i in range(n):
        mean_hat = (0.5 + sum_x) / (1.0 + i)
        var_hat = (0.25 + pred_sq) / (1.0 + i)
        gamma = 1.0 / (sqrt_n * np.sqrt(var_hat))
        remaining_variance = 1.0 - i / float(n)
        beta_plus = gamma * heat_payoff_delta(
            S_plus, remaining_variance, strike
        )
        beta_minus = gamma * heat_payoff_delta(
            S_minus, remaining_variance, strike
        )

        if m > 1e-14:
            cap_plus = c * M_plus / m
            if beta_plus > cap_plus:
                beta_plus = cap_plus
                clipped_plus += 1
        one_m = 1.0 - m
        if one_m > 1e-14:
            cap_minus = c * M_minus / one_m
            if beta_minus > cap_minus:
                beta_minus = cap_minus
                clipped_minus += 1

        xi = X[i] - m
        M_plus += beta_plus * xi
        M_minus -= beta_minus * xi
        S_plus += gamma * xi
        S_minus -= gamma * xi

        residual = X[i] - mean_hat
        sum_x += X[i]
        pred_sq += residual * residual

    return clipped_plus / float(n), clipped_minus / float(n)


def fixed_permutations(X, B, rng):
    """Generate common random permutations to reuse for every candidate m."""
    return np.asarray([rng.permutation(X) for _ in range(B)])


def _interval_component(statistic, threshold, center, scan_points=24, iterations=35):
    """Invert a continuous U-shaped statistic around an accepted center.

    The short outward scan detects the nearest crossing on each side; bisection
    then avoids the ``sqrt(n)``-amplified discretization error of a fixed grid.
    """
    cache = {}

    def accepted(m):
        key = float(m)
        if key not in cache:
            cache[key] = float(statistic(key)) < threshold
        return cache[key]

    if not accepted(center):
        raise ValueError("the supplied center is not in the confidence set")

    def boundary(outer):
        previous = center
        bracket = None
        for current in np.linspace(center, outer, scan_points + 1)[1:]:
            current = float(current)
            if not accepted(current):
                bracket = (previous, current)
                break
            previous = current
        if bracket is None:
            return float(outer)

        inside, outside = bracket
        for _ in range(iterations):
            mid = 0.5 * (inside + outside)
            if accepted(mid):
                inside = mid
            else:
                outside = mid
        return 0.5 * (inside + outside)

    return boundary(0.0), boundary(1.0)


def heat_ci_endpoints(X, delta, strike, initial_wealth, B=20, rng=None):
    """Endpoints of the corrected CI, using common permutation randomness."""
    if rng is None:
        rng = np.random.default_rng()
    X_perms = fixed_permutations(np.asarray(X), B, rng)
    threshold = initial_wealth / delta
    center = float(np.mean(X))

    def statistic(m):
        return compute_M_bar_heat(X_perms, m, strike, initial_wealth)

    return _interval_component(statistic, threshold, center)


def legacy_ci_endpoints(X, delta, strike, initial_wealth, B=20, rng=None):
    """Endpoints of the stopped legacy CI, using common permutation randomness."""
    if rng is None:
        rng = np.random.default_rng()
    X = np.asarray(X)
    X_perms = fixed_permutations(X, B, rng)
    threshold = initial_wealth / delta
    lam_per_step = strike / len(X)
    center = float(np.mean(X))

    def statistic(m):
        return compute_M_bar_n2_fixed(
            X_perms, m, lam_per_step, initial_wealth
        )

    return _interval_component(statistic, threshold, center)





@njit
def clip_fraction_path(X, m, lam_per_step, I2_val, c=0.5):
    """
    Runs the same loop as compute_Mt2_path but returns the fraction of the 2n
    individual bets (n for the + arm, n for the - arm) where the clip condition
    was active, i.e. where tilde_gamma > c_i^{+/-} and gamma was reduced.
    """
    n            = len(X)
    sqrt_n       = np.sqrt(float(n))
    S_plus       = 0.0
    S_minus      = 0.0
    Lambda_plus  = 0.0
    Lambda_minus = 0.0
    M_plus       = I2_val
    M_minus      = I2_val
    sum_x         = 0.0
    pred_sq       = 0.0
    n_clipped    = 0

    for i in range(n):
        mean_hat    = (0.5 + sum_x) / (1.0 + i)
        var_hat     = (0.25 + pred_sq) / (1.0 + i)
        tilde_gamma = 1.0 / (sqrt_n * np.sqrt(var_hat))

        Wp = max(S_plus  - Lambda_plus,  0.0)
        Wm = max(S_minus - Lambda_minus, 0.0)

        if Wp > 1e-14 and m > 1e-14:
            c_plus     = c * M_plus / (2.0 * Wp * m)
            gamma_plus = min(tilde_gamma, c_plus)
            if tilde_gamma > c_plus:
                n_clipped += 1
        else:
            gamma_plus = tilde_gamma

        one_m = 1.0 - m
        if Wm > 1e-14 and one_m > 1e-14:
            c_minus     = c * M_minus / (2.0 * Wm * one_m)
            gamma_minus = min(tilde_gamma, c_minus)
            if tilde_gamma > c_minus:
                n_clipped += 1
        else:
            gamma_minus = tilde_gamma

        xi           = X[i] - m
        M_plus      += 2.0 * gamma_plus  *  xi * Wp
        M_minus     += 2.0 * gamma_minus * (-xi) * Wm
        S_plus      += gamma_plus  *  xi
        S_minus     += gamma_minus * (-xi)
        Lambda_plus  += lam_per_step * (gamma_plus  / tilde_gamma)
        Lambda_minus += lam_per_step * (gamma_minus / tilde_gamma)

        residual = X[i] - mean_hat
        sum_x += X[i]
        pred_sq += residual * residual

    return n_clipped / (2.0 * n)   # denominator: n bets per arm


@njit
def clip_fraction_two_m(X, m1, m2, lam_per_step, I2_val, B):
    """
    Clip fraction for Construction 2, averaged over {m1, m2} and B permutations.
    m1, m2 should be the two predicted CI boundaries for the current sample.
    """
    total = 0.0
    for _ in range(B):
        X_perm = np.random.permutation(X)
        total += clip_fraction_path(X_perm, m1, lam_per_step, I2_val)
        total += clip_fraction_path(X_perm, m2, lam_per_step, I2_val)
    return total / (2.0 * B)


@njit
def wsr_clip_fraction_path(X, m, delta, c=0.5):
    """
    Fraction of the 2n bets in M_{n,inf}(m) where the clip condition fired:
      M^+ arm: tilde_lam > c / m
      M^- arm: tilde_lam > c / (1-m)
    Returns n_clipped / (2n).
    Note: tilde_lam uses the predictable residual variance, so different data
    orderings yield different clip fractions.
    """
    n       = len(X)
    sum_x   = 0.0
    pred_sq = 0.0
    log2d   = np.log(2.0 / delta)
    n_clipped = 0

    for i in range(n):
        mean_hat  = (0.5 + sum_x) / (1.0 + i)
        var_hat   = (0.25 + pred_sq) / (1.0 + i)
        tilde_lam = np.sqrt(2.0 * log2d / (n * var_hat))

        if tilde_lam > c / (m       + 1e-14):
            n_clipped += 1
        if tilde_lam > c / (1.0 - m + 1e-14):
            n_clipped += 1

        residual = X[i] - mean_hat
        sum_x += X[i]
        pred_sq += residual * residual

    return n_clipped / (2.0 * n)


@njit
def wsr_clip_fraction_two_m(X, m1, m2, delta, B, c=0.5):
    """
    Clip fraction for W&R, averaged over {m1, m2} and B permutations.
    m1, m2 should be the two predicted CI boundaries for the current sample.
    """
    total = 0.0
    for _ in range(B):
        X_perm = np.random.permutation(X)
        total += wsr_clip_fraction_path(X_perm, m1, delta, c)
        total += wsr_clip_fraction_path(X_perm, m2, delta, c)
    return total / (2.0 * B)


@njit
def bankruptcy_fraction_two_m(X, m1, m2, lam_per_step, I2_val, B, thresh_frac=0.01):
    """
    Bankruptcy fraction for Construction 2, averaged over {m1, m2} and B permutations.
    A path is 'bankrupt' when its final capital < thresh_frac * I2_val.
    m1, m2 should be the two predicted CI boundaries for the current sample.
    """
    n_bankrupt = 0
    for _ in range(B):
        X_perm = np.random.permutation(X)
        if compute_Mt2_path(X_perm, m1, lam_per_step, I2_val) < thresh_frac * I2_val:
            n_bankrupt += 1
        if compute_Mt2_path(X_perm, m2, lam_per_step, I2_val) < thresh_frac * I2_val:
            n_bankrupt += 1
    return n_bankrupt / (2.0 * B)


# ------------------------------------------------------------------
# 4a. Wealth-path trajectories (for diagnostics)
# ------------------------------------------------------------------

@njit
def compute_M_inf_path(X, m, delta, c=1):
    """
    Same computation as compute_M_inf but stores M_t at every step.
    Returns array of shape (n+1,): M_0=1, M_1, ..., M_n.
    """
    n       = len(X)
    path    = np.empty(n + 1)
    M_plus  = 1.0
    M_minus = 1.0
    path[0] = 1.0
    sum_x   = 0.0
    pred_sq = 0.0
    log2d   = np.log(2.0 / delta)

    for i in range(n):
        mean_hat  = (0.5 + sum_x) / (1.0 + i)
        var_hat   = (0.25 + pred_sq) / (1.0 + i)
        tilde_lam = np.sqrt(2.0 * log2d / (n * var_hat))
        lam_plus  = min(tilde_lam, c / (m       + 1e-14))
        lam_minus = min(tilde_lam, c / (1.0 - m + 1e-14))
        M_plus  *= 1.0 + lam_plus  * (X[i] - m)
        M_minus *= 1.0 - lam_minus * (X[i] - m)
        path[i + 1] = 0.5 * M_plus + 0.5 * M_minus
        residual = X[i] - mean_hat
        sum_x += X[i]
        pred_sq += residual * residual

    return path


@njit
def compute_Mt2_single_path_traj(X, m, lam_per_step, I2_val, c=0.5):
    """
    Same as compute_Mt2_path but stores M_t at every step.
    Returns array of shape (n+1,): M_0=I2_val, M_1, ..., M_n.
    """
    n            = len(X)
    path         = np.empty(n + 1)
    sqrt_n       = np.sqrt(float(n))
    S_plus       = 0.0
    S_minus      = 0.0
    Lambda_plus  = 0.0
    Lambda_minus = 0.0
    M_plus       = I2_val
    M_minus      = I2_val
    path[0]      = I2_val
    sum_x         = 0.0
    pred_sq       = 0.0

    for i in range(n):
        mean_hat    = (0.5 + sum_x) / (1.0 + i)
        var_hat     = (0.25 + pred_sq) / (1.0 + i)
        tilde_gamma = 1.0 / (sqrt_n * np.sqrt(var_hat))
        Wp = max(S_plus  - Lambda_plus,  0.0)
        Wm = max(S_minus - Lambda_minus, 0.0)
        if Wp > 1e-14 and m > 1e-14:
            gamma_plus = min(tilde_gamma, c * M_plus / (2.0 * Wp * m))
        else:
            gamma_plus = tilde_gamma
        one_m = 1.0 - m
        if Wm > 1e-14 and one_m > 1e-14:
            gamma_minus = min(tilde_gamma, c * M_minus / (2.0 * Wm * one_m))
        else:
            gamma_minus = tilde_gamma
        xi           = X[i] - m
        M_plus      += 2.0 * gamma_plus  *  xi * Wp
        M_minus     += 2.0 * gamma_minus * (-xi) * Wm
        S_plus      += gamma_plus  *  xi
        S_minus     += gamma_minus * (-xi)
        Lambda_plus  += lam_per_step * (gamma_plus  / tilde_gamma)
        Lambda_minus += lam_per_step * (gamma_minus / tilde_gamma)
        residual = X[i] - mean_hat
        sum_x += X[i]
        pred_sq += residual * residual
        path[i + 1] = 0.5 * M_plus + 0.5 * M_minus

    return path


@njit
def compute_M_bar_n2_path_traj(X, m, lam_per_step, I2_val, B):
    """
    Wealth path of M_bar_{n,2}: average of B permutation paths at each step.
    Returns array of shape (n+1,).
    """
    n     = len(X)
    total = np.zeros(n + 1)
    for _ in range(B):
        total += compute_Mt2_single_path_traj(
            np.random.permutation(X), m, lam_per_step, I2_val
        )
    return total / B


# ------------------------------------------------------------------
# 4.  Lebesgue measure of a CI for one sample
# ------------------------------------------------------------------

def ci_lebesgue(X, delta, lam_per_step, I2_star, m_grid, B):
    """
    For a single sample X, returns (width_inf, width_bar2):
    the approximate Lebesgue measure of each CI, computed as
    (number of accepted grid points) * (grid step size).
    """
    thresh_inf  = 1.0 / delta
    thresh_bar2 = I2_star / delta
    step        = m_grid[1] - m_grid[0]

    acc_inf  = 0
    acc_bar2 = 0
    for m in m_grid:
        if compute_M_inf(X, m, delta) < thresh_inf:
            acc_inf += 1
        if compute_M_bar_n2(X, m, lam_per_step, I2_star, B) < thresh_bar2:
            acc_bar2 += 1

    return acc_inf * step, acc_bar2 * step


# ------------------------------------------------------------------
# 5.  Asymptotic limits of sqrt(n) * CI width
# ------------------------------------------------------------------

# True sigma and mean for each distribution
_TRUE_SIGMAS = {
    "Beta(2,2)  [mu=0.50, symmetric]":    np.sqrt(2 * 2 / (4**2 * 5)),   # 1/sqrt(20)
    "Beta(1,5)  [mu=0.17, right-skewed]": np.sqrt(1 * 5 / (6**2 * 7)),   # sqrt(5/252)
    "Bernoulli(0.5)":                     0.5,
}
_TRUE_MEANS = {
    "Beta(2,2)  [mu=0.50, symmetric]":    2.0 / (2.0 + 2.0),
    "Beta(1,5)  [mu=0.17, right-skewed]": 1.0 / (1.0 + 5.0),
    "Bernoulli(0.5)":                     0.5,
}


def asymptotic_limit_wsr(delta):
    """sqrt(n) * half-width limit for M_{n,inf}: sqrt(2 log(2/delta))."""
    return np.sqrt(2.0 * np.log(2.0 / delta))


def asymptotic_limit_mt2(delta):
    """
    sqrt(n) * half-width limit for M_bar_{n,2}: inf_lambda U_{2, delta/2}(lambda).

    Derivation: at the CI boundary, (M^+ + M_{0,2})/2 = M_{0,2}/delta, so
    M^+ ~= 2*M_{0,2}/delta = I2(lam*) / (delta/2).  The dominant term in M^+ is
    (S_n - lam*)_+^2 = I2(lam*) / (delta/2), giving S_n = lam* + sqrt(I2/(delta/2))
    = U_{2,delta/2}(lam*).  With S_n = sqrt(n)*d/sigma, full-width limit = 2*sigma*U.
    """
    lam_star, I2_val = get_optimal_lambda(delta / 2)
    return U2(lam_star, delta / 2)   # = lam* + sqrt(I2(lam*) / (delta/2))


def run_convergence_experiment(
    delta=0.01,
    B=5,
    num_sims=40,
    n_values=(100, 500, 2000, 5000),
    seed=42,
):
    """Diagnose the stopping-time bug and validate the heat-flow repair.

    Unlike ``run_experiment``, this routine uses common permutations for every
    candidate value of ``m`` and bisection rather than a fixed grid.  Those two
    choices isolate the martingale construction from Monte Carlo roughness and
    from a grid error that is magnified by ``sqrt(n)``.
    """
    rng = np.random.default_rng(seed)
    n_values = list(n_values)
    strike, initial_wealth = get_optimal_lambda(delta / 2.0)
    half_width_factor = U2(strike, delta / 2.0)

    distributions = {
        "Beta(2,2)": (
            lambda n: rng.beta(2, 2, n),
            np.sqrt(1.0 / 20.0),
        ),
        "Beta(1,5)": (
            lambda n: rng.beta(1, 5, n),
            np.sqrt(5.0 / 252.0),
        ),
        "Bernoulli(0.5)": (
            lambda n: rng.binomial(1, 0.5, n).astype(float),
            0.5,
        ),
    }
    results = {
        name: {
            "target_heat": 2.0 * sigma * half_width_factor,
            "target_wsr": 2.0 * sigma * asymptotic_limit_wsr(delta),
            "legacy": [],
            "heat": [],
            "wsr": [],
            "legacy_clip": [],
            "heat_clip": [],
        }
        for name, (_, sigma) in distributions.items()
    }

    # Compile the kernels before the timed simulation loop.
    warm = rng.uniform(0.0, 1.0, 20)
    warm_perms = fixed_permutations(warm, 2, rng)
    compute_M_bar_n2_fixed(warm_perms, 0.5, strike / len(warm), initial_wealth)
    compute_M_bar_heat(warm_perms, 0.5, strike, initial_wealth)
    heat_clip_fractions(warm, 0.5, strike, initial_wealth)

    for n in n_values:
        print(f"n={n}")
        for name, (sample, _) in distributions.items():
            widths_legacy = []
            widths_heat = []
            widths_wsr = []
            clips_legacy = []
            clips_heat = []

            for _ in range(num_sims):
                X = sample(n)
                X_perms = fixed_permutations(X, B, rng)
                center = float(np.mean(X))
                threshold = initial_wealth / delta

                def legacy_statistic(m):
                    return compute_M_bar_n2_fixed(
                        X_perms, m, strike / n, initial_wealth
                    )

                def heat_statistic(m):
                    return compute_M_bar_heat(
                        X_perms, m, strike, initial_wealth
                    )

                def wsr_statistic(m):
                    return compute_M_inf(X, m, delta)

                legacy_lo, legacy_hi = _interval_component(
                    legacy_statistic, threshold, center
                )
                widths_legacy.append(np.sqrt(n) * (legacy_hi - legacy_lo))

                heat_lo, heat_hi = _interval_component(
                    heat_statistic, threshold, center
                )
                widths_heat.append(np.sqrt(n) * (heat_hi - heat_lo))

                wsr_lo, wsr_hi = _interval_component(
                    wsr_statistic, 1.0 / delta, center
                )
                widths_wsr.append(np.sqrt(n) * (wsr_hi - wsr_lo))

                # Measure clipping at each method's actual endpoints.  For the
                # heat hedge, only the rejection-relevant arm is counted.
                old_clip = 0.0
                active_heat_clip = 0.0
                for X_perm in X_perms:
                    old_clip += 0.5 * (
                        clip_fraction_path(
                            X_perm, legacy_lo, strike / n, initial_wealth
                        )
                        + clip_fraction_path(
                            X_perm, legacy_hi, strike / n, initial_wealth
                        )
                    )
                    lower_clips = heat_clip_fractions(
                        X_perm, heat_lo, strike, initial_wealth
                    )
                    upper_clips = heat_clip_fractions(
                        X_perm, heat_hi, strike, initial_wealth
                    )
                    active_heat_clip += 0.5 * (
                        lower_clips[0] + upper_clips[1]
                    )
                clips_legacy.append(old_clip / B)
                clips_heat.append(active_heat_clip / B)

            for key, values in (
                ("legacy", widths_legacy),
                ("heat", widths_heat),
                ("wsr", widths_wsr),
                ("legacy_clip", clips_legacy),
                ("heat_clip", clips_heat),
            ):
                arr = np.asarray(values)
                results[name][key].append(
                    {
                        "mean": float(np.mean(arr)),
                        "lo": float(np.quantile(arr, 0.1)),
                        "hi": float(np.quantile(arr, 0.9)),
                    }
                )

            print(
                f"  {name:16s} target={results[name]['target_heat']:.3f}  "
                f"legacy={np.mean(widths_legacy):.3f}  "
                f"heat={np.mean(widths_heat):.3f}  "
                f"WSR={np.mean(widths_wsr):.3f}  "
                f"clip(old/new)={np.mean(clips_legacy):.3f}/"
                f"{np.mean(clips_heat):.3f}"
            )

    fig, axes = plt.subplots(1, len(distributions), figsize=(16, 4.5))
    for ax, name in zip(axes, distributions):
        target_heat = results[name]["target_heat"]
        target_wsr = results[name]["target_wsr"]
        ax.axhline(target_heat, color="black", ls=":", lw=1.8,
                   label=f"heat theory = {target_heat:.3f}")
        ax.axhline(target_wsr, color="gray", ls="--", lw=1.5,
                   label=f"WSR theory = {target_wsr:.3f}")
        for key, color, marker, label in (
            ("legacy", "coral", "o", "legacy stopped hedge"),
            ("heat", "steelblue", "^", "heat-flow hedge"),
            ("wsr", "seagreen", "s", "product martingale (WSR)"),
        ):
            means = [row["mean"] for row in results[name][key]]
            lows = [row["lo"] for row in results[name][key]]
            highs = [row["hi"] for row in results[name][key]]
            ax.plot(n_values, means, color=color, marker=marker, lw=2,
                    label=label)
            ax.fill_between(n_values, lows, highs, color=color, alpha=0.14)
        ax.set_xscale("log")
        ax.set_title(name)
        ax.set_xlabel("n (log scale)")
        ax.set_ylabel(r"$\sqrt{n}\times$ CI width")
        ax.grid(True, ls="--", alpha=0.35)
        ax.legend(fontsize=8)

    fig.suptitle(
        rf"Stopping-time failure and heat-flow repair "
        rf"[$\delta={delta}$, B={B}, sims={num_sims}]"
    )
    plt.tight_layout()
    os.makedirs("plots", exist_ok=True)
    out = "plots/ci_width_convergence_corrected.png"
    plt.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved to {out}")
    return results


# ------------------------------------------------------------------
# 6.  Experiment driver
# ------------------------------------------------------------------

def run_experiment(
    delta=0.1,
    B=50,
    num_sims=50,
    grid_size=200,
    n_values=(50, 100, 250, 500),
    seed=42,
    diagnostics=True,
):
    """
    For each n in n_values and each of three bounded distributions, run num_sims
    independent trials and record sqrt(n) * Lebesgue measure of each CI.
    Plots mean +/- 90% empirical band vs n.
    """
    np.random.seed(seed)
    n_values = list(n_values)

    # M_bar_{n,2} uses lambda^*(delta/2) — see Construction 2 in content.tex
    lam_star, I2_star = get_optimal_lambda(delta / 2)
    print(f"delta={delta}")
    print(f"  lambda^*(delta/2={delta/2})  = {lam_star:.4f}")
    print(f"  I2(lambda^*)                 = {I2_star:.4f}")
    print(f"  threshold M_inf  : 1/delta   = {1/delta:.4f}")
    print(f"  threshold M_bar  : I2/delta  = {I2_star/delta:.4f}\n")

    m_grid = np.linspace(0.01, 0.99, grid_size)

    dist_names = [
        "Beta(2,2)  [mu=0.50, symmetric]",
        "Beta(1,5)  [mu=0.17, right-skewed]",
        "Bernoulli(0.5)",
    ]
    methods = ["M_inf", "M_bar_n2"]

    results = {
        name: {meth: {"mean": [], "lo": [], "hi": []} for meth in methods}
        for name in dist_names
    }
    clip_results     = {name: {"mean": [], "lo": [], "hi": []} for name in dist_names}
    wsr_clip_results = {name: {"mean": [], "lo": [], "hi": []} for name in dist_names}
    bankrupt_results = {name: {"mean": [], "lo": [], "hi": []} for name in dist_names}

    # JIT warmup: compile before timing
    _d  = np.random.uniform(0, 1, 20).astype(np.float64)
    _mg = np.linspace(0.01, 0.99, 5)
    compute_M_inf(_d, 0.5, 0.1)
    compute_Mt2_path(_d, 0.5, 0.001, 0.1)
    compute_M_bar_n2(_d, 0.5, 0.001, 0.1, 2)
    compute_M_inf_path(_d, 0.5, 0.1)
    compute_Mt2_single_path_traj(_d, 0.5, 0.001, 0.1)
    compute_M_bar_n2_path_traj(_d, 0.5, 0.001, 0.1, 2)
    if diagnostics:
        clip_fraction_path(_d, 0.5, 0.001, 0.1)
        clip_fraction_two_m(_d, 0.3, 0.7, 0.001, 0.1, 2)
        wsr_clip_fraction_path(_d, 0.5, 0.1)
        wsr_clip_fraction_two_m(_d, 0.3, 0.7, 0.1, 2)
        bankruptcy_fraction_two_m(_d, 0.3, 0.7, 0.001, 0.1, 2)
    print("JIT warmup complete.\n")

    # Asymptotic half-width factors (used both in the plot and for m boundaries)
    h_wsr = asymptotic_limit_wsr(delta)
    h_mt2 = asymptotic_limit_mt2(delta)

    for n in n_values:
        lam_per_step = lam_star / n
        raw          = {name: {meth: [] for meth in methods} for name in dist_names}
        raw_clip     = {name: [] for name in dist_names}
        raw_wsr_clip = {name: [] for name in dist_names}
        raw_bankrupt = {name: [] for name in dist_names}

        for _ in tqdm(range(num_sims), desc=f"n={n}"):
            data = {
                dist_names[0]: np.random.beta(2, 2, n),
                dist_names[1]: np.random.beta(1, 5, n),
                dist_names[2]: np.random.binomial(1, 0.5, n).astype(float),
            }
            sqrt_n = np.sqrt(n)
            for name, X in data.items():
                w_inf, w_bar2 = ci_lebesgue(X, delta, lam_per_step, I2_star, m_grid, B)
                raw[name]["M_inf"].append(w_inf * sqrt_n)
                raw[name]["M_bar_n2"].append(w_bar2 * sqrt_n)
                if diagnostics:
                    # Predicted CI boundaries from sample mean and std
                    xbar    = float(np.mean(X))
                    sig_hat = float(np.std(X))
                    half_wsr = h_wsr * sig_hat / sqrt_n
                    half_mt2 = h_mt2 * sig_hat / sqrt_n
                    m_wsr_lo = np.clip(xbar - half_wsr, 0.01, 0.99)
                    m_wsr_hi = np.clip(xbar + half_wsr, 0.01, 0.99)
                    m_mt2_lo = np.clip(xbar - half_mt2, 0.01, 0.99)
                    m_mt2_hi = np.clip(xbar + half_mt2, 0.01, 0.99)
                    raw_clip[name].append(
                        clip_fraction_two_m(X, m_mt2_lo, m_mt2_hi, lam_per_step, I2_star, B)
                    )
                    raw_wsr_clip[name].append(
                        wsr_clip_fraction_two_m(X, m_wsr_lo, m_wsr_hi, delta, B)
                    )
                    raw_bankrupt[name].append(
                        bankruptcy_fraction_two_m(X, m_mt2_lo, m_mt2_hi, lam_per_step, I2_star, B)
                    )

        for name in dist_names:
            for meth in methods:
                arr = np.array(raw[name][meth])
                results[name][meth]["mean"].append(float(np.mean(arr)))
                lo, hi = np.percentile(arr, [5.0, 95.0])
                results[name][meth]["lo"].append(float(lo))
                results[name][meth]["hi"].append(float(hi))
            if diagnostics:
                for arr_raw, store in [
                    (raw_clip[name],     clip_results[name]),
                    (raw_wsr_clip[name], wsr_clip_results[name]),
                    (raw_bankrupt[name], bankrupt_results[name]),
                ]:
                    arr = np.array(arr_raw)
                    store["mean"].append(float(np.mean(arr)))
                    lo, hi = np.percentile(arr, [5.0, 95.0])
                    store["lo"].append(float(lo))
                    store["hi"].append(float(hi))

    # --- Plot ---
    fig, axes = plt.subplots(1, len(dist_names), figsize=(6 * len(dist_names), 5))
    fig.suptitle(
        rf"$\sqrt{{n}}\times$ Lebesgue measure of $(1-\delta)$-CI  [$\delta$={delta}, B={B}, sims={num_sims}]",
        fontsize=14,
    )
    styles = {
        "M_inf":    {"color": "coral",     "marker": "o", "ls": "-"},
        "M_bar_n2": {"color": "steelblue", "marker": "^", "ls": "--"},
    }
    labels = {
        "M_inf":    r"$M_{n,\infty}$ (Waudby-Smith & Ramdas)",
        "M_bar_n2": r"$\bar{M}_{n,2}$ (Construction 2)",
    }
    limit_labels = {
        "M_inf":    r"$2\sigma\sqrt{2\log(2/\delta)}$",
        "M_bar_n2": r"$2\sigma\,\inf_\lambda U_{2,\delta/2}(\lambda)$",
    }
    limits = {
        "M_inf":    h_wsr,
        "M_bar_n2": h_mt2,
    }

    for ax, name in zip(axes, dist_names):
        sigma = _TRUE_SIGMAS[name]
        for meth in methods:
            d = results[name][meth]
            s = styles[meth]
            ax.plot(
                n_values, d["mean"],
                marker=s["marker"], color=s["color"], ls=s["ls"], lw=2,
                label=labels[meth],
            )
            ax.fill_between(n_values, d["lo"], d["hi"], color=s["color"], alpha=0.15)
            # Asymptotic reference line
            lim_val = 2 * sigma * limits[meth]
            ax.axhline(lim_val, color=s["color"], ls=":", lw=1.5, alpha=0.75,
                       label=f"limit: {limit_labels[meth]} = {lim_val:.3f}")
        ax.set_title(name, fontsize=11)
        ax.set_xlabel("n")
        ax.set_ylabel(r"$\sqrt{n}\times$ CI width")
        ax.set_xticks(n_values)
        ax.grid(True, ls="--", alpha=0.5)
        ax.legend(fontsize=8)

    plt.tight_layout()
    os.makedirs("plots", exist_ok=True)
    out = "plots/ci_lebesgue_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved to {out}")
    plt.show()

    if diagnostics:
        # --- Second figure: clip fraction vs n (both methods) ---
        fig2, axes2 = plt.subplots(1, len(dist_names), figsize=(6 * len(dist_names), 4))
        fig2.suptitle(
            rf"Fraction of bets clipped  [$\delta$={delta}, B={B}, sims={num_sims}]",
            fontsize=14,
        )
        for ax2, name in zip(axes2, dist_names):
            d_c = clip_results[name]
            ax2.plot(n_values, d_c["mean"], marker="s", color="steelblue", lw=2,
                     label=r"$\bar{M}_{n,2}$ (Constr. 2)")
            ax2.fill_between(n_values, d_c["lo"], d_c["hi"], color="steelblue", alpha=0.15)

            d_w = wsr_clip_results[name]
            ax2.plot(n_values, d_w["mean"], marker="o", color="coral", lw=2,
                     label=r"$M_{n,\infty}$ (W\&R)")
            ax2.fill_between(n_values, d_w["lo"], d_w["hi"], color="coral", alpha=0.15)

            ax2.axhline(0, color="gray", ls=":", lw=1)
            ax2.set_title(name, fontsize=11)
            ax2.set_xlabel("n")
            ax2.set_ylabel("clip fraction (avg over $m$-grid, permutations)")
            ax2.set_xticks(n_values)
            ax2.set_ylim(-0.02, 1.02)
            ax2.legend(fontsize=8)
            ax2.grid(True, ls="--", alpha=0.5)

        plt.tight_layout()
        out2 = "plots/clip_fraction.png"
        plt.savefig(out2, dpi=150, bbox_inches="tight")
        print(f"Saved to {out2}")
        plt.show()

        # --- Third figure: bankruptcy fraction vs n ---
        fig3, axes3 = plt.subplots(1, len(dist_names), figsize=(6 * len(dist_names), 4))
        fig3.suptitle(
            rf"Fraction of $\bar{{M}}_{{n,2}}$ paths that go bankrupt (<1\% of $I_2(\lambda^*)$)  "
            rf"[$\delta$={delta}, B={B}, sims={num_sims}]",
            fontsize=13,
        )
        for ax3, name in zip(axes3, dist_names):
            d = bankrupt_results[name]
            ax3.plot(n_values, d["mean"], marker="D", color="mediumpurple", lw=2,
                     label="bankrupt fraction")
            ax3.fill_between(n_values, d["lo"], d["hi"], color="mediumpurple", alpha=0.15,
                             label="5–95% across sims")
            ax3.axhline(0, color="gray", ls=":", lw=1)
            ax3.set_title(name, fontsize=11)
            ax3.set_xlabel("n")
            ax3.set_ylabel(r"fraction of $(B \times |m\text{-grid}|)$ paths bankrupt")
            ax3.set_xticks(n_values)
            ax3.set_ylim(-0.02, 1.02)
            ax3.legend(fontsize=8)
            ax3.grid(True, ls="--", alpha=0.5)

        plt.tight_layout()
        out3 = "plots/bankruptcy_fraction.png"
        plt.savefig(out3, dpi=150, bbox_inches="tight")
        print(f"Saved to {out3}")
        plt.show()

        # --- Fourth figure: wealth processes at predicted CI boundaries ---
        # Each method evaluated at its own two predicted boundaries:
        #   m_lo = mu - h * sigma / sqrt(n),  m_hi = mu + h * sigma / sqrt(n)
        # (using true mu and sigma so the m values are fixed across simulations).
        # Wealth normalized by threshold: both processes start at delta and should
        # reach 1 to reject.  Solid = lower boundary, dashed = upper boundary.
        n_show      = n_values[-1]
        lam_show    = lam_star / n_show
        thresh_inf  = 1.0 / delta
        thresh_bar2 = I2_star / delta

        def _sample_dist(name, n):
            if "Beta(2,2)" in name:
                return np.random.beta(2, 2, n)
            elif "Beta(1,5)" in name:
                return np.random.beta(1, 5, n)
            else:
                return np.random.binomial(1, 0.5, n).astype(float)

        # Fixed m values (true mu/sigma) for coherent path aggregation
        m_bounds = {}
        for name in dist_names:
            mu_t  = _TRUE_MEANS[name]
            sig_t = _TRUE_SIGMAS[name]
            half_wsr = h_wsr * sig_t / np.sqrt(n_show)
            half_mt2 = h_mt2 * sig_t / np.sqrt(n_show)
            m_bounds[name] = {
                "inf_lo":  np.clip(mu_t - half_wsr, 0.01, 0.99),
                "inf_hi":  np.clip(mu_t + half_wsr, 0.01, 0.99),
                "bar2_lo": np.clip(mu_t - half_mt2, 0.01, 0.99),
                "bar2_hi": np.clip(mu_t + half_mt2, 0.01, 0.99),
            }

        # Collect: for each method×boundary, a list of (n_show+1,) normalized arrays
        path_sims = {
            name: {"inf_lo": [], "inf_hi": [], "bar2_lo": [], "bar2_hi": []}
            for name in dist_names
        }
        for _ in tqdm(range(num_sims), desc="wealth paths"):
            for name in dist_names:
                X   = _sample_dist(name, n_show)
                mb  = m_bounds[name]
                path_sims[name]["inf_lo"].append(
                    compute_M_inf_path(X, mb["inf_lo"], delta) / thresh_inf
                )
                path_sims[name]["inf_hi"].append(
                    compute_M_inf_path(X, mb["inf_hi"], delta) / thresh_inf
                )
                path_sims[name]["bar2_lo"].append(
                    compute_M_bar_n2_path_traj(X, mb["bar2_lo"], lam_show, I2_star, B)
                    / thresh_bar2
                )
                path_sims[name]["bar2_hi"].append(
                    compute_M_bar_n2_path_traj(X, mb["bar2_hi"], lam_show, I2_star, B)
                    / thresh_bar2
                )

        fig4, axes4 = plt.subplots(
            2, len(dist_names), figsize=(6 * len(dist_names), 8), sharey="row"
        )
        fig4.suptitle(
            rf"Normalized wealth paths at predicted CI boundaries  "
            rf"[$n={n_show}$, $\delta$={delta}, B={B}, sims={num_sims}]"
            "\n"
            r"solid = lower boundary $\mu - h\hat\sigma/\sqrt{n}$, "
            r"dashed = upper $\mu + h\hat\sigma/\sqrt{n}$",
            fontsize=11,
        )
        steps = np.arange(n_show + 1)
        row_cfg = [
            ("inf",  "coral",     r"$M_{n,\infty}$ (W\&R)"),
            ("bar2", "steelblue", r"$\bar{M}_{n,2}$ (Constr.2)"),
        ]

        for row, (key, color, label) in enumerate(row_cfg):
            for col, name in enumerate(dist_names):
                ax  = axes4[row, col]
                mb  = m_bounds[name]

                for side, ls, side_label in [
                    (f"{key}_lo", "-",  f"lo ($m={mb[key+'_lo']:.3f}$)"),
                    (f"{key}_hi", "--", f"hi ($m={mb[key+'_hi']:.3f}$)"),
                ]:
                    arr       = np.array(path_sims[name][side])  # (num_sims, n_show+1)
                    mean_path = np.mean(arr, axis=0)
                    lo_path   = np.percentile(arr,  5, axis=0)
                    hi_path   = np.percentile(arr, 95, axis=0)
                    ax.plot(steps, mean_path, color=color, lw=2, ls=ls,
                            label=f"mean {side_label}")
                    ax.fill_between(steps, lo_path, hi_path,
                                    color=color, alpha=0.12)

                ax.axhline(1.0,   color="black", ls="--", lw=1.5, label="threshold")
                ax.axhline(delta, color="gray",  ls=":",  lw=1.0,
                           label=rf"start ($\delta$)")
                if col == 0:
                    ax.set_ylabel(f"{label}\n$M_t\\,/$ threshold")
                if row == 0:
                    ax.set_title(name, fontsize=10)
                ax.set_xlabel("step $t$")
                ax.legend(fontsize=7, loc="upper left")
                ax.grid(True, ls="--", alpha=0.4)

        plt.tight_layout()
        out4 = "plots/wealth_processes.png"
        plt.savefig(out4, dpi=150, bbox_inches="tight")
        print(f"Saved to {out4}")
        plt.show()


if __name__ == "__main__":
    run_convergence_experiment()
