#!/usr/bin/env python3
"""Confidence-interval experiments for the current betting constructions.

The legacy polynomial Constructions 1 and 2 live in legacy_constructions.py.
This module contains the Waudby-Smith--Ramdas product martingale, its
target-recalculating STaR variant, the buffered randomized Probit-STaR rule,
a digital-payoff DP hedge, the exact Bernoulli DP benchmark, the nonnegative
heat-flow Construction 3, confidence-set inversion, and the current
comparison experiment.
"""

import math
import os
import json
from collections.abc import Mapping

import matplotlib.pyplot as plt
import numpy as np
from numba import njit, prange
from scipy.optimize import brentq, minimize_scalar
from scipy.stats import binom, norm


# ------------------------------------------------------------------
# Gaussian squared-hinge payoff and optimal strike
# ------------------------------------------------------------------

def I2(lam):
    """Return E[(Z-lam)_+^2] for Z standard normal."""
    return (1.0 + lam**2) * norm.sf(lam) - lam * norm.pdf(lam)


def U2(lam, delta):
    """Return lam + sqrt(I2(lam)/delta)."""
    return lam + np.sqrt(max(I2(lam), 0.0) / delta)


def get_optimal_lambda(delta):
    """Numerically minimize U2(lam, delta) over nonnegative lam."""
    result = minimize_scalar(
        lambda lam: U2(lam, delta),
        bounds=(1e-6, 20.0),
        method="bounded",
    )
    strike = float(result.x)
    return strike, float(I2(strike))


def _build_target_strike_table():
    """Tabulate the inverse of I1(a)^2/I2(a) on a logit-alpha grid."""
    strike_axis = np.linspace(-12.0, 12.0, 65_537)
    survival = norm.sf(strike_axis)
    density = norm.pdf(strike_axis)
    i1 = density - strike_axis * survival
    i2 = (
        (1.0 + strike_axis * strike_axis) * survival
        - strike_axis * density
    )
    ratios = np.clip(i1 * i1 / i2, 1e-300, 1.0 - 1e-15)
    ratio_logits = np.log(ratios / (1.0 - ratios))

    logit_min = np.log(1e-12 / (1.0 - 1e-12))
    logit_max = np.log(0.99 / 0.01)
    logit_grid = np.linspace(logit_min, logit_max, 16_385)
    strikes = np.interp(
        logit_grid,
        ratio_logits[::-1],
        strike_axis[::-1],
    )
    return logit_min, logit_max, strikes


(
    _TARGET_LOGIT_MIN,
    _TARGET_LOGIT_MAX,
    _TARGET_STRIKE_TABLE,
) = _build_target_strike_table()


_CAPPED_HINGE_RAMP = 1.0


def _build_capped_hinge_leverage_table(ramp=_CAPPED_HINGE_RAMP):
    """Tabulate the delta-to-price ratio of a target-capped hinge claim."""
    strike = np.linspace(-8.0, 12.0, 65_537)
    upper = strike + ramp
    density_lower = norm.pdf(strike)
    density_upper = norm.pdf(upper)
    interval_probability = norm.cdf(upper) - norm.cdf(strike)
    first_moment = (
        density_lower
        - density_upper
        - strike * interval_probability
    )
    second_moment = (
        (1.0 + strike * strike) * interval_probability
        - strike * density_lower
        + (strike - ramp) * density_upper
    )
    price = norm.sf(upper) + second_moment / (ramp * ramp)
    delta = 2.0 * first_moment / (ramp * ramp)
    price = np.clip(price, 1e-300, 1.0 - 1e-15)
    leverage = np.maximum(delta, 0.0) / price
    price_logits = np.log(price / (1.0 - price))

    logit_min = np.log(1e-12 / (1.0 - 1e-12))
    logit_max = np.log((1.0 - 1e-12) / 1e-12)
    logit_grid = np.linspace(logit_min, logit_max, 16_385)
    leverage_grid = np.interp(
        logit_grid,
        price_logits[::-1],
        leverage[::-1],
    )
    return logit_min, logit_max, leverage_grid


(
    _CAPPED_HINGE_LOGIT_MIN,
    _CAPPED_HINGE_LOGIT_MAX,
    _CAPPED_HINGE_LEVERAGE_TABLE,
) = _build_capped_hinge_leverage_table()


def _capped_exponential_price_delta(strike, slope):
    """Gaussian price and delta of min(exp(slope * (Z-strike)), 1)."""
    tilted_tail = (
        np.exp(-slope * strike + 0.5 * slope * slope)
        * norm.cdf(strike - slope)
    )
    price = norm.sf(strike) + tilted_tail
    delta = slope * tilted_tail
    return price, delta


def _build_capped_exponential_leverage_table():
    """Tabulate the direct target-capped analogue of product STaR."""
    logit_min = np.log(1e-12 / (1.0 - 1e-12))
    logit_max = np.log((1.0 - 1e-12) / 1e-12)
    logit_grid = np.linspace(logit_min, logit_max, 16_385)
    probabilities = 1.0 / (1.0 + np.exp(-logit_grid))
    leverage_grid = np.empty_like(probabilities)

    for index, probability in enumerate(probabilities):
        slope = np.sqrt(max(2.0 * np.log(1.0 / probability), 1e-30))

        def price_gap(strike):
            price, _ = _capped_exponential_price_delta(strike, slope)
            return price - probability

        strike = brentq(price_gap, -16.0, 16.0)
        price, delta = _capped_exponential_price_delta(strike, slope)
        leverage_grid[index] = delta / price

    return logit_min, logit_max, leverage_grid


(
    _CAPPED_EXPONENTIAL_LOGIT_MIN,
    _CAPPED_EXPONENTIAL_LOGIT_MAX,
    _CAPPED_EXPONENTIAL_LEVERAGE_TABLE,
) = _build_capped_exponential_leverage_table()


@njit
def _normal_cdf(x):
    return 0.5 * math.erfc(-x / np.sqrt(2.0))


@njit
def _normal_pdf(x):
    return np.exp(-0.5 * x * x) / np.sqrt(2.0 * np.pi)


@njit
def _normal_ppf(probability):
    """Machine-precision inverse standard-normal CDF.

    This is the Acklam rational approximation followed by one Halley step.
    It avoids a fixed probability grid or an asymptotically active clamp.
    """
    if probability <= 0.0:
        return -np.inf
    if probability >= 1.0:
        return np.inf

    reflect = probability > 0.5
    probability_left = 1.0 - probability if reflect else probability
    p_low = 0.02425

    if probability_left < p_low:
        q = np.sqrt(-2.0 * np.log(probability_left))
        value = (
            (((((-7.784894002430293e-3 * q - 3.223964580411365e-1) * q
                - 2.400758277161838) * q - 2.549732539343734) * q
              + 4.374664141464968) * q + 2.938163982698783)
            / (((((7.784695709041462e-3 * q + 3.224671290700398e-1) * q
                  + 2.445134137142996) * q + 3.754408661907416) * q)
                + 1.0)
        )
    else:
        q = probability_left - 0.5
        r = q * q
        value = (
            (((((-3.969683028665376e1 * r + 2.209460984245205e2) * r
                - 2.759285104469687e2) * r + 1.383577518672690e2) * r
              - 3.066479806614716e1) * r + 2.506628277459239)
            * q
            / (((((-5.447609879822406e1 * r + 1.615858368580409e2) * r
                  - 1.556989798598866e2) * r + 6.680131188771972e1) * r
                - 1.328068155288572e1) * r + 1.0)
        )

    density = _normal_pdf(value)
    if density > 0.0:
        correction = (_normal_cdf(value) - probability_left) / density
        value -= correction / (1.0 + 0.5 * value * correction)
    return -value if reflect else value


@njit
def heat_payoff_value(x, remaining_variance, strike):
    """Gaussian continuation value E[(x + sqrt(v) Z - strike)_+^2]."""
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
    """Derivative in x of heat_payoff_value."""
    a = x - strike
    if remaining_variance <= 1e-15:
        return 2.0 * max(a, 0.0)
    sd = np.sqrt(remaining_variance)
    d = a / sd
    return 2.0 * (a * _normal_cdf(d) + sd * _normal_pdf(d))


@njit
def _hinge_I1(strike):
    """Return E[(Z-strike)_+] for Z standard normal."""
    survival = _normal_cdf(-strike)
    return _normal_pdf(strike) - strike * survival


@njit
def _hinge_I2(strike):
    """Numba-compatible E[(Z-strike)_+^2]."""
    survival = _normal_cdf(-strike)
    return max(
        (1.0 + strike * strike) * survival
        - strike * _normal_pdf(strike),
        1e-300,
    )


@njit
def target_recalculating_strike(effective_alpha):
    """Minimize a + sqrt(I2(a)/effective_alpha).

    The first-order equation is I1(a)^2 = effective_alpha I2(a).
    This is the squared-hinge analogue of recomputing the remaining
    log-wealth gap in STaR-Bets.
    """
    alpha = min(max(effective_alpha, 1e-12), 1.0 - 1e-12)
    if alpha > 0.99:
        return -np.sqrt(alpha / (1.0 - alpha))
    logit = np.log(alpha / (1.0 - alpha))
    position = (
        (logit - _TARGET_LOGIT_MIN)
        / (_TARGET_LOGIT_MAX - _TARGET_LOGIT_MIN)
        * (_TARGET_STRIKE_TABLE.size - 1)
    )
    lower = int(np.floor(position))
    lower = min(max(lower, 0), _TARGET_STRIKE_TABLE.size - 2)
    weight = position - lower
    return (
        (1.0 - weight) * _TARGET_STRIKE_TABLE[lower]
        + weight * _TARGET_STRIKE_TABLE[lower + 1]
    )


@njit
def hinge_target_leverage(effective_alpha):
    """Standardized delta-to-wealth ratio of the optimized hinge claim."""
    strike = target_recalculating_strike(effective_alpha)
    return 2.0 * _hinge_I1(strike) / _hinge_I2(strike)


@njit
def exponential_target_leverage(effective_alpha):
    """Standardized square-root leverage used by product STaR."""
    alpha = min(max(effective_alpha, 1e-300), 1.0)
    return np.sqrt(max(2.0 * np.log(1.0 / alpha), 0.0))


@njit
def capped_hinge_target_leverage(target_fraction):
    """Delta-to-price ratio for the unit-width target-capped hinge ramp."""
    probability = min(max(target_fraction, 1e-12), 1.0 - 1e-12)
    logit = np.log(probability / (1.0 - probability))
    position = (
        (logit - _CAPPED_HINGE_LOGIT_MIN)
        / (_CAPPED_HINGE_LOGIT_MAX - _CAPPED_HINGE_LOGIT_MIN)
        * (_CAPPED_HINGE_LEVERAGE_TABLE.size - 1)
    )
    lower = int(np.floor(position))
    lower = min(max(lower, 0), _CAPPED_HINGE_LEVERAGE_TABLE.size - 2)
    weight = position - lower
    return (
        (1.0 - weight) * _CAPPED_HINGE_LEVERAGE_TABLE[lower]
        + weight * _CAPPED_HINGE_LEVERAGE_TABLE[lower + 1]
    )


@njit
def capped_exponential_target_leverage(target_fraction):
    """Delta-to-price ratio of the capped original-STaR planning claim."""
    probability = min(max(target_fraction, 1e-12), 1.0 - 1e-12)
    logit = np.log(probability / (1.0 - probability))
    position = (
        (logit - _CAPPED_EXPONENTIAL_LOGIT_MIN)
        / (
            _CAPPED_EXPONENTIAL_LOGIT_MAX
            - _CAPPED_EXPONENTIAL_LOGIT_MIN
        )
        * (_CAPPED_EXPONENTIAL_LEVERAGE_TABLE.size - 1)
    )
    lower = int(np.floor(position))
    lower = min(
        max(lower, 0),
        _CAPPED_EXPONENTIAL_LEVERAGE_TABLE.size - 2,
    )
    weight = position - lower
    return (
        (1.0 - weight) * _CAPPED_EXPONENTIAL_LEVERAGE_TABLE[lower]
        + weight * _CAPPED_EXPONENTIAL_LEVERAGE_TABLE[lower + 1]
    )


@njit
def probit_target_leverage(target_fraction):
    """Return the Gaussian-digital leverage phi(Phi^{-1}(p))/p.

    Here p is current wealth as a fraction of the absorbing target.  This is
    the delta-to-wealth ratio of a Gaussian digital continuation value,
    before division by the square root of the remaining variance.  It is
    evaluated directly to machine precision rather than through a fixed grid.
    """
    if target_fraction <= 0.0:
        return np.inf
    if target_fraction >= 1.0:
        return 0.0
    quantile = _normal_ppf(target_fraction)
    return np.exp(
        -0.5 * quantile * quantile
        - 0.5 * np.log(2.0 * np.pi)
        - np.log(target_fraction)
    )


@njit
def digital_payoff_value(x, remaining_variance, boundary):
    """Gaussian probability of finishing above a terminal boundary."""
    if remaining_variance <= 1e-15:
        return 1.0 if x >= boundary else 0.0
    sd = np.sqrt(remaining_variance)
    return _normal_cdf((x - boundary) / sd)


@njit
def digital_payoff_delta(x, remaining_variance, boundary):
    """State derivative of digital_payoff_value."""
    if remaining_variance <= 1e-15:
        return 0.0
    sd = np.sqrt(remaining_variance)
    return _normal_pdf((x - boundary) / sd) / sd


# ------------------------------------------------------------------
# Product-martingale comparators
# ------------------------------------------------------------------

@njit
def compute_M_inf(X, m, delta, c=0.5):
    """Two-sided Waudby-Smith--Ramdas product martingale at time n."""
    n = len(X)
    M_plus = 1.0
    M_minus = 1.0
    sum_x = 0.0
    pred_sq = 0.0
    log2d = np.log(2.0 / delta)

    for i in range(n):
        mean_hat = (0.5 + sum_x) / (1.0 + i)
        var_hat = (0.25 + pred_sq) / (1.0 + i)
        raw_bet = np.sqrt(2.0 * log2d / (n * var_hat))

        bet_plus = min(raw_bet, c / (m + 1e-14))
        bet_minus = min(raw_bet, c / (1.0 - m + 1e-14))

        centered = X[i] - m
        M_plus *= 1.0 + bet_plus * centered
        M_minus *= 1.0 - bet_minus * centered

        residual = X[i] - mean_hat
        sum_x += X[i]
        pred_sq += residual * residual

    return 0.5 * (M_plus + M_minus)


@njit
def _compute_M_product_target(
    X,
    m,
    delta,
    regularization,
    c,
    recalculate,
):
    """Shared implementation for matched fixed and recalculating bets."""
    n = len(X)
    alpha = delta / 2.0
    target = 1.0 / alpha
    log_target = np.log(target)
    eps = 1e-12

    M_plus = 1.0
    M_minus = 1.0
    centered_sq_sum = 0.0

    for i in range(n):
        remaining = float(n - i) if recalculate else float(n)
        cap_variance = m * (1.0 - m) + eps

        if i == 0:
            v_plus = cap_variance
            v_minus = cap_variance
        else:
            past = float(i)
            empirical_second = centered_sq_sum / past
            v_plus = min(
                empirical_second
                + regularization * (m + eps) * n / (past * past),
                cap_variance,
            )
            v_minus = min(
                empirical_second
                + regularization * (1.0 - m + eps) * n
                / (past * past),
                cap_variance,
            )
            v_plus = max(v_plus, eps)
            v_minus = max(v_minus, eps)

        if M_plus > 0.0 and M_plus < target:
            gap_plus = log_target
            if recalculate:
                gap_plus = max(log_target - np.log(M_plus), 0.0)
            bet_plus = np.sqrt(
                2.0 * gap_plus / (remaining * v_plus)
            )
            bet_plus = min(bet_plus, c / (m + eps))
        else:
            bet_plus = 0.0

        if M_minus > 0.0 and M_minus < target:
            gap_minus = log_target
            if recalculate:
                gap_minus = max(log_target - np.log(M_minus), 0.0)
            bet_minus = np.sqrt(
                2.0 * gap_minus / (remaining * v_minus)
            )
            bet_minus = min(bet_minus, c / (1.0 - m + eps))
        else:
            bet_minus = 0.0

        centered = X[i] - m
        M_plus *= max(1.0 + bet_plus * centered, 0.0)
        M_minus *= max(1.0 - bet_minus * centered, 0.0)
        centered_sq_sum += centered * centered

    return 0.5 * (M_plus + M_minus)


@njit
def compute_M_bets(X, m, delta, regularization=1.0, c=1.0):
    """Matched non-recalculating product strategy.

    This comparator uses the same candidate-centered second-moment estimate,
    solvency cap, and absorbing target as compute_M_star.  It differs only by
    retaining the original log target and horizon in every round.
    """
    return _compute_M_product_target(
        X, m, delta, regularization, c, False
    )


@njit
def compute_M_star(X, m, delta, regularization=1.0, c=1.0):
    """Two-sided STaR target-recalculating product supermartingale.

    This follows the public STaR-Bets implementation: at every round the bet
    recomputes the remaining log-wealth gap, remaining horizon, and a
    regularized estimate of E[(X-m)^2]. Once a one-sided arm reaches its
    target 2/delta, its achieved (possibly overshooting) wealth is frozen.
    """
    return _compute_M_product_target(
        X, m, delta, regularization, c, True
    )


@njit
def _compute_M_recalculating_feedback(
    X,
    m,
    delta,
    regularization,
    c,
    feedback_kind,
):
    """Matched STaR recursion differing only in its feedback map."""
    if feedback_kind < 0 or feedback_kind > 3:
        raise ValueError("feedback_kind must be 0, 1, 2, or 3")
    n = len(X)
    alpha = delta / 2.0
    target = 1.0 / alpha
    eps = 1e-12

    M_plus = 1.0
    M_minus = 1.0
    centered_sq_sum = 0.0

    for i in range(n):
        remaining = float(n - i)
        cap_variance = m * (1.0 - m) + eps

        if i == 0:
            v_plus = cap_variance
            v_minus = cap_variance
        else:
            past = float(i)
            empirical_second = centered_sq_sum / past
            v_plus = min(
                empirical_second
                + regularization * (m + eps) * n / (past * past),
                cap_variance,
            )
            v_minus = min(
                empirical_second
                + regularization * (1.0 - m + eps) * n
                / (past * past),
                cap_variance,
            )
            v_plus = max(v_plus, eps)
            v_minus = max(v_minus, eps)

        if M_plus > 0.0 and M_plus < target:
            target_fraction = alpha * M_plus
            if feedback_kind == 0:
                leverage_plus = exponential_target_leverage(
                    target_fraction
                )
            elif feedback_kind == 1:
                leverage_plus = hinge_target_leverage(target_fraction)
            elif feedback_kind == 2:
                leverage_plus = capped_hinge_target_leverage(
                    target_fraction
                )
            else:
                leverage_plus = capped_exponential_target_leverage(
                    target_fraction
                )
            bet_plus = leverage_plus / np.sqrt(remaining * v_plus)
            bet_plus = min(bet_plus, c / (m + eps))
        else:
            bet_plus = 0.0

        if M_minus > 0.0 and M_minus < target:
            target_fraction = alpha * M_minus
            if feedback_kind == 0:
                leverage_minus = exponential_target_leverage(
                    target_fraction
                )
            elif feedback_kind == 1:
                leverage_minus = hinge_target_leverage(target_fraction)
            elif feedback_kind == 2:
                leverage_minus = capped_hinge_target_leverage(
                    target_fraction
                )
            else:
                leverage_minus = capped_exponential_target_leverage(
                    target_fraction
                )
            bet_minus = leverage_minus / np.sqrt(remaining * v_minus)
            bet_minus = min(bet_minus, c / (1.0 - m + eps))
        else:
            bet_minus = 0.0

        centered = X[i] - m
        M_plus *= max(1.0 + bet_plus * centered, 0.0)
        M_minus *= max(1.0 - bet_minus * centered, 0.0)
        centered_sq_sum += centered * centered

    return 0.5 * (M_plus + M_minus)


@njit
def compute_M_hinge_feedback_star(
    X, m, delta, regularization=1.0, c=1.0
):
    """Matched product recursion with optimized squared-hinge feedback."""
    return _compute_M_recalculating_feedback(
        X, m, delta, regularization, c, 1
    )


@njit
def compute_M_capped_feedback_star(
    X, m, delta, regularization=1.0, c=1.0
):
    """Matched product recursion with target-capped quadratic feedback."""
    return _compute_M_recalculating_feedback(
        X, m, delta, regularization, c, 2
    )


@njit
def compute_M_capped_exponential_feedback_star(
    X, m, delta, regularization=1.0, c=1.0
):
    """Matched product recursion with capped original feedback."""
    return _compute_M_recalculating_feedback(
        X, m, delta, regularization, c, 3
    )


@njit(parallel=True)
def _recalculating_feedback_scores(X, means, delta, feedback_kind):
    """Evaluate one matched feedback rule over candidate means in parallel."""
    scores = np.empty(len(means))
    for j in prange(len(means)):
        scores[j] = _compute_M_recalculating_feedback(
            X, means[j], delta, 1.0, 1.0, feedback_kind
        )
    return scores


@njit
def compute_M_probit_star_arms(
    X,
    m,
    delta,
    regularization=1.0,
    c=1.0,
    buffer_rounds=0.0,
):
    """Return the two arms of the buffered probit STaR approximation.

    The raw leverage is the inverse-Mills ratio at current target fraction
    alpha*M, divided by the estimated remaining standard deviation.  A
    positive buffer_rounds regularizes the terminal singularity.  With
    sqrt(n) = o(buffer_rounds) and buffer_rounds = o(n), terminal
    all-or-nothing randomization yields the Gaussian-efficiency result
    described in the paper.  The inverse Mills ratio is evaluated directly
    to machine precision.
    """
    n = len(X)
    alpha = delta / 2.0
    target = 1.0 / alpha
    eps = 1e-12

    M_plus = 1.0
    M_minus = 1.0
    centered_sq_sum = 0.0

    for i in range(n):
        remaining = float(n - i) + max(buffer_rounds, 0.0)
        cap_variance = m * (1.0 - m) + eps

        if i == 0:
            v_plus = cap_variance
            v_minus = cap_variance
        else:
            past = float(i)
            empirical_second = centered_sq_sum / past
            v_plus = min(
                empirical_second
                + regularization * (m + eps) * n / (past * past),
                cap_variance,
            )
            v_minus = min(
                empirical_second
                + regularization * (1.0 - m + eps) * n
                / (past * past),
                cap_variance,
            )
            v_plus = max(v_plus, eps)
            v_minus = max(v_minus, eps)

        if M_plus > 0.0 and M_plus < target:
            target_fraction = alpha * M_plus
            bet_plus = (
                probit_target_leverage(target_fraction)
                / np.sqrt(remaining * v_plus)
            )
            bet_plus = min(bet_plus, c / (m + eps))
        else:
            bet_plus = 0.0

        if M_minus > 0.0 and M_minus < target:
            target_fraction = alpha * M_minus
            bet_minus = (
                probit_target_leverage(target_fraction)
                / np.sqrt(remaining * v_minus)
            )
            bet_minus = min(bet_minus, c / (1.0 - m + eps))
        else:
            bet_minus = 0.0

        centered = X[i] - m
        M_plus *= max(1.0 + bet_plus * centered, 0.0)
        M_minus *= max(1.0 - bet_minus * centered, 0.0)
        centered_sq_sum += centered * centered

    return M_plus, M_minus


@njit
def compute_M_probit_star(
    X,
    m,
    delta,
    regularization=1.0,
    c=1.0,
    buffer_rounds=0.0,
):
    """Two-sided wealth from the inverse-Mills/probit STaR rule."""
    M_plus, M_minus = compute_M_probit_star_arms(
        X, m, delta, regularization, c, buffer_rounds
    )
    return 0.5 * (M_plus + M_minus)


@njit(parallel=True)
def _probit_randomized_scores(
    X,
    means,
    delta,
    buffer_rounds,
    u_plus,
    u_minus,
):
    """Evaluate randomized Probit rejection scores over candidate means."""
    alpha = delta / 2.0
    scores = np.empty(len(means))
    for j in prange(len(means)):
        M_plus, M_minus = compute_M_probit_star_arms(
            X,
            means[j],
            delta,
            buffer_rounds=buffer_rounds,
        )
        scores[j] = max(
            alpha * M_plus / u_plus,
            alpha * M_minus / u_minus,
        )
    return scores


# ------------------------------------------------------------------
# Digital-payoff DP hedge
# ------------------------------------------------------------------

@njit
def compute_M_digital_dp(X, m, delta, boundary, c=1.0):
    """Two-sided bounded-data hedge of a Gaussian digital terminal payoff.

    The ideal stake is the delta of the fixed-variance Gaussian continuation
    probability. It is transferred to bounded observations, clipped for
    solvency, and stopped after reaching the one-sided target 2/delta.
    Predictability of the stake makes the resulting process valid even though
    the Gaussian transition law is used only as a design model.
    """
    n = len(X)
    sqrt_n = np.sqrt(float(n))
    alpha = delta / 2.0
    target = 1.0 / alpha
    eps = 1e-14

    S_plus = 0.0
    S_minus = 0.0
    M_plus = 1.0
    M_minus = 1.0
    sum_x = 0.0
    pred_sq = 0.0

    for i in range(n):
        mean_hat = (0.5 + sum_x) / (1.0 + i)
        var_hat = (0.25 + pred_sq) / (1.0 + i)
        gamma = 1.0 / (sqrt_n * np.sqrt(var_hat))
        remaining_variance = 1.0 - i / float(n)

        if M_plus < target:
            beta_plus = (
                gamma
                * digital_payoff_delta(
                    S_plus, remaining_variance, boundary
                )
                / alpha
            )
            if m > eps:
                beta_plus = min(beta_plus, c * M_plus / m)
        else:
            beta_plus = 0.0

        if M_minus < target:
            beta_minus = (
                gamma
                * digital_payoff_delta(
                    S_minus, remaining_variance, boundary
                )
                / alpha
            )
            if 1.0 - m > eps:
                beta_minus = min(
                    beta_minus, c * M_minus / (1.0 - m)
                )
        else:
            beta_minus = 0.0

        centered = X[i] - m
        M_plus += beta_plus * centered
        M_minus -= beta_minus * centered
        S_plus += gamma * centered
        S_minus -= gamma * centered

        residual = X[i] - mean_hat
        sum_x += X[i]
        pred_sq += residual * residual

    return 0.5 * (M_plus + M_minus)


# ------------------------------------------------------------------
# Construction 3: nonnegative heat-flow hedge
# ------------------------------------------------------------------

@njit
def compute_M_heat_path(X, m, strike, initial_wealth, c=1.0):
    """Terminal Construction 3 wealth for one fixed data ordering.

    The raw amount invested is the delta of the Gaussian continuation value.
    It is clipped to the largest one-step-safe amount. The standardized score
    itself is never clipped.
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

        if m > 1e-14:
            beta_plus = min(beta_plus, c * M_plus / m)
        one_m = 1.0 - m
        if one_m > 1e-14:
            beta_minus = min(beta_minus, c * M_minus / one_m)

        centered = X[i] - m
        M_plus += beta_plus * centered
        M_minus -= beta_minus * centered

        # Preserve the full statistical information clock after a wealth clip.
        S_plus += gamma * centered
        S_minus -= gamma * centered

        residual = X[i] - mean_hat
        sum_x += X[i]
        pred_sq += residual * residual

    return 0.5 * (M_plus + M_minus)


@njit
def _compute_M_heat_star_path(
    X,
    m,
    delta,
    initial_wealth,
    c,
    feedback_kind,
):
    """Shared target-recalculating squared-hinge recursion."""
    if feedback_kind < 0 or feedback_kind > 1:
        raise ValueError("feedback_kind must be 0 or 1")
    n = len(X)
    sqrt_n = np.sqrt(float(n))
    target = 2.0 * initial_wealth / delta
    M_plus = initial_wealth
    M_minus = initial_wealth
    sum_x = 0.0
    pred_sq = 0.0

    for i in range(n):
        mean_hat = (0.5 + sum_x) / (1.0 + i)
        var_hat = (0.25 + pred_sq) / (1.0 + i)
        gamma = 1.0 / (sqrt_n * np.sqrt(var_hat))
        remaining_variance = 1.0 - i / float(n)
        sqrt_remaining = np.sqrt(max(remaining_variance, 1e-15))

        beta_plus = 0.0
        if 0.0 < M_plus < target:
            effective_alpha = M_plus / target
            if feedback_kind == 0:
                standardized_leverage = hinge_target_leverage(
                    effective_alpha
                )
            else:
                standardized_leverage = capped_hinge_target_leverage(
                    effective_alpha
                )
            beta_plus = (
                gamma
                * M_plus
                * standardized_leverage
                / sqrt_remaining
            )
            if m > 1e-14:
                beta_plus = min(beta_plus, c * M_plus / m)

        beta_minus = 0.0
        if 0.0 < M_minus < target:
            effective_alpha = M_minus / target
            if feedback_kind == 0:
                standardized_leverage = hinge_target_leverage(
                    effective_alpha
                )
            else:
                standardized_leverage = capped_hinge_target_leverage(
                    effective_alpha
                )
            beta_minus = (
                gamma
                * M_minus
                * standardized_leverage
                / sqrt_remaining
            )
            one_m = 1.0 - m
            if one_m > 1e-14:
                beta_minus = min(
                    beta_minus, c * M_minus / one_m
                )

        centered = X[i] - m
        M_plus += beta_plus * centered
        M_minus -= beta_minus * centered

        residual = X[i] - mean_hat
        sum_x += X[i]
        pred_sq += residual * residual

    return 0.5 * (M_plus + M_minus)


@njit
def compute_M_heat_star_path(X, m, delta, initial_wealth, c=1.0):
    """Target-recalculating squared-hinge/Bentkus supermartingale.

    At each round, each arm re-prices a locally centered squared-hinge claim.
    Its effective error level is current wealth divided by the absorbing
    one-sided target 2*initial_wealth/delta. The resulting strike minimizes
    the remaining Gaussian score required to reach that target.
    """
    return _compute_M_heat_star_path(
        X, m, delta, initial_wealth, c, 0
    )


@njit
def compute_M_heat_capped_star_path(
    X, m, delta, initial_wealth, c=1.0
):
    """Target-capped quadratic-ramp STaR supermartingale.

    The local Gaussian planning payoff is
    ``min(((Z-a)_+ / eta)**2, 1)`` with unit ramp width.  Capping at the
    rejection target removes squared-hinge overshoots and nearly digitalizes
    the continuation value while retaining a smooth quadratic transition.
    This is Bentkus-inspired rather than a convex Bentkus test function.
    """
    return _compute_M_heat_star_path(
        X, m, delta, initial_wealth, c, 1
    )


@njit
def compute_M_heat_trajectory(X, m, strike, initial_wealth, c=1.0):
    """Return the plus and minus wealth trajectories for Ville diagnostics."""
    n = len(X)
    sqrt_n = np.sqrt(float(n))
    plus_path = np.empty(n + 1)
    minus_path = np.empty(n + 1)
    plus_path[0] = initial_wealth
    minus_path[0] = initial_wealth

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
        if m > 1e-14:
            beta_plus = min(beta_plus, c * M_plus / m)
        one_m = 1.0 - m
        if one_m > 1e-14:
            beta_minus = min(beta_minus, c * M_minus / one_m)

        centered = X[i] - m
        M_plus += beta_plus * centered
        M_minus -= beta_minus * centered
        S_plus += gamma * centered
        S_minus -= gamma * centered

        residual = X[i] - mean_hat
        sum_x += X[i]
        pred_sq += residual * residual

        plus_path[i + 1] = M_plus
        minus_path[i + 1] = M_minus

    return plus_path, minus_path


@njit
def heat_clip_fractions(X, m, strike, initial_wealth, c=1.0):
    """Return plus- and minus-arm fractions of clipped bets."""
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

        centered = X[i] - m
        M_plus += beta_plus * centered
        M_minus -= beta_minus * centered
        S_plus += gamma * centered
        S_minus -= gamma * centered

        residual = X[i] - mean_hat
        sum_x += X[i]
        pred_sq += residual * residual

    return clipped_plus / float(n), clipped_minus / float(n)


# ------------------------------------------------------------------
# Confidence-set inversion
# ------------------------------------------------------------------

def _interval_component(
    statistic,
    threshold,
    center,
    scan_points=24,
    geometric_scan=False,
    batch_statistic=None,
):
    """Find the accepted component containing center by scan and root solving."""
    cache = {}

    def centered_statistic(m):
        key = float(m)
        if key not in cache:
            cache[key] = float(statistic(key)) - threshold
        return cache[key]

    if centered_statistic(center) >= 0.0:
        raise ValueError("the supplied center is not in the confidence set")

    def boundary(outer):
        previous = center
        bracket = None
        span = abs(float(outer) - center)
        if span == 0.0:
            return float(outer)
        if geometric_scan:
            direction = 1.0 if outer > center else -1.0
            fractions = np.geomspace(1e-8, 1.0, scan_points)
            candidates = center + direction * span * fractions
        else:
            candidates = np.linspace(center, outer, scan_points + 1)[1:]
        if batch_statistic is not None:
            values = np.asarray(batch_statistic(candidates), dtype=float)
            if values.shape != candidates.shape:
                raise ValueError(
                    "batch_statistic must return one value per candidate"
                )
            for current, value in zip(candidates, values):
                cache[float(current)] = float(value) - threshold
        for current in candidates:
            current = float(current)
            if centered_statistic(current) >= 0.0:
                bracket = (previous, current)
                break
            previous = current
        if bracket is None:
            return float(outer)

        left, right = sorted(bracket)
        return float(
            brentq(
                centered_statistic,
                left,
                right,
                xtol=1e-9,
                rtol=1e-10,
                maxiter=100,
            )
        )

    return boundary(0.0), boundary(1.0)


def heat_ci_endpoints(X, delta, strike, initial_wealth):
    """Invert the chronological fixed-horizon Construction 3 e-value."""
    X = np.asarray(X)
    center = float(np.mean(X))
    return _interval_component(
        lambda m: compute_M_heat_path(X, m, strike, initial_wealth),
        initial_wealth / delta,
        center,
    )


def heat_star_ci_endpoints(X, delta, initial_wealth):
    """Invert the chronological target-recalculating Bentkus hedge."""
    X = np.asarray(X)
    center = float(np.mean(X))
    return _interval_component(
        lambda m: compute_M_heat_star_path(
            X, m, delta, initial_wealth
        ),
        initial_wealth / delta,
        center,
    )


def heat_capped_star_ci_endpoints(X, delta, initial_wealth):
    """Invert the chronological target-capped quadratic-ramp STaR."""
    X = np.asarray(X)
    center = float(np.mean(X))
    return _interval_component(
        lambda m: compute_M_heat_capped_star_path(
            X, m, delta, initial_wealth
        ),
        initial_wealth / delta,
        center,
        scan_points=56,
        geometric_scan=True,
    )


def wsr_ci_endpoints(X, delta):
    """Invert the product martingale around the sample mean."""
    X = np.asarray(X)
    center = float(np.mean(X))
    return _interval_component(
        lambda m: compute_M_inf(X, m, delta),
        1.0 / delta,
        center,
    )


def bets_ci_endpoints(X, delta):
    """Invert the matched non-recalculating product strategy."""
    X = np.asarray(X)
    center = float(np.mean(X))
    return _interval_component(
        lambda m: compute_M_bets(X, m, delta),
        1.0 / delta,
        center,
    )


def star_ci_endpoints(X, delta):
    """Invert the two-sided target-recalculating product supermartingale."""
    X = np.asarray(X)
    center = float(np.mean(X))
    return _interval_component(
        lambda m: compute_M_star(X, m, delta),
        1.0 / delta,
        center,
        batch_statistic=lambda means: _recalculating_feedback_scores(
            X, np.asarray(means), delta, 0
        ),
    )


def hinge_feedback_star_ci_endpoints(X, delta):
    """Invert the matched optimized-hinge STaR feedback."""
    X = np.asarray(X)
    center = float(np.mean(X))
    return _interval_component(
        lambda m: compute_M_hinge_feedback_star(X, m, delta),
        1.0 / delta,
        center,
        batch_statistic=lambda means: _recalculating_feedback_scores(
            X, np.asarray(means), delta, 1
        ),
    )


def capped_feedback_star_ci_endpoints(X, delta):
    """Invert matched target-capped quadratic-ramp STaR feedback."""
    X = np.asarray(X)
    center = float(np.mean(X))
    return _interval_component(
        lambda m: compute_M_capped_feedback_star(X, m, delta),
        1.0 / delta,
        center,
        scan_points=56,
        geometric_scan=True,
        batch_statistic=lambda means: _recalculating_feedback_scores(
            X, np.asarray(means), delta, 2
        ),
    )


def capped_exponential_feedback_star_ci_endpoints(X, delta):
    """Invert matched capped original STaR feedback."""
    X = np.asarray(X)
    center = float(np.mean(X))
    return _interval_component(
        lambda m: compute_M_capped_exponential_feedback_star(X, m, delta),
        1.0 / delta,
        center,
        scan_points=56,
        geometric_scan=True,
        batch_statistic=lambda means: _recalculating_feedback_scores(
            X, np.asarray(means), delta, 3
        ),
    )


def probit_star_ci_endpoints(
    X,
    delta,
    buffer_rounds=0.0,
    randomizers=None,
):
    """Return the sample-mean component of the probit acceptance set.

    The full inverted set has the finite-sample coverage guarantee.  This
    endpoint helper reports only its component containing the sample mean;
    asymptotic equivalence requires the paper's local-crossing condition.

    If a pair of randomizers is supplied, each terminal arm is projected to
    zero or its one-sided target using that uniform randomizer.  A diverging
    buffer satisfying sqrt(n) = o(buffer_rounds) and buffer_rounds = o(n)
    should be paired with this randomized inversion for the
    Gaussian-efficiency construction.  The same randomizers must be retained
    while scanning all candidate means.
    """
    X = np.asarray(X)
    center = float(np.mean(X))

    if randomizers is None:
        return _interval_component(
            lambda m: compute_M_probit_star(
                X, m, delta, buffer_rounds=buffer_rounds
            ),
            1.0 / delta,
            center,
            scan_points=56,
            geometric_scan=True,
        )

    u_plus, u_minus = (float(value) for value in randomizers)
    if not (0.0 < u_plus < 1.0 and 0.0 < u_minus < 1.0):
        raise ValueError("randomizers must lie strictly between zero and one")
    alpha = delta / 2.0

    def randomized_rejection_score(m):
        M_plus, M_minus = compute_M_probit_star_arms(
            X, m, delta, buffer_rounds=buffer_rounds
        )
        return max(
            alpha * M_plus / u_plus,
            alpha * M_minus / u_minus,
        )

    def batched_randomized_rejection_score(means):
        return _probit_randomized_scores(
            X,
            np.asarray(means),
            delta,
            buffer_rounds,
            u_plus,
            u_minus,
        )

    return _interval_component(
        randomized_rejection_score,
        1.0,
        center,
        scan_points=56,
        geometric_scan=True,
        batch_statistic=batched_randomized_rejection_score,
    )


def probit_star_randomized_ci_endpoints(
    X,
    delta,
    buffer_rounds=None,
    rng=None,
):
    """Return the randomized probit set component containing the sample mean."""
    X = np.asarray(X)
    if buffer_rounds is None:
        # This satisfies the theorem's buffer-rate inequalities.  The theorem
        # states additional tracking and local-crossing conditions separately.
        buffer_rounds = float(len(X)) ** (2.0 / 3.0)
    if rng is None:
        rng = np.random.default_rng()
    randomizers = tuple(rng.uniform(0.0, 1.0, size=2))
    return probit_star_ci_endpoints(
        X,
        delta,
        buffer_rounds=buffer_rounds,
        randomizers=randomizers,
    )


def _probit_star_experiment_component(X, delta, rng):
    """Return the randomized center component and whether it is empty.

    A randomized confidence set may reject the sample mean.  The experiments
    report the component containing the sample mean for every method, so in
    that event the relevant component is empty and is encoded by the
    zero-width pair ``(mean(X), mean(X))``.  The terminal uniforms are not
    redrawn: doing so until the center is accepted would change the method.
    """
    X = np.asarray(X)
    center = float(np.mean(X))
    try:
        endpoints = probit_star_randomized_ci_endpoints(
            X, delta, rng=rng
        )
    except ValueError as error:
        if str(error) != "the supplied center is not in the confidence set":
            raise
        return center, center, True
    return endpoints[0], endpoints[1], False


def digital_dp_ci_endpoints(X, delta):
    """Invert the clipped Gaussian digital-delta product martingale."""
    X = np.asarray(X)
    center = float(np.mean(X))
    boundary = float(norm.isf(delta / 2.0))
    return _interval_component(
        lambda m: compute_M_digital_dp(
            X, m, delta, boundary
        ),
        1.0 / delta,
        center,
        scan_points=56,
        geometric_scan=True,
    )


def bernoulli_dp_ci_endpoints(
    X,
    delta,
    upper_randomizer=0.5,
    lower_randomizer=0.5,
):
    """Invert the exact randomized Bernoulli terminal-event DP.

    The upper-tail p-value is P_m(Bin(n,m)>s)+U_+P_m(Bin(n,m)=s);
    the lower-tail p-value is defined analogously. Their backward
    continuation probabilities are the exact binary-tree DP values.
    """
    X = np.asarray(X)
    if np.any((X != 0.0) & (X != 1.0)):
        raise ValueError("the exact Bernoulli DP requires binary data")
    if not 0.0 <= upper_randomizer <= 1.0:
        raise ValueError("upper_randomizer must be in [0,1]")
    if not 0.0 <= lower_randomizer <= 1.0:
        raise ValueError("lower_randomizer must be in [0,1]")

    n = len(X)
    successes = int(np.sum(X))
    alpha = delta / 2.0

    def upper_tail_p(m):
        return (
            binom.sf(successes, n, m)
            + upper_randomizer * binom.pmf(successes, n, m)
        )

    def lower_tail_p(m):
        return (
            binom.cdf(successes - 1, n, m)
            + lower_randomizer * binom.pmf(successes, n, m)
        )

    lower_at_zero = upper_tail_p(0.0) - alpha
    if lower_at_zero >= 0.0:
        lower = 0.0
    elif upper_tail_p(1.0) - alpha <= 0.0:
        lower = 1.0
    else:
        lower = float(
            brentq(
                lambda m: upper_tail_p(m) - alpha,
                0.0,
                1.0,
                xtol=1e-12,
                rtol=1e-13,
            )
        )

    upper_at_one = lower_tail_p(1.0) - alpha
    if upper_at_one >= 0.0:
        upper = 1.0
    elif lower_tail_p(0.0) - alpha <= 0.0:
        upper = 0.0
    else:
        upper = float(
            brentq(
                lambda m: lower_tail_p(m) - alpha,
                0.0,
                1.0,
                xtol=1e-12,
                rtol=1e-13,
            )
        )

    return lower, upper


# ------------------------------------------------------------------
# Asymptotic constants and current comparison experiment
# ------------------------------------------------------------------

MAX_EXPERIMENT_N = 1_000_000
DEFAULT_N_VALUES = (
    10,
    50,
    100,
    500,
    1000,
    5000,
    10_000,
    50_000,
    100_000,
    500_000,
    MAX_EXPERIMENT_N,
)

PUBLICATION_SIMULATION_COUNTS = {
    n: (50 if n <= 10_000 else 30 if n <= 100_000 else 20)
    for n in DEFAULT_N_VALUES
}


def _validated_n_values(n_values):
    """Validate the common sample-size grid used by all experiments."""
    values = list(n_values)
    if not values:
        raise ValueError("n_values must be nonempty")
    if any(int(n) != n or n <= 0 for n in values):
        raise ValueError("n_values must contain positive integers")
    if max(values) > MAX_EXPERIMENT_N:
        raise ValueError(
            f"experiment sample sizes may not exceed {MAX_EXPERIMENT_N}"
        )
    return [int(n) for n in values]


def _simulation_counts(n_values, num_sims):
    """Return a validated simulation count for each sample size."""
    if isinstance(num_sims, Mapping):
        counts = {}
        for n in n_values:
            if n in num_sims:
                value = num_sims[n]
            elif str(n) in num_sims:
                value = num_sims[str(n)]
            else:
                raise ValueError(f"num_sims has no entry for n={n}")
            if int(value) != value or value <= 0:
                raise ValueError("simulation counts must be positive integers")
            counts[n] = int(value)
        return counts

    if int(num_sims) != num_sims or num_sims <= 0:
        raise ValueError("num_sims must be a positive integer or mapping")
    return {n: int(num_sims) for n in n_values}


def _simulation_label(num_sims_by_n):
    counts = list(num_sims_by_n.values())
    if min(counts) == max(counts):
        return str(counts[0])
    return f"{min(counts)}\N{EN DASH}{max(counts)}"

_TRUE_SIGMAS = {
    "Beta(2,2)": np.sqrt(1.0 / 20.0),
    "Beta(1,5)": np.sqrt(5.0 / 252.0),
    "Bernoulli(0.5)": 0.5,
    "Uniform(0,1)": np.sqrt(1.0 / 12.0),
    "Beta(0.5,0.5)": np.sqrt(1.0 / 8.0),
    "Bernoulli(0.1)": 0.3,
}


def asymptotic_limit_wsr(delta):
    """Product-martingale half-width factor."""
    return np.sqrt(2.0 * np.log(2.0 / delta))


def asymptotic_limit_heat(delta):
    """Construction 3 half-width factor U_{2,delta/2}(lambda*)."""
    strike, _ = get_optimal_lambda(delta / 2.0)
    return U2(strike, delta / 2.0)


def asymptotic_limit_digital(delta):
    """Gaussian equal-tail half-width factor for the digital DP."""
    return float(norm.isf(delta / 2.0))


def run_dp_experiment(
    delta=0.01,
    num_sims=50,
    n_values=DEFAULT_N_VALUES,
    seed=42,
):
    """Retained diagnostic comparing the pure-DP strategies."""
    rng = np.random.default_rng(seed)
    probit_rng = np.random.default_rng(
        np.random.SeedSequence(seed).spawn(1)[0]
    )
    n_values = _validated_n_values(n_values)
    num_sims_by_n = _simulation_counts(n_values, num_sims)
    simulation_label = _simulation_label(num_sims_by_n)
    strike, initial_wealth = get_optimal_lambda(delta / 2.0)
    heat_factor = U2(strike, delta / 2.0)
    wsr_factor = asymptotic_limit_wsr(delta)
    digital_factor = asymptotic_limit_digital(delta)
    digital_boundary = digital_factor

    samplers = {
        "Beta(2,2)": lambda n: rng.beta(2, 2, n),
        "Beta(1,5)": lambda n: rng.beta(1, 5, n),
        "Bernoulli(0.5)": (
            lambda n: rng.binomial(1, 0.5, n).astype(float)
        ),
        "Uniform(0,1)": lambda n: rng.uniform(0.0, 1.0, n),
        "Beta(0.5,0.5)": lambda n: rng.beta(0.5, 0.5, n),
        "Bernoulli(0.1)": (
            lambda n: rng.binomial(1, 0.1, n).astype(float)
        ),
    }
    results = {
        name: {
            "target_heat": 2.0 * _TRUE_SIGMAS[name] * heat_factor,
            "target_wsr": 2.0 * _TRUE_SIGMAS[name] * wsr_factor,
            "target_digital": (
                2.0 * _TRUE_SIGMAS[name] * digital_factor
            ),
            "wsr": [],
            "wsr_raw": [],
            "star": [],
            "star_raw": [],
            "probit_star": [],
            "probit_star_raw": [],
            "probit_empty_rate": [],
            "digital_dp": [],
            "digital_dp_raw": [],
            "heat": [],
            "heat_raw": [],
            **(
                {"exact_dp": [], "exact_dp_raw": []}
                if name.startswith("Bernoulli")
                else {}
            ),
        }
        for name in samplers
    }

    warm = rng.uniform(0.0, 1.0, 20)
    compute_M_inf(warm, 0.5, delta)
    compute_M_star(warm, 0.5, delta)
    compute_M_probit_star(
        warm,
        0.5,
        delta,
        buffer_rounds=float(len(warm)) ** (2.0 / 3.0),
    )
    compute_M_digital_dp(
        warm, 0.5, delta, digital_boundary
    )
    compute_M_heat_path(warm, 0.5, strike, initial_wealth)

    for n in n_values:
        sims_at_n = num_sims_by_n[n]
        print(f"n={n}  sims={sims_at_n}")
        for name, sample in samplers.items():
            heat_widths = []
            heat_raw_widths = []
            wsr_widths = []
            wsr_raw_widths = []
            star_widths = []
            star_raw_widths = []
            probit_widths = []
            probit_raw_widths = []
            probit_empty = []
            digital_widths = []
            digital_raw_widths = []
            exact_widths = []
            exact_raw_widths = []

            for _ in range(sims_at_n):
                X = sample(n)
                center = float(np.mean(X))

                heat_lo, heat_hi = _interval_component(
                    lambda m: compute_M_heat_path(
                        X, m, strike, initial_wealth
                    ),
                    initial_wealth / delta,
                    center,
                )
                heat_raw_width = heat_hi - heat_lo
                heat_raw_widths.append(heat_raw_width)
                heat_widths.append(np.sqrt(n) * heat_raw_width)

                wsr_lo, wsr_hi = wsr_ci_endpoints(X, delta)
                wsr_raw_width = wsr_hi - wsr_lo
                wsr_raw_widths.append(wsr_raw_width)
                wsr_widths.append(np.sqrt(n) * wsr_raw_width)

                star_lo, star_hi = star_ci_endpoints(X, delta)
                star_raw_width = star_hi - star_lo
                star_raw_widths.append(star_raw_width)
                star_widths.append(np.sqrt(n) * star_raw_width)

                (
                    probit_lo,
                    probit_hi,
                    probit_is_empty,
                ) = _probit_star_experiment_component(
                    X, delta, probit_rng
                )
                probit_raw_width = probit_hi - probit_lo
                probit_raw_widths.append(probit_raw_width)
                probit_widths.append(
                    np.sqrt(n) * probit_raw_width
                )
                probit_empty.append(probit_is_empty)

                digital_lo, digital_hi = digital_dp_ci_endpoints(
                    X, delta
                )
                digital_raw_width = digital_hi - digital_lo
                digital_raw_widths.append(digital_raw_width)
                digital_widths.append(
                    np.sqrt(n) * digital_raw_width
                )

                if name.startswith("Bernoulli"):
                    exact_lo, exact_hi = bernoulli_dp_ci_endpoints(
                        X,
                        delta,
                        upper_randomizer=float(rng.random()),
                        lower_randomizer=float(rng.random()),
                    )
                    exact_raw_width = exact_hi - exact_lo
                    exact_raw_widths.append(exact_raw_width)
                    exact_widths.append(
                        np.sqrt(n) * exact_raw_width
                    )

            summaries = [
                ("heat", heat_widths),
                ("heat_raw", heat_raw_widths),
                ("wsr", wsr_widths),
                ("wsr_raw", wsr_raw_widths),
                ("star", star_widths),
                ("star_raw", star_raw_widths),
                ("probit_star", probit_widths),
                ("probit_star_raw", probit_raw_widths),
                ("digital_dp", digital_widths),
                ("digital_dp_raw", digital_raw_widths),
            ]
            if name.startswith("Bernoulli"):
                summaries.extend(
                    (
                        ("exact_dp", exact_widths),
                        ("exact_dp_raw", exact_raw_widths),
                    )
                )
            for key, values in summaries:
                values = np.asarray(values)
                results[name][key].append(
                    {
                        "mean": float(np.mean(values)),
                        "lo": float(np.quantile(values, 0.1)),
                        "hi": float(np.quantile(values, 0.9)),
                    }
                )
            results[name]["probit_empty_rate"].append(
                float(np.mean(probit_empty))
            )

            heat_report = f"C3={np.mean(heat_widths):.3f}"
            print(
                f"  {name:16s} "
                f"{heat_report} "
                f"(target {results[name]['target_heat']:.3f})  "
                f"WSR={np.mean(wsr_widths):.3f} "
                f"STaR={np.mean(star_widths):.3f} "
                f"Probit-STaR={np.mean(probit_widths):.3f} "
                f"(empty={np.mean(probit_empty):.1%}) "
                f"DigDP={np.mean(digital_widths):.3f}"
                + (
                    f" ExactDP={np.mean(exact_widths):.3f}"
                    if exact_widths
                    else ""
                )
            )

    os.makedirs("plots", exist_ok=True)

    def plot_widths(scaled):
        suffix = "" if scaled else "_raw"
        n_array = np.asarray(n_values, dtype=float)
        fig, axes = plt.subplots(2, 3, figsize=(16, 9))
        for axis, name in zip(axes.ravel(), samplers):
            target_heat = results[name]["target_heat"]
            target_wsr = results[name]["target_wsr"]
            heat_target_curve = (
                np.full_like(n_array, target_heat)
                if scaled
                else target_heat / np.sqrt(n_array)
            )
            wsr_target_curve = (
                np.full_like(n_array, target_wsr)
                if scaled
                else target_wsr / np.sqrt(n_array)
            )
            axis.plot(
                n_values,
                heat_target_curve,
                color="steelblue",
                ls=":",
                lw=1.8,
                label="Construction 3 theory",
            )
            axis.plot(
                n_values,
                wsr_target_curve,
                color="seagreen",
                ls=":",
                lw=1.8,
                label="product theory",
            )

            methods = [
                (
                    f"heat{suffix}",
                    "navy",
                    "o",
                    "Construction 3",
                ),
                (
                    f"wsr{suffix}",
                    "seagreen",
                    "s",
                    "product martingale",
                ),
            ]
            for key, color, marker, label in methods:
                rows = results[name][key]
                means = [row["mean"] for row in rows]
                lows = [row["lo"] for row in rows]
                highs = [row["hi"] for row in rows]
                axis.plot(
                    n_values,
                    means,
                    color=color,
                    marker=marker,
                    ms=4.5,
                    lw=2,
                    label=label,
                )
                axis.fill_between(
                    n_values,
                    lows,
                    highs,
                    color=color,
                    alpha=0.10,
                )

            axis.set_xscale("log")
            if not scaled:
                axis.set_yscale("log")
            axis.set_title(name)
            axis.set_xlabel("n (log scale)")
            axis.set_ylabel(
                r"$\sqrt{n}\times$ CI width"
                if scaled
                else "CI width (log scale)"
            )
            axis.grid(True, ls="--", alpha=0.35)
            axis.legend(fontsize=7.5)

        scale_label = "scaled" if scaled else "raw"
        fig.suptitle(
            rf"Construction 3 versus product martingale: {scale_label} widths "
            rf"[$\delta={delta},\ \mathrm{{sims}}={simulation_label}$]"
        )
        plt.tight_layout()
        output = (
            "plots/ci_width_construction3_vs_wsr.png"
            if scaled
            else "plots/ci_width_raw_construction3_vs_wsr.png"
        )
        plt.savefig(output, dpi=180, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved to {output}")

    def plot_dp_widths(scaled):
        """Plot the target-aware strategies against the original product."""
        suffix = "" if scaled else "_raw"
        n_array = np.asarray(n_values, dtype=float)
        fig, axes = plt.subplots(2, 3, figsize=(16, 9))

        for axis, name in zip(axes.ravel(), samplers):
            target_digital = results[name]["target_digital"]
            target_curve = (
                np.full_like(n_array, target_digital)
                if scaled
                else target_digital / np.sqrt(n_array)
            )
            axis.plot(
                n_values,
                target_curve,
                color="black",
                ls=":",
                lw=1.8,
                label="Gaussian digital limit",
            )

            methods = [
                (
                    f"heat{suffix}",
                    "navy",
                    "o",
                    "Construction 3",
                ),
                (
                    f"wsr{suffix}",
                    "seagreen",
                    "s",
                    "product martingale",
                ),
                (
                    f"star{suffix}",
                    "darkorange",
                    "P",
                    "STaR product",
                ),
                (
                    f"probit_star{suffix}",
                    "purple",
                    "v",
                    "Probit STaR (randomized)",
                ),
                (
                    f"digital_dp{suffix}",
                    "crimson",
                    "D",
                    "digital-DP hedge",
                ),
            ]
            if name.startswith("Bernoulli"):
                methods.append(
                    (
                        f"exact_dp{suffix}",
                        "black",
                        "X",
                        "exact Bernoulli DP",
                    )
                )

            for key, color, marker, label in methods:
                rows = results[name][key]
                means = [row["mean"] for row in rows]
                lows = [row["lo"] for row in rows]
                highs = [row["hi"] for row in rows]
                axis.plot(
                    n_values,
                    means,
                    color=color,
                    marker=marker,
                    ms=4.5,
                    lw=2,
                    label=label,
                )
                axis.fill_between(
                    n_values,
                    lows,
                    highs,
                    color=color,
                    alpha=0.08,
                )

            axis.set_xscale("log")
            if not scaled:
                axis.set_yscale("log")
            axis.set_title(name)
            axis.set_xlabel("n (log scale)")
            axis.set_ylabel(
                r"$\sqrt{n}\times$ CI width"
                if scaled
                else "CI width (log scale)"
            )
            axis.grid(True, ls="--", alpha=0.35)
            axis.legend(fontsize=7.2)

        scale_label = "scaled" if scaled else "raw"
        fig.suptitle(
            rf"Target-aware DP and STaR strategies: {scale_label} widths "
            rf"[$\delta={delta},\ \mathrm{{sims}}={simulation_label}$]"
        )
        plt.tight_layout()
        output = (
            "plots/ci_width_dp_star.png"
            if scaled
            else "plots/ci_width_raw_dp_star.png"
        )
        plt.savefig(output, dpi=180, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved to {output}")

    plot_widths(scaled=True)
    plot_widths(scaled=False)
    plot_dp_widths(scaled=True)
    plot_dp_widths(scaled=False)

    summary_output = "plots/ci_width_construction3_vs_wsr.json"
    with open(summary_output, "w", encoding="utf-8") as output_file:
        json.dump(
            {
                "delta": delta,
                "seed": seed,
                "num_sims_by_n": {
                    str(n): num_sims_by_n[n] for n in n_values
                },
                "n_values": n_values,
                "strike": strike,
                "initial_wealth": initial_wealth,
                "digital_boundary": digital_boundary,
                "probit_buffer_power": 2.0 / 3.0,
                "probit_terminal_randomization": True,
                "results": results,
            },
            output_file,
            indent=2,
        )
    print(f"Saved to {summary_output}")
    return results


def run_experiment(
    delta=0.01,
    num_sims=50,
    n_values=DEFAULT_N_VALUES,
    seed=42,
):
    """Compare fixed, square-root STaR, and probit-STaR intervals."""
    rng = np.random.default_rng(seed)
    probit_rng = np.random.default_rng(
        np.random.SeedSequence(seed).spawn(1)[0]
    )
    n_values = _validated_n_values(n_values)
    num_sims_by_n = _simulation_counts(n_values, num_sims)
    simulation_label = _simulation_label(num_sims_by_n)
    strike, initial_wealth = get_optimal_lambda(delta / 2.0)
    heat_factor = U2(strike, delta / 2.0)
    product_factor = asymptotic_limit_wsr(delta)
    probit_factor = asymptotic_limit_digital(delta)

    samplers = {
        "Beta(2,2)": lambda n: rng.beta(2, 2, n),
        "Beta(1,5)": lambda n: rng.beta(1, 5, n),
        "Bernoulli(0.5)": (
            lambda n: rng.binomial(1, 0.5, n).astype(float)
        ),
        "Uniform(0,1)": lambda n: rng.uniform(0.0, 1.0, n),
        "Beta(0.5,0.5)": lambda n: rng.beta(0.5, 0.5, n),
        "Bernoulli(0.1)": (
            lambda n: rng.binomial(1, 0.1, n).astype(float)
        ),
    }
    results = {
        name: {
            "target_heat": 2.0 * _TRUE_SIGMAS[name] * heat_factor,
            "target_product": (
                2.0 * _TRUE_SIGMAS[name] * product_factor
            ),
            "target_probit": (
                2.0 * _TRUE_SIGMAS[name] * probit_factor
            ),
            "heat_original": [],
            "heat_original_raw": [],
            "heat_star": [],
            "heat_star_raw": [],
            "product_original": [],
            "product_original_raw": [],
            "product_star": [],
            "product_star_raw": [],
            "hinge_feedback_star": [],
            "hinge_feedback_star_raw": [],
            "capped_feedback_star": [],
            "capped_feedback_star_raw": [],
            "capped_exponential_feedback_star": [],
            "capped_exponential_feedback_star_raw": [],
            "probit_star": [],
            "probit_star_raw": [],
            "probit_empty_rate": [],
        }
        for name in samplers
    }

    warm = rng.uniform(0.0, 1.0, 20)
    compute_M_inf(warm, 0.5, delta)
    compute_M_star(warm, 0.5, delta)
    compute_M_hinge_feedback_star(warm, 0.5, delta)
    compute_M_probit_star(
        warm,
        0.5,
        delta,
        buffer_rounds=float(len(warm)) ** (2.0 / 3.0),
    )
    compute_M_heat_path(warm, 0.5, strike, initial_wealth)
    compute_M_heat_star_path(
        warm, 0.5, delta, initial_wealth
    )
    compute_M_capped_feedback_star(warm, 0.5, delta)
    compute_M_capped_exponential_feedback_star(warm, 0.5, delta)

    for n in n_values:
        sims_at_n = num_sims_by_n[n]
        print(f"n={n}  sims={sims_at_n}")
        for name, sample in samplers.items():
            widths = {
                "heat_original": [],
                "heat_star": [],
                "product_original": [],
                "product_star": [],
                "hinge_feedback_star": [],
                "capped_feedback_star": [],
                "capped_exponential_feedback_star": [],
                "probit_star": [],
            }
            raw_widths = {key: [] for key in widths}
            probit_empty = []

            for _ in range(sims_at_n):
                X = sample(n)
                center = float(np.mean(X))

                heat_lo, heat_hi = _interval_component(
                    lambda m: compute_M_heat_path(
                        X, m, strike, initial_wealth
                    ),
                    initial_wealth / delta,
                    center,
                )
                heat_star_lo, heat_star_hi = _interval_component(
                    lambda m: compute_M_heat_star_path(
                        X, m, delta, initial_wealth
                    ),
                    initial_wealth / delta,
                    center,
                )
                product_lo, product_hi = wsr_ci_endpoints(X, delta)
                product_star_lo, product_star_hi = star_ci_endpoints(
                    X, delta
                )
                hinge_feedback_lo, hinge_feedback_hi = (
                    hinge_feedback_star_ci_endpoints(X, delta)
                )
                capped_feedback_lo, capped_feedback_hi = (
                    capped_feedback_star_ci_endpoints(X, delta)
                )
                capped_exponential_lo, capped_exponential_hi = (
                    capped_exponential_feedback_star_ci_endpoints(X, delta)
                )
                (
                    probit_lo,
                    probit_hi,
                    probit_is_empty,
                ) = _probit_star_experiment_component(
                    X, delta, probit_rng
                )
                probit_empty.append(probit_is_empty)

                endpoints = {
                    "heat_original": (heat_lo, heat_hi),
                    "heat_star": (heat_star_lo, heat_star_hi),
                    "product_original": (product_lo, product_hi),
                    "product_star": (
                        product_star_lo,
                        product_star_hi,
                    ),
                    "hinge_feedback_star": (
                        hinge_feedback_lo,
                        hinge_feedback_hi,
                    ),
                    "capped_feedback_star": (
                        capped_feedback_lo,
                        capped_feedback_hi,
                    ),
                    "capped_exponential_feedback_star": (
                        capped_exponential_lo,
                        capped_exponential_hi,
                    ),
                    "probit_star": (probit_lo, probit_hi),
                }
                for key, (lower, upper) in endpoints.items():
                    raw = upper - lower
                    raw_widths[key].append(raw)
                    widths[key].append(np.sqrt(n) * raw)

            for key in widths:
                for result_key, values in (
                    (key, widths[key]),
                    (f"{key}_raw", raw_widths[key]),
                ):
                    values = np.asarray(values)
                    results[name][result_key].append(
                        {
                            "mean": float(np.mean(values)),
                            "lo": float(np.quantile(values, 0.1)),
                            "hi": float(np.quantile(values, 0.9)),
                        }
                    )
            results[name]["probit_empty_rate"].append(
                float(np.mean(probit_empty))
            )

            print(
                f"  {name:16s} "
                f"Bentkus={np.mean(widths['heat_original']):.3f} "
                f"Bentkus-STaR={np.mean(widths['heat_star']):.3f}  "
                f"WSR={np.mean(widths['product_original']):.3f} "
                f"STaR-Bets={np.mean(widths['product_star']):.3f} "
                f"Matched-Hinge={np.mean(widths['hinge_feedback_star']):.3f} "
                f"Capped-STaR={np.mean(widths['capped_feedback_star']):.3f} "
                f"Capped-exp-STaR={np.mean(widths['capped_exponential_feedback_star']):.3f} "
                f"Probit-STaR={np.mean(widths['probit_star']):.3f} "
                f"(empty={np.mean(probit_empty):.1%})"
            )

    os.makedirs("plots", exist_ok=True)

    def plot_widths(scaled):
        suffix = "" if scaled else "_raw"
        n_array = np.asarray(n_values, dtype=float)
        fig, axes = plt.subplots(2, 3, figsize=(16, 9))

        for axis, name in zip(axes.ravel(), samplers):
            heat_target = results[name]["target_heat"]
            product_target = results[name]["target_product"]
            probit_target = results[name]["target_probit"]
            axis.plot(
                n_values,
                (
                    np.full_like(n_array, heat_target)
                    if scaled
                    else heat_target / np.sqrt(n_array)
                ),
                color="navy",
                ls=":",
                lw=1.6,
                label="Bentkus theory",
            )
            axis.plot(
                n_values,
                (
                    np.full_like(n_array, product_target)
                    if scaled
                    else product_target / np.sqrt(n_array)
                ),
                color="seagreen",
                ls=":",
                lw=1.6,
                label="product theory",
            )
            axis.plot(
                n_values,
                (
                    np.full_like(n_array, probit_target)
                    if scaled
                    else probit_target / np.sqrt(n_array)
                ),
                color="black",
                ls=":",
                lw=1.6,
                label="Gaussian limit",
            )

            methods = (
                (
                    f"heat_original{suffix}",
                    "navy",
                    "o",
                    "Bentkus fixed claim",
                ),
                (
                    f"heat_star{suffix}",
                    "crimson",
                    "D",
                    "Bentkus STaR",
                ),
                (
                    f"product_original{suffix}",
                    "seagreen",
                    "s",
                    "WSR product comparator",
                ),
                (
                    f"product_star{suffix}",
                    "darkorange",
                    "P",
                    "product STaR-Bets",
                ),
                (
                    f"capped_feedback_star{suffix}",
                    "deeppink",
                    "*",
                    "target-capped quadratic STaR",
                ),
                (
                    f"capped_exponential_feedback_star{suffix}",
                    "teal",
                    "X",
                    "capped original STaR",
                ),
                (
                    f"probit_star{suffix}",
                    "purple",
                    "v",
                    "Probit STaR (randomized)",
                ),
            )
            for key, color, marker, label in methods:
                rows = results[name][key]
                means = [row["mean"] for row in rows]
                lows = [row["lo"] for row in rows]
                highs = [row["hi"] for row in rows]
                axis.plot(
                    n_values,
                    means,
                    color=color,
                    marker=marker,
                    ms=4.5,
                    lw=2,
                    label=label,
                )
                axis.fill_between(
                    n_values,
                    lows,
                    highs,
                    color=color,
                    alpha=0.08,
                )

            axis.set_xscale("log")
            if not scaled:
                axis.set_yscale("log")
            axis.set_title(name)
            axis.set_xlabel("n (log scale)")
            axis.set_ylabel(
                r"$\sqrt{n}\times$ CI width"
                if scaled
                else "CI width (log scale)"
            )
            axis.grid(True, ls="--", alpha=0.35)
            axis.legend(fontsize=7.2)

        scale_label = "scaled" if scaled else "raw"
        fig.suptitle(
            f"Fixed-plan versus STaR betting: {scale_label} widths "
            f"[\N{GREEK SMALL LETTER DELTA}={delta}, "
            f"sims/n={simulation_label}]"
        )
        plt.tight_layout()
        output = (
            "plots/ci_width_original_vs_star.png"
            if scaled
            else "plots/ci_width_raw_original_vs_star.png"
        )
        plt.savefig(output, dpi=180, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved to {output}")

    def plot_feedback_ablation(scaled):
        """Compare feedback maps with clock, cap, and stopping held fixed."""
        suffix = "" if scaled else "_raw"
        n_array = np.asarray(n_values, dtype=float)
        fig, axes = plt.subplots(2, 3, figsize=(16, 9))
        methods = (
            (f"product_star{suffix}", "darkorange", "P",
             "square-root/product feedback"),
            (f"hinge_feedback_star{suffix}", "crimson", "D",
             "squared-hinge feedback"),
            (f"capped_feedback_star{suffix}", "deeppink", "*",
             "target-capped quadratic feedback"),
            (f"capped_exponential_feedback_star{suffix}", "teal", "X",
             "capped original feedback"),
            (f"probit_star{suffix}", "purple", "v",
             "buffered randomized Probit"),
        )

        for axis, name in zip(axes.ravel(), samplers):
            gaussian_target = results[name]["target_probit"]
            axis.plot(
                n_values,
                (
                    np.full_like(n_array, gaussian_target)
                    if scaled
                    else gaussian_target / np.sqrt(n_array)
                ),
                color="black", ls=":", lw=1.6,
                label="Gaussian limit",
            )
            for key, color, marker, label in methods:
                rows = results[name][key]
                means = [row["mean"] for row in rows]
                lows = [row["lo"] for row in rows]
                highs = [row["hi"] for row in rows]
                axis.plot(
                    n_values, means, color=color, marker=marker,
                    ms=4.5, lw=2, label=label,
                )
                axis.fill_between(
                    n_values, lows, highs, color=color, alpha=0.08,
                )
            axis.set_xscale("log")
            if not scaled:
                axis.set_yscale("log")
            axis.set_title(name)
            axis.set_xlabel("n (log scale)")
            axis.set_ylabel(
                r"$\sqrt{n}\times$ CI width"
                if scaled else "CI width (log scale)"
            )
            axis.grid(True, ls="--", alpha=0.35)
            axis.legend(fontsize=7.2)

        scale_label = "scaled" if scaled else "raw"
        fig.suptitle(
            f"Chronological STaR feedback comparison: {scale_label} widths "
            f"[\N{GREEK SMALL LETTER DELTA}={delta}, "
            f"sims/n={simulation_label}]"
        )
        plt.tight_layout()
        output = (
            "plots/ci_width_feedback_ablation.png"
            if scaled else "plots/ci_width_raw_feedback_ablation.png"
        )
        plt.savefig(output, dpi=180, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved to {output}")

    plot_widths(scaled=True)
    plot_widths(scaled=False)
    plot_feedback_ablation(scaled=True)
    plot_feedback_ablation(scaled=False)

    summary_output = "plots/ci_width_original_vs_star.json"
    with open(summary_output, "w", encoding="utf-8") as output_file:
        json.dump(
            {
                "delta": delta,
                "seed": seed,
                "num_sims_by_n": {
                    str(n): num_sims_by_n[n] for n in n_values
                },
                "n_values": n_values,
                "strike": strike,
                "initial_wealth": initial_wealth,
                "probit_buffer_power": 2.0 / 3.0,
                "probit_terminal_randomization": True,
                "capped_ramp_width": _CAPPED_HINGE_RAMP,
                "capped_exponential_slope": "sqrt(2 log(1/p))",
                "capped_endpoint_scan": {
                    "kind": "geometric_first_crossing",
                    "points": 56,
                },
                "results": results,
            },
            output_file,
            indent=2,
        )
    print(f"Saved to {summary_output}")
    return results


# Backward-compatible name for callers of the earlier diagnostic.
run_convergence_experiment = run_experiment


if __name__ == "__main__":
    run_experiment(num_sims=PUBLICATION_SIMULATION_COUNTS)
