#!/usr/bin/env python3
"""Anytime-valid confidence sequences from product and heat-flow betting.

This module deliberately excludes the STaR and dynamic-programming strategies
in :mod:`betting`.  It implements five chronological e-processes:

* a hedged grid-Kelly (finite method-of-mixtures) product martingale;
* a horizon-free geometric scale mixture of product martingales;
* the signed approximate-GRAPA product martingale;
* a mixture of stopped, fixed-maturity Bentkus/heat-flow hedges; and
* the same heat mixture with every dollar stake capped by aGRAPA's stake.

All paths below include time zero, so an input of length ``n`` produces an
array of length ``n + 1`` whose first entry is zero on the log-e scale.
Retrospective permutation averaging is intentionally absent: it is not an
online predictable operation.
"""

from __future__ import annotations

import math
import time

import numpy as np
from numba import njit, prange
from scipy.optimize import brentq
from scipy.special import zeta

try:  # Importable both as a loose module and as ``betting.confidence_sequences``.
    from .betting import (
        _hinge_I2,
        get_optimal_lambda,
        heat_payoff_delta,
    )
except ImportError:  # pragma: no cover - depends on the caller's import path.
    try:
        from betting import _hinge_I2, get_optimal_lambda, heat_payoff_delta
    except ImportError:
        from betting.betting import (
            _hinge_I2,
            get_optimal_lambda,
            heat_payoff_delta,
        )


_LOG_ZERO = -math.inf


def _as_observations(X):
    values = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
    if values.ndim != 1:
        raise ValueError("X must be one-dimensional")
    if np.any(~np.isfinite(values)) or np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("X must contain finite observations in [0, 1]")
    return values


def _as_candidate_means(means):
    values = np.ascontiguousarray(np.asarray(means, dtype=np.float64))
    if values.ndim != 1 or not len(values):
        raise ValueError("means must be a nonempty one-dimensional array")
    if np.any(~np.isfinite(values)) or np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("candidate means must lie in [0, 1]")
    return values


def _as_times(times, n):
    values = np.asarray(times)
    if values.ndim != 1 or not len(values):
        raise ValueError("times must be a nonempty one-dimensional array")
    if np.any(~np.isfinite(values)) or np.any(values != values.astype(np.int64)):
        raise ValueError("times must contain integers")
    values = np.ascontiguousarray(values.astype(np.int64))
    if np.any(values < 0) or np.any(values > n):
        raise ValueError("times must lie between zero and len(X)")
    if np.any(values[1:] <= values[:-1]):
        raise ValueError("times must be strictly increasing")
    return values


def _validate_level(delta):
    value = float(delta)
    if not 0.0 < value < 1.0:
        raise ValueError("delta must lie strictly between zero and one")
    return value


@njit
def _logsumexp_equal_weight(log_values):
    maximum = _LOG_ZERO
    for value in log_values:
        if value > maximum:
            maximum = value
    if maximum == _LOG_ZERO:
        return _LOG_ZERO
    if math.isinf(maximum):
        return maximum
    total = 0.0
    for value in log_values:
        total += math.exp(value - maximum)
    return maximum + math.log(total / len(log_values))


@njit
def _hgkelly_log_e_path_kernel(X, m, G):
    """Hedged grid-Kelly with G positive and G negative components."""
    n = len(X)
    path = np.empty(n + 1)
    path[0] = 0.0
    component_logs = np.zeros(2 * G)

    for i in range(n):
        x = X[i]
        centered = x - m
        for g_index in range(G):
            fraction = (g_index + 1.0) / (G + 1.0)

            # At an endpoint the corresponding point null is degenerate.  The
            # singular arm stays as cash on a compatible observation and
            # rejects logically on an incompatible one.
            if m <= 0.0:
                if x > 0.0:
                    component_logs[g_index] = math.inf
            else:
                multiplier_plus = 1.0 + fraction * centered / m
                if multiplier_plus <= 0.0:
                    component_logs[g_index] = _LOG_ZERO
                elif component_logs[g_index] != _LOG_ZERO:
                    component_logs[g_index] += math.log(multiplier_plus)

            minus_index = G + g_index
            if m >= 1.0:
                if x < 1.0:
                    component_logs[minus_index] = math.inf
            else:
                multiplier_minus = 1.0 - fraction * centered / (1.0 - m)
                if multiplier_minus <= 0.0:
                    component_logs[minus_index] = _LOG_ZERO
                elif component_logs[minus_index] != _LOG_ZERO:
                    component_logs[minus_index] += math.log(multiplier_minus)

        path[i + 1] = _logsumexp_equal_weight(component_logs)
    return path


def hgkelly_log_e_path(X, m, G=20):
    """Return the log e-process of the hedged grid-Kelly mixture."""
    X = _as_observations(X)
    m = float(m)
    if not 0.0 <= m <= 1.0:
        raise ValueError("m must lie in [0, 1]")
    if int(G) != G or G <= 0:
        raise ValueError("G must be a positive integer")
    return _hgkelly_log_e_path_kernel(X, m, int(G))


def product_scale_schedule(
    max_time,
    weight_power=2.0,
    horizon_overshoot=2.0,
    scale_ratio=2.0,
):
    """Return the finite prefix of a horizon-free product scale mixture.

    Scale ``j`` uses the safe fraction ``r_j = scale_ratio**(-j / 2)`` and
    receives total (two-sided) mass
    ``1 / (zeta(weight_power) * j**weight_power)``.  Its nominal quadratic
    scale satisfies ``r_j**(-2) = scale_ratio**j``, matching the geometric
    maturity index of the Bentkus mixture; candidate-mean and variance factors
    still enter the component's statistical horizon.  The returned prefix
    ends at the first scale satisfying

    ``r_j <= 1 / sqrt(horizon_overshoot * max_time)``.

    All later scales from the countable mixture are retained as unbet cash.
    Thus truncation is exact, rather than a renormalization that changes the
    prior or silently spends the omitted mass.
    """
    if int(max_time) != max_time or max_time <= 0:
        raise ValueError("max_time must be a positive integer")
    weight_power = float(weight_power)
    horizon_overshoot = float(horizon_overshoot)
    scale_ratio = float(scale_ratio)
    if not math.isfinite(weight_power) or weight_power <= 1.0:
        raise ValueError("weight_power must exceed one")
    if (
        not math.isfinite(horizon_overshoot)
        or horizon_overshoot < 1.0
    ):
        raise ValueError("horizon_overshoot must be at least one")
    if not math.isfinite(scale_ratio) or scale_ratio <= 1.0:
        raise ValueError("scale_ratio must exceed one")

    terminal_scale = 1.0 / math.sqrt(
        horizon_overshoot * int(max_time)
    )
    fractions = [scale_ratio ** (-0.5)]
    while fractions[-1] > terminal_scale:
        scale_index = len(fractions) + 1
        fractions.append(scale_ratio ** (-0.5 * scale_index))
    fractions = np.asarray(fractions, dtype=np.float64)

    indices = np.arange(1, len(fractions) + 1, dtype=np.float64)
    weights = 1.0 / (
        float(zeta(weight_power, 1.0)) * indices**weight_power
    )
    cash_weight = max(1.0 - float(np.sum(weights)), 0.0)
    return fractions, weights, cash_weight


def _validate_product_scale_schedule(fractions, weights, cash_weight):
    fractions = np.ascontiguousarray(
        np.asarray(fractions, dtype=np.float64)
    )
    weights = np.ascontiguousarray(np.asarray(weights, dtype=np.float64))
    cash_weight = float(cash_weight)
    if fractions.ndim != 1 or weights.ndim != 1 or not len(fractions):
        raise ValueError(
            "fractions and weights must be nonempty one-dimensional arrays"
        )
    if len(fractions) != len(weights):
        raise ValueError("fractions and weights must have equal length")
    if (
        np.any(~np.isfinite(fractions))
        or np.any(fractions <= 0.0)
        or np.any(fractions > 1.0)
    ):
        raise ValueError("fractions must be finite and lie in (0, 1]")
    if np.any(fractions[1:] >= fractions[:-1]):
        raise ValueError("fractions must be strictly decreasing")
    if np.any(~np.isfinite(weights)) or np.any(weights <= 0.0):
        raise ValueError("weights must be finite and positive")
    if not np.isfinite(cash_weight) or cash_weight < -1e-14:
        raise ValueError("cash_weight must be nonnegative")
    total = float(np.sum(weights)) + cash_weight
    if not np.isclose(total, 1.0, rtol=0.0, atol=2e-12):
        raise ValueError("weights plus cash_weight must sum to one")
    return fractions, weights, max(cash_weight, 0.0)


@njit
def _product_scale_mixture_log_value(
    component_logs,
    log_direction_weights,
    log_cash_weight,
):
    maximum = log_cash_weight
    for j in range(len(log_direction_weights)):
        plus = component_logs[j] + log_direction_weights[j]
        minus = (
            component_logs[len(log_direction_weights) + j]
            + log_direction_weights[j]
        )
        if plus > maximum:
            maximum = plus
        if minus > maximum:
            maximum = minus
    if maximum == _LOG_ZERO:
        return _LOG_ZERO
    if math.isinf(maximum):
        return maximum

    total = 0.0
    if log_cash_weight != _LOG_ZERO:
        total += math.exp(log_cash_weight - maximum)
    for j in range(len(log_direction_weights)):
        total += math.exp(
            component_logs[j] + log_direction_weights[j] - maximum
        )
        total += math.exp(
            component_logs[len(log_direction_weights) + j]
            + log_direction_weights[j]
            - maximum
        )
    return maximum + math.log(total)


@njit
def _product_scale_mixture_log_e_path_kernel(
    X,
    m,
    fractions,
    weights,
    cash_weight,
):
    n = len(X)
    scale_count = len(fractions)
    component_logs = np.zeros(2 * scale_count)
    log_direction_weights = np.empty(scale_count)
    for j in range(scale_count):
        # Each scale divides its prior mass equally between directions.
        log_direction_weights[j] = math.log(0.5 * weights[j])
    log_cash_weight = (
        math.log(cash_weight) if cash_weight > 0.0 else _LOG_ZERO
    )

    path = np.empty(n + 1)
    path[0] = 0.0
    for i in range(n):
        x = X[i]
        centered = x - m
        for j in range(scale_count):
            fraction = fractions[j]
            if m <= 0.0:
                if x > 0.0:
                    component_logs[j] = math.inf
            else:
                multiplier_plus = 1.0 + fraction * centered / m
                if multiplier_plus <= 0.0:
                    component_logs[j] = _LOG_ZERO
                elif component_logs[j] != _LOG_ZERO:
                    component_logs[j] += math.log(multiplier_plus)

            minus_index = scale_count + j
            if m >= 1.0:
                if x < 1.0:
                    component_logs[minus_index] = math.inf
            else:
                multiplier_minus = 1.0 - fraction * centered / (1.0 - m)
                if multiplier_minus <= 0.0:
                    component_logs[minus_index] = _LOG_ZERO
                elif component_logs[minus_index] != _LOG_ZERO:
                    component_logs[minus_index] += math.log(multiplier_minus)

        path[i + 1] = _product_scale_mixture_log_value(
            component_logs,
            log_direction_weights,
            log_cash_weight,
        )
    return path


def product_scale_mixture_log_e_path(
    X,
    m,
    fractions=None,
    weights=None,
    cash_weight=None,
    weight_power=2.0,
    horizon_overshoot=2.0,
    scale_ratio=2.0,
):
    """Return the two-sided geometric product-mixture log e-process.

    With no explicit schedule, the finite prefix is tuned to ``len(X)`` by
    :func:`product_scale_schedule`.  To reuse a single schedule across paths,
    pass all three of ``fractions``, ``weights``, and ``cash_weight``.
    Prefix-by-prefix callers should likewise construct one schedule at their
    declared maximum time and reuse it, rather than changing the component
    menu retrospectively as the observed prefix grows.
    """
    X = _as_observations(X)
    m = float(m)
    if not 0.0 <= m <= 1.0:
        raise ValueError("m must lie in [0, 1]")
    provided = (
        fractions is not None,
        weights is not None,
        cash_weight is not None,
    )
    if any(provided) and not all(provided):
        raise ValueError(
            "fractions, weights, and cash_weight must be provided together"
        )
    if not any(provided):
        fractions, weights, cash_weight = product_scale_schedule(
            max(len(X), 1),
            weight_power=weight_power,
            horizon_overshoot=horizon_overshoot,
            scale_ratio=scale_ratio,
        )
    else:
        fractions, weights, cash_weight = (
            _validate_product_scale_schedule(
                fractions, weights, cash_weight
            )
        )
    return _product_scale_mixture_log_e_path_kernel(
        X, m, fractions, weights, cash_weight
    )


@njit
def _agrapa_log_e_path_kernel(
    X,
    m,
    c,
    prior_mean,
    prior_variance,
    fake_obs,
):
    n = len(X)
    path = np.empty(n + 1)
    path[0] = 0.0
    log_wealth = 0.0
    sum_x = 0.0
    variance_numerator = 0.0

    for i in range(n):
        denominator = fake_obs + i
        mean_hat = (fake_obs * prior_mean + sum_x) / denominator
        variance_hat = (
            fake_obs * prior_variance + variance_numerator
        ) / denominator
        difference = mean_hat - m
        raw_bet = difference / max(
            variance_hat + difference * difference, 1e-300
        )

        if m > 0.0:
            raw_bet = min(raw_bet, c / m)
        if m < 1.0:
            raw_bet = max(raw_bet, -c / (1.0 - m))

        x = X[i]
        if (m <= 0.0 and x > 0.0) or (m >= 1.0 and x < 1.0):
            log_wealth = math.inf
        elif not math.isinf(log_wealth):
            multiplier = 1.0 + raw_bet * (x - m)
            if multiplier <= 0.0:
                log_wealth = _LOG_ZERO
            elif log_wealth != _LOG_ZERO:
                log_wealth += math.log(multiplier)
        path[i + 1] = log_wealth

        sum_x += x
        updated_mean = (
            fake_obs * prior_mean + sum_x
        ) / (fake_obs + i + 1.0)
        residual = x - updated_mean
        variance_numerator += residual * residual
    return path


def agrapa_log_e_path(
    X,
    m,
    c=0.5,
    prior_mean=0.5,
    prior_variance=0.25,
    fake_obs=1.0,
):
    """Return the signed approximate-GRAPA product log e-process."""
    X = _as_observations(X)
    m = float(m)
    c = float(c)
    prior_mean = float(prior_mean)
    prior_variance = float(prior_variance)
    fake_obs = float(fake_obs)
    if not 0.0 <= m <= 1.0:
        raise ValueError("m must lie in [0, 1]")
    if not 0.0 < c <= 1.0:
        raise ValueError("c must lie in (0, 1]")
    if not 0.0 <= prior_mean <= 1.0:
        raise ValueError("prior_mean must lie in [0, 1]")
    if prior_variance <= 0.0 or fake_obs <= 0.0:
        raise ValueError("prior_variance and fake_obs must be positive")
    return _agrapa_log_e_path_kernel(
        X, m, c, prior_mean, prior_variance, fake_obs
    )


def bentkus_horizon_schedule(
    max_time,
    delta,
    horizon_ratio=2.0,
    weight_power=2.0,
    horizon_overshoot=2.0,
):
    """Construct a polynomially weighted geometric maturity schedule.

    The returned weights retain their infinite-sequence normalization.  The
    uninstantiated tail is returned as ``cash_weight`` and remains unbet cash,
    so the implemented finite mixture still starts exactly at one and remains
    valid after its final expert has stopped.
    """
    if int(max_time) != max_time or max_time <= 0:
        raise ValueError("max_time must be a positive integer")
    delta = _validate_level(delta)
    horizon_ratio = float(horizon_ratio)
    weight_power = float(weight_power)
    horizon_overshoot = float(horizon_overshoot)
    if horizon_ratio <= 1.0:
        raise ValueError("horizon_ratio must exceed one")
    if weight_power <= 1.0:
        raise ValueError("weight_power must exceed one")
    if horizon_overshoot < 1.0:
        raise ValueError("horizon_overshoot must be at least one")

    target = int(math.ceil(horizon_overshoot * int(max_time)))
    horizons = [1]
    while horizons[-1] < target:
        next_horizon = int(math.ceil(horizons[-1] * horizon_ratio))
        horizons.append(max(next_horizon, horizons[-1] + 1))
    horizons = np.asarray(horizons, dtype=np.int64)

    indices = np.arange(1, len(horizons) + 1, dtype=np.float64)
    weights = 1.0 / (float(zeta(weight_power, 1.0)) * indices**weight_power)
    cash_weight = max(1.0 - float(np.sum(weights)), 0.0)
    strikes = np.asarray(
        [get_optimal_lambda(delta * weight / 2.0)[0] for weight in weights],
        dtype=np.float64,
    )
    return horizons, strikes, weights, cash_weight


def _validate_heat_schedule(horizons, strikes, weights, cash_weight):
    horizons_raw = np.asarray(horizons)
    if horizons_raw.ndim != 1 or not len(horizons_raw):
        raise ValueError("horizons must be a nonempty one-dimensional array")
    if np.any(horizons_raw != horizons_raw.astype(np.int64)):
        raise ValueError("horizons must contain integers")
    horizons = np.ascontiguousarray(horizons_raw.astype(np.int64))
    strikes = np.ascontiguousarray(np.asarray(strikes, dtype=np.float64))
    weights = np.ascontiguousarray(np.asarray(weights, dtype=np.float64))
    cash_weight = float(cash_weight)
    if strikes.ndim != 1 or weights.ndim != 1:
        raise ValueError("strikes and weights must be one-dimensional")
    if not (len(horizons) == len(strikes) == len(weights)):
        raise ValueError("horizons, strikes, and weights must have equal length")
    if np.any(horizons <= 0) or np.any(horizons[1:] <= horizons[:-1]):
        raise ValueError("horizons must be positive and strictly increasing")
    if np.any(~np.isfinite(strikes)):
        raise ValueError("strikes must be finite")
    if np.any(~np.isfinite(weights)) or np.any(weights <= 0.0):
        raise ValueError("weights must be finite and positive")
    if not np.isfinite(cash_weight) or cash_weight < -1e-14:
        raise ValueError("cash_weight must be nonnegative")
    total = float(np.sum(weights)) + cash_weight
    if not np.isclose(total, 1.0, rtol=0.0, atol=2e-12):
        raise ValueError("weights plus cash_weight must sum to one")
    return horizons, strikes, weights, max(cash_weight, 0.0)


@njit
def _mixture_log_value(
    M_plus,
    M_minus,
    initial_wealth,
    weights,
    cash_weight,
):
    value = cash_weight
    for k in range(len(weights)):
        value += weights[k] * 0.5 * (
            M_plus[k] + M_minus[k]
        ) / initial_wealth[k]
    if value <= 0.0:
        return _LOG_ZERO
    return math.log(value)


@njit
def _checked_linear_wealth_update(old_wealth, signed_gain):
    """Apply a linear update, clipping only a negative roundoff residue."""
    updated = old_wealth + signed_gain
    if updated >= 0.0:
        return updated
    scale = max(abs(old_wealth), abs(signed_gain), 1e-300)
    tolerance = 64.0 * np.finfo(np.float64).eps * scale
    if updated >= -tolerance:
        return 0.0
    raise ValueError("linear wealth update violated its solvency bound")


@njit
def _bentkus_mixture_log_e_path_kernel(
    X,
    m,
    horizons,
    strikes,
    weights,
    cash_weight,
    solvency_fraction,
):
    n = len(X)
    expert_count = len(horizons)
    initial = np.empty(expert_count)
    M_plus = np.empty(expert_count)
    M_minus = np.empty(expert_count)
    S_plus = np.zeros(expert_count)
    S_minus = np.zeros(expert_count)
    for k in range(expert_count):
        initial[k] = _hinge_I2(strikes[k])
        M_plus[k] = initial[k]
        M_minus[k] = initial[k]

    path = np.empty(n + 1)
    path[0] = 0.0
    sum_x = 0.0
    predicted_square_sum = 0.0

    for i in range(n):
        mean_hat = (0.5 + sum_x) / (1.0 + i)
        variance_hat = (0.25 + predicted_square_sum) / (1.0 + i)
        x = X[i]
        centered = x - m

        for k in range(expert_count):
            horizon = horizons[k]
            if i >= horizon:
                continue
            gamma = 1.0 / math.sqrt(horizon * variance_hat)
            remaining_variance = 1.0 - i / float(horizon)
            beta_plus = gamma * heat_payoff_delta(
                S_plus[k], remaining_variance, strikes[k]
            )
            beta_minus = gamma * heat_payoff_delta(
                S_minus[k], remaining_variance, strikes[k]
            )

            old_plus = M_plus[k]
            old_minus = M_minus[k]
            plus_cap_bound = False
            minus_cap_bound = False
            if m > 0.0:
                cap_plus = solvency_fraction * old_plus / m
                if beta_plus >= cap_plus:
                    beta_plus = cap_plus
                    plus_cap_bound = True
            if m < 1.0:
                cap_minus = (
                    solvency_fraction * old_minus / (1.0 - m)
                )
                if beta_minus >= cap_minus:
                    beta_minus = cap_minus
                    minus_cap_bound = True

            # Avoid a cancellation residue when a unit cap binds exactly at
            # its worst endpoint; otherwise preserve the linear recursion.
            if plus_cap_bound and x == 0.0:
                M_plus[k] = (1.0 - solvency_fraction) * old_plus
            else:
                M_plus[k] = _checked_linear_wealth_update(
                    old_plus, beta_plus * centered
                )
            if minus_cap_bound and x == 1.0:
                M_minus[k] = (1.0 - solvency_fraction) * old_minus
            else:
                M_minus[k] = _checked_linear_wealth_update(
                    old_minus, -beta_minus * centered
                )
            # The standardized score follows the full information clock.
            S_plus[k] += gamma * centered
            S_minus[k] -= gamma * centered

        path[i + 1] = _mixture_log_value(
            M_plus, M_minus, initial, weights, cash_weight
        )
        residual = x - mean_hat
        sum_x += x
        predicted_square_sum += residual * residual
    return path


def bentkus_mixture_log_e_path(
    X,
    m,
    horizons,
    strikes,
    weights,
    cash_weight,
    solvency_fraction=1.0,
):
    """Return the stopped fixed-maturity Bentkus-mixture log e-process."""
    X = _as_observations(X)
    m = float(m)
    if not 0.0 <= m <= 1.0:
        raise ValueError("m must lie in [0, 1]")
    horizons, strikes, weights, cash_weight = _validate_heat_schedule(
        horizons, strikes, weights, cash_weight
    )
    solvency_fraction = float(solvency_fraction)
    if not 0.0 < solvency_fraction <= 1.0:
        raise ValueError("solvency_fraction must lie in (0, 1]")
    return _bentkus_mixture_log_e_path_kernel(
        X,
        m,
        horizons,
        strikes,
        weights,
        cash_weight,
        solvency_fraction,
    )


@njit
def _heat_constrained_agrapa_log_e_path_kernel(
    X,
    m,
    horizons,
    strikes,
    weights,
    cash_weight,
    agrapa_c,
    solvency_fraction,
):
    n = len(X)
    expert_count = len(horizons)
    initial = np.empty(expert_count)
    M_plus = np.empty(expert_count)
    M_minus = np.empty(expert_count)
    S_plus = np.zeros(expert_count)
    S_minus = np.zeros(expert_count)
    for k in range(expert_count):
        initial[k] = _hinge_I2(strikes[k])
        M_plus[k] = initial[k]
        M_minus[k] = initial[k]

    path = np.empty(n + 1)
    path[0] = 0.0
    sum_x = 0.0
    predicted_square_sum = 0.0
    agrapa_variance_numerator = 0.0

    for i in range(n):
        mean_hat = (0.5 + sum_x) / (1.0 + i)
        heat_variance_hat = (
            0.25 + predicted_square_sum
        ) / (1.0 + i)
        agrapa_variance_hat = (
            0.25 + agrapa_variance_numerator
        ) / (1.0 + i)
        difference = mean_hat - m
        signed_agrapa = difference / max(
            agrapa_variance_hat + difference * difference, 1e-300
        )
        if m > 0.0:
            signed_agrapa = min(signed_agrapa, agrapa_c / m)
        if m < 1.0:
            signed_agrapa = max(
                signed_agrapa, -agrapa_c / (1.0 - m)
            )
        lambda_plus = max(signed_agrapa, 0.0)
        lambda_minus = max(-signed_agrapa, 0.0)

        x = X[i]
        centered = x - m
        for k in range(expert_count):
            horizon = horizons[k]
            if i >= horizon:
                continue
            gamma = 1.0 / math.sqrt(horizon * heat_variance_hat)
            remaining_variance = 1.0 - i / float(horizon)
            raw_plus = gamma * heat_payoff_delta(
                S_plus[k], remaining_variance, strikes[k]
            )
            raw_minus = gamma * heat_payoff_delta(
                S_minus[k], remaining_variance, strikes[k]
            )
            beta_plus = min(raw_plus, M_plus[k] * lambda_plus)
            beta_minus = min(raw_minus, M_minus[k] * lambda_minus)
            old_plus = M_plus[k]
            old_minus = M_minus[k]
            plus_cap_bound = False
            minus_cap_bound = False
            if m > 0.0:
                cap_plus = solvency_fraction * old_plus / m
                if beta_plus >= cap_plus:
                    beta_plus = cap_plus
                    plus_cap_bound = True
            if m < 1.0:
                cap_minus = (
                    solvency_fraction * old_minus / (1.0 - m)
                )
                if beta_minus >= cap_minus:
                    beta_minus = cap_minus
                    minus_cap_bound = True

            if plus_cap_bound and x == 0.0:
                M_plus[k] = (1.0 - solvency_fraction) * old_plus
            else:
                M_plus[k] = _checked_linear_wealth_update(
                    old_plus, beta_plus * centered
                )
            if minus_cap_bound and x == 1.0:
                M_minus[k] = (1.0 - solvency_fraction) * old_minus
            else:
                M_minus[k] = _checked_linear_wealth_update(
                    old_minus, -beta_minus * centered
                )
            S_plus[k] += gamma * centered
            S_minus[k] -= gamma * centered

        path[i + 1] = _mixture_log_value(
            M_plus, M_minus, initial, weights, cash_weight
        )
        residual = x - mean_hat
        sum_x += x
        predicted_square_sum += residual * residual
        updated_mean = (0.5 + sum_x) / (2.0 + i)
        agrapa_residual = x - updated_mean
        agrapa_variance_numerator += agrapa_residual * agrapa_residual
    return path


def heat_constrained_agrapa_log_e_path(
    X,
    m,
    horizons,
    strikes,
    weights,
    cash_weight,
    agrapa_c=0.5,
    solvency_fraction=1.0,
):
    """Return the aGRAPA-capped fixed-claim heat-mixture log e-process."""
    X = _as_observations(X)
    m = float(m)
    if not 0.0 <= m <= 1.0:
        raise ValueError("m must lie in [0, 1]")
    horizons, strikes, weights, cash_weight = _validate_heat_schedule(
        horizons, strikes, weights, cash_weight
    )
    agrapa_c = float(agrapa_c)
    solvency_fraction = float(solvency_fraction)
    if not 0.0 < agrapa_c <= 1.0:
        raise ValueError("agrapa_c must lie in (0, 1]")
    if not 0.0 < solvency_fraction <= 1.0:
        raise ValueError("solvency_fraction must lie in (0, 1]")
    return _heat_constrained_agrapa_log_e_path_kernel(
        X,
        m,
        horizons,
        strikes,
        weights,
        cash_weight,
        agrapa_c,
        solvency_fraction,
    )


@njit
def _sample_running_maxima(path, times):
    result = np.empty(len(times))
    running = path[0]
    time_index = 0
    for t in range(len(path)):
        if path[t] > running:
            running = path[t]
        while time_index < len(times) and times[time_index] == t:
            result[time_index] = running
            time_index += 1
    return result


@njit(parallel=True)
def _hgkelly_running_surface(X, means, times, G):
    result = np.empty((len(means), len(times)))
    for j in prange(len(means)):
        result[j] = _sample_running_maxima(
            _hgkelly_log_e_path_kernel(X, means[j], G), times
        )
    return result


@njit(parallel=True)
def _product_scale_running_surface(
    X,
    means,
    times,
    fractions,
    weights,
    cash_weight,
):
    result = np.empty((len(means), len(times)))
    for j in prange(len(means)):
        result[j] = _sample_running_maxima(
            _product_scale_mixture_log_e_path_kernel(
                X,
                means[j],
                fractions,
                weights,
                cash_weight,
            ),
            times,
        )
    return result


@njit(parallel=True)
def _agrapa_running_surface(
    X, means, times, c, prior_mean, prior_variance, fake_obs
):
    result = np.empty((len(means), len(times)))
    for j in prange(len(means)):
        result[j] = _sample_running_maxima(
            _agrapa_log_e_path_kernel(
                X,
                means[j],
                c,
                prior_mean,
                prior_variance,
                fake_obs,
            ),
            times,
        )
    return result


@njit(parallel=True)
def _bentkus_running_surface(
    X,
    means,
    times,
    horizons,
    strikes,
    weights,
    cash_weight,
    solvency_fraction,
):
    result = np.empty((len(means), len(times)))
    for j in prange(len(means)):
        result[j] = _sample_running_maxima(
            _bentkus_mixture_log_e_path_kernel(
                X,
                means[j],
                horizons,
                strikes,
                weights,
                cash_weight,
                solvency_fraction,
            ),
            times,
        )
    return result


@njit(parallel=True)
def _constrained_running_surface(
    X,
    means,
    times,
    horizons,
    strikes,
    weights,
    cash_weight,
    agrapa_c,
    solvency_fraction,
):
    result = np.empty((len(means), len(times)))
    for j in prange(len(means)):
        result[j] = _sample_running_maxima(
            _heat_constrained_agrapa_log_e_path_kernel(
                X,
                means[j],
                horizons,
                strikes,
                weights,
                cash_weight,
                agrapa_c,
                solvency_fraction,
            ),
            times,
        )
    return result


def _heat_config(config):
    required = ("horizons", "strikes", "weights", "cash_weight")
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"heat method_config is missing {missing}")
    return _validate_heat_schedule(
        config["horizons"],
        config["strikes"],
        config["weights"],
        config["cash_weight"],
    )


def _product_scale_config(config, max_time):
    required = ("fractions", "weights", "cash_weight")
    provided = [key in config for key in required]
    if any(provided) and not all(provided):
        missing = [key for key in required if key not in config]
        raise ValueError(
            f"product scale method_config is missing {missing}"
        )
    if all(provided):
        return _validate_product_scale_schedule(
            config["fractions"],
            config["weights"],
            config["cash_weight"],
        )
    return product_scale_schedule(
        max(int(max_time), 1),
        weight_power=float(config.get("weight_power", 2.0)),
        horizon_overshoot=float(config.get("horizon_overshoot", 2.0)),
        scale_ratio=float(config.get("scale_ratio", 2.0)),
    )


def running_log_e_at_times(X, means, times, method, method_config=None):
    """Evaluate ``max_{s <= t} log E_s(m)`` on a candidate-mean grid."""
    X = _as_observations(X)
    means = _as_candidate_means(means)
    times = _as_times(times, len(X))
    config = {} if method_config is None else dict(method_config)
    method = str(method).lower().replace("-", "_")

    if method in ("hgkelly", "product_mixture"):
        raw_G = config.get("G", 20)
        if int(raw_G) != raw_G or raw_G <= 0:
            raise ValueError("G must be a positive integer")
        G = int(raw_G)
        return _hgkelly_running_surface(X, means, times, G)
    if method == "product_scale_mixture":
        fractions, weights, cash_weight = _product_scale_config(
            config, len(X)
        )
        return _product_scale_running_surface(
            X,
            means,
            times,
            fractions,
            weights,
            cash_weight,
        )
    if method in ("agrapa", "product_agrapa"):
        c = float(config.get("c", 0.5))
        prior_mean = float(config.get("prior_mean", 0.5))
        prior_variance = float(config.get("prior_variance", 0.25))
        fake_obs = float(config.get("fake_obs", 1.0))
        if not 0.0 < c <= 1.0:
            raise ValueError("c must lie in (0, 1]")
        if not 0.0 <= prior_mean <= 1.0:
            raise ValueError("prior_mean must lie in [0, 1]")
        if prior_variance <= 0.0 or fake_obs <= 0.0:
            raise ValueError("prior_variance and fake_obs must be positive")
        return _agrapa_running_surface(
            X,
            means,
            times,
            c,
            prior_mean,
            prior_variance,
            fake_obs,
        )
    if method in ("bentkus", "bentkus_mixture"):
        horizons, strikes, weights, cash_weight = _heat_config(config)
        solvency = float(config.get("solvency_fraction", 1.0))
        if not 0.0 < solvency <= 1.0:
            raise ValueError("solvency_fraction must lie in (0, 1]")
        return _bentkus_running_surface(
            X,
            means,
            times,
            horizons,
            strikes,
            weights,
            cash_weight,
            solvency,
        )
    if method in (
        "heat_constrained_agrapa",
        "bentkus_constrained_agrapa",
    ):
        horizons, strikes, weights, cash_weight = _heat_config(config)
        agrapa_c = float(config.get("agrapa_c", 0.5))
        solvency = float(config.get("solvency_fraction", 1.0))
        if not 0.0 < agrapa_c <= 1.0:
            raise ValueError("agrapa_c must lie in (0, 1]")
        if not 0.0 < solvency <= 1.0:
            raise ValueError("solvency_fraction must lie in (0, 1]")
        return _constrained_running_surface(
            X,
            means,
            times,
            horizons,
            strikes,
            weights,
            cash_weight,
            agrapa_c,
            solvency,
        )
    raise ValueError(f"unknown confidence-sequence method {method!r}")


def _path_function(X, method, config):
    method = str(method).lower().replace("-", "_")
    if method in ("hgkelly", "product_mixture"):
        G = int(config.get("G", 20))
        return lambda m, prefix: _hgkelly_log_e_path_kernel(
            X[:prefix], m, G
        )
    if method == "product_scale_mixture":
        fractions, weights, cash_weight = _product_scale_config(
            config, len(X)
        )
        return lambda m, prefix: _product_scale_mixture_log_e_path_kernel(
            X[:prefix],
            m,
            fractions,
            weights,
            cash_weight,
        )
    if method in ("agrapa", "product_agrapa"):
        c = float(config.get("c", 0.5))
        prior_mean = float(config.get("prior_mean", 0.5))
        prior_variance = float(config.get("prior_variance", 0.25))
        fake_obs = float(config.get("fake_obs", 1.0))
        return lambda m, prefix: _agrapa_log_e_path_kernel(
            X[:prefix],
            m,
            c,
            prior_mean,
            prior_variance,
            fake_obs,
        )
    if method in ("bentkus", "bentkus_mixture"):
        horizons, strikes, weights, cash_weight = _heat_config(config)
        solvency = float(config.get("solvency_fraction", 1.0))
        return lambda m, prefix: _bentkus_mixture_log_e_path_kernel(
            X[:prefix],
            m,
            horizons,
            strikes,
            weights,
            cash_weight,
            solvency,
        )
    if method in (
        "heat_constrained_agrapa",
        "bentkus_constrained_agrapa",
    ):
        horizons, strikes, weights, cash_weight = _heat_config(config)
        agrapa_c = float(config.get("agrapa_c", 0.5))
        solvency = float(config.get("solvency_fraction", 1.0))
        return lambda m, prefix: _heat_constrained_agrapa_log_e_path_kernel(
            X[:prefix],
            m,
            horizons,
            strikes,
            weights,
            cash_weight,
            agrapa_c,
            solvency,
        )
    raise ValueError(f"unknown confidence-sequence method {method!r}")


def _component_count(mask):
    if not np.any(mask):
        return 0
    return int(np.sum(mask & ~np.r_[False, mask[:-1]]))


def confidence_sequence_endpoints(
    X,
    delta,
    times,
    method,
    method_config=None,
    topology_grid_size=129,
    root_xtol=1e-8,
    return_diagnostics=True,
):
    """Invert a running e-process and return convex-hull CS endpoints.

    A global topology grid is augmented by every reporting-time sample mean.
    The outer accepted-to-rejected transitions are refined by Brent's method.
    Convex hulling is conservative for disconnected accepted sets.  The
    topology diagnostics make the finite numerical scan explicit; they are
    not a proof that an arbitrarily narrow component cannot lie between grid
    points.
    """
    X = _as_observations(X)
    delta = _validate_level(delta)
    times = _as_times(times, len(X))
    if int(topology_grid_size) != topology_grid_size or topology_grid_size < 3:
        raise ValueError("topology_grid_size must be an integer at least three")
    root_xtol = float(root_xtol)
    if root_xtol <= 0.0:
        raise ValueError("root_xtol must be positive")
    config = {} if method_config is None else dict(method_config)

    cumulative = np.r_[0.0, np.cumsum(X)]
    centers = np.asarray(
        [0.5 if t == 0 else cumulative[t] / t for t in times],
        dtype=np.float64,
    )
    base_grid = np.linspace(0.0, 1.0, int(topology_grid_size))
    means = np.unique(np.r_[base_grid, centers])
    surface = running_log_e_at_times(
        X, means, times, method, method_config=config
    )
    threshold = math.log(1.0 / delta)
    path_function = _path_function(X, method, config)

    lower = np.empty(len(times))
    upper = np.empty(len(times))
    empty = np.zeros(len(times), dtype=bool)
    components = np.zeros(len(times), dtype=np.int64)
    uncertain = np.zeros(len(times), dtype=bool)
    accepted_counts = np.zeros(len(times), dtype=np.int64)

    for time_index, time in enumerate(times):
        accepted = surface[:, time_index] < threshold
        accepted_counts[time_index] = int(np.sum(accepted))
        components[time_index] = _component_count(accepted)
        if not np.any(accepted):
            lower[time_index] = math.nan
            upper[time_index] = math.nan
            empty[time_index] = True
            uncertain[time_index] = True
            continue

        indices = np.flatnonzero(accepted)
        first = int(indices[0])
        last = int(indices[-1])
        # A singleton grid hit or any disconnected set merits a denser audit.
        uncertain[time_index] = (
            components[time_index] > 1
            or np.any(
                np.diff(indices) > 1
            )
            or (last - first + 1 <= 1)
        )

        def centered_running_log_e(candidate):
            path = path_function(float(candidate), int(time))
            return float(np.max(path)) - threshold

        if first == 0:
            lower[time_index] = 0.0
        else:
            rejected_mean = float(means[first - 1])
            accepted_mean = float(means[first])
            f_rejected = centered_running_log_e(rejected_mean)
            f_accepted = centered_running_log_e(accepted_mean)
            if f_rejected >= 0.0 and f_accepted < 0.0:
                lower[time_index] = brentq(
                    centered_running_log_e,
                    rejected_mean,
                    accepted_mean,
                    xtol=root_xtol,
                    rtol=1e-12,
                    maxiter=100,
                )
            else:
                lower[time_index] = accepted_mean
                uncertain[time_index] = True

        if last == len(means) - 1:
            upper[time_index] = 1.0
        else:
            accepted_mean = float(means[last])
            rejected_mean = float(means[last + 1])
            f_accepted = centered_running_log_e(accepted_mean)
            f_rejected = centered_running_log_e(rejected_mean)
            if f_accepted < 0.0 and f_rejected >= 0.0:
                upper[time_index] = brentq(
                    centered_running_log_e,
                    accepted_mean,
                    rejected_mean,
                    xtol=root_xtol,
                    rtol=1e-12,
                    maxiter=100,
                )
            else:
                upper[time_index] = accepted_mean
                uncertain[time_index] = True

    result = {
        "times": times,
        "lower": lower,
        "upper": upper,
        # The diameter of the empty set is zero.  Keep NaN endpoints so an
        # empty set cannot be mistaken for the singleton {0}.
        "width": np.where(empty, 0.0, upper - lower),
    }
    if return_diagnostics:
        result.update(
            {
                "empty": empty,
                "component_count": components,
                "accepted_grid_count": accepted_counts,
                "topology_uncertain": uncertain,
                "topology_grid_size": int(topology_grid_size),
                "candidate_grid": means,
            }
        )
    return result


def default_cs_times(max_time, count=32):
    """Return early integer times plus a logarithmic reporting grid."""
    if int(max_time) != max_time or max_time <= 0:
        raise ValueError("max_time must be a positive integer")
    if int(count) != count or count < 2:
        raise ValueError("count must be an integer at least two")
    max_time = int(max_time)
    early = np.arange(1, min(max_time, 9) + 1, dtype=np.int64)
    if max_time <= 9:
        return early
    logarithmic = np.rint(
        np.geomspace(10.0, float(max_time), int(count))
    ).astype(np.int64)
    return np.unique(np.r_[early, logarithmic, max_time])


_DISTRIBUTION_NAMES = (
    "Beta(2,2)",
    "Beta(1,5)",
    "Bernoulli(0.5)",
    "Uniform(0,1)",
    "Beta(0.5,0.5)",
    "Bernoulli(0.1)",
)

_TRUE_MEANS = {
    "Beta(2,2)": 0.5,
    "Beta(1,5)": 1.0 / 6.0,
    "Bernoulli(0.5)": 0.5,
    "Uniform(0,1)": 0.5,
    "Beta(0.5,0.5)": 0.5,
    "Bernoulli(0.1)": 0.1,
}


def _sample_distribution(rng, name, size):
    if name == "Beta(2,2)":
        return rng.beta(2.0, 2.0, size)
    if name == "Beta(1,5)":
        return rng.beta(1.0, 5.0, size)
    if name == "Bernoulli(0.5)":
        return rng.binomial(1, 0.5, size).astype(float)
    if name == "Uniform(0,1)":
        return rng.uniform(0.0, 1.0, size)
    if name == "Beta(0.5,0.5)":
        return rng.beta(0.5, 0.5, size)
    if name == "Bernoulli(0.1)":
        return rng.binomial(1, 0.1, size).astype(float)
    raise ValueError(f"unknown distribution {name!r}")


def _summary_rows(values):
    values = np.asarray(values, dtype=np.float64)
    return [
        {
            "mean": float(np.nanmean(values[:, index])),
            "median": float(np.nanmedian(values[:, index])),
            "lo": float(np.nanquantile(values[:, index], 0.1)),
            "hi": float(np.nanquantile(values[:, index], 0.9)),
        }
        for index in range(values.shape[1])
    ]


def _scalar_summary(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.nanmean(values)),
        "median": float(np.nanmedian(values)),
        "lo": float(np.nanquantile(values, 0.1)),
        "hi": float(np.nanquantile(values, 0.9)),
    }


def _finite_positive_denominator_ratio(numerator, denominator):
    """Return ratios and the mask of finite pairs with positive denominator."""
    numerator = np.asarray(numerator, dtype=np.float64)
    denominator = np.asarray(denominator, dtype=np.float64)
    if numerator.shape != denominator.shape:
        raise ValueError("paired arrays must have the same shape")
    mask = (
        np.isfinite(numerator)
        & np.isfinite(denominator)
        & (denominator > 0.0)
    )
    return numerator[mask] / denominator[mask], mask


def _summary_or_none(values):
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return {key: None for key in ("mean", "median", "lo", "hi")}
    return _scalar_summary(values)


def _wilson_interval(successes, total, z_value=1.959963984540054):
    if total <= 0:
        return math.nan, math.nan
    probability = successes / float(total)
    z2 = z_value * z_value
    denominator = 1.0 + z2 / total
    center = (probability + z2 / (2.0 * total)) / denominator
    radius = z_value / denominator * math.sqrt(
        probability * (1.0 - probability) / total
        + z2 / (4.0 * total * total)
    )
    return max(center - radius, 0.0), min(center + radius, 1.0)


def _warm_confidence_sequence_methods(method_configs, delta):
    """Compile scalar, parallel-surface, and inversion paths before timing."""
    warm_X = np.asarray([0.0, 1.0, 0.25, 0.75], dtype=np.float64)
    warm_means = np.asarray([0.25, 0.5, 0.75], dtype=np.float64)
    warm_times = np.asarray([2, 4], dtype=np.int64)
    for method, config in method_configs.items():
        _path_function(warm_X, method, config)(0.5, len(warm_X))
        running_log_e_at_times(
            warm_X,
            warm_means,
            warm_times,
            method,
            method_config=config,
        )
        confidence_sequence_endpoints(
            warm_X,
            delta,
            warm_times,
            method,
            method_config=config,
            topology_grid_size=5,
        )


def run_confidence_sequence_experiment(
    delta=0.01,
    max_time=100_000,
    times=None,
    num_width_sims=50,
    coverage_max_time=10_000,
    num_coverage_sims=0,
    seed=20260717,
    product_grid_size=20,
    topology_grid_size=129,
    progress=False,
):
    """Run the paired five-method confidence-sequence benchmark.

    The return value contains only ordinary Python scalars, lists, and dicts,
    and can therefore be passed directly to ``json.dump``.  Production-scale
    defaults are provided for widths, while coverage simulation is opt-in so
    a bare call does not unexpectedly add thousands of paths.  Reported
    runtimes cover width inversion only: all method kernels are warmed before
    timing, and coverage paths are excluded.
    """
    delta = _validate_level(delta)
    if int(max_time) != max_time or max_time <= 0:
        raise ValueError("max_time must be a positive integer")
    if int(coverage_max_time) != coverage_max_time or coverage_max_time <= 0:
        raise ValueError("coverage_max_time must be a positive integer")
    if int(num_width_sims) != num_width_sims or num_width_sims <= 0:
        raise ValueError("num_width_sims must be a positive integer")
    if int(num_coverage_sims) != num_coverage_sims or num_coverage_sims < 0:
        raise ValueError("num_coverage_sims must be a nonnegative integer")
    if int(product_grid_size) != product_grid_size or product_grid_size <= 0:
        raise ValueError("product_grid_size must be a positive integer")
    max_time = int(max_time)
    coverage_max_time = int(coverage_max_time)
    num_width_sims = int(num_width_sims)
    num_coverage_sims = int(num_coverage_sims)
    product_grid_size = int(product_grid_size)
    if times is None:
        times = default_cs_times(max_time)
    else:
        times = _as_times(times, max_time)

    schedule_time = max(
        max_time,
        coverage_max_time if num_coverage_sims else max_time,
    )
    horizons, strikes, weights, cash = bentkus_horizon_schedule(
        schedule_time, delta
    )
    scale_fractions, scale_weights, scale_cash = product_scale_schedule(
        schedule_time,
        weight_power=2.0,
        horizon_overshoot=2.0,
        scale_ratio=2.0,
    )
    heat_config = {
        "horizons": horizons,
        "strikes": strikes,
        "weights": weights,
        "cash_weight": cash,
    }
    method_configs = {
        "hgkelly": {"G": product_grid_size},
        "product_scale_mixture": {
            "fractions": scale_fractions,
            "weights": scale_weights,
            "cash_weight": scale_cash,
        },
        "agrapa": {"c": 0.5},
        "bentkus_mixture": heat_config,
        "heat_constrained_agrapa": heat_config,
    }
    method_names = tuple(method_configs)
    _warm_confidence_sequence_methods(method_configs, delta)

    seed_sequence = np.random.SeedSequence(int(seed))
    width_seed, coverage_seed = seed_sequence.spawn(2)
    width_rng = np.random.default_rng(width_seed)
    coverage_rng = np.random.default_rng(coverage_seed)
    threshold = math.log(1.0 / delta)
    auc_mask = times >= min(10, max_time)
    auc_times = times[auc_mask].astype(np.float64)

    results = {}
    for distribution in _DISTRIBUTION_NAMES:
        if progress:
            print(f"widths: {distribution}", flush=True)
        widths = {
            method: np.full((num_width_sims, len(times)), np.nan)
            for method in method_names
        }
        empty = {
            method: np.zeros((num_width_sims, len(times)), dtype=bool)
            for method in method_names
        }
        disconnected = {
            method: np.zeros((num_width_sims, len(times)), dtype=bool)
            for method in method_names
        }
        uncertain = {
            method: np.zeros((num_width_sims, len(times)), dtype=bool)
            for method in method_names
        }
        runtime = {method: 0.0 for method in method_names}

        for simulation in range(num_width_sims):
            X = _sample_distribution(width_rng, distribution, max_time)
            for method, config in method_configs.items():
                started = time.perf_counter()
                inverted = confidence_sequence_endpoints(
                    X,
                    delta,
                    times,
                    method,
                    method_config=config,
                    topology_grid_size=topology_grid_size,
                )
                runtime[method] += time.perf_counter() - started
                widths[method][simulation] = inverted["width"]
                empty[method][simulation] = inverted["empty"]
                disconnected[method][simulation] = (
                    inverted["component_count"] > 1
                )
                uncertain[method][simulation] = inverted[
                    "topology_uncertain"
                ]

        distribution_results = {}
        auc_values = {}
        for method in method_names:
            if len(auc_times) <= 1:
                auc = widths[method][:, -1]
            else:
                auc = np.trapezoid(
                    widths[method][:, auc_mask],
                    x=np.log(auc_times),
                    axis=1,
                ) / math.log(auc_times[-1] / auc_times[0])
            auc_values[method] = auc
            distribution_results[method] = {
                "width": _summary_rows(widths[method]),
                "terminal_width": _scalar_summary(widths[method][:, -1]),
                "log_time_auc": _scalar_summary(auc),
                "empty_rate": np.mean(empty[method], axis=0).tolist(),
                "disconnected_rate": np.mean(
                    disconnected[method], axis=0
                ).tolist(),
                "topology_uncertain_rate": np.mean(
                    uncertain[method], axis=0
                ).tolist(),
                "runtime_seconds": float(runtime[method]),
                "runtime_scope": "width_inversion_excluding_jit_and_coverage",
                "per_path_widths": widths[method].tolist(),
                "per_path_empty": empty[method].tolist(),
                "per_path_disconnected": disconnected[method].tolist(),
                "per_path_topology_uncertain": uncertain[method].tolist(),
                "coverage": None,
            }

        pair_specs = {
            "bentkus_over_hgkelly": (
                "bentkus_mixture",
                "hgkelly",
            ),
            "bentkus_over_product_scale_mixture": (
                "bentkus_mixture",
                "product_scale_mixture",
            ),
            "constrained_heat_over_agrapa": (
                "heat_constrained_agrapa",
                "agrapa",
            ),
        }
        paired = {}
        for label, (numerator, denominator) in pair_specs.items():
            terminal_numerator = widths[numerator][:, -1]
            terminal_denominator = widths[denominator][:, -1]
            terminal_ratio, terminal_mask = (
                _finite_positive_denominator_ratio(
                    terminal_numerator, terminal_denominator
                )
            )
            auc_ratio, auc_pair_mask = _finite_positive_denominator_ratio(
                auc_values[numerator], auc_values[denominator]
            )
            terminal_pair_count = int(np.sum(terminal_mask))
            auc_pair_count = int(np.sum(auc_pair_mask))
            paired[label] = {
                "terminal_width_ratio": _summary_or_none(terminal_ratio),
                "log_time_auc_ratio": _summary_or_none(auc_ratio),
                "terminal_win_rate": (
                    float(
                        np.mean(
                            terminal_numerator[terminal_mask]
                            < terminal_denominator[terminal_mask]
                        )
                    )
                    if terminal_pair_count
                    else None
                ),
                "terminal_finite_pair_count": terminal_pair_count,
                "log_time_auc_finite_pair_count": auc_pair_count,
            }

        if num_coverage_sims:
            if progress:
                print(f"coverage: {distribution}", flush=True)
            crossings = {method: 0 for method in method_names}
            true_mean = _TRUE_MEANS[distribution]
            for _ in range(num_coverage_sims):
                X = _sample_distribution(
                    coverage_rng, distribution, coverage_max_time
                )
                for method, config in method_configs.items():
                    path = _path_function(X, method, config)(
                        true_mean, coverage_max_time
                    )
                    crossings[method] += int(np.max(path) >= threshold)
            for method in method_names:
                lower, upper = _wilson_interval(
                    crossings[method], num_coverage_sims
                )
                distribution_results[method]["coverage"] = {
                    "crossings": int(crossings[method]),
                    "total": num_coverage_sims,
                    "crossing_rate": (
                        crossings[method] / float(num_coverage_sims)
                    ),
                    "wilson_lower": float(lower),
                    "wilson_upper": float(upper),
                }

        distribution_results["paired"] = paired
        results[distribution] = distribution_results

    return {
        "delta": delta,
        "max_time": max_time,
        "times": times.tolist(),
        "num_width_sims": num_width_sims,
        "coverage_max_time": coverage_max_time,
        "num_coverage_sims": num_coverage_sims,
        "seed": int(seed),
        "topology_grid_size": int(topology_grid_size),
        "runtime_scope": "width_inversion_excluding_jit_and_coverage",
        "methods": {
            "hgkelly": {"G": product_grid_size},
            "product_scale_mixture": {
                "scale_ratio": 2.0,
                "weight_power": 2.0,
                "horizon_overshoot": 2.0,
            },
            "agrapa": {"c": 0.5},
            "bentkus_mixture": {
                "solvency_fraction": 1.0,
                "maturity_ratio": 2.0,
                "weight_power": 2.0,
            },
            "heat_constrained_agrapa": {
                "agrapa_c": 0.5,
                "solvency_fraction": 1.0,
                "maturity_ratio": 2.0,
                "weight_power": 2.0,
            },
        },
        "schedule": {
            "schedule_time": schedule_time,
            "horizons": horizons.tolist(),
            "strikes": strikes.tolist(),
            "weights": weights.tolist(),
            "cash_weight": float(cash),
            "product_scale_fractions": scale_fractions.tolist(),
            "product_scale_weights": scale_weights.tolist(),
            "product_scale_cash_weight": float(scale_cash),
        },
        "true_means": dict(_TRUE_MEANS),
        "results": results,
    }


def run_confidence_sequence_smoke_experiment(
    delta=0.05,
    max_time=100,
    num_sims=1,
    seed=20260717,
    topology_grid_size=17,
):
    """Run a tiny, in-memory five-method integration experiment.

    This helper intentionally writes no files and is not the publication-scale
    driver.  It exists to exercise shared paths, schedules, inversion, and the
    result schema before a costly simulation is launched.
    """
    delta = _validate_level(delta)
    if int(max_time) != max_time or max_time <= 0:
        raise ValueError("max_time must be a positive integer")
    if int(num_sims) != num_sims or num_sims <= 0:
        raise ValueError("num_sims must be a positive integer")
    rng = np.random.default_rng(seed)
    times = default_cs_times(int(max_time), count=8)
    horizons, strikes, weights, cash = bentkus_horizon_schedule(
        int(max_time), delta
    )
    heat_config = {
        "horizons": horizons,
        "strikes": strikes,
        "weights": weights,
        "cash_weight": cash,
    }
    scale_fractions, scale_weights, scale_cash = product_scale_schedule(
        int(max_time)
    )
    scale_config = {
        "fractions": scale_fractions,
        "weights": scale_weights,
        "cash_weight": scale_cash,
    }
    methods = {
        "hgkelly": {"G": 20},
        "product_scale_mixture": scale_config,
        "agrapa": {"c": 0.5},
        "bentkus_mixture": heat_config,
        "heat_constrained_agrapa": heat_config,
    }
    samplers = {
        "Beta(2,2)": lambda: rng.beta(2.0, 2.0, int(max_time)),
        "Beta(1,5)": lambda: rng.beta(1.0, 5.0, int(max_time)),
        "Bernoulli(0.5)": lambda: rng.binomial(
            1, 0.5, int(max_time)
        ).astype(float),
        "Uniform(0,1)": lambda: rng.uniform(0.0, 1.0, int(max_time)),
        "Beta(0.5,0.5)": lambda: rng.beta(
            0.5, 0.5, int(max_time)
        ),
        "Bernoulli(0.1)": lambda: rng.binomial(
            1, 0.1, int(max_time)
        ).astype(float),
    }
    results = {}
    for distribution, sampler in samplers.items():
        method_widths = {method: [] for method in methods}
        for _ in range(int(num_sims)):
            X = sampler()
            for method, config in methods.items():
                cs = confidence_sequence_endpoints(
                    X,
                    delta,
                    times,
                    method,
                    method_config=config,
                    topology_grid_size=topology_grid_size,
                )
                method_widths[method].append(cs["width"])
        results[distribution] = {
            method: np.nanmean(np.asarray(widths), axis=0).tolist()
            for method, widths in method_widths.items()
        }
    return {
        "delta": delta,
        "max_time": int(max_time),
        "num_sims": int(num_sims),
        "seed": int(seed),
        "times": times.tolist(),
        "horizons": horizons.tolist(),
        "weights": weights.tolist(),
        "cash_weight": float(cash),
        "product_scale_fractions": scale_fractions.tolist(),
        "product_scale_weights": scale_weights.tolist(),
        "product_scale_cash_weight": float(scale_cash),
        "results": results,
    }


__all__ = [
    "agrapa_log_e_path",
    "bentkus_horizon_schedule",
    "bentkus_mixture_log_e_path",
    "confidence_sequence_endpoints",
    "default_cs_times",
    "heat_constrained_agrapa_log_e_path",
    "hgkelly_log_e_path",
    "product_scale_mixture_log_e_path",
    "product_scale_schedule",
    "run_confidence_sequence_smoke_experiment",
    "run_confidence_sequence_experiment",
    "running_log_e_at_times",
]
