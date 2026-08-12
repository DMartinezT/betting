#!/usr/bin/env python3
"""Fixed-event survival confidence-interval comparison.

The experiment compares four intervals evaluated only after a prespecified
number of failures:

* the classical logrank score interval;
* the horizon-aware GE-logrank interval;
* the prequential plug-in AV-logrank interval of ter Schure et al.; and
* their two-sided conditionally-GROW construction, tuned to HR 0.70; and
* a mixture, made once at time zero, of the two directional point-alternative
  AV e-processes at reciprocal hazard ratios.

No method is monitored before the terminal event count.  The AV procedures
are included as finite-sample e-value comparators, not for their
time-uniform guarantee.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numba import njit, prange
from scipy.stats import norm


METHOD_KEYS = (
    "classical",
    "ge",
    "av_preq",
    "av_grow",
    "av_path_mix",
)
METHOD_LABELS = {
    "classical": "Classical logrank",
    "ge": "GE-logrank",
    "av_preq": "AV-prequential",
    "av_grow": "AV conditional-GROW",
    "av_path_mix": "AV path mixture",
}
METHOD_COLORS = {
    "classical": "#222222",
    "ge": "#0072B2",
    "av_preq": "#D55E00",
    "av_grow": "#009E73",
    "av_path_mix": "#CC79A7",
}
METHOD_MARKERS = {
    "classical": "o",
    "ge": "s",
    "av_preq": "^",
    "av_grow": "D",
    "av_path_mix": "P",
}


@dataclass(frozen=True)
class HazardScenario:
    key: str
    label: str
    treatment_hazard_early: float
    treatment_hazard_late: float
    change_time: float
    true_log_hr: float | None


@dataclass(frozen=True)
class CensorScenario:
    key: str
    label: str
    control_rate: float
    treatment_rate: float


HAZARD_SCENARIOS = (
    HazardScenario("null", "Null", 1.0, 1.0, 0.35, 0.0),
    HazardScenario(
        "ph_070",
        "PH 0.70",
        0.70,
        0.70,
        0.35,
        math.log(0.70),
    ),
    HazardScenario(
        "delayed",
        "Delayed 1.00→0.45",
        1.0,
        0.45,
        0.35,
        None,
    ),
    HazardScenario(
        "crossing",
        "Crossing 0.45→1.80",
        0.45,
        1.80,
        0.35,
        None,
    ),
)

CENSOR_SCENARIOS = (
    CensorScenario("none", "No censoring", 0.0, 0.0),
    CensorScenario("balanced", "Balanced censoring", 0.45, 0.45),
    CensorScenario("differential", "Differential censoring", 0.20, 0.75),
)


# ---------------------------------------------------------------------------
# Numerical primitives
# ---------------------------------------------------------------------------


@njit(cache=True)
def _normal_cdf(x):
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


@njit(cache=True)
def _normal_pdf(x):
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


@njit(cache=True)
def _normal_ppf(probability):
    """Acklam inverse-normal approximation followed by one Halley step."""
    if probability <= 0.0:
        return -np.inf
    if probability >= 1.0:
        return np.inf

    reflect = probability > 0.5
    probability_left = 1.0 - probability if reflect else probability
    p_low = 0.02425

    if probability_left < p_low:
        q = math.sqrt(-2.0 * math.log(probability_left))
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


@njit(cache=True)
def _probit_leverage(target_fraction):
    if target_fraction <= 0.0:
        return np.inf
    if target_fraction >= 1.0:
        return 0.0
    quantile = _normal_ppf(target_fraction)
    return math.exp(
        -0.5 * quantile * quantile
        - 0.5 * math.log(2.0 * math.pi)
        - math.log(target_fraction)
    )


@njit(cache=True)
def _risk_probability(risk_control, risk_treatment, log_hr):
    if risk_treatment <= 0:
        return 0.0
    if risk_control <= 0:
        return 1.0
    log_odds = (
        log_hr
        + math.log(float(risk_treatment))
        - math.log(float(risk_control))
    )
    if log_odds >= 0.0:
        return 1.0 / (1.0 + math.exp(-log_odds))
    odds = math.exp(log_odds)
    return odds / (1.0 + odds)


@njit(cache=True)
def _bernoulli_log_probability(outcome, probability):
    probability = min(max(probability, 1e-300), 1.0 - 1e-15)
    if outcome == 1:
        return math.log(probability)
    return math.log1p(-probability)


# ---------------------------------------------------------------------------
# Event-time data generation
# ---------------------------------------------------------------------------


@njit(cache=True)
def _piecewise_exponential_time(
    exponential_draw,
    hazard_early,
    hazard_late,
    change_time,
):
    early_integrated_hazard = hazard_early * change_time
    if exponential_draw <= early_integrated_hazard:
        return exponential_draw / hazard_early
    return (
        change_time
        + (exponential_draw - early_integrated_hazard) / hazard_late
    )


@njit(cache=True)
def _risk_set_sequence(
    treatment,
    event_times,
    censor_times,
    event_horizon,
):
    """Return risk sets and failing arms for the first event_horizon failures."""
    subject_count = treatment.size
    observed_times = np.empty(subject_count)
    event_sort_key = np.empty(subject_count)
    event_indicators = np.empty(subject_count, dtype=np.int8)

    risk_time_control = np.empty(subject_count)
    risk_time_treatment = np.empty(subject_count)
    control_count = 0
    treatment_count = 0
    event_count = 0

    for subject in range(subject_count):
        is_event = event_times[subject] <= censor_times[subject]
        event_indicators[subject] = 1 if is_event else 0
        observed = (
            event_times[subject] if is_event else censor_times[subject]
        )
        observed_times[subject] = observed
        event_sort_key[subject] = event_times[subject] if is_event else np.inf
        if is_event:
            event_count += 1
        if treatment[subject] == 1:
            risk_time_treatment[treatment_count] = observed
            treatment_count += 1
        else:
            risk_time_control[control_count] = observed
            control_count += 1

    if event_count < event_horizon:
        return (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int8),
            np.nan,
            np.nan,
        )

    sorted_control = np.sort(risk_time_control[:control_count])
    sorted_treatment = np.sort(risk_time_treatment[:treatment_count])
    event_order = np.argsort(event_sort_key)

    risk_control = np.empty(event_horizon, dtype=np.int64)
    risk_treatment = np.empty(event_horizon, dtype=np.int64)
    failing_arm = np.empty(event_horizon, dtype=np.int8)

    for event_index in range(event_horizon):
        subject = event_order[event_index]
        event_time = event_times[subject]
        risk_control[event_index] = (
            control_count - np.searchsorted(sorted_control, event_time)
        )
        risk_treatment[event_index] = (
            treatment_count - np.searchsorted(sorted_treatment, event_time)
        )
        failing_arm[event_index] = treatment[subject]

    terminal_time = event_times[event_order[event_horizon - 1]]
    censoring_exits = 0
    total_exits = event_horizon
    for subject in range(subject_count):
        if (
            event_indicators[subject] == 0
            and observed_times[subject] <= terminal_time
        ):
            censoring_exits += 1
            total_exits += 1
    censoring_exit_fraction = (
        float(censoring_exits) / float(total_exits)
        if total_exits > 0
        else 0.0
    )
    return (
        risk_control,
        risk_treatment,
        failing_arm,
        terminal_time,
        censoring_exit_fraction,
    )


@njit(cache=True)
def _make_trial(
    treatment_uniforms,
    event_uniforms,
    censor_uniforms,
    event_horizon,
    hazard_early,
    hazard_late,
    change_time,
    censor_control,
    censor_treatment,
):
    subject_count = treatment_uniforms.size
    treatment = np.empty(subject_count, dtype=np.int8)
    event_times = np.empty(subject_count)
    censor_times = np.empty(subject_count)

    for subject in range(subject_count):
        arm = 1 if treatment_uniforms[subject] < 0.5 else 0
        treatment[subject] = arm
        exponential_draw = -math.log(
            max(event_uniforms[subject], 1e-300)
        )
        if arm == 1:
            event_times[subject] = _piecewise_exponential_time(
                exponential_draw,
                hazard_early,
                hazard_late,
                change_time,
            )
            censor_rate = censor_treatment
        else:
            event_times[subject] = exponential_draw
            censor_rate = censor_control
        if censor_rate > 0.0:
            censor_times[subject] = (
                -math.log(max(censor_uniforms[subject], 1e-300))
                / censor_rate
            )
        else:
            censor_times[subject] = np.inf

    return _risk_set_sequence(
        treatment,
        event_times,
        censor_times,
        event_horizon,
    )


# ---------------------------------------------------------------------------
# Candidate tests
# ---------------------------------------------------------------------------


@njit(cache=True)
def _classical_score(
    risk_control,
    risk_treatment,
    failing_arm,
    candidate,
    normal_cutoff,
):
    score = 0.0
    information = 0.0
    for event_index in range(failing_arm.size):
        probability = _risk_probability(
            risk_control[event_index],
            risk_treatment[event_index],
            candidate,
        )
        score += failing_arm[event_index] - probability
        information += probability * (1.0 - probability)
    if information <= 0.0:
        return np.inf
    return abs(score) / math.sqrt(information) - normal_cutoff


@njit(cache=True)
def _ge_wealths(
    risk_control,
    risk_treatment,
    failing_arm,
    candidate,
    delta,
    solvency_constant,
):
    alpha = delta / 2.0
    target = 1.0 / alpha
    upper_wealth = 1.0
    lower_wealth = 1.0
    event_horizon = failing_arm.size

    for event_index in range(event_horizon):
        probability = _risk_probability(
            risk_control[event_index],
            risk_treatment[event_index],
            candidate,
        )
        null_probability = _risk_probability(
            risk_control[event_index],
            risk_treatment[event_index],
            0.0,
        )
        current_information = max(
            null_probability * (1.0 - null_probability),
            1e-12,
        )
        remaining_information = (
            float(event_horizon - event_index) * current_information
        )

        if 0.0 < upper_wealth < target:
            upper_leverage = (
                _probit_leverage(alpha * upper_wealth)
                / math.sqrt(remaining_information)
            )
            upper_leverage = min(
                upper_leverage,
                solvency_constant / max(probability, 1e-15),
            )
        else:
            upper_leverage = 0.0

        if 0.0 < lower_wealth < target:
            lower_leverage = (
                _probit_leverage(alpha * lower_wealth)
                / math.sqrt(remaining_information)
            )
            lower_leverage = min(
                lower_leverage,
                solvency_constant / max(1.0 - probability, 1e-15),
            )
        else:
            lower_leverage = 0.0

        increment = failing_arm[event_index] - probability
        upper_wealth = min(
            max(upper_wealth * (1.0 + upper_leverage * increment), 0.0),
            target,
        )
        lower_wealth = min(
            max(lower_wealth * (1.0 - lower_leverage * increment), 0.0),
            target,
        )

    return upper_wealth, lower_wealth


@njit(cache=True)
def _ge_score(
    risk_control,
    risk_treatment,
    failing_arm,
    candidate,
    delta,
    solvency_constant,
):
    upper, lower = _ge_wealths(
        risk_control,
        risk_treatment,
        failing_arm,
        candidate,
        delta,
        solvency_constant,
    )
    target = 2.0 / delta
    return max(upper, lower) / target - 1.0


@njit(cache=True)
def _partial_log_likelihood(
    risk_control,
    risk_treatment,
    failing_arm,
    candidate,
):
    value = 0.0
    for event_index in range(failing_arm.size):
        probability = _risk_probability(
            risk_control[event_index],
            risk_treatment[event_index],
            candidate,
        )
        value += _bernoulli_log_probability(
            failing_arm[event_index],
            probability,
        )
    return value


@njit(cache=True)
def _smoothed_score_information(
    risk_control,
    risk_treatment,
    failing_arm,
    observed_events,
    initial_control,
    initial_treatment,
    candidate,
):
    exp_candidate = math.exp(min(max(candidate, -20.0), 20.0))
    treatment_augmented = float(initial_treatment + 1) * exp_candidate
    probability_first = treatment_augmented / (
        float(initial_control + 1) + treatment_augmented
    )
    probability_second = treatment_augmented / (
        float(initial_control) + treatment_augmented
    )
    score = 1.0 - probability_first - probability_second
    information = (
        probability_first * (1.0 - probability_first)
        + probability_second * (1.0 - probability_second)
    )

    for event_index in range(observed_events):
        probability = _risk_probability(
            risk_control[event_index],
            risk_treatment[event_index],
            candidate,
        )
        score += failing_arm[event_index] - probability
        information += probability * (1.0 - probability)
    return score, information


@njit(cache=True)
def _prequential_log_numerator(
    risk_control,
    risk_treatment,
    failing_arm,
):
    initial_control = int(risk_control[0])
    initial_treatment = int(risk_treatment[0])
    estimate = 0.0
    log_numerator = 0.0

    for event_index in range(failing_arm.size):
        for _ in range(12):
            score, information = _smoothed_score_information(
                risk_control,
                risk_treatment,
                failing_arm,
                event_index,
                initial_control,
                initial_treatment,
                estimate,
            )
            if information <= 1e-15:
                break
            step = score / information
            estimate = min(max(estimate + step, -12.0), 12.0)
            if abs(step) < 1e-10:
                break

        probability = _risk_probability(
            risk_control[event_index],
            risk_treatment[event_index],
            estimate,
        )
        log_numerator += _bernoulli_log_probability(
            failing_arm[event_index],
            probability,
        )
    return log_numerator


@njit(cache=True)
def _av_prequential_score(
    risk_control,
    risk_treatment,
    failing_arm,
    candidate,
    delta,
    prequential_log_numerator,
):
    denominator = _partial_log_likelihood(
        risk_control,
        risk_treatment,
        failing_arm,
        candidate,
    )
    return (
        prequential_log_numerator
        - denominator
        - math.log(1.0 / delta)
    )


@njit(cache=True)
def _av_grow_score(
    risk_control,
    risk_treatment,
    failing_arm,
    candidate,
    delta,
    grow_offset,
):
    """Published two-sided conditional-GROW product.

    The two point-alternative likelihood ratios are averaged separately at
    every event, as in the construction following equation (3.6) of ter
    Schure et al.  This is different from mixing the two complete directional
    e-processes once at time zero.
    """
    log_e_value = 0.0
    for event_index in range(failing_arm.size):
        probability_null = _risk_probability(
            risk_control[event_index],
            risk_treatment[event_index],
            candidate,
        )
        probability_lower = _risk_probability(
            risk_control[event_index],
            risk_treatment[event_index],
            candidate - grow_offset,
        )
        probability_upper = _risk_probability(
            risk_control[event_index],
            risk_treatment[event_index],
            candidate + grow_offset,
        )
        log_null = _bernoulli_log_probability(
            failing_arm[event_index],
            probability_null,
        )
        lower_ratio = (
            _bernoulli_log_probability(
                failing_arm[event_index],
                probability_lower,
            )
            - log_null
        )
        upper_ratio = (
            _bernoulli_log_probability(
                failing_arm[event_index],
                probability_upper,
            )
            - log_null
        )
        maximum = max(lower_ratio, upper_ratio)
        log_e_value += (
            maximum
            + math.log(
                0.5 * math.exp(lower_ratio - maximum)
                + 0.5 * math.exp(upper_ratio - maximum)
            )
        )
    return log_e_value - math.log(1.0 / delta)


@njit(cache=True)
def _av_path_mixture_score(
    risk_control,
    risk_treatment,
    failing_arm,
    candidate,
    delta,
    grow_offset,
):
    """Mixture of the two complete directional point e-processes."""
    log_lower_ratio = 0.0
    log_upper_ratio = 0.0
    for event_index in range(failing_arm.size):
        probability_null = _risk_probability(
            risk_control[event_index],
            risk_treatment[event_index],
            candidate,
        )
        probability_lower = _risk_probability(
            risk_control[event_index],
            risk_treatment[event_index],
            candidate - grow_offset,
        )
        probability_upper = _risk_probability(
            risk_control[event_index],
            risk_treatment[event_index],
            candidate + grow_offset,
        )
        log_null = _bernoulli_log_probability(
            failing_arm[event_index],
            probability_null,
        )
        log_lower_ratio += (
            _bernoulli_log_probability(
                failing_arm[event_index],
                probability_lower,
            )
            - log_null
        )
        log_upper_ratio += (
            _bernoulli_log_probability(
                failing_arm[event_index],
                probability_upper,
            )
            - log_null
        )
    maximum = max(log_lower_ratio, log_upper_ratio)
    log_e_value = (
        maximum
        + math.log(
            0.5 * math.exp(log_lower_ratio - maximum)
            + 0.5 * math.exp(log_upper_ratio - maximum)
        )
    )
    return log_e_value - math.log(1.0 / delta)


@njit(cache=True)
def _candidate_score(
    method_index,
    risk_control,
    risk_treatment,
    failing_arm,
    candidate,
    delta,
    normal_cutoff,
    solvency_constant,
    grow_offset,
    prequential_log_numerator,
):
    if method_index == 0:
        return _classical_score(
            risk_control,
            risk_treatment,
            failing_arm,
            candidate,
            normal_cutoff,
        )
    if method_index == 1:
        return _ge_score(
            risk_control,
            risk_treatment,
            failing_arm,
            candidate,
            delta,
            solvency_constant,
        )
    if method_index == 2:
        return _av_prequential_score(
            risk_control,
            risk_treatment,
            failing_arm,
            candidate,
            delta,
            prequential_log_numerator,
        )
    if method_index == 3:
        return _av_grow_score(
            risk_control,
            risk_treatment,
            failing_arm,
            candidate,
            delta,
            grow_offset,
        )
    return _av_path_mixture_score(
        risk_control,
        risk_treatment,
        failing_arm,
        candidate,
        delta,
        grow_offset,
    )


@njit(cache=True)
def _refine_boundary(
    method_index,
    risk_control,
    risk_treatment,
    failing_arm,
    rejected_candidate,
    accepted_candidate,
    delta,
    normal_cutoff,
    solvency_constant,
    grow_offset,
    prequential_log_numerator,
):
    rejected = rejected_candidate
    accepted = accepted_candidate
    for _ in range(36):
        midpoint = 0.5 * (rejected + accepted)
        score = _candidate_score(
            method_index,
            risk_control,
            risk_treatment,
            failing_arm,
            midpoint,
            delta,
            normal_cutoff,
            solvency_constant,
            grow_offset,
            prequential_log_numerator,
        )
        if score < 0.0:
            accepted = midpoint
        else:
            rejected = midpoint
    return accepted


@njit(cache=True)
def _confidence_hull(
    method_index,
    risk_control,
    risk_treatment,
    failing_arm,
    delta,
    normal_cutoff,
    solvency_constant,
    grow_offset,
    prequential_log_numerator,
    candidate_bound,
    grid_size,
):
    grid = np.linspace(-candidate_bound, candidate_bound, grid_size)
    accepted = np.zeros(grid_size, dtype=np.int8)
    first_accepted = -1
    last_accepted = -1
    component_count = 0
    previous_accepted = False

    for grid_index in range(grid_size):
        score = _candidate_score(
            method_index,
            risk_control,
            risk_treatment,
            failing_arm,
            grid[grid_index],
            delta,
            normal_cutoff,
            solvency_constant,
            grow_offset,
            prequential_log_numerator,
        )
        is_accepted = score < 0.0
        if is_accepted:
            accepted[grid_index] = 1
            if first_accepted < 0:
                first_accepted = grid_index
            last_accepted = grid_index
            if not previous_accepted:
                component_count += 1
        previous_accepted = is_accepted

    if first_accepted < 0:
        return np.nan, np.nan, 0, 0

    boundary_hit = 0
    if first_accepted == 0:
        lower = -candidate_bound
        boundary_hit = 1
    else:
        lower = _refine_boundary(
            method_index,
            risk_control,
            risk_treatment,
            failing_arm,
            grid[first_accepted - 1],
            grid[first_accepted],
            delta,
            normal_cutoff,
            solvency_constant,
            grow_offset,
            prequential_log_numerator,
        )

    if last_accepted == grid_size - 1:
        upper = candidate_bound
        boundary_hit = 1
    else:
        # _refine_boundary expects its first point rejected.
        upper = _refine_boundary(
            method_index,
            risk_control,
            risk_treatment,
            failing_arm,
            grid[last_accepted + 1],
            grid[last_accepted],
            delta,
            normal_cutoff,
            solvency_constant,
            grow_offset,
            prequential_log_numerator,
        )
    return lower, upper, component_count, boundary_hit


# ---------------------------------------------------------------------------
# Parallel experiment
# ---------------------------------------------------------------------------


@njit(parallel=True, cache=True)
def _run_scenario_numba(
    treatment_uniforms,
    event_uniforms,
    censor_uniforms,
    event_horizon,
    hazard_early,
    hazard_late,
    change_time,
    censor_control,
    censor_treatment,
    delta,
    normal_cutoff,
    solvency_constant,
    grow_offset,
    candidate_bound,
    grid_size,
):
    repetitions = treatment_uniforms.shape[0]
    method_count = len(METHOD_KEYS)
    lower = np.full((repetitions, method_count), np.nan)
    upper = np.full((repetitions, method_count), np.nan)
    components = np.zeros((repetitions, method_count), dtype=np.int16)
    boundary_hits = np.zeros((repetitions, method_count), dtype=np.int8)
    terminal_times = np.full(repetitions, np.nan)
    censor_fractions = np.full(repetitions, np.nan)
    null_information = np.full(repetitions, np.nan)
    failures = np.zeros(repetitions, dtype=np.int8)

    for repetition in prange(repetitions):
        (
            risk_control,
            risk_treatment,
            failing_arm,
            terminal_time,
            censor_fraction,
        ) = _make_trial(
            treatment_uniforms[repetition],
            event_uniforms[repetition],
            censor_uniforms[repetition],
            event_horizon,
            hazard_early,
            hazard_late,
            change_time,
            censor_control,
            censor_treatment,
        )
        if failing_arm.size != event_horizon:
            failures[repetition] = 1
            continue

        terminal_times[repetition] = terminal_time
        censor_fractions[repetition] = censor_fraction
        information = 0.0
        for event_index in range(event_horizon):
            probability = _risk_probability(
                risk_control[event_index],
                risk_treatment[event_index],
                0.0,
            )
            information += probability * (1.0 - probability)
        null_information[repetition] = information

        prequential_log_numerator = _prequential_log_numerator(
            risk_control,
            risk_treatment,
            failing_arm,
        )
        for method_index in range(method_count):
            (
                lower[repetition, method_index],
                upper[repetition, method_index],
                components[repetition, method_index],
                boundary_hits[repetition, method_index],
            ) = _confidence_hull(
                method_index,
                risk_control,
                risk_treatment,
                failing_arm,
                delta,
                normal_cutoff,
                solvency_constant,
                grow_offset,
                prequential_log_numerator,
                candidate_bound,
                grid_size,
            )

    return (
        lower,
        upper,
        components,
        boundary_hits,
        terminal_times,
        censor_fractions,
        null_information,
        failures,
    )


def _scenario_seed(base_seed, hazard_index, censor_index):
    return int(base_seed + 10_000 * hazard_index + 1_000_000 * censor_index)


def _run_one_scenario(
    rng,
    repetitions,
    subject_count,
    event_horizon,
    hazard,
    censor,
    delta,
    normal_cutoff,
    solvency_constant,
    grow_offset,
    candidate_bound,
    grid_size,
):
    shape = (repetitions, subject_count)
    treatment_uniforms = rng.random(shape)
    event_uniforms = rng.random(shape)
    censor_uniforms = rng.random(shape)
    return _run_scenario_numba(
        treatment_uniforms,
        event_uniforms,
        censor_uniforms,
        event_horizon,
        hazard.treatment_hazard_early,
        hazard.treatment_hazard_late,
        hazard.change_time,
        censor.control_rate,
        censor.treatment_rate,
        delta,
        normal_cutoff,
        solvency_constant,
        grow_offset,
        candidate_bound,
        grid_size,
    )


def _method_summary(
    lower,
    upper,
    components,
    boundary_hits,
    true_log_hr,
):
    finite = np.isfinite(lower) & np.isfinite(upper)
    width = upper - lower
    excludes_zero = finite & ((upper < 0.0) | (lower > 0.0))
    repetitions = lower.size
    rejection_probability = float(np.mean(excludes_zero))
    summary = {
        "repetitions": int(repetitions),
        "finite_interval_rate": float(np.mean(finite)),
        "rejection_probability_at_zero": rejection_probability,
        "rejection_mcse": float(
            math.sqrt(
                rejection_probability
                * (1.0 - rejection_probability)
                / repetitions
            )
        ),
        "mean_log_hr_width": float(np.nanmean(width)),
        "median_log_hr_width": float(np.nanmedian(width)),
        "q10_log_hr_width": float(np.nanquantile(width, 0.10)),
        "q90_log_hr_width": float(np.nanquantile(width, 0.90)),
        "grid_fragmentation_rate": float(np.mean(components > 1)),
        "candidate_boundary_rate": float(np.mean(boundary_hits > 0)),
    }
    if true_log_hr is not None:
        covered = finite & (lower <= true_log_hr) & (true_log_hr <= upper)
        coverage = float(np.mean(covered))
        summary["coverage"] = coverage
        summary["coverage_mcse"] = float(
            math.sqrt(coverage * (1.0 - coverage) / repetitions)
        )
    else:
        summary["coverage"] = None
        summary["coverage_mcse"] = None
    return summary


def _plot_results(result, output_path):
    summaries = result["summaries"]
    fig, axes = plt.subplots(
        2,
        len(CENSOR_SCENARIOS),
        figsize=(9.0, 6.2),
        sharex="col",
    )
    x = np.arange(len(HAZARD_SCENARIOS), dtype=float)
    offsets = np.linspace(-0.18, 0.18, len(METHOD_KEYS))

    for censor_index, censor in enumerate(CENSOR_SCENARIOS):
        power_axis = axes[0, censor_index]
        width_axis = axes[1, censor_index]
        for method_index, method in enumerate(METHOD_KEYS):
            power = []
            power_error = []
            width = []
            width_low = []
            width_high = []
            for hazard in HAZARD_SCENARIOS:
                method_result = summaries[f"{hazard.key}__{censor.key}"][
                    "methods"
                ][method]
                power.append(
                    method_result["rejection_probability_at_zero"]
                )
                power_error.append(1.96 * method_result["rejection_mcse"])
                width.append(method_result["median_log_hr_width"])
                width_low.append(method_result["q10_log_hr_width"])
                width_high.append(method_result["q90_log_hr_width"])

            positions = x + offsets[method_index]
            power_axis.errorbar(
                positions,
                power,
                yerr=power_error,
                color=METHOD_COLORS[method],
                marker=METHOD_MARKERS[method],
                markersize=5,
                linewidth=1.4,
                capsize=2,
                label=METHOD_LABELS[method],
            )
            width = np.asarray(width)
            width_low = np.asarray(width_low)
            width_high = np.asarray(width_high)
            width_axis.errorbar(
                positions,
                width,
                yerr=np.vstack((width - width_low, width_high - width)),
                color=METHOD_COLORS[method],
                marker=METHOD_MARKERS[method],
                markersize=5,
                linewidth=1.4,
                capsize=2,
            )

        power_axis.axhline(
            result["config"]["delta"],
            color="#888888",
            linestyle=":",
            linewidth=1.0,
        )
        power_axis.set_ylim(-0.02, 1.02)
        power_axis.set_title(censor.label, fontsize=10.5)
        power_axis.grid(alpha=0.2)
        width_axis.grid(alpha=0.2)
        power_axis.tick_params(axis="both", labelsize=8.5)
        width_axis.tick_params(axis="y", labelsize=8.5)
        width_axis.set_xticks(x)
        width_axis.set_xticklabels(
            [hazard.label for hazard in HAZARD_SCENARIOS],
            rotation=24,
            ha="right",
            fontsize=8.0,
        )

    axes[0, 0].set_ylabel(
        "Probability CI excludes 0",
        fontsize=9.0,
    )
    axes[1, 0].set_ylabel(
        "Log-HR CI width",
        fontsize=9.0,
    )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    # Matplotlib fills multirow figure legends columnwise.  This order displays
    # classical, GE, and prequential on row one, then the two AV mixtures.
    legend_order = (0, 3, 1, 4, 2)
    handles = [handles[index] for index in legend_order]
    labels = [labels[index] for index in legend_order]
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        fontsize=8.5,
        frameon=False,
        bbox_to_anchor=(0.5, 1.015),
    )
    fig.suptitle(
        (
            f"Fixed-event 95% intervals after "
            f"{result['config']['event_horizon']} failures"
        ),
        y=0.925,
        fontsize=11.5,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.88))
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def run_experiment(
    repetitions=10_000,
    event_horizon=200,
    subject_multiplier=3,
    delta=0.05,
    seed=20260812,
    solvency_constant=0.5,
    grow_hazard_ratio=0.70,
    candidate_bound=5.0,
    grid_size=81,
    output_dir=None,
    copy_to_paper=True,
):
    if repetitions <= 0:
        raise ValueError("repetitions must be positive")
    if event_horizon <= 0:
        raise ValueError("event_horizon must be positive")
    if subject_multiplier < 2:
        raise ValueError("subject_multiplier must be at least two")
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must lie in (0,1)")
    if grid_size < 9 or grid_size % 2 == 0:
        raise ValueError("grid_size must be an odd integer of at least nine")

    script_dir = Path(__file__).resolve().parent
    output_dir = (
        Path(output_dir)
        if output_dir is not None
        else script_dir / "plots" / "survival_fixed_event"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    subject_count = int(subject_multiplier * event_horizon)
    normal_cutoff = float(norm.ppf(1.0 - delta / 2.0))
    grow_offset = float(abs(math.log(grow_hazard_ratio)))

    # Warm up every compiled path before starting the timed experiment.
    warm_rng = np.random.default_rng(seed - 1)
    _run_one_scenario(
        warm_rng,
        2,
        30,
        min(event_horizon, 10),
        HAZARD_SCENARIOS[0],
        CENSOR_SCENARIOS[0],
        delta,
        normal_cutoff,
        solvency_constant,
        grow_offset,
        candidate_bound,
        min(grid_size, 9),
    )

    started = time.perf_counter()
    summaries = {}
    raw_arrays = {}
    for censor_index, censor in enumerate(CENSOR_SCENARIOS):
        for hazard_index, hazard in enumerate(HAZARD_SCENARIOS):
            scenario_key = f"{hazard.key}__{censor.key}"
            scenario_rng = np.random.default_rng(
                _scenario_seed(seed, hazard_index, censor_index)
            )
            (
                lower,
                upper,
                components,
                boundary_hits,
                terminal_times,
                censor_fractions,
                null_information,
                failures,
            ) = _run_one_scenario(
                scenario_rng,
                repetitions,
                subject_count,
                event_horizon,
                hazard,
                censor,
                delta,
                normal_cutoff,
                solvency_constant,
                grow_offset,
                candidate_bound,
                grid_size,
            )
            if np.any(failures):
                raise RuntimeError(
                    f"{scenario_key}: {int(np.sum(failures))} trials "
                    "did not reach the event horizon"
                )

            method_results = {}
            for method_index, method in enumerate(METHOD_KEYS):
                method_results[method] = _method_summary(
                    lower[:, method_index],
                    upper[:, method_index],
                    components[:, method_index],
                    boundary_hits[:, method_index],
                    hazard.true_log_hr,
                )
            summaries[scenario_key] = {
                "hazard": asdict(hazard),
                "censoring": asdict(censor),
                "mean_terminal_time": float(np.mean(terminal_times)),
                "mean_censoring_exit_fraction": float(
                    np.mean(censor_fractions)
                ),
                "mean_null_information": float(np.mean(null_information)),
                "methods": method_results,
            }
            raw_arrays[f"{scenario_key}__lower"] = lower
            raw_arrays[f"{scenario_key}__upper"] = upper
            raw_arrays[f"{scenario_key}__components"] = components
            raw_arrays[f"{scenario_key}__boundary_hits"] = boundary_hits
            raw_arrays[f"{scenario_key}__terminal_times"] = terminal_times
            raw_arrays[f"{scenario_key}__censor_fractions"] = (
                censor_fractions
            )
            raw_arrays[f"{scenario_key}__null_information"] = (
                null_information
            )

    elapsed_seconds = time.perf_counter() - started
    result = {
        "config": {
            "repetitions_per_scenario": int(repetitions),
            "event_horizon": int(event_horizon),
            "subject_count": int(subject_count),
            "subject_multiplier": int(subject_multiplier),
            "delta": float(delta),
            "seed": int(seed),
            "solvency_constant": float(solvency_constant),
            "grow_hazard_ratio": float(grow_hazard_ratio),
            "candidate_bound": float(candidate_bound),
            "grid_size": int(grid_size),
            "method_labels": METHOD_LABELS,
            "fixed_terminal_analysis_only": True,
        },
        "elapsed_seconds_excluding_compilation": float(elapsed_seconds),
        "summaries": summaries,
    }

    summary_path = output_dir / "summary.json"
    raw_path = output_dir / "replicates.npz"
    figure_path = output_dir / "fixed_event_survival_comparison.png"
    summary_path.write_text(json.dumps(result, indent=2) + "\n")
    np.savez_compressed(raw_path, **raw_arrays)
    _plot_results(result, figure_path)

    if copy_to_paper:
        paper_figure = (
            script_dir.parent
            / "paper"
            / "plots"
            / "fixed_event_survival_comparison.png"
        )
        shutil.copy2(figure_path, paper_figure)
        result["paper_figure"] = str(paper_figure)

    return result


def _print_compact_summary(result):
    print(
        "elapsed_seconds_excluding_compilation="
        f"{result['elapsed_seconds_excluding_compilation']:.2f}"
    )
    for censor in CENSOR_SCENARIOS:
        print(f"\n{censor.label}")
        for hazard in HAZARD_SCENARIOS:
            scenario = result["summaries"][f"{hazard.key}__{censor.key}"]
            values = []
            for method in METHOD_KEYS:
                method_result = scenario["methods"][method]
                values.append(
                    f"{method}: power={method_result['rejection_probability_at_zero']:.3f}, "
                    f"median_width={method_result['median_log_hr_width']:.3f}"
                )
            print(f"  {hazard.label}: " + " | ".join(values))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repetitions", type=int, default=10_000)
    parser.add_argument("--events", type=int, default=200)
    parser.add_argument("--subject-multiplier", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--grid-size", type=int, default=81)
    parser.add_argument("--candidate-bound", type=float, default=5.0)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--no-copy", action="store_true")
    args = parser.parse_args()

    result = run_experiment(
        repetitions=args.repetitions,
        event_horizon=args.events,
        subject_multiplier=args.subject_multiplier,
        seed=args.seed,
        grid_size=args.grid_size,
        candidate_bound=args.candidate_bound,
        output_dir=args.output_dir,
        copy_to_paper=not args.no_copy,
    )
    _print_compact_summary(result)


if __name__ == "__main__":
    main()
