#!/usr/bin/env python3
"""Planned-window confidence sequences for bounded means."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from numba import njit
from scipy.optimize import brentq
from scipy.stats import norm

from betting import probit_target_leverage
from confidence_sequences import agrapa_log_e_path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "plots" / "horizon_free_wr_cs"
DEFAULT_PAPER_PLOT_DIR = ROOT.parent / "paper" / "plots"

POPULATIONS = (
    ("Uniform(0,1)", "uniform", (0.0, 1.0), 0.5, 1.0 / 12.0),
    ("Beta(0.5,0.5)", "beta", (0.5, 0.5), 0.5, 0.125),
    ("Bernoulli(0.1)", "bernoulli", (0.1,), 0.1, 0.09),
    ("Beta(2,2)", "beta", (2.0, 2.0), 0.5, 1.0 / 20.0),
    ("Beta(1,5)", "beta", (1.0, 5.0), 1.0 / 6.0, 5.0 / 252.0),
    ("Bernoulli(0.5)", "bernoulli", (0.5,), 0.5, 0.25),
    ("Beta(50,50)", "beta", (50.0, 50.0), 0.5, 1.0 / 404.0),
    (
        "Beta(20,80)",
        "beta",
        (20.0, 80.0),
        0.2,
        1600.0 / 1_010_000.0,
    ),
    (
        "Uniform(0.45,0.55)",
        "uniform",
        (0.45, 0.55),
        0.5,
        0.1**2 / 12.0,
    ),
)
RECORD_TIMES = (10, 20, 40, 75, 125, 200, 350, 500, 1_000, 2_000, 5_000, 10_000)
METHODS = ("Stitched GE", "aGRAPA", "PRECiSE-A-CO96", "Hedged-CS")
METHOD_STYLES = {
    "Stitched GE": {"color": "#2ca02c", "marker": "h"},
    "aGRAPA": {"color": "#ff7f0e", "marker": "^"},
    "PRECiSE-A-CO96": {"color": "#1f77b4", "marker": "D"},
    "Hedged-CS": {"color": "#9467bd", "marker": "s"},
}
METHOD_LABELS = {
    "Stitched GE": "Stitched GE-betting",
    "aGRAPA": "aGRAPA",
    "PRECiSE-A-CO96": "PRECiSE-A-CO96",
    "Hedged-CS": "Hedged-CS",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--run-experiments", action="store_true")
    mode.add_argument("--plot-only", action="store_true")
    parser.add_argument("--max-time", type=int, default=10_000)
    parser.add_argument("--num-simulations", type=int, default=40)
    parser.add_argument("--seed", type=int, default=260_813)
    parser.add_argument("--delta", type=float, default=0.05)
    parser.add_argument("--solvency-c", type=float, default=1.0)
    parser.add_argument("--competitor-solvency-c", type=float, default=0.5)
    parser.add_argument("--checkpoint-ratio", type=float, default=1.3)
    parser.add_argument("--first-checkpoint", type=int, default=5)
    parser.add_argument("--bisection-steps", type=int, default=24)
    parser.add_argument("--agrapa-grid-size", type=int, default=129)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--paper-plot-dir", type=Path, default=DEFAULT_PAPER_PLOT_DIR
    )
    parser.add_argument("--no-paper-copy", action="store_true")
    return parser.parse_args()


def checkpoint_grid(
    max_time: int, first_checkpoint: int, ratio: float
) -> np.ndarray:
    """Return integer checkpoints approximately geometric in time."""
    if not 1 <= first_checkpoint <= max_time:
        raise ValueError("first-checkpoint must lie in {1,...,max-time}")
    if ratio <= 1.0:
        raise ValueError("checkpoint-ratio must exceed one")
    checkpoints = [first_checkpoint]
    while checkpoints[-1] < max_time:
        next_checkpoint = max(
            checkpoints[-1] + 1,
            int(math.ceil(ratio * checkpoints[-1])),
        )
        checkpoints.append(min(max_time, next_checkpoint))
    return np.asarray(checkpoints, dtype=np.int64)


def sample_population(
    rng: np.random.Generator,
    family: str,
    parameters: tuple[float, ...],
    size: int,
) -> np.ndarray:
    if family == "beta":
        return rng.beta(parameters[0], parameters[1], size)
    if family == "bernoulli":
        return rng.binomial(1, parameters[0], size).astype(float)
    if family == "uniform":
        return rng.uniform(parameters[0], parameters[1], size)
    raise ValueError(f"unknown family: {family}")


@njit(cache=True)
def predictable_variances(values: np.ndarray) -> np.ndarray:
    result = np.empty(values.size, dtype=np.float64)
    running_sum = 0.0
    squared_residual_sum = 0.0
    for index in range(values.size):
        mean_hat = (0.5 + running_sum) / (index + 1.0)
        result[index] = (0.25 + squared_residual_sum) / (index + 1.0)
        residual = values[index] - mean_hat
        running_sum += values[index]
        squared_residual_sum += residual * residual
    return result


@njit(cache=True)
def stitched_arm_wealth(
    values: np.ndarray,
    variances: np.ndarray,
    checkpoints: np.ndarray,
    weights: np.ndarray,
    candidate: float,
    sample_size: int,
    delta: float,
    solvency_c: float,
    upper_tail: bool,
) -> float:
    target = 2.0 / delta
    aggregate = 0.0
    for account_index in range(checkpoints.size):
        horizon = checkpoints[account_index]
        wealth = weights[account_index]
        terminal_time = min(sample_size, horizon)
        for index in range(terminal_time):
            if not (0.0 < wealth < target):
                continue
            scale = 1.0 / math.sqrt(
                (horizon - index) * variances[index]
            )
            raw_fraction = (
                probit_target_leverage((delta / 2.0) * wealth) * scale
            )
            if upper_tail:
                cap = math.inf if candidate <= 0.0 else solvency_c / candidate
                fraction = min(raw_fraction, cap)
                increment = values[index] - candidate
            else:
                cap = (
                    math.inf
                    if candidate >= 1.0
                    else solvency_c / (1.0 - candidate)
                )
                fraction = min(raw_fraction, cap)
                increment = candidate - values[index]
            wealth = min(
                target,
                max(0.0, wealth * (1.0 + fraction * increment)),
            )
        aggregate += wealth
    return aggregate


@njit(cache=True)
def hedged_arm_wealth(
    values: np.ndarray,
    variances: np.ndarray,
    candidate: float,
    sample_size: int,
    delta: float,
    solvency_c: float,
    upper_tail: bool,
) -> float:
    target = 2.0 / delta
    log_target = math.log(target)
    wealth = 1.0
    for index in range(sample_size):
        if not (0.0 < wealth < target):
            continue
        draw = index + 1
        raw_fraction = math.sqrt(
            2.0
            * log_target
            / (variances[index] * draw * math.log(draw + 1.0))
        )
        if upper_tail:
            cap = math.inf if candidate <= 0.0 else solvency_c / candidate
            fraction = min(raw_fraction, cap)
            increment = values[index] - candidate
        else:
            cap = (
                math.inf
                if candidate >= 1.0
                else solvency_c / (1.0 - candidate)
            )
            fraction = min(raw_fraction, cap)
            increment = candidate - values[index]
        wealth = min(
            target,
            max(0.0, wealth * (1.0 + fraction * increment)),
        )
    return wealth


@njit(cache=True)
def arm_wealth(
    values: np.ndarray,
    variances: np.ndarray,
    checkpoints: np.ndarray,
    weights: np.ndarray,
    candidate: float,
    sample_size: int,
    delta: float,
    solvency_c: float,
    upper_tail: bool,
    stitched: bool,
) -> float:
    if stitched:
        return stitched_arm_wealth(
            values,
            variances,
            checkpoints,
            weights,
            candidate,
            sample_size,
            delta,
            solvency_c,
            upper_tail,
        )
    return hedged_arm_wealth(
        values,
        variances,
        candidate,
        sample_size,
        delta,
        solvency_c,
        upper_tail,
    )


@njit(cache=True)
def current_betting_interval(
    values: np.ndarray,
    variances: np.ndarray,
    checkpoints: np.ndarray,
    weights: np.ndarray,
    sample_size: int,
    delta: float,
    solvency_c: float,
    stitched: bool,
    bisection_steps: int,
) -> tuple[float, float]:
    target = 2.0 / delta
    if arm_wealth(
        values,
        variances,
        checkpoints,
        weights,
        0.0,
        sample_size,
        delta,
        solvency_c,
        True,
        stitched,
    ) < target:
        lower = 0.0
    else:
        rejected = 0.0
        accepted = 1.0
        for _ in range(bisection_steps):
            midpoint = 0.5 * (rejected + accepted)
            if arm_wealth(
                values,
                variances,
                checkpoints,
                weights,
                midpoint,
                sample_size,
                delta,
                solvency_c,
                True,
                stitched,
            ) >= target:
                rejected = midpoint
            else:
                accepted = midpoint
        lower = accepted

    if arm_wealth(
        values,
        variances,
        checkpoints,
        weights,
        1.0,
        sample_size,
        delta,
        solvency_c,
        False,
        stitched,
    ) < target:
        upper = 1.0
    else:
        accepted = 0.0
        rejected = 1.0
        for _ in range(bisection_steps):
            midpoint = 0.5 * (accepted + rejected)
            if arm_wealth(
                values,
                variances,
                checkpoints,
                weights,
                midpoint,
                sample_size,
                delta,
                solvency_c,
                False,
                stitched,
            ) >= target:
                rejected = midpoint
            else:
                accepted = midpoint
        upper = accepted
    return lower, upper


@njit(cache=True)
def betting_intervals(
    values: np.ndarray,
    variances: np.ndarray,
    checkpoints: np.ndarray,
    weights: np.ndarray,
    times: np.ndarray,
    delta: float,
    solvency_c: float,
    stitched: bool,
    bisection_steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    lowers = np.empty(times.size)
    uppers = np.empty(times.size)
    for time_index in range(times.size):
        lower, upper = current_betting_interval(
            values,
            variances,
            checkpoints,
            weights,
            times[time_index],
            delta,
            solvency_c,
            stitched,
            bisection_steps,
        )
        lowers[time_index] = lower
        uppers[time_index] = upper
    return lowers, uppers


def agrapa_current_endpoints(
    values: np.ndarray,
    times: np.ndarray,
    delta: float,
    c: float,
    topology_grid_size: int,
) -> dict[str, np.ndarray]:
    """Invert current aGRAPA wealth, returning its numerical convex hull."""
    cumulative = np.r_[0.0, np.cumsum(values)]
    centers = cumulative[times] / times
    means = np.unique(
        np.r_[np.linspace(0.0, 1.0, topology_grid_size), centers]
    )
    surface = np.empty((means.size, times.size))
    for mean_index, candidate in enumerate(means):
        surface[mean_index] = agrapa_log_e_path(
            values, float(candidate), c=c
        )[times]

    threshold = math.log(1.0 / delta)
    lower = np.empty(times.size)
    upper = np.empty(times.size)
    components = np.empty(times.size, dtype=np.int64)
    uncertain = np.zeros(times.size, dtype=bool)
    for time_index, sample_size in enumerate(times):
        accepted = surface[:, time_index] < threshold
        components[time_index] = int(
            np.sum(accepted & ~np.r_[False, accepted[:-1]])
        )
        if not np.any(accepted):
            lower[time_index] = math.nan
            upper[time_index] = math.nan
            uncertain[time_index] = True
            continue

        accepted_indices = np.flatnonzero(accepted)
        first = int(accepted_indices[0])
        last = int(accepted_indices[-1])
        uncertain[time_index] = (
            components[time_index] > 1
            or np.any(np.diff(accepted_indices) > 1)
            or last == first
        )

        def centered_log_wealth(candidate: float) -> float:
            return float(
                agrapa_log_e_path(
                    values[:sample_size], candidate, c=c
                )[-1]
                - threshold
            )

        if first == 0:
            lower[time_index] = 0.0
        else:
            rejected = float(means[first - 1])
            accepted_mean = float(means[first])
            if (
                centered_log_wealth(rejected) >= 0.0
                and centered_log_wealth(accepted_mean) < 0.0
            ):
                lower[time_index] = brentq(
                    centered_log_wealth,
                    rejected,
                    accepted_mean,
                    xtol=1e-8,
                    rtol=1e-12,
                )
            else:
                lower[time_index] = accepted_mean
                uncertain[time_index] = True

        if last == means.size - 1:
            upper[time_index] = 1.0
        else:
            accepted_mean = float(means[last])
            rejected = float(means[last + 1])
            if (
                centered_log_wealth(accepted_mean) < 0.0
                and centered_log_wealth(rejected) >= 0.0
            ):
                upper[time_index] = brentq(
                    centered_log_wealth,
                    accepted_mean,
                    rejected,
                    xtol=1e-8,
                    rtol=1e-12,
                )
            else:
                upper[time_index] = accepted_mean
                uncertain[time_index] = True

    return {
        "lower": lower,
        "upper": upper,
        "component_count": components,
        "topology_uncertain": uncertain,
    }


@njit(cache=True)
def kl_divergence(p: float, q: float) -> float:
    if q <= 0.0:
        return 0.0 if p <= 0.0 else math.inf
    if q >= 1.0:
        return 0.0 if p >= 1.0 else math.inf
    result = 0.0
    if p > 0.0:
        result += p * math.log(p / q)
    if p < 1.0:
        result += (1.0 - p) * math.log((1.0 - p) / (1.0 - q))
    return result


@njit(cache=True)
def precise_fan_value(
    candidate: float,
    mean: float,
    variance: float,
    sample_size: int,
    lower_side: bool,
) -> float:
    if abs(candidate - mean) <= 1e-15:
        return 0.0
    if lower_side:
        if candidate <= 0.0:
            denominator = variance + mean * mean
            return (
                0.0
                if denominator <= 0.0
                else 0.5 * mean * mean / denominator * sample_size
            )
        difference = mean - candidate
        scale = candidate
    else:
        if candidate >= 1.0:
            difference = 1.0 - mean
            denominator = variance + difference * difference
            return (
                0.0
                if denominator <= 0.0
                else 0.5 * difference * difference / denominator * sample_size
            )
        difference = candidate - mean
        scale = 1.0 - candidate
    a_value = difference / scale
    b_value = (variance + difference * difference) / (scale * scale)
    denominator = a_value + b_value
    if denominator <= 0.0:
        return 0.0
    lam = a_value / denominator
    penalty = -math.log1p(-lam) - lam
    return (
        a_value * a_value / denominator - penalty * b_value
    ) * sample_size


@njit(cache=True)
def precise_test_value(
    candidate: float,
    mean: float,
    variance: float,
    sample_size: int,
    lower_side: bool,
) -> float:
    fan = precise_fan_value(
        candidate, mean, variance, sample_size, lower_side
    )
    bernoulli_kl = sample_size * kl_divergence(mean, candidate)
    return max(fan, bernoulli_kl)


@njit(cache=True)
def precise_a_intervals(
    values: np.ndarray,
    times: np.ndarray,
    delta: float,
    bisection_steps: int,
    true_mean: float,
) -> tuple[np.ndarray, np.ndarray, bool]:
    lowers = np.empty(times.size)
    uppers = np.empty(times.size)
    running_sum = 0.0
    running_square_sum = 0.0
    time_index = 0
    covered = True
    for index in range(values.size):
        value = values[index]
        running_sum += value
        running_square_sum += value * value
        sample_size = index + 1
        mean = running_sum / sample_size
        variance = max(running_square_sum / sample_size - mean * mean, 0.0)
        boundary = (
            0.5 * math.log(math.pi)
            + math.lgamma(sample_size + 1.0)
            - math.lgamma(sample_size + 0.5)
            + math.log(1.0 / delta)
        )

        if precise_test_value(0.0, mean, variance, sample_size, True) <= boundary:
            current_lower = 0.0
        else:
            rejected = 0.0
            accepted = mean
            for _ in range(bisection_steps):
                midpoint = 0.5 * (rejected + accepted)
                if precise_test_value(
                    midpoint, mean, variance, sample_size, True
                ) > boundary:
                    rejected = midpoint
                else:
                    accepted = midpoint
            current_lower = accepted

        if precise_test_value(1.0, mean, variance, sample_size, False) <= boundary:
            current_upper = 1.0
        else:
            accepted = mean
            rejected = 1.0
            for _ in range(bisection_steps):
                midpoint = 0.5 * (accepted + rejected)
                if precise_test_value(
                    midpoint, mean, variance, sample_size, False
                ) > boundary:
                    rejected = midpoint
                else:
                    accepted = midpoint
            current_upper = accepted

        if not (current_lower <= true_mean <= current_upper):
            covered = False
        if time_index < times.size and sample_size == times[time_index]:
            lowers[time_index] = current_lower
            uppers[time_index] = current_upper
            time_index += 1
    return lowers, uppers, covered


@njit(cache=True)
def betting_simultaneous_coverage(
    values: np.ndarray,
    variances: np.ndarray,
    checkpoints: np.ndarray,
    weights: np.ndarray,
    true_mean: float,
    delta: float,
    solvency_c: float,
    stitched: bool,
) -> bool:
    target = 2.0 / delta
    if stitched:
        upper_accounts = weights.copy()
        lower_accounts = weights.copy()
        for index in range(values.size):
            draw = index + 1
            increment = values[index] - true_mean
            for account_index in range(checkpoints.size):
                horizon = checkpoints[account_index]
                if draw > horizon:
                    continue
                scale = 1.0 / math.sqrt(
                    (horizon - index) * variances[index]
                )
                upper_wealth = upper_accounts[account_index]
                if 0.0 < upper_wealth < target:
                    raw = probit_target_leverage(
                        (delta / 2.0) * upper_wealth
                    ) * scale
                    cap = (
                        math.inf
                        if true_mean <= 0.0
                        else solvency_c / true_mean
                    )
                    upper_accounts[account_index] = min(
                        target,
                        max(0.0, upper_wealth * (1.0 + min(raw, cap) * increment)),
                    )
                lower_wealth = lower_accounts[account_index]
                if 0.0 < lower_wealth < target:
                    raw = probit_target_leverage(
                        (delta / 2.0) * lower_wealth
                    ) * scale
                    cap = (
                        math.inf
                        if true_mean >= 1.0
                        else solvency_c / (1.0 - true_mean)
                    )
                    lower_accounts[account_index] = min(
                        target,
                        max(0.0, lower_wealth * (1.0 - min(raw, cap) * increment)),
                    )
            if np.sum(upper_accounts) >= target or np.sum(lower_accounts) >= target:
                return False
        return True

    upper_wealth = 1.0
    lower_wealth = 1.0
    log_target = math.log(target)
    for index in range(values.size):
        draw = index + 1
        increment = values[index] - true_mean
        raw = math.sqrt(
            2.0
            * log_target
            / (variances[index] * draw * math.log(draw + 1.0))
        )
        if 0.0 < upper_wealth < target:
            cap = math.inf if true_mean <= 0.0 else solvency_c / true_mean
            upper_wealth = min(
                target,
                max(0.0, upper_wealth * (1.0 + min(raw, cap) * increment)),
            )
        if 0.0 < lower_wealth < target:
            cap = (
                math.inf
                if true_mean >= 1.0
                else solvency_c / (1.0 - true_mean)
            )
            lower_wealth = min(
                target,
                max(0.0, lower_wealth * (1.0 - min(raw, cap) * increment)),
            )
        if upper_wealth >= target or lower_wealth >= target:
            return False
    return True


def write_rows(path: Path, rows: Sequence[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty table")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_experiments(
    args: argparse.Namespace, output_dir: Path
) -> list[dict[str, object]]:
    if args.max_time <= 0 or args.num_simulations <= 0:
        raise ValueError("max-time and num-simulations must be positive")
    if not 0.0 < args.delta < 1.0:
        raise ValueError("delta must lie in (0,1)")
    if not 0.0 < args.solvency_c <= 1.0:
        raise ValueError("solvency-c must lie in (0,1]")
    if not 0.0 < args.competitor_solvency_c <= 1.0:
        raise ValueError("competitor-solvency-c must lie in (0,1]")

    record_times = np.asarray(
        [time for time in RECORD_TIMES if time <= args.max_time],
        dtype=np.int64,
    )
    if not record_times.size or record_times[-1] != args.max_time:
        record_times = np.unique(np.r_[record_times, args.max_time]).astype(
            np.int64
        )
    checkpoints = checkpoint_grid(
        args.max_time, args.first_checkpoint, args.checkpoint_ratio
    )
    weights = np.full(checkpoints.size, 1.0 / checkpoints.size)

    # Compile the numerical kernels before timing the experiment.
    warm_values = np.array([0.0, 1.0, 0.25, 0.75], dtype=float)
    warm_variances = predictable_variances(warm_values)
    warm_checkpoints = np.array([2, 4], dtype=np.int64)
    warm_weights = np.array([0.5, 0.5])
    betting_intervals(
        warm_values,
        warm_variances,
        warm_checkpoints,
        warm_weights,
        np.array([2, 4], dtype=np.int64),
        args.delta,
        args.solvency_c,
        True,
        3,
    )
    precise_a_intervals(
        warm_values,
        np.array([2, 4], dtype=np.int64),
        args.delta,
        3,
        0.5,
    )
    agrapa_current_endpoints(
        warm_values,
        np.array([2, 4], dtype=np.int64),
        args.delta,
        args.competitor_solvency_c,
        5,
    )

    path_rows: list[dict[str, object]] = []
    coverage_rows: list[dict[str, object]] = []
    topology_rows: list[dict[str, object]] = []
    for population_index, (
        name,
        family,
        parameters,
        true_mean,
        true_variance,
    ) in enumerate(POPULATIONS):
        print(f"Running {name} ({args.num_simulations} paths)", flush=True)
        for replication in range(args.num_simulations):
            rng = np.random.default_rng(
                np.random.SeedSequence(
                    [args.seed, population_index, replication]
                )
            )
            values = sample_population(
                rng, family, parameters, args.max_time
            )
            variances = predictable_variances(values)

            method_intervals: dict[str, tuple[np.ndarray, np.ndarray]] = {}
            stitched_lower, stitched_upper = betting_intervals(
                values,
                variances,
                checkpoints,
                weights,
                record_times,
                args.delta,
                args.solvency_c,
                True,
                args.bisection_steps,
            )
            method_intervals["Stitched GE"] = (
                stitched_lower,
                stitched_upper,
            )
            hedged_lower, hedged_upper = betting_intervals(
                values,
                variances,
                checkpoints,
                weights,
                record_times,
                args.delta,
                args.competitor_solvency_c,
                False,
                args.bisection_steps,
            )
            method_intervals["Hedged-CS"] = (hedged_lower, hedged_upper)
            precise_lower, precise_upper, precise_covered = precise_a_intervals(
                values,
                record_times,
                args.delta,
                args.bisection_steps,
                true_mean,
            )
            method_intervals["PRECiSE-A-CO96"] = (
                precise_lower,
                precise_upper,
            )
            agrapa = agrapa_current_endpoints(
                values,
                record_times,
                args.delta,
                args.competitor_solvency_c,
                args.agrapa_grid_size,
            )
            method_intervals["aGRAPA"] = (
                np.asarray(agrapa["lower"]),
                np.asarray(agrapa["upper"]),
            )
            for time_index, sample_size in enumerate(record_times):
                topology_rows.append(
                    {
                        "distribution": name,
                        "replication": replication,
                        "n": int(sample_size),
                        "component_count": int(
                            agrapa["component_count"][time_index]
                        ),
                        "topology_uncertain": int(
                            agrapa["topology_uncertain"][time_index]
                        ),
                    }
                )

            coverage = {
                "Stitched GE": betting_simultaneous_coverage(
                    values,
                    variances,
                    checkpoints,
                    weights,
                    true_mean,
                    args.delta,
                    args.solvency_c,
                    True,
                ),
                "Hedged-CS": betting_simultaneous_coverage(
                    values,
                    variances,
                    checkpoints,
                    weights,
                    true_mean,
                    args.delta,
                    args.competitor_solvency_c,
                    False,
                ),
                "aGRAPA": bool(
                    np.max(
                        agrapa_log_e_path(
                            values,
                            true_mean,
                            c=args.competitor_solvency_c,
                        )
                    )
                    < math.log(1.0 / args.delta)
                ),
                "PRECiSE-A-CO96": bool(precise_covered),
            }
            for method in METHODS:
                coverage_rows.append(
                    {
                        "distribution": name,
                        "replication": replication,
                        "method": method,
                        "simultaneously_covered": int(coverage[method]),
                    }
                )
                lower, upper = method_intervals[method]
                for time_index, sample_size in enumerate(record_times):
                    gaussian_width = (
                        2.0
                        * norm.ppf(1.0 - args.delta / 2.0)
                        * math.sqrt(true_variance / sample_size)
                    )
                    width = max(
                        0.0, float(upper[time_index] - lower[time_index])
                    )
                    path_rows.append(
                        {
                            "distribution": name,
                            "true_mean": true_mean,
                            "true_variance": true_variance,
                            "replication": replication,
                            "n": int(sample_size),
                            "sampling_fraction": sample_size / args.max_time,
                            "method": method,
                            "lower": float(lower[time_index]),
                            "upper": float(upper[time_index]),
                            "width": width,
                            "normalized_width": width / gaussian_width,
                        }
                    )

    write_rows(output_dir / "pathwise_widths.csv", path_rows)
    write_rows(output_dir / "pathwise_coverage.csv", coverage_rows)
    write_rows(output_dir / "agrapa_topology.csv", topology_rows)

    grouped: dict[tuple[str, int, str], list[float]] = {}
    metadata: dict[tuple[str, int, str], dict[str, object]] = {}
    for row in path_rows:
        key = (str(row["distribution"]), int(row["n"]), str(row["method"]))
        grouped.setdefault(key, []).append(float(row["normalized_width"]))
        metadata[key] = row
    summary_rows: list[dict[str, object]] = []
    for key, widths in grouped.items():
        row = metadata[key]
        summary_rows.append(
            {
                "distribution": row["distribution"],
                "n": row["n"],
                "sampling_fraction": row["sampling_fraction"],
                "method": row["method"],
                "normalized_mean_width": float(np.mean(widths)),
                "normalized_standard_error": (
                    float(np.std(widths, ddof=1) / math.sqrt(len(widths)))
                    if len(widths) > 1
                    else 0.0
                ),
                "num_simulations": len(widths),
            }
        )
    distribution_order = [spec[0] for spec in POPULATIONS]
    summary_rows.sort(
        key=lambda row: (
            distribution_order.index(str(row["distribution"])),
            METHODS.index(str(row["method"])),
            int(row["n"]),
        )
    )
    write_rows(output_dir / "summary.csv", summary_rows)

    configuration = {
        "max_time": int(args.max_time),
        "record_times": record_times.tolist(),
        "num_simulations": int(args.num_simulations),
        "seed": int(args.seed),
        "delta": float(args.delta),
        "solvency_c": float(args.solvency_c),
        "competitor_solvency_c": float(args.competitor_solvency_c),
        "checkpoint_ratio": float(args.checkpoint_ratio),
        "first_checkpoint": int(args.first_checkpoint),
        "checkpoints": checkpoints.tolist(),
        "checkpoint_weights": "uniform",
        "bisection_steps": int(args.bisection_steps),
        "agrapa_grid_size": int(args.agrapa_grid_size),
        "reported_interval": "raw current-time confidence sequence for every method",
        "methods": list(METHODS),
        "normalization": "2*z_(1-delta/2)*sigma/sqrt(t)",
    }
    with (output_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(configuration, handle, indent=2)
        handle.write("\n")
    return summary_rows


def load_summary(path: Path) -> list[dict[str, object]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["n"] = int(row["n"])
        row["sampling_fraction"] = float(row["sampling_fraction"])
        row["normalized_mean_width"] = float(row["normalized_mean_width"])
        row["normalized_standard_error"] = float(
            row["normalized_standard_error"]
        )
        row["num_simulations"] = int(row["num_simulations"])
    return rows


def make_plot(
    output_dir: Path, summary_rows: Sequence[dict[str, object]]
) -> Path:
    figure, axes = plt.subplots(3, 3, figsize=(13.5, 11.3), sharex=True)
    handles: dict[str, object] = {}
    for axis, (name, _, _, _, _) in zip(axes.ravel(), POPULATIONS):
        axis.axhline(1.0, color="black", linestyle=":", linewidth=1.5)
        rows = [row for row in summary_rows if row["distribution"] == name]
        for method in METHODS:
            method_rows = sorted(
                (row for row in rows if row["method"] == method),
                key=lambda row: float(row["sampling_fraction"]),
            )
            style = METHOD_STYLES[method]
            handle = axis.plot(
                [float(row["sampling_fraction"]) for row in method_rows],
                [float(row["normalized_mean_width"]) for row in method_rows],
                color=style["color"],
                marker=style["marker"],
                markerfacecolor=style["color"],
                markeredgecolor=style["color"],
                markeredgewidth=0.8,
                markersize=3.8,
                linewidth=1.7,
            )[0]
            handles.setdefault(method, handle)
        axis.set_title(name, fontsize=10.5)
        axis.set_xscale("log")
        axis.set_xlabel(r"fraction of planned window $t/n$")
        axis.set_ylabel("normalized mean width")
        axis.set_xlim(8.0 / max(row["n"] for row in rows), 1.08)
        axis.set_ylim(bottom=0.9)
        axis.grid(True, which="major", linestyle="--", alpha=0.30)

    figure.suptitle(
        "Planned-window confidence sequences under sampling with replacement",
        fontsize=13.5,
    )
    figure.legend(
        [handles[method] for method in METHODS],
        [METHOD_LABELS[method] for method in METHODS],
        loc="lower center",
        bbox_to_anchor=(0.40, 0.005),
        ncol=4,
        fontsize=8.0,
        title="Construction (color and marker)",
        title_fontsize=8.3,
        frameon=False,
    )
    figure.legend(
        [Line2D([0], [0], color="black", linestyle=":", linewidth=1.6)],
        ["Pointwise Gaussian width"],
        loc="lower center",
        bbox_to_anchor=(0.86, 0.005),
        fontsize=8.0,
        title="Scale reference (dotted)",
        title_fontsize=8.3,
        frameon=False,
    )
    figure.tight_layout(rect=(0.0, 0.09, 1.0, 0.96))
    output = output_dir / "horizon_free_wr_cs_widths.png"
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)
    return output


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.csv"
    should_run = args.run_experiments or (
        not args.plot_only and not summary_path.exists()
    )
    if should_run:
        summary_rows = run_experiments(args, output_dir)
    else:
        if not summary_path.exists():
            raise FileNotFoundError(
                f"No saved results at {summary_path}; use --run-experiments."
            )
        summary_rows = load_summary(summary_path)
    plot = make_plot(output_dir, summary_rows)
    if not args.no_paper_copy:
        paper_plot_dir = args.paper_plot_dir.resolve()
        paper_plot_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(plot, paper_plot_dir / plot.name)


if __name__ == "__main__":
    main()
