#!/usr/bin/env python3
"""Run and plot the sampling-without-replacement experiments."""

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
from numba import njit
from scipy.optimize import brentq
from scipy.stats import beta as beta_distribution
from scipy.stats import hypergeom, norm
from matplotlib.lines import Line2D

from betting import probit_target_leverage


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = ROOT / "plots" / "wor"
DEFAULT_PAPER_PLOT_DIR = ROOT.parent / "paper" / "plots"

POPULATIONS = (
    ("Uniform(0,1)", "uniform", (0.0, 1.0)),
    ("Beta(0.5,0.5)", "beta", (0.5, 0.5)),
    ("Bernoulli(0.1)", "bernoulli", (0.1,)),
    ("Beta(2,2)", "beta", (2.0, 2.0)),
    ("Beta(1,5)", "beta", (1.0, 5.0)),
    ("Bernoulli(0.5)", "bernoulli", (0.5,)),
    ("Beta(50,50)", "beta", (50.0, 50.0)),
    ("Beta(20,80)", "beta", (20.0, 80.0)),
    ("Uniform(0.45,0.55)", "uniform", (0.45, 0.55)),
)
FRACTIONS = (0.1, 0.3, 0.5, 0.7, 0.8, 0.9)

METHOD_ORDER = (
    "Bridge-efficient betting",
    "WSR running intersection",
    "Exact hypergeometric",
    "Bardenet--Maillard EBS",
    "Shekhar--Ramdas AS-CI",
    "Shekhar--Ramdas rate CI",
)
BETTING_METHODS = frozenset(
    {"Bridge-efficient betting", "WSR running intersection"}
)
CALIBRATION_ORDER = (
    "Deterministic Markov",
    "Uniformly randomized Markov",
    "As published",
)
LOWER_ROW_OMITTED_METHODS = frozenset(
    {
        "Bardenet--Maillard EBS",
        "Shekhar--Ramdas AS-CI",
        "Shekhar--Ramdas rate CI",
    }
)
METHOD_STYLES = {
    "Bridge-efficient betting": {
        "color": "#2ca02c",
        "marker": "h",
        "linestyle": "-",
        "linewidth": 2.2,
        "label": "GE-betting",
    },
    "WSR running intersection": {
        "color": "#9467bd",
        "marker": "s",
        "linestyle": "--",
        "linewidth": 1.55,
        "label": "WSR running intersection",
    },
    "Exact hypergeometric": {
        "color": "#222222",
        "marker": "D",
        "linestyle": ":",
        "linewidth": 1.55,
        "label": "Exact hypergeometric",
    },
    "Bardenet--Maillard EBS": {
        "color": "#1976b9",
        "marker": "^",
        "linestyle": "-.",
        "linewidth": 1.45,
        "label": "Bardenet--Maillard EBS",
    },
    "Shekhar--Ramdas AS-CI": {
        "color": "darkorange",
        "marker": "P",
        "linestyle": "--",
        "linewidth": 1.45,
        "label": "Shekhar--Ramdas AS-CI",
    },
    "Shekhar--Ramdas rate CI": {
        "color": "#c44e52",
        "marker": "X",
        "linestyle": ":",
        "linewidth": 1.55,
        "label": "Shekhar--Ramdas rate CI",
    },
}

CS_FRACTIONS = (0.020, 0.125, 0.500, 0.800, 0.950)
CONFIDENCE_SEQUENCE_RESULTS = {
    "Bernoulli(0.5)": {
        "Stitched bridge": (1.546, 1.534, 1.586, 1.615, 1.598),
        "Hedged-WoR": (1.303, 1.598, 2.080, 2.711, 3.092),
    },
    "Beta(1,5)": {
        "Stitched bridge": (2.793, 1.853, 1.645, 1.658, 1.752),
        "Hedged-WoR": (2.777, 1.805, 1.709, 2.103, 3.009),
    },
}

GREY = "#707070"
BENCHMARK_COLOR = "black"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run or redraw the fixed-horizon sampling-without-replacement "
            "comparison."
        )
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--run-experiments",
        action="store_true",
        help="Rerun the simulations before drawing the figure.",
    )
    mode.add_argument(
        "--plot-only",
        action="store_true",
        help="Redraw from the saved summary without rerunning simulations.",
    )
    parser.add_argument("--population-size", type=int, default=4_000)
    parser.add_argument("--num-simulations", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20_260_813)
    parser.add_argument("--delta", type=float, default=0.01)
    parser.add_argument("--solvency-c", type=float, default=1.0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for figures, configuration, and source data.",
    )
    parser.add_argument(
        "--paper-plot-dir",
        type=Path,
        default=DEFAULT_PAPER_PLOT_DIR,
        help="Paper plot directory receiving Figure 4.",
    )
    parser.add_argument(
        "--no-paper-copy",
        action="store_true",
        help="Do not copy Figure 4 into the paper repository.",
    )
    return parser.parse_args()


@njit(cache=True)
def predictable_variances(values: np.ndarray) -> np.ndarray:
    """Variance sequence from the shared estimator used in the manuscript."""
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
def bridge_arm_wealth(
    values: np.ndarray,
    prefix: np.ndarray,
    variances: np.ndarray,
    candidate: float,
    sample_size: int,
    population_size: int,
    delta: float,
    solvency_c: float,
    upper_tail: bool,
) -> float:
    target = 2.0 / delta
    wealth = 1.0
    for index in range(sample_size):
        if not (0.0 < wealth < target):
            continue
        draw = index + 1
        remaining_mean = (
            population_size * candidate - prefix[index]
        ) / (population_size - index)
        relative_wealth = (delta / 2.0) * wealth
        scale = math.sqrt(
            (population_size - sample_size)
            / (
                (population_size - draw)
                * (sample_size - index)
                * variances[index]
            )
        )
        raw_fraction = probit_target_leverage(relative_wealth) * scale
        if upper_tail:
            cap = (
                math.inf
                if remaining_mean <= 0.0
                else solvency_c / remaining_mean
            )
            fraction = min(raw_fraction, cap)
            increment = values[index] - remaining_mean
        else:
            cap = (
                math.inf
                if remaining_mean >= 1.0
                else solvency_c / (1.0 - remaining_mean)
            )
            fraction = min(raw_fraction, cap)
            increment = remaining_mean - values[index]
        wealth = min(
            target,
            max(0.0, wealth * (1.0 + fraction * increment)),
        )
    return wealth


@njit(cache=True)
def wsr_arm_wealth(
    values: np.ndarray,
    prefix: np.ndarray,
    variances: np.ndarray,
    candidate: float,
    sample_size: int,
    population_size: int,
    delta: float,
    solvency_c: float,
    upper_tail: bool,
) -> float:
    """Fixed-horizon Hedged-WoR arm with running intersection."""
    target = 2.0 / delta
    wealth = 1.0
    log_target = math.log(2.0 / delta)
    for index in range(sample_size):
        if not (0.0 < wealth < target):
            continue
        remaining_mean = (
            population_size * candidate - prefix[index]
        ) / (population_size - index)
        raw_fraction = math.sqrt(
            2.0 * log_target / (sample_size * variances[index])
        )
        if upper_tail:
            cap = (
                math.inf
                if remaining_mean <= 0.0
                else solvency_c / remaining_mean
            )
            fraction = min(raw_fraction, cap)
            increment = values[index] - remaining_mean
        else:
            cap = (
                math.inf
                if remaining_mean >= 1.0
                else solvency_c / (1.0 - remaining_mean)
            )
            fraction = min(raw_fraction, cap)
            increment = remaining_mean - values[index]
        wealth = min(
            target,
            max(0.0, wealth * (1.0 + fraction * increment)),
        )
    return wealth


@njit(cache=True)
def betting_interval(
    values: np.ndarray,
    prefix: np.ndarray,
    variances: np.ndarray,
    sample_size: int,
    population_size: int,
    delta: float,
    solvency_c: float,
    bridge: bool,
    upper_tail_randomizer: float = 1.0,
    lower_tail_randomizer: float = 1.0,
    bisection_steps: int = 30,
) -> tuple[float, float]:
    feasible_lower = prefix[sample_size] / population_size
    feasible_upper = (
        prefix[sample_size] + population_size - sample_size
    ) / population_size
    if not (
        0.0 < upper_tail_randomizer <= 1.0
        and 0.0 < lower_tail_randomizer <= 1.0
    ):
        raise ValueError("randomizers must lie in (0,1]")
    upper_tail_target = 2.0 * upper_tail_randomizer / delta
    lower_tail_target = 2.0 * lower_tail_randomizer / delta

    if bridge:
        lower_outer_wealth = bridge_arm_wealth(
            values,
            prefix,
            variances,
            feasible_lower,
            sample_size,
            population_size,
            delta,
            solvency_c,
            True,
        )
    else:
        lower_outer_wealth = wsr_arm_wealth(
            values,
            prefix,
            variances,
            feasible_lower,
            sample_size,
            population_size,
            delta,
            solvency_c,
            True,
        )

    if lower_outer_wealth < upper_tail_target:
        lower_endpoint = feasible_lower
    else:
        rejected = feasible_lower
        accepted = feasible_upper
        for _ in range(bisection_steps):
            midpoint = 0.5 * (rejected + accepted)
            if bridge:
                wealth = bridge_arm_wealth(
                    values,
                    prefix,
                    variances,
                    midpoint,
                    sample_size,
                    population_size,
                    delta,
                    solvency_c,
                    True,
                )
            else:
                wealth = wsr_arm_wealth(
                    values,
                    prefix,
                    variances,
                    midpoint,
                    sample_size,
                    population_size,
                    delta,
                    solvency_c,
                    True,
                )
            if wealth >= upper_tail_target:
                rejected = midpoint
            else:
                accepted = midpoint
        lower_endpoint = accepted

    if bridge:
        upper_outer_wealth = bridge_arm_wealth(
            values,
            prefix,
            variances,
            feasible_upper,
            sample_size,
            population_size,
            delta,
            solvency_c,
            False,
        )
    else:
        upper_outer_wealth = wsr_arm_wealth(
            values,
            prefix,
            variances,
            feasible_upper,
            sample_size,
            population_size,
            delta,
            solvency_c,
            False,
        )

    if upper_outer_wealth < lower_tail_target:
        upper_endpoint = feasible_upper
    else:
        accepted = feasible_lower
        rejected = feasible_upper
        for _ in range(bisection_steps):
            midpoint = 0.5 * (accepted + rejected)
            if bridge:
                wealth = bridge_arm_wealth(
                    values,
                    prefix,
                    variances,
                    midpoint,
                    sample_size,
                    population_size,
                    delta,
                    solvency_c,
                    False,
                )
            else:
                wealth = wsr_arm_wealth(
                    values,
                    prefix,
                    variances,
                    midpoint,
                    sample_size,
                    population_size,
                    delta,
                    solvency_c,
                    False,
                )
            if wealth >= lower_tail_target:
                rejected = midpoint
            else:
                accepted = midpoint
        upper_endpoint = accepted

    return lower_endpoint, upper_endpoint


def intersected_width(
    lower: float,
    upper: float,
    feasible_lower: float,
    feasible_upper: float,
) -> float:
    return max(
        0.0,
        min(1.0, upper, feasible_upper)
        - max(0.0, lower, feasible_lower),
    )


def exact_hypergeometric_width(
    successes: int,
    sample_size: int,
    population_size: int,
    delta: float,
) -> float:
    """Width of the equal-tail hypergeometric inversion."""
    smallest_total = successes
    largest_total = population_size - sample_size + successes
    tail_probability = delta / 2.0

    if hypergeom.sf(
        successes - 1,
        population_size,
        smallest_total,
        sample_size,
    ) > tail_probability:
        lower_total = smallest_total
    else:
        rejected = smallest_total
        accepted = largest_total
        while accepted - rejected > 1:
            midpoint = (rejected + accepted) // 2
            if hypergeom.sf(
                successes - 1,
                population_size,
                midpoint,
                sample_size,
            ) > tail_probability:
                accepted = midpoint
            else:
                rejected = midpoint
        lower_total = accepted

    if hypergeom.cdf(
        successes,
        population_size,
        largest_total,
        sample_size,
    ) > tail_probability:
        upper_total = largest_total
    else:
        accepted = smallest_total
        rejected = largest_total
        while rejected - accepted > 1:
            midpoint = (accepted + rejected + 1) // 2
            if hypergeom.cdf(
                successes,
                population_size,
                midpoint,
                sample_size,
            ) > tail_probability:
                accepted = midpoint
            else:
                rejected = midpoint
        upper_total = accepted

    return max(0.0, (upper_total - lower_total) / population_size)


def bardenet_maillard_width(
    sample_mean: float,
    sample_variance: float,
    sample_size: int,
    population_size: int,
    delta: float,
    feasible_lower: float,
    feasible_upper: float,
) -> float:
    """The empirical Bernstein--Serfling interval of Bardenet--Maillard."""
    if sample_size <= population_size / 2:
        rho = 1.0 - (sample_size - 1.0) / population_size
    else:
        rho = (
            1.0 - sample_size / population_size
        ) * (1.0 + 1.0 / sample_size)
    log_term = math.log(10.0 / delta)
    kappa = 7.0 / 3.0 + 3.0 / math.sqrt(2.0)
    half_width = math.sqrt(sample_variance) * math.sqrt(
        2.0 * rho * log_term / sample_size
    ) + kappa * log_term / sample_size
    return intersected_width(
        sample_mean - half_width,
        sample_mean + half_width,
        feasible_lower,
        feasible_upper,
    )


def binary_kl(probability: float, reference: float) -> float:
    if probability <= 0.0:
        first = 0.0
    elif reference <= 0.0:
        return math.inf
    else:
        first = probability * math.log(probability / reference)
    if probability >= 1.0:
        second = 0.0
    elif reference >= 1.0:
        return math.inf
    else:
        second = (1.0 - probability) * math.log(
            (1.0 - probability) / (1.0 - reference)
        )
    return first + second


def shekhar_ramdas_as_width(
    sample_mean: float,
    sample_size: int,
    population_size: int,
    delta: float,
    feasible_lower: float,
    feasible_upper: float,
) -> float:
    """Width of the empirical AS-CI in Shekhar--Ramdas (2026)."""
    beta = sample_size / population_size
    target = (
        math.log(
            2.36
            * math.sqrt((1.0 - beta) * sample_size)
            / delta
        )
        / population_size
        + population_size ** (-1.1)
    )
    if sample_mean <= 0.0:
        half_width = 0.0
    else:
        maximum_rate = sample_mean * math.log(1.0 / beta)
        if target >= maximum_rate:
            tilted_probability = 1.0
        else:
            tilted_probability = brentq(
                lambda value: (
                    sample_mean * binary_kl(value, beta) - target
                ),
                beta,
                1.0 - 1e-14,
            )
        y_value = sample_mean * (tilted_probability - beta)
        half_width = y_value / beta
    return intersected_width(
        sample_mean - half_width,
        sample_mean + half_width,
        feasible_lower,
        feasible_upper,
    )


def binary_wor_rate(
    sample_probability: float,
    beta: float,
    population_probability: float,
) -> float:
    remaining_probability = (
        population_probability - beta * sample_probability
    ) / (1.0 - beta)
    return binary_kl(
        sample_probability, population_probability
    ) + (1.0 - beta) / beta * binary_kl(
        remaining_probability, population_probability
    )


def shekhar_ramdas_rate_width(
    sample_mean: float,
    sample_size: int,
    population_size: int,
    delta: float,
    feasible_lower: float,
    feasible_upper: float,
) -> float:
    """Width of the finite-alphabet empirical-rate interval."""
    beta = sample_size / population_size
    support_size = 2
    r_n = (support_size + 1.0) * math.log(population_size + 1.0)
    target = (
        math.log(2.0 / delta) + 2.0 * r_n
    ) / sample_size

    if sample_mean <= feasible_lower + 1e-14:
        lower = feasible_lower
    elif (
        binary_wor_rate(sample_mean, beta, feasible_lower)
        < target
    ):
        lower = feasible_lower
    else:
        lower = brentq(
            lambda value: (
                binary_wor_rate(sample_mean, beta, value) - target
            ),
            feasible_lower,
            sample_mean,
        )

    if sample_mean >= feasible_upper - 1e-14:
        upper = feasible_upper
    elif (
        binary_wor_rate(sample_mean, beta, feasible_upper)
        < target
    ):
        upper = feasible_upper
    else:
        upper = brentq(
            lambda value: (
                binary_wor_rate(sample_mean, beta, value) - target
            ),
            sample_mean,
            feasible_upper,
        )

    return max(0.0, upper - lower)


def row_seed(seed: int, population_index: int, replication: int) -> np.random.SeedSequence:
    return np.random.SeedSequence([seed, population_index, replication])


def make_population(
    population_size: int,
    family: str,
    parameters: tuple[float, ...],
) -> np.ndarray:
    """Construct a deterministic finite population from distribution quantiles."""
    if family == "bernoulli":
        number_of_ones = int(round(parameters[0] * population_size))
        population = np.zeros(population_size, dtype=float)
        population[:number_of_ones] = 1.0
        return population

    probabilities = (
        np.arange(population_size, dtype=float) + 0.5
    ) / population_size
    if family == "uniform":
        lower, upper = parameters
        return lower + (upper - lower) * probabilities
    if family == "beta":
        shape_a, shape_b = parameters
        return np.asarray(
            beta_distribution.ppf(probabilities, shape_a, shape_b),
            dtype=float,
        )
    raise ValueError(f"unknown population family: {family}")


def run_experiments(args: argparse.Namespace, output_dir: Path) -> list[dict[str, object]]:
    population_size = int(args.population_size)
    if population_size < 20:
        raise ValueError("population-size must be at least 20")
    if args.num_simulations <= 0:
        raise ValueError("num-simulations must be positive")
    if not 0.0 < args.delta < 1.0:
        raise ValueError("delta must lie in (0,1)")
    if not 0.0 < args.solvency_c <= 1.0:
        raise ValueError("solvency-c must lie in (0,1]")

    sample_sizes = tuple(
        int(round(fraction * population_size))
        for fraction in FRACTIONS
    )
    path_rows: list[dict[str, object]] = []

    # Compile the numerical kernels before timing the experiment.
    warm_values = np.array([0.0, 1.0, 0.0, 1.0], dtype=float)
    warm_prefix = np.concatenate(([0.0], np.cumsum(warm_values)))
    warm_variances = predictable_variances(warm_values)
    betting_interval(
        warm_values,
        warm_prefix,
        warm_variances,
        2,
        8,
        args.delta,
        args.solvency_c,
        True,
    )

    for population_index, (population_name, family, parameters) in enumerate(
        POPULATIONS
    ):
        population = make_population(population_size, family, parameters)
        population_mean = float(np.mean(population))
        population_variance = float(
            np.mean((population - population_mean) ** 2)
        )
        population_sd = math.sqrt(population_variance)
        print(f"Running {population_name} ({args.num_simulations} orders)")

        for replication in range(args.num_simulations):
            rng = np.random.default_rng(
                row_seed(args.seed, population_index, replication)
            )
            values = rng.permutation(population)
            prefix = np.concatenate(([0.0], np.cumsum(values)))
            squared_prefix = np.concatenate(
                ([0.0], np.cumsum(values * values))
            )
            variances = predictable_variances(values)
            randomizer_rng = np.random.default_rng(
                np.random.SeedSequence(
                    [args.seed, population_index, replication, 1]
                )
            )

            for sample_size in sample_sizes:
                sample_sum = prefix[sample_size]
                sample_mean = sample_sum / sample_size
                sample_variance = max(
                    0.0,
                    squared_prefix[sample_size] / sample_size
                    - sample_mean * sample_mean,
                )
                feasible_lower = sample_sum / population_size
                feasible_upper = (
                    sample_sum + population_size - sample_size
                ) / population_size
                standard_error = population_sd * math.sqrt(
                    (population_size - sample_size)
                    / (sample_size * (population_size - 1.0))
                )
                gaussian_width = (
                    2.0
                    * norm.ppf(1.0 - args.delta / 2.0)
                    * standard_error
                )
                u_plus, u_minus = randomizer_rng.random(2)
                positive_floor = np.finfo(float).tiny
                u_plus = max(float(u_plus), positive_floor)
                u_minus = max(float(u_minus), positive_floor)

                bridge_lower, bridge_upper = betting_interval(
                    values,
                    prefix,
                    variances,
                    sample_size,
                    population_size,
                    args.delta,
                    args.solvency_c,
                    True,
                )
                (
                    bridge_randomized_lower,
                    bridge_randomized_upper,
                ) = betting_interval(
                    values,
                    prefix,
                    variances,
                    sample_size,
                    population_size,
                    args.delta,
                    args.solvency_c,
                    True,
                    u_plus,
                    u_minus,
                )
                wsr_lower, wsr_upper = betting_interval(
                    values,
                    prefix,
                    variances,
                    sample_size,
                    population_size,
                    args.delta,
                    args.solvency_c,
                    False,
                )
                wsr_randomized_lower, wsr_randomized_upper = betting_interval(
                    values,
                    prefix,
                    variances,
                    sample_size,
                    population_size,
                    args.delta,
                    args.solvency_c,
                    False,
                    u_plus,
                    u_minus,
                )
                widths = {
                    ("Bridge-efficient betting", "Deterministic Markov"): max(
                        0.0, bridge_upper - bridge_lower
                    ),
                    ("Bridge-efficient betting", "Uniformly randomized Markov"): max(
                        0.0,
                        bridge_randomized_upper - bridge_randomized_lower,
                    ),
                    ("WSR running intersection", "Deterministic Markov"): max(
                        0.0, wsr_upper - wsr_lower
                    ),
                    ("WSR running intersection", "Uniformly randomized Markov"): max(
                        0.0, wsr_randomized_upper - wsr_randomized_lower
                    ),
                    ("Bardenet--Maillard EBS", "As published"): bardenet_maillard_width(
                        sample_mean,
                        sample_variance,
                        sample_size,
                        population_size,
                        args.delta,
                        feasible_lower,
                        feasible_upper,
                    ),
                    ("Shekhar--Ramdas AS-CI", "As published"): shekhar_ramdas_as_width(
                        sample_mean,
                        sample_size,
                        population_size,
                        args.delta,
                        feasible_lower,
                        feasible_upper,
                    ),
                }
                if family == "bernoulli":
                    widths[
                        ("Exact hypergeometric", "As published")
                    ] = exact_hypergeometric_width(
                        int(round(sample_sum)),
                        sample_size,
                        population_size,
                        args.delta,
                    )
                    widths[
                        ("Shekhar--Ramdas rate CI", "As published")
                    ] = shekhar_ramdas_rate_width(
                        sample_mean,
                        sample_size,
                        population_size,
                        args.delta,
                        feasible_lower,
                        feasible_upper,
                    )

                for (method, calibration), width in widths.items():
                    path_rows.append(
                        {
                            "distribution": population_name,
                            "population_mean": population_mean,
                            "population_variance": population_variance,
                            "replication": replication,
                            "n": sample_size,
                            "sampling_fraction": (
                                sample_size / population_size
                            ),
                            "method": method,
                            "width": width,
                            "calibration": calibration,
                            "normalized_width": width / gaussian_width,
                        }
                    )

    path_file = output_dir / "fixed_horizon_path_widths.csv"
    fieldnames = list(path_rows[0])
    with path_file.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(path_rows)

    grouped: dict[tuple[str, int, str, str], list[float]] = {}
    metadata: dict[tuple[str, int, str, str], dict[str, object]] = {}
    for row in path_rows:
        key = (
            str(row["distribution"]),
            int(row["n"]),
            str(row["method"]),
            str(row["calibration"]),
        )
        grouped.setdefault(key, []).append(float(row["normalized_width"]))
        metadata[key] = row

    summary_rows: list[dict[str, object]] = []
    for key, normalized_widths in grouped.items():
        row = metadata[key]
        summary_rows.append(
            {
                "distribution": row["distribution"],
                "population_mean": row["population_mean"],
                "population_variance": row["population_variance"],
                "n": row["n"],
                "sampling_fraction": row["sampling_fraction"],
                "method": row["method"],
                "calibration": row["calibration"],
                "normalized_mean_width": float(
                    np.mean(normalized_widths)
                ),
                "normalized_standard_error": float(
                    np.std(normalized_widths, ddof=1)
                    / math.sqrt(len(normalized_widths))
                ),
                "num_simulations": len(normalized_widths),
            }
        )
    summary_rows.sort(
        key=lambda row: (
            [name for name, _, _ in POPULATIONS].index(
                str(row["distribution"])
            ),
            METHOD_ORDER.index(str(row["method"])),
            CALIBRATION_ORDER.index(str(row["calibration"])),
            int(row["n"]),
        )
    )
    write_rows(output_dir / "fixed_horizon_widths.csv", summary_rows)

    configuration = {
        "population_size": population_size,
        "populations": [
            {"name": name, "family": family, "parameters": parameters}
            for name, family, parameters in POPULATIONS
        ],
        "sampling_fractions": list(FRACTIONS),
        "sample_sizes": list(sample_sizes),
        "num_simulations": int(args.num_simulations),
        "seed": int(args.seed),
        "delta": float(args.delta),
        "solvency_c": float(args.solvency_c),
        "randomization": {
            "methods": sorted(BETTING_METHODS),
            "calibration": "uniformly randomized Markov",
            "pairing": (
                "one independent uniform pair per population, reveal order, "
                "and sampling fraction; shared across betting methods"
            ),
        },
        "bisection_steps": 30,
        "normalization": (
            "2 * z_(1-delta/2) * sigma_N * "
            "sqrt((N-n)/(n*(N-1)))"
        ),
    }
    with (output_dir / "fixed_horizon_config.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(configuration, handle, indent=2)
        handle.write("\n")
    return summary_rows


def write_rows(path: Path, rows: Sequence[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty result table")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_summary(path: Path) -> list[dict[str, object]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        if not row.get("calibration"):
            row["calibration"] = (
                "Deterministic Markov"
                if row["method"] in BETTING_METHODS
                else "As published"
            )
        for key in (
            "population_mean",
            "population_variance",
            "sampling_fraction",
            "normalized_mean_width",
            "normalized_standard_error",
        ):
            row[key] = float(row[key])
        row["n"] = int(row["n"])
        row["num_simulations"] = int(row["num_simulations"])
    return rows


def add_benchmark(axis: plt.Axes) -> None:
    axis.axhline(
        1.0,
        color=BENCHMARK_COLOR,
        linestyle=":",
        linewidth=1.5,
        zorder=0,
    )
    axis.grid(
        axis="y",
        color="#D9D9D9",
        linewidth=0.55,
        alpha=0.75,
    )
    axis.set_axisbelow(True)


def fixed_horizon_plot(
    output_dir: Path,
    summary_rows: Sequence[dict[str, object]],
) -> Path:
    figure, axes = plt.subplots(3, 3, figsize=(13.5, 11.3))

    handles_by_method: dict[str, object] = {}
    for panel_index, (axis, (population_name, _, _)) in enumerate(
        zip(axes.ravel(), POPULATIONS)
    ):
        add_benchmark(axis)
        population_rows = [
            row
            for row in summary_rows
            if row["distribution"] == population_name
        ]
        panel_minimum = math.inf
        for method in METHOD_ORDER:
            if (
                panel_index >= 3
                and method in LOWER_ROW_OMITTED_METHODS
            ):
                continue
            calibrations = (
                (
                    "Deterministic Markov",
                    "Uniformly randomized Markov",
                )
                if method in BETTING_METHODS
                else ("As published",)
            )
            for calibration in calibrations:
                method_rows = sorted(
                    (
                        row
                        for row in population_rows
                        if (
                            row["method"] == method
                            and row["calibration"] == calibration
                        )
                    ),
                    key=lambda row: float(row["sampling_fraction"]),
                )
                if not method_rows:
                    continue
                style = METHOD_STYLES[method]
                x_values = [
                    float(row["sampling_fraction"])
                    for row in method_rows
                ]
                y_values = [
                    float(row["normalized_mean_width"])
                    for row in method_rows
                ]
                panel_minimum = min(panel_minimum, min(y_values))
                is_deterministic = calibration == "Deterministic Markov"
                is_randomized = (
                    calibration == "Uniformly randomized Markov"
                )
                if is_deterministic:
                    linestyle = "--"
                    linewidth = 1.55
                    marker_face = "white"
                    zorder = 2
                elif is_randomized:
                    linestyle = "-"
                    linewidth = 2.15
                    marker_face = style["color"]
                    zorder = 4
                else:
                    linestyle = style["linestyle"]
                    linewidth = style["linewidth"]
                    marker_face = style["color"]
                    zorder = 3
                handle = axis.plot(
                    x_values,
                    y_values,
                    color=style["color"],
                    marker=style["marker"],
                    linestyle=linestyle,
                    linewidth=linewidth,
                    markerfacecolor=marker_face,
                    markeredgecolor=style["color"],
                    markeredgewidth=0.9,
                    markersize=4.2,
                    zorder=zorder,
                    label=style["label"],
                )[0]
                if not is_deterministic:
                    handles_by_method[method] = handle

        axis.set_title(population_name)
        axis.set_xlabel(r"Sampling fraction $\rho=n/N$")
        axis.set_xticks(FRACTIONS)
        axis.set_xticklabels([".1", ".3", ".5", ".7", ".8", ".9"])
        axis.set_xlim(0.065, 0.935)
        axis.set_ylim(bottom=min(0.88, max(0.0, 0.96 * panel_minimum)))
        axis.set_ylabel("Normalized mean width")
        axis.grid(True, ls="--", alpha=0.3)

    method_handles = [
        handles_by_method[method]
        for method in METHOD_ORDER
        if method in handles_by_method
    ]
    method_labels = [
        METHOD_STYLES[method]["label"]
        for method in METHOD_ORDER
        if method in handles_by_method
    ]
    calibration_handles = [
        Line2D(
            [0], [0],
            color="0.25",
            linestyle="--",
            linewidth=1.55,
            marker="o",
            markerfacecolor="white",
            markeredgecolor="0.25",
        ),
        Line2D(
            [0], [0],
            color="0.25",
            linestyle="-",
            linewidth=2.15,
            marker="o",
            markerfacecolor="0.25",
            markeredgecolor="0.25",
        ),
    ]
    reference_handles = [
        Line2D(
            [0], [0], color=BENCHMARK_COLOR, linestyle=":", linewidth=1.7
        )
    ]
    figure.suptitle(
        "Confidence intervals under sampling without replacement",
        fontsize=15,
    )
    figure.legend(
        method_handles,
        method_labels,
        loc="lower center",
        bbox_to_anchor=(0.27, 0.012),
        ncol=3,
        fontsize=8.4,
        title="Construction (color and marker)",
        title_fontsize=8.6,
        frameon=False,
    )
    figure.legend(
        calibration_handles,
        ["Deterministic", "Uniformly randomized"],
        loc="lower center",
        bbox_to_anchor=(0.65, 0.028),
        ncol=2,
        fontsize=8.4,
        title="Calibration (line and marker fill)",
        title_fontsize=8.6,
        frameon=False,
    )
    figure.legend(
        reference_handles,
        ["Gaussian finite-population benchmark"],
        loc="lower center",
        bbox_to_anchor=(0.89, 0.028),
        ncol=1,
        fontsize=8.4,
        title="Asymptotic reference (dotted)",
        title_fontsize=8.6,
        frameon=False,
    )
    figure.tight_layout(rect=(0.0, 0.12, 1.0, 0.955))
    output_path = output_dir / "fixed_horizon_widths.png"
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)
    return output_path


def write_confidence_sequence_csv(path: Path) -> None:
    rows: list[dict[str, object]] = []
    for population, methods in CONFIDENCE_SEQUENCE_RESULTS.items():
        for method, values in methods.items():
            for fraction, value in zip(CS_FRACTIONS, values):
                rows.append(
                    {
                        "population": population,
                        "sampling_fraction": fraction,
                        "method": method,
                        "normalized_mean_width": value,
                    }
                )
    write_rows(path, rows)


def confidence_sequence_plot(output_dir: Path) -> Path:
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(6.45, 3.25),
        sharex=True,
        sharey=True,
        gridspec_kw={"wspace": 0.12},
    )
    for axis, (population, values) in zip(
        axes, CONFIDENCE_SEQUENCE_RESULTS.items()
    ):
        add_benchmark(axis)
        axis.plot(
            CS_FRACTIONS,
            values["Stitched bridge"],
            color="#2ca02c",
            marker="o",
            label="Stitched bridge",
            zorder=3,
        )
        axis.plot(
            CS_FRACTIONS,
            values["Hedged-WoR"],
            color="#9467bd",
            marker="s",
            linestyle="--",
            markerfacecolor="white",
            label="Hedged-WoR",
            zorder=3,
        )
        axis.set_title(population)
        axis.set_xlabel(r"Sampling fraction \(t/N\)")
        axis.set_xticks(CS_FRACTIONS)
        axis.set_xticklabels([".02", ".125", ".50", ".80", ".95"])
        axis.set_xlim(0, 0.98)
        axis.set_ylim(0.9, 3.2)
    axes[0].set_ylabel("Normalized mean width")
    axes[1].text(
        0.955,
        1.03,
        "Pointwise Gaussian benchmark",
        color=GREY,
        fontsize=7.3,
        ha="right",
        va="bottom",
    )
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        frameon=False,
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
    )
    figure.subplots_adjust(top=0.82)
    output_path = output_dir / "confidence_sequence_widths.pdf"
    figure.savefig(output_path)
    plt.close(figure)
    return output_path


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "fixed_horizon_widths.csv"

    should_run = args.run_experiments or (
        not args.plot_only and not summary_path.exists()
    )
    if should_run:
        summary_rows = run_experiments(args, output_dir)
    else:
        if not summary_path.exists():
            raise FileNotFoundError(
                f"No saved WOR summary found at {summary_path}; "
                "run with --run-experiments."
            )
        summary_rows = load_summary(summary_path)

    fixed_path = fixed_horizon_plot(output_dir, summary_rows)
    confidence_sequence_plot(output_dir)
    write_confidence_sequence_csv(
        output_dir / "confidence_sequence_widths.csv"
    )

    if not args.no_paper_copy:
        paper_plot_dir = args.paper_plot_dir.resolve()
        paper_plot_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(
            fixed_path,
            paper_plot_dir / "wor_fixed_horizon_widths.png",
        )


if __name__ == "__main__":
    main()
