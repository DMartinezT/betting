#!/usr/bin/env python3
"""Run the horizon-free finite-population confidence-sequence experiment."""

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
from scipy.stats import beta as beta_distribution
from scipy.stats import norm

from betting import probit_target_leverage


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = ROOT / "plots" / "horizon_free_cs"
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
RECORD_TIMES = (10, 20, 40, 75, 125, 200, 350, 500, 650, 800, 900, 950)
METHODS = ("Stitched bridge-efficient", "Hedged-WoR")
METHOD_STYLES = {
    "Stitched bridge-efficient": {
        "color": "#2ca02c",
        "marker": "h",
    },
    "Hedged-WoR": {
        "color": "#9467bd",
        "marker": "s",
    },
}
METHOD_LABELS = {
    "Stitched bridge-efficient": "Stitched GE-betting",
    "Hedged-WoR": "Hedged-WoR",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--run-experiments", action="store_true")
    mode.add_argument("--plot-only", action="store_true")
    parser.add_argument("--population-size", type=int, default=1_000)
    parser.add_argument("--num-simulations", type=int, default=40)
    parser.add_argument("--seed", type=int, default=260_810)
    parser.add_argument("--delta", type=float, default=0.05)
    parser.add_argument("--solvency-c", type=float, default=1.0)
    parser.add_argument("--plugin-solvency-c", type=float, default=0.5)
    parser.add_argument("--checkpoint-ratio", type=float, default=1.3)
    parser.add_argument("--first-checkpoint", type=int, default=5)
    parser.add_argument("--bisection-steps", type=int, default=24)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--paper-plot-dir", type=Path, default=DEFAULT_PAPER_PLOT_DIR
    )
    parser.add_argument("--no-paper-copy", action="store_true")
    return parser.parse_args()


def checkpoint_grid(
    population_size: int,
    first_checkpoint: int,
    ratio: float,
) -> np.ndarray:
    """Return checkpoints geometric in bridge variance time t/(N-t)."""
    if not 1 <= first_checkpoint < population_size:
        raise ValueError("first-checkpoint must lie in {1,...,N-1}")
    if ratio <= 1.0:
        raise ValueError("checkpoint-ratio must exceed one")
    checkpoints = [first_checkpoint]
    while checkpoints[-1] < population_size - 1:
        current = checkpoints[-1]
        target_clock = ratio * current / (population_size - current)
        next_checkpoint = math.ceil(
            population_size * target_clock / (1.0 + target_clock)
        )
        next_checkpoint = max(current + 1, next_checkpoint)
        checkpoints.append(min(population_size - 1, next_checkpoint))
    return np.asarray(checkpoints, dtype=np.int64)


def make_population(
    population_size: int,
    family: str,
    parameters: tuple[float, ...],
) -> np.ndarray:
    if family == "bernoulli":
        number_of_ones = int(round(parameters[0] * population_size))
        population = np.zeros(population_size, dtype=float)
        population[:number_of_ones] = 1.0
        return population
    probabilities = (
        np.arange(population_size, dtype=float) + 0.5
    ) / population_size
    if family == "beta":
        return np.asarray(
            beta_distribution.ppf(probabilities, *parameters), dtype=float
        )
    if family == "uniform":
        lower, upper = parameters
        return lower + (upper - lower) * probabilities
    raise ValueError(f"unknown population family: {family}")


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
    prefix: np.ndarray,
    variances: np.ndarray,
    checkpoints: np.ndarray,
    weights: np.ndarray,
    candidate: float,
    sample_size: int,
    population_size: int,
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
            draw = index + 1
            remaining_mean = (
                population_size * candidate - prefix[index]
            ) / (population_size - index)
            relative_wealth = (delta / 2.0) * wealth
            scale = math.sqrt(
                (population_size - horizon)
                / (
                    (population_size - draw)
                    * (horizon - index)
                    * variances[index]
                )
            )
            raw_fraction = (
                probit_target_leverage(relative_wealth) * scale
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
        aggregate += wealth
    return aggregate


@njit(cache=True)
def plugin_arm_wealth(
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
    log_target = math.log(2.0 / delta)
    for index in range(sample_size):
        if not (0.0 < wealth < target):
            continue
        draw = index + 1
        remaining_mean = (
            population_size * candidate - prefix[index]
        ) / (population_size - index)
        raw_fraction = math.sqrt(
            2.0
            * log_target
            / (variances[index] * draw * math.log(draw + 1.0))
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
def arm_wealth(
    values: np.ndarray,
    prefix: np.ndarray,
    variances: np.ndarray,
    checkpoints: np.ndarray,
    weights: np.ndarray,
    candidate: float,
    sample_size: int,
    population_size: int,
    delta: float,
    solvency_c: float,
    upper_tail: bool,
    stitched: bool,
) -> float:
    if stitched:
        return stitched_arm_wealth(
            values,
            prefix,
            variances,
            checkpoints,
            weights,
            candidate,
            sample_size,
            population_size,
            delta,
            solvency_c,
            upper_tail,
        )
    return plugin_arm_wealth(
        values,
        prefix,
        variances,
        candidate,
        sample_size,
        population_size,
        delta,
        solvency_c,
        upper_tail,
    )


@njit(cache=True)
def confidence_interval(
    values: np.ndarray,
    prefix: np.ndarray,
    variances: np.ndarray,
    checkpoints: np.ndarray,
    weights: np.ndarray,
    sample_size: int,
    population_size: int,
    delta: float,
    solvency_c: float,
    stitched: bool,
    bisection_steps: int,
) -> tuple[float, float]:
    feasible_lower = prefix[sample_size] / population_size
    feasible_upper = (
        prefix[sample_size] + population_size - sample_size
    ) / population_size
    target = 2.0 / delta

    lower_outer = arm_wealth(
        values,
        prefix,
        variances,
        checkpoints,
        weights,
        feasible_lower,
        sample_size,
        population_size,
        delta,
        solvency_c,
        True,
        stitched,
    )
    if lower_outer < target:
        lower = feasible_lower
    else:
        rejected = feasible_lower
        accepted = feasible_upper
        for _ in range(bisection_steps):
            midpoint = 0.5 * (rejected + accepted)
            wealth = arm_wealth(
                values,
                prefix,
                variances,
                checkpoints,
                weights,
                midpoint,
                sample_size,
                population_size,
                delta,
                solvency_c,
                True,
                stitched,
            )
            if wealth >= target:
                rejected = midpoint
            else:
                accepted = midpoint
        lower = accepted

    upper_outer = arm_wealth(
        values,
        prefix,
        variances,
        checkpoints,
        weights,
        feasible_upper,
        sample_size,
        population_size,
        delta,
        solvency_c,
        False,
        stitched,
    )
    if upper_outer < target:
        upper = feasible_upper
    else:
        accepted = feasible_lower
        rejected = feasible_upper
        for _ in range(bisection_steps):
            midpoint = 0.5 * (accepted + rejected)
            wealth = arm_wealth(
                values,
                prefix,
                variances,
                checkpoints,
                weights,
                midpoint,
                sample_size,
                population_size,
                delta,
                solvency_c,
                False,
                stitched,
            )
            if wealth >= target:
                rejected = midpoint
            else:
                accepted = midpoint
        upper = accepted
    return lower, upper


@njit(cache=True)
def simultaneous_coverage(
    values: np.ndarray,
    prefix: np.ndarray,
    variances: np.ndarray,
    checkpoints: np.ndarray,
    weights: np.ndarray,
    true_mean: float,
    population_size: int,
    delta: float,
    solvency_c: float,
    stitched: bool,
) -> bool:
    target = 2.0 / delta
    if stitched:
        upper_accounts = weights.copy()
        lower_accounts = weights.copy()
        for index in range(population_size - 1):
            draw = index + 1
            remaining_mean = (
                population_size * true_mean - prefix[index]
            ) / (population_size - index)
            increment = values[index] - remaining_mean
            for account_index in range(checkpoints.size):
                horizon = checkpoints[account_index]
                if draw > horizon:
                    continue
                scale = math.sqrt(
                    (population_size - horizon)
                    / (
                        (population_size - draw)
                        * (horizon - index)
                        * variances[index]
                    )
                )
                upper_wealth = upper_accounts[account_index]
                if 0.0 < upper_wealth < target:
                    raw_fraction = probit_target_leverage(
                        (delta / 2.0) * upper_wealth
                    ) * scale
                    cap = (
                        math.inf
                        if remaining_mean <= 0.0
                        else solvency_c / remaining_mean
                    )
                    upper_accounts[account_index] = min(
                        target,
                        max(
                            0.0,
                            upper_wealth
                            * (1.0 + min(raw_fraction, cap) * increment),
                        ),
                    )
                lower_wealth = lower_accounts[account_index]
                if 0.0 < lower_wealth < target:
                    raw_fraction = probit_target_leverage(
                        (delta / 2.0) * lower_wealth
                    ) * scale
                    cap = (
                        math.inf
                        if remaining_mean >= 1.0
                        else solvency_c / (1.0 - remaining_mean)
                    )
                    lower_accounts[account_index] = min(
                        target,
                        max(
                            0.0,
                            lower_wealth
                            * (1.0 - min(raw_fraction, cap) * increment),
                        ),
                    )
            if np.sum(upper_accounts) >= target:
                return False
            if np.sum(lower_accounts) >= target:
                return False
        return True

    upper_wealth = 1.0
    lower_wealth = 1.0
    log_target = math.log(2.0 / delta)
    for index in range(population_size - 1):
        draw = index + 1
        remaining_mean = (
            population_size * true_mean - prefix[index]
        ) / (population_size - index)
        increment = values[index] - remaining_mean
        raw_fraction = math.sqrt(
            2.0
            * log_target
            / (variances[index] * draw * math.log(draw + 1.0))
        )
        if 0.0 < upper_wealth < target:
            cap = (
                math.inf
                if remaining_mean <= 0.0
                else solvency_c / remaining_mean
            )
            upper_wealth = min(
                target,
                max(
                    0.0,
                    upper_wealth
                    * (1.0 + min(raw_fraction, cap) * increment),
                ),
            )
        if 0.0 < lower_wealth < target:
            cap = (
                math.inf
                if remaining_mean >= 1.0
                else solvency_c / (1.0 - remaining_mean)
            )
            lower_wealth = min(
                target,
                max(
                    0.0,
                    lower_wealth
                    * (1.0 - min(raw_fraction, cap) * increment),
                ),
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
    population_size = int(args.population_size)
    record_times = tuple(t for t in RECORD_TIMES if t < population_size)
    if not record_times:
        raise ValueError("population-size is too small for the recording grid")
    checkpoints = checkpoint_grid(
        population_size, args.first_checkpoint, args.checkpoint_ratio
    )
    weights = np.full(checkpoints.size, 1.0 / checkpoints.size)

    # Compile all numerical kernels before the experiment begins.
    warm_values = np.array([0.0, 1.0, 0.0, 1.0], dtype=float)
    warm_prefix = np.concatenate(([0.0], np.cumsum(warm_values)))
    warm_variances = predictable_variances(warm_values)
    confidence_interval(
        warm_values,
        warm_prefix,
        warm_variances,
        np.array([2, 3], dtype=np.int64),
        np.array([0.5, 0.5]),
        2,
        4,
        args.delta,
        args.solvency_c,
        True,
        3,
    )

    path_rows: list[dict[str, object]] = []
    coverage_rows: list[dict[str, object]] = []
    for population_index, (name, family, parameters) in enumerate(POPULATIONS):
        population = make_population(population_size, family, parameters)
        true_mean = float(np.mean(population))
        true_variance = float(np.mean((population - true_mean) ** 2))
        print(f"Running {name} ({args.num_simulations} reveal orders)", flush=True)
        for replication in range(args.num_simulations):
            rng = np.random.default_rng(
                np.random.SeedSequence(
                    [args.seed, population_index, replication]
                )
            )
            values = rng.permutation(population)
            prefix = np.concatenate(([0.0], np.cumsum(values)))
            variances = predictable_variances(values)

            for method in METHODS:
                stitched = method == "Stitched bridge-efficient"
                method_solvency_c = (
                    args.solvency_c
                    if stitched
                    else args.plugin_solvency_c
                )
                covered = simultaneous_coverage(
                    values,
                    prefix,
                    variances,
                    checkpoints,
                    weights,
                    true_mean,
                    population_size,
                    args.delta,
                    method_solvency_c,
                    stitched,
                )
                coverage_rows.append(
                    {
                        "distribution": name,
                        "replication": replication,
                        "method": method,
                        "simultaneously_covered": int(covered),
                    }
                )
                for sample_size in record_times:
                    lower, upper = confidence_interval(
                        values,
                        prefix,
                        variances,
                        checkpoints,
                        weights,
                        sample_size,
                        population_size,
                        args.delta,
                        method_solvency_c,
                        stitched,
                        args.bisection_steps,
                    )
                    standard_error = math.sqrt(true_variance) * math.sqrt(
                        (population_size - sample_size)
                        / (sample_size * (population_size - 1.0))
                    )
                    gaussian_width = (
                        2.0
                        * norm.ppf(1.0 - args.delta / 2.0)
                        * standard_error
                    )
                    width = max(0.0, upper - lower)
                    path_rows.append(
                        {
                            "distribution": name,
                            "population_mean": true_mean,
                            "population_variance": true_variance,
                            "replication": replication,
                            "n": sample_size,
                            "sampling_fraction": sample_size / population_size,
                            "method": method,
                            "lower": lower,
                            "upper": upper,
                            "width": width,
                            "normalized_width": width / gaussian_width,
                        }
                    )

    write_rows(output_dir / "pathwise_widths.csv", path_rows)
    write_rows(output_dir / "pathwise_coverage.csv", coverage_rows)

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
                "normalized_standard_error": float(
                    np.std(widths, ddof=1) / math.sqrt(len(widths))
                ),
                "num_simulations": len(widths),
            }
        )
    summary_rows.sort(
        key=lambda row: (
            [spec[0] for spec in POPULATIONS].index(str(row["distribution"])),
            METHODS.index(str(row["method"])),
            int(row["n"]),
        )
    )
    write_rows(output_dir / "summary.csv", summary_rows)

    configuration = {
        "population_size": population_size,
        "populations": [
            {"name": name, "family": family, "parameters": parameters}
            for name, family, parameters in POPULATIONS
        ],
        "record_times": list(record_times),
        "num_simulations": int(args.num_simulations),
        "seed": int(args.seed),
        "delta": float(args.delta),
        "solvency_c": float(args.solvency_c),
        "plugin_solvency_c": float(args.plugin_solvency_c),
        "checkpoint_ratio": float(args.checkpoint_ratio),
        "first_checkpoint": int(args.first_checkpoint),
        "checkpoints": checkpoints.tolist(),
        "checkpoint_weights": "uniform",
        "bisection_steps": int(args.bisection_steps),
        "reported_interval": (
            "raw simultaneously valid interval J_t; no running intersection"
        ),
        "normalization": (
            "2*z_(1-delta/2)*sigma_N*sqrt((N-n)/(n*(N-1)))"
        ),
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
    figure, axes = plt.subplots(3, 3, figsize=(13.5, 11.3), sharey=False)
    handles: dict[str, object] = {}
    for axis, (name, _, _) in zip(axes.ravel(), POPULATIONS):
        axis.axhline(1.0, color="black", linestyle=":", linewidth=1.6)
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
                linestyle="-",
            )[0]
            handles.setdefault(method, handle)
        axis.set_title(name, fontsize=10.5)
        axis.set_xlabel(r"sampling fraction $t/N$")
        axis.set_ylabel("normalized mean width")
        axis.set_xlim(0.0, 0.97)
        axis.set_ylim(bottom=0.9)
        axis.grid(True, linestyle="--", alpha=0.3)

    method_handles = [handles[method] for method in METHODS]
    reference_handle = [
        Line2D([0], [0], color="black", linestyle=":", linewidth=1.7)
    ]
    figure.suptitle(
        "Horizon-free confidence sequences under sampling without replacement",
        fontsize=14,
    )
    figure.legend(
        method_handles,
        [METHOD_LABELS[method] for method in METHODS],
        loc="lower center",
        bbox_to_anchor=(0.38, 0.005),
        ncol=2,
        fontsize=8.4,
        title="Construction (color and marker)",
        title_fontsize=8.6,
        frameon=False,
    )
    figure.legend(
        reference_handle,
        ["Pointwise Gaussian width"],
        loc="lower center",
        bbox_to_anchor=(0.82, 0.005),
        ncol=1,
        fontsize=8.4,
        title="Scale reference (dotted)",
        title_fontsize=8.6,
        frameon=False,
    )
    figure.tight_layout(rect=(0.0, 0.09, 1.0, 0.96))
    output = output_dir / "horizon_free_cs_widths.png"
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
