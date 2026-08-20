#!/usr/bin/env python3
"""Create sqrt(n)-scaled sampling-without-replacement comparisons.

This module leaves the original normalized-width Figure 4 pipeline in
``wor.py`` unchanged.  It produces two additional figures:

1. the existing fixed-N experiment, redrawn on the sqrt(n) * CI-width scale;
2. a new experiment fixing n/N = 0.5 and varying N over Figure 3's grid.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import multiprocessing as mp
import os
import shutil
import time
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy.stats import norm

from wor import (
    BETTING_METHODS,
    CALIBRATION_ORDER,
    LOWER_ROW_OMITTED_METHODS,
    METHOD_ORDER,
    METHOD_STYLES,
    POPULATIONS,
    bardenet_maillard_width,
    betting_interval,
    exact_hypergeometric_width,
    make_population,
    predictable_variances,
    shekhar_ramdas_as_width,
    shekhar_ramdas_rate_width,
    write_rows,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = ROOT / "plots" / "wor"
DEFAULT_PAPER_PLOT_DIR = ROOT.parent / "paper" / "plots"

FIGURE3_POPULATION_SIZES = (
    50,
    100,
    500,
    1_000,
    5_000,
    10_000,
    50_000,
    100_000,
    500_000,
    1_000_000,
)
FIXED_FRACTION_ROW_MINIMUM_N = (50, 100, 1_000)
MINI_POPULATION_ORDER = (
    "Bernoulli(0.5)",
    "Beta(50,50)",
    "Beta(1,5)",
)
SAMPLING_FRACTION = 0.5
DELTA = 0.01
SOLVENCY_C = 1.0
SEED = 20_260_813
DEFAULT_BISECTION_STEPS = 24
BENCHMARK_COLOR = "black"

_POPULATION_CACHE: dict[tuple[int, int], np.ndarray] = {}
_WORKER_BISECTION_STEPS = DEFAULT_BISECTION_STEPS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Draw sqrt(n)-scaled WoR figures, including the fixed-fraction "
            "varying-population experiment."
        )
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--run-experiments",
        action="store_true",
        help="Run or resume the fixed-fraction varying-N simulations.",
    )
    mode.add_argument(
        "--plot-only",
        action="store_true",
        help="Redraw both figures from saved path and summary data.",
    )
    parser.add_argument(
        "--population-sizes",
        nargs="+",
        type=int,
        default=list(FIGURE3_POPULATION_SIZES),
        help="Population-size grid for the fixed-fraction experiment.",
    )
    parser.add_argument(
        "--max-replications",
        type=int,
        help="Optional cap on Figure 3's replication schedule.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(16, os.cpu_count() or 1),
        help="Number of forked simulation workers.",
    )
    parser.add_argument(
        "--bisection-steps",
        type=int,
        default=DEFAULT_BISECTION_STEPS,
        help="Endpoint bisection steps for the varying-N experiment.",
    )
    parser.add_argument(
        "--current-data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory containing Figure 4's saved path-level widths.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory receiving new data and figures.",
    )
    parser.add_argument(
        "--paper-plot-dir",
        type=Path,
        default=DEFAULT_PAPER_PLOT_DIR,
        help="Paper plot directory receiving copies of the new figures.",
    )
    parser.add_argument(
        "--no-paper-copy",
        action="store_true",
        help="Do not copy the new figures into the paper repository.",
    )
    return parser.parse_args()


def replications_for_population_size(population_size: int) -> int:
    """Use the replication schedule of Figure 3."""
    if population_size <= 10_000:
        return 50
    if population_size <= 100_000:
        return 30
    return 20


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _scaled_gaussian_width(
    *,
    population_variance: float,
    sample_size: int,
    population_size: int,
    delta: float = DELTA,
) -> float:
    """Return sqrt(n) times the finite-population Gaussian width."""
    return (
        2.0
        * norm.ppf(1.0 - delta / 2.0)
        * math.sqrt(population_variance)
        * math.sqrt(
            (population_size - sample_size) / (population_size - 1.0)
        )
    )


def summarize_scaled_widths(
    path_rows: Sequence[dict[str, object]],
) -> list[dict[str, object]]:
    grouped: dict[
        tuple[str, int, int, str, str], list[float]
    ] = {}
    metadata: dict[
        tuple[str, int, int, str, str], dict[str, object]
    ] = {}

    for row in path_rows:
        population_size = int(row["population_size"])
        sample_size = int(row["n"])
        key = (
            str(row["distribution"]),
            population_size,
            sample_size,
            str(row["method"]),
            str(row["calibration"]),
        )
        scaled_width = math.sqrt(sample_size) * float(row["width"])
        grouped.setdefault(key, []).append(scaled_width)
        metadata[key] = row

    summary_rows: list[dict[str, object]] = []
    for key, values in grouped.items():
        row = metadata[key]
        population_size = int(row["population_size"])
        sample_size = int(row["n"])
        standard_error = (
            float(np.std(values, ddof=1) / math.sqrt(len(values)))
            if len(values) > 1
            else 0.0
        )
        summary_rows.append(
            {
                "distribution": row["distribution"],
                "population_mean": float(row["population_mean"]),
                "population_variance": float(row["population_variance"]),
                "population_size": population_size,
                "n": sample_size,
                "sampling_fraction": sample_size / population_size,
                "method": row["method"],
                "calibration": row["calibration"],
                "sqrt_n_mean_width": float(np.mean(values)),
                "sqrt_n_standard_error": standard_error,
                "gaussian_sqrt_n_width": _scaled_gaussian_width(
                    population_variance=float(row["population_variance"]),
                    sample_size=sample_size,
                    population_size=population_size,
                ),
                "num_simulations": len(values),
            }
        )

    population_order = [name for name, _, _ in POPULATIONS]
    summary_rows.sort(
        key=lambda row: (
            population_order.index(str(row["distribution"])),
            METHOD_ORDER.index(str(row["method"])),
            CALIBRATION_ORDER.index(str(row["calibration"])),
            int(row["population_size"]),
            int(row["n"]),
        )
    )
    return summary_rows


def load_current_figure_paths(
    data_dir: Path,
) -> list[dict[str, object]]:
    config_path = data_dir / "fixed_horizon_config.json"
    path_path = data_dir / "fixed_horizon_path_widths.csv"
    with config_path.open(encoding="utf-8") as handle:
        configuration = json.load(handle)
    population_size = int(configuration["population_size"])

    output: list[dict[str, object]] = []
    for row in _read_csv(path_path):
        output.append(
            {
                "distribution": row["distribution"],
                "population_mean": float(row["population_mean"]),
                "population_variance": float(row["population_variance"]),
                "population_size": population_size,
                "replication": int(row["replication"]),
                "n": int(row["n"]),
                "sampling_fraction": float(row["sampling_fraction"]),
                "method": row["method"],
                "calibration": row["calibration"],
                "width": float(row["width"]),
            }
        )
    return output


def _initialize_worker(bisection_steps: int) -> None:
    global _WORKER_BISECTION_STEPS
    _WORKER_BISECTION_STEPS = bisection_steps


def _prepare_population_cache(population_sizes: Iterable[int]) -> None:
    for population_size in population_sizes:
        for population_index, (_, family, parameters) in enumerate(
            POPULATIONS
        ):
            _POPULATION_CACHE[(population_size, population_index)] = (
                make_population(population_size, family, parameters)
            )


def _warm_numerical_kernels(bisection_steps: int) -> None:
    values = np.array([0.0, 1.0, 0.0, 1.0], dtype=float)
    prefix = np.concatenate(([0.0], np.cumsum(values)))
    variances = predictable_variances(values)
    for bridge in (True, False):
        betting_interval(
            values,
            prefix,
            variances,
            2,
            8,
            DELTA,
            SOLVENCY_C,
            bridge,
            1.0,
            1.0,
            bisection_steps,
        )


def _simulate_varying_population_task(
    task: tuple[int, int, int],
) -> list[dict[str, object]]:
    population_size, population_index, replication = task
    population_name, family, _ = POPULATIONS[population_index]
    sample_size = int(round(SAMPLING_FRACTION * population_size))
    population = _POPULATION_CACHE[(population_size, population_index)]
    population_mean = float(np.mean(population))
    population_variance = float(
        np.mean((population - population_mean) ** 2)
    )

    data_rng = np.random.default_rng(
        np.random.SeedSequence(
            [SEED, population_size, population_index, replication]
        )
    )
    indices = data_rng.choice(
        population_size,
        size=sample_size,
        replace=False,
        shuffle=True,
    )
    values = np.ascontiguousarray(population[indices], dtype=float)
    prefix = np.empty(sample_size + 1, dtype=float)
    prefix[0] = 0.0
    np.cumsum(values, out=prefix[1:])
    variances = predictable_variances(values)

    sample_sum = float(prefix[sample_size])
    sample_mean = sample_sum / sample_size
    sample_variance = max(
        0.0,
        float(np.mean(values * values)) - sample_mean * sample_mean,
    )
    feasible_lower = sample_sum / population_size
    feasible_upper = (
        sample_sum + population_size - sample_size
    ) / population_size

    randomizer_rng = np.random.default_rng(
        np.random.SeedSequence(
            [SEED, population_size, population_index, replication, 1]
        )
    )
    positive_floor = np.finfo(float).tiny
    u_plus, u_minus = (
        max(float(value), positive_floor)
        for value in randomizer_rng.random(2)
    )

    bridge_lower, bridge_upper = betting_interval(
        values,
        prefix,
        variances,
        sample_size,
        population_size,
        DELTA,
        SOLVENCY_C,
        True,
        bisection_steps=_WORKER_BISECTION_STEPS,
    )
    bridge_randomized_lower, bridge_randomized_upper = betting_interval(
        values,
        prefix,
        variances,
        sample_size,
        population_size,
        DELTA,
        SOLVENCY_C,
        True,
        u_plus,
        u_minus,
        _WORKER_BISECTION_STEPS,
    )
    wsr_lower, wsr_upper = betting_interval(
        values,
        prefix,
        variances,
        sample_size,
        population_size,
        DELTA,
        SOLVENCY_C,
        False,
        bisection_steps=_WORKER_BISECTION_STEPS,
    )
    wsr_randomized_lower, wsr_randomized_upper = betting_interval(
        values,
        prefix,
        variances,
        sample_size,
        population_size,
        DELTA,
        SOLVENCY_C,
        False,
        u_plus,
        u_minus,
        _WORKER_BISECTION_STEPS,
    )

    widths = {
        ("Bridge-efficient betting", "Deterministic Markov"): max(
            0.0, bridge_upper - bridge_lower
        ),
        (
            "Bridge-efficient betting",
            "Uniformly randomized Markov",
        ): max(
            0.0,
            bridge_randomized_upper - bridge_randomized_lower,
        ),
        ("WSR running intersection", "Deterministic Markov"): max(
            0.0, wsr_upper - wsr_lower
        ),
        (
            "WSR running intersection",
            "Uniformly randomized Markov",
        ): max(
            0.0,
            wsr_randomized_upper - wsr_randomized_lower,
        ),
        ("Bardenet--Maillard EBS", "As published"): (
            bardenet_maillard_width(
                sample_mean,
                sample_variance,
                sample_size,
                population_size,
                DELTA,
                feasible_lower,
                feasible_upper,
            )
        ),
        ("Shekhar--Ramdas AS-CI", "As published"): (
            shekhar_ramdas_as_width(
                sample_mean,
                sample_size,
                population_size,
                DELTA,
                feasible_lower,
                feasible_upper,
            )
        ),
    }
    if family == "bernoulli":
        widths[("Exact hypergeometric", "As published")] = (
            exact_hypergeometric_width(
                int(round(sample_sum)),
                sample_size,
                population_size,
                DELTA,
            )
        )
        widths[("Shekhar--Ramdas rate CI", "As published")] = (
            shekhar_ramdas_rate_width(
                sample_mean,
                sample_size,
                population_size,
                DELTA,
                feasible_lower,
                feasible_upper,
            )
        )

    return [
        {
            "distribution": population_name,
            "population_mean": population_mean,
            "population_variance": population_variance,
            "population_size": population_size,
            "replication": replication,
            "n": sample_size,
            "sampling_fraction": sample_size / population_size,
            "method": method,
            "calibration": calibration,
            "width": width,
            "sqrt_n_width": math.sqrt(sample_size) * width,
        }
        for (method, calibration), width in widths.items()
    ]


def _path_sort_key(row: dict[str, object]) -> tuple[object, ...]:
    population_order = [name for name, _, _ in POPULATIONS]
    return (
        int(row["population_size"]),
        population_order.index(str(row["distribution"])),
        int(row["replication"]),
        METHOD_ORDER.index(str(row["method"])),
        CALIBRATION_ORDER.index(str(row["calibration"])),
    )


def run_varying_population_experiment(
    *,
    output_dir: Path,
    population_sizes: Sequence[int],
    max_replications: int | None,
    workers: int,
    bisection_steps: int,
) -> list[dict[str, object]]:
    if any(population_size < 20 for population_size in population_sizes):
        raise ValueError("all population sizes must be at least 20")
    if any(population_size % 2 for population_size in population_sizes):
        raise ValueError("population sizes must be even when n/N = 0.5")
    if max_replications is not None and max_replications <= 0:
        raise ValueError("max-replications must be positive")
    if workers <= 0:
        raise ValueError("workers must be positive")
    if bisection_steps < 12:
        raise ValueError("bisection-steps must be at least 12")

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "fixed_fraction_path_widths_checkpoint.csv"
    final_path = output_dir / "fixed_fraction_path_widths.csv"

    existing_rows: list[dict[str, object]] = []
    resume_path = checkpoint_path if checkpoint_path.exists() else final_path
    if resume_path.exists():
        existing_rows = [dict(row) for row in _read_csv(resume_path)]
        for row in existing_rows:
            for key in (
                "population_mean",
                "population_variance",
                "sampling_fraction",
                "width",
                "sqrt_n_width",
            ):
                row[key] = float(row[key])
            for key in ("population_size", "replication", "n"):
                row[key] = int(row[key])

    completed = {
        (
            int(row["population_size"]),
            [name for name, _, _ in POPULATIONS].index(
                str(row["distribution"])
            ),
            int(row["replication"]),
        )
        for row in existing_rows
    }
    tasks: list[tuple[int, int, int]] = []
    replications_by_size: dict[int, int] = {}
    for population_size in population_sizes:
        replications = replications_for_population_size(population_size)
        if max_replications is not None:
            replications = min(replications, max_replications)
        replications_by_size[population_size] = replications
        for population_index in range(len(POPULATIONS)):
            for replication in range(replications):
                task = (population_size, population_index, replication)
                if task not in completed:
                    tasks.append(task)

    if tasks:
        needed_sizes = sorted({task[0] for task in tasks})
        print(
            f"Preparing {len(needed_sizes)} population sizes and "
            f"{len(tasks)} simulation tasks."
        )
        _prepare_population_cache(needed_sizes)
        _warm_numerical_kernels(bisection_steps)
        start = time.perf_counter()

        if workers == 1:
            _initialize_worker(bisection_steps)
            iterator = map(_simulate_varying_population_task, tasks)
            pool = None
        else:
            context = mp.get_context("fork")
            pool = context.Pool(
                processes=workers,
                initializer=_initialize_worker,
                initargs=(bisection_steps,),
            )
            iterator = pool.imap_unordered(
                _simulate_varying_population_task,
                tasks,
                chunksize=1,
            )

        try:
            for completed_count, task_rows in enumerate(iterator, start=1):
                existing_rows.extend(task_rows)
                if completed_count % 25 == 0 or completed_count == len(tasks):
                    existing_rows.sort(key=_path_sort_key)
                    write_rows(checkpoint_path, existing_rows)
                    elapsed = time.perf_counter() - start
                    print(
                        f"Completed {completed_count}/{len(tasks)} tasks "
                        f"in {elapsed:.1f}s.",
                        flush=True,
                    )
        finally:
            if pool is not None:
                pool.close()
                pool.join()

    existing_rows.sort(key=_path_sort_key)
    write_rows(final_path, existing_rows)
    summary_rows = summarize_scaled_widths(existing_rows)
    write_rows(
        output_dir / "fixed_fraction_sqrt_n_widths.csv",
        summary_rows,
    )

    configuration = {
        "population_sizes": list(population_sizes),
        "sampling_fraction": SAMPLING_FRACTION,
        "sample_sizes": [
            int(round(SAMPLING_FRACTION * size))
            for size in population_sizes
        ],
        "replications_by_population_size": replications_by_size,
        "seed": SEED,
        "delta": DELTA,
        "solvency_c": SOLVENCY_C,
        "bisection_steps": bisection_steps,
        "scale": "sqrt(n) * confidence-interval width",
        "population_size_grid_note": (
            "Figure 3 grid with N=10 omitted because the WoR experiment "
            "requires N >= 20"
        ),
        "randomization": (
            "one independent uniform pair per population, population size, "
            "and reveal order; shared across betting methods"
        ),
    }
    with (output_dir / "fixed_fraction_config.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(configuration, handle, indent=2)
        handle.write("\n")
    return summary_rows


def load_scaled_summary(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = [dict(row) for row in _read_csv(path)]
    for row in rows:
        for key in (
            "population_mean",
            "population_variance",
            "sampling_fraction",
            "sqrt_n_mean_width",
            "sqrt_n_standard_error",
            "gaussian_sqrt_n_width",
        ):
            row[key] = float(row[key])
        for key in ("population_size", "n", "num_simulations"):
            row[key] = int(row[key])
    return rows


def _add_axis_grid(axis: plt.Axes) -> None:
    axis.grid(True, linestyle="--", alpha=0.3)
    axis.set_axisbelow(True)


def scaled_width_plot(
    *,
    summary_rows: Sequence[dict[str, object]],
    output_path: Path,
    varying_population_size: bool,
) -> Path:
    figure, axes = plt.subplots(3, 3, figsize=(13.5, 11.3))
    handles_by_method: dict[str, object] = {}

    for panel_index, (axis, (population_name, _, _)) in enumerate(
        zip(axes.ravel(), POPULATIONS)
    ):
        population_rows = [
            row
            for row in summary_rows
            if row["distribution"] == population_name
        ]
        if varying_population_size:
            minimum_population_size = FIXED_FRACTION_ROW_MINIMUM_N[
                panel_index // 3
            ]
            population_rows = [
                row
                for row in population_rows
                if int(row["population_size"]) >= minimum_population_size
            ]
        x_key = (
            "population_size"
            if varying_population_size
            else "sampling_fraction"
        )
        reference_by_x = {
            float(row[x_key]): float(row["gaussian_sqrt_n_width"])
            for row in population_rows
        }
        reference_x = sorted(reference_by_x)
        reference_y = [reference_by_x[value] for value in reference_x]
        axis.plot(
            reference_x,
            reference_y,
            color=BENCHMARK_COLOR,
            linestyle=":",
            linewidth=1.5,
            zorder=0,
        )
        panel_values = list(reference_y)

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
                    key=lambda row: float(row[x_key]),
                )
                if not method_rows:
                    continue
                style = METHOD_STYLES[method]
                x_values = [float(row[x_key]) for row in method_rows]
                y_values = [
                    float(row["sqrt_n_mean_width"])
                    for row in method_rows
                ]
                panel_values.extend(y_values)
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
        if varying_population_size:
            displayed_sizes = sorted(
                {int(row["population_size"]) for row in population_rows}
            )
            axis.set_xscale("log")
            axis.set_xlim(
                min(displayed_sizes) / 1.25,
                max(displayed_sizes) * 1.25,
            )
            axis.set_xlabel(r"Population size $N$ ($n/N=0.5$)")
        else:
            fractions = sorted(
                {float(row["sampling_fraction"]) for row in population_rows}
            )
            axis.set_xticks(fractions)
            axis.set_xticklabels(
                [f"{fraction:.1f}".lstrip("0") for fraction in fractions]
            )
            axis.set_xlim(0.065, 0.935)
            axis.set_xlabel(r"Sampling fraction $\rho=n/N$")
        axis.set_ylabel(r"$\sqrt{n}\times$ CI width")
        if panel_values:
            axis.set_ylim(bottom=max(0.0, 0.94 * min(panel_values)))
        _add_axis_grid(axis)

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
            [0],
            [0],
            color="0.25",
            linestyle="--",
            linewidth=1.55,
            marker="o",
            markerfacecolor="white",
            markeredgecolor="0.25",
        ),
        Line2D(
            [0],
            [0],
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
            [0],
            [0],
            color=BENCHMARK_COLOR,
            linestyle=":",
            linewidth=1.7,
        )
    ]
    title = (
        r"Confidence intervals at sampling fraction $n/N=0.5$"
        if varying_population_size
        else r"Confidence intervals under sampling without replacement "
        r"($N=4000$)"
    )
    figure.suptitle(title, fontsize=15)
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
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)
    return output_path


def fixed_fraction_mini_plot(
    *,
    summary_rows: Sequence[dict[str, object]],
    output_path: Path,
) -> Path:
    """Plot the fixed-fraction comparison in Figure 1's 2-by-3 layout."""
    figure, axes = plt.subplots(2, 3, figsize=(11.2, 6.4))
    calibration_rows = (
        ("Deterministic Markov", "Deterministic", "--", False, 5.2, 4),
        ("Uniformly randomized Markov", "Randomized", "-", True, 3.8, 3),
    )
    displayed_sizes = sorted(
        {int(row["population_size"]) for row in summary_rows}
    )

    for row_index, calibration_row in enumerate(calibration_rows):
        (
            calibration,
            _,
            linestyle,
            filled_marker,
            marker_size,
            zorder,
        ) = calibration_row
        for axis, population_name in zip(
            axes[row_index], MINI_POPULATION_ORDER
        ):
            population_rows = [
                row
                for row in summary_rows
                if row["distribution"] == population_name
            ]
            reference_by_size = {
                int(row["population_size"]): float(
                    row["gaussian_sqrt_n_width"]
                )
                for row in population_rows
            }
            reference_x = sorted(reference_by_size)
            axis.plot(
                reference_x,
                [reference_by_size[value] for value in reference_x],
                color=BENCHMARK_COLOR,
                linestyle=":",
                linewidth=1.6,
                label="_nolegend_",
                zorder=2,
            )

            for method in (
                "Bridge-efficient betting",
                "WSR running intersection",
            ):
                method_rows = sorted(
                    (
                        row
                        for row in population_rows
                        if (
                            row["method"] == method
                            and row["calibration"] == calibration
                        )
                    ),
                    key=lambda row: int(row["population_size"]),
                )
                style = METHOD_STYLES[method]
                axis.plot(
                    [int(row["population_size"]) for row in method_rows],
                    [
                        float(row["sqrt_n_mean_width"])
                        for row in method_rows
                    ],
                    color=style["color"],
                    marker=style["marker"],
                    markerfacecolor=(
                        style["color"] if filled_marker else "none"
                    ),
                    markeredgecolor=style["color"],
                    markeredgewidth=0.9,
                    linestyle=linestyle,
                    linewidth=1.9,
                    markersize=marker_size,
                    label="_nolegend_",
                    zorder=zorder,
                )

            axis.set_xscale("log")
            axis.set_xlim(
                min(displayed_sizes) / 1.25,
                max(displayed_sizes) * 1.25,
            )
            if row_index == 0:
                axis.set_title(population_name)
            axis.set_xlabel(r"Population size $N$ ($n/N=0.5$)")
            axis.set_ylabel(r"$\sqrt{n}\times$ CI width")
            _add_axis_grid(axis)

    for row_index, calibration_row in enumerate(calibration_rows):
        row_label = calibration_row[1]
        axes[row_index, -1].annotate(
            row_label,
            xy=(1.07, 0.5),
            xycoords="axes fraction",
            ha="center",
            va="center",
            rotation=-90,
            fontsize=9.5,
            annotation_clip=False,
        )

    method_handles = [
        Line2D(
            [0],
            [0],
            color=METHOD_STYLES[method]["color"],
            marker=METHOD_STYLES[method]["marker"],
            linewidth=2.0,
            markerfacecolor=METHOD_STYLES[method]["color"],
            markeredgecolor=METHOD_STYLES[method]["color"],
            markersize=4.5,
        )
        for method in (
            "Bridge-efficient betting",
            "WSR running intersection",
        )
    ]
    calibration_handles = [
        Line2D(
            [0],
            [0],
            color="0.25",
            linestyle="--",
            linewidth=2.0,
            marker="o",
            markerfacecolor="none",
            markeredgecolor="0.25",
            markersize=5.2,
        ),
        Line2D(
            [0],
            [0],
            color="0.25",
            linestyle="-",
            linewidth=2.0,
            marker="o",
            markerfacecolor="0.25",
            markeredgecolor="0.25",
            markersize=3.8,
        ),
    ]
    reference_handles = [
        Line2D(
            [0],
            [0],
            color=BENCHMARK_COLOR,
            linestyle=":",
            linewidth=1.7,
        )
    ]
    figure.legend(
        method_handles,
        [
            "GE-betting",
            "WSR running intersection",
        ],
        loc="lower center",
        bbox_to_anchor=(0.22, 0.012),
        ncol=2,
        fontsize=8.4,
        title="Construction (color and marker)",
        title_fontsize=8.6,
        frameon=False,
    )
    figure.legend(
        calibration_handles,
        ["Deterministic", "Randomized"],
        loc="lower center",
        bbox_to_anchor=(0.61, 0.028),
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
        bbox_to_anchor=(0.87, 0.028),
        ncol=1,
        fontsize=8.4,
        title="Reference (dotted)",
        title_fontsize=8.6,
        frameon=False,
    )
    figure.tight_layout(rect=(0.0, 0.12, 0.98, 1.0))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)
    return output_path


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    current_data_dir = args.current_data_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    current_paths = load_current_figure_paths(current_data_dir)
    current_summary = summarize_scaled_widths(current_paths)
    write_rows(
        output_dir / "fixed_horizon_sqrt_n_widths.csv",
        current_summary,
    )
    fixed_n_path = scaled_width_plot(
        summary_rows=current_summary,
        output_path=output_dir / "fixed_horizon_sqrt_n_widths.png",
        varying_population_size=False,
    )

    population_sizes = tuple(sorted(set(args.population_sizes)))
    varying_summary_path = output_dir / "fixed_fraction_sqrt_n_widths.csv"
    should_run = args.run_experiments or (
        not args.plot_only and not varying_summary_path.exists()
    )
    if should_run:
        varying_summary = run_varying_population_experiment(
            output_dir=output_dir,
            population_sizes=population_sizes,
            max_replications=args.max_replications,
            workers=args.workers,
            bisection_steps=args.bisection_steps,
        )
    else:
        if not varying_summary_path.exists():
            raise FileNotFoundError(
                f"No saved varying-N summary at {varying_summary_path}; "
                "run with --run-experiments."
            )
        varying_summary = load_scaled_summary(varying_summary_path)

    fixed_fraction_path = scaled_width_plot(
        summary_rows=varying_summary,
        output_path=output_dir / "fixed_fraction_sqrt_n_widths.png",
        varying_population_size=True,
    )
    fixed_fraction_mini_path = fixed_fraction_mini_plot(
        summary_rows=varying_summary,
        output_path=output_dir / "fixed_fraction_intro_comparison.png",
    )

    if not args.no_paper_copy:
        paper_plot_dir = args.paper_plot_dir.resolve()
        paper_plot_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(
            fixed_n_path,
            paper_plot_dir / "wor_fixed_horizon_sqrt_n_widths.png",
        )
        shutil.copy2(
            fixed_fraction_path,
            paper_plot_dir / "wor_fixed_fraction_sqrt_n_widths.png",
        )
        shutil.copy2(
            fixed_fraction_mini_path,
            paper_plot_dir / "wor_fixed_fraction_intro_comparison.png",
        )

    print(fixed_n_path)
    print(fixed_fraction_path)
    print(fixed_fraction_mini_path)


if __name__ == "__main__":
    main()
