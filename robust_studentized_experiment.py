#!/usr/bin/env python3
"""Finite-sample audit of symmetric studentized terminal inference.

The experiment separates four objects:

1. chronological common-clock GE, the paper's proposed procedure;
2. exact binary inversion after unbiased Bernoulli rounding;
3. a fully order-invariant, martingale-valid sample-variance inequality;
4. an optimistic interval using only the correction forced by a particular
   near-degenerate null.  The fourth object is not a confidence interval.  It
   is narrower than any valid member of the hard-boundary family whenever
   the rare-path obstruction is active, so losing to it rules out an
   improvement by that family.

The script also computes exact finite-grid Bellman prices for the terminal
event A_n >= z sqrt(R_n)+b.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import shutil
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import betting
import robust_studentized_dp as robust_dp


METHOD_GE = "Chronological GE"
METHOD_BERNOULLI_ROUNDING = "Exact Bernoulli rounding"
METHOD_VALID_EMPIRICAL = "Symmetric empirical-variance bound"
METHOD_OPTIMISTIC = "Optimistic hard-boundary benchmark (not valid)"


def symmetric_empirical_variance_ci_endpoints(X, delta):
    """Invert a sample-variance-only martingale inequality.

    This is Corollary 1 of Yuan (2025), specialized to observations in [0,1].
    Its two-sided failure probability is at most delta after setting the
    source's auxiliary tail probability to delta/3 and using |X_i-m| <= 1.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 1 or X.size == 0:
        raise ValueError("X must be a nonempty vector")
    if np.any((X < 0.0) | (X > 1.0)):
        raise ValueError("X must lie in [0,1]")
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must lie in (0,1)")
    n = X.size
    tail_log = math.log(3.0 / delta)
    denominator = 1.0 - math.sqrt(2.0 * tail_log / n)
    if denominator <= 0.0:
        return 0.0, 1.0, False
    center = float(np.mean(X))
    empirical_sum_squares = float(np.sum((X - center) ** 2))
    radius = (
        math.sqrt(2.0 * empirical_sum_squares * tail_log) / n
        + 3.15 * tail_log / n
    ) / denominator
    return max(0.0, center - radius), min(1.0, center + radius), False


def _samplers(rng):
    return {
        "Beta(1,5)": (
            lambda n: rng.beta(1.0, 5.0, n),
            math.sqrt(5.0 / 252.0),
        ),
        "Bernoulli(0.5)": (
            lambda n: rng.binomial(1, 0.5, n).astype(float),
            0.5,
        ),
        "Uniform(0.45,0.55)": (
            lambda n: rng.uniform(0.45, 0.55, n),
            math.sqrt(0.1**2 / 12.0),
        ),
    }


def _summarize(values):
    values = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(values)),
        "standard_error": float(np.std(values, ddof=1) / math.sqrt(values.size)),
        "q10": float(np.quantile(values, 0.1)),
        "q50": float(np.quantile(values, 0.5)),
        "q90": float(np.quantile(values, 0.9)),
    }


def run_width_study(
    *,
    output_dir,
    delta,
    n_values,
    repetitions,
    seed,
):
    """Run the paired width comparison."""
    rng = np.random.default_rng(seed)
    calibration_rng = np.random.default_rng(seed + 1)
    rounding_rng = np.random.default_rng(seed + 2)
    rows = []
    z = betting.asymptotic_limit_digital(delta)
    started = time.perf_counter()

    for distribution, (sample, sigma) in _samplers(rng).items():
        for n in n_values:
            print(
                f"widths: {distribution}, n={n}, reps={repetitions}",
                flush=True,
            )
            scale = math.sqrt(n) / (2.0 * sigma * z)
            for repetition in range(repetitions):
                X = sample(int(n))
                randomizers = calibration_rng.uniform(size=2)
                rounded = robust_dp.unbiased_grid_quantize(
                    X, 1, rounding_rng
                )
                rounded_lower, rounded_upper = (
                    betting.bernoulli_dp_ci_endpoints(
                        rounded,
                        delta,
                        upper_randomizer=randomizers[0],
                        lower_randomizer=randomizers[1],
                    )
                )
                methods = {
                    METHOD_GE: betting.probit_common_clock_ci_endpoints(
                        X,
                        delta,
                        randomizers=randomizers,
                        c=1.0,
                    ),
                    METHOD_BERNOULLI_ROUNDING: (
                        rounded_lower,
                        rounded_upper,
                        False,
                    ),
                    METHOD_VALID_EMPIRICAL:
                        symmetric_empirical_variance_ci_endpoints(X, delta),
                    METHOD_OPTIMISTIC:
                        robust_dp.optimistic_studentized_ci_endpoints(X, delta),
                }
                for method, (lower, upper, empty) in methods.items():
                    width = float(upper - lower)
                    rows.append({
                        "distribution": distribution,
                        "n": int(n),
                        "repetition": int(repetition),
                        "method": method,
                        "lower": float(lower),
                        "upper": float(upper),
                        "empty": bool(empty),
                        "width": width,
                        "normalized_width": float(width * scale),
                    })

    path = output_dir / "path_widths.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    summaries = []
    paired = []
    for distribution in _samplers(np.random.default_rng(0)):
        for n in n_values:
            selected_cell = [
                row
                for row in rows
                if row["distribution"] == distribution and row["n"] == n
            ]
            ge_by_repetition = {
                row["repetition"]: row["normalized_width"]
                for row in selected_cell
                if row["method"] == METHOD_GE
            }
            for method in (
                METHOD_GE,
                METHOD_BERNOULLI_ROUNDING,
                METHOD_VALID_EMPIRICAL,
                METHOD_OPTIMISTIC,
            ):
                method_rows = [
                    row for row in selected_cell if row["method"] == method
                ]
                summaries.append({
                    "distribution": distribution,
                    "n": int(n),
                    "method": method,
                    **_summarize(
                        [row["normalized_width"] for row in method_rows]
                    ),
                })
                if method != METHOD_GE:
                    differences = np.asarray([
                        row["normalized_width"]
                        - ge_by_repetition[row["repetition"]]
                        for row in method_rows
                    ])
                    paired.append({
                        "distribution": distribution,
                        "n": int(n),
                        "method": method,
                        "mean_difference": float(np.mean(differences)),
                        "standard_error": float(
                            np.std(differences, ddof=1)
                            / math.sqrt(differences.size)
                        ),
                        "ge_win_rate": float(np.mean(differences > 0.0)),
                    })

    for filename, records in (
        ("summary.csv", summaries),
        ("paired_summary.csv", paired),
    ):
        with (output_dir / filename).open(
            "w", newline="", encoding="utf-8"
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=records[0].keys())
            writer.writeheader()
            writer.writerows(records)
    return rows, summaries, paired, time.perf_counter() - started


def run_bellman_audit(*, output_dir, delta, bisection_steps):
    """Compute exact restricted-grid Bellman calibrations."""
    alpha = delta / 2.0
    z = betting.asymptotic_limit_digital(delta)
    configurations = (
        (12, 6), (12, 12), (12, 24),
        (20, 6), (20, 12), (20, 24),
        (30, 8), (30, 12), (30, 16),
    )
    rows = []
    started = time.perf_counter()
    for n, grid_intervals in configurations:
        print(
            f"Bellman grid: n={n}, J={grid_intervals}",
            flush=True,
        )
        layers = robust_dp.reachable_state_layers(n, grid_intervals)
        for m in (0.2, 0.5, 0.8):
            row = robust_dp.calibrate_grid_correction(
                n=n,
                m=m,
                alpha=alpha,
                z=z,
                grid_intervals=grid_intervals,
                bisection_steps=bisection_steps,
                layers=layers,
            )
            row["rare_event_lower_bound"] = robust_dp.rare_event_lower_bound(
                n, m, alpha, z
            )
            row["excess_over_rare_bound"] = (
                row["correction"] - row["rare_event_lower_bound"]
            )
            rows.append(row)
        del layers
        gc.collect()

    path = output_dir / "bellman_calibration.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    return rows, time.perf_counter() - started


def _plot_widths(summaries, output_dir, delta):
    distributions = list(dict.fromkeys(row["distribution"] for row in summaries))
    styles = (
        (METHOD_GE, "#2ca02c", "o", "-"),
        (METHOD_BERNOULLI_ROUNDING, "#1f77b4", "D", "-."),
        (METHOD_OPTIMISTIC, "#d62728", "s", "--"),
        (METHOD_VALID_EMPIRICAL, "#7f7f7f", "^", ":"),
    )
    fig, axes = plt.subplots(
        1,
        len(distributions),
        figsize=(13.2, 4.7),
        sharey=True,
    )
    for axis, distribution in zip(np.atleast_1d(axes), distributions):
        for method, color, marker, linestyle in styles:
            selected = [
                row
                for row in summaries
                if row["distribution"] == distribution
                and row["method"] == method
            ]
            selected.sort(key=lambda row: row["n"])
            axis.errorbar(
                [row["n"] for row in selected],
                [row["mean"] for row in selected],
                yerr=[1.96 * row["standard_error"] for row in selected],
                color=color,
                marker=marker,
                linestyle=linestyle,
                linewidth=1.8,
                capsize=2.5,
                label=method,
            )
        axis.axhline(1.0, color="black", linestyle="-.", linewidth=1.0)
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_title(distribution)
        axis.set_xlabel("sample size")
        axis.grid(True, linestyle="--", alpha=0.3)
    axes[0].set_ylabel("mean width / Gaussian benchmark")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=2,
        frameon=False,
    )
    fig.suptitle(
        (
            "Symmetric studentized terminal-event audit "
            f"({1.0-delta:.1%} confidence)"
        ),
        y=1.10,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.82))
    path = output_dir / "robust_studentized_widths.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def run_experiment(
    *,
    output_dir="plots/robust_studentized",
    delta=0.01,
    n_values=(50, 200, 1000, 5000),
    repetitions=200,
    seed=20260814,
    bisection_steps=15,
    paper_plot="../paper/plots/robust_studentized_widths.png",
):
    """Run the full time-bounded research audit."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    _, summaries, paired, width_seconds = run_width_study(
        output_dir=output_dir,
        delta=delta,
        n_values=tuple(int(n) for n in n_values),
        repetitions=int(repetitions),
        seed=int(seed),
    )
    bellman_rows, bellman_seconds = run_bellman_audit(
        output_dir=output_dir,
        delta=delta,
        bisection_steps=int(bisection_steps),
    )
    plot_path = _plot_widths(summaries, output_dir, delta)
    paper_plot = Path(paper_plot)
    paper_plot.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(plot_path, paper_plot)

    metadata = {
        "delta": float(delta),
        "n_values": [int(n) for n in n_values],
        "repetitions": int(repetitions),
        "seed": int(seed),
        "bellman_bisection_steps": int(bisection_steps),
        "width_seconds": float(width_seconds),
        "bellman_seconds": float(bellman_seconds),
        "elapsed_seconds": float(time.perf_counter() - started),
        "interpretation": {
            METHOD_GE: "finite-sample martingale-valid and order-dependent",
            METHOD_BERNOULLI_ROUNDING: (
                "finite-sample valid and order-invariant in distribution; "
                "uses external observation-level randomization"
            ),
            METHOD_VALID_EMPIRICAL:
                "finite-sample martingale-valid and order-invariant",
            METHOD_OPTIMISTIC: (
                "not valid; optimistic lower-width diagnostic for the "
                "hard-boundary studentized family"
            ),
        },
        "all_optimistic_differences_positive": bool(all(
            row["mean_difference"] > 0.0
            for row in paired
            if row["method"] == METHOD_OPTIMISTIC
        )),
        "bellman_rows": len(bellman_rows),
    }
    with (output_dir / "config.json").open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2)
    print(
        f"saved {output_dir} in {metadata['elapsed_seconds']:.1f} seconds",
        flush=True,
    )
    return metadata


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="plots/robust_studentized")
    parser.add_argument("--delta", type=float, default=0.01)
    parser.add_argument(
        "--n-values",
        type=int,
        nargs="+",
        default=[50, 200, 1000, 5000],
    )
    parser.add_argument("--repetitions", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--bisection-steps", type=int, default=15)
    parser.add_argument(
        "--paper-plot",
        default="../paper/plots/robust_studentized_widths.png",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = _parse_args()
    run_experiment(
        output_dir=arguments.output_dir,
        delta=arguments.delta,
        n_values=tuple(arguments.n_values),
        repetitions=arguments.repetitions,
        seed=arguments.seed,
        bisection_steps=arguments.bisection_steps,
        paper_plot=arguments.paper_plot,
    )
