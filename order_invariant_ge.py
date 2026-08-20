#!/usr/bin/env python3
"""Permutation-integrated Gaussian-efficient betting.

The chronological GE procedure is valid for the constant-conditional-mean
model, but its terminal wealth depends on the reveal order.  Under iid
sampling, a uniform permutation of the observations has the same law as the
original sample.  Consequently, an arithmetic average of GE e-values over
permutations is again an e-value and is invariant to the original ordering
when the full permutation average is used.

This module implements one finite Monte Carlo construction and one diagnostic
variant:

* ``capped`` averages the target-capped wealth used by the paper and applies
  uniformly randomized Markov calibration;
* ``overshoot`` freezes each arm at its raw wealth on first crossing the
  target and averages these uncapped stopped e-values.  It diagnoses a
  tempting deterministic construction, but is not inverted as a confidence
  interval because the averaged arms need not be monotone in the candidate
  mean.

The finite permutation sample is part of the external randomization.  It is
valid under iid sampling and invariant in distribution, but it does not retain
the paper's guarantee for arbitrary martingale-dependent observations.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numba import njit, prange

import betting


METHOD_CHRONOLOGICAL = "Chronological GE"
METHOD_PERMUTATION_CAPPED = "Permutation-integrated GE (randomized)"
METHOD_PERMUTATION_OVERSHOOT = "Permutation-integrated GE (overshoot)"
METHOD_EXACT_BINOMIAL = "Exact binomial (randomized)"


@njit
def compute_M_probit_common_clock_stopped_arms(
    X,
    m,
    delta,
    c=1.0,
    buffer_rounds=0.0,
):
    """Return GE arms stopped at their untruncated first crossing.

    Immediately before a crossing the stake is predictable and solvent.  We
    retain the raw post-update wealth and then stop, rather than replacing it
    by the target.  Optional stopping therefore leaves each arm an e-value and
    gives a permutation average enough headroom for deterministic Markov
    calibration.
    """
    n = len(X)
    alpha = delta / 2.0
    target = 1.0 / alpha
    eps = 1e-12

    M_plus = 1.0
    M_minus = 1.0
    sum_x = 0.0
    pred_sq = 0.0

    for i in range(n):
        mean_hat = (0.5 + sum_x) / (1.0 + i)
        var_hat = max((0.25 + pred_sq) / (1.0 + i), eps)
        remaining = float(n - i) + max(buffer_rounds, 0.0)

        if 0.0 < M_plus < target:
            target_fraction = alpha * M_plus
            bet_plus = (
                betting.probit_target_leverage(target_fraction)
                / np.sqrt(remaining * var_hat)
            )
            bet_plus = min(bet_plus, c / (m + eps))
        else:
            bet_plus = 0.0

        if 0.0 < M_minus < target:
            target_fraction = alpha * M_minus
            bet_minus = (
                betting.probit_target_leverage(target_fraction)
                / np.sqrt(remaining * var_hat)
            )
            bet_minus = min(bet_minus, c / (1.0 - m + eps))
        else:
            bet_minus = 0.0

        centered = X[i] - m
        M_plus = max(M_plus * (1.0 + bet_plus * centered), 0.0)
        M_minus = max(M_minus * (1.0 - bet_minus * centered), 0.0)

        residual = X[i] - mean_hat
        sum_x += X[i]
        pred_sq += residual * residual

    return M_plus, M_minus


@njit(parallel=True)
def permutation_average_arms(
    permuted_samples,
    m,
    delta,
    retain_overshoot,
    c=1.0,
):
    """Average the two GE arms over rows of ``permuted_samples``."""
    repetitions = permuted_samples.shape[0]
    plus = np.empty(repetitions)
    minus = np.empty(repetitions)
    for index in prange(repetitions):
        if retain_overshoot:
            arm_plus, arm_minus = compute_M_probit_common_clock_stopped_arms(
                permuted_samples[index], m, delta, c=c
            )
        else:
            arm_plus, arm_minus = betting.compute_M_probit_common_clock_arms(
                permuted_samples[index], m, delta, c=c
            )
        plus[index] = arm_plus
        minus[index] = arm_minus
    return np.mean(plus), np.mean(minus)


@njit(parallel=True)
def permutation_average_arm_scores(
    permuted_samples,
    means,
    delta,
    retain_overshoot,
    u_plus,
    u_minus,
    c=1.0,
):
    """Evaluate normalized averaged-arm scores over candidate means."""
    alpha = delta / 2.0
    plus_scores = np.empty(means.size)
    minus_scores = np.empty(means.size)
    for index in prange(means.size):
        plus_sum = 0.0
        minus_sum = 0.0
        for permutation in range(permuted_samples.shape[0]):
            if retain_overshoot:
                plus, minus = compute_M_probit_common_clock_stopped_arms(
                    permuted_samples[permutation],
                    means[index],
                    delta,
                    c=c,
                )
            else:
                plus, minus = betting.compute_M_probit_common_clock_arms(
                    permuted_samples[permutation],
                    means[index],
                    delta,
                    c=c,
                )
            plus_sum += plus
            minus_sum += minus
        plus_scores[index] = (
            alpha * plus_sum / (permuted_samples.shape[0] * u_plus)
        )
        minus_scores[index] = (
            alpha * minus_sum / (permuted_samples.shape[0] * u_minus)
        )
    return plus_scores, minus_scores


def draw_permuted_samples(X, permutation_count, rng):
    """Return independent uniform permutations of one observed sample."""
    X = np.asarray(X, dtype=float)
    if int(permutation_count) != permutation_count or permutation_count <= 0:
        raise ValueError("permutation_count must be a positive integer")
    output = np.empty((int(permutation_count), X.size), dtype=float)
    for index in range(int(permutation_count)):
        output[index] = X[rng.permutation(X.size)]
    return output


def _ordered_endpoints_from_scores(
    center,
    score_arms,
    tolerance=1e-8,
):
    """Invert monotone upper and lower arm scores by bisection."""
    cache = {}

    def arms(m):
        key = float(m)
        if key not in cache:
            plus, minus = score_arms(np.asarray([key], dtype=float))
            cache[key] = (float(plus[0] - 1.0), float(minus[0] - 1.0))
        return cache[key]

    if arms(center)[0] >= 0.0 or arms(center)[1] >= 0.0:
        return center, center, True

    if arms(0.0)[0] < 0.0:
        lower = 0.0
    else:
        rejected, accepted = 0.0, center
        while accepted - rejected > tolerance:
            midpoint = 0.5 * (rejected + accepted)
            if arms(midpoint)[0] >= 0.0:
                rejected = midpoint
            else:
                accepted = midpoint
        lower = 0.5 * (rejected + accepted)

    if arms(1.0)[1] < 0.0:
        upper = 1.0
    else:
        accepted, rejected = center, 1.0
        while rejected - accepted > tolerance:
            midpoint = 0.5 * (accepted + rejected)
            if arms(midpoint)[1] < 0.0:
                accepted = midpoint
            else:
                rejected = midpoint
        upper = 0.5 * (accepted + rejected)

    if lower > upper:
        return center, center, True
    return float(lower), float(upper), False


def permutation_integrated_ci_endpoints(
    permuted_samples,
    delta,
    *,
    retain_overshoot=False,
    randomizers=(1.0, 1.0),
    c=1.0,
):
    """Invert capped, permutation-averaged GE arms.

    Capped arms are paired with terminal uniforms.  The stopped-overshoot
    diagnostic is deliberately excluded here because its arms can be
    nonmonotone in the candidate mean; use audit_arm_monotonicity to examine
    that construction.
    """
    permuted_samples = np.asarray(permuted_samples, dtype=float)
    if permuted_samples.ndim != 2 or permuted_samples.shape[1] == 0:
        raise ValueError("permuted_samples must be a nonempty matrix")
    if retain_overshoot:
        raise ValueError(
            "stopped-overshoot arms are not guaranteed to be monotone and "
            "cannot be inverted with the ordered-endpoint routine"
        )
    u_plus, u_minus = (float(value) for value in randomizers)
    if not (0.0 < u_plus <= 1.0 and 0.0 < u_minus <= 1.0):
        raise ValueError("randomizers must lie in (0,1]")
    center = float(np.mean(permuted_samples[0]))

    def scores(means):
        return permutation_average_arm_scores(
            permuted_samples,
            np.asarray(means, dtype=float),
            delta,
            retain_overshoot,
            u_plus,
            u_minus,
            c,
        )

    return _ordered_endpoints_from_scores(center, scores)


def audit_arm_monotonicity(
    permuted_samples,
    delta,
    *,
    retain_overshoot,
    grid_size=257,
    tolerance=1e-10,
    c=1.0,
):
    """Check the expected upper/lower arm ordering on a dense mean grid."""
    means = np.linspace(0.0, 1.0, int(grid_size))
    plus, minus = permutation_average_arm_scores(
        np.asarray(permuted_samples, dtype=float),
        means,
        delta,
        retain_overshoot,
        1.0,
        1.0,
        c,
    )
    plus_violations = int(np.sum(np.diff(plus) > tolerance))
    minus_violations = int(np.sum(np.diff(minus) < -tolerance))
    return {
        "plus_violations": plus_violations,
        "minus_violations": minus_violations,
        "max_plus_increase": float(max(np.max(np.diff(plus)), 0.0)),
        "max_minus_decrease": float(max(-np.min(np.diff(minus)), 0.0)),
    }


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
        "standard_error": float(np.std(values, ddof=1) / np.sqrt(values.size)),
        "q10": float(np.quantile(values, 0.1)),
        "q50": float(np.quantile(values, 0.5)),
        "q90": float(np.quantile(values, 0.9)),
    }


def run_order_invariance_experiment(
    *,
    output_dir="plots/order_invariant_ge",
    delta=0.01,
    n_values=(50, 200, 1000),
    repetitions=40,
    permutation_count=24,
    seed=20260813,
):
    """Compare chronological and permutation-integrated GE intervals."""
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must lie in (0,1)")
    if int(repetitions) != repetitions or repetitions <= 0:
        raise ValueError("repetitions must be a positive integer")
    if int(permutation_count) != permutation_count or permutation_count <= 0:
        raise ValueError("permutation_count must be a positive integer")
    raw_n_values = tuple(n_values)
    if (
        not raw_n_values
        or any(int(n) != n for n in raw_n_values)
        or any(n <= 0 for n in raw_n_values)
    ):
        raise ValueError("n_values must contain positive integers")
    n_values = tuple(int(n) for n in raw_n_values)
    repetitions = int(repetitions)
    permutation_count = int(permutation_count)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    permutation_rng = np.random.default_rng(seed + 1)
    calibration_rng = np.random.default_rng(seed + 2)

    # Trigger compilation outside the timed loop.
    warm = draw_permuted_samples(np.linspace(0.1, 0.9, 12), 3, permutation_rng)
    permutation_average_arm_scores(
        warm,
        np.asarray([0.4, 0.5, 0.6]),
        delta,
        False,
        0.5,
        0.5,
    )
    permutation_average_arm_scores(
        warm,
        np.asarray([0.4, 0.5, 0.6]),
        delta,
        True,
        1.0,
        1.0,
    )

    rows = []
    topology_rows = []
    order_rows = []
    started = time.perf_counter()
    for distribution, (sample, sigma) in _samplers(rng).items():
        for n in n_values:
            print(f"{distribution}: n={n}, reps={repetitions}", flush=True)
            for repetition in range(repetitions):
                X = sample(int(n))
                permutations = draw_permuted_samples(
                    X, permutation_count, permutation_rng
                )
                u_plus, u_minus = calibration_rng.uniform(size=2)

                chronological = betting.probit_common_clock_ci_endpoints(
                    X,
                    delta,
                    randomizers=(u_plus, u_minus),
                    c=1.0,
                )
                integrated = permutation_integrated_ci_endpoints(
                    permutations,
                    delta,
                    retain_overshoot=False,
                    randomizers=(u_plus, u_minus),
                    c=1.0,
                )

                methods = {
                    METHOD_CHRONOLOGICAL: chronological,
                    METHOD_PERMUTATION_CAPPED: integrated,
                }
                if distribution == "Bernoulli(0.5)":
                    exact_lower, exact_upper = betting.bernoulli_dp_ci_endpoints(
                        X,
                        delta,
                        upper_randomizer=u_plus,
                        lower_randomizer=u_minus,
                    )
                    methods[METHOD_EXACT_BINOMIAL] = (
                        exact_lower,
                        exact_upper,
                        False,
                    )
                for method, endpoints in methods.items():
                    lower, upper, empty = endpoints[:3]
                    width = float(upper - lower)
                    rows.append({
                        "distribution": distribution,
                        "n": int(n),
                        "repetition": repetition,
                        "method": method,
                        "lower": float(lower),
                        "upper": float(upper),
                        "empty": bool(empty),
                        "width": width,
                        "scaled_width": float(np.sqrt(n) * width),
                        "normalized_width": float(
                            np.sqrt(n) * width
                            / (2.0 * sigma * betting.asymptotic_limit_digital(delta))
                        ),
                    })

                if repetition < min(10, repetitions):
                    audit = audit_arm_monotonicity(
                        permutations,
                        delta,
                        retain_overshoot=True,
                        grid_size=129,
                    )
                    topology_rows.append({
                        "distribution": distribution,
                        "n": int(n),
                        "repetition": repetition,
                        **audit,
                    })
                    for order_index in range(min(8, permutation_count)):
                        order_endpoints = (
                            betting.probit_common_clock_ci_endpoints(
                                permutations[order_index],
                                delta,
                                randomizers=(u_plus, u_minus),
                                c=1.0,
                            )
                        )
                        order_lower, order_upper = order_endpoints[:2]
                        order_rows.append({
                            "distribution": distribution,
                            "n": int(n),
                            "repetition": repetition,
                            "order_index": order_index,
                            "width": float(order_upper - order_lower),
                            "scaled_midpoint": float(
                                np.sqrt(n)
                                * (0.5 * (order_lower + order_upper) - np.mean(X))
                            ),
                        })

    elapsed = time.perf_counter() - started
    path_rows = output_dir / "path_widths.csv"
    with path_rows.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    path_topology = output_dir / "overshoot_monotonicity.csv"
    with path_topology.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=topology_rows[0].keys())
        writer.writeheader()
        writer.writerows(topology_rows)

    path_orders = output_dir / "order_sensitivity.csv"
    with path_orders.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=order_rows[0].keys())
        writer.writeheader()
        writer.writerows(order_rows)

    summaries = []
    for distribution in _samplers(np.random.default_rng(0)):
        for n in n_values:
            methods = [METHOD_CHRONOLOGICAL, METHOD_PERMUTATION_CAPPED]
            if distribution == "Bernoulli(0.5)":
                methods.append(METHOD_EXACT_BINOMIAL)
            for method in methods:
                selected = [
                    row for row in rows
                    if row["distribution"] == distribution
                    and row["n"] == n
                    and row["method"] == method
                ]
                summary = _summarize(
                    [row["normalized_width"] for row in selected]
                )
                summaries.append({
                    "distribution": distribution,
                    "n": int(n),
                    "method": method,
                    **summary,
                    "empty_rate": float(np.mean([row["empty"] for row in selected])),
                })

    summary_path = output_dir / "summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=summaries[0].keys())
        writer.writeheader()
        writer.writerows(summaries)

    paired_summaries = []
    for distribution in _samplers(np.random.default_rng(0)):
        for n in n_values:
            chronological_values = {
                row["repetition"]: row["normalized_width"]
                for row in rows
                if row["distribution"] == distribution
                and row["n"] == n
                and row["method"] == METHOD_CHRONOLOGICAL
            }
            comparators = [METHOD_PERMUTATION_CAPPED]
            if distribution == "Bernoulli(0.5)":
                comparators.append(METHOD_EXACT_BINOMIAL)
            for method in comparators:
                differences = np.asarray([
                    row["normalized_width"]
                    - chronological_values[row["repetition"]]
                    for row in rows
                    if row["distribution"] == distribution
                    and row["n"] == n
                    and row["method"] == method
                ])
                paired_summaries.append({
                    "distribution": distribution,
                    "n": int(n),
                    "method": method,
                    "mean_difference": float(np.mean(differences)),
                    "standard_error": float(
                        np.std(differences, ddof=1) / np.sqrt(differences.size)
                    ),
                    "win_rate": float(np.mean(differences < 0.0)),
                })

    paired_path = output_dir / "paired_summary.csv"
    with paired_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=paired_summaries[0].keys())
        writer.writeheader()
        writer.writerows(paired_summaries)

    order_summaries = []
    order_keys = sorted({
        (row["distribution"], row["n"], row["repetition"])
        for row in order_rows
    })
    for distribution, n, repetition in order_keys:
        selected = [
            row for row in order_rows
            if row["distribution"] == distribution
            and row["n"] == n
            and row["repetition"] == repetition
        ]
        widths = np.asarray([row["width"] for row in selected])
        midpoints = np.asarray([row["scaled_midpoint"] for row in selected])
        order_summaries.append({
            "distribution": distribution,
            "n": int(n),
            "repetition": repetition,
            "width_range": float(np.ptp(widths)),
            "width_sd": float(np.std(widths, ddof=1)),
            "scaled_midpoint_range": float(np.ptp(midpoints)),
            "scaled_midpoint_sd": float(np.std(midpoints, ddof=1)),
        })
    order_summary_path = output_dir / "order_sensitivity_summary.csv"
    with order_summary_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=order_summaries[0].keys())
        writer.writeheader()
        writer.writerows(order_summaries)

    metadata = {
        "delta": delta,
        "n_values": [int(n) for n in n_values],
        "repetitions": int(repetitions),
        "permutation_count": int(permutation_count),
        "seed": int(seed),
        "elapsed_seconds": elapsed,
        "validity_scope": (
            "permutation-integrated methods are finite-sample valid under iid "
            "sampling, including the sampled permutations and terminal uniforms"
        ),
    }
    with (output_dir / "config.json").open("w", encoding="utf-8") as output:
        json.dump(metadata, output, indent=2)

    _plot_results(summaries, paired_summaries, output_dir, delta)
    print(f"Saved results to {output_dir} in {elapsed:.1f} seconds")
    return rows, summaries, topology_rows


def _plot_results(summaries, paired_summaries, output_dir, delta):
    distributions = list(dict.fromkeys(row["distribution"] for row in summaries))
    methods = (
        (METHOD_CHRONOLOGICAL, "#2ca02c", "o", "-"),
        (METHOD_PERMUTATION_CAPPED, "#1f77b4", "s", "-"),
        (METHOD_EXACT_BINOMIAL, "black", "^", "--"),
    )
    fig, axes = plt.subplots(1, len(distributions), figsize=(13.2, 4.1), sharey=True)
    for axis, distribution in zip(np.atleast_1d(axes), distributions):
        for method, color, marker, linestyle in methods:
            selected = [
                row for row in summaries
                if row["distribution"] == distribution and row["method"] == method
            ]
            selected.sort(key=lambda row: row["n"])
            if not selected:
                continue
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
        axis.axhline(1.0, color="black", linestyle=":", linewidth=1.2)
        axis.set_xscale("log")
        axis.set_title(distribution)
        axis.set_xlabel("sample size")
        axis.grid(True, linestyle="--", alpha=0.3)
    axes[0].set_ylabel("mean width / Gaussian benchmark")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.suptitle(
        f"Order-invariant GE diagnostics ({1.0-delta:.1%} confidence)",
        y=1.04,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(
        output_dir / "order_invariant_ge_widths.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)

    fig, axes = plt.subplots(1, len(distributions), figsize=(13.2, 3.8), sharey=True)
    for axis, distribution in zip(np.atleast_1d(axes), distributions):
        selected = [
            row for row in paired_summaries
            if row["distribution"] == distribution
            and row["method"] == METHOD_PERMUTATION_CAPPED
        ]
        selected.sort(key=lambda row: row["n"])
        axis.errorbar(
            [row["n"] for row in selected],
            [row["mean_difference"] for row in selected],
            yerr=[1.96 * row["standard_error"] for row in selected],
            color="#1f77b4",
            marker="s",
            linewidth=1.8,
            capsize=2.5,
        )
        axis.axhline(0.0, color="black", linestyle=":", linewidth=1.2)
        axis.set_xscale("log")
        axis.set_title(distribution)
        axis.set_xlabel("sample size")
        axis.grid(True, linestyle="--", alpha=0.3)
    axes[0].set_ylabel("paired normalized-width difference")
    fig.suptitle("Permutation-integrated minus chronological GE", y=1.02)
    fig.tight_layout()
    fig.savefig(
        output_dir / "order_invariant_ge_paired_difference.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="plots/order_invariant_ge")
    parser.add_argument("--delta", type=float, default=0.01)
    parser.add_argument("--n-values", type=int, nargs="+", default=[50, 200, 1000])
    parser.add_argument("--repetitions", type=int, default=40)
    parser.add_argument("--permutations", type=int, default=24)
    parser.add_argument("--seed", type=int, default=20260813)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = _parse_args()
    run_order_invariance_experiment(
        output_dir=arguments.output_dir,
        delta=arguments.delta,
        n_values=tuple(arguments.n_values),
        repetitions=arguments.repetitions,
        permutation_count=arguments.permutations,
        seed=arguments.seed,
    )
