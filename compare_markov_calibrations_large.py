#!/usr/bin/env python3
"""Paired large-sample comparison of Markov calibration regimes.

Every betting strategy is inverted armwise at level ``delta / 2``.  The
deterministic regime uses ``u_plus=u_minus=1``; the uniformly randomized
Markov regime uses one independent pair of uniforms per dataset, held fixed
over candidate means and shared by all betting strategies.  Gaffke is shown
both in its ordinary form and with the randomized product-orthant refinement
of Ming et al. (2026).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from numba import njit, prange
from scipy.stats import norm


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import betting  # noqa: E402
from gaffke_comparison.compare_star_probit_gaffke import DISTRIBUTIONS  # noqa: E402
from gaffke_comparison.large_sample_feedback_gaffke import (  # noqa: E402
    _is_binary_distribution,
    fast_gaffke_ci,
    invert_local_component,
)


SAMPLE_SIZES = (1_000, 10_000, 100_000, 1_000_000, 10_000_000)
BETTING_METHODS = (
    "Original STaR",
    "Squared-hinge STaR",
    "Efficient betting",
)
MAIN_BETTING_METHODS = (
    "Original STaR",
    "Efficient betting",
)
CALIBRATIONS = ("deterministic_markov", "randomized_markov")
METHOD_STYLES = {
    "Original STaR": ("darkorange", "P"),
    "Squared-hinge STaR": ("crimson", "D"),
    "Efficient betting": ("#2ca02c", "h"),
    "Gaffke": ("#1976b9", "o"),
    "Randomized Gaffke": ("#00a6d6", "x"),
}
METHOD_LABELS = {
    "Original STaR": "STaR betting",
    "Efficient betting": "GE-betting",
}


def _randomized_product_orthant_lower(
    x,
    tail_probability,
    uniform,
    deterministic_lower,
):
    """Invert the randomized Gaffke rule on its product orthant.

    For a lower-tail candidate ``theta <= min(x)``, all scaled e-values
    ``x_i / theta`` are at least one.  The randomized product-orthant
    p-value is then ``uniform * theta**n / prod(x)``.  Outside this region it
    equals the ordinary Gaffke p-value, so the endpoint changes only when the
    deterministic lower endpoint lies strictly below ``min(x)``.
    """
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        raise ValueError("x must be nonempty")
    if not (0.0 < tail_probability < 1.0):
        raise ValueError("tail_probability must lie in (0,1)")
    if not (0.0 < uniform <= 1.0):
        raise ValueError("uniform must lie in (0,1]")

    sample_minimum = float(np.min(x))
    deterministic_lower = float(deterministic_lower)
    if sample_minimum <= 0.0 or deterministic_lower >= sample_minimum:
        return deterministic_lower

    log_candidate = (
        math.log(tail_probability)
        - math.log(uniform)
        + float(np.sum(np.log(x)))
    ) / x.size
    candidate = math.exp(log_candidate)
    return max(
        deterministic_lower,
        min(sample_minimum, candidate),
    )


def randomized_product_orthant_gaffke_ci(
    x,
    delta,
    deterministic_interval,
    randomizers,
):
    """Invert the randomized product-orthant Gaffke p-values.

    One independent uniform is used for each equal-tail arm and held fixed
    throughout inversion.  The upper endpoint is obtained by applying the
    lower-endpoint formula to the reflected sample ``1-x``.
    """
    x = np.asarray(x, dtype=float)
    lower, upper = (float(value) for value in deterministic_interval)
    u_plus, u_minus = (float(value) for value in randomizers)
    tail_probability = delta / 2.0
    randomized_lower = _randomized_product_orthant_lower(
        x,
        tail_probability,
        u_plus,
        lower,
    )
    reflected_lower = _randomized_product_orthant_lower(
        1.0 - x,
        tail_probability,
        u_minus,
        1.0 - upper,
    )
    randomized_upper = 1.0 - reflected_lower
    return randomized_lower, randomized_upper


@njit(parallel=True)
def _feedback_scores(x, means, delta, feedback_kind, u_plus, u_minus):
    """Armwise calibrated rejection scores for one feedback strategy."""
    output = np.empty(means.size)
    alpha = delta / 2.0
    for index in prange(means.size):
        plus, minus = betting.compute_M_recalculating_feedback_arms(
            x, means[index], delta, feedback_kind
        )
        output[index] = max(
            alpha * plus / u_plus,
            alpha * minus / u_minus,
        ) - 1.0
    return output


def _feedback_interval(x, delta, feedback_kind, randomizers):
    u_plus, u_minus = randomizers

    def batch(means):
        return _feedback_scores(
            x,
            np.asarray(means, dtype=float),
            delta,
            feedback_kind,
            u_plus,
            u_minus,
        )

    return invert_local_component(
        x,
        lambda mean: float(batch(np.asarray([mean]))[0]),
        batch,
    )


def _outside_grid(lower, upper, standard_error):
    candidates = [0.0, 1.0]
    first = max(standard_error / 8.0, 1.0e-12)
    for boundary, endpoint, direction in (
        (lower, 0.0, -1.0),
        (upper, 1.0, 1.0),
    ):
        distance = first
        while distance < abs(endpoint - boundary):
            candidates.append(boundary + direction * distance)
            distance *= 2.0
    grid = np.unique(np.clip(candidates, 0.0, 1.0))
    return grid[(grid < lower) | (grid > upper)]


def _topology_checked_feedback_interval(
    x,
    delta,
    feedback_kind,
    randomizers,
    score_work_budget,
):
    lower, upper, empty, evaluations = _feedback_interval(
        x, delta, feedback_kind, randomizers
    )
    if empty:
        return lower, upper, True, evaluations, False

    u_plus, u_minus = randomizers

    def batch(means):
        return _feedback_scores(
            x,
            np.asarray(means, dtype=float),
            delta,
            feedback_kind,
            u_plus,
            u_minus,
        )

    standard_error = betting._sample_standard_error(x)
    interior = np.linspace(lower, upper, 33)[1:-1]
    outside = _outside_grid(lower, upper, standard_error)
    screen = np.unique(np.r_[interior, outside])
    accepted = batch(screen) < 0.0
    outside_mask = (screen < lower) | (screen > upper)
    fragmentation = bool(
        np.any(~accepted[~outside_mask]) or np.any(accepted[outside_mask])
    )
    evaluations += int(screen.size)
    if not fragmentation:
        return lower, upper, False, evaluations, False

    parameters = betting._topology_scan_parameters(
        len(x),
        score_work_budget=score_work_budget,
        minimum_verification_points=9,
        maximum_verification_points=2049,
    )
    components, diagnostics = betting._adaptive_confidence_set_components(
        lambda mean: float(batch(np.asarray([mean]))[0]),
        0.0,
        float(np.mean(x)),
        standard_error,
        batch_statistic=batch,
        local_radius=6.0,
        base_points_per_se=1,
        global_scan_points=9,
        geometric_tail_points=6,
        boundary_tolerance=max(1.0e-8, standard_error * 1.0e-4),
        **parameters,
    )
    if not components:
        center = float(np.mean(x))
        return center, center, True, evaluations, True
    return (
        float(components[0][0]),
        float(components[-1][1]),
        False,
        evaluations + int(diagnostics["evaluation_count"]),
        True,
    )


def _record(distribution, n, rep, method, calibration, lower, upper, empty,
            seconds, evaluations, escalated):
    width = max(float(upper - lower), 0.0)
    return {
        "distribution": distribution.name,
        "true_mean": distribution.mean,
        "true_variance": distribution.variance,
        "n": int(n),
        "rep": int(rep),
        "method": method,
        "calibration": calibration,
        "lower": float(lower),
        "upper": float(upper),
        "width": width,
        "sqrt_n_width": math.sqrt(n) * width,
        "covered": bool(lower <= distribution.mean <= upper),
        "empty": bool(empty),
        "runtime_seconds": float(seconds),
        "score_evaluations": int(evaluations),
        "topology_escalated": bool(escalated),
    }


def _summarize(frame):
    return (
        frame.groupby(
            ["distribution", "n", "method", "calibration"],
            as_index=False,
        )
        .agg(
            replications=("width", "size"),
            mean_sqrt_n_width=("sqrt_n_width", "mean"),
            q10_sqrt_n_width=(
                "sqrt_n_width", lambda values: np.quantile(values, 0.10)
            ),
            q90_sqrt_n_width=(
                "sqrt_n_width", lambda values: np.quantile(values, 0.90)
            ),
            coverage=("covered", "mean"),
            empty_rate=("empty", "mean"),
            escalation_rate=("topology_escalated", "mean"),
        )
    )


def _plot(frame, output, delta):
    summary = _summarize(frame)
    fig, axes = plt.subplots(3, 3, figsize=(13, 11), sharex=True)
    for axis, distribution in zip(axes.ravel(), DISTRIBUTIONS):
        ddf = summary[summary["distribution"] == distribution.name]
        n_values = np.sort(ddf["n"].unique())
        gaussian = 2.0 * math.sqrt(distribution.variance) * norm.isf(
            delta / 2.0
        )
        axis.plot(
            n_values,
            np.full(n_values.size, gaussian),
            color="black",
            ls=":",
            lw=1.7,
            label="Gaussian limit",
        )
        for method in MAIN_BETTING_METHODS:
            color, marker = METHOD_STYLES[method]
            for calibration, linestyle, current_marker, fill in (
                ("deterministic_markov", "--", None, False),
                ("randomized_markov", "-", marker, True),
            ):
                mdf = ddf[
                    (ddf["method"] == method)
                    & (ddf["calibration"] == calibration)
                ].sort_values("n")
                axis.plot(
                    mdf["n"],
                    mdf["mean_sqrt_n_width"],
                    color=color,
                    ls=linestyle,
                    marker=current_marker,
                    ms=4.7,
                    lw=2.0,
                    label=(
                        METHOD_LABELS[method]
                        if calibration == "randomized_markov"
                        else "_nolegend_"
                    ),
                )
                if fill:
                    axis.fill_between(
                        mdf["n"],
                        mdf["q10_sqrt_n_width"],
                        mdf["q90_sqrt_n_width"],
                        color=color,
                        alpha=0.08,
                        linewidth=0.0,
                    )
        gdf = ddf[ddf["method"] == "Gaffke"].sort_values("n")
        axis.plot(
            gdf["n"],
            gdf["mean_sqrt_n_width"],
            color=METHOD_STYLES["Gaffke"][0],
            marker=METHOD_STYLES["Gaffke"][1],
            lw=2.0,
            ms=4.7,
            label="Gaffke (ordinary)",
        )
        rgdf = ddf[ddf["method"] == "Randomized Gaffke"].sort_values("n")
        axis.plot(
            rgdf["n"],
            rgdf["mean_sqrt_n_width"],
            color=METHOD_STYLES["Randomized Gaffke"][0],
            marker=METHOD_STYLES["Randomized Gaffke"][1],
            ls="-.",
            lw=1.7,
            ms=4.7,
            label="Gaffke (randomized product orthant)",
        )
        axis.set_xscale("log")
        axis.set_title(distribution.name)
        axis.set_xlabel("sample size $n$")
        axis.set_ylabel(r"$\sqrt{n}\times$ CI width")
        axis.grid(True, ls="--", alpha=0.3)

    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    handles.extend([
        Line2D([0], [0], color="0.25", ls="--", lw=2),
        Line2D([0], [0], color="0.25", ls="-", marker="o", lw=2),
    ])
    labels.extend([
        "deterministic Markov",
        "uniformly randomized Markov",
    ])
    fig.suptitle(
        "Large-sample matched Markov-calibration comparison",
        fontsize=15,
    )
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=4,
        fontsize=8.8,
        frameon=False,
    )
    fig.tight_layout(rect=(0.0, 0.09, 1.0, 0.96))
    destination = output / "scaled_width_large_markov_calibrations.png"
    fig.savefig(destination, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return destination


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("gaffke_comparison/markov_calibration_results"),
    )
    parser.add_argument("--delta", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--score-work-budget", type=int, default=200_000)
    parser.add_argument("--progress-every", type=int, default=3)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output / "results_checkpoint.csv"
    rows = []
    completed = set()
    if args.resume and checkpoint.exists():
        previous = pd.read_csv(checkpoint)
        rows = previous.to_dict("records")
        completed = {
            (row["distribution"], int(row["n"]), int(row["rep"]),
             row["method"], row["calibration"])
            for row in rows
        }
    row_by_key = {
        (row["distribution"], int(row["n"]), int(row["rep"]),
         row["method"], row["calibration"]): row
        for row in rows
    }

    warm = np.asarray([0.2, 0.8], dtype=float)
    _feedback_scores(warm, np.asarray([0.5]), args.delta, 0, 1.0, 1.0)
    betting.probit_common_clock_batched_ci_endpoints(
        warm, args.delta, randomizers=(1.0, 1.0)
    )

    start = time.time()
    cells = 0
    for distribution_index, distribution in enumerate(DISTRIBUTIONS):
        print(f"\n=== {distribution.name} ===", flush=True)
        for rep in range(args.reps):
            seed_sequence = np.random.SeedSequence(
                [args.seed, distribution_index, rep]
            )
            data_seed, auxiliary_seed = seed_sequence.spawn(2)
            data_rng = np.random.default_rng(data_seed)
            auxiliary_rng = np.random.default_rng(auxiliary_seed)
            path = np.asarray(
                distribution.sampler(data_rng, max(SAMPLE_SIZES)),
                dtype=float,
            )
            for n in SAMPLE_SIZES:
                x = np.ascontiguousarray(path[:n])
                random_uniforms = tuple(
                    float(value) for value in auxiliary_rng.uniform(size=2)
                )
                for calibration in CALIBRATIONS:
                    uniforms = (
                        (1.0, 1.0)
                        if calibration == "deterministic_markov"
                        else random_uniforms
                    )
                    for method, kind in (
                        ("Original STaR", 0),
                        ("Squared-hinge STaR", 1),
                    ):
                        key = (distribution.name, n, rep, method, calibration)
                        if key in completed:
                            continue
                        t0 = time.perf_counter()
                        result = _topology_checked_feedback_interval(
                            x,
                            args.delta,
                            kind,
                            uniforms,
                            args.score_work_budget,
                        )
                        lower, upper, empty, evaluations, escalated = result
                        rows.append(_record(
                            distribution, n, rep, method, calibration,
                            lower, upper, empty,
                            time.perf_counter() - t0,
                            evaluations, escalated,
                        ))
                        completed.add(key)

                    method = "Efficient betting"
                    key = (distribution.name, n, rep, method, calibration)
                    if key not in completed:
                        t0 = time.perf_counter()
                        lower, upper, empty, evaluations = (
                            betting.probit_common_clock_batched_ci_endpoints(
                                x,
                                args.delta,
                                buffer_rounds=0.0,
                                randomizers=uniforms,
                                return_diagnostics=True,
                            )
                        )
                        rows.append(_record(
                            distribution, n, rep, method, calibration,
                            lower, upper, empty,
                            time.perf_counter() - t0,
                            evaluations, False,
                        ))
                        completed.add(key)

                method = "Gaffke"
                calibration = "not_applicable"
                key = (distribution.name, n, rep, method, calibration)
                if key not in completed:
                    t0 = time.perf_counter()
                    lower, upper, backend = fast_gaffke_ci(
                        x,
                        args.delta,
                        binary=_is_binary_distribution(distribution.name),
                        exact_cutoff=3_000,
                    )
                    rows.append(_record(
                        distribution, n, rep, method, calibration,
                        lower, upper, False,
                        time.perf_counter() - t0,
                        0, False,
                    ))
                    rows[-1]["backend"] = backend
                    completed.add(key)
                    row_by_key[key] = rows[-1]

                ordinary_gaffke = row_by_key[key]
                method = "Randomized Gaffke"
                calibration = "product_orthant_randomized"
                randomized_key = (
                    distribution.name, n, rep, method, calibration
                )
                if randomized_key not in completed:
                    t0 = time.perf_counter()
                    lower, upper = randomized_product_orthant_gaffke_ci(
                        x,
                        args.delta,
                        (
                            ordinary_gaffke["lower"],
                            ordinary_gaffke["upper"],
                        ),
                        random_uniforms,
                    )
                    rows.append(_record(
                        distribution, n, rep, method, calibration,
                        lower, upper, False,
                        time.perf_counter() - t0,
                        0, False,
                    ))
                    rows[-1]["backend"] = (
                        "randomized-product-orthant-"
                        + str(ordinary_gaffke.get("backend", "gaffke"))
                    )
                    completed.add(randomized_key)
                    row_by_key[randomized_key] = rows[-1]

                cells += 1
                if cells % args.progress_every == 0:
                    print(
                        f"n={n:>10,d} rep={rep + 1}/{args.reps} "
                        f"cells={cells} elapsed={(time.time() - start) / 60:.1f} min",
                        flush=True,
                    )
                    pd.DataFrame(rows).to_csv(checkpoint, index=False)

    frame = pd.DataFrame(rows)
    frame.to_csv(args.output / "results.csv", index=False)
    frame.to_csv(checkpoint, index=False)
    _summarize(frame).to_csv(args.output / "summary.csv", index=False)
    config = {
        "delta": args.delta,
        "seed": args.seed,
        "sample_sizes": list(SAMPLE_SIZES),
        "replications": args.reps,
        "betting_methods": list(BETTING_METHODS),
        "calibrations": {
            "deterministic_markov": "u_plus = u_minus = 1",
            "randomized_markov": (
                "shared independent Uniform(0,1) pair per dataset, fixed over m"
            ),
            "arm_level": "delta / 2",
        },
        "gaffke": (
            "ordinary equal-tail interval shown as the deterministic baseline"
        ),
        "randomized_gaffke": (
            "Ming--Ramdas--Shen--Wang--Waudby-Smith randomized product-orthant "
            "p-value, inverted with the shared armwise uniform pair"
        ),
        "topology": (
            "local bisection plus 31 interior checks and exponentially spaced "
            "outside probes; adaptive global inversion after any anomaly"
        ),
    }
    with (args.output / "config.json").open("w", encoding="utf-8") as stream:
        json.dump(config, stream, indent=2)
    destination = _plot(frame, args.output, args.delta)
    print(f"saved {destination}")
    return frame


if __name__ == "__main__":
    main()
