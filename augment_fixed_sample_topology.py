#!/usr/bin/env python3
"""Add full-set diameter and largest-component widths to Figure 1 results.

The script exactly replays the data and terminal-randomizer streams used by
``betting.run_experiment``.  Candidate means are evaluated with the adaptive
multiresolution topology inverter: a standard-error-scale mesh, global and
geometric safety probes, a sample-size-aware verification mesh, and
exponential refinement whenever fragmentation is detected.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from numba import njit, prange

import betting


METHODS = (
    "heat_original",
    "heat_star",
    "product_original",
    "product_star",
    "probit_star_unbuffered",
    "probit_common_clock",
)


@njit(parallel=True)
def _scores(
    x,
    means,
    delta,
    strike,
    initial_wealth,
    kind,
    u_plus,
    u_minus,
):
    output = np.empty(means.size)
    alpha = delta / 2.0
    for index in prange(means.size):
        mean = means[index]
        if kind == 0:
            value = betting.compute_M_heat_path(
                x, mean, strike, initial_wealth
            )
        elif kind == 1:
            value = betting.compute_M_heat_star_path(
                x, mean, delta, initial_wealth
            )
        elif kind == 2:
            value = betting.compute_M_inf(x, mean, delta)
        elif kind == 3:
            value = betting.compute_M_star(x, mean, delta)
        elif kind == 4:
            plus, minus = betting.compute_M_probit_star_arms(
                x, mean, delta, buffer_rounds=0.0
            )
            value = max(
                alpha * plus / u_plus,
                alpha * minus / u_minus,
            )
        else:
            plus, minus = betting.compute_M_probit_common_clock_arms(
                x, mean, delta, buffer_rounds=0.0
            )
            value = max(
                alpha * plus / u_plus,
                alpha * minus / u_minus,
            )
        output[index] = value
    return output


def _samplers(rng):
    return {
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
        "Beta(50,50)": lambda n: rng.beta(50, 50, n),
        "Beta(20,80)": lambda n: rng.beta(20, 80, n),
        "Uniform(0.45,0.55)": lambda n: rng.uniform(0.45, 0.55, n),
    }


def _summary(values):
    values = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(values)),
        "lo": float(np.quantile(values, 0.1)),
        "hi": float(np.quantile(values, 0.9)),
    }


def _topology_summary(
    x,
    delta,
    strike,
    initial_wealth,
    kind,
    u_plus,
    u_minus,
    score_work_budget,
):
    if kind < 2:
        threshold = initial_wealth / delta
    elif kind < 4:
        threshold = 1.0 / delta
    else:
        threshold = 1.0

    def batch(means):
        return _scores(
            x,
            np.asarray(means),
            delta,
            strike,
            initial_wealth,
            kind,
            u_plus,
            u_minus,
        )

    small_sample = len(x) <= 1000
    effective_budget = (
        max(score_work_budget, 1_000_000)
        if small_sample else score_work_budget
    )
    parameters = betting._topology_scan_parameters(
        len(x),
        score_work_budget=effective_budget,
        minimum_verification_points=65 if small_sample else 9,
    )
    adaptive_options = {}
    if not small_sample:
        adaptive_options = {
            "local_radius": 4.0,
            "base_points_per_se": 1,
            "global_scan_points": 5,
            "geometric_tail_points": 4,
        }
    _, diagnostics = betting._adaptive_confidence_set_components(
        lambda mean: float(batch(np.asarray([mean]))[0]),
        threshold,
        float(np.mean(x)),
        betting._sample_standard_error(x),
        batch_statistic=batch,
        **adaptive_options,
        **parameters,
    )
    return diagnostics


def _common_clock_summary(x, delta, u_plus, u_minus):
    """Return exact topology diagnostics from the monotone arm boundaries."""
    lower, upper, empty, evaluations = (
        betting.probit_common_clock_ci_endpoints(
            x,
            delta,
            buffer_rounds=0.0,
            randomizers=(u_plus, u_minus),
            return_diagnostics=True,
        )
    )
    components = () if empty else ((lower, upper),)
    diagnostics = betting._confidence_set_widths(
        components, center=float(np.mean(x))
    )
    diagnostics.update({
        "evaluation_count": int(evaluations),
        "scan_point_count": 0,
        "refinement_levels": 0,
        "fragmentation_detected": False,
        "point_budget_reached": False,
        "final_mesh_resolution": 0.0,
        "standard_error": betting._sample_standard_error(x),
        "finite_mesh": False,
        "analytic_connectedness": True,
    })
    return diagnostics


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("plots/ci_width_original_vs_star.json"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("plots/ci_width_topology_checkpoint.json"),
    )
    parser.add_argument(
        "--score-work-budget",
        type=int,
        default=300_000,
        help="rough observation-by-candidate budget per method and dataset",
    )
    parser.add_argument("--large-reps", type=int, default=5)
    parser.add_argument("--medium-reps", type=int, default=10)
    parser.add_argument("--progress-every", type=int, default=10)
    return parser.parse_args()


def main():
    args = parse_args()
    with args.input.open(encoding="utf-8") as stream:
        payload = json.load(stream)

    delta = float(payload["delta"])
    seed = int(payload["seed"])
    n_values = [int(value) for value in payload["n_values"]]
    counts = {
        int(n): int(count)
        for n, count in payload["num_sims_by_n"].items()
    }
    strike = float(payload["strike"])
    initial_wealth = float(payload["initial_wealth"])

    completed = {}
    if args.checkpoint.exists():
        with args.checkpoint.open(encoding="utf-8") as stream:
            completed = json.load(stream).get("cells", {})

    rng = np.random.default_rng(seed)
    randomizer_seed = np.random.SeedSequence(seed).spawn(1)[0]
    regularized_rng = np.random.default_rng(randomizer_seed)
    unbuffered_rng = np.random.default_rng(randomizer_seed)
    rng.uniform(0.0, 1.0, 20)
    samplers = _samplers(rng)

    cell_count = 0
    topology_counts = {
        n: (
            counts[n]
            if n <= 1000
            else min(args.medium_reps, counts[n])
            if n <= 10_000
            else min(args.large_reps, counts[n])
        )
        for n in n_values
    }
    for n in n_values:
        for distribution, sampler in samplers.items():
            for replication in range(counts[n]):
                x = np.asarray(sampler(n), dtype=float)
                regularized_rng.uniform(size=2)
                u_plus, u_minus = (
                    float(value) for value in unbuffered_rng.uniform(size=2)
                )
                key = f"{distribution}|{n}|{replication}"
                if replication >= topology_counts[n]:
                    continue
                method_results = dict(completed.get(key, {}))
                for kind, method in enumerate(METHODS):
                    if method in method_results:
                        continue
                    if method == "probit_common_clock":
                        method_results[method] = _common_clock_summary(
                            x, delta, u_plus, u_minus
                        )
                    else:
                        method_results[method] = _topology_summary(
                            x,
                            delta,
                            strike,
                            initial_wealth,
                            kind,
                            u_plus,
                            u_minus,
                            args.score_work_budget,
                        )
                completed[key] = method_results
                cell_count += 1
                if cell_count % args.progress_every == 0:
                    print(
                        f"completed {cell_count} new cells: "
                        f"{distribution}, n={n}, rep={replication + 1}",
                        flush=True,
                    )
                if cell_count % args.progress_every == 0:
                    args.checkpoint.parent.mkdir(
                        parents=True, exist_ok=True
                    )
                    with args.checkpoint.open(
                        "w", encoding="utf-8"
                    ) as stream:
                        json.dump(
                            {
                                "score_work_budget": (
                                    args.score_work_budget
                                ),
                                "cells": completed,
                            },
                            stream,
                        )

    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    with args.checkpoint.open("w", encoding="utf-8") as stream:
        json.dump(
            {
                "score_work_budget": args.score_work_budget,
                "cells": completed,
            },
            stream,
        )

    for distribution in samplers:
        result = payload["results"][distribution]
        for method in METHODS:
            diameter_by_n = []
            largest_by_n = []
            diameter_raw_by_n = []
            largest_raw_by_n = []
            component_counts_by_n = []
            budget_rates = []
            for n in n_values:
                rows = [
                    completed[f"{distribution}|{n}|{replication}"][method]
                    for replication in range(topology_counts[n])
                ]
                diameter = np.asarray(
                    [row["hull_width"] for row in rows], dtype=float
                )
                largest = np.asarray(
                    [row["largest_component_width"] for row in rows],
                    dtype=float,
                )
                diameter_by_n.append(_summary(np.sqrt(n) * diameter))
                diameter_raw_by_n.append(_summary(diameter))
                largest_raw_by_n.append(_summary(largest))
                largest_by_n.append(_summary(np.sqrt(n) * largest))
                component_counts_by_n.append(
                    _summary([row["component_count"] for row in rows])
                )
                budget_rates.append(
                    float(np.mean(
                        [row["point_budget_reached"] for row in rows]
                    ))
                )
            result[f"{method}_diameter"] = diameter_by_n
            result[f"{method}_diameter_raw"] = diameter_raw_by_n
            result[f"{method}_largest_component_raw"] = largest_raw_by_n
            result[f"{method}_largest_component"] = largest_by_n
            result[f"{method}_component_count"] = component_counts_by_n
            result[f"{method}_topology_budget_rate"] = budget_rates

    payload["base_inversion_width"] = payload.get("reported_width")
    payload["reported_width"] = (
        "mesh-resolved full-set diameter; largest-component width also stored"
    )
    payload["topology_inversion"] = {
        "method": "adaptive multiresolution global finite-mesh inversion",
        "score_work_budget": args.score_work_budget,
        "topology_reps_by_n": {
            str(n): topology_counts[n] for n in n_values
        },
        "reported_widths": [
            "full-set diameter",
            "largest accepted-component width",
        ],
        "common_clock_inversion": (
            "exact interval from the two monotone one-sided arm boundaries; "
            "no discovery mesh"
        ),
        "finite_mesh_caveat": (
            "Two or more crossings contained in one final mesh cell can be "
            "missed; point-budget rates and component counts are retained."
        ),
        "checkpoint": str(args.checkpoint),
    }
    with args.input.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)
    print(f"updated {args.input}")


if __name__ == "__main__":
    main()
