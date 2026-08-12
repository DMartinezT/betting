#!/usr/bin/env python3
"""Add matched Markov-calibration widths to the fixed-sample figures.

The script replays the data and terminal-randomizer streams stored by
``betting.run_experiment`` and uses independently indexed streams when the
requested figure count exceeds the base count.  Every method is calibrated
armwise at level ``delta / 2``.  The deterministic regime uses thresholds
``u_plus=u_minus=1``;
the uniformly randomized Markov regime uses the same independent uniforms
for every method on a given dataset.  Candidate means without an analytic
ordering guarantee are evaluated with the adaptive multiresolution topology
inverter.
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
    "product_original",
    "product_star_common_clock",
    "probit_common_clock",
)

# These identifiers are consumed by the compiled score evaluator below.  Keep
# the mapping explicit: the plotted subset need not have the same order as the
# complete method list for which the evaluator was originally written.
SCORE_KIND_BY_METHOD = {
    "heat_original": 0,
    "product_original": 2,
}
CHECKPOINT_VERSION = 3

CALIBRATIONS = (
    "deterministic_markov",
    "randomized_markov",
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
    solvency_c=0.5,
):
    if kind < 0 or kind > 5:
        raise ValueError("unknown score kind")
    output = np.empty(means.size)
    alpha = delta / 2.0
    for index in prange(means.size):
        mean = means[index]
        if kind == 0:
            plus, minus = betting.compute_M_heat_path_arms(
                x, mean, strike, initial_wealth
            )
            scale = initial_wealth
        elif kind == 1:
            plus, minus = betting.compute_M_heat_star_arms(
                x, mean, delta, initial_wealth
            )
            scale = initial_wealth
        elif kind == 2:
            plus, minus = betting.compute_M_inf_arms(
                x, mean, delta, c=solvency_c
            )
            scale = 1.0
        elif kind == 3:
            plus, minus = betting.compute_M_star_arms(
                x, mean, delta, c=solvency_c
            )
            scale = 1.0
        elif kind == 4:
            plus, minus = betting.compute_M_star_common_clock_arms(
                x, mean, delta, c=solvency_c
            )
            scale = 1.0
        else:
            plus, minus = betting.compute_M_probit_common_clock_arms(
                x, mean, delta, c=solvency_c, buffer_rounds=0.0
            )
            scale = 1.0
        output[index] = max(
            alpha * plus / (scale * u_plus),
            alpha * minus / (scale * u_minus),
        )
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
    solvency_c,
):
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
            solvency_c,
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


def _star_common_clock_summary(
    x, delta, u_plus, u_minus, solvency_c
):
    """Return exact STaR diagnostics from its ordered arm boundaries."""
    lower, upper, empty, evaluations = (
        betting.star_common_clock_batched_ci_endpoints(
            x,
            delta,
            randomizers=(u_plus, u_minus),
            return_diagnostics=True,
            c=solvency_c,
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


def _common_clock_summary(
    x, delta, u_plus, u_minus, solvency_c
):
    """Return exact topology diagnostics from the monotone arm boundaries."""
    lower, upper, empty, evaluations = (
        betting.probit_common_clock_batched_ci_endpoints(
            x,
            delta,
            buffer_rounds=0.0,
            randomizers=(u_plus, u_minus),
            return_diagnostics=True,
            c=solvency_c,
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
        "--output",
        type=Path,
        help="write the augmented JSON here instead of overwriting --input",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("plots/ci_width_topology_checkpoint.json"),
    )
    parser.add_argument(
        "--score-work-budget",
        type=int,
        default=100_000,
        help="rough observation-by-candidate budget per method and dataset",
    )
    parser.add_argument(
        "--solvency-c",
        type=float,
        default=1.0,
        help=(
            "fraction of the one-step solvency limit used by STaR and "
            "efficient betting"
        ),
    )
    parser.add_argument(
        "--product-solvency-c",
        type=float,
        default=0.5,
        help="fraction of the one-step solvency limit used by product betting",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=METHODS,
        default=list(METHODS),
        help="methods to recompute; unselected series are retained from --input",
    )
    parser.add_argument("--small-reps", type=int, default=120)
    parser.add_argument("--large-reps", type=int, default=30)
    parser.add_argument("--medium-reps", type=int, default=60)
    parser.add_argument(
        "--max-n",
        type=int,
        help="restrict the augmented output to horizons at most this value",
    )
    parser.add_argument("--progress-every", type=int, default=10)
    return parser.parse_args()


def main():
    args = parse_args()
    with args.input.open(encoding="utf-8") as stream:
        payload = json.load(stream)
    output_path = args.output if args.output is not None else args.input
    selected_methods = tuple(dict.fromkeys(args.methods))

    delta = float(payload["delta"])
    seed = int(payload["seed"])
    n_values = [
        int(value)
        for value in payload["n_values"]
        if args.max_n is None or int(value) <= args.max_n
    ]
    if not n_values:
        raise ValueError("--max-n excludes every configured horizon")
    counts = {
        int(n): int(count)
        for n, count in payload["num_sims_by_n"].items()
    }
    strike = float(payload["strike"])
    initial_wealth = float(payload["initial_wealth"])

    completed = {}
    checkpoint_version = 0
    if args.checkpoint.exists():
        with args.checkpoint.open(encoding="utf-8") as stream:
            checkpoint_payload = json.load(stream)
        completed = checkpoint_payload.get("cells", {})
        checkpoint_version = int(checkpoint_payload.get("version", 0))
        checkpoint_solvency_c = float(
            checkpoint_payload.get("solvency_c", 1.0)
        )
        if not np.isclose(checkpoint_solvency_c, args.solvency_c):
            raise ValueError(
                f"checkpoint uses c={checkpoint_solvency_c}, but "
                f"--solvency-c={args.solvency_c}"
            )
        checkpoint_product_c = float(
            checkpoint_payload.get("product_solvency_c", 0.5)
        )
        if not np.isclose(checkpoint_product_c, args.product_solvency_c):
            raise ValueError(
                f"checkpoint uses product c={checkpoint_product_c}, but "
                f"--product-solvency-c={args.product_solvency_c}"
            )

    # Version 1 paired ``enumerate(METHODS)`` with numeric score codes.  Once
    # the plotted method list was shortened, this made product betting use
    # score kind 1 (squared-hinge STaR) instead of score kind 2 (the product
    # e-process).  Discard only those stale entries; all other cached results
    # remain valid.
    if checkpoint_version < CHECKPOINT_VERSION:
        stale_keys = {
            f"product_original|{calibration}"
            for calibration in CALIBRATIONS
        }
        invalidated = 0
        for method_results in completed.values():
            for result_key in stale_keys:
                invalidated += int(
                    method_results.pop(result_key, None) is not None
                )
        if invalidated:
            print(
                f"invalidated {invalidated} product-betting checkpoint entries "
                "computed with the obsolete positional dispatch",
                flush=True,
            )

    rng = np.random.default_rng(seed)
    randomizer_seed = np.random.SeedSequence(seed).spawn(1)[0]
    regularized_rng = np.random.default_rng(randomizer_seed)
    unbuffered_rng = np.random.default_rng(randomizer_seed)
    rng.uniform(0.0, 1.0, 20)
    samplers = _samplers(rng)

    cell_count = 0
    if min(args.small_reps, args.medium_reps, args.large_reps) <= 0:
        raise ValueError("replication counts must be positive")
    if not 0.0 < args.solvency_c <= 1.0:
        raise ValueError("--solvency-c must lie in (0,1]")
    if not 0.0 < args.product_solvency_c <= 1.0:
        raise ValueError("--product-solvency-c must lie in (0,1]")
    topology_counts = {
        n: (
            args.small_reps
            if n <= 1000
            else args.medium_reps
            if n <= 10_000
            else args.large_reps
        )
        for n in n_values
    }
    for n in n_values:
        for distribution_index, (distribution, sampler) in enumerate(
            samplers.items()
        ):
            for replication in range(max(counts[n], topology_counts[n])):
                if replication < counts[n]:
                    x = np.asarray(sampler(n), dtype=float)
                    regularized_rng.uniform(size=2)
                    u_plus, u_minus = (
                        float(value)
                        for value in unbuffered_rng.uniform(size=2)
                    )
                else:
                    extension_seed = np.random.SeedSequence(
                        [seed, 8675309, n, distribution_index, replication]
                    )
                    data_seed, auxiliary_seed = extension_seed.spawn(2)
                    extension_rng = np.random.default_rng(data_seed)
                    extension_sampler = _samplers(extension_rng)[distribution]
                    x = np.asarray(extension_sampler(n), dtype=float)
                    extension_auxiliary_rng = np.random.default_rng(
                        auxiliary_seed
                    )
                    u_plus, u_minus = (
                        float(value)
                        for value in extension_auxiliary_rng.uniform(size=2)
                    )
                key = f"{distribution}|{n}|{replication}"
                if replication >= topology_counts[n]:
                    continue
                method_results = dict(completed.get(key, {}))
                cell_updated = False
                for calibration in CALIBRATIONS:
                    calibration_uniforms = (
                        (1.0, 1.0)
                        if calibration == "deterministic_markov"
                        else (u_plus, u_minus)
                    )
                    for method in selected_methods:
                        result_key = f"{method}|{calibration}"
                        if result_key in method_results:
                            continue
                        cell_updated = True
                        method_solvency_c = (
                            args.product_solvency_c
                            if method == "product_original"
                            else args.solvency_c
                        )
                        if method == "probit_common_clock":
                            method_results[result_key] = _common_clock_summary(
                                x, delta, *calibration_uniforms,
                                solvency_c=args.solvency_c
                            )
                        elif method == "product_star_common_clock":
                            method_results[result_key] = (
                                _star_common_clock_summary(
                                    x, delta, *calibration_uniforms,
                                    solvency_c=args.solvency_c
                                )
                            )
                        else:
                            method_results[result_key] = _topology_summary(
                                x,
                                delta,
                                strike,
                                initial_wealth,
                                SCORE_KIND_BY_METHOD[method],
                                *calibration_uniforms,
                                args.score_work_budget,
                                method_solvency_c,
                            )
                if not cell_updated:
                    continue
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
                                "version": CHECKPOINT_VERSION,
                                "solvency_c": args.solvency_c,
                                "product_solvency_c": args.product_solvency_c,
                                "score_kind_by_method": SCORE_KIND_BY_METHOD,
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
                "version": CHECKPOINT_VERSION,
                "solvency_c": args.solvency_c,
                "product_solvency_c": args.product_solvency_c,
                "score_kind_by_method": SCORE_KIND_BY_METHOD,
                "score_work_budget": args.score_work_budget,
                "cells": completed,
            },
            stream,
        )

    for distribution in samplers:
        result = payload["results"][distribution]
        for calibration in CALIBRATIONS:
            for method in selected_methods:
                diameter_by_n = []
                largest_by_n = []
                diameter_raw_by_n = []
                largest_raw_by_n = []
                component_counts_by_n = []
                budget_rates = []
                result_key = f"{method}|{calibration}"
                for n in n_values:
                    rows = [
                        completed[f"{distribution}|{n}|{replication}"][result_key]
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
                prefix = f"{method}_{calibration}"
                result[f"{prefix}_diameter"] = diameter_by_n
                result[f"{prefix}_diameter_raw"] = diameter_raw_by_n
                result[f"{prefix}_largest_component_raw"] = largest_raw_by_n
                result[f"{prefix}_largest_component"] = largest_by_n
                result[f"{prefix}_component_count"] = component_counts_by_n
                result[f"{prefix}_topology_budget_rate"] = budget_rates

    payload["base_inversion_width"] = payload.get("reported_width")
    payload["n_values"] = n_values
    payload["num_sims_by_n"] = {
        str(n): counts[n] for n in n_values
    }
    payload["solvency_c"] = args.solvency_c
    payload["product_solvency_c"] = args.product_solvency_c
    payload["recomputed_methods"] = list(selected_methods)
    payload["reported_width"] = (
        "mesh-resolved full-set diameter; largest-component width also stored"
    )
    payload["topology_inversion"] = {
        "method": "matched armwise Markov-calibration comparison",
        "score_work_budget": args.score_work_budget,
        "base_replayed_reps_by_n": {
            str(n): min(counts[n], topology_counts[n]) for n in n_values
        },
        "extension_sampling": (
            "Replications beyond the base JSON count use independent "
            "SeedSequence streams indexed by horizon, distribution, and "
            "replication."
        ),
        "topology_reps_by_n": {
            str(n): topology_counts[n] for n in n_values
        },
        "reported_widths": [
            "full-set diameter",
            "largest accepted-component width",
        ],
        "calibrations": {
            "deterministic_markov": "u_plus = u_minus = 1",
            "randomized_markov": (
                "independent Uniform(0,1) thresholds, fixed over candidate means; "
                "the same pair is used by every betting method on a dataset"
            ),
            "arm_level": "delta / 2",
        },
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
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)
    print(f"updated {output_path}")


if __name__ == "__main__":
    main()
