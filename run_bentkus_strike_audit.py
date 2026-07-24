#!/usr/bin/env python3
"""Paired audit of fixed strike choices in the Bentkus maturity mixture.

The audit deliberately uses only the public confidence-sequence interfaces.
All strike schedules share the same maturity horizons, mixture weights, cash
tail, observations, and inversion settings, so the comparison isolates the
fixed strike choice.  Strike choice does not affect e-process validity.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from betting import get_optimal_lambda
from confidence_sequences import (
    bentkus_horizon_schedule,
    bentkus_mixture_log_e_path,
    confidence_sequence_endpoints,
    product_scale_schedule,
)


_DISTRIBUTIONS = (
    "Beta(2,2)",
    "Beta(1,5)",
    "Bernoulli(0.5)",
    "Uniform(0,1)",
    "Beta(0.5,0.5)",
    "Bernoulli(0.1)",
)

_TRUE_MEANS = {
    "Beta(2,2)": 0.5,
    "Beta(1,5)": 1.0 / 6.0,
    "Bernoulli(0.5)": 0.5,
    "Uniform(0,1)": 0.5,
    "Beta(0.5,0.5)": 0.5,
    "Bernoulli(0.1)": 0.1,
}


def _sample(rng, name, size):
    if name == "Beta(2,2)":
        return rng.beta(2.0, 2.0, size)
    if name == "Beta(1,5)":
        return rng.beta(1.0, 5.0, size)
    if name == "Bernoulli(0.5)":
        return rng.binomial(1, 0.5, size).astype(np.float64)
    if name == "Uniform(0,1)":
        return rng.uniform(0.0, 1.0, size)
    if name == "Beta(0.5,0.5)":
        return rng.beta(0.5, 0.5, size)
    if name == "Bernoulli(0.1)":
        return rng.binomial(1, 0.1, size).astype(np.float64)
    raise ValueError(f"unknown distribution {name!r}")


def _summary(values):
    values = np.asarray(values, dtype=np.float64)
    if not len(values) or np.any(~np.isfinite(values)):
        raise ValueError("summaries require finite nonempty values")
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def _wilson(successes, total, z=1.959963984540054):
    probability = successes / float(total)
    z2 = z * z
    denominator = 1.0 + z2 / total
    center = (probability + z2 / (2.0 * total)) / denominator
    radius = z / denominator * math.sqrt(
        probability * (1.0 - probability) / total
        + z2 / (4.0 * total * total)
    )
    return {
        "crossings": int(successes),
        "total": int(total),
        "crossing_rate": float(probability),
        "wilson_lower": float(max(center - radius, 0.0)),
        "wilson_upper": float(min(center + radius, 1.0)),
    }


def _strike_schedules(delta, weights, default_strikes):
    schedules = {
        "default_delta_weight_over_2": np.asarray(
            default_strikes, dtype=np.float64
        ),
        "effective_delta_sqrt_weight_over_2": np.asarray(
            [
                get_optimal_lambda(delta * math.sqrt(weight / 2.0))[0]
                for weight in weights
            ],
            dtype=np.float64,
        ),
        "common_delta_over_2": np.full(
            len(weights), get_optimal_lambda(delta / 2.0)[0]
        ),
    }
    for multiplier in (0.5, 0.75, 1.25, 1.5, 2.0):
        schedules[f"default_times_{multiplier:g}"] = (
            multiplier * np.asarray(default_strikes, dtype=np.float64)
        )
    return schedules


def run_audit(
    *,
    delta=0.01,
    max_time=10_000,
    num_width_paths=3,
    num_crossing_paths=300,
    seed=20260719,
    topology_grid_size=65,
):
    horizons, default_strikes, weights, cash_weight = (
        bentkus_horizon_schedule(
            max_time,
            delta,
            horizon_ratio=2.0,
            weight_power=1.25,
            horizon_overshoot=2.0,
        )
    )
    schedules = _strike_schedules(delta, weights, default_strikes)
    configs = {
        label: {
            "horizons": horizons,
            "strikes": strikes,
            "weights": weights,
            "cash_weight": cash_weight,
        }
        for label, strikes in schedules.items()
    }
    scale_fractions, scale_weights, scale_cash = product_scale_schedule(
        max_time,
        weight_power=2.0,
        horizon_overshoot=2.0,
        scale_ratio=2.0,
    )
    comparator_configs = {
        "product_scale_mixture": {
            "fractions": scale_fractions,
            "weights": scale_weights,
            "cash_weight": scale_cash,
        },
        "agrapa": {"c": 0.5},
    }
    times = np.asarray([100, 1_000, max_time], dtype=np.int64)
    if max_time < 1_000:
        times = np.unique(
            np.maximum(
                1,
                np.rint(np.geomspace(1, max_time, 3)).astype(np.int64),
            )
        )

    # Compile once before generating or timing any audit observations.
    warm_x = np.asarray([0.0, 1.0, 0.25, 0.75])
    first_config = next(iter(configs.values()))
    bentkus_mixture_log_e_path(
        warm_x, 0.5, **first_config
    )
    confidence_sequence_endpoints(
        warm_x,
        delta,
        np.asarray([4]),
        "bentkus_mixture",
        method_config=first_config,
        topology_grid_size=5,
    )
    for method, config in comparator_configs.items():
        confidence_sequence_endpoints(
            warm_x,
            delta,
            np.asarray([4]),
            method,
            method_config=config,
            topology_grid_size=5,
        )

    seed_sequence = np.random.SeedSequence(seed)
    child_seeds = seed_sequence.spawn(2 * len(_DISTRIBUTIONS))
    results = {}
    pooled_crossings = {label: 0 for label in schedules}
    pooled_width_ratios = {label: [] for label in schedules}
    pooled_comparator_ratios = {
        diagnostic: {method: [] for method in comparator_configs}
        for diagnostic in ("fixed_1.5", "per_distribution_oracle")
    }
    default_label = "default_delta_weight_over_2"

    for distribution_index, distribution in enumerate(_DISTRIBUTIONS):
        print(f"widths: {distribution}", flush=True)
        width_rng = np.random.default_rng(child_seeds[2 * distribution_index])
        crossing_rng = np.random.default_rng(
            child_seeds[2 * distribution_index + 1]
        )
        width_paths = [
            _sample(width_rng, distribution, max_time)
            for _ in range(num_width_paths)
        ]
        widths = {
            label: np.empty((num_width_paths, len(times)))
            for label in schedules
        }
        empty = {
            label: np.zeros((num_width_paths, len(times)), dtype=bool)
            for label in schedules
        }
        uncertain = {
            label: np.zeros((num_width_paths, len(times)), dtype=bool)
            for label in schedules
        }
        comparator_widths = {
            method: np.empty((num_width_paths, len(times)))
            for method in comparator_configs
        }
        comparator_empty = {
            method: np.zeros((num_width_paths, len(times)), dtype=bool)
            for method in comparator_configs
        }
        for path_index, observations in enumerate(width_paths):
            for label, config in configs.items():
                inversion = confidence_sequence_endpoints(
                    observations,
                    delta,
                    times,
                    "bentkus_mixture",
                    method_config=config,
                    topology_grid_size=topology_grid_size,
                )
                widths[label][path_index] = inversion["width"]
                empty[label][path_index] = inversion["empty"]
                uncertain[label][path_index] = inversion[
                    "topology_uncertain"
                ]
            for method, config in comparator_configs.items():
                inversion = confidence_sequence_endpoints(
                    observations,
                    delta,
                    times,
                    method,
                    method_config=config,
                    topology_grid_size=topology_grid_size,
                )
                comparator_widths[method][path_index] = inversion["width"]
                comparator_empty[method][path_index] = inversion["empty"]

        print(f"crossings: {distribution}", flush=True)
        crossing_counts = {label: 0 for label in schedules}
        true_mean = _TRUE_MEANS[distribution]
        threshold = math.log(1.0 / delta)
        for _ in range(num_crossing_paths):
            observations = _sample(crossing_rng, distribution, max_time)
            for label, config in configs.items():
                log_e_path = bentkus_mixture_log_e_path(
                    observations, true_mean, **config
                )
                crossing_counts[label] += int(
                    float(np.max(log_e_path)) >= threshold
                )

        method_results = {}
        default_width = widths[default_label]
        for label in schedules:
            ratios = widths[label] / default_width
            if np.any(~np.isfinite(ratios)):
                raise ValueError("nonfinite paired width ratio")
            pooled_width_ratios[label].extend(ratios[:, -1].tolist())
            pooled_crossings[label] += crossing_counts[label]
            method_results[label] = {
                "mean_width_by_time": np.mean(
                    widths[label], axis=0
                ).tolist(),
                "terminal_width": _summary(widths[label][:, -1]),
                "width_ratio_to_default_by_time": [
                    _summary(ratios[:, time_index])
                    for time_index in range(len(times))
                ],
                "per_path_widths": widths[label].tolist(),
                "empty_rate_by_time": np.mean(
                    empty[label], axis=0
                ).tolist(),
                "topology_uncertain_rate_by_time": np.mean(
                    uncertain[label], axis=0
                ).tolist(),
                "true_mean_crossing": _wilson(
                    crossing_counts[label], num_crossing_paths
                ),
            }

        comparator_results = {}
        for method in comparator_configs:
            comparator_results[method] = {
                "mean_width_by_time": np.mean(
                    comparator_widths[method], axis=0
                ).tolist(),
                "terminal_width": _summary(
                    comparator_widths[method][:, -1]
                ),
                "per_path_widths": comparator_widths[method].tolist(),
                "empty_rate_by_time": np.mean(
                    comparator_empty[method], axis=0
                ).tolist(),
            }

        oracle_candidates = ("default_times_1.25", "default_times_1.5")
        oracle_label = min(
            oracle_candidates,
            key=lambda label: float(np.mean(widths[label][:, -1])),
        )
        diagnostic_widths = {
            "fixed_1.5": widths["default_times_1.5"],
            "per_distribution_oracle": widths[oracle_label],
        }
        product_comparison = {
            "oracle_selected_strike_schedule": oracle_label,
            "oracle_warning": (
                "Selected after seeing the three evaluation paths; this is "
                "an optimistic diagnostic ceiling, not an implementable "
                "data-independent strategy."
            ),
            "comparators": comparator_results,
            "diagnostics": {},
        }
        for diagnostic, diagnostic_width in diagnostic_widths.items():
            product_comparison["diagnostics"][diagnostic] = {}
            for method, comparator_width in comparator_widths.items():
                if np.any(comparator_width <= 0.0):
                    raise ValueError(
                        "zero comparator width prevents a paired ratio"
                    )
                ratio = diagnostic_width / comparator_width
                pooled_comparator_ratios[diagnostic][method].extend(
                    ratio[:, -1].tolist()
                )
                product_comparison["diagnostics"][diagnostic][method] = {
                    "width_ratio_by_time": [
                        _summary(ratio[:, time_index])
                        for time_index in range(len(times))
                    ],
                    "terminal_width_ratio": _summary(ratio[:, -1]),
                }

        results[distribution] = {
            "strike_variants": method_results,
            "product_comparison": product_comparison,
        }

    pooled_total = num_crossing_paths * len(_DISTRIBUTIONS)
    pooled = {
        label: {
            "terminal_width_ratio_to_default": _summary(
                pooled_width_ratios[label]
            ),
            "true_mean_crossing": _wilson(
                pooled_crossings[label], pooled_total
            ),
        }
        for label in schedules
    }
    pooled_product_comparison = {
        diagnostic: {
            method: {
                "terminal_width_ratio": _summary(values),
                "finite_pair_count": len(values),
            }
            for method, values in by_method.items()
        }
        for diagnostic, by_method in pooled_comparator_ratios.items()
    }
    return {
        "audit": "bentkus_fixed_strike_sensitivity",
        "delta": float(delta),
        "max_time": int(max_time),
        "times": times.tolist(),
        "num_width_paths_per_distribution": int(num_width_paths),
        "num_crossing_paths_per_distribution": int(num_crossing_paths),
        "seed": int(seed),
        "topology_grid_size": int(topology_grid_size),
        "maturity_ratio": 2.0,
        "weight_power": 1.25,
        "horizon_overshoot": 2.0,
        "horizons": horizons.tolist(),
        "weights": weights.tolist(),
        "cash_weight": float(cash_weight),
        "invested_weight": float(np.sum(weights)),
        "strike_schedules": {
            label: strikes.tolist() for label, strikes in schedules.items()
        },
        "comparator_methods": {
            "product_scale_mixture": {
                "weight_power": 2.0,
                "scale_ratio": 2.0,
                "horizon_overshoot": 2.0,
                "fractions": scale_fractions.tolist(),
                "weights": scale_weights.tolist(),
                "cash_weight": float(scale_cash),
            },
            "agrapa": {"c": 0.5},
        },
        "true_means": dict(_TRUE_MEANS),
        "results": results,
        "pooled": pooled,
        "pooled_product_comparison": pooled_product_comparison,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="plots/bentkus_strike_audit.json",
        help="output JSON path (default: %(default)s)",
    )
    parser.add_argument("--seed", type=int, default=20260719)
    parser.add_argument("--num-width-paths", type=int, default=3)
    parser.add_argument("--num-crossing-paths", type=int, default=300)
    parser.add_argument("--topology-grid-size", type=int, default=65)
    args = parser.parse_args()
    result = run_audit(
        seed=args.seed,
        num_width_paths=args.num_width_paths,
        num_crossing_paths=args.num_crossing_paths,
        topology_grid_size=args.topology_grid_size,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(f"wrote {output}", flush=True)


if __name__ == "__main__":
    main()
