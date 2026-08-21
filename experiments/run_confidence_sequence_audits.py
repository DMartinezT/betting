#!/usr/bin/env python3
"""Run reproducible numerical audits for confidence-sequence experiments.

Two audits are available.

``topology``
    Re-invert all five confidence sequences on the same paths using several
    candidate-mean grids.  The default comparison uses grids 33, 65, and 129
    through time 10,000 and records the complete endpoint diagnostics.

``sensitivity``
    Compare the Bentkus maturity mixture and heat-constrained aGRAPA under
    several fixed maturity schedules.  Width paths and true-mean crossing
    indicators are paired across schedules by reusing the same chronological
    observations.

The output files use strict JSON: nonfinite numerical diagnostics (notably the
endpoints of an empty confidence set) are represented by ``null`` rather than
the nonstandard tokens ``NaN`` or ``Infinity``.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from confidence_sequences import (
    bentkus_horizon_schedule,
    bentkus_mixture_log_e_path,
    confidence_sequence_endpoints,
    default_cs_times,
    heat_constrained_agrapa_log_e_path,
    product_scale_schedule,
)


HERE = Path(__file__).resolve().parents[1]
OUTPUT_DIRECTORY = HERE / "plots"

DISTRIBUTIONS = (
    "Beta(2,2)",
    "Beta(1,5)",
    "Bernoulli(0.5)",
    "Uniform(0,1)",
    "Beta(0.5,0.5)",
    "Bernoulli(0.1)",
)

TRUE_MEANS = {
    "Beta(2,2)": 0.5,
    "Beta(1,5)": 1.0 / 6.0,
    "Bernoulli(0.5)": 0.5,
    "Uniform(0,1)": 0.5,
    "Beta(0.5,0.5)": 0.5,
    "Bernoulli(0.1)": 0.1,
}

METHODS = (
    "hgkelly",
    "product_scale_mixture",
    "agrapa",
    "bentkus_mixture",
    "heat_constrained_agrapa",
)

HEAT_METHODS = (
    "bentkus_mixture",
    "heat_constrained_agrapa",
)

SENSITIVITY_CONFIGURATIONS = (
    ("ratio2_p2_default", 2.0, 2.0),
    ("ratio2_p1p25", 2.0, 1.25),
    ("ratio2_p1p5", 2.0, 1.5),
    ("ratio2_p3", 2.0, 3.0),
    ("ratio1p5_p2", 1.5, 2.0),
    ("ratio3_p2", 3.0, 2.0),
)


def _sample_distribution(rng: np.random.Generator, name: str, size: int):
    """Draw one path without relying on private benchmark helpers."""
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


def _path_rng(seed: int, audit_code: int, distribution: int, path: int):
    words = [int(seed), int(audit_code), int(distribution), int(path)]
    return np.random.default_rng(np.random.SeedSequence(words)), words


def _serializable(value):
    """Recursively convert NumPy values and nonfinite floats for strict JSON."""
    if isinstance(value, dict):
        return {str(key): _serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serializable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _serializable(value.tolist())
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    return value


def _write_strict_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(
            _serializable(payload),
            handle,
            indent=2,
            sort_keys=False,
            allow_nan=False,
        )
        handle.write("\n")
    temporary.replace(path)


def _heat_schedule_dict(
    max_time: int,
    delta: float,
    horizon_ratio: float,
    weight_power: float,
    horizon_overshoot: float,
):
    horizons, strikes, weights, cash_weight = bentkus_horizon_schedule(
        max_time,
        delta,
        horizon_ratio=horizon_ratio,
        weight_power=weight_power,
        horizon_overshoot=horizon_overshoot,
    )
    return {
        "horizons": horizons,
        "strikes": strikes,
        "weights": weights,
        "cash_weight": float(cash_weight),
    }


def _schedule_record(config: dict) -> dict:
    return {
        "horizons": config["horizons"].tolist(),
        "strikes": config["strikes"].tolist(),
        "weights": config["weights"].tolist(),
        "cash_weight": float(config["cash_weight"]),
        "expert_count": int(len(config["horizons"])),
        "total_instantiated_weight": float(np.sum(config["weights"])),
    }


def _topology_method_configs(
    max_time: int,
    delta: float,
    product_grid_size: int,
    horizon_ratio: float,
    weight_power: float,
    horizon_overshoot: float,
):
    heat = _heat_schedule_dict(
        max_time,
        delta,
        horizon_ratio,
        weight_power,
        horizon_overshoot,
    )
    fractions, scale_weights, scale_cash = product_scale_schedule(
        max_time,
        weight_power=weight_power,
        horizon_overshoot=horizon_overshoot,
        scale_ratio=horizon_ratio,
    )
    scale = {
        "fractions": fractions,
        "weights": scale_weights,
        "cash_weight": float(scale_cash),
    }
    methods = {
        "hgkelly": {"G": int(product_grid_size)},
        "product_scale_mixture": scale,
        "agrapa": {"c": 0.5},
        "bentkus_mixture": heat,
        "heat_constrained_agrapa": {
            **heat,
            "agrapa_c": 0.5,
            "solvency_fraction": 1.0,
        },
    }
    schedules = {
        "bentkus": _schedule_record(heat),
        "product_scale": {
            "fractions": fractions.tolist(),
            "weights": scale_weights.tolist(),
            "cash_weight": float(scale_cash),
            "scale_count": int(len(fractions)),
            "total_instantiated_weight": float(np.sum(scale_weights)),
        },
    }
    return methods, schedules


def _raw_endpoint_record(inverted: dict, path_index: int, seed_words: list[int]):
    return {
        "path_index": int(path_index),
        "seed_words": list(seed_words),
        "lower": inverted["lower"].tolist(),
        "upper": inverted["upper"].tolist(),
        "width": inverted["width"].tolist(),
        "empty": inverted["empty"].tolist(),
        "component_count": inverted["component_count"].tolist(),
        "accepted_grid_count": inverted["accepted_grid_count"].tolist(),
        "topology_uncertain": inverted["topology_uncertain"].tolist(),
    }


def _max_finite_absolute_difference(left, right):
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    finite = np.isfinite(left) & np.isfinite(right)
    if not np.any(finite):
        return None, 0
    return float(np.max(np.abs(left[finite] - right[finite]))), int(
        np.sum(finite)
    )


def _resolution_comparison(low_records: list[dict], high_records: list[dict]):
    per_path = []
    pooled = {"lower": [], "upper": [], "width": []}
    empty_mismatches = 0
    component_mismatches = 0
    topology_mismatches = 0
    total_points = 0

    for low, high in zip(low_records, high_records):
        row = {"path_index": int(low["path_index"])}
        for field in pooled:
            low_values = np.asarray(low[field], dtype=np.float64)
            high_values = np.asarray(high[field], dtype=np.float64)
            maximum, count = _max_finite_absolute_difference(
                low_values, high_values
            )
            row[f"max_{field}_absolute_difference"] = maximum
            row[f"finite_{field}_comparison_count"] = count
            pooled[field].append((low_values, high_values))

        low_empty = np.asarray(low["empty"], dtype=bool)
        high_empty = np.asarray(high["empty"], dtype=bool)
        low_components = np.asarray(low["component_count"], dtype=np.int64)
        high_components = np.asarray(high["component_count"], dtype=np.int64)
        low_uncertain = np.asarray(low["topology_uncertain"], dtype=bool)
        high_uncertain = np.asarray(high["topology_uncertain"], dtype=bool)
        row["empty_flag_mismatch_count"] = int(
            np.sum(low_empty != high_empty)
        )
        row["component_count_mismatch_count"] = int(
            np.sum(low_components != high_components)
        )
        row["topology_flag_mismatch_count"] = int(
            np.sum(low_uncertain != high_uncertain)
        )
        empty_mismatches += row["empty_flag_mismatch_count"]
        component_mismatches += row["component_count_mismatch_count"]
        topology_mismatches += row["topology_flag_mismatch_count"]
        total_points += len(low_empty)
        per_path.append(row)

    aggregate = {}
    for field, pairs in pooled.items():
        left = np.concatenate([pair[0] for pair in pairs])
        right = np.concatenate([pair[1] for pair in pairs])
        maximum, count = _max_finite_absolute_difference(left, right)
        aggregate[f"max_{field}_absolute_difference"] = maximum
        aggregate[f"finite_{field}_comparison_count"] = count
    endpoint_values = [
        aggregate["max_lower_absolute_difference"],
        aggregate["max_upper_absolute_difference"],
    ]
    endpoint_values = [value for value in endpoint_values if value is not None]
    aggregate["max_endpoint_absolute_difference"] = (
        max(endpoint_values) if endpoint_values else None
    )
    aggregate.update(
        {
            "empty_flag_mismatch_count": int(empty_mismatches),
            "component_count_mismatch_count": int(component_mismatches),
            "topology_flag_mismatch_count": int(topology_mismatches),
            "total_path_time_points": int(total_points),
        }
    )
    return {"aggregate": aggregate, "per_path": per_path}


def _diagnostic_summary(records: list[dict]) -> dict:
    empty = np.asarray([row["empty"] for row in records], dtype=bool)
    components = np.asarray(
        [row["component_count"] for row in records], dtype=np.int64
    )
    uncertain = np.asarray(
        [row["topology_uncertain"] for row in records], dtype=bool
    )
    return {
        "path_time_count": int(empty.size),
        "empty_count": int(np.sum(empty)),
        "disconnected_count": int(np.sum(components > 1)),
        "topology_uncertain_count": int(np.sum(uncertain)),
        "maximum_component_count": int(np.max(components)),
    }


def _warm_topology(method_configs: dict, delta: float) -> None:
    observations = np.asarray([0.0, 1.0, 0.25, 0.75], dtype=np.float64)
    for method, config in method_configs.items():
        confidence_sequence_endpoints(
            observations,
            delta,
            np.asarray([4], dtype=np.int64),
            method,
            method_config=config,
            topology_grid_size=5,
        )


def run_topology_audit(
    *,
    delta: float,
    max_time: int,
    num_paths: int,
    seed: int,
    grid_sizes: tuple[int, ...],
    comparison_grids: tuple[int, int],
    time_count: int,
    product_grid_size: int,
    horizon_ratio: float,
    weight_power: float,
    horizon_overshoot: float,
    progress: bool,
):
    times = default_cs_times(max_time, count=time_count)
    method_configs, schedules = _topology_method_configs(
        max_time,
        delta,
        product_grid_size,
        horizon_ratio,
        weight_power,
        horizon_overshoot,
    )
    _warm_topology(method_configs, delta)

    results = {}
    runtime = {
        distribution: {
            method: {str(grid): 0.0 for grid in grid_sizes}
            for method in METHODS
        }
        for distribution in DISTRIBUTIONS
    }
    for distribution_index, distribution in enumerate(DISTRIBUTIONS):
        if progress:
            print(f"topology: {distribution}", flush=True)
        raw = {
            method: {str(grid): [] for grid in grid_sizes}
            for method in METHODS
        }
        for path_index in range(num_paths):
            rng, seed_words = _path_rng(
                seed, 101, distribution_index, path_index
            )
            observations = _sample_distribution(
                rng, distribution, max_time
            )
            for method in METHODS:
                config = method_configs[method]
                for grid in grid_sizes:
                    started = time.perf_counter()
                    inverted = confidence_sequence_endpoints(
                        observations,
                        delta,
                        times,
                        method,
                        method_config=config,
                        topology_grid_size=grid,
                    )
                    runtime[distribution][method][str(grid)] += (
                        time.perf_counter() - started
                    )
                    raw[method][str(grid)].append(
                        _raw_endpoint_record(
                            inverted, path_index, seed_words
                        )
                    )

        distribution_result = {}
        low_grid, high_grid = comparison_grids
        for method in METHODS:
            grids = {}
            for grid in grid_sizes:
                records = raw[method][str(grid)]
                grids[str(grid)] = {
                    "raw_paths": records,
                    "diagnostic_summary": _diagnostic_summary(records),
                    "runtime_seconds": float(
                        runtime[distribution][method][str(grid)]
                    ),
                }
            distribution_result[method] = {
                "grids": grids,
                f"comparison_{low_grid}_vs_{high_grid}": (
                    _resolution_comparison(
                        raw[method][str(low_grid)],
                        raw[method][str(high_grid)],
                    )
                ),
            }
        results[distribution] = distribution_result

    return {
        "audit": "confidence_sequence_topology_resolution",
        "schema_version": 1,
        "parameters": {
            "delta": float(delta),
            "max_time": int(max_time),
            "num_paths_per_distribution": int(num_paths),
            "seed": int(seed),
            "seed_recipe": [
                "global_seed",
                "audit_code=101",
                "distribution_index",
                "path_index",
            ],
            "grid_sizes": list(grid_sizes),
            "comparison_grids": list(comparison_grids),
            "reporting_times": times.tolist(),
            "time_count_argument": int(time_count),
            "chronological_observations": True,
            "paths_reused_across_grids_and_methods": True,
            "fixed_schedules_declared_before_sampling": True,
            "root_refinement_tolerance": 1e-8,
            "product_grid_size": int(product_grid_size),
            "horizon_ratio": float(horizon_ratio),
            "weight_power": float(weight_power),
            "horizon_overshoot": float(horizon_overshoot),
            "agrapa_c": 0.5,
            "heat_solvency_fraction": 1.0,
        },
        "methods": list(METHODS),
        "distributions": list(DISTRIBUTIONS),
        "true_means": dict(TRUE_MEANS),
        "schedules": schedules,
        "runtime_scope": "endpoint_inversion_after_numba_warmup",
        "results": results,
    }


def _scalar_summary(values) -> dict:
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    if not len(finite):
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "q10": None,
            "q90": None,
            "minimum": None,
            "maximum": None,
        }
    return {
        "count": int(len(finite)),
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "q10": float(np.quantile(finite, 0.1)),
        "q90": float(np.quantile(finite, 0.9)),
        "minimum": float(np.min(finite)),
        "maximum": float(np.max(finite)),
    }


def _wilson_interval(successes: int, total: int):
    if total <= 0:
        return None, None
    z_value = 1.959963984540054
    probability = successes / float(total)
    z2 = z_value * z_value
    denominator = 1.0 + z2 / total
    center = (probability + z2 / (2.0 * total)) / denominator
    radius = z_value / denominator * math.sqrt(
        probability * (1.0 - probability) / total
        + z2 / (4.0 * total * total)
    )
    return max(center - radius, 0.0), min(center + radius, 1.0)


def _heat_log_e_path(
    method: str,
    observations: np.ndarray,
    candidate_mean: float,
    config: dict,
):
    arguments = (
        observations,
        candidate_mean,
        config["horizons"],
        config["strikes"],
        config["weights"],
        config["cash_weight"],
    )
    if method == "bentkus_mixture":
        return bentkus_mixture_log_e_path(*arguments, solvency_fraction=1.0)
    if method == "heat_constrained_agrapa":
        return heat_constrained_agrapa_log_e_path(
            *arguments,
            agrapa_c=0.5,
            solvency_fraction=1.0,
        )
    raise ValueError(f"unknown heat method {method!r}")


def _warm_sensitivity(schedules: dict, delta: float) -> None:
    observations = np.asarray([0.0, 1.0, 0.25, 0.75], dtype=np.float64)
    for config in schedules.values():
        for method in HEAT_METHODS:
            _heat_log_e_path(method, observations, 0.5, config)
            confidence_sequence_endpoints(
                observations,
                delta,
                np.asarray([4], dtype=np.int64),
                method,
                method_config=config,
                topology_grid_size=5,
            )


def run_schedule_sensitivity_audit(
    *,
    delta: float,
    max_time: int,
    num_width_paths: int,
    num_coverage_paths: int,
    seed: int,
    topology_grid_size: int,
    horizon_overshoot: float,
    progress: bool,
):
    schedules = {}
    schedule_metadata = {}
    for name, horizon_ratio, weight_power in SENSITIVITY_CONFIGURATIONS:
        config = _heat_schedule_dict(
            max_time,
            delta,
            horizon_ratio,
            weight_power,
            horizon_overshoot,
        )
        schedules[name] = config
        schedule_metadata[name] = {
            "horizon_ratio": float(horizon_ratio),
            "weight_power": float(weight_power),
            "horizon_overshoot": float(horizon_overshoot),
            **_schedule_record(config),
        }
    _warm_sensitivity(schedules, delta)

    width_results = {}
    width_times = np.asarray([max_time], dtype=np.int64)
    for distribution_index, distribution in enumerate(DISTRIBUTIONS):
        if progress:
            print(f"sensitivity widths: {distribution}", flush=True)
        raw = {
            name: {method: [] for method in HEAT_METHODS}
            for name in schedules
        }
        runtime = {
            name: {method: 0.0 for method in HEAT_METHODS}
            for name in schedules
        }
        for path_index in range(num_width_paths):
            rng, seed_words = _path_rng(
                seed, 201, distribution_index, path_index
            )
            observations = _sample_distribution(
                rng, distribution, max_time
            )
            for name, config in schedules.items():
                for method in HEAT_METHODS:
                    started = time.perf_counter()
                    inverted = confidence_sequence_endpoints(
                        observations,
                        delta,
                        width_times,
                        method,
                        method_config=config,
                        topology_grid_size=topology_grid_size,
                    )
                    runtime[name][method] += time.perf_counter() - started
                    raw[name][method].append(
                        _raw_endpoint_record(
                            inverted, path_index, seed_words
                        )
                    )

        distribution_result = {}
        for name in schedules:
            configuration_result = {}
            for method in HEAT_METHODS:
                records = raw[name][method]
                widths = [record["width"][0] for record in records]
                configuration_result[method] = {
                    "raw_paths": records,
                    "terminal_width_summary": _scalar_summary(widths),
                    "diagnostic_summary": _diagnostic_summary(records),
                    "runtime_seconds": float(runtime[name][method]),
                }
            bentkus_widths = np.asarray(
                [
                    row["width"][0]
                    for row in raw[name]["bentkus_mixture"]
                ],
                dtype=np.float64,
            )
            constrained_widths = np.asarray(
                [
                    row["width"][0]
                    for row in raw[name]["heat_constrained_agrapa"]
                ],
                dtype=np.float64,
            )
            valid_ratio = (
                np.isfinite(constrained_widths)
                & np.isfinite(bentkus_widths)
                & (bentkus_widths > 0.0)
            )
            ratios = constrained_widths[valid_ratio] / bentkus_widths[
                valid_ratio
            ]
            configuration_result["paired_comparison"] = {
                "heat_constrained_over_bentkus_width_ratio": (
                    _scalar_summary(ratios)
                ),
                "heat_constrained_narrower_count": int(
                    np.sum(
                        constrained_widths[valid_ratio]
                        < bentkus_widths[valid_ratio]
                    )
                ),
                "finite_positive_bentkus_pair_count": int(
                    np.sum(valid_ratio)
                ),
            }
            distribution_result[name] = configuration_result
        width_results[distribution] = distribution_result

    coverage_results = {}
    threshold = math.log(1.0 / delta)
    for distribution_index, distribution in enumerate(DISTRIBUTIONS):
        if progress:
            print(f"sensitivity coverage: {distribution}", flush=True)
        flags = {
            name: {method: [] for method in HEAT_METHODS}
            for name in schedules
        }
        true_mean = TRUE_MEANS[distribution]
        for path_index in range(num_coverage_paths):
            if progress and path_index and path_index % 100 == 0:
                print(
                    f"  {distribution}: {path_index}/{num_coverage_paths}",
                    flush=True,
                )
            rng, _ = _path_rng(seed, 202, distribution_index, path_index)
            observations = _sample_distribution(
                rng, distribution, max_time
            )
            for name, config in schedules.items():
                for method in HEAT_METHODS:
                    log_path = _heat_log_e_path(
                        method, observations, true_mean, config
                    )
                    flags[name][method].append(
                        bool(np.max(log_path) >= threshold)
                    )

        distribution_result = {}
        for name in schedules:
            configuration_result = {}
            for method in HEAT_METHODS:
                method_flags = flags[name][method]
                crossings = int(np.sum(method_flags))
                lower, upper = _wilson_interval(
                    crossings, num_coverage_paths
                )
                configuration_result[method] = {
                    "raw_crossing_flags": method_flags,
                    "crossings": crossings,
                    "total": int(num_coverage_paths),
                    "crossing_rate": (
                        crossings / float(num_coverage_paths)
                        if num_coverage_paths
                        else None
                    ),
                    "wilson_95_lower": lower,
                    "wilson_95_upper": upper,
                }
            bentkus_flags = np.asarray(
                flags[name]["bentkus_mixture"], dtype=bool
            )
            constrained_flags = np.asarray(
                flags[name]["heat_constrained_agrapa"], dtype=bool
            )
            configuration_result["paired_comparison"] = {
                "both_crossed": int(
                    np.sum(bentkus_flags & constrained_flags)
                ),
                "bentkus_only_crossed": int(
                    np.sum(bentkus_flags & ~constrained_flags)
                ),
                "heat_constrained_only_crossed": int(
                    np.sum(~bentkus_flags & constrained_flags)
                ),
                "neither_crossed": int(
                    np.sum(~bentkus_flags & ~constrained_flags)
                ),
            }
            distribution_result[name] = configuration_result
        coverage_results[distribution] = distribution_result

    return {
        "audit": "bentkus_schedule_sensitivity",
        "schema_version": 1,
        "parameters": {
            "delta": float(delta),
            "max_time": int(max_time),
            "width_reporting_times": [int(max_time)],
            "num_width_paths_per_distribution": int(num_width_paths),
            "num_coverage_paths_per_distribution": int(
                num_coverage_paths
            ),
            "seed": int(seed),
            "width_seed_recipe": [
                "global_seed",
                "audit_code=201",
                "distribution_index",
                "path_index",
            ],
            "coverage_seed_recipe": [
                "global_seed",
                "audit_code=202",
                "distribution_index",
                "path_index",
            ],
            "topology_grid_size_for_widths": int(topology_grid_size),
            "horizon_overshoot": float(horizon_overshoot),
            "agrapa_c": 0.5,
            "heat_solvency_fraction": 1.0,
            "chronological_observations": True,
            "width_paths_reused_across_schedules_and_methods": True,
            "coverage_paths_reused_across_schedules_and_methods": True,
            "fixed_schedules_declared_before_sampling": True,
            "crossing_rule": "max_{0<=t<=max_time} log(E_t) >= log(1/delta)",
        },
        "methods": list(HEAT_METHODS),
        "configuration_order": [
            name for name, _, _ in SENSITIVITY_CONFIGURATIONS
        ],
        "distributions": list(DISTRIBUTIONS),
        "true_means": dict(TRUE_MEANS),
        "schedules": schedule_metadata,
        "width_runtime_scope": "terminal_endpoint_inversion_after_numba_warmup",
        "width_results": width_results,
        "coverage_results": coverage_results,
    }


def _positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _nonnegative_integer(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be a nonnegative integer")
    return parsed


def _probability(value: str) -> float:
    parsed = float(value)
    if not 0.0 < parsed < 1.0:
        raise argparse.ArgumentTypeError("must lie strictly between zero and one")
    return parsed


def _grid_size(value: str) -> int:
    parsed = int(value)
    if parsed < 3:
        raise argparse.ArgumentTypeError("must be an integer at least three")
    return parsed


def _integer_at_least_two(value: str) -> int:
    parsed = int(value)
    if parsed < 2:
        raise argparse.ArgumentTypeError("must be an integer at least two")
    return parsed


def _output_prefix(value: str) -> str:
    if not value or value in {".", ".."} or Path(value).name != value:
        raise argparse.ArgumentTypeError("must be a file prefix, not a path")
    if "/" in value or "\\" in value:
        raise argparse.ArgumentTypeError("must be a file prefix, not a path")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--audit", choices=("all", "topology", "sensitivity"), default="all"
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--delta", type=_probability, default=0.01)
    parser.add_argument("--seed", type=_nonnegative_integer, default=20260717)
    parser.add_argument(
        "--output-prefix",
        type=_output_prefix,
        default="confidence_sequence_audits",
    )
    parser.add_argument(
        "--topology-max-time", type=_positive_integer, default=10_000
    )
    parser.add_argument(
        "--topology-num-paths", type=_positive_integer, default=3
    )
    parser.add_argument(
        "--topology-grid-sizes",
        type=_grid_size,
        nargs="+",
        default=(33, 65, 129),
    )
    parser.add_argument(
        "--topology-comparison-grids",
        type=_grid_size,
        nargs=2,
        default=(65, 129),
    )
    parser.add_argument(
        "--topology-time-count", type=_integer_at_least_two, default=32
    )
    parser.add_argument(
        "--product-grid-size", type=_positive_integer, default=20
    )
    parser.add_argument(
        "--sensitivity-max-time", type=_positive_integer, default=10_000
    )
    parser.add_argument(
        "--sensitivity-num-width-paths", type=_positive_integer, default=5
    )
    parser.add_argument(
        "--sensitivity-num-coverage-paths",
        type=_nonnegative_integer,
        default=500,
    )
    parser.add_argument(
        "--sensitivity-topology-grid-size", type=_grid_size, default=65
    )
    parser.add_argument(
        "--topology-horizon-ratio", type=float, default=2.0
    )
    parser.add_argument(
        "--topology-weight-power", type=float, default=2.0
    )
    parser.add_argument("--horizon-overshoot", type=float, default=2.0)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.topology_horizon_ratio <= 1.0:
        raise SystemExit("--topology-horizon-ratio must exceed one")
    if args.topology_weight_power <= 1.0:
        raise SystemExit("--topology-weight-power must exceed one")
    if args.horizon_overshoot < 1.0:
        raise SystemExit("--horizon-overshoot must be at least one")

    grid_sizes = tuple(dict.fromkeys(args.topology_grid_sizes))
    comparison_grids = tuple(args.topology_comparison_grids)
    missing = [grid for grid in comparison_grids if grid not in grid_sizes]
    if missing:
        raise SystemExit(
            "--topology-comparison-grids must both occur in "
            f"--topology-grid-sizes; missing {missing}"
        )

    topology_max_time = args.topology_max_time
    topology_num_paths = args.topology_num_paths
    topology_time_count = args.topology_time_count
    sensitivity_max_time = args.sensitivity_max_time
    sensitivity_num_width_paths = args.sensitivity_num_width_paths
    sensitivity_num_coverage_paths = args.sensitivity_num_coverage_paths
    sensitivity_grid_size = args.sensitivity_topology_grid_size
    output_prefix = args.output_prefix
    if args.smoke:
        topology_max_time = min(topology_max_time, 64)
        topology_num_paths = 1
        topology_time_count = min(topology_time_count, 8)
        grid_sizes = (9, 17, 33)
        comparison_grids = (17, 33)
        sensitivity_max_time = min(sensitivity_max_time, 64)
        sensitivity_num_width_paths = 1
        sensitivity_num_coverage_paths = min(
            sensitivity_num_coverage_paths, 4
        )
        sensitivity_grid_size = 17
        output_prefix += "_smoke"

    written = []
    if args.audit in ("all", "topology"):
        payload = run_topology_audit(
            delta=args.delta,
            max_time=topology_max_time,
            num_paths=topology_num_paths,
            seed=args.seed,
            grid_sizes=grid_sizes,
            comparison_grids=comparison_grids,
            time_count=topology_time_count,
            product_grid_size=args.product_grid_size,
            horizon_ratio=args.topology_horizon_ratio,
            weight_power=args.topology_weight_power,
            horizon_overshoot=args.horizon_overshoot,
            progress=args.progress,
        )
        output_path = OUTPUT_DIRECTORY / f"{output_prefix}_topology.json"
        _write_strict_json(payload, output_path)
        written.append(output_path)

    if args.audit in ("all", "sensitivity"):
        payload = run_schedule_sensitivity_audit(
            delta=args.delta,
            max_time=sensitivity_max_time,
            num_width_paths=sensitivity_num_width_paths,
            num_coverage_paths=sensitivity_num_coverage_paths,
            seed=args.seed,
            topology_grid_size=sensitivity_grid_size,
            horizon_overshoot=args.horizon_overshoot,
            progress=args.progress,
        )
        output_path = (
            OUTPUT_DIRECTORY
            / f"{output_prefix}_schedule_sensitivity.json"
        )
        _write_strict_json(payload, output_path)
        written.append(output_path)

    for path in written:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
