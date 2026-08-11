#!/usr/bin/env python3
"""Generate the Gaffke curves used in the paper's main comparison figure."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd


BETTING_DIR = Path(__file__).resolve().parents[1]
if str(BETTING_DIR) not in sys.path:
    sys.path.insert(0, str(BETTING_DIR))

from gaffke_comparison.compare_star_probit_gaffke import DISTRIBUTIONS
from gaffke_comparison.large_sample_feedback_gaffke import fast_gaffke_ci


HERE = BETTING_DIR / "gaffke_comparison"
DEFAULT_OUTPUT = HERE / "figure2_gaffke_results"
DELTA = 0.01
SEED = 20260810
SAMPLE_SIZES = (
    10,
    50,
    100,
    500,
    1_000,
    3_000,
    10_000,
    30_000,
    100_000,
    300_000,
    1_000_000,
)
REPS_BY_N = {
    n: 120 if n <= 1_000 else 60 if n <= 10_000 else 30
    for n in SAMPLE_SIZES
}
LOW_VARIANCE_DISTRIBUTIONS = {
    "Beta(50,50)",
    "Beta(20,80)",
    "Uniform(0.45,0.55)",
}


def _randomized_lower_endpoint(
    *,
    n: int,
    sample_minimum: float,
    log_product: float,
    uniform: float,
    ordinary_endpoint: float,
    tail_probability: float,
) -> float:
    if sample_minimum <= 0.0 or ordinary_endpoint >= sample_minimum:
        return ordinary_endpoint
    log_candidate = (
        math.log(tail_probability) - math.log(uniform) + log_product
    ) / n
    candidate = math.exp(log_candidate)
    return max(ordinary_endpoint, min(sample_minimum, candidate))


def _randomized_endpoints(
    x: np.ndarray,
    lower: float,
    upper: float,
    u_plus: float,
    u_minus: float,
    delta: float,
) -> tuple[float, float]:
    positive = x > 0.0
    reflected = 1.0 - x
    reflected_positive = reflected > 0.0
    log_product = (
        float(np.sum(np.log(x))) if bool(np.all(positive)) else -math.inf
    )
    log_reflected_product = (
        float(np.sum(np.log(reflected)))
        if bool(np.all(reflected_positive))
        else -math.inf
    )
    randomized_lower = _randomized_lower_endpoint(
        n=x.size,
        sample_minimum=float(np.min(x)),
        log_product=log_product,
        uniform=u_plus,
        ordinary_endpoint=lower,
        tail_probability=delta / 2.0,
    )
    reflected_lower = _randomized_lower_endpoint(
        n=x.size,
        sample_minimum=float(np.min(reflected)),
        log_product=log_reflected_product,
        uniform=u_minus,
        ordinary_endpoint=1.0 - upper,
        tail_probability=delta / 2.0,
    )
    return randomized_lower, 1.0 - reflected_lower


def _record(
    distribution: str,
    n: int,
    replication: int,
    method: str,
    lower: float,
    upper: float,
    backend: str,
) -> dict[str, object]:
    width = max(float(upper - lower), 0.0)
    return {
        "distribution": distribution,
        "n": n,
        "rep": replication,
        "method": method,
        "lower": lower,
        "upper": upper,
        "width": width,
        "sqrt_n_width": math.sqrt(n) * width,
        "backend": backend,
    }


def run(output: Path, progress_every: int) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    completed_paths = 0
    for distribution_index, distribution in enumerate(DISTRIBUTIONS):
        available_sizes = [
            n
            for n in SAMPLE_SIZES
            if distribution.name not in LOW_VARIANCE_DISTRIBUTIONS or n >= 1_000
        ]
        for replication in range(max(REPS_BY_N.values())):
            eligible = [
                n for n in available_sizes if replication < REPS_BY_N[n]
            ]
            if not eligible:
                continue
            seed_sequence = np.random.SeedSequence(
                [SEED, distribution_index, replication]
            )
            data_seed, auxiliary_seed = seed_sequence.spawn(2)
            data_rng = np.random.default_rng(data_seed)
            auxiliary_rng = np.random.default_rng(auxiliary_seed)
            path = np.asarray(
                distribution.sampler(data_rng, max(eligible)), dtype=float
            )
            for n in eligible:
                x = np.ascontiguousarray(path[:n])
                u_plus, u_minus = (
                    float(value) for value in auxiliary_rng.uniform(size=2)
                )
                lower, upper, backend = fast_gaffke_ci(
                    x,
                    delta=DELTA,
                    binary=distribution.name.startswith("Bernoulli"),
                    exact_cutoff=3_000,
                )
                rows.append(
                    _record(
                        distribution.name,
                        n,
                        replication,
                        "Gaffke",
                        lower,
                        upper,
                        backend,
                    )
                )
                randomized_lower, randomized_upper = _randomized_endpoints(
                    x, lower, upper, u_plus, u_minus, DELTA
                )
                rows.append(
                    _record(
                        distribution.name,
                        n,
                        replication,
                        "Randomized Gaffke",
                        randomized_lower,
                        randomized_upper,
                        f"{backend}+product-orthant",
                    )
                )
            completed_paths += 1
            if progress_every > 0 and completed_paths % progress_every == 0:
                print(
                    f"completed {completed_paths} paths: "
                    f"{distribution.name}, replication {replication + 1}",
                    flush=True,
                )

    results = pd.DataFrame(rows)
    results_path = output / "results.csv"
    results.to_csv(results_path, index=False)
    with (output / "config.json").open("w", encoding="utf-8") as stream:
        json.dump(
            {
                "delta": DELTA,
                "seed": SEED,
                "sample_sizes": list(SAMPLE_SIZES),
                "reps_by_n": {
                    str(n): REPS_BY_N[n] for n in SAMPLE_SIZES
                },
                "distributions": [
                    distribution.name for distribution in DISTRIBUTIONS
                ],
                "low_variance_minimum_n": 1_000,
                "gaffke_exact_cutoff": 3_000,
                "randomization": (
                    "product-orthant refinement with one uniform per arm"
                ),
            },
            stream,
            indent=2,
        )
    print(f"saved {results_path}")
    return results_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--progress-every", type=int, default=20)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    run(arguments.output, arguments.progress_every)
