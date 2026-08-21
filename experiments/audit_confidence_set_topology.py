"""Empirical topology audit for the fixed-sample confidence sets.

The audit never assumes quasiconvexity.  It evaluates each rejection statistic
on a global candidate-mean grid, counts every accepted run visible on that
grid, and compares center-component length, total accepted length, and full-set
diameter.  Successive grid sizes should be compared because a finite mesh can
miss narrow components.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from numba import njit, prange

import betting


METHODS = (
    "WSR",
    "STaR",
    "Fixed hinge",
    "Hinge STaR",
    "Efficient betting",
)
PUBLICATION_N = (10, 50, 100, 500, 1000)


@njit(parallel=True)
def _score_matrix(
    paths,
    means,
    delta,
    strike,
    initial_wealth,
    kind,
    u_plus,
    u_minus,
):
    rows = paths.shape[0]
    cols = means.size
    output = np.empty((rows, cols))
    for flat_index in prange(rows * cols):
        row = flat_index // cols
        col = flat_index - row * cols
        x = paths[row]
        m = means[col]
        if kind == 0:
            value = betting.compute_M_inf(x, m, delta)
        elif kind == 1:
            value = betting.compute_M_star(x, m, delta)
        elif kind == 2:
            value = betting.compute_M_heat_path(x, m, strike, initial_wealth)
        elif kind == 3:
            value = betting.compute_M_heat_star_path(x, m, delta, initial_wealth)
        else:
            plus, minus = betting.compute_M_probit_star_arms(
                x, m, delta, buffer_rounds=0.0
            )
            alpha = delta / 2.0
            value = max(
                alpha * plus / u_plus[row], alpha * minus / u_minus[row]
            )
        output[row, col] = value
    return output


def _threshold(kind, delta, initial_wealth):
    if kind < 2:
        return 1.0 / delta
    if kind < 4:
        return initial_wealth / delta
    return 1.0


def _component_count(accepted):
    return int(np.sum(accepted & np.r_[True, ~accepted[:-1]]))


def _publication_samplers(rng):
    return (
        ("Beta(2,2)", lambda n: rng.beta(2.0, 2.0, n)),
        ("Beta(1,5)", lambda n: rng.beta(1.0, 5.0, n)),
        ("Bernoulli(0.5)", lambda n: rng.binomial(1, 0.5, n).astype(float)),
        ("Uniform(0,1)", lambda n: rng.uniform(0.0, 1.0, n)),
        ("Beta(0.5,0.5)", lambda n: rng.beta(0.5, 0.5, n)),
        ("Bernoulli(0.1)", lambda n: rng.binomial(1, 0.1, n).astype(float)),
        ("Beta(50,50)", lambda n: rng.beta(50.0, 50.0, n)),
        ("Beta(20,80)", lambda n: rng.beta(20.0, 80.0, n)),
        ("Uniform(0.45,0.55)", lambda n: rng.uniform(0.45, 0.55, n)),
    )


def publication_audit(seed, delta, scan_points, repetitions):
    rng = np.random.default_rng(seed)
    randomizer_seed = np.random.SeedSequence(seed).spawn(1)[0]
    randomizer_rng = np.random.default_rng(randomizer_seed)
    rng.uniform(0.0, 1.0, 20)  # same warm-up draw as betting.run_experiment
    means = np.linspace(0.0, 1.0, scan_points)
    strike, initial_wealth = betting.get_optimal_lambda(delta / 2.0)
    counts = {
        method: {"empty": 0, "connected": 0, "disconnected": 0}
        for method in METHODS
    }
    by_n = {}
    total_paths = 0
    for n in PUBLICATION_N:
        by_n[str(n)] = {
            method: {"empty": 0, "connected": 0, "disconnected": 0}
            for method in METHODS
        }
        for _, sampler in _publication_samplers(rng):
            paths = []
            u_plus = []
            u_minus = []
            for _ in range(repetitions):
                paths.append(sampler(n))
                uniforms = randomizer_rng.uniform(size=2)
                u_plus.append(uniforms[0])
                u_minus.append(uniforms[1])
            paths = np.asarray(paths, dtype=float)
            u_plus = np.asarray(u_plus)
            u_minus = np.asarray(u_minus)
            total_paths += repetitions
            for kind, method in enumerate(METHODS):
                values = _score_matrix(
                    paths,
                    means,
                    delta,
                    strike,
                    initial_wealth,
                    kind,
                    u_plus,
                    u_minus,
                )
                threshold = _threshold(kind, delta, initial_wealth)
                for row in values:
                    component_count = _component_count(row < threshold)
                    label = (
                        "empty" if component_count == 0
                        else "connected" if component_count == 1
                        else "disconnected"
                    )
                    counts[method][label] += 1
                    by_n[str(n)][method][label] += 1
    return {
        "seed": seed,
        "sample_sizes": list(PUBLICATION_N),
        "repetitions_per_distribution_and_n": repetitions,
        "distribution_count": 9,
        "candidate_grid_size": scan_points,
        "sets_checked_per_method": total_paths,
        "counts": counts,
        "counts_by_n": by_n,
    }


def _component_summary(
    statistic,
    threshold,
    center,
    grids,
    batch_statistic=None,
):
    output = {}
    for grid in grids:
        components = betting._confidence_set_components(
            statistic,
            threshold,
            scan_points=grid,
            batch_statistic=batch_statistic,
            boundary_tolerance=2e-9,
        )
        summary = betting._confidence_set_widths(components, center=center)
        summary["components"] = [list(component) for component in components]
        output[str(grid)] = summary
    return output


def counterexample_audit(delta, grids):
    strike, initial_wealth = betting.get_optimal_lambda(delta / 2.0)
    fixed_x = np.zeros(100)
    fixed_one_locations = (2, 12, 17, 23, 28, 30, 33, 34, 68)
    fixed_x[np.asarray(fixed_one_locations)] = 1.0
    threshold = initial_wealth / delta
    fixed = _component_summary(
        lambda m: betting.compute_M_heat_path(
            fixed_x, m, strike, initial_wealth
        ),
        threshold,
        float(np.mean(fixed_x)),
        grids,
    )
    descending_sparse = np.r_[np.ones(11), np.zeros(89)]
    hinge_star = _component_summary(
        lambda m: betting.compute_M_heat_star_path(
            descending_sparse, m, delta, initial_wealth
        ),
        threshold,
        float(np.mean(descending_sparse)),
        grids,
    )

    efficient_x = np.r_[np.zeros(107), np.ones(93)]
    u_plus = 0.9961808459752498
    u_minus = 0.36866436376777545
    alpha = delta / 2.0

    def efficient_score(m):
        plus, minus = betting.compute_M_probit_star_arms(
            efficient_x, m, delta, buffer_rounds=0.0
        )
        return max(alpha * plus / u_plus, alpha * minus / u_minus)

    efficient = _component_summary(
        efficient_score,
        1.0,
        float(np.mean(efficient_x)),
        grids,
        batch_statistic=lambda means: betting._probit_randomized_scores(
            efficient_x,
            np.asarray(means),
            delta,
            0.0,
            u_plus,
            u_minus,
        ),
    )
    star_x = np.r_[1.0, np.zeros(19)]
    star_means = (0.047125, 0.997, 0.997875)
    star_values = [
        float(betting.compute_M_star(star_x, mean, delta))
        for mean in star_means
    ]
    return {
        "star_quasiconvexity_witness": {
            "data": "one followed by 19 zeros",
            "candidate_means": list(star_means),
            "statistic_values": star_values,
            "violates_quasiconvexity": bool(
                star_values[1] > max(star_values[0], star_values[2])
            ),
        },
        "fixed_hinge_data": "nine ones among 100 observations",
        "fixed_hinge_one_locations_zero_indexed": list(fixed_one_locations),
        "fixed_hinge": fixed,
        "hinge_star_data": "11 ones followed by 89 zeros",
        "hinge_star": hinge_star,
        "efficient_data": "107 zeros followed by 93 ones",
        "efficient_randomizers": [u_plus, u_minus],
        "efficient_betting": efficient,
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--delta", type=float, default=0.01)
    parser.add_argument("--scan-points", type=int, default=2001)
    parser.add_argument("--repetitions", type=int, default=50)
    parser.add_argument(
        "--counterexample-grids",
        type=int,
        nargs="+",
        default=(2001, 4001, 8001, 16001, 32001),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("plots/confidence_set_topology_audit.json"),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    result = {
        "width_convention": {
            "total_length": "sum of accepted-component lengths",
            "hull_width": "supremum minus infimum of the full accepted set",
            "largest_component_width": (
                "maximum length among all accepted components"
            ),
            "center_component_width": (
                "length of the accepted component containing the sample mean"
            ),
        },
        "publication_protocol": publication_audit(
            args.seed, args.delta, args.scan_points, args.repetitions
        ),
        "counterexamples": counterexample_audit(
            args.delta, tuple(args.counterexample_grids)
        ),
        "caveat": (
            "A finite topology grid is an empirical audit, not a proof that "
            "arbitrarily narrow additional components do not occur "
            "between grid points."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
    print(f"saved {args.output}")
    for method, counts in result["publication_protocol"]["counts"].items():
        print(method, counts)


if __name__ == "__main__":
    main()
