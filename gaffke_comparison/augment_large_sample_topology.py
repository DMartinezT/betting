#!/usr/bin/env python3
"""Augment the large-sample results with topology-aware betting widths.

The expensive topology audit is paired across methods and restricted to the
first ``--rep-limit`` paths per distribution and horizon.  This matches the
existing Efficient-betting budget.  Full-set diameters and largest-component
widths use the adaptive multiresolution inverter from ``betting.py``.
"""

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

import betting as methods  # noqa: E402
from gaffke_comparison.compare_star_probit_gaffke import (  # noqa: E402
    DISTRIBUTIONS,
)
from gaffke_comparison.large_sample_feedback_gaffke import (  # noqa: E402
    summarize,
)


BETTING_METHODS = (
    "Efficient betting",
)


def _score_functions(
    x, method_name, delta, u_plus, u_minus, solvency_c
):
    threshold = 1.0 / delta
    if method_name == "Square-root feedback":
        return (
            lambda m: methods.compute_M_star(x, m, delta),
            lambda means: methods._recalculating_feedback_scores(
                x, np.asarray(means), delta, 0
            ),
            threshold,
        )
    if method_name == "Squared-hinge feedback":
        return (
            lambda m: methods.compute_M_hinge_feedback_star(
                x, m, delta
            ),
            lambda means: methods._recalculating_feedback_scores(
                x, np.asarray(means), delta, 1
            ),
            threshold,
        )
    if method_name == "Efficient betting":
        alpha = delta / 2.0

        def scalar(mean):
            plus, minus = methods.compute_M_probit_star_arms(
                x, mean, delta, c=solvency_c, buffer_rounds=0.0
            )
            return max(
                alpha * plus / u_plus,
                alpha * minus / u_minus,
            )

        return (
            scalar,
            lambda means: methods._probit_randomized_scores(
                x,
                np.asarray(means),
                delta,
                0.0,
                u_plus,
                u_minus,
                solvency_c,
            ),
            1.0,
        )
    raise ValueError(f"unknown method {method_name}")


def _exponential_outside_grid(lower, upper, standard_error):
    """Probe outside a known component on a logarithmic distance scale."""
    candidates = [0.0, 1.0]
    first_distance = max(standard_error / 8.0, 1.0e-12)
    for boundary, endpoint, direction in (
        (lower, 0.0, -1.0),
        (upper, 1.0, 1.0),
    ):
        available = abs(endpoint - boundary)
        distance = first_distance
        while distance < available:
            candidates.append(boundary + direction * distance)
            distance *= 2.0
    grid = np.unique(np.clip(candidates, 0.0, 1.0))
    return grid[(grid < lower) | (grid > upper)]


def _topology(
    x,
    method_name,
    delta,
    u_plus,
    u_minus,
    score_work_budget,
    local_lower,
    local_upper,
    local_empty,
    solvency_c,
):
    scalar, batch, threshold = _score_functions(
        x, method_name, delta, u_plus, u_minus, solvency_c
    )
    standard_error = methods._sample_standard_error(x)

    # At moderate n a full multiresolution scan is inexpensive.  At large n,
    # reuse the already-bisected component, screen its interior for gaps, and
    # search logarithmically for accepted islands outside it.  This avoids
    # dozens of O(n) scalar bisections when the set is interval-like.
    if len(x) <= 10_000 or local_empty:
        parameters = methods._topology_scan_parameters(
            len(x),
            score_work_budget=score_work_budget,
            minimum_verification_points=9,
            maximum_verification_points=2049,
        )
        components, diagnostics = (
            methods._adaptive_confidence_set_components(
                scalar,
                threshold,
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
        )
        diagnostics = dict(diagnostics)
        diagnostics["screening_only"] = False
    else:
        outside_grid = _exponential_outside_grid(
            local_lower, local_upper, standard_error
        )
        interior_grid = np.linspace(local_lower, local_upper, 33)[1:-1]
        screen_grid = np.unique(np.r_[outside_grid, interior_grid])
        screen_scores = np.asarray(batch(screen_grid), dtype=float)
        outside_mask = (screen_grid < local_lower) | (
            screen_grid > local_upper
        )
        accepted = np.isfinite(screen_scores) & (
            screen_scores < threshold
        )
        extra_acceptance = np.any(accepted[outside_mask])
        interior_rejection = np.any(~accepted[~outside_mask])
        if extra_acceptance or interior_rejection:
            parameters = methods._topology_scan_parameters(
                len(x),
                score_work_budget=score_work_budget,
                minimum_verification_points=9,
                maximum_verification_points=2049,
            )
            components, diagnostics = (
                methods._adaptive_confidence_set_components(
                    scalar,
                    threshold,
                    float(np.mean(x)),
                    standard_error,
                    batch_statistic=batch,
                    local_radius=6.0,
                    base_points_per_se=1,
                    global_scan_points=9,
                    geometric_tail_points=6,
                    boundary_tolerance=max(
                        1.0e-8, standard_error * 1.0e-4
                    ),
                    **parameters,
                )
            )
            diagnostics = dict(diagnostics)
            diagnostics["screening_only"] = False
            diagnostics["outside_screen_points"] = int(
                outside_grid.size
            )
            diagnostics["interior_screen_points"] = int(
                interior_grid.size
            )
        else:
            components = ((float(local_lower), float(local_upper)),)
            diagnostics = {
                **methods._confidence_set_widths(
                    components, center=float(np.mean(x))
                ),
                "evaluation_count": int(screen_grid.size),
                "scan_point_count": int(screen_grid.size),
                "refinement_levels": 0,
                "fragmentation_detected": False,
                "point_budget_reached": False,
                "final_mesh_resolution": float(
                    np.max(np.diff(np.unique(np.r_[
                        screen_grid, local_lower, local_upper
                    ])))
                    if screen_grid.size > 1 else 1.0
                ),
                "standard_error": standard_error,
                "finite_mesh": True,
                "screening_only": True,
                "outside_screen_points": int(outside_grid.size),
                "interior_screen_points": int(interior_grid.size),
            }

    diagnostics["lower"] = (
        float(components[0][0]) if components else math.nan
    )
    diagnostics["upper"] = (
        float(components[-1][1]) if components else math.nan
    )
    return diagnostics


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(__file__).with_name("large_sample_gaffke_results"),
    )
    parser.add_argument("--rep-limit", type=int, default=30)
    parser.add_argument("--score-work-budget", type=int, default=300_000)
    parser.add_argument("--progress-every", type=int, default=5)
    return parser.parse_args()


def main():
    args = parse_args()
    results_dir = args.results_dir.resolve()
    with (results_dir / "config.json").open(encoding="utf-8") as stream:
        config = json.load(stream)
    solvency_c = float(config.get("solvency_c", 1.0))
    results_path = results_dir / "results.csv"
    frame = pd.read_csv(results_path)
    checkpoint_path = results_dir / "topology_checkpoint.json"
    completed = {}
    if checkpoint_path.exists():
        with checkpoint_path.open(encoding="utf-8") as stream:
            checkpoint_payload = json.load(stream)
        checkpoint_c = float(checkpoint_payload.get("solvency_c", 1.0))
        if not np.isclose(checkpoint_c, solvency_c):
            raise ValueError(
                f"checkpoint uses c={checkpoint_c}, but results use "
                f"c={solvency_c}"
            )
        completed = checkpoint_payload.get("cells", {})

    sample_sizes = [int(value) for value in config["sample_sizes"]]
    reps_by_n = {
        int(n): int(value) for n, value in config["reps_by_n"].items()
    }
    delta = float(config["delta"])
    seed = int(config["seed"])
    distribution_by_name = {
        distribution.name: (index, distribution)
        for index, distribution in enumerate(DISTRIBUTIONS)
    }
    distribution_pairs = [
        distribution_by_name[name]
        for name in config["distributions"]
    ]

    new_cells = 0
    for distribution_index, distribution in distribution_pairs:
        maximum_replications = min(
            args.rep_limit, max(reps_by_n.values())
        )
        for replication in range(maximum_replications):
            eligible = [
                n
                for n in sample_sizes
                if replication < reps_by_n[n]
            ]
            if not eligible:
                continue
            seed_sequence = np.random.SeedSequence(
                [seed, distribution_index, replication]
            )
            data_seed, auxiliary_seed = seed_sequence.spawn(2)
            data_rng = np.random.default_rng(data_seed)
            auxiliary_rng = np.random.default_rng(auxiliary_seed)
            path = np.asarray(
                distribution.sampler(data_rng, max(eligible)),
                dtype=np.float64,
            )

            for n in eligible:
                u_plus, u_minus = (
                    float(value)
                    for value in auxiliary_rng.uniform(size=2)
                )
                x = np.ascontiguousarray(path[:n])
                for method_name in BETTING_METHODS:
                    key = (
                        f"{distribution.name}|{n}|"
                        f"{replication}|{method_name}"
                    )
                    needs_interior_upgrade = (
                        key in completed
                        and completed[key].get("screening_only", False)
                        and "interior_screen_points" not in completed[key]
                    )
                    if key in completed and not needs_interior_upgrade:
                        continue
                    row_mask = (
                        (frame["distribution"] == distribution.name)
                        & (frame["n"] == n)
                        & (frame["rep"] == replication)
                        & (frame["method"] == method_name)
                    )
                    if int(row_mask.sum()) != 1:
                        raise RuntimeError(
                            f"could not uniquely locate {key}"
                        )
                    row = frame.loc[row_mask].iloc[0]
                    completed[key] = _topology(
                        x,
                        method_name,
                        delta,
                        u_plus,
                        u_minus,
                        args.score_work_budget,
                        float(row["lower"]),
                        float(row["upper"]),
                        bool(row["empty_center_component"]),
                        solvency_c,
                    )
                    new_cells += 1
                    if new_cells % args.progress_every == 0:
                        print(
                            f"completed {new_cells} new method-cells: "
                            f"{distribution.name}, n={n}, "
                            f"rep={replication + 1}, {method_name}",
                            flush=True,
                        )
                        with checkpoint_path.open(
                            "w", encoding="utf-8"
                        ) as stream:
                            json.dump(
                                {
                                    "rep_limit": args.rep_limit,
                                    "solvency_c": solvency_c,
                                    "score_work_budget": (
                                        args.score_work_budget
                                    ),
                                    "cells": completed,
                                },
                                stream,
                            )

    with checkpoint_path.open("w", encoding="utf-8") as stream:
        json.dump(
            {
                "rep_limit": args.rep_limit,
                "solvency_c": solvency_c,
                "score_work_budget": args.score_work_budget,
                "cells": completed,
            },
            stream,
        )

    for column in (
        "adjacent_component_width",
        "full_set_diameter",
        "largest_component_width",
        "topology_component_count",
        "topology_scan_points",
        "topology_point_budget_reached",
    ):
        if column not in frame:
            frame[column] = np.nan

    # Gaffke is an ordinary interval, so all three widths coincide.
    gaffke = frame["method"] == "Gaffke"
    frame.loc[gaffke, "adjacent_component_width"] = frame.loc[
        gaffke, "width"
    ]
    frame.loc[gaffke, "full_set_diameter"] = frame.loc[
        gaffke, "width"
    ]
    frame.loc[gaffke, "largest_component_width"] = frame.loc[
        gaffke, "width"
    ]
    frame.loc[gaffke, "topology_component_count"] = 1.0

    for key, diagnostics in completed.items():
        distribution, n, replication, method_name = key.split("|", 3)
        mask = (
            (frame["distribution"] == distribution)
            & (frame["n"] == int(n))
            & (frame["rep"] == int(replication))
            & (frame["method"] == method_name)
        )
        if int(mask.sum()) != 1:
            raise RuntimeError(f"could not uniquely locate {key}")
        saved_adjacent = frame.loc[
            mask, "adjacent_component_width"
        ].iloc[0]
        old_width = (
            float(saved_adjacent)
            if np.isfinite(saved_adjacent)
            else float(frame.loc[mask, "width"].iloc[0])
        )
        diameter = float(diagnostics["hull_width"])
        largest = float(diagnostics["largest_component_width"])
        variance = float(frame.loc[mask, "true_variance"].iloc[0])
        frame.loc[mask, "adjacent_component_width"] = old_width
        frame.loc[mask, "full_set_diameter"] = diameter
        frame.loc[mask, "largest_component_width"] = largest
        frame.loc[mask, "topology_component_count"] = float(
            diagnostics["component_count"]
        )
        frame.loc[mask, "topology_scan_points"] = float(
            diagnostics["scan_point_count"]
        )
        frame.loc[mask, "topology_point_budget_reached"] = float(
            diagnostics["point_budget_reached"]
        )
        frame.loc[mask, "lower"] = diagnostics["lower"]
        frame.loc[mask, "upper"] = diagnostics["upper"]
        frame.loc[mask, "width"] = diameter
        frame.loc[mask, "sqrt_n_width"] = math.sqrt(int(n)) * diameter
        frame.loc[mask, "normalized_halfwidth"] = (
            math.sqrt(int(n)) * diameter / (2.0 * math.sqrt(variance))
        )
        true_mean = float(frame.loc[mask, "true_mean"].iloc[0])
        frame.loc[mask, "covered"] = (
            diagnostics["lower"] <= true_mean <= diagnostics["upper"]
        )
        frame.loc[mask, "backend"] = "adaptive-topology-multiresolution"

    frame.to_csv(results_path, index=False)
    frame.to_csv(results_dir / "results_checkpoint.csv", index=False)
    summarize(frame).to_csv(results_dir / "summary.csv", index=False)
    config["base_inversion_width"] = config.get("reported_width")
    config["reported_width"] = (
        "mesh-resolved full-set diameter; largest-component width also stored"
    )
    config["topology_inversion"] = {
        "rep_limit": args.rep_limit,
        "score_work_budget": args.score_work_budget,
        "reported_widths": [
            "full-set diameter",
            "largest accepted-component width",
        ],
        "large_n_screen": (
            "known local component plus 31 interior probes and "
            "endpoint-anchored outside probes at standard-error distances "
            "1/8, 1/4, 1/2, 1, 2, ...; full multiresolution inversion on "
            "any interior rejection or outside acceptance"
        ),
        "finite_mesh": True,
        "checkpoint": checkpoint_path.name,
    }
    with (results_dir / "config.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(config, stream, indent=2)
    print(f"updated {results_path}")


if __name__ == "__main__":
    main()
