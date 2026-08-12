#!/usr/bin/env python3
"""Large-sample comparison of matched betting feedbacks and Gaffke's CI.

This experiment extends the paper's chronological-feedback ablation.  It keeps
the matched feedback rules, adds candidate-dependent and common-clock
Efficient betting and the equal-tail Gaffke interval, and focuses on n from
10^3 through 10^7.  Two otherwise identical figures are produced:
one with pointwise empirical 10--90% width intervals and one with means only.

The generic paper code scans candidate means across all of [0,1].  At the
large horizons used here, every endpoint is local to the sample mean.  This
script therefore brackets each adjacent crossing from the sample mean and
then applies paired bisection.  The underlying wealth functions are imported
unchanged from betting.py.  The --audit flag compares this local inversion
against the original global inversion on a small sample.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from numba import njit, prange
from scipy.stats import beta as beta_dist
from scipy.stats import norm


BETTING_DIR = Path(__file__).resolve().parents[1]
PAPER_PLOT_DIR = BETTING_DIR.parent / "paper" / "plots"
if str(BETTING_DIR) not in sys.path:
    sys.path.insert(0, str(BETTING_DIR))

import betting as methods  # noqa: E402
from gaffke_comparison.compare_star_probit_gaffke import (  # noqa: E402
    DISTRIBUTIONS,
    gaffke_ci,
)


METHOD_ORDER = (
    "Square-root feedback",
    "Capped original feedback",
    "Squared-hinge feedback",
    "Target-capped feedback",
    "Regularized Efficient betting",
    "Efficient betting",
    "Common-clock Efficient betting",
    "STaR betting",
    "Common-clock STaR betting",
    "Gaffke",
)

MAIN_METHOD_ORDER = tuple(
    method_name for method_name in METHOD_ORDER
    if method_name not in {
        "Capped original feedback",
        "Target-capped feedback",
        "Regularized Efficient betting",
        # The candidate-dependent version is retained in the dedicated
        # convexification appendix plot below.  The main figure uses the
        # pathwise interval-valued common-clock implementation.
        "Efficient betting",
        "STaR betting",
        "Common-clock STaR betting",
    }
)

TARGET_CAPPING_METHOD_ORDER = (
    "Square-root feedback",
    "Capped original feedback",
    "Squared-hinge feedback",
    "Target-capped feedback",
    "Efficient betting",
)

METHOD_STYLES = {
    "Square-root feedback": ("darkorange", "P"),
    "Capped original feedback": ("teal", "X"),
    "Squared-hinge feedback": ("crimson", "D"),
    "Target-capped feedback": ("deeppink", "*"),
    "Efficient betting": ("#8c564b", "^"),
    "Common-clock Efficient betting": ("#2ca02c", "h"),
    "STaR betting": ("#c44e52", "P"),
    "Common-clock STaR betting": ("darkorange", "o"),
    "Regularized Efficient betting": ("purple", "v"),
    "Gaffke": ("#1976b9", "o"),
}

METHOD_LABELS = {
    "Regularized Efficient betting": r"Efficient betting ($b_n=n^{2/3}$)",
    "Efficient betting": "Efficient betting",
    "Common-clock Efficient betting": "Efficient betting (shared estimator)",
    "STaR betting": "STaR (candidate-centered)",
    "Common-clock STaR betting": "STaR (shared estimator)",
    "Square-root feedback": "Original STaR-Bets (square-root)",
    "Capped original feedback": "Capped original STaR",
}

FEEDBACK_METHODS = (
    "Square-root feedback",
    "Squared-hinge feedback",
    "Target-capped feedback",
)
FEEDBACK_KINDS = (0, 1, 2)

SAMPLE_SIZES = (
    1_000,
    3_000,
    10_000,
    30_000,
    100_000,
    300_000,
    1_000_000,
    3_000_000,
    10_000_000,
)

BASE_REPS_BY_N = {
    1_000: 50,
    3_000: 50,
    10_000: 50,
    30_000: 40,
    100_000: 40,
    300_000: 30,
    1_000_000: 20,
    3_000_000: 15,
    10_000_000: 10,
}
REPS_BY_N = {
    n: max(base_reps, 30)
    for n, base_reps in BASE_REPS_BY_N.items()
}
FIGURE3_METHODS = {
    "Efficient betting",
    "Common-clock Efficient betting",
    "STaR betting",
    "Common-clock STaR betting",
}


@njit(cache=True)
def _power_sums(x: np.ndarray) -> tuple[float, float, float, float]:
    """Return the first four raw power sums without large temporaries."""
    s1 = 0.0
    s2 = 0.0
    s3 = 0.0
    s4 = 0.0
    for value in x:
        value2 = value * value
        s1 += value
        s2 += value2
        s3 += value2 * value
        s4 += value2 * value2
    return s1, s2, s3, s4


def _dirichlet_cf_from_sums(
    n: int,
    sums: tuple[float, float, float, float],
    endpoint: float,
    q: float,
) -> float:
    """Cornish--Fisher quantile for x augmented by one endpoint knot."""
    m = float(n + 1)
    s1, s2, s3, s4 = sums
    s1 += endpoint
    s2 += endpoint**2
    s3 += endpoint**3
    s4 += endpoint**4
    mu = s1 / m

    p2 = max(s2 - m * mu * mu, 0.0)
    if p2 <= 1.0e-30:
        return mu
    p3 = s3 - 3.0 * mu * s2 + 2.0 * m * mu**3
    p4 = s4 - 4.0 * mu * s3 + 6.0 * mu * mu * s2 - 3.0 * m * mu**4

    var = p2 / (m * (m + 1.0))
    sd = math.sqrt(max(var, 0.0))
    cm3 = 2.0 * p3 / (m * (m + 1.0) * (m + 2.0))
    cm4 = (
        3.0 * p2 * p2 + 6.0 * p4
    ) / (m * (m + 1.0) * (m + 2.0) * (m + 3.0))
    skew = cm3 / sd**3
    excess = cm4 / sd**4 - 3.0

    z = float(norm.ppf(q))
    z_cf = (
        z
        + (skew / 6.0) * (z * z - 1.0)
        + (excess / 24.0) * (z**3 - 3.0 * z)
        - (skew * skew / 36.0) * (2.0 * z**3 - 5.0 * z)
    )
    return mu + sd * z_cf


def fast_gaffke_ci(
    x: np.ndarray,
    delta: float,
    binary: bool,
    exact_cutoff: int,
) -> tuple[float, float, str]:
    """Gaffke interval with an O(n)-memory-safe large-n approximation."""
    n = x.size
    q = delta / 2.0
    if binary:
        k = int(np.sum(x))
        lower = 0.0 if k == 0 else float(beta_dist.ppf(q, k, n + 1 - k))
        upper = 1.0 if k == n else float(beta_dist.ppf(1.0 - q, k + 1, n - k))
        return lower, upper, "beta-exact"

    if n <= exact_cutoff:
        return gaffke_ci(x, delta=delta, exact_cutoff=exact_cutoff)

    sums = _power_sums(x)
    lower = _dirichlet_cf_from_sums(n, sums, endpoint=0.0, q=q)
    upper = _dirichlet_cf_from_sums(n, sums, endpoint=1.0, q=1.0 - q)
    return (
        float(np.clip(lower, 0.0, 1.0)),
        float(np.clip(upper, 0.0, 1.0)),
        "cornish-fisher-moments",
    )


def invert_local_component(
    x: np.ndarray,
    rejection_score: Callable[[float], float],
    batch_rejection_score: Callable[[np.ndarray], np.ndarray] | None = None,
    initial_se_multiplier: float = 3.25,
) -> tuple[float, float, bool, int]:
    """Invert the accepted component adjacent to the sample mean.

    ``rejection_score(m)`` is negative for acceptance and nonnegative for
    rejection.  Search starts a few empirical standard errors from the sample
    mean and expands geometrically only when necessary.
    """
    center = float(np.mean(x))
    n = x.size
    empirical_sd = float(np.std(x, ddof=1)) if n > 1 else 0.0
    standard_error = max(empirical_sd / math.sqrt(n), 1.0 / n)
    initial_step = initial_se_multiplier * standard_error
    center_score = float(rejection_score(center))
    evaluations = 1
    if center_score >= 0.0:
        return center, center, True, evaluations

    def evaluate(candidates: np.ndarray) -> np.ndarray:
        nonlocal evaluations
        candidates = np.asarray(candidates, dtype=np.float64)
        evaluations += candidates.size
        if batch_rejection_score is None:
            return np.asarray(
                [rejection_score(float(value)) for value in candidates],
                dtype=np.float64,
            )
        values = np.asarray(batch_rejection_score(candidates), dtype=np.float64)
        if values.shape != candidates.shape:
            raise ValueError("batch_rejection_score returned the wrong shape")
        return values

    # Brackets are stored in increasing mean order, with endpoint scores of
    # opposite sign.  Both sides are evaluated in one parallel batch.
    inner = {-1.0: center, 1.0: center}
    step = {-1.0: initial_step, 1.0: initial_step}
    brackets: dict[float, list[float]] = {}
    boundary_values: dict[float, float] = {}
    unresolved = [-1.0, 1.0]
    for _ in range(12):
        if not unresolved:
            break
        candidates = np.asarray(
            [np.clip(center + direction * step[direction], 0.0, 1.0)
             for direction in unresolved],
            dtype=np.float64,
        )
        values = evaluate(candidates)
        next_unresolved: list[float] = []
        for direction, current, value in zip(unresolved, candidates, values):
            current = float(current)
            value = float(value)
            if value >= 0.0:
                if direction < 0:
                    brackets[direction] = [current, inner[direction], value, center_score]
                else:
                    brackets[direction] = [inner[direction], current, center_score, value]
                continue
            outer_limit = 0.0 if direction < 0 else 1.0
            if current == outer_limit:
                boundary_values[direction] = outer_limit
                continue
            inner[direction] = current
            step[direction] *= 1.8
            next_unresolved.append(direction)
        unresolved = next_unresolved
    if unresolved:
        raise RuntimeError("failed to bracket a local confidence endpoint")

    root_tol = max(2.0e-10, 2.0e-4 * standard_error)
    active = sorted(brackets)
    for _ in range(24):
        active = [
            direction
            for direction in active
            if brackets[direction][1] - brackets[direction][0] > root_tol
        ]
        if not active:
            break
        midpoints = np.asarray(
            [0.5 * (brackets[d][0] + brackets[d][1]) for d in active]
        )
        values = evaluate(midpoints)
        for direction, midpoint, value in zip(active, midpoints, values):
            left, right, f_left, f_right = brackets[direction]
            if f_left * value <= 0.0:
                brackets[direction] = [left, float(midpoint), f_left, float(value)]
            else:
                brackets[direction] = [float(midpoint), right, float(value), f_right]

    endpoints = {}
    for direction in (-1.0, 1.0):
        if direction in boundary_values:
            endpoints[direction] = boundary_values[direction]
        else:
            endpoints[direction] = 0.5 * (
                brackets[direction][0] + brackets[direction][1]
            )
    return endpoints[-1.0], endpoints[1.0], False, evaluations


@njit(cache=True, parallel=True)
def _feedback_scores_by_kind(
    x: np.ndarray,
    candidate_means: np.ndarray,
    feedback_kinds: np.ndarray,
    delta: float,
) -> np.ndarray:
    """Evaluate different feedback maps and candidate means concurrently."""
    threshold = 1.0 / delta
    out = np.empty(candidate_means.size)
    for index in prange(candidate_means.size):
        out[index] = methods._compute_M_recalculating_feedback(
            x,
            candidate_means[index],
            delta,
            1.0,
            1.0,
            int(feedback_kinds[index]),
        ) - threshold
    return out


def invert_feedback_components(
    x: np.ndarray,
    delta: float,
    initial_se_multiplier: float = 3.25,
) -> dict[str, tuple[float, float, bool, int]]:
    """Invert all three deterministic feedback maps in shared parallel batches."""
    center = float(np.mean(x))
    n = x.size
    empirical_sd = float(np.std(x, ddof=1)) if n > 1 else 0.0
    standard_error = max(empirical_sd / math.sqrt(n), 1.0 / n)
    initial_step = initial_se_multiplier * standard_error
    method_count = len(FEEDBACK_METHODS)
    evaluations = np.ones(method_count, dtype=np.int64)

    center_scores = _feedback_scores_by_kind(
        x,
        np.full(method_count, center),
        np.asarray(FEEDBACK_KINDS, dtype=np.int64),
        delta,
    )
    rejected_center = center_scores >= 0.0

    inner = {(method, direction): center
             for method in range(method_count) for direction in (-1, 1)}
    step = {(method, direction): initial_step
            for method in range(method_count) for direction in (-1, 1)}
    brackets: dict[tuple[int, int], list[float]] = {}
    boundary_values: dict[tuple[int, int], float] = {}
    unresolved = [
        (method, direction)
        for method in range(method_count)
        if not rejected_center[method]
        for direction in (-1, 1)
    ]

    for _ in range(12):
        if not unresolved:
            break
        means = np.asarray([
            np.clip(center + direction * step[(method, direction)], 0.0, 1.0)
            for method, direction in unresolved
        ])
        kinds = np.asarray(
            [FEEDBACK_KINDS[method] for method, _ in unresolved],
            dtype=np.int64,
        )
        values = _feedback_scores_by_kind(x, means, kinds, delta)
        next_unresolved: list[tuple[int, int]] = []
        for key, current, value in zip(unresolved, means, values):
            method, direction = key
            evaluations[method] += 1
            current = float(current)
            value = float(value)
            if value >= 0.0:
                if direction < 0:
                    brackets[key] = [current, inner[key], value, float(center_scores[method])]
                else:
                    brackets[key] = [inner[key], current, float(center_scores[method]), value]
                continue
            outer_limit = 0.0 if direction < 0 else 1.0
            if current == outer_limit:
                boundary_values[key] = outer_limit
                continue
            inner[key] = current
            step[key] *= 1.8
            next_unresolved.append(key)
        unresolved = next_unresolved
    if unresolved:
        raise RuntimeError("failed to bracket a feedback confidence endpoint")

    root_tol = max(2.0e-10, 2.0e-4 * standard_error)
    active = sorted(brackets)
    for _ in range(24):
        active = [key for key in active if brackets[key][1] - brackets[key][0] > root_tol]
        if not active:
            break
        means = np.asarray([0.5 * (brackets[key][0] + brackets[key][1]) for key in active])
        kinds = np.asarray(
            [FEEDBACK_KINDS[key[0]] for key in active], dtype=np.int64
        )
        values = _feedback_scores_by_kind(x, means, kinds, delta)
        for key, midpoint, value in zip(active, means, values):
            method = key[0]
            evaluations[method] += 1
            left, right, f_left, f_right = brackets[key]
            if f_left * value <= 0.0:
                brackets[key] = [left, float(midpoint), f_left, float(value)]
            else:
                brackets[key] = [float(midpoint), right, float(value), f_right]

    output: dict[str, tuple[float, float, bool, int]] = {}
    for method, method_name in enumerate(FEEDBACK_METHODS):
        if rejected_center[method]:
            output[method_name] = (center, center, True, int(evaluations[method]))
            continue
        endpoints = []
        for direction in (-1, 1):
            key = (method, direction)
            if key in boundary_values:
                endpoints.append(boundary_values[key])
            else:
                endpoints.append(0.5 * (brackets[key][0] + brackets[key][1]))
        output[method_name] = (
            float(endpoints[0]),
            float(endpoints[1]),
            False,
            int(evaluations[method]),
        )
    return output


def _betting_interval(
    x: np.ndarray,
    method_name: str,
    delta: float,
    u_plus: float,
    u_minus: float,
    solvency_c: float,
) -> tuple[float, float, bool, int]:
    threshold = 1.0 / delta

    if method_name == "STaR betting":
        alpha = delta / 2.0

        def randomized_score(m: float) -> float:
            plus, minus = methods.compute_M_star_arms(
                x, m, delta, c=solvency_c
            )
            return max(
                alpha * plus / u_plus,
                alpha * minus / u_minus,
            ) - 1.0

        def randomized_scores(ms: np.ndarray) -> np.ndarray:
            return methods._star_randomized_scores(
                x, ms, delta, u_plus, u_minus, solvency_c
            ) - 1.0

        return invert_local_component(
            x, randomized_score, randomized_scores
        )
    if method_name == "Common-clock STaR betting":
        return methods.star_common_clock_batched_ci_endpoints(
            x,
            delta,
            randomizers=(u_plus, u_minus),
            return_diagnostics=True,
            c=solvency_c,
        )
    if method_name == "Square-root feedback":
        return invert_local_component(
            x,
            lambda m: methods.compute_M_star(x, m, delta) - threshold,
            lambda ms: methods._recalculating_feedback_scores(x, ms, delta, 0)
            - threshold,
        )
    if method_name == "Capped original feedback":
        return invert_local_component(
            x,
            lambda m: methods.compute_M_capped_exponential_feedback_star(
                x, m, delta
            ) - threshold,
            lambda ms: methods._recalculating_feedback_scores(x, ms, delta, 3)
            - threshold,
        )
    if method_name == "Squared-hinge feedback":
        return invert_local_component(
            x,
            lambda m: methods.compute_M_hinge_feedback_star(x, m, delta)
            - threshold,
            lambda ms: methods._recalculating_feedback_scores(x, ms, delta, 1)
            - threshold,
        )
    if method_name == "Target-capped feedback":
        return invert_local_component(
            x,
            lambda m: methods.compute_M_capped_feedback_star(x, m, delta)
            - threshold,
            lambda ms: methods._recalculating_feedback_scores(x, ms, delta, 2)
            - threshold,
        )
    if method_name in (
        "Regularized Efficient betting",
        "Efficient betting",
    ):
        alpha = delta / 2.0
        buffer_rounds = (
            float(x.size) ** (2.0 / 3.0)
            if method_name == "Regularized Efficient betting"
            else 0.0
        )

        def randomized_score(m: float) -> float:
            plus, minus = methods.compute_M_probit_star_arms(
                x, m, delta, c=solvency_c,
                buffer_rounds=buffer_rounds
            )
            return max(alpha * plus / u_plus, alpha * minus / u_minus) - 1.0

        def randomized_scores(ms: np.ndarray) -> np.ndarray:
            return methods._probit_randomized_scores(
                x, ms, delta, buffer_rounds, u_plus, u_minus, solvency_c
            ) - 1.0

        return invert_local_component(x, randomized_score, randomized_scores)
    if method_name == "Common-clock Efficient betting":
        return methods.probit_common_clock_batched_ci_endpoints(
            x,
            delta,
            buffer_rounds=0.0,
            randomizers=(u_plus, u_minus),
            return_diagnostics=True,
            c=solvency_c,
        )
    raise ValueError(f"unknown method {method_name}")


def audit_local_inversion(
    delta: float, seed: int, solvency_c: float
) -> None:
    """Check local endpoints against the paper's global inversion helpers."""
    rng = np.random.default_rng(seed)
    x = rng.beta(2.0, 2.0, 2_000)
    u_plus, u_minus = rng.uniform(size=2)
    feedback_local = invert_feedback_components(x, delta)
    comparisons = (
        (
            "Square-root feedback",
            lambda: methods.star_ci_endpoints(x, delta),
        ),
        (
            "STaR betting",
            lambda: invert_local_component(
                x,
                lambda m: max(
                    delta
                    * methods.compute_M_star_arms(
                        x, m, delta, c=solvency_c
                    )[0]
                    / (2.0 * u_plus),
                    delta
                    * methods.compute_M_star_arms(
                        x, m, delta, c=solvency_c
                    )[1]
                    / (2.0 * u_minus),
                ) - 1.0,
            )[:2],
        ),
        (
            "Common-clock STaR betting",
            lambda: methods.star_common_clock_ci_endpoints(
                x,
                delta,
                randomizers=(u_plus, u_minus),
                c=solvency_c,
            )[:2],
        ),
        (
            "Capped original feedback",
            lambda: methods.capped_exponential_feedback_star_ci_endpoints(
                x, delta
            ),
        ),
        (
            "Squared-hinge feedback",
            lambda: methods.hinge_feedback_star_ci_endpoints(x, delta),
        ),
        (
            "Target-capped feedback",
            lambda: methods.capped_feedback_star_ci_endpoints(x, delta),
        ),
        (
            "Regularized Efficient betting",
            lambda: methods.probit_star_ci_endpoints(
                x,
                delta,
                buffer_rounds=float(x.size) ** (2.0 / 3.0),
                randomizers=(u_plus, u_minus),
                c=solvency_c,
            ),
        ),
        (
            "Efficient betting",
            lambda: methods.probit_star_ci_endpoints(
                x,
                delta,
                buffer_rounds=0.0,
                randomizers=(u_plus, u_minus),
                c=solvency_c,
            ),
        ),
        (
            "Common-clock Efficient betting",
            lambda: methods.probit_common_clock_ci_endpoints(
                x,
                delta,
                buffer_rounds=0.0,
                randomizers=(u_plus, u_minus),
                c=solvency_c,
            )[:2],
        ),
    )

    for name, global_fn in comparisons:
        if name in feedback_local:
            local = feedback_local[name][:2]
        else:
            local = _betting_interval(
                x, name, delta, u_plus, u_minus, solvency_c
            )[:2]
        global_endpoints = global_fn()
        error = max(abs(local[0] - global_endpoints[0]), abs(local[1] - global_endpoints[1]))
        print(f"audit {name:36s} max endpoint error={error:.3e}")
        if error > 2.0e-6:
            raise AssertionError(f"local inversion disagrees for {name}")


def _path_max_n(rep: int, sample_sizes: list[int], reps_by_n: dict[int, int]) -> int:
    eligible = [n for n in sample_sizes if rep < reps_by_n[n]]
    return max(eligible) if eligible else 0


def _is_binary_distribution(name: str) -> bool:
    return name.startswith("Bernoulli")


def _record_row(
    distribution: str,
    true_mean: float,
    true_variance: float,
    n: int,
    rep: int,
    method_name: str,
    lower: float,
    upper: float,
    empty: bool,
    backend: str,
    seconds: float,
    score_evaluations: int,
) -> dict[str, object]:
    width = max(float(upper - lower), 0.0)
    sigma = math.sqrt(true_variance)
    return {
        "distribution": distribution,
        "true_mean": true_mean,
        "true_variance": true_variance,
        "n": n,
        "rep": rep,
        "method": method_name,
        "lower": lower,
        "upper": upper,
        "width": width,
        "sqrt_n_width": math.sqrt(n) * width,
        "normalized_halfwidth": math.sqrt(n) * width / (2.0 * sigma),
        "covered": lower <= true_mean <= upper,
        "empty_center_component": empty,
        "backend": backend,
        "runtime_seconds": seconds,
        "score_evaluations": score_evaluations,
    }


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "largest_component_width" not in df:
        df["largest_component_width"] = df["width"]
    else:
        df["largest_component_width"] = df[
            "largest_component_width"
        ].fillna(df["width"])
    df["sqrt_n_largest_component_width"] = (
        np.sqrt(df["n"].to_numpy(dtype=float))
        * df["largest_component_width"].to_numpy(dtype=float)
    )
    return (
        df.groupby(["distribution", "n", "method"], as_index=False)
        .agg(
            replications=("width", "size"),
            coverage=("covered", "mean"),
            mean_width=("width", "mean"),
            q10_width=("width", lambda z: np.quantile(z, 0.10)),
            q90_width=("width", lambda z: np.quantile(z, 0.90)),
            mean_sqrt_n_width=("sqrt_n_width", "mean"),
            q10_sqrt_n_width=(
                "sqrt_n_width", lambda z: np.quantile(z, 0.10)
            ),
            q90_sqrt_n_width=(
                "sqrt_n_width", lambda z: np.quantile(z, 0.90)
            ),
            mean_sqrt_n_largest_component=(
                "sqrt_n_largest_component_width", "mean"
            ),
            mean_normalized_halfwidth=("normalized_halfwidth", "mean"),
            mean_runtime_seconds=("runtime_seconds", "mean"),
            mean_score_evaluations=("score_evaluations", "mean"),
            empty_rate=("empty_center_component", "mean"),
        )
    )


def make_plots(df: pd.DataFrame, output: Path, delta: float) -> list[Path]:
    plot_dir = output / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize(df)
    has_candidate_topology = (
        "full_set_diameter" in df
        and df.loc[
            df["method"] == "Efficient betting",
            "full_set_diameter",
        ].notna().any()
    )
    if has_candidate_topology:
        topology_betting = df[
            (df["method"] == "Efficient betting")
            & df["full_set_diameter"].notna()
        ]
        paired_keys = topology_betting[
            ["distribution", "n", "rep"]
        ].drop_duplicates()
        topology_df = df[df["method"].isin(MAIN_METHOD_ORDER)].merge(
            paired_keys,
            on=["distribution", "n", "rep"],
            how="inner",
        )
        main_summary = summarize(topology_df)
    else:
        main_summary = summary
    outputs: list[Path] = []

    for show_intervals in (True, False):
        fig, axes = plt.subplots(3, 3, figsize=(13, 11), sharex=True)
        legend_handles = None
        legend_labels = None

        for axis, dist in zip(axes.ravel(), DISTRIBUTIONS):
            ddf = main_summary[
                main_summary["distribution"] == dist.name
            ]
            n_values = np.sort(ddf["n"].unique())
            gaussian_limit = 2.0 * math.sqrt(dist.variance) * norm.isf(delta / 2.0)
            axis.plot(
                n_values,
                np.full(n_values.size, gaussian_limit),
                color="black",
                linestyle=":",
                linewidth=1.8,
                label="Gaussian limit",
            )

            for method_name in MAIN_METHOD_ORDER:
                mdf = ddf[ddf["method"] == method_name].sort_values("n")
                color, marker = METHOD_STYLES[method_name]
                axis.plot(
                    mdf["n"],
                    mdf["mean_sqrt_n_width"],
                    color=color,
                    marker=marker,
                    markersize=5.0,
                    linewidth=2.0,
                    label=METHOD_LABELS.get(method_name, method_name),
                )
                if show_intervals:
                    axis.fill_between(
                        mdf["n"],
                        mdf["q10_sqrt_n_width"],
                        mdf["q90_sqrt_n_width"],
                        color=color,
                        alpha=0.09,
                        linewidth=0.0,
                    )

            axis.set_xscale("log")
            axis.set_title(dist.name)
            axis.set_xlabel("sample size $n$")
            axis.set_ylabel(r"$\sqrt{n}\times$ CI width")
            axis.grid(True, linestyle="--", alpha=0.3)
            if legend_handles is None:
                legend_handles, legend_labels = axis.get_legend_handles_labels()

        qualifier = "means with empirical 10--90% intervals" if show_intervals else "means only"
        fig.suptitle(
            f"Large-sample convex-CI and Gaffke comparison: {qualifier} "
            f"($1-\\delta={1.0-delta:.2f}$)",
            fontsize=15,
        )
        fig.legend(
            legend_handles,
            legend_labels,
            loc="lower center",
            ncol=3,
            fontsize=9.2,
            frameon=False,
        )
        fig.tight_layout(rect=(0.0, 0.09, 1.0, 0.95))
        suffix = "bands" if show_intervals else "means"
        destination = plot_dir / f"scaled_width_large_gaffke_{suffix}.png"
        fig.savefig(destination, dpi=220, bbox_inches="tight")
        plt.close(fig)
        outputs.append(destination)

    unbuffered_name = "Efficient betting"
    pair_columns = ["distribution", "n", "rep"]
    pair_keys = df.loc[
        df["method"] == unbuffered_name, pair_columns
    ].drop_duplicates()
    regularized_name = "Regularized Efficient betting"
    regularization_methods = (regularized_name, unbuffered_name)
    regularization_df = df.loc[
        df["method"].isin(regularization_methods)
    ].merge(pair_keys, on=pair_columns, how="inner")
    regularization_summary = summarize(regularization_df)

    fig, axes = plt.subplots(3, 3, figsize=(13, 11), sharex=True)
    legend_handles = None
    legend_labels = None
    regularization_labels = {
        regularized_name: r"Regularized Efficient betting ($b_n=n^{2/3}$)",
        unbuffered_name: r"Efficient betting ($b_n=0$)",
    }
    for axis, dist in zip(axes.ravel(), DISTRIBUTIONS):
        ddf = regularization_summary[
            regularization_summary["distribution"] == dist.name
        ]
        n_values = np.sort(ddf["n"].unique())
        gaussian_limit = 2.0 * math.sqrt(dist.variance) * norm.isf(
            delta / 2.0
        )
        axis.plot(
            n_values,
            np.full(n_values.size, gaussian_limit),
            color="black",
            linestyle=":",
            linewidth=1.8,
            label="Gaussian limit",
        )
        for method_name in regularization_methods:
            mdf = ddf[ddf["method"] == method_name].sort_values("n")
            color, marker = METHOD_STYLES[method_name]
            axis.plot(
                mdf["n"],
                mdf["mean_sqrt_n_width"],
                color=color,
                marker=marker,
                markersize=5.5,
                linewidth=2.2,
                label=regularization_labels[method_name],
            )
        axis.set_xscale("log")
        axis.set_title(dist.name)
        axis.set_xlabel("sample size $n$")
        axis.set_ylabel(r"$\sqrt{n}\times$ CI width")
        axis.grid(True, linestyle="--", alpha=0.3)
        if legend_handles is None:
            legend_handles, legend_labels = axis.get_legend_handles_labels()

    fig.suptitle(
        "Residual-variance regularization: paired means "
        f"($1-\\delta={1.0-delta:.2f}$)",
        fontsize=15,
    )
    fig.legend(
        legend_handles,
        legend_labels,
        loc="lower center",
        ncol=3,
        fontsize=9.5,
        frameon=False,
    )
    fig.tight_layout(rect=(0.0, 0.08, 1.0, 0.95))
    destination = plot_dir / "scaled_width_probit_regularization_means.png"
    fig.savefig(destination, dpi=220, bbox_inches="tight")
    plt.close(fig)
    outputs.append(destination)

    focused_styles = {
        "STaR betting": ("darkorange", "P", "--", False, 5.2),
        "Common-clock STaR betting": (
            "darkorange", "P", "-", True, 3.8,
        ),
        unbuffered_name: ("#2ca02c", "h", "--", False, 5.2),
        "Common-clock Efficient betting": (
            "#2ca02c", "h", "-", True, 3.8,
        ),
    }
    fig, axes = plt.subplots(3, 3, figsize=(13, 11), sharex=True)
    for axis, dist in zip(axes.ravel(), DISTRIBUTIONS):
        ddf = summary[summary["distribution"] == dist.name]
        n_values = np.sort(ddf["n"].unique())
        gaussian_limit = 2.0 * math.sqrt(dist.variance) * norm.isf(
            delta / 2.0
        )
        axis.plot(
            n_values,
            np.full(n_values.size, gaussian_limit),
            color="black",
            linestyle=":",
            linewidth=1.8,
            label="Gaussian limit",
        )
        for method_name in TARGET_CAPPING_METHOD_ORDER:
            mdf = ddf[ddf["method"] == method_name].sort_values("n")
            color, marker = METHOD_STYLES[method_name]
            axis.plot(
                mdf["n"],
                mdf["mean_sqrt_n_width"],
                color=color,
                marker=marker,
                markersize=5.0,
                linewidth=2.0,
                label=METHOD_LABELS.get(method_name, method_name),
            )
        axis.set_xscale("log")
        axis.set_title(dist.name)
        axis.set_xlabel("sample size $n$")
        axis.set_ylabel(r"$\sqrt{n}\times$ CI width")
        axis.grid(True, linestyle="--", alpha=0.3)
        if legend_handles is None:
            legend_handles, legend_labels = axis.get_legend_handles_labels()

    fig.suptitle(
        "Target-capping feedback comparison: means only "
        f"($1-\\delta={1.0-delta:.2f}$)",
        fontsize=15,
    )
    fig.legend(
        legend_handles,
        legend_labels,
        loc="lower center",
        ncol=3,
        fontsize=9.2,
        frameon=False,
    )
    fig.tight_layout(rect=(0.0, 0.09, 1.0, 0.95))
    destination = plot_dir / "scaled_width_target_capping_means.png"
    fig.savefig(destination, dpi=220, bbox_inches="tight")
    plt.close(fig)
    outputs.append(destination)

    unbuffered_name = "Efficient betting"
    focused_methods = (
        "STaR betting",
        "Common-clock STaR betting",
        unbuffered_name,
        "Common-clock Efficient betting",
    )
    pair_columns = ["distribution", "n", "rep"]
    focused_pair_mask = df["method"] == unbuffered_name
    if has_candidate_topology:
        focused_pair_mask &= df["full_set_diameter"].notna()
    pair_keys = df.loc[
        focused_pair_mask, pair_columns
    ].drop_duplicates()
    focused_df = df.loc[df["method"].isin(focused_methods)].merge(
        pair_keys, on=pair_columns, how="inner"
    )
    focused_summary = summarize(focused_df)

    fig, axes = plt.subplots(3, 3, figsize=(13, 11), sharex=True)
    legend_handles = None
    legend_labels = None
    for axis, dist in zip(axes.ravel(), DISTRIBUTIONS):
        ddf = focused_summary[
            focused_summary["distribution"] == dist.name
        ]
        for method_name in focused_methods:
            mdf = ddf[ddf["method"] == method_name].sort_values("n")
            color, marker, linestyle, filled_marker, marker_size = (
                focused_styles[method_name]
            )
            axis.plot(
                mdf["n"],
                mdf["mean_sqrt_n_width"],
                color=color,
                marker=marker,
                markerfacecolor=color if filled_marker else "none",
                markeredgecolor=color,
                markeredgewidth=0.9,
                markersize=marker_size,
                linestyle=linestyle,
                linewidth=1.9,
                label="_nolegend_",
            )
            if method_name == unbuffered_name:
                axis.plot(
                    mdf["n"],
                    mdf["mean_sqrt_n_largest_component"],
                    color=color,
                    linestyle=":",
                    linewidth=1.6,
                    label="_nolegend_",
                )
        axis.set_xscale("log")
        axis.set_title(dist.name)
        axis.set_xlabel("sample size $n$")
        axis.set_ylabel(r"$\sqrt{n}\times$ CI width")
        axis.grid(True, linestyle="--", alpha=0.3)

    construction_handles = [
        Line2D(
            [0], [0], color=color, marker=marker, lw=2,
            markerfacecolor=color, markeredgecolor=color, ms=4.5,
        )
        for color, marker in (("darkorange", "P"), ("#2ca02c", "h"))
    ]
    construction_labels = ["STaR betting", "Efficient betting"]
    estimator_handles = [
        Line2D(
            [0], [0], color="0.25", ls="--", marker="o", lw=2,
            markerfacecolor="none", markeredgecolor="0.25", ms=5.2,
        ),
        Line2D(
            [0], [0], color="0.25", ls="-", marker="o", lw=2,
            markerfacecolor="0.25", markeredgecolor="0.25", ms=3.8,
        ),
    ]
    estimator_labels = ["Candidate-specific", "Shared"]
    component_handles = [
        Line2D([0], [0], color="#2ca02c", ls=":", lw=1.7),
    ]
    component_labels = ["Largest connected piece"]
    fig.suptitle(
        "STaR and Efficient betting: mean scaled confidence-interval widths "
        f"($1-\\delta={1.0-delta:.2f}$)",
        fontsize=15,
    )
    fig.legend(
        construction_handles,
        construction_labels,
        loc="lower center",
        bbox_to_anchor=(0.23, 0.012),
        ncol=2,
        fontsize=8.4,
        title="Construction (color and marker)",
        title_fontsize=8.6,
        frameon=False,
    )
    fig.legend(
        estimator_handles,
        estimator_labels,
        loc="lower center",
        bbox_to_anchor=(0.61, 0.012),
        ncol=2,
        fontsize=8.4,
        title="Variance estimator (line and marker fill)",
        title_fontsize=8.6,
        frameon=False,
    )
    fig.legend(
        component_handles,
        component_labels,
        loc="lower center",
        bbox_to_anchor=(0.88, 0.012),
        ncol=1,
        fontsize=8.4,
        title="Additional width (dotted)",
        title_fontsize=8.6,
        frameon=False,
    )
    fig.tight_layout(rect=(0.0, 0.08, 1.0, 0.95))
    filename = "scaled_width_star_efficient_shared_estimator_comparison.png"
    destination = plot_dir / filename
    PAPER_PLOT_DIR.mkdir(parents=True, exist_ok=True)
    for figure_path in (destination, PAPER_PLOT_DIR / filename):
        fig.savefig(figure_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    outputs.append(destination)

    return outputs


def run_experiment(args: argparse.Namespace) -> pd.DataFrame:
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    sample_sizes = sorted(args.sample_sizes or SAMPLE_SIZES)
    if args.reps is None:
        reps_by_n = {n: REPS_BY_N[n] for n in sample_sizes}
    else:
        reps_by_n = {n: args.reps for n in sample_sizes}
    unbuffered_reps_by_n = {
        n: min(args.unbuffered_reps, reps_by_n[n]) for n in sample_sizes
    }

    def required_methods_for_cell(n: int, rep: int) -> set[str]:
        required = set(METHOD_ORDER)
        if args.reps is None and rep >= BASE_REPS_BY_N[n]:
            required.intersection_update(FIGURE3_METHODS)
        if rep >= unbuffered_reps_by_n[n]:
            required.difference_update(FIGURE3_METHODS)
        return required
    if args.distribution_indices is None:
        distribution_pairs = list(enumerate(
            DISTRIBUTIONS[: args.distribution_limit]
        ))
    else:
        distribution_pairs = [
            (index, DISTRIBUTIONS[index])
            for index in args.distribution_indices
        ]

    if args.audit:
        audit_local_inversion(args.delta, args.seed, args.solvency_c)

    checkpoint = output / "results_checkpoint.csv"
    rows: list[dict[str, object]] = []
    observed_methods: dict[tuple[str, int, int], set[str]] = {}
    completed: set[tuple[str, int, int]] = set()
    if args.resume and checkpoint.exists():
        previous = pd.read_csv(checkpoint)
        previous = previous[previous["method"].isin(METHOD_ORDER)].copy()
        rows = previous.to_dict("records")
        method_sets = previous.groupby(
            ["distribution", "n", "rep"]
        )["method"].agg(set)
        observed_methods = method_sets.to_dict()
        completed = {
            key for key, observed in observed_methods.items()
            if required_methods_for_cell(int(key[1]), int(key[2])).issubset(observed)
        }
        print(f"resuming with {len(completed)} completed dataset/horizon cells")

    # Trigger compilation outside timed regions.
    warm = np.linspace(0.1, 0.9, 64)
    methods.compute_M_star(warm, 0.5, args.delta, c=args.solvency_c)
    methods.compute_M_star_common_clock_arms(
        warm, 0.5, args.delta, c=args.solvency_c
    )
    methods._star_randomized_scores(
        warm, np.asarray([0.5]), args.delta, 0.5, 0.5,
        args.solvency_c
    )
    methods._star_common_clock_arm_randomized_scores(
        warm, np.asarray([0.5]), args.delta, 0.5, 0.5,
        args.solvency_c
    )
    methods.compute_M_hinge_feedback_star(warm, 0.5, args.delta)
    methods.compute_M_capped_feedback_star(warm, 0.5, args.delta)
    methods.compute_M_capped_exponential_feedback_star(
        warm, 0.5, args.delta
    )
    methods.compute_M_probit_star_arms(
        warm, 0.5, args.delta, c=args.solvency_c,
        buffer_rounds=float(warm.size) ** (2.0 / 3.0)
    )
    methods.compute_M_probit_star_arms(
        warm, 0.5, args.delta, c=args.solvency_c,
        buffer_rounds=0.0
    )
    methods.compute_M_probit_common_clock_arms(
        warm, 0.5, args.delta, c=args.solvency_c,
        buffer_rounds=0.0
    )
    _feedback_scores_by_kind(
        warm,
        np.full(len(FEEDBACK_METHODS), 0.5),
        np.asarray(FEEDBACK_KINDS, dtype=np.int64),
        args.delta,
    )
    _power_sums(warm)

    max_reps = max(reps_by_n.values())
    start = time.time()
    cells_done = 0

    for dist_idx, dist in distribution_pairs:
        print(f"\n=== {dist.name} ===", flush=True)
        for rep in range(max_reps):
            max_n = _path_max_n(rep, sample_sizes, reps_by_n)
            if max_n == 0:
                continue
            seed_sequence = np.random.SeedSequence([args.seed, dist_idx, rep])
            data_seed, aux_seed = seed_sequence.spawn(2)
            data_rng = np.random.default_rng(data_seed)
            aux_rng = np.random.default_rng(aux_seed)
            path = np.asarray(dist.sampler(data_rng, max_n), dtype=np.float64)

            for n in sample_sizes:
                if rep >= reps_by_n[n]:
                    continue
                u_plus, u_minus = (float(value) for value in aux_rng.uniform(size=2))
                key = (dist.name, n, rep)
                if key in completed:
                    continue
                x = np.ascontiguousarray(path[:n])

                observed = observed_methods.get(key, set())
                required_methods = required_methods_for_cell(n, rep)
                missing = required_methods.difference(observed)
                incremental_methods = {
                    "Capped original feedback",
                    "Efficient betting",
                    "Common-clock Efficient betting",
                    "STaR betting",
                    "Common-clock STaR betting",
                }
                if missing and missing.issubset(incremental_methods):
                    for method_name in sorted(missing):
                        t0 = time.perf_counter()
                        lower, upper, empty, evaluations = _betting_interval(
                            x, method_name, args.delta, u_plus, u_minus,
                            args.solvency_c,
                        )
                        row = _record_row(
                            dist.name,
                            dist.mean,
                            dist.variance,
                            n,
                            rep,
                            method_name,
                            lower,
                            upper,
                            empty,
                            (
                                "monotone-common-clock-inversion"
                                if method_name in {
                                    "Common-clock Efficient betting",
                                    "Common-clock STaR betting",
                                }
                                else "local-batched-bisection"
                            ),
                            time.perf_counter() - t0,
                            evaluations,
                        )
                        if method_name in {
                            "Common-clock Efficient betting",
                            "Common-clock STaR betting",
                        }:
                            width = max(float(upper - lower), 0.0)
                            row.update({
                                "adjacent_component_width": width,
                                "full_set_diameter": width,
                                "largest_component_width": width,
                                "topology_component_count": (
                                    0.0 if empty else 1.0
                                ),
                                "topology_scan_points": 0.0,
                                "topology_point_budget_reached": False,
                            })
                        rows.append(row)
                    cells_done += 1
                    if cells_done % args.progress_every == 0:
                        elapsed = (time.time() - start) / 60.0
                        print(
                            f"n={n:>10,d} rep={rep+1:>3d}/{reps_by_n[n]} "
                            f"cells={cells_done} elapsed={elapsed:.1f} min",
                            flush=True,
                        )
                    pd.DataFrame(rows).to_csv(checkpoint, index=False)
                    continue

                cell_rows: list[dict[str, object]] = []
                t0 = time.perf_counter()
                feedback_results = invert_feedback_components(x, args.delta)
                feedback_seconds = (time.perf_counter() - t0) / len(FEEDBACK_METHODS)
                for method_name in FEEDBACK_METHODS:
                    lower, upper, empty, evaluations = feedback_results[method_name]
                    cell_rows.append(
                        _record_row(
                            dist.name,
                            dist.mean,
                            dist.variance,
                            n,
                            rep,
                            method_name,
                            lower,
                            upper,
                            empty,
                            "local-batched-bisection",
                            feedback_seconds,
                            evaluations,
                        )
                    )

                method_name = "Regularized Efficient betting"
                t0 = time.perf_counter()
                lower, upper, empty, evaluations = _betting_interval(
                    x, method_name, args.delta, u_plus, u_minus,
                    args.solvency_c,
                )
                cell_rows.append(
                    _record_row(
                        dist.name,
                        dist.mean,
                        dist.variance,
                        n,
                        rep,
                        method_name,
                        lower,
                        upper,
                        empty,
                        "local-batched-bisection",
                        time.perf_counter() - t0,
                        evaluations,
                    )
                )
                if "Efficient betting" in required_methods:
                    method_name = "Efficient betting"
                    t0 = time.perf_counter()
                    lower, upper, empty, evaluations = _betting_interval(
                        x, method_name, args.delta, u_plus, u_minus,
                        args.solvency_c,
                    )
                    cell_rows.append(
                        _record_row(
                            dist.name,
                            dist.mean,
                            dist.variance,
                            n,
                            rep,
                            method_name,
                            lower,
                            upper,
                            empty,
                            "local-batched-bisection",
                            time.perf_counter() - t0,
                            evaluations,
                        )
                    )

                if "Common-clock Efficient betting" in required_methods:
                    method_name = "Common-clock Efficient betting"
                    t0 = time.perf_counter()
                    lower, upper, empty, evaluations = _betting_interval(
                        x, method_name, args.delta, u_plus, u_minus,
                        args.solvency_c,
                    )
                    row = _record_row(
                        dist.name,
                        dist.mean,
                        dist.variance,
                        n,
                        rep,
                        method_name,
                        lower,
                        upper,
                        empty,
                        "monotone-common-clock-inversion",
                        time.perf_counter() - t0,
                        evaluations,
                    )
                    width = max(float(upper - lower), 0.0)
                    row.update({
                        "adjacent_component_width": width,
                        "full_set_diameter": width,
                        "largest_component_width": width,
                        "topology_component_count": 0.0 if empty else 1.0,
                        "topology_scan_points": 0.0,
                        "topology_point_budget_reached": False,
                    })
                    cell_rows.append(row)


                for method_name in (
                    "STaR betting",
                    "Common-clock STaR betting",
                ):
                    if method_name not in required_methods:
                        continue
                    t0 = time.perf_counter()
                    lower, upper, empty, evaluations = _betting_interval(
                        x, method_name, args.delta, u_plus, u_minus,
                        args.solvency_c,
                    )
                    row = _record_row(
                        dist.name,
                        dist.mean,
                        dist.variance,
                        n,
                        rep,
                        method_name,
                        lower,
                        upper,
                        empty,
                        (
                            "monotone-common-clock-inversion"
                            if method_name == "Common-clock STaR betting"
                            else "local-batched-bisection"
                        ),
                        time.perf_counter() - t0,
                        evaluations,
                    )
                    if method_name == "Common-clock STaR betting":
                        width = max(float(upper - lower), 0.0)
                        row.update({
                            "adjacent_component_width": width,
                            "full_set_diameter": width,
                            "largest_component_width": width,
                            "topology_component_count": (
                                0.0 if empty else 1.0
                            ),
                            "topology_scan_points": 0.0,
                            "topology_point_budget_reached": False,
                        })
                    cell_rows.append(row)


                method_name = "Capped original feedback"
                t0 = time.perf_counter()
                lower, upper, empty, evaluations = _betting_interval(
                    x, method_name, args.delta, u_plus, u_minus,
                    args.solvency_c,
                )
                cell_rows.append(
                    _record_row(
                        dist.name,
                        dist.mean,
                        dist.variance,
                        n,
                        rep,
                        method_name,
                        lower,
                        upper,
                        empty,
                        "local-batched-bisection",
                        time.perf_counter() - t0,
                        evaluations,
                    )
                )

                t0 = time.perf_counter()
                lower, upper, backend = fast_gaffke_ci(
                    x,
                    delta=args.delta,
                    binary=_is_binary_distribution(dist.name),
                    exact_cutoff=args.gaffke_exact_cutoff,
                )
                cell_rows.append(
                    _record_row(
                        dist.name,
                        dist.mean,
                        dist.variance,
                        n,
                        rep,
                        "Gaffke",
                        lower,
                        upper,
                        False,
                        backend,
                        time.perf_counter() - t0,
                        0,
                    )
                )
                rows.extend(cell_rows)
                cells_done += 1

                if cells_done % args.progress_every == 0:
                    elapsed = (time.time() - start) / 60.0
                    print(
                        f"n={n:>10,d} rep={rep+1:>3d}/{reps_by_n[n]} "
                        f"cells={cells_done} elapsed={elapsed:.1f} min",
                        flush=True,
                    )
                pd.DataFrame(rows).to_csv(checkpoint, index=False)

    df = pd.DataFrame(rows)
    df.to_csv(output / "results.csv", index=False)
    summary = summarize(df)
    summary.to_csv(output / "summary.csv", index=False)
    config = {
        "delta": args.delta,
        "solvency_c": args.solvency_c,
        "seed": args.seed,
        "sample_sizes": sample_sizes,
        "reps_by_n": {str(n): reps_by_n[n] for n in sample_sizes},
        "unbuffered_reps_by_n": {
            str(n): unbuffered_reps_by_n[n] for n in sample_sizes
        },
        "distributions": [
            dist.name for _, dist in distribution_pairs
        ],
        "methods": list(METHOD_ORDER),
        "gaffke_exact_cutoff": args.gaffke_exact_cutoff,
        "continuous_gaffke_large_n_backend": "fourth-order Cornish-Fisher from exact Dirichlet moments",
        "betting_inversion": (
            "candidate-dependent methods use a local geometric bracket plus "
            "paired bisection"
        ),
        "common_clock_inversion": (
            "parallel multisection of the two globally monotone arm "
            "boundaries; the reported set is an exact interval"
        ),
        "reported_width": "accepted component adjacent to the sample mean",
        "full_set_topology": (
            "not assumed; use ../audit_confidence_set_topology.py for the "
            "global finite-mesh audit"
        ),
        "empirical_interval": "pointwise 10th--90th percentiles of CI widths",
    }
    with (output / "config.json").open("w", encoding="utf-8") as stream:
        json.dump(config, stream, indent=2)
    for plot in make_plots(df, output, args.delta):
        print(f"saved {plot}")
    print(f"saved results to {output}")
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default=str(Path(__file__).with_name("large_sample_gaffke_results")),
    )
    parser.add_argument("--delta", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--solvency-c", type=float, default=1.0)
    parser.add_argument("--sample-sizes", nargs="+", type=int)
    parser.add_argument("--reps", type=int)
    parser.add_argument("--unbuffered-reps", type=int, default=30)
    parser.add_argument("--distribution-limit", type=int, default=len(DISTRIBUTIONS))
    parser.add_argument("--distribution-indices", nargs="+", type=int)
    parser.add_argument("--gaffke-exact-cutoff", type=int, default=3_000)
    parser.add_argument("--progress-every", type=int, default=5)
    parser.add_argument("--audit", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    if args.unbuffered_reps <= 0:
        parser.error("--unbuffered-reps must be positive")
    if not 0.0 < args.solvency_c <= 1.0:
        parser.error("--solvency-c must lie in (0,1]")
    if args.reps is not None and args.reps <= 0:
        parser.error("--reps must be positive")
    if args.distribution_limit <= 0 or args.distribution_limit > len(DISTRIBUTIONS):
        parser.error("--distribution-limit is out of range")
    if args.distribution_indices is not None:
        if len(set(args.distribution_indices)) != len(args.distribution_indices):
            parser.error("--distribution-indices must be unique")
        if any(
            index < 0 or index >= len(DISTRIBUTIONS)
            for index in args.distribution_indices
        ):
            parser.error("--distribution-indices contains an out-of-range index")
    return args


def main() -> None:
    args = parse_args()
    if args.plot_only:
        results_path = Path(args.output).resolve() / "results.csv"
        df = pd.read_csv(results_path)
        for plot in make_plots(df, Path(args.output).resolve(), args.delta):
            print(f"saved {plot}")
        return
    run_experiment(args)


if __name__ == "__main__":
    main()
