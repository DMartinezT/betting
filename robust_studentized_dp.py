#!/usr/bin/env python3
"""Discrete-state audit of a symmetric studentized martingale event.

For a candidate mean m, put

    A_n = sum_i (X_i - m),   R_n = sum_i (X_i - m)^2.

The upper terminal event is A_n >= z sqrt(R_n) + b.  Its sharp null price
over processes with E[X_i | F_{i-1}] = m is a two-state Bellman problem.
This module solves the problem exactly when X_i is restricted to an equally
spaced grid.  The grid calculation is a diagnostic for the unrounded
continuous-state problem.  It also becomes an exact, externally randomized
procedure for arbitrary X_i in [0,1] after unbiased stochastic quantization
to that grid; see ``unbiased_grid_quantize``.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
from numba import njit, prange
from scipy.stats import norm


def unbiased_grid_quantize(X, grid_intervals, rng):
    """Round observations to a grid without changing conditional means.

    If ``scaled = J * X``, the output equals ``floor(scaled) / J`` or the
    adjacent upper grid point, with probabilities chosen so that its
    conditional expectation given X is exactly X.  Independent randomizers
    therefore preserve the constant-conditional-mean null.
    """
    X = np.asarray(X, dtype=float)
    if np.any((X < 0.0) | (X > 1.0)):
        raise ValueError("X must lie in [0,1]")
    if int(grid_intervals) != grid_intervals or grid_intervals <= 0:
        raise ValueError("grid_intervals must be a positive integer")
    grid_intervals = int(grid_intervals)
    scaled = grid_intervals * X
    lower = np.floor(scaled).astype(np.int64)
    upper_probability = scaled - lower
    rounded_index = lower + (rng.random(X.shape) < upper_probability)
    return rounded_index.astype(float) / grid_intervals


def reachable_state_layers(n, grid_intervals):
    """Return compact arrays of reachable (sum j, sum j^2) grid states."""
    if int(n) != n or n <= 0:
        raise ValueError("n must be a positive integer")
    if int(grid_intervals) != grid_intervals or grid_intervals <= 0:
        raise ValueError("grid_intervals must be a positive integer")
    n = int(n)
    grid_intervals = int(grid_intervals)
    layer = {(0, 0)}
    layers = [
        (
            np.asarray([0], dtype=np.int32),
            np.asarray([0], dtype=np.int32),
        )
    ]
    increments = tuple(
        (value, value * value) for value in range(grid_intervals + 1)
    )
    for _ in range(n):
        layer = {
            (total + value, squares + square)
            for total, squares in layer
            for value, square in increments
        }
        ordered = sorted(layer)
        layers.append(
            (
                np.asarray([state[0] for state in ordered], dtype=np.int32),
                np.asarray([state[1] for state in ordered], dtype=np.int32),
            )
        )
    return layers


@njit(parallel=True)
def _set_terminal_values(
    values,
    sums,
    squares,
    n,
    grid_intervals,
    m,
    z,
    correction,
):
    for index in prange(sums.size):
        total = sums[index] / grid_intervals
        raw_squares = squares[index] / (grid_intervals * grid_intervals)
        centered_sum = total - n * m
        centered_squares = (
            raw_squares - 2.0 * m * total + n * m * m
        )
        if centered_squares < 0.0:
            centered_squares = 0.0
        if centered_sum >= z * math.sqrt(centered_squares) + correction:
            values[sums[index], squares[index]] = 1.0
        else:
            values[sums[index], squares[index]] = 0.0


@njit(parallel=True)
def _bellman_layer(
    current,
    future,
    sums,
    squares,
    grid_intervals,
    m,
):
    target = m * grid_intervals
    lower_max = int(math.floor(target + 1e-14))
    upper_min = int(math.ceil(target - 1e-14))
    on_grid = abs(target - round(target)) <= 1e-12
    grid_index = int(round(target))

    for index in prange(sums.size):
        total = sums[index]
        raw_squares = squares[index]
        best = 0.0
        if on_grid:
            best = future[
                total + grid_index,
                raw_squares + grid_index * grid_index,
            ]
        for lower in range(lower_max + 1):
            lower_value = future[
                total + lower,
                raw_squares + lower * lower,
            ]
            for upper in range(upper_min, grid_intervals + 1):
                if upper == lower:
                    continue
                upper_value = future[
                    total + upper,
                    raw_squares + upper * upper,
                ]
                upper_weight = (target - lower) / (upper - lower)
                candidate = (
                    (1.0 - upper_weight) * lower_value
                    + upper_weight * upper_value
                )
                if candidate > best:
                    best = candidate
        current[total, raw_squares] = best


def grid_robust_tail_probability(
    *,
    n,
    m,
    z,
    correction,
    grid_intervals=8,
    layers=None,
):
    """Compute the sharp price on an equally spaced observation grid."""
    if not 0.0 <= m <= 1.0:
        raise ValueError("m must lie in [0,1]")
    if correction < 0.0:
        raise ValueError("correction must be nonnegative")
    n = int(n)
    grid_intervals = int(grid_intervals)
    if layers is None:
        layers = reachable_state_layers(n, grid_intervals)
    if len(layers) != n + 1:
        raise ValueError("layers do not match n")

    shape = (
        n * grid_intervals + 1,
        n * grid_intervals * grid_intervals + 1,
    )
    future = np.empty(shape, dtype=np.float64)
    current = np.empty(shape, dtype=np.float64)
    terminal_sums, terminal_squares = layers[n]
    _set_terminal_values(
        future,
        terminal_sums,
        terminal_squares,
        n,
        grid_intervals,
        m,
        z,
        correction,
    )

    for step in range(n - 1, -1, -1):
        sums, squares = layers[step]
        _bellman_layer(
            current,
            future,
            sums,
            squares,
            grid_intervals,
            m,
        )
        future, current = current, future
    return float(future[0, 0])


def calibrate_grid_correction(
    *,
    n,
    m,
    alpha,
    z=None,
    grid_intervals=8,
    bisection_steps=24,
    layers=None,
):
    """Find the smallest grid correction whose Bellman price is at most alpha."""
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0,1)")
    if z is None:
        z = float(norm.isf(alpha))
    if layers is None:
        layers = reachable_state_layers(n, grid_intervals)

    lower = 0.0
    upper = float(n)
    upper_price = grid_robust_tail_probability(
        n=n,
        m=m,
        z=z,
        correction=upper,
        grid_intervals=grid_intervals,
        layers=layers,
    )
    if upper_price > alpha:
        raise RuntimeError("failed to bracket a valid correction")

    for _ in range(int(bisection_steps)):
        midpoint = 0.5 * (lower + upper)
        price = grid_robust_tail_probability(
            n=n,
            m=m,
            z=z,
            correction=midpoint,
            grid_intervals=grid_intervals,
            layers=layers,
        )
        if price <= alpha:
            upper = midpoint
        else:
            lower = midpoint
    final_price = grid_robust_tail_probability(
        n=n,
        m=m,
        z=z,
        correction=upper,
        grid_intervals=grid_intervals,
        layers=layers,
    )
    return {
        "n": int(n),
        "m": float(m),
        "alpha": float(alpha),
        "z": float(z),
        "grid_intervals": int(grid_intervals),
        "correction": float(upper),
        "grid_price": float(final_price),
    }


def rare_event_lower_bound(n, m, alpha, z):
    """Necessary upper-tail correction from a two-point iid null.

    Put mass m/(m+d) at m+d and the remaining mass at zero.  The all-m+d
    sample has probability exceeding alpha whenever

        d < m (alpha^(-1/n) - 1).

    Feasibility also requires d <= 1-m.  On that sample the uncorrected
    margin A_n-z*sqrt(R_n) equals d*(n-z*sqrt(n)).
    """
    if int(n) != n or n <= 0:
        raise ValueError("n must be a positive integer")
    if not 0.0 <= m <= 1.0:
        raise ValueError("m must lie in [0,1]")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0,1)")
    if z < 0.0:
        raise ValueError("z must be nonnegative")
    n = int(n)
    denominator = n - z * math.sqrt(n)
    if denominator <= 0.0:
        return 0.0
    largest_admissible_step = min(
        1.0 - m,
        m * (alpha ** (-1.0 / n) - 1.0),
    )
    return denominator * max(largest_admissible_step, 0.0)


def optimistic_studentized_ci_endpoints(X, delta):
    """Invert the unavoidable-correction boundary.

    This is an optimistic lower-bound diagnostic, not a valid confidence
    interval: the rare-path correction is necessary but need not be
    sufficient for controlling every null path.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 1 or X.size == 0:
        raise ValueError("X must be a nonempty vector")
    if np.any((X < 0.0) | (X > 1.0)):
        raise ValueError("X must lie in [0,1]")
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must lie in (0,1)")
    n = X.size
    alpha = delta / 2.0
    z = float(norm.isf(alpha))
    total = float(np.sum(X))
    raw_squares = float(np.dot(X, X))
    center = total / n

    def upper_margin(m):
        centered_sum = total - n * m
        centered_squares = raw_squares - 2.0 * m * total + n * m * m
        correction = rare_event_lower_bound(n, m, alpha, z)
        return (
            centered_sum
            - z * math.sqrt(max(centered_squares, 0.0))
            - correction
        )

    def lower_margin(m):
        centered_sum = n * m - total
        centered_squares = raw_squares - 2.0 * m * total + n * m * m
        correction = rare_event_lower_bound(n, 1.0 - m, alpha, z)
        return (
            centered_sum
            - z * math.sqrt(max(centered_squares, 0.0))
            - correction
        )

    def boundary(rejected, accepted, margin):
        if margin(rejected) < 0.0:
            return rejected
        for _ in range(60):
            midpoint = 0.5 * (rejected + accepted)
            if margin(midpoint) >= 0.0:
                rejected = midpoint
            else:
                accepted = midpoint
        return 0.5 * (rejected + accepted)

    lower = boundary(0.0, center, upper_margin)
    upper = boundary(1.0, center, lower_margin)
    return float(lower), float(upper), False


def run_calibration_audit(
    *,
    output="plots/robust_studentized_dp/calibration.json",
    n_values=(10, 20),
    m_values=(0.2, 0.5, 0.8),
    delta=0.01,
    grid_intervals=8,
    bisection_steps=18,
):
    """Calibrate several grid problems and save a reproducible audit."""
    alpha = delta / 2.0
    z = float(norm.isf(alpha))
    started = time.perf_counter()
    rows = []
    for n in n_values:
        print(f"building states: n={n}, J={grid_intervals}", flush=True)
        layers = reachable_state_layers(int(n), int(grid_intervals))
        for m in m_values:
            print(f"calibrating: n={n}, m={m:.3f}", flush=True)
            row = calibrate_grid_correction(
                n=int(n),
                m=float(m),
                alpha=alpha,
                z=z,
                grid_intervals=int(grid_intervals),
                bisection_steps=int(bisection_steps),
                layers=layers,
            )
            row["rare_event_lower_bound"] = rare_event_lower_bound(
                int(n), float(m), alpha, z
            )
            rows.append(row)
    payload = {
        "description": (
            "Exact Bellman values for observations restricted to the stated "
            "grid; these are diagnostics, not continuous-support certificates."
        ),
        "delta": float(delta),
        "elapsed_seconds": float(time.perf_counter() - started),
        "rows": rows,
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)
    print(f"saved {output} in {payload['elapsed_seconds']:.1f} seconds")
    return payload


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="plots/robust_studentized_dp/calibration.json",
    )
    parser.add_argument("--n-values", type=int, nargs="+", default=[10, 20])
    parser.add_argument(
        "--m-values",
        type=float,
        nargs="+",
        default=[0.2, 0.5, 0.8],
    )
    parser.add_argument("--delta", type=float, default=0.01)
    parser.add_argument("--grid-intervals", type=int, default=8)
    parser.add_argument("--bisection-steps", type=int, default=18)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = _parse_args()
    run_calibration_audit(
        output=arguments.output,
        n_values=tuple(arguments.n_values),
        m_values=tuple(arguments.m_values),
        delta=arguments.delta,
        grid_intervals=arguments.grid_intervals,
        bisection_steps=arguments.bisection_steps,
    )
