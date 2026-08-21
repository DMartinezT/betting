#!/usr/bin/env python3
"""Reproduce the local-Gaussian feedback comparison used in the paper.

For feedback ``f`` and ``sigma(p) = p f(p)``, this solves

    .5 sigma(p)^2 q''(p) - .5 q(p) = -sigma(p),  q(0)=q(1)=0,

on a truncated, uniform logit mesh.  The reported refinement differences
are numerical diagnostics, not interval-certified error bounds.
"""

from __future__ import annotations

import math
import time
from functools import lru_cache

import numpy as np
from scipy.linalg import solve_banded
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq
from scipy.special import expit, logit
from scipy.stats import norm


ALPHA = 0.005
ETA = 1.0
LOGIT_LIMIT = 28.0
MESH_SIZES = (1501, 3001, 6001)


def f_product(p: np.ndarray) -> np.ndarray:
    """Product-STaR feedback."""
    return np.sqrt(2.0 * np.log(1.0 / p))


def _hinge_moments(a: float) -> tuple[float, float]:
    """I1=E[(Z-a)+] and I2=E[(Z-a)+^2]."""
    survival = norm.sf(a)
    density = norm.pdf(a)
    i1 = density - a * survival
    i2 = (1.0 + a * a) * survival - a * density
    return i1, i2


@lru_cache(maxsize=None)
def _hinge_leverage_scalar(p: float) -> float:
    """Exact scalar inversion of I1(a)^2/I2(a)=p."""
    lower = -max(8.0, 2.0 * math.sqrt(p / (1.0 - p)))

    def equation(a: float) -> float:
        i1, i2 = _hinge_moments(a)
        return i1 * i1 / i2 - p

    a = brentq(equation, lower, 8.0, xtol=2e-12, rtol=5e-15)
    i1, i2 = _hinge_moments(a)
    return 2.0 * i1 / i2


def f_hinge(p: np.ndarray) -> np.ndarray:
    """Squared-hinge Bentkus-STaR feedback."""
    return np.fromiter(
        (_hinge_leverage_scalar(float(value)) for value in p),
        dtype=float,
        count=p.size,
    )


def _capped_moments(a: float, eta: float = ETA) -> tuple[float, float]:
    """Return P_eta(a) and D_eta(a) for the capped quadratic ramp."""
    b = a + eta
    if a >= 0.0:
        interval = norm.sf(a) - norm.sf(b)
    else:
        interval = norm.cdf(b) - norm.cdf(a)
    phi_a, phi_b = norm.pdf(a), norm.pdf(b)
    j1 = phi_a - phi_b - a * interval
    j2 = (
        (1.0 + a * a) * interval
        - a * phi_a
        + (a - eta) * phi_b
    )
    price = norm.sf(b) + j2 / (eta * eta)
    delta = 2.0 * j1 / (eta * eta)
    return price, delta


@lru_cache(maxsize=None)
def _capped_leverage_scalar(p: float, eta: float = ETA) -> float:
    """Invert P_eta(a)=p and return D_eta(a)/P_eta(a)."""
    a = brentq(
        lambda strike: _capped_moments(strike, eta)[0] - p,
        -8.0 - eta,
        8.0,
        xtol=2e-12,
        rtol=5e-15,
    )
    price, delta = _capped_moments(a, eta)
    return delta / price


def f_capped(p: np.ndarray) -> np.ndarray:
    """Unit-width target-capped quadratic-ramp feedback."""
    return np.fromiter(
        (_capped_leverage_scalar(float(value), ETA) for value in p),
        dtype=float,
        count=p.size,
    )


def _capped_exponential_moments(
    a: float, slope: float
) -> tuple[float, float]:
    """Price and delta of min(exp(slope * (Z-a)), 1)."""
    tilted = math.exp(-slope * a + 0.5 * slope * slope) * norm.cdf(
        a - slope
    )
    return norm.sf(a) + tilted, slope * tilted


@lru_cache(maxsize=None)
def _capped_exponential_leverage_scalar(p: float) -> float:
    """Preserve original STaR slope, cap its payoff, and re-price it."""
    slope = math.sqrt(2.0 * math.log(1.0 / p))
    a = brentq(
        lambda strike: _capped_exponential_moments(strike, slope)[0] - p,
        -16.0,
        16.0,
        xtol=2e-12,
        rtol=5e-15,
    )
    price, delta = _capped_exponential_moments(a, slope)
    return delta / price


def f_capped_exponential(p: np.ndarray) -> np.ndarray:
    """Target-capped original-STaR feedback."""
    return np.fromiter(
        (_capped_exponential_leverage_scalar(float(value)) for value in p),
        dtype=float,
        count=p.size,
    )


def f_digital(p: np.ndarray) -> np.ndarray:
    """Digital/Probit feedback, whose exact local slope is phi(Phi^-1(p))."""
    return norm.pdf(norm.isf(p)) / p


FEEDBACKS = {
    "product": f_product,
    "capped_exp": f_capped_exponential,
    "hinge": f_hinge,
    "capped_eta1": f_capped,
    "digital": f_digital,
}


def solve_local_power(feedback, mesh_size: int) -> float:
    """Second-order finite-difference solve in x=logit(p)."""
    x = np.linspace(-LOGIT_LIMIT, LOGIT_LIMIT, mesh_size)
    p = expit(x)
    h = x[1] - x[0]

    p_i = p[1:-1]
    r_i = p_i * (1.0 - p_i)
    sigma_i = p_i * feedback(p_i)
    drift_i = 1.0 - 2.0 * p_i
    potential_i = (r_i / sigma_i) ** 2

    inverse_h2 = 1.0 / (h * h)
    lower = inverse_h2 + drift_i / (2.0 * h)
    diagonal = -2.0 * inverse_h2 - potential_i
    upper = inverse_h2 - drift_i / (2.0 * h)
    rhs = -2.0 * r_i * r_i / sigma_i

    banded = np.zeros((3, mesh_size - 2))
    banded[0, 1:] = upper[:-1]
    banded[1, :] = diagonal
    banded[2, :-1] = lower[1:]
    q = np.zeros(mesh_size)
    q[1:-1] = solve_banded((1, 1), banded, rhs)

    return float(CubicSpline(x, q)(logit(ALPHA)))


def leverage_crossing() -> float:
    """Nontrivial crossing of squared-hinge and product feedbacks."""
    return brentq(
        lambda p: _hinge_leverage_scalar(float(p))
        - float(f_product(np.array([p]))[0]),
        0.01,
        0.10,
        xtol=5e-15,
        rtol=5e-15,
    )


def main() -> None:
    started = time.perf_counter()
    rows: list[dict[str, float]] = []
    for mesh_size in MESH_SIZES:
        rows.append(
            {
                name: solve_local_power(feedback, mesh_size)
                for name, feedback in FEEDBACKS.items()
            }
        )

    endpoint = expit(-LOGIT_LIMIT)
    print(
        f"alpha={ALPHA:g}, eta={ETA:g}, "
        f"logit truncation=[-{LOGIT_LIMIT:g},{LOGIT_LIMIT:g}], "
        f"p_min={endpoint:.3e}"
    )
    print("mesh  " + "  ".join(f"{name:>14s}" for name in FEEDBACKS))
    for mesh_size, row in zip(MESH_SIZES, rows):
        print(
            f"{mesh_size:4d}  "
            + "  ".join(f"{row[name]:.12f}" for name in FEEDBACKS)
        )

    print("\nrefinement differences (new minus previous)")
    for index in range(1, len(rows)):
        differences = {
            name: rows[index][name] - rows[index - 1][name]
            for name in FEEDBACKS
        }
        print(
            f"{MESH_SIZES[index - 1]}->{MESH_SIZES[index]}  "
            + "  ".join(f"{name}={differences[name]:+.3e}" for name in FEEDBACKS)
        )

    # The nested meshes halve their spacing, so the leading O(h^2) term can
    # be removed for a useful point estimate.  This remains a numerical
    # extrapolation, not a certified error enclosure.
    finest = {
        name: (4.0 * rows[-1][name] - rows[-2][name]) / 3.0
        for name in FEEDBACKS
    }
    print("\nRichardson estimates from the last two meshes")
    print("  ".join(f"{name}={finest[name]:.12f}" for name in FEEDBACKS))
    coarser = {
        name: (4.0 * rows[-2][name] - rows[-3][name]) / 3.0
        for name in FEEDBACKS
    }
    print("Richardson refinement (fine minus coarse)")
    print(
        "  ".join(
            f"{name}={finest[name] - coarser[name]:+.3e}"
            for name in FEEDBACKS
        )
    )

    exact_digital = norm.pdf(norm.isf(ALPHA))
    print(f"\nproduct - hinge at alpha: {finest['product'] - finest['hinge']:.12e}")
    print(f"leverage crossing p_c:    {leverage_crossing():.15f}")
    print(f"digital analytic value:   {exact_digital:.12f}")
    print(f"digital BVP minus exact:  {finest['digital'] - exact_digital:+.3e}")
    print(f"runtime: {time.perf_counter() - started:.2f} seconds")


if __name__ == "__main__":
    main()
