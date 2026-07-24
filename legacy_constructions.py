#!/usr/bin/env python3
"""Archived implementations of polynomial Constructions 1 and 2.

These functions are retained to reproduce and study the stopping/time-change
failure described in the paper appendix. They are intentionally not imported by
or plotted in the current Construction 3 experiments.
"""

import numpy as np
from numba import njit

# 3.  M_{n,2}: Construction 2 (one fixed ordering)
# ------------------------------------------------------------------

@njit
def compute_Mt2_path(X, m, lam_per_step, I2_val, c=0.5):
    """
    Single-path M_{n,2}(m) = 0.5 * M^+(m) + 0.5 * M^-(m).

    Initialization: M_0^+ = M_0^- = I2_val  (= I2(lam^*))

    Recursive update for i = 0, ..., n-1  (step i+1 in 1-indexed notation):

      W^+  = (S^+ - Lambda^+)_+   [arm-specific positive part, computed before update]
      W^-  = (S^- - Lambda^-)_+

      tilde_gamma = 1 / (sqrt(n) * sigma_hat_{i-1})

      gamma^+ = min(tilde_gamma, c * M^+ / (2 * W^+ * m))       if W^+, m > 0
      gamma^- = min(tilde_gamma, c * M^- / (2 * W^- * (1-m)))   if W^-, (1-m) > 0

      M^+      += 2 * gamma^+ * (X[i] - m) * W^+
      M^-      += 2 * gamma^- * (m - X[i]) * W^-

      S^+      += gamma^+ * (X[i] - m)
      S^-      += gamma^- * (m - X[i])
      Lambda^+ += lam_per_step * (gamma^+ / tilde_gamma)   [rescaled per paper eq. 5.5]
      Lambda^- += lam_per_step * (gamma^- / tilde_gamma)

    Clipping ensures M^+, M^- >= 0 at every step (worst case X_i = 0 for M^+,
    X_i = 1 for M^-).  Lambda is rescaled arm-by-arm so the shift/scale ratio
    gamma/lambda stays constant when clipping is active.

    CI: {m : M_bar_{n,2}(m) < I2_val / delta}.
    """
    n            = len(X)
    sqrt_n       = np.sqrt(float(n))
    S_plus       = 0.0
    S_minus      = 0.0
    Lambda_plus  = 0.0
    Lambda_minus = 0.0
    M_plus       = I2_val
    M_minus      = I2_val
    sum_x         = 0.0
    pred_sq       = 0.0

    for i in range(n):
        # Predictable variance estimate sigma_hat_{i-1}^2
        mean_hat    = (0.5 + sum_x) / (1.0 + i)
        var_hat     = (0.25 + pred_sq) / (1.0 + i)
        tilde_gamma = 1.0 / (sqrt_n * np.sqrt(var_hat))

        Wp = max(S_plus  - Lambda_plus,  0.0)   # (S^+_{i-1} - Lambda^+_{i-1})_+
        Wm = max(S_minus - Lambda_minus, 0.0)   # (S^-_{i-1} - Lambda^-_{i-1})_+

        # Clip gamma^+ to keep M^+ nonneg
        if Wp > 1e-14 and m > 1e-14:
            gamma_plus = min(tilde_gamma, c * M_plus / (2.0 * Wp * m))
        else:
            gamma_plus = tilde_gamma

        # Clip gamma^- to keep M^- nonneg
        one_m = 1.0 - m
        if Wm > 1e-14 and one_m > 1e-14:
            gamma_minus = min(tilde_gamma, c * M_minus / (2.0 * Wm * one_m))
        else:
            gamma_minus = tilde_gamma

        xi = X[i] - m   # signed deviation

        M_plus  += 2.0 * gamma_plus  *  xi * Wp
        M_minus += 2.0 * gamma_minus * (-xi) * Wm

        S_plus       += gamma_plus  *  xi
        S_minus      += gamma_minus * (-xi)
        Lambda_plus  += lam_per_step * (gamma_plus  / tilde_gamma)
        Lambda_minus += lam_per_step * (gamma_minus / tilde_gamma)

        residual = X[i] - mean_hat
        sum_x += X[i]
        pred_sq += residual * residual

    return 0.5 * M_plus + 0.5 * M_minus


@njit
def compute_M_bar_n2(X, m, lam_per_step, I2_val, B):
    """
    M_bar_{n,2}(m) = (1/B) * sum_{b=1}^B M_{n,2}(m ; X_{pi_b})

    Rao-Blackwellization: averaging over B uniform random permutations
    integrates out path-dependent noise while preserving the e-value property:
        E_{H_0}[M_bar_{n,2}(m)] = I2_val  =>  CI at level delta via Markov.
    """
    total = 0.0
    for _ in range(B):
        total += compute_Mt2_path(np.random.permutation(X), m, lam_per_step, I2_val)
    return total / B


@njit
def compute_M_bar_n2_fixed(X_perms, m, lam_per_step, I2_val):
    """Legacy permutation average with common randomness across candidate m."""
    B = X_perms.shape[0]
    total = 0.0
    for b in range(B):
        total += compute_Mt2_path(X_perms[b], m, lam_per_step, I2_val)
    return total / B


# ------------------------------------------------------------------
# 3a. Corrected construction: hedge the terminal Bentkus payoff
# ------------------------------------------------------------------

@njit
def clip_fraction_path(X, m, lam_per_step, I2_val, c=0.5):
    """
    Runs the same loop as compute_Mt2_path but returns the fraction of the 2n
    individual bets (n for the + arm, n for the - arm) where the clip condition
    was active, i.e. where tilde_gamma > c_i^{+/-} and gamma was reduced.
    """
    n            = len(X)
    sqrt_n       = np.sqrt(float(n))
    S_plus       = 0.0
    S_minus      = 0.0
    Lambda_plus  = 0.0
    Lambda_minus = 0.0
    M_plus       = I2_val
    M_minus      = I2_val
    sum_x         = 0.0
    pred_sq       = 0.0
    n_clipped    = 0

    for i in range(n):
        mean_hat    = (0.5 + sum_x) / (1.0 + i)
        var_hat     = (0.25 + pred_sq) / (1.0 + i)
        tilde_gamma = 1.0 / (sqrt_n * np.sqrt(var_hat))

        Wp = max(S_plus  - Lambda_plus,  0.0)
        Wm = max(S_minus - Lambda_minus, 0.0)

        if Wp > 1e-14 and m > 1e-14:
            c_plus     = c * M_plus / (2.0 * Wp * m)
            gamma_plus = min(tilde_gamma, c_plus)
            if tilde_gamma > c_plus:
                n_clipped += 1
        else:
            gamma_plus = tilde_gamma

        one_m = 1.0 - m
        if Wm > 1e-14 and one_m > 1e-14:
            c_minus     = c * M_minus / (2.0 * Wm * one_m)
            gamma_minus = min(tilde_gamma, c_minus)
            if tilde_gamma > c_minus:
                n_clipped += 1
        else:
            gamma_minus = tilde_gamma

        xi           = X[i] - m
        M_plus      += 2.0 * gamma_plus  *  xi * Wp
        M_minus     += 2.0 * gamma_minus * (-xi) * Wm
        S_plus      += gamma_plus  *  xi
        S_minus     += gamma_minus * (-xi)
        Lambda_plus  += lam_per_step * (gamma_plus  / tilde_gamma)
        Lambda_minus += lam_per_step * (gamma_minus / tilde_gamma)

        residual = X[i] - mean_hat
        sum_x += X[i]
        pred_sq += residual * residual

    return n_clipped / (2.0 * n)   # denominator: n bets per arm


@njit
def clip_fraction_two_m(X, m1, m2, lam_per_step, I2_val, B):
    """
    Clip fraction for Construction 2, averaged over {m1, m2} and B permutations.
    m1, m2 should be the two predicted CI boundaries for the current sample.
    """
    total = 0.0
    for _ in range(B):
        X_perm = np.random.permutation(X)
        total += clip_fraction_path(X_perm, m1, lam_per_step, I2_val)
        total += clip_fraction_path(X_perm, m2, lam_per_step, I2_val)
    return total / (2.0 * B)




# Construction 1's Taylor-remainder representation and Construction 2's direct
# additive representation are algebraically identical when driven by the same
# clipped predictable bets. This explicit alias makes that relationship clear
# while keeping one tested implementation of the recursion.
compute_construction1_path = compute_Mt2_path
compute_construction2_path = compute_Mt2_path
