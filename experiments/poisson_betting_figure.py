#!/usr/bin/env python3
"""Build Figure 6, the small-sample comparison with Poisson betting."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import NormalDist

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

import betting
from gaffke_comparison.compare_star_probit_gaffke import (
    DISTRIBUTIONS as GAFFKE_DISTRIBUTIONS,
)
from gaffke_comparison.figure2_gaffke import (
    DELTA as GAFFKE_DELTA,
    SEED as GAFFKE_SEED,
    _randomized_endpoints,
)
from gaffke_comparison.large_sample_feedback_gaffke import fast_gaffke_ci


HERE = Path(__file__).resolve().parents[1]
PAPER_PLOTS = HERE.parent / "paper" / "plots"
DEFAULT_INPUT = HERE / "plots" / "ci_width_original_vs_star.json"
DEFAULT_GAFFKE = (
    HERE / "gaffke_comparison" / "figure2_gaffke_results" / "results.csv"
)
DEFAULT_OUTPUT = HERE / "plots" / "poisson_betting"
PLOT_NAME = "ci_width_poisson_betting.png"
CONSIDERED_N = (10, 50, 100, 500, 1_000, 5_000, 10_000)
GAFFKE_METHODS = ("Gaffke", "Randomized Gaffke")
GAFFKE_EXTENSION_SEED_TAG = 602214076
PANEL_ORDER = (
    "Uniform(0,1)",
    "Beta(0.5,0.5)",
    "Bernoulli(0.1)",
    "Beta(2,2)",
    "Beta(1,5)",
    "Bernoulli(0.5)",
    "Beta(50,50)",
    "Beta(20,80)",
    "Uniform(0.45,0.55)",
)
VARIANCES = {
    "Uniform(0,1)": 1.0 / 12.0,
    "Beta(0.5,0.5)": 1.0 / 8.0,
    "Bernoulli(0.1)": 0.09,
    "Beta(2,2)": 1.0 / 20.0,
    "Beta(1,5)": 5.0 / 252.0,
    "Bernoulli(0.5)": 0.25,
    "Beta(50,50)": 1.0 / 404.0,
    "Beta(20,80)": 1600.0 / 1_010_000.0,
    "Uniform(0.45,0.55)": 0.01 / 12.0,
}
TRUE_MEANS = {
    "Uniform(0,1)": 0.5,
    "Beta(0.5,0.5)": 0.5,
    "Bernoulli(0.1)": 0.1,
    "Beta(2,2)": 0.5,
    "Beta(1,5)": 1.0 / 6.0,
    "Bernoulli(0.5)": 0.5,
    "Beta(50,50)": 0.5,
    "Beta(20,80)": 0.2,
    "Uniform(0.45,0.55)": 0.5,
}
CALIBRATIONS = {
    "Deterministic": (1.0, 1.0),
    "Uniformly randomized": None,
}
SAVED_METHODS = (
    ("product_original", "#9467bd", "s", "Product betting"),
    (
        "product_star_common_clock",
        "darkorange",
        "P",
        "STaR betting",
    ),
    ("probit_common_clock", "#2ca02c", "h", "GE-betting"),
)
PE_STYLE = ("#d62728", "X", "PE-betting")
GAFFKE_STYLE = ("#1976b9", "o", "Gaffke")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--gaffke-results", type=Path, default=DEFAULT_GAFFKE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--plot-name", default=PLOT_NAME)
    parser.add_argument("--reps", type=int, default=120)
    parser.add_argument("--progress-every", type=int, default=20)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--solvency-c", type=float, default=0.5)
    parser.add_argument(
        "--third-moment-shrinkage",
        type=float,
        default=betting.POISSON_THIRD_MOMENT_SHRINKAGE,
    )
    parser.add_argument(
        "--skewness-epsilon",
        type=float,
        default=betting.POISSON_SKEWNESS_EPSILON,
    )
    return parser.parse_args()


def _samplers(rng: np.random.Generator) -> dict[str, object]:
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


def _configuration_matches(
    frame: pd.DataFrame,
    *,
    c: float,
    shrinkage: float,
    epsilon: float,
) -> bool:
    if frame.empty:
        return True
    return (
        np.allclose(frame["solvency_c"], c)
        and np.allclose(frame["third_moment_shrinkage"], shrinkage)
        and np.allclose(frame["skewness_epsilon"], epsilon)
    )


def _run_pe(
    payload: dict[str, object],
    checkpoint: Path,
    *,
    reps: int,
    resume: bool,
    progress_every: int,
    c: float,
    shrinkage: float,
    epsilon: float,
) -> pd.DataFrame:
    if reps <= 0:
        raise ValueError("--reps must be positive")
    if not 0.0 < c <= 1.0:
        raise ValueError("--solvency-c must lie in (0,1]")
    delta = float(payload["delta"])
    seed = int(payload["seed"])
    n_values = [
        int(n) for n in payload["n_values"]
        if int(n) <= betting.POISSON_MAX_N
    ]
    base_counts = {
        int(n): int(count)
        for n, count in payload["num_sims_by_n"].items()
    }

    if resume and checkpoint.exists():
        frame = pd.read_csv(checkpoint)
        if not _configuration_matches(
            frame, c=c, shrinkage=shrinkage, epsilon=epsilon
        ):
            raise ValueError("checkpoint PE tuning does not match this run")
        rows = frame.to_dict("records")
        completed = {
            (
                str(row.distribution),
                int(row.n),
                int(row.rep),
                str(row.calibration),
            )
            for row in frame.itertuples()
        }
        print(f"resuming with {len(completed)} completed PE intervals")
    else:
        rows = []
        completed = set()

    rng = np.random.default_rng(seed)
    randomizer_seed = np.random.SeedSequence(seed).spawn(1)[0]
    regularized_rng = np.random.default_rng(randomizer_seed)
    unbuffered_rng = np.random.default_rng(randomizer_seed)
    rng.uniform(0.0, 1.0, 20)
    samplers = _samplers(rng)

    warm = np.linspace(0.1, 0.9, 64)
    betting.compute_M_poisson_common_clock_arms(
        warm,
        0.5,
        delta,
        c=c,
        third_moment_shrinkage=shrinkage,
        skewness_epsilon=epsilon,
    )

    new_intervals = 0
    for n in n_values:
        for distribution_index, (distribution, sampler) in enumerate(
            samplers.items()
        ):
            for replication in range(max(base_counts[n], reps)):
                if replication < base_counts[n]:
                    x = np.ascontiguousarray(
                        np.asarray(sampler(n), dtype=float)
                    )
                    regularized_rng.uniform(size=2)
                    u_plus, u_minus = (
                        float(value)
                        for value in unbuffered_rng.uniform(size=2)
                    )
                else:
                    extension_seed = np.random.SeedSequence([
                        seed,
                        8675309,
                        n,
                        distribution_index,
                        replication,
                    ])
                    data_seed, auxiliary_seed = extension_seed.spawn(2)
                    extension_rng = np.random.default_rng(data_seed)
                    extension_sampler = _samplers(extension_rng)[distribution]
                    x = np.ascontiguousarray(
                        np.asarray(extension_sampler(n), dtype=float)
                    )
                    auxiliary_rng = np.random.default_rng(auxiliary_seed)
                    u_plus, u_minus = (
                        float(value)
                        for value in auxiliary_rng.uniform(size=2)
                    )

                if replication >= reps:
                    continue
                for calibration in CALIBRATIONS:
                    key = (distribution, n, replication, calibration)
                    if key in completed:
                        continue
                    randomizers = (
                        (1.0, 1.0)
                        if calibration == "Deterministic"
                        else (u_plus, u_minus)
                    )
                    lower, upper, empty, evaluations = (
                        betting.poisson_common_clock_ci_endpoints(
                            x,
                            delta,
                            randomizers=randomizers,
                            return_diagnostics=True,
                            c=c,
                            third_moment_shrinkage=shrinkage,
                            skewness_epsilon=epsilon,
                        )
                    )
                    width = max(float(upper - lower), 0.0)
                    sample_mean = float(np.mean(x))
                    true_mean = TRUE_MEANS[distribution]
                    clocks = betting._poisson_common_clock_parameters(
                        x, shrinkage, epsilon
                    )
                    rows.append({
                        "distribution": distribution,
                        "n": n,
                        "rep": replication,
                        "method": "PE-betting",
                        "calibration": calibration,
                        "lower": lower,
                        "upper": upper,
                        "width": width,
                        "sqrt_n_width": math.sqrt(n) * width,
                        "empty": bool(empty),
                        "noncoverage": not (
                            lower <= true_mean <= upper
                        ) or bool(empty),
                        "sample_mean_excluded": not (
                            lower <= sample_mean <= upper
                        ) or bool(empty),
                        "poisson_round_fraction": float(np.mean(clocks[3])),
                        "evaluations": int(evaluations),
                        "solvency_c": c,
                        "third_moment_shrinkage": shrinkage,
                        "skewness_epsilon": epsilon,
                    })
                    completed.add(key)
                    new_intervals += 1
                    if (
                        progress_every > 0
                        and new_intervals % progress_every == 0
                    ):
                        checkpoint.parent.mkdir(parents=True, exist_ok=True)
                        pd.DataFrame(rows).to_csv(checkpoint, index=False)
                        print(
                            f"completed {new_intervals} new PE intervals: "
                            f"{distribution}, n={n}, rep={replication + 1}",
                            flush=True,
                        )

    output = pd.DataFrame(rows).drop_duplicates(
        ["distribution", "n", "rep", "method", "calibration"],
        keep="last",
    )
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(checkpoint, index=False)
    return output


def _summarize_pe(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.groupby(
            ["distribution", "n", "method", "calibration"],
            as_index=False,
        )
        .agg(
            mean=("sqrt_n_width", "mean"),
            lo=("sqrt_n_width", lambda x: np.quantile(x, 0.1)),
            hi=("sqrt_n_width", lambda x: np.quantile(x, 0.9)),
            replications=("sqrt_n_width", "size"),
            noncoverage=("noncoverage", "mean"),
            empty_rate=("empty", "mean"),
            sample_mean_exclusion_rate=("sample_mean_excluded", "mean"),
            poisson_round_fraction=("poisson_round_fraction", "mean"),
        )
    )


def _validate_pe_grid(
    frame: pd.DataFrame,
    *,
    n_values: tuple[int, ...],
    reps: int,
) -> None:
    counts = frame.groupby(
        ["distribution", "n", "calibration"], as_index=False
    ).agg(replications=("rep", "nunique"))
    indexed = counts.set_index(["distribution", "n", "calibration"])[
        "replications"
    ]
    missing = []
    for distribution in PANEL_ORDER:
        for n in n_values:
            for calibration in CALIBRATIONS:
                count = int(indexed.get((distribution, n, calibration), 0))
                if count != reps:
                    missing.append(
                        f"{distribution}, n={n}, {calibration}: "
                        f"{count}/{reps}"
                    )
    if missing:
        raise ValueError(
            "incomplete PE grid:\n" + "\n".join(missing[:20])
        )


def _validate_saved_betting_grid(
    payload: dict[str, object],
    *,
    n_values: tuple[int, ...],
) -> None:
    payload_n = [int(n) for n in payload["n_values"]]
    indices = [payload_n.index(n) for n in n_values]
    missing = []
    for distribution in PANEL_ORDER:
        model_results = payload["results"].get(distribution, {})
        for method, _, _, _ in SAVED_METHODS:
            for suffix in (
                "deterministic_markov_diameter",
                "randomized_markov_diameter",
            ):
                key = f"{method}_{suffix}"
                rows = model_results.get(key, [])
                for n, index in zip(n_values, indices):
                    if (
                        index >= len(rows)
                        or not math.isfinite(float(rows[index]["mean"]))
                    ):
                        missing.append(f"{distribution}, n={n}, {key}")
    if missing:
        raise ValueError(
            "incomplete saved-betting grid:\n" + "\n".join(missing[:20])
        )


def _gaffke_record(
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


def _validate_gaffke_grid(
    frame: pd.DataFrame,
    *,
    n_values: tuple[int, ...],
    reps: int,
) -> None:
    counts = frame.groupby(
        ["distribution", "n", "method"], as_index=False
    ).agg(replications=("rep", "nunique"))
    indexed = counts.set_index(["distribution", "n", "method"])[
        "replications"
    ]
    missing = []
    for distribution in PANEL_ORDER:
        for n in n_values:
            for method in GAFFKE_METHODS:
                count = int(indexed.get((distribution, n, method), 0))
                if count != reps:
                    missing.append(
                        f"{distribution}, n={n}, {method}: {count}/{reps}"
                    )
    if missing:
        raise ValueError(
            "incomplete Gaffke grid:\n" + "\n".join(missing[:20])
        )


def _complete_gaffke_results(
    source: Path,
    cache: Path,
    *,
    n_values: tuple[int, ...],
    reps: int,
    progress_every: int,
) -> pd.DataFrame:
    if not math.isclose(GAFFKE_DELTA, 0.01):
        raise ValueError("unexpected Gaffke confidence level")
    frames = [pd.read_csv(source)]
    if cache.exists():
        frames.append(pd.read_csv(cache))
    frame = pd.concat(frames, ignore_index=True).drop_duplicates(
        ["distribution", "n", "rep", "method"], keep="last"
    )
    frame = frame[
        frame["distribution"].isin(PANEL_ORDER)
        & frame["n"].isin(n_values)
        & (frame["rep"] < reps)
        & frame["method"].isin(GAFFKE_METHODS)
    ].copy()
    completed = {
        (
            str(row.distribution),
            int(row.n),
            int(row.rep),
            str(row.method),
        )
        for row in frame.itertuples()
    }
    rows = frame.to_dict("records")
    new_intervals = 0
    distributions = {
        distribution.name: (index, distribution)
        for index, distribution in enumerate(GAFFKE_DISTRIBUTIONS)
    }
    for distribution_name in PANEL_ORDER:
        distribution_index, distribution = distributions[distribution_name]
        for replication in range(reps):
            missing_n = [
                n
                for n in n_values
                if any(
                    (distribution_name, n, replication, method)
                    not in completed
                    for method in GAFFKE_METHODS
                )
            ]
            if not missing_n:
                continue
            seed_sequence = np.random.SeedSequence(
                [GAFFKE_SEED, distribution_index, replication]
            )
            data_seed, _ = seed_sequence.spawn(2)
            data_rng = np.random.default_rng(data_seed)
            path = np.asarray(
                distribution.sampler(data_rng, max(n_values)), dtype=float
            )
            for n in missing_n:
                x = np.ascontiguousarray(path[:n])
                lower, upper, backend = fast_gaffke_ci(
                    x,
                    delta=GAFFKE_DELTA,
                    binary=distribution_name.startswith("Bernoulli"),
                    exact_cutoff=3_000,
                )
                deterministic_key = (
                    distribution_name,
                    n,
                    replication,
                    "Gaffke",
                )
                if deterministic_key not in completed:
                    rows.append(_gaffke_record(
                        distribution_name,
                        n,
                        replication,
                        "Gaffke",
                        lower,
                        upper,
                        backend,
                    ))
                    completed.add(deterministic_key)
                    new_intervals += 1
                randomized_key = (
                    distribution_name,
                    n,
                    replication,
                    "Randomized Gaffke",
                )
                if randomized_key not in completed:
                    auxiliary_seed = np.random.SeedSequence([
                        GAFFKE_SEED,
                        GAFFKE_EXTENSION_SEED_TAG,
                        distribution_index,
                        replication,
                        n,
                    ])
                    auxiliary_rng = np.random.default_rng(auxiliary_seed)
                    u_plus, u_minus = (
                        float(value)
                        for value in auxiliary_rng.uniform(size=2)
                    )
                    randomized_lower, randomized_upper = (
                        _randomized_endpoints(
                            x,
                            lower,
                            upper,
                            u_plus,
                            u_minus,
                            GAFFKE_DELTA,
                        )
                    )
                    rows.append(_gaffke_record(
                        distribution_name,
                        n,
                        replication,
                        "Randomized Gaffke",
                        randomized_lower,
                        randomized_upper,
                        f"{backend}+product-orthant-indexed",
                    ))
                    completed.add(randomized_key)
                    new_intervals += 1
                if (
                    progress_every > 0
                    and new_intervals % progress_every == 0
                ):
                    print(
                        f"completed {new_intervals} missing Gaffke intervals: "
                        f"{distribution_name}, n={n}, "
                        f"rep={replication + 1}",
                        flush=True,
                    )

    output = pd.DataFrame(rows).drop_duplicates(
        ["distribution", "n", "rep", "method"], keep="last"
    ).sort_values(["distribution", "n", "rep", "method"])
    _validate_gaffke_grid(output, n_values=n_values, reps=reps)
    cache.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(cache, index=False)
    return output


def _summarize_gaffke(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["calibration"] = frame["method"].map({
        "Gaffke": "Deterministic",
        "Randomized Gaffke": "Uniformly randomized",
    })
    return (
        frame.groupby(["distribution", "n", "calibration"], as_index=False)
        .agg(mean=("sqrt_n_width", "mean"))
    )


def _plot_saved_series(
    axis: plt.Axes,
    n_values: np.ndarray,
    rows: list[dict[str, float]],
    *,
    color: str,
    marker: str,
    randomized: bool,
) -> None:
    axis.plot(
        n_values,
        [row["mean"] for row in rows],
        color=color,
        marker=marker,
        markerfacecolor=color if randomized else "none",
        markeredgecolor=color,
        markeredgewidth=0.9,
        ms=3.8 if randomized else 5.2,
        lw=1.8,
        ls="-" if randomized else "--",
        zorder=3 if randomized else 4,
    )


def _make_figure(
    payload: dict[str, object],
    pe_summary: pd.DataFrame,
    gaffke: pd.DataFrame,
    destination: Path,
    solvency_c: float,
) -> None:
    all_n = np.asarray(payload["n_values"], dtype=int)
    indices = np.flatnonzero(all_n <= betting.POISSON_MAX_N)
    n_values = all_n[indices]
    delta = float(payload["delta"])
    gaussian_quantile = NormalDist().inv_cdf(1.0 - delta / 2.0)
    fig, axes = plt.subplots(3, 3, figsize=(13.5, 11.3))

    for axis, distribution in zip(axes.ravel(), PANEL_ORDER):
        model_results = payload["results"][distribution]
        gaussian_limit = (
            2.0
            * math.sqrt(VARIANCES[distribution])
            * gaussian_quantile
        )
        axis.plot(
            n_values,
            np.full(n_values.size, gaussian_limit),
            color="black",
            ls=":",
            lw=1.5,
        )
        axis.plot(
            n_values,
            np.full(
                n_values.size,
                float(model_results["target_product"]),
            ),
            color="#9467bd",
            ls=":",
            lw=1.5,
        )

        for key, color, marker, _ in SAVED_METHODS:
            for calibration, randomized in (
                ("Uniformly randomized", True),
                ("Deterministic", False),
            ):
                suffix = (
                    "randomized_markov_diameter"
                    if randomized
                    else "deterministic_markov_diameter"
                )
                rows = model_results[f"{key}_{suffix}"]
                _plot_saved_series(
                    axis,
                    n_values,
                    [rows[index] for index in indices],
                    color=color,
                    marker=marker,
                    randomized=randomized,
                )

        pe_color, pe_marker, _ = PE_STYLE
        for calibration, randomized in (
            ("Uniformly randomized", True),
            ("Deterministic", False),
        ):
            rows = pe_summary[
                (pe_summary["distribution"] == distribution)
                & (pe_summary["calibration"] == calibration)
            ].sort_values("n")
            axis.plot(
                rows["n"],
                rows["mean"],
                color=pe_color,
                marker=pe_marker,
                markerfacecolor=pe_color if randomized else "none",
                markeredgecolor=pe_color,
                markeredgewidth=0.9,
                ms=3.8 if randomized else 5.2,
                lw=1.8,
                ls="-" if randomized else "--",
                zorder=5,
            )

        gaffke_color, gaffke_marker, _ = GAFFKE_STYLE
        for calibration, randomized in (
            ("Uniformly randomized", True),
            ("Deterministic", False),
        ):
            rows = gaffke[
                (gaffke["distribution"] == distribution)
                & (gaffke["calibration"] == calibration)
            ].sort_values("n")
            axis.plot(
                rows["n"],
                rows["mean"],
                color=gaffke_color,
                marker=gaffke_marker,
                markerfacecolor=(
                    gaffke_color if randomized else "none"
                ),
                markeredgecolor=gaffke_color,
                markeredgewidth=0.9,
                ms=3.8 if randomized else 5.2,
                lw=1.8,
                ls="-" if randomized else "--",
                zorder=4,
            )

        axis.set_xscale("log")
        axis.set_xlim(10, betting.POISSON_MAX_N)
        axis.set_title(distribution)
        axis.set_xlabel("sample size $n$")
        axis.set_ylabel(r"$\sqrt{n}\times$ CI width")
        axis.grid(True, ls="--", alpha=0.3)

    method_styles = [
        ("#9467bd", "s", "Product betting"),
        ("darkorange", "P", "STaR betting"),
        ("#2ca02c", "h", "GE-betting"),
        (PE_STYLE[0], PE_STYLE[1], PE_STYLE[2]),
        (GAFFKE_STYLE[0], GAFFKE_STYLE[1], GAFFKE_STYLE[2]),
    ]
    method_handles = [
        Line2D(
            [0], [0], color=color, marker=marker, lw=2,
            markerfacecolor=color, markeredgecolor=color, ms=4.5,
        )
        for color, marker, _ in method_styles
    ]
    calibration_handles = [
        Line2D(
            [0], [0], color="0.25", ls="--", marker="o", lw=2,
            markerfacecolor="none", markeredgecolor="0.25", ms=5.2,
        ),
        Line2D(
            [0], [0], color="0.25", ls="-", marker="o", lw=2,
            markerfacecolor="0.25", markeredgecolor="0.25", ms=3.8,
        ),
    ]
    reference_handles = [
        Line2D([0], [0], color="#9467bd", ls=":", lw=1.7),
        Line2D([0], [0], color="black", ls=":", lw=1.7),
    ]
    fig.suptitle(
        "Finite-sample betting and Gaffke confidence intervals "
        f"($c={solvency_c:g}$)",
        fontsize=15,
    )
    fig.legend(
        method_handles,
        [label for _, _, label in method_styles],
        loc="lower center",
        bbox_to_anchor=(0.31, 0.012),
        ncol=5,
        fontsize=8.2,
        title="Construction (color and marker)",
        title_fontsize=8.5,
        frameon=False,
    )
    fig.legend(
        calibration_handles,
        ["Deterministic", "Uniformly randomized"],
        loc="lower center",
        bbox_to_anchor=(0.72, 0.012),
        ncol=2,
        fontsize=8.2,
        title="Calibration (line and marker fill)",
        title_fontsize=8.5,
        frameon=False,
    )
    fig.legend(
        reference_handles,
        ["Product limit", "Gaussian limit"],
        loc="lower center",
        bbox_to_anchor=(0.93, 0.012),
        fontsize=8.2,
        title="Reference (dotted)",
        title_fontsize=8.5,
        frameon=False,
    )
    fig.tight_layout(rect=(0.0, 0.09, 1.0, 0.955))
    PAPER_PLOTS.mkdir(parents=True, exist_ok=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    for path in (destination, PAPER_PLOTS / destination.name):
        fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    with args.input.open(encoding="utf-8") as stream:
        payload = json.load(stream)
    payload_c = float(payload.get("solvency_c", args.solvency_c))
    product_c = float(payload.get("product_solvency_c", payload_c))
    if not (
        math.isclose(payload_c, args.solvency_c)
        and math.isclose(product_c, args.solvency_c)
    ):
        raise ValueError(
            "saved betting results do not use the requested common "
            f"solvency c={args.solvency_c:g}: STaR/GE use {payload_c:g} "
            f"and product betting uses {product_c:g}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output_dir / "pe_intervals.csv"
    if args.plot_only:
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)
        pe = pd.read_csv(checkpoint)
    else:
        pe = _run_pe(
            payload,
            checkpoint,
            reps=args.reps,
            resume=args.resume,
            progress_every=args.progress_every,
            c=args.solvency_c,
            shrinkage=args.third_moment_shrinkage,
            epsilon=args.skewness_epsilon,
        )
    n_values = tuple(
        int(n) for n in payload["n_values"]
        if int(n) <= betting.POISSON_MAX_N
    )
    if n_values != CONSIDERED_N:
        raise ValueError(
            f"expected considered sample sizes {CONSIDERED_N}, got {n_values}"
        )
    _validate_saved_betting_grid(payload, n_values=n_values)
    _validate_pe_grid(pe, n_values=n_values, reps=args.reps)
    pe_summary = _summarize_pe(pe)
    pe_summary.to_csv(args.output_dir / "pe_summary.csv", index=False)
    complete_gaffke_path = args.output_dir / "gaffke_complete.csv"
    gaffke_intervals = _complete_gaffke_results(
        args.gaffke_results,
        complete_gaffke_path,
        n_values=n_values,
        reps=args.reps,
        progress_every=args.progress_every,
    )
    gaffke = _summarize_gaffke(gaffke_intervals)
    _make_figure(
        payload,
        pe_summary,
        gaffke,
        args.output_dir / args.plot_name,
        args.solvency_c,
    )
    with (args.output_dir / "config.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump({
            "delta": float(payload["delta"]),
            "betting_seed": int(payload["seed"]),
            "n_values": [
                int(n) for n in payload["n_values"]
                if int(n) <= betting.POISSON_MAX_N
            ],
            "replications": int(args.reps),
            "solvency_c": float(args.solvency_c),
            "third_moment_shrinkage": float(
                args.third_moment_shrinkage
            ),
            "skewness_epsilon": float(args.skewness_epsilon),
            "poisson_lookup": (
                "real-order regularized-incomplete-gamma inversion; "
                "linear interpolation in target probability and log lambda"
            ),
            "randomization": (
                "one independent uniform per arm, fixed over candidate "
                "means and shared with the replayed Figure 3 betting paths"
            ),
            "gaffke_results": str(args.gaffke_results),
            "gaffke_complete_results": str(complete_gaffke_path),
            "gaffke_extension_seed": int(GAFFKE_SEED),
            "gaffke_extension_seed_tag": int(GAFFKE_EXTENSION_SEED_TAG),
            "gaffke_extension_randomization": (
                "independently indexed by distribution, replication, and n"
            ),
            "complete_method_grid": True,
            "plot_name": args.plot_name,
        }, stream, indent=2)
    print(f"saved PE results and {args.plot_name} to {args.output_dir}")


if __name__ == "__main__":
    main()
