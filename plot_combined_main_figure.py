"""Build the paper's combined betting--Gaffke confidence-interval figure.

The betting curves come from the saved fixed-horizon experiment.  Ordinary
Gaffke curves come from the two saved Gaffke experiments.  The randomized
product-orthant Gaffke endpoints are reconstructed from those experiments'
documented seeds and ordinary endpoints, without rerunning any betting
inversions.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


HERE = Path(__file__).resolve().parent
PAPER_DIR = HERE.parent / "paper"
BETTING_RESULTS = HERE / "plots" / "ci_width_original_vs_star.json"
SMALL_GAFFKE_DIR = next(
    directory
    for directory in (HERE / "gaffke_comparison").glob(
        "star_*_gaffke_results"
    )
    if (directory / "config.json").exists()
)
LARGE_GAFFKE_DIR = (
    HERE / "gaffke_comparison" / "large_sample_gaffke_results"
)
GAFFKE_CACHE = (
    HERE / "gaffke_comparison" / "combined_main_figure_gaffke.csv"
)
PAPER_OUTPUT = PAPER_DIR / "plots" / "ci_width_original_vs_star.png"
BETTING_OUTPUT = HERE / "plots" / "ci_width_original_vs_star.png"

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
ROW_MINIMUM_N = (10, 100, 1_000)
MAXIMUM_N = 1_000_000
GAFFKE_RANDOMIZED_ORDER = (
    "Uniform(0,1)",
    "Beta(1,5)",
    "Beta(0.5,0.5)",
    "Bernoulli(0.1)",
    "Beta(50,50)",
    "Beta(20,80)",
    "Beta(2,2)",
    "Bernoulli(0.5)",
    "Uniform(0.45,0.55)",
)

METHODS = (
    ("product_original", "#9467bd", "s", "Product betting"),
    (
        "product_star_common_clock",
        "darkorange",
        "P",
        "STaR betting",
    ),
)


@dataclass(frozen=True)
class DistributionSpec:
    name: str
    variance: float
    sampler: object


DISTRIBUTIONS = (
    DistributionSpec(
        "Beta(2,2)",
        1.0 / 20.0,
        lambda rng, n: rng.beta(2.0, 2.0, n),
    ),
    DistributionSpec(
        "Beta(1,5)",
        5.0 / 252.0,
        lambda rng, n: rng.beta(1.0, 5.0, n),
    ),
    DistributionSpec(
        "Bernoulli(0.5)",
        0.25,
        lambda rng, n: rng.binomial(1, 0.5, n).astype(float),
    ),
    DistributionSpec(
        "Uniform(0,1)",
        1.0 / 12.0,
        lambda rng, n: rng.random(n),
    ),
    DistributionSpec(
        "Beta(0.5,0.5)",
        1.0 / 8.0,
        lambda rng, n: rng.beta(0.5, 0.5, n),
    ),
    DistributionSpec(
        "Bernoulli(0.1)",
        0.09,
        lambda rng, n: rng.binomial(1, 0.1, n).astype(float),
    ),
    DistributionSpec(
        "Beta(50,50)",
        1.0 / 404.0,
        lambda rng, n: rng.beta(50.0, 50.0, n),
    ),
    DistributionSpec(
        "Beta(20,80)",
        1600.0 / 1_010_000.0,
        lambda rng, n: rng.beta(20.0, 80.0, n),
    ),
    DistributionSpec(
        "Uniform(0.45,0.55)",
        0.1**2 / 12.0,
        lambda rng, n: rng.uniform(0.45, 0.55, n),
    ),
)
DISTRIBUTION_BY_NAME = {
    distribution.name: distribution for distribution in DISTRIBUTIONS
}


def _efficient_saved_key(model_results: dict[str, object]) -> str:
    suffix = "_common_clock_randomized_markov_diameter"
    matches = [
        key.removesuffix("_randomized_markov_diameter")
        for key in model_results
        if key.startswith("probit_") and key.endswith(suffix)
    ]
    if len(matches) != 1:
        raise ValueError(
            "Expected one shared-scale efficient-betting series, "
            f"found {len(matches)}"
        )
    return matches[0]


def _path_max_n(
    rep: int,
    sample_sizes: list[int],
    reps_by_n: dict[int, int],
    maximum_n: int,
) -> int:
    eligible = [
        n
        for n in sample_sizes
        if n <= maximum_n and rep < reps_by_n[n]
    ]
    return max(eligible) if eligible else 0


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
        math.log(tail_probability)
        - math.log(uniform)
        + log_product
    ) / n
    candidate = math.exp(log_candidate)
    return max(ordinary_endpoint, min(sample_minimum, candidate))


def _reconstruct_source(
    directory: Path,
    *,
    minimum_n: int,
    maximum_n: int,
    allowed_distributions: set[str],
) -> pd.DataFrame:
    with (directory / "config.json").open(encoding="utf-8") as stream:
        config = json.load(stream)
    results = pd.read_csv(directory / "results.csv")
    ordinary = results[results["method"] == "Gaffke"].copy()

    sample_sizes = [
        int(n)
        for n in config["sample_sizes"]
        if minimum_n <= int(n) <= maximum_n
    ]
    all_sample_sizes = [int(n) for n in config["sample_sizes"]]
    reps_by_n = {
        int(n): int(count)
        for n, count in config["reps_by_n"].items()
    }
    ordinary = ordinary[
        ordinary["distribution"].isin(allowed_distributions)
        & ordinary["n"].isin(sample_sizes)
    ]
    endpoint_rows = {
        (row.distribution, int(row.n), int(row.rep)): row
        for row in ordinary.itertuples()
    }

    distribution_by_name = {
        distribution.name: (index, distribution)
        for index, distribution in enumerate(DISTRIBUTIONS)
    }
    tail_probability = float(config["delta"]) / 2.0
    randomized_rows: list[dict[str, object]] = []

    for distribution_name in GAFFKE_RANDOMIZED_ORDER:
        if distribution_name not in allowed_distributions:
            continue
        distribution_index, distribution = distribution_by_name[
            distribution_name
        ]
        relevant = ordinary[
            ordinary["distribution"] == distribution_name
        ]
        if relevant.empty:
            continue
        maximum_rep = int(relevant["rep"].max()) + 1
        for rep in range(maximum_rep):
            maximum_path_n = _path_max_n(
                rep,
                all_sample_sizes,
                reps_by_n,
                maximum_n,
            )
            if maximum_path_n == 0:
                continue
            seed_sequence = np.random.SeedSequence(
                [int(config["seed"]), distribution_index, rep]
            )
            data_seed, auxiliary_seed = seed_sequence.spawn(2)
            data_rng = np.random.default_rng(data_seed)
            auxiliary_rng = np.random.default_rng(auxiliary_seed)
            path = np.asarray(
                distribution.sampler(data_rng, maximum_path_n),
                dtype=float,
            )

            running_minimum = math.inf
            running_maximum = -math.inf
            log_product = 0.0
            log_reflected_product = 0.0
            lower_product_zero = False
            upper_product_zero = False
            previous_n = 0

            for n in all_sample_sizes:
                if n > maximum_n or rep >= reps_by_n[n]:
                    continue
                u_plus, u_minus = (
                    float(value)
                    for value in auxiliary_rng.uniform(size=2)
                )
                chunk = path[previous_n:n]
                previous_n = n
                if chunk.size:
                    running_minimum = min(
                        running_minimum, float(np.min(chunk))
                    )
                    running_maximum = max(
                        running_maximum, float(np.max(chunk))
                    )
                    if not lower_product_zero:
                        if np.any(chunk <= 0.0):
                            lower_product_zero = True
                            log_product = -math.inf
                        else:
                            log_product += float(np.sum(np.log(chunk)))
                    if not upper_product_zero:
                        reflected = 1.0 - chunk
                        if np.any(reflected <= 0.0):
                            upper_product_zero = True
                            log_reflected_product = -math.inf
                        else:
                            log_reflected_product += float(
                                np.sum(np.log(reflected))
                            )

                if n < minimum_n:
                    continue
                key = (distribution_name, n, rep)
                if key not in endpoint_rows:
                    continue
                ordinary_row = endpoint_rows[key]
                lower = _randomized_lower_endpoint(
                    n=n,
                    sample_minimum=running_minimum,
                    log_product=log_product,
                    uniform=u_plus,
                    ordinary_endpoint=float(ordinary_row.lower),
                    tail_probability=tail_probability,
                )
                reflected_lower = _randomized_lower_endpoint(
                    n=n,
                    sample_minimum=1.0 - running_maximum,
                    log_product=log_reflected_product,
                    uniform=u_minus,
                    ordinary_endpoint=1.0 - float(ordinary_row.upper),
                    tail_probability=tail_probability,
                )
                upper = 1.0 - reflected_lower
                randomized_rows.append(
                    {
                        "distribution": distribution_name,
                        "n": n,
                        "rep": rep,
                        "method": "Randomized Gaffke",
                        "width": max(upper - lower, 0.0),
                        "sqrt_n_width": math.sqrt(n)
                        * max(upper - lower, 0.0),
                    }
                )

    ordinary = ordinary[
        ["distribution", "n", "rep", "method", "width", "sqrt_n_width"]
    ]
    return pd.concat(
        [ordinary, pd.DataFrame(randomized_rows)],
        ignore_index=True,
    )


def _load_gaffke_results() -> pd.DataFrame:
    first_six = set(PANEL_ORDER[:6])
    low_variance = set(PANEL_ORDER[6:])
    small = _reconstruct_source(
        SMALL_GAFFKE_DIR,
        minimum_n=10,
        maximum_n=999,
        allowed_distributions=first_six,
    )
    large = _reconstruct_source(
        LARGE_GAFFKE_DIR,
        minimum_n=1_000,
        maximum_n=MAXIMUM_N,
        allowed_distributions=set(PANEL_ORDER),
    )
    output = pd.concat([small, large], ignore_index=True)
    output.to_csv(GAFFKE_CACHE, index=False)
    return output


def _summarize_gaffke(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.groupby(["distribution", "n", "method"], as_index=False)
        .agg(
            mean=("sqrt_n_width", "mean"),
            lo=("sqrt_n_width", lambda values: np.quantile(values, 0.10)),
            hi=("sqrt_n_width", lambda values: np.quantile(values, 0.90)),
        )
    )


def _plot_saved_rows(
    axis,
    n_values: np.ndarray,
    rows: list[dict[str, float]],
    *,
    color: str,
    marker: str | None,
    label: str,
    linestyle: str,
    show_band: bool,
    filled_marker: bool,
    marker_size: float,
    zorder: float,
) -> None:
    means = np.asarray([row["mean"] for row in rows], dtype=float)
    axis.plot(
        n_values,
        means,
        color=color,
        marker=marker,
        markerfacecolor=color if filled_marker else "none",
        markeredgecolor=color,
        markeredgewidth=0.9,
        ms=marker_size,
        lw=1.9,
        ls=linestyle,
        label=label,
        zorder=zorder,
    )
    if show_band:
        lows = np.asarray([row["lo"] for row in rows], dtype=float)
        highs = np.asarray([row["hi"] for row in rows], dtype=float)
        axis.fill_between(
            n_values,
            lows,
            highs,
            color=color,
            alpha=0.07,
            linewidth=0.0,
            zorder=1,
        )


def make_figure() -> Path:
    with BETTING_RESULTS.open(encoding="utf-8") as stream:
        payload = json.load(stream)
    results = payload["results"]
    all_n = np.asarray(payload["n_values"], dtype=int)
    delta = float(payload["delta"])
    gaussian_quantile = NormalDist().inv_cdf(1.0 - delta / 2.0)
    gaffke = _summarize_gaffke(_load_gaffke_results())

    fig, axes = plt.subplots(3, 3, figsize=(13.5, 11.3))
    for panel_index, (axis, name) in enumerate(
        zip(axes.ravel(), PANEL_ORDER)
    ):
        model_results = results[name]
        minimum_n = ROW_MINIMUM_N[panel_index // 3]
        betting_mask = (all_n >= minimum_n) & (all_n <= MAXIMUM_N)
        betting_n = all_n[betting_mask]
        betting_indices = np.flatnonzero(betting_mask)

        gaussian_limit = (
            2.0
            * math.sqrt(DISTRIBUTION_BY_NAME[name].variance)
            * gaussian_quantile
        )
        reference_lines = [
            (gaussian_limit, "black", "Gaussian limit"),
        ]
        if panel_index < 3:
            reference_lines.insert(
                0,
                (
                    model_results["target_product"],
                    "#9467bd",
                    "Product limit",
                ),
            )
        for target_value, color, label in reference_lines:
            axis.plot(
                betting_n,
                np.full(betting_n.size, target_value),
                color=color,
                ls=":",
                lw=1.5,
                label="_nolegend_",
            )

        row_methods = METHODS if panel_index < 3 else METHODS[1:]
        panel_methods = row_methods + (
            (
                _efficient_saved_key(model_results),
                "#2ca02c",
                "h",
                "Efficient betting",
            ),
        )
        for key, color, marker, label in panel_methods:
            deterministic = model_results[
                f"{key}_deterministic_markov_diameter"
            ]
            randomized = model_results[
                f"{key}_randomized_markov_diameter"
            ]
            _plot_saved_rows(
                axis,
                betting_n,
                [randomized[index] for index in betting_indices],
                color=color,
                marker=marker,
                label="_nolegend_",
                linestyle="-",
                show_band=True,
                filled_marker=True,
                marker_size=3.8,
                zorder=3,
            )
            _plot_saved_rows(
                axis,
                betting_n,
                [deterministic[index] for index in betting_indices],
                color=color,
                marker=marker,
                label="_nolegend_",
                linestyle="--",
                show_band=False,
                filled_marker=False,
                marker_size=5.2,
                zorder=4,
            )

        panel_gaffke = gaffke[
            (gaffke["distribution"] == name)
            & (gaffke["n"] >= minimum_n)
            & (gaffke["n"] <= MAXIMUM_N)
        ]
        for method, linestyle, filled_marker, marker_size, zorder in (
            (
                "Randomized Gaffke",
                "-",
                True,
                3.8,
                3,
            ),
            (
                "Gaffke",
                "--",
                False,
                5.2,
                4,
            ),
        ):
            method_rows = panel_gaffke[
                panel_gaffke["method"] == method
            ].sort_values("n")
            axis.plot(
                method_rows["n"],
                method_rows["mean"],
                color="#1976b9",
                marker="o",
                markerfacecolor=(
                    "#1976b9" if filled_marker else "none"
                ),
                markeredgecolor="#1976b9",
                markeredgewidth=0.9,
                ls=linestyle,
                lw=1.8,
                ms=marker_size,
                label="_nolegend_",
                zorder=zorder,
            )

        axis.set_xscale("log")
        axis.set_xlim(minimum_n, MAXIMUM_N)
        axis.set_title(name)
        axis.set_xlabel("sample size $n$")
        axis.set_ylabel(r"$\sqrt{n}\times$ CI width")
        axis.grid(True, ls="--", alpha=0.3)

    method_handles = [
        Line2D(
            [0], [0], color=color, marker=marker, lw=2,
            markerfacecolor=color, markeredgecolor=color, ms=4.5,
        )
        for color, marker in (
            ("#9467bd", "s"),
            ("darkorange", "P"),
            ("#2ca02c", "h"),
            ("#1976b9", "o"),
        )
    ]
    method_labels = [
        "Product betting",
        "STaR betting",
        "Efficient betting",
        "Gaffke",
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
    calibration_labels = ["Deterministic", "Uniformly randomized"]
    reference_handles = [
        Line2D([0], [0], color="#9467bd", ls=":", lw=1.7),
        Line2D([0], [0], color="black", ls=":", lw=1.7),
    ]
    reference_labels = ["Product limit", "Gaussian limit"]
    fig.suptitle(
        "Betting and Gaffke confidence intervals",
        fontsize=15,
    )
    fig.legend(
        method_handles,
        method_labels,
        loc="lower center",
        bbox_to_anchor=(0.27, 0.012),
        ncol=4,
        fontsize=8.4,
        title="Construction (color and marker)",
        title_fontsize=8.6,
        frameon=False,
    )
    fig.legend(
        calibration_handles,
        calibration_labels,
        loc="lower center",
        bbox_to_anchor=(0.65, 0.012),
        ncol=2,
        fontsize=8.4,
        title="Calibration (line and marker fill)",
        title_fontsize=8.6,
        frameon=False,
    )
    fig.legend(
        reference_handles,
        reference_labels,
        loc="lower center",
        bbox_to_anchor=(0.89, 0.012),
        ncol=2,
        fontsize=8.4,
        title="Asymptotic reference (dotted)",
        title_fontsize=8.6,
        frameon=False,
    )
    fig.tight_layout(rect=(0.0, 0.09, 1.0, 0.955))
    for destination in (PAPER_OUTPUT, BETTING_OUTPUT):
        destination.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(destination, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return PAPER_OUTPUT


if __name__ == "__main__":
    print(make_figure())
