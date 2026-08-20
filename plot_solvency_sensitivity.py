#!/usr/bin/env python3
"""Combine the deterministic solvency-sensitivity comparisons."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parent
PAPER_ROOT = ROOT.parent / "paper"
OUTPUT = ROOT / "plots" / "solvency_c_comparison" / (
    "ci_width_solvency_combined.png"
)
PAPER_OUTPUT = PAPER_ROOT / "plots" / "ci_width_solvency_combined.png"

EXPERIMENTS = {
    0.5: ROOT / "plots" / "ci_width_original_vs_star_c05.json",
    0.75: ROOT / "plots" / "ci_width_original_vs_star_c075.json",
    1.0: ROOT / "plots" / "solvency_c_comparison" / (
        "ci_width_all_methods_c1.json"
    ),
}

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

METHODS = (
    (
        "product_original",
        "#9467bd",
        "Product betting",
    ),
    (
        "product_star_common_clock",
        "darkorange",
        "STaR betting",
    ),
    (
        "probit_common_clock",
        "#2ca02c",
        "GE-betting",
    ),
)

C_STYLES = {
    0.5: {
        "linestyle": "-",
        "marker": "o",
        "markerfacecolor": "color",
        "markersize": 3.8,
        "label": r"$c=1/2$",
    },
    0.75: {
        "linestyle": "--",
        "marker": "D",
        "markerfacecolor": "white",
        "markersize": 4.1,
        "label": r"$c=3/4$",
    },
    1.0: {
        "linestyle": ":",
        "marker": "^",
        "markerfacecolor": "white",
        "markersize": 4.5,
        "label": r"$c=1$",
    },
}


def load_experiments() -> dict[float, dict[str, object]]:
    payloads: dict[float, dict[str, object]] = {}
    reference_n: list[int] | None = None
    for c_value, path in EXPERIMENTS.items():
        with path.open(encoding="utf-8") as stream:
            payload = json.load(stream)

        saved_c = float(payload.get("solvency_c", c_value))
        product_c = float(payload.get("product_solvency_c", saved_c))
        if not np.isclose(saved_c, c_value) or not np.isclose(
            product_c, c_value
        ):
            raise ValueError(
                f"{path} does not contain c={c_value:g} for every method"
            )

        n_values = [int(value) for value in payload["n_values"]]
        if reference_n is None:
            reference_n = n_values
        elif n_values != reference_n:
            raise ValueError("saved experiments use different horizon grids")
        payloads[c_value] = payload
    return payloads


def make_figure(output_paths: Sequence[Path] | None = None) -> Path:
    payloads = load_experiments()
    all_n = np.asarray(payloads[0.5]["n_values"], dtype=int)
    figure, axes = plt.subplots(3, 3, figsize=(13.5, 11.3))

    for panel_index, (axis, distribution) in enumerate(
        zip(axes.ravel(), PANEL_ORDER)
    ):
        minimum_n = ROW_MINIMUM_N[panel_index // 3]
        mask = (all_n >= minimum_n) & (all_n <= MAXIMUM_N)
        n_values = all_n[mask]
        indices = np.flatnonzero(mask)
        displayed_values: list[float] = []

        for method_key, color, _ in METHODS:
            result_key = (
                f"{method_key}_deterministic_markov_diameter"
            )
            for zorder, (c_value, c_style) in enumerate(
                C_STYLES.items(), start=3
            ):
                rows = payloads[c_value]["results"][distribution][result_key]
                means = np.asarray(
                    [float(rows[index]["mean"]) for index in indices]
                )
                displayed_values.extend(means.tolist())
                marker_face = (
                    color
                    if c_style["markerfacecolor"] == "color"
                    else c_style["markerfacecolor"]
                )
                axis.plot(
                    n_values,
                    means,
                    color=color,
                    linestyle=c_style["linestyle"],
                    linewidth=1.65,
                    marker=c_style["marker"],
                    markerfacecolor=marker_face,
                    markeredgecolor=color,
                    markeredgewidth=0.9,
                    markersize=c_style["markersize"],
                    zorder=zorder,
                )

        lower = min(displayed_values)
        upper = max(displayed_values)
        padding = max(0.06 * (upper - lower), 0.008 * abs(upper))
        axis.set_ylim(lower - padding, upper + padding)
        axis.set_xscale("log")
        axis.set_xlim(minimum_n, MAXIMUM_N)
        axis.set_title(distribution)
        axis.set_xlabel("sample size $n$")
        axis.set_ylabel(r"$\sqrt{n}\times$ CI width")
        axis.grid(True, linestyle="--", alpha=0.3)

    method_handles = [
        Line2D([0], [0], color=color, linewidth=2.2)
        for _, color, _ in METHODS
    ]
    method_labels = [label for _, _, label in METHODS]
    c_handles = [
        Line2D(
            [0],
            [0],
            color="0.25",
            linestyle=style["linestyle"],
            linewidth=1.8,
            marker=style["marker"],
            markerfacecolor=(
                "0.25"
                if style["markerfacecolor"] == "color"
                else style["markerfacecolor"]
            ),
            markeredgecolor="0.25",
            markeredgewidth=0.9,
            markersize=style["markersize"] + 0.4,
        )
        for style in C_STYLES.values()
    ]
    c_labels = [style["label"] for style in C_STYLES.values()]

    figure.suptitle(
        "Sensitivity to the solvency fraction",
        fontsize=15,
    )
    figure.legend(
        method_handles,
        method_labels,
        loc="lower center",
        bbox_to_anchor=(0.31, 0.012),
        ncol=3,
        fontsize=8.6,
        title="Betting construction (color)",
        title_fontsize=8.8,
        frameon=False,
    )
    figure.legend(
        c_handles,
        c_labels,
        loc="lower center",
        bbox_to_anchor=(0.75, 0.012),
        ncol=3,
        fontsize=8.6,
        title=r"Solvency fraction (line and marker)",
        title_fontsize=8.8,
        frameon=False,
    )
    figure.tight_layout(rect=(0.0, 0.09, 1.0, 0.955))

    destinations = tuple(
        (OUTPUT, PAPER_OUTPUT) if output_paths is None else output_paths
    )
    if not destinations:
        raise ValueError("output_paths must contain at least one path")
    for destination in destinations:
        destination.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(destination, dpi=220, bbox_inches="tight")
    plt.close(figure)
    return destinations[0]


if __name__ == "__main__":
    print(make_figure())
