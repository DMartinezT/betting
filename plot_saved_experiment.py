"""Regenerate the main STaR figures from a saved experiment JSON file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


HERE = Path(__file__).resolve().parent
DEFAULT_INPUT = (
    HERE / "plots" / "solvency_c_comparison" / "ci_width_all_methods_c1.json"
)
PAPER_PLOT_DIR = HERE.parent / "paper" / "plots"


def _simulation_label(counts_by_n: dict[str, int]) -> str:
    counts = list(counts_by_n.values())
    if min(counts) == max(counts):
        return str(counts[0])
    return f"{min(counts)}\N{EN DASH}{max(counts)}"


def _series(
    axis,
    n_values,
    rows,
    color,
    marker,
    label,
    linestyle="-",
    fill=True,
    filled_marker=None,
    marker_size=4.5,
):
    means = [row["mean"] for row in rows]
    lows = [row["lo"] for row in rows]
    highs = [row["hi"] for row in rows]
    marker_kwargs = {}
    if filled_marker is not None:
        marker_kwargs = {
            "markerfacecolor": color if filled_marker else "none",
            "markeredgecolor": color,
            "markeredgewidth": 0.9,
        }
    axis.plot(
        n_values,
        means,
        color=color,
        marker=marker,
        ms=marker_size,
        lw=2,
        ls=linestyle,
        label=label,
        **marker_kwargs,
    )
    if fill:
        axis.fill_between(n_values, lows, highs, color=color, alpha=0.08)


def _largest_component_series(axis, n_values, rows, color):
    means = [row["mean"] for row in rows]
    axis.plot(
        n_values,
        means,
        color=color,
        ls="--",
        lw=1.35,
        alpha=0.95,
        label="_nolegend_",
    )


def _finish_axis(axis, name, scaled):
    axis.set_xscale("log")
    if not scaled:
        axis.set_yscale("log")
    axis.set_title(name)
    axis.set_xlabel("n (log scale)")
    axis.set_ylabel(
        r"$\sqrt{n}\times$ CI width" if scaled else "CI width (log scale)"
    )
    axis.grid(True, ls="--", alpha=0.35)


def plot_saved_experiment(input_path: Path) -> list[Path]:
    with input_path.open(encoding="utf-8") as input_file:
        payload = json.load(input_file)

    results = payload["results"]
    n_values = payload["n_values"]
    n_array = np.asarray(n_values, dtype=float)
    delta = payload["delta"]
    topology_counts = payload.get("topology_inversion", {}).get(
        "topology_reps_by_n", payload["num_sims_by_n"]
    )
    simulation_label = _simulation_label(topology_counts)
    output_dir = input_path.parent
    outputs = []

    for scaled in (True, False):
        suffix = "" if scaled else "_raw"
        fig, axes = plt.subplots(3, 3, figsize=(13, 11))
        for axis, (name, model_results) in zip(axes.ravel(), results.items()):
            for target_key, color, label in (
                ("target_product", "#9467bd", "Product limit"),
                ("target_probit", "black", "Gaussian limit"),
            ):
                target = model_results[target_key]
                axis.plot(
                    n_values,
                    (
                        np.full_like(n_array, target)
                        if scaled
                        else target / np.sqrt(n_array)
                    ),
                    color=color,
                    ls=":",
                    lw=1.6,
                    label=label,
                )

            # The main figure reports confidence intervals.  For methods
            # whose raw inversion was audited on a mesh, ``diameter`` is the
            # width of its convex hull; the common-clock Efficient interval
            # is interval-valued before post-processing.  The raw-set versus
            # convex-hull comparison is documented in the appendix.
            methods = (
                ("product_original", "#9467bd", "s", "Product betting"),
                ("product_star", "darkorange", "P", "STaR betting"),
                (
                    "probit_common_clock",
                    "#2ca02c",
                    "h",
                    "Efficient betting (common clock)",
                ),
            )
            for key, color, marker, label in methods:
                deterministic_key = (
                    f"{key}_deterministic_markov_diameter{suffix}"
                )
                randomized_key = (
                    f"{key}_randomized_markov_diameter{suffix}"
                )
                _series(
                    axis,
                    n_values,
                    model_results[deterministic_key],
                    color,
                    None,
                    "_nolegend_",
                    linestyle="--",
                    fill=False,
                )
                _series(
                    axis,
                    n_values,
                    model_results[randomized_key],
                    color,
                    marker,
                    label,
                )
            _finish_axis(axis, name, scaled)

        legend_handles, legend_labels = axes.ravel()[0].get_legend_handles_labels()
        legend_handles.extend([
            Line2D([0], [0], color="0.25", ls="--", lw=2),
            Line2D([0], [0], color="0.25", ls="-", marker="o", lw=2),
        ])
        legend_labels.extend([
            "deterministic Markov",
            "uniformly randomized Markov",
        ])
        scale_label = "scaled" if scaled else "raw"
        fig.suptitle(
            f"Matched Markov calibration: {scale_label} confidence-interval widths "
            f"[\N{GREEK SMALL LETTER DELTA}={delta}, "
            f"sims/n={simulation_label}]",
            fontsize=14,
        )
        fig.legend(
            legend_handles,
            legend_labels,
            loc="lower center",
            ncol=5,
            fontsize=8.0,
            frameon=False,
        )
        fig.tight_layout(rect=(0.0, 0.11, 1.0, 0.95))
        output = output_dir / (
            "ci_width_original_vs_star.png"
            if scaled
            else "ci_width_raw_original_vs_star.png"
        )
        fig.savefig(output, dpi=180, bbox_inches="tight")
        plt.close(fig)
        outputs.append(output)

    # Compare the two fixed-horizon test functions separately in the
    # appendix: Section 4.1's exponential/product construction and
    # Appendix D.1's fixed squared-hinge construction.
    fig, axes = plt.subplots(3, 3, figsize=(13, 11))
    for axis, (name, model_results) in zip(axes.ravel(), results.items()):
        for target_key, color, label in (
            ("target_heat", "navy", "Squared-hinge limit"),
            ("target_product", "#9467bd", "Product limit"),
        ):
            axis.plot(
                n_values,
                np.full_like(n_array, model_results[target_key]),
                color=color,
                ls=":",
                lw=1.6,
                label="_nolegend_",
            )

        for key, color, marker, label in (
            ("heat_original", "navy", "o", "Fixed squared-hinge betting"),
            ("product_original", "#9467bd", "s", "Product betting"),
        ):
            _series(
                axis,
                n_values,
                model_results[f"{key}_deterministic_markov_diameter"],
                color,
                marker,
                "_nolegend_",
                linestyle="--",
                fill=False,
                filled_marker=False,
                marker_size=5.2,
            )
            _series(
                axis,
                n_values,
                model_results[f"{key}_randomized_markov_diameter"],
                color,
                marker,
                "_nolegend_",
                fill=False,
                filled_marker=True,
                marker_size=3.8,
            )
        _finish_axis(axis, name, True)

    construction_handles = [
        Line2D(
            [0], [0], color=color, marker=marker, lw=2,
            markerfacecolor=color, markeredgecolor=color, ms=4.5,
        )
        for color, marker in (("navy", "o"), ("#9467bd", "s"))
    ]
    construction_labels = ["Fixed squared-hinge betting", "Product betting"]
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
        Line2D([0], [0], color="navy", ls=":", lw=1.7),
        Line2D([0], [0], color="#9467bd", ls=":", lw=1.7),
    ]
    reference_labels = ["Squared-hinge limit", "Product limit"]
    fig.suptitle(
        "Fixed squared-hinge versus product betting: scaled confidence-interval widths",
        fontsize=14,
    )
    fig.legend(
        construction_handles,
        construction_labels,
        loc="lower center",
        bbox_to_anchor=(0.25, 0.012),
        ncol=2,
        fontsize=8.4,
        title="Construction (color and marker)",
        title_fontsize=8.6,
        frameon=False,
    )
    fig.legend(
        calibration_handles,
        calibration_labels,
        loc="lower center",
        bbox_to_anchor=(0.64, 0.012),
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
    fig.tight_layout(rect=(0.0, 0.09, 1.0, 0.95))
    output = output_dir / "ci_width_fixed_hinge_vs_product.png"
    PAPER_PLOT_DIR.mkdir(parents=True, exist_ok=True)
    for figure_path in (output, PAPER_PLOT_DIR / output.name):
        fig.savefig(figure_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    outputs.append(output)

    # Keep the replanned squared-hinge comparison out of the main figure and
    # report it separately in the appendix.
    fig, axes = plt.subplots(3, 3, figsize=(13, 11))
    for axis, (name, model_results) in zip(axes.ravel(), results.items()):
        for key, color, marker, label in (
            ("heat_star", "crimson", "D", "Bentkus STaR"),
            ("product_star", "darkorange", "P", "STaR betting"),
        ):
            _series(
                axis,
                n_values,
                model_results[f"{key}_deterministic_markov_diameter"],
                color,
                None,
                "_nolegend_",
                linestyle="--",
                fill=False,
            )
            _series(
                axis,
                n_values,
                model_results[f"{key}_randomized_markov_diameter"],
                color,
                marker,
                label,
            )
        _finish_axis(axis, name, True)

    legend_handles, legend_labels = axes.ravel()[0].get_legend_handles_labels()
    legend_handles.extend([
        Line2D([0], [0], color="0.25", ls="--", lw=2),
        Line2D([0], [0], color="0.25", ls="-", marker="o", lw=2),
    ])
    legend_labels.extend([
        "deterministic Markov",
        "uniformly randomized Markov",
    ])
    fig.suptitle(
        "Bentkus STaR versus STaR betting: scaled confidence-interval widths",
        fontsize=14,
    )
    fig.legend(
        legend_handles,
        legend_labels,
        loc="lower center",
        ncol=4,
        fontsize=9.0,
        frameon=False,
    )
    fig.tight_layout(rect=(0.0, 0.09, 1.0, 0.95))
    output = output_dir / "ci_width_bentkus_star_vs_original_star.png"
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    outputs.append(output)

    fig, axes = plt.subplots(3, 3, figsize=(13, 11))
    for axis, (name, model_results) in zip(axes.ravel(), results.items()):
        gaussian_target = model_results["target_probit"]
        axis.plot(
            n_values,
            np.full_like(n_array, gaussian_target),
            color="black",
            ls=":",
            lw=1.6,
            label="Gaussian limit",
        )
        for key, color, marker, label in (
            ("probit_star", "purple", "v", r"Regularized Efficient betting ($b_n=n^{2/3}$)"),
            ("probit_star_unbuffered", "#8c564b", "^", r"Efficient betting ($b_n=0$)"),
        ):
            _series(axis, n_values, model_results[key], color, marker, label)
        _finish_axis(axis, name, True)

    legend_handles, legend_labels = axes.ravel()[0].get_legend_handles_labels()
    fig.suptitle(
        "Residual-variance regularization: scaled widths "
        f"[\N{GREEK SMALL LETTER DELTA}={delta}, sims/n={simulation_label}]",
        fontsize=14,
    )
    fig.legend(
        legend_handles,
        legend_labels,
        loc="lower center",
        ncol=3,
        fontsize=9.0,
        frameon=False,
    )
    fig.tight_layout(rect=(0.0, 0.09, 1.0, 0.95))
    output = output_dir / "ci_width_probit_regularization.png"
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    outputs.append(output)

    for scaled in (True, False):
        suffix = "" if scaled else "_raw"
        fig, axes = plt.subplots(3, 3, figsize=(13, 11))
        for axis, (name, model_results) in zip(axes.ravel(), results.items()):
            gaussian_target = model_results["target_probit"]
            axis.plot(
                n_values,
                (
                    np.full_like(n_array, gaussian_target)
                    if scaled
                    else gaussian_target / np.sqrt(n_array)
                ),
                color="black",
                ls=":",
                lw=1.6,
                label="Gaussian limit",
            )
            for key, color, marker, label in (
                (f"product_star{suffix}", "darkorange", "P", "square-root/product feedback"),
                (f"hinge_feedback_star{suffix}", "crimson", "D", "squared-hinge feedback"),
                (f"capped_feedback_star{suffix}", "deeppink", "*", "target-capped quadratic feedback"),
                (f"capped_exponential_feedback_star{suffix}", "teal", "X", "capped original feedback"),
                (f"probit_star{suffix}", "purple", "v", "Regularized Efficient betting"),
                (f"probit_star_unbuffered{suffix}", "#8c564b", "^", r"Efficient betting ($b_n=0$)"),
            ):
                _series(axis, n_values, model_results[key], color, marker, label)
            _finish_axis(axis, name, scaled)

        legend_handles, legend_labels = axes.ravel()[0].get_legend_handles_labels()
        scale_label = "scaled" if scaled else "raw"
        fig.suptitle(
            f"Chronological STaR feedback comparison: {scale_label} widths "
            f"[\N{GREEK SMALL LETTER DELTA}={delta}, "
            f"sims/n={simulation_label}]",
            fontsize=14,
        )
        fig.legend(
            legend_handles,
            legend_labels,
            loc="lower center",
            ncol=3,
            fontsize=8.5,
            frameon=False,
        )
        fig.tight_layout(rect=(0.0, 0.11, 1.0, 0.95))
        output = output_dir / (
            "ci_width_feedback_ablation.png"
            if scaled
            else "ci_width_raw_feedback_ablation.png"
        )
        fig.savefig(output, dpi=180, bbox_inches="tight")
        plt.close(fig)
        outputs.append(output)

    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", type=Path, default=DEFAULT_INPUT)
    args = parser.parse_args()
    for output in plot_saved_experiment(args.input.resolve()):
        print(f"Saved to {output}")


if __name__ == "__main__":
    main()
