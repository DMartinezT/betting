"""Regenerate the main STaR figures from a saved experiment JSON file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
DEFAULT_INPUT = HERE / "plots" / "ci_width_original_vs_star.json"


def _simulation_label(counts_by_n: dict[str, int]) -> str:
    counts = list(counts_by_n.values())
    if min(counts) == max(counts):
        return str(counts[0])
    return f"{min(counts)}\N{EN DASH}{max(counts)}"


def _series(axis, n_values, rows, color, marker, label):
    means = [row["mean"] for row in rows]
    lows = [row["lo"] for row in rows]
    highs = [row["hi"] for row in rows]
    axis.plot(
        n_values,
        means,
        color=color,
        marker=marker,
        ms=4.5,
        lw=2,
        label=label,
    )
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
                ("target_heat", "navy", "Bentkus theory"),
                ("target_product", "seagreen", "product theory"),
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
            # convex-hull comparison is kept in the appendix figure generated
            # by the large-sample script.
            for key, color, marker, label in (
                ("heat_original", "navy", "o", "Bentkus fixed claim"),
                ("heat_star", "crimson", "D", "Bentkus STaR"),
                ("product_original", "seagreen", "s", "WSR product comparator"),
                ("product_star", "darkorange", "P", "product STaR-Bets"),
                (
                    "probit_common_clock",
                    "#2ca02c",
                    "h",
                    "Efficient betting (common clock)",
                ),
            ):
                diameter_key = f"{key}_diameter{suffix}"
                _series(
                    axis,
                    n_values,
                    model_results[diameter_key],
                    color,
                    marker,
                    label,
                )
            _finish_axis(axis, name, scaled)

        legend_handles, legend_labels = axes.ravel()[0].get_legend_handles_labels()
        scale_label = "scaled" if scaled else "raw"
        fig.suptitle(
            f"Fixed-plan versus STaR betting: {scale_label} confidence-interval widths "
            f"[\N{GREEK SMALL LETTER DELTA}={delta}, "
            f"sims/n={simulation_label}]",
            fontsize=14,
        )
        fig.legend(
            legend_handles,
            legend_labels,
            loc="lower center",
            ncol=5,
            fontsize=8.5,
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
