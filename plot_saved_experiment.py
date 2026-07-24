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
    axis.legend(fontsize=7.2)


def plot_saved_experiment(input_path: Path) -> list[Path]:
    with input_path.open(encoding="utf-8") as input_file:
        payload = json.load(input_file)

    results = payload["results"]
    n_values = payload["n_values"]
    n_array = np.asarray(n_values, dtype=float)
    delta = payload["delta"]
    simulation_label = _simulation_label(payload["num_sims_by_n"])
    output_dir = input_path.parent
    outputs = []

    for scaled in (True, False):
        suffix = "" if scaled else "_raw"
        fig, axes = plt.subplots(2, 3, figsize=(16, 9))
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

            for key, color, marker, label in (
                (f"heat_original{suffix}", "navy", "o", "Bentkus fixed claim"),
                (f"heat_star{suffix}", "crimson", "D", "Bentkus STaR"),
                (f"product_original{suffix}", "seagreen", "s", "WSR product comparator"),
                (f"product_star{suffix}", "darkorange", "P", "product STaR-Bets"),
                (f"capped_feedback_star{suffix}", "deeppink", "*", "target-capped quadratic STaR"),
                (f"capped_exponential_feedback_star{suffix}", "teal", "X", "capped original STaR"),
                (f"probit_star{suffix}", "purple", "v", "Probit STaR (randomized)"),
            ):
                _series(axis, n_values, model_results[key], color, marker, label)
            _finish_axis(axis, name, scaled)

        scale_label = "scaled" if scaled else "raw"
        fig.suptitle(
            f"Fixed-plan versus STaR betting: {scale_label} widths "
            f"[\N{GREEK SMALL LETTER DELTA}={delta}, "
            f"sims/n={simulation_label}]"
        )
        fig.tight_layout()
        output = output_dir / (
            "ci_width_original_vs_star.png"
            if scaled
            else "ci_width_raw_original_vs_star.png"
        )
        fig.savefig(output, dpi=180, bbox_inches="tight")
        plt.close(fig)
        outputs.append(output)

    for scaled in (True, False):
        suffix = "" if scaled else "_raw"
        fig, axes = plt.subplots(2, 3, figsize=(16, 9))
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
                (f"probit_star{suffix}", "purple", "v", "buffered randomized Probit"),
            ):
                _series(axis, n_values, model_results[key], color, marker, label)
            _finish_axis(axis, name, scaled)

        scale_label = "scaled" if scaled else "raw"
        fig.suptitle(
            f"Chronological STaR feedback comparison: {scale_label} widths "
            f"[\N{GREEK SMALL LETTER DELTA}={delta}, "
            f"sims/n={simulation_label}]"
        )
        fig.tight_layout()
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
