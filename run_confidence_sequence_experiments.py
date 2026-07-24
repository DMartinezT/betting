#!/usr/bin/env python3
"""Run and plot the five-method confidence-sequence benchmark.

The numerical work is delegated to
``confidence_sequences.run_confidence_sequence_experiment``.  This driver
only supplies a reproducible command-line interface, saves its complete
JSON return value, and creates the publication figures.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

if __package__:
    from .confidence_sequences import run_confidence_sequence_experiment
else:
    from confidence_sequences import run_confidence_sequence_experiment


HERE = Path(__file__).resolve().parent
OUTPUT_DIRECTORY = HERE / "plots"
DEFAULT_OUTPUT_PREFIX = "confidence_sequences"

METHOD_STYLE = {
    "hgkelly": {
        "label": "Finite grid-Kelly",
        "color": "#999933",
        "linestyle": ":",
    },
    "product_scale_mixture": {
        "label": "Product scale mixture",
        "color": "#4477AA",
        "linestyle": "--",
    },
    "agrapa": {
        "label": "Product aGRAPA",
        "color": "#228833",
        "linestyle": "-",
    },
    "bentkus_mixture": {
        "label": "Bentkus maturity mixture",
        "color": "#CC6677",
        "linestyle": "-",
    },
    "heat_constrained_agrapa": {
        "label": "Heat-constrained aGRAPA",
        "color": "#AA3377",
        "linestyle": "-.",
    },
}


def _positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _nonnegative_integer(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be a nonnegative integer")
    return parsed


def _probability(value: str) -> float:
    parsed = float(value)
    if not 0.0 < parsed < 1.0:
        raise argparse.ArgumentTypeError("must lie strictly between zero and one")
    return parsed


def _topology_grid_size(value: str) -> int:
    parsed = int(value)
    if parsed < 3:
        raise argparse.ArgumentTypeError("must be an integer at least three")
    return parsed


def _checkpoint_times(value: str) -> list[int]:
    try:
        parsed = [int(item.strip()) for item in value.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "must be a comma-separated list of integers"
        ) from error
    if not parsed or any(time <= 0 for time in parsed):
        raise argparse.ArgumentTypeError("all checkpoint times must be positive")
    if any(right <= left for left, right in zip(parsed, parsed[1:])):
        raise argparse.ArgumentTypeError(
            "checkpoint times must be strictly increasing"
        )
    return parsed


def _output_prefix(value: str) -> str:
    if not value or value in {".", ".."}:
        raise argparse.ArgumentTypeError("must be a nonempty file prefix")
    if Path(value).name != value or "/" in value or "\\" in value:
        raise argparse.ArgumentTypeError(
            "must be a file prefix, not a path"
        )
    return value


def _ordered_distributions(payload: dict) -> list[str]:
    return list(payload["true_means"])


def _configure_axis(axis: plt.Axes, title: str) -> None:
    axis.set_title(title, fontsize=11)
    axis.set_xscale("log")
    axis.grid(True, which="major", color="0.85", linewidth=0.7)
    axis.grid(True, which="minor", color="0.92", linewidth=0.45)
    axis.tick_params(labelsize=9)


def _save_figure(figure: plt.Figure, path: Path) -> None:
    figure.savefig(
        path,
        dpi=200,
        bbox_inches="tight",
        metadata={"Software": "run_confidence_sequence_experiments.py"},
    )
    plt.close(figure)


def plot_absolute_widths(payload: dict, output_path: Path) -> None:
    """Plot mean width and its marginal 10--90 percent simulation band."""
    times = np.asarray(payload["times"], dtype=float)
    figure, axes = plt.subplots(2, 3, figsize=(13.5, 7.6), sharex=True)

    for axis, distribution in zip(
        axes.ravel(), _ordered_distributions(payload)
    ):
        distribution_results = payload["results"][distribution]
        for method, style in METHOD_STYLE.items():
            rows = distribution_results[method]["width"]
            mean = np.asarray([row["mean"] for row in rows], dtype=float)
            lower = np.asarray([row["lo"] for row in rows], dtype=float)
            upper = np.asarray([row["hi"] for row in rows], dtype=float)
            axis.plot(
                times,
                mean,
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=1.9,
                label=style["label"],
            )
            axis.fill_between(
                times,
                lower,
                upper,
                color=style["color"],
                alpha=0.09,
                linewidth=0,
            )
        axis.set_yscale("log")
        _configure_axis(axis, distribution)

    figure.suptitle(
        "Mean confidence-sequence width (shaded: 10th–90th percentiles)",
        fontsize=13,
    )
    figure.supxlabel("Time $t$ (log scale)", fontsize=11)
    figure.supylabel("Width (log scale)", fontsize=11)
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        ncol=5,
        frameon=False,
        fontsize=9.5,
        bbox_to_anchor=(0.5, -0.005),
    )
    figure.tight_layout(rect=(0.025, 0.065, 1.0, 0.96))
    _save_figure(figure, output_path)


def plot_relative_widths(payload: dict, output_path: Path) -> None:
    """Plot each marginal mean width relative to product aGRAPA."""
    times = np.asarray(payload["times"], dtype=float)
    figure, axes = plt.subplots(2, 3, figsize=(13.5, 7.6), sharex=True)

    for axis, distribution in zip(
        axes.ravel(), _ordered_distributions(payload)
    ):
        distribution_results = payload["results"][distribution]
        reference = np.asarray(
            [
                row["mean"]
                for row in distribution_results["agrapa"]["width"]
            ],
            dtype=float,
        )
        axis.axhline(
            1.0, color="0.25", linestyle=":", linewidth=1.0, zorder=0
        )
        for method, style in METHOD_STYLE.items():
            mean = np.asarray(
                [
                    row["mean"]
                    for row in distribution_results[method]["width"]
                ],
                dtype=float,
            )
            ratio = np.divide(
                mean,
                reference,
                out=np.full_like(mean, np.nan),
                where=reference > 0.0,
            )
            axis.plot(
                times,
                ratio,
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=1.9,
                label=style["label"],
            )
        axis.set_yscale("log")
        _configure_axis(axis, distribution)

    figure.suptitle("Mean width relative to product aGRAPA", fontsize=13)
    figure.supxlabel("Time $t$ (log scale)", fontsize=11)
    figure.supylabel("Mean-width ratio (log scale)", fontsize=11)
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        ncol=5,
        frameon=False,
        fontsize=9.5,
        bbox_to_anchor=(0.5, -0.005),
    )
    figure.tight_layout(rect=(0.025, 0.065, 1.0, 0.96))
    _save_figure(figure, output_path)


def plot_crossing_summary(payload: dict, output_path: Path) -> None:
    """Plot true-mean crossing rates and Wilson confidence intervals."""
    distributions = _ordered_distributions(payload)
    methods = list(METHOD_STYLE)
    x_positions = np.arange(len(distributions), dtype=float)
    group_width = 0.82
    bar_width = group_width / len(methods)
    offsets = (
        np.arange(len(methods), dtype=float) - (len(methods) - 1.0) / 2.0
    ) * bar_width

    figure, axis = plt.subplots(figsize=(12.8, 5.3))
    upper_extent = float(payload["delta"])
    for offset, method in zip(offsets, methods):
        coverage_rows = [
            payload["results"][distribution][method]["coverage"]
            for distribution in distributions
        ]
        rates = np.asarray(
            [row["crossing_rate"] for row in coverage_rows], dtype=float
        )
        lowers = np.asarray(
            [row["wilson_lower"] for row in coverage_rows], dtype=float
        )
        uppers = np.asarray(
            [row["wilson_upper"] for row in coverage_rows], dtype=float
        )
        upper_extent = max(upper_extent, float(np.nanmax(uppers)))
        style = METHOD_STYLE[method]
        axis.bar(
            x_positions + offset,
            rates,
            width=0.92 * bar_width,
            color=style["color"],
            alpha=0.86,
            label=style["label"],
            yerr=np.maximum(
                np.vstack((rates - lowers, uppers - rates)), 0.0
            ),
            capsize=2.5,
            error_kw={"elinewidth": 0.9, "capthick": 0.9},
        )

    axis.axhline(
        payload["delta"],
        color="black",
        linestyle=":",
        linewidth=1.4,
        label=r"Nominal $\delta$",
    )
    axis.set_xticks(x_positions, distributions, rotation=16, ha="right")
    axis.set_ylabel("True-mean crossing probability")
    axis.set_ylim(0.0, max(0.04, 1.12 * upper_extent))
    axis.set_title(
        "Empirical anytime error (coverage equals one minus crossing rate)"
    )
    axis.grid(True, axis="y", color="0.88", linewidth=0.7)
    axis.set_axisbelow(True)
    axis.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.20),
        ncol=5,
        frameon=False,
        fontsize=9.2,
    )
    figure.tight_layout(rect=(0.0, 0.09, 1.0, 1.0))
    _save_figure(figure, output_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--delta", type=_probability, default=0.01)
    parser.add_argument("--max-time", type=_positive_integer, default=100_000)
    parser.add_argument(
        "--num-width-sims", type=_positive_integer, default=50
    )
    parser.add_argument(
        "--coverage-max-time", type=_positive_integer, default=10_000
    )
    parser.add_argument(
        "--num-coverage-sims", type=_nonnegative_integer, default=0
    )
    parser.add_argument(
        "--topology-grid-size", type=_topology_grid_size, default=129
    )
    parser.add_argument(
        "--times",
        type=_checkpoint_times,
        default=None,
        help=(
            "optional comma-separated reporting times; useful for sparse "
            "high-horizon audits"
        ),
    )
    parser.add_argument(
        "--product-grid-size", type=_positive_integer, default=20
    )
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument(
        "--output-prefix", type=_output_prefix, default=DEFAULT_OUTPUT_PREFIX
    )
    parser.add_argument(
        "--progress",
        action="store_true",
        help="print progress as each distribution is processed",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    payload = run_confidence_sequence_experiment(
        delta=args.delta,
        max_time=args.max_time,
        times=args.times,
        num_width_sims=args.num_width_sims,
        coverage_max_time=args.coverage_max_time,
        num_coverage_sims=args.num_coverage_sims,
        seed=args.seed,
        product_grid_size=args.product_grid_size,
        topology_grid_size=args.topology_grid_size,
        progress=args.progress,
    )

    json_path = OUTPUT_DIRECTORY / f"{args.output_prefix}.json"
    with json_path.open("w", encoding="utf-8") as output_file:
        json.dump(
            payload,
            output_file,
            indent=2,
            sort_keys=False,
            allow_nan=False,
        )
        output_file.write("\n")

    output_paths = [json_path]
    absolute_path = OUTPUT_DIRECTORY / f"{args.output_prefix}_widths.png"
    plot_absolute_widths(payload, absolute_path)
    output_paths.append(absolute_path)

    relative_path = OUTPUT_DIRECTORY / f"{args.output_prefix}_relative_widths.png"
    plot_relative_widths(payload, relative_path)
    output_paths.append(relative_path)

    if args.num_coverage_sims:
        coverage_path = OUTPUT_DIRECTORY / f"{args.output_prefix}_coverage.png"
        plot_crossing_summary(payload, coverage_path)
        output_paths.append(coverage_path)

    for path in output_paths:
        print(f"Saved to {path}")


if __name__ == "__main__":
    main()
