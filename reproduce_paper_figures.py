#!/usr/bin/env python3
"""Regenerate every figure used by the companion paper.

The default command redraws the figures from committed summaries and
replication-level results.  It does not rerun any Monte Carlo experiment.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import pandas as pd


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = ROOT / "paper_figures"


@dataclass(frozen=True)
class FigureSpec:
    """Metadata for one manuscript figure asset."""

    filename: str
    group: str
    location: str
    description: str


FIGURES = (
    FigureSpec(
        "intro_both_comparison.png",
        "iid",
        "Introduction",
        "Three-distribution deterministic/randomized iid comparison.",
    ),
    FigureSpec(
        "wor_fixed_fraction_intro_comparison.png",
        "without_replacement",
        "Introduction",
        "Three-distribution fixed-fraction comparison without replacement.",
    ),
    FigureSpec(
        "efficient_betting_function.pdf",
        "betting_function",
        "GE-betting construction",
        "Inverse-Mills betting fraction and amount bet.",
    ),
    FigureSpec(
        "ci_width_original_vs_star.png",
        "iid",
        "Fixed-horizon iid experiments",
        "Nine-distribution fixed-horizon iid comparison.",
    ),
    FigureSpec(
        "wor_fixed_horizon_widths.png",
        "without_replacement",
        "Sampling without replacement",
        "Fixed population size with varying sampling fraction.",
    ),
    FigureSpec(
        "wor_fixed_fraction_sqrt_n_widths.png",
        "without_replacement",
        "Sampling without replacement",
        "Fixed sampling fraction with varying population size.",
    ),
    FigureSpec(
        "scaled_width_star_efficient_shared_estimator_comparison.png",
        "large_sample",
        "Appendix: estimator comparison",
        "Candidate-specific and shared-estimator large-sample comparison.",
    ),
    FigureSpec(
        "ci_width_solvency_combined.png",
        "solvency",
        "Appendix: solvency sensitivity",
        "Sensitivity to the solvency fraction.",
    ),
    FigureSpec(
        "ci_width_fixed_hinge_vs_product.png",
        "fixed_hinge",
        "Appendix: betting constructions",
        "Fixed squared-hinge and product-betting comparison.",
    ),
    FigureSpec(
        "horizon_free_wr_cs_widths.png",
        "horizon_free_wr",
        "Appendix: confidence sequences",
        "Planned-window confidence sequences with replacement.",
    ),
    FigureSpec(
        "horizon_free_cs_widths.png",
        "horizon_free_wor",
        "Appendix: confidence sequences",
        "Horizon-free confidence sequences without replacement.",
    ),
)

FIGURE_BY_NAME = {figure.filename: figure for figure in FIGURES}

GROUP_INPUTS = {
    "iid": (
        ROOT / "plots/solvency_c_comparison/ci_width_all_methods_c1.json",
        ROOT / "gaffke_comparison/figure2_gaffke_results/results.csv",
    ),
    "betting_function": (),
    "without_replacement": (
        ROOT / "plots/wor/fixed_horizon_widths.csv",
        ROOT / "plots/wor/fixed_fraction_sqrt_n_widths.csv",
    ),
    "large_sample": (
        ROOT / "gaffke_comparison/large_sample_gaffke_results_c1/results.csv",
        ROOT / "gaffke_comparison/large_sample_gaffke_results_c1/config.json",
    ),
    "solvency": (
        ROOT / "plots/ci_width_original_vs_star_c05.json",
        ROOT / "plots/ci_width_original_vs_star_c075.json",
        ROOT / "plots/solvency_c_comparison/ci_width_all_methods_c1.json",
    ),
    "fixed_hinge": (
        ROOT / "plots/solvency_c_comparison/ci_width_all_methods_c1.json",
    ),
    "horizon_free_wr": (
        ROOT / "plots/horizon_free_wr_cs/summary.csv",
    ),
    "horizon_free_wor": (
        ROOT / "plots/horizon_free_cs/summary.csv",
    ),
}


def _generate_iid(output_dir: Path) -> dict[str, Path]:
    import plot_combined_main_figure as figures

    intro = output_dir / "intro_both_comparison.png"
    main = output_dir / "ci_width_original_vs_star.png"
    figures.make_intro_both_figure((intro,))
    figures.make_figure((main,))
    return {intro.name: intro, main.name: main}


def _generate_betting_function(output_dir: Path) -> dict[str, Path]:
    import plot_efficient_betting_function as figure

    output = output_dir / "efficient_betting_function.pdf"
    figure.make_figure((output,))
    return {output.name: output}


def _generate_without_replacement(output_dir: Path) -> dict[str, Path]:
    import wor
    import wor_scaled_plots as scaled

    fixed_rows = wor.load_summary(GROUP_INPUTS["without_replacement"][0])
    temporary = wor.fixed_horizon_plot(output_dir, fixed_rows)
    fixed = output_dir / "wor_fixed_horizon_widths.png"
    shutil.copy2(temporary, fixed)

    fraction_rows = scaled.load_scaled_summary(
        GROUP_INPUTS["without_replacement"][1]
    )
    fraction = scaled.scaled_width_plot(
        summary_rows=fraction_rows,
        output_path=output_dir / "wor_fixed_fraction_sqrt_n_widths.png",
        varying_population_size=True,
    )
    intro = scaled.fixed_fraction_mini_plot(
        summary_rows=fraction_rows,
        output_path=output_dir / "wor_fixed_fraction_intro_comparison.png",
    )
    return {
        fixed.name: fixed,
        fraction.name: fraction,
        intro.name: intro,
    }


def _generate_large_sample(output_dir: Path) -> dict[str, Path]:
    from gaffke_comparison import large_sample_feedback_gaffke as experiment

    results_path, config_path = GROUP_INPUTS["large_sample"]
    frame = pd.read_csv(results_path)
    with config_path.open(encoding="utf-8") as stream:
        delta = float(json.load(stream)["delta"])
    work_dir = output_dir / "large_sample_work"
    outputs = experiment.make_plots(
        frame,
        work_dir,
        delta,
        copy_to_paper=False,
    )
    source = next(
        path
        for path in outputs
        if path.name
        == "scaled_width_star_efficient_shared_estimator_comparison.png"
    )
    destination = output_dir / source.name
    shutil.copy2(source, destination)
    return {destination.name: destination}


def _generate_solvency(output_dir: Path) -> dict[str, Path]:
    import plot_solvency_sensitivity as figure

    output = output_dir / "ci_width_solvency_combined.png"
    figure.make_figure((output,))
    return {output.name: output}


def _generate_fixed_hinge(output_dir: Path) -> dict[str, Path]:
    import plot_saved_experiment as figures

    work_dir = output_dir / "fixed_hinge_work"
    outputs = figures.plot_saved_experiment(
        GROUP_INPUTS["fixed_hinge"][0],
        output_dir=work_dir,
        paper_plot_dir=None,
    )
    source = next(
        path
        for path in outputs
        if path.name == "ci_width_fixed_hinge_vs_product.png"
    )
    destination = output_dir / source.name
    shutil.copy2(source, destination)
    return {destination.name: destination}


def _generate_horizon_free_wr(output_dir: Path) -> dict[str, Path]:
    import horizon_free_wr_cs as experiment

    rows = experiment.load_summary(GROUP_INPUTS["horizon_free_wr"][0])
    output = experiment.make_plot(output_dir, rows)
    return {output.name: output}


def _generate_horizon_free_wor(output_dir: Path) -> dict[str, Path]:
    import horizon_free_cs as experiment

    rows = experiment.load_summary(GROUP_INPUTS["horizon_free_wor"][0])
    output = experiment.make_plot(output_dir, rows)
    return {output.name: output}


GENERATORS: dict[str, Callable[[Path], dict[str, Path]]] = {
    "iid": _generate_iid,
    "betting_function": _generate_betting_function,
    "without_replacement": _generate_without_replacement,
    "large_sample": _generate_large_sample,
    "solvency": _generate_solvency,
    "fixed_hinge": _generate_fixed_hinge,
    "horizon_free_wr": _generate_horizon_free_wr,
    "horizon_free_wor": _generate_horizon_free_wor,
}


def validate_inputs(groups: Sequence[str]) -> None:
    """Raise a useful error if a committed plotting input is absent."""
    missing = sorted(
        path
        for group in groups
        for path in GROUP_INPUTS[group]
        if not path.is_file()
    )
    if missing:
        formatted = "\n".join(f"  - {path.relative_to(ROOT)}" for path in missing)
        raise FileNotFoundError(f"Missing cached figure inputs:\n{formatted}")


def reproduce(
    output_dir: Path,
    filenames: Sequence[str] | None = None,
) -> list[Path]:
    """Regenerate selected manuscript figures from committed results."""
    requested = (
        set(FIGURE_BY_NAME)
        if filenames is None
        else set(filenames)
    )
    unknown = sorted(requested.difference(FIGURE_BY_NAME))
    if unknown:
        raise ValueError("Unknown figure name(s): " + ", ".join(unknown))

    groups = tuple(
        dict.fromkeys(
            figure.group for figure in FIGURES if figure.filename in requested
        )
    )
    validate_inputs(groups)
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    generated: dict[str, Path] = {}
    with tempfile.TemporaryDirectory(prefix="ge-betting-figures-") as temporary:
        work_dir = Path(temporary)
        for group in groups:
            started = time.perf_counter()
            print(f"[{group}] generating...", flush=True)
            generated.update(GENERATORS[group](work_dir))
            elapsed = time.perf_counter() - started
            print(f"[{group}] done in {elapsed:.1f}s", flush=True)

        outputs: list[Path] = []
        for figure in FIGURES:
            if figure.filename not in requested:
                continue
            source = generated.get(figure.filename)
            if source is None or not source.is_file() or source.stat().st_size == 0:
                raise RuntimeError(f"Generator did not create {figure.filename}")
            destination = output_dir / figure.filename
            shutil.copy2(source, destination)
            outputs.append(destination)
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Destination directory (default: %(default)s).",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        metavar="FILENAME",
        help="Regenerate only the named manuscript figure files.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List manuscript figures and exit.",
    )
    parser.add_argument(
        "--check-inputs",
        action="store_true",
        help="Verify all cached plotting inputs and exit.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.list:
        for figure in FIGURES:
            print(f"{figure.filename:62s} {figure.location}")
        return 0
    if args.check_inputs:
        validate_inputs(tuple(GENERATORS))
        print(f"All cached inputs for {len(FIGURES)} manuscript figures are present.")
        return 0

    outputs = reproduce(args.output_dir, args.only)
    print(f"\nWrote {len(outputs)} figure(s) to {args.output_dir.resolve()}:")
    for output in outputs:
        print(f"  {output.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
