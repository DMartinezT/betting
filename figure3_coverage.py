#!/usr/bin/env python3
"""Check finite-sample noncoverage for the methods shown in Figure 3.

The betting samples and terminal uniforms exactly replay the streams used by
``augment_fixed_sample_topology.py``.  Membership is evaluated directly at
the true mean, which avoids any numerical inversion error.  The saved Gaffke
endpoints are read from their dedicated Figure 3 experiment.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

import betting
from gaffke_comparison.compare_star_probit_gaffke import DISTRIBUTIONS


HERE = Path(__file__).resolve().parent
PAPER_PLOTS = HERE.parent / "paper" / "plots"
DEFAULT_INPUT = HERE / "plots" / "ci_width_original_vs_star.json"
DEFAULT_GAFFKE = (
    HERE / "gaffke_comparison" / "figure2_gaffke_results" / "results.csv"
)
DEFAULT_OUTPUT = HERE / "plots" / "figure3_coverage"

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
METHOD_STYLES = {
    "Product betting": ("#9467bd", "s"),
    "STaR betting": ("darkorange", "P"),
    "Efficient betting": ("#2ca02c", "h"),
    "Gaffke": ("#1976b9", "o"),
}
CALIBRATION_STYLES = {
    "Deterministic": ("--", False, 5.2),
    "Uniformly randomized": ("-", True, 3.8),
}
BETTING_C_VALUES = (0.5, 1.0)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--gaffke-results", type=Path, default=DEFAULT_GAFFKE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--progress-every", type=int, default=20)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _rejected(
    plus: float,
    minus: float,
    delta: float,
    u_plus: float,
    u_minus: float,
) -> bool:
    alpha = delta / 2.0
    return max(alpha * plus / u_plus, alpha * minus / u_minus) >= 1.0


def _betting_rows(
    distribution: str,
    n: int,
    replication: int,
    delta: float,
    x: np.ndarray,
    true_mean: float,
    u_plus: float,
    u_minus: float,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    product_plus, product_minus = betting.compute_M_inf_arms(
        x, true_mean, delta
    )
    for calibration, uniforms in (
        ("Deterministic", (1.0, 1.0)),
        ("Uniformly randomized", (u_plus, u_minus)),
    ):
        rows.append({
            "source": "betting",
            "distribution": distribution,
            "n": n,
            "rep": replication,
            "method": "Product betting",
            "calibration": calibration,
            "solvency_c": 0.5,
            "rejected": _rejected(
                product_plus, product_minus, delta, *uniforms
            ),
        })

    for solvency_c in BETTING_C_VALUES:
        star_plus, star_minus = betting.compute_M_star_common_clock_arms(
            x, true_mean, delta, c=solvency_c
        )
        efficient_plus, efficient_minus = (
            betting.compute_M_probit_common_clock_arms(
                x, true_mean, delta, c=solvency_c, buffer_rounds=0.0
            )
        )
        for calibration, uniforms in (
            ("Deterministic", (1.0, 1.0)),
            ("Uniformly randomized", (u_plus, u_minus)),
        ):
            for method, arms in (
                ("STaR betting", (star_plus, star_minus)),
                ("Efficient betting", (efficient_plus, efficient_minus)),
            ):
                rows.append({
                    "source": "betting",
                    "distribution": distribution,
                    "n": n,
                    "rep": replication,
                    "method": method,
                    "calibration": calibration,
                    "solvency_c": solvency_c,
                    "rejected": _rejected(*arms, delta, *uniforms),
                })
    return rows


def _completed_cells(frame: pd.DataFrame) -> set[tuple[str, int, int]]:
    if frame.empty:
        return set()
    counts = frame.groupby(["distribution", "n", "rep"]).size()
    return {
        (str(distribution), int(n), int(replication))
        for (distribution, n, replication), count in counts.items()
        if count == 10
    }


def _run_betting(
    payload: dict[str, object],
    checkpoint: Path,
    resume: bool,
    progress_every: int,
) -> pd.DataFrame:
    delta = float(payload["delta"])
    seed = int(payload["seed"])
    n_values = [int(value) for value in payload["n_values"]]
    base_counts = {
        int(n): int(count)
        for n, count in payload["num_sims_by_n"].items()
    }
    topology_counts = {
        int(n): int(count)
        for n, count in payload["topology_inversion"][
            "topology_reps_by_n"
        ].items()
    }

    if resume and checkpoint.exists():
        frame = pd.read_csv(checkpoint)
        rows = frame.to_dict("records")
        completed = _completed_cells(frame)
        print(f"resuming with {len(completed)} completed betting cells")
    else:
        rows = []
        completed = set()

    rng = np.random.default_rng(seed)
    randomizer_seed = np.random.SeedSequence(seed).spawn(1)[0]
    regularized_rng = np.random.default_rng(randomizer_seed)
    unbuffered_rng = np.random.default_rng(randomizer_seed)
    rng.uniform(0.0, 1.0, 20)

    warm = np.linspace(0.1, 0.9, 64)
    betting.compute_M_inf_arms(warm, 0.5, delta)
    for solvency_c in BETTING_C_VALUES:
        betting.compute_M_star_common_clock_arms(
            warm, 0.5, delta, c=solvency_c
        )
        betting.compute_M_probit_common_clock_arms(
            warm, 0.5, delta, c=solvency_c, buffer_rounds=0.0
        )

    new_cells = 0
    for n in n_values:
        for distribution_index, distribution in enumerate(DISTRIBUTIONS):
            for replication in range(
                max(base_counts[n], topology_counts[n])
            ):
                if replication < base_counts[n]:
                    x = np.asarray(distribution.sampler(rng, n), dtype=float)
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
                    x = np.asarray(
                        distribution.sampler(extension_rng, n), dtype=float
                    )
                    auxiliary_rng = np.random.default_rng(auxiliary_seed)
                    u_plus, u_minus = (
                        float(value)
                        for value in auxiliary_rng.uniform(size=2)
                    )

                if replication >= topology_counts[n]:
                    continue
                key = (distribution.name, n, replication)
                if key in completed:
                    continue
                rows.extend(_betting_rows(
                    distribution.name,
                    n,
                    replication,
                    delta,
                    np.ascontiguousarray(x),
                    float(distribution.mean),
                    u_plus,
                    u_minus,
                ))
                new_cells += 1
                if progress_every > 0 and new_cells % progress_every == 0:
                    checkpoint.parent.mkdir(parents=True, exist_ok=True)
                    pd.DataFrame(rows).to_csv(checkpoint, index=False)
                    print(
                        f"completed {new_cells} new betting cells: "
                        f"{distribution.name}, n={n}, rep={replication + 1}",
                        flush=True,
                    )

    output = pd.DataFrame(rows).drop_duplicates(
        [
            "source",
            "distribution",
            "n",
            "rep",
            "method",
            "calibration",
            "solvency_c",
        ],
        keep="last",
    )
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(checkpoint, index=False)
    return output


def _gaffke_rows(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    means = {distribution.name: distribution.mean for distribution in DISTRIBUTIONS}
    output = frame[["distribution", "n", "rep", "method", "lower", "upper"]].copy()
    output["source"] = "gaffke"
    output["calibration"] = output["method"].map({
        "Gaffke": "Deterministic",
        "Randomized Gaffke": "Uniformly randomized",
    })
    output["method"] = "Gaffke"
    output["solvency_c"] = np.nan
    true_means = output["distribution"].map(means)
    output["rejected"] = ~(
        (output["lower"] <= true_means)
        & (true_means <= output["upper"])
    )
    return output[
        [
            "source",
            "distribution",
            "n",
            "rep",
            "method",
            "calibration",
            "solvency_c",
            "rejected",
        ]
    ]


def _summarize(frame: pd.DataFrame) -> pd.DataFrame:
    summary = (
        frame.groupby(
            [
                "source",
                "distribution",
                "n",
                "method",
                "calibration",
                "solvency_c",
            ],
            dropna=False,
            as_index=False,
        )
        .agg(rejections=("rejected", "sum"), replications=("rejected", "size"))
    )
    count = summary["replications"].to_numpy(dtype=float)
    rate = summary["rejections"].to_numpy(dtype=float) / count
    z = 1.959963984540054
    denominator = 1.0 + z * z / count
    center = (rate + z * z / (2.0 * count)) / denominator
    half = (
        z
        * np.sqrt(rate * (1.0 - rate) / count + z * z / (4.0 * count**2))
        / denominator
    )
    summary["noncoverage"] = rate
    summary["wilson_lower"] = np.maximum(center - half, 0.0)
    summary["wilson_upper"] = np.minimum(center + half, 1.0)
    return summary


def _main_c05(summary: pd.DataFrame) -> pd.DataFrame:
    betting_mask = (summary["source"] == "betting") & (
        np.isclose(summary["solvency_c"], 0.5)
    )
    return summary[betting_mask | (summary["source"] == "gaffke")].copy()


def _pooled_by_n(summary: pd.DataFrame) -> pd.DataFrame:
    pooled = (
        _main_c05(summary)
        .groupby(["n", "method", "calibration"], as_index=False)
        .agg(
            rejections=("rejections", "sum"),
            replications=("replications", "sum"),
        )
    )
    pooled["noncoverage"] = (
        pooled["rejections"] / pooled["replications"]
    )
    return pooled


def _legend(fig: plt.Figure, reference_title: str) -> None:
    method_handles = [
        Line2D(
            [0], [0], color=color, marker=marker, lw=2,
            markerfacecolor=color, markeredgecolor=color, ms=4.5,
        )
        for color, marker in METHOD_STYLES.values()
    ]
    calibration_handles = [
        Line2D(
            [0], [0], color="0.25", ls=linestyle, marker="o", lw=2,
            markerfacecolor="0.25" if filled else "none",
            markeredgecolor="0.25", ms=marker_size,
        )
        for linestyle, filled, marker_size in CALIBRATION_STYLES.values()
    ]
    fig.legend(
        method_handles,
        list(METHOD_STYLES),
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
        list(CALIBRATION_STYLES),
        loc="lower center",
        bbox_to_anchor=(0.68, 0.012),
        ncol=2,
        fontsize=8.4,
        title="Calibration (line and marker fill)",
        title_fontsize=8.6,
        frameon=False,
    )
    fig.legend(
        [Line2D([0], [0], color="black", ls=":", lw=1.7)],
        [r"Nominal $\delta$"],
        loc="lower center",
        bbox_to_anchor=(0.91, 0.012),
        fontsize=8.4,
        title=reference_title,
        title_fontsize=8.6,
        frameon=False,
    )


def _plot_panels(summary: pd.DataFrame, delta: float, destination: Path) -> None:
    selected = _main_c05(summary)
    maximum = max(float(selected["noncoverage"].max()), delta)
    y_max = max(0.05, 1.18 * maximum)
    fig, axes = plt.subplots(3, 3, figsize=(13.5, 11.3), sharey=True)
    for panel_index, (axis, distribution) in enumerate(
        zip(axes.ravel(), PANEL_ORDER)
    ):
        minimum_n = ROW_MINIMUM_N[panel_index // 3]
        methods = list(METHOD_STYLES)
        if panel_index >= 3:
            methods.remove("Product betting")
        for method in methods:
            color, marker = METHOD_STYLES[method]
            for calibration, (
                linestyle, filled, marker_size
            ) in CALIBRATION_STYLES.items():
                rows = selected[
                    (selected["distribution"] == distribution)
                    & (selected["method"] == method)
                    & (selected["calibration"] == calibration)
                    & (selected["n"] >= minimum_n)
                    & (selected["n"] <= 1_000_000)
                ].sort_values("n")
                axis.plot(
                    rows["n"],
                    rows["noncoverage"],
                    color=color,
                    marker=marker,
                    markerfacecolor=color if filled else "none",
                    markeredgecolor=color,
                    markeredgewidth=0.9,
                    ms=marker_size,
                    lw=1.8,
                    ls=linestyle,
                )
        axis.axhline(delta, color="black", ls=":", lw=1.5)
        axis.set_xscale("log")
        axis.set_xlim(minimum_n, 1_000_000)
        axis.set_ylim(0.0, y_max)
        axis.set_title(distribution)
        axis.set_xlabel("sample size $n$")
        axis.set_ylabel("noncoverage proportion")
        axis.grid(True, ls="--", alpha=0.3)

    fig.suptitle(
        "Finite-sample noncoverage of the Figure 3 methods "
        r"(STaR and efficient betting: $c=0.5$)",
        fontsize=15,
    )
    _legend(fig, "Validity reference (dotted)")
    fig.tight_layout(rect=(0.0, 0.09, 1.0, 0.955))
    destination.parent.mkdir(parents=True, exist_ok=True)
    PAPER_PLOTS.mkdir(parents=True, exist_ok=True)
    for path in (destination, PAPER_PLOTS / destination.name):
        fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_pooled(summary: pd.DataFrame, delta: float, destination: Path) -> None:
    pooled = _pooled_by_n(summary)
    maximum = max(float(pooled["noncoverage"].max()), delta)
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.0), sharey=True)
    for axis, calibration in zip(axes, CALIBRATION_STYLES):
        for method, (color, marker) in METHOD_STYLES.items():
            rows = pooled[
                (pooled["method"] == method)
                & (pooled["calibration"] == calibration)
            ].sort_values("n")
            axis.plot(
                rows["n"],
                rows["noncoverage"],
                color=color,
                marker=marker,
                ms=4.4,
                lw=1.9,
                label=method,
            )
        axis.axhline(delta, color="black", ls=":", lw=1.5)
        axis.set_xscale("log")
        axis.set_ylim(0.0, max(0.03, 1.18 * maximum))
        axis.set_title(calibration)
        axis.set_xlabel("sample size $n$")
        axis.set_ylabel("pooled noncoverage proportion")
        axis.grid(True, ls="--", alpha=0.3)
    axes[0].legend(frameon=False, fontsize=8.8)
    fig.suptitle(
        "Noncoverage pooled across bounded distributions "
        r"(STaR and efficient betting: $c=0.5$)",
        fontsize=13.5,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
    destination.parent.mkdir(parents=True, exist_ok=True)
    PAPER_PLOTS.mkdir(parents=True, exist_ok=True)
    for path in (destination, PAPER_PLOTS / destination.name):
        fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    with args.input.open(encoding="utf-8") as stream:
        payload = json.load(stream)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output_dir / "betting_checkpoint.csv"
    betting_rows = _run_betting(
        payload, checkpoint, args.resume, args.progress_every
    )
    frame = pd.concat(
        [betting_rows, _gaffke_rows(args.gaffke_results)],
        ignore_index=True,
    )
    frame.to_csv(args.output_dir / "results.csv", index=False)
    summary = _summarize(frame)
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    _pooled_by_n(summary).to_csv(
        args.output_dir / "pooled_summary.csv", index=False
    )
    delta = float(payload["delta"])
    _plot_panels(
        summary, delta, args.output_dir / "figure3_noncoverage.png"
    )
    _plot_pooled(
        summary, delta, args.output_dir / "pooled_noncoverage.png"
    )
    with (args.output_dir / "config.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(
            {
                "delta": delta,
                "betting_seed": int(payload["seed"]),
                "betting_reps_by_n": payload["topology_inversion"][
                    "topology_reps_by_n"
                ],
                "solvency_c_values": list(BETTING_C_VALUES),
                "main_figure_solvency_c": 0.5,
                "membership_check": (
                    "terminal rejection evaluated at the true mean"
                ),
                "gaffke_results": str(args.gaffke_results),
            },
            stream,
            indent=2,
        )
    print(f"saved coverage results and plots to {args.output_dir}")


if __name__ == "__main__":
    main()
