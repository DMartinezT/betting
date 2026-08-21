"""Plot the GE-betting fraction and the corresponding amount bet."""

from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm


HERE = Path(__file__).resolve().parents[1]
PAPER_OUTPUT = (
    HERE.parent / "paper" / "plots" / "efficient_betting_function.pdf"
)
CODE_OUTPUT = HERE / "plots" / "efficient_betting_function.pdf"


def make_figure(output_paths: Sequence[Path] | None = None) -> Path:
    """Write the figure and return its first output path."""
    p = np.linspace(1e-4, 1.0 - 1e-4, 4_000)
    q = norm.ppf(p)
    amount_bet = norm.pdf(q)
    betting_fraction = amount_bet / p

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "legend.fontsize": 9,
        }
    )
    figure, axes = plt.subplots(1, 2, figsize=(8.3, 3.0))

    axes[0].plot(p, betting_fraction, color="#0072B2", linewidth=2.1)
    axes[0].set_title("Fraction of current wealth")
    axes[0].set_ylabel(r"$\psi(p)=\phi\{\Phi^{-1}(p)\}/p$")

    axes[1].plot(p, amount_bet, color="#D55E00", linewidth=2.1)
    axes[1].set_title("Amount bet relative to target")
    axes[1].set_ylabel(r"$p\psi(p)=\phi\{\Phi^{-1}(p)\}$")

    for axis in axes:
        axis.set_xlabel(r"Current wealth divided by target, $p$")
        axis.set_xlim(0.0, 1.0)
        axis.set_ylim(bottom=0.0)
        axis.grid(alpha=0.22, linewidth=0.6)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    figure.tight_layout(w_pad=2.2)
    destinations = tuple(
        (CODE_OUTPUT, PAPER_OUTPUT)
        if output_paths is None
        else output_paths
    )
    if not destinations:
        raise ValueError("output_paths must contain at least one path")
    for output in destinations:
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, bbox_inches="tight")
    plt.close(figure)
    return destinations[0]


def main() -> None:
    """Write the figure to the code and paper repositories."""
    print(make_figure())


if __name__ == "__main__":
    main()
