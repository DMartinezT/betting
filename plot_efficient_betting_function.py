"""Plot the efficient-betting fraction and its target-normalized exposure."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm


HERE = Path(__file__).resolve().parent
PAPER_OUTPUT = (
    HERE.parent / "paper" / "plots" / "efficient_betting_function.pdf"
)
CODE_OUTPUT = HERE / "plots" / "efficient_betting_function.pdf"


def main() -> None:
    """Write the figure to the code and paper repositories."""
    p = np.linspace(1e-4, 1.0 - 1e-4, 4_000)
    normal_score = norm.ppf(p)
    exposure = norm.pdf(normal_score)
    betting_fraction = exposure / p

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

    axes[1].plot(p, exposure, color="#D55E00", linewidth=2.1)
    axes[1].set_title("Target-normalized exposure")
    axes[1].set_ylabel(r"$p\psi(p)=\phi\{\Phi^{-1}(p)\}$")

    for axis in axes:
        axis.set_xlabel(r"Current wealth divided by target, $p$")
        axis.set_xlim(0.0, 1.0)
        axis.set_ylim(bottom=0.0)
        axis.grid(alpha=0.22, linewidth=0.6)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    figure.tight_layout(w_pad=2.2)
    for output in (CODE_OUTPUT, PAPER_OUTPUT):
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
