"""Figure generation for the report.

This is the ONLY module in the package permitted to import matplotlib. The
project description allows visualisation libraries for visualisation only, and
tests/test_spec_compliance.py enforces that rule mechanically.

Figures are written to reports/figures/ by EDA.py, so anyone can regenerate them.
"""

import matplotlib.pyplot as plt
import numpy as np

from mlproject.config import FIGURES_DIR


def save_figure(fig: plt.Figure, name: str) -> None:
    """Save a figure to reports/figures/<name>.png."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    output_path = FIGURES_DIR / f"{name}.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Figure saved to: {output_path}")


def _missing_ratio(x: np.ndarray, feature_names: list[str]) -> np.ndarray:
    """Return the fraction of missing values per feature."""
    if x.shape[1] != len(feature_names):
        raise ValueError(
            "The number of feature names must match the number of columns."
        )

    return np.mean(np.isnan(x), axis=0)


def plot_missingness(x: np.ndarray, feature_names: list[str]) -> plt.Figure:
    """Plot the distribution of missing-value fractions across features."""
    missing_ratio = _missing_ratio(x, feature_names)

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.hist(
        missing_ratio,
        bins=30,
        range=(0, 1),
        edgecolor="black",
    )

    ax.set_title("Missing-value distribution across features")
    ax.set_xlabel("Fraction of missing values")
    ax.set_ylabel("Number of features")
    ax.set_xlim(0, 1)

    fig.tight_layout()
    return fig


def plot_missingness_with_cutoff(
    x: np.ndarray, feature_names: list[str], threshold: float = 0.80
) -> plt.Figure:
    """Plot the missing-value distribution with the preprocessing cutoff.

    A dashed line marks the missingness cutoff used by the preprocessing;
    features to its right are dropped. Each bar is labelled with its count.
    """
    missing_ratio = _missing_ratio(x, feature_names)
    n_dropped = int(np.sum(missing_ratio > threshold))

    fig, ax = plt.subplots(figsize=(7, 3.5))

    counts, _, bars = ax.hist(
        missing_ratio,
        bins=20,
        range=(0, 1),
        edgecolor="black",
    )
    labels = [str(int(c)) if c > 0 else "" for c in counts]
    ax.bar_label(bars, labels=labels, fontsize=8, padding=2)
    ax.set_ylim(0, 1.15 * counts.max())

    ax.axvline(threshold, color="tab:red", linestyle="--", linewidth=1.5)
    ax.text(
        threshold - 0.01,
        counts.max(),
        f"cutoff {threshold:.2f}\n{n_dropped} features dropped",
        color="tab:red",
        fontsize=9,
        ha="right",
        va="top",
    )

    ax.set_title("Missing-value distribution across features")
    ax.set_xlabel("Fraction of missing values")
    ax.set_ylabel("Number of features")
    ax.set_xlim(0, 1)

    fig.tight_layout()
    return fig
