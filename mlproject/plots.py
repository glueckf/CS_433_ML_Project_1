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


def plot_missingness(x: np.ndarray, feature_names: list[str]) -> plt.Figure:
    """Plot the distribution of missing-value fractions across features."""
    if x.shape[1] != len(feature_names):
        raise ValueError(
            "The number of feature names must match the number of columns."
        )

    missing_ratio = np.mean(np.isnan(x), axis=0)

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
