"""Regenerate every figure used in the report.

Run this rather than exporting images out of a notebook by hand. Figures in
the report must be reproducible by any of us with a single command:

    python scripts/make_figures.py
"""


def main() -> None:
    """Build and save all report figures.

    Load the data, call the plot_* helpers in ``mlproject.plots``, and pass
    each figure to ``mlproject.plots.save_figure``.
    """
    # *** YOUR CODE HERE ***
    raise NotImplementedError


if __name__ == "__main__":
    main()
