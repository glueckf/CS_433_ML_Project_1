"""Feature engineering: building better inputs from the cleaned columns.

The grading rubric rewards scientific contribution, and feature work is where
most of it is available on this dataset. Every transformation added here
should be switchable, so the report can carry an ablation study showing which
ones actually mattered.
"""

import numpy as np


def one_hot_encode(
    x: np.ndarray,
    column_indices: list[int],
    categories: dict[int, np.ndarray] | None = None,
) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    """Expand categorical columns into binary indicator columns.

    Many BRFSS columns are categorical codes rather than quantities; treating
    them as ordinal numbers imposes an ordering that does not exist.

    Args:
        x: Feature matrix, shape ``(N, D)``.
        column_indices: Indices of the columns to expand.
        categories: Category values seen at fit time, keyed by column index.
            If ``None``, they are learned from ``x``. Reuse the fitted value on
            the test set so the column count matches.

    Returns:
        A tuple ``(x_encoded, categories)``.
    """
    # *** YOUR CODE HERE ***
    raise NotImplementedError


def add_polynomial_features(x: np.ndarray, degree: int) -> np.ndarray:
    """Augment the feature matrix with powers of each column.

    Args:
        x: Feature matrix, shape ``(N, D)``.
        degree: Highest power to include. ``degree=1`` returns ``x`` unchanged.

    Returns:
        The expanded feature matrix.
    """
    # *** YOUR CODE HERE ***
    raise NotImplementedError


def add_missingness_indicators(x: np.ndarray) -> np.ndarray:
    """Append a binary column per feature marking where values were missing.

    On survey data, whether someone answered a question is often predictive in
    itself, so this information is worth keeping rather than imputing away.

    Args:
        x: Feature matrix, shape ``(N, D)``, with missing values as ``nan``.

    Returns:
        Feature matrix with indicator columns appended.
    """
    # *** YOUR CODE HERE ***
    raise NotImplementedError


def build_features(
    x: np.ndarray, fitted_state: dict | None = None
) -> tuple[np.ndarray, dict]:
    """Run the full feature pipeline end to end.

    This is the single entry point that ``run.py`` calls, so training and
    inference cannot drift apart. Call it once with ``fitted_state=None`` on
    the training set, then pass the returned state when transforming the test
    set.

    Args:
        x: Raw feature matrix, shape ``(N, D)``.
        fitted_state: State learned on the training set, or ``None`` to fit.

    Returns:
        A tuple ``(x_transformed, fitted_state)``.
    """
    # *** YOUR CODE HERE ***
    raise NotImplementedError
