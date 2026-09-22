"""Cleaning the raw feature matrix: missing values, constants, scaling.

The BRFSS data has 321 raw features, heavy missingness, and survey-specific
sentinel codes (7 and 9 commonly mean "don't know" and "refused", 77/99 and
777/999 the same on wider scales). Treating those as real numeric values is
the single most common way to get a bad score on this dataset.
"""

import numpy as np


def drop_constant_columns(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Remove columns with zero variance, which carry no information.

    Args:
        x: Feature matrix, shape ``(N, D)``.

    Returns:
        A tuple ``(x_reduced, keep_mask)`` where ``keep_mask`` has shape
        ``(D,)`` and must be reused verbatim on the test set.
    """
    # *** YOUR CODE HERE ***
    raise NotImplementedError


def drop_high_missingness_columns(
    x: np.ndarray, max_missing_fraction: float
) -> tuple[np.ndarray, np.ndarray]:
    """Remove columns whose fraction of missing values exceeds a threshold.

    Args:
        x: Feature matrix, shape ``(N, D)``, with missing values as ``nan``.
        max_missing_fraction: Columns missing more than this fraction of
            values are dropped.

    Returns:
        A tuple ``(x_reduced, keep_mask)``. Reuse ``keep_mask`` on the test
        set rather than recomputing it there.
    """
    # *** YOUR CODE HERE ***
    raise NotImplementedError


def impute_missing(
    x: np.ndarray, fill_values: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Replace missing values with a per-column statistic.

    Fit the fill values on the training set only and pass them in when
    transforming validation or test data. Recomputing them on the test set
    leaks information and quietly inflates local scores.

    Args:
        x: Feature matrix, shape ``(N, D)``, with missing values as ``nan``.
        fill_values: Precomputed per-column fill values, shape ``(D,)``. If
            ``None``, they are computed from ``x``.

    Returns:
        A tuple ``(x_filled, fill_values)``.
    """
    # *** YOUR CODE HERE ***
    raise NotImplementedError


def standardize(
    x: np.ndarray,
    mean: np.ndarray | None = None,
    std: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Centre and scale features to zero mean and unit variance.

    Gradient descent on raw BRFSS columns barely converges, because feature
    scales span several orders of magnitude. As with imputation, fit on train
    and reuse the statistics everywhere else.

    Args:
        x: Feature matrix, shape ``(N, D)``.
        mean: Precomputed per-column mean, shape ``(D,)``, or ``None``.
        std: Precomputed per-column standard deviation, shape ``(D,)``, or
            ``None``. Guard against division by zero.

    Returns:
        A tuple ``(x_standardized, mean, std)``.
    """
    # *** YOUR CODE HERE ***
    raise NotImplementedError


def add_bias_column(x: np.ndarray) -> np.ndarray:
    """Prepend a column of ones so the model can learn an intercept.

    Args:
        x: Feature matrix, shape ``(N, D)``.

    Returns:
        Feature matrix of shape ``(N, D + 1)``.
    """
    # *** YOUR CODE HERE ***
    raise NotImplementedError
