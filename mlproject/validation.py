"""Cross-validation and hyper-parameter search.

The project description is explicit that the AIcrowd leaderboard is not an
evaluation tool: with five submissions a day, tuning against it is both slow
and a good way to overfit the public split. Everything gets decided locally
here, and the leaderboard is only ever a sanity check.
"""

from collections.abc import Callable

import numpy as np

from mlproject.config import N_FOLDS, RANDOM_SEED


def build_k_fold_indices(
    n_samples: int, k: int = N_FOLDS, seed: int = RANDOM_SEED
) -> np.ndarray:
    """Build the row indices for each cross-validation fold.

    Args:
        n_samples: Number of rows in the training set.
        k: Number of folds.
        seed: Seed for the shuffle, so folds are identical across runs and
            across all three of our machines.

    Returns:
        An integer array of shape ``(k, n_samples // k)``.
    """
    # *** YOUR CODE HERE ***
    raise NotImplementedError


def cross_validate(
    x: np.ndarray,
    y: np.ndarray,
    fit_predict: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
    k: int = N_FOLDS,
    seed: int = RANDOM_SEED,
) -> dict[str, float]:
    """Run k-fold cross-validation and average the metrics across folds.

    Preprocessing must be fitted inside each fold, never once on the whole
    training set beforehand. Fitting outside the loop leaks the held-out fold
    into the imputation and scaling statistics, which inflates the local score
    and does not transfer to the leaderboard.

    Args:
        x: Feature matrix, shape ``(N, D)``.
        y: Labels, shape ``(N,)``.
        fit_predict: Callable taking ``(x_tr, y_tr, x_val)`` and returning
            predictions for ``x_val``.
        k: Number of folds.
        seed: Seed controlling the fold assignment.

    Returns:
        A dict of metric name to mean value across folds.
    """
    # *** YOUR CODE HERE ***
    raise NotImplementedError


def grid_search(
    x: np.ndarray,
    y: np.ndarray,
    build_model: Callable[..., Callable],
    param_grid: dict[str, list],
    k: int = N_FOLDS,
) -> tuple[dict, dict[str, float]]:
    """Search a hyper-parameter grid by cross-validation.

    Args:
        x: Feature matrix, shape ``(N, D)``.
        y: Labels, shape ``(N,)``.
        build_model: Callable that takes hyper-parameters as keyword arguments
            and returns a ``fit_predict`` callable.
        param_grid: Mapping of parameter name to the list of values to try.
        k: Number of folds.

    Returns:
        A tuple ``(best_params, best_metrics)``.
    """
    # *** YOUR CODE HERE ***
    raise NotImplementedError
