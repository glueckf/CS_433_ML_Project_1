"""Cross-validation and hyper-parameter search.

The project description is explicit that the AIcrowd leaderboard is not an
evaluation tool: with five submissions a day, tuning against it is both slow
and a good way to overfit the public split. Everything gets decided locally
here, and the leaderboard is only ever a sanity check.
"""

from __future__ import annotations

from collections.abc import Callable
from itertools import product

import numpy as np

from mlproject.config import N_FOLDS, RANDOM_SEED
from mlproject.metrics import accuracy, precision_recall_f1


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
    if n_samples <= 0:
        raise ValueError("n_samples must be positive.")
    if k <= 1 or k > n_samples:
        raise ValueError("k must be greater than 1 and at most n_samples.")
    if n_samples % k:
        raise ValueError("n_samples must be divisible by k.")

    indices = np.arange(n_samples)
    np.random.default_rng(seed).shuffle(indices)
    return indices.reshape(k, n_samples // k)


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
    x = np.asarray(x)
    y = np.asarray(y)
    if x.ndim != 2 or y.ndim != 1:
        raise ValueError("x must be 2-D and y must be 1-D.")
    if x.shape[0] != y.shape[0]:
        raise ValueError("x and y must contain the same number of rows.")

    folds = build_k_fold_indices(x.shape[0], k=k, seed=seed)
    fold_metrics: list[dict[str, float]] = []

    for validation_indices in folds:
        training_mask = np.ones(x.shape[0], dtype=bool)
        training_mask[validation_indices] = False

        predictions = np.asarray(
            fit_predict(x[training_mask], y[training_mask], x[validation_indices])
        )
        if predictions.shape != y[validation_indices].shape:
            raise ValueError("fit_predict returned predictions with the wrong shape.")

        precision, recall, f1 = precision_recall_f1(y[validation_indices], predictions)
        fold_metrics.append(
            {
                "accuracy": accuracy(y[validation_indices], predictions),
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )

    return {
        metric: float(np.mean([fold[metric] for fold in fold_metrics]))
        for metric in fold_metrics[0]
    }


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
    if not param_grid or any(not values for values in param_grid.values()):
        raise ValueError("param_grid must contain at least one value per parameter.")

    parameter_names = list(param_grid)
    best_params: dict = {}
    best_metrics: dict[str, float] | None = None

    for values in product(*(param_grid[name] for name in parameter_names)):
        params = dict(zip(parameter_names, values))
        metrics = cross_validate(x, y, build_model(**params), k=k)
        if best_metrics is None or metrics["f1"] > best_metrics["f1"]:
            best_params = params
            best_metrics = metrics

    assert best_metrics is not None
    return best_params, best_metrics
