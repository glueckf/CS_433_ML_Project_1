"""Tests for cross-validation and hyper-parameter search."""

import numpy as np
import pytest

from mlproject.validation import (
    build_k_fold_indices,
    cross_validate,
    grid_search,
)


def test_build_k_fold_indices_is_reproducible_and_disjoint() -> None:
    """Create reproducible folds that cover every row exactly once."""
    folds = build_k_fold_indices(n_samples=12, k=3, seed=7)

    assert folds.shape == (3, 4)
    assert np.array_equal(
        folds,
        build_k_fold_indices(n_samples=12, k=3, seed=7),
    )
    assert np.array_equal(np.sort(folds.ravel()), np.arange(12))


def test_build_k_fold_indices_rejects_invalid_sizes() -> None:
    """Reject fold counts that cannot produce equal-sized folds."""
    with pytest.raises(ValueError):
        build_k_fold_indices(n_samples=10, k=3)

    with pytest.raises(ValueError):
        build_k_fold_indices(n_samples=4, k=1)


def test_cross_validate_averages_classification_metrics() -> None:
    """Average metrics returned by a deterministic prediction function."""

    def predict_negative(
        x_train: np.ndarray, y_train: np.ndarray, x_validation: np.ndarray
    ) -> np.ndarray:
        del x_train, y_train
        return np.full(x_validation.shape[0], -1)

    x = np.arange(12, dtype=float).reshape(6, 2)
    y = np.array([1, -1, 1, -1, 1, -1])

    metrics = cross_validate(x, y, predict_negative, k=3, seed=7)

    assert metrics == {
        "accuracy": 0.5,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
    }


def test_grid_search_selects_highest_f1_configuration() -> None:
    """Select the parameter value whose predictions have the best F1."""

    def build_model(label: int):
        def predict(
            x_train: np.ndarray,
            y_train: np.ndarray,
            x_validation: np.ndarray,
        ) -> np.ndarray:
            del x_train, y_train
            return np.full(x_validation.shape[0], label)

        return predict

    x = np.arange(12, dtype=float).reshape(6, 2)
    y = np.array([1, 1, 1, -1, -1, -1])

    best_params, best_metrics = grid_search(
        x,
        y,
        build_model,
        {"label": [-1, 1]},
        k=3,
    )

    assert best_params == {"label": 1}
    assert best_metrics["f1"] > 0.0
