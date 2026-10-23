"""Tests for classification metrics."""

import numpy as np
import pytest

from mlproject.metrics import (
    accuracy,
    best_threshold,
    confusion_counts,
    precision_recall_f1,
)


def test_accuracy_and_confusion_counts() -> None:
    """Compute accuracy and all four confusion-matrix counts."""
    y_true = np.array([1, 1, -1, -1])
    y_pred = np.array([1, -1, 1, -1])

    assert accuracy(y_true, y_pred) == 0.5
    assert confusion_counts(y_true, y_pred) == (1, 1, 1, 1)


def test_precision_recall_f1() -> None:
    """Compute precision, recall, and F1 from confusion counts."""
    y_true = np.array([1, 1, -1, -1])
    y_pred = np.array([1, -1, 1, -1])

    assert precision_recall_f1(y_true, y_pred) == (0.5, 0.5, 0.5)


def test_precision_recall_f1_returns_zero_without_positive_predictions() -> None:
    """Avoid division by zero when a classifier predicts only negatives."""
    y_true = np.array([1, -1, 1, -1])
    y_pred = np.array([-1, -1, -1, -1])

    assert precision_recall_f1(y_true, y_pred) == (0.0, 0.0, 0.0)


def test_best_threshold_maximises_f1() -> None:
    """Choose a threshold that separates the positive examples."""
    y_true = np.array([-1, -1, 1, 1])
    scores = np.array([-2.0, -1.0, 1.0, 2.0])

    threshold, f1 = best_threshold(y_true, scores)

    assert threshold == 1.0
    assert f1 == 1.0


@pytest.mark.parametrize(
    ("y_true", "scores"),
    [
        (np.array([1, -1]), np.array([0.0])),
        (np.array([]), np.array([])),
        (np.array([1, -1]), np.array([])),
    ],
)
def test_best_threshold_rejects_invalid_shapes(
    y_true: np.ndarray, scores: np.ndarray
) -> None:
    """Reject empty or shape-mismatched threshold inputs."""
    with pytest.raises(ValueError):
        best_threshold(y_true, scores)
