"""Classification metrics.

About 8.8 percent of the training labels are positive, so accuracy is close to
useless on its own: predicting the majority class everywhere already scores
roughly 0.91 with an F1 of zero. AIcrowd ranks on accuracy and F1, so report
both, and tune the decision threshold rather than leaving it at 0.5.
"""

import numpy as np


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute the fraction of correct predictions.

    Args:
        y_true: Ground truth labels, shape ``(N,)``.
        y_pred: Predicted labels using the same convention, shape ``(N,)``.

    Returns:
        Accuracy in ``[0, 1]``.
    """
    # *** YOUR CODE HERE ***
    raise NotImplementedError


def confusion_counts(
    y_true: np.ndarray, y_pred: np.ndarray, positive_label: int = 1
) -> tuple[int, int, int, int]:
    """Count true positives, false positives, true negatives, false negatives.

    Args:
        y_true: Ground truth labels, shape ``(N,)``.
        y_pred: Predicted labels, shape ``(N,)``.
        positive_label: Value that denotes the positive class.

    Returns:
        A tuple ``(tp, fp, tn, fn)``.
    """
    # *** YOUR CODE HERE ***
    raise NotImplementedError


def precision_recall_f1(
    y_true: np.ndarray, y_pred: np.ndarray, positive_label: int = 1
) -> tuple[float, float, float]:
    """Compute precision, recall and their harmonic mean.

    Return zero rather than raising when a denominator is zero, which happens
    whenever a model predicts no positives at all.

    Args:
        y_true: Ground truth labels, shape ``(N,)``.
        y_pred: Predicted labels, shape ``(N,)``.
        positive_label: Value that denotes the positive class.

    Returns:
        A tuple ``(precision, recall, f1)``.
    """
    # *** YOUR CODE HERE ***
    raise NotImplementedError


def best_threshold(
    y_true: np.ndarray, scores: np.ndarray, positive_label: int = 1
) -> tuple[float, float]:
    """Find the decision threshold that maximises F1 on a validation set.

    Args:
        y_true: Ground truth labels, shape ``(N,)``.
        scores: Continuous model scores, shape ``(N,)``.
        positive_label: Value that denotes the positive class.

    Returns:
        A tuple ``(threshold, f1)`` for the best threshold found.
    """
    # *** YOUR CODE HERE ***
    raise NotImplementedError
