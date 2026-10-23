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
    return float(np.mean(y_true == y_pred))


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
    true_positive = y_true == positive_label
    predicted_positive = y_pred == positive_label

    tp = int(np.sum(true_positive & predicted_positive))
    fp = int(np.sum(~true_positive & predicted_positive))
    tn = int(np.sum(~true_positive & ~predicted_positive))
    fn = int(np.sum(true_positive & ~predicted_positive))
    return tp, fp, tn, fn


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
    tp, fp, tn, fn = confusion_counts(y_true, y_pred, positive_label)

    precision_denominator = tp + fp
    recall_denominator = tp + fn
    precision = tp / precision_denominator if precision_denominator else 0.0
    recall = tp / recall_denominator if recall_denominator else 0.0

    f1_denominator = precision + recall
    f1 = 2 * precision * recall / f1_denominator if f1_denominator else 0.0
    return float(precision), float(recall), float(f1)


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
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    if y_true.shape != scores.shape:
        raise ValueError("y_true and scores must have the same shape.")
    if scores.ndim != 1 or scores.size == 0:
        raise ValueError("scores must be a non-empty one-dimensional array.")

    negative_label = -positive_label
    best_score = -1.0
    best_value = float(scores[0])

    for threshold in np.unique(scores):
        predictions = np.where(scores >= threshold, positive_label, negative_label)
        _, _, f1 = precision_recall_f1(y_true, predictions, positive_label)
        if f1 > best_score:
            best_score = f1
            best_value = float(threshold)

    return best_value, float(best_score)
