"""Cleaning the raw feature matrix: missing values, constants, scaling.

The BRFSS data has 321 raw features, heavy missingness, and survey-specific
sentinel codes (7 and 9 commonly mean "don't know" and "refused", 77/99 and
777/999 the same on wider scales). Treating those as real numeric values is
the single most common way to get a bad score on this dataset.
"""

import csv

import numpy as np

from mlproject.special_code_brfss2015 import (
    replace_special_codes,
)


def detect_special_code_candidates(x, feature_names, max_unique_values=20):
    """Detect potential special missing codes in features.

    Features with 20 or more unique values are ignored.
    For other features, report possible special codes.
    No values are replaced.
    """
    candidates = {}

    possible_codes = {7, 9, 77, 99, 98, 88, 777, 999, 888, 7777, 9999, 8888, 99999}

    for j, name in enumerate(feature_names):
        col = x[:, j]
        col = col[~np.isnan(col)]

        if len(col) == 0:
            continue

        unique_values, counts = np.unique(col, return_counts=True)

        # Ignore features with at least 20 distinct values
        if len(unique_values) >= max_unique_values:
            continue

        values = set(unique_values.tolist())
        observed_candidates = values.intersection(possible_codes)

        if observed_candidates:
            candidates[name] = {
                "unique_values": sorted(values),
                "possible_codes": sorted(observed_candidates),
                "counts": {
                    value.item() if isinstance(value, np.generic) else value: int(count)
                    for value, count in zip(unique_values, counts, strict=True)
                },
            }

    return candidates


def load_feature_names(csv_path):
    """Read feature names, excluding the first ID column."""
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        header = next(reader)

    # load_csv_data() removes the first column (ID) from x_train/x_test.
    return [name.strip() for name in header[1:]]


def drop_constant_columns(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Remove columns with zero variance, which carry no information.

    Args:
        x: Feature matrix, shape ``(N, D)``.

    Returns:
        A tuple ``(x_reduced, keep_mask)`` where ``keep_mask`` has shape
        ``(D,)`` and must be reused verbatim on the test set.
    """
    # *** YOUR CODE HERE ***
    n_features = x.shape[1]
    keep_mask = np.ones(n_features, dtype=bool)

    for j in range(n_features):
        observed = x[~np.isnan(x[:, j]), j]

        if observed.size == 0 or np.all(observed == observed[0]):
            keep_mask[j] = False

    return x[:, keep_mask], keep_mask


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
    if not 0.0 <= max_missing_fraction <= 1.0:
        raise ValueError("max_missing_fraction must be between 0 and 1.")

    missing_fraction = np.mean(np.isnan(x), axis=0)

    # print("Missing fraction percentiles:")
    # print(np.percentile(missing_fraction, [0, 25, 50, 75, 90, 95, 100]))

    keep_mask = missing_fraction <= max_missing_fraction

    # print(f"Threshold: {max_missing_fraction:.0%} |
    #  "f"Kept: {np.sum(keep_mask)}
    # / {x.shape[1]} | "f"Removed: {np.sum(~keep_mask)}")

    return x[:, keep_mask], keep_mask


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
    x_filled = x.copy()

    if fill_values is None:
        fill_values = np.zeros(x.shape[1], dtype=float)

        for j in range(x.shape[1]):
            observed = x[~np.isnan(x[:, j]), j]

            if observed.size == 0:
                # An entirely missing column has no estimable median.
                # Use zero as a fallback; preferably remove this column
                # before calling this function.
                fill_values[j] = 0.0
            else:
                fill_values[j] = np.median(observed)
    else:
        fill_values = np.asarray(fill_values, dtype=float)

        if fill_values.shape != (x.shape[1],):
            raise ValueError("fill_values must have one value per feature.")

    rows, cols = np.where(np.isnan(x_filled))
    x_filled[rows, cols] = fill_values[cols]

    return x_filled, fill_values


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
    mean = np.mean(x, axis=0) if mean is None else np.asarray(mean, dtype=float)

    std = np.std(x, axis=0) if std is None else np.asarray(std, dtype=float)

    if mean.shape != (x.shape[1],) or std.shape != (x.shape[1],):
        raise ValueError("mean and std must have one value per feature.")

    std = np.array(std, dtype=float, copy=True)
    std[std < 1e-12] = 1.0

    x_standardized = (x - mean) / std

    return x_standardized, mean, std


def add_bias_column(x: np.ndarray) -> np.ndarray:
    """Prepend a column of ones so the model can learn an intercept.

    Args:
        x: Feature matrix, shape ``(N, D)``.

    Returns:
        Feature matrix of shape ``(N, D + 1)``.
    """
    # *** YOUR CODE HERE ***
    ones = np.ones((x.shape[0], 1), dtype=x.dtype)
    return np.concatenate((ones, x), axis=1)


def stratified_split(x, y, validation_fraction=0.2, seed=42):
    """Split data while approximately preserving class proportions."""
    rng = np.random.default_rng(seed)

    train_indices = []
    val_indices = []

    for label in np.unique(y):
        indices = np.where(y == label)[0]
        rng.shuffle(indices)

        n_val = int(round(validation_fraction * len(indices)))

        val_indices.extend(indices[:n_val])
        train_indices.extend(indices[n_val:])

    train_indices = np.array(train_indices, dtype=int)
    val_indices = np.array(val_indices, dtype=int)

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)

    return (
        x[train_indices],
        y[train_indices],
        x[val_indices],
        y[val_indices],
    )


def drop_duplicate_columns(
    x_train: np.ndarray,
    x_val: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Remove exact duplicate columns using training data only."""
    n_features = x_train.shape[1]
    keep_mask = np.ones(n_features, dtype=bool)

    for j in range(n_features):
        if not keep_mask[j]:
            continue

        for k in range(j + 1, n_features):
            if keep_mask[k] and np.array_equal(x_train[:, j], x_train[:, k]):
                keep_mask[k] = False

    x_train = x_train[:, keep_mask]
    x_val = x_val[:, keep_mask]

    return x_train, x_val, keep_mask


def preprocess_data(
    x_train,
    x_val,
    csv_path,
    threshold=0.97,
):
    """Preprocess training and validation data."""
    print("\n===== PREPROCESSING =====")

    # 1. Load feature names.
    feature_names = load_feature_names(csv_path)

    if len(feature_names) != x_train.shape[1]:
        raise ValueError(
            f"Feature mismatch: {len(feature_names)} names "
            f"for {x_train.shape[1]} columns."
        )

    if x_train.shape[1] != x_val.shape[1]:
        raise ValueError("Training and validation have different feature counts.")

    # 2. Replace documented special codes with NaN.
    x_train = replace_special_codes(x_train, feature_names)
    x_val = replace_special_codes(x_val, feature_names)

    # print("Special codes replaced.")

    # 3. Remove columns with too many missing values.
    x_train, keep_mask = drop_high_missingness_columns(
        x_train,
        max_missing_fraction=threshold,
    )

    x_val = x_val[:, keep_mask]

    # Fit imputation values on training data only.
    x_train, fill_values = impute_missing(x_train)

    # Apply the same values to validation data.
    x_val, _ = impute_missing(x_val, fill_values)

    # 5. Remove constant columns.
    # Remove constant columns based on training data only.
    x_train, keep_mask = drop_constant_columns(x_train)

    # Apply the same mask to validation data.
    x_val = x_val[:, keep_mask]

    # Remove exact duplicate columns.
    x_train, x_val, duplicate_mask = drop_duplicate_columns(x_train, x_val)

    print(
        "Duplicate columns removed:",
        np.sum(~duplicate_mask),
    )
    # 6. Standardize using training statistics only.
    #

    x_train, mean, std = standardize(x_train)
    x_val, _, _ = standardize(x_val, mean=mean, std=std)
    print("Data standardized.")

    # 7. Add bias column.
    x_train = add_bias_column(x_train)
    x_val = add_bias_column(x_val)
    print("Bias column added.")
    # 8. Check the output.
    if not np.all(np.isfinite(x_train)):
        raise ValueError("x_train contains NaN or infinite values.")

    if not np.all(np.isfinite(x_val)):
        raise ValueError("x_val contains NaN or infinite values.")

    print("Final training shape:", x_train.shape)
    print("Final validation shape:", x_val.shape)

    return x_train, x_val


# for TEST must be delete
def evaluate_baseline(y_true, predictions, threshold=0):
    """Print basic classification metrics using threshold zero."""
    y_pred = np.where(predictions >= threshold, 1, -1)

    accuracy = np.mean(y_pred == y_true)

    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == -1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == -1))

    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    )
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
