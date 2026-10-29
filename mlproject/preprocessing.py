"""Cleaning the raw feature matrix: missing values, constants, scaling.

The BRFSS data has 321 raw features, heavy missingness, and survey-specific
sentinel codes (7 and 9 commonly mean "don't know" and "refused", 77/99 and
777/999 the same on wider scales). Treating those as real numeric values is
the single most common way to get a bad score on this dataset.
"""

from __future__ import annotations

import csv

import numpy as np

from mlproject.special_code_brfss2015 import (
    CATEGORICAL_FEATURES,
    QUANTITATIVE_FEATURES,
    replace_special_codes,
)


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
    if not 0.0 <= max_missing_fraction <= 1.0:
        raise ValueError("max_missing_fraction must be between 0 and 1.")

    missing_fraction = np.mean(np.isnan(x), axis=0)

    keep_mask = missing_fraction <= max_missing_fraction

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


def fit_preprocessor(
    x_train,
    csv_path,
    threshold=0.97,
    use_one_hot=False,
    use_bias=True,
    use_special_codes=True,
    use_augmented_input=False,
):
    """Fit preprocessing on training data and return transformed data and state."""
    feature_names = load_feature_names(csv_path)
    x = np.asarray(x_train, dtype=float).copy()

    if x.shape[1] != len(feature_names):
        raise ValueError("Feature names do not match x_train.")

    # 1. Replace documented special codes.
    if use_special_codes:
        x = replace_special_codes(x, feature_names)

    # 2. Remove columns with too many missing values.
    x, missing_mask = drop_high_missingness_columns(x, threshold)
    feature_names = [name for name, keep in zip(feature_names, missing_mask) if keep]

    # 3. Learn imputation values from training data.
    x, fill_values = impute_missing(x)

    # 4. Remove constant columns.
    x, constant_mask = drop_constant_columns(x)
    feature_names = [name for name, keep in zip(feature_names, constant_mask) if keep]

    # 5. Remove duplicate columns using training data only.
    x, _, duplicate_mask = drop_duplicate_columns(x, x)
    feature_names = [name for name, keep in zip(feature_names, duplicate_mask) if keep]

    # 6. Learn categorical values from training data.
    categorical_set = set(CATEGORICAL_FEATURES)
    categories_by_feature = {}

    if use_one_hot:
        for j, name in enumerate(feature_names):
            if name in categorical_set:
                categories_by_feature[name] = np.unique(x[:, j])

    # 7. Augment quantitative features if requested.
    if use_augmented_input:
        x, _, feature_names = augment_inputs_train_val(x, x, feature_names)

    # 8. Apply one-hot encoding using the learned categories.
    if use_one_hot:
        x, feature_names = _encode_with_categories(
            x, feature_names, categories_by_feature
        )

    # 9. Learn scaling statistics from training data.
    x, mean, std = standardize(x)

    # 10. Add the bias column last.
    if use_bias:
        x = add_bias_column(x)

    state = {
        "feature_names": load_feature_names(csv_path),
        "missing_mask": missing_mask,
        "constant_mask": constant_mask,
        "duplicate_mask": duplicate_mask,
        "fill_values": fill_values,
        "categories_by_feature": categories_by_feature,
        "use_one_hot": use_one_hot,
        "use_bias": use_bias,
        "use_special_codes": use_special_codes,
        "use_augmented_input": use_augmented_input,
        "mean": mean,
        "std": std,
    }

    return x, state


def _encode_with_categories(x, feature_names, categories_by_feature):
    """Encode categorical features using categories learned on training data."""
    categorical_set = set(categories_by_feature)
    numerical_indices = [
        i for i, name in enumerate(feature_names) if name not in categorical_set
    ]

    parts = [x[:, numerical_indices]]
    encoded_names = [feature_names[i] for i in numerical_indices]

    for name, categories in categories_by_feature.items():
        idx = feature_names.index(name)
        encoded_categories = categories[1:]

        encoded = (x[:, idx, None] == encoded_categories[None, :]).astype(float)

        parts.append(encoded)
        encoded_names.extend(f"{name}_{category:g}" for category in encoded_categories)

    return np.concatenate(parts, axis=1), encoded_names


def augment_inputs_train_val(
    x_train,
    x_val,
    feature_names,
    quantitative_features=QUANTITATIVE_FEATURES,
):
    """Augment the input features with their squared values.

    Categorical features are excluded from the augmentation.
    The original features are preserved.

    Parameters
    ----------
    x_train : ndarray, shape (n_train, n_features)
        Training feature matrix.
    x_val : ndarray, shape (n_val, n_features)
        Validation feature matrix.
    feature_names : list of str
        Names of the input features.
    quantitative_features : list of str
    Quantitative features whose squared values are appended.

    Returns:
    -------
    x_train_augmented : ndarray
        Training matrix with original and squared features.
    x_val_augmented : ndarray
        Validation matrix with original and squared features.
    augmented_feature_names : list of str
        Names of the original and squared features.
    """
    feature_names = list(feature_names)
    quantitative_set = set(quantitative_features)

    if x_train.shape[1] != len(feature_names):
        raise ValueError("The number of feature names does not match x_train.")

    if x_val.shape[1] != len(feature_names):
        raise ValueError("The number of feature names does not match x_val.")

    numeric_indices = [
        i for i, name in enumerate(feature_names) if name in quantitative_set
    ]

    if not numeric_indices:
        print("No features selected for augmentation.")
        return x_train, x_val, feature_names

    # Extract the features to square.
    x_train_numeric = x_train[:, numeric_indices]
    x_val_numeric = x_val[:, numeric_indices]

    # Compute squared features.
    x_train_squared = x_train_numeric**2
    x_val_squared = x_val_numeric**2

    # Append squared features to the original matrices.
    x_train_augmented = np.concatenate(
        [x_train, x_train_squared],
        axis=1,
    )

    x_val_augmented = np.concatenate(
        [x_val, x_val_squared],
        axis=1,
    )

    squared_feature_names = [f"{feature_names[i]}_squared" for i in numeric_indices]

    augmented_feature_names = feature_names + squared_feature_names

    print(f"Added {len(squared_feature_names)} squared features.")

    return (
        x_train_augmented,
        x_val_augmented,
        augmented_feature_names,
    )


def transform_preprocessor(x, state):
    """Transform validation/test data using fitted training parameters."""
    feature_names = state["feature_names"]
    x = np.asarray(x, dtype=float).copy()

    if x.shape[1] != len(feature_names):
        raise ValueError("Feature names do not match input data.")

    # 1. Apply the same special-code rules.
    if state["use_special_codes"]:
        x = replace_special_codes(x, feature_names)

    # 2. Apply the same column-selection masks.
    x = x[:, state["missing_mask"]]

    # 3. Impute using training-set values.
    x, _ = impute_missing(x, state["fill_values"])

    # 4. Remove the same constant and duplicate columns.
    x = x[:, state["constant_mask"]]
    x = x[:, state["duplicate_mask"]]

    # Reconstruct the feature names after column selection.
    feature_names = [
        name for name, keep in zip(feature_names, state["missing_mask"]) if keep
    ]
    feature_names = [
        name for name, keep in zip(feature_names, state["constant_mask"]) if keep
    ]
    feature_names = [
        name for name, keep in zip(feature_names, state["duplicate_mask"]) if keep
    ]

    # 5. Apply the same feature augmentation.
    if state["use_augmented_input"]:
        x, _, feature_names = augment_inputs_train_val(x, x, feature_names)

    # 6. Apply the categories learned from training data.
    if state["use_one_hot"]:
        x, _ = _encode_with_categories(x, feature_names, state["categories_by_feature"])

    # 7. Reuse training means and standard deviations.
    x, _, _ = standardize(x, mean=state["mean"], std=state["std"])

    # 8. Add the bias column if requested.
    if state["use_bias"]:
        x = add_bias_column(x)

    if not np.all(np.isfinite(x)):
        raise ValueError("Processed data contains NaN or infinite values.")

    return x
