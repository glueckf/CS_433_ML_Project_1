"""Tests for preprocessing functions."""

import numpy as np

import mlproject.preprocessing as preprocessing
from mlproject.preprocessing import (
    drop_constant_columns,
    drop_duplicate_columns,
    drop_high_missingness_columns,
    impute_missing,
    standardize,
)


def test_drop_high_missingness_columns():
    x = np.array(
        [
            [1.0, np.nan, 3.0],
            [2.0, np.nan, 4.0],
            [3.0, 5.0, 5.0],
            [4.0, np.nan, 6.0],
        ]
    )

    x_reduced, keep_mask = drop_high_missingness_columns(x, max_missing_fraction=0.5)

    assert np.array_equal(keep_mask, [True, False, True])
    assert x_reduced.shape == (4, 2)


def test_impute_missing_uses_training_values():
    x_train = np.array(
        [
            [1.0, 10.0],
            [3.0, np.nan],
            [5.0, 30.0],
        ]
    )

    x_train_filled, fill_values = impute_missing(x_train)

    x_val = np.array([[np.nan, 1000.0]])
    x_val_filled, _ = impute_missing(x_val, fill_values)

    assert np.allclose(fill_values, [3.0, 20.0])
    assert np.allclose(x_train_filled[1], [3.0, 20.0])
    assert np.allclose(x_val_filled, [[3.0, 1000.0]])


def test_drop_constant_columns():
    x = np.array(
        [
            [1.0, 5.0, 2.0],
            [2.0, 5.0, 3.0],
            [3.0, 5.0, 4.0],
        ]
    )

    x_reduced, keep_mask = drop_constant_columns(x)

    assert np.array_equal(keep_mask, [True, False, True])
    assert x_reduced.shape == (3, 2)


def test_standardize_uses_training_statistics():
    x_train = np.array(
        [
            [1.0, 10.0],
            [3.0, 20.0],
            [5.0, 30.0],
        ]
    )

    x_train_scaled, mean, std = standardize(x_train)

    x_val = np.array([[7.0, 40.0]])
    x_val_scaled, _, _ = standardize(x_val, mean=mean, std=std)

    assert np.allclose(mean, [3.0, 20.0])
    assert np.allclose(x_train_scaled.mean(axis=0), [0.0, 0.0])
    assert np.allclose(
        x_val_scaled,
        [[np.sqrt(6), np.sqrt(6)]],
    )


def test_drop_duplicate_columns():
    x_train = np.array(
        [
            [1.0, 2.0, 1.0],
            [3.0, 4.0, 3.0],
            [5.0, 6.0, 5.0],
        ]
    )
    x_val = np.array(
        [
            [7.0, 8.0, 7.0],
            [9.0, 10.0, 9.0],
        ]
    )

    x_train_reduced, x_val_reduced, keep_mask = drop_duplicate_columns(x_train, x_val)

    assert np.array_equal(keep_mask, [True, True, False])
    assert x_train_reduced.shape == (3, 2)
    assert x_val_reduced.shape == (2, 2)


def _fit_small_preprocessor(monkeypatch):
    """Create a small dataset for integration tests."""
    # Simulate feature names without loading the real BRFSS dataset.
    feature_names = [
        "numeric",
        "cat",
        "high_missing",
        "constant",
        "duplicate",
    ]

    monkeypatch.setattr(
        preprocessing,
        "load_feature_names",
        lambda csv_path: feature_names,
    )

    # Treat "cat" as a categorical feature.
    monkeypatch.setattr(
        preprocessing,
        "CATEGORICAL_FEATURES",
        ["cat"],
    )

    x_train = np.array(
        [
            [1.0, 0.0, np.nan, 5.0, 1.0],
            [2.0, 1.0, np.nan, 5.0, 2.0],
            [3.0, 0.0, 1.0, 5.0, 3.0],
            [4.0, 1.0, np.nan, 5.0, 4.0],
        ]
    )

    x_val = np.array(
        [
            [10.0, 2.0, np.nan, 5.0, 10.0],
        ]
    )

    x_train_processed, state = preprocessing.fit_preprocessor(
        x_train,
        csv_path="dummy.csv",
        threshold=0.5,
        use_one_hot=True,
        use_bias=True,
        use_special_codes=False,
        use_augmented_input=False,
    )

    return x_train, x_val, x_train_processed, state


def test_fit_and_transform_produce_compatible_finite_matrices(
    monkeypatch,
):
    """Train and validation must have compatible finite features."""
    _, x_val, x_train_processed, state = _fit_small_preprocessor(monkeypatch)

    x_val_processed = preprocessing.transform_preprocessor(x_val, state)

    assert x_train_processed.shape[1] == x_val_processed.shape[1]
    assert np.all(np.isfinite(x_train_processed))
    assert np.all(np.isfinite(x_val_processed))

    # The bias column must contain only ones.
    assert np.all(x_train_processed[:, 0] == 1.0)
    assert np.all(x_val_processed[:, 0] == 1.0)


def test_transform_reuses_training_standardization_statistics(
    monkeypatch,
):
    """Validation data must not change the fitted scaling statistics."""
    _, x_val, x_train_processed, state = _fit_small_preprocessor(monkeypatch)

    mean_before = state["mean"].copy()
    std_before = state["std"].copy()

    x_val_processed = preprocessing.transform_preprocessor(x_val, state)

    # The validation value of the numerical feature is 10.
    expected_scaled_value = (10.0 - mean_before[0]) / std_before[0]

    # Column 0 is the bias; column 1 is the numerical feature.
    assert np.isclose(
        x_val_processed[0, 1],
        expected_scaled_value,
    )

    # Transforming validation must not modify the fitted statistics.
    assert np.array_equal(state["mean"], mean_before)
    assert np.array_equal(state["std"], std_before)


def test_transform_handles_unseen_categorical_values(monkeypatch):
    """An unseen category must not break one-hot encoding."""
    _, x_val, x_train_processed, state = _fit_small_preprocessor(monkeypatch)

    # Category 2 was not present in the training data (categories 0 and 1).
    x_val_processed = preprocessing.transform_preprocessor(x_val, state)

    assert x_val_processed.shape[1] == x_train_processed.shape[1]
    assert np.all(np.isfinite(x_val_processed))
