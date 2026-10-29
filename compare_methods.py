"""Reproduce the method comparison table of the report.

Trains each of the six methods from implementations.py on the same stratified
training split and preprocessing as run.py, tunes the decision threshold for F1
on the validation split, and prints the results.

Usage:
    python compare_methods.py
"""

import numpy as np

import implementations as impl
from helpers import load_csv_data
from mlproject.config import DATA_DIR, RANDOM_SEED, VALIDATION_FRACTION
from mlproject.metrics import best_threshold
from mlproject.preprocessing import (
    fit_preprocessor,
    stratified_split,
    transform_preprocessor,
)

PREPROCESSING_OPTIONS = {
    "csv_path": str(DATA_DIR / "x_train.csv"),
    "threshold": 0.80,
    "use_one_hot": True,
    "use_bias": True,
    "use_special_codes": True,
    "use_augmented_input": False,
}


def train_all_methods(y, tx):
    """Fit every method with the hyperparameters reported in the table.

    Args:
        y: Training labels in {-1, 1}, shape (N,).
        tx: Preprocessed training features, shape (N, D).

    Returns:
        A list of (name, parameters, weights) tuples.
    """
    y01 = (y == 1).astype(float)
    w0 = np.zeros(tx.shape[1])

    np.random.seed(RANDOM_SEED)
    w_sgd, _ = impl.mean_squared_error_sgd(y, tx, w0, 500, 0.01)

    return [
        ("Least Squares", "-", impl.least_squares(y, tx)[0]),
        (
            "MSE GD",
            "100 iters, gamma=0.001",
            impl.mean_squared_error_gd(y, tx, w0, 100, 0.001)[0],
        ),
        ("MSE SGD", "500 iters, gamma=0.01", w_sgd),
        (
            "Logistic Reg.",
            "100 iters, gamma=0.1",
            impl.logistic_regression(y01, tx, w0, 100, 0.1)[0],
        ),
        (
            "Reg. Logistic",
            "lambda=1e-5, 100 iters, gamma=0.1",
            impl.reg_logistic_regression(y01, tx, 1e-5, w0, 100, 0.1)[0],
        ),
        ("Ridge Reg.", "lambda=1e-5", impl.ridge_regression(y, tx, 1e-5)[0]),
    ]


def main() -> None:
    """Train all six methods and print their validation F1 scores."""
    x_train, _, y_train, _, _ = load_csv_data(str(DATA_DIR))
    x_tr, y_tr, x_val, y_val = stratified_split(
        x_train, y_train, validation_fraction=VALIDATION_FRACTION, seed=RANDOM_SEED
    )
    tx_tr, state = fit_preprocessor(x_tr, **PREPROCESSING_OPTIONS)
    tx_val = transform_preprocessor(x_val, state)

    print(f"\n{'Model':15s} {'Parameters':35s} {'Threshold':>10s} {'F1':>7s}")
    for name, params, w in train_all_methods(y_tr, tx_tr):
        threshold, f1 = best_threshold(y_val, tx_val @ w)
        print(f"{name:15s} {params:35s} {threshold:10.4f} {f1:7.4f}")


if __name__ == "__main__":
    main()
