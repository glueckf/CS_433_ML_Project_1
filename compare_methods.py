"""Reproduce the method comparison table and the ablation study of the report.

Trains each of the six methods from implementations.py on the same stratified
training split and preprocessing as run.py, tunes the decision threshold for F1
on the validation split, and prints the results. It then re-evaluates the final
ridge model with one preprocessing step changed at a time.

Usage:
    python compare_methods.py
"""

import numpy as np

import implementations as impl
from helpers import load_csv_data
from mlproject.config import DATA_DIR, RANDOM_SEED, VALIDATION_FRACTION
from mlproject.metrics import accuracy, best_threshold, precision_recall_f1
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

RIDGE_LAMBDA = 1e-5

# Each ablation changes one preprocessing option of the final pipeline.
ABLATIONS = [
    ("Full pipeline", {}),
    ("No special-code replacement", {"use_special_codes": False}),
    ("No one-hot encoding", {"use_one_hot": False}),
    ("Missingness cutoff 97%", {"threshold": 0.97}),
    ("+ squared quantitative features", {"use_augmented_input": True}),
]


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
        ("Ridge Reg.", "lambda=1e-5", impl.ridge_regression(y, tx, RIDGE_LAMBDA)[0]),
    ]


def compare_methods(x_tr, y_tr, x_val, y_val):
    """Print the validation threshold and F1 of all six methods.

    Args:
        x_tr: Raw training features.
        y_tr: Training labels in {-1, 1}.
        x_val: Raw validation features.
        y_val: Validation labels in {-1, 1}.
    """
    tx_tr, state = fit_preprocessor(x_tr, **PREPROCESSING_OPTIONS)
    tx_val = transform_preprocessor(x_val, state)

    print(f"\n{'Model':15s} {'Parameters':35s} {'Threshold':>10s} {'F1':>7s}")
    for name, params, w in train_all_methods(y_tr, tx_tr):
        threshold, f1 = best_threshold(y_val, tx_val @ w)
        print(f"{name:15s} {params:35s} {threshold:10.4f} {f1:7.4f}")


def ablation_study(x_tr, y_tr, x_val, y_val):
    """Print the ridge model's validation scores with one step changed at a time.

    The first row is the full pipeline; the second keeps it but uses the
    default threshold 0 instead of the F1-tuned one.

    Args:
        x_tr: Raw training features.
        y_tr: Training labels in {-1, 1}.
        x_val: Raw validation features.
        y_val: Validation labels in {-1, 1}.
    """
    print(
        f"\n{'Ablation':35s} {'Features':>8s} {'Threshold':>10s} {'F1':>7s} {'Acc':>7s}"
    )
    for name, change in ABLATIONS:
        options = {**PREPROCESSING_OPTIONS, **change}
        tx_tr, state = fit_preprocessor(x_tr, **options)
        tx_val = transform_preprocessor(x_val, state)
        scores = tx_val @ impl.ridge_regression(y_tr, tx_tr, RIDGE_LAMBDA)[0]

        thresholds = [best_threshold(y_val, scores)[0]]
        if not change:
            thresholds.append(0.0)
        for threshold in thresholds:
            y_pred = np.where(scores >= threshold, 1, -1)
            f1 = precision_recall_f1(y_val, y_pred)[2]
            label = name if threshold != 0.0 else f"{name}, threshold 0"
            print(
                f"{label:35s} {tx_tr.shape[1]:8d} {threshold:10.4f} "
                f"{f1:7.4f} {accuracy(y_val, y_pred):7.4f}"
            )


def main() -> None:
    """Run the method comparison and the ablation study."""
    x_train, _, y_train, _, _ = load_csv_data(str(DATA_DIR))
    x_tr, y_tr, x_val, y_val = stratified_split(
        x_train, y_train, validation_fraction=VALIDATION_FRACTION, seed=RANDOM_SEED
    )
    compare_methods(x_tr, y_tr, x_val, y_val)
    ablation_study(x_tr, y_tr, x_val, y_val)


if __name__ == "__main__":
    main()
