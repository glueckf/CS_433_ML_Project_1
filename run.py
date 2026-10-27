"""Reproduce our best AIcrowd submission from scratch.

The project description requires this script to produce exactly the csv file
we submitted. That means it has to be deterministic end to end: every random
draw is seeded from ``mlproject.config.RANDOM_SEED``, nothing depends on the
wall clock or on filesystem ordering, and the dependency versions are pinned
in requirements.txt.

Usage:
    python run.py

Prerequisites:
    The three raw csv files must be present in ``data/``. See README.md, and
    verify them with ``make check-data``.
"""

import numpy as np

from helpers import create_csv_submission, load_csv_data
from implementations import ridge_regression
from mlproject.config import DATA_DIR, RANDOM_SEED, SUBMISSIONS_DIR
from mlproject.metrics import accuracy, best_threshold, precision_recall_f1
from mlproject.preprocessing import (
    fit_preprocessor,
    stratified_split,
    transform_preprocessor,
)


def main() -> None:
    """Train the final model and write the submission csv."""
    np.random.seed(RANDOM_SEED)

    # 1. Load the raw data.                  -> mlproject.data
    # 2. Clean and impute.                   -> mlproject.preprocessing
    # 3. Build features.                     -> mlproject.features
    # 4. Fit the final model on all of it.   -> implementations
    # 5. Predict, threshold, map to {-1, 1}. -> mlproject.metrics, mlproject.data
    # 6. Write the csv.                      -> helpers.create_csv_submission
    #
    # Keep this function a readable sequence of calls into mlproject. If a step
    # here grows past a couple of lines, it belongs in a module instead.
    #
    # The last two lines, once steps 1 to 5 produce test_ids and y_pred, are:
    #
    #     output_path = SUBMISSIONS_DIR / "submission.csv"
    #     create_csv_submission(test_ids, y_pred, str(output_path))
    #
    # *** YOUR CODE HERE ***

    # 1. Load the dataset.
    print("\n===== STEP 1/6: LOADING DATA =====")
    print("Reading the training data and the test data...")
    x_train, x_test, y_train, train_ids, test_ids = load_csv_data(str(DATA_DIR))
    print("\n===== STEP 2/6: SPLITTING TRAINING DATA =====")
    print("Creating a training set and a validation set...")
    # 2. Create a stratified training/validation split.
    x_tr, y_tr, x_val, y_val = stratified_split(
        x_train,
        y_train,
        validation_fraction=0.2,
        seed=RANDOM_SEED,
    )

    # ==================================================
    # DEVELOPMENT PHASE: PREPROCESSING
    # ==================================================
    # Fit preprocessing rules on x_tr only.
    # Apply the same rules to x_val to avoid data leakage.
    #
    # The best preprocessing parameters, model
    # hyperparameters and classification threshold
    # will be selected using validation performance.

    use_one_hot = True
    use_bias = True
    use_special_codes = True
    use_augmented_input = False
    ridge_lambda = 1e-5

    preprocessing_options = {
        "csv_path": str(DATA_DIR / "x_train.csv"),
        "threshold": 0.80,  # To be optimized
        "use_one_hot": use_one_hot,
        "use_bias": use_bias,
        "use_special_codes": use_special_codes,
        "use_augmented_input": use_augmented_input,
    }

    print("\n===== STEP 3/6: PREPROCESSING VALIDATION DATA =====")

    x_tr_processed, preprocessing_state = fit_preprocessor(
        x_tr,
        **preprocessing_options,
    )

    x_val_processed = transform_preprocessor(
        x_val,
        preprocessing_state,
    )

    print("Validation preprocessing completed.")
    print(f"Samples used to train the validation model: {len(x_tr_processed)}")
    print(f"Validation samples: {len(x_val_processed)}")
    print(f"Features used by the model: {x_tr_processed.shape[1]}")

    w, train_loss = ridge_regression(y_tr, x_tr_processed, ridge_lambda)
    scores = x_val_processed @ w
    classification_threshold, best_f1 = best_threshold(y_val, scores)
    y_val_pred = np.where(scores >= classification_threshold, 1, -1)
    precision, recall, f1 = precision_recall_f1(y_val, y_val_pred)
    metrics = {
        "accuracy": accuracy(y_val, y_val_pred),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

    print("\n===== STEP 4/6: EVALUATING VALIDATION PERFORMANCE =====")
    print(f"Model: ridge regression (lambda={ridge_lambda:g})")
    print(
        "Best classification threshold found on validation data:",
        f"{classification_threshold:.4f}",
    )
    print(f"Validation accuracy:  {metrics['accuracy']:.4f}")
    print(f"Validation precision: {metrics['precision']:.4f}")
    print(f"Validation recall:    {metrics['recall']:.4f}")
    print(f"Validation F1-score:  {metrics['f1']:.4f}")

    # Refit preprocessing on all labeled training data after validation
    # decisions are complete.
    print("\n===== STEP 5/6: TRAINING FINAL MODEL =====")
    print("Preprocessing the complete training dataset...")
    x_train_processed, final_preprocessing_state = fit_preprocessor(
        x_train,
        **preprocessing_options,
    )
    print(f"Complete training samples: {len(x_train_processed)}")
    print(f"Features used by the final model: {x_train_processed.shape[1]}")

    print("Applying the same preprocessing rules to the test data...")
    x_test_processed = transform_preprocessor(
        x_test,
        final_preprocessing_state,
    )
    print(f"Test samples: {len(x_test_processed)}")
    print(f"Test features: {x_test_processed.shape[1]}")

    print(f"Training the final ridge regression model (lambda={ridge_lambda:g})...")
    final_w, final_train_loss = ridge_regression(
        y_train,
        x_train_processed,
        ridge_lambda,
    )
    test_scores = x_test_processed @ final_w
    print(f"Training loss on the complete training data: {final_train_loss:.4f}")
    print(f"Test prediction scores generated: {len(test_scores)}")

    y_test_pred = np.where(test_scores >= classification_threshold, 1, -1)
    negative_count = int(np.sum(y_test_pred == -1))
    positive_count = int(np.sum(y_test_pred == 1))
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    submission_path = SUBMISSIONS_DIR / "submission.csv"
    create_csv_submission(test_ids, y_test_pred, str(submission_path))
    print("\n===== STEP 6/6: CREATING SUBMISSION FILE =====")
    print(f"Total predictions: {len(y_test_pred)}")
    print(f"Predictions labeled -1: {negative_count}")
    print(f"Predictions labeled  1: {positive_count}")
    print("Submission file saved to:")
    print(submission_path)


if __name__ == "__main__":
    main()
