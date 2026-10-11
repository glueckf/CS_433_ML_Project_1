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

from mlproject.config import RANDOM_SEED, DATA_DIR
from helpers import load_csv_data
from mlproject.preprocessing import (
    stratified_split,
    preprocess_data,
    evaluate_baseline,
)
from implementations import least_squares




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
    print("Loading dataset...")
    x_train, x_test, y_train, train_ids, test_ids = load_csv_data(
        str(DATA_DIR)
    )
    print("\n===== START PREPROCESS =====")
    # 2. Create a stratified training/validation split.
    x_tr, y_tr, x_val, y_val = stratified_split(
        x_train,
        y_train,
        validation_fraction=0.2,
        seed=RANDOM_SEED,
    )

   # 3. Preprocess training and validation data. #ratio will be optimize
    x_tr_processed, x_val_processed = preprocess_data(x_tr,
    x_val,
    csv_path=str(DATA_DIR / "x_train.csv"),
    threshold=0.97,
    )
    print("Preprocessing completed.")
    print("Training shape:", x_tr_processed.shape)
    print("Validation shape:", x_val_processed.shape)

    print("\n===== END PREPROCESS =====")

    #for testing preprocess you can modify/change/delete it
    w, train_loss = least_squares(y_tr, x_tr_processed)
    print(f"Training loss: {train_loss:.6f}")
    scores = x_val_processed @ w
    classification_threshold = -0.6 #to test for preprocess
    metrics = evaluate_baseline(y_val,scores,threshold=classification_threshold)
    print("\n===== RESULTS =====")
    print(f"Threshold: {classification_threshold:.4f}")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-score:  {metrics['f1']:.4f}")
    


if __name__ == "__main__":
    main()
