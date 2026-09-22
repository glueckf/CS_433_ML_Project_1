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

from mlproject.config import RANDOM_SEED


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
    raise NotImplementedError


if __name__ == "__main__":
    main()
