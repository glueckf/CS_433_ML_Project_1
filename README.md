# CS-433 Machine Learning — Project 1: Predicting coronary heart disease

Binary classification of coronary heart disease (MICHD) from the
[BRFSS 2015](https://www.cdc.gov/brfss/annual_data/annual_2015.html) health
survey, implemented with NumPy and the Python standard library only.

**Final model:** ridge regression (λ = 1e-5) on 219 preprocessed features, with
a decision threshold tuned for F1 on a stratified validation split.

|                                         | Accuracy | F1    |
| --------------------------------------- | -------- | ----- |
| Local validation (20% stratified split) | 0.856    | 0.422 |
| AIcrowd test set                        | 0.858    | 0.425 |

## Reproducing the submission

Tested with Python 3.11 (NumPy 2.3) and Python 3.9 (NumPy 1.23).

1. Download `x_train.csv`, `x_test.csv` and `y_train.csv` from the
   [AIcrowd challenge](https://www.aicrowd.com/challenges/epfl-machine-learning-project-1)
   into `data/`, keeping the filenames. The data is not part of this
   repository.
2. Install the dependencies and verify the data:

   ```bash
   make setup && source .venv/bin/activate
   make check-data      # compares the files against data/MANIFEST.sha256
   ```

3. Run:

   ```bash
   python run.py
   ```

   This writes `submissions/submission.csv` in about 2 minutes. All randomness
   is seeded (`RANDOM_SEED = 42` in `mlproject/config.py`), so repeated runs
   produce a byte-identical file.

`python compare_methods.py` reproduces the comparison of all six methods and the
ablation study from the report (about 5 minutes), and `python EDA.py`
regenerates the exploratory statistics and the missing-value figure in
`reports/`.

## Pipeline

1. Replace BRFSS special codes ("don't know", "refused", …) with missing
   values, per feature, following the codebook
   (`mlproject/special_code_brfss2015.py`).
2. Drop features with more than 80% missing values, impute the rest with the
   training median, and drop constant and duplicate features.
3. One-hot encode categorical features, standardize, and add a bias column.
4. Fit ridge regression, choose the decision threshold that maximizes F1 on
   the validation split, refit on all training data, and predict the test set.

All preprocessing statistics are fitted on training data only and reused for
the validation and test data.

## Repository layout

| Path                                  | Contents                                                    |
| ------------------------------------- | ----------------------------------------------------------- |
| `implementations.py`                  | The six required methods                                    |
| `run.py`                              | Reproduces the final submission                             |
| `compare_methods.py`                  | Reproduces the method comparison table and ablation study   |
| `EDA.py`                              | Exploratory data analysis and report figure                 |
| `mlproject/preprocessing.py`          | Cleaning, imputation, encoding, standardization             |
| `mlproject/special_code_brfss2015.py` | Per-feature special codes from the BRFSS codebook           |
| `mlproject/metrics.py`                | Accuracy, precision, recall, F1, threshold search           |
| `mlproject/validation.py`             | k-fold cross-validation and grid search                     |
| `mlproject/plots.py`                  | Figure helpers (the only module that uses matplotlib)       |
| `tests/`                              | Unit tests, including checks of the course rules            |
| `helpers.py`                          | Provided by the course, unmodified                          |

Run the tests with `make check`.

## Team

Finn Glück · Ruimin Zhang · Matthieu Willy Schmid

## Use of AI tools

We used Claude Code to set up the repository tooling (CI, linting, Makefile,
contributor workflow). During development, we used Claude Code and GitHub
Copilot to review some pull requests and spot inconsistencies, and later to
clean up unused code and redundant comments. Claude Code also helped write the unit tests and the method-comparison script. All implementations were written by the team.
