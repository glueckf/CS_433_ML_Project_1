# CS-433 Machine Learning — Project 1

Predicting the risk of coronary heart disease (MICHD) from the
[BRFSS 2015](https://www.cdc.gov/brfss/annual_data/annual_2015.html) health
survey, using only NumPy and the Python standard library.

The repository holds the six required method implementations, the full data
pipeline, and the code that reproduces our competition submission.

- Course: [CS-433 Machine Learning](https://www.epfl.ch/labs/mlo/machine-learning-cs-433/)
- Competition: [EPFL ML Project 1 on AIcrowd](https://www.aicrowd.com/challenges/epfl-machine-learning-project-1)
- Assignment: [project1_description.pdf](project1_description.pdf)

## Quick start

```bash
make setup                    # create .venv, install dependencies, install git hooks
source .venv/bin/activate
make check                    # lint and tests, the same checks CI runs
```

Download `x_train.csv`, `x_test.csv` and `y_train.csv` from AIcrowd into
`data/` (keep the filenames as they are), then verify them:

```bash
make check-data
python run.py                 # regenerate the submission csv
```

## Layout

| Path                | Contents                                                  |
| ------------------- | --------------------------------------------------------- |
| `implementations.py`| The six graded methods. Self-contained, NumPy only.        |
| `run.py`            | Reproduces our best submission end to end.                 |
| `mlproject/`        | Everything else: loading, preprocessing, features, metrics, cross-validation, plots. |
| `scripts/`          | Entry points that produce artefacts, such as report figures. |
| `notebooks/`        | Exploration, one folder per person.                        |
| `tests/`            | Local tests, plus the checks that enforce the course rules. |
| `helpers.py`        | Provided by the course, kept unmodified.                   |

## Contributing

Read [WORKFLOW.md](WORKFLOW.md) first. It covers setup, the branch and review
process, and the handful of project rules that CI enforces.

## Team

Finn Glück · <!-- teammate --> · <!-- teammate -->
