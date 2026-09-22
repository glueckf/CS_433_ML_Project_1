"""Loading the raw dataset and converting between label conventions.

The raw csv files are never committed. Download them from AIcrowd into
``data/`` and verify them with ``make check-data``. See README.md.
"""

import numpy as np

from mlproject.config import RANDOM_SEED


def load_raw() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load the raw training and test data from ``data/``.

    Thin wrapper around the course-provided ``helpers.load_csv_data``, which we
    keep byte-identical to the upstream file. Parsing 227 MB of csv with
    ``np.genfromtxt`` takes a while, so call this once and cache the arrays.

    Returns:
        A tuple ``(x_train, x_test, y_train, train_ids, test_ids)``. Feature
        matrices have the ``Id`` column already stripped, and ``y_train`` uses
        the ``{-1, 1}`` convention.
    """
    # *** YOUR CODE HERE ***
    raise NotImplementedError


def to_zero_one(y: np.ndarray) -> np.ndarray:
    """Convert labels from the ``{-1, 1}`` convention to ``{0, 1}``.

    Args:
        y: Labels in ``{-1, 1}``, shape ``(N,)``.

    Returns:
        Labels in ``{0, 1}``, shape ``(N,)``.
    """
    # *** YOUR CODE HERE ***
    raise NotImplementedError


def to_plus_minus_one(y: np.ndarray) -> np.ndarray:
    """Convert labels from the ``{0, 1}`` convention back to ``{-1, 1}``.

    The submission csv must use ``{-1, 1}``; ``helpers.create_csv_submission``
    raises a ``ValueError`` on anything else.

    Args:
        y: Labels in ``{0, 1}``, shape ``(N,)``.

    Returns:
        Labels in ``{-1, 1}``, shape ``(N,)``.
    """
    # *** YOUR CODE HERE ***
    raise NotImplementedError


def train_validation_split(
    x: np.ndarray,
    y: np.ndarray,
    validation_fraction: float,
    seed: int = RANDOM_SEED,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split the data into a training and a validation set.

    Only about 8.8 percent of the labels are positive, so the split should be
    stratified: an unstratified split makes validation scores noisy on a
    minority class this small.

    Args:
        x: Feature matrix, shape ``(N, D)``.
        y: Labels, shape ``(N,)``.
        validation_fraction: Fraction of rows held out for validation.
        seed: Seed for the shuffle, so the split is reproducible.

    Returns:
        A tuple ``(x_tr, y_tr, x_val, y_val)``.
    """
    # *** YOUR CODE HERE ***
    raise NotImplementedError
