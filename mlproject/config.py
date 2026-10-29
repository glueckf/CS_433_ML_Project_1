"""Paths, constants and the global random seed.

Anything that would otherwise be a magic number scattered across modules
belongs here, so that a single edit changes the whole pipeline consistently.
"""

from pathlib import Path

# Repo root, resolved relative to this file so it works from any cwd.
ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "data"

# These two are created on demand by whatever writes into them, so the repo
# carries no empty placeholder directories.
SUBMISSIONS_DIR = ROOT / "submissions"
FIGURES_DIR = ROOT / "reports" / "figures"

# Every source of randomness in the pipeline derives from this seed. The
# project description requires run.py to reproduce our submission exactly, so
# nothing may depend on an unseeded generator, on dict ordering, or on the
# wall clock.
RANDOM_SEED = 42

# Fraction of the training set held out for local validation.
VALIDATION_FRACTION = 0.2

# Number of folds for k-fold cross-validation.
N_FOLDS = 5
