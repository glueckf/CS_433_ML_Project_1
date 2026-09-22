"""Pytest configuration.

Its mere presence at the repository root puts the root on ``sys.path``, so
tests can ``import implementations`` and ``import mlproject`` exactly the way
the graders do, with no install step.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
