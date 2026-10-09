"""Compatibility entrypoint for the unseen-evaluator experiment."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from wmt25_esa.analysis.run_unseen_evaluator_test import *  # noqa: F403, E402
from wmt25_esa.analysis.run_unseen_evaluator_test import main as _main  # noqa: E402

if __name__ == "__main__":
    _main()
