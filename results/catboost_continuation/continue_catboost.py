"""Compatibility entrypoint for CatBoost continuation analysis."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from wmt25_esa.analysis.continue_catboost import *  # noqa: F403, E402
from wmt25_esa.analysis.continue_catboost import main as _main  # noqa: E402

if __name__ == "__main__":
    _main()
