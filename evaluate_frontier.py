"""Compatibility entrypoint for frontier-judge evaluation."""

from wmt25_esa.evaluate_frontier import *  # noqa: F403
from wmt25_esa.evaluate_frontier import main as _main

if __name__ == "__main__":
    _main()
