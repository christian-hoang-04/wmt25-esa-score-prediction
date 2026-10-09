"""Compatibility entrypoint for the optional frontier judge."""

from wmt25_esa.run_frontier_judge import *  # noqa: F403
from wmt25_esa.run_frontier_judge import main as _main

if __name__ == "__main__":
    _main()
