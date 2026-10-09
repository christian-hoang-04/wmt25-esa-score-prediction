"""Compatibility entrypoint for experiment evaluation."""

from wmt25_esa.evaluate import *  # noqa: F403
from wmt25_esa.evaluate import main as _main

if __name__ == "__main__":
    _main()
