"""Compatibility entrypoint for model training."""

from wmt25_esa.train import *  # noqa: F403
from wmt25_esa.train import main as _main

if __name__ == "__main__":
    _main()
