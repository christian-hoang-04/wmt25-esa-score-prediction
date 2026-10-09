"""Compatibility entrypoint for data preparation."""

from wmt25_esa.prepare_data import *  # noqa: F403
from wmt25_esa.prepare_data import main as _main

if __name__ == "__main__":
    _main()
