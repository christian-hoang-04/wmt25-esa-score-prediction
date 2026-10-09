"""Compatibility entrypoint for building the Kaggle package."""

from wmt25_esa.build_kaggle_kernel import *  # noqa: F403
from wmt25_esa.build_kaggle_kernel import build as _build

if __name__ == "__main__":
    _build()
