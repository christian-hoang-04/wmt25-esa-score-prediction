"""Compatibility entrypoint for the data downloader."""

from wmt25_esa.download_data import *  # noqa: F403
from wmt25_esa.download_data import main as _main

if __name__ == "__main__":
    _main()
