"""Download and validate the official WMT25 human evaluation JSONL."""

from __future__ import annotations

import argparse
import json
import shutil
import urllib.error
import urllib.request
from pathlib import Path


DATA_URL = (
    "https://github.com/wmt-conference/wmt25-general-mt/raw/refs/heads/main/"
    "data/wmt25-genmt-humeval.jsonl"
)
LFS_MEDIA_URL = (
    "https://media.githubusercontent.com/media/wmt-conference/"
    "wmt25-general-mt/main/data/wmt25-genmt-humeval.jsonl"
)
MIN_DATA_BYTES = 10_000_000


def validate_jsonl(path: Path) -> None:
    """Reject HTML, Git LFS pointer files, and malformed first JSONL records."""
    if not path.is_file():
        raise FileNotFoundError(path)
    size = path.stat().st_size
    if size < MIN_DATA_BYTES:
        head = path.read_bytes()[:500].decode("utf-8", errors="replace")
        if head.startswith("version https://git-lfs.github.com/spec/v1"):
            raise ValueError(f"{path} is a Git LFS pointer, not the dataset")
        raise ValueError(f"Dataset is unexpectedly small ({size} bytes): {head!r}")

    with path.open("rb") as stream:
        first_line = stream.readline()
    try:
        record = json.loads(first_line)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"First line is not JSON: {path}") from exc
    required = {"doc_id", "src_text", "tgt_text", "scores"}
    missing = required.difference(record)
    if missing:
        raise ValueError(f"First JSON record is missing fields: {sorted(missing)}")
    if not isinstance(record["scores"], dict) or not isinstance(record["tgt_text"], dict):
        raise ValueError("First JSON record has unexpected scores/tgt_text types")


def _download(url: str, destination: Path) -> None:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "wmt25-esa-score-experiment/1.0"},
    )
    temp_path = destination.with_suffix(destination.suffix + ".part")
    try:
        with urllib.request.urlopen(request, timeout=90) as response, temp_path.open("wb") as out:
            if response.status != 200:
                raise urllib.error.HTTPError(
                    url, response.status, "unexpected HTTP status", response.headers, None
                )
            print(
                f"Downloading {url} ({response.headers.get('Content-Length', 'unknown')} bytes)",
                flush=True,
            )
            while chunk := response.read(1024 * 1024):
                out.write(chunk)
        try:
            validate_jsonl(temp_path)
        except ValueError as exc:
            temp_path.unlink(missing_ok=True)
            raise ValueError(f"Downloaded response failed data validation: {exc}") from exc
        temp_path.replace(destination)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise


def download_data(output: Path, local_input: Path | None = None) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    if local_input is not None:
        local_input = local_input.expanduser().resolve()
        validate_jsonl(local_input)
        if local_input != output.resolve():
            shutil.copyfile(local_input, output)
        validate_jsonl(output)
        print(f"Using validated local dataset: {output.resolve()} ({output.stat().st_size} bytes)")
        return output

    if output.exists():
        try:
            validate_jsonl(output)
            print(f"Using existing validated dataset: {output.resolve()}")
            return output
        except ValueError:
            output.unlink()

    try:
        _download(DATA_URL, output)
    except Exception as first_error:
        print(f"Primary URL failed validation/download: {first_error}", flush=True)
        _download(LFS_MEDIA_URL, output)
    validate_jsonl(output)
    print(f"Dataset ready: {output.resolve()} ({output.stat().st_size} bytes)")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/raw/wmt25-genmt-humeval.jsonl"),
        help="Destination path for the validated JSONL file.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        help="Optional local copy of the JSONL; bypasses network download.",
    )
    args = parser.parse_args()
    download_data(args.output, args.input)


if __name__ == "__main__":
    main()
