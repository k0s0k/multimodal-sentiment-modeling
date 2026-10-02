"""Fetch the fixed public, frozen BERT dependency and verify its SHA256."""

from pathlib import Path
import argparse
import hashlib
import json
import shutil

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--local-file",
        type=Path,
        help="Copy an existing matching public weight instead of downloading",
    )
    args = parser.parse_args()
    directory = ROOT / "models/bert-base-uncased"
    config = json.loads((directory / "download.json").read_text("utf-8"))
    destination = directory / config["filename"]
    if destination.exists():
        if sha(destination) != config["sha256"]:
            raise ValueError(
                "Existing public BERT weight differs from the fixed revision"
            )
        print("Public BERT verified")
        return
    source = args.local_file
    if source is None:
        from huggingface_hub import hf_hub_download

        source = Path(
            hf_hub_download(
                repo_id=config["repo"],
                revision=config["revision"],
                filename=config["filename"],
            )
        )
    if sha(source) != config["sha256"]:
        raise ValueError("Public weight checksum differs from the fixed revision")
    shutil.copyfile(source, destination)
    print("Public BERT ready")


if __name__ == "__main__":
    main()
