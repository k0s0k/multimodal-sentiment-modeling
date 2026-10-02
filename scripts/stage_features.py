"""Prepare a fresh Q1 extraction workspace from the 100 official videos."""

from pathlib import Path
import argparse
import hashlib
import json
import shutil

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--official-root",
        type=Path,
        required=True,
        help="Root of the official E_data folder",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Use a new extraction directory")
    records = [
        json.loads(line)
        for line in (ROOT / "q1/data/manifest.jsonl").read_text("utf-8").splitlines()
    ]
    sources = []
    for row in records:
        relative = Path(row["source_relpath"])
        if relative.parts[0] == "E_data":
            relative = Path(*relative.parts[1:])
        source = args.official_root / relative
        if hashlib.sha256(source.read_bytes()).hexdigest() != row["source_sha256"]:
            raise ValueError("Official video identity mismatch: " + row["sample_key"])
        sources.append(source)
    for name in ("feature_extraction", "tools", "review"):
        shutil.copytree(ROOT / "q1" / name, args.output / name)
    for name in (
        "reproduce.sh",
        "setup_openface.sh",
        "setup_alignment_dependencies.sh",
        "config.json",
    ):
        shutil.copyfile(ROOT / "q1" / name, args.output / name)
    (args.output / "data/media").mkdir(parents=True)
    for name in ("manifest.jsonl", "labels.csv"):
        shutil.copyfile(ROOT / "q1/data" / name, args.output / "data" / name)
    for source, row in zip(sources, records):
        shutil.copyfile(source, args.output / "data" / row["media_path"])
    print(
        "100 official videos verified and staged; run reproduce.sh in the new directory."
    )


if __name__ == "__main__":
    main()
