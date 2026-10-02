"""Rebuild the selected three-member model from official attachment 2."""

from pathlib import Path
import argparse
import gzip
import json
import shutil
import subprocess
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from prepare_training_data import stage
from sentiment.cache import FrozenBertEncoder, build_cache
from sentiment.data import file_sha256, load_official, subset_data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--bert", type=Path, default=ROOT / "models/bert-base-uncased")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Use a new training directory")
    expected = json.loads(
        (ROOT / "configs/validation_split_v1.json").read_text("utf-8")
    )
    if file_sha256(args.source) != expected["source_sha256"]:
        raise ValueError("Official aligned_50.pkl checksum mismatch")
    args.output.mkdir(parents=True)
    provenance = stage(args.source, args.output / "data")
    source = args.output / "data/compact_aligned.pkl"
    with gzip.open(
        args.output / "data/compact_aligned.pkl.gz", "rb"
    ) as stream, source.open("xb") as target:
        shutil.copyfileobj(stream, target)
    if file_sha256(source) != provenance["working_copy_sha256"]:
        raise ValueError("Working copy decompression checksum mismatch")
    encoder = FrozenBertEncoder(
        str(args.bert), device=args.device, local_files_only=True
    )
    for split, bank in (("train", "train-block"), ("valid", "core")):
        data = load_official(source, split)
        data["original_source_sha256"] = provenance["original_source_sha256"]
        if split == "valid":
            entry = expected["splits"]["valid_tune"]
            data = subset_data(data, entry["indices_in_official_valid"], "valid_tune")
            if data["ids"] != entry["ids"]:
                raise ValueError("Validation group identity mismatch")
        build_cache(
            data,
            args.output / "cache",
            bank,
            encoder,
            seed=31,
            views=12,
            batch_size=128,
            storage_dtype="float16",
        )
    del encoder
    for member in range(1, 4):
        command = [
            sys.executable,
            "-m",
            "sentiment.train",
            "--config",
            str(ROOT / f"configs/member_{member}.json"),
            "--source",
            str(source),
            "--cache",
            str(args.output / "cache"),
            "--output",
            str(args.output / f"member_{member}"),
            "--device",
            args.device,
            "--valid-bank",
            "core",
        ]
        subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
