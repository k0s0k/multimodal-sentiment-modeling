"""Explicit held-out boundary checks; unlike assert, these survive python -O."""

from pathlib import Path
import hashlib, json


def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def verify_freeze(path, checkpoints=None, data_source_sha256=None):
    if not path:
        raise ValueError("Held-out access requires a frozen protocol file")
    record = json.loads(Path(path).read_text(encoding="utf-8"))
    if record.get("status") != "frozen":
        raise ValueError("Protocol is not frozen")
    if checkpoints is not None:
        hashes = [file_sha256(x) for x in checkpoints]
        if not hashes or len(hashes) != len(set(hashes)):
            raise ValueError("Empty or duplicate checkpoint list")
        if not set(hashes) <= set(record.get("allowed_checkpoint_sha256", [])):
            raise ValueError("Checkpoint not registered before held-out evaluation")
        combinations = [
            sorted(p["checkpoint_sha256"])
            for p in record.get("allowed_predictors", [])
            if p.get("weights") == "equal"
        ]
        if sorted(hashes) not in combinations:
            raise ValueError("Exact predictor combination was not frozen")
    if data_source_sha256 is not None and data_source_sha256 not in record.get(
        "allowed_cache_source_sha256", []
    ):
        raise ValueError("Held-out source hash not registered before evaluation")
    return record
