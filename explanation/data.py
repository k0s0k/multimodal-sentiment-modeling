"""Audited aligned data adapters. Special samples are unavailable before freeze."""

from pathlib import Path
import json
import hashlib
import numpy as np
from sentiment.data import (
    load_official,
    subset_data as _subset,
    RestrictedUnpickler,
    sanitize_tokens,
    observed_masks,
    file_sha256,
    fit_scaler,
    apply_scaler,
    build_structure,
)

ROOT = Path(__file__).resolve().parents[1]


def load_split(source, split):
    data = load_official(source, split)
    with open(source, "rb") as f:
        original = RestrictedUnpickler(f).load()[split]
    data["raw_text"] = [str(x) for x in original["raw_text"]]
    data["official_indices"] = list(range(len(data["ids"])))
    data["official_split"] = split
    return data


def subset(data, indices, name=None):
    out = _subset(data, indices, name)
    out["raw_text"] = [data["raw_text"][i] for i in indices]
    out["official_indices"] = [
        data.get("official_indices", list(range(len(data["ids"]))))[i] for i in indices
    ]
    out["official_split"] = data.get("official_split", data["split"])
    return out


def load_training_data(
    source,
    train_indices=None,
    valid_indices=None,
    subset_name="valid_tune",
    validation_split="valid",
):
    if validation_split not in ("valid", "train"):
        raise ValueError(
            "Only official valid or held training folds may guide this operation"
        )
    train = load_split(source, "train")
    valid = train if validation_split == "train" else load_split(source, "valid")
    if validation_split == "train" and (train_indices is None or valid_indices is None):
        raise ValueError("OOF must explicitly specify both training and held indices")
    if train_indices is not None:
        train = subset(train, train_indices, "train")
    if valid_indices is None:
        spec = json.loads(
            (ROOT / "configs/validation_split_v1.json").read_text("utf-8")
        )
        valid_indices = spec["splits"][subset_name]["indices_in_official_valid"]
    valid = subset(valid, valid_indices, subset_name)
    if set(train["ids"]) & set(valid["ids"]):
        raise ValueError("Train/validation sample overlap")
    if set(train["video_ids"]) & set(valid["video_ids"]):
        raise ValueError("Train/validation video overlap")
    return (train, valid)


def load_special(directory, freeze_path):
    freeze_path = Path(freeze_path)
    if not freeze_path.is_file():
        raise RuntimeError(
            "Special emotion inference requires a finalized selection freeze"
        )
    freeze = json.loads(freeze_path.read_text("utf-8"))
    if freeze.get("status") != "frozen":
        raise RuntimeError("Selection has not been frozen")
    if freeze.get("numeric_profile") != "q3_fp32_v1":
        raise ValueError("Unsupported special prediction numerical profile")
    directory = Path(directory)
    paths = sorted(directory.glob("*.pkl"))
    if [p.stem for p in paths] != [f"{i:02}" for i in range(1, 21)]:
        raise ValueError("Exactly the 20 attachment4 aligned samples are required")
    rows = []
    manifest = json.loads(
        (directory.parent / "special_staging_manifest.json").read_text("utf-8")
    )
    expected = {r["sample_id"]: r["pickle_sha256"] for r in manifest["samples"]}
    for p in paths:
        if file_sha256(p) != expected.get(p.stem):
            raise ValueError(
                "Special sample SHA mismatch against official staging manifest"
            )
        with p.open("rb") as f:
            s = RestrictedUnpickler(f).load()
        if any(
            (
                k in s
                for k in ("classification_labels", "regression_labels", "test", "train")
            )
        ):
            raise ValueError("Unexpected labels or wrapped special schema")
        if str(s["id"]) != p.stem:
            raise ValueError("Filename/sample ID mismatch")
        rows.append(s)
    tokens = sanitize_tokens(np.stack([s["text_bert"] for s in rows]))
    audio = np.stack([s["audio"] for s in rows]).astype(np.float32)
    vision = np.stack([s["vision"] for s in rows]).astype(np.float32)
    observed_masks(tokens, audio, vision)
    hashes = {p.name: file_sha256(p) for p in paths}
    return dict(
        tokens=tokens,
        audio=audio,
        vision=vision,
        ids=[p.stem for p in paths],
        video_ids=[p.stem for p in paths],
        raw_text=[str(s["raw_text"]) for s in rows],
        split="special",
        official_split="special",
        official_indices=list(range(20)),
        source=str(directory),
        source_sha256=hashlib.sha256(
            json.dumps(hashes, sort_keys=True).encode()
        ).hexdigest(),
        source_hashes=hashes,
        freeze_sha256=file_sha256(freeze_path),
    )
