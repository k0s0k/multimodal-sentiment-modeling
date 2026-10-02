"""Strict aligned-50 input and train-only scaling; no clean metadata in model inputs."""

from __future__ import annotations
import hashlib
import pickle
import re
from pathlib import Path
import numpy as np

LENGTH = 50
MODALITIES = ("text", "audio", "vision")
STUDENT_KEYS = ("text", "audio", "vision", "mask", "structure")


class RestrictedUnpickler(pickle.Unpickler):
    """Official NumPy/basic-container pickle format; no arbitrary class imports."""

    def find_class(self, module, name):
        try:
            import numpy._core.multiarray as ma
            import numpy._core.numeric as nu
        except ImportError:
            import numpy.core.multiarray as ma
            import numpy.core.numeric as nu
        allowed = {
            ("numpy", "ndarray"): np.ndarray,
            ("numpy", "dtype"): np.dtype,
            ("numpy", "asarray"): np.asarray,
            ("builtins", "set"): set,
            ("builtins", "frozenset"): frozenset,
            ("builtins", "slice"): slice,
            ("builtins", "complex"): complex,
        }
        for prefix in ("numpy.core", "numpy._core"):
            allowed[prefix + ".multiarray", "_reconstruct"] = ma._reconstruct
            allowed[prefix + ".multiarray", "scalar"] = ma.scalar
            allowed[prefix + ".numeric", "_frombuffer"] = nu._frombuffer
        if (module, name) not in allowed:
            raise pickle.UnpicklingError(f"Forbidden pickle global: {module}.{name}")
        return allowed[module, name]

    def persistent_load(self, pid):
        raise pickle.UnpicklingError("Persistent pickle references forbidden")


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def stable_seed(*parts):
    import json

    value = json.dumps(
        parts, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode()
    return int.from_bytes(hashlib.sha256(value).digest()[:8], "little")


def sanitize_tokens(tokens):
    a = np.asarray(tokens)
    if a.ndim != 3 or a.shape[1:] != (3, LENGTH):
        raise ValueError(f"tokens must have shape N,3,50, got {a.shape}")
    if not np.isfinite(a).all() or not np.equal(a, np.floor(a)).all():
        raise ValueError(
            "token channels must be finite integer values before conversion"
        )
    a = a.astype(np.int64, copy=True)
    if not ((a[:, 0] >= 0) & (a[:, 0] < 30522)).all():
        raise ValueError("token ID outside pinned BERT vocabulary")
    if not np.isin(a[:, 1], [0, 1]).all() or not np.isin(a[:, 2], [0, 1]).all():
        raise ValueError("attention/type channels must be binary")
    if ((a[:, 0] == 0) & (a[:, 1] == 1)).any():
        raise ValueError("PAD ID with visible attention is inconsistent")
    hidden = a[:, 1] == 0
    a[:, 0][hidden] = 0
    a[:, 2][hidden] = 0
    return a


def observed_masks(tokens, audio, vision):
    tokens = sanitize_tokens(tokens)
    result = [(tokens[:, 1] == 1) & ~np.isin(tokens[:, 0], [0, 101, 102])]
    for name, arr, width in (("audio", audio, 74), ("vision", vision, 35)):
        arr = np.asarray(arr)
        if arr.shape != (len(tokens), LENGTH, width) or not np.isfinite(arr).all():
            raise ValueError(f"{name} must be finite N,50,{width}, got {arr.shape}")
        result.append(np.any(arr != 0, axis=-1))
    return np.stack(result, axis=1)


def build_structure(mask):
    """Only damaged visibility: observed,left,right,invalid-run,position,coverage."""
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 3 or mask.shape[1:] != (3, LENGTH):
        raise ValueError("mask must be N,3,50")
    shape = mask.shape
    flat = mask.reshape(-1, LENGTH)
    pos = np.arange(LENGTH)[None, :]
    left = np.maximum.accumulate(np.where(flat, pos, -LENGTH), axis=1)
    right = np.minimum.accumulate(np.where(flat, pos, 2 * LENGTH)[:, ::-1], axis=1)[
        :, ::-1
    ]
    dl = np.minimum(pos - left, LENGTH - 1) / (LENGTH - 1)
    dr = np.minimum(right - pos, LENGTH - 1) / (LENGTH - 1)
    runs = np.zeros_like(flat, dtype=np.float32)
    for i, row in enumerate(flat):
        edges = np.diff(np.r_[False, ~row, False].astype(np.int8))
        for start, end in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
            runs[i, start:end] = (end - start) / LENGTH
    out = np.stack(
        [
            flat,
            dl,
            dr,
            runs,
            np.broadcast_to(pos / 49, flat.shape),
            np.broadcast_to(flat.mean(1)[:, None], flat.shape),
        ],
        axis=-1,
    )
    return out.reshape(*shape, 6).astype(np.float32)


def _one_official(path, split):
    with open(path, "rb") as f:
        source = RestrictedUnpickler(f).load()
    s = source[split]
    tokens = sanitize_tokens(s["text_bert"])
    audio = np.asarray(s["audio"], dtype=np.float32)
    vision = np.asarray(s["vision"], dtype=np.float32)
    observed_masks(tokens, audio, vision)
    ids = [
        str(x)
        for x in s.get("id", [f"{Path(path).name}#{i}" for i in range(len(tokens))])
    ]
    if len(ids) != len(tokens) or len(set(ids)) != len(ids):
        raise ValueError("IDs must be unique and match the sample axis")
    out = {
        "tokens": tokens,
        "audio": audio,
        "vision": vision,
        "ids": ids,
        "video_ids": [x.rsplit("$_$", 1)[0] if "$_$" in x else x for x in ids],
        "split": split,
        "source": str(Path(path).resolve()),
        "source_sha256": file_sha256(path),
    }
    if "classification_labels" in s or "regression_labels" in s:
        labels = np.asarray(s["classification_labels"])
        targets = np.asarray(s["regression_labels"])
        if labels.shape != (len(ids),) or targets.shape != (len(ids),):
            raise ValueError("label shape mismatch")
        expected = np.where(targets < 0, 0, np.where(targets > 0, 2, 1))
        if (
            not np.isin(labels, [0, 1, 2]).all()
            or not np.isfinite(targets).all()
            or (not (np.abs(targets) <= 3).all())
            or (not np.array_equal(labels, expected))
        ):
            raise ValueError(
                "label contract must be 0 negative, 1 zero, 2 positive; intensity [-3,3]"
            )
        out.update(labels=labels.astype(np.int64), targets=targets.astype(np.float32))
    return out


def load_official(path, split):
    """Load one official aligned pickle, or an explicitly selected special-test directory.

    Old text cache and raw_text are never returned. No unaligned fallback is allowed.
    """
    path = Path(path)
    if path.is_file():
        return _one_official(path, split)
    paths = sorted(
        path.glob("*.pkl"),
        key=lambda p: [
            int(t) if t.isdigit() else t for t in re.split("(\\d+)", p.name)
        ],
    )
    if not paths:
        raise FileNotFoundError(f"no official pickle files: {path}")
    items = [_one_official(p, split) for p in paths]
    if any(("labels" in item for item in items)):
        raise ValueError("directory input is reserved for unlabeled special-test files")
    return {
        "tokens": np.concatenate([x["tokens"] for x in items]),
        "audio": np.concatenate([x["audio"] for x in items]),
        "vision": np.concatenate([x["vision"] for x in items]),
        "ids": sum([x["ids"] for x in items], []),
        "video_ids": sum([x["video_ids"] for x in items], []),
        "split": split,
        "source": str(path.resolve()),
        "source_sha256": hashlib.sha256(
            "".join((x["source_sha256"] for x in items)).encode()
        ).hexdigest(),
    }


def subset_data(data, indices, name=None):
    indices = np.asarray(indices, dtype=np.int64)
    if (
        indices.ndim != 1
        or len(set(indices.tolist())) != len(indices)
        or np.any(indices < 0)
        or np.any(indices >= len(data["ids"]))
    ):
        raise ValueError("subset indices must be unique, in-bounds, 1D")
    out = dict(data)
    for key in ("tokens", "audio", "vision", "labels", "targets"):
        if key in out:
            out[key] = out[key][indices]
    for key in ("ids", "video_ids"):
        out[key] = [data[key][i] for i in indices]
    out["official_indices"] = indices.tolist()
    out["official_split"] = data["split"]
    if name:
        out["split"] = name
    return out


def fit_scaler(train, eps=1e-06):
    if train.get("split") != "train":
        raise ValueError("scaler fitting is restricted to explicitly tagged train")
    mask = observed_masks(train["tokens"], train["audio"], train["vision"])
    out = {
        "version": 1,
        "fitted_split": "train",
        "source_sha256": train.get("source_sha256"),
        "ids_sha256": hashlib.sha256("\n".join(train["ids"]).encode()).hexdigest(),
        "eps": eps,
    }
    for j, name in enumerate(("audio", "vision"), 1):
        valid = train[name][mask[:, j]].astype(np.float64)
        if len(valid):
            mean, std = (valid.mean(0), valid.std(0))
        else:
            mean, std = (
                np.zeros(train[name].shape[-1]),
                np.ones(train[name].shape[-1]),
            )
        constant = std < eps
        std[constant] = 1.0
        out[name] = {
            "mean": mean.tolist(),
            "std": std.tolist(),
            "count": len(valid),
            "constant": constant.tolist(),
        }
    return out


def apply_scaler(batch, scaler):
    """Keep supplied raw visibility even when a real standardized vector becomes all zero."""
    if scaler.get("fitted_split") != "train":
        raise ValueError("invalid scaler provenance")
    out = dict(batch)
    mask = batch.get("mask")
    if mask is None:
        mask = observed_masks(batch["tokens"], batch["audio"], batch["vision"])
    mask = np.asarray(mask, dtype=bool)
    out["mask"] = mask
    for j, name in enumerate(("audio", "vision"), 1):
        arr = np.asarray(batch[name], dtype=np.float32)
        if not np.isfinite(arr).all():
            raise ValueError(f"nonfinite {name}")
        arr = (arr - np.asarray(scaler[name]["mean"], np.float32)) / np.asarray(
            scaler[name]["std"], np.float32
        )
        out[name] = np.where(mask[:, j, :, None], arr, 0).astype(np.float32)
    out["structure"] = build_structure(mask)
    return out
