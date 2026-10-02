"""Copy exact official Q3 inputs and audit tokenizer offsets; no model forward."""

from __future__ import annotations
import argparse
import hashlib
import json
import pickle
import re
import shutil
import sys
from pathlib import Path
import numpy as np

BASE = Path(__file__).resolve().parents[1]
REVISION = "86b5e0934494bd15c9632b12f734a8a67f723594"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)


class RestrictedUnpickler(pickle.Unpickler):

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
        raise pickle.UnpicklingError("Persistent references forbidden")


def load_sample(path):
    with Path(path).open("rb") as f:
        sample = RestrictedUnpickler(f).load()
    expected = {"raw_text", "id", "text", "text_bert", "audio", "vision"}
    if set(sample) != expected:
        raise ValueError(f"Unexpected aligned Q3 fields in {path}: {list(sample)}")
    for name, shape in {
        "text": (50, 768),
        "text_bert": (3, 50),
        "audio": (50, 74),
        "vision": (50, 35),
    }.items():
        a = np.asarray(sample[name])
        if a.shape != shape or not np.isfinite(a).all():
            raise ValueError(f"Invalid {name} shape/finite values")
    t = np.asarray(sample["text_bert"])
    if not np.equal(t, np.floor(t)).all() or not np.isin(t[1:], [0, 1]).all():
        raise ValueError("Invalid integer token channels")
    if not ((t[0] >= 0) & (t[0] < 30522)).all() or not np.array_equal(
        t[0] == 0, t[1] == 0
    ):
        raise ValueError("Token vocabulary or attention mismatch")
    if str(sample["id"]) != Path(path).stem:
        raise ValueError("Sample id does not equal official filename")
    return sample


def token_mapping(sample, tokenizer):
    text = str(sample["raw_text"])
    enc = tokenizer.encode(text)
    actual = np.asarray(sample["text_bert"], dtype=np.int64)
    expected = np.asarray([enc.ids, enc.attention_mask, enc.type_ids], dtype=np.int64)
    exact = expected.shape == actual.shape and bool(np.array_equal(expected, actual))
    words = [
        {
            "word_id": i,
            "text": m.group(),
            "char_start": m.start(),
            "char_end": m.end(),
            "token_positions": [],
            "lexical": any((c.isalnum() for c in m.group())),
        }
        for i, m in enumerate(re.finditer("\\S+", text))
    ]
    records = []
    for pos in range(50):
        token_id = int(actual[0, pos])
        observed = bool(actual[1, pos] and token_id not in (0, 101, 102))
        offset = list(enc.offsets[pos]) if exact and observed else None
        candidates = [
            w
            for w in words
            if offset
            and offset[0] >= w["char_start"]
            and (offset[1] <= w["char_end"])
            and (offset[1] > offset[0])
        ]
        word_id = candidates[0]["word_id"] if len(candidates) == 1 else None
        if word_id is not None:
            words[word_id]["token_positions"].append(pos)
        records.append(
            {
                "position": pos,
                "token_id": token_id,
                "attention": int(actual[1, pos]),
                "token": enc.tokens[pos] if exact else None,
                "observed_content": observed,
                "audio_observed": (
                    bool(np.any(np.asarray(sample["audio"])[pos] != 0))
                    if "audio" in sample
                    else None
                ),
                "vision_observed": (
                    bool(np.any(np.asarray(sample["vision"])[pos] != 0))
                    if "vision" in sample
                    else None
                ),
                "char_start": offset[0] if offset else None,
                "char_end": offset[1] if offset else None,
                "word_id": word_id,
                "mapping_level": "raw_text_exact" if word_id is not None else "unknown",
            }
        )
    return {
        "raw_text": text,
        "raw_text_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "all_three_channels_exact": exact,
        "channels_exact": (
            np.all(expected == actual, axis=1).tolist()
            if expected.shape == actual.shape
            else [False] * 3
        ),
        "tokens": records,
        "words": words,
        "content_positions": sum((x["observed_content"] for x in records)),
        "mapped_content_positions": sum(
            (x["observed_content"] and x["word_id"] is not None for x in records)
        ),
        "truncated_words": [
            w["word_id"] for w in words if w["lexical"] and (not w["token_positions"])
        ],
        "timing_status": "not_run_unknown",
        "audio_video_feature_time_mapping": "no_official_timestamps",
    }


def copy_verified(source, target):
    digest = sha(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if sha(target) != digest:
            raise ValueError(f"Refusing to overwrite different staged bytes: {target}")
    else:
        shutil.copyfile(source, target)
    if sha(target) != digest:
        raise ValueError(f"Staged hash mismatch: {target}")
    return digest


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=BASE.parent)
    p.add_argument("--source", type=Path)
    p.add_argument("--output", type=Path, default=BASE / "data")
    p.add_argument("--tokenizer", type=Path)
    p.add_argument("--tokenizer-deps", type=Path)
    args = p.parse_args(argv)
    source = (
        args.source
        or args.root
        / "E_data/附件4-可解释专项视频样本与特征文件/附件4-可解释专项视频样本与特征文件/对齐版本"
    )
    tokenizer_dir = (
        args.tokenizer or args.root / "Q2_workplace/models/bert-base-uncased"
    )
    deps = args.tokenizer_deps or args.root / "Q2_workplace/_work/tokenizer_deps"
    if deps.is_dir():
        sys.path.insert(0, str(deps))
    from tokenizers import Tokenizer, __version__ as tokenizers_version

    tokenizer = Tokenizer.from_file(str(tokenizer_dir / "tokenizer.json"))
    tokenizer.enable_truncation(max_length=50)
    tokenizer.enable_padding(length=50, pad_id=0, pad_token="[PAD]")
    expected = [f"{i:02}" for i in range(1, 21)]
    if (
        sorted((x.stem for x in source.glob("*.pkl"))) != expected
        or sorted((x.stem for x in (source / "videos").glob("*.mp4"))) != expected
    ):
        raise ValueError(
            "Official selected version must contain exact 01..20 feature/video pairs"
        )
    records = []
    for sid in expected:
        source_pkl, source_video = (
            source / f"{sid}.pkl",
            source / "videos" / f"{sid}.mp4",
        )
        sample = load_sample(source_pkl)
        pkl = args.output / "special_aligned" / f"{sid}.pkl"
        video = args.output / "media" / f"{sid}.mp4"
        pkl_sha, video_sha = (
            copy_verified(source_pkl, pkl),
            copy_verified(source_video, video),
        )
        metadata = token_mapping(sample, tokenizer)
        metadata.update(
            sample_id=sid,
            sample_key=f"sample_{sid}",
            feature_version="aligned_50",
            source_pkl_sha256=pkl_sha,
            source_video_sha256=video_sha,
        )
        metadata_path = args.output / "metadata" / f"{sid}.json"
        write_json(metadata_path, metadata)
        records.append(
            {
                "sample_id": sid,
                "sample_key": f"sample_{sid}",
                "source_pickle": source_pkl.relative_to(args.root).as_posix(),
                "source_video": source_video.relative_to(args.root).as_posix(),
                "pickle": pkl.relative_to(args.output.parent).as_posix(),
                "video": video.relative_to(args.output.parent).as_posix(),
                "metadata": metadata_path.relative_to(args.output.parent).as_posix(),
                "pickle_sha256": pkl_sha,
                "video_sha256": video_sha,
                "metadata_sha256": sha(metadata_path),
                "all_three_channels_exact": metadata["all_three_channels_exact"],
                "content_positions": metadata["content_positions"],
                "mapped_content_positions": metadata["mapped_content_positions"],
            }
        )
    manifest = {
        "schema_version": 1,
        "status": "staged_schema_and_token_offsets_only",
        "task": "Q3 attachment4 aligned 20 independent samples",
        "sample_count": len(records),
        "source_unchanged": True,
        "sentiment_forward_executed": False,
        "alignment_executed": False,
        "attachment2_matching_performed": False,
        "tokenizer": {
            "model": "google-bert/bert-base-uncased",
            "revision": REVISION,
            "tokenizers_version": tokenizers_version,
            "files_sha256": {
                n: sha(tokenizer_dir / n)
                for n in ("tokenizer.json", "vocab.txt", "tokenizer_config.json")
            },
        },
        "token_channels_exact_samples": sum(
            (r["all_three_channels_exact"] for r in records)
        ),
        "content_positions": sum((r["content_positions"] for r in records)),
        "mapped_content_positions": sum(
            (r["mapped_content_positions"] for r in records)
        ),
        "implementation_sha256": sha(Path(__file__)),
        "samples": records,
    }
    write_json(args.output / "special_staging_manifest.json", manifest)
    print(
        json.dumps(
            {k: v for k, v in manifest.items() if k not in ("samples", "tokenizer")},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
