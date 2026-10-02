"""Create a loss-audited working copy; existing audited artifacts are immutable.

Defaults preserve the original workspace layout. A portable Q2 checkout uses:
 python tools/stage_compact_data.py --source /path/aligned_50.pkl --output-dir /new/data
Only NumPy/basic-container pickle globals are accepted. No fitting or split
semantics are applied; retained fields and protocol-4/gzip settings are unchanged.
"""

from pathlib import Path
import argparse, gzip, hashlib, json, pickle, sys
import numpy as np

BASE = Path(__file__).resolve().parents[1]
ROOT = BASE.parent
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))
from sentiment.data import RestrictedUnpickler, file_sha256

FIELDS = [
    "text_bert",
    "audio",
    "vision",
    "id",
    "raw_text",
    "classification_labels",
    "regression_labels",
    "annotations",
]
DEFAULT_SOURCE = ROOT / "E_data/附件2-数据集特征文件/aligned_50.pkl"


def convert_object(obj):
    """Preserve historical field ordering, FP32 conversion and rounding audit."""
    if not isinstance(obj, dict):
        raise ValueError("Expected a split-keyed dictionary")
    compact = {}
    details = {}
    for split, part in obj.items():
        if not isinstance(part, dict):
            raise ValueError("Expected per-split field dictionaries")
        compact[split] = {k: part[k] for k in FIELDS if k in part}
        details[split] = {}
        for name in ["audio", "vision"]:
            raw = np.asarray(part[name])
            converted = raw.astype(np.float32)
            details[split][name] = {
                "original_dtype": str(raw.dtype),
                "working_dtype": str(converted.dtype),
                "max_abs_rounding_error": float(
                    np.max(np.abs(raw - converted.astype(raw.dtype)))
                ),
                "nonzero_rows_preserved": bool(
                    np.array_equal(
                        np.any(raw != 0, axis=-1), np.any(converted != 0, axis=-1)
                    )
                ),
            }
            if not details[split][name]["nonzero_rows_preserved"]:
                raise ValueError(f"{split}/{name}: conversion changes nonzero row mask")
            if not np.isfinite(converted).all():
                raise ValueError(f"{split}/{name}: nonfinite FP32 result")
            compact[split][name] = converted
    return (compact, details)


def source_label(source):
    try:
        return source.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return source.as_posix()


def stage(source=DEFAULT_SOURCE, output_dir=BASE / "data"):
    source = Path(source).resolve()
    dest = Path(output_dir).resolve()
    output = dest / "compact_aligned.pkl.gz"
    manifest = dest / "working_copy_manifest.json"
    lock = dest / ".stage_compact_data.lock"
    protected = (output, manifest, dest / "compact_aligned.pkl")
    if any((p.exists() or p.is_symlink() for p in (*protected, lock))):
        raise FileExistsError(
            "Existing working-copy/audit/lock artifact; use a new --output-dir"
        )
    if source in protected:
        raise ValueError("Source and destination may not overlap")
    if not source.is_file():
        raise FileNotFoundError(source)
    dest.mkdir(parents=True, exist_ok=True)
    with lock.open("x", encoding="utf-8") as f:
        f.write("one working-copy writer; source=" + source.as_posix())
    created_output = created_manifest = False
    try:
        if any((p.exists() or p.is_symlink() for p in protected)):
            raise FileExistsError("Destination appeared after preflight")
        source_hash = file_sha256(source)
        with source.open("rb") as stream:
            obj = RestrictedUnpickler(stream).load()
        compact, details = convert_object(obj)
        with output.open("xb") as f:
            created_output = True
            with gzip.GzipFile(
                filename="", mode="wb", fileobj=f, mtime=0, compresslevel=5
            ) as z:
                pickle.dump(compact, z, protocol=4)
        raw_hash = hashlib.sha256()
        with gzip.open(output, "rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                raw_hash.update(block)
        if file_sha256(source) != source_hash:
            raise ValueError("Source changed during working-copy conversion")
        result = {
            "original_source_path": source_label(source),
            "original_source_sha256": source_hash,
            "working_copy_sha256": raw_hash.hexdigest(),
            "transform": "remove unused text embedding; A/V float64 to float32 with exact-value and row-mask check",
            "source": source_label(source),
            "source_sha256": source_hash,
            "working_file": output.name,
            "working_sha256": file_sha256(output),
            "bytes": output.stat().st_size,
            "fields_retained": FIELDS,
            "removed_field": "text: unused precomputed full-context BERT cache",
            "source_modified": False,
            "conversion": details,
            "official_split_counts": {k: len(v["id"]) for k, v in compact.items()},
            "instruction": "This file is a working transformation, not the original official file. Use original SHA for provenance.",
        }
        with manifest.open("x", encoding="utf-8") as f:
            created_manifest = True
            f.write(json.dumps(result, ensure_ascii=False, indent=2))
        return result
    except Exception:
        if created_manifest:
            manifest.unlink(missing_ok=True)
        if created_output:
            output.unlink(missing_ok=True)
        raise
    finally:
        lock.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=BASE / "data")
    args = parser.parse_args()
    print(json.dumps(stage(args.source, args.output_dir), ensure_ascii=False))


if __name__ == "__main__":
    main()
