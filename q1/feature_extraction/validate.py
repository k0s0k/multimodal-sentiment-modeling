"""Read-only Q1 artifact validation; no model, GPU, or source-label dependency.

Examples::

    python -m feature_extraction.validate --manifest stage/manifest.jsonl --output out --report-dir reports/all
    python -m feature_extraction.validate --manifest stage/manifest.jsonl --output out --report-dir reports/pilot --ids sample_001 --branches text,audio,acoustic

CSV always contains every manifest row. --ids limits deep validation and the
requested-scope verdict; unselected artifacts are explicitly NOT certified.
The full pipeline status always includes all seven branches. Exit 0 means the
requested artifacts passed structural validation (alignment.partial is allowed),
1 means incomplete requested outputs, and 2 means a data/schema/identity error.
Quality warnings and word coverage still need review before expanding a pilot.
"""

from __future__ import annotations
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import re
from pathlib import Path
import struct
from typing import Any
import numpy as np

BRANCHES = ("media", "alignment", "text", "audio", "acoustic", "face", "video")
FILES = {
    "media": ("media.json", "audio16k.wav", "frame_pts.npy", "source_time.npz"),
    "alignment": ("alignment.json",),
    "text": ("text.npz", "text.meta.json"),
    "audio": ("audio.npz", "audio.meta.json"),
    "acoustic": ("acoustic.npz", "acoustic.meta.json"),
    "face": ("face.npz", "face_frames.npz", "face.json", "face_status.json"),
    "video": ("video.npz", "video_timing.npz", "video.json", "video_status.json"),
}
DIMENSIONS = {"text": 1024, "audio": 1024, "acoustic": 25, "face": 44, "video": 768}
TIME_TOL = 2 / 16000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"{path.name}: expected a JSON object")
    return value


def json_safe(value):
    """Keep a corrupt metric from preventing delivery of its failure report."""
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)) and (not math.isfinite(value)):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def load_manifest(path: Path) -> list[dict]:
    if path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as stream:
            return list(csv.DictReader(stream))
    text = path.read_text(encoding="utf-8-sig")
    rows = (
        json.loads(text)
        if text.lstrip().startswith("[")
        else [json.loads(x) for x in text.splitlines() if x.strip()]
    )
    if not isinstance(rows, list) or not all((isinstance(row, dict) for row in rows)):
        raise ValueError("Manifest must be a JSON array, JSONL, or CSV of records")
    return rows


def aggregate(states: list[str]) -> str:
    if "failed" in states:
        return "failed"
    if states and all((state == "complete" for state in states)):
        return "complete"
    if not states or all((state == "not_run" for state in states)):
        return "not_run"
    return "partial"


class Checks:

    def __init__(self):
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.metrics: dict[str, Any] = {}

    def require(self, condition, message):
        if not bool(condition):
            self.errors.append(message)
        return bool(condition)

    def identity(self, doc, row, required=("sample_key",)):
        for key in ("sample_key", "sample_id", "video_id", "clip_id", "media_path"):
            if key in required or key in doc:
                self.require(
                    str(doc.get(key)) == str(row.get(key)), f"identity mismatch: {key}"
                )

    def probability(self, values, name):
        values = np.asarray(values)
        self.require(
            np.isfinite(values).all()
            and np.all((values >= -1e-06) & (values <= 1 + 1e-06)),
            f"{name}: expected finite values in [0,1]",
        )

    def mask(self, value, shape, name):
        a = np.asarray(value)
        okay = self.require(a.shape == shape, f"{name}: shape {a.shape} != {shape}")
        self.require(
            a.dtype.kind in "biu" and np.isin(a, [0, 1]).all(),
            f"{name}: expected boolean/integer 0/1",
        )
        return a.astype(bool) if okay else np.zeros(shape, bool)

    def times(
        self,
        values,
        duration,
        name,
        *,
        unknown=False,
        sorted_rows=False,
        disjoint=False,
        vector=False,
    ):
        a = np.asarray(values)
        self.require(
            a.dtype == np.float64,
            f"{name}: time dtype must be float64, found {a.dtype}",
        )
        if vector:
            if not self.require(
                a.ndim == 1, f"{name}: expected one-dimensional timestamps"
            ):
                return np.zeros(len(a), bool)
            known = np.isfinite(a)
            self.require(known.all(), f"{name}: nonfinite timestamps")
            self.require(
                np.all(a[known] >= -TIME_TOL)
                and np.all(a[known] <= duration + TIME_TOL),
                f"{name}: out of common-axis bounds",
            )
            if sorted_rows:
                self.require(
                    np.all(np.diff(a[known]) > 0),
                    f"{name}: timestamps must strictly increase",
                )
            return known
        if not self.require(
            a.ndim == 2 and a.shape[1] == 2, f"{name}: expected (N,2) intervals"
        ):
            return np.zeros(len(a), bool)
        finite = np.isfinite(a).all(axis=1)
        absent = np.isnan(a).all(axis=1) | np.all(a == -1, axis=1)
        known = finite & ~absent
        self.require(
            np.all(known | absent) if unknown else np.all(known),
            f"{name}: malformed or missing intervals",
        )
        b = a[known]
        self.require(
            np.all(b[:, 0] >= -TIME_TOL)
            and np.all(b[:, 1] <= duration + TIME_TOL)
            and np.all(b[:, 1] > b[:, 0]),
            f"{name}: reversed/empty/out-of-bounds intervals",
        )
        if sorted_rows or disjoint:
            self.require(
                np.all(np.diff(b[:, 0]) >= -1e-09),
                f"{name}: interval starts not monotone",
            )
        if disjoint:
            self.require(
                np.all(b[1:, 0] >= b[:-1, 1] - 1e-08),
                f"{name}: allocation/observation intervals overlap",
            )
        return known


def load_npz(path: Path, check: Checks, required=()) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        data = {name: archive[name] for name in archive.files}
    check.require(
        set(required) <= data.keys(),
        f"{path.name}: missing arrays {sorted(set(required) - data.keys())}",
    )
    return data


def check_csr(data, prefix, rows, source_count, check):
    keys = (f"{prefix}_offsets", f"{prefix}_indices", f"{prefix}_weights")
    if prefix == "merge":
        keys = ("merge_offsets", "merge_raw_token_indices", "merge_weights")
    if not check.require(
        all((key in data for key in keys)), f"missing {prefix} sparse provenance"
    ):
        return
    offsets, indices, weights = (data[key] for key in keys)
    if not check.require(
        offsets.shape == (rows + 1,) and indices.ndim == weights.ndim == 1,
        f"{prefix}: invalid sparse shapes",
    ):
        return
    okay = (
        offsets.dtype.kind in "iu"
        and indices.dtype.kind in "iu"
        and (weights.dtype == np.float32)
    )
    okay = (
        okay
        and offsets[0] == 0
        and np.all(np.diff(offsets) >= 0)
        and (offsets[-1] == len(indices) == len(weights))
    )
    okay = okay and np.all(indices >= 0) and np.all(indices < source_count)
    if not check.require(okay, f"{prefix}: sparse indices, offsets, or dtypes invalid"):
        return
    check.require(
        np.isfinite(weights).all() and np.all(weights > 0),
        f"{prefix}: weights must be finite and positive",
    )
    for a, b in zip(offsets[:-1], offsets[1:]):
        if b > a:
            check.require(
                abs(float(weights[a:b].sum()) - 1) <= 2e-05,
                f"{prefix}: nonempty row weights do not sum to one",
            )


def check_wav(path, metadata, check):
    """Parse the RIFF header without requiring soundfile or reading PCM data."""
    with path.open("rb") as stream:
        head = stream.read(12)
        if not check.require(
            len(head) == 12 and head[:4] == b"RIFF" and (head[8:] == b"WAVE"),
            "audio16k.wav: invalid RIFF/WAVE header",
        ):
            return
        fmt = None
        data_bytes = None
        while True:
            header = stream.read(8)
            if len(header) < 8:
                break
            kind, size = struct.unpack("<4sI", header)
            start = stream.tell()
            if not check.require(
                start + size <= path.stat().st_size, "audio16k.wav: truncated chunk"
            ):
                break
            if kind == b"fmt ":
                body = stream.read(min(size, 40))
                if len(body) >= 16:
                    code, channels, rate, byte_rate, block_align, bits = struct.unpack(
                        "<HHIIHH", body[:16]
                    )
                    if code == 65534 and len(body) >= 40:
                        code = struct.unpack("<H", body[24:26])[0]
                    fmt = (code, channels, rate, block_align, bits, byte_rate)
            elif kind == b"data":
                data_bytes = size
            stream.seek(start + size + size % 2)
        if check.require(
            fmt is not None and data_bytes is not None,
            "audio16k.wav: missing fmt/data chunk",
        ):
            code, channels, rate, block_align, bits, byte_rate = fmt
            check.require(
                (code, channels, rate, block_align, bits, byte_rate)
                == (3, 1, 16000, 4, 32, 64000),
                "audio16k.wav: expected mono 16000 Hz IEEE float32 PCM",
            )
            check.require(
                block_align > 0
                and data_bytes % block_align == 0
                and (
                    data_bytes // max(block_align, 1) == metadata.get("audio_samples")
                ),
                "audio16k.wav: sample count differs from media.json",
            )


def check_hash(path, expected, check, label):
    if expected and path.is_file():
        check.require(
            sha256(path) == expected,
            f"{label}: SHA256 mismatch (stale or changed artifact)",
        )


def validate_media(folder, row, check):
    doc = read_json(folder / "media.json")
    check.identity(doc, row, ("sample_key", "sample_id"))
    check.require(
        doc.get("schema_version") in (2, 3) and doc.get("status") == "complete",
        "media: expected successful schema_version 2 or 3",
    )
    duration = float(doc["duration_s"])
    check.require(
        math.isfinite(duration) and duration > 0, "media: invalid common duration"
    )
    check.require(
        doc.get("audio_rate_hz") == 16000, "media: audio rate must be 16000 Hz"
    )
    check.require(
        abs(int(doc["audio_samples"]) / 16000 - duration) <= TIME_TOL,
        "media: waveform extent disagrees with common duration",
    )
    if row.get("source_sha256"):
        check.require(
            doc.get("source_sha256") == row["source_sha256"],
            "media source_sha256 differs from manifest",
        )
    for kind in ("audio", "video"):
        observed = np.asarray(doc[f"{kind}_observed_intervals"], np.float64).reshape(
            -1, 2
        )
        check.times(
            observed, duration, f"media.{kind}_observed_intervals", disjoint=True
        )
        check.metrics[f"{kind}_observed_fraction"] = float(
            np.diff(observed, axis=1).sum() / duration
        )
    pts = np.load(folder / "frame_pts.npy", allow_pickle=False)
    check.times(pts, duration, "frame_pts.npy", vector=True, sorted_rows=True)
    check.require(
        len(pts) == doc["decode_frame_count"] and len(pts) > 0,
        "media: decoded frame count mismatch",
    )
    source = load_npz(
        folder / "source_time.npz",
        check,
        ("audio_pts", "audio_nb_samples", "video_pts", "video_duration"),
    )
    for name in ("audio_pts", "video_pts", "video_duration"):
        a = source[name]
        check.require(
            a.dtype == np.float64 and a.ndim == 1 and np.isfinite(a).all(),
            f"source_time.{name}: expected finite float64 vector",
        )
    check.require(
        len(source["video_pts"]) == len(pts) == len(source["video_duration"]),
        "source_time: video lengths mismatch",
    )
    check.require(
        len(source["audio_pts"]) == len(source["audio_nb_samples"]),
        "source_time: audio lengths mismatch",
    )
    check.require(
        np.all(source["audio_nb_samples"] > 0),
        "source_time: invalid audio frame sample counts",
    )
    check.require(
        np.allclose(
            source["video_pts"] - float(doc["origin_pts_s"]), pts, rtol=0, atol=1e-07
        ),
        "source_time: video origin mapping differs from frame_pts",
    )
    check_wav(folder / "audio16k.wav", doc, check)
    check.metrics.update(duration_s=duration, decoded_frames=len(pts))
    check.warnings.extend((str(x) for x in doc.get("warnings", [])))


def validate_alignment(folder, row, duration, check):
    doc = read_json(folder / "alignment.json")
    check.identity(doc, row, ("sample_key", "sample_id"))
    check.require(
        doc.get("schema_version") == 1, "alignment: unsupported schema_version"
    )
    status = doc.get("status")
    check.require(
        status in ("complete", "partial", "unaligned", "no_lexical_text"),
        f"alignment backend status: {status}",
    )
    check.require(
        doc.get("text") == row["text"], "alignment: official text differs from manifest"
    )
    words = doc["words"]
    expected_words = list(re.finditer("\\S+", row["text"]))
    check.require(
        len(words) == len(expected_words),
        "alignment: official whitespace-word inventory changed",
    )
    check.require(
        [(w.get("char_start"), w.get("char_end")) for w in words]
        == [(m.start(), m.end()) for m in expected_words],
        "alignment: official word spans missing, inserted, or reordered",
    )
    bounds, known, last_char_end = ([], [], 0)
    for i, word in enumerate(words):
        a, b = (word["char_start"], word["char_end"])
        check.require(
            isinstance(a, int)
            and isinstance(b, int)
            and (0 <= a < b <= len(row["text"])),
            f"word {i}: invalid character bounds",
        )
        original = word.get("original", word.get("text", word.get("word")))
        check.require(
            a >= last_char_end and row["text"][a:b] == original,
            f"word {i}: character mapping/order differs from official text",
        )
        last_char_end = b
        valid = bool(word.get("alignment_valid", False))
        x, y = (word.get("start"), word.get("end"))
        check.require(
            valid == (x is not None and y is not None),
            f"word {i}: alignment_valid disagrees with nullable timing",
        )
        bounds.append([np.nan if x is None else x, np.nan if y is None else y])
        known.append(valid)
        if word.get("audio_observed_fraction") is not None:
            check.probability(
                [word["audio_observed_fraction"]], f"word {i} audio_observed_fraction"
            )
    times = np.asarray(bounds, np.float64).reshape(-1, 2)
    check.times(times, duration, "alignment.words", unknown=True, disjoint=True)
    lexical = sum((w.get("status") != "nonlexical" for w in words))
    aligned = sum(known)
    check.require(
        doc.get("word_count") == len(words)
        and doc.get("aligned_word_count") == aligned,
        "alignment: word counts disagree with words",
    )
    if status == "complete":
        check.require(
            aligned == lexical, "alignment claims complete with unaligned lexical words"
        )
    if status == "partial":
        check.require(
            0 < aligned < lexical, "alignment.partial inconsistent with word counts"
        )
    check.metrics.update(
        word_count=len(words),
        lexical_word_count=lexical,
        aligned_word_count=aligned,
        aligned_word_fraction=aligned / lexical if lexical else None,
        alignment_source_status=status,
    )
    if status in ("partial", "unaligned"):
        check.warnings.append(
            f"{aligned}/{lexical} lexical words aligned; alignment partial is not a backend failure"
        )
    provenance = doc.get("provenance", {})
    check.identity(provenance, row)
    for name in ("audio16k", "media_json"):
        filename = "audio16k.wav" if name == "audio16k" else "media.json"
        check_hash(
            folder / filename,
            provenance.get(f"{name}_sha256"),
            check,
            f"alignment input {filename}",
        )
    if not doc.get("model_manifest", {}).get("loaded_state_dict_sha256"):
        check.warnings.append("alignment model loaded-state SHA256 is absent")
    return status


def check_word_rows(data, branch, words, known_times, check):
    count = len(words)
    rows = len(data["features"])
    check.require(
        rows == count if branch == "text" else rows >= count,
        f"{branch}: missing or extra word rows",
    )
    if rows < count:
        return
    if "word_index" in data:
        check.require(
            np.array_equal(data["word_index"][:count], np.arange(count)),
            f"{branch}: word_index not manifest word order",
        )
        check.require(
            np.all(data["word_index"][count:] == -1),
            f"{branch}: gap word_index must be -1",
        )
    kind_key = "node_type" if branch == "audio" else "node_types"
    if kind_key in data:
        check.require(
            np.all(data[kind_key][:count] == "word")
            and np.all(data[kind_key][count:] != "word"),
            f"{branch}: word/gap node types disagree with alignment",
        )
    expected = np.asarray([bool(w.get("alignment_valid", False)) for w in words])
    check.require(
        np.array_equal(known_times[:count], expected),
        f"{branch}: known word times disagree with alignment",
    )
    tkey = "times" if branch == "video" else "time_s"
    actual = data[tkey][:count][expected]
    wanted = np.asarray(
        [[w["start"], w["end"]] for w in words if w.get("alignment_valid")], np.float64
    ).reshape(-1, 2)
    check.require(
        actual.shape == wanted.shape
        and np.allclose(actual, wanted, atol=1e-07, rtol=0),
        f"{branch}: word times differ from alignment.json",
    )
    for name, part in (("words", data[tkey][:count]), ("gaps", data[tkey][count:])):
        if len(part):
            check.times(
                part,
                float(check.metrics["duration_s"]),
                f"{branch}.{name}",
                unknown=True,
                disjoint=True,
            )


def validate_features(folder, row, branch, duration, check):
    encoder = branch in ("text", "audio", "acoustic")
    meta_name = f"{branch}.meta.json" if encoder else f"{branch}.json"
    doc = read_json(folder / meta_name)
    check.identity(
        doc,
        row,
        (
            ("sample_key", "sample_id", "video_id", "clip_id", "media_path")
            if encoder
            else ("sample_key",)
        ),
    )
    check.require(
        doc.get("schema_version") == 1
        and doc.get("branch") == branch
        and (doc.get("status") == "complete"),
        f"{meta_name}: expected complete schema_version 1 for {branch}",
    )
    if encoder:
        check.require(
            doc.get("labels_used") is False, f"{branch}: labels_used must be false"
        )
        check.require(
            bool(doc.get("output_sha256")), f"{branch}: missing output SHA256"
        )
        check_hash(
            folder / f"{branch}.npz", doc.get("output_sha256"), check, f"{branch}.npz"
        )
        input_hashes = doc.get("input_hashes", {})
        for name, filename in (
            ("media_json_sha256", "media.json"),
            ("alignment_json_sha256", "alignment.json"),
            ("audio16k_sha256", "audio16k.wav"),
        ):
            check_hash(
                folder / filename,
                input_hashes.get(name),
                check,
                f"{branch} input {filename}",
            )
        if input_hashes.get("official_text_sha256"):
            check.require(
                hashlib.sha256(row["text"].encode("utf-8")).hexdigest()
                == input_hashes["official_text_sha256"],
                f"{branch}: official text SHA256 mismatch",
            )
    else:
        status = read_json(folder / f"{branch}_status.json")
        check.identity(status, row)
        check.require(
            status.get("status") == "complete" and status.get("branch") == branch,
            f"{branch}: last run failed or status invalid",
        )
    tkey, mkey = ("time_s", "valid_mask") if encoder else ("times", "valid")
    required = ["features", tkey, mkey]
    if branch != "text":
        required.append("valid_time_fraction" if branch == "face" else "coverage")
    data = load_npz(folder / f"{branch}.npz", check, required)
    features = data["features"]
    if not check.require(
        features.ndim == 2 and features.shape[1] == DIMENSIONS[branch],
        f"{branch}: expected (N,{DIMENSIONS[branch]}) features, got {features.shape}",
    ):
        return
    n = len(features)
    check.require(
        features.dtype == np.float32,
        f"{branch}: features must be float32, found {features.dtype}",
    )
    mask_shape = features.shape if branch == "acoustic" else (n,)
    valid = check.mask(data[mkey], mask_shape, f"{branch}.{mkey}")
    broadcast = (
        valid
        if branch == "acoustic"
        else np.broadcast_to(valid[:, None], features.shape)
    )
    check.require(
        np.isfinite(features[broadcast]).all(),
        f"{branch}: nonfinite features under valid mask",
    )
    check.require(
        data[tkey].shape == (n, 2), f"{branch}: time rows differ from feature rows"
    )
    known = check.times(
        data[tkey],
        duration,
        f"{branch}.{tkey}",
        unknown=branch in ("text", "audio", "video"),
        disjoint=branch in ("acoustic", "face"),
    )
    row_valid = valid.any(axis=1) if branch == "acoustic" else valid
    if branch != "text" and len(known) == n:
        check.require(
            not np.any(row_valid & ~known),
            f"{branch}: valid modality row has unknown time",
        )
    check.metrics.update(
        feature_rows=n,
        valid_rows=int(row_valid.sum()),
        valid_row_fraction=float(row_valid.mean()) if n else 0,
        feature_values=int(features.size),
        duration_s=duration,
    )
    if not row_valid.any():
        check.warnings.append(
            f"{branch}: no valid modality features; completed extraction is not evidence of availability"
        )
    if branch != "text":
        ckey = "valid_time_fraction" if branch == "face" else "coverage"
        coverage = data[ckey]
        check.require(
            coverage.shape == mask_shape and coverage.dtype == np.float32,
            f"{branch}.{ckey}: shape/dtype mismatch",
        )
        check.probability(coverage, f"{branch}.{ckey}")
        if coverage.shape == valid.shape:
            check.require(
                np.array_equal(coverage > 0, valid),
                f"{branch}: coverage>0 disagrees with valid mask",
            )
        check.metrics["mean_coverage"] = float(coverage.mean()) if coverage.size else 0
    if branch in ("text", "audio", "video"):
        alignment = read_json(folder / "alignment.json")
        words = alignment["words"]
        check_word_rows(data, branch, words, known, check)
        count = len(words)
        expected = np.asarray([bool(w.get("alignment_valid", False)) for w in words])
        if count <= n:
            check.metrics["valid_word_count"] = int(row_valid[:count].sum())
            check.metrics["valid_word_fraction"] = (
                float(row_valid[:count].mean()) if count else None
            )
            check.metrics["aligned_words_with_modality_fraction"] = (
                float(row_valid[:count][expected].mean()) if expected.any() else None
            )
    if branch == "text":
        alignment_known = check.mask(
            data["alignment_known"], (n,), "text.alignment_known"
        )
        check.require(
            np.array_equal(alignment_known, known), "text: alignment_known mismatch"
        )
        check.probability(data["character_coverage"], "text.character_coverage")
        check.require(
            data["character_coverage"].shape == (n,)
            and data["character_coverage"].dtype == np.float32,
            "text: invalid character_coverage shape/dtype",
        )
        expected = np.asarray(
            [[w["char_start"], w["char_end"]] for w in words], np.int64
        ).reshape(-1, 2)
        check.require(
            np.array_equal(data["char_bounds"], expected),
            "text: character bounds disagree with alignment",
        )
    if branch in ("audio", "acoustic"):
        anchors, cells = (data["anchors_s"], data["allocation_bounds_s"])
        check.times(
            anchors, duration, f"{branch}.anchors_s", vector=True, sorted_rows=True
        )
        check.times(cells, duration, f"{branch}.allocation_bounds_s", disjoint=True)
        check.require(
            cells.shape == (len(anchors), 2),
            f"{branch}: allocation/anchor count mismatch",
        )
        skey = (
            "receptive_field_bounds_s"
            if branch == "audio"
            else "analysis_support_bounds_s"
        )
        check.times(data[skey], duration, f"{branch}.{skey}", sorted_rows=True)
        check.require(
            data[skey].shape == cells.shape, f"{branch}: source support count mismatch"
        )
        check.times(
            data["observed_intervals_s"],
            duration,
            f"{branch}.observed_intervals_s",
            disjoint=True,
        )
        observed = np.asarray(
            read_json(folder / "media.json")["audio_observed_intervals"], np.float64
        ).reshape(-1, 2)
        check.require(
            observed.shape == data["observed_intervals_s"].shape
            and np.allclose(observed, data["observed_intervals_s"], atol=1e-08),
            f"{branch}: observed intervals differ from media",
        )
        if branch == "audio":
            check.require(
                np.allclose(np.diff(anchors), 0.02, atol=1e-08),
                "audio: WavLM anchor stride is not 20ms",
            )
            check.require(
                np.allclose(np.diff(data[skey], axis=1), 0.025, atol=1e-08),
                "audio: convolution support is not 25ms",
            )
            if doc.get("details", {}).get("context_contains_inserted_audio"):
                check.warnings.append(
                    "WavLM context contains inserted unobserved audio; pooling mask alone does not remove contextual influence"
                )
        else:
            names = data["feature_names"]
            check.require(
                names.shape == (25,) and len(set(names.tolist())) == 25,
                "acoustic: expected 25 distinct LLD names",
            )
            check.probability(data["voiced_fraction"], "acoustic.voiced_fraction")
            voiced_valid = check.mask(
                data["voiced_fraction_valid"], (n,), "acoustic.voiced_fraction_valid"
            )
            check.require(
                data["voiced_fraction"].shape == (n,)
                and data["voiced_fraction"].dtype == np.float32,
                "acoustic: voiced_fraction shape/dtype mismatch",
            )
            f0 = [i for i, name in enumerate(names) if "F0semitone" in str(name)]
            check.require(len(f0) == 1, "acoustic: expected one F0semitone channel")
            if len(f0) == 1:
                pitch = f0[0]
                check.require(
                    np.all(features[valid[:, pitch], pitch] > 0),
                    "acoustic: valid F0 contains nonpositive/unvoiced measurement",
                )
                check.require(
                    not np.any(
                        valid[:, pitch]
                        & (~voiced_valid | (data["voiced_fraction"] <= 0))
                    ),
                    "acoustic: pitch valid without voiced evidence",
                )
            check.metrics["per_feature_valid_fraction"] = (
                valid.mean(axis=0).tolist() if n else [0] * 25
            )
    if branch in ("face", "acoustic"):
        expected_starts = np.arange(n) * 0.1
        check.require(
            np.allclose(data[tkey][:, 0], expected_starts, atol=1e-08, rtol=0)
            and (n > 0 and abs(data[tkey][-1, 1] - duration) < TIME_TOL),
            f"{branch}: grid is not 100ms from zero through duration",
        )
        check.require(
            np.allclose(
                data[tkey][:, 1],
                np.minimum(expected_starts + 0.1, duration),
                atol=1e-08,
            ),
            f"{branch}: incorrect 100ms cell ends",
        )
    if branch == "face":
        check.require(
            n == int(math.ceil(duration / 0.1 - 1e-10)),
            "face: floating-point boundary created a spurious grid cell",
        )
        frame = load_npz(
            folder / "face_frames.npz",
            check,
            (
                "times",
                "allocation_support",
                "observed_support",
                "support_frame_indices",
                "valid",
                "confidence",
                "bbox_xyxy",
                "detection_success",
            ),
        )
        pts = np.load(folder / "frame_pts.npy", allow_pickle=False)
        check.times(
            frame["times"], duration, "face_frames.times", vector=True, sorted_rows=True
        )
        check.require(
            np.array_equal(frame["times"], pts),
            "face: frame timestamps differ from prepared frame_pts",
        )
        fvalid = check.mask(frame["valid"], (len(pts),), "face_frames.valid")
        check.mask(
            frame["detection_success"], (len(pts),), "face_frames.detection_success"
        )
        check.times(
            frame["allocation_support"],
            duration,
            "face_frames.allocation_support",
            disjoint=True,
        )
        check.times(
            frame["observed_support"],
            duration,
            "face_frames.observed_support",
            disjoint=True,
        )
        indices = frame["support_frame_indices"]
        check.require(
            indices.shape == (len(frame["observed_support"]),)
            and indices.dtype.kind in "iu"
            and np.all(indices >= 0)
            and np.all(indices < len(pts)),
            "face: invalid observation-to-frame mapping",
        )
        check.require(
            frame["confidence"].dtype == np.float32
            and frame["confidence"].shape == (len(pts),),
            "face: confidence shape/dtype mismatch",
        )
        check.probability(
            frame["confidence"][fvalid], "face confidence under valid mask"
        )
        check.require(
            frame["bbox_xyxy"].shape == (len(pts), 4)
            and frame["bbox_xyxy"].dtype == np.float32,
            "face: bbox shape/dtype mismatch",
        )
        check.require(
            np.isfinite(frame["bbox_xyxy"][fvalid]).all(),
            "face: nonfinite bbox under valid mask",
        )
        check.require(
            np.all(features[valid, :17] >= -1e-05)
            and np.all(features[valid, :17] <= 5.0001),
            "face: AU intensity outside [0,5]",
        )
        check.probability(features[valid, 17:35], "face AU presence duration fraction")
        check_csr(data, "source", n, len(pts), check)
        check.metrics["valid_frame_fraction"] = (
            float(fvalid.mean()) if len(fvalid) else 0
        )
        if doc.get("quality_warning"):
            check.warnings.append(str(doc["quality_warning"]))
    if branch == "video":
        timing = load_npz(
            folder / "video_timing.npz", check, ("support", "global_ticks")
        )
        check.times(timing["support"], duration, "video_timing.support", disjoint=True)
        check.require(
            timing["global_ticks"].shape == (len(timing["support"]),),
            "video: global tick count mismatch",
        )
        check_csr(data, "source", n, len(timing["support"]), check)
        check_csr(
            timing,
            "merge",
            len(timing["support"]),
            len(doc.get("windows", [])) * 8,
            check,
        )
        alignment_valid = check.mask(
            data["alignment_valid"], (n,), "video.alignment_valid"
        )
        check.require(
            np.array_equal(alignment_valid, known),
            "video: alignment_valid disagrees with node time",
        )
        check.metrics["crop_types"] = doc.get("crop_types", {})


def branch_result(folder, row, branch, selected=True):
    paths = [folder / name for name in FILES[branch]]
    error_file = folder / "media.error.json" if branch == "media" else None
    available = [p.name for p in paths if p.is_file()]
    missing = [p.name for p in paths if not p.is_file()]
    result = {
        "status": "not_run",
        "validated": selected,
        "missing_files": missing,
        "files_present": available,
        "errors": [],
        "warnings": [],
        "metrics": {},
        "artifact_bytes": sum(
            (p.stat().st_size for p in paths if p.is_file() and p.suffix != ".wav")
        ),
    }
    if error_file and error_file.is_file():
        result["artifact_bytes"] += error_file.stat().st_size
    if not available and (not (error_file and error_file.is_file())):
        return result
    if not selected:
        result.update(
            status="partial",
            warnings=["not_selected: existing artifacts were not validated"],
        )
        return result
    check = Checks()
    partial_alignment = False
    try:
        if branch == "media" and (folder / "media.json").is_file():
            current = read_json(folder / "media.json")
            if current.get("status") == "processing":
                result.update(
                    status="partial",
                    warnings=[
                        "media preparation is in progress; existing arrays are not certified"
                    ],
                )
                return result
        status_path = folder / f"{branch}_status.json"
        if status_path.is_file():
            run = read_json(status_path)
            if run.get("status") in ("failed", "error"):
                check.errors.append(
                    f"last {branch} run failed: {run.get('error', run.get('error_type', 'unknown error'))}"
                )
        if (
            error_file
            and error_file.is_file()
            and (
                not (folder / "media.json").is_file()
                or error_file.stat().st_mtime_ns
                >= (folder / "media.json").stat().st_mtime_ns
            )
        ):
            check.errors.append(
                f"last media run failed: {read_json(error_file).get('error', 'see media.error.json')}"
            )
        if not missing:
            if branch == "media":
                validate_media(folder, row, check)
            else:
                duration = float(read_json(folder / "media.json")["duration_s"])
                if branch == "alignment":
                    status = validate_alignment(folder, row, duration, check)
                    partial_alignment = status in ("partial", "unaligned")
                else:
                    validate_features(folder, row, branch, duration, check)
    except Exception as exc:
        check.errors.append(f"{type(exc).__name__}: {exc}")
    result.update(errors=check.errors, warnings=check.warnings, metrics=check.metrics)
    result["status"] = (
        "failed"
        if check.errors
        else "partial" if missing or partial_alignment else "complete"
    )
    return result


def validate(manifest: Path, output: Path, report_dir: Path, branches=None, ids=None):
    rows = load_manifest(manifest)
    required = set(branches or BRANCHES)
    if not required <= set(BRANCHES):
        raise ValueError(f"Unknown branches: {sorted(required - set(BRANCHES))}")
    if required - {"media"}:
        required.add("media")
    if required & {"text", "audio", "video"}:
        required.add("alignment")
    keys = [str(row.get("sample_key", "")) for row in rows]
    selected = set(ids) if ids else set(keys)
    if selected - set(keys):
        raise ValueError(f"Unknown sample keys: {sorted(selected - set(keys))}")
    manifest_errors, manifest_warnings = ([], [])
    for field in ("sample_key", "sample_id"):
        values = [str(row.get(field, "")) for row in rows]
        duplicate = [key for key, count in Counter(values).items() if count > 1]
        if duplicate:
            manifest_errors.append(f"Duplicate {field}: {duplicate}")
    if len(rows) != 100:
        manifest_warnings.append(
            f"Manifest has {len(rows)} rows; full competition delivery expects 100"
        )
    results = []
    for row in rows:
        key = str(row.get("sample_key", ""))
        safe = (
            bool(key)
            and key not in (".", "..")
            and ("/" not in key)
            and ("\\" not in key)
            and (":" not in key)
        )
        fields = (
            "sample_key",
            "sample_id",
            "video_id",
            "clip_id",
            "media_path",
            "text",
            "duration_s",
        )
        identity_errors = [f"manifest: missing {f}" for f in fields if f not in row]
        if not safe:
            identity_errors.append("manifest: unsafe sample_key")
        folder = output / key if safe else output / "__invalid_manifest_sample__"
        states = (
            {b: branch_result(folder, row, b, key in selected) for b in BRANCHES}
            if not identity_errors
            else {
                b: {
                    "status": "failed",
                    "validated": False,
                    "errors": identity_errors,
                    "warnings": [],
                    "metrics": {},
                    "missing_files": list(FILES[b]),
                    "files_present": [],
                    "artifact_bytes": 0,
                }
                for b in BRANCHES
            }
        )
        manifest_errors.extend((f"{key}: {e}" for e in identity_errors))
        results.append(
            {
                "sample_key": key,
                "sample_id": row.get("sample_id"),
                "video_id": row.get("video_id"),
                "clip_id": row.get("clip_id"),
                "selected": key in selected,
                "pipeline_status": aggregate([states[b]["status"] for b in BRANCHES]),
                "requested_status": aggregate([states[b]["status"] for b in required]),
                "artifact_bytes": sum((states[b]["artifact_bytes"] for b in BRANCHES)),
                "branches": states,
            }
        )
    chosen = [r for r in results if r["selected"]]
    missing = [
        {
            "sample_key": r["sample_key"],
            "branch": b,
            "status": r["branches"][b]["status"],
            "required_in_this_run": r["selected"] and b in required,
            "missing_files": r["branches"][b]["missing_files"],
            "errors": r["branches"][b]["errors"],
        }
        for r in results
        for b in BRANCHES
        if r["branches"][b]["status"] != "complete"
    ]

    def accepted(branch, result):
        return result["status"] == "complete" or (
            branch == "alignment"
            and result["status"] == "partial"
            and (not result["errors"])
            and (not result["missing_files"])
        )

    passed = (
        bool(chosen)
        and (not manifest_errors)
        and all((accepted(b, r["branches"][b]) for r in chosen for b in required))
    )
    failures = bool(manifest_errors) or any(
        (r["branches"][b]["status"] == "failed" for r in chosen for b in required)
    )
    quality = [
        {
            "sample_key": r["sample_key"],
            "branch": b,
            "warnings": r["branches"][b]["warnings"],
        }
        for r in chosen
        for b in required
        if r["branches"][b]["warnings"]
    ]
    summary = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "manifest": str(manifest.resolve()),
        "output": str(output.resolve()),
        "manifest_rows": len(rows),
        "expected_full_delivery_rows": 100,
        "selected_rows": len(chosen),
        "manifest_errors": manifest_errors,
        "manifest_warnings": manifest_warnings,
        "required_branches": [b for b in BRANCHES if b in required],
        "status_definitions": {
            "not_run": "No branch artifacts found",
            "partial": "Missing artifacts, partial alignment, or present but not selected for validation",
            "failed": "Latest run failed or an identity/schema/value/hash check failed",
            "complete": "All expected files present and validation passed",
        },
        "pipeline_status_counts_all_manifest": dict(
            Counter((r["pipeline_status"] for r in results))
        ),
        "requested_status_counts_selected": dict(
            Counter((r["requested_status"] for r in chosen))
        ),
        "branch_status_counts_selected": {
            b: dict(Counter((r["branches"][b]["status"] for r in chosen)))
            for b in BRANCHES
        },
        "requested_structural_validation_passed": passed,
        "pilot_decision": (
            "ready_for_quality_review"
            if passed
            else "not_ready_missing_or_invalid_artifacts"
        ),
        "pilot_note": "Structural success does not approve alignment accuracy, speaking-person identity, or useful modality coverage. Review per-word/per-modality coverage and warnings before expansion.",
        "full_100_sample_delivery_complete": len(rows) == 100
        and (not manifest_errors)
        and all((r["pipeline_status"] == "complete" for r in results)),
        "quality_review": quality,
        "missing_branches_count": len(missing),
        "artifact_bytes": sum((r["artifact_bytes"] for r in results)),
        "artifact_decimal_mb": sum((r["artifact_bytes"] for r in results)) / 1000000,
        "byte_accounting": {
            "included": "Exact on-disk bytes of named final NPZ/NPY/JSON outputs and per-branch status/error JSON, including partial outputs",
            "excluded": "audio16k.wav; source videos; checkpoints/model caches; openface_native intermediate CSV/logs; report files; top-level model manifests and any other unlisted files",
            "not_a_complete_submission_package_size": True,
        },
        "samples": results,
    }
    summary, missing = (json_safe(summary), json_safe(missing))
    report_dir.mkdir(parents=True, exist_ok=True)
    for name, value in (("summary.json", summary), ("missing_branches.json", missing)):
        target = report_dir / name
        temp = target.with_name(target.name + ".tmp")
        temp.write_text(
            json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temp.replace(target)
    fields = [
        "sample_key",
        "sample_id",
        "video_id",
        "clip_id",
        "selected",
        "pipeline_status",
        "requested_status",
        "artifact_bytes",
    ]
    fields += [f"{b}_status" for b in BRANCHES]
    fields += [
        "word_count",
        "aligned_word_count",
        "aligned_word_fraction",
        "text_valid_word_fraction",
        "audio_valid_word_fraction",
        "video_valid_word_fraction",
        "face_valid_frame_fraction",
        "acoustic_valid_row_fraction",
        "missing_branches",
        "errors",
        "warnings",
    ]
    with (report_dir / "samples.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for r in results:
            entry = {k: r[k] for k in fields if k in r}
            entry.update({f"{b}_status": r["branches"][b]["status"] for b in BRANCHES})
            for metric in ("word_count", "aligned_word_count", "aligned_word_fraction"):
                entry[metric] = r["branches"]["alignment"]["metrics"].get(metric)
            for b, metric in (
                ("text", "valid_word_fraction"),
                ("audio", "valid_word_fraction"),
                ("video", "valid_word_fraction"),
                ("face", "valid_frame_fraction"),
                ("acoustic", "valid_row_fraction"),
            ):
                entry[f"{b}_{metric}"] = r["branches"][b]["metrics"].get(metric)
            entry["missing_branches"] = ",".join(
                (b for b in BRANCHES if r["branches"][b]["status"] != "complete")
            )
            for kind in ("errors", "warnings"):
                entry[kind] = " | ".join(
                    (
                        f"{b}: {message}"
                        for b in BRANCHES
                        for message in r["branches"][b][kind]
                    )
                )
            writer.writerow(entry)
    return (summary, 2 if failures else 0 if passed else 1)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--branches", default=",".join(BRANCHES))
    parser.add_argument("--report-dir", required=True, type=Path)
    parser.add_argument("--ids", default="")
    args = parser.parse_args(argv)
    try:
        summary, code = validate(
            args.manifest,
            args.output,
            args.report_dir,
            [x.strip() for x in args.branches.split(",") if x.strip()],
            [x.strip() for x in args.ids.split(",") if x.strip()],
        )
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    print(
        json.dumps(
            {
                key: summary[key]
                for key in (
                    "manifest_rows",
                    "selected_rows",
                    "required_branches",
                    "requested_structural_validation_passed",
                    "pilot_decision",
                    "artifact_bytes",
                )
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print(f"Reports: {args.report_dir.resolve()}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
