"""Validate sample identity and physical clocks before final extraction."""

from __future__ import annotations
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import tempfile

SCHEMA = 1
TIME_TOL = 1e-09
MAX_MEMBER_BYTES = 16 * 1024 * 1024
MAX_TOTAL_BYTES = 256 * 1024 * 1024
INDEX_NAME = "transfer_manifest.json"
CLOCK_FIELDS = (
    "duration_s",
    "origin_pts_s",
    "audio_original_start_s",
    "video_original_start_s",
    "audio_rate_hz",
    "audio_samples",
    "audio_source_rate_hz",
    "audio_source_channels",
    "audio_observed_intervals",
    "video_observed_intervals",
    "decode_frame_count",
    "width",
    "height",
    "video_time_base",
    "audio_time_base",
    "audio_clock",
    "edit_list_policy",
    "mono_policy",
)
IDENTITY_FIELDS = ("sample_key", "sample_id", "video_id", "clip_id", "media_path")


def digest_bytes(data):
    return hashlib.sha256(data).hexdigest()


def digest_file(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def json_bytes(value):
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def parse_json(data, label):

    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{label}: duplicate JSON key {key!r}")
            result[key] = value
        return result

    def bad_constant(value):
        raise ValueError(f"{label}: nonfinite JSON constant {value}")

    value = json.loads(
        data.decode("utf-8-sig"), object_pairs_hook=unique, parse_constant=bad_constant
    )
    if not isinstance(value, dict):
        raise ValueError(f"{label}: expected a JSON object")
    return value


def safe_key(key):
    if not isinstance(key, str) or not re.fullmatch("[A-Za-z0-9_-]+", key):
        raise ValueError(f"Unsafe sample_key: {key!r}")
    if key.upper() in {
        "CON",
        "PRN",
        "AUX",
        "NUL",
        *[f"COM{i}" for i in range(1, 10)],
        *[f"LPT{i}" for i in range(1, 10)],
    }:
        raise ValueError(f"Reserved sample_key: {key!r}")
    return key


def inside(root, relative):
    relative = str(relative)
    posix = PurePosixPath(relative.replace("\\", "/"))
    if (
        posix.is_absolute()
        or ":" in relative
        or any((part in ("..", ".") for part in posix.parts))
    ):
        raise ValueError(f"Unsafe relative path: {relative!r}")
    root = root.resolve()
    path = root.joinpath(*posix.parts)
    if not path.resolve().is_relative_to(root):
        raise ValueError(f"Path escapes expected root: {relative!r}")
    return path


def regular_file(path):
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"Expected an existing regular nonsymlink file: {path}")


def load_manifest(path):
    rows = []
    seen, seen_ids = (set(), set())
    for line_number, line in enumerate(path.read_bytes().splitlines(), 1):
        if not line.strip():
            continue
        row = parse_json(line, f"manifest line {line_number}")
        key = safe_key(row.get("sample_key"))
        for field in (*IDENTITY_FIELDS, "text", "duration_s", "source_sha256"):
            if field not in row:
                raise ValueError(f"{key}: missing manifest field {field}")
        if key in seen or row["sample_id"] in seen_ids:
            raise ValueError(f"Duplicate manifest sample_key/sample_id: {key}")
        if not isinstance(row["text"], str) or not re.fullmatch(
            "[0-9a-f]{64}", row["source_sha256"]
        ):
            raise ValueError(f"{key}: invalid text or source_sha256")
        inside(path.parent, row["media_path"])
        seen.add(key)
        seen_ids.add(row["sample_id"])
        rows.append(
            {
                field: row[field]
                for field in (*IDENTITY_FIELDS, "text", "duration_s", "source_sha256")
            }
        )
    if not rows:
        raise ValueError("Empty manifest")
    return rows


def compare(left, right, label):
    """Exact categorical equality and absolute 1e-9 tolerance for clock numbers."""
    if isinstance(left, bool) or isinstance(right, bool):
        if type(left) is not type(right) or left != right:
            raise ValueError(f"{label}: boolean mismatch")
    elif isinstance(left, (int, float)) and isinstance(right, (int, float)):
        if (
            not math.isfinite(left)
            or not math.isfinite(right)
            or abs(left - right) > TIME_TOL
        ):
            raise ValueError(f"{label}: numeric mismatch {left!r} != {right!r}")
    elif isinstance(left, dict) and isinstance(right, dict):
        if left.keys() != right.keys():
            raise ValueError(f"{label}: metadata fields differ")
        for key in left:
            compare(left[key], right[key], f"{label}.{key}")
    elif isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            raise ValueError(f"{label}: list length mismatch")
        for i, (a, b) in enumerate(zip(left, right)):
            compare(a, b, f"{label}[{i}]")
    elif type(left) is not type(right) or left != right:
        raise ValueError(f"{label}: value mismatch {left!r} != {right!r}")


def clock_metadata(media, key):
    if media.get("schema_version") not in (2, 3) or media.get("status") != "complete":
        raise ValueError(f"{key}: media preparation is not complete schema 2")
    missing = set(CLOCK_FIELDS) - media.keys()
    if missing:
        raise ValueError(f"{key}: missing media clock fields: {sorted(missing)}")
    clock = {field: media[field] for field in CLOCK_FIELDS}
    duration = clock["duration_s"]
    if (
        not isinstance(duration, (float, int))
        or not math.isfinite(duration)
        or duration <= 0
    ):
        raise ValueError(f"{key}: invalid common duration")
    if (
        clock["audio_rate_hz"] != 16000
        or abs(clock["audio_samples"] / 16000 - duration) > 1 / 16000 + 1e-12
    ):
        raise ValueError(f"{key}: prepared audio clock disagrees with duration")
    for name in ("audio_observed_intervals", "video_observed_intervals"):
        intervals = clock[name]
        if not isinstance(intervals, list) or not intervals:
            raise ValueError(f"{key}.{name}: no observation intervals")
        previous = -TIME_TOL
        for interval in intervals:
            if not isinstance(interval, list) or len(interval) != 2:
                raise ValueError(f"{key}.{name}: invalid observation interval")
            a, b = interval
            if not all(
                (isinstance(v, (int, float)) and math.isfinite(v) for v in interval)
            ) or not (
                a >= -TIME_TOL
                and a >= previous - TIME_TOL
                and (a < b <= duration + TIME_TOL)
            ):
                raise ValueError(
                    f"{key}.{name}: invalid/nonmonotone/out-of-bounds observations"
                )
            previous = b
    compare(clock, clock, f"{key}.clock")
    return clock


def local_record(manifest, output, row):
    key = row["sample_key"]
    folder = inside(output, key)
    if folder.is_symlink():
        raise ValueError(f"{key}: sample output directory cannot be a symlink")
    source = inside(manifest.parent, row["media_path"])
    paths = {
        "source_media_sha256": source,
        "media_json_sha256": folder / "media.json",
        "frame_pts_sha256": folder / "frame_pts.npy",
        "audio16k_sha256": folder / "audio16k.wav",
    }
    for path in paths.values():
        regular_file(path)
    media_bytes = paths["media_json_sha256"].read_bytes()
    media = parse_json(media_bytes, f"{key}/media.json")
    if media.get("sample_key") != key or media.get("sample_id") != row["sample_id"]:
        raise ValueError(f"{key}: media sample identity mismatch")
    hashes = {
        name: digest_file(path)
        for name, path in paths.items()
        if name != "media_json_sha256"
    }
    hashes["media_json_sha256"] = digest_bytes(media_bytes)
    if (
        not hashes["source_media_sha256"]
        == row["source_sha256"]
        == media.get("source_sha256")
    ):
        raise ValueError(f"{key}: actual source media SHA256 != manifest/media.json")
    return {
        "identity": {field: row[field] for field in IDENTITY_FIELDS},
        "official_text_sha256": digest_bytes(row["text"].encode("utf-8")),
        "clock": clock_metadata(media, key),
        **hashes,
    }


def alignment_document(payload, row, local):
    key = row["sample_key"]
    doc = parse_json(payload, f"{key}/alignment.json")
    if doc.get("schema_version") != 1 or doc.get("status") not in (
        "complete",
        "partial",
        "unaligned",
        "no_lexical_text",
    ):
        raise ValueError(
            f"{key}: alignment is not a successful/partial schema 1 result"
        )
    if (
        doc.get("sample_key") != key
        or doc.get("sample_id") != row["sample_id"]
        or doc.get("text") != row["text"]
    ):
        raise ValueError(
            f"{key}: alignment identity or official text differs from manifest"
        )
    waveform_duration = (
        local["clock"]["audio_samples"] / local["clock"]["audio_rate_hz"]
    )
    if abs(waveform_duration - local["clock"]["duration_s"]) > 1 / 16000 + 1e-12:
        raise ValueError(
            f"{key}: WAV extent differs from media duration by more than one sample"
        )
    compare(doc.get("duration_s"), waveform_duration, f"{key}.alignment WAV duration")
    if not re.fullmatch("[0-9a-f]{64}", str(doc.get("input_fingerprint", ""))):
        raise ValueError(f"{key}: source inference fingerprint is absent or invalid")
    provenance = doc.get("provenance", {})
    for field in IDENTITY_FIELDS:
        if provenance.get(field) != row[field]:
            raise ValueError(f"{key}: alignment provenance identity differs: {field}")
    for field in ("media_json_sha256", "audio16k_sha256"):
        if provenance.get(field) != local[field]:
            raise ValueError(f"{key}: alignment provenance {field} is stale")
    if provenance.get("clock") != "prepared_audio_sample_zero_is_public_time_zero":
        raise ValueError(f"{key}: alignment public clock is unknown")
    compare(provenance.get("offset_applied_by_aligner_s"), 0.0, f"{key}.aligner offset")
    if "audio_observed_intervals" in doc:
        compare(
            doc["audio_observed_intervals"],
            local["clock"]["audio_observed_intervals"],
            f"{key}.alignment observations",
        )
    words = doc.get("words")
    if not isinstance(words, list) or doc.get("word_count") != len(words):
        raise ValueError(f"{key}: alignment word list/count invalid")
    matches = list(re.finditer("\\S+", row["text"]))
    if len(words) != len(matches):
        raise ValueError(f"{key}: alignment word count differs from official text")
    prior_end = 0.0
    aligned = 0
    for i, (word, match) in enumerate(zip(words, matches)):
        if word.get("text", word.get("original")) != match.group() or [
            word.get("char_start"),
            word.get("char_end"),
        ] != [match.start(), match.end()]:
            raise ValueError(
                f"{key}: word {i} character mapping differs from official text"
            )
        if word.get("alignment_valid"):
            a, b = (word.get("start"), word.get("end"))
            if not all(
                (isinstance(v, (int, float)) and math.isfinite(v) for v in (a, b))
            ) or not (
                a >= prior_end - 1e-06 and 0 <= a < b <= waveform_duration + 1e-06
            ):
                raise ValueError(f"{key}: word {i} timing invalid")
            prior_end, aligned = (b, aligned + 1)
        elif word.get("start") is not None or word.get("end") is not None:
            raise ValueError(f"{key}: word {i} unknown alignment has nonnull timing")
    if doc.get("aligned_word_count") != aligned:
        raise ValueError(f"{key}: aligned word count inconsistent")
    return doc


def atomic_bytes(path, payload):
    with tempfile.NamedTemporaryFile(
        prefix="." + path.name + ".", suffix=".tmp", dir=path.parent, delete=False
    ) as stream:
        temp = Path(stream.name)
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)
