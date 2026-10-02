"""Align the official transcript to the prepared 16 kHz public media clock.

Only this module's CLI loads WhisperX; helpers and unit tests need stdlib only.
No ASR, timestamp interpolation, transcript replacement, or label access occurs.
"""

from __future__ import annotations
import argparse
from contextlib import contextmanager
import hashlib
import importlib.metadata
import inspect
import json
import math
import os
from pathlib import Path
import re
import sys
import time
import unicodedata
import wave
from typing import Any

SCHEMA_VERSION = 1
DEFAULT_MODEL = "WAV2VEC2_ASR_BASE_960H"
SAMPLE_RATE = 16000
DEFAULT_DICTIONARY = set("abcdefghijklmnopqrstuvwxyz'|")
PACKAGE_NAMES = (
    "whisperx",
    "torch",
    "torchaudio",
    "transformers",
    "numpy",
    "nltk",
    "soundfile",
)
VERIFIED_WHISPERX_VERSION = "3.7.4"
VERIFIED_ALIGNMENT_SHA256 = (
    "f740b561d4fec513f802e837981b146882d62372996947c77fd789e2b84d6b74"
)
SILENT_PEAK_THRESHOLD = 1e-07


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def digest_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode(
            "utf-8"
        )
    ).hexdigest()


def package_versions() -> dict[str, str | None]:
    versions = {}
    for name in PACKAGE_NAMES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def finite_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def merge_intervals(intervals: list[list[float]], duration: float) -> list[list[float]]:
    validated = []
    for interval in intervals:
        if len(interval) != 2:
            raise ValueError("Observed intervals must contain [start, end] pairs")
        start, end = map(finite_number, interval)
        if (
            start is None
            or end is None
            or start < -1e-06
            or (end < start)
            or (end > duration + 0.001)
        ):
            raise ValueError(f"Invalid observed interval: {interval}")
        if end > start:
            validated.append([max(0.0, start), min(duration, end)])
    merged: list[list[float]] = []
    for start, end in sorted(validated):
        if merged and start <= merged[-1][1] + 1e-09:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged


def interval_coverage(start: float, end: float, observed: list[list[float]]) -> float:
    if end <= start:
        return 0.0
    covered = sum(
        (max(0.0, min(end, stop) - max(start, begin)) for begin, stop in observed)
    )
    return min(1.0, covered / (end - start))


def prepare_transcript(text: str, dictionary: set[str] | None = None) -> dict[str, Any]:
    """Keep official whitespace-token spans; reversible normalization is separate.

    Digits/foreign letters are NOT guessed or passed as wildcard acoustic tokens.
    Such entire words remain in the output with unknown timing. Punctuation is
    removed, hyphens become spaces, and curly apostrophes become ASCII apostrophes.
    """
    if not isinstance(text, str):
        raise TypeError("Manifest text must be a string")
    dictionary = {c.lower() for c in dictionary or DEFAULT_DICTIONARY}
    words = []
    normalized_parts, normalized_map = ([], [])
    align_parts, alignment_map = ([], [])
    normalized_cursor = alignment_cursor = 0
    for word_id, match in enumerate(re.finditer("\\S+", text)):
        chars: list[str] = []
        indices: list[int] = []
        for offset, source in enumerate(match.group()):
            source_index = match.start() + offset
            for char in unicodedata.normalize("NFKC", source).casefold():
                if char in "’‘ʼ`":
                    char = "'"
                if char in "-‐‑‒–—":
                    char = " "
                if char.isalnum() or char == "'" or char == " ":
                    if char == " " and (not chars or chars[-1] == " "):
                        continue
                    chars.append(char)
                    indices.append(source_index)
        while chars and chars[-1] == " ":
            chars.pop()
            indices.pop()
        candidate = "".join(chars)
        lexical = any((c.isalnum() for c in candidate))
        unsupported = sorted({c for c in candidate if c != " " and c not in dictionary})
        normalized_span = None
        if candidate:
            if normalized_parts:
                normalized_cursor += 1
                normalized_map.append(None)
            normalized_span = [normalized_cursor, normalized_cursor + len(candidate)]
            normalized_cursor += len(candidate)
            normalized_parts.append(candidate)
            normalized_map.extend(indices)
        alignment_span = None
        if lexical and (not unsupported):
            if align_parts:
                alignment_cursor += 1
                alignment_map.append(None)
            alignment_span = [alignment_cursor, alignment_cursor + len(candidate)]
            alignment_cursor += len(candidate)
            align_parts.append(candidate)
            alignment_map.extend(indices)
        status = (
            "pending"
            if alignment_span
            else "unsupported_text" if lexical else "nonlexical"
        )
        words.append(
            {
                "word_id": word_id,
                "text": match.group(),
                "char_start": match.start(),
                "char_end": match.end(),
                "original_char_span": [match.start(), match.end()],
                "normalized_text": candidate,
                "normalized_char_span": normalized_span,
                "alignment_char_span": alignment_span,
                "unsupported_characters": unsupported,
                "start": None,
                "end": None,
                "score": None,
                "status": status,
                "alignment_valid": False,
                "audio_observed_fraction": None,
                "review_flags": [],
            }
        )
    return {
        "text": text,
        "normalized_text": " ".join(normalized_parts),
        "alignment_text": " ".join(align_parts),
        "words": words,
        "normalization": {
            "version": 1,
            "rules": [
                "official_whitespace_tokens_preserved",
                "per_character_nfkc_casefold",
                "curly_apostrophe_to_ascii",
                "hyphen_to_space",
                "other_punctuation_removed",
                "unsupported_lexical_tokens_excluded_without_guessing_pronunciation",
            ],
            "normalized_char_to_original": normalized_map,
            "alignment_char_to_original": alignment_map,
            "official_text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        },
    }


def sanitize_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): sanitize_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize_json(v) for v in value]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if hasattr(value, "item"):
        return sanitize_json(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    raise TypeError(f"Cannot serialize alignment value of type {type(value).__name__}")


def project_alignment(
    prepared: dict[str, Any],
    result: dict[str, Any],
    duration: float,
    observed: list[list[float]],
    min_observed_fraction: float = 0.95,
) -> dict[str, Any]:
    """Project exact returned character order onto official word spans.

    No string-search match or interpolation is used, so repeated words cannot
    silently acquire a neighbour's timestamp. Unexpected character output fails
    closed while retaining the original words and raw segments for review.
    """
    document = json.loads(json.dumps(prepared, ensure_ascii=False))
    segments = sanitize_json(result.get("segments", []))
    characters = [char for segment in segments for char in segment.get("chars") or []]
    returned_text = "".join((str(char.get("char", "")) for char in characters))
    exact_character_mapping = returned_text == prepared["alignment_text"]
    words = document["words"]
    for word in words:
        span = word["alignment_char_span"]
        if span is None:
            continue
        if result.get("alignment_skip_reason") == "no_observed_audio":
            word["status"] = "unobserved_audio"
            continue
        if result.get("waveform_quality", {}).get("strict_numeric_silence"):
            word["status"] = "silent_audio"
            continue
        if not exact_character_mapping:
            word["status"] = "character_mapping_failed"
            continue
        selected = [
            (pos, characters[pos])
            for pos in range(*span)
            if prepared["alignment_text"][pos] != " "
        ]
        timestamps = []
        scores = []
        for _, char in selected:
            start, end = (
                finite_number(char.get("start")),
                finite_number(char.get("end")),
            )
            if start is not None and end is not None and (end > start):
                timestamps.append((start, end))
            score = finite_number(char.get("score"))
            if score is not None:
                scores.append(score)
        word["timed_character_count"] = len(timestamps)
        word["expected_character_count"] = len(selected)
        if len(timestamps) != len(selected) or not timestamps:
            word["status"] = "unaligned_characters"
            continue
        start, end = (min((t[0] for t in timestamps)), max((t[1] for t in timestamps)))
        score = sum(scores) / len(scores) if scores else None
        word["candidate_start"], word["candidate_end"] = (start, end)
        word["score"] = score
        if start < 0 or end > duration + 1e-06 or end <= start:
            word["status"] = "invalid_bounds"
            continue
        if any(
            (
                timestamps[i + 1][0] < timestamps[i][0] - 1e-06
                for i in range(len(timestamps) - 1)
            )
        ):
            word["status"] = "nonmonotonic_characters"
            continue
        coverage = interval_coverage(start, end, observed)
        word["audio_observed_fraction"] = coverage
        character_coverage = [interval_coverage(a, b, observed) for a, b in timestamps]
        word["minimum_character_observed_fraction"] = min(character_coverage)
        if (
            coverage < min_observed_fraction
            or min(character_coverage) < min_observed_fraction
        ):
            word["status"] = "unobserved_audio"
            continue
        word.update(start=start, end=end, status="aligned", alignment_valid=True)
        if end - start < 0.02:
            word["review_flags"].append("duration_under_20ms")
        if end - start > 2.0:
            word["review_flags"].append("duration_over_2s")
        if coverage < 1.0 - 1e-06:
            word["review_flags"].append("partially_unobserved_audio")
        if score is None:
            word["review_flags"].append("score_unavailable")
        if any((char.get("ctc_end_at_input_boundary") for _, char in selected)):
            word["review_flags"].append(
                "ctc_path_reaches_input_end_requires_boundary_review"
            )
    previous = None
    for word in words:
        if not word["alignment_valid"]:
            continue
        if previous is not None and word["start"] < previous["end"] - 0.002:
            for conflicting in (previous, word):
                conflicting.update(
                    start=None,
                    end=None,
                    status="overlapping_words",
                    alignment_valid=False,
                )
            previous = None
        else:
            previous = word
    intervals = merge_intervals(
        [[w["start"], w["end"]] for w in words if w["alignment_valid"]], duration
    )
    gaps = []
    cursor = 0.0
    for start, end in intervals + [[duration, duration]]:
        if start > cursor + 1e-09:
            coverage = interval_coverage(cursor, start, observed)
            gaps.append(
                {
                    "start": cursor,
                    "end": start,
                    "duration_s": start - cursor,
                    "status": "unassigned",
                    "speech_present": None,
                    "audio_observed_fraction": coverage,
                    "eligible_for_nonverbal_review": start - cursor >= 0.3,
                }
            )
        cursor = max(cursor, end)
    eligible = [w for w in words if w["status"] != "nonlexical"]
    aligned_count = sum((w["alignment_valid"] for w in words))
    status = (
        ("complete" if aligned_count == len(eligible) else "partial")
        if eligible
        else "no_lexical_text"
    )
    if eligible and aligned_count == 0:
        status = "unaligned"
    document.update(
        {
            "status": status,
            "duration_s": duration,
            "segments": segments,
            "gaps": gaps,
            "word_count": len(words),
            "aligned_word_count": aligned_count,
            "unaligned_word_count": len(eligible) - aligned_count,
            "exact_character_mapping": exact_character_mapping,
            "audio_observed_intervals": observed,
            "score_semantics": "uncalibrated_aligner_score_not_boundary_correctness_probability",
            "timestamp_mapping": sanitize_json(result.get("timestamp_mapping")),
            "waveform_quality": sanitize_json(result.get("waveform_quality")),
            "ctc_input": sanitize_json(result.get("ctc_input")),
        }
    )
    if eligible and result.get("alignment_skip_reason"):
        document["reason"] = result["alignment_skip_reason"]
    elif eligible and result.get("waveform_quality", {}).get("strict_numeric_silence"):
        document["reason"] = "silent_audio"
    return document


def read_media(sample_dir: Path) -> tuple[dict[str, Any], float, list[list[float]]]:
    media = json.loads((sample_dir / "media.json").read_text(encoding="utf-8"))
    wav = sample_dir / "audio16k.wav"
    try:
        with wave.open(str(wav), "rb") as stream:
            rate, channels, samples = (
                stream.getframerate(),
                stream.getnchannels(),
                stream.getnframes(),
            )
            if stream.getcomptype() != "NONE":
                raise ValueError("audio16k.wav must be uncompressed PCM")
    except wave.Error:
        import soundfile as sf

        info = sf.info(wav)
        if info.format not in ("WAV", "WAVEX", "RF64"):
            raise ValueError("Prepared audio must be a PCM WAV container")
        rate, channels, samples = (info.samplerate, info.channels, info.frames)
    if rate != SAMPLE_RATE or channels != 1:
        raise ValueError(
            "audio16k.wav must already be mono 16000 Hz on the common clock"
        )
    duration = samples / SAMPLE_RATE
    if samples <= 0:
        raise ValueError("Prepared audio contains no samples")
    if (
        media.get("audio_rate_hz") != SAMPLE_RATE
        or int(media.get("audio_samples", -1)) != samples
    ):
        raise ValueError("Prepared WAV and media.json sample rate/count disagree")
    media_duration = finite_number(media.get("duration_s"))
    if media_duration is None or abs(media_duration - duration) > 0.002:
        raise ValueError(
            "Prepared WAV must cover media.json duration_s on the public time axis"
        )
    if "audio_observed_intervals" not in media:
        raise ValueError(
            "media.json must distinguish observed audio from inserted padding"
        )
    observed = merge_intervals(media["audio_observed_intervals"], duration)
    return (media, duration, observed)


@contextmanager
def no_timestamp_interpolation(alignment_module):
    """WhisperX 3.7.4 lacks the 'ignore' utility branch; enforce it explicitly.

    The CLI is single-threaded. Restore the dependency's function even on failure;
    never patch its installed files or change handling of other interpolation modes.
    """
    original = alignment_module.interpolate_nans

    def guarded(series, method="nearest"):
        return series if method == "ignore" else original(series, method=method)

    alignment_module.interpolate_nans = guarded
    try:
        yield
    finally:
        alignment_module.interpolate_nans = original


def require_verified_timestamp_source(version: str | None, source_sha256: str) -> None:
    if (
        version != VERIFIED_WHISPERX_VERSION
        or source_sha256 != VERIFIED_ALIGNMENT_SHA256
    ):
        raise RuntimeError(
            f"CTC timestamp correction requires the audited WhisperX 3.7.4 alignment.py (version={version!r}, sha256={source_sha256}); inspect unknown builds first"
        )


@contextmanager
def capture_ctc_frames(alignment_module):
    """Observe official inference/path output without modifying trellis or search.

    One official-transcript segment is supported. Captures integer half-open CTC
    frame bounds before WhisperX's incorrect D/(T-1) and millisecond rounding.
    """
    original_trellis = alignment_module.get_trellis
    original_merge = alignment_module.merge_repeats
    captured = {"trellises": [], "paths": []}

    def get_trellis(emission, *args, **kwargs):
        result = original_trellis(emission, *args, **kwargs)
        captured["trellises"].append(
            {
                "emission_frames": int(emission.size(0)),
                "trellis_rows": int(result.size(0)),
            }
        )
        return result

    def merge_repeats(path, transcript):
        result = original_merge(path, transcript)
        captured["paths"].append(
            [
                {
                    "char": item.label,
                    "start_frame": int(item.start),
                    "end_frame": int(item.end),
                    "score": float(item.score),
                }
                for item in result
            ]
        )
        return result

    alignment_module.get_trellis = get_trellis
    alignment_module.merge_repeats = merge_repeats
    try:
        yield captured
    finally:
        alignment_module.get_trellis = original_trellis
        alignment_module.merge_repeats = original_merge


def correct_ctc_timestamps(
    result: dict[str, Any],
    captured: dict[str, Any],
    text: str,
    input_samples: int,
    requested_duration: float,
) -> dict[str, Any]:
    """Allocate T real emission bins over the actual input, never clamp a word.

    This convention maps half-open boundary T to input_samples/16000. It is a
    timestamp allocation convention, not a claim of phoneme-boundary accuracy.
    """
    corrected = sanitize_json(result)
    if not captured["paths"]:
        corrected["timestamp_mapping"] = {"status": "no_ctc_path_no_correction"}
        return corrected
    if len(captured["trellises"]) != 1 or len(captured["paths"]) != 1:
        raise ValueError("Expected exactly one whole-transcript CTC alignment")
    info, path = (captured["trellises"][0], captured["paths"][0])
    frames = info["emission_frames"]
    if frames <= 1 or info["trellis_rows"] != frames or input_samples < 400:
        raise ValueError("Unexpected audited CTC shape or padded short-input case")
    segments = corrected.get("segments", [])
    characters = [char for segment in segments for char in segment.get("chars") or []]
    if (
        "".join((c.get("char", "") for c in characters)) != text
        or "".join((p["char"].replace("|", " ") for p in path)) != text
        or len(path) != len(characters)
    ):
        raise ValueError(
            "CTC frame path must match every official normalized character exactly"
        )
    seconds = input_samples / SAMPLE_RATE
    for char, item in zip(characters, path):
        start, end = (item["start_frame"], item["end_frame"])
        if not 0 <= start < end <= frames:
            raise ValueError(
                "CTC character frame bounds are outside the actual emission tensor"
            )
        char["upstream_start"], char["upstream_end"] = (
            char.get("start"),
            char.get("end"),
        )
        char.update(
            start=start * seconds / frames,
            end=end * seconds / frames,
            ctc_start_frame=start,
            ctc_end_frame=end,
            ctc_end_at_input_boundary=end == frames,
        )
    all_words = []
    for segment in segments:
        chars = segment.get("chars") or []
        lexical = [c for c in chars if not c["char"].isspace()]
        segment["upstream_start"], segment["upstream_end"] = (
            segment.get("start"),
            segment.get("end"),
        )
        segment["upstream_words"] = segment.get("words", [])
        if lexical:
            segment["start"] = min((c["start"] for c in lexical))
            segment["end"] = max((c["end"] for c in lexical))
        reconstructed = []
        for match in re.finditer("\\S+", "".join((c["char"] for c in chars))):
            selected = chars[match.start() : match.end()]
            scores = [
                c["score"]
                for c in selected
                if finite_number(c.get("score")) is not None
            ]
            reconstructed.append(
                {
                    "word": match.group(),
                    "start": min((c["start"] for c in selected)),
                    "end": max((c["end"] for c in selected)),
                    "score": sum(scores) / len(scores) if scores else None,
                }
            )
        segment["words"] = reconstructed
        all_words.extend(reconstructed)
    corrected["word_segments"] = all_words
    corrected["timestamp_mapping"] = {
        "status": "corrected_from_integer_ctc_bounds",
        "emission_frames": frames,
        "trellis_rows": info["trellis_rows"],
        "input_samples": input_samples,
        "sample_rate_hz": SAMPLE_RATE,
        "input_start_s": 0.0,
        "input_end_s": seconds,
        "upstream_formula": "round(frame_boundary * requested_duration / (T - 1), 3)",
        "upstream_requested_duration_s": requested_duration,
        "corrected_formula": "frame_boundary * (input_samples / 16000) / T",
        "convention": "T_half_open_uniform_input_allocation_bins_not_exact_phoneme_boundaries",
        "rounding_applied": False,
        "clamping_applied": False,
        "ctc_path_modified": False,
        "observation_policy": "word_and_each_character_must_meet_min_observed_fraction",
    }
    return corrected


def waveform_quality(audio) -> dict[str, Any]:
    import numpy as np

    values = np.asarray(audio, dtype=np.float64)
    peak = float(np.max(np.abs(values))) if values.size else 0.0
    rms = float(np.sqrt(np.mean(values * values))) if values.size else 0.0
    return {
        "peak_abs": peak,
        "rms": rms,
        "strict_numeric_silence": peak <= SILENT_PEAK_THRESHOLD,
        "silence_peak_threshold": SILENT_PEAK_THRESHOLD,
        "meaning": "numeric_silence_gate_only_not_speech_activity_detection",
    }


def ctc_input_range(common_samples: int, observed: list[list[float]]) -> dict[str, Any]:
    """Select a public-clock prefix using observation metadata, never signal VAD.

    Only the final unobserved tail is excluded. Leading fill, internal holes and
    all observed silence retain their original indices. Floor conservatively
    discards at most one boundary sample; no unobserved sample is appended.
    """
    if common_samples <= 0:
        raise ValueError("Common waveform must contain samples")
    common_duration = common_samples / SAMPLE_RATE
    intervals = merge_intervals(observed, common_duration)
    observed_end = intervals[-1][1] if intervals else 0.0
    samples = (
        common_samples
        if observed_end == common_duration
        else min(common_samples, math.floor(observed_end * SAMPLE_RATE))
    )
    end = samples / SAMPLE_RATE
    observed_seconds = sum((max(0.0, min(end, b) - a) for a, b in intervals if a < end))
    return {
        "policy": "remove_only_unobserved_suffix_keep_public_zero_and_internal_holes",
        "observation_source": "media.json.audio_observed_intervals",
        "common_samples": common_samples,
        "common_duration_s": common_duration,
        "observed_intervals_s": intervals,
        "observed_end_s": observed_end,
        "sample_start_in_common": 0,
        "sample_end_exclusive_in_common": samples,
        "input_samples": samples,
        "input_start_s": 0.0,
        "input_end_s": end,
        "sample_rate_hz": SAMPLE_RATE,
        "removed_suffix_samples": common_samples - samples,
        "removed_suffix_s": common_duration - end,
        "unobserved_tail_s": common_duration - observed_end,
        "observed_boundary_quantization_discard_s": observed_end - end,
        "retained_unobserved_prefix_or_internal_s": max(0.0, end - observed_seconds),
        "real_silence_trimmed": False,
        "internal_holes_removed": False,
        "offset_applied_s": 0.0,
    }


class WhisperXBackend:

    def __init__(self, model_name: str, device: str, model_dir: Path | None = None):
        import torch
        import whisperx
        from whisperx import alignment as alignment_module

        self.whisperx = whisperx
        self.alignment_module = alignment_module
        source_file = Path(inspect.getfile(alignment_module))
        source_hash = sha256_file(source_file)
        require_verified_timestamp_source(
            importlib.metadata.version("whisperx"), source_hash
        )
        self.device = (
            ("cuda" if torch.cuda.is_available() else "cpu")
            if device == "auto"
            else device
        )
        if self.device.startswith("cuda") and (not torch.cuda.is_available()):
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
        self.model, self.metadata = whisperx.load_align_model(
            language_code="en",
            device=self.device,
            model_name=model_name,
            model_dir=str(model_dir) if model_dir else None,
        )
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        dictionary = self.metadata.get("dictionary")
        if self.metadata.get("language") != "en" or not isinstance(dictionary, dict):
            raise RuntimeError("WhisperX returned invalid English alignment metadata")
        self.dictionary = set(dictionary)
        state_hash = hashlib.sha256()
        for name, tensor in sorted(self.model.state_dict().items()):
            array = tensor.detach().cpu().contiguous().numpy()
            state_hash.update(name.encode("utf-8"))
            state_hash.update(str(array.dtype).encode("ascii"))
            state_hash.update(str(array.shape).encode("ascii"))
            state_hash.update(array.tobytes())
        self.manifest = {
            "tool": "whisperx",
            "mode": "known_transcript_forced_alignment",
            "asr_executed": False,
            "packages": package_versions(),
            "requested_model": model_name,
            "backend_type": self.metadata.get("type"),
            "model_class": type(self.model).__name__,
            "language": "en",
            "device": self.device,
            "loaded_state_dict_sha256": state_hash.hexdigest(),
            "state_hash_definition": "sorted_name_dtype_shape_contiguous_cpu_tensor_bytes",
            "dictionary_sha256": digest_json(dictionary),
            "whisperx_alignment_source_sha256": sha256_file(source_file),
            "resolved_hf_revision": getattr(
                getattr(self.model, "config", None), "_commit_hash", None
            ),
            "interpolate_method": "ignore",
            "return_char_alignments": True,
            "compatibility_shim": "ignore_interpolation_returns_original_series_including_missing_values",
            "waveform_loader": "soundfile_float32_no_resampling_no_amplitude_normalization",
            "ctc_timestamp_fix": {
                "audited_version": VERIFIED_WHISPERX_VERSION,
                "audited_source_sha256": source_hash,
                "upstream_source": "https://github.com/m-bain/whisperX/blob/v3.7.4/whisperx/alignment.py",
                "reason": "T-row trellis and half-open end T require D/T, not D/(T-1)",
                "method": "capture_official_integer_path_and_rebuild_all_character_times_without_clamping",
            },
            "strict_silence_peak_threshold": SILENT_PEAK_THRESHOLD,
            "ctc_input_policy": "prefix_0_to_floor_last_observed_audio_end_no_VAD_no_internal_hole_removal",
        }
        if self.metadata.get("type") == "torchaudio":
            import torchaudio

            bundle = getattr(torchaudio.pipelines, model_name)
            artifact = getattr(bundle, "_path", None)
            self.manifest["torchaudio_bundle_artifact"] = artifact
            if artifact:
                cached = (
                    model_dir or Path(torch.hub.get_dir()) / "checkpoints"
                ) / Path(artifact).name
                if cached.is_file():
                    self.manifest["checkpoint_file"] = str(cached.resolve())
                    self.manifest["checkpoint_file_sha256"] = sha256_file(cached)

    def align(
        self, text: str, wav: Path, duration: float, observed: list[list[float]]
    ) -> dict[str, Any]:
        import torch
        import numpy as np
        import soundfile as sf

        audio, rate = sf.read(wav, dtype="float32", always_2d=False)
        if (
            rate != SAMPLE_RATE
            or audio.ndim != 1
            or len(audio) != round(duration * SAMPLE_RATE)
        ):
            raise ValueError(
                "Prepared audio sample rate/count/channels changed after media validation"
            )
        if not np.isfinite(audio).all():
            raise ValueError("Prepared waveform contains nonfinite samples")
        quality = waveform_quality(audio)
        input_range = ctc_input_range(len(audio), observed)
        input_samples = input_range["input_samples"]
        if input_samples == 0:
            return {
                "segments": [],
                "word_segments": [],
                "waveform_quality": quality,
                "ctc_input": input_range,
                "alignment_skip_reason": "no_observed_audio",
            }
        if not text or quality["strict_numeric_silence"]:
            return {
                "segments": [],
                "word_segments": [],
                "waveform_quality": quality,
                "ctc_input": input_range,
            }
        if input_samples < 400:
            raise ValueError(
                "CTC timestamp correction does not support input shorter than 400 samples"
            )
        audio = audio[:input_samples]
        segment_end = input_samples / SAMPLE_RATE
        if int(segment_end * SAMPLE_RATE) < input_samples:
            segment_end = math.nextafter(segment_end, math.inf)
        if int(segment_end * SAMPLE_RATE) != input_samples:
            raise ValueError(
                "Cannot represent the exact upstream waveform slice endpoint"
            )
        input_range["whisperx_segment_end_s"] = segment_end
        input_range["segment_end_float_roundtrip_guard_s"] = (
            segment_end - input_samples / SAMPLE_RATE
        )
        with torch.inference_mode(), no_timestamp_interpolation(
            self.alignment_module
        ), capture_ctc_frames(self.alignment_module) as captured:
            result = self.whisperx.align(
                [{"start": 0.0, "end": segment_end, "text": text}],
                self.model,
                self.metadata,
                audio,
                self.device,
                interpolate_method="ignore",
                return_char_alignments=True,
            )
        result = correct_ctc_timestamps(
            result, captured, text, input_samples, segment_end
        )
        result["waveform_quality"] = quality
        result["ctc_input"] = input_range
        return result


def load_manifest(
    path: Path, requested_ids: set[str] | None = None
) -> list[dict[str, Any]]:
    records, seen = ([], set())
    with path.open(encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            key = record.get("sample_key", "")
            if not re.fullmatch("[A-Za-z0-9_-]+", key) or key in seen:
                raise ValueError(
                    f"Unsafe or duplicate sample_key at manifest line {line_number}"
                )
            seen.add(key)
            for field in (
                "sample_id",
                "video_id",
                "clip_id",
                "media_path",
                "text",
                "duration_s",
            ):
                if field not in record:
                    raise ValueError(f"Missing manifest field {field} for {key}")
            if not isinstance(record["text"], str):
                raise ValueError(f"Manifest text must be a string for {key}")
            if requested_ids is None or key in requested_ids:
                records.append(
                    {
                        k: record[k]
                        for k in (
                            "sample_key",
                            "sample_id",
                            "video_id",
                            "clip_id",
                            "media_path",
                            "text",
                            "duration_s",
                        )
                    }
                )
    if requested_ids and requested_ids - seen:
        raise ValueError(f"Unknown --ids: {sorted(requested_ids - seen)}")
    return records


def validate_saved(
    document: dict[str, Any],
    fingerprint: str,
    text: str,
    retry_incomplete: bool = False,
) -> bool:
    if (
        document.get("schema_version") != SCHEMA_VERSION
        or document.get("input_fingerprint") != fingerprint
    ):
        return False
    if document.get("status") in (None, "error") or document.get("text") != text:
        return False
    if retry_incomplete and document.get("status") not in (
        "complete",
        "no_lexical_text",
    ):
        return False
    words = document.get("words", [])
    if len(words) != len(list(re.finditer("\\S+", text))) or document.get(
        "word_count"
    ) != len(words):
        return False
    if not document.get("model_manifest", {}).get("loaded_state_dict_sha256"):
        return False
    duration = finite_number(document.get("duration_s"))
    if duration is None:
        return False
    for word, match in zip(words, re.finditer("\\S+", text)):
        if word.get("text") != match.group() or word.get("original_char_span") != [
            match.start(),
            match.end(),
        ]:
            return False
        if word.get("alignment_valid"):
            start, end = (
                finite_number(word.get("start")),
                finite_number(word.get("end")),
            )
            if (
                start is None
                or end is None
                or (not 0 <= start < end <= duration + 1e-06)
            ):
                return False
    return True


def run(args: argparse.Namespace) -> int:
    manifest = Path(args.manifest).resolve()
    data_root, output = (Path(args.data_root).resolve(), Path(args.output).resolve())
    requested = (
        {s.strip() for s in args.ids.split(",") if s.strip()} if args.ids else None
    )
    records = load_manifest(manifest, requested)
    settings = {
        "model_name": args.model_name,
        "device": args.device,
        "min_observed_fraction": args.min_observed_fraction,
        "interpolate_method": "ignore",
        "return_char_alignments": True,
        "module_sha256": sha256_file(Path(__file__)),
        "packages": package_versions(),
    }
    backend = None
    initialization_error = None
    counts = {"processed": 0, "skipped": 0, "partial": 0, "errors": 0}
    for record in records:
        key = record["sample_key"]
        sample_dir = output / key
        destination = sample_dir / "alignment.json"
        fingerprint = None
        started = time.monotonic()
        try:
            source = (data_root / record["media_path"].replace("\\", "/")).resolve()
            source.relative_to(data_root)
            if not source.is_file():
                raise FileNotFoundError(f"Manifest media file is missing for {key}")
            media, duration, observed = read_media(sample_dir)
            if media.get("sample_key") != key or media.get("status") in (
                "error",
                "failed",
            ):
                raise ValueError(
                    "media.json is not a successful preparation of this sample"
                )
            provenance = {
                "sample_key": key,
                "sample_id": record["sample_id"],
                "video_id": record["video_id"],
                "clip_id": record["clip_id"],
                "media_path": record["media_path"],
                "audio16k_sha256": sha256_file(sample_dir / "audio16k.wav"),
                "media_json_sha256": sha256_file(sample_dir / "media.json"),
                "clock": "prepared_audio_sample_zero_is_public_time_zero",
                "offset_applied_by_aligner_s": 0.0,
            }
            fingerprint = digest_json(
                {"record": record, "provenance": provenance, "settings": settings}
            )
            if destination.is_file() and (not args.force):
                try:
                    saved = json.loads(destination.read_text(encoding="utf-8"))
                    if validate_saved(
                        saved, fingerprint, record["text"], args.retry_incomplete
                    ):
                        counts["skipped"] += 1
                        print(
                            json.dumps(
                                {"sample_key": key, "status": "skipped_valid_output"}
                            ),
                            flush=True,
                        )
                        continue
                except (ValueError, TypeError, KeyError):
                    pass
            if backend is None:
                if initialization_error is not None:
                    raise RuntimeError(
                        f"Alignment backend initialization previously failed: {initialization_error}"
                    )
                try:
                    backend = WhisperXBackend(
                        args.model_name,
                        args.device,
                        (
                            Path(args.model_cache_dir).resolve()
                            if args.model_cache_dir
                            else None
                        ),
                    )
                    atomic_json(
                        output / "alignment_model_manifest.json", backend.manifest
                    )
                except Exception as error:
                    initialization_error = f"{type(error).__name__}: {error}"
                    raise
            prepared = prepare_transcript(record["text"], backend.dictionary)
            result = backend.align(
                prepared["alignment_text"],
                sample_dir / "audio16k.wav",
                duration,
                observed,
            )
            document = project_alignment(
                prepared, result, duration, observed, args.min_observed_fraction
            )
            document.update(
                {
                    "schema_version": SCHEMA_VERSION,
                    "sample_key": key,
                    "sample_id": record["sample_id"],
                    "input_fingerprint": fingerprint,
                    "provenance": provenance,
                    "settings": settings,
                    "model_manifest": backend.manifest,
                    "elapsed_s": time.monotonic() - started,
                }
            )
            atomic_json(destination, document)
            counts["processed"] += 1
            if document["status"] not in ("complete", "no_lexical_text"):
                counts["partial"] += 1
            print(
                json.dumps(
                    {
                        "sample_key": key,
                        "status": document["status"],
                        "words": document["word_count"],
                        "aligned": document["aligned_word_count"],
                    }
                ),
                flush=True,
            )
        except Exception as error:
            prepared = prepare_transcript(record["text"])
            for word in prepared["words"]:
                if word["status"] == "pending":
                    word["status"] = "not_attempted_due_to_error"
            prepared.update(
                {
                    "schema_version": SCHEMA_VERSION,
                    "sample_key": key,
                    "sample_id": record["sample_id"],
                    "status": "error",
                    "word_count": len(prepared["words"]),
                    "aligned_word_count": 0,
                    "segments": [],
                    "gaps": [],
                    "input_fingerprint": fingerprint,
                    "error": {"type": type(error).__name__, "message": str(error)},
                    "settings": settings,
                    "elapsed_s": time.monotonic() - started,
                }
            )
            atomic_json(destination, prepared)
            counts["errors"] += 1
            print(
                json.dumps({"sample_key": key, "status": "error", "error": str(error)}),
                flush=True,
            )
    print(json.dumps({"alignment_summary": counts}), flush=True)
    return 1 if counts["errors"] else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--ids", help="Comma-separated sample_key values")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--model-cache-dir")
    parser.add_argument(
        "--min-observed-fraction",
        type=float,
        default=0.95,
        help="Engineering validity threshold, not a calibrated probability",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--retry-incomplete", action="store_true")
    args = parser.parse_args(argv)
    if not 0 <= args.min_observed_fraction <= 1:
        parser.error("--min-observed-fraction must be between 0 and 1")
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
