"""Independent Qwen forced-alignment audit; NEVER replaces alignment.json.

python -m feature_extraction.align_qwen_check --manifest DATA/manifest.jsonl --output OUT     --model models/qwen_aligner --device cuda:1 --ids sample_098,sample_099,sample_100 --text-mode refined

Dependencies: official qwen-asr 0.0.6 / transformers 4.57.6; use a project venv
with inherited Torch and install explicitly with --no-deps. No ASR call occurs.
"""

from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import inspect
import json
import math
from pathlib import Path
import re
import time
import unicodedata
import numpy as np

MODEL_ID = "Qwen/Qwen3-ForcedAligner-0.6B"
SILENCE_PEAK = 1e-07


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temp.replace(path)


def normalized_chars(text):
    return "".join(
        (c for c in unicodedata.normalize("NFKC", str(text)).casefold() if c.isalnum())
    )


def transcript_mapping(official, refined=None, mode="official"):
    words = [
        dict(word_id=i, text=m.group(), char_start=m.start(), char_end=m.end())
        for i, m in enumerate(re.finditer("\\S+", official))
    ]
    if mode == "refined":
        if not refined or refined.get("text") != official:
            raise ValueError("Refined transcript must preserve the exact official text")
        spoken = refined.get("alignment_text")
        spans = refined.get("spoken_char_to_original_span")
        if spans is None:
            spans = refined.get("normalization", {}).get(
                "alignment_char_to_original_span"
            )
        if (
            not isinstance(spoken, str)
            or not isinstance(spans, list)
            or len(spans) != len(spoken)
        ):
            raise ValueError(
                "Refined text requires a recorded per-character original-span relation; cannot guess numeric expansion"
            )
    else:
        spoken, spans = (official, [[i, i + 1] for i in range(len(official))])
    canonical, owners = ([], [])
    for char, span in zip(spoken, spans):
        cleaned = normalized_chars(char)
        owner = None
        if span is not None:
            if not (
                isinstance(span, list)
                and len(span) == 2
                and all((isinstance(v, int) for v in span))
                and (0 <= span[0] < span[1] <= len(official))
            ):
                raise ValueError("Malformed expanded-text original character span")
            matches = [
                w["word_id"]
                for w in words
                if w["char_start"] <= span[0] and span[1] <= w["char_end"]
            ]
            if len(matches) == 1:
                owner = matches[0]
        canonical.extend(cleaned)
        owners.extend([owner] * len(cleaned))
    return dict(
        words=words,
        spoken_text=spoken,
        spoken_char_to_original_span=spans,
        canonical_text="".join(canonical),
        canonical_char_owner=owners,
        mode=mode,
    )


def observed_fraction(start, end, observed):
    if end <= start:
        return 0.0
    spans = sorted(((max(start, float(a)), min(end, float(b))) for a, b in observed))
    covered, last = (0.0, start)
    for a, b in spans:
        a = max(a, last)
        if b > a:
            covered += b - a
            last = b
    return min(1.0, max(0.0, covered / (end - start)))


def project_native(mapping, native, duration, observed, silent=False):
    """Exact normalized character correspondence; NEVER zip unequal word lists."""
    tokens = [normalized_chars(row.get("text", "")) for row in native]
    exact = "".join(tokens) == mapping["canonical_text"]
    by_word = {w["word_id"]: [] for w in mapping["words"]}
    mapping_issues, cursor = ([], 0)
    for i, (token, item) in enumerate(zip(tokens, native)):
        owned = (
            mapping["canonical_char_owner"][cursor : cursor + len(token)]
            if exact
            else []
        )
        cursor += len(token)
        owners = set(owned)
        if token and len(owners) == 1 and (None not in owners):
            by_word[next(iter(owners))].append(i)
        else:
            mapping_issues.append(
                dict(
                    native_index=i,
                    reason=(
                        "empty_or_cross_word_mapping"
                        if exact
                        else "normalized_transcript_mismatch"
                    ),
                )
            )
    result = []
    for word in mapping["words"]:
        wid = word["word_id"]
        indices = by_word[wid]
        expected = mapping["canonical_char_owner"].count(wid)
        assigned = sum((len(tokens[i]) for i in indices))
        row = dict(
            word,
            native_indices=indices,
            qwen_start=None,
            qwen_end=None,
            mapping_valid=False,
            timing_candidate_valid=False,
            observed_fraction=None,
            status="silent_no_time" if silent else "mapping_uncertain",
        )
        if silent:
            result.append(row)
            continue
        if exact and expected > 0 and (assigned == expected) and indices:
            row["mapping_valid"] = True
            spans = [
                [native[i].get("start_time"), native[i].get("end_time")]
                for i in indices
            ]
            valid = all(
                (
                    all(
                        (isinstance(v, (int, float)) and math.isfinite(v) for v in pair)
                    )
                    and 0 <= pair[0] < pair[1] <= duration + 1e-09
                    for pair in spans
                )
            )
            if valid and all(
                (spans[j][0] >= spans[j - 1][1] - 1e-09 for j in range(1, len(spans)))
            ):
                a, b = (spans[0][0], spans[-1][1])
                coverage = observed_fraction(a, b, observed)
                row["observed_fraction"] = coverage
                if coverage >= 0.98:
                    row.update(
                        qwen_start=a,
                        qwen_end=b,
                        timing_candidate_valid=True,
                        status="mapped_candidate_requires_review",
                    )
                else:
                    row["status"] = "candidate_crosses_unobserved_audio"
            else:
                row["status"] = "native_zero_reversed_overlapping_or_out_of_bounds"
        elif expected == 0:
            row["status"] = "no_spoken_mapping"
        result.append(row)
    previous = None
    conflicts = set()
    for i, row in enumerate(result):
        if row["timing_candidate_valid"]:
            if (
                previous is not None
                and row["qwen_start"] < result[previous]["qwen_end"] - 1e-09
            ):
                conflicts.update((previous, i))
            previous = i
    for i in conflicts:
        result[i].update(
            qwen_start=None,
            qwen_end=None,
            timing_candidate_valid=False,
            status="native_cross_word_overlap",
        )
    return (
        result,
        dict(
            exact_normalized_character_stream_match=exact,
            native_word_count=len(native),
            official_word_count=len(result),
            mapping_issues=mapping_issues,
            mapping_rule="NFKC/casefold/alphanumeric character stream, recorded original-span ownership, complete word coverage; ambiguous ownership remains null",
        ),
    )


def compare_ctc(words, refined):
    lookup = {
        (w.get("char_start"), w.get("char_end")): w
        for w in (refined or {}).get("words", [])
    }
    differences = []
    for word in words:
        ctc = lookup.get((word["char_start"], word["char_end"]))
        row = dict(
            word_id=word["word_id"],
            char_start=word["char_start"],
            char_end=word["char_end"],
            ctc_status=ctc.get("status") if ctc else None,
            ctc_alignment_valid=bool(ctc and ctc.get("alignment_valid")),
            ctc_start=ctc.get("start") if ctc else None,
            ctc_end=ctc.get("end") if ctc else None,
            qwen_start=word["qwen_start"],
            qwen_end=word["qwen_end"],
            start_difference_s=None,
            end_difference_s=None,
        )
        if ctc and ctc.get("text", ctc.get("original")) != word["text"]:
            row["comparison_status"] = "character_span_text_mismatch"
        elif row["ctc_alignment_valid"] and word["timing_candidate_valid"]:
            row.update(
                start_difference_s=word["qwen_start"] - ctc["start"],
                end_difference_s=word["qwen_end"] - ctc["end"],
                comparison_status="both_available_not_accuracy",
            )
        else:
            row["comparison_status"] = "one_or_both_unavailable"
        differences.append(row)
    paired = [x for x in differences if x["start_difference_s"] is not None]
    return (
        differences,
        {
            "paired_words": len(paired),
            "meaning": "Inter-aligner disagreement only; neither output is human reference or correctness probability",
            "median_absolute_start_difference_s": (
                float(np.median([abs(x["start_difference_s"]) for x in paired]))
                if paired
                else None
            ),
            "median_absolute_end_difference_s": (
                float(np.median([abs(x["end_difference_s"]) for x in paired]))
                if paired
                else None
            ),
        },
    )


def model_provenance(model_path, backend, requested_revision):
    hashes = {
        str(p.relative_to(model_path)): sha(p)
        for p in model_path.rglob("*")
        if p.is_file()
        and ".cache" not in p.parts
        and (p.suffix in (".json", ".safetensors", ".bin", ".txt", ".model"))
    }
    if not any((name.endswith((".safetensors", ".bin")) for name in hashes)):
        raise ValueError("Local model directory has no checkpoint weights")
    commits = set()
    for path in (model_path / ".cache" / "huggingface" / "download").rglob(
        "*.metadata"
    ):
        lines = path.read_text(encoding="utf-8").splitlines()
        if lines and re.fullmatch("[0-9a-f]{40}", lines[0]):
            commits.add(lines[0])
    config_commit = getattr(backend.model.config, "_commit_hash", None)
    if config_commit:
        commits.add(config_commit)
    if len(commits) > 1:
        raise ValueError("Local checkpoint files contain multiple HF commit revisions")
    resolved = next(iter(commits)) if commits else None
    if requested_revision and resolved and (requested_revision != resolved):
        raise ValueError(
            "Requested model revision differs from local checkpoint metadata"
        )
    versions = {}
    for package in (
        "qwen-asr",
        "transformers",
        "torch",
        "torchaudio",
        "accelerate",
        "numpy",
        "soundfile",
        "nagisa",
    ):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    source = Path(inspect.getfile(type(backend)))
    return dict(
        model_id=MODEL_ID,
        model_path=str(model_path.resolve()),
        requested_revision=requested_revision,
        resolved_revision=resolved,
        revision_note=(
            None
            if resolved
            else "HF commit metadata absent; exact local file hashes are authoritative"
        ),
        files_sha256=hashes,
        packages=versions,
        wrapper_source_sha256=sha(source),
        source_url="https://github.com/QwenLM/Qwen3-ASR",
        forced_alignment_only=True,
        asr_executed=False,
        frozen=True,
        eval=True,
        native_timestamp_note="Official qwen-asr applies its own timestamp monotonicity repair and rounds seconds to 3 decimals; native results retained without further repair",
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path("models/qwen_aligner"))
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--ids", default="")
    parser.add_argument(
        "--text-mode", choices=("official", "refined"), default="official"
    )
    parser.add_argument("--refined-name", default="alignment.refined.json")
    parser.add_argument("--revision")
    args = parser.parse_args(argv)
    if Path(args.refined_name).name != args.refined_name:
        parser.error("--refined-name must be a local filename")
    rows = [
        json.loads(s)
        for s in args.manifest.read_text(encoding="utf-8-sig").splitlines()
        if s.strip()
    ]
    selected = {x.strip() for x in args.ids.split(",") if x.strip()}
    if selected - {r["sample_key"] for r in rows}:
        parser.error("Unknown sample_key in --ids")
    backend = provenance = None
    failures = 0
    for row in rows:
        key = row["sample_key"]
        if not re.fullmatch("[A-Za-z0-9_-]+", key):
            raise ValueError("Unsafe sample_key")
        if selected and key not in selected:
            continue
        folder = args.output / key
        target = folder / "alignment.qwen.json"
        start = time.monotonic()
        try:
            import soundfile as sf

            media = json.loads((folder / "media.json").read_text(encoding="utf-8"))
            if media.get("sample_key") != key or media.get("status") != "complete":
                raise ValueError("Media identity/status mismatch")
            refined_path = folder / args.refined_name
            refined = (
                json.loads(refined_path.read_text(encoding="utf-8"))
                if refined_path.is_file()
                else None
            )
            if refined and (
                refined.get("text") != row["text"] or refined.get("sample_key") != key
            ):
                raise ValueError(
                    "Refined CTC result does not match official text/sample"
                )
            wav_path = folder / "audio16k.wav"
            audio, rate = sf.read(wav_path, dtype="float32", always_2d=False)
            if rate != 16000 or audio.ndim != 1 or (not np.isfinite(audio).all()):
                raise ValueError("Expected finite mono float32 16kHz prepared audio")
            observed = media["audio_observed_intervals"]
            last_observed = max((float(b) for _, b in observed))
            input_samples = min(
                len(audio), int(math.floor(last_observed * rate + 1e-08))
            )
            audio = np.ascontiguousarray(audio[:input_samples])
            if not len(audio):
                raise ValueError("No observed audio extent")
            duration = len(audio) / rate
            peak = float(np.max(np.abs(audio)))
            silent = peak <= SILENCE_PEAK
            mapping = transcript_mapping(
                row["text"], refined, "official" if silent else args.text_mode
            )
            native = []
            if not silent:
                if backend is None:
                    if not args.model.is_dir():
                        raise ValueError(
                            "Download the exact official model to --model before running"
                        )
                    import torch
                    from qwen_asr import Qwen3ForcedAligner

                    backend = Qwen3ForcedAligner.from_pretrained(
                        str(args.model.resolve()),
                        dtype=torch.bfloat16,
                        device_map=args.device,
                        local_files_only=True,
                    )
                    backend.model.requires_grad_(False).eval()
                    provenance = model_provenance(args.model, backend, args.revision)
                result = backend.align(
                    audio=(audio, rate), text=mapping["spoken_text"], language="English"
                )
                if len(result) != 1:
                    raise ValueError("Qwen returned unexpected batch length")
                native = [
                    dict(
                        native_index=i,
                        text=str(item.text),
                        start_time=float(item.start_time),
                        end_time=float(item.end_time),
                    )
                    for i, item in enumerate(result[0])
                ]
                if not all(
                    (
                        math.isfinite(x[field])
                        for x in native
                        for field in ("start_time", "end_time")
                    )
                ):
                    raise ValueError("Qwen returned nonfinite native timestamps")
            words, mapping_audit = project_native(
                mapping, native, duration, observed, silent
            )
            comparison, comparison_summary = compare_ctc(words, refined)
            document = dict(
                schema_version=1,
                status="skip_silent" if silent else "audit_complete",
                sample_key=key,
                sample_id=row["sample_id"],
                official_text=row["text"],
                input_text=mapping["spoken_text"],
                text_mode=mapping["mode"],
                requested_text_mode=args.text_mode,
                text_mode_note=(
                    "official inventory only; no spoken expansion attempted for silence"
                    if silent
                    else None
                ),
                words=words,
                native_words=native,
                mapping=mapping,
                mapping_audit=mapping_audit,
                comparison_to_refined_ctc=comparison,
                comparison_summary=comparison_summary,
                formal_alignment_modified=False,
                automatically_accepted_as_ground_truth=False,
                qwen_candidate_words=sum((w["timing_candidate_valid"] for w in words)),
                official_word_count=len(words),
                waveform=dict(
                    input_samples=input_samples,
                    sample_rate=rate,
                    input_time_s=[0.0, duration],
                    peak_absolute=peak,
                    rms=float(np.sqrt(np.mean(audio.astype(float) ** 2))),
                    strict_numeric_silence=peak == 0.0,
                    silent_threshold=SILENCE_PEAK,
                    audio_observed_intervals=observed,
                    trim_policy="prefix through floor(last observed audio end); no leading offset, VAD, or internal-gap removal",
                    normalization_note="Official Qwen normalizes peaks only if >1 and clips to [-1,1]; original WAV unchanged",
                ),
                provenance=dict(
                    media_json_sha256=sha(folder / "media.json"),
                    audio16k_sha256=sha(wav_path),
                    refined_ctc_sha256=sha(refined_path) if refined else None,
                    official_text_sha256=hashlib.sha256(
                        row["text"].encode()
                    ).hexdigest(),
                    implementation_sha256=sha(Path(__file__)),
                    model=provenance,
                    model_inference_skipped_for_silence=silent,
                ),
                created_utc=datetime.now(timezone.utc).isoformat(),
                elapsed_seconds=time.monotonic() - start,
            )
            write_json(target, document)
            print(
                json.dumps(
                    dict(
                        sample_key=key,
                        status=document["status"],
                        qwen_candidates=document["qwen_candidate_words"],
                        official_words=len(words),
                        paired_ctc_words=comparison_summary["paired_words"],
                    )
                ),
                flush=True,
            )
        except Exception as exc:
            failures += 1
            folder.mkdir(parents=True, exist_ok=True)
            write_json(
                target,
                dict(
                    schema_version=1,
                    sample_key=key,
                    status="audit_failed",
                    official_text=row["text"],
                    error_type=type(exc).__name__,
                    error=str(exc),
                    formal_alignment_modified=False,
                ),
            )
            print(
                json.dumps(dict(sample_key=key, status="audit_failed", error=str(exc))),
                flush=True,
            )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
