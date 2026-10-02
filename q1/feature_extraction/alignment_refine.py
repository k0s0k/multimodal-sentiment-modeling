"""Offline, auditable CTC refinement; never overwrite alignment.json.

python -m feature_extraction.alignment_refine --manifest data/manifest.jsonl   --output outputs --model-cache-dir models/ctc --device cuda [--ids sample_001]

Explicit blank states prevent leading/trailing blank runs becoming letters.
Finite numeric pronunciations are hypotheses compared on the same acoustic
emissions, not replacement transcripts or calibrated correctness probabilities.
"""

from __future__ import annotations
import argparse
import copy
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import re
import time
import numpy as np
from .align import (
    DEFAULT_MODEL,
    SAMPLE_RATE,
    WhisperXBackend,
    atomic_json,
    ctc_input_range,
    digest_json,
    interval_coverage,
    load_manifest,
    merge_intervals,
    prepare_transcript,
    read_media,
    sha256_file,
    waveform_quality,
)

ALGORITHM_VERSION = "explicit_blank_ctc_numeric_candidates_v1"
DEFAULT_REFINEMENT_MODEL = "facebook/wav2vec2-base-960h"
PRONUNCIATION_VERSION = "auditable_q1_numeric_catalog_v1"
NUMBER_READINGS = {
    "2008": [
        "two thousand eight",
        "two thousand and eight",
        "twenty oh eight",
        "two zero zero eight",
    ],
    "1500": ["one thousand five hundred", "fifteen hundred", "a thousand five hundred"],
    "100000": ["one hundred thousand", "a hundred thousand"],
    "10th": ["tenth"],
    "20000": ["twenty thousand"],
    "50": ["fifty"],
    "1989": [
        "nineteen eighty nine",
        "one thousand nine hundred eighty nine",
        "nineteen hundred and eighty nine",
        "one nine eight nine",
    ],
}
DEFAULTS = {
    "min_candidate_margin_nats": 2.0,
    "min_observed_fraction": 0.95,
    "min_character_top1_fraction": 0.4,
    "min_mean_label_logp": -5.0,
    "review_boundary_shift_s": 0.2,
    "max_candidates": 64,
}


def numeric_options(surface: str):
    core = re.search("\\d+(?:,\\d{3})*(?:st|nd|rd|th)?", surface.casefold())
    if not core or any(
        (c.isalnum() for c in surface[: core.start()] + surface[core.end() :])
    ):
        return None
    key = core.group().replace(",", "")
    if key not in NUMBER_READINGS:
        return None
    return {
        "key": key,
        "local_original_span": [core.start(), core.end()],
        "readings": NUMBER_READINGS[key],
    }


def transcript_variants(text: str, dictionary: dict[str, int], max_candidates=64):
    """Expanded characters map to a source SPAN, never to invented source letters."""
    prepared = prepare_transcript(text, set(dictionary))
    numbers = {w["word_id"]: numeric_options(w["text"]) for w in prepared["words"]}
    numbers = {i: spec for i, spec in numbers.items() if spec is not None}
    ids = sorted(numbers)
    size = math.prod((len(numbers[i]["readings"]) for i in ids))
    if size > max_candidates:
        raise ValueError(
            f"Numeric candidate count {size} exceeds explicit cap {max_candidates}"
        )
    variants = []
    for options in itertools.product(*(numbers[i]["readings"] for i in ids)):
        choices = dict(zip(ids, options))
        chars, owners, source_spans, spans = ([], [], [], {})
        for word in prepared["words"]:
            wid = word["word_id"]
            spoken = choices.get(
                wid, word["normalized_text"] if word["alignment_char_span"] else ""
            )
            if not spoken:
                continue
            if any((c != " " and c not in dictionary for c in spoken)):
                raise ValueError(
                    f"Candidate contains characters outside the acoustic dictionary: {spoken}"
                )
            if chars:
                chars.append(" ")
                owners.append(-1)
                source_spans.append(None)
            begin = len(chars)
            for j, char in enumerate(spoken):
                chars.append(char)
                owners.append(wid)
                if wid in numbers:
                    lo, hi = numbers[wid]["local_original_span"]
                    source_spans.append(
                        [word["char_start"] + lo, word["char_start"] + hi]
                    )
                else:
                    index = prepared["normalization"]["normalized_char_to_original"][
                        word["normalized_char_span"][0] + j
                    ]
                    source_spans.append(
                        [index, index + 1] if index is not None else None
                    )
            spans[wid] = [begin, len(chars)]
        spoken_text = "".join(chars)
        variants.append(
            {
                "spoken_text": spoken_text,
                "choices": choices,
                "word_spoken_spans": spans,
                "char_owner": owners,
                "spoken_char_to_original_span": source_spans,
                "tokens": [dictionary["|" if c == " " else c] for c in chars],
            }
        )
    return (prepared, numbers, variants)


def ctc_dynamic_program(log_probs, tokens, blank_id=0, with_path=False):
    """Exact standard CTC forward sum and optional Viterbi path, including blanks.

    Expanded states are blank,y0,blank,y1,...,blank. Skip two states only when
    entering a nonblank label different from the label two states earlier.
    All acoustic frames are consumed; terminal blank is a separate legal state.
    """
    x = np.asarray(log_probs, dtype=np.float64)
    tokens = np.asarray(tokens, dtype=np.int64)
    if x.ndim != 2 or not len(x) or (not np.isfinite(x).all()):
        raise ValueError("Expected a finite T by vocabulary log-probability matrix")
    if (
        not 0 <= blank_id < x.shape[1]
        or np.any((tokens < 0) | (tokens >= x.shape[1]))
        or np.any(tokens == blank_id)
    ):
        raise ValueError("Invalid CTC target/blank IDs")
    labels = np.full(2 * len(tokens) + 1, blank_id, dtype=np.int64)
    labels[1::2] = tokens
    states, frames = (len(labels), len(x))
    skip = np.zeros(states, dtype=bool)
    skip[2:] = (labels[2:] != blank_id) & (labels[2:] != labels[:-2])
    alpha = np.full(states, -np.inf)
    alpha[0] = x[0, blank_id]
    if len(tokens):
        alpha[1] = x[0, tokens[0]]
    best = alpha.copy()
    pointers = np.zeros((frames, states), dtype=np.uint8) if with_path else None
    for t in range(1, frames):
        one = np.r_[-np.inf, alpha[:-1]]
        two = np.full(states, -np.inf)
        if states > 2:
            two[2:] = alpha[:-2]
        two[~skip] = -np.inf
        alpha = np.logaddexp(np.logaddexp(alpha, one), two) + x[t, labels]
        if with_path:
            one = np.r_[-np.inf, best[:-1]]
            two = np.full(states, -np.inf)
            if states > 2:
                two[2:] = best[:-2]
            two[~skip] = -np.inf
            options = np.stack([best, one, two])
            choices = np.argmax(options, axis=0)
            pointers[t] = choices
            best = options[choices, np.arange(states)] + x[t, labels]
    ends = [states - 1] if not len(tokens) else [states - 1, states - 2]
    total = float(np.logaddexp.reduce(alpha[ends]))
    result = {"log_likelihood": total, "state_labels": labels}
    if with_path:
        last = ends[int(np.argmax(best[ends]))]
        score = float(best[last])
        if not math.isfinite(score):
            result.update(viterbi_log_score=score, path=None)
            return result
        path = np.empty(frames, dtype=np.int32)
        for t in range(frames - 1, -1, -1):
            path[t] = last
            if t:
                last -= int(pointers[t, last])
        result.update(viterbi_log_score=score, path=path)
    return result


def greedy_text(log_probs, labels, blank_id):
    previous, output = (None, [])
    for label in np.argmax(log_probs, axis=1):
        label = int(label)
        if label != previous and label != blank_id:
            output.append(labels[label])
        previous = label
    return "".join(output).replace("|", " ")


def select_candidates(log_probs, variants, numbers, blank_id):
    scores = [
        ctc_dynamic_program(log_probs, v["tokens"], blank_id)["log_likelihood"]
        for v in variants
    ]
    valid = [i for i, score in enumerate(scores) if math.isfinite(score)]
    if not valid:
        raise ValueError("No finite CTC path for any explicit transcript candidate")
    selected = max(valid, key=lambda i: scores[i])
    margins = {}
    for wid in numbers:
        by_reading = {}
        for i in valid:
            reading = variants[i]["choices"][wid]
            by_reading[reading] = max(by_reading.get(reading, -np.inf), scores[i])
        ranking = sorted(by_reading.items(), key=lambda pair: pair[1], reverse=True)
        margins[wid] = {
            "selected_reading": variants[selected]["choices"][wid],
            "ranking": [
                {"reading": r, "best_full_transcript_log_likelihood": s}
                for r, s in ranking
            ],
            "margin_nats": ranking[0][1] - ranking[1][1] if len(ranking) > 1 else None,
            "single_hypothesis": len(numbers[wid]["readings"]) == 1,
            "selection_rule": "maximum_full_transcript_CTC_forward_log_likelihood_equal_candidate_priors",
        }
    return (selected, scores, margins)


def interval_diagnostics(bounds, log_probs, cells, blank_id, path=None):
    if bounds is None or any((v is None for v in bounds)) or bounds[1] <= bounds[0]:
        return None
    weights = np.maximum(
        0, np.minimum(cells[:, 1], bounds[1]) - np.maximum(cells[:, 0], bounds[0])
    )
    denominator = weights.sum()
    if denominator <= 0:
        return None
    greedy_blank = np.argmax(log_probs, axis=1) == blank_id
    result = {
        "greedy_blank_fraction": float(weights @ greedy_blank / denominator),
        "mean_blank_posterior": float(
            weights @ np.exp(log_probs[:, blank_id]) / denominator
        ),
        "duration_s": float(bounds[1] - bounds[0]),
    }
    if path is not None:
        result["viterbi_blank_state_fraction"] = float(
            weights @ (path % 2 == 0) / denominator
        )
    return result


def signal_evidence(waveform, start, end):
    """Independent PCM measurements, not an independent speech recognizer/VAD."""
    if waveform is None:
        return None

    def rms(a, b):
        left = max(0, min(len(waveform), int(math.floor(a * SAMPLE_RATE))))
        right = max(left, min(len(waveform), int(math.ceil(b * SAMPLE_RATE))))
        values = np.asarray(waveform[left:right], dtype=np.float64)
        return float(np.sqrt(np.mean(values * values))) if len(values) else None

    return {
        "interval_rms": rms(start, end),
        "before_start_100ms_rms": rms(start - 0.1, start),
        "after_start_100ms_rms": rms(start, min(end, start + 0.1)),
        "before_end_100ms_rms": rms(max(start, end - 0.1), end),
        "after_end_100ms_rms": rms(end, end + 0.1),
        "semantics": "raw_PCM_energy_context_not_speech_probability",
    }


def refine_document(
    prepared,
    baseline,
    numbers,
    variant,
    numeric_evidence,
    log_probs,
    blank_id,
    labels,
    input_samples,
    duration,
    observed,
    settings,
    waveform=None,
):
    dp = ctc_dynamic_program(log_probs, variant["tokens"], blank_id, with_path=True)
    if dp["path"] is None:
        raise ValueError("Chosen transcript has no Viterbi path")
    path = dp["path"]
    frames = len(log_probs)
    edges = (
        np.arange(frames + 1, dtype=np.float64) * (input_samples / SAMPLE_RATE) / frames
    )
    cells = np.column_stack((edges[:-1], edges[1:]))
    greedy = np.argmax(log_probs, axis=1)
    words = copy.deepcopy(prepared["words"])
    char_evidence = []
    for i, (char, token) in enumerate(zip(variant["spoken_text"], variant["tokens"])):
        positions = np.flatnonzero(path == 2 * i + 1)
        if not len(positions):
            raise ValueError("Viterbi path omitted a target character")
        start, end = (float(edges[positions[0]]), float(edges[positions[-1] + 1]))
        selected = log_probs[positions, token]
        char_evidence.append(
            {
                "char": char,
                "owner_word_index": variant["char_owner"][i],
                "original_char_span": variant["spoken_char_to_original_span"][i],
                "start": start,
                "end": end,
                "start_frame": int(positions[0]),
                "end_frame": int(positions[-1] + 1),
                "label_state_frame_count": len(positions),
                "mean_label_logp": float(selected.mean()),
                "mean_label_minus_blank_logp": float(
                    (selected - log_probs[positions, blank_id]).mean()
                ),
                "top1_label_frame_fraction": float(np.mean(greedy[positions] == token)),
                "any_top1_label_frame": bool(np.any(greedy[positions] == token)),
                "audio_observed_fraction": interval_coverage(start, end, observed),
            }
        )
    for word in words:
        wid = word["word_id"]
        old = baseline["words"][wid]
        word["baseline"] = {
            key: old.get(key)
            for key in ("status", "start", "end", "score", "review_flags")
        }
        word["human_verified"] = False
        span = variant["word_spoken_spans"].get(wid)
        if span is None:
            word["verification_status"] = "no_supported_spoken_candidate"
            continue
        selected = [c for c in char_evidence[span[0] : span[1]] if c["char"] != " "]
        start, end = (
            min((c["start"] for c in selected)),
            max((c["end"] for c in selected)),
        )
        top1 = float(np.mean([c["any_top1_label_frame"] for c in selected]))
        mean_logp = float(np.mean([c["mean_label_logp"] for c in selected]))
        coverage = interval_coverage(start, end, observed)
        min_coverage = min((c["audio_observed_fraction"] for c in selected))
        baseline_bounds = [old.get("start"), old.get("end")]
        old_diag = interval_diagnostics(
            baseline_bounds, log_probs, cells, blank_id, path
        )
        new_diag = interval_diagnostics([start, end], log_probs, cells, blank_id, path)
        shift = (
            max(abs(start - baseline_bounds[0]), abs(end - baseline_bounds[1]))
            if all((v is not None for v in baseline_bounds))
            else None
        )
        evidence = {
            "character_top1_supported_fraction": top1,
            "mean_character_label_logp": mean_logp,
            "minimum_character_observed_fraction": min_coverage,
            "baseline_interval": old_diag,
            "refined_interval": new_diag,
            "max_boundary_change_s": shift,
            "waveform_refined": signal_evidence(waveform, start, end),
            "waveform_baseline": (
                signal_evidence(waveform, *baseline_bounds)
                if all((v is not None for v in baseline_bounds))
                else None
            ),
            "independence_warning": "greedy_and_forced_CTC_share_one_acoustic_model_not_independent_validation",
        }
        word.update(
            alignment_char_span=span,
            spoken_text=variant["spoken_text"][span[0] : span[1]],
            candidate_start=start,
            candidate_end=end,
            evidence=evidence,
            score=float(math.exp(mean_logp)),
            audio_observed_fraction=coverage,
            review_flags=[],
            verification_status="acoustic_supported_not_human_verified",
        )
        if wid in numbers:
            word["numeric_hypotheses"] = numeric_evidence[wid]
            if not numeric_evidence[wid]["single_hypothesis"] and (
                numeric_evidence[wid]["margin_nats"] is None
                or numeric_evidence[wid]["margin_nats"]
                < settings["min_candidate_margin_nats"]
            ):
                word["status"] = "ambiguous_numeric_reading"
                word["review_flags"].append("numeric_candidate_margin_insufficient")
                continue
        if (
            coverage < settings["min_observed_fraction"]
            or min_coverage < settings["min_observed_fraction"]
        ):
            word["status"] = "unobserved_audio"
            continue
        if (
            top1 < settings["min_character_top1_fraction"]
            or mean_logp < settings["min_mean_label_logp"]
        ):
            word["status"] = "weak_acoustic_evidence"
            word["verification_status"] = "not_acoustically_supported"
            continue
        if shift is not None and shift > settings["review_boundary_shift_s"]:
            word["review_flags"].append("large_change_from_baseline_requires_listening")
        if old_diag and old_diag["greedy_blank_fraction"] > 0.7:
            word["review_flags"].append("baseline_interval_blank_dominated")
        if end - start > 2 or (
            new_diag and new_diag["viterbi_blank_state_fraction"] > 0.7
        ):
            word["review_flags"].append("long_or_blank_dominated_refined_interval")
        if top1 < 0.8:
            word["review_flags"].append("incomplete_unconstrained_label_agreement")
        if wid in numbers:
            word["review_flags"].append(
                "numeric_reading_selected_by_acoustic_comparison"
            )
        word.update(start=start, end=end, alignment_valid=True, status="aligned")
    intervals = merge_intervals(
        [[w["start"], w["end"]] for w in words if w["alignment_valid"]], duration
    )
    gaps, cursor = ([], 0.0)
    for start, end in intervals + [[duration, duration]]:
        if start > cursor:
            gaps.append(
                {
                    "start": cursor,
                    "end": start,
                    "status": "unassigned",
                    "speech_present": None,
                    "audio_observed_fraction": interval_coverage(
                        cursor, start, observed
                    ),
                    "eligible_for_nonverbal_review": start - cursor >= 0.3,
                }
            )
        cursor = max(cursor, end)
    aligned = sum((w["alignment_valid"] for w in words))
    eligible = sum((w["status"] != "nonlexical" for w in words))
    normalization = {
        **prepared["normalization"],
        "version": 2,
        "baseline_alignment_char_to_original": prepared["normalization"][
            "alignment_char_to_original"
        ],
        "alignment_char_to_original": [
            s[0] if s is not None and s[1] - s[0] == 1 else None
            for s in variant["spoken_char_to_original_span"]
        ],
        "alignment_char_to_original_span": variant["spoken_char_to_original_span"],
        "expanded_numeric_mapping": "one_to_many_span_relation_not_invented_literal_characters",
    }
    return (
        {
            **prepared,
            "normalization": normalization,
            "words": words,
            "alignment_text": variant["spoken_text"],
            "spoken_char_to_original_span": variant["spoken_char_to_original_span"],
            "duration_s": duration,
            "audio_observed_intervals": observed,
            "status": (
                "complete"
                if aligned == eligible
                else "partial" if aligned else "unaligned"
            ),
            "word_count": len(words),
            "aligned_word_count": aligned,
            "unaligned_word_count": eligible - aligned,
            "gaps": gaps,
            "segments": [{"text": variant["spoken_text"], "chars": char_evidence}],
            "ctc_forward_log_likelihood": dp["log_likelihood"],
            "ctc_viterbi_log_score": dp["viterbi_log_score"],
            "diagnostic_greedy_text": greedy_text(log_probs, labels, blank_id),
            "score_semantics": "uncalibrated_acoustic_support_not_correctness_probability",
            "boundary_semantics": "first_to_last_nonblank_label_state_allocation_bins_not_exact_phoneme_edges",
            "verification_status": "requires_flagged_word_listening_no_human_validation_in_this_module",
        },
        cells,
    )


class OfflineCTC:

    def __init__(
        self, device, cache, model_name=DEFAULT_REFINEMENT_MODEL, revision="main"
    ):
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        import torch
        import torchaudio

        torch.manual_seed(20260923)
        bundle = getattr(torchaudio.pipelines, model_name, None)
        if bundle is not None:
            cache = Path(cache) if cache else Path(torch.hub.get_dir()) / "checkpoints"
            checkpoint = cache / Path(bundle._path).name
            if not checkpoint.is_file():
                raise FileNotFoundError(
                    f"Offline cached CTC checkpoint required; no download attempted: {checkpoint}"
                )
            load_name = model_name
            model_files = {checkpoint.name: sha256_file(checkpoint)}
            resolved = None
        else:
            from huggingface_hub import snapshot_download

            snapshot = Path(
                snapshot_download(
                    repo_id=model_name,
                    revision=revision,
                    cache_dir=str(cache) if cache else None,
                    local_files_only=True,
                )
            )
            load_name = str(snapshot)
            model_files = {
                p.name: sha256_file(p)
                for p in snapshot.iterdir()
                if p.is_file()
                and (
                    p.suffix in (".json", ".safetensors")
                    or p.name.startswith("pytorch_model")
                )
            }
            if not any(
                (name.endswith((".bin", ".safetensors")) for name in model_files)
            ):
                raise FileNotFoundError(
                    f"Cached HF snapshot has no CTC weights: {snapshot}"
                )
            resolved = snapshot.name
        self.backend = WhisperXBackend(
            load_name, device, Path(cache) if cache else None
        )
        self.torch = torch
        torch.set_float32_matmul_precision("highest")
        self.dictionary = self.backend.metadata["dictionary"]
        self.labels = [None] * (max(self.dictionary.values()) + 1)
        for char, index in self.dictionary.items():
            self.labels[index] = char
        self.blank = next(
            (
                self.dictionary[c]
                for c in ("[pad]", "<pad>", "-")
                if c in self.dictionary
            ),
            0,
        )
        self.manifest = {
            **self.backend.manifest,
            "refinement_algorithm": ALGORITHM_VERSION,
            "ctc_blank_id": self.blank,
            "ctc_labels": self.labels,
            "requested_model": model_name,
            "requested_revision": revision,
            "resolved_hf_revision": resolved,
            "loaded_from": load_name,
            "offline_checkpoint_verified_present": True,
            "cached_model_files_sha256": model_files,
            "asr_transcript_used": False,
            "greedy_decode_diagnostic_only": True,
            "decoder": "exact_CTC_forward_and_explicit_blank_state_Viterbi",
            "algorithm_sources": [
                "https://docs.pytorch.org/audio/2.8/tutorials/ctc_forced_alignment_api_tutorial.html"
            ],
        }

    def emissions(self, waveform):
        torch = self.torch
        signal = torch.from_numpy(waveform).unsqueeze(0).to(self.backend.device)
        with torch.inference_mode():
            output = self.backend.model(signal)
            logits = (
                output[0]
                if self.backend.metadata["type"] == "torchaudio"
                else output.logits
            )
            logp = torch.log_softmax(logits[0], dim=-1).float().cpu().numpy()
        if len(logp) != (len(waveform) - 400) // 320 + 1:
            raise ValueError("Unexpected fixed wav2vec2-base receptive field/stride")
        if logp.dtype != np.float32 or not np.isfinite(logp).all():
            raise ValueError("Invalid acoustic emission matrix")
        return logp


def save_emissions(path, log_probs, cells, input_samples):
    starts = np.arange(len(log_probs), dtype=np.float64) * 320 / SAMPLE_RATE
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            log_probs=log_probs.astype(np.float32),
            allocation_bounds_s=cells,
            conv_receptive_field_bounds_s=np.column_stack(
                (starts, starts + 400 / SAMPLE_RATE)
            ),
            conv_anchors_s=starts + 200 / SAMPLE_RATE,
            transformer_context_s=np.asarray(
                [0.0, input_samples / SAMPLE_RATE], dtype=np.float64
            ),
        )
    os.replace(temporary, path)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--ids", default="")
    p.add_argument("--device", default="cuda")
    p.add_argument("--model-cache-dir")
    p.add_argument("--force", action="store_true")
    p.add_argument("--model-name", default=DEFAULT_REFINEMENT_MODEL)
    p.add_argument("--model-revision", default="main")
    for name, default in DEFAULTS.items():
        p.add_argument(
            "--" + name.replace("_", "-"),
            default=default,
            type=int if isinstance(default, int) else float,
        )
    a = p.parse_args(argv)
    settings = {key: getattr(a, key) for key in DEFAULTS}
    settings.update(model_name=a.model_name, model_revision=a.model_revision)
    ids = set(a.ids.split(",")) if a.ids else None
    rows = load_manifest(Path(a.manifest), ids)
    output = Path(a.output)
    backend = None
    failures = 0
    implementation_hash = sha256_file(Path(__file__))
    for row in rows:
        key = row["sample_key"]
        directory = output / key
        target = directory / "alignment.refined.json"
        started = time.monotonic()
        try:
            media, duration, observed = read_media(directory)
            baseline = json.loads(
                (directory / "alignment.json").read_text(encoding="utf-8")
            )
            if baseline.get("text") != row["text"] or baseline.get("sample_key") != key:
                raise ValueError(
                    "Baseline alignment does not match the original manifest"
                )
            provenance = {
                "media_json_sha256": sha256_file(directory / "media.json"),
                "audio16k_sha256": sha256_file(directory / "audio16k.wav"),
                "baseline_alignment_sha256": sha256_file(directory / "alignment.json"),
                "official_text_sha256": hashlib.sha256(
                    row["text"].encode("utf-8")
                ).hexdigest(),
                "implementation_sha256": implementation_hash,
            }
            provenance.update(
                {
                    k: row[k]
                    for k in (
                        "sample_key",
                        "sample_id",
                        "video_id",
                        "clip_id",
                        "media_path",
                    )
                }
            )
            provenance.update(
                clock="prepared_audio_sample_zero_is_public_time_zero",
                offset_applied_by_aligner_s=0.0,
            )
            signature = digest_json(
                {
                    "provenance": provenance,
                    "settings": settings,
                    "catalog": NUMBER_READINGS,
                }
            )
            if target.exists() and (not a.force):
                old = json.loads(target.read_text(encoding="utf-8"))
                proof = directory / "alignment_refinement_emissions.npz"
                if (
                    old.get("input_signature") == signature
                    and old.get("status") != "error"
                    and (
                        old.get("reason") in ("silent_audio", "no_observed_audio")
                        or (
                            proof.is_file()
                            and old.get("emissions_sha256") == sha256_file(proof)
                        )
                    )
                ):
                    print(
                        json.dumps({"sample_key": key, "status": "cached"}), flush=True
                    )
                    continue
            import soundfile as sf

            waveform, rate = sf.read(
                directory / "audio16k.wav", dtype="float32", always_2d=False
            )
            if (
                rate != SAMPLE_RATE
                or waveform.ndim != 1
                or (not np.isfinite(waveform).all())
            ):
                raise ValueError("Expected finite 16kHz mono prepared PCM")
            input_range = ctc_input_range(len(waveform), observed)
            quality = waveform_quality(waveform)
            if quality["strict_numeric_silence"] or not input_range["input_samples"]:
                result = copy.deepcopy(baseline)
                reason = (
                    "no_observed_audio"
                    if not input_range["input_samples"]
                    else "silent_audio"
                )
                for word in result["words"]:
                    word.update(
                        start=None,
                        end=None,
                        alignment_valid=False,
                        status=reason,
                        human_verified=False,
                    )
                result.update(
                    status="unaligned",
                    reason=reason,
                    aligned_word_count=0,
                    unaligned_word_count=len(result["words"]),
                    waveform_quality=quality,
                )
            else:
                if backend is None:
                    backend = OfflineCTC(
                        a.device, a.model_cache_dir, a.model_name, a.model_revision
                    )
                    atomic_json(
                        output / "alignment_refinement_model_manifest.json",
                        backend.manifest,
                    )
                prepared, numbers, variants = transcript_variants(
                    row["text"], backend.dictionary, settings["max_candidates"]
                )
                waveform = waveform[: input_range["input_samples"]]
                log_probs = backend.emissions(waveform)
                selected, scores, margins = select_candidates(
                    log_probs, variants, numbers, backend.blank
                )
                result, cells = refine_document(
                    prepared,
                    baseline,
                    numbers,
                    variants[selected],
                    margins,
                    log_probs,
                    backend.blank,
                    backend.labels,
                    len(waveform),
                    duration,
                    observed,
                    settings,
                    waveform,
                )
                proof = directory / "alignment_refinement_emissions.npz"
                save_emissions(proof, log_probs, cells, len(waveform))
                result.update(
                    model_manifest=backend.manifest,
                    emissions_sha256=sha256_file(proof),
                    waveform_quality=quality,
                    selected_candidate_index=selected,
                    candidate_comparison=[
                        {
                            "index": i,
                            "spoken_text": v["spoken_text"],
                            "numeric_choices": v["choices"],
                            "CTC_log_likelihood": (
                                score if math.isfinite(score) else None
                            ),
                            "log_likelihood_per_frame": (
                                score / len(log_probs) if math.isfinite(score) else None
                            ),
                        }
                        for i, (v, score) in enumerate(zip(variants, scores))
                    ],
                )
            result.update(
                schema_version=1,
                sample_key=key,
                sample_id=row["sample_id"],
                refinement_algorithm=ALGORITHM_VERSION,
                pronunciation_catalog_version=PRONUNCIATION_VERSION,
                settings=settings,
                provenance=provenance,
                input_signature=signature,
                input_fingerprint=signature,
                ctc_input=input_range,
                labels_used=False,
                elapsed_s=time.monotonic() - started,
            )
            atomic_json(target, result)
            print(
                json.dumps(
                    {
                        "sample_key": key,
                        "status": result["status"],
                        "aligned": result["aligned_word_count"],
                        "words": result["word_count"],
                    }
                ),
                flush=True,
            )
        except Exception as error:
            failures += 1
            failure = {
                "sample_key": key,
                "status": "error",
                "error_type": type(error).__name__,
                "message": str(error),
                "implementation_sha256": implementation_hash,
            }
            atomic_json(directory / "alignment.refine.error.json", failure)
            atomic_json(target, failure)
            print(
                json.dumps({"sample_key": key, "status": "error", "error": str(error)}),
                flush=True,
            )
    return int(failures > 0)


if __name__ == "__main__":
    raise SystemExit(main())
