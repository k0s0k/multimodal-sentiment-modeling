"""Build conservative alignment.final.json candidates; NEVER promotes alignment.json.

Segment masking is a valid terminal representation of uncertainty. Official
inventory, text and labels are immutable. Qwen agreement is not ground truth.
"""

from __future__ import annotations
import argparse
from collections import Counter
import copy
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import unicodedata

DEFAULT_POLICY = {
    "schema_version": 1,
    "policy_id": "conservative_segment_gate_v1",
    "normalized_token_edit_min": 0.8,
    "ctc_supported_fraction_max": 0.5,
    "minimum_lexical_words": 5,
    "language_score_advisory_min": 0.9,
    "qwen_boundary_review_s": 0.25,
    "qwen_rescue_unsupported": False,
    "qwen_disagreement_action": "flag_only",
    "prior_suspect_samples": [
        f"sample_{i:03}"
        for i in [23, 24, *range(54, 61), 66, 67, 70, 71, 72, 98, 99, 100]
    ],
}
RATIONALE = "先在片段层面检查官方文本能否与现有音视频建立逐词对应，避免强制对齐把常见词偶合当成证据。人工确认不一致/非英语、严格静音和无观测片段保留原词但关闭词时间；既有可疑清单，或独立tiny转写与官方文本的归一化词编辑距离≥0.80且CTC支持率≤0.50（至少5个词），标为未核实配对并关闭词时间。这些阈值未经准确率校准，只是保守屏蔽规则，不证明官方错配或模型识别语言为真。其余片段仅保留refined CTC已支持的合法时间；Qwen不救回不支持词、不替换边界。两模型任一边界差>250ms加复核标记，并另存双端≤250ms的一致性mask；该mask不等于正确率或人工确认。未知时间是合法终态，独立音视频观察及文本表示仍可保留，不能将真实静音等同于未观测。"


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def digest_json(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
    ).hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def atomic_json(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temp.replace(path)


def tokens(text):
    return re.findall("[^\\W_]+", unicodedata.normalize("NFKC", text).casefold())


def normalized_edit(left, right):
    previous = list(range(len(right) + 1))
    for i, a in enumerate(left, 1):
        current = [i]
        for j, b in enumerate(right, 1):
            current.append(
                min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (a != b))
            )
        previous = current
    return previous[-1] / max(len(left), len(right), 1)


def lexical(word):
    return any((c.isalnum() for c in word.get("text", word.get("original", ""))))


def valid_bounds(word, duration):
    a, b = (word.get("start"), word.get("end"))
    return bool(
        word.get("alignment_valid")
        and all((isinstance(v, (int, float)) and math.isfinite(v) for v in (a, b)))
        and (0 <= a < b <= duration + 1e-06)
    )


def coverage(a, b, observed):
    end, total = (a, 0.0)
    for lo, hi in sorted(observed):
        lo, hi = (max(a, lo, end), min(b, hi))
        if hi > lo:
            total += hi - lo
            end = hi
    return max(0.0, min(1.0, total / (b - a))) if b > a else 0.0


def decide_segment(refined, language, manual, policy):
    words = [w for w in refined["words"] if lexical(w)]
    supported = sum((valid_bounds(w, refined["duration_s"]) for w in words))
    fraction = supported / len(words) if words else 0.0
    official, asr = (tokens(refined["text"]), tokens(language.get("text", "")))
    edit = normalized_edit(official, asr)
    char_edit = normalized_edit(list("".join(official)), list("".join(asr)))
    lang, score = (language.get("language"), language.get("model_language_score"))
    verdict = (manual or {}).get("verdict")
    flags = []
    if lang and lang != "en":
        flags.append(
            "model_only_non_english_diagnostic_not_confirmed"
            if isinstance(score, (int, float))
            and score >= policy["language_score_advisory_min"]
            else "low_score_language_output_unreliable"
        )
    segments = language.get("segments", [])
    if any((s.get("no_speech_prob", 0) >= 0.8 for s in segments)):
        flags.append("ASR_high_no_speech_output_may_be_hallucinated")
    if len(asr) >= 12 and len(set(asr)) / len(asr) < 0.25:
        flags.append("ASR_repetitive_output_unreliable")
    if not asr:
        flags.append("ASR_empty_output_not_proof_of_silence")
    silent = (
        refined.get("reason") == "silent_audio"
        or refined.get("waveform_quality", {}).get("strict_numeric_silence") is True
    )
    absent = refined.get("reason") == "no_observed_audio"
    rule = (
        len(words) >= policy["minimum_lexical_words"]
        and edit >= policy["normalized_token_edit_min"]
        and (fraction <= policy["ctc_supported_fraction_max"])
    )
    prior = refined["sample_key"] in policy["prior_suspect_samples"]
    if absent:
        state, blocked = ("no_observed_audio", True)
    elif silent:
        state, blocked = ("observed_numeric_silence_no_word_alignment", True)
    elif verdict == "transcript_av_mismatch":
        state, blocked = ("human_confirmed_clip_text_av_mismatch", True)
    elif verdict == "not_english":
        state, blocked = ("human_confirmed_not_english_no_literal_word_alignment", True)
    elif verdict == "transcript_basically_consistent":
        state, blocked = ("human_clip_content_consistent_boundaries_unverified", False)
    elif rule or prior:
        state, blocked = ("unverified_pairing_masked_model_diagnostics_only", True)
    else:
        state, blocked = ("not_flagged_boundaries_not_human_verified", False)
    return dict(
        state=state,
        mask_word_alignment=blocked,
        legal_terminal_state=True,
        official_text_replaced=False,
        official_labels_changed=False,
        lexical_words=len(words),
        ctc_supported_words=supported,
        ctc_supported_fraction=fraction,
        normalized_token_edit_distance=edit,
        normalized_character_edit_distance=char_edit,
        diagnostic_edit_rule_triggered=rule,
        registered_prior_suspicion=prior,
        model_language=lang,
        model_language_score=score,
        model_language_is_human_confirmed=False,
        manual_verdict=verdict,
        manual_scope=(
            "clip_content_and_language_only_not_word_boundary_ground_truth"
            if manual
            else None
        ),
        flags=flags,
        thresholds_are_uncalibrated=True,
    )


def qwen_comparison(word, qwen_word, duration, threshold):
    result = dict(
        candidate_available=False,
        qwen_start=None,
        qwen_end=None,
        start_difference_s=None,
        end_difference_s=None,
        both_within_threshold=False,
        status="qwen_candidate_missing_or_invalid",
        threshold_s=threshold,
        semantics="inter_aligner_agreement_not_accuracy_or_human_verification",
    )
    if not qwen_word or not qwen_word.get("timing_candidate_valid"):
        return result
    a, b = (qwen_word.get("qwen_start"), qwen_word.get("qwen_end"))
    if (
        not all((isinstance(v, (int, float)) and math.isfinite(v) for v in (a, b)))
        or not 0 <= a < b <= duration + 1e-06
    ):
        return result
    if qwen_word.get("text") != word.get("text", word.get("original")):
        result["status"] = "qwen_character_span_text_mismatch"
        return result
    result.update(
        candidate_available=True,
        qwen_start=a,
        qwen_end=b,
        status="CTC_unavailable_not_rescued",
    )
    if valid_bounds(word, duration):
        da, db = (a - word["start"], b - word["end"])
        agreement = abs(da) <= threshold and abs(db) <= threshold
        result.update(
            start_difference_s=da,
            end_difference_s=db,
            both_within_threshold=agreement,
            status=(
                "both_available_within_threshold_not_confirmed"
                if agreement
                else "boundary_disagreement_requires_review"
            ),
        )
    return result


def finalize_document(refined, language, manual, policy, source_hashes, qwen=None):
    if refined.get("status") not in (
        "complete",
        "partial",
        "unaligned",
        "no_lexical_text",
    ):
        raise ValueError("Cannot finalize failed/incomplete refinement")
    duration = float(refined["duration_s"])
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("Invalid refined common-axis duration")
    gate = decide_segment(refined, language, manual, policy)
    qwords = {
        (w.get("char_start"), w.get("char_end")): w
        for w in (qwen or {}).get("words", [])
    }
    keep_fields = (
        "schema_version",
        "sample_key",
        "sample_id",
        "text",
        "normalized_text",
        "alignment_text",
        "normalization",
        "spoken_char_to_original_span",
        "duration_s",
        "audio_observed_intervals",
        "model_manifest",
        "settings",
        "provenance",
        "input_signature",
        "input_fingerprint",
        "ctc_input",
        "waveform_quality",
        "refinement_algorithm",
        "pronunciation_catalog_version",
        "score_semantics",
        "boundary_semantics",
    )
    result = {k: copy.deepcopy(refined[k]) for k in keep_fields if k in refined}
    source_supported = source_paired = source_agree = 0
    words = []
    compact_word_fields = (
        "word_id",
        "text",
        "char_start",
        "char_end",
        "original_char_span",
        "normalized_text",
        "normalized_char_span",
        "alignment_char_span",
        "spoken_text",
        "unsupported_characters",
        "numeric_hypotheses",
        "acoustic_evidence",
        "evidence",
        "baseline",
        "candidate_start",
        "candidate_end",
    )
    for original in refined["words"]:
        word = {
            k: copy.deepcopy(original[k]) for k in compact_word_fields if k in original
        }
        if "text" not in word:
            word["text"] = original.get("original", "")
        is_lexical = lexical(original)
        supported = is_lexical and valid_bounds(original, duration)
        comparison = qwen_comparison(
            original,
            qwords.get((word.get("char_start"), word.get("char_end"))),
            duration,
            policy["qwen_boundary_review_s"],
        )
        if supported:
            source_supported += 1
            source_paired += int(comparison["candidate_available"])
            source_agree += int(comparison["both_within_threshold"])
        keep = supported and (not gate["mask_word_alignment"])
        flags = list(original.get("review_flags", []))
        if (
            supported
            and comparison["status"] == "boundary_disagreement_requires_review"
        ):
            flags.append("boundary_review_independent_aligner_disagreement")
        if supported and (not comparison["candidate_available"]):
            flags.append("independent_qwen_candidate_unavailable_not_confirmed")
        status = (
            "nonlexical"
            if not is_lexical
            else (
                "aligned"
                if keep
                else (
                    "segment_alignment_unavailable"
                    if gate["mask_word_alignment"]
                    else original.get("status", "unaligned")
                )
            )
        )
        if status == "aligned" and (not keep):
            status = "invalid_refined_bounds"
        word.update(
            start=original["start"] if keep else None,
            end=original["end"] if keep else None,
            alignment_valid=keep,
            status=status,
            score=original.get("score") if keep else None,
            audio_observed_fraction=(
                original.get("audio_observed_fraction") if keep else None
            ),
            source_refined_status=original.get("status"),
            source_refined_alignment_valid=bool(original.get("alignment_valid")),
            source_refined_boundary={
                "start": original.get("start"),
                "end": original.get("end"),
                "semantics": "audit_only_source_CTC_before_segment_gate_not_final_available_time",
            },
            human_verified=False,
            verification_status=(
                "CTC_supported_boundaries_not_human_verified"
                if keep
                else "no_available_word_time"
            ),
            segment_gate_reason=(
                gate["state"] if gate["mask_word_alignment"] and is_lexical else None
            ),
            review_flags=sorted(set(flags)),
            qwen_comparison=comparison,
            timing_agreement_mask=bool(keep and comparison["both_within_threshold"]),
        )
        words.append(word)
    timed = [w for w in words if w["alignment_valid"]]
    if any((b["start"] < a["end"] - 1e-06 for a, b in zip(timed, timed[1:]))):
        raise ValueError("Refined word intervals overlap; refuse final candidate")
    observed = result["audio_observed_intervals"]
    gaps, cursor = ([], 0.0)
    for word in timed + [dict(start=duration, end=duration)]:
        if word["start"] > cursor:
            gaps.append(
                dict(
                    start=cursor,
                    end=word["start"],
                    status="unassigned",
                    type="unknown",
                    speech_present=None,
                    audio_observed_fraction=coverage(cursor, word["start"], observed),
                    eligible_for_nonverbal_review=word["start"] - cursor >= 0.3,
                    semantics="unassigned_is_not_nonverbal; unobserved portions remain masked",
                )
            )
        cursor = max(cursor, word["end"])
    eligible = sum((lexical(w) for w in words))
    retained = len(timed)
    agreement = sum((w["timing_agreement_mask"] for w in words))
    paired = sum(
        (
            w["alignment_valid"] and w["qwen_comparison"]["candidate_available"]
            for w in words
        )
    )
    status = (
        "no_lexical_text"
        if not eligible
        else (
            "complete"
            if retained == eligible
            else "partial" if retained else "unaligned"
        )
    )
    result.update(
        schema_version=1,
        status=status,
        words=words,
        word_count=len(words),
        aligned_word_count=retained,
        unaligned_word_count=eligible - retained,
        gaps=gaps,
        segments=[],
        labels_used=False,
        human_word_boundaries_verified=False,
        segment_gate=gate,
        timing_agreement_mask=[w["timing_agreement_mask"] for w in words],
        timing_agreement=dict(
            threshold_s=policy["qwen_boundary_review_s"],
            retained_CTC_words=retained,
            retained_with_qwen_candidate=paired,
            retained_without_qwen_candidate=retained - paired,
            retained_both_endpoints_within_threshold=agreement,
            agreement_fraction_all_retained=agreement / retained if retained else None,
            agreement_fraction_paired_retained=agreement / paired if paired else None,
            source_refined_supported_words_before_segment_gate=source_supported,
            source_refined_paired_before_segment_gate=source_paired,
            source_refined_agreed_before_segment_gate=source_agree,
            meaning="agreement is not correctness; missing candidate is not agreement; CTC boundaries are retained rather than replaced",
        ),
        verification_status="conservative_available_times_plus_explicit_unknowns_no_word_boundary_ground_truth",
        finalization=dict(
            policy=copy.deepcopy(policy),
            rationale=RATIONALE,
            source_hashes=source_hashes,
            source_input_fingerprint_retained=refined.get("input_fingerprint"),
            fingerprint_semantics="input_fingerprint identifies source CTC inference; finalization_fingerprint separately binds gating/audit inputs",
            manual_review=copy.deepcopy(manual),
            automatic_rescued_words=0,
            formal_promotion_performed=False,
            qwen_audit_status=(qwen or {}).get("status", "not_available"),
            qwen_audit_error=(qwen or {}).get("error"),
            refinement_evidence_retained_separately=True,
        ),
    )
    result["finalization_fingerprint"] = digest_json(
        dict(
            source_input_fingerprint=refined.get("input_fingerprint"),
            policy=policy,
            source_hashes=source_hashes,
            segment_gate=gate,
        )
    )
    return result


def indexed(document, name):
    rows = document.get("records", [])
    result = {r["sample_key"]: r for r in rows}
    if len(result) != len(rows):
        raise ValueError(f"Duplicate sample keys in {name}")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Existing prepared sample root containing media and alignment.refined.json",
    )
    parser.add_argument("--language-audit", type=Path, required=True)
    parser.add_argument("--manual-review", type=Path, required=True)
    parser.add_argument(
        "--qwen-root",
        type=Path,
        help="Root of sample_NNN/alignment.qwen.json; defaults to --output",
    )
    parser.add_argument("--policy", type=Path)
    parser.add_argument("--report-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    policy = copy.deepcopy(DEFAULT_POLICY)
    if args.policy:
        overrides = read_json(args.policy)
        if set(overrides) - policy.keys():
            parser.error("Unknown policy fields")
        policy.update(overrides)
    if (
        policy["qwen_rescue_unsupported"]
        or policy["qwen_disagreement_action"] != "flag_only"
    ):
        parser.error(
            "This implementation never rescues unsupported words or replaces CTC times using Qwen"
        )
    for field in (
        "normalized_token_edit_min",
        "ctc_supported_fraction_max",
        "language_score_advisory_min",
    ):
        if not 0 <= policy[field] <= 1:
            parser.error(f"Invalid {field}")
    if (
        not isinstance(policy["minimum_lexical_words"], int)
        or policy["minimum_lexical_words"] < 1
    ):
        parser.error("minimum_lexical_words must be a positive integer")
    if (
        not isinstance(policy["qwen_boundary_review_s"], (int, float))
        or not 0 < policy["qwen_boundary_review_s"] < math.inf
    ):
        parser.error("qwen_boundary_review_s must be finite and positive")
    language_doc, manual_doc = (
        read_json(args.language_audit),
        read_json(args.manual_review),
    )
    if (
        manual_doc.get("scope")
        != "clip_content_and_language_only_not_word_boundary_ground_truth"
    ):
        parser.error(
            "Manual review scope must explicitly exclude word-boundary ground truth"
        )
    language, manual = (
        indexed(language_doc, "language audit"),
        indexed(manual_doc, "manual review"),
    )
    rows = [
        json.loads(line)
        for line in args.manifest.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
    keys = [r["sample_key"] for r in rows]
    if (
        not keys
        or len(set(keys)) != len(keys)
        or set(language) != set(keys)
        or set(manual) - set(keys)
    ):
        parser.error("Manifest/audit identity sets do not match")
    if any((not re.fullmatch("[A-Za-z0-9_-]+", key) for key in keys)):
        parser.error("Unsafe sample_key")
    sources = {
        "language_audit_sha256": sha(args.language_audit),
        "manual_review_sha256": sha(args.manual_review),
        "manifest_sha256": sha(args.manifest),
        "implementation_sha256": sha(Path(__file__)),
    }
    plans, summaries, disagreements = ([], [], [])
    for row in rows:
        key, folder = (row["sample_key"], args.output / row["sample_key"])
        refined_path, media_path, wav_path = (
            folder / "alignment.refined.json",
            folder / "media.json",
            folder / "audio16k.wav",
        )
        refined, media = (read_json(refined_path), read_json(media_path))
        if (
            refined.get("sample_key") != key
            or refined.get("sample_id") != row["sample_id"]
            or refined.get("text") != row["text"]
        ):
            raise ValueError(f"{key}: refined identity/text mismatch")
        if media.get("sample_key") != key or media.get("status") != "complete":
            raise ValueError(f"{key}: media identity/status mismatch")
        provenance = refined["provenance"]
        media_hash, wav_hash = (sha(media_path), sha(wav_path))
        if (
            provenance.get("media_json_sha256") != media_hash
            or provenance.get("audio16k_sha256") != wav_hash
        ):
            raise ValueError(f"{key}: refined media/WAV provenance is stale")
        if language[key].get("audio_sha256") != wav_hash:
            raise ValueError(
                f"{key}: independent language audit used a different waveform"
            )
        if not re.fullmatch("[0-9a-f]{64}", str(refined.get("input_fingerprint", ""))):
            raise ValueError(
                f"{key}: missing source input_fingerprint needed for transfer"
            )
        matches = list(re.finditer("\\S+", row["text"]))
        if len(matches) != len(refined["words"]):
            raise ValueError(f"{key}: original word inventory changed")
        for match, word in zip(matches, refined["words"]):
            if word.get("text") != match.group() or [
                word.get("char_start"),
                word.get("char_end"),
            ] != [match.start(), match.end()]:
                raise ValueError(f"{key}: original character mapping changed")
        qpath = (args.qwen_root or args.output) / key / "alignment.qwen.json"
        qwen = read_json(qpath) if qpath.is_file() else None
        hashes = dict(
            sources,
            refined_ctc_sha256=sha(refined_path),
            media_json_sha256=media_hash,
            audio16k_sha256=wav_hash,
            qwen_audit_sha256=sha(qpath) if qwen else None,
        )
        if qwen:
            qp = qwen.get("provenance", {})
            if (
                qwen.get("sample_key") != key
                or qwen.get("official_text") != row["text"]
            ):
                raise ValueError(
                    f"{key}: Qwen audit identity/input hashes do not match refined source"
                )
            if qwen.get("status") == "audit_failed":
                if qwen.get("words") or qwen.get("native_words"):
                    raise ValueError(
                        f"{key}: failed Qwen audit must not supply usable candidates"
                    )
            elif (
                qp.get("audio16k_sha256") != wav_hash
                or qp.get("refined_ctc_sha256") != hashes["refined_ctc_sha256"]
            ):
                raise ValueError(
                    f"{key}: Qwen audit identity/input hashes do not match refined source"
                )
        doc = finalize_document(
            refined, language[key], manual.get(key), policy, hashes, qwen
        )
        plans.append((folder / "alignment.final.json", doc))
        summaries.append(
            dict(
                sample_key=key,
                sample_id=row["sample_id"],
                status=doc["status"],
                official_words=doc["word_count"],
                retained_word_times=doc["aligned_word_count"],
                segment_gate=doc["segment_gate"]["state"],
                masked=doc["segment_gate"]["mask_word_alignment"],
                **doc["timing_agreement"],
            )
        )
        for word in doc["words"]:
            comparison = word["qwen_comparison"]
            if (
                word["alignment_valid"]
                and comparison["status"] == "boundary_disagreement_requires_review"
            ):
                disagreements.append(
                    dict(
                        sample_key=key,
                        word_id=word["word_id"],
                        text=word["text"],
                        char_start=word["char_start"],
                        char_end=word["char_end"],
                        CTC_start=word["start"],
                        CTC_end=word["end"],
                        qwen_start=comparison["qwen_start"],
                        qwen_end=comparison["qwen_end"],
                        start_difference_s=comparison["start_difference_s"],
                        end_difference_s=comparison["end_difference_s"],
                    )
                )
    for path, doc in plans:
        atomic_json(path, doc)
    totals = {
        name: sum((row[name] for row in summaries))
        for name in (
            "official_words",
            "retained_word_times",
            "retained_with_qwen_candidate",
            "retained_without_qwen_candidate",
            "retained_both_endpoints_within_threshold",
            "source_refined_supported_words_before_segment_gate",
            "source_refined_paired_before_segment_gate",
            "source_refined_agreed_before_segment_gate",
        )
    }
    retained, paired = (
        totals["retained_word_times"],
        totals["retained_with_qwen_candidate"],
    )
    agreed = totals["retained_both_endpoints_within_threshold"]
    report = dict(
        schema_version=1,
        created_utc=datetime.now(timezone.utc).isoformat(),
        status="candidates_complete_not_promoted",
        samples=len(plans),
        segment_states=dict(Counter((row["segment_gate"] for row in summaries))),
        totals=totals,
        agreement_fraction_all_retained=agreed / retained if retained else None,
        agreement_fraction_paired_retained=agreed / paired if paired else None,
        source_hashes=sources,
        policy=policy,
        rationale=RATIONALE,
        records=summaries,
        unknown_is_valid_terminal_state=True,
        formal_promotion_performed=False,
        original_inventory_text_labels_unchanged=True,
        automatic_rescued_words=0,
        important_example="sample_001 Polymer/nearby words have approximately one-second CTC-Qwen disagreement; clip content consistency is not word-boundary ground truth",
    )
    args.report_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(args.report_dir / "finalization_summary.json", report)
    atomic_json(args.report_dir / "resolved_policy.json", policy)
    atomic_json(args.report_dir / "boundary_review.json", disagreements)
    with (args.report_dir / "samples.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    print(
        json.dumps(
            {
                "samples": len(plans),
                "segment_states": report["segment_states"],
                "totals": totals,
                "agreement_fraction_all_retained": report[
                    "agreement_fraction_all_retained"
                ],
                "formal_promotion_performed": False,
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
