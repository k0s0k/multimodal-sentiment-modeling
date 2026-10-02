"""Project independently verified word intervals onto the official feature grid."""

import copy
import math
import numpy as np


def project_grid(metadata, final, media, frame_pts):
    """Link raw-text offsets to supported words, preserving unknown time as null."""
    if metadata["raw_text"] != final["text"]:
        raise ValueError("Alignment text differs from the official staged text")
    duration = float(media["duration_s"])
    pts = np.asarray(frame_pts, dtype=float)
    if pts.ndim != 1 or not np.isfinite(pts).all() or np.any(np.diff(pts) <= 0):
        raise ValueError("Frame PTS must be finite and strictly increasing")
    if len(pts) and (pts[0] < -1e-06 or pts[-1] >= duration + 1e-06):
        raise ValueError("Frame PTS outside common duration")
    words = {(w["char_start"], w["char_end"]): w for w in final["words"]}
    original_words = {w["word_id"]: w for w in metadata["words"]}
    positions = []
    for token in metadata["tokens"]:
        item = copy.deepcopy(token)
        item.update(
            start_sec=None,
            end_sec=None,
            time_mapping_level="unknown",
            frame_indices=[],
            frame_pts_sec=[],
            timing_agreement_mask=False,
            review_flags=[],
            word_time_status="not_a_mapped_content_token",
        )
        word = original_words.get(token["word_id"])
        aligned = words.get((word["char_start"], word["char_end"])) if word else None
        if aligned is not None:
            if aligned["text"] != word["text"]:
                raise ValueError("Original character span has mismatched word text")
            item["word_time_status"] = aligned.get("status", "unknown")
            item["review_flags"] = list(aligned.get("review_flags", []))
        if (
            token["observed_content"]
            and token["mapping_level"] == "raw_text_exact"
            and aligned
            and aligned.get("alignment_valid")
        ):
            a, b = (aligned.get("start"), aligned.get("end"))
            if not (
                isinstance(a, (int, float))
                and isinstance(b, (int, float))
                and math.isfinite(a)
                and math.isfinite(b)
                and (0 <= a < b <= duration + 1e-06)
            ):
                raise ValueError("An available word has invalid bounds")
            available = np.flatnonzero((pts >= a) & (pts < b))
            picks = (
                sorted(
                    {
                        int(available[j])
                        for j in (0, len(available) // 2, len(available) - 1)
                    }
                )
                if len(available)
                else []
            )
            item.update(
                start_sec=float(a),
                end_sec=float(b),
                time_mapping_level="word_time_supported",
                feature_time_mapping_level="grid_anchor_approximate",
                frame_indices=picks,
                frame_pts_sec=[float(pts[i]) for i in picks],
                timing_agreement_mask=bool(aligned.get("timing_agreement_mask")),
            )
            if not picks:
                item["review_flags"].append(
                    "no_actual_video_frame_inside_word_interval"
                )
        positions.append(item)
    return {
        "schema_version": 1,
        "sample_id": metadata["sample_id"],
        "sample_key": metadata["sample_key"],
        "raw_text": metadata["raw_text"],
        "duration_s": duration,
        "origin_pts_s": media["origin_pts_s"],
        "status": final["status"],
        "segment_gate": final["segment_gate"],
        "word_count": final["word_count"],
        "aligned_word_count": final["aligned_word_count"],
        "positions": positions,
        "words": final["words"],
        "timed_content_positions": sum((x["start_sec"] is not None for x in positions)),
        "mapping_limit": "Word-anchored replay only; original official audio/visual feature pooling windows and per-frame extraction times are unavailable.",
        "text_replaced": False,
        "predictor_inputs_changed": False,
        "human_boundary_ground_truth_available": False,
        "timing_agreement": final["timing_agreement"],
        "source_pkl_sha256": metadata["source_pkl_sha256"],
        "source_video_sha256": metadata["source_video_sha256"],
        "raw_text_sha256": metadata["raw_text_sha256"],
    }
