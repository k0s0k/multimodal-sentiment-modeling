"""Map budgeted evidence to official characters, seconds and frames."""

import numpy as np
from .faithfulness import select_evidence

MODALITIES = ("T", "A", "V")


def split_atom_runs(groups, atom_indices, values=None, by_sign=False):
    """Break display runs at actual-grid holes; regression signs stay separate."""
    runs = []
    for atom in atom_indices:
        sign = int(np.sign(values[atom])) if by_sign else None
        previous = runs[-1][-1] if runs else None
        contiguous = (
            previous is not None
            and atom == previous + 1
            and (max(groups[previous]) + 1 == min(groups[atom]))
        )
        same_sign = not by_sign or (
            previous is not None and int(np.sign(values[previous])) == sign
        )
        if not runs or not contiguous or (not same_sign):
            runs.append([int(atom)])
        else:
            runs[-1].append(int(atom))
    return runs


def evidence_for_selection(exp, metadata, alignment, selection, kind, scope):
    positions_by_grid = {int(p["position"]): p for p in alignment["positions"]}
    source_tokens = {int(p["position"]): p for p in metadata["tokens"]}
    target_index = 3 if kind == "intensity_absolute" else int(exp["predicted_class"])
    rows = []
    for mi, m in enumerate(MODALITIES):
        groups = exp["groups"][m]
        values = np.asarray(exp["local"][m]["values"], dtype=float).reshape(-1, 4)[
            :, target_index
        ]
        for atoms in split_atom_runs(
            groups, selection["atom_indices"][m], values, kind == "intensity_absolute"
        ):
            grid = sorted({p for atom in atoms for p in groups[atom]})
            tokens = [source_tokens[p] for p in grid]
            mapped = [positions_by_grid[p] for p in grid]
            offsets = [
                (p["char_start"], p["char_end"])
                for p in tokens
                if p.get("char_start") is not None and p.get("char_end") is not None
            ]
            char_start = (
                min((a for a, _ in offsets)) if len(offsets) == len(tokens) else None
            )
            char_end = (
                max((b for _, b in offsets)) if len(offsets) == len(tokens) else None
            )
            quote = (
                metadata["raw_text"][char_start:char_end]
                if char_start is not None
                else None
            )
            timed = all(
                (
                    p.get("start_sec") is not None and p.get("end_sec") is not None
                    for p in mapped
                )
            )
            start = min((p["start_sec"] for p in mapped)) if timed else None
            end = max((p["end_sec"] for p in mapped)) if timed else None
            if timed and (not 0 <= start < end <= alignment["duration_s"] + 1e-06):
                raise ValueError("Mapped evidence interval outside video")
            frames = {
                int(i): float(t)
                for p in mapped
                for i, t in zip(p.get("frame_indices", []), p.get("frame_pts_sec", []))
            }
            score = float(values[atoms].sum())
            target = (
                kind
                if kind != "intensity_absolute"
                else (
                    "intensity_upward"
                    if score > 0
                    else "intensity_downward" if score < 0 else "intensity_zero"
                )
            )
            flags = sorted({flag for p in mapped for flag in p.get("review_flags", [])})
            if not timed:
                flags.append("one_or_more_selected_positions_have_unknown_time")
            if m == "V" and (not frames):
                flags.append("no_verified_keyframe_for_selected_evidence")
            if (
                any((values[a] < 0 for a in atoms))
                and kind == "class_support"
                or (
                    any((values[a] > 0 for a in atoms))
                    and kind == "class_counterevidence"
                )
            ):
                flags.append("contains_opposing_sign_bridge_atom")
            level = (
                ("word_time_supported" if m == "T" else "grid_anchor_approximate")
                if timed
                else "raw_text_exact" if quote is not None else "unknown"
            )
            subspans = sorted(
                {
                    (p["start_sec"], p["end_sec"])
                    for p in mapped
                    if p.get("start_sec") is not None
                }
            )
            rows.append(
                {
                    "sample_id": metadata["sample_id"],
                    "selection_kind": kind,
                    "target": target,
                    "target_output_index": target_index,
                    "scope": scope,
                    "modality": m,
                    "atomic_unit_ids": [f"{m}:{a}" for a in atoms],
                    "atom_indices": atoms,
                    "token_indices": grid if m == "T" else [],
                    "grid_indices": grid,
                    "char_start": char_start,
                    "char_end": char_end,
                    "text": quote,
                    "signed_contribution": score,
                    "atom_signed_contributions": [float(values[a]) for a in atoms],
                    "start_s": start,
                    "end_s": end,
                    "timed_subspans": [list(x) for x in subspans],
                    "frame_indices": sorted(frames),
                    "frame_pts_s": [frames[i] for i in sorted(frames)],
                    "mapping_level": level,
                    "feature_time_mapping_level": (
                        "grid_anchor_approximate" if timed else "unknown"
                    ),
                    "source_sha256": metadata["source_video_sha256"],
                    "source_feature_sha256": metadata["source_pkl_sha256"],
                    "alignment_method": (
                        "Q1_refined_CTC_conservative_Q3_gate_Qwen_disagreement_audit"
                        if timed
                        else None
                    ),
                    "quality_flags": sorted(set(flags)),
                    "timing_agreement_mask": bool(
                        timed and all((p.get("timing_agreement_mask") for p in mapped))
                    ),
                    "budget_fraction": selection["budget_fraction"],
                    "allowed_atom_budget": selection["allowed_atom_budget"],
                    "selection_actual_atom_count": selection["actual_atom_count"],
                    "selection_actual_grid_position_counts": selection[
                        "actual_grid_position_counts"
                    ],
                    "interval_coordinate_system": "actual_grid_contiguous_display_segments_from_ordered_observed_atom_selection",
                    "time_range_includes_between_selected_word_gaps": len(subspans) > 1,
                    "all_selected_positions_observed_in_this_modality": bool(
                        np.asarray(exp["observed_mask"])[mi, grid].all()
                    ),
                }
            )
    return rows


def gather_evidence(exp, metadata, alignment):
    selections, spans = ({}, [])
    for scope in ("joint", *MODALITIES):
        for kind in ("class_support", "class_counterevidence", "intensity_absolute"):
            selected = select_evidence(
                exp,
                budget_fraction=0.2,
                kind=kind,
                modality=None if scope == "joint" else scope,
                max_intervals=3,
            )
            key = f"{scope}:{kind}"
            selections[key] = selected
            spans.extend(
                evidence_for_selection(exp, metadata, alignment, selected, kind, scope)
            )
    for i, span in enumerate(spans, 1):
        span["evidence_id"] = f"{metadata['sample_id']}_e{i:03}"
    return (selections, spans)
