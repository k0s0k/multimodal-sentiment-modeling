"""Additional VALIDATION diagnostics, independent of the frozen main mask/cache path.

Equal-budget outputs always retain all N samples. Compare a modality/rate's four
methods on its common_eligible intersection; ineligible rows carry the unchanged
input and explicit reasons, not a silently relaxed deletion budget.

The splitter uses ranks only to select observed groups. Every damaged array and
every reported interval remains on the original 0..49 axis; no sequence is packed.
"""

from __future__ import annotations
import numpy as np
from .data import LENGTH, sanitize_tokens, observed_masks, build_structure, stable_seed

DIAGNOSTICS_VERSION = "q2_equal_budget_v1"
MODALITIES = "TAV"


def equal_budget_conditions():
    """24 preregistered conditions: .3/.5 x T/A/V x 1/2/4 blocks/point."""
    return _conditions((0.3, 0.5))


def _conditions(rates):
    result = []
    for rate in rates:
        if not 0 < float(rate) < 1:
            raise ValueError("equal-budget rates must lie strictly in (0,1)")
        for modality in MODALITIES:
            for blocks in (1, 2, 4, None):
                method = "point" if blocks is None else f"blocks{blocks}"
                result.append(
                    {
                        "name": f"eq_{modality}_r{float(rate):g}_{method}",
                        "family": "equal_budget",
                        "modality": modality,
                        "rate": float(rate),
                        "method": method,
                        "blocks": blocks,
                    }
                )
    return result


def _budget(n, rate):
    return min(n - 1, max(1, int(np.floor(rate * n + 0.5)))) if n >= 2 else 0


def _eligibility(n, budget, blocks):
    if n < 2 or budget < 1:
        return (False, "fewer_than_two_observed_positions")
    if budget >= n:
        return (False, "no_observation_would_remain")
    if blocks is None:
        return (True, "eligible")
    if budget < blocks:
        return (False, "deletion_budget_smaller_than_block_count")
    if n - budget < blocks - 1:
        return (False, "too_few_retained_observations_between_blocks")
    return (True, "eligible")


def _positive_composition(total, parts, rng):
    if parts == 1:
        return np.array([total], np.int64)
    cuts = np.sort(rng.choice(np.arange(1, total), parts - 1, replace=False))
    return np.diff(np.r_[0, cuts, total])


def _weak_composition(total, parts, rng):
    if parts == 1:
        return np.array([total], np.int64)
    bars = np.sort(rng.choice(total + parts - 1, parts - 1, replace=False))
    return np.diff(np.r_[-1, bars, total + parts - 1]) - 1


def _runs(mask):
    edges = np.diff(np.r_[False, mask, False].astype(np.int8))
    return [
        [int(a), int(b)]
        for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))
    ]


def _select(observed, budget, blocks, rng):
    """Select exactly budget observed sites, with >=1 kept observed site per block gap."""
    positions = np.flatnonzero(observed)
    drop = np.zeros(LENGTH, bool)
    if blocks is None:
        selected = np.sort(rng.choice(positions, budget, replace=False))
        drop[selected] = True
        return (drop, _runs(drop), [], [])
    sizes = _positive_composition(budget, blocks, rng)
    surplus = len(positions) - budget - (blocks - 1)
    gaps = _weak_composition(surplus, blocks + 1, rng)
    gaps[1:-1] += 1
    cursor = int(gaps[0])
    intervals, observed_groups = ([], [])
    for b, size in enumerate(sizes):
        group = positions[cursor : cursor + size]
        start, end = (int(group[0]), int(group[-1] + 1))
        drop[start:end] = True
        intervals.append([start, end])
        observed_groups.append(group.tolist())
        cursor += int(size + gaps[b + 1])
    retained_between = [
        np.flatnonzero(observed[left[1] : right[0]]).size
        for left, right in zip(intervals, intervals[1:])
    ]
    if int((drop & observed).sum()) != budget or any((x < 1 for x in retained_between)):
        raise AssertionError("Equal-budget block invariant failed")
    return (drop, intervals, list(map(int, retained_between)), observed_groups)


def _validate_inputs(tokens, audio, vision, ids):
    tokens = sanitize_tokens(tokens)
    audio = np.asarray(audio, np.float32)
    vision = np.asarray(vision, np.float32)
    original = observed_masks(tokens, audio, vision)
    if len(ids) != len(tokens) or len(set(map(str, ids))) != len(ids):
        raise ValueError("Unique stable IDs must match the sample axis")
    return (tokens, audio, vision, original)


def _apply(tokens, audio, vision, drop):
    t, a, v = (tokens.copy(), audio.copy(), vision.copy())
    for channel in range(3):
        t[:, channel][drop[:, 0]] = 0
    a[drop[:, 1]] = 0
    v[drop[:, 2]] = 0
    mask = observed_masks(t, a, v)
    return {
        "tokens": t,
        "audio": a,
        "vision": v,
        "mask": mask,
        "structure": build_structure(mask),
    }


def make_equal_budget_views(tokens, audio, vision, seed, ids, rates=(0.3, 0.5)):
    """Return {views, diagnostics, comparison_groups}, with all N rows in every view.

    views[name] contains damaged tokens/audio/vision/mask/structure only.
    diagnostics[name] contains eligibility arrays, drop masks, and per-row records.
    comparison_groups[group].common_eligible is the SAME subset for all four methods.
    There are no labels and no model-dependent selection decisions in this API.
    """
    tokens, audio, vision, original = _validate_inputs(tokens, audio, vision, ids)
    conditions = _conditions(rates)
    n = len(ids)
    views, diagnostics, groups = ({}, {}, {})
    for c in conditions:
        m = MODALITIES.index(c["modality"])
        group_name = f"eq_{c['modality']}_r{c['rate']:g}"
        if group_name not in groups:
            counts = original[:, m].sum(1)
            budgets = np.array(
                [_budget(int(count), c["rate"]) for count in counts], np.int64
            )
            eligibility = [
                _eligibility(int(count), int(budget), 4)
                for count, budget in zip(counts, budgets)
            ]
            common = np.array([x[0] for x in eligibility], bool)
            groups[group_name] = {
                "modality": c["modality"],
                "rate": c["rate"],
                "condition_names": [],
                "common_eligible": common,
                "eligible_ids": [str(ids[i]) for i in np.flatnonzero(common)],
                "excluded": [
                    {
                        "id": str(ids[i]),
                        "reason": eligibility[i][1],
                        "original_observed_count": int(counts[i]),
                        "deletion_budget": int(budgets[i]),
                    }
                    for i in np.flatnonzero(~common)
                ],
                "n_total": n,
                "n_common_eligible": int(common.sum()),
                "coverage": float(common.mean()) if n else None,
                "comparison_rule": "All four methods use this same common_eligible subset; main evaluation still retains every sample.",
            }
        common = groups[group_name]["common_eligible"]
        groups[group_name]["condition_names"].append(c["name"])
        drop = np.zeros_like(original)
        eligible = np.zeros(n, bool)
        records = []
        for i, sample_id in enumerate(ids):
            count = int(original[i, m].sum())
            budget = _budget(count, c["rate"])
            ok, reason = _eligibility(count, budget, c["blocks"])
            eligible[i] = ok
            intervals, retained_between, observed_groups = ([], [], [])
            if ok:
                rng = np.random.default_rng(
                    stable_seed(DIAGNOSTICS_VERSION, int(seed), str(sample_id), c)
                )
                selected, intervals, retained_between, observed_groups = _select(
                    original[i, m], budget, c["blocks"], rng
                )
                drop[i, m] = selected
            deleted = np.flatnonzero(drop[i, m] & original[i, m])
            records.append(
                {
                    "id": str(sample_id),
                    "eligible": bool(ok),
                    "reason": reason,
                    "common_eligible": bool(common[i]),
                    "modality": c["modality"],
                    "target_rate": c["rate"],
                    "original_observed_count": count,
                    "deletion_budget": budget,
                    "actual_deleted_count": int(len(deleted)),
                    "actual_deleted_fraction": (
                        float(len(deleted) / count) if count else None
                    ),
                    "retained_observed_count": count - len(deleted),
                    "intervals_half_open": intervals,
                    "interval_span_positions": [b - a for a, b in intervals],
                    "deleted_observed_positions": deleted.tolist(),
                    "observed_groups": observed_groups,
                    "retained_observations_between_blocks": retained_between,
                    "unchanged_due_to_ineligibility": not ok,
                }
            )
        views[c["name"]] = _apply(tokens, audio, vision, drop)
        diagnostics[c["name"]] = {
            "condition": c,
            "eligible": eligible,
            "common_eligible": common.copy(),
            "drop_mask": drop,
            "records": records,
        }
    return {
        "version": DIAGNOSTICS_VERSION,
        "seed": int(seed),
        "ids": list(map(str, ids)),
        "views": views,
        "diagnostics": diagnostics,
        "comparison_groups": groups,
    }


def make_boundary_fixtures():
    """Synthetic numerical fixtures, not labeled evaluation examples.

    'full50' means all 50 BERT attention positions: CLS + 48 contents + SEP.
    Dummy non-special token 2000 is a numerical fixture, not imported emotion data.
    """
    output = {}
    for name, content_count in (("one_content", 1), ("full50", 48)):
        t = np.zeros((1, 3, 50), np.int64)
        t[0, 0, 0] = 101
        t[0, 0, 1 : content_count + 1] = 2000
        t[0, 0, content_count + 1] = 102
        t[0, 1, : content_count + 2] = 1
        a = np.zeros((1, 50, 74), np.float32)
        v = np.zeros((1, 50, 35), np.float32)
        a[:, 1 : content_count + 1] = 1.0
        v[:, 1 : content_count + 1] = 2.0
        mask = observed_masks(t, a, v)
        output[name] = {
            "inputs": {
                "tokens": t,
                "audio": a,
                "vision": v,
                "mask": mask,
                "structure": build_structure(mask),
            },
            "metadata": {
                "synthetic": True,
                "ids": ["synthetic_" + name],
                "purpose": "numerical_boundary_only",
                "labels_available": False,
                "content_positions": content_count,
                "attention_positions": content_count + 2,
            },
        }
    return output


def make_stress_views(tokens, audio, vision, seed, ids):
    """Separate boundary suite; excluded from local-missingness aggregate metrics.

    special_token_columns erases all visible CLS/SEP columns of text_bert together.
    whole_T/A/V and all_empty are entire-modality stress cases. one_observed_content
    retains one original absolute index in all modalities, plus still-visible text
    specials; absent modalities are never synthesized. Synthetic full50/one-content
    fixtures are under boundary_fixtures, with no inherited labels or sample IDs.
    """
    tokens, audio, vision, original = _validate_inputs(tokens, audio, vision, ids)
    patterns = {}
    special = np.zeros_like(original)
    special[:, 0] = (tokens[:, 1] == 1) & np.isin(tokens[:, 0], [101, 102])
    patterns["special_token_columns"] = special
    for m, modality in enumerate(MODALITIES):
        drop = np.zeros_like(original)
        drop[:, m] = True
        patterns["whole_" + modality] = drop
    patterns["all_empty"] = np.ones_like(original)
    one = np.ones_like(original)
    retained = []
    for i, sample_id in enumerate(ids):
        candidates = np.flatnonzero(original[i, 0])
        if not len(candidates):
            candidates = np.flatnonzero(original[i].any(0))
        specials = (tokens[i, 1] == 1) & np.isin(tokens[i, 0], [101, 102])
        one[i, 0, specials] = False
        if len(candidates):
            rng = np.random.default_rng(
                stable_seed(
                    DIAGNOSTICS_VERSION, "one_content", int(seed), str(sample_id)
                )
            )
            chosen = int(rng.choice(candidates))
            one[i, :, chosen] = False
            retained.append(chosen)
        else:
            retained.append(None)
    patterns["one_observed_content"] = one
    views, diagnostics = ({}, {})
    for name, drop in patterns.items():
        damaged = _apply(tokens, audio, vision, drop)
        views[name] = damaged
        diagnostics[name] = {
            "stress_only": True,
            "include_in_main_average": False,
            "drop_mask": drop,
            "records": [
                {
                    "id": str(sample_id),
                    "original_observed_count": original[i].sum(1).tolist(),
                    "remaining_observed_count": damaged["mask"][i].sum(1).tolist(),
                    "retained_absolute_position": (
                        retained[i] if name == "one_observed_content" else None
                    ),
                    "has_original_content": bool(original[i].any()),
                }
                for i, sample_id in enumerate(ids)
            ],
        }
    return {
        "version": DIAGNOSTICS_VERSION,
        "seed": int(seed),
        "ids": list(map(str, ids)),
        "views": views,
        "diagnostics": diagnostics,
        "boundary_fixtures": make_boundary_fixtures(),
    }
