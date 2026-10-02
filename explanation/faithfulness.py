"""Deterministic budgeted evidence selection and frozen-model perturbation tests.

Intervals are contiguous in the ordered *observed atomic units*, not guessed
seconds. WordPiece groups count as one atom; random controls additionally match
the number of covered original grid positions in each modality.
"""

from __future__ import annotations
import math
import numpy as np
from .attribution import MODALITIES, MaskEvaluator, normalize_groups, validate_observed


def _better(candidate, incumbent):
    if incumbent is None:
        return True
    if candidate[0] > incumbent[0] + 1e-12:
        return True
    if candidate[0] < incumbent[0] - 1e-12:
        return False
    return candidate[1:] < incumbent[1:]


def _intervals(indices):
    intervals = []
    for index in indices:
        if not intervals or index != intervals[-1][1]:
            intervals.append([int(index), int(index) + 1])
        else:
            intervals[-1][1] += 1
    return intervals


def interval_profile(scores, budget, max_intervals=3):
    """Best signed sum for every exact atom count, with at most k intervals.

    Negative bridge atoms consume budget and contribute their signed score.
    Tie order for equal count: fewer intervals, lexicographically earlier atoms.
    """
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 1 or not np.isfinite(scores).all():
        raise ValueError("scores must be a finite vector")
    if (
        int(budget) != budget
        or budget < 0
        or int(max_intervals) != max_intervals
        or (max_intervals < 1)
    ):
        raise ValueError(
            "budget must be nonnegative and max_intervals positive integers"
        )
    budget, max_intervals = (min(int(budget), len(scores)), int(max_intervals))
    states = {(0, 0, False): (0.0, ())}
    for index, value in enumerate(scores):
        updated = {}
        for (used, intervals, inside), (score, indices) in states.items():
            skipped_key = (used, intervals, False)
            skipped = (score, indices)
            if _better(skipped, updated.get(skipped_key)):
                updated[skipped_key] = skipped
            added_intervals = intervals + int(not inside)
            if used < budget and added_intervals <= max_intervals:
                selected_key = (used + 1, added_intervals, True)
                selected = (score + float(value), indices + (index,))
                if _better(selected, updated.get(selected_key)):
                    updated[selected_key] = selected
        states = updated
    profile = {}
    for (used, count, _), (score, indices) in states.items():
        candidate = (score, count, indices)
        if _better(candidate, profile.get(used)):
            profile[used] = candidate
    return {
        used: {"score": value[0], "interval_count": value[1], "indices": value[2]}
        for used, value in profile.items()
    }


def selection_from_indices(
    groups,
    observed_mask,
    indices_by_modality,
    *,
    objective_score=None,
    allowed_budget=None,
    budget_fraction=None,
    scope="joint",
):
    observed = validate_observed(observed_mask)
    keep = np.zeros_like(observed)
    atoms, spans, position_counts = ({}, {}, {})
    for mi, modality in enumerate(MODALITIES):
        indices = sorted(set((int(i) for i in indices_by_modality.get(modality, []))))
        if any((i < 0 or i >= len(groups[modality]) for i in indices)):
            raise ValueError("selected atom index outside groups")
        atoms[modality] = indices
        spans[modality] = _intervals(indices)
        for i in indices:
            keep[mi, groups[modality][i]] = True
        position_counts[modality] = int(keep[mi].sum())
    keep &= observed
    return {
        "scope": scope,
        "budget_fraction": budget_fraction,
        "allowed_atom_budget": allowed_budget,
        "actual_atom_count": sum(map(len, atoms.values())),
        "atom_indices": atoms,
        "atom_intervals_end_exclusive": spans,
        "actual_grid_position_counts": position_counts,
        "actual_interval_count": sum((len(s) for s in spans.values())),
        "objective_score": objective_score,
        "keep_mask": keep.tolist(),
        "interval_coordinate_system": "ordered_observed_atoms",
    }


def select_spans(
    scores,
    groups,
    observed_mask,
    *,
    budget_fraction=0.2,
    modality=None,
    max_intervals=3,
):
    """Joint atom-budget DP, or conditional within one named modality.

    ``scores`` contains the signed objective vectors for T/A/V. Use a for class
    support, -a for counterevidence, abs(a_reg) for intensity reconstruction.
    """
    observed = validate_observed(observed_mask)
    normalized, added = normalize_groups(observed, groups)
    if any(added.values()):
        raise ValueError(
            "selection groups must cover observations; use explanation['groups']"
        )
    if not 0 <= budget_fraction <= 1 or modality not in (None, *MODALITIES):
        raise ValueError("invalid budget fraction or modality")
    active = MODALITIES if modality is None else (modality,)
    total = sum((len(normalized[m]) for m in active))
    budget = math.ceil(float(budget_fraction) * total)
    profiles = {}
    for m in active:
        values = np.asarray(scores[m], dtype=np.float64)
        if values.shape != (len(normalized[m]),):
            raise ValueError(f"score count does not match {m} groups")
        profiles[m] = interval_profile(values, budget, max_intervals)
    joint = {0: (0.0, 0, (), ())}
    for m in active:
        updated = {}
        for used, (score, span_count, global_indices, allocations) in joint.items():
            for count, local in profiles[m].items():
                if used + count <= budget:
                    candidate = (
                        score + local["score"],
                        span_count + local["interval_count"],
                        global_indices
                        + tuple(((MODALITIES.index(m), j) for j in local["indices"])),
                        allocations + (local["indices"],),
                    )
                    if _better(candidate, updated.get(used + count)):
                        updated[used + count] = candidate
        joint = updated
    best = None
    for used, (score, spans, global_indices, allocation) in joint.items():
        candidate = (score, used, spans, global_indices, allocation)
        if _better(candidate, best):
            best = candidate
    score, _, _, _, allocation = best
    selected = dict(zip(active, allocation))
    return selection_from_indices(
        normalized,
        observed,
        selected,
        objective_score=float(score),
        allowed_budget=budget,
        budget_fraction=float(budget_fraction),
        scope="joint" if modality is None else modality,
    )


def evidence_scores(explanation, kind):
    target = int(explanation["predicted_class"])
    values = {
        m: np.asarray(explanation["local"][m]["values"], dtype=np.float64).reshape(
            -1, 4
        )
        for m in MODALITIES
    }
    if kind == "class_support":
        return {m: values[m][:, target] for m in MODALITIES}
    if kind == "class_counterevidence":
        return {m: -values[m][:, target] for m in MODALITIES}
    if kind == "intensity_absolute":
        return {m: np.abs(values[m][:, 3]) for m in MODALITIES}
    raise ValueError(
        "kind must be class_support, class_counterevidence, or intensity_absolute"
    )


def select_evidence(
    explanation,
    *,
    budget_fraction=0.2,
    kind="class_support",
    modality=None,
    max_intervals=3,
):
    return select_spans(
        evidence_scores(explanation, kind),
        explanation["groups"],
        explanation["observed_mask"],
        budget_fraction=budget_fraction,
        modality=modality,
        max_intervals=max_intervals,
    )


def _sample_intervals(rng, n, lengths):
    """Uniform starts for fixed ordered lengths with >=1 atom between runs."""
    k = len(lengths)
    if not k:
        return []
    slack = n - sum(lengths) - (k - 1)
    if slack < 0:
        raise ValueError("interval lengths cannot fit the observed atoms")
    bars = np.sort(rng.choice(slack + k, size=k, replace=False))
    gaps = np.diff(np.concatenate(([-1], bars, [slack + k]))) - 1
    position, indices = (int(gaps[0]), [])
    for i, length in enumerate(lengths):
        indices.extend(range(position, position + length))
        position += length + 1 + int(gaps[i + 1])
    return indices


def matched_random_selections(
    selection, groups, observed_mask, *, count=20, seed=20260927, max_attempts=20000
):
    """Sample with replacement, matching atoms, runs, lengths and grid coverage.

    Rejection only enforces original-grid coverage for variable-sized word groups.
    A shortage is reported explicitly instead of relaxing the matching contract.
    """
    if int(count) != count or count < 0 or max_attempts < count:
        raise ValueError("invalid random count or attempt budget")
    observed = validate_observed(observed_mask)
    rng = np.random.default_rng(seed)
    lengths = {
        m: [end - start for start, end in selection["atom_intervals_end_exclusive"][m]]
        for m in MODALITIES
    }
    controls, attempts, unique = ([], 0, set())
    while len(controls) < count and attempts < max_attempts:
        attempts += 1
        chosen = {
            m: _sample_intervals(rng, len(groups[m]), lengths[m]) for m in MODALITIES
        }
        candidate = selection_from_indices(
            groups,
            observed,
            chosen,
            allowed_budget=selection["actual_atom_count"],
            budget_fraction=selection["budget_fraction"],
            scope=selection["scope"],
        )
        if (
            candidate["actual_grid_position_counts"]
            != selection["actual_grid_position_counts"]
        ):
            continue
        controls.append(candidate)
        unique.add(tuple((tuple(candidate["atom_indices"][m]) for m in MODALITIES)))
    return (
        controls,
        {
            "requested": int(count),
            "obtained": len(controls),
            "attempts": attempts,
            "unique_controls": len(unique),
            "sampling_with_replacement": True,
            "matched": [
                "per_modality_atom_count",
                "interval_count",
                "ordered_interval_lengths",
                "grid_position_count",
            ],
            "status": (
                "empty_evidence"
                if selection["actual_atom_count"] == 0
                else "complete" if len(controls) == count else "matching_shortage"
            ),
            "skipped_count": int(count) - len(controls),
        },
    )


def _intervention_masks(selection, observed, panel, modality):
    evidence = np.asarray(selection["keep_mask"], dtype=bool) & observed
    if panel == "joint":
        return {"delete": observed & ~evidence, "keep": evidence}
    if panel == "dominant_delete":
        return {"delete": observed & ~evidence}
    if panel == "dominant_keep":
        keep = observed.copy()
        mi = MODALITIES.index(modality)
        keep[mi] = evidence[mi]
        return {"keep": keep}
    raise ValueError("unknown evaluation panel")


def _metrics(full, output, fixed_class, operation):
    delta = np.asarray(full) - np.asarray(output)
    return {
        "comprehensiveness" if operation == "delete" else "sufficiency": float(
            delta[fixed_class]
        ),
        "class_probability_delta": float(delta[fixed_class]),
        "class_flipped": bool(np.argmax(output[:3]) != fixed_class),
        "intensity_delta_full_minus_intervened": float(delta[3]),
        "intensity_absolute_change": float(abs(delta[3])),
    }


def _summary(metrics):
    if not metrics:
        return {"n": 0, "mean": {}, "sample_sd": {}}
    keys = metrics[0].keys()
    return {
        "n": len(metrics),
        "mean": {key: float(np.mean([m[key] for m in metrics])) for key in keys},
        "sample_sd": {
            key: (
                float(np.std([m[key] for m in metrics], ddof=1))
                if len(metrics) > 1
                else None
            )
            for key in keys
        },
    }


def evaluate_faithfulness(
    predict_masks,
    observed_mask,
    explanation,
    *,
    budgets=(0.1, 0.2, 0.3),
    random_controls=20,
    seed=20260927,
    batch_size=64,
    max_intervals=3,
    kinds=("class_support", "class_counterevidence", "intensity_absolute"),
    intervention_id="primary_zero_mask",
    endpoint_tolerance=1e-05,
):
    """Three named panels, fixed original class, and matched random controls.

    Primary intervention endpoints must reproduce explanation['full_output'].
    A different callback may implement a secondary intervention, but retains the
    original target and is labelled by its separate intervention_id.
    """
    observed = validate_observed(observed_mask)
    if not np.array_equal(observed, np.asarray(explanation["observed_mask"], bool)):
        raise ValueError("faithfulness and explanation observations differ")
    if len(set(budgets)) != len(budgets) or any((not 0 <= b <= 1 for b in budgets)):
        raise ValueError("budgets must be distinct fractions in [0, 1]")
    evaluator = MaskEvaluator(predict_masks, observed, batch_size)
    full = evaluator.evaluate(observed[None])[0]
    endpoint_delta = float(
        np.max(np.abs(full - np.asarray(explanation["full_output"])))
    )
    if endpoint_delta > endpoint_tolerance:
        raise ValueError(
            "full-input callback does not reproduce the explained model endpoint"
        )
    fixed_class = int(explanation["predicted_class"])
    records, skipped = ([], [])
    for ki, kind in enumerate(kinds):
        scores = evidence_scores(explanation, kind)
        dominance = (
            explanation["regression_dominance"]
            if kind == "intensity_absolute"
            else explanation["classification_dominance"]
        )
        dominant = dominance["modality"]
        for bi, budget in enumerate(budgets):
            for pi, panel in enumerate(("joint", "dominant_delete", "dominant_keep")):
                modality = None if panel == "joint" else dominant
                if modality not in (None, *MODALITIES):
                    skipped.append(
                        {
                            "kind": kind,
                            "budget_fraction": budget,
                            "panel": panel,
                            "reason": "unresolved_dominant_modality",
                        }
                    )
                    continue
                evidence = select_spans(
                    scores,
                    explanation["groups"],
                    observed,
                    budget_fraction=budget,
                    modality=modality,
                    max_intervals=max_intervals,
                )
                seed_components = [int(seed), ki, bi, int(panel != "joint")]
                control_seed = np.random.SeedSequence(seed_components)
                controls, random_info = matched_random_selections(
                    evidence,
                    explanation["groups"],
                    observed,
                    count=random_controls,
                    seed=control_seed,
                )
                selections = [evidence] + controls
                operations = list(
                    _intervention_masks(evidence, observed, panel, modality)
                )
                masks = [
                    _intervention_masks(selection, observed, panel, modality)[operation]
                    for selection in selections
                    for operation in operations
                ]
                outputs = evaluator.evaluate(np.stack(masks)).reshape(
                    len(selections), len(operations), 4
                )
                record = {
                    "kind": kind,
                    "budget_fraction": float(budget),
                    "panel": panel,
                    "dominant_modality": dominant,
                    "fixed_class": fixed_class,
                    "intervention_id": intervention_id,
                    "evidence": evidence,
                    "random_matching": random_info,
                    "random_seed_components": seed_components,
                    "random_atom_indices": [
                        control["atom_indices"] for control in controls
                    ],
                    "operations": {},
                }
                for oi, operation in enumerate(operations):
                    random_metrics = [
                        _metrics(full, output, fixed_class, operation)
                        for output in outputs[1:, oi]
                    ]
                    record["operations"][operation] = {
                        "output": outputs[0, oi].tolist(),
                        "metrics": _metrics(
                            full, outputs[0, oi], fixed_class, operation
                        ),
                        "random_outputs": outputs[1:, oi].tolist(),
                        "random_summary": _summary(random_metrics),
                    }
                records.append(record)
    aopc = []
    for kind in kinds:
        for panel in ("joint", "dominant_delete", "dominant_keep"):
            matched = [r for r in records if r["kind"] == kind and r["panel"] == panel]
            if not matched:
                continue
            for operation in matched[0]["operations"]:
                measurements = [r["operations"][operation]["metrics"] for r in matched]
                random_means = [
                    r["operations"][operation]["random_summary"]["mean"]
                    for r in matched
                    if r["operations"][operation]["random_summary"]["n"]
                ]
                aopc.append(
                    {
                        "kind": kind,
                        "panel": panel,
                        "operation": operation,
                        "definition": "unweighted mean across the registered discrete budget fractions",
                        "budget_count": len(matched),
                        "budgets": [r["budget_fraction"] for r in matched],
                        "mean_metrics": _summary(measurements)["mean"],
                        "mean_random_metrics": _summary(random_means)["mean"],
                    }
                )
    return {
        "schema_version": 1,
        "intervention_id": intervention_id,
        "fixed_class": fixed_class,
        "full_output": full.tolist(),
        "full_endpoint_max_abs_difference": endpoint_delta,
        "seed": int(seed),
        "budgets": list(budgets),
        "kinds": list(kinds),
        "records": records,
        "skipped": skipped,
        "aopc": aopc,
        "cache": evaluator.stats(),
        "negative_metrics_preserved": True,
        "random_controls_are_not_independent_dataset_examples": True,
    }
