"""Model-agnostic exact modality Shapley and grouped permutation Owen values.

The callback receives boolean (batch, 3, length) *keep* masks and returns
(batch, 4) outputs: negative/neutral/positive probability and intensity.
Only small numeric outputs are cached here; encoder tensors belong to runtime.
"""

from __future__ import annotations
import math
import hashlib
from collections.abc import Callable, Mapping, Sequence
import numpy as np

MODALITIES = ("T", "A", "V")
OUTPUTS = ("p_negative", "p_neutral", "p_positive", "intensity")
PERMUTATION_HASH_FORMAT = "sha256(header UTF8 'q3_owen_antithetic_orders_v1\\0', n_atoms as little-endian uint32, then actual forward/reverse orders as C-order little-endian int32 rows in sampling order); contexts share each order"


def validate_observed(observed_mask) -> np.ndarray:
    raw = np.asarray(observed_mask)
    if raw.ndim != 2 or raw.shape[0] != 3 or raw.shape[1] < 1:
        raise ValueError("observed_mask must have shape (3, length)")
    if not np.isin(raw, (0, 1)).all():
        raise ValueError("observed_mask must contain only boolean/0/1 values")
    return raw.astype(bool, copy=True)


def normalize_groups(observed_mask, groups=None) -> tuple[dict, dict]:
    """Partition observations, preserving unmatched tokens as singleton players.

    Supplied groups are intersected with original observations. Out-of-range
    indices and overlapping observed positions are errors, not silent merges.
    """
    observed = validate_observed(observed_mask)
    groups = {} if groups is None else groups
    if not isinstance(groups, Mapping) or set(groups) - set(MODALITIES):
        raise ValueError("groups must be a mapping with T/A/V keys")
    result, added = ({}, {})
    for mi, modality in enumerate(MODALITIES):
        units, used = ([], set())
        for group in groups.get(modality, []):
            positions = []
            for position in group:
                if isinstance(position, (bool, np.bool_)) or int(position) != position:
                    raise ValueError("group positions must be integer indices")
                position = int(position)
                if position < 0 or position >= observed.shape[1]:
                    raise ValueError("group position is outside the sequence")
                if observed[mi, position]:
                    positions.append(position)
            positions = sorted(set(positions))
            if used.intersection(positions):
                raise ValueError("an observed position belongs to overlapping groups")
            if positions:
                units.append(positions)
                used.update(positions)
        missing = sorted(set(np.flatnonzero(observed[mi]).tolist()) - used)
        units.extend([[position] for position in missing])
        units.sort(key=lambda unit: tuple(unit))
        result[modality] = units
        added[modality] = missing
    return (result, added)


class MaskEvaluator:
    """Deduplicate masks and batch a callback without retaining feature tensors."""

    def __init__(self, predict_masks: Callable, observed_mask, batch_size=64):
        if (
            not callable(predict_masks)
            or int(batch_size) != batch_size
            or batch_size < 1
        ):
            raise ValueError(
                "a callback and a positive integer batch_size are required"
            )
        self.predict_masks = predict_masks
        self.observed = validate_observed(observed_mask)
        self.batch_size = int(batch_size)
        self.cache: dict[bytes, np.ndarray] = {}
        self.callback_calls = 0
        self.requests = 0

    def evaluate(self, masks) -> np.ndarray:
        raw = np.asarray(masks)
        if raw.ndim != 3 or raw.shape[1:] != self.observed.shape:
            raise ValueError("keep masks must have shape (batch, 3, length)")
        if not np.isin(raw, (0, 1)).all():
            raise ValueError("keep masks must contain only boolean/0/1 values")
        masks = raw.astype(bool) & self.observed[None]
        self.requests += len(masks)
        keys = [np.packbits(mask, bitorder="little").tobytes() for mask in masks]
        pending: dict[bytes, np.ndarray] = {}
        for key, mask in zip(keys, masks):
            if key not in self.cache and key not in pending:
                pending[key] = mask
        items = list(pending.items())
        for start in range(0, len(items), self.batch_size):
            chunk = items[start : start + self.batch_size]
            batch = np.stack([mask for _, mask in chunk])
            outputs = np.asarray(self.predict_masks(batch), dtype=np.float64)
            self.callback_calls += 1
            if outputs.shape != (len(chunk), 4) or not np.isfinite(outputs).all():
                raise ValueError("predict_masks must return finite (batch, 4) outputs")
            for (key, _), output in zip(chunk, outputs):
                self.cache[key] = output.copy()
        if not keys:
            return np.empty((0, 4), dtype=np.float64)
        return np.stack([self.cache[key] for key in keys])

    def stats(self):
        return {
            "unique_masks": len(self.cache),
            "requested_masks": self.requests,
            "callback_calls": self.callback_calls,
            "callback_batch_size": self.batch_size,
            "numeric_cache_bytes": sum(
                (len(k) + v.nbytes for k, v in self.cache.items())
            ),
        }


def coalition_masks(observed_mask) -> np.ndarray:
    observed = validate_observed(observed_mask)
    return np.stack(
        [
            observed & np.array([code >> m & 1 for m in range(3)], bool)[:, None]
            for code in range(8)
        ]
    )


def exact_modality_shapley(coalition_outputs) -> np.ndarray:
    values = np.asarray(coalition_outputs, dtype=np.float64)
    if values.shape != (8, 4) or not np.isfinite(values).all():
        raise ValueError("coalition_outputs must be finite (8, 4), bit order T,A,V")
    attribution = np.zeros((3, 4), dtype=np.float64)
    for modality in range(3):
        bit = 1 << modality
        for subset in range(8):
            if not subset & bit:
                size = subset.bit_count()
                weight = math.factorial(size) * math.factorial(2 - size) / 6
                attribution[modality] += weight * (
                    values[subset | bit] - values[subset]
                )
    return attribution


def anchored_interactions(coalition_outputs) -> np.ndarray:
    values = np.asarray(coalition_outputs, dtype=np.float64)
    if values.shape != (8, 4) or not np.isfinite(values).all():
        raise ValueError("coalition_outputs must be finite (8, 4)")
    dividends = values.copy()
    for bit in (1, 2, 4):
        for subset in range(8):
            if subset & bit:
                dividends[subset] -= dividends[subset ^ bit]
    return dividends


def _pair_marginals(evaluator, mi, units, permutation):
    """One independent antithetic pair, combined over all four contexts."""
    n = len(units)
    pair = np.zeros((n, 4), dtype=np.float64)
    for subset in range(8):
        if subset & 1 << mi:
            continue
        size = subset.bit_count()
        weight = math.factorial(size) * math.factorial(2 - size) / 6
        base = (
            evaluator.observed
            & np.array([subset >> m & 1 for m in range(3)], bool)[:, None]
        )
        for order in (permutation, permutation[::-1]):
            state = base.copy()
            states = [state.copy()]
            for atom in order:
                state[mi, units[int(atom)]] = True
                states.append(state.copy())
            values = evaluator.evaluate(np.stack(states))
            deltas = np.diff(values, axis=0)
            pair[np.asarray(order, dtype=int)] += weight / 2 * deltas
    return pair


def _average_ranks(values):
    values = np.asarray(values)
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and values[order[j]] == values[order[i]]:
            j += 1
        ranks[order[i:j]] = (i + j - 1) / 2
        i = j
    return ranks


def _rank_correlation(left, right):
    if len(left) < 2:
        return None
    left, right = (_average_ranks(left), _average_ranks(right))
    left -= left.mean()
    right -= right.mean()
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    return None if denom <= 1e-14 else float(np.dot(left, right) / denom)


def stability_diagnostics(
    pair_samples,
    *,
    targets=(0, 1, 2, 3),
    jaccard_min=0.7,
    spearman_min=0.8,
    relative_se=0.2,
    mass_se=0.01,
    near_zero=1e-12,
):
    """Fixed-K diagnostics; adaptive results are not nominal confidence intervals."""
    samples = np.asarray(pair_samples, dtype=np.float64)
    if samples.ndim != 3 or samples.shape[2] != 4 or samples.shape[0] < 2:
        raise ValueError(
            "at least two complete pair samples (pairs, atoms, 4) are required"
        )
    count, n, _ = samples.shape
    mean = samples.mean(0)
    se = samples.std(0, ddof=1) / math.sqrt(count)
    split = count // 2
    halves = (samples[:split].mean(0), samples[split:].mean(0))
    details, upgrade = ({}, False)
    for target in targets:
        target = int(target)
        if not 0 <= target < 4:
            raise ValueError("stability target must be in [0, 3]")
        score = mean[:, target]
        mass = float(np.abs(score).sum())
        top_count = max(1, math.ceil(n * 0.2)) if n else 0
        ordering = np.argsort(-np.abs(score), kind="stable")
        top = ordering[:top_count]
        threshold = np.maximum(relative_se * np.abs(score[top]), mass_se * mass)
        imprecise = bool(np.any(se[top, target] > np.maximum(threshold, near_zero)))
        jaccard = correlation = None
        rank_status = "defined"
        if n < 5:
            rank_status = "fewer_than_five_atoms"
        elif mass <= near_zero or any(
            (np.ptp(h[:, target]) <= near_zero for h in halves)
        ):
            rank_status = "zero_or_near_constant"
        else:
            first = set(
                np.argsort(-np.abs(halves[0][:, target]), kind="stable")[
                    :top_count
                ].tolist()
            )
            second = set(
                np.argsort(-np.abs(halves[1][:, target]), kind="stable")[
                    :top_count
                ].tolist()
            )
            jaccard = len(first & second) / len(first | second)
            correlation = _rank_correlation(halves[0][:, target], halves[1][:, target])
        rank_failed = (
            jaccard is not None
            and jaccard < jaccard_min
            or (correlation is not None and correlation < spearman_min)
        )
        upgrade |= bool(rank_failed or imprecise)
        details[OUTPUTS[target]] = {
            "rank_status": rank_status,
            "top20_jaccard": jaccard,
            "signed_spearman": correlation,
            "top_atom_precision_failed": imprecise,
            "top_atom_indices": top.tolist(),
            "top_atom_se_thresholds": threshold.tolist(),
            "requires_upgrade": bool(rank_failed or imprecise),
        }
    return (
        {
            "independent_pairs": count,
            "half_pair_counts": [split, count - split],
            "targets": details,
            "requires_upgrade": bool(upgrade),
            "se_is_valid_95pct_interval": False,
        },
        mean,
        se,
    )


def _dominance(phi, target, near_zero=1e-06):
    scores = np.asarray(phi)[:, target]
    denom = float(np.abs(scores).sum())
    if denom <= near_zero:
        return {
            "modality": "unresolved",
            "absolute_shares": None,
            "positive_support": "none_positive",
            "co_dominant": False,
        }
    shares = np.abs(scores) / denom
    order = np.argsort(-shares, kind="stable")
    support = int(np.argmax(scores))
    return {
        "modality": MODALITIES[int(order[0])],
        "absolute_shares": shares.tolist(),
        "positive_support": (
            MODALITIES[support] if scores[support] > 0 else "none_positive"
        ),
        "co_dominant": bool(shares[order[0]] - shares[order[1]] < 0.02),
    }


def explain(
    predict_masks: Callable,
    observed_mask,
    groups=None,
    *,
    seed=20260927,
    K=16,
    max_K=64,
    adaptive=True,
    batch_size=64,
    stability_targets=(0, 1, 2, 3),
    tolerance=1e-05,
    near_zero_total=1e-06,
    include_pair_samples=False,
) -> dict:
    """Explain one sample using exact outer games and grouped antithetic Owen.

    K is the TOTAL number of permutations, including reversals. The initial
    K must be divisible by four so the independent pairs split evenly. Adaptive
    doubling reuses earlier samples. No output-dependent class retargeting occurs.
    """
    if int(K) != K or int(max_K) != max_K:
        raise ValueError("K and max_K must be integers")
    K, max_K = (int(K), int(max_K))
    if K < 4 or K % 4 or max_K < K or max_K % K or max_K // K & max_K // K - 1:
        raise ValueError("K must be a multiple of 4; max_K/K must be a power of two")
    observed = validate_observed(observed_mask)
    units, added = normalize_groups(observed, groups)
    evaluator = MaskEvaluator(predict_masks, observed, batch_size)
    coalition_values = evaluator.evaluate(coalition_masks(observed))
    phi = exact_modality_shapley(coalition_values)
    dividends = anchored_interactions(coalition_values)
    class_order = np.argsort(-coalition_values[7, :3], kind="stable")
    fixed_class, runner_up = (int(class_order[0]), int(class_order[1]))
    locals_out = {}
    for mi, modality in enumerate(MODALITIES):
        modality_units = units[modality]
        n = len(modality_units)
        permutation_digest = hashlib.sha256(
            b"q3_owen_antithetic_orders_v1\x00" + np.asarray([n], dtype="<u4").tobytes()
        )
        if not n:
            locals_out[modality] = {
                "groups": [],
                "values": [],
                "pair_se": [],
                "K": 0,
                "independent_pairs": 0,
                "history": [],
                "status": "empty_modality",
                "sum_residual": (-phi[mi]).tolist(),
                "total_abs_local": [0.0] * 4,
                "contrastive_values": [],
                "contrastive_pair_se": [],
                "permutation_sha256": permutation_digest.hexdigest(),
                "permutation_hash_format": PERMUTATION_HASH_FORMAT,
                "seed_sequence_entropy": [int(seed), mi],
                "bit_generator": "PCG64",
                "rank_diagnostics_may_be_undefined": True,
            }
            continue
        rng = np.random.default_rng(np.random.SeedSequence([int(seed), mi]))
        samples, history = ([], [])
        current_K = int(K)
        while True:
            while len(samples) < current_K // 2:
                permutation = rng.permutation(n)
                for order in (permutation, permutation[::-1]):
                    permutation_digest.update(
                        np.asarray(order, dtype="<i4").tobytes(order="C")
                    )
                samples.append(
                    _pair_marginals(evaluator, mi, modality_units, permutation)
                )
            diagnostic, local, se = stability_diagnostics(
                np.stack(samples), targets=stability_targets
            )
            history.append({"K": current_K, **diagnostic})
            if not adaptive or not diagnostic["requires_upgrade"] or current_K >= max_K:
                break
            current_K *= 2
        residual = local.sum(0) - phi[mi]
        if np.max(np.abs(residual)) > tolerance:
            raise ArithmeticError(
                f"Owen/Shapley efficiency failed for {modality}: {residual}"
            )
        rank_undefined = any(
            (v["rank_status"] != "defined" for v in diagnostic["targets"].values())
        )
        status = (
            "low_stability"
            if diagnostic["requires_upgrade"]
            else (
                "precision_satisfied_rank_partly_undefined"
                if rank_undefined
                else "diagnostics_satisfied"
            )
        )
        record = {
            "groups": modality_units,
            "values": local.tolist(),
            "pair_se": se.tolist(),
            "K": current_K,
            "independent_pairs": len(samples),
            "history": history,
            "status": status,
            "permutation_sha256": permutation_digest.hexdigest(),
            "permutation_hash_format": PERMUTATION_HASH_FORMAT,
            "seed_sequence_entropy": [int(seed), mi],
            "bit_generator": "PCG64",
            "rank_diagnostics_may_be_undefined": rank_undefined,
            "sum_residual": residual.tolist(),
            "total_abs_local": np.abs(local).sum(0).tolist(),
            "contrastive_values": (
                local[:, fixed_class] - local[:, runner_up]
            ).tolist(),
            "contrastive_pair_se": (
                (
                    np.stack(samples)[:, :, fixed_class]
                    - np.stack(samples)[:, :, runner_up]
                ).std(0, ddof=1)
                / math.sqrt(len(samples))
            ).tolist(),
        }
        if include_pair_samples:
            record["pair_samples"] = np.stack(samples).tolist()
        locals_out[modality] = record
    residual = phi.sum(0) - (coalition_values[7] - coalition_values[0])
    if np.max(np.abs(residual)) > tolerance:
        raise ArithmeticError("modality Shapley efficiency failed")
    regression_dominance = _dominance(phi, 3, near_zero_total)
    regression_dominance["largest_upward_modality"] = regression_dominance.pop(
        "positive_support"
    )
    if regression_dominance["largest_upward_modality"] == "none_positive":
        regression_dominance["largest_upward_modality"] = "none_upward"
    down = int(np.argmin(phi[:, 3]))
    regression_dominance["largest_downward_modality"] = (
        MODALITIES[down] if phi[down, 3] < 0 else "none_downward"
    )
    return {
        "schema_version": 1,
        "method": "exact_modality_shapley_grouped_owen",
        "output_names": list(OUTPUTS),
        "modality_order": list(MODALITIES),
        "observed_mask": observed.tolist(),
        "groups": units,
        "unmatched_observed_singletons_added": added,
        "seed": int(seed),
        "seed_hash": hashlib.sha256(
            f"q3_owen_seed_v1:{int(seed)}".encode("ascii")
        ).hexdigest(),
        "seed_hash_format": "sha256 ASCII 'q3_owen_seed_v1:' followed by decimal seed",
        "permutation_sha256": {
            m: locals_out[m]["permutation_sha256"] for m in MODALITIES
        },
        "initial_K": int(K),
        "maximum_K": int(max_K),
        "adaptive": bool(adaptive),
        "se_is_valid_95pct_interval": False,
        "predicted_class": fixed_class,
        "runner_up_class": runner_up,
        "full_output": coalition_values[7].tolist(),
        "baseline_output": coalition_values[0].tolist(),
        "coalition_bit_order": "T=1,A=2,V=4",
        "coalition_outputs": coalition_values.tolist(),
        "modality_values": phi.tolist(),
        "anchored_interactions": dividends.tolist(),
        "leave_one_modality_out": np.stack(
            [coalition_values[7] - coalition_values[7 ^ 1 << mi] for mi in range(3)]
        ).tolist(),
        "contrastive_modality_values": (
            phi[:, fixed_class] - phi[:, runner_up]
        ).tolist(),
        "classification_dominance": _dominance(phi, fixed_class, near_zero_total),
        "regression_dominance": regression_dominance,
        "local": locals_out,
        "efficiency_residual": residual.tolist(),
        "cache": evaluator.stats(),
    }
