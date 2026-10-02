"""Stable-ID, absolute-position missingness banks. Clean masks are diagnostics only."""

from __future__ import annotations
import itertools
import numpy as np
from .data import LENGTH, stable_seed, sanitize_tokens, observed_masks, build_structure

MASK_VERSION = "q2-block-v1"
SETS = ("T", "A", "V", "TA", "TV", "AV", "TAV")


def condition(
    name, modalities="", rate=0.0, mechanism="block", sync=True, position="all", **kw
):
    return dict(
        name=name,
        modalities=modalities,
        rate=float(rate),
        mechanism=mechanism,
        sync=bool(sync),
        position=position,
        **kw,
    )


def make_core_conditions():
    out = [condition("clean", mechanism="clean")]
    for mods in SETS:
        for sync in [True] if len(mods) == 1 else [True, False]:
            for rate in (0.2, 0.4, 0.6):
                out.append(
                    condition(
                        f"{mods}_{('sync' if sync else 'async')}_r{rate:.1f}",
                        mods,
                        rate,
                        sync=sync,
                    )
                )
    return out


def make_grid_conditions():
    out = [condition("clean", mechanism="clean")]
    for mods in SETS:
        for sync in [True] if len(mods) == 1 else [True, False]:
            for rate in (0.1, 0.3, 0.5, 0.7):
                for position in ("front", "middle", "back"):
                    out.append(
                        condition(
                            f"{mods}_{('sync' if sync else 'async')}_r{rate:.1f}_{position}",
                            mods,
                            rate,
                            sync=sync,
                            position=position,
                        )
                    )
    return out


def make_training_condition(family, view):
    if family not in ("point", "block"):
        raise ValueError("training family must be point or block")
    return condition(
        f"train_{family}_{view:02d}", mechanism="train", family=family, view=int(view)
    )


def _draw_training(c, rng):
    if rng.random() < 0.2:
        return condition(c["name"], mechanism="clean")
    mods = SETS[int(rng.integers(len(SETS)))]
    mechanism = (
        "point"
        if c["family"] == "point"
        else str(rng.choice(["block", "two_block", "point"], p=[0.5, 0.3, 0.2]))
    )
    return condition(
        c["name"],
        mods,
        float(rng.choice(np.arange(1, 8) / 10)),
        mechanism,
        sync=True if len(mods) == 1 else bool(rng.integers(2)),
    )


def _start(rng, max_start, position):
    ranges = {
        "all": (0.0, 1.0),
        "front": (0.0, 1 / 3),
        "middle": (1 / 3, 2 / 3),
        "back": (2 / 3, 1.0),
    }
    if position not in ranges:
        raise ValueError(f"unknown position {position}")
    a, b = ranges[position]
    return min(max_start, int(np.floor(rng.uniform(a, b) * (max_start + 1))))


def _draw_pattern(rng, start, stop, rate, mechanism, position):
    width = stop - start
    bits = np.zeros(LENGTH, dtype=bool)
    if width < 2:
        return bits
    k = min(width - 1, max(1, int(np.floor(rate * width + 0.5))))
    if mechanism == "point":
        bits[start + rng.choice(width, k, replace=False)] = True
    elif mechanism == "block":
        offset = _start(rng, width - k, position)
        bits[start + offset : start + offset + k] = True
    elif mechanism == "two_block":
        if k < 2 or width < 3:
            return _draw_pattern(rng, start, stop, rate, "block", position)
        first = k // 2
        second = k - first
        gap = int(rng.integers(1, width - k + 1))
        offset = _start(rng, width - k - gap, position)
        bits[start + offset : start + offset + first] = True
        right = start + offset + first + gap
        bits[right : right + second] = True
    else:
        raise ValueError(f"unknown mechanism {mechanism}")
    return bits


def _runs(bits):
    edges = np.diff(np.r_[False, bits, False].astype(np.int8))
    return [
        [int(a), int(b)]
        for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))
    ]


def make_masks(tokens, audio, vision, condition, seed, ids):
    """Return damaged inputs and independent diagnostics; never expose clean masks to student.

    Bounded rejection checks actual sparse observations. If no valid block is found in
    64 attempts, the relevant block stays empty and is explicitly recorded, preserving
    all samples. Synchronised patterns are shared exactly, including conservative skip.
    """
    tok = sanitize_tokens(tokens)
    audio, vision = (
        np.asarray(audio, np.float32).copy(),
        np.asarray(vision, np.float32).copy(),
    )
    original = observed_masks(tok, audio, vision)
    if len(ids) != len(tok) or len(set(map(str, ids))) != len(ids):
        raise ValueError("stable unique IDs required")
    drop = np.zeros_like(original)
    records = []
    for i, sample_id in enumerate(ids):
        rng = np.random.default_rng(
            stable_seed(MASK_VERSION, int(seed), str(sample_id), condition)
        )
        c = (
            _draw_training(condition, rng)
            if condition["mechanism"] == "train"
            else dict(condition)
        )
        if not 0 <= c["rate"] <= 1:
            raise ValueError("rate must lie in [0,1]")
        mods = ["TAV".index(m) for m in c["modalities"]]
        content = np.flatnonzero(original[i, 0])
        if not len(content):
            content = np.flatnonzero(original[i].any(0))
        start, stop = (
            (int(content[0]), int(content[-1] + 1)) if len(content) else (0, 0)
        )
        statuses = []
        if c["mechanism"] != "clean" and c["rate"] > 0 and mods:
            groups = [mods] if c["sync"] else [[m] for m in mods]
            for group in groups:
                capable = [m for m in group if original[i, m].sum() >= 2]
                selected = None
                if not capable or any((original[i, m].sum() == 1 for m in group)):
                    statuses.append("short_or_single_observation_skip")
                else:
                    for _ in range(64):
                        candidate = _draw_pattern(
                            rng, start, stop, c["rate"], c["mechanism"], c["position"]
                        )
                        if all(
                            (
                                0
                                < (candidate & original[i, m]).sum()
                                < original[i, m].sum()
                                for m in capable
                            )
                        ):
                            selected = candidate
                            break
                    if selected is None:
                        statuses.append("no_feasible_sampled_pattern_skip")
                if selected is not None:
                    for m in group:
                        drop[i, m] = selected
            if c["mechanism"] == "two_block" and any(
                (len(_runs(drop[i, m])) == 1 for m in mods)
            ):
                statuses.append("two_block_degenerated_to_one")
        elif c["mechanism"] == "clean":
            statuses.append("clean")
        initial = original[i].sum(1)
        removed = (drop[i] & original[i]).sum(1)
        records.append(
            {
                "id": str(sample_id),
                "condition": c,
                "sampling_span": [start, stop],
                "original_count": initial.tolist(),
                "newly_deleted_count": removed.tolist(),
                "actual_new_rate": [
                    float(removed[m] / initial[m]) if initial[m] else None
                    for m in range(3)
                ],
                "final_unavailable_fraction_50": (
                    1 - (initial - removed) / LENGTH
                ).tolist(),
                "intervals_half_open": [_runs(x) for x in drop[i]],
                "deleted_observed_positions": [
                    np.flatnonzero(drop[i, m] & original[i, m]).tolist()
                    for m in range(3)
                ],
                "status": statuses,
                "short_position_groups_can_coincide": stop - start < 4,
            }
        )
    tok[:, 0][drop[:, 0]] = 0
    tok[:, 1][drop[:, 0]] = 0
    tok[:, 2][drop[:, 0]] = 0
    audio[drop[:, 1]] = 0
    vision[drop[:, 2]] = 0
    mask = observed_masks(tok, audio, vision)
    return {
        "tokens": tok,
        "audio": audio,
        "vision": vision,
        "mask": mask,
        "structure": build_structure(mask),
        "diagnostics": {
            "version": MASK_VERSION,
            "seed": int(seed),
            "drop_mask": drop,
            "records": records,
        },
    }
