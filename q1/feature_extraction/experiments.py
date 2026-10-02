"""Predeclared grouped OOF diagnostics for Q1 frozen features, never Q2 training.

Requires numpy, scipy and scikit-learn. No encoder is trained or downloaded.
Run with an unused --tag for a changed feature snapshot; the input-hashed
protocol is written before any predictive model is fitted.
"""

from __future__ import annotations
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import time
import warnings
import numpy as np
import scipy
from scipy.stats import spearmanr
import sklearn
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.exceptions import ConvergenceWarning

SEED = 20260923
CLASSES = ("Negative", "Neutral", "Positive")
BRANCHES = ("text", "audio", "acoustic", "face", "video")
DIMENSIONS = dict(text=1024, audio=1024, acoustic=25, face=44, video=768)
SCENARIOS = (
    "clean",
    "drop_text",
    "drop_audio",
    "drop_visual",
    "contiguous_av_30pct",
    "shift_av_plus200ms",
    "shift_av_minus200ms",
)
VARIANTS = {
    "majority_mean": (),
    "T": ("text",),
    "TA": ("text", "audio"),
    "TAV": ("text", "audio", "video"),
    "LOW": ("acoustic", "face"),
    "F0": ("text", "audio", "acoustic", "face"),
    "F0_zero_impute_invalid": (
        "text_unmasked",
        "audio_unmasked",
        "acoustic_unmasked",
        "face_unmasked",
    ),
    "F1": ("text", "audio", "acoustic", "face", "video"),
    "F1_sync": (
        "text",
        "audio",
        "acoustic",
        "face",
        "video",
        "TA_sync",
        "TV_sync",
        "AV_sync",
    ),
    "F1_correspondence_removed": (
        "text",
        "audio",
        "acoustic",
        "face",
        "video",
        "TA_removed",
        "TV_removed",
        "AV_removed",
    ),
}
COMPARISONS = (
    ("T", "majority_mean"),
    ("TA", "T"),
    ("TAV", "TA"),
    ("LOW", "majority_mean"),
    ("F0", "TA"),
    ("F0", "F0_zero_impute_invalid"),
    ("F1", "F0"),
    ("F1_sync", "F1"),
    ("F1_sync", "F1_correspondence_removed"),
)


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_hash(value):
    value = np.ascontiguousarray(value)
    return hashlib.sha256(
        str(value.shape).encode() + str(value.dtype).encode() + value.tobytes()
    ).hexdigest()


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and (not math.isfinite(value)):
        return None
    return value


def write_json(path, value):
    path.write_text(
        json.dumps(json_safe(value), ensure_ascii=False, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"No rows for {path}")
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(json_safe(rows))


def count_classes(y, folds):
    return np.asarray([np.bincount(y[folds == f], minlength=3) for f in range(5)])


def repair_class_coverage(y, groups, folds):
    """Deterministic whole-group swaps; uses labels/counts, never predictions."""
    folds = folds.copy()
    unique = sorted(set(groups))
    group_rows = {g: np.flatnonzero(groups == g) for g in unique}
    logs = []
    for _ in range(15):
        counts = count_classes(y, folds)
        absent = int((counts == 0).sum())
        if absent == 0:
            return (folds, logs)
        target = np.bincount(y, minlength=3) / 5
        best = None
        for i, left in enumerate(unique):
            li = group_rows[left]
            lf = int(folds[li[0]])
            for right in unique[i + 1 :]:
                ri = group_rows[right]
                rf = int(folds[ri[0]])
                if lf == rf:
                    continue
                changed = counts.copy()
                lc, rc = (
                    np.bincount(y[li], minlength=3),
                    np.bincount(y[ri], minlength=3),
                )
                changed[lf] += rc - lc
                changed[rf] += lc - rc
                missing = int((changed == 0).sum())
                if missing >= absent:
                    continue
                imbalance = float(np.sum((changed - target) ** 2 / (target + 1)))
                key = (missing, imbalance, left, right)
                if best is None or key < best[0]:
                    best = (key, left, right, lf, rf)
        if best is None:
            raise ValueError(
                "Cannot obtain all-three-class validation folds by declared group-swap repair; redesign must precede predictive evaluation"
            )
        _, left, right, lf, rf = best
        folds[group_rows[left]], folds[group_rows[right]] = (rf, lf)
        logs.append(
            dict(
                left_video_id=left,
                right_video_id=right,
                left_original_fold=lf,
                right_original_fold=rf,
                missing_class_cells_before=absent,
                missing_class_cells_after=best[0][0],
                count_imbalance_after=best[0][1],
            )
        )
    raise ValueError("Fold coverage repair exceeded its fixed iteration bound")


def make_folds(y, groups):
    original = np.full(len(y), -1, int)
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    for fold, (_, test) in enumerate(splitter.split(np.zeros((len(y), 1)), y, groups)):
        original[test] = fold
    folds, repairs = repair_class_coverage(y, groups, original)
    for fold in range(5):
        train, test = (np.flatnonzero(folds != fold), np.flatnonzero(folds == fold))
        if (
            set(groups[train]) & set(groups[test])
            or len(set(y[train])) != 3
            or len(set(y[test])) != 3
        ):
            raise ValueError("Fold identity leakage or missing-class coverage")
    return (original, folds, repairs)


def load_inputs(index_path, label_path, manifest_path):
    index = read_csv(index_path)
    labels = read_csv(label_path)
    manifests = [
        json.loads(s)
        for s in manifest_path.read_text(encoding="utf-8-sig").splitlines()
        if s.strip()
    ]
    tables = []
    for rows, name in ((index, "index"), (labels, "labels"), (manifests, "manifest")):
        keys = [r["sample_key"] for r in rows]
        if len(set(keys)) != len(keys):
            raise ValueError(f"Duplicate sample_key in {name}")
        tables.append({r["sample_key"]: r for r in rows})
    ix, lab, manifest = tables
    if not set(ix) == set(lab) == set(manifest):
        raise ValueError(
            "Index, labels, and manifest must contain the exact same sample keys"
        )
    if len(ix) != 100:
        raise ValueError("This predeclared Q1 diagnosis requires all 100 samples")
    records, hashes = ([], {})
    for key in sorted(ix):
        r, l, m = (ix[key], lab[key], manifest[key])
        if not r["official_sample_id"] == l["sample_id"] == m["sample_id"]:
            raise ValueError(f"{key}: official sample identity mismatch")
        if (
            r["source_sha256"] != m["source_sha256"]
            or m["sample_id"].split("$_$")[0] != m["video_id"]
        ):
            raise ValueError(f"{key}: source/group identity mismatch")
        if (
            l["annotation"] not in CLASSES
            or not math.isfinite(float(l["label"]))
            or (not -3 <= float(l["label"]) <= 3)
        ):
            raise ValueError(f"{key}: unknown official annotation/intensity")
        paths = {b: (index_path.parent / r[b + "_path"]).resolve() for b in BRANCHES}
        paths["alignment"] = (index_path.parent / r["alignment_path"]).resolve()
        for name, path in paths.items():
            if not path.is_file():
                raise ValueError(f"{key}: missing {name} file {path}")
            hashes[str(path)] = file_hash(path)
        records.append(
            dict(
                sample_key=key,
                sample_id=m["sample_id"],
                video_id=m["video_id"],
                y_class=CLASSES.index(l["annotation"]),
                y_value=float(l["label"]),
                paths=paths,
            )
        )
    return (records, hashes)


def read_features(record):
    alignment = json.loads(record["paths"]["alignment"].read_text(encoding="utf-8"))
    duration = float(alignment["duration_s"])
    words = alignment["words"]
    word_times = np.asarray(
        [
            [w["start"], w["end"]] if w.get("alignment_valid") else [np.nan, np.nan]
            for w in words
        ],
        float,
    ).reshape(-1, 2)
    result = {
        "key": record["sample_key"],
        "duration": duration,
        "word_times": word_times,
        "branches": {},
    }
    for branch in BRANCHES:
        with np.load(record["paths"][branch], allow_pickle=False) as archive:
            data = {k: archive[k] for k in archive.files}
        values = data["features"]
        if (
            values.dtype != np.float32
            or values.ndim != 2
            or values.shape[1] != DIMENSIONS[branch]
        ):
            raise ValueError(f"{record['sample_key']}: unexpected {branch} features")
        time_key, mask_key = (
            ("time_s", "valid_mask")
            if branch in ("text", "audio", "acoustic")
            else ("times", "valid")
        )
        times = data[time_key]
        if times.dtype != np.float64 or times.shape != (len(values), 2):
            raise ValueError(f"{record['sample_key']}: invalid {branch} timestamps")
        valid = data[mask_key].astype(bool)
        if valid.ndim == 1:
            valid = np.broadcast_to(valid[:, None], values.shape).copy()
        if valid.shape != values.shape or not np.isfinite(values[valid]).all():
            raise ValueError(f"{record['sample_key']}: invalid {branch} mask/value")
        if branch == "text":
            coverage = np.broadcast_to(
                data["character_coverage"][:, None], values.shape
            ).copy()
        else:
            coverage = data["valid_time_fraction" if branch == "face" else "coverage"]
            if coverage.ndim == 1:
                coverage = np.broadcast_to(coverage[:, None], values.shape).copy()
        if (
            not np.isfinite(coverage).all()
            or np.any(coverage < 0)
            or np.any(coverage > 1 + 1e-06)
        ):
            raise ValueError(f"{record['sample_key']}: invalid {branch} coverage")
        if branch in ("text", "audio", "video") and len(values) < len(words):
            raise ValueError(
                f"{record['sample_key']}: {branch} lost original word rows"
            )
        result["branches"][branch] = dict(
            values=values.astype(float),
            times=times.copy(),
            valid=valid,
            coverage=coverage.astype(float),
        )
    return result


def weighted_stats(values, weights, std=False):
    weights = np.where(np.isfinite(values), weights, 0.0)
    values = np.where(np.isfinite(values), values, 0.0)
    denominator = weights.sum(axis=0)
    mean = np.divide(
        (values * weights).sum(axis=0),
        denominator,
        out=np.full(values.shape[1], np.nan),
        where=denominator > 0,
    )
    if not std:
        return mean
    variance = np.divide(
        ((values - np.nan_to_num(mean)) ** 2 * weights).sum(axis=0),
        denominator,
        out=np.full_like(mean, np.nan),
        where=denominator > 0,
    )
    return np.r_[mean, np.sqrt(np.maximum(variance, 0))]


def transformed_branch(sample, branch, scenario):
    raw = sample["branches"][branch]
    values, valid = (raw["values"], raw["valid"].copy())
    times, coverage = (raw["times"].copy(), raw["coverage"])
    drop = (
        scenario == "drop_text"
        and branch == "text"
        or (scenario == "drop_audio" and branch in ("audio", "acoustic"))
        or (scenario == "drop_visual" and branch in ("face", "video"))
    )
    if drop:
        valid[:] = False
    if branch != "text":
        known = (
            np.isfinite(times).all(axis=1)
            & (times[:, 0] >= 0)
            & (times[:, 1] > times[:, 0])
        )
        valid &= known[:, None]
        if scenario == "contiguous_av_30pct":
            middle = times.mean(axis=1)
            valid[
                (middle >= 0.35 * sample["duration"])
                & (middle < 0.65 * sample["duration"])
            ] = False
        if scenario in ("shift_av_plus200ms", "shift_av_minus200ms"):
            times[known] += 0.2 if scenario == "shift_av_plus200ms" else -0.2
        times[known] = np.clip(times[known], 0, sample["duration"])
        span = np.where(known, np.maximum(0, times[:, 1] - times[:, 0]), 0)
        weights = valid * coverage * span[:, None]
    else:
        weights = valid * coverage
    return (values, times, valid, coverage, weights)


def word_vectors(sample, branch, transformed, scenario):
    values, times, valid, coverage, weights = transformed
    n = len(sample["word_times"])
    if branch == "text" or scenario not in (
        "shift_av_plus200ms",
        "shift_av_minus200ms",
    ):
        return (values[:n], valid[:n].all(axis=1) & (weights[:n].sum(axis=1) > 0))
    output = np.zeros((n, values.shape[1]))
    available = np.zeros(n, bool)
    for i, (a, b) in enumerate(sample["word_times"]):
        if not np.isfinite([a, b]).all():
            continue
        overlap = np.maximum(
            0.0, np.minimum(b, times[:, 1]) - np.maximum(a, times[:, 0])
        )
        overlap = np.where(np.isfinite(overlap), overlap, 0.0)
        row_weights = overlap * valid.all(axis=1) * coverage.mean(axis=1)
        if row_weights.sum() > 0:
            output[i] = (np.nan_to_num(values) * row_weights[:, None]).sum(
                axis=0
            ) / row_weights.sum()
            available[i] = True
    return (output, available)


def projected_words(values, branch):
    branch_seed = SEED + {"text": 11, "audio": 23, "video": 37}[branch]
    projection = np.random.default_rng(branch_seed).normal(
        size=(values.shape[1], 8)
    ) / math.sqrt(values.shape[1])
    reduced = np.nan_to_num(values) @ projection
    return reduced / np.maximum(np.linalg.norm(reduced, axis=1, keepdims=True), 1e-12)


def covariance_block(left, right, valid):
    if valid.sum() < 2:
        return np.full(64, np.nan)
    a, b = (left[valid], right[valid])
    return ((a - a.mean(axis=0)).T @ (b - b.mean(axis=0)) / len(a)).reshape(-1)


def describe(sample, scenario):
    descriptors, projected, masks = ({}, {}, {})
    for branch in BRANCHES:
        transformed = transformed_branch(sample, branch, scenario)
        values, _, _, _, weights = transformed
        descriptors[branch] = weighted_stats(
            values, weights, std=branch in ("acoustic", "face")
        )
        if branch != "video":
            _, times, valid, _, _ = transformed
            naive_values = np.where(valid & np.isfinite(values), values, 0.0)
            if branch == "text":
                naive_weights = np.ones_like(values)
            else:
                known = (
                    np.isfinite(times).all(axis=1)
                    & (times[:, 0] >= 0)
                    & (times[:, 1] > times[:, 0])
                )
                spans = np.where(known, np.maximum(times[:, 1] - times[:, 0], 0.0), 0.0)
                naive_weights = np.broadcast_to(spans[:, None], values.shape)
            descriptors[branch + "_unmasked"] = weighted_stats(
                naive_values, naive_weights, std=branch in ("acoustic", "face")
            )
        if branch in ("text", "audio", "video"):
            word_values, valid = word_vectors(sample, branch, transformed, scenario)
            projected[branch], masks[branch] = (
                projected_words(word_values, branch),
                valid,
            )
    removed, removed_masks = ({}, {})
    for branch in projected:
        n = len(projected[branch])
        if branch == "text" or n < 2:
            lag = 0
        else:
            token = f"{SEED}|{sample['key']}|{branch}".encode()
            lag = 1 + int.from_bytes(hashlib.sha256(token).digest()[:8], "big") % (
                n - 1
            )
        removed[branch] = np.roll(projected[branch], lag, axis=0)
        removed_masks[branch] = np.roll(masks[branch], lag)
    for prefix, left, right in (
        ("TA", "text", "audio"),
        ("TV", "text", "video"),
        ("AV", "audio", "video"),
    ):
        descriptors[prefix + "_sync"] = covariance_block(
            projected[left], projected[right], masks[left] & masks[right]
        )
        descriptors[prefix + "_removed"] = covariance_block(
            removed[left], removed[right], removed_masks[left] & removed_masks[right]
        )
    return descriptors


class TrainOnlyBlocks:
    """Training-only mean imputation/scaling, equal expected block energy.

    All-missing test blocks impute training means; one availability indicator per
    block is appended. No PCA is used anywhere in this protocol.
    """

    def fit(self, blocks):
        self.statistics = {}
        for key, values in blocks.items():
            finite = np.isfinite(values)
            counts = finite.sum(axis=0)
            mean = np.divide(
                np.where(finite, values, 0).sum(axis=0),
                counts,
                out=np.zeros(values.shape[1]),
                where=counts > 0,
            )
            centered = np.where(finite, values - mean, 0)
            scale = np.sqrt(
                np.divide(
                    (centered**2).sum(axis=0),
                    counts,
                    out=np.zeros_like(mean),
                    where=counts > 0,
                )
            )
            scale[scale < 1e-12] = 1.0
            available_mean = finite.any(axis=1).mean()
            self.statistics[key] = (mean, scale, available_mean)
        return self

    def transform(self, blocks):
        output = []
        for key, (mean, scale, available_mean) in self.statistics.items():
            values = blocks[key]
            finite = np.isfinite(values)
            normal = (
                (np.where(finite, values, mean) - mean)
                / scale
                / math.sqrt(values.shape[1])
            )
            output.append(
                np.column_stack(
                    (normal, finite.any(axis=1).astype(float) - available_mean)
                )
            )
        return np.concatenate(output, axis=1)

    def audit(self):
        return {
            k: {
                "mean_sha256": array_hash(v[0]),
                "scale_sha256": array_hash(v[1]),
                "dimensions": len(v[0]),
                "training_available_fraction": v[2],
            }
            for k, v in self.statistics.items()
        }


def metric_values(y, truth, predicted, estimate):
    confusion = np.bincount(y * 3 + predicted, minlength=9).reshape(3, 3)
    tp = np.diag(confusion).astype(float)
    precision = np.divide(
        tp, confusion.sum(axis=0), out=np.zeros(3), where=confusion.sum(axis=0) > 0
    )
    recall = np.divide(
        tp, confusion.sum(axis=1), out=np.zeros(3), where=confusion.sum(axis=1) > 0
    )
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros(3),
        where=precision + recall > 0,
    )
    pearson = (
        float(np.corrcoef(truth, estimate)[0, 1])
        if np.std(truth) > 1e-12 and np.std(estimate) > 1e-12
        else np.nan
    )
    return (
        {
            "accuracy": float((y == predicted).mean()),
            "macro_f1": float(f1.mean()),
            "balanced_accuracy": float(recall.mean()),
            "mae": float(np.abs(truth - estimate).mean()),
            "rmse": float(np.sqrt(np.mean((truth - estimate) ** 2))),
            "pearson": pearson,
        },
        confusion,
        precision,
        recall,
        f1,
    )


def bootstrap_indices(groups, repeats):
    unique = np.asarray(sorted(set(groups)))
    by_group = {g: np.flatnonzero(groups == g) for g in unique}
    rng = np.random.default_rng(SEED + 101)
    return [
        np.concatenate(
            [by_group[g] for g in rng.choice(unique, len(unique), replace=True)]
        )
        for _ in range(repeats)
    ]


def interval(values):
    values = np.asarray(values)
    finite = values[np.isfinite(values)]
    return (
        [float(x) for x in np.percentile(finite, [2.5, 97.5])]
        if len(finite)
        else [None, None]
    )


def protocol_document(args, records, hashes, original, folds, repairs):
    return {
        "schema_version": 1,
        "purpose": "Q1 frozen-feature diagnostic only; labels never enter feature extraction or Q2 training",
        "registration_stage": "Written before descriptor construction and before any predictive fit",
        "seed": SEED,
        "bootstrap_seed": SEED + 101,
        "bootstrap_repeats": args.bootstrap,
        "samples": len(records),
        "groups": len({r["video_id"] for r in records}),
        "group_definition": "official video_id; source video group, not a verified person identity",
        "class_definition": "Official annotation as given: Negative/Neutral/Positive; no regression thresholding",
        "regression_target": "Official label directly, nominal MOSEI intensity [-3,3]; predictions clipped to that fixed scale",
        "class_counts": dict(Counter((CLASSES[r["y_class"]] for r in records))),
        "fold_method": "5-fold StratifiedGroupKFold fixed seed, then deterministic whole-group swaps solely for missing-class coverage; never seed/fold/model-result selection",
        "fold_repairs": repairs,
        "initial_fold_assignments": original.tolist(),
        "locked_fold_assignments": folds.tolist(),
        "classification_head": {
            "class": "LogisticRegression",
            "C": 1.0,
            "penalty": "l2",
            "solver": "lbfgs",
            "class_weight": "balanced",
            "max_iter": 3000,
            "tol": 1e-08,
        },
        "regression_head": {
            "class": "Ridge",
            "alpha": 1.0,
            "solver": "lsqr",
            "tol": 1e-08,
            "max_iter": 3000,
        },
        "preprocessing": "Training-fold-only mean imputation and per-coordinate population standard deviation; each block divided by sqrt(d); append availability minus its training mean. No PCA or hyperparameter search.",
        "descriptors": {
            "text": "1024D masked character-coverage-weighted word mean",
            "audio": "1024D duration*coverage weighted mean over stored word and >=300ms gap nodes",
            "video": "768D duration*coverage weighted mean over stored nodes",
            "acoustic": "25 masked means + 25 standard deviations",
            "face": "44 masked means + 44 standard deviations",
            "interaction": "For each T/A/V pair: fixed label-independent Gaussian projection to 8D, per-word L2 normalization, centered 8x8 covariance over available matching word pairs; not a learned encoder",
        },
        "variants": {k: list(v) for k, v in VARIANTS.items()},
        "clean_comparisons": COMPARISONS,
        "scenarios": list(SCENARIOS),
        "robustness_training": "Always train on clean training folds; perturb only the held-out samples, with fixed conditions and no retraining/tuning",
        "correspondence_control": "F1_correspondence_removed has identical marginal descriptors to F1_sync; only interaction word pairing is cyclically shifted by fixed sample-key hashes during both training and testing",
        "mask_control": "F0_zero_impute_invalid replaces invalid coordinates by zero, drops validity/coverage weighting, and counts all known row durations; unknown times remain excluded. Its availability indicators are consequently constant when any timed row exists. This is a deliberately naive pooling comparator, not a claim that unvoiced F0=0 is a physical observation.",
        "contiguous_mask": "Drop cached A/LLD/V/Face rows whose time midpoint falls in [35%,65%) of clip duration; text unchanged",
        "timestamp_shift": "Shift cached A/LLD/V/Face support by +/-200ms, clip to clip extent, and re-pool cached A/V vectors to original word intervals for interactions",
        "uncertainty": "2000 by default: resample all video_id clusters with replacement, include every clip in each sampled cluster, common resamples for all paired contrasts; percentile 95% intervals",
        "limitations": [
            "N=100 and only 37 source-video groups; exploratory Q1 information diagnostic, no SOTA claim or independent test-set estimate",
            "Deep inputs have already been word-pooled; removing correspondence is a controlled proxy, not a true alignment-free re-extraction",
            "Perturbations operate on cached contextual representations. Unmasked vectors can retain full-clip context: this is not raw missing-modality training or Q2 benchmark performance",
            "Fold count repair depends only on class/group counts, which is allowed for stratification; it never uses predictions",
            "Cluster-bootstrap intervals condition on these OOF models and folds, not the entire model-selection or training process",
            "Initial numeric/silent alignment states are retained; no sample is excluded based on quality or label",
            "Projection to 8D may discard synchrony information; absence of benefit does not prove absence of useful alignment",
        ],
        "input_files_sha256": hashes,
        "label_file_sha256": file_hash(args.labels),
        "index_file_sha256": file_hash(args.index),
        "manifest_sha256": file_hash(args.manifest),
        "implementation_sha256": file_hash(Path(__file__)),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "sklearn": sklearn.__version__,
        },
    }


def run(args):
    started = time.monotonic()
    if args.bootstrap < 100:
        raise ValueError(
            "Use at least 100 cluster bootstrap replicates; default is 2000"
        )
    if not args.tag or any(
        (
            c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
            for c in args.tag
        )
    ):
        raise ValueError("--tag must be a simple ASCII run name")
    destination = args.output / args.tag
    records, hashes = load_inputs(args.index, args.labels, args.manifest)
    y = np.asarray([r["y_class"] for r in records], int)
    truth = np.asarray([r["y_value"] for r in records], float)
    groups = np.asarray([r["video_id"] for r in records])
    original, folds, repairs = make_folds(y, groups)
    protocol = protocol_document(args, records, hashes, original, folds, repairs)
    protocol_path = destination / "protocol.json"
    if protocol_path.exists():
        existing = json.loads(protocol_path.read_text(encoding="utf-8"))
        existing.pop("registered_utc", None)
        if existing != json_safe(protocol):
            raise ValueError(
                "Existing run protocol differs; use a new --tag, do not overwrite a prior registration"
            )
    else:
        destination.mkdir(parents=True, exist_ok=True)
        registered = dict(
            protocol, registered_utc=datetime.now(timezone.utc).isoformat()
        )
        write_json(protocol_path, registered)
    fold_rows = [
        dict(
            sample_key=r["sample_key"],
            sample_id=r["sample_id"],
            video_id=r["video_id"],
            initial_fold=int(original[i]),
            fold=int(folds[i]),
            official_annotation=CLASSES[y[i]],
            official_intensity=truth[i],
        )
        for i, r in enumerate(records)
    ]
    write_csv(destination / "folds.csv", fold_rows)
    fold_summary = [
        dict(
            fold=f,
            train_samples=int((folds != f).sum()),
            test_samples=int((folds == f).sum()),
            train_groups=len(set(groups[folds != f])),
            test_groups=len(set(groups[folds == f])),
            test_negative=int(((folds == f) & (y == 0)).sum()),
            test_neutral=int(((folds == f) & (y == 1)).sum()),
            test_positive=int(((folds == f) & (y == 2)).sum()),
            group_overlap=0,
        )
        for f in range(5)
    ]
    write_csv(destination / "fold_summary.csv", fold_summary)
    print(
        f"Protocol locked: {destination}; N={len(y)}, groups={len(set(groups))}, count-only fold repairs={len(repairs)}",
        flush=True,
    )
    samples = [read_features(r) for r in records]
    descriptors = {}
    for scenario in SCENARIOS:
        described = [describe(sample, scenario) for sample in samples]
        descriptors[scenario] = {
            name: np.stack([item[name] for item in described]) for name in described[0]
        }
    audits = []
    for record, sample in zip(records, samples):
        audits.append(
            dict(
                sample_key=record["sample_key"],
                aligned_word_fraction=float(
                    np.isfinite(sample["word_times"]).all(axis=1).mean()
                ),
                branches={
                    b: {
                        "feature_rows": len(raw["values"]),
                        "valid_feature_fraction": float(raw["valid"].mean()),
                    }
                    for b, raw in sample["branches"].items()
                },
            )
        )
    write_json(destination / "inputs_audit.json", audits)
    predictions, preprocessing, convergence = ({}, [], [])
    for variant, blocks in VARIANTS.items():
        for scenario in SCENARIOS:
            predictions[variant, scenario] = {
                "class": np.full(len(y), -1, int),
                "value": np.full(len(y), np.nan),
                "probability": np.full((len(y), 3), np.nan),
            }
        for fold in range(5):
            train, test = (np.flatnonzero(folds != fold), np.flatnonzero(folds == fold))
            if blocks:
                raw_train = {b: descriptors["clean"][b][train] for b in blocks}
                scaler = TrainOnlyBlocks().fit(raw_train)
                x_train = scaler.transform(raw_train)
                classifier = LogisticRegression(
                    C=1.0,
                    penalty="l2",
                    solver="lbfgs",
                    class_weight="balanced",
                    max_iter=3000,
                    tol=1e-08,
                    random_state=SEED,
                )
                regressor = Ridge(alpha=1.0, solver="lsqr", tol=1e-08, max_iter=3000)
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always", ConvergenceWarning)
                    classifier.fit(x_train, y[train])
                    regressor.fit(x_train, truth[train])
                convergence.extend(
                    (
                        {"variant": variant, "fold": fold, "warning": str(w.message)}
                        for w in caught
                    )
                )
                preprocessing.append(
                    dict(
                        variant=variant,
                        fold=fold,
                        training_sample_keys=[records[i]["sample_key"] for i in train],
                        test_sample_keys=[records[i]["sample_key"] for i in test],
                        training_transformed_sha256=array_hash(x_train),
                        blocks=scaler.audit(),
                        classifier_iterations=classifier.n_iter_.tolist(),
                    )
                )
            for scenario in SCENARIOS:
                output = predictions[variant, scenario]
                if blocks:
                    x_test = scaler.transform(
                        {b: descriptors[scenario][b][test] for b in blocks}
                    )
                    output["class"][test] = classifier.predict(x_test)
                    output["value"][test] = np.clip(
                        regressor.predict(x_test), -3.0, 3.0
                    )
                    output["probability"][test] = classifier.predict_proba(x_test)
                else:
                    majority = int(np.bincount(y[train], minlength=3).argmax())
                    output["class"][test] = majority
                    output["value"][test] = truth[train].mean()
                    output["probability"][test] = np.eye(3)[majority]
        clean = predictions[variant, "clean"]
        clean_metrics = metric_values(y, truth, clean["class"], clean["value"])[0]
        print(
            json.dumps(
                dict(
                    stage="clean_oof_point_scores_before_bootstrap",
                    variant=variant,
                    **clean_metrics,
                )
            ),
            flush=True,
        )
    write_json(destination / "preprocessing_audit.json", preprocessing)
    bootstrap = bootstrap_indices(groups, args.bootstrap)
    metric_rows, fold_metric_rows, oof_rows, details, distributions = (
        [],
        [],
        [],
        {},
        {},
    )
    for (variant, scenario), output in predictions.items():
        if (
            np.any(output["class"] < 0)
            or not np.isfinite(output["value"]).all()
            or (not np.isfinite(output["probability"]).all())
        ):
            raise RuntimeError("Missing OOF predictions")
        values, confusion, precision, recall, f1 = metric_values(
            y, truth, output["class"], output["value"]
        )
        samples_metrics = {k: np.empty(len(bootstrap)) for k in values}
        for j, selection in enumerate(bootstrap):
            measured = metric_values(
                y[selection],
                truth[selection],
                output["class"][selection],
                output["value"][selection],
            )[0]
            for metric in values:
                samples_metrics[metric][j] = measured[metric]
        distributions[variant, scenario] = samples_metrics
        row = dict(
            variant=variant,
            scenario=scenario,
            n=len(y),
            video_groups=len(set(groups)),
            **values,
        )
        for metric, distribution in samples_metrics.items():
            row[metric + "_ci_low"], row[metric + "_ci_high"] = interval(distribution)
        metric_rows.append(row)
        details[variant + "/" + scenario] = dict(
            confusion_matrix=confusion,
            confusion_labels=CLASSES,
            per_class=[
                dict(
                    label=CLASSES[k],
                    precision=precision[k],
                    recall=recall[k],
                    f1=f1[k],
                    support=int((y == k).sum()),
                )
                for k in range(3)
            ],
            spearman=(
                float(spearmanr(truth, output["value"]).statistic)
                if np.std(output["value"]) > 1e-12
                else None
            ),
        )
        for fold in range(5):
            selected = folds == fold
            measured = metric_values(
                y[selected],
                truth[selected],
                output["class"][selected],
                output["value"][selected],
            )[0]
            fold_metric_rows.append(
                dict(
                    variant=variant,
                    scenario=scenario,
                    fold=fold,
                    n=int(selected.sum()),
                    **measured,
                )
            )
        for i, r in enumerate(records):
            oof_rows.append(
                dict(
                    sample_key=r["sample_key"],
                    sample_id=r["sample_id"],
                    video_id=r["video_id"],
                    fold=int(folds[i]),
                    variant=variant,
                    scenario=scenario,
                    official_annotation=CLASSES[y[i]],
                    official_intensity=truth[i],
                    predicted_annotation=CLASSES[output["class"][i]],
                    predicted_intensity=output["value"][i],
                    probability_negative=output["probability"][i, 0],
                    probability_neutral=output["probability"][i, 1],
                    probability_positive=output["probability"][i, 2],
                )
            )
    contrasts = [
        (left, "clean", right, "clean", "clean_ablation") for left, right in COMPARISONS
    ]
    contrasts += [
        (variant, scenario, variant, "clean", "heldout_robustness")
        for variant in VARIANTS
        if variant != "majority_mean"
        for scenario in SCENARIOS
        if scenario != "clean"
    ]
    point = {(r["variant"], r["scenario"]): r for r in metric_rows}
    contrast_rows = []
    for left, lscenario, right, rscenario, kind in contrasts:
        for metric in ("accuracy", "macro_f1", "mae", "pearson"):
            delta = (
                distributions[left, lscenario][metric]
                - distributions[right, rscenario][metric]
            )
            low, high = interval(delta)
            contrast_rows.append(
                dict(
                    kind=kind,
                    left_variant=left,
                    left_scenario=lscenario,
                    right_variant=right,
                    right_scenario=rscenario,
                    metric=metric,
                    difference_left_minus_right=point[left, lscenario][metric]
                    - point[right, rscenario][metric],
                    ci_low=low,
                    ci_high=high,
                    favorable_direction="negative" if metric == "mae" else "positive",
                )
            )
    write_csv(destination / "oof_predictions.csv", oof_rows)
    write_csv(destination / "metrics.csv", metric_rows)
    write_csv(destination / "fold_metrics.csv", fold_metric_rows)
    write_csv(destination / "paired_comparisons.csv", contrast_rows)
    write_json(destination / "metric_details.json", details)
    summary = dict(
        status="complete",
        samples=len(records),
        groups=len(set(groups)),
        variants=len(VARIANTS),
        scenarios=len(SCENARIOS),
        oof_rows=len(oof_rows),
        bootstrap_repeats=args.bootstrap,
        fold_repairs=repairs,
        convergence_warnings=convergence,
        elapsed_seconds=time.monotonic() - started,
        protocol_sha256=file_hash(protocol_path),
        label_use="Q1 supervised diagnostic only; not passed to extraction/Q2",
        output=str(destination.resolve()),
        interpretation="Use pooled OOF metrics and paired cluster-bootstrap intervals. No best fold/seed/model is selected, and a point-score gain with an interval crossing zero is inconclusive.",
    )
    write_json(destination / "run_summary.json", summary)
    print(json.dumps(json_safe(summary), ensure_ascii=False, indent=2), flush=True)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--index", type=Path, default=Path("features/samples_index.csv")
    )
    parser.add_argument("--labels", type=Path, default=Path("labels/q1_labels.csv"))
    parser.add_argument("--manifest", type=Path, default=Path("data/manifest.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("experiments"))
    parser.add_argument("--tag", default="reproduced")
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
