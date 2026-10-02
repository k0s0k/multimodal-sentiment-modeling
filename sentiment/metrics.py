"""Three-class, continuous-score and paired video-cluster evaluation (NumPy only).

Classes are fixed at 0=Negative, 1=Neutral (target exactly zero), 2=Positive.
Mask repetitions are averaged at the metric level before selecting worst conditions.
Bootstrap intervals condition on fitted models; they are not training-seed variance.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import numpy as np

CLASS_NAMES = ("Negative", "Neutral", "Positive")
SCALAR_METRICS = ("accuracy", "macro_f1", "weighted_f1", "mae", "pearson", "rmse")


@dataclass
class PredictionSet:
    probs: np.ndarray
    pred: np.ndarray
    ids: np.ndarray
    video_ids: np.ndarray
    labels: np.ndarray | None = None
    targets: np.ndarray | None = None
    representation: np.ndarray | None = None

    def __post_init__(self):
        self.probs = np.asarray(self.probs, dtype=np.float64)
        self.pred = np.asarray(self.pred, dtype=np.float64).reshape(-1)
        self.ids = np.asarray(self.ids, dtype=str).reshape(-1)
        self.video_ids = np.asarray(self.video_ids, dtype=str).reshape(-1)
        n = len(self.pred)
        if not n or self.probs.shape != (n, 3):
            raise ValueError("nonempty probabilities must have shape N,3")
        if not np.isfinite(self.probs).all() or not np.isfinite(self.pred).all():
            raise ValueError("predictions must be finite")
        if (self.probs < -1e-07).any() or (self.probs > 1 + 1e-07).any():
            raise ValueError("probabilities outside [0,1]")
        if not np.allclose(self.probs.sum(1), 1, atol=1e-06, rtol=0):
            raise ValueError("probability rows must sum to one")
        if len(self.ids) != n or len(self.video_ids) != n:
            raise ValueError("ID lengths disagree")
        if (
            len(set(self.ids.tolist())) != n
            or np.any(self.ids == "")
            or np.any(self.video_ids == "")
        ):
            raise ValueError("sample IDs must be unique; all identifiers nonempty")
        if (self.labels is None) != (self.targets is None):
            raise ValueError("labels and targets must both be supplied or both absent")
        if self.labels is not None:
            lab = np.asarray(self.labels).reshape(-1)
            self.targets = np.asarray(self.targets, dtype=np.float64).reshape(-1)
            if (
                len(lab) != n
                or len(self.targets) != n
                or (not np.isin(lab, [0, 1, 2]).all())
            ):
                raise ValueError("labels must be N integers in {0,1,2}")
            self.labels = lab.astype(np.int64)
            if not np.isfinite(self.targets).all():
                raise ValueError("targets must be finite")
            if (np.abs(self.targets) > 3 + 1e-06).any():
                raise ValueError("official intensity target outside [-3,3]")
            expected = np.where(self.targets < 0, 0, np.where(self.targets > 0, 2, 1))
            if not np.array_equal(self.labels, expected):
                raise ValueError(
                    "labels disagree with exact-zero neutral target mapping"
                )
        if self.representation is not None:
            self.representation = np.asarray(self.representation)
            if (
                len(self.representation) != n
                or not np.isfinite(self.representation).all()
            ):
                raise ValueError("invalid representation")

    def subset(self, indices):
        k = np.asarray(indices)
        return PredictionSet(
            self.probs[k],
            self.pred[k],
            self.ids[k],
            self.video_ids[k],
            None if self.labels is None else self.labels[k],
            None if self.targets is None else self.targets[k],
            None if self.representation is None else self.representation[k],
        )


def _divide(a, b):
    return np.divide(
        a,
        b,
        out=np.zeros(np.broadcast_shapes(np.shape(a), np.shape(b)), dtype=float),
        where=np.asarray(b) != 0,
    )


def _from_sufficient(stats):
    """Last dimension: confusion(9), abs/sq error, centered x/y/xx/yy/xy sums."""
    s = np.asarray(stats, dtype=np.float64)
    cm = s[..., :9].reshape(s.shape[:-1] + (3, 3))
    support, predicted = (cm.sum(-1), cm.sum(-2))
    tp = np.diagonal(cm, axis1=-2, axis2=-1)
    n = support.sum(-1)
    f1 = _divide(2 * tp, support + predicted)
    covariance = s[..., 15] - _divide(s[..., 11] * s[..., 12], n)
    vx = np.maximum(0, s[..., 13] - _divide(s[..., 11] ** 2, n))
    vy = np.maximum(0, s[..., 14] - _divide(s[..., 12] ** 2, n))
    tx = 64 * np.finfo(float).eps * np.maximum(np.abs(s[..., 13]), np.finfo(float).tiny)
    ty = 64 * np.finfo(float).eps * np.maximum(np.abs(s[..., 14]), np.finfo(float).tiny)
    valid = (n >= 2) & (vx > tx) & (vy > ty)
    pearson = np.full(np.shape(n), np.nan)
    np.divide(covariance, np.sqrt(vx * vy), out=pearson, where=valid)
    pearson = np.clip(pearson, -1, 1)
    return {
        "accuracy": _divide(tp.sum(-1), n),
        "macro_f1": f1.mean(-1),
        "weighted_f1": _divide((f1 * support).sum(-1), n),
        "mae": _divide(s[..., 9], n),
        "rmse": np.sqrt(_divide(s[..., 10], n)),
        "pearson": pearson,
    }


def _sample_sufficient(p):
    if p.labels is None:
        raise ValueError("metrics require labels and targets")
    n = len(p.pred)
    stats = np.zeros((n, 16), dtype=np.float64)
    stats[np.arange(n), 3 * p.labels + np.argmax(p.probs, axis=1)] = 1
    error = p.pred - p.targets
    x, y = (p.targets - p.targets.mean(), p.pred - p.pred.mean())
    stats[:, 9:] = np.stack((np.abs(error), error**2, x, y, x**2, y**2, x * y), axis=1)
    return stats


def compute_metrics(probs, pred, labels, targets, sample_weight=None):
    """Fixed-label macro/weighted F1, accuracy, MAE, Pearson and extra RMSE.

    Absent-class F1 is zero. Undefined Pearson is None with an explicit reason.
    No clipping or threshold fitting is performed here.
    """
    n = len(np.asarray(pred).reshape(-1))
    p = PredictionSet(
        probs, pred, np.arange(n).astype(str), np.arange(n).astype(str), labels, targets
    )
    w = np.ones(n) if sample_weight is None else np.asarray(sample_weight, dtype=float)
    if w.shape != (n,) or not np.isfinite(w).all() or (w < 0).any() or (w.sum() <= 0):
        raise ValueError(
            "sample weights must be finite, nonnegative and have positive sum"
        )
    stats = (w[:, None] * _sample_sufficient(p)).sum(0)
    values = _from_sufficient(stats)
    result = {
        key: None if not np.isfinite(value) else float(value)
        for key, value in values.items()
    }
    cm = stats[:9].reshape(3, 3)
    support, predicted = (cm.sum(1), cm.sum(0))
    tp = np.diag(cm)
    result.update(
        n=n,
        weighted_n=float(w.sum()),
        confusion_matrix=cm.tolist(),
        class_order=list(CLASS_NAMES),
        zero_division=0,
    )
    result["per_class"] = [
        {
            "label": i,
            "name": name,
            "support": float(support[i]),
            "precision": float(_divide(tp[i], predicted[i])),
            "recall": float(_divide(tp[i], support[i])),
            "f1": float(_divide(2 * tp[i], support[i] + predicted[i])),
        }
        for i, name in enumerate(CLASS_NAMES)
    ]
    result["pearson_reason"] = (
        None
        if result["pearson"] is not None
        else "fewer_than_two_weighted_observations_or_constant_target_or_prediction"
    )
    return result


def prediction_metrics(predictions):
    return compute_metrics(
        predictions.probs, predictions.pred, predictions.labels, predictions.targets
    )


def condition_loss(metrics):
    return 0.5 * (1 - float(metrics["macro_f1"])) + 0.5 * float(metrics["mae"]) / 3


def aggregate_conditions(records):
    """Input records: condition_id, mask_seed, is_clean, metrics[, training_seed]."""
    if not records:
        raise ValueError("no evaluation records")
    seeds = {str(r.get("training_seed", "unspecified")) for r in records}
    if len(seeds) != 1:
        raise ValueError("training seeds must be summarized separately")
    grouped, seen = ({}, set())
    for record in records:
        key = (str(record["condition_id"]), str(record["mask_seed"]))
        if key in seen:
            raise ValueError(f"duplicate condition/mask repeat: {key}")
        seen.add(key)
        grouped.setdefault(key[0], []).append(record)
    output = []
    for name, rows in sorted(grouped.items()):
        if len({bool(r["is_clean"]) for r in rows}) != 1:
            raise ValueError("condition cannot mix clean and missing views")
        if rows[0]["is_clean"] and len(rows) != 1:
            raise ValueError(
                "clean is evaluated once, not replicated as extra observations"
            )
        means, stds, valid_counts = ({}, {}, {})
        for metric in SCALAR_METRICS:
            values = [
                float(r["metrics"][metric])
                for r in rows
                if r["metrics"][metric] is not None
            ]
            valid_counts[metric] = len(values)
            means[metric] = float(np.mean(values)) if len(values) == len(rows) else None
            stds[metric] = (
                float(np.std(values, ddof=1))
                if len(values) == len(rows) and len(values) > 1
                else None
            )
        per_class = None
        if all(("per_class" in r["metrics"] for r in rows)):
            per_class = [
                {
                    "label": i,
                    "name": label,
                    **{
                        field: float(
                            np.mean([r["metrics"]["per_class"][i][field] for r in rows])
                        )
                        for field in ("support", "precision", "recall", "f1")
                    },
                }
                for i, label in enumerate(CLASS_NAMES)
            ]
        output.append(
            {
                "condition_id": name,
                "is_clean": bool(rows[0]["is_clean"]),
                "mask_seeds": [r["mask_seed"] for r in rows],
                "repeat_count": len(rows),
                "metrics": means,
                "mask_repeat_std": stds,
                "valid_repeat_counts": valid_counts,
                "per_class_repeat_mean": per_class,
                "loss": condition_loss(means),
            }
        )
    return output


def risk_score(conditions, expected_missing=33, tail_count=7):
    clean = [c for c in conditions if c["is_clean"]]
    missing = [c for c in conditions if not c["is_clean"]]
    if (
        len(clean) != 1
        or len(missing) != expected_missing
        or (not 1 <= tail_count <= len(missing))
    ):
        raise ValueError(
            f"risk requires one clean and {expected_missing} missing conditions"
        )
    if len({c["condition_id"] for c in conditions}) != len(conditions):
        raise ValueError("condition IDs must be unique")
    if all(("mask_seeds" in c for c in missing)):
        panels = {tuple(sorted(map(str, c["mask_seeds"]))) for c in missing}
        if len(panels) != 1:
            raise ValueError(
                "all risk conditions must use the identical mask-repeat seed panel"
            )
    losses = np.array([condition_loss(c["metrics"]) for c in missing])
    tail = np.argsort(-losses, kind="stable")[:tail_count]
    original = condition_loss(clean[0]["metrics"])
    return {
        "R": float(0.2 * original + 0.6 * losses.mean() + 0.2 * losses[tail].mean()),
        "clean_loss": original,
        "mean_missing_loss": float(losses.mean()),
        "worst_tail_loss": float(losses[tail].mean()),
        "worst_condition_ids": [missing[i]["condition_id"] for i in tail],
        "weights": [0.2, 0.6, 0.2],
        "missing_condition_count": len(missing),
        "tail_count": tail_count,
        "meaning": "engineering selection criterion; not an official competition score",
    }


@dataclass
class EvalView:
    condition_id: str
    mask_seed: int | str
    is_clean: bool
    predictions: PredictionSet
    training_seed: int | str = "unspecified"


def summarize_views(views, include_risk=False):
    records = [
        {
            "condition_id": v.condition_id,
            "mask_seed": v.mask_seed,
            "is_clean": v.is_clean,
            "training_seed": v.training_seed,
            "metrics": prediction_metrics(v.predictions),
        }
        for v in views
    ]
    conditions = aggregate_conditions(records)
    return {
        "views": records,
        "conditions": conditions,
        "risk": risk_score(conditions) if include_risk else None,
    }


def training_seed_summary(seed_values):
    """Independent training-seed variability, never folded into bootstrap intervals."""
    if not seed_values:
        raise ValueError("no training-seed results")
    values = np.asarray(list(seed_values.values()), dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("training-seed summaries require finite values")
    return {
        "seeds": list(map(str, seed_values)),
        "values": values.tolist(),
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)) if len(values) > 1 else None,
        "meaning": "between-training-seed variability; separate from conditional video bootstrap",
    }


def _interval(values, point):
    finite = np.asarray(values)[np.isfinite(values)]
    return {
        "point": None if not np.isfinite(point) else float(point),
        "lower": float(np.percentile(finite, 2.5)) if len(finite) else None,
        "upper": float(np.percentile(finite, 97.5)) if len(finite) else None,
        "valid_replicates": int(len(finite)),
    }


def paired_video_bootstrap(
    left_views, right_views, n_boot=2000, seed=20260924, include_risk=False
):
    """Paired percentile CIs for LEFT minus RIGHT, resampling entire video groups.

    The identical video multiplicities are used for both models, every condition,
    and every fixed mask repeat. Mask metrics are averaged within condition in each
    resample. The mask panel is held fixed; this does not estimate unseen-mask or
    training randomness. All views must contain the same sample set; callers of
    equal-rate panels must explicitly select a shared feasible subset first.
    """
    if n_boot < 2:
        raise ValueError("at least two bootstrap replicates required")

    def index(views):
        if len({str(v.training_seed) for v in views}) != 1:
            raise ValueError("one fitted training seed per side required")
        result = {(v.condition_id, str(v.mask_seed)): v for v in views}
        if len(result) != len(views) or not views:
            raise ValueError("empty or duplicate views")
        return result

    li, ri = (index(left_views), index(right_views))
    if li.keys() != ri.keys():
        raise ValueError("paired models must have identical condition/mask keys")
    keys = sorted(li)
    reference = li[keys[0]].predictions
    ids = sorted(reference.ids.tolist())
    ref_lookup = {s: i for i, s in enumerate(reference.ids)}
    reference = reference.subset([ref_lookup[s] for s in ids])
    if reference.labels is None:
        raise ValueError("bootstrap requires ground truth")
    groups, group_index = np.unique(reference.video_ids, return_inverse=True)
    if len(groups) < 2:
        raise ValueError("at least two video clusters required")
    side_arrays = []
    normalized_sides = []
    for collection in (li, ri):
        values = np.zeros((len(groups), len(keys), 16))
        normalized = []
        for j, key in enumerate(keys):
            view = collection[key]
            p = view.predictions
            if set(p.ids.tolist()) != set(ids):
                raise ValueError("no silent sample intersection is allowed")
            lookup = {s: i for i, s in enumerate(p.ids)}
            p = p.subset([lookup[s] for s in ids])
            if (
                not np.array_equal(p.video_ids, reference.video_ids)
                or not np.array_equal(p.labels, reference.labels)
                or (not np.array_equal(p.targets, reference.targets))
            ):
                raise ValueError(
                    "paired IDs have different video grouping or ground truth"
                )
            if view.is_clean != li[key].is_clean:
                raise ValueError("clean/missing flags differ")
            np.add.at(values[:, j, :], group_index, _sample_sufficient(p))
            normalized.append(
                EvalView(
                    view.condition_id,
                    view.mask_seed,
                    view.is_clean,
                    p,
                    view.training_seed,
                )
            )
        side_arrays.append(values)
        normalized_sides.append(normalized)
    summaries = [summarize_views(v, include_risk) for v in normalized_sides]
    condition_names = [c["condition_id"] for c in summaries[0]["conditions"]]
    locations = [
        np.array([i for i, key in enumerate(keys) if key[0] == name])
        for name in condition_names
    ]
    clean_flags = np.array([c["is_clean"] for c in summaries[0]["conditions"]])
    rng = np.random.Generator(np.random.PCG64(seed))
    deltas = {m: np.empty((n_boot, len(locations))) for m in SCALAR_METRICS}
    risks = np.empty(n_boot) if include_risk else None
    for start in range(0, n_boot, 128):
        count = min(128, n_boot - start)
        weights = rng.multinomial(
            len(groups), np.full(len(groups), 1 / len(groups)), size=count
        )
        aggregated = []
        for array in side_arrays:
            sufficient = (weights @ array.reshape(len(groups), -1)).reshape(
                count, len(keys), 16
            )
            measures = _from_sufficient(sufficient)
            aggregated.append(
                {
                    m: np.stack(
                        [measures[m][:, loc].mean(1) for loc in locations], axis=1
                    )
                    for m in SCALAR_METRICS
                }
            )
        for metric in SCALAR_METRICS:
            deltas[metric][start : start + count] = (
                aggregated[0][metric] - aggregated[1][metric]
            )
        if include_risk:
            side_risks = []
            for metrics in aggregated:
                loss = 0.5 * (1 - metrics["macro_f1"]) + 0.5 * metrics["mae"] / 3
                missing = loss[:, ~clean_flags]
                side_risks.append(
                    0.2 * loss[:, clean_flags][:, 0]
                    + 0.6 * missing.mean(1)
                    + 0.2 * np.sort(missing, axis=1)[:, -7:].mean(1)
                )
            risks[start : start + count] = side_risks[0] - side_risks[1]
    output = []
    for j, name in enumerate(condition_names):
        comparison = {}
        for metric in SCALAR_METRICS:
            l = summaries[0]["conditions"][j]["metrics"][metric]
            r = summaries[1]["conditions"][j]["metrics"][metric]
            interval = _interval(
                deltas[metric][:, j], np.nan if l is None or r is None else l - r
            )
            interval.update(left=l, right=r)
            comparison[metric] = interval
        output.append(
            {
                "condition_id": name,
                "is_clean": bool(clean_flags[j]),
                "left_minus_right": comparison,
            }
        )
    return {
        "method": "paired_video_cluster_percentile",
        "replicates": n_boot,
        "seed": seed,
        "video_clusters": len(groups),
        "samples": len(ids),
        "conditions": output,
        "risk_difference": (
            _interval(risks, summaries[0]["risk"]["R"] - summaries[1]["risk"]["R"])
            if include_risk
            else None
        ),
        "uncertainty_scope": "fixed fitted models and fixed mask-repeat panel; video sampling only; training-seed variability must be reported separately",
    }
