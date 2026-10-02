"""Video-cluster bootstrap intervals for prediction and explanation validation."""

from collections import defaultdict
import hashlib
import json
import numpy as np
from sentiment.metrics import (
    PredictionSet,
    compute_metrics,
    _sample_sufficient,
    _from_sufficient,
)

METRICS = ("accuracy", "macro_f1", "mae", "pearson")
GROUP_FIELDS = ("split", "method", "kind", "panel", "budget", "operation", "metric")


class ClusterBootstrap:
    """Percentile intervals, resampling whole videos with equal cluster probability.

    Point estimates and each replicate are sample-weighted. A repeated video
    contributes all its clips together. Fixed-mask random draws are averaged
    within the sample before bootstrapping, never counted as additional clips.
    """

    def __init__(self, repeats=2000, seed=20260927):
        if repeats != 2000:
            raise ValueError("Registered analysis requires exactly 2000 replicates")
        self.repeats, self.seed, self.cache = (repeats, int(seed), {})

    def weights(self, videos):
        groups, inverse = np.unique(np.asarray(videos, str), return_inverse=True)
        key = tuple(groups.tolist())
        if key not in self.cache:
            digest = hashlib.sha256(
                json.dumps([self.seed, key], ensure_ascii=False).encode()
            ).digest()
            rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
            self.cache[key] = rng.multinomial(
                len(groups), np.full(len(groups), 1 / len(groups)), size=self.repeats
            ).astype(np.float64)
        return (groups, inverse, self.cache[key])

    @staticmethod
    def interval(point, draws, n, clusters):
        valid = np.asarray(draws)[np.isfinite(draws)]
        enough = clusters >= 2 and len(valid) >= 1000
        return {
            "point": float(point) if point is not None and np.isfinite(point) else None,
            "lower": float(np.quantile(valid, 0.025)) if enough else None,
            "upper": float(np.quantile(valid, 0.975)) if enough else None,
            "n_samples": int(n),
            "n_video_clusters": int(clusters),
            "bootstrap_replicates": 2000,
            "valid_replicates": int(len(valid)),
            "confidence_level": 0.95,
            "status": "ok" if enough else "insufficient_clusters_or_valid_replicates",
        }

    def mean(self, values, videos):
        values = np.asarray(values, float)
        videos = np.asarray(videos, str)
        valid = np.isfinite(values)
        values, videos = (values[valid], videos[valid])
        if not len(values):
            return self.interval(None, [], 0, 0)
        groups, inverse, weights = self.weights(videos)
        sums, counts = (
            np.zeros(len(groups)),
            np.bincount(inverse, minlength=len(groups)),
        )
        np.add.at(sums, inverse, values)
        draws = weights @ sums / (weights @ counts)
        return self.interval(values.mean(), draws, len(values), len(groups))

    def predictions(self, prediction):
        groups, inverse, weights = self.weights(prediction.video_ids)
        sufficient = np.zeros((len(groups), 16))
        np.add.at(sufficient, inverse, _sample_sufficient(prediction))
        replicates = _from_sufficient(weights @ sufficient)
        point = compute_metrics(
            prediction.probs, prediction.pred, prediction.labels, prediction.targets
        )
        return {
            "metrics": point,
            "intervals": {
                m: self.interval(
                    point[m], replicates[m], len(prediction.ids), len(groups)
                )
                for m in METRICS
            },
        }

    def paired_predictions(self, left, right):
        if set(left.ids) != set(right.ids):
            raise ValueError(
                "Paired prediction sets require exactly the same sample IDs"
            )
        ids = sorted(left.ids)
        sides = []
        for p in (left, right):
            lookup = {sid: i for i, sid in enumerate(p.ids)}
            sides.append(p.subset([lookup[sid] for sid in ids]))
        left, right = sides
        if any(
            (
                not np.array_equal(getattr(left, key), getattr(right, key))
                for key in ("video_ids", "labels", "targets")
            )
        ):
            raise ValueError(
                "Paired prediction IDs have inconsistent videos or ground truth"
            )
        groups, inverse, weights = self.weights(left.video_ids)
        estimates, draws = ([], [])
        for p in sides:
            stats = np.zeros((len(groups), 16))
            np.add.at(stats, inverse, _sample_sufficient(p))
            estimates.append(_from_sufficient(stats.sum(0)))
            draws.append(_from_sufficient(weights @ stats))
        return {
            m: self.interval(
                estimates[0][m] - estimates[1][m],
                draws[0][m] - draws[1][m],
                len(ids),
                len(groups),
            )
            for m in METRICS
        }


def intended_direction(kind, operation, metric):
    if metric in ("comprehensiveness", "sufficiency") and kind.startswith("class_"):
        direction = 1 if operation == "delete" else -1
        return direction if kind == "class_support" else -direction
    if metric == "intensity_absolute_change" and kind == "intensity_absolute":
        return 1 if operation == "delete" else -1
    return None


def flatten_faithfulness(rows):
    flat, unavailable, skipped = ([], [], [])
    unique = set()
    for row in rows:
        identity = (row["split"], row["sample_id"], row["method"])
        if identity in unique:
            raise ValueError(f"Duplicate faithfulness row: {identity}")
        unique.add(identity)
        result = row.get("result", {})
        if (
            row.get("status") == "not_applicable"
            or result.get("status") == "not_applicable"
        ):
            unavailable.append(
                {
                    "split": row["split"],
                    "sample_id": row["sample_id"],
                    "method": row["method"],
                    "reason": result.get(
                        "reason", row.get("method_metadata", {}).get("reason")
                    ),
                }
            )
            continue
        skipped.extend(
            (
                {
                    "split": row["split"],
                    "sample_id": row["sample_id"],
                    "method": row["method"],
                    **entry,
                }
                for entry in result.get("skipped", [])
            )
        )
        for record in result.get("records", []):
            for operation, output in record["operations"].items():
                for metric, value in output["metrics"].items():
                    if metric == "class_probability_delta":
                        continue
                    random = output.get("random_summary", {})
                    flat.append(
                        {
                            "split": row["split"],
                            "sample_id": row["sample_id"],
                            "video_id": str(row["video_id"]),
                            "method": row["method"],
                            "kind": record["kind"],
                            "panel": record["panel"],
                            "budget": float(record["budget_fraction"]),
                            "operation": operation,
                            "metric": metric,
                            "value": float(value),
                            "random_mean": random.get("mean", {}).get(metric),
                            "random_count": random.get("n", 0),
                            "actual_atoms": record["evidence"]["actual_atom_count"],
                            "allowed_atoms": record["evidence"]["allowed_atom_budget"],
                            "random_requested": record.get("random_matching", {}).get(
                                "requested", 20
                            ),
                        }
                    )
    by_sample = defaultdict(list)
    for row in flat:
        key = tuple(
            (
                row[k]
                for k in (
                    "split",
                    "sample_id",
                    "method",
                    "kind",
                    "panel",
                    "operation",
                    "metric",
                )
            )
        )
        by_sample[key].append(row)
    for members in by_sample.values():
        if sorted((r["budget"] for r in members)) == [0.1, 0.2, 0.3]:
            row = dict(
                members[0],
                budget="AOPC",
                value=float(np.mean([r["value"] for r in members])),
                random_mean=(
                    float(np.mean([r["random_mean"] for r in members]))
                    if all((r["random_mean"] is not None for r in members))
                    else None
                ),
                random_count=min((r["random_count"] for r in members)),
                random_requested=max((r["random_requested"] for r in members)),
                random_count_semantics="minimum_across_three_budgets_for_completeness_check",
                actual_atoms=float(np.mean([r["actual_atoms"] for r in members])),
                allowed_atoms=float(np.mean([r["allowed_atoms"] for r in members])),
            )
            flat.append(row)
    return (flat, unavailable, skipped)


def summarize_faithfulness(flat, bootstrap):
    groups, lookup = (defaultdict(list), {})
    for row in flat:
        key = tuple((row[k] for k in GROUP_FIELDS))
        groups[key].append(row)
        lookup[row["sample_id"], *key] = row
    summary = []
    for key, rows in groups.items():
        record = dict(zip(GROUP_FIELDS, key))
        direction = intended_direction(
            record["kind"], record["operation"], record["metric"]
        )
        videos, values = ([r["video_id"] for r in rows], [r["value"] for r in rows])
        record.update(
            value=bootstrap.mean(values, videos),
            intended_direction=direction,
            actual_atoms_mean=float(np.mean([r["actual_atoms"] for r in rows])),
            allowed_atoms_mean=float(np.mean([r["allowed_atoms"] for r in rows])),
            random_shortage_samples=sum(
                (r["random_count"] < r["random_requested"] for r in rows)
            ),
        )
        paired = [r for r in rows if r["random_mean"] is not None]
        differences = [r["value"] - r["random_mean"] for r in paired]
        record["matched_random"] = bootstrap.mean(
            [r["random_mean"] for r in paired], [r["video_id"] for r in paired]
        )
        record["paired_raw_minus_random"] = bootstrap.mean(
            differences, [r["video_id"] for r in paired]
        )
        record["paired_gain_over_random"] = (
            bootstrap.mean(
                np.asarray(differences) * direction, [r["video_id"] for r in paired]
            )
            if direction is not None
            else None
        )
        reference_key = list(key)
        reference_key[1] = "owen"
        pairs = [(r, lookup.get((r["sample_id"], *reference_key))) for r in rows]
        pairs = [(r, ref) for r, ref in pairs if ref is not None]
        if any((r["video_id"] != ref["video_id"] for r, ref in pairs)):
            raise ValueError("Paired method samples disagree on video ID")
        delta = np.array([r["value"] - ref["value"] for r, ref in pairs])
        record["paired_raw_minus_owen"] = bootstrap.mean(
            delta, [r["video_id"] for r, _ in pairs]
        )
        record["paired_gain_over_owen"] = (
            bootstrap.mean(delta * direction, [r["video_id"] for r, _ in pairs])
            if direction is not None
            else None
        )
        record["owen_pairing_missing_samples"] = len(rows) - len(pairs)
        summary.append(record)
    return summary


def scalar_summary(values):
    values = np.asarray([v for v in values if v is not None], float)
    values = values[np.isfinite(values)]
    return {
        "n": len(values),
        "mean": float(values.mean()) if len(values) else None,
        "median": float(np.median(values)) if len(values) else None,
        "min": float(values.min()) if len(values) else None,
        "max": float(values.max()) if len(values) else None,
    }
