"""Inference-only evaluation adapter and auditable per-sample prediction files.

No masking, normalization, cached clean representation, or threshold fitting is
performed here. Data code owns the construction of damaged student inputs.
"""

from __future__ import annotations
import csv
import json
from contextlib import nullcontext
from pathlib import Path
import numpy as np
from .metrics import CLASS_NAMES, PredictionSet, EvalView, summarize_views

MODEL_KEYS = ("text", "audio", "vision", "mask", "structure")


def _numpy(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu()
        if str(value.dtype) == "torch.bfloat16":
            value = value.float()
        value = value.numpy()
    return np.asarray(value)


def _to_device(value, device, torch):
    if isinstance(value, dict):
        return {key: _to_device(v, device, torch) for key, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)((_to_device(v, device, torch) for v in value))
    if isinstance(value, np.ndarray):
        value = torch.from_numpy(value)
    return value.to(device) if hasattr(value, "to") else value


def evaluate_model(model, batch_iter, device="cpu", forward_fn=None):
    """Return PredictionSet from model({text,audio,vision,mask,structure}).

    Batches also contain ``ids`` and ``video_ids``, and optionally ``labels`` and
    ``targets``; these are NEVER passed to the model. The model returns a mapping
    with ``logits`` (N,3), ``regression`` (N or N,1) and optional representation.
    ``forward_fn(model, inputs)`` adapts other forward signatures. With device=None
    and a forward_fn, this supports NumPy test doubles without importing torch.
    Model training/eval state is restored even when inference fails.
    """
    torch = None
    if device is not None or forward_fn is None:
        import torch
    context = torch.inference_mode() if torch is not None else nullcontext()
    was_training = getattr(model, "training", None)
    if hasattr(model, "eval"):
        model.eval()
    arrays = {
        key: []
        for key in (
            "probs",
            "pred",
            "ids",
            "video_ids",
            "labels",
            "targets",
            "representation",
        )
    }
    labeled = has_representation = None
    try:
        with context:
            for batch in batch_iter:
                absent = [
                    key for key in MODEL_KEYS + ("ids", "video_ids") if key not in batch
                ]
                if absent:
                    raise ValueError(f"batch missing required fields {absent}")
                inputs = {key: batch[key] for key in MODEL_KEYS}
                if torch is not None and device is not None:
                    inputs = _to_device(inputs, device, torch)
                result = forward_fn(model, inputs) if forward_fn else model(inputs)
                if (
                    not isinstance(result, dict)
                    or not {"logits", "regression"} <= result.keys()
                ):
                    raise ValueError("model must return logits and regression")
                logits = _numpy(result["logits"]).astype(np.float64)
                if (
                    logits.ndim != 2
                    or logits.shape[1] != 3
                    or (not np.isfinite(logits).all())
                ):
                    raise ValueError("model logits must be finite N,3")
                exp = np.exp(logits - logits.max(axis=1, keepdims=True))
                arrays["probs"].append(exp / exp.sum(axis=1, keepdims=True))
                arrays["pred"].append(_numpy(result["regression"]).reshape(-1))
                for name in ("ids", "video_ids"):
                    arrays[name].append(_numpy(batch[name]).astype(str).reshape(-1))
                this_labeled = "labels" in batch and "targets" in batch
                if ("labels" in batch) != ("targets" in batch):
                    raise ValueError("labels and targets must occur together")
                if labeled is not None and labeled != this_labeled:
                    raise ValueError("cannot mix labeled and unlabeled batches")
                labeled = this_labeled
                if labeled:
                    for name in ("labels", "targets"):
                        arrays[name].append(_numpy(batch[name]).reshape(-1))
                this_representation = result.get("representation") is not None
                if (
                    has_representation is not None
                    and has_representation != this_representation
                ):
                    raise ValueError(
                        "representation must be present in every batch or none"
                    )
                has_representation = this_representation
                if has_representation:
                    arrays["representation"].append(_numpy(result["representation"]))
                n = len(logits)
                for name in arrays:
                    if arrays[name] and len(arrays[name][-1]) != n:
                        raise ValueError(f"batch {name} length differs from logits")
    finally:
        if was_training is not None and hasattr(model, "train"):
            model.train(was_training)
    if not arrays["pred"]:
        raise ValueError("evaluation iterator produced no batches")
    joined = {
        key: np.concatenate(values, axis=0) if values else None
        for key, values in arrays.items()
    }
    return PredictionSet(**joined)


def save_predictions(
    predictions,
    output_prefix,
    condition_id="original",
    mask_seed="none",
    training_seed="unspecified",
):
    """Save detailed CSV plus no-pickle NPZ; no labels invented for inference data."""
    prefix = Path(output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path, npz_path = (Path(str(prefix) + ".csv"), Path(str(prefix) + ".npz"))
    p = predictions
    columns = [
        "sample_id",
        "video_id",
        "condition_id",
        "mask_seed",
        "training_seed",
        "prob_negative",
        "prob_neutral",
        "prob_positive",
        "predicted_class",
        "polarity",
        "intensity",
    ]
    if p.labels is not None:
        columns += ["label", "target"]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for i in range(len(p.pred)):
            label = int(np.argmax(p.probs[i]))
            row = dict(
                zip(
                    columns[:5],
                    [p.ids[i], p.video_ids[i], condition_id, mask_seed, training_seed],
                )
            )
            row.update(
                prob_negative=float(p.probs[i, 0]),
                prob_neutral=float(p.probs[i, 1]),
                prob_positive=float(p.probs[i, 2]),
                predicted_class=label,
                polarity=CLASS_NAMES[label],
                intensity=float(p.pred[i]),
            )
            if p.labels is not None:
                row.update(label=int(p.labels[i]), target=float(p.targets[i]))
            writer.writerow(row)
    values = {
        name: getattr(p, name)
        for name in (
            "probs",
            "pred",
            "ids",
            "video_ids",
            "labels",
            "targets",
            "representation",
        )
        if getattr(p, name) is not None
    }
    values.update(
        condition_id=np.asarray(str(condition_id)),
        mask_seed=np.asarray(str(mask_seed)),
        training_seed=np.asarray(str(training_seed)),
    )
    np.savez_compressed(npz_path, **values)
    return {"csv": str(csv_path), "npz": str(npz_path), "samples": len(p.pred)}


def load_predictions(path):
    with np.load(path, allow_pickle=False) as data:
        return PredictionSet(
            **{
                name: data[name]
                for name in (
                    "probs",
                    "pred",
                    "ids",
                    "video_ids",
                    "labels",
                    "targets",
                    "representation",
                )
                if name in data
            }
        )


def save_official_predictions(predictions, source_files, output_csv):
    """Project output contract; official problem does not mandate these column names."""
    p = predictions
    sources = np.asarray(source_files, dtype=str).reshape(-1)
    if len(sources) != len(p.pred) or np.any(sources == ""):
        raise ValueError("one nonempty source filename per sample is required")
    if (np.abs(p.pred) > 3 + 1e-06).any():
        raise ValueError(
            "submission intensity must be in [-3,3]; repair the model, do not silently clip"
        )
    destination = Path(output_csv)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["sample_id", "source_file", "polarity", "intensity"])
        for i in range(len(p.pred)):
            writer.writerow(
                [
                    p.ids[i],
                    sources[i],
                    CLASS_NAMES[int(np.argmax(p.probs[i]))],
                    float(p.pred[i]),
                ]
            )
    return destination


def evaluate_conditions(
    model,
    specifications,
    batch_factory,
    device="cpu",
    forward_fn=None,
    training_seed="unspecified",
    include_risk=False,
    output_dir=None,
):
    """Evaluate fixed views without knowing or changing the masking protocol.

    Each spec supplies condition_id,mask_seed,is_clean. batch_factory(spec) returns
    its iterator. Supports core 34 views (early stop), core 100 views (3 repeats),
    or grid 397 views (133 conditions); include_risk only accepts the core panel.
    """
    views, paths = ([], [])
    for index, spec in enumerate(specifications):
        prediction = evaluate_model(model, batch_factory(spec), device, forward_fn)
        view = EvalView(
            str(spec["condition_id"]),
            spec["mask_seed"],
            bool(spec["is_clean"]),
            prediction,
            training_seed,
        )
        views.append(view)
        if output_dir is not None:
            paths.append(
                save_predictions(
                    prediction,
                    Path(output_dir) / f"view_{index:04d}",
                    view.condition_id,
                    view.mask_seed,
                    training_seed,
                )
            )
    summary = summarize_views(views, include_risk)
    if output_dir is not None:
        destination = Path(output_dir) / "evaluation_summary.json"
        destination.write_text(
            json.dumps(
                {**summary, "prediction_files": paths},
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            ),
            encoding="utf-8",
        )
    return (views, summary)
