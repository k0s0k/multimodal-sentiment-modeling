"""Train complete-input Q3 models and fold-local fixed-recipe replicas.

No official test/special data enters this engine. OOF holdout metrics are
computed once after the fixed training schedule and never select an epoch.
"""

from __future__ import annotations
import argparse
import contextlib
import copy
import hashlib
import json
import math
import time
from pathlib import Path
import numpy as np
import torch
from sentiment.common import append_jsonl, now, save_json, seed_everything, sha256
from sentiment.data import apply_scaler, fit_scaler
from sentiment.losses import (
    compute_loss as vendor_loss,
    inverse_sqrt_class_weights,
    reconstruction_loss,
    simsiam_loss,
    supervised_loss,
)
from sentiment.metrics import compute_metrics, condition_loss
from .models import OnlineBertFusion, make_model, normalize_model_config

MODEL_KEYS = ("text", "audio", "vision", "mask", "structure")


def set_numeric_profile():
    """Same public FP32 evaluation profile for every candidate and coalition."""
    torch.set_float32_matmul_precision("highest")
    if hasattr(torch.backends, "cuda"):
        torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False


def _amp(device, enabled):
    return (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if enabled and str(device).startswith("cuda")
        else contextlib.nullcontext()
    )


def _take_output(output, indices):

    def take(value):
        if isinstance(value, dict):
            return {k: take(v) for k, v in value.items()}
        return value[indices] if torch.is_tensor(value) and value.ndim else value

    return {k: take(v) for k, v in output.items()}


def compute_training_loss(
    full_output,
    labels,
    targets,
    config,
    *,
    damaged_output=None,
    full_batch=None,
    damaged_batch=None,
    aux_valid=None,
    class_weights=None,
):
    """Keep Lfull, Ldamaged and both restoration coefficients independent.

    Failed augmentation views contribute zero auxiliary loss. Multiplication
    by their valid fraction preserves that zero rather than amplifying the
    remaining examples to compensate. Original full supervision is retained.
    """
    c = dict(config)
    if c.get("kd_mode", "none") not in ("none", "off"):
        raise ValueError("Q3 engine has no hidden/full-train teacher path")
    beta = float(c.get("smooth_l1_beta", 0.5))
    reg_weight = float(c.get("lambda_regression", 1.0))
    full_loss, parts = supervised_loss(
        full_output,
        labels,
        targets,
        class_weights=class_weights,
        lambda_regression=reg_weight,
        beta=beta,
    )
    parts = {"full_" + k: v for k, v in parts.items()}
    parts["full_supervised"] = full_loss
    weights = (
        float(c.get("lambda_mask", 0)),
        float(c.get("lambda_reconstruction", 0)),
        float(c.get("lambda_hlfr", 0)),
    )
    if any((w < 0 for w in weights)):
        raise ValueError("negative loss coefficients")
    if damaged_output is None:
        if any(weights):
            raise ValueError("auxiliary loss requested without damaged forward")
        parts["total"] = full_loss
        return (full_loss, parts)
    if full_batch is None or damaged_batch is None:
        raise ValueError("damaged training requires explicit full/damaged inputs")
    if c.get("regime") == "legacy_q2":
        legacy = dict(c)
        legacy.setdefault("lambda_kd", 0)
        legacy.setdefault("kd_mode", "none")
        return vendor_loss(
            damaged_output,
            labels,
            targets,
            legacy,
            original_outputs=full_output,
            recon_targets={m: full_batch[m] for m in ("text", "audio", "vision")},
            recon_mask=full_batch["mask"] & ~damaged_batch["mask"],
            original_observed=full_batch["mask"],
            current_observed=damaged_batch["mask"],
            class_weights=class_weights,
        )
    if aux_valid is None:
        aux_valid = torch.ones(len(labels), device=labels.device, dtype=torch.bool)
    if aux_valid.dtype != torch.bool or aux_valid.shape != labels.shape:
        raise ValueError("aux_valid must be bool [B]")
    selected = torch.nonzero(aux_valid, as_tuple=False).flatten()
    fraction = aux_valid.float().mean()
    parts["aux_valid_fraction"] = fraction
    total = full_loss
    if len(selected):
        d = _take_output(damaged_output, selected)
        f = _take_output(full_output, selected)
        original_mask, current_mask = (
            full_batch["mask"][selected],
            damaged_batch["mask"][selected],
        )
        if weights[0]:
            value, _ = supervised_loss(
                d,
                labels[selected],
                targets[selected],
                class_weights=class_weights,
                lambda_regression=reg_weight,
                beta=beta,
            )
            parts["damaged_supervised"] = fraction * value
            total = total + weights[0] * fraction * value
        if weights[1]:
            value, _ = reconstruction_loss(
                d.get("recon", {}),
                {m: full_batch[m][selected] for m in ("text", "audio", "vision")},
                original_mask & ~current_mask,
                original_observed=original_mask,
                current_observed=current_mask,
                beta=float(c.get("reconstruction_beta", 1.0)),
                text_mode=c.get("text_reconstruction_mode", "smooth_l1"),
            )
            parts["reconstruction"] = fraction * value
            total = total + weights[1] * fraction * value
        if weights[2]:
            value, _ = simsiam_loss(d, f, original_observed=original_mask)
            parts["hlfr"] = fraction * value
            total = total + weights[2] * fraction * value
    parts["total"] = total
    return (total, parts)


def _ids_hash(ids):
    return hashlib.sha256("\n".join(map(str, ids)).encode("utf-8")).hexdigest()


def training_identity(raw):
    return {
        "ids_sha256": _ids_hash(raw["ids"]),
        "video_ids_sha256": _ids_hash(raw["video_ids"]),
        "n": len(raw["ids"]),
        "source_sha256": raw.get("source_sha256"),
        "official_indices": list(
            map(int, raw.get("official_indices", range(len(raw["ids"]))))
        ),
    }


def validate_initialization(state, identity, scaler):
    """Fail before any optimizer step if FT initialization saw other groups."""
    if state.get("bert_delta") is not None:
        raise ValueError("FT initialization must be a frozen-BERT fusion checkpoint")
    previous = state.get("provenance", {}).get("train_identity", {})
    for key in (
        "ids_sha256",
        "video_ids_sha256",
        "n",
        "source_sha256",
        "official_indices",
    ):
        if previous.get(key) != identity.get(key):
            raise ValueError(f"FT train identity mismatch: {key}")
    if state.get("scaler") != scaler:
        raise ValueError("FT scaler differs from its own fold-local pretraining")


def validate_split_config(config):
    split = config.get("validation_split", "valid")
    fixed = config.get("oof_fixed_epochs")
    if split not in ("valid", "train"):
        raise ValueError(
            "training validation must be valid_tune or a fixed train OOF fold"
        )
    if split == "train":
        if (
            fixed is None
            or int(fixed) < 1
            or config.get("train_indices") is None
            or (config.get("valid_indices") is None)
        ):
            raise ValueError(
                "OOF requires explicit train/holdout indices and fixed epochs"
            )
        if set(config["train_indices"]) & set(config["valid_indices"]):
            raise ValueError("OOF row overlap")
    elif config.get("subset_name", "valid_tune") != "valid_tune":
        raise ValueError("development epoch selection is limited to valid_tune")
    if fixed is not None and int(fixed) < 1:
        raise ValueError("fixed epochs must be positive")


def _batch(data, indices, device):
    result = {}
    for key in (*MODEL_KEYS, "tokens", "labels", "targets", "aux_valid"):
        if key not in data:
            continue
        value = np.asarray(data[key])[indices]
        result[key] = torch.as_tensor(np.array(value, copy=True), device=device)
    if "tokens" in result:
        result["tokens"] = result["tokens"].long()
    if "labels" in result:
        result["labels"] = result["labels"].long()
    if "targets" in result:
        result["targets"] = result["targets"].float()
    return result


def _forward(model, batch):
    keys = (
        ("tokens", "audio", "vision", "mask", "structure")
        if isinstance(model, OnlineBertFusion)
        else MODEL_KEYS
    )
    return model({k: batch[k] for k in keys})


@torch.inference_mode()
def predict(model, prepared, device, batch_size=64):
    model.eval()
    probs, pred = ([], [])
    for start in range(0, len(prepared["ids"]), batch_size):
        batch = _batch(
            prepared,
            np.arange(start, min(start + batch_size, len(prepared["ids"]))),
            device,
        )
        output = _forward(model, batch)
        probs.append(output["logits"].float().softmax(-1).cpu().numpy())
        pred.append(output["regression"].float().cpu().numpy())
    return {
        "probs": np.concatenate(probs),
        "pred": np.concatenate(pred),
        "labels": np.asarray(prepared["labels"]),
        "targets": np.asarray(prepared["targets"]),
        "ids": np.asarray(prepared["ids"]),
        "video_ids": np.asarray(prepared["video_ids"]),
        "indices": np.asarray(
            prepared.get("official_indices", np.arange(len(prepared["ids"]))),
            dtype=np.int64,
        ),
    }


def checkpoint_state(model, config, scaler, provenance, epoch, score):
    fusion = model.fusion if isinstance(model, OnlineBertFusion) else model
    state = {
        "format": "q3_predictor_v1",
        "model_state": fusion.state_dict(),
        "model_config": copy.deepcopy(fusion.q3_config),
        "scaler": scaler,
        "config": config,
        "epoch": int(epoch),
        "selection_score": score,
        "provenance": copy.deepcopy(provenance),
        "numeric_profile": "q3_fp32_v1",
    }
    if isinstance(model, OnlineBertFusion):
        state.update(
            bert_delta=model.delta_state(),
            bert_metadata=model.bert_metadata,
            bert_trainable_layers=[10, 11],
        )
    return state


def _restore(model, state):
    fusion = model.fusion if isinstance(model, OnlineBertFusion) else model
    fusion.load_state_dict(state["model_state"], strict=True)
    if isinstance(model, OnlineBertFusion):
        model.load_delta(state["bert_delta"])


def train(
    config,
    source,
    bert_path,
    output,
    device="cuda:0",
    *,
    data_loader=None,
    encoder_factory=None,
    augmenter=None,
):
    """Public API; injectable readers/encoder enable small meaningful tests."""
    from .data import load_training_data
    from .encoding import FrozenEncoder
    from .augmentation import augment_epoch

    data_loader = data_loader or load_training_data
    encoder_factory = encoder_factory or FrozenEncoder
    augmenter = augmenter or augment_epoch
    config = copy.deepcopy(config)
    validate_split_config(config)
    outdir = Path(output)
    outdir.mkdir(parents=True, exist_ok=True)
    if (outdir / "result.json").exists() or (outdir / "best.pt").exists():
        raise FileExistsError(
            "run already has a checkpoint/result; choose a new run ID"
        )
    seed = int(config.get("seed", 20260927))
    seed_everything(seed)
    set_numeric_profile()
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.init()
        torch.cuda.reset_peak_memory_stats(device)
    raw_train, raw_valid = data_loader(
        source,
        train_indices=config.get("train_indices"),
        valid_indices=config.get("valid_indices"),
        subset_name=config.get("subset_name", "valid_tune"),
        validation_split=config.get("validation_split", "valid"),
    )
    if not len(raw_train["ids"]) or not len(raw_valid["ids"]):
        raise ValueError("empty train/validation partition")
    if config.get("validation_split", "valid") == "train":
        if set(raw_train["video_ids"]) & set(raw_valid["video_ids"]):
            raise ValueError(
                "OOF video overlap: clips from one source may not cross folds"
            )
    scaler = fit_scaler(raw_train)
    identity = training_identity(raw_train)
    counts = np.bincount(raw_train["labels"], minlength=3).astype(np.float64)
    mean = float(np.mean(raw_train["targets"]))
    weights = (
        inverse_sqrt_class_weights(counts=counts).to(device)
        if config.get("class_weights", "inverse_sqrt") != "none"
        else None
    )
    fusion = make_model(config["model"]).to(device)
    fusion.set_priors(counts / counts.sum(), mean)
    finetune = bool(config.get("finetune", False))
    initialization_sha = None
    if finetune:
        if not config.get("initialize"):
            raise ValueError(
                "fine-tuning requires its own frozen pretraining checkpoint"
            )
        initial = torch.load(
            config["initialize"], map_location="cpu", weights_only=False
        )
        validate_initialization(initial, identity, scaler)
        if normalize_model_config(initial["model_config"]) != normalize_model_config(
            config["model"]
        ):
            raise ValueError("FT model recipe differs from pretraining")
        fusion.load_state_dict(initial["model_state"], strict=True)
        initialization_sha = sha256(config["initialize"])
    encoder = encoder_factory(
        bert_path, device=str(device), batch_size=int(config.get("bert_batch_size", 64))
    )
    clean = apply_scaler(raw_train, scaler)
    clean["text"] = np.asarray(encoder.encode(raw_train["tokens"]), dtype=np.float32)
    valid = apply_scaler(raw_valid, scaler)
    if not finetune:
        valid["text"] = np.asarray(
            encoder.encode(raw_valid["tokens"]), dtype=np.float32
        )
    model = (
        OnlineBertFusion(fusion, encoder.model, getattr(encoder, "metadata", {})).to(
            device
        )
        if finetune
        else fusion
    )
    loss_config = dict(config.get("loss", {}))
    aux_enabled = not config.get("original_only", False) and (
        loss_config.get("regime") == "legacy_q2"
        or any(
            (
                float(loss_config.get(k, 0)) > 0
                for k in ("lambda_mask", "lambda_reconstruction", "lambda_hlfr")
            )
        )
    )
    lr = float(config.get("learning_rate", 0.0001 if finetune else 0.0003))
    if finetune:
        groups = [
            {"params": fusion.parameters(), "lr": lr},
            {
                "params": [p for p in model.bert.parameters() if p.requires_grad],
                "lr": float(config.get("bert_learning_rate", 1e-05)),
            },
        ]
    else:
        groups = fusion.parameters()
    optimizer = torch.optim.AdamW(
        groups, lr=lr, weight_decay=float(config.get("weight_decay", 0.01))
    )
    batch_size = int(config.get("batch_size", 32 if finetune else 64))
    eval_batch = int(config.get("eval_batch_size", 64))
    schedule_epochs = int(config.get("max_epochs", 20 if finetune else 50))
    fixed = config.get("oof_fixed_epochs")
    epochs = int(fixed) if fixed is not None else schedule_epochs
    if epochs < 1 or batch_size < 1 or eval_batch < 1 or (schedule_epochs < epochs):
        raise ValueError(
            "invalid batch/epoch budget or fixed epoch beyond registered schedule"
        )
    steps_per_epoch = math.ceil(len(raw_train["ids"]) / batch_size)
    total_steps = schedule_epochs * steps_per_epoch
    warmup = max(1, int(total_steps * float(config.get("warmup_fraction", 0.05))))

    def lr_scale(step):
        if step < warmup:
            return (step + 1) / warmup
        progress = min(1.0, (step - warmup) / max(1, total_steps - warmup))
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_scale)
    provenance = {
        "started_utc": now(),
        "train_identity": identity,
        "train_target_mean": mean,
        "counts": counts.tolist(),
        "train_samples": len(raw_train["ids"]),
        "validation_samples": len(raw_valid["ids"]),
        "validation_ids_sha256": _ids_hash(raw_valid["ids"]),
        "candidate_id": config.get("candidate_id"),
        "seed": seed,
        "finetune": finetune,
        "initialization_sha256": initialization_sha,
        "encoder": getattr(encoder, "metadata", {}),
        "numeric_profile": "q3_fp32_v1",
        "training_amp": bool(config.get("training_amp", True)),
        "oof_fixed_epochs": fixed,
        "heldout_used_for_epoch_selection": fixed is None,
        "scheduler_horizon_epochs": schedule_epochs,
        "trainable_parameters": sum(
            (p.numel() for p in model.parameters() if p.requires_grad)
        ),
        "engine_sha256": sha256(__file__),
        "model_source_sha256": sha256(Path(__file__).with_name("models.py")),
        "source_sha256": raw_train.get("source_sha256"),
        "status": "running",
    }
    save_json(outdir / "config.json", config)
    save_json(outdir / "scaler.json", scaler)
    save_json(outdir / "status.json", provenance)
    start = time.perf_counter()
    best, best_epoch, stale, step = (math.inf, 0, 0, 0)
    step_times = []
    validation_calls = 0
    print(
        json.dumps(
            {
                "event": "started",
                "run": outdir.name,
                "epochs": epochs,
                "fixed_epochs": fixed,
                "parameters": provenance["trainable_parameters"],
            }
        ),
        flush=True,
    )
    try:
        for epoch in range(1, epochs + 1):
            epoch_start = time.perf_counter()
            damaged = None
            augmentation_seconds = 0.0
            augmentation_diagnostics = None
            if aux_enabled:
                tick = time.perf_counter()
                raw_damaged = augmenter(
                    raw_train,
                    seed=seed,
                    epoch=epoch,
                    mode=config.get("augmentation_mode", "q3_mixture"),
                )
                if len(raw_damaged["tokens"]) != len(raw_train["ids"]):
                    raise ValueError("augmentation changed sample axis")
                damaged = apply_scaler(raw_damaged, scaler)
                damaged["aux_valid"] = np.asarray(raw_damaged["aux_valid"], dtype=bool)
                if not finetune:
                    damaged["text"] = np.asarray(
                        encoder.encode(raw_damaged["tokens"]), dtype=np.float32
                    )
                augmentation_diagnostics = raw_damaged.get("diagnostics")
                augmentation_seconds = time.perf_counter() - tick
            model.train()
            order = np.random.default_rng(seed + epoch).permutation(
                len(raw_train["ids"])
            )
            sums, n = ({}, 0)
            for begin in range(0, len(order), batch_size):
                tick = time.perf_counter()
                indices = order[begin : begin + batch_size]
                full_batch = _batch(clean, indices, device)
                damaged_batch = (
                    _batch(damaged, indices, device) if damaged is not None else None
                )
                optimizer.zero_grad(set_to_none=True)
                with _amp(device, bool(config.get("training_amp", True))):
                    full_output = _forward(model, full_batch)
                    damaged_output = (
                        _forward(model, damaged_batch)
                        if damaged_batch is not None
                        else None
                    )
                    loss, components = compute_training_loss(
                        full_output,
                        full_batch["labels"],
                        full_batch["targets"],
                        loss_config,
                        damaged_output=damaged_output,
                        full_batch=full_batch,
                        damaged_batch=damaged_batch,
                        aux_valid=(
                            damaged_batch.get("aux_valid") if damaged_batch else None
                        ),
                        class_weights=weights,
                    )
                if not torch.isfinite(loss):
                    raise FloatingPointError("nonfinite training loss")
                loss.backward()
                gradient = torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    float(config.get("grad_clip", 1.0)),
                )
                if not torch.isfinite(gradient):
                    raise FloatingPointError("nonfinite training gradient")
                optimizer.step()
                scheduler.step()
                step += 1
                for key, value in components.items():
                    sums[key] = sums.get(key, 0.0) + float(
                        value.detach().float().cpu()
                    ) * len(indices)
                n += len(indices)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                step_times.append(time.perf_counter() - tick)
            train_seconds = time.perf_counter() - epoch_start
            row = {
                "epoch": epoch,
                "total_steps": step,
                "train_seconds": train_seconds,
                "augmentation_seconds": augmentation_seconds,
                "augmentation": augmentation_diagnostics,
                "train_loss": {k: v / n for k, v in sums.items()},
                "elapsed_seconds": time.perf_counter() - start,
            }
            if fixed is None:
                tick = time.perf_counter()
                prediction = predict(model, valid, device, eval_batch)
                validation_calls += 1
                metrics = compute_metrics(
                    prediction["probs"],
                    prediction["pred"],
                    prediction["labels"],
                    prediction["targets"],
                )
                score = float(condition_loss(metrics))
                if score < best - float(config.get("min_delta", 0.0001)):
                    best, best_epoch, stale = (score, epoch, 0)
                    torch.save(
                        checkpoint_state(
                            model, config, scaler, provenance, epoch, score
                        ),
                        outdir / "best.pt",
                    )
                else:
                    stale += 1
                row.update(
                    selection_score=score,
                    clean_metrics=metrics,
                    best_epoch=best_epoch,
                    validation_seconds=time.perf_counter() - tick,
                )
            else:
                row.update(selection_score=None, heldout_evaluation_skipped=True)
            append_jsonl(outdir / "epochs.jsonl", row)
            print(
                json.dumps(
                    {
                        "event": "epoch",
                        "run": outdir.name,
                        "epoch": epoch,
                        "score": row["selection_score"],
                        "seconds": train_seconds,
                    }
                ),
                flush=True,
            )
            if (
                fixed is None
                and epoch
                >= int(config.get("earliest_stop_epoch", 5 if finetune else 10))
                and (stale >= int(config.get("patience", 5 if finetune else 8)))
            ):
                break
        if fixed is not None:
            best_epoch = epoch
            torch.save(
                checkpoint_state(model, config, scaler, provenance, epoch, None),
                outdir / "best.pt",
            )
        state = torch.load(outdir / "best.pt", map_location="cpu", weights_only=False)
        _restore(model, state)
        prediction = predict(model, valid, device, eval_batch)
        validation_calls += 1
        metrics = compute_metrics(
            prediction["probs"],
            prediction["pred"],
            prediction["labels"],
            prediction["targets"],
        )
        np.savez_compressed(outdir / "validation_predictions.npz", **prediction)
        result = {
            **provenance,
            "status": "complete",
            "completed_utc": now(),
            "best_epoch": best_epoch,
            "epochs_run": epoch,
            "total_steps": step,
            "selection_score": None if fixed is not None else best,
            "evaluation_score": float(condition_loss(metrics)),
            "clean_metrics": metrics,
            "validation_calls": validation_calls,
            "elapsed_seconds": time.perf_counter() - start,
            "checkpoint_sha256": sha256(outdir / "best.pt"),
            "mean_step_seconds": float(np.mean(step_times)),
            "max_gpu_allocated_bytes": (
                torch.cuda.max_memory_allocated(device)
                if device.type == "cuda"
                else None
            ),
            "max_gpu_reserved_bytes": (
                torch.cuda.max_memory_reserved(device)
                if device.type == "cuda"
                else None
            ),
        }
        save_json(outdir / "result.json", result)
        save_json(outdir / "status.json", result)
        print(
            json.dumps(
                {
                    "event": "complete",
                    "run": outdir.name,
                    "epochs": epoch,
                    "best_epoch": best_epoch,
                    "metrics": metrics,
                }
            ),
            flush=True,
        )
        return result
    except Exception as exc:
        save_json(
            outdir / "status.json",
            {
                **provenance,
                "status": "failed",
                "error": repr(exc),
                "total_steps": step,
                "elapsed_seconds": time.perf_counter() - start,
            },
        )
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--bert", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8-sig"))
    train(config, args.source, args.bert, args.output, args.device)


if __name__ == "__main__":
    main()
