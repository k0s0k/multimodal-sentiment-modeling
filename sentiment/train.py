"""Train official-train-only Q2 models; select on the fixed valid_tune bank."""

from __future__ import annotations
import argparse, contextlib, copy, hashlib, json, math, time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from .common import (
    MODEL_KEYS,
    save_json,
    append_jsonl,
    now,
    sha256,
    seed_everything,
    to_device,
    source_manifest,
)
from .models import Model
from .losses import compute_loss, inverse_sqrt_class_weights
from .data import load_official, fit_scaler
from .cache import CacheDataset
from .metrics import compute_metrics, aggregate_conditions, risk_score, condition_loss


def loader(dataset, batch_size, shuffle=False):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )


def amp(device, enabled=True):
    return (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if enabled and str(device).startswith("cuda")
        else contextlib.nullcontext()
    )


def _source_hashes(record):
    return {
        str(record[k])
        for k in ("source_sha256", "original_source_sha256")
        if record.get(k)
    }


def _array_hash(value, dtype):
    return hashlib.sha256(np.asarray(value, dtype=dtype).tobytes(order="C")).hexdigest()


def _assert_scaler_identity(first, second, allowed_sources):
    """Exact train statistics; permit a verified original/working source alias."""
    for value in (first, second):
        if (
            value.get("fitted_split") != "train"
            or value.get("source_sha256") not in allowed_sources
        ):
            raise ValueError("Scaler must be fitted on this train source")
    a, b = (dict(first), dict(second))
    a.pop("source_sha256", None)
    b.pop("source_sha256", None)
    if a != b:
        raise ValueError("Scaler statistics or ordered train IDs differ")


def validate_training_identity(raw, dataset, scaler, cache_scaler=None):
    """Fail before optimization if CLI source and cached train data differ.

    cache.py verifies the original/working SHA mapping when building metadata.
    Paths are deliberately not identity: copies may live on different hosts.
    """
    meta, manifest = (dataset.metadata, dataset.manifest)
    if (
        raw.get("split") != "train"
        or meta.get("split") != "train"
        or meta.get("official_split") != "train"
        or (manifest.get("split") != "train")
        or (not meta.get("labeled"))
    ):
        raise ValueError("Training identity requires the labeled official train split")
    aliases = _source_hashes(meta)
    if not aliases or not _source_hashes(raw) & aliases:
        raise ValueError("Raw/cache train source SHA mismatch")
    if meta.get("ids") != raw["ids"] or meta.get("n") != len(raw["ids"]):
        raise ValueError("Raw/cache ordered sample IDs differ")
    if meta.get("video_ids") != raw["video_ids"]:
        raise ValueError("Raw/cache video IDs differ")
    for key in ("labels", "targets"):
        if key not in dataset.base or not np.array_equal(raw[key], dataset.base[key]):
            raise ValueError(f"Raw/cache train {key} differ")
    if cache_scaler is None:
        cache_scaler = json.loads(
            (dataset.root / "scaler.json").read_text(encoding="utf-8")
        )
    _assert_scaler_identity(scaler, cache_scaler, aliases)
    return {
        "version": 1,
        "split": "train",
        "n": len(raw["ids"]),
        "source_sha256": raw["source_sha256"],
        "original_source_sha256": meta.get("original_source_sha256"),
        "cache_source_sha256": meta["source_sha256"],
        "source_aliases": sorted(aliases),
        "ids_sha256": hashlib.sha256("\n".join(raw["ids"]).encode()).hexdigest(),
        "labels_sha256": _array_hash(raw["labels"], "<i8"),
        "targets_sha256": _array_hash(raw["targets"], "<f4"),
    }


def validate_teacher_identity(state, identity, scaler):
    config, provenance = (state.get("config", {}), state.get("provenance", {}))
    loss = config.get("loss", {})
    if (
        not config.get("original_only")
        or not config.get("selection_clean_only")
        or loss.get("kd_mode", "none") not in ("none", "off")
        or any(
            (
                float(loss.get(k, 0)) != 0
                for k in (
                    "lambda_original_view",
                    "lambda_reconstruction",
                    "lambda_hlfr",
                )
            )
        )
        or (provenance.get("training_role") != "original_view_teacher")
    ):
        raise ValueError(
            "Checkpoint is not a verified original-view, train-only teacher"
        )
    prior = provenance.get("train_identity", {})
    for key in (
        "version",
        "split",
        "n",
        "ids_sha256",
        "labels_sha256",
        "targets_sha256",
    ):
        if prior.get(key) != identity.get(key):
            raise ValueError(f"Teacher train identity differs: {key}")
    aliases = set(identity["source_aliases"])
    if (
        not set(prior.get("source_aliases", [])) & aliases
        or provenance.get("input_source_sha256") not in aliases
    ):
        raise ValueError("Teacher train source differs")
    _assert_scaler_identity(state.get("scaler", {}), scaler, aliases)


def validate_core_bank(dataset, mask_seed):
    manifest = dataset.manifest
    if (
        manifest.get("split") != "valid_tune"
        or dataset.metadata.get("split") != "valid_tune"
    ):
        raise ValueError("Checkpoint selection only accepts valid_tune")
    if manifest.get("bank") != "core" or manifest.get("mask_seed") != int(mask_seed):
        raise ValueError(
            "Core validation bank or precomputed mask seed differs from requested protocol"
        )
    names = dataset.get_view_names()
    if len(names) != 34 or len(set(names)) != 34 or names[0] != "clean":
        raise ValueError(
            "Core selection bank requires clean plus 33 unique missing conditions"
        )


@torch.inference_mode()
def predict(model, dataset, device, batch_size=256, autocast=True):
    model.eval()
    probs = []
    pred = []
    labels = []
    targets = []
    indices = []
    for batch in loader(dataset, batch_size):
        inputs = to_device(batch, device, model_only=True)
        with amp(device, autocast):
            out = model(inputs)
        probs.append(out["logits"].float().softmax(-1).cpu().numpy())
        pred.append(out["regression"].float().cpu().numpy().reshape(-1))
        if "labels" in batch:
            labels.append(np.asarray(batch["labels"]))
        if "targets" in batch:
            targets.append(np.asarray(batch["targets"]))
        indices.append(np.asarray(batch["index"]))
    return dict(
        probs=np.concatenate(probs),
        pred=np.concatenate(pred),
        labels=np.concatenate(labels) if labels else None,
        targets=np.concatenate(targets) if targets else None,
        indices=np.concatenate(indices),
    )


_VALID_DATASETS = {}


def evaluate_core(
    model,
    cache,
    scaler,
    device,
    bank="core",
    mask_seed=31,
    batch_size=256,
    only_clean=False,
    save_dir=None,
):
    cache_key = (str(cache), bank, mask_seed, id(scaler))
    if cache_key not in _VALID_DATASETS:
        _VALID_DATASETS[cache_key] = CacheDataset(
            cache,
            split="valid_tune",
            bank=bank,
            view="clean",
            seed=mask_seed,
            scaler=scaler,
        )
    prototype = _VALID_DATASETS[cache_key]
    validate_core_bank(prototype, mask_seed)
    names = ["clean"] if only_clean else prototype.get_view_names()
    records = []
    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
    for name in names:
        if hasattr(prototype, "with_view"):
            dataset = prototype.with_view(name)
        else:
            dataset = copy.copy(prototype)
            dataset.view = prototype.names.index(name)
        prediction = predict(model, dataset, device, batch_size)
        met = compute_metrics(
            prediction["probs"],
            prediction["pred"],
            prediction["labels"],
            prediction["targets"],
        )
        records.append(
            {
                "condition_id": name,
                "mask_seed": mask_seed,
                "is_clean": name == "clean",
                "metrics": met,
            }
        )
        if save_dir:
            np.savez_compressed(Path(save_dir) / f"{name}.npz", **prediction)
    conditions = aggregate_conditions(records)
    score = (
        condition_loss(records[0]["metrics"])
        if only_clean
        else risk_score(conditions)["R"]
    )
    return (float(score), records, conditions)


def initialize_model(config, counts, mean, device):
    model = Model(config).to(device)
    model.set_priors(counts / counts.sum(), float(mean))
    return model


@torch.inference_mode()
def teacher_predictions(path, train, cache, scaler, device, batch_size, train_identity):
    state = torch.load(path, map_location="cpu", weights_only=False)
    validate_teacher_identity(state, train_identity, scaler)
    device = torch.device(device)
    devices = (
        [device.index if device.index is not None else torch.cuda.current_device()]
        if device.type == "cuda"
        else []
    )
    with torch.random.fork_rng(devices=devices):
        model = Model(state["model_config"]).to(device)
        model.load_state_dict(state["model_state"])
        model.eval()
        data = CacheDataset(
            cache, split="train", bank=train, view="clean", scaler=scaler
        )
        logits = torch.empty((len(data), 3), device=device)
        reg = torch.empty(len(data), device=device)
        for batch in loader(data, batch_size):
            with amp(device):
                out = model(to_device(batch, device, model_only=True))
            idx = batch["index"].to(device)
            logits[idx] = out["logits"].float()
            reg[idx] = out["regression"].float().reshape(-1)
        del model
    return {"logits": logits, "regression": reg}


def run(args):
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    outdir = Path(args.output).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    if (outdir / "result.json").exists():
        raise FileExistsError(
            "Finished run exists; use a new run ID rather than overwrite."
        )
    seed = int(config.get("seed", 20260924))
    seed_everything(seed)
    device = torch.device(args.device)
    raw = load_official(args.source, "train")
    scaler = fit_scaler(raw)
    counts = np.bincount(raw["labels"], minlength=3).astype(np.float64)
    mean = float(np.mean(raw["targets"]))
    model = initialize_model(config["model"], counts, mean, device)
    train_bank = config.get("train_bank", "train-block")
    train_view = "clean" if config.get("original_only", False) else None
    dataset = CacheDataset(
        args.cache,
        split="train",
        bank=train_bank,
        view=train_view,
        seed=seed,
        scaler=scaler,
    )
    train_identity = validate_training_identity(raw, dataset, scaler)
    batch_size = int(config.get("batch_size", 64))
    train_loader = loader(dataset, batch_size, shuffle=True)
    loss_cfg = config["loss"]
    weights = (
        inverse_sqrt_class_weights(counts=counts).to(device)
        if config.get("class_weights", "inverse_sqrt") == "inverse_sqrt"
        else None
    )
    teacher = None
    if loss_cfg.get("kd_mode", "none") != "none":
        if not args.teacher:
            raise ValueError("KD requested without a frozen train-only teacher.")
        teacher = teacher_predictions(
            args.teacher,
            train_bank,
            args.cache,
            scaler,
            device,
            batch_size * 2,
            train_identity,
        )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.get("learning_rate", 0.0003),
        weight_decay=config.get("weight_decay", 0.01),
    )
    max_epochs = int(config.get("max_epochs", 50))
    steps_per_epoch = len(train_loader)
    total = max_epochs * steps_per_epoch
    warm = max(1, int(total * 0.05))

    def lr_scale(step):
        if step < warm:
            return (step + 1) / warm
        return 0.5 * (
            1 + math.cos(math.pi * min(1, (step - warm) / max(1, total - warm)))
        )

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_scale)
    save_json(outdir / "scaler.json", scaler)
    save_json(outdir / "config.json", config)
    params = sum((p.numel() for p in model.parameters() if p.requires_grad))
    start = time.perf_counter()
    best = float("inf")
    best_epoch = 0
    stale = 0
    global_step = 0
    step_times = []
    provenance = {
        "started_utc": now(),
        "seed": seed,
        "train_samples": len(dataset),
        "counts": counts,
        "train_target_mean": mean,
        "train_identity": train_identity,
        "training_role": (
            "original_view_teacher"
            if config.get("original_only") and config.get("selection_clean_only")
            else "student"
        ),
        "trainable_parameters": params,
        "input_source_sha256": sha256(args.source),
        "config_sha256": sha256(args.config),
        "sources": source_manifest(Path(__file__).resolve().parents[1]),
        "teacher_sha256": sha256(args.teacher) if args.teacher else None,
        "training_precision": "bf16_autocast_float32_parameters",
        "deterministic_algorithms": False,
        "model_provenance": (
            model.provenance()
            if callable(getattr(model, "provenance", None))
            else getattr(model, "provenance", None)
        ),
        "status": "running",
    }
    save_json(outdir / "status.json", provenance)
    print(
        json.dumps(
            {
                "event": "started",
                "run": outdir.name,
                "parameters": params,
                "epochs": max_epochs,
                "steps_per_epoch": steps_per_epoch,
            }
        ),
        flush=True,
    )
    try:
        for epoch in range(1, max_epochs + 1):
            dataset.set_epoch(epoch)
            model.train()
            epoch_start = time.perf_counter()
            sums = {}
            n = 0
            for batch in train_loader:
                tick = time.perf_counter()
                batch_device = to_device(batch, device)
                idx = batch["index"].long()
                need_original = (
                    loss_cfg.get("lambda_original_view", 0) > 0
                    or loss_cfg.get("lambda_reconstruction", 0) > 0
                    or loss_cfg.get("lambda_hlfr", 0) > 0
                    or (teacher is not None)
                )
                clean = (
                    to_device(dataset.get_clean_batch(idx.tolist()), device)
                    if need_original
                    else None
                )
                optimizer.zero_grad(set_to_none=True)
                with amp(device):
                    output = model({k: batch_device[k] for k in MODEL_KEYS})
                    original_output = (
                        model({k: clean[k] for k in MODEL_KEYS})
                        if clean is not None
                        and (
                            loss_cfg.get("lambda_original_view", 0) > 0
                            or loss_cfg.get("lambda_hlfr", 0) > 0
                        )
                        else None
                    )
                    teacher_out = (
                        {k: v[idx.to(device)] for k, v in teacher.items()}
                        if teacher is not None
                        else None
                    )
                    recon_mask = (
                        clean["mask"] & ~batch_device["mask"]
                        if clean is not None
                        else None
                    )
                    loss, components = compute_loss(
                        output,
                        batch_device["labels"],
                        batch_device["targets"],
                        loss_cfg,
                        teacher_outputs=teacher_out,
                        original_outputs=original_output,
                        recon_targets=(
                            {k: clean[k] for k in ["text", "audio", "vision"]}
                            if clean is not None
                            else None
                        ),
                        recon_mask=recon_mask,
                        original_observed=clean["mask"] if clean is not None else None,
                        current_observed=batch_device["mask"],
                        class_weights=weights,
                    )
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite loss")
                loss.backward()
                grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if not torch.isfinite(grad):
                    raise FloatingPointError("Non-finite gradient")
                optimizer.step()
                scheduler.step()
                global_step += 1
                for k, v in components.items():
                    scalar = float(v.detach().cpu()) if torch.is_tensor(v) else float(v)
                    sums[k] = sums.get(k, 0.0) + scalar * len(idx)
                n += len(idx)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                step_times.append(time.perf_counter() - tick)
                if global_step == 200:
                    save_json(
                        outdir / "benchmark_200_steps.json",
                        {
                            "steps": global_step,
                            "mean_last150_seconds": float(np.mean(step_times[-150:])),
                            "median_last150_seconds": float(
                                np.median(step_times[-150:])
                            ),
                            "max_gpu_allocated_bytes": (
                                torch.cuda.max_memory_allocated(device)
                                if device.type == "cuda"
                                else None
                            ),
                        },
                    )
            train_seconds = time.perf_counter() - epoch_start
            val_start = time.perf_counter()
            score, records, conditions = evaluate_core(
                model,
                args.cache,
                scaler,
                device,
                bank=args.valid_bank,
                batch_size=config.get("eval_batch_size", 256),
                only_clean=config.get("selection_clean_only", False),
            )
            validation_seconds = time.perf_counter() - val_start
            improved = score < best - 0.0001
            if improved:
                best = score
                best_epoch = epoch
                stale = 0
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "model_config": config["model"],
                        "scaler": scaler,
                        "config": config,
                        "epoch": epoch,
                        "selection_score": score,
                        "optimizer_state": optimizer.state_dict(),
                        "scheduler_state": scheduler.state_dict(),
                        "provenance": provenance,
                    },
                    outdir / "best.pt",
                )
            else:
                stale += 1
            row = {
                "epoch": epoch,
                "selection_score": score,
                "best_score": best,
                "best_epoch": best_epoch,
                "train_loss": {k: v / n for k, v in sums.items()},
                "train_seconds": train_seconds,
                "validation_seconds": validation_seconds,
                "clean_metrics": records[0]["metrics"],
                "elapsed_seconds": time.perf_counter() - start,
            }
            append_jsonl(outdir / "epochs.jsonl", row)
            save_json(
                outdir / "latest_validation.json",
                {"epoch": epoch, "records": records, "conditions": conditions},
            )
            print(
                json.dumps(
                    {
                        "event": "epoch",
                        "run": outdir.name,
                        **{
                            k: row[k]
                            for k in [
                                "epoch",
                                "selection_score",
                                "best_epoch",
                                "train_seconds",
                                "validation_seconds",
                            ]
                        },
                    }
                ),
                flush=True,
            )
            if epoch >= config.get("earliest_stop_epoch", 10) and stale >= config.get(
                "patience", 8
            ):
                break
        saved = torch.load(outdir / "best.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(saved["model_state"])
        score, records, conditions = evaluate_core(
            model,
            args.cache,
            scaler,
            device,
            bank=args.valid_bank,
            batch_size=config.get("eval_batch_size", 256),
            save_dir=outdir / "tune_core_seed31",
        )
        result = {
            **provenance,
            "status": "complete",
            "completed_utc": now(),
            "best_epoch": best_epoch,
            "epochs_run": epoch,
            "selection_score": best,
            "core_R": score,
            "records": records,
            "conditions": conditions,
            "elapsed_seconds": time.perf_counter() - start,
            "max_gpu_allocated_bytes": (
                torch.cuda.max_memory_allocated(device)
                if device.type == "cuda"
                else None
            ),
            "checkpoint_sha256": sha256(outdir / "best.pt"),
            "total_steps": global_step,
        }
        save_json(outdir / "result.json", result)
        save_json(outdir / "status.json", result)
        print(
            json.dumps(
                {
                    "event": "complete",
                    "run": outdir.name,
                    "best_epoch": best_epoch,
                    "core_R": score,
                    "seconds": result["elapsed_seconds"],
                }
            ),
            flush=True,
        )
    except Exception as exc:
        save_json(
            outdir / "status.json",
            {
                **provenance,
                "status": "failed",
                "error": repr(exc),
                "steps": global_step,
                "elapsed_seconds": time.perf_counter() - start,
            },
        )
        raise


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--source", required=True)
    p.add_argument("--cache", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--teacher")
    p.add_argument("--valid-bank", default="core")
    p.add_argument("--device", default="cuda:0")
    a = p.parse_args()
    run(a)


if __name__ == "__main__":
    main()
