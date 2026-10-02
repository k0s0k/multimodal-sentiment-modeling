"""Optional, separately budgeted last-two-BERT-layer fine-tuning.

Examples (from Q2_workplace):
 python -m sentiment.finetune train --config FT.json --source aligned.pkl    --cache CACHE --initialize frozen_best.pt --output RUN --model-path BERT_DIR
 python -m sentiment.finetune infer --checkpoint RUN/best.pt --cache CACHE    --split valid_tune --bank core_seed47 --output PRED --model-path BERT_DIR

Student text is ALWAYS encoded online from damaged tokens. Frozen cache text
is used only as a detached LLFR target, never as student input. Held-out
inference requires a frozen protocol JSON naming the exact checkpoint SHA.
"""

from __future__ import annotations
import argparse
import copy
import json
import math
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from .cache import CacheDataset, FrozenBertEncoder
from .common import (
    save_json,
    append_jsonl,
    sha256,
    now,
    seed_everything,
    to_device,
    source_manifest,
)
from .data import load_official, fit_scaler, stable_seed
from .models import Model
from .losses import compute_loss, inverse_sqrt_class_weights
from .metrics import compute_metrics, aggregate_conditions, risk_score
from .protocol import verify_freeze
from .train import (
    amp,
    validate_training_identity,
    validate_core_bank,
    teacher_predictions,
    _assert_scaler_identity,
    predict as predict_frozen,
)

ONLINE_KEYS = ("tokens", "audio", "vision", "mask", "structure")
FORMAT = "q2_last_two_bert_layers_v1"


class OnlineDataset:
    """Select the exact existing corruption view without reading its text shard."""

    def __init__(self, root=None, *, cached=None, **kwargs):
        self.cached = cached if cached is not None else CacheDataset(root, **kwargs)

    def __len__(self):
        return len(self.cached)

    def set_epoch(self, epoch):
        self.cached.set_epoch(epoch)

    def with_view(self, name):
        return OnlineDataset(cached=self.cached.with_view(name))

    def _item(self, index, view):
        data = self.cached
        v = data.view_arrays[view]
        drop = np.asarray(v["drop"][index], dtype=bool)
        mask = np.array(v["mask"][index], dtype=bool, copy=True)
        tokens = np.array(data.base["tokens"][index], dtype=np.int64, copy=True)
        tokens[:, drop[0]] = 0
        tokens[0, tokens[1] == 0] = 0
        tokens[2, tokens[1] == 0] = 0
        content = (tokens[1] == 1) & ~np.isin(tokens[0], [0, 101, 102])
        if not np.array_equal(content, mask[0]):
            raise ValueError("Online tokens and cached damaged text mask differ")
        item = {
            "tokens": torch.from_numpy(tokens),
            "mask": torch.from_numpy(mask),
            "structure": torch.from_numpy(np.array(v["structure"][index], copy=True)),
            "index": int(index),
            "ids": data.metadata["ids"][index],
            "video_ids": data.metadata["video_ids"][index],
        }
        for j, name in enumerate(("audio", "vision"), 1):
            raw = np.array(data.base[name][index], copy=True)
            raw[drop[j]] = 0
            value = np.where(
                mask[j, :, None], (raw - data._mean[name]) / data._std[name], 0
            )
            item[name] = torch.from_numpy(value.astype(np.float32))
        if "labels" in data.base:
            item["labels"] = torch.tensor(
                int(data.base["labels"][index]), dtype=torch.long
            )
            item["targets"] = torch.tensor(
                float(data.base["targets"][index]), dtype=torch.float32
            )
        return item

    def __getitem__(self, index):
        data = self.cached
        view = data.view
        if view is None:
            candidates = [i for i, name in enumerate(data.names) if name != "clean"]
            view = candidates[
                stable_seed(data.seed, data.epoch, data.metadata["ids"][index])
                % len(candidates)
            ]
        return self._item(int(index), int(view))

    def get_clean_batch(self, indices):
        return torch.utils.data.default_collate(
            [self._item(int(i), self.cached.names.index("clean")) for i in indices]
        )


def load_public_bert(model_path, device):
    encoder = FrozenBertEncoder(
        model_path=model_path, device=device, local_files_only=True
    )
    encoder.model.pooler = None
    metadata = dict(encoder.metadata)
    metadata["frozen_reference_compute_dtype"] = metadata.pop("compute_dtype")
    metadata["parameter_dtype"] = "float32"
    metadata["online_compute_dtype"] = "cuda_bfloat16_autocast_cpu_float32"
    return (encoder.model, metadata)


def assert_encoder_identity(metadata, cache_metadata):
    for key in ("repo", "revision", "weights_sha256"):
        if not metadata.get(key) or metadata[key] != cache_metadata.get(key):
            raise ValueError(f"Online/frozen-target BERT identity differs: {key}")


class OnlineBertFusion(nn.Module):

    def __init__(self, fusion, bert, bert_metadata):
        super().__init__()
        self.fusion, self.bert = (fusion, bert)
        self.bert_metadata = dict(bert_metadata)
        self.bert.pooler = None
        layers = self.bert.encoder.layer
        if len(layers) != 12:
            raise ValueError("Pinned BERT must contain 12 encoder layers")
        self.trainable_layer_indices = (10, 11)
        self.bert.requires_grad_(False)
        for i in self.trainable_layer_indices:
            layers[i].requires_grad_(True)
        self.train(self.training)

    def train(self, mode=True):
        super().train(mode)
        self.bert.eval()
        for i in self.trainable_layer_indices:
            self.bert.encoder.layer[i].train(mode)
        self.fusion.train(mode)
        return self

    def encode_text(self, tokens, content):
        if (
            tokens.ndim != 3
            or tokens.shape[1:] != (3, 50)
            or tokens.dtype != torch.long
        ):
            raise ValueError("Online tokens must be int64 [B,3,50]")
        actual = (
            (tokens[:, 1] == 1)
            & (tokens[:, 0] != 0)
            & (tokens[:, 0] != 101)
            & (tokens[:, 0] != 102)
        )
        if content.dtype != torch.bool or not torch.equal(actual, content):
            raise ValueError("Content mask does not match damaged tokens")
        indices = torch.nonzero(content.any(-1), as_tuple=False).flatten()
        result = torch.zeros(
            (*content.shape, 768), device=tokens.device, dtype=torch.float32
        )
        if not len(indices):
            return result
        selected = tokens[indices]
        attention = selected[:, 1]
        ids = torch.where(attention.bool(), selected[:, 0], 0)
        types = torch.where(attention.bool(), selected[:, 2], 0)
        h = self.bert(
            input_ids=ids,
            attention_mask=attention,
            token_type_ids=types,
            position_ids=torch.arange(50, device=tokens.device)[None].expand(
                len(indices), -1
            ),
            return_dict=True,
        ).last_hidden_state
        h = torch.where(
            content[indices, :, None],
            h.float(),
            torch.zeros_like(h, dtype=torch.float32),
        )
        return result.index_copy(0, indices, h)

    def forward(self, batch):
        text = self.encode_text(batch["tokens"], batch["mask"][:, 0])
        return self.fusion(
            {
                "text": text,
                **{k: batch[k] for k in ("audio", "vision", "mask", "structure")},
            }
        )

    def delta_state(self):
        prefixes = tuple((f"encoder.layer.{i}." for i in self.trainable_layer_indices))
        return {
            k: v.detach().cpu().clone()
            for k, v in self.bert.state_dict().items()
            if k.startswith(prefixes)
        }

    def load_delta(self, delta):
        expected = set(self.delta_state())
        if set(delta) != expected:
            raise ValueError(
                "BERT delta must contain exactly the final two encoder layers"
            )
        result = self.bert.load_state_dict(delta, strict=False)
        if result.unexpected_keys:
            raise ValueError("Unexpected BERT delta keys")


def online_inputs(batch, device):
    return to_device({k: batch[k] for k in ONLINE_KEYS}, device)


def online_loader(data, batch_size, shuffle=False):
    return DataLoader(
        data,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )


@torch.inference_mode()
def predict_online(model, data, device, batch_size=64):
    was_training = model.training
    model.eval()
    pieces = {k: [] for k in ("probs", "pred", "indices", "labels", "targets")}
    try:
        for batch in online_loader(data, batch_size):
            with amp(device):
                output = model(online_inputs(batch, device))
            pieces["probs"].append(output["logits"].float().softmax(-1).cpu().numpy())
            pieces["pred"].append(output["regression"].float().cpu().numpy())
            for key, source in (
                ("indices", "index"),
                ("labels", "labels"),
                ("targets", "targets"),
            ):
                if source in batch:
                    pieces[key].append(np.asarray(batch[source]))
        return {k: np.concatenate(v) for k, v in pieces.items() if v}
    finally:
        model.train(was_training)


def evaluate_online_core(model, data, device, batch_size=64, save_dir=None):
    validate_core_bank(data.cached, 31)
    records = []
    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
    for name in data.cached.get_view_names():
        prediction = predict_online(model, data.with_view(name), device, batch_size)
        metrics = compute_metrics(
            prediction["probs"],
            prediction["pred"],
            prediction["labels"],
            prediction["targets"],
        )
        records.append(
            {
                "condition_id": name,
                "mask_seed": 31,
                "is_clean": name == "clean",
                "metrics": metrics,
            }
        )
        if save_dir:
            np.savez_compressed(Path(save_dir) / f"{name}.npz", **prediction)
    conditions = aggregate_conditions(records)
    return (float(risk_score(conditions)["R"]), records, conditions)


def _validate_initialization(state, identity, scaler):
    p = state.get("provenance", {})
    previous = p.get("train_identity", {})
    for key in (
        "version",
        "split",
        "n",
        "ids_sha256",
        "labels_sha256",
        "targets_sha256",
    ):
        if previous.get(key) != identity.get(key):
            raise ValueError(f"Frozen fusion train identity differs: {key}")
    if p.get("input_source_sha256") not in identity["source_aliases"]:
        raise ValueError("Frozen fusion source differs")
    _assert_scaler_identity(
        state.get("scaler", {}), scaler, set(identity["source_aliases"])
    )


def save_checkpoint(path, model, config, scaler, provenance, epoch, score):
    torch.save(
        {
            "format": FORMAT,
            "fusion_state": model.fusion.state_dict(),
            "model_config": model.fusion.config,
            "bert_delta": model.delta_state(),
            "bert_metadata": model.bert_metadata,
            "trainable_layer_indices": [10, 11],
            "config": config,
            "scaler": scaler,
            "provenance": provenance,
            "epoch": epoch,
            "selection_score": score,
        },
        path,
    )


def load_finetuned(checkpoint, *, model_path=None, device="cuda:0", bert_loader=None):
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if state.get("format") != FORMAT or state.get("trainable_layer_indices") != [
        10,
        11,
    ]:
        raise ValueError("Unsupported fine-tuned checkpoint format")
    bert, metadata = (bert_loader or load_public_bert)(model_path, device)
    assert_encoder_identity(metadata, state["bert_metadata"])
    fusion = Model(state["model_config"]).to(device)
    fusion.load_state_dict(state["fusion_state"], strict=True)
    model = OnlineBertFusion(fusion, bert, metadata).to(device)
    model.load_delta(state["bert_delta"])
    return (model, state)


def _check_frozen_protocol(path, checkpoint, data_source_sha256=None):
    verify_freeze(path, checkpoints=[checkpoint], data_source_sha256=data_source_sha256)
    return sha256(path)


def infer_finetuned(
    checkpoint,
    cache_root,
    *,
    split="valid_tune",
    bank="core",
    view=None,
    model_path=None,
    device="cuda:0",
    batch_size=64,
    frozen_protocol=None,
    expected_mask_seed=None,
    output_dir=None,
):
    """Online inference; bank can be core/grid with any precomputed mask seed.

    Other than train/valid_tune, the shared protocol must register this exact
    singleton predictor AND cache source SHA. No held-out metric selects here.
    """
    freeze_hash = None
    if split not in ("train", "valid_tune"):
        _check_frozen_protocol(frozen_protocol, checkpoint)
        metadata_path = Path(cache_root) / split / "base" / "metadata.json"
        source_hash = json.loads(metadata_path.read_text(encoding="utf-8")).get(
            "source_sha256"
        )
        if not source_hash:
            raise ValueError("Held-out cache metadata lacks source SHA256")
        freeze_hash = _check_frozen_protocol(frozen_protocol, checkpoint, source_hash)
    model, state = load_finetuned(checkpoint, model_path=model_path, device=device)
    data = OnlineDataset(
        cache_root, split=split, bank=bank, view="clean", scaler=state["scaler"]
    )
    if split == "special" and (
        data.cached.manifest["bank"] != "clean" or data.cached.names != ["clean"]
    ):
        raise ValueError(
            "Attachment 3 inference must preserve only its provided damaged view"
        )
    assert_encoder_identity(model.bert_metadata, data.cached.manifest["encoder"])
    actual_seed = data.cached.manifest["mask_seed"]
    if expected_mask_seed is not None and actual_seed != int(expected_mask_seed):
        raise ValueError("Requested inference seed differs from cached corruption seed")
    names = [view] if view is not None else data.cached.get_view_names()
    results = {}
    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    for name in names:
        prediction = predict_online(model, data.with_view(name), device, batch_size)
        results[name] = prediction
        if output_dir:
            np.savez_compressed(Path(output_dir) / f"{name}.npz", **prediction)
    if output_dir:
        save_json(
            Path(output_dir) / "inference_manifest.json",
            {
                "format": FORMAT,
                "split": split,
                "bank": bank,
                "mask_seed": actual_seed,
                "views": names,
                "checkpoint_sha256": sha256(checkpoint),
                "frozen_protocol_sha256": freeze_hash,
                "ids": data.cached.metadata["ids"],
                "bert_metadata": model.bert_metadata,
                "student_text": "online_damaged_tokens_never_cached_embeddings",
            },
        )
    return results


def initialization_smoke(model, valid, device, batch_size, output_dir):
    """Clean valid_tune numerical comparison, zero optimizer steps, no selection.

    Compares the same initialized fusion using frozen cached text versus online
    public BERT under BF16. This separates numerical-path changes from training.
    """
    validate_core_bank(valid.cached, 31)
    device = torch.device(device)
    devices = (
        [device.index if device.index is not None else torch.cuda.current_device()]
        if device.type == "cuda"
        else []
    )
    was_training = model.training
    try:
        with torch.random.fork_rng(devices=devices):
            old = predict_frozen(
                model.fusion, valid.cached.with_view("clean"), device, batch_size
            )
            new = predict_online(model, valid.with_view("clean"), device, batch_size)
    finally:
        model.train(was_training)
    if not np.array_equal(old["indices"], new["indices"]):
        raise ValueError("Initialization comparison sample order differs")
    result = {
        "optimizer_steps": 0,
        "split": "valid_tune",
        "view": "clean",
        "used_for_selection": False,
        "frozen_cached_text": compute_metrics(
            old["probs"], old["pred"], old["labels"], old["targets"]
        ),
        "online_public_bert": compute_metrics(
            new["probs"], new["pred"], new["labels"], new["targets"]
        ),
        "max_absolute_probability_difference": float(
            np.max(np.abs(new["probs"] - old["probs"]))
        ),
        "mean_absolute_regression_difference": float(
            np.mean(np.abs(new["pred"] - old["pred"]))
        ),
        "max_absolute_regression_difference": float(
            np.max(np.abs(new["pred"] - old["pred"]))
        ),
        "class_prediction_agreement": float(
            np.mean(new["probs"].argmax(-1) == old["probs"].argmax(-1))
        ),
    }
    output_dir = Path(output_dir)
    save_json(output_dir / "initialization_smoke.json", result)
    np.savez_compressed(output_dir / "initialization_frozen.npz", **old)
    np.savez_compressed(output_dir / "initialization_online.npz", **new)
    return result


def run(args):
    requested = json.loads(Path(args.config).read_text(encoding="utf-8"))
    base = torch.load(args.initialize, map_location="cpu", weights_only=False)
    config = {**copy.deepcopy(base["config"]), **requested}
    if requested.get("model", base["model_config"]) != base["model_config"]:
        raise ValueError("Fine-tuning must preserve initialized fusion architecture")
    config["model"] = base["model_config"]
    defaults = dict(
        bert_learning_rate=1e-05,
        learning_rate=0.0001,
        batch_size=32,
        eval_batch_size=64,
        max_epochs=20,
        earliest_stop_epoch=5,
        patience=5,
    )
    for key, value in defaults.items():
        config[key] = requested.get(key, value)
    if config.get("original_only") or config.get("selection_clean_only"):
        raise ValueError(
            "Fine-tuning candidates use damaged training and core-R selection"
        )
    if config.get("train_bank", "train-block") != "train-block":
        raise ValueError(
            "Planned fine-tuning candidates are B4/structure block-bank models"
        )
    outdir = Path(args.output).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    if (outdir / "best.pt").exists() or (outdir / "status.json").exists():
        raise FileExistsError("Use a new fine-tuning run directory")
    seed = int(config.get("seed", 20260924))
    seed_everything(seed)
    device = torch.device(args.device)
    raw = load_official(args.source, "train")
    scaler = fit_scaler(raw)
    cached = CacheDataset(
        args.cache,
        split="train",
        bank="train-block",
        view=None,
        seed=seed,
        scaler=scaler,
    )
    identity = validate_training_identity(raw, cached, scaler)
    _validate_initialization(base, identity, scaler)
    data = OnlineDataset(cached=cached)
    valid = OnlineDataset(
        args.cache,
        split="valid_tune",
        bank=args.valid_bank,
        view="clean",
        scaler=scaler,
    )
    validate_core_bank(valid.cached, 31)
    bert, bert_metadata = load_public_bert(args.model_path, device)
    assert_encoder_identity(bert_metadata, cached.manifest["encoder"])
    assert_encoder_identity(bert_metadata, valid.cached.manifest["encoder"])
    fusion = Model(config["model"]).to(device)
    fusion.load_state_dict(base["model_state"], strict=True)
    model = OnlineBertFusion(fusion, bert, bert_metadata).to(device)
    loss_config = config["loss"]
    teacher = None
    if loss_config.get("kd_mode", "none") not in ("none", "off"):
        if not args.teacher:
            raise ValueError(
                "KD fine-tuning requires the frozen original-view teacher checkpoint"
            )
        previous_teacher = base["provenance"].get("teacher_sha256")
        if previous_teacher and sha256(args.teacher) != previous_teacher:
            raise ValueError(
                "Fine-tuning teacher differs from initialized student teacher"
            )
        teacher = teacher_predictions(
            args.teacher,
            "train-block",
            args.cache,
            scaler,
            device,
            config["eval_batch_size"],
            identity,
        )
    weights = inverse_sqrt_class_weights(
        counts=np.bincount(raw["labels"], minlength=3)
    ).to(device)
    if config.get("class_weights", "inverse_sqrt") != "inverse_sqrt":
        weights = None
    optimizer = torch.optim.AdamW(
        [
            {
                "params": [p for p in model.bert.parameters() if p.requires_grad],
                "lr": config["bert_learning_rate"],
            },
            {"params": model.fusion.parameters(), "lr": config["learning_rate"]},
        ],
        weight_decay=config.get("weight_decay", 0.01),
    )
    batches = online_loader(data, config["batch_size"], shuffle=True)
    total = config["max_epochs"] * len(batches)
    warmup = max(1, int(0.05 * total))

    def schedule(step):
        return (
            (step + 1) / warmup
            if step < warmup
            else 0.5
            * (1 + math.cos(math.pi * min(1, (step - warmup) / max(1, total - warmup))))
        )

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    provenance = {
        "started_utc": now(),
        "training_role": "last_two_layer_finetune_student",
        "train_identity": identity,
        "input_source_sha256": sha256(args.source),
        "initialization_sha256": sha256(args.initialize),
        "teacher_sha256": sha256(args.teacher) if args.teacher else None,
        "config_sha256": sha256(args.config),
        "sources": source_manifest(Path(__file__).resolve().parents[1]),
        "bert_metadata": bert_metadata,
        "bert_trainable_layers": [10, 11],
        "bert_trainable_parameters": sum(
            (p.numel() for p in model.bert.parameters() if p.requires_grad)
        ),
        "fusion_trainable_parameters": sum(
            (p.numel() for p in fusion.parameters() if p.requires_grad)
        ),
        "trainable_parameters": sum(
            (p.numel() for p in model.parameters() if p.requires_grad)
        ),
        "student_text": "online_damaged_tokens",
        "llfr_text_target": "detached_frozen_public_BERT_clean_cache",
        "original_student_view": "separate_online_forward",
        "precision": "bf16_autocast_fp32_parameters",
        "frozen_prefix_dropout": False,
        "seed": seed,
        "status": "running",
    }
    save_json(outdir / "config.json", config)
    save_json(outdir / "status.json", provenance)
    start = time.perf_counter()
    best = math.inf
    best_epoch = 0
    stale = 0
    step = 0
    try:
        if config.get("initialization_smoke", True):
            initialization_smoke(
                model, valid, device, config["eval_batch_size"], outdir
            )
        for epoch in range(1, config["max_epochs"] + 1):
            data.set_epoch(epoch)
            model.train()
            epoch_start = time.perf_counter()
            sums = {}
            n = 0
            for batch in batches:
                indices = batch["index"].long()
                current = to_device(batch, device)
                need_original = (
                    any(
                        (
                            loss_config.get(k, 0) > 0
                            for k in (
                                "lambda_original_view",
                                "lambda_reconstruction",
                                "lambda_hlfr",
                            )
                        )
                    )
                    or teacher is not None
                )
                original = (
                    to_device(data.get_clean_batch(indices.tolist()), device)
                    if need_original
                    else None
                )
                frozen = (
                    to_device(cached.get_clean_batch(indices.tolist()), device)
                    if loss_config.get("lambda_reconstruction", 0) > 0
                    else None
                )
                optimizer.zero_grad(set_to_none=True)
                with amp(device):
                    output = model({k: current[k] for k in ONLINE_KEYS})
                    original_output = (
                        model({k: original[k] for k in ONLINE_KEYS})
                        if original is not None
                        and (
                            loss_config.get("lambda_original_view", 0) > 0
                            or loss_config.get("lambda_hlfr", 0) > 0
                        )
                        else None
                    )
                    teacher_output = (
                        {k: v[indices.to(device)] for k, v in teacher.items()}
                        if teacher is not None
                        else None
                    )
                    loss, components = compute_loss(
                        output,
                        current["labels"],
                        current["targets"],
                        loss_config,
                        original_outputs=original_output,
                        teacher_outputs=teacher_output,
                        original_observed=(
                            original["mask"] if original is not None else None
                        ),
                        current_observed=current["mask"],
                        class_weights=weights,
                        recon_mask=(
                            original["mask"] & ~current["mask"]
                            if frozen is not None
                            else None
                        ),
                        recon_targets=(
                            {k: frozen[k] for k in ("text", "audio", "vision")}
                            if frozen is not None
                            else None
                        ),
                    )
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite fine-tuning loss")
                loss.backward()
                gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if not torch.isfinite(gradient):
                    raise FloatingPointError("Nonfinite fine-tuning gradient")
                optimizer.step()
                scheduler.step()
                step += 1
                for key, value in components.items():
                    sums[key] = sums.get(key, 0.0) + float(value.detach().cpu()) * len(
                        indices
                    )
                n += len(indices)
            train_seconds = time.perf_counter() - epoch_start
            valid_start = time.perf_counter()
            score, records, conditions = evaluate_online_core(
                model, valid, device, config["eval_batch_size"]
            )
            if score < best - 0.0001:
                best, best_epoch, stale = (score, epoch, 0)
                save_checkpoint(
                    outdir / "best.pt", model, config, scaler, provenance, epoch, score
                )
            else:
                stale += 1
            row = {
                "epoch": epoch,
                "selection_score": score,
                "best_epoch": best_epoch,
                "train_loss": {k: v / n for k, v in sums.items()},
                "train_seconds": train_seconds,
                "validation_seconds": time.perf_counter() - valid_start,
            }
            append_jsonl(outdir / "epochs.jsonl", row)
            save_json(
                outdir / "latest_validation.json",
                {"records": records, "conditions": conditions},
            )
            print(json.dumps({"event": "finetune_epoch", **row}), flush=True)
            if epoch >= config["earliest_stop_epoch"] and stale >= config["patience"]:
                break
        best_state = torch.load(
            outdir / "best.pt", map_location="cpu", weights_only=False
        )
        model.fusion.load_state_dict(best_state["fusion_state"])
        model.load_delta(best_state["bert_delta"])
        score, records, conditions = evaluate_online_core(
            model, valid, device, config["eval_batch_size"], outdir / "tune_core_seed31"
        )
        result = {
            **provenance,
            "status": "complete",
            "completed_utc": now(),
            "core_R": score,
            "best_epoch": best_epoch,
            "epochs_run": epoch,
            "steps": step,
            "records": records,
            "conditions": conditions,
            "elapsed_seconds": time.perf_counter() - start,
            "max_gpu_allocated_bytes": (
                torch.cuda.max_memory_allocated(device)
                if device.type == "cuda"
                else None
            ),
            "checkpoint_sha256": sha256(outdir / "best.pt"),
        }
        save_json(outdir / "result.json", result)
        save_json(outdir / "status.json", result)
    except Exception as exc:
        save_json(
            outdir / "status.json",
            {**provenance, "status": "failed", "error": repr(exc), "steps": step},
        )
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train")
    for flag in ("config", "source", "cache", "initialize", "output"):
        training.add_argument("--" + flag, required=True)
    training.add_argument("--teacher")
    training.add_argument("--valid-bank", default="core")
    inference = commands.add_parser("infer")
    for flag in ("checkpoint", "cache", "output"):
        inference.add_argument("--" + flag, required=True)
    inference.add_argument("--split", default="valid_tune")
    inference.add_argument("--bank", default="core")
    inference.add_argument("--view")
    inference.add_argument("--batch-size", type=int, default=64)
    inference.add_argument("--expected-mask-seed", type=int)
    inference.add_argument("--frozen-protocol")
    for command in (training, inference):
        command.add_argument("--model-path")
        command.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.command == "train":
        run(args)
    else:
        infer_finetuned(
            args.checkpoint,
            args.cache,
            split=args.split,
            bank=args.bank,
            view=args.view,
            model_path=args.model_path,
            device=args.device,
            batch_size=args.batch_size,
            frozen_protocol=args.frozen_protocol,
            expected_mask_seed=args.expected_mask_seed,
            output_dir=args.output,
        )


if __name__ == "__main__":
    main()
