"""Reproduce frozen predictions directly from the 30 provided aligned pickles.

No cache, training, extra masking, pairing, or label-based selection is used.
Example:
 python tools/predict_submission.py --input SPECIAL_ALIGNED_DIR    --checkpoints best1.pt best2.pt best3.pt --model-path PINNED_BERT_DIR    --freeze frozen_protocol.json --output submission.csv    --details-prefix review/submission --device cuda:0

The injectable predictor_loader/encoder_factory API supports synthetic tests
without Torch or downloaded weights. The CLI always uses the real backends.
"""

from __future__ import annotations
import argparse
import contextlib
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from types import SimpleNamespace

Q2_ROOT = Path(__file__).resolve().parents[1]
if str(Q2_ROOT) not in sys.path:
    sys.path.insert(0, str(Q2_ROOT))
import numpy as np
from sentiment.common import save_json, sha256, now
from sentiment.data import load_official, observed_masks, apply_scaler
from sentiment.evaluate import save_official_predictions, save_predictions
from sentiment.metrics import PredictionSet
from sentiment.protocol import verify_freeze

FINE_FORMAT = "q2_last_two_bert_layers_v1"


def normalize_device(value):
    value = str(value).lower()
    if re.fullmatch("cuda\\d+", value):
        value = "cuda:" + value[4:]
    if not re.fullmatch("cpu|cuda(?::\\d+)?", value):
        raise ValueError("Device must be cpu, cuda, or cuda:N")
    return value


def inspect_sources(directory):
    """Hash bytes only, before the freeze gate permits pickle deserialization."""
    directory = Path(directory).resolve()
    if not directory.is_dir():
        raise ValueError("--input must be the selected aligned pickle directory")
    paths = sorted(
        directory.glob("*.pkl"),
        key=lambda p: [
            int(t) if t.isdigit() else t for t in re.split("(\\d+)", p.name)
        ],
    )
    if not paths:
        raise ValueError("No aligned pickle files found")
    records = [{"file": p.name, "path": str(p), "sha256": sha256(p)} for p in paths]
    digest = hashlib.sha256(
        "".join((r["sha256"] for r in records)).encode()
    ).hexdigest()
    return (paths, records, digest)


def load_provided_inputs(paths, records, expected_count):
    items, source_files = ([], [])
    for path, record in zip(paths, records):
        item = load_official(path, "test")
        if item["source_sha256"] != record["sha256"]:
            raise ValueError("Input bytes changed after the frozen source check")
        if "labels" in item or "targets" in item:
            raise ValueError("Submission inputs must be unlabeled test records")
        if len(item["ids"]) != 1 or item["ids"] != [path.name + "#0"]:
            raise ValueError(
                "Each special file must be one sample identified as sourcebasename#0"
            )
        record["sample_id"] = item["ids"][0]
        items.append(item)
        source_files.append(path.name)
    if len(items) != int(expected_count) or int(expected_count) < 1:
        raise ValueError(
            f"Expected {expected_count} complete special samples, found {len(items)}"
        )
    data = {
        key: np.concatenate([x[key] for x in items])
        for key in ("tokens", "audio", "vision")
    }
    data["ids"] = sum([x["ids"] for x in items], [])
    data["video_ids"] = sum([x["video_ids"] for x in items], [])
    if len(set(data["ids"])) != len(items):
        raise ValueError("Duplicate special sample IDs")
    data["mask"] = observed_masks(data["tokens"], data["audio"], data["vision"])
    return (data, source_files)


def validate_scaler(scaler):
    if (
        scaler.get("fitted_split") != "train"
        or not scaler.get("source_sha256")
        or (not scaler.get("ids_sha256"))
    ):
        raise ValueError("Checkpoint must carry a source-identified, train-only scaler")
    for name, width in (("audio", 74), ("vision", 35)):
        mean, std = (np.asarray(scaler[name]["mean"]), np.asarray(scaler[name]["std"]))
        if (
            mean.shape != (width,)
            or std.shape != (width,)
            or (not np.isfinite(mean).all())
            or (not np.isfinite(std).all())
            or (std <= 0).any()
        ):
            raise ValueError(f"Invalid train scaler: {name}")


def load_predictor(checkpoint, model_path, device):
    """Real backend; returns kind/scaler/metadata/predict for a whole input batch."""
    import torch
    from sentiment.common import seed_everything
    from sentiment.models import Model
    from sentiment.train import amp

    seed_everything(20260924)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if state.get("format") == FINE_FORMAT:
        from sentiment.finetune import load_finetuned

        model, state = load_finetuned(checkpoint, model_path=model_path, device=device)
        kind = "online"
        keys = ("tokens", "audio", "vision", "mask", "structure")
        bert_metadata = model.bert_metadata
    else:
        model = Model(state["model_config"]).to(device)
        model.load_state_dict(state["model_state"], strict=True)
        kind = "frozen"
        keys = ("text", "audio", "vision", "mask", "structure")
        bert_metadata = None
    model.eval()

    @torch.inference_mode()
    def predict(batch):
        tensors = {k: torch.as_tensor(batch[k], device=device) for k in keys}
        with amp(device):
            output = model(tensors)
        return {
            "probs": output["logits"].float().softmax(-1).cpu().numpy(),
            "pred": output["regression"].float().cpu().numpy().reshape(-1),
        }

    runtime = {
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "device": str(device),
        "fusion_autocast": "bfloat16" if device.startswith("cuda") else None,
        "softmax_dtype": "float32",
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "nvidia_tf32_override": os.environ.get("NVIDIA_TF32_OVERRIDE"),
        "hardware": (
            torch.cuda.get_device_name(torch.device(device))
            if device.startswith("cuda")
            else "CPU"
        ),
    }
    return SimpleNamespace(
        kind=kind,
        scaler=state["scaler"],
        predict=predict,
        metadata={
            "format": state.get("format", "frozen_features_fusion"),
            "kind": kind,
            "training_seed": state.get("config", {}).get("seed"),
            "bert_metadata": bert_metadata,
            "training_provenance": state.get("provenance", {}),
            "runtime": runtime,
        },
    )


def make_frozen_encoder(model_path, device):
    import torch
    from sentiment.cache import FrozenBertEncoder

    encoder = FrozenBertEncoder(
        model_path=model_path, device=device, local_files_only=True
    )

    def encode(tokens):
        precision = torch.get_float32_matmul_precision()
        matmul, cudnn = (
            torch.backends.cuda.matmul.allow_tf32,
            torch.backends.cudnn.allow_tf32,
        )
        try:
            torch.set_float32_matmul_precision("highest")
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = True
            no_amp = (
                torch.autocast(device_type="cuda", enabled=False)
                if device.startswith("cuda")
                else contextlib.nullcontext()
            )
            with no_amp:
                return encoder(tokens)
        finally:
            torch.set_float32_matmul_precision(precision)
            torch.backends.cuda.matmul.allow_tf32 = matmul
            torch.backends.cudnn.allow_tf32 = cudnn

    return SimpleNamespace(
        encode=encode,
        metadata={
            **encoder.metadata,
            "autocast": False,
            "float32_matmul_precision": "highest",
            "matmul_allow_tf32": False,
            "cudnn_allow_tf32": True,
            "nvidia_tf32_override": os.environ.get("NVIDIA_TF32_OVERRIDE"),
            "roundtrip": "float32_encoder_to_float16_to_float32",
        },
    )


def _destinations(output_csv, details_prefix):
    output_csv = Path(output_csv).resolve()
    prefix = (
        Path(details_prefix).resolve()
        if details_prefix
        else output_csv.with_name(output_csv.stem + "_details")
    )
    paths = {
        "official_csv": output_csv,
        "details_csv": Path(str(prefix) + ".csv"),
        "details_npz": Path(str(prefix) + ".npz"),
        "members_npz": Path(str(prefix) + ".members.npz"),
        "provenance": Path(str(prefix) + ".provenance.json"),
        "provenance_sha256": Path(str(prefix) + ".provenance.sha256"),
    }
    paths = {name: path.resolve() for name, path in paths.items()}
    if len(set(paths.values())) != len(paths):
        raise ValueError("Output CSV and detail paths must be distinct")
    return (prefix, paths)


def predict_submission(
    input_dir,
    checkpoints,
    model_path,
    freeze,
    output_csv,
    *,
    details_prefix=None,
    device="cuda:0",
    expected_count=30,
    overwrite=False,
    predictor_loader=None,
    encoder_factory=None,
):
    """Run only a pre-registered predictor. Injected backends are for tests."""
    device = normalize_device(device)
    checkpoints = [Path(p).resolve() for p in checkpoints]
    freeze = Path(freeze).resolve()
    prefix, outputs = _destinations(output_csv, details_prefix)
    paths, sources, source_hash = inspect_sources(input_dir)
    protected = set(paths + checkpoints + [freeze])
    if any((p in protected for p in outputs.values())):
        raise ValueError(
            "Outputs may not overwrite input data, checkpoints, or the freeze protocol"
        )
    existing = [str(p) for p in outputs.values() if p.exists()]
    if existing and (not overwrite):
        raise FileExistsError(
            "Output exists; use --overwrite explicitly: " + ", ".join(existing)
        )
    verify_freeze(freeze, checkpoints=checkpoints, data_source_sha256=source_hash)
    freeze_digest = sha256(freeze)
    checkpoint_digests = [sha256(path) for path in checkpoints]
    data, source_files = load_provided_inputs(paths, sources, expected_count)
    predictors = [
        (predictor_loader or load_predictor)(p, model_path, device) for p in checkpoints
    ]
    if not predictors:
        raise ValueError("At least one frozen checkpoint is required")
    for predictor in predictors:
        if predictor.kind not in ("frozen", "online"):
            raise ValueError("Unknown prediction branch")
        validate_scaler(predictor.scaler)
    scaler = predictors[0].scaler
    if any(
        (
            json.dumps(p.scaler, sort_keys=True) != json.dumps(scaler, sort_keys=True)
            for p in predictors
        )
    ):
        raise ValueError("Ensemble checkpoints have different train scalers")
    prepared = apply_scaler(data, scaler)
    common = {k: prepared[k] for k in ("audio", "vision", "mask", "structure")}
    text, encoder_metadata = (None, None)
    if any((p.kind == "frozen" for p in predictors)):
        encoder = (encoder_factory or make_frozen_encoder)(model_path, device)
        encoded = np.asarray(encoder.encode(data["tokens"]), dtype=np.float32)
        if (
            encoded.shape != (len(data["ids"]), 50, 768)
            or not np.isfinite(encoded).all()
        ):
            raise ValueError("Frozen text encoder returned invalid features")
        text = encoded.astype(np.float16).astype(np.float32)
        text = np.where(data["mask"][:, 0, :, None], text, 0).astype(np.float32)
        if not np.isfinite(text).all():
            raise ValueError("FP16 frozen-text roundtrip overflowed")
        encoder_metadata = encoder.metadata
    member_probs, member_pred, member_metadata = ([], [], [])
    for checkpoint, predictor in zip(checkpoints, predictors):
        inputs = {key: value.copy() for key, value in common.items()}
        if predictor.kind == "frozen":
            inputs["text"] = text.copy()
        else:
            inputs["tokens"] = data["tokens"].copy()
        result = predictor.predict(inputs)
        probs = np.asarray(result["probs"], dtype=np.float32)
        pred = np.asarray(result["pred"], dtype=np.float32).reshape(-1)
        PredictionSet(
            probs, pred, np.asarray(data["ids"]), np.asarray(data["video_ids"])
        )
        if (np.abs(pred) > 3 + 1e-06).any():
            raise ValueError("Member intensity outside [-3,3]; do not silently clip")
        member_probs.append(probs)
        member_pred.append(pred)
        member_metadata.append(
            {
                "checkpoint": str(checkpoint),
                "sha256": sha256(checkpoint),
                **predictor.metadata,
            }
        )
    probs = np.mean(np.stack(member_probs), axis=0)
    pred = np.mean(np.stack(member_pred), axis=0)
    bundle = PredictionSet(
        probs, pred, np.asarray(data["ids"]), np.asarray(data["video_ids"])
    )
    if (
        sha256(freeze) != freeze_digest
        or [sha256(path) for path in checkpoints] != checkpoint_digests
    ):
        raise ValueError("Frozen protocol or checkpoint bytes changed during inference")
    if inspect_sources(input_dir)[2] != source_hash:
        raise ValueError("Special input directory changed during inference")
    for path in outputs.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    save_official_predictions(bundle, source_files, outputs["official_csv"])
    training_seed = "equal_ensemble:" + ",".join(
        (str(p.metadata.get("training_seed")) for p in predictors)
    )
    save_predictions(
        bundle,
        prefix,
        condition_id="provided_special_input",
        mask_seed="none",
        training_seed=training_seed,
    )
    np.savez_compressed(
        outputs["members_npz"],
        probs=np.stack(member_probs),
        pred=np.stack(member_pred),
        ids=np.asarray(data["ids"]),
        source_files=np.asarray(source_files),
        checkpoint_sha256=np.asarray([m["sha256"] for m in member_metadata]),
    )
    output_hashes = {
        k: {"path": str(v), "sha256": sha256(v)}
        for k, v in outputs.items()
        if k not in ("provenance", "provenance_sha256")
    }
    provenance = {
        "schema_version": 1,
        "status": "complete",
        "created_utc": now(),
        "purpose": "frozen_submission_reproduction_not_model_selection",
        "input_directory": str(Path(input_dir).resolve()),
        "source_sha256": source_hash,
        "sources": sources,
        "samples": len(data["ids"]),
        "expected_count": int(expected_count),
        "freeze": {"path": str(freeze), "sha256": freeze_digest},
        "members": member_metadata,
        "weights": [1 / len(predictors)] * len(predictors),
        "frozen_text_encoder": encoder_metadata,
        "scaler": scaler,
        "device": device,
        "input_policy": "provided_damaged_tokens_and_av_only_no_additional_mask_or_pairing",
        "positions": "original_0_to_49",
        "student_observed_counts": data["mask"].sum(axis=-1).tolist(),
        "labels_used": False,
        "softmax_and_ensemble_dtype": "float32",
        "outputs": output_hashes,
        "implementation_sha256": sha256(__file__),
    }
    save_json(outputs["provenance"], provenance)
    digest = sha256(outputs["provenance"])
    outputs["provenance_sha256"].write_text(
        digest + "  " + outputs["provenance"].name + "\n", encoding="utf-8"
    )
    return {
        "samples": len(data["ids"]),
        "output_csv": str(outputs["official_csv"]),
        "provenance": str(outputs["provenance"]),
        "provenance_sha256": digest,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True)
    p.add_argument("--checkpoints", nargs="+", required=True)
    p.add_argument("--model-path", required=True)
    p.add_argument("--freeze", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--details-prefix")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--expected-count", type=int, default=30)
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args()
    result = predict_submission(
        a.input,
        a.checkpoints,
        a.model_path,
        a.freeze,
        a.output,
        details_prefix=a.details_prefix,
        device=a.device,
        expected_count=a.expected_count,
        overwrite=a.overwrite,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
