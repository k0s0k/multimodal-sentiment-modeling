"""Evaluate fixed checkpoints or an equal-weight ensemble on immutable cache panels."""

from pathlib import Path
import argparse, json, time
import numpy as np
import torch
from .cache import CacheDataset
from .models import Model
from .train import predict
from .common import save_json, sha256, seed_everything
from .metrics import PredictionSet, compute_metrics, aggregate_conditions, risk_score
from .evaluate import save_predictions
from .protocol import verify_freeze


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoints", nargs="+", required=True)
    p.add_argument("--cache", required=True)
    p.add_argument("--split", default="valid_tune")
    p.add_argument("--banks", nargs="+", default=["core", "core_seed47", "core_seed73"])
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--freeze")
    p.add_argument("--save-csv", action="store_true")
    p.add_argument("--model-path")
    a = p.parse_args()
    if a.split != "valid_tune":
        verify_freeze(a.freeze, checkpoints=a.checkpoints)
    if len({sha256(x) for x in a.checkpoints}) != len(a.checkpoints):
        raise ValueError(
            "Duplicate checkpoints create implicit unequal ensemble weights"
        )
    seed_everything(20260924)
    outdir = Path(a.output)
    outdir.mkdir(parents=True, exist_ok=True)
    states = [
        torch.load(x, map_location="cpu", weights_only=False) for x in a.checkpoints
    ]
    scaler = states[0]["scaler"]
    if any(
        (
            json.dumps(s["scaler"], sort_keys=True)
            != json.dumps(scaler, sort_keys=True)
            for s in states
        )
    ):
        raise ValueError("Ensemble scaler mismatch")
    models = []
    for checkpoint, s in zip(a.checkpoints, states):
        if s.get("format") == "q2_last_two_bert_layers_v1":
            from .finetune import (
                load_finetuned,
                OnlineDataset,
                predict_online,
                assert_encoder_identity,
            )

            m, _ = load_finetuned(checkpoint, model_path=a.model_path, device=a.device)
            models.append(("online", m))
        else:
            m = Model(s["model_config"]).to(a.device)
            m.load_state_dict(s["model_state"])
            m.eval()
            models.append(("frozen", m))
    training_seed = (
        states[0]["config"]["seed"]
        if len(states) == 1
        else "equal_ensemble:" + ",".join((str(s["config"]["seed"]) for s in states))
    )
    source = {
        "checkpoint_sha256": [sha256(x) for x in a.checkpoints],
        "training_seed": training_seed,
        "ensemble_weights": [1 / len(states)] * len(states),
        "split": a.split,
        "freeze_sha256": sha256(a.freeze) if a.freeze else None,
        "banks": a.banks,
        "precision": "bf16_autocast_FP32_softmax",
    }
    if (outdir / "evaluation_summary.json").exists():
        raise FileExistsError("Evaluation output already exists")
    start = time.perf_counter()
    records = []
    clean_seen = False
    counts = []
    for bank in a.banks:
        ds = CacheDataset(
            a.cache, split=a.split, bank=bank, view="clean", scaler=scaler
        )
        mask_seed = int(ds.manifest["mask_seed"])
        if a.split != "valid_tune":
            verify_freeze(a.freeze, data_source_sha256=ds.metadata["source_sha256"])
        if a.split == "special" and (
            len(a.banks) != 1 or ds.manifest["bank"] != "clean" or ds.names != ["clean"]
        ):
            raise ValueError(
                "Attachment 3 inference must preserve only its provided damaged view"
            )
        bank_dir = outdir / f"{a.split}_{ds.manifest['bank']}_seed{mask_seed}"
        bank_dir.mkdir(exist_ok=True)
        for condition in ds.manifest["views"]:
            name = condition["name"]
            is_clean = name == "clean"
            if is_clean and clean_seen:
                continue
            clean_seen |= is_clean
            view = ds.with_view(name)
            predictions = []
            for kind, m in models:
                if kind == "online":
                    assert_encoder_identity(m.bert_metadata, ds.manifest["encoder"])
                    q = predict_online(
                        m, OnlineDataset(cached=view), a.device, a.batch_size
                    )
                    q.setdefault("labels", None)
                    q.setdefault("targets", None)
                else:
                    q = predict(m, view, a.device, a.batch_size)
                predictions.append(q)
            first = predictions[0]
            for q in predictions[1:]:
                for key in ("indices", "labels", "targets"):
                    if not np.array_equal(first[key], q[key]):
                        raise ValueError("Ensemble sample identity mismatch")
            probs = np.mean([q["probs"] for q in predictions], axis=0)
            pred = np.mean([q["pred"] for q in predictions], axis=0)
            bundle = PredictionSet(
                probs,
                pred,
                np.asarray(ds.metadata["ids"]),
                np.asarray(ds.metadata["video_ids"]),
                first["labels"],
                first["targets"],
            )
            arrays = {
                "probs": probs,
                "pred": pred,
                "ids": bundle.ids,
                "video_ids": bundle.video_ids,
                "indices": first["indices"],
                "condition_id": np.asarray(name),
                "mask_seed": np.asarray(mask_seed),
                "training_seed": np.asarray(str(training_seed)),
            }
            if bundle.labels is not None:
                arrays.update(labels=bundle.labels, targets=bundle.targets)
            if a.save_csv:
                save_predictions(
                    bundle,
                    bank_dir / name,
                    condition_id=name,
                    mask_seed=mask_seed,
                    training_seed=training_seed,
                )
            np.savez_compressed(bank_dir / f"{name}.npz", **arrays)
            row = {
                "condition_id": name,
                "mask_seed": mask_seed,
                "is_clean": is_clean,
                "training_seed": training_seed,
                "condition": condition["condition"],
            }
            if bundle.labels is not None:
                row["metrics"] = compute_metrics(
                    probs, pred, bundle.labels, bundle.targets
                )
            records.append(row)
        counts.append(
            {
                "bank": bank,
                "n": len(ds),
                "views": len(ds.names),
                "manifest_sha256": sha256(ds.directory / "manifest.json"),
                "source_sha256": ds.metadata["source_sha256"],
                "original_source_sha256": ds.metadata.get("original_source_sha256"),
                "base_metadata_sha256": ds.manifest["base_metadata_sha256"],
                "encoder": ds.manifest["encoder"],
            }
        )
        print(
            json.dumps(
                {
                    "event": "bank_complete",
                    "split": a.split,
                    "bank": bank,
                    "checkpoint_count": len(models),
                }
            ),
            flush=True,
        )
    conditions = aggregate_conditions(records) if "metrics" in records[0] else None
    risk = risk_score(conditions) if conditions and len(conditions) == 34 else None
    save_json(
        outdir / "evaluation_summary.json",
        {
            **source,
            "records": records,
            "conditions": conditions,
            "risk": risk,
            "cache_banks": counts,
            "elapsed_seconds": time.perf_counter() - start,
        },
    )


if __name__ == "__main__":
    main()
