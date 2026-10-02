"""Predict all attachment-4 samples and recompute their Shapley/Owen evidence."""

from pathlib import Path
import argparse
import csv
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from sentiment.data import observed_masks
from explanation.attribution import explain
from explanation.runtime import Predictor
from explanation.localization import gather_evidence
from token_mapping import load_sample, token_mapping


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sample_seed(seed, sample_id):
    value = json.dumps(
        [int(seed), "special", str(sample_id)],
        ensure_ascii=False,
        sort_keys=True,
        allow_nan=False,
    )
    return int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:16], 16) % 2**32


def build_row(sample_id, explanation, metadata, alignment, protocol):
    """Join freshly computed contributions to the independently measured timeline."""
    _, spans = gather_evidence(explanation, metadata, alignment)
    classes = ("Negative", "Neutral", "Positive")
    target = int(explanation["predicted_class"])
    full = explanation["full_output"]
    phi = explanation["modality_values"]
    dominant = explanation["classification_dominance"]
    regression = explanation["regression_dominance"]
    scope = dominant["modality"] if dominant["modality"] in "TAV" else "joint"
    primary = [
        s
        for s in spans
        if s["scope"] == scope and s["selection_kind"] == "class_support"
    ]
    if not primary:
        primary = [
            s
            for s in spans
            if s["scope"] == scope and s["selection_kind"] == "class_counterevidence"
        ]
    primary_ok = bool(primary) and all(
        (
            (s["char_start"] is not None and s["char_end"] is not None)
            if s["modality"] == "T"
            else (
                s["start_s"] is not None
                and (s["modality"] != "V" or bool(s["frame_indices"]))
            )
        )
        for s in primary
    )
    localization = (
        "主要证据可回看_原文定位"
        if primary_ok and all(s["modality"] == "T" for s in primary)
        else "主要证据可回看_词锚点近似" if primary_ok else "主要证据定位未完全通过"
    )
    row = dict(
        sample_id=sample_id,
        source_feature=f"official/attachment4/aligned_50/{sample_id}.pkl",
        source_video=f"official/attachment4/video/{sample_id}.mp4",
        model_id=protocol["candidate_id"],
        freeze_sha256=protocol["original_protocol_sha256"],
        numeric_profile=protocol["numeric_profile"],
        polarity=classes[target],
        intensity=full[3],
        prob_negative=full[0],
        prob_neutral=full[1],
        prob_positive=full[2],
        class_target=classes[target],
        dominant_modality=dominant["modality"],
        dominant_support=dominant["positive_support"],
        co_dominant=dominant["co_dominant"],
        dominant_intensity_modality=regression["modality"],
        base_class_probability=explanation["baseline_output"][target],
        base_intensity=explanation["baseline_output"][3],
        localization_status=localization,
        explanation_stability={m: explanation["local"][m]["status"] for m in "TAV"},
        details_jsonl_key=sample_id,
        explanation_card=f"cards/{sample_id}.svg",
    )
    for index, modality in enumerate("TAV"):
        row.update(
            {
                f"shap_class_{modality}": phi[index][target],
                f"shap_intensity_{modality}": phi[index][3],
                f"share_class_abs_{modality}": (
                    dominant["absolute_shares"][index]
                    if dominant["absolute_shares"]
                    else None
                ),
                f"share_intensity_abs_{modality}": (
                    regression["absolute_shares"][index]
                    if regression["absolute_shares"]
                    else None
                ),
                f"evidence_{modality}_json": [
                    s for s in spans if s["scope"] == modality
                ],
            }
        )
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--bert", type=Path, default=ROOT / "models/bert-base-uncased")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--predictions-only", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("Choose an empty output directory")
    args.output.mkdir(parents=True, exist_ok=True)
    protocol = read_json(ROOT / "configs/explanation_protocol.json")
    manifest = read_json(ROOT / "models/manifest.json")
    checkpoints = [ROOT / member["file"] for member in manifest["members"]]
    if [file_hash(p) for p in checkpoints] != protocol["checkpoint_sha256"]:
        raise ValueError("Model parameter checksum mismatch")
    samples = []
    for index in range(1, 21):
        path = args.input / f"{index:02}.pkl"
        if file_hash(path) != protocol["special_input_sha256"][path.name]:
            raise ValueError(f"Official input checksum mismatch: {path.name}")
        samples.append(load_sample(path))
    raw = {
        key: np.stack([np.asarray(s[field]) for s in samples])
        for key, field in (
            ("tokens", "text_bert"),
            ("audio", "audio"),
            ("vision", "vision"),
        )
    }
    raw["tokens"] = raw["tokens"].astype(np.int64)
    raw["audio"] = raw["audio"].astype(np.float32)
    raw["vision"] = raw["vision"].astype(np.float32)
    predictor = Predictor(checkpoints, args.bert, device=args.device, batch_size=64)
    outputs = predictor.predict(raw)
    np.savez_compressed(
        args.output / "predictions.npz",
        ids=np.asarray([f"{i:02}" for i in range(1, 21)]),
        outputs=outputs,
    )
    if args.predictions_only:
        return
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(args.bert / "tokenizer.json"))
    tokenizer.enable_truncation(max_length=50)
    tokenizer.enable_padding(length=50, pad_id=0, pad_token="[PAD]")
    alignments = {
        row["sample_id"]: row
        for row in map(
            json.loads,
            (ROOT / "results/attachment4_alignment.jsonl")
            .read_text("utf-8")
            .splitlines(),
        )
    }
    observed = observed_masks(raw["tokens"], raw["audio"], raw["vision"])
    rows = []
    options = protocol["options"]
    for index, sample in enumerate(samples):
        sample_id = f"{index + 1:02}"
        mapping = token_mapping(sample, tokenizer)
        metadata = read_json(ROOT / "results/token_mapping" / f"{sample_id}.json")
        if (
            not mapping["all_three_channels_exact"]
            or mapping["tokens"] != metadata["tokens"]
        ):
            raise ValueError("Official token/character correspondence differs")
        groups = {
            "T": [
                word["token_positions"]
                for word in mapping["words"]
                if word["token_positions"]
            ]
        }
        seed = sample_seed(options["seed"], sample_id)
        callback = predictor.callback(
            raw, index, max_cache_bytes=options["text_cache_bytes"]
        )
        result = explain(
            callback,
            observed[index],
            groups=groups,
            seed=seed,
            K=options["initial_K"],
            max_K=options["max_K"],
            adaptive=True,
            batch_size=options["batch_size"],
        )
        (args.output / f"{sample_id}.json").write_text(
            json.dumps(result, ensure_ascii=False, allow_nan=False), encoding="utf-8"
        )
        rows.append(
            build_row(sample_id, result, metadata, alignments[sample_id], protocol)
        )
        print(json.dumps({"completed": index + 1, "samples": 20}), flush=True)
        del callback
    columns = next(
        csv.reader(
            (ROOT / "results/attachment4_predictions_explanations.csv").open(
                encoding="utf-8-sig"
            )
        )
    )
    with (args.output / "attachment4_predictions_explanations.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: (
                        json.dumps(value, ensure_ascii=False, separators=(",", ":"))
                        if isinstance(value, (dict, list))
                        else value
                    )
                    for key, value in row.items()
                }
            )


if __name__ == "__main__":
    main()
