"""Frozen Q1 encoders. No labels are read and no training is performed.

python -m feature_extraction.encode --manifest manifest.jsonl --data-root E_data     --output outputs/q1 --branch text --device cuda:0

Inputs per sample: media.json, audio16k.wav on the common zero-based axis,
and alignment.json (words with original character offsets and nullable times).
Outputs: <branch>.npz and <branch>.meta.json, without pickle/object arrays.
Model IDs are fixed; --revision or Q1_TEXT_REVISION/Q1_AUDIO_REVISION can pin a
revision. HF_ENDPOINT/HF_HOME are respected. A first resolved revision is locked
in the output directory and reused on subsequent runs of the same request.
"""

from __future__ import annotations
import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile
from zipfile import BadZipFile
from datetime import datetime, timezone
from urllib.parse import urlsplit, urlunsplit
import numpy as np

SCHEMA_VERSION = 1
MODEL_IDS = {"text": "FacebookAI/roberta-large", "audio": "microsoft/wavlm-large"}
GAP_KINDS = {"silence", "nonverbal", "unaligned_speech", "unknown"}
MIN_GAP_S = 0.3
GRID_S = 0.1


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_hash(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode(
            "utf-8"
        )
    ).hexdigest()


def read_json(path: Path):
    with path.open(encoding="utf-8-sig") as handle:
        return json.load(handle)


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def versions(*names: str) -> dict:
    result = {"python": sys.version.split()[0], "numpy": np.__version__}
    for name in names:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = "not-installed"
    return result


def merge_intervals(intervals, lower: float, upper: float) -> np.ndarray:
    """Union valid observed intervals; never double-count overlap."""
    rows = []
    for pair in intervals:
        if len(pair) != 2:
            raise ValueError("An interval must contain exactly two endpoints")
        a, b = (float(pair[0]), float(pair[1]))
        if not np.isfinite([a, b]).all() or b < a:
            raise ValueError(f"Invalid interval: {pair}")
        a, b = (max(lower, a), min(upper, b))
        if b > a:
            rows.append((a, b))
    merged = []
    for a, b in sorted(rows):
        if merged and a <= merged[-1][1] + 1e-10:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return np.asarray(merged, dtype=np.float64).reshape(-1, 2)


def overlap_with_observed(
    cells: np.ndarray, interval, observed: np.ndarray
) -> np.ndarray:
    """Seconds in each non-overlapping cell intersecting target and observations."""
    weights = np.zeros(len(cells), dtype=np.float64)
    a, b = interval
    if not np.isfinite([a, b]).all() or b <= a:
        return weights
    for lo, hi in observed:
        weights += np.maximum(
            0.0,
            np.minimum(cells[:, 1], min(b, hi)) - np.maximum(cells[:, 0], max(a, lo)),
        )
    return weights


def allocation_cells(anchors, lower: float, upper: float) -> np.ndarray:
    """Midpoint allocation cells J, distinct from overlapping receptive fields S."""
    anchors = np.asarray(anchors, dtype=np.float64)
    if len(anchors) == 0:
        return np.empty((0, 2), dtype=np.float64)
    if not np.isfinite(anchors).all() or np.any(np.diff(anchors) <= 0):
        raise ValueError("Feature anchors must be finite and strictly increasing")
    if anchors[0] < lower - 1e-08 or anchors[-1] > upper + 1e-08:
        raise ValueError("Feature anchors fall outside their observed processing range")
    edges = np.concatenate(([lower], (anchors[:-1] + anchors[1:]) / 2, [upper]))
    return np.column_stack((edges[:-1], edges[1:])).astype(np.float64)


def pool_features(values, cells, targets, feature_valid, observed):
    """Duration means; FP32 accumulation, per-feature masks and union coverage."""
    values = np.asarray(values, dtype=np.float32)
    valid = np.asarray(feature_valid, dtype=bool)
    if valid.ndim == 1:
        valid = np.broadcast_to(valid[:, None], values.shape)
    if values.ndim != 2 or valid.shape != values.shape or len(cells) != len(values):
        raise ValueError("Pooling feature/time/mask shape mismatch")
    valid = valid & np.isfinite(values)
    clean = np.where(valid, values, 0).astype(np.float32)
    output = np.zeros((len(targets), values.shape[1]), dtype=np.float32)
    out_valid = np.zeros(output.shape, dtype=np.uint8)
    coverage = np.zeros(output.shape, dtype=np.float32)
    for i, target in enumerate(targets):
        weights = overlap_with_observed(cells, target, observed).astype(np.float32)
        weighted_mask = weights[:, None] * valid
        denom = weighted_mask.sum(axis=0, dtype=np.float32)
        usable = denom > 0
        output[i, usable] = (clean * weights[:, None]).sum(axis=0, dtype=np.float32)[
            usable
        ] / denom[usable]
        out_valid[i, usable] = 1
        if np.isfinite(target).all() and target[1] > target[0]:
            coverage[i] = np.minimum(1.0, denom / np.float32(target[1] - target[0]))
    return (output, out_valid, coverage)


def word_time(word, duration: float) -> list[float]:
    a, b = (word.get("start"), word.get("end"))
    if a is None or b is None:
        return [math.nan, math.nan]
    a, b = (float(a), float(b))
    if not np.isfinite([a, b]).all() or a < 0 or b <= a or (b > duration + 1 / 16000):
        return [math.nan, math.nan]
    return [a, min(b, duration)]


def build_nodes(
    words: list[dict], duration: float, alignment: dict
) -> tuple[list[dict], np.ndarray]:
    """All words plus unassigned >=300 ms intervals. Unlabelled gaps stay unknown."""
    nodes = [
        {
            "node_id": f"word_{i:04d}",
            "node_type": "word",
            "word_index": i,
            "original": str(
                word.get("original", word.get("text", word.get("word", "")))
            ),
        }
        for i, word in enumerate(words)
    ]
    times = [word_time(word, duration) for word in words]
    covered = merge_intervals([p for p in times if np.isfinite(p).all()], 0, duration)
    gaps, cursor = ([], 0.0)
    for a, b in covered:
        if a - cursor >= MIN_GAP_S - 1e-09:
            gaps.append((cursor, float(a)))
        cursor = max(cursor, float(b))
    if duration - cursor >= MIN_GAP_S - 1e-09:
        gaps.append((cursor, duration))
    annotations = []
    for entry in list(alignment.get("gaps", [])) + list(alignment.get("nodes", [])):
        kind = entry.get("node_type", entry.get("type", "unknown"))
        if kind in GAP_KINDS and kind != "unknown":
            bounds = word_time(entry, duration)
            if np.isfinite(bounds).all():
                annotations.append((bounds[0], bounds[1], kind))
    gap_id = 0
    for a, b in gaps:
        edges = sorted(
            {a, b}
            | {float(x) for lo, hi, _ in annotations for x in (lo, hi) if a < x < b}
        )
        for lo, hi in zip(edges[:-1], edges[1:]):
            kinds = {
                kind
                for s, e, kind in annotations
                if s <= lo + 1e-09 and e >= hi - 1e-09
            }
            kind = next(iter(kinds)) if len(kinds) == 1 else "unknown"
            nodes.append(
                {
                    "node_id": f"gap_{gap_id:04d}",
                    "node_type": kind,
                    "word_index": -1,
                    "classification_source": (
                        "alignment_annotation"
                        if kind != "unknown"
                        else "unassigned_time_not_automatically_nonverbal"
                    ),
                }
            )
            times.append([lo, hi])
            gap_id += 1
    return (nodes, np.asarray(times, dtype=np.float64).reshape(-1, 2))


def character_pool(token_features, token_offsets, words, text_length: int):
    """Last-layer subword vectors, weighted by exact original character overlap."""
    features = np.asarray(token_features, dtype=np.float32)
    offsets = np.asarray(token_offsets, dtype=np.int64).reshape(-1, 2)
    output = np.zeros((len(words), features.shape[1]), dtype=np.float32)
    mask = np.zeros(len(words), dtype=np.uint8)
    char_coverage = np.zeros(len(words), dtype=np.float32)
    word_spans = np.full((len(words), 2), -1, dtype=np.int64)
    for i, word in enumerate(words):
        a, b = (word.get("char_start"), word.get("char_end"))
        if a is None or b is None:
            continue
        if isinstance(a, bool) or isinstance(b, bool) or int(a) != a or (int(b) != b):
            raise ValueError("Character offsets must be integers")
        a, b = (int(a), int(b))
        if not 0 <= a < b <= text_length:
            continue
        word_spans[i] = [a, b]
        overlap = np.maximum(
            0, np.minimum(offsets[:, 1], b) - np.maximum(offsets[:, 0], a)
        )
        weight = overlap.astype(np.float32)
        if weight.sum() > 0:
            output[i] = (features * weight[:, None]).sum(
                axis=0, dtype=np.float32
            ) / weight.sum()
            mask[i] = 1
            union = merge_intervals(
                [[max(a, x), min(b, y)] for x, y in offsets if min(b, y) > max(a, x)],
                a,
                b,
            )
            char_coverage[i] = np.sum(union[:, 1] - union[:, 0]) / (b - a)
    return (output, mask, char_coverage, word_spans)


def wavlm_timing(n_samples: int, kernels, strides, sample_rate=16000):
    length, jump, receptive = (n_samples, 1, 1)
    for kernel, stride in zip(kernels, strides):
        length = (length - int(kernel)) // int(stride) + 1
        receptive += (int(kernel) - 1) * jump
        jump *= int(stride)
    if jump != 320 or receptive != 400 or length <= 0:
        raise ValueError(
            f"Unexpected WavLM timing: stride={jump}, RF={receptive}, T={length}"
        )
    starts = np.arange(length, dtype=np.float64) * jump / sample_rate
    receptive_fields = np.column_stack((starts, starts + receptive / sample_rate))
    anchors = receptive_fields.mean(axis=1)
    cells = allocation_cells(anchors, 0, n_samples / sample_rate)
    return (anchors, cells, receptive_fields)


def acoustic_validity(values, names, analysis_support, observed):
    """Do not treat unvoiced F0 (or pitch-conditioned sma3nz values) as measurements."""
    valid = np.isfinite(values)
    support_duration = analysis_support[:, 1] - analysis_support[:, 0]
    observed_duration = np.zeros(len(values), dtype=np.float64)
    for a, b in observed:
        observed_duration += np.maximum(
            0,
            np.minimum(analysis_support[:, 1], b)
            - np.maximum(analysis_support[:, 0], a),
        )
    valid &= (observed_duration >= support_duration - 1e-08)[:, None]
    f0_indices = [i for i, name in enumerate(names) if name.startswith("F0semitone")]
    if len(f0_indices) != 1:
        raise ValueError("eGeMAPSv02 must have exactly one F0semitone LLD")
    voiced = np.isfinite(values[:, f0_indices[0]]) & (values[:, f0_indices[0]] > 0)
    for i, name in enumerate(names):
        if name.endswith("sma3nz") or name.startswith("F0semitone"):
            valid[:, i] &= voiced
    analysis_observed = observed_duration >= support_duration - 1e-08
    return (valid, voiced, analysis_observed)


def _safe_endpoint() -> str:
    endpoint = urlsplit(os.environ.get("HF_ENDPOINT", "https://huggingface.co"))
    return urlunsplit((endpoint.scheme, endpoint.hostname or "", endpoint.path, "", ""))


def _hf_snapshot(model_id: str, requested_revision: str, lock_path: Path):
    """Download only this exact checkpoint's required files; never substitute a model."""
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError, LocalEntryNotFoundError

    revision = requested_revision
    if lock_path.exists():
        locked = read_json(lock_path)
        if (
            locked.get("model_id") == model_id
            and locked.get("requested_revision") == requested_revision
        ):
            revision = locked["resolved_revision"]
    config_path = Path(hf_hub_download(model_id, "config.json", revision=revision))
    snapshot = config_path.parent
    resolved = snapshot.name
    if not re.fullmatch("[0-9a-f]{40}", resolved):
        raise RuntimeError(f"Cannot establish immutable HF commit from {snapshot}")
    files = {"config.json": config_path}

    def download(name, optional=False):
        try:
            path = Path(hf_hub_download(model_id, name, revision=resolved))
            files[name] = path
            return path
        except (EntryNotFoundError, LocalEntryNotFoundError):
            if optional:
                return None
            raise

    if model_id == MODEL_IDS["text"]:
        for name in (
            "tokenizer_config.json",
            "tokenizer.json",
            "special_tokens_map.json",
            "vocab.json",
            "merges.txt",
            "added_tokens.json",
        ):
            download(name, optional=True)
        if "tokenizer.json" not in files and (
            not {"vocab.json", "merges.txt"} <= files.keys()
        ):
            raise RuntimeError("The pinned RoBERTa tokenizer files are unavailable")
    else:
        download("preprocessor_config.json")
    weight_format = None
    for name, fmt in (
        ("model.safetensors", "safetensors"),
        ("model.safetensors.index.json", "safetensors"),
        ("pytorch_model.bin", "pytorch"),
        ("pytorch_model.bin.index.json", "pytorch"),
    ):
        path = download(name, optional=True)
        if path is not None:
            weight_format = fmt
            if name.endswith(".index.json"):
                for shard in sorted(set(read_json(path)["weight_map"].values())):
                    download(shard)
            break
    if weight_format is None:
        raise RuntimeError(f"No supported weights found for {model_id}@{resolved}")
    provenance = {
        "model_id": model_id,
        "requested_revision": requested_revision,
        "resolved_revision": resolved,
        "source_url": f"https://huggingface.co/{model_id}/tree/{resolved}",
        "download_endpoint": _safe_endpoint(),
        "weight_format": weight_format,
        "files_sha256": {
            name: sha256_file(path) for name, path in sorted(files.items())
        },
    }
    atomic_json(lock_path, provenance)
    return (snapshot, provenance)


class HFEncoder:

    def __init__(self, branch: str, device: str, revision: str, output: Path):
        import torch
        from transformers import (
            AutoFeatureExtractor,
            AutoTokenizer,
            RobertaModel,
            WavLMModel,
        )

        self.torch, self.device, self.branch = (torch, device, branch)
        snapshot, self.provenance = _hf_snapshot(
            MODEL_IDS[branch], revision, output / f"{branch}.model.lock.json"
        )
        kwargs = {
            "local_files_only": True,
            "use_safetensors": self.provenance["weight_format"] == "safetensors",
        }
        if branch == "text":
            self.processor = AutoTokenizer.from_pretrained(
                snapshot, use_fast=True, local_files_only=True
            )
            if not self.processor.is_fast:
                raise RuntimeError(
                    "A fast RoBERTa tokenizer is required for original character offsets"
                )
            self.model = RobertaModel.from_pretrained(
                snapshot, add_pooling_layer=False, **kwargs
            )
        else:
            self.processor = AutoFeatureExtractor.from_pretrained(
                snapshot, local_files_only=True
            )
            self.model = WavLMModel.from_pretrained(snapshot, **kwargs)
            if self.processor.sampling_rate != 16000 or self.model.config.add_adapter:
                raise RuntimeError(
                    "Unexpected WavLM preprocessing/adapter configuration"
                )
        if (
            self.model.config.hidden_size != 1024
            or self.model.config.num_hidden_layers != 24
        ):
            raise RuntimeError(
                "Checkpoint architecture does not match the fixed 1024-dimensional large model"
            )
        self.model.to(device=device, dtype=torch.float32)
        torch.set_float32_matmul_precision("highest")
        self.model.eval()
        self.model.requires_grad_(False)
        self.provenance.update(
            {
                "frozen": True,
                "eval": True,
                "inference_dtype": "float32",
                "library_versions": versions(
                    "torch", "transformers", "huggingface-hub", "tokenizers"
                ),
            }
        )

    def text_tokens(self, text: str):
        torch = self.torch
        encoded = self.processor(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
            truncation=False,
        )
        token_ids, offsets = (encoded["input_ids"], encoded["offset_mapping"])
        count = len(token_ids)
        features = np.zeros((count, 1024), dtype=np.float32)
        totals = np.zeros(count, dtype=np.float32)
        if count == 0:
            return (features, np.asarray(offsets, dtype=np.int64).reshape(-1, 2))
        start = 0
        with torch.inference_mode():
            while start < count:
                stop = min(start + 510, count)
                content = token_ids[start:stop]
                ids = self.processor.build_inputs_with_special_tokens(content)
                if ids != [self.processor.bos_token_id] + content + [
                    self.processor.eos_token_id
                ]:
                    raise RuntimeError(
                        "Pinned RoBERTa tokenizer has an unexpected BOS/EOS layout"
                    )
                input_ids = torch.tensor([ids], dtype=torch.long, device=self.device)
                result = self.model(
                    input_ids=input_ids,
                    attention_mask=torch.ones_like(input_ids),
                    return_dict=True,
                ).last_hidden_state[0]
                hidden = result[1:-1].float().cpu().numpy()
                if hidden.shape != (stop - start, 1024):
                    raise RuntimeError("Unexpected RoBERTa special-token layout")
                weights = np.minimum(
                    np.arange(1, stop - start + 1), np.arange(stop - start, 0, -1)
                ).astype(np.float32)
                features[start:stop] += hidden * weights[:, None]
                totals[start:stop] += weights
                if stop == count:
                    break
                start = stop - 64
        return (features / totals[:, None], np.asarray(offsets, dtype=np.int64))

    def audio_frames(self, waveform: np.ndarray):
        torch = self.torch
        inputs = self.processor(
            waveform, sampling_rate=16000, return_tensors="pt", padding=False
        )
        inputs = {name: value.to(self.device) for name, value in inputs.items()}
        with torch.inference_mode():
            result = self.model(**inputs, output_hidden_states=True, return_dict=True)
            hidden = torch.stack(result.hidden_states[-4:], dim=0).mean(dim=0)[0]
            features = hidden.float().cpu().numpy()
        anchors, cells, receptive = wavlm_timing(
            len(waveform), self.model.config.conv_kernel, self.model.config.conv_stride
        )
        if features.shape != (len(anchors), 1024):
            raise RuntimeError(
                f"WavLM output shape {features.shape} disagrees with its convolution timing"
            )
        return (features, anchors, cells, receptive)


class AcousticEncoder:

    def __init__(self):
        import opensmile

        self.smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02,
            feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
        )
        self.names = list(self.smile.feature_names)
        if len(self.names) != 25:
            raise RuntimeError(
                f"Expected 25 eGeMAPSv02 LLDs, received {len(self.names)}"
            )
        package = Path(opensmile.__file__).parent
        digest = hashlib.sha256()
        file_count = 0
        for path in sorted(package.rglob("*")):
            if path.is_file() and (
                ".conf" in path.name or path.suffix in {".inc", ".so", ".dll", ".dylib"}
            ):
                digest.update(
                    str(path.relative_to(package)).replace("\\", "/").encode()
                )
                digest.update(bytes.fromhex(sha256_file(path)))
                file_count += 1
        self.provenance = {
            "tool": "opensmile",
            "feature_set": "eGeMAPSv02",
            "feature_level": "LowLevelDescriptors",
            "feature_names": self.names,
            "source_url": "https://github.com/audeering/opensmile-python",
            "package_config_and_binary_tree_sha256": digest.hexdigest(),
            "hashed_package_file_count": file_count,
            "library_versions": versions(
                "opensmile", "pandas", "audinterface", "soundfile"
            ),
        }

    def frames(self, waveform):
        result = self.smile.process_signal(waveform, sampling_rate=16000)
        values = result[self.names].to_numpy(dtype=np.float32)
        starts = (
            result.index.get_level_values("start")
            .total_seconds()
            .to_numpy(dtype=np.float64)
        )
        ends = (
            result.index.get_level_values("end")
            .total_seconds()
            .to_numpy(dtype=np.float64)
        )
        support = np.column_stack((starts, ends)).astype(np.float64)
        if len(values) == 0 or not np.isfinite(support).all() or np.any(ends <= starts):
            raise RuntimeError("openSMILE returned no valid timed LLD frames")
        return (values, support)


def load_manifest(path: Path) -> list[dict]:
    if path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
    else:
        content = path.read_text(encoding="utf-8-sig")
        rows = (
            json.loads(content)
            if content.lstrip().startswith("[")
            else [json.loads(line) for line in content.splitlines() if line.strip()]
        )
    keys = set()
    for row in rows:
        for name in (
            "sample_key",
            "sample_id",
            "video_id",
            "clip_id",
            "media_path",
            "text",
            "duration_s",
        ):
            if name not in row:
                raise ValueError(f"Manifest row lacks {name}")
        key = row["sample_key"]
        if (
            not isinstance(key, str)
            or not re.fullmatch("[A-Za-z0-9_-]+", key)
            or key in keys
        ):
            raise ValueError(f"Unsafe or duplicate sample_key: {key!r}")
        keys.add(key)
        if not isinstance(row["text"], str):
            raise ValueError("Manifest text must be the official transcript string")
    return rows


def load_sample(row, root: Path, output: Path, branch: str):
    directory = output / row["sample_key"]
    media_path = directory / "media.json"
    media = read_json(media_path)
    duration = float(media["duration_s"])
    if not np.isfinite(duration) or duration <= 0:
        raise ValueError("media.duration_s must be positive")
    if media.get("sample_key", row["sample_key"]) != row["sample_key"]:
        raise ValueError("media.json belongs to a different sample")
    if "audio_observed_intervals" not in media:
        raise ValueError("media.json must explicitly declare audio_observed_intervals")
    observed = merge_intervals(media["audio_observed_intervals"], 0, duration)
    alignment_path = directory / "alignment.json"
    alignment = read_json(alignment_path) if branch != "acoustic" else {}
    words = alignment.get("words", [])
    if branch != "acoustic" and (not isinstance(words, list)):
        raise ValueError("alignment.words must be a list")
    source = Path(row["media_path"])
    if not source.is_absolute():
        source = root / source
    hashes = {
        "source_media_sha256": sha256_file(source),
        "media_json_sha256": sha256_file(media_path),
        "official_text_sha256": hashlib.sha256(row["text"].encode("utf-8")).hexdigest(),
    }
    if branch != "acoustic":
        hashes["alignment_json_sha256"] = sha256_file(alignment_path)
    if branch != "text":
        hashes["audio16k_sha256"] = sha256_file(directory / "audio16k.wav")
    return (directory, media, duration, observed, alignment, words, hashes)


def load_waveform(directory: Path, media, duration):
    import soundfile as sf

    waveform, rate = sf.read(
        directory / "audio16k.wav", dtype="float32", always_2d=True
    )
    if rate != 16000 or waveform.shape[1] != 1 or (not np.isfinite(waveform).all()):
        raise ValueError("audio16k.wav must be finite mono PCM at 16000 Hz")
    waveform = waveform[:, 0]
    if abs(len(waveform) / rate - duration) > 2 / rate:
        raise ValueError(
            "audio16k.wav is not padded/truncated to the declared common-axis duration"
        )
    if media.get("audio_samples") is not None and int(media["audio_samples"]) != len(
        waveform
    ):
        raise ValueError("media.audio_samples disagrees with audio16k.wav")
    return waveform


def validate_arrays(arrays: dict, branch: str) -> None:
    features = arrays["features"]
    dim = 25 if branch == "acoustic" else 1024
    if (
        features.dtype != np.float32
        or features.ndim != 2
        or features.shape[1] != dim
        or (not np.isfinite(features).all())
    ):
        raise ValueError("Invalid float32 feature matrix")
    time_s = arrays["time_s"]
    if time_s.dtype != np.float64 or time_s.shape != (len(features), 2):
        raise ValueError("Invalid float64 target intervals")
    known = np.isfinite(time_s).all(axis=1)
    unknown = np.isnan(time_s).all(axis=1)
    if (
        not np.all(known | unknown)
        or np.any(time_s[known, 0] < 0)
        or np.any(time_s[known, 1] <= time_s[known, 0])
    ):
        raise ValueError(
            "Target intervals must be increasing finite pairs or two NaNs for unknown word times"
        )
    mask = arrays["valid_mask"]
    if (
        mask.dtype != np.uint8
        or mask.shape not in {(len(features),), features.shape}
        or (not np.isin(mask, [0, 1]).all())
    ):
        raise ValueError("Invalid explicit validity mask")
    for name, array in arrays.items():
        if array.dtype.hasobject:
            raise ValueError(f"Object arrays are prohibited: {name}")
        if (
            name.endswith("_time_s")
            or name.endswith("_bounds_s")
            or name in {"anchors_s", "observed_intervals_s"}
        ):
            if array.dtype != np.float64:
                raise ValueError(f"Time precision must be float64: {name}")


def valid_existing(directory: Path, branch: str, signature: str) -> bool:
    data_path, meta_path = (
        directory / f"{branch}.npz",
        directory / f"{branch}.meta.json",
    )
    if not data_path.is_file() or not meta_path.is_file():
        return False
    try:
        metadata = read_json(meta_path)
        if (
            metadata.get("status") != "complete"
            or metadata.get("input_signature") != signature
        ):
            return False
        if metadata.get("output_sha256") != sha256_file(data_path):
            return False
        with np.load(data_path, allow_pickle=False) as loaded:
            arrays = {name: loaded[name] for name in loaded.files}
        validate_arrays(arrays, branch)
        return True
    except (ValueError, KeyError, OSError, EOFError, BadZipFile, json.JSONDecodeError):
        return False


def save_result(directory: Path, branch: str, arrays: dict, metadata: dict):
    validate_arrays(arrays, branch)
    target = directory / f"{branch}.npz"
    descriptor, temporary = tempfile.mkstemp(
        prefix=target.name + ".", suffix=".tmp", dir=directory
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez_compressed(handle, **arrays)
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    metadata.update(
        {
            "status": "complete",
            "output_sha256": sha256_file(target),
            "arrays": {
                key: {"shape": list(value.shape), "dtype": str(value.dtype)}
                for key, value in arrays.items()
            },
        }
    )
    atomic_json(directory / f"{branch}.meta.json", metadata)


def encode_sample(row, data, branch, encoder):
    directory, media, duration, observed, alignment, words, _ = data
    if branch == "text":
        tokens, offsets = encoder.text_tokens(row["text"])
        features, valid, coverage, spans = character_pool(
            tokens, offsets, words, len(row["text"])
        )
        times = np.asarray(
            [word_time(w, duration) for w in words], dtype=np.float64
        ).reshape(-1, 2)
        arrays = {
            "features": features,
            "valid_mask": valid,
            "time_s": times,
            "alignment_known": np.isfinite(times).all(axis=1).astype(np.uint8),
            "word_index": np.arange(len(words), dtype=np.int32),
            "char_bounds": spans,
            "character_coverage": coverage,
            "token_char_bounds": offsets,
        }
        detail = {
            "hidden_layer": "last",
            "pooling": "original_character_overlap_weighted_mean",
            "long_text": {
                "max_content_tokens": 510,
                "overlap_tokens": 64,
                "merge": "triangular_distance_to_window_boundary",
            },
            "context_scope": "whole_transcript_or_recorded_overlapping_windows",
            "words": [
                {
                    "original": w.get("original", w.get("text", w.get("word", ""))),
                    "char_start": w.get("char_start"),
                    "char_end": w.get("char_end"),
                }
                for w in words
            ],
        }
        return (arrays, detail)
    waveform = load_waveform(directory, media, duration)
    observed_end = float(observed[:, 1].max())
    waveform = waveform[: int(math.floor(observed_end * 16000 + 1e-07))]
    context_end = len(waveform) / 16000
    if branch == "audio":
        frame_features, anchors, cells, receptive = encoder.audio_frames(waveform)
        nodes, times = build_nodes(words, duration, alignment)
        features, valid, coverage = pool_features(
            frame_features, cells, times, np.isfinite(frame_features), observed
        )
        row_valid = valid.all(axis=1).astype(np.uint8)
        features[row_valid == 0] = 0
        arrays = {
            "features": features,
            "valid_mask": row_valid,
            "time_s": times,
            "coverage": coverage.min(axis=1),
            "word_index": np.asarray([n["word_index"] for n in nodes], dtype=np.int32),
            "node_type": np.asarray([n["node_type"] for n in nodes], dtype="U24"),
            "anchors_s": anchors,
            "allocation_bounds_s": cells,
            "receptive_field_bounds_s": receptive,
            "observed_intervals_s": observed,
        }
        observed_seconds = float(np.sum(observed[:, 1] - observed[:, 0]))
        detail = {
            "hidden_layers": [-4, -3, -2, -1],
            "layer_pooling": "equal_mean",
            "nodes": nodes,
            "allocation": "nonoverlapping_anchor_midpoint_cells_intersect_real_observations",
            "local_conv_receptive_field_samples": 400,
            "stride_samples": 320,
            "context_scope": "common_origin_to_last_observed_audio_sample",
            "context_time_s": [0.0, context_end],
            "context_contains_inserted_audio": observed_seconds
            < context_end - 1 / 16000,
            "context_warning": "Unobserved tail is excluded before encoding. Any internal filled gap remains in transformer context and is excluded from pooling.",
            "waveform_normalization": "checkpoint_feature_extractor_on_common_axis_waveform",
            "attention_mask_policy": "No internal-gap mask passed as a right-padding length mask",
        }
        return (arrays, detail)
    values, support = encoder.frames(waveform)
    anchors = support.mean(axis=1)
    cells = allocation_cells(anchors, 0.0, len(waveform) / 16000)
    frame_valid, voiced, analysis_observed = acoustic_validity(
        values, encoder.names, support, observed
    )
    count = int(math.ceil(duration / GRID_S - 1e-10))
    edges = np.minimum(np.arange(count + 1, dtype=np.float64) * GRID_S, duration)
    targets = np.column_stack((edges[:-1], edges[1:]))
    features, valid, coverage = pool_features(
        values, cells, targets, frame_valid, observed
    )
    voiced_fraction, voiced_valid, _ = pool_features(
        voiced[:, None].astype(np.float32), cells, targets, analysis_observed, observed
    )
    arrays = {
        "features": features,
        "valid_mask": valid,
        "time_s": targets,
        "coverage": coverage,
        "feature_names": np.asarray(encoder.names, dtype="U96"),
        "anchors_s": anchors,
        "allocation_bounds_s": cells,
        "analysis_support_bounds_s": support,
        "conservative_context_bounds_s": np.tile(
            [0.0, context_end], (len(values), 1)
        ).astype(np.float64),
        "observed_intervals_s": observed,
        "voiced_fraction": voiced_fraction[:, 0],
        "voiced_fraction_valid": voiced_valid[:, 0],
    }
    detail = {
        "grid_step_s": GRID_S,
        "grid_origin_s": 0.0,
        "pooling": "duration_mean_over_disjoint_midpoint_cells",
        "native_timing": "actual_openSMILE_output_index_start_end_preserved_separately_from_allocation",
        "analysis_support_bounds_semantics": "Legacy array name: openSMILE output-index bounds, not the full signal dependency of every feature",
        "context_warning": "Full utterance context is conservatively recorded because F0, smoothing and Viterbi exceed output-index support. Unobserved tail is removed before encoding; any internal filled gaps are disclosed.",
        "context_time_s": [0.0, context_end],
        "context_contains_inserted_audio": float(
            np.sum(observed[:, 1] - observed[:, 0])
        )
        < context_end - 1 / 16000,
        "undefined_value_policy": "zero_storage_plus_per_feature_mask; no fabricated zero pitch",
        "pitch_validity": "F0semitone>0; all sma3nz channels require voiced F0; output-index support must be observed",
        "waveform_normalization": "none; original PCM amplitude preserved",
    }
    return (arrays, detail)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--branch", choices=("text", "audio", "acoustic"), required=True
    )
    parser.add_argument("--ids", help="Comma-separated manifest sample_key values")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--revision", help="Explicit HF revision for the fixed text/audio checkpoint"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute even when a valid matching output exists",
    )
    args = parser.parse_args(argv)
    rows = load_manifest(args.manifest)
    if args.ids:
        selected = {value.strip() for value in args.ids.split(",") if value.strip()}
        missing = selected - {row["sample_key"] for row in rows}
        if missing:
            raise ValueError(
                f"Requested sample keys absent from manifest: {sorted(missing)}"
            )
        rows = [row for row in rows if row["sample_key"] in selected]
    args.output.mkdir(parents=True, exist_ok=True)
    revision = args.revision or os.environ.get(
        f"Q1_{args.branch.upper()}_REVISION", "main"
    )
    encoder = (
        AcousticEncoder()
        if args.branch == "acoustic"
        else HFEncoder(args.branch, args.device, revision, args.output)
    )
    code_hash = sha256_file(Path(__file__))
    failures = 0
    for row in rows:
        try:
            data = load_sample(row, args.data_root, args.output, args.branch)
            identity = {
                name: row[name]
                for name in (
                    "sample_key",
                    "sample_id",
                    "video_id",
                    "clip_id",
                    "media_path",
                )
            }
            signature = json_hash(
                {
                    "schema": SCHEMA_VERSION,
                    "branch": args.branch,
                    "identity": identity,
                    "input_hashes": data[-1],
                    "encoder": encoder.provenance,
                    "implementation_sha256": code_hash,
                }
            )
            if not args.force and valid_existing(data[0], args.branch, signature):
                print(f"SKIP {row['sample_key']} {args.branch}: validated", flush=True)
                continue
            arrays, detail = encode_sample(row, data, args.branch, encoder)
            metadata = {
                "schema_version": SCHEMA_VERSION,
                "branch": args.branch,
                **identity,
                "input_signature": signature,
                "input_hashes": data[-1],
                "encoder": encoder.provenance,
                "implementation_sha256": code_hash,
                "details": detail,
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "labels_used": False,
                "feature_dtype": "float32",
                "timestamp_dtype": "float64",
            }
            save_result(data[0], args.branch, arrays, metadata)
            print(
                f"OK {row['sample_key']} {args.branch} {arrays['features'].shape}",
                flush=True,
            )
        except Exception as exc:
            failures += 1
            print(
                f"ERROR {row['sample_key']} {args.branch}: {type(exc).__name__}: {exc}",
                file=sys.stderr,
                flush=True,
            )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
