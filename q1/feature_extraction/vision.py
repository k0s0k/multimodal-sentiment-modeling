"""Q1 OpenFace and frozen VideoMAE extraction on the prepared common time axis.

Face needs only numpy and an installed OpenFace FeatureExtraction executable.
Video additionally needs torch, transformers, pillow and av. No model is loaded
at module import. Outputs are numeric NPZ + JSON; source media remain read-only.
"""

from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
import uuid
import numpy as np

AU_R = [1, 2, 4, 5, 6, 7, 9, 10, 12, 14, 15, 17, 20, 23, 25, 26, 45]
AU_C = [1, 2, 4, 5, 6, 7, 9, 10, 12, 14, 15, 17, 20, 23, 25, 26, 28, 45]
FACE_FIELDS = (
    [f"AU{x:02d}_r" for x in AU_R]
    + [f"AU{x:02d}_c" for x in AU_C]
    + ["pose_Rx", "pose_Ry", "pose_Rz"]
    + [f"gaze_{eye}_{axis}" for eye in [0, 1] for axis in ["x", "y", "z"]]
)
VIDEO_MODEL = "MCG-NJU/videomae-base"
VIDEO_REVISION = "dc740ceda42fce44faed2ea03c6d447db72f6af9"


def write_json(path: Path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    temp.replace(path)


def save_npz(path: Path, **arrays):
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temp.replace(path)


def file_hash(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def manifest_rows(path: Path):
    if path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
    else:
        text = path.read_text(encoding="utf-8-sig")
        rows = (
            json.loads(text)
            if text.lstrip().startswith("[")
            else [json.loads(x) for x in text.splitlines() if x.strip()]
        )
    keys = [r["sample_key"] for r in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate sample_key in manifest")
    if any(
        (
            not k or Path(k).name != k or k in [".", ".."] or ("/" in k) or ("\\" in k)
            for k in keys
        )
    ):
        raise ValueError("Every sample_key must be one safe directory name")
    return rows


def load_prepared(row, output: Path):
    key = row["sample_key"]
    if Path(key).name != key or key in [".", ".."]:
        raise ValueError("sample_key must be a single safe directory name")
    folder = output / key
    media = json.loads((folder / "media.json").read_text(encoding="utf-8"))
    if media.get("status") not in [None, "ok", "complete", "completed", "success"]:
        raise ValueError(f"Media preparation has not succeeded: {media.get('status')}")
    pts = np.load(folder / "frame_pts.npy", allow_pickle=False).astype(np.float64)
    duration = float(
        media["duration_s"] if "duration_s" in media else row["duration_s"]
    )
    if media.get("sample_key", key) != key:
        raise ValueError("media.json sample_key disagrees with the manifest")
    if row.get("source_sha256") and media.get("source_sha256") != row["source_sha256"]:
        raise ValueError("Prepared source SHA256 disagrees with the manifest")
    if (
        pts.ndim != 1
        or not len(pts)
        or (not np.isfinite(pts).all())
        or np.any(np.diff(pts) <= 0)
    ):
        raise ValueError(
            "frame_pts must be finite, nonempty and strictly increasing presentation timestamps"
        )
    if not np.isfinite(duration) or pts[0] < -1e-05 or duration <= pts[-1]:
        raise ValueError("Frame PTS do not lie within prepared common duration")
    if media.get("decode_frame_count", len(pts)) != len(pts):
        raise ValueError("media.json decoded frame count disagrees with frame_pts.npy")
    return (folder, media, pts, duration)


def frame_cells(pts, duration):
    """Midpoint allocation, restricted to the actually observed video extent."""
    pts = np.asarray(pts, dtype=np.float64)
    delta = float(np.median(np.diff(pts))) if len(pts) > 1 else duration - pts[0]
    boundaries = np.r_[pts[0], (pts[:-1] + pts[1:]) / 2, min(duration, pts[-1] + delta)]
    return np.column_stack([boundaries[:-1], boundaries[1:]])


def grid_cells(duration, step=0.1):
    starts = np.arange(int(math.ceil(duration / step - 1e-10)), dtype=np.float64) * step
    return np.column_stack([starts, np.minimum(starts + step, duration)])


def pool_intervals(values, support, targets, valid=None):
    """Duration-only means and union coverage for nonoverlapping support cells."""
    values = np.asarray(values, dtype=np.float32)
    support = np.asarray(support, dtype=np.float64).reshape(-1, 2)
    targets = np.asarray(targets, dtype=np.float64).reshape(-1, 2)
    valid = (
        np.ones(len(values), dtype=bool)
        if valid is None
        else np.asarray(valid, dtype=bool)
    )
    if values.ndim != 2 or len(values) != len(support) or valid.shape != (len(values),):
        raise ValueError("Values, support and validity have inconsistent shapes")
    if not np.isfinite(values[valid]).all() or not np.isfinite(support).all():
        raise ValueError("Valid feature values and all support times must be finite")
    if np.any(support[:, 1] <= support[:, 0]) or np.any(
        support[1:, 0] < support[:-1, 1] - 1e-08
    ):
        raise ValueError("Support must be sorted, positive-duration and nonoverlapping")
    result = np.zeros((len(targets), values.shape[1]), dtype=np.float32)
    coverage = np.zeros(len(targets), dtype=np.float32)
    offsets, indices, weights = ([0], [], [])
    for i, (start, end) in enumerate(targets):
        if not (
            np.isfinite(start) and np.isfinite(end) and (start >= 0) and (end > start)
        ):
            offsets.append(len(indices))
            continue
        overlap = np.maximum(
            0, np.minimum(end, support[:, 1]) - np.maximum(start, support[:, 0])
        )
        overlap *= valid
        idx = np.flatnonzero(overlap > 0)
        denom = overlap[idx].sum()
        if denom > 0:
            w = (overlap[idx] / denom).astype(np.float32)
            result[i] = (values[idx] * w[:, None]).sum(axis=0)
            coverage[i] = min(1.0, denom / (end - start))
            indices.extend(idx.tolist())
            weights.extend(w.tolist())
        offsets.append(len(indices))
    return (
        result,
        coverage,
        np.asarray(offsets, np.int64),
        np.asarray(indices, np.int32),
        np.asarray(weights, np.float32),
    )


def unwrap_valid_runs(values, valid):
    out = values.copy()
    positions = np.flatnonzero(valid)
    for run in np.split(positions, np.flatnonzero(np.diff(positions) > 1) + 1):
        if len(run):
            out[run, 35:38] = np.unwrap(out[run, 35:38], axis=0)
    return out


def read_openface(csv_path, pts, threshold):
    with csv_path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, skipinitialspace=True)
        headers = [x.strip() for x in reader.fieldnames or []]
        reader.fieldnames = headers
        missing = sorted(
            set(["frame", "success", "confidence"] + FACE_FIELDS) - set(headers)
        )
        if missing:
            raise ValueError(f"OpenFace output is missing required columns: {missing}")
        rows = list(reader)
    if len(rows) != len(pts):
        raise ValueError(
            f"OpenFace rows {len(rows)} != prepared decoded frames {len(pts)}; cannot assume frame correspondence"
        )
    frames = np.asarray([float(r["frame"]) for r in rows], np.float64)
    if not np.array_equal(frames, np.arange(1, len(pts) + 1)):
        raise ValueError(
            "OpenFace frame sequence must be exactly 1..N without duplicates"
        )
    values = np.asarray(
        [[float(r[k]) for k in FACE_FIELDS] for r in rows], dtype=np.float32
    )
    confidence = np.asarray([float(r["confidence"]) for r in rows], np.float32)
    success = np.asarray([float(r["success"]) == 1 for r in rows])
    finite = np.isfinite(values).all(axis=1) & np.isfinite(confidence)
    valid = success & (confidence >= threshold) & (confidence <= 1.0001) & finite
    valid &= np.all((values[:, :17] >= -0.0001) & (values[:, :17] <= 5.0001), axis=1)
    valid &= np.all(np.isin(values[:, 17:35], [0, 1]), axis=1)
    boxes = np.full((len(rows), 4), -1, dtype=np.float32)
    xs, ys = ([f"x_{j}" for j in range(68)], [f"y_{j}" for j in range(68)])
    if set(xs + ys).issubset(headers):
        for i, r in enumerate(rows):
            x = np.asarray([float(r[k]) for k in xs])
            y = np.asarray([float(r[k]) for k in ys])
            if valid[i] and np.isfinite(x).all() and np.isfinite(y).all():
                boxes[i] = [x.min(), y.min(), x.max(), y.max()]
    values[~finite] = 0
    confidence = np.nan_to_num(confidence, nan=0.0, posinf=0.0, neginf=0.0)
    return (values, valid, confidence, success, boxes)


def extract_face(row, args):
    folder, media, pts, duration = load_prepared(row, args.output)
    executable = args.openface_bin or os.environ.get("OPENFACE_BIN")
    if not executable:
        raise RuntimeError(
            "OpenFace is required: set OPENFACE_BIN or --openface-bin; no replacement model is used"
        )
    resolved = (
        shutil.which(executable)
        if not Path(executable).is_file()
        else str(Path(executable).resolve())
    )
    if not resolved:
        raise FileNotFoundError(f"OpenFace executable not found: {executable}")
    source = (args.data_root / row["media_path"]).resolve()
    native = folder / "openface_native" / ("attempt-" + uuid.uuid4().hex)
    native.mkdir(parents=True, exist_ok=False)
    raw_csv = native / "features.csv"
    command = [
        resolved,
        "-f",
        str(source),
        "-out_dir",
        str(native.resolve()),
        "-of",
        "features.csv",
        "-2Dfp",
        "-pose",
        "-aus",
        "-gaze",
    ]
    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    with (native / "run.log").open("w", encoding="utf-8") as log:
        process = subprocess.run(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
            cwd=str(Path(resolved).resolve().parent),
            timeout=args.timeout,
        )
    if process.returncode:
        raise RuntimeError(
            f"OpenFace exit {process.returncode}; see {native / 'run.log'}"
        )
    if not raw_csv.exists():
        raise FileNotFoundError(f"OpenFace did not produce {raw_csv}; see run.log")
    values, valid, confidence, success, boxes = read_openface(
        raw_csv, pts, args.face_confidence
    )
    allocation, grid = (frame_cells(pts, duration), grid_cells(duration))
    observed = observed_intervals(media, pts, duration)
    support, source_map = ([], [])
    for index, (start, end) in enumerate(allocation):
        for a, b in observed:
            lo, hi = (max(start, a), min(end, b))
            if hi > lo:
                support.append([lo, hi])
                source_map.append(index)
    support = np.asarray(support, np.float64).reshape(-1, 2)
    source_map = np.asarray(source_map, np.int32)
    values = unwrap_valid_runs(values, valid)
    features, coverage, offsets, indices, weights = pool_intervals(
        values[source_map], support, grid, valid[source_map]
    )
    indices = source_map[indices]
    gaze_norm = np.stack(
        [
            np.linalg.norm(features[:, 38:41], axis=1),
            np.linalg.norm(features[:, 41:44], axis=1),
        ],
        axis=1,
    )
    save_npz(
        folder / "face.npz",
        features=features,
        times=grid,
        valid=(coverage > 0).astype(np.uint8),
        valid_time_fraction=coverage,
        gaze_mean_norm=gaze_norm.astype(np.float32),
        feature_names=np.asarray(FACE_FIELDS),
        source_offsets=offsets,
        source_indices=indices,
        source_weights=weights,
    )
    save_npz(
        folder / "face_frames.npz",
        times=pts,
        allocation_support=allocation,
        observed_support=support,
        support_frame_indices=source_map,
        valid=valid.astype(np.uint8),
        detection_success=success.astype(np.uint8),
        confidence=confidence,
        bbox_xyxy=boxes,
    )
    info = {
        "schema_version": 1,
        "branch": "face",
        "status": "complete",
        "sample_key": row["sample_key"],
        "dimensions": 44,
        "fields": FACE_FIELDS,
        "dtype": "float32",
        "time_dtype": "float64",
        "grid_step_s": 0.1,
        "source_csv": str(raw_csv),
        "source_csv_sha256": file_hash(raw_csv),
        "command": command,
        "frame_mapping": "OpenFace one-based frame -> prepared presentation PTS at frame-1; counts and sequence checked",
        "validity": f"success == 1 and confidence >= {args.face_confidence} and finite/in-range features",
        "au_presence_pooling": "observed valid duration fraction; confidence is a validity gate, never a weight",
        "continuous_pooling": "valid duration mean; pose angles unwrapped within uninterrupted valid runs",
        "subject_rule": "OpenFace FeatureExtraction single tracked face; multi-person identity is not independently verified",
        "valid_frames": int(valid.sum()),
        "frames": len(pts),
        "grid_cells": len(grid),
        "all_face_missing": bool(not valid.any()),
        "quality_warning": "no_reliable_face" if not valid.any() else None,
    }
    write_json(folder / "face.json", info)
    return info


def run_signature(row, args):
    folder = args.output / row["sample_key"]
    source = (args.data_root / row["media_path"]).resolve()
    if not source.is_relative_to(args.data_root.resolve()):
        raise ValueError("media_path resolves outside --data-root")
    paths = [source, folder / "media.json", folder / "frame_pts.npy", Path(__file__)]
    if args.branch == "video":
        if not getattr(args, "shape_smoke", False):
            paths.append(folder / "alignment.json")
        if (folder / "face_frames.npz").exists():
            paths.append(folder / "face_frames.npz")
        if (folder / "face_status.json").exists():
            paths.append(folder / "face_status.json")
    else:
        executable = args.openface_bin or os.environ.get("OPENFACE_BIN", "")
        resolved = shutil.which(executable) if executable else None
        if resolved:
            paths.append(Path(resolved))
        elif executable and Path(executable).is_file():
            paths.append(Path(executable))
    hashes = {str(p): file_hash(p) for p in paths}
    if row.get("source_sha256") and hashes[str(source)] != row["source_sha256"]:
        raise ValueError("Input source SHA256 disagrees with the manifest")
    payload = {
        "sample_id": row.get("sample_id"),
        "source": str(source),
        "branch": args.branch,
        "file_sha256": hashes,
        "shape_smoke": getattr(args, "shape_smoke", False),
        "face_confidence": args.face_confidence,
        "model": args.model,
        "revision": args.revision,
        "crop": args.crop,
        "device": args.device,
        "batch_size": args.batch_size,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def load_video_model(args):
    import torch
    import transformers
    from transformers import VideoMAEImageProcessor, VideoMAEModel

    processor = VideoMAEImageProcessor.from_pretrained(
        args.model, revision=args.revision, local_files_only=args.local_files_only
    )
    model, loading = VideoMAEModel.from_pretrained(
        args.model,
        revision=args.revision,
        local_files_only=args.local_files_only,
        output_loading_info=True,
        use_safetensors=True,
    )
    expected = {
        "hidden_size": 768,
        "num_frames": 16,
        "tubelet_size": 2,
        "patch_size": 16,
        "image_size": 224,
        "num_channels": 3,
    }
    for key, value in expected.items():
        if getattr(model.config, key, None) != value:
            raise ValueError(
                f"Unexpected VideoMAE {key}: {getattr(model.config, key, None)} != {value}"
            )
    if (
        loading.get("missing_keys")
        or loading.get("mismatched_keys")
        or loading.get("error_msgs")
    ):
        raise RuntimeError(f"Encoder checkpoint did not load completely: {loading}")
    if args.device.startswith("cuda") and (not torch.cuda.is_available()):
        raise RuntimeError(
            "CUDA requested but unavailable; choose --device cpu explicitly to permit CPU execution"
        )
    model.requires_grad_(False).eval().to(args.device)
    return (
        model,
        processor,
        {
            "model": args.model,
            "requested_revision": args.revision,
            "resolved_revision": getattr(model.config, "_commit_hash", None),
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "config": model.config.to_dict(),
            "processor": processor.to_dict(),
            "loading_info": loading,
            "source": "https://huggingface.co/MCG-NJU/videomae-base",
            "pretraining": "official-org HF VideoMAE base, Kinetics-400 self-supervised pretraining; no task fine-tuning",
        },
    )


def nearest_frame_indices(pts, queries):
    right = np.searchsorted(pts, queries).clip(0, len(pts) - 1)
    left = (right - 1).clip(0, len(pts) - 1)
    return np.where(
        np.abs(pts[left] - queries) <= np.abs(pts[right] - queries), left, right
    ).astype(np.int32)


def observed_intervals(media, pts, duration):
    cells = frame_cells(pts, duration)
    intervals = media.get("video_observed_intervals") or [[cells[0, 0], cells[-1, 1]]]
    result = []
    for item in intervals:
        if isinstance(item, dict):
            a, b = (
                item.get("start", item.get("start_s")),
                item.get("end", item.get("end_s")),
            )
        else:
            a, b = item
        a, b = (max(0.0, float(a)), min(duration, float(b)))
        if b > a:
            result.append([a, b])
    result.sort()
    if any((result[i][0] < result[i - 1][1] - 1e-08 for i in range(1, len(result)))):
        raise ValueError(
            "Observed video intervals overlap; coverage would be counted twice"
        )
    if not result:
        raise ValueError("No observed video interval")
    return np.asarray(result, np.float64)


def video_window_plan(pts, duration, observed):
    records = []
    for index, start in enumerate(np.arange(0.0, duration, 0.4)):
        query = start + (np.arange(16, dtype=np.float64) + 0.5) / 10.0
        valid = np.any(
            (query[:, None] >= observed[:, 0]) & (query[:, None] < observed[:, 1]),
            axis=1,
        )
        indices = nearest_frame_indices(pts, query)
        records.append(
            {
                "window_index": index,
                "context_start_s": float(start),
                "context_end_s": float(start + 1.6),
                "requested_frame_times_s": query.tolist(),
                "source_frame_indices": indices.tolist(),
                "source_frame_pts_s": pts[indices].tolist(),
                "padding_mask": (~valid).astype(int).tolist(),
                "unique_real_frames": int(len(set(indices[valid].tolist()))),
            }
        )
    return records


def decode_selected_frames(source, selected, expected_pts, origin):
    import av

    selected = set(map(int, selected))
    frames = {}
    count = 0
    with av.open(str(source)) as container:
        stream = container.streams.video[0]
        for index, frame in enumerate(container.decode(stream)):
            if index >= len(expected_pts) or frame.pts is None:
                raise ValueError(
                    "Video decode has missing PTS or more frames than prepared frame_pts"
                )
            common_pts = float(frame.pts * frame.time_base) - origin
            if abs(common_pts - expected_pts[index]) > 0.0005:
                raise ValueError(
                    f"Decoder PTS disagreement at frame {index}: {common_pts} vs {expected_pts[index]}"
                )
            if index in selected:
                frames[index] = frame.to_ndarray(format="rgb24")
            count += 1
    if count != len(expected_pts) or set(frames) != selected:
        raise ValueError("Decoded frame inventory disagrees with prepared media")
    return frames


def choose_crop(record, frames, face, requested):
    idx = np.asarray(record["source_frame_indices"], np.int32)
    height, width = frames[int(idx[0])].shape[:2]
    full = [0, 0, width, height]
    if requested == "full":
        return (full, "full_frame_baseline_requested")
    if face is None:
        return (full, "full_frame_baseline_no_face_track")
    real = np.asarray(record["padding_mask"]) == 0
    valid = face["valid"][idx].astype(bool) & real
    bbox = face["bbox_xyxy"][idx]
    valid &= (bbox[:, 2] > bbox[:, 0]) & (bbox[:, 3] > bbox[:, 1])
    if valid.sum() < max(4, math.ceil(real.sum() / 2)):
        return (full, "full_frame_baseline_insufficient_face_track")
    boxes = bbox[valid]
    centers = (boxes[:, :2] + boxes[:, 2:]) / 2
    median_center = np.median(centers, axis=0)
    if np.max(np.linalg.norm(centers - median_center, axis=1)) > 0.15 * math.hypot(
        width, height
    ):
        return (full, "full_frame_baseline_unstable_track")
    b = np.median(boxes, axis=0)
    bw, bh = (b[2] - b[0], b[3] - b[1])
    if min(bw, bh) < 12:
        return (full, "full_frame_baseline_tiny_face")
    cx, cy = ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2 + 0.25 * bh)
    side = max(2 * bw, 2.4 * bh)
    crop = [
        max(0, int(math.floor(cx - side / 2))),
        max(0, int(math.floor(cy - side / 2))),
        min(width, int(math.ceil(cx + side / 2))),
        min(height, int(math.ceil(cy + side / 2))),
    ]
    if min(crop[2] - crop[0], crop[3] - crop[1]) < 24:
        return (full, "full_frame_baseline_invalid_crop")
    return (crop, "tracked_face_head_shoulders_heuristic")


def alignment_nodes(path, duration):
    alignment = json.loads(path.read_text(encoding="utf-8"))
    nodes = []
    for i, word in enumerate(alignment.get("words", [])):
        a, b = (
            word.get("start", word.get("start_s")),
            word.get("end", word.get("end_s")),
        )
        known = word.get("alignment_valid", True) and a is not None and (b is not None)
        if known:
            known = (
                np.isfinite([a, b]).all()
                and 0 <= float(a) < float(b) <= duration + 1e-05
            )
        nodes.append(
            {
                "node_id": "word:" + str(word.get("word_id", i)),
                "node_type": "word",
                "start": float(a) if known else -1.0,
                "end": min(float(b), duration) if known else -1.0,
                "alignment_valid": bool(known),
                "word_id": word.get("word_id", i),
            }
        )
    for i, gap in enumerate(alignment.get("gaps", [])):
        if isinstance(gap, dict):
            a, b = (
                gap.get("start", gap.get("start_s")),
                gap.get("end", gap.get("end_s")),
            )
        else:
            a, b = gap
        if a is None or b is None or (not np.isfinite([a, b]).all()):
            continue
        a, b = (max(0.0, float(a)), min(duration, float(b)))
        if b - a >= 0.3 - 1e-08:
            nodes.append(
                {
                    "node_id": "gap:" + str(i),
                    "node_type": "unassigned_gap_unknown",
                    "start": a,
                    "end": b,
                    "alignment_valid": True,
                }
            )
    if not nodes:
        raise ValueError("alignment.json contains no word or gap nodes")
    return nodes


def merge_tubelets(raw_values, records, observed):
    """Merge same global 200ms cell across overlapping windows before node pooling."""
    groups = {}
    for wi, record in enumerate(records):
        real = 1 - np.asarray(record["padding_mask"], dtype=float)
        for tubelet in range(8):
            fraction = real[tubelet * 2 : tubelet * 2 + 2].mean()
            if fraction <= 0:
                continue
            tick = record["window_index"] * 2 + tubelet
            center = (tubelet + 0.5) * 0.2
            weight = max(0.05, 1 - abs(center - 0.8) / 0.8) * fraction
            groups.setdefault(tick, []).append((wi * 8 + tubelet, weight))
    features, support, ticks = ([], [], [])
    offsets, indices, weights = ([0], [], [])
    flat = np.asarray(raw_values, np.float32).reshape(-1, 768)
    for tick, group in sorted(groups.items()):
        ids = np.asarray([x[0] for x in group], np.int32)
        w = np.asarray([x[1] for x in group], np.float32)
        w /= w.sum()
        merged = (flat[ids] * w[:, None]).sum(axis=0)
        for a, b in observed:
            lo, hi = (max(tick * 0.2, a), min((tick + 1) * 0.2, b))
            if hi <= lo:
                continue
            features.append(merged)
            support.append([lo, hi])
            ticks.append(tick)
            indices.extend(ids.tolist())
            weights.extend(w.tolist())
            offsets.append(len(indices))
    if not features:
        raise ValueError("No real observed tubelet could be constructed")
    return (
        np.asarray(features, np.float32),
        np.asarray(support, np.float64),
        np.asarray(ticks, np.int32),
        np.asarray(offsets, np.int64),
        np.asarray(indices, np.int32),
        np.asarray(weights, np.float32),
    )


def crop_provenance(records, merge_offsets, merge_indices, merge_weights):
    """Same convex weights as embeddings; a valid full-frame feature is not a face observation."""
    is_face = np.asarray(
        [r["crop_type"] == "tracked_face_head_shoulders_heuristic" for r in records],
        np.float32,
    )
    fractions = np.zeros((len(merge_offsets) - 1, 2), np.float32)
    for i, (a, b) in enumerate(zip(merge_offsets[:-1], merge_offsets[1:])):
        fraction = float(np.sum(is_face[merge_indices[a:b] // 8] * merge_weights[a:b]))
        fractions[i] = [
            min(1.0, max(0.0, fraction)),
            min(1.0, max(0.0, 1.0 - fraction)),
        ]
    return fractions


def extract_video(row, args, runtime):
    import torch
    from PIL import Image

    model, processor, model_info = runtime
    folder, media, pts, duration = load_prepared(row, args.output)
    smoke = getattr(args, "shape_smoke", False)
    nodes = None if smoke else alignment_nodes(folder / "alignment.json", duration)
    observed = observed_intervals(media, pts, duration)
    records = [
        r
        for r in video_window_plan(pts, duration, observed)
        if r["unique_real_frames"] > 0
    ]
    if not records:
        raise ValueError("No observed video frame at any requested sampling point")
    if smoke:
        records = records[:1]
    selected = {j for r in records for j in r["source_frame_indices"]}
    frames = decode_selected_frames(
        (args.data_root / row["media_path"]).resolve(),
        selected,
        pts,
        float(media["origin_pts_s"]),
    )
    face = None
    face_complete = False
    if (folder / "face_status.json").exists():
        face_complete = (
            json.loads((folder / "face_status.json").read_text(encoding="utf-8")).get(
                "status"
            )
            == "complete"
        )
    if face_complete and (folder / "face_frames.npz").exists():
        with np.load(folder / "face_frames.npz", allow_pickle=False) as f:
            face = {k: f[k] for k in ["times", "valid", "bbox_xyxy"]}
        if (
            face["times"].shape != pts.shape
            or not np.allclose(face["times"], pts, atol=1e-08, rtol=0)
            or face["valid"].shape != pts.shape
            or (face["bbox_xyxy"].shape != (len(pts), 4))
        ):
            raise ValueError("Face track uses a different PTS axis")
    for record in records:
        crop, typ = choose_crop(record, frames, face, args.crop)
        record["crop_xyxy"] = crop
        record["crop_type"] = typ
    values = []
    forward_shapes = []
    with torch.inference_mode():
        for begin in range(0, len(records), args.batch_size):
            inputs = []
            for record in records[begin : begin + args.batch_size]:
                x1, y1, x2, y2 = record["crop_xyxy"]
                images = [
                    Image.fromarray(frames[j][y1:y2, x1:x2])
                    for j in record["source_frame_indices"]
                ]
                inputs.append(processor(images, return_tensors="pt")["pixel_values"])
            batch = torch.cat(inputs, dim=0).to(args.device)
            hidden = model(pixel_values=batch, bool_masked_pos=None).last_hidden_state
            if tuple(hidden.shape[1:]) != (1568, 768):
                raise ValueError(
                    f"Expected 8x14x14 patch tokens without CLS; received {tuple(hidden.shape)}"
                )
            features = (
                hidden.reshape(len(inputs), 8, 14, 14, 768)
                .mean(dim=(2, 3))
                .float()
                .cpu()
                .numpy()
            )
            if not np.isfinite(features).all():
                raise ValueError("VideoMAE returned nonfinite representations")
            forward_shapes.append(
                {
                    "pixel_values": list(batch.shape),
                    "last_hidden_state": list(hidden.shape),
                    "spatial_mean_tubelets": list(features.shape),
                }
            )
            values.extend(features)
    if smoke:
        values = np.asarray(values, np.float32)
        info = {
            "schema_version": 1,
            "branch": "video_smoke",
            "status": "complete",
            "sample_key": row["sample_key"],
            "purpose": "One actual source-video window through the frozen encoder; not a completed Q1 feature artifact",
            "alignment_used": False,
            "word_features_created": False,
            "model": model_info,
            "device": args.device,
            "prepared_frame_count": len(pts),
            "decoded_frame_count_verified": len(pts),
            "selected_source_frames": len(selected),
            "windows": records,
            "forward_shapes": forward_shapes,
            "output_finite": bool(np.isfinite(values).all()),
            "output_mean": float(values.mean()),
            "output_std": float(values.std()),
            "output_min": float(values.min()),
            "output_max": float(values.max()),
        }
        if args.device.startswith("cuda"):
            device = torch.cuda.get_device_properties(args.device)
            info["gpu"] = {
                "name": device.name,
                "total_memory_bytes": device.total_memory,
            }
        write_json(folder / "video_smoke.json", info)
        return info
    merged, support, ticks, merge_offsets, merge_indices, merge_weights = (
        merge_tubelets(values, records, observed)
    )
    times = np.asarray([[n["start"], n["end"]] for n in nodes], np.float64)
    pooled, coverage, offsets, indices, weights = pool_intervals(merged, support, times)
    crop_fractions = crop_provenance(
        records, merge_offsets, merge_indices, merge_weights
    )
    node_crop_fractions, _, *_ = pool_intervals(crop_fractions, support, times)
    full_frame_contribution = (coverage > 0) & (node_crop_fractions[:, 1] > 1e-06)
    save_npz(
        folder / "video.npz",
        features=pooled,
        times=times,
        valid=(coverage > 0).astype(np.uint8),
        coverage=coverage,
        node_ids=np.asarray([n["node_id"] for n in nodes]),
        node_types=np.asarray([n["node_type"] for n in nodes]),
        alignment_valid=np.asarray([n["alignment_valid"] for n in nodes], np.uint8),
        face_crop_fraction=node_crop_fractions[:, 0],
        full_frame_fraction=node_crop_fractions[:, 1],
        has_full_frame_contribution=full_frame_contribution.astype(np.uint8),
        face_crop_only=((coverage > 0) & ~full_frame_contribution).astype(np.uint8),
        source_offsets=offsets,
        source_indices=indices,
        source_weights=weights,
    )
    save_npz(
        folder / "video_timing.npz",
        support=support,
        global_ticks=ticks,
        merge_offsets=merge_offsets,
        merge_raw_token_indices=merge_indices,
        merge_weights=merge_weights,
        face_crop_fraction=crop_fractions[:, 0],
        full_frame_fraction=crop_fractions[:, 1],
    )
    info = {
        "schema_version": 1,
        "branch": "video",
        "status": "complete",
        "sample_key": row["sample_key"],
        "model": model_info,
        "dimensions": 768,
        "feature_dtype": "float32",
        "time_dtype": "float64",
        "spatial_pooling": "mean of 14x14 patch tokens per two-frame tubelet; no CLS token",
        "window_sampling": "16 samples at cell centers, 10 Hz, 1.6s span, 0.4s hop; nearest actual PTS",
        "temporal_merge": "same global 200ms allocation cell: normalized window-centrality times real-frame-fraction; merge before node pooling",
        "context_warning": "Every tubelet can attend to the entire window; allocation interval is not its full receptive field",
        "boundary_padding": "nearest real frame; padding mask and unique real frame counts retained",
        "crop_policy": "one stable crop per window when face track passes explicit heuristics; otherwise named full-frame baseline",
        "valid_mask_semantics": "valid means time-aligned observed video, including full-frame baseline; it does not mean a face is observed",
        "crop_fraction_semantics": "face_crop_fraction and full_frame_fraction use the same normalized convex merge and interval-pooling weights as features; sum 1 for valid nodes, 0 otherwise",
        "full_frame_mask_semantics": "has_full_frame_contribution marks any non-negligible full-frame weight; face_crop_only excludes it, but does not establish speaker identity",
        "crop_types": {
            k: sum((x["crop_type"] == k for x in records))
            for k in sorted({x["crop_type"] for x in records})
        },
        "nodes": nodes,
        "windows": records,
        "merged_support_rows": len(support),
        "crop_quality_note": "Single-face tracker does not independently establish speaking-person identity; face crop is a recorded heuristic",
    }
    write_json(folder / "video.json", info)
    return info


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--branch", choices=["face", "video"], required=True)
    parser.add_argument("--ids", default="")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--openface-bin")
    parser.add_argument("--face-confidence", type=float, default=0.8)
    parser.add_argument("--timeout", type=int, default=7200)
    parser.add_argument("--model", default=VIDEO_MODEL)
    parser.add_argument("--revision", default=VIDEO_REVISION)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--crop", choices=["face", "full"], default="face")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--shape-smoke",
        action="store_true",
        help="Video only: run one real window for exactly one selected sample; no alignment or word features",
    )
    args = parser.parse_args(argv)
    if not 0 <= args.face_confidence <= 1 or args.batch_size < 1:
        parser.error("confidence must be in [0,1] and batch size positive")
    rows = manifest_rows(args.manifest)
    selected = set(args.ids.split(",")) - {""}
    if selected:
        absent = selected - {r["sample_key"] for r in rows}
        if absent:
            parser.error(f"Unknown sample keys: {sorted(absent)}")
        rows = [r for r in rows if r["sample_key"] in selected]
    if args.shape_smoke and (args.branch != "video" or len(rows) != 1):
        parser.error(
            "--shape-smoke requires --branch video and exactly one sample selected with --ids"
        )
    args.output.mkdir(parents=True, exist_ok=True)
    runtime = None
    failures = 0
    for row in rows:
        folder = args.output / row["sample_key"]
        folder.mkdir(parents=True, exist_ok=True)
        artifact_branch = "video_smoke" if args.shape_smoke else args.branch
        status_path = folder / f"{artifact_branch}_status.json"
        artifact_path = folder / (
            "video_smoke.json" if args.shape_smoke else f"{args.branch}.npz"
        )
        start = time.monotonic()
        signature = None
        try:
            signature = run_signature(row, args)
            if not args.overwrite and artifact_path.exists() and status_path.exists():
                prior = json.loads(status_path.read_text(encoding="utf-8"))
                if (
                    prior.get("status") == "complete"
                    and prior.get("run_signature") == signature
                ):
                    print(
                        json.dumps(
                            {
                                "sample_key": row["sample_key"],
                                "branch": artifact_branch,
                                "status": "skipped_complete",
                            }
                        ),
                        flush=True,
                    )
                    continue
            if args.branch == "face":
                result = extract_face(row, args)
            else:
                if runtime is None:
                    runtime = load_video_model(args)
                result = extract_video(row, args, runtime)
            status = {
                "sample_key": row["sample_key"],
                "branch": artifact_branch,
                "status": "complete",
                "seconds": time.monotonic() - start,
                "run_signature": signature,
            }
        except Exception as exc:
            failures += 1
            status = {
                "sample_key": row["sample_key"],
                "branch": artifact_branch,
                "status": "failed",
                "seconds": time.monotonic() - start,
                "run_signature": signature,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        write_json(status_path, status)
        print(
            json.dumps(
                {k: v for k, v in status.items() if k != "traceback"},
                ensure_ascii=False,
            ),
            flush=True,
        )
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
