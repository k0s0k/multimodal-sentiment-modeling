"""Decode real presentation timestamps and construct a common, masked clock."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time
from fractions import Fraction
import numpy as np

SCHEMA = 3


def video_merge_tolerance(time_base):
    """Allow two native timestamp ticks, but retain genuinely missing frames."""
    return max(1e-05, 2 * float(Fraction(time_base)))


def run(command):
    return subprocess.run(
        command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )


def merge_intervals(rows, tolerance=1e-06):
    output = []
    for s, e in sorted(rows):
        if e <= s:
            continue
        if output and s <= output[-1][1] + tolerance:
            output[-1][1] = max(output[-1][1], e)
        else:
            output.append([float(s), float(e)])
    return output


def probe(path):
    fields = "stream=index,codec_type,time_base,start_time,duration,sample_rate,channels,width,height,avg_frame_rate:format=duration,start_time:frame=media_type,stream_index,best_effort_timestamp_time,pts_time,pkt_duration_time,duration_time,nb_samples"
    result = run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_streams",
            "-show_format",
            "-show_frames",
            "-show_entries",
            fields,
            "-of",
            "json",
            str(path),
        ]
    )
    return json.loads(result.stdout)


def number(row, *keys):
    for key in keys:
        try:
            value = float(row[key])
            if np.isfinite(value):
                return value
        except (KeyError, ValueError, TypeError):
            pass
    return None


def process(row, data_root, output, force=False):
    folder = output / row["sample_key"]
    folder.mkdir(parents=True, exist_ok=True)
    dest = folder / "media.json"
    if dest.exists() and (not force):
        old = json.loads(dest.read_text())
        if (
            old.get("status") == "complete"
            and old.get("schema_version") == SCHEMA
            and (old.get("source_sha256") == row["source_sha256"])
            and all(
                (
                    (folder / n).exists()
                    for n in ["audio16k.wav", "frame_pts.npy", "source_time.npz"]
                )
            )
        ):
            return {"sample_key": row["sample_key"], "status": "cached"}
    started = time.time()
    dest.write_text(
        json.dumps(
            {
                "schema_version": SCHEMA,
                "status": "processing",
                "sample_key": row["sample_key"],
            }
        ),
        encoding="utf-8",
    )
    source = data_root / row["media_path"]
    actual_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    if actual_hash != row["source_sha256"]:
        raise ValueError("Input SHA256 mismatch")
    raw = probe(source)
    videos = [s for s in raw["streams"] if s["codec_type"] == "video"]
    audios = [s for s in raw["streams"] if s["codec_type"] == "audio"]
    if not videos or not audios:
        raise ValueError("Expected audio and video streams")
    video, audio = (videos[0], audios[0])
    vs = [f for f in raw["frames"] if f.get("stream_index") == video["index"]]
    aus = [f for f in raw["frames"] if f.get("stream_index") == audio["index"]]
    vt = [number(f, "best_effort_timestamp_time", "pts_time") for f in vs]
    at = [number(f, "best_effort_timestamp_time", "pts_time") for f in aus]
    if not vt or not at or any((t is None for t in vt + at)):
        raise ValueError("Missing decoded frame PTS; cannot invent time")
    if np.any(np.diff(vt) <= 0):
        raise ValueError("Non-increasing video presentation timestamps")
    sr = int(audio["sample_rate"])
    audio_intervals = [[t, t + int(f["nb_samples"]) / sr] for t, f in zip(at, aus)]
    vstep = (
        float(np.median(np.diff(vt)))
        if len(vt) > 1
        else number(vs[0], "duration_time", "pkt_duration_time")
    )
    if not vstep or vstep <= 0:
        raise ValueError("Cannot determine video frame support")
    video_intervals = [
        [t, t + (number(f, "duration_time", "pkt_duration_time") or vstep)]
        for t, f in zip(vt, vs)
    ]
    origin = min(min(vt), min(at))
    end = max(
        max((e for _, e in video_intervals)), max((e for _, e in audio_intervals))
    )
    duration = end - origin
    audio_observed = merge_intervals(
        [[s - origin, e - origin] for s, e in audio_intervals], 2 / sr
    )
    video_tolerance = video_merge_tolerance(video["time_base"])
    video_observed = merge_intervals(
        [[s - origin, e - origin] for s, e in video_intervals], video_tolerance
    )
    np.save(
        folder / "frame_pts.npy",
        np.asarray(vt, dtype=np.float64) - origin,
        allow_pickle=False,
    )
    np.savez_compressed(
        folder / "source_time.npz",
        audio_pts=np.asarray(at, dtype=np.float64),
        audio_nb_samples=np.asarray([f["nb_samples"] for f in aus], dtype=np.int32),
        video_pts=np.asarray(vt, dtype=np.float64),
        video_duration=np.asarray(
            [e - s for s, e in video_intervals], dtype=np.float64
        ),
    )
    channels = int(audio["channels"])
    downmix = "pan=mono|c0=" + "+".join(
        (f"{1 / channels:.12g}*c{i}" for i in range(channels))
    )
    phase_ratio = None
    mono_policy = "channel_mean"
    if channels > 1:
        check = run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-i",
                str(source),
                "-map",
                "0:a:0",
                "-t",
                "5",
                "-ar",
                "16000",
                "-f",
                "f32le",
                "pipe:1",
            ]
        ).stdout
        values = np.frombuffer(check, dtype="<f4").reshape(-1, channels)
        if not values.size:
            raise ValueError("Empty audio in channel quality check")
        energies = np.mean(values.astype(np.float64) ** 2, axis=0)
        phase_ratio = float(
            np.mean(np.mean(values, axis=1).astype(np.float64) ** 2)
            / (max(energies) + 1e-20)
        )
        if max(energies) > 1e-10 and phase_ratio < 0.05:
            selected = int(np.argmax(energies))
            downmix = f"pan=mono|c0=c{selected}"
            mono_policy = f"channel_{selected}_phase_cancellation_fallback"
    samples = int(np.ceil(duration * 16000))
    filters = f"{downmix},asetpts=PTS-({origin:.12f})/TB,aresample=16000:async=1:first_pts=0:min_comp=0.0000625:min_hard_comp=0.0000625:max_soft_comp=0,apad,atrim=end_sample={samples}"
    wav_tmp = folder / "audio16k.tmp.wav"
    cmd = [
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-copyts",
        "-i",
        str(source),
        "-map",
        "0:a:0",
        "-vn",
        "-af",
        filters,
        "-ar",
        "16000",
        "-ac",
        "1",
        "-c:a",
        "pcm_f32le",
        str(wav_tmp),
    ]
    run(cmd)
    import soundfile as sf

    info = sf.info(wav_tmp)
    if info.samplerate != 16000 or info.channels != 1 or abs(info.frames - samples) > 1:
        raise ValueError("Audio output clock/shape mismatch")
    wav_tmp.replace(folder / "audio16k.wav")
    warnings = []
    if len(audio_observed) > 1:
        warnings.append("audio_pts_discontinuity")
    if len(video_observed) > 1:
        warnings.append("video_support_gap")
    if mono_policy != "channel_mean":
        warnings.append("phase_cancellation_fallback")
    if len(vt) > 1 and np.max(np.diff(vt)) - np.min(np.diff(vt)) > 0.002:
        warnings.append("variable_video_timestamps")
    record = {
        "schema_version": SCHEMA,
        "status": "complete",
        "sample_key": row["sample_key"],
        "sample_id": row["sample_id"],
        "source_sha256": actual_hash,
        "duration_s": duration,
        "container_duration_s": number(raw.get("format", {}), "duration"),
        "origin_pts_s": origin,
        "audio_original_start_s": at[0],
        "video_original_start_s": vt[0],
        "audio_rate_hz": 16000,
        "audio_samples": info.frames,
        "audio_source_rate_hz": sr,
        "audio_source_channels": channels,
        "audio_observed_intervals": audio_observed,
        "video_observed_intervals": video_observed,
        "decode_frame_count": len(vt),
        "video_interval_merge_tolerance_s": video_tolerance,
        "width": video["width"],
        "height": video["height"],
        "video_time_base": video["time_base"],
        "audio_time_base": audio["time_base"],
        "audio_clock": "common_origin_with_pts_gap_fill_and_observation_mask",
        "edit_list_policy": "ffmpeg_demuxer_default_once",
        "mono_policy": mono_policy,
        "phase_energy_ratio_first_5s": phase_ratio,
        "ffmpeg_command": cmd,
        "warnings": warnings,
        "elapsed_s": time.time() - started,
        "ffmpeg_version": run(["ffmpeg", "-version"]).stdout.decode().splitlines()[0],
    }
    temp = dest.with_suffix(".tmp")
    temp.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(dest)
    return {
        "sample_key": row["sample_key"],
        "status": "complete",
        "duration_s": duration,
        "frames": len(vt),
        "warnings": warnings,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--ids", default="")
    p.add_argument("--force", action="store_true")
    a = p.parse_args()
    ids = set(a.ids.split(",")) if a.ids else None
    rows = [
        json.loads(s)
        for s in Path(a.manifest).read_text(encoding="utf-8").splitlines()
        if s.strip()
    ]
    failed = 0
    for row in rows:
        if ids and row["sample_key"] not in ids:
            continue
        try:
            result = process(row, Path(a.data_root), Path(a.output), a.force)
        except Exception as exc:
            failed += 1
            result = {
                "sample_key": row["sample_key"],
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            folder = Path(a.output) / row["sample_key"]
            folder.mkdir(parents=True, exist_ok=True)
            (folder / "media.error.json").write_text(
                json.dumps(result), encoding="utf-8"
            )
        print(json.dumps(result), flush=True)
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
