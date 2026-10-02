"""Build a portable, verified 100-sample index without rewriting source metadata."""

import argparse, csv, hashlib, json, math
from collections import Counter
from pathlib import Path
import numpy as np

BRANCHES = {
    "text": ("text_audio", 1024),
    "audio": ("text_audio", 1024),
    "acoustic": ("acoustic_visual", 25),
    "face": ("acoustic_visual", 44),
    "video": ("acoustic_visual", 768),
}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def union_length(rows):
    total = 0.0
    end = -math.inf
    for a, b in sorted(rows):
        total += max(0.0, b - max(a, end))
        end = max(end, b)
    return total


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--features", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    a = p.parse_args()
    rows = [
        json.loads(l)
        for l in a.manifest.read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]
    assert len(rows) == 100 and len({r["sample_key"] for r in rows}) == 100
    result = []
    inventory = []
    statuses = Counter()
    reasons = Counter()
    branch_rows = Counter()
    branch_valid = Counter()
    numeric = []
    for row in rows:
        key = row["sample_key"]
        ta = a.features / "text_audio" / key
        av = a.features / "acoustic_visual" / key
        ma, mb = (read(ta / "media.json"), read(av / "media.json"))
        al = read(ta / "alignment.json")
        bl = read(av / "alignment.json")
        assert al["text"] == bl["text"] == row["text"] and al["words"] == bl["words"], (
            key + " alignment mismatch"
        )
        assert ma["source_sha256"] == mb["source_sha256"] == row["source_sha256"]
        for name in ["frame_pts.npy", "source_time.npz"]:
            if name == "frame_pts.npy":
                assert sha(ta / name) == sha(av / name), key + " PTS mismatch"
            else:
                with np.load(ta / name) as x, np.load(av / name) as y:
                    assert x.files == y.files and all(
                        (np.array_equal(x[k], y[k]) for k in x.files)
                    ), (key + " source clocks mismatch")
        for f in [
            "duration_s",
            "origin_pts_s",
            "audio_samples",
            "decode_frame_count",
            "audio_observed_intervals",
            "video_observed_intervals",
        ]:
            assert ma[f] == mb[f], key + " " + f + " mismatch"
        words = al["words"]
        w = len(words)
        valid_words = sum((bool(x["alignment_valid"]) for x in words))
        assert [x["text"] for x in words] == row["text"].split(), (
            key + " official words changed"
        )
        rec = {
            "sample_key": key,
            "official_sample_id": row["sample_id"],
            "video_id": row["video_id"],
            "clip_id": row["clip_id"],
            "source_sha256": row["source_sha256"],
            "duration_s": ma["duration_s"],
            "container_duration_s": ma["container_duration_s"],
            "audio_observed_s": union_length(ma["audio_observed_intervals"]),
            "video_observed_s": union_length(ma["video_observed_intervals"]),
            "words": w,
            "lexical_words": sum((x.get("status") != "nonlexical" for x in words)),
            "aligned_words": valid_words,
            "alignment_status": al["status"],
            "decoded_frames": ma["decode_frame_count"],
            "alignment_path": (ta / "alignment.json")
            .relative_to(a.features)
            .as_posix(),
            "media_path": (ta / "media.json").relative_to(a.features).as_posix(),
        }
        statuses[al["status"]] += 1
        reasons.update((x["status"] for x in words))
        for word in words:
            if word.get("numeric_hypotheses"):
                numeric.append(
                    {
                        "sample_key": key,
                        "word_id": word["word_id"],
                        "original": word["text"],
                        "reading": word.get("spoken_text"),
                        "start": word["start"],
                        "end": word["end"],
                        "status": word["status"],
                        "margin_nats": word["numeric_hypotheses"].get("margin_nats"),
                        "human_verified": word.get("human_verified", False),
                    }
                )
        for branch, (kind, dim) in BRANCHES.items():
            path = a.features / kind / key / (branch + ".npz")
            with np.load(path, allow_pickle=False) as z:
                feat = z["features"]
                mask = (
                    z["valid_mask"]
                    if branch in ["text", "audio", "acoustic"]
                    else z["valid"]
                )
                assert (
                    feat.dtype == np.float32
                    and feat.ndim == 2
                    and (feat.shape[1] == dim)
                    and np.isfinite(feat).all()
                ), str(path)
                assert len(mask) == len(feat), str(path)
                t = (
                    z["time_s"]
                    if branch in ["text", "audio", "acoustic"]
                    else z["times"]
                ).copy()
                assert t.shape == (len(feat), 2)
                if branch == "video":
                    absent = np.all(t == -1, axis=1)
                    assert not np.any(mask[absent]) and np.all(
                        z["coverage"][absent] == 0
                    ), (str(path) + " unknown video mask")
                    t[absent] = np.nan
                finite = np.isfinite(t).all(axis=1)
                assert np.all(finite | np.isnan(t).all(axis=1)), (
                    str(path) + " malformed time"
                )
                assert (
                    np.all(t[finite, 0] >= 0)
                    and np.all(t[finite, 1] > t[finite, 0])
                    and np.all(t[finite, 1] <= ma["duration_s"] + 1 / 16000 + 1e-07)
                )
                if branch in ["text", "audio", "video"]:
                    assert len(feat) >= w
                    expected = np.array(
                        [
                            (
                                [x["start"], x["end"]]
                                if x["alignment_valid"]
                                else [np.nan, np.nan]
                            )
                            for x in words
                        ]
                    )
                    assert np.allclose(
                        t[:w], expected, atol=1e-07, rtol=0, equal_nan=True
                    ), (str(path) + " word clock")
                valid = np.any(mask, axis=1) if mask.ndim == 2 else mask.astype(bool)
                rec[branch + "_rows"] = len(feat)
                rec[branch + "_valid_rows"] = int(valid.sum())
                rec[branch + "_dim"] = dim
                rec[branch + "_path"] = path.relative_to(a.features).as_posix()
                rec[branch + "_sha256"] = sha(path)
                branch_rows[branch] += len(feat)
                branch_valid[branch] += int(valid.sum())
                inventory.append(
                    {
                        "sample_key": key,
                        "branch": branch,
                        "path": rec[branch + "_path"],
                        "sha256": rec[branch + "_sha256"],
                        "arrays": {
                            k: {"shape": list(z[k].shape), "dtype": str(z[k].dtype)}
                            for k in z.files
                        },
                    }
                )
        assert (
            rec["audio_rows"] == rec["video_rows"]
            and rec["acoustic_rows"] == rec["face_rows"]
        )
        face = read(av / "face.json")
        rec["face_valid_frames"] = face["valid_frames"]
        rec["face_frame_fraction"] = face["valid_frames"] / face["frames"]
        rec["acoustic_context_has_inserted_audio"] = read(av / "acoustic.meta.json")[
            "details"
        ]["context_contains_inserted_audio"]
        rec["audio_context_has_inserted_audio"] = read(ta / "audio.meta.json")[
            "details"
        ]["context_contains_inserted_audio"]
        result.append(rec)
    summary = {
        "samples": len(result),
        "video_groups": len({r["video_id"] for r in rows}),
        "official_words": sum((r["words"] for r in result)),
        "lexical_words": sum((r["lexical_words"] for r in result)),
        "aligned_words": sum((r["aligned_words"] for r in result)),
        "alignment_status": dict(statuses),
        "word_status": dict(reasons),
        "branch_rows": dict(branch_rows),
        "branch_valid_rows": dict(branch_valid),
        "common_duration_s": sum((r["duration_s"] for r in result)),
        "container_duration_s": sum((r["container_duration_s"] for r in result)),
        "audio_observed_s": sum((r["audio_observed_s"] for r in result)),
        "video_frames": sum((r["decoded_frames"] for r in result)),
        "face_valid_frames": sum((r["face_valid_frames"] for r in result)),
        "all_face_missing": [
            r["sample_key"] for r in result if not r["face_valid_frames"]
        ],
        "word_alignment_none": [
            r["sample_key"] for r in result if not r["aligned_words"]
        ],
        "all_cross_machine_clock_and_word_checks": True,
        "sample_count_preserved": True,
        "original_word_inventory_preserved": True,
        "context_contains_inserted_audio_samples": [
            r["sample_key"]
            for r in result
            if r["audio_context_has_inserted_audio"]
            or r["acoustic_context_has_inserted_audio"]
        ],
        "numeric_words": numeric,
        "index_builder_sha256": sha(__file__),
        "manifest_sha256": sha(a.manifest),
    }
    with (a.features / "samples_index.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as f:
        writer = csv.DictWriter(f, fieldnames=list(result[0]))
        writer.writeheader()
        writer.writerows(result)
    for name, value in [("summary.json", summary), ("array_inventory.json", inventory)]:
        (a.features / name).write_text(
            json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
