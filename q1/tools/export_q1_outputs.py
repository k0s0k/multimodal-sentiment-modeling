"""Export exact branch outputs only after full structural validation."""

import argparse, hashlib, json, tarfile, io
from pathlib import Path

COMMON = ["media.json", "source_time.npz", "frame_pts.npy", "alignment.json"]
FILES = {
    "text_audio": ["text.npz", "text.meta.json", "audio.npz", "audio.meta.json"],
    "acoustic_visual": [
        "acoustic.npz",
        "acoustic.meta.json",
        "face.npz",
        "face_frames.npz",
        "face.json",
        "face_status.json",
        "video.npz",
        "video_timing.npz",
        "video.json",
        "video_status.json",
    ],
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--role", choices=FILES, required=True)
    p.add_argument("--validation", type=Path, required=True)
    p.add_argument("--destination", type=Path, required=True)
    a = p.parse_args()
    report = json.loads(a.validation.read_text())
    assert (
        report["selected_rows"] == 100
        and report["requested_structural_validation_passed"]
    )
    rows = [json.loads(l) for l in a.manifest.read_text().splitlines() if l.strip()]
    assert len(rows) == 100
    files = []
    for r in rows:
        for name in COMMON + FILES[a.role]:
            path = a.output / r["sample_key"] / name
            assert path.is_file(), path
            files.append(path)
    files += sorted(a.output.glob("*.model.lock.json"))
    for name in [
        "alignment_model_manifest.json",
        "alignment_refinement_model_manifest.json",
        "transfer_provenance.json",
        "video_model_manifest.json",
        "final_alignment_summary.json",
    ]:
        if (a.output / name).is_file():
            files.append(a.output / name)
    manifest = {
        "role": a.role,
        "samples": 100,
        "validation_sha256": hashlib.sha256(a.validation.read_bytes()).hexdigest(),
        "files": [
            {
                "path": f.relative_to(a.output).as_posix(),
                "bytes": f.stat().st_size,
                "sha256": hashlib.sha256(f.read_bytes()).hexdigest(),
            }
            for f in files
        ],
    }
    a.destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = a.destination.with_suffix(".tmp")
    with tarfile.open(tmp, "w:gz") as t:
        for f in files:
            t.add(f, arcname=f.relative_to(a.output), recursive=False)
        b = json.dumps(manifest, indent=2).encode()
        info = tarfile.TarInfo("export_manifest.json")
        info.size = len(b)
        t.addfile(info, io.BytesIO(b))
    tmp.replace(a.destination)
    print(
        json.dumps(
            {
                "role": a.role,
                "files": len(files),
                "archive_bytes": a.destination.stat().st_size,
                "archive_sha256": hashlib.sha256(
                    a.destination.read_bytes()
                ).hexdigest(),
            }
        )
    )


if __name__ == "__main__":
    main()
