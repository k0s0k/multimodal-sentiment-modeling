"""Retrieve exactly the audit/alignment checkpoints used in Q1, never latest."""

import argparse, hashlib, json
from pathlib import Path
from huggingface_hub import snapshot_download

MODELS = {
    "ctc": ("facebook/wav2vec2-base-960h", "22aad52d435eb6dbaf354bdad9b0da84ce7d6156"),
    "qwen_aligner": (
        "Qwen/Qwen3-ForcedAligner-0.6B",
        "c7cbfc2048c462b0d63a45797104fc9db3ad62b7",
    ),
    "whisper_tiny": (
        "Systran/faster-whisper-tiny",
        "d90ca5fe260221311c53c58e660288d3deb8d356",
    ),
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", default=",".join(MODELS))
    p.add_argument("--root", type=Path, default=Path("models"))
    a = p.parse_args()
    for key in a.models.split(","):
        repo, rev = MODELS[key]
        kw = (
            {"cache_dir": str(a.root / "hub")}
            if key == "ctc"
            else {"local_dir": str(a.root / key)}
        )
        target = Path(
            snapshot_download(
                repo,
                revision=rev,
                allow_patterns=["*.json", "*.txt", "*.safetensors", "*.bin"],
                max_workers=3,
                **kw
            )
        )
        files = {
            f.relative_to(target).as_posix(): hashlib.sha256(f.read_bytes()).hexdigest()
            for f in target.rglob("*")
            if f.is_file() and ".cache" not in f.parts
        }
        result = {"repo_id": repo, "revision": rev, "files_sha256": files}
        (a.root / (key + ".download.json")).write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )
        print(json.dumps({"model": key, "files": len(files)}), flush=True)


if __name__ == "__main__":
    main()
