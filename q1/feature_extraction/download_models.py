"""Fetch only selected public inference weights, pinning exact repository commits."""

import argparse
import json
import os
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download

MODELS = {
    "text": "FacebookAI/roberta-large",
    "audio": "microsoft/wavlm-large",
    "video": "MCG-NJU/videomae-base",
}
REVISIONS = {
    "text": "722cf37b1afa9454edce342e7895e588b6ff1d59",
    "audio": "c1423ed94bb01d80a3f5ce5bc39f6026a0f4828c",
    "video": "dc740ceda42fce44faed2ea03c6d447db72f6af9",
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--branches", required=True)
    p.add_argument("--cache", required=True)
    a = p.parse_args()
    Path(a.cache).mkdir(parents=True, exist_ok=True)
    api = HfApi()
    for branch in a.branches.split(","):
        repo = MODELS[branch]
        info = api.model_info(repo, revision=REVISIONS[branch], files_metadata=False)
        if info.sha != REVISIONS[branch]:
            raise RuntimeError("Pinned checkpoint resolution mismatch")
        names = [x.rfilename for x in info.siblings]
        weights = (
            ["model.safetensors"]
            if "model.safetensors" in names
            else ["pytorch_model.bin"]
        )
        if weights[0] not in names:
            raise RuntimeError(f"No verified standard weight file in {repo}")
        wanted = weights + [
            "config.json",
            "preprocessor_config.json",
            "tokenizer_config.json",
            "tokenizer.json",
            "vocab.json",
            "merges.txt",
            "special_tokens_map.json",
            "added_tokens.json",
        ]
        local = snapshot_download(
            repo_id=repo,
            revision=info.sha,
            cache_dir=a.cache,
            allow_patterns=wanted,
            max_workers=3,
        )
        result = {
            "branch": branch,
            "repo_id": repo,
            "revision": info.sha,
            "local_snapshot": local,
            "endpoint": api.endpoint,
            "weight_files": weights,
        }
        (Path(a.cache) / (branch + ".download.json")).write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
