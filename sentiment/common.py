"""Small reproducibility utilities shared by the execution controller."""

from pathlib import Path
import datetime, hashlib, json, os, random
import numpy as np

MODEL_KEYS = ("text", "audio", "vision", "mask", "structure")


def plain(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, Path):
        return str(x)
    raise TypeError(type(x).__name__)


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=plain, allow_nan=False),
        encoding="utf-8",
    )
    os.replace(tmp, path)


def append_jsonl(path, value):
    with Path(path).open("a", encoding="utf-8") as f:
        f.write(
            json.dumps(value, ensure_ascii=False, default=plain, allow_nan=False) + "\n"
        )


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def seed_everything(seed):
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")


def to_device(batch, device, model_only=False):
    import torch

    keys = MODEL_KEYS if model_only else batch.keys()
    return {
        k: (
            batch[k].to(device, non_blocking=True)
            if isinstance(batch[k], torch.Tensor)
            else batch[k]
        )
        for k in keys
        if k in batch
    }


def source_manifest(root):
    root = Path(root)
    return {
        str(p.relative_to(root)).replace("\\", "/"): sha256(p)
        for p in sorted((root / "sentiment").rglob("*.py"))
    }
