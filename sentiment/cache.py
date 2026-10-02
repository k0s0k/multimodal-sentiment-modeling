"""Frozen, mask-before-BERT cache CLI and training/evaluation dataset.

Examples (working directory Q2_workplace):
 python -m sentiment.cache --source aligned.pkl --split train --bank train-point --output CACHE
 python -m sentiment.cache --source aligned.pkl --split train --bank train-block --output CACHE
 python -m sentiment.cache --source aligned.pkl --split valid --subset-file configs/validation_split_v1.json --subset-name valid_tune --bank core --seed 31 --output CACHE

CPU unit tests inject an encoder; this module never downloads weights on import.
"""

from __future__ import annotations
import argparse
from collections import OrderedDict
import contextlib
import copy
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np
from .data import (
    load_official,
    subset_data,
    fit_scaler,
    apply_scaler,
    observed_masks,
    sanitize_tokens,
    build_structure,
    stable_seed,
    file_sha256,
    STUDENT_KEYS,
)
from .masking import (
    MASK_VERSION,
    make_masks,
    make_core_conditions,
    make_grid_conditions,
    make_training_condition,
)

BERT_REPO = "google-bert/bert-base-uncased"
BERT_REVISION = "86b5e0934494bd15c9632b12f734a8a67f723594"
CACHE_VERSION = 1


def _json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), "utf-8"
    )
    os.replace(tmp, path)


def _npy(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    with tmp.open("wb") as f:
        np.save(f, value, allow_pickle=False)
    os.replace(tmp, path)


def _read_json(path):
    return json.loads(Path(path).read_text("utf-8"))


def _canonical_hash(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def token_hash(tokens, encoder_metadata):
    a = sanitize_tokens(np.asarray(tokens)[None])[0]
    h = hashlib.sha256(_canonical_hash(encoder_metadata).encode())
    h.update(a.astype("<i8", copy=False).tobytes())
    return h.hexdigest()


class FrozenBertEncoder:
    """Pinned official BERT, final hidden state, explicit absolute positions, eval only.

    A custom local directory requires q2_checkpoint_provenance.json with repo,
    revision and weights_sha256; this prevents quietly substituting task weights.
    """

    def __init__(self, model_path=None, device="cuda:0", local_files_only=False):
        import torch
        import transformers
        from transformers import BertModel
        from transformers.utils.hub import cached_file

        source = model_path or BERT_REPO
        local = Path(source).is_dir()
        if local:
            weight = Path(source) / "model.safetensors"
            provenance = Path(source) / "q2_checkpoint_provenance.json"
            if not provenance.exists():
                raise ValueError(
                    "Local BERT requires q2_checkpoint_provenance.json: repo,revision,weights_sha256"
                )
            p = _read_json(provenance)
            if (
                p.get("repo") != BERT_REPO
                or p.get("revision") != BERT_REVISION
                or p.get("weights_sha256") != file_sha256(weight)
            ):
                raise ValueError(
                    "Local BERT provenance/hash does not match pinned official source"
                )
        elif source != BERT_REPO:
            raise ValueError("Only the pinned official BERT repository is accepted")
        self.model = (
            BertModel.from_pretrained(
                source,
                revision=BERT_REVISION,
                local_files_only=local_files_only,
                use_safetensors=True,
            )
            .to(device)
            .eval()
        )
        self.model.requires_grad_(False)
        if not local:
            if self.model.config._commit_hash != BERT_REVISION:
                raise ValueError("Resolved model revision is not pinned BERT revision")
            weight = Path(
                cached_file(
                    BERT_REPO,
                    "model.safetensors",
                    revision=BERT_REVISION,
                    local_files_only=True,
                )
            )
        if (
            self.model.config.hidden_size != 768
            or self.model.config.vocab_size != 30522
        ):
            raise ValueError("Unexpected BERT config")
        self.device = str(device)
        self.metadata = {
            "repo": BERT_REPO,
            "revision": BERT_REVISION,
            "weights_sha256": file_sha256(weight),
            "hidden_layer": "last_hidden_state",
            "position_ids": "fixed_0_to_49",
            "output_content_only": True,
            "compute_dtype": "float32",
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        }

    def __call__(self, tokens):
        import torch

        tokens = sanitize_tokens(tokens)
        content = (tokens[:, 1] == 1) & ~np.isin(tokens[:, 0], [0, 101, 102])
        output = np.zeros((len(tokens), 50, 768), dtype=np.float32)
        use = np.flatnonzero(content.any(1))
        if not len(use):
            return output
        t = torch.as_tensor(tokens[use], device=self.device, dtype=torch.long)
        with torch.inference_mode():
            h = self.model(
                input_ids=t[:, 0],
                attention_mask=t[:, 1],
                token_type_ids=t[:, 2],
                position_ids=torch.arange(50, device=self.device)[None].expand(
                    len(use), -1
                ),
                return_dict=True,
            ).last_hidden_state
        output[use] = h.float().cpu().numpy()
        output *= content[:, :, None]
        if not np.isfinite(output).all():
            raise FloatingPointError("Nonfinite frozen BERT output")
        return output


def _conditions(bank, views):
    if bank in ("train-point", "train-block"):
        family = bank.split("-", 1)[1]
        return make_core_conditions()[:1] + [
            make_training_condition(family, v) for v in range(views)
        ]
    if bank == "core":
        return make_core_conditions()
    if bank == "grid":
        return make_grid_conditions()
    if bank == "clean":
        return make_core_conditions()[:1]
    raise ValueError(f"Unknown bank {bank}")


def build_cache(
    data,
    root,
    bank,
    encoder,
    seed=31,
    views=12,
    batch_size=64,
    storage_dtype="float16",
    conditions=None,
    cache_name=None,
):
    """Build one complete bank, resume hashed feature shards; one writer per root.

    Encoder injection is for deterministic tests. Production CLI always uses pinned BERT.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    lock = root / ".cache_writer.lock"
    try:
        fd = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    except FileExistsError:
        raise RuntimeError(
            f"Cache writer lock exists; concurrent writers forbidden: {lock}"
        )
    os.write(fd, str(os.getpid()).encode())
    os.close(fd)
    try:
        return _build_cache(
            data,
            root,
            bank,
            encoder,
            seed,
            views,
            batch_size,
            storage_dtype,
            conditions,
            cache_name,
        )
    finally:
        lock.unlink()


def _build_cache(
    data,
    root,
    bank,
    encoder,
    seed,
    views,
    batch_size,
    storage_dtype,
    conditions,
    cache_name,
):
    if storage_dtype not in ("float16", "float32") or batch_size < 1 or views < 1:
        raise ValueError("invalid cache storage/batch/views")
    if bank.startswith("train-") and data["split"] != "train":
        raise ValueError("training banks may only be built on train")
    conds = conditions if conditions is not None else _conditions(bank, views)
    if (
        not conds
        or conds[0]["name"] != "clean"
        or len({x["name"] for x in conds}) != len(conds)
    ):
        raise ValueError("Unique conditions starting with clean are required")
    if any((not all((ch.isalnum() or ch in "_.-" for ch in c["name"])) for c in conds)):
        raise ValueError("Unsafe condition name")
    split = data["split"]
    if not split.replace("_", "").isalnum():
        raise ValueError("Unsafe split name")
    cache_name = cache_name or bank
    if not all((ch.isalnum() or ch in "_.-" for ch in cache_name)) or cache_name in (
        ".",
        "..",
    ):
        raise ValueError("Unsafe cache name")
    base = root / split / "base"
    base_meta = {
        "version": CACHE_VERSION,
        "split": split,
        "ids": data["ids"],
        "video_ids": data["video_ids"],
        "source": data.get("source"),
        "source_sha256": data.get("source_sha256"),
        "original_source_sha256": data.get("original_source_sha256"),
        "official_indices": data.get("official_indices"),
        "official_split": data.get("official_split", split),
        "labeled": "labels" in data,
        "n": len(data["ids"]),
    }
    if (base / "metadata.json").exists():
        old = _read_json(base / "metadata.json")
        if old != base_meta:
            raise ValueError(
                "Existing base differs in source/subset/order; use a new cache root"
            )
    else:
        for key in ("tokens", "audio", "vision", "labels", "targets"):
            if key in data:
                _npy(base / (key + ".npy"), data[key])
        _json(base / "metadata.json", base_meta)
    if split == "train":
        scaler = fit_scaler(data)
        sp = root / "scaler.json"
        if sp.exists() and _read_json(sp) != scaler:
            raise ValueError("Existing scaler does not match train data")
        _json(sp, scaler)
    store_dir = root / "text_store"
    store_path = store_dir / "index.json"
    signature = {
        **encoder.metadata,
        "storage_dtype": storage_dtype,
        "cache_version": CACHE_VERSION,
    }
    if store_path.exists():
        store = _read_json(store_path)
        if store["encoder"] != signature:
            raise ValueError("Global text cache encoder/dtype differs")
        for shard in store["shards"]:
            if file_sha256(store_dir / shard["file"]) != shard["sha256"]:
                raise ValueError(f"Corrupt text shard: {shard['file']}")
    else:
        store = {"encoder": signature, "keys": {}, "entries": [], "shards": []}
    needed, view_records = ({}, [])
    bank_dir = root / split / cache_name
    configuration = {
        "version": CACHE_VERSION,
        "base_metadata": base_meta,
        "bank": bank,
        "cache_name": cache_name,
        "conditions": conds,
        "seed": seed,
        "mask_version": MASK_VERSION,
        "encoder": signature,
    }
    fingerprint = _canonical_hash(configuration)
    if (bank_dir / "manifest.json").exists():
        completed = _read_json(bank_dir / "manifest.json")
        if completed.get("configuration_fingerprint") != fingerprint:
            raise ValueError(
                "Existing bank configuration/seed differs; choose a different --cache-name"
            )
        for view in completed["views"]:
            directory = bank_dir / "views" / view["name"]
            for kind in ("drop", "mask", "structure", "text_index"):
                if file_sha256(directory / (kind + ".npy")) != view[kind + "_sha256"]:
                    raise ValueError(f"Existing bank corrupt: {view['name']}/{kind}")
        return {**completed, "new_unique_text_inputs": 0, "reused_complete_bank": True}
    t0 = time.perf_counter()
    for c in conds:
        damaged = make_masks(
            data["tokens"], data["audio"], data["vision"], c, seed, data["ids"]
        )
        directory = bank_dir / "views" / c["name"]
        _npy(directory / "drop.npy", damaged["diagnostics"]["drop_mask"])
        _npy(directory / "mask.npy", damaged["mask"])
        _npy(directory / "structure.npy", damaged["structure"])
        keys = []
        for tokens in damaged["tokens"]:
            key = token_hash(tokens, signature)
            keys.append(key)
            if key not in store["keys"] and key not in needed:
                needed[key] = tokens.copy()
        _npy(directory / "text_keys.npy", np.asarray(keys, dtype="U64"))
        record_path = directory / "diagnostics.jsonl"
        tmp = record_path.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            for record in damaged["diagnostics"]["records"]:
                f.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        os.replace(tmp, record_path)
        view_records.append(
            {
                "name": c["name"],
                "condition": c,
                "drop_sha256": file_sha256(directory / "drop.npy"),
                "mask_sha256": file_sha256(directory / "mask.npy"),
                "structure_sha256": file_sha256(directory / "structure.npy"),
                "diagnostics_sha256": file_sha256(record_path),
            }
        )
    all_keys = list(needed)
    for chunk_start in range(0, len(all_keys), 256):
        chunk_keys = all_keys[chunk_start : chunk_start + 256]
        outputs = []
        for offset in range(0, len(chunk_keys), batch_size):
            selected = chunk_keys[offset : offset + batch_size]
            inputs = np.stack([needed[k] for k in selected])
            h = np.asarray(encoder(inputs), dtype=np.float32)
            if h.shape != (len(inputs), 50, 768) or not np.isfinite(h).all():
                raise ValueError("Encoder must return finite B,50,768")
            content = (inputs[:, 1] == 1) & ~np.isin(inputs[:, 0], [0, 101, 102])
            h *= content[:, :, None]
            outputs.append(h)
        features = np.concatenate(outputs).astype(storage_dtype)
        if not np.isfinite(features).all():
            raise ValueError("Cache precision overflow")
        shard_name = f"shard_{len(store['shards']):06d}.npy"
        _npy(store_dir / shard_name, features)
        for row, key in enumerate(chunk_keys):
            store["keys"][key] = len(store["entries"])
            store["entries"].append([shard_name, row])
        store["shards"].append(
            {
                "file": shard_name,
                "sha256": file_sha256(store_dir / shard_name),
                "n": len(chunk_keys),
            }
        )
        _json(store_path, store)
        print(
            f"Encoded new unique text {min(chunk_start + 256, len(all_keys))}/{len(all_keys)}",
            flush=True,
        )
    if not store_path.exists():
        _json(store_path, store)
    for view in view_records:
        directory = bank_dir / "views" / view["name"]
        keys = np.load(directory / "text_keys.npy", allow_pickle=False)
        indices = np.asarray([store["keys"][str(k)] for k in keys], np.int64)
        _npy(directory / "text_index.npy", indices)
        view["text_index_sha256"] = file_sha256(directory / "text_index.npy")
    manifest = {
        "version": CACHE_VERSION,
        "status": "complete",
        "split": split,
        "bank": bank,
        "cache_name": cache_name,
        "configuration_fingerprint": fingerprint,
        "n": len(data["ids"]),
        "mask_version": MASK_VERSION,
        "mask_seed": seed,
        "views": view_records,
        "encoder": signature,
        "base_metadata_sha256": file_sha256(base / "metadata.json"),
        "new_unique_text_inputs": len(all_keys),
        "elapsed_seconds": time.perf_counter() - t0,
        "training_selection": "stable_id_epoch_choose_training_views_only; clean probability inside each view",
        "student_no_oracle": "Only damaged mask and derived structure; clean/drop/diagnostics not student tensors",
    }
    _json(bank_dir / "manifest.json", manifest)
    return manifest


class CacheDataset:
    """Map-style torch DataLoader-compatible dataset. No torch import before item access.

    clean_index/index are local selected-subset indices. get_clean_batch is explicit;
    teacher tensors are never automatically attached to a student item.
    """

    def __init__(
        self,
        root,
        split="train",
        bank="train-block",
        view=None,
        seed=20260924,
        scaler=None,
    ):
        self.root, self.split, self.bank, self.seed = (
            Path(root),
            split,
            bank,
            int(seed),
        )
        self.directory = self.root / split / bank
        self.manifest = _read_json(self.directory / "manifest.json")
        if self.manifest.get("status") != "complete":
            raise ValueError("Incomplete cache bank")
        base = self.root / split / "base"
        if file_sha256(base / "metadata.json") != self.manifest["base_metadata_sha256"]:
            raise ValueError("Cache base metadata hash changed")
        self.metadata = _read_json(base / "metadata.json")
        self.base = {
            k: np.load(base / (k + ".npy"), mmap_mode="r", allow_pickle=False)
            for k in ("tokens", "audio", "vision", "labels", "targets")
            if (base / (k + ".npy")).exists()
        }
        self.names = [v["name"] for v in self.manifest["views"]]
        self.view = self.names.index(view) if isinstance(view, str) else view
        if self.view is None and (not self.manifest["bank"].startswith("train-")):
            raise ValueError("Evaluation requires explicit view, including clean")
        if self.view is not None and (not 0 <= int(self.view) < len(self.names)):
            raise ValueError("View index outside bank")
        self.view_arrays = []
        for record in self.manifest["views"]:
            directory = self.directory / "views" / record["name"]
            for kind in ("drop", "mask", "text_index", "structure"):
                if file_sha256(directory / (kind + ".npy")) != record[kind + "_sha256"]:
                    raise ValueError(f"Corrupt {record['name']}/{kind}")
            self.view_arrays.append(
                {
                    k: np.load(
                        directory / (k + ".npy"), mmap_mode="r", allow_pickle=False
                    )
                    for k in ("drop", "mask", "text_index", "structure")
                }
            )
        self.store = _read_json(self.root / "text_store" / "index.json")
        if self.store["encoder"] != self.manifest["encoder"]:
            raise ValueError("Text cache encoder differs from bank")
        self.scaler = (
            scaler if scaler is not None else _read_json(self.root / "scaler.json")
        )
        if self.scaler.get("fitted_split") != "train":
            raise ValueError("Only train-fitted scaler allowed")
        self._mean = {
            k: np.asarray(self.scaler[k]["mean"], np.float32)
            for k in ("audio", "vision")
        }
        self._std = {
            k: np.asarray(self.scaler[k]["std"], np.float32)
            for k in ("audio", "vision")
        }
        self.epoch = 0
        self._shards = OrderedDict()
        self._verified_shards = set()

    def __len__(self):
        return self.metadata["n"]

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def get_view_names(self):
        return list(self.names)

    def get_metadata(self):
        return {
            **self.metadata,
            "bank": self.bank,
            "views": self.get_view_names(),
            "encoder": self.manifest["encoder"],
        }

    def with_view(self, name):
        """Shallow view selection; shares validated mmap arrays and shard handles."""
        other = copy.copy(self)
        other.view = self.names.index(name) if isinstance(name, str) else int(name)
        if not 0 <= other.view < len(self.names):
            raise ValueError("View index outside bank")
        return other

    def close(self):
        """Release mmap handles after every shallow view has finished using them."""
        arrays = list(self.base.values()) + list(self._shards.values())
        arrays += [a for view in self.view_arrays for a in view.values()]
        for arr in arrays:
            mm = getattr(arr, "_mmap", None)
            if mm is not None and (not mm.closed):
                mm.close()
        self._shards.clear()

    def _feature(self, index):
        name, row = self.store["entries"][int(index)]
        if name not in self._shards:
            if name not in self._verified_shards:
                expected = next(
                    (s["sha256"] for s in self.store["shards"] if s["file"] == name)
                )
                if file_sha256(self.root / "text_store" / name) != expected:
                    raise ValueError(f"Corrupt text feature shard {name}")
                self._verified_shards.add(name)
            self._shards[name] = np.load(
                self.root / "text_store" / name, mmap_mode="r", allow_pickle=False
            )
        self._shards.move_to_end(name)
        while len(self._shards) > 8:
            self._shards.popitem(last=False)
        return np.array(self._shards[name][row], dtype=np.float32, copy=True)

    def _item(self, index, view_index):
        import torch

        v = self.view_arrays[view_index]
        drop = v["drop"][index]
        audio, vision = (
            np.array(self.base["audio"][index], copy=True),
            np.array(self.base["vision"][index], copy=True),
        )
        audio[drop[1]], vision[drop[2]] = (0.0, 0.0)
        mask = np.asarray(v["mask"][index], dtype=bool)
        audio = np.where(
            mask[1, :, None], (audio - self._mean["audio"]) / self._std["audio"], 0
        ).astype(np.float32)
        vision = np.where(
            mask[2, :, None], (vision - self._mean["vision"]) / self._std["vision"], 0
        ).astype(np.float32)
        item = {
            "text": torch.from_numpy(self._feature(v["text_index"][index])),
            "audio": torch.from_numpy(audio),
            "vision": torch.from_numpy(vision),
            "mask": torch.from_numpy(mask.copy()),
            "structure": torch.from_numpy(np.array(v["structure"][index], copy=True)),
            "index": index,
            "clean_index": index,
            "view_index": view_index,
            "ids": self.metadata["ids"][index],
            "video_ids": self.metadata["video_ids"][index],
        }
        if "labels" in self.base:
            item["labels"] = torch.tensor(
                int(self.base["labels"][index]), dtype=torch.long
            )
            item["targets"] = torch.tensor(
                float(self.base["targets"][index]), dtype=torch.float32
            )
        return item

    def __getitem__(self, index):
        v = self.view
        if v is None:
            candidates = [j for j, name in enumerate(self.names) if name != "clean"]
            v = candidates[
                stable_seed(self.seed, self.epoch, self.metadata["ids"][index])
                % len(candidates)
            ]
        return self._item(int(index), int(v))

    def get_clean_batch(self, indices):
        from torch.utils.data._utils.collate import default_collate

        return default_collate(
            [self._item(int(i), self.names.index("clean")) for i in indices]
        )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", required=True, type=Path)
    ap.add_argument("--split", required=True, choices=["train", "valid", "test"])
    ap.add_argument("--source-provenance", type=Path)
    ap.add_argument("--subset-file", type=Path)
    ap.add_argument("--subset-name")
    ap.add_argument(
        "--bank",
        required=True,
        choices=["train-point", "train-block", "core", "grid", "clean"],
    )
    ap.add_argument(
        "--cache-name",
        "--bank-name",
        dest="cache_name",
        help="Distinct on-disk bank name, e.g. core_seed31",
    )
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--seed", type=int, default=31)
    ap.add_argument("--views", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument(
        "--storage-dtype", choices=["float16", "float32"], default="float16"
    )
    ap.add_argument("--model-path")
    ap.add_argument("--local-files-only", action="store_true")
    ap.add_argument(
        "--final-evaluation",
        action="store_true",
        help="Explicitly enable held-out/whole-valid/special inference after freezing",
    )
    args = ap.parse_args()
    if (
        args.split == "test"
        or (args.split == "valid" and args.subset_name != "valid_tune")
    ) and (not args.final_evaluation):
        ap.error(
            "Held-out/whole-valid/special cache requires --final-evaluation after model protocol is frozen"
        )
    data = load_official(args.source, args.split)
    if args.source_provenance:
        provenance = _read_json(args.source_provenance)
        if provenance["working_copy_sha256"] != data["source_sha256"]:
            raise ValueError("Working-copy source hash does not match provenance")
        data["original_source_sha256"] = provenance["original_source_sha256"]
    if args.subset_file or args.subset_name:
        if not args.subset_file or not args.subset_name or args.split != "valid":
            ap.error(
                "--subset-file and --subset-name must be provided together for official valid"
            )
        definition = _read_json(args.subset_file)
        if definition["source_sha256"] != data.get(
            "original_source_sha256", data["source_sha256"]
        ):
            raise ValueError("Subset definition official source hash mismatch")
        entry = definition["splits"][args.subset_name]
        data = subset_data(data, entry["indices_in_official_valid"], args.subset_name)
        if "sample_ids" in entry and data["ids"] != entry["sample_ids"]:
            raise ValueError("Subset IDs do not match official indices")
    encoder = FrozenBertEncoder(args.model_path, args.device, args.local_files_only)
    result = build_cache(
        data,
        args.output,
        args.bank,
        encoder,
        args.seed,
        args.views,
        args.batch_size,
        args.storage_dtype,
        cache_name=args.cache_name,
    )
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "status",
                    "split",
                    "bank",
                    "n",
                    "new_unique_text_inputs",
                    "elapsed_seconds",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
