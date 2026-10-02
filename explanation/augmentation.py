"""Stable sample/epoch keyed Q3 visibility interventions before encoding/scaling."""

import numpy as np
from sentiment.data import stable_seed, sanitize_tokens, observed_masks, build_structure
from sentiment.masking import make_training_condition, make_masks

VERSION = "q3-mixture-v1"
SETS = ("T", "A", "V", "TA", "TV", "AV")


def apply_keep(raw, keep):
    tokens = sanitize_tokens(raw["tokens"])
    audio = np.asarray(raw["audio"], np.float32).copy()
    vision = np.asarray(raw["vision"], np.float32).copy()
    original = observed_masks(tokens, audio, vision)
    keep = np.asarray(keep, dtype=bool) & original
    drop = original & ~keep
    for channel in range(3):
        tokens[:, channel][drop[:, 0]] = 0
    audio[~keep[:, 1]] = 0
    vision[~keep[:, 2]] = 0
    return dict(
        tokens=tokens,
        audio=audio,
        vision=vision,
        mask=keep,
        structure=build_structure(keep),
    )


def augment_epoch(raw_train, seed, epoch, mode="q3_mixture"):
    n = len(raw_train["ids"])
    orig = observed_masks(raw_train["tokens"], raw_train["audio"], raw_train["vision"])
    if mode == "legacy_q2_block":
        view = np.array(
            [stable_seed(int(seed), int(epoch), str(x)) % 12 for x in raw_train["ids"]]
        )
        out = apply_keep(raw_train, orig)
        diagnostics = []
        for v in range(12):
            ix = np.flatnonzero(view == v)
            if not len(ix):
                continue
            batch = make_masks(
                raw_train["tokens"][ix],
                raw_train["audio"][ix],
                raw_train["vision"][ix],
                make_training_condition("block", v),
                31,
                [raw_train["ids"][i] for i in ix],
            )
            for key in ("tokens", "audio", "vision", "mask", "structure"):
                out[key][ix] = batch[key]
            diagnostics.extend(batch["diagnostics"]["records"])
        out["aux_valid"] = np.ones(n, dtype=bool)
        out["diagnostics"] = dict(version="q2-block-v1", mode=mode, records=diagnostics)
        return out
    if mode != "q3_mixture":
        raise ValueError("Unknown augmentation mode")
    keep = orig.copy()
    valid = np.zeros(n, dtype=bool)
    records = []
    for i, sample_id in enumerate(raw_train["ids"]):
        rng = np.random.default_rng(
            stable_seed(VERSION, int(seed), int(epoch), str(sample_id), 0)
        )
        details = {"id": str(sample_id), "attempts": 0, "mode": "skip"}
        for attempt in range(8):
            u = rng.random()
            candidate = orig[i].copy()
            if u < 0.5:
                chosen = SETS[int(rng.integers(6))]
                candidate &= np.array([m in chosen for m in "TAV"])[:, None]
                details.update(mode="whole_modality", kept_modalities=chosen)
            elif u < 0.75:
                p = float(rng.choice([0.25, 0.5, 0.75]))
                candidate &= rng.random((3, 50)) < p
                details.update(mode="independent_atoms", retention=p)
            else:
                chosen = (SETS + ("TAV",))[int(rng.integers(7))]
                rate = float(rng.choice([0.2, 0.4, 0.6]))
                length = int(np.ceil(50 * rate))
                start = int(rng.integers(51 - length))
                for m in chosen:
                    candidate["TAV".index(m), start : start + length] = False
                details.update(
                    mode="synchronous_block",
                    removed_modalities=chosen,
                    interval=[start, start + length],
                    rate=rate,
                )
            details["attempts"] = attempt + 1
            if candidate.any() and (orig[i] & ~candidate).any():
                keep[i] = candidate
                valid[i] = True
                break
        details.update(
            aux_valid=bool(valid[i]),
            newly_deleted_count=(orig[i] & ~keep[i]).sum(1).tolist(),
        )
        records.append(details)
    out = apply_keep(raw_train, keep)
    out["aux_valid"] = valid
    out["diagnostics"] = dict(
        version=VERSION,
        mode=mode,
        seed=int(seed),
        epoch=int(epoch),
        accepted=int(valid.sum()),
        skipped=int((~valid).sum()),
        records=records,
    )
    return out
