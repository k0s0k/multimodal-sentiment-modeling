"""One numerical prediction path for selection, export and all interventions."""

from collections import OrderedDict
from pathlib import Path
import time
import numpy as np
import torch
from sentiment.data import observed_masks, apply_scaler
from .augmentation import apply_keep
from .encoding import FrozenEncoder, set_numeric_profile
from .models import load_checkpoint, OnlineBertFusion

KEYS = ("text", "audio", "vision", "mask", "structure")


def tensor_batch(data, device):
    return {
        k: torch.as_tensor(v, device=device)
        for k, v in data.items()
        if k in (*KEYS, "tokens")
    }


class Predictor:

    def __init__(self, checkpoints, bert_path, device="cuda:0", batch_size=64):
        set_numeric_profile()
        self.device = device
        self.batch_size = int(batch_size)
        self.paths = [str(Path(p)) for p in checkpoints]
        pairs = [load_checkpoint(p, bert_path, device) for p in checkpoints]
        self.models, self.states = map(list, zip(*pairs))
        self.encoder = (
            FrozenEncoder(bert_path, device, self.batch_size)
            if any((not isinstance(m, OnlineBertFusion) for m in self.models))
            else None
        )
        self.member_count = len(self.models)
        self.stats = dict(
            fusion_sample_members=0, bert_unique_states=0, wall_seconds=0.0
        )

    @torch.inference_mode()
    @torch.autocast("cuda", enabled=False)
    def predict(self, raw, keep=None, return_members=False):
        original = observed_masks(raw["tokens"], raw["audio"], raw["vision"])
        masked = apply_keep(raw, original if keep is None else keep)
        result = []
        for start in range(0, len(raw["tokens"]), self.batch_size):
            batch = {k: v[start : start + self.batch_size] for k, v in masked.items()}
            text = self.encoder.encode(batch["tokens"]) if self.encoder else None
            members = []
            for model, state in zip(self.models, self.states):
                scaled = apply_scaler(batch, state["scaler"])
                if not isinstance(model, OnlineBertFusion):
                    scaled["text"] = text
                args = tensor_batch(scaled, self.device)
                out = model(args)
                members.append(
                    np.column_stack(
                        (
                            out["logits"].float().softmax(-1).cpu().numpy(),
                            out["regression"].float().cpu().numpy(),
                        )
                    )
                )
            result.append(np.stack(members, 1))
        values = np.concatenate(result)
        return values if return_members else values.mean(1, dtype=np.float64)

    def callback(self, raw, index, max_cache_bytes=4 * 1024**3):
        return MaskPredictor(self, raw, index, max_cache_bytes)


class MaskPredictor:
    """Per-example cache keyed by exact current text visibility, never original text."""

    def __init__(self, predictor, raw, index, max_cache_bytes):
        self.predictor = predictor
        self.raw = {
            k: np.asarray(raw[k])[index : index + 1]
            for k in ("tokens", "audio", "vision")
        }
        self.observed = observed_masks(**self.raw)[0]
        self.cache = OrderedDict()
        self.max_cache_items = max(1, int(max_cache_bytes) // (50 * 768 * 4))
        self.stats = dict(
            calls=0, queried_masks=0, bert_new_states=0, fusion_sample_members=0
        )

    def _encoder_key(self, member):
        return (
            member
            if isinstance(self.predictor.models[member], OnlineBertFusion)
            else "public"
        )

    @torch.inference_mode()
    @torch.autocast("cuda", enabled=False)
    def _texts(self, tokens, member):
        p = self.predictor
        encoder_key = self._encoder_key(member)
        keys = [(encoder_key, t.tobytes()) for t in tokens]
        missing = {}
        for k, t in zip(keys, tokens):
            if k not in self.cache:
                missing[k] = t
        if missing:
            missing_keys = list(missing)
            ts = np.stack(list(missing.values()))
            for start in range(0, len(ts), p.batch_size):
                sub = ts[start : start + p.batch_size]
                if encoder_key == "public":
                    text = p.encoder.encode(sub)
                else:
                    model = p.models[member]
                    tt = torch.as_tensor(sub, device=p.device, dtype=torch.long)
                    content = (
                        tt[:, 1].bool()
                        & (tt[:, 0] != 0)
                        & (tt[:, 0] != 101)
                        & (tt[:, 0] != 102)
                    )
                    text = model.encode_text(tt, content).float().cpu().numpy()
                for k, value in zip(missing_keys[start : start + p.batch_size], text):
                    self.cache[k] = value.copy()
                self.stats["bert_new_states"] += len(sub)
        result = np.stack([self.cache[k] for k in keys])
        for k in keys:
            self.cache.move_to_end(k)
        while len(self.cache) > self.max_cache_items:
            self.cache.popitem(last=False)
        return result

    @torch.inference_mode()
    @torch.autocast("cuda", enabled=False)
    def __call__(self, keep_masks):
        p = self.predictor
        set_numeric_profile()
        keep_masks = np.asarray(keep_masks, dtype=bool)
        if keep_masks.ndim != 3 or keep_masks.shape[1:] != (3, 50):
            raise ValueError("Callback requires [B,3,50] keep masks")
        if (keep_masks & ~self.observed).any():
            raise ValueError("Cannot make originally absent features observable")
        result = []
        begin = time.perf_counter()
        for start in range(0, len(keep_masks), p.batch_size):
            keep = keep_masks[start : start + p.batch_size]
            raw = {k: np.repeat(v, len(keep), axis=0) for k, v in self.raw.items()}
            batch = apply_keep(raw, keep)
            members = []
            for j, (model, state) in enumerate(zip(p.models, p.states)):
                text = self._texts(batch["tokens"], j)
                scaled = apply_scaler(batch, state["scaler"])
                scaled["text"] = text
                fusion = model.fusion if isinstance(model, OnlineBertFusion) else model
                out = fusion(tensor_batch(scaled, p.device))
                members.append(
                    np.column_stack(
                        (
                            out["logits"].float().softmax(-1).cpu().numpy(),
                            out["regression"].float().cpu().numpy(),
                        )
                    )
                )
            result.append(np.mean(members, axis=0, dtype=np.float64))
        self.stats["calls"] += 1
        self.stats["queried_masks"] += len(keep_masks)
        self.stats["fusion_sample_members"] += len(keep_masks) * p.member_count
        p.stats["wall_seconds"] += time.perf_counter() - begin
        return np.concatenate(result)

    @torch.autocast("cuda", enabled=False)
    def gradient_scores(self, target_class, steps=32):
        """Representation-space IG, keeping visibility/positions fixed.

        This is a comparator, not a token-removal explanation. A zero vector
        baseline is used inside the fusion model; BERT is evaluated once.
        """
        p = self.predictor
        full = apply_keep(self.raw, self.observed[None])
        all_ixg, all_ig, residuals = ([], [], [])
        for j, (model, state) in enumerate(zip(p.models, p.states)):
            scaled = apply_scaler(full, state["scaler"])
            scaled["text"] = self._texts(full["tokens"], j)
            fusion = model.fusion if isinstance(model, OnlineBertFusion) else model
            values = tensor_batch(scaled, p.device)
            features = [values[k].detach() for k in ("text", "audio", "vision")]

            def evaluate(alpha, grad=True):
                xs = [(x * alpha).requires_grad_(grad) for x in features]
                out = fusion({**values, **dict(zip(("text", "audio", "vision"), xs))})
                score = out["logits"].float().softmax(-1)[0, int(target_class)]
                if grad:
                    gradients = torch.autograd.grad(score, xs)
                    return [g.detach() for g in gradients]
                return float(score.detach().cpu())

            with torch.enable_grad():
                ixg = [
                    (g * x).sum(-1)[0].detach().cpu().numpy()
                    for g, x in zip(evaluate(1), features)
                ]
                sums = [torch.zeros_like(x) for x in features]
                for alpha in (np.arange(steps) + 0.5) / steps:
                    grads = evaluate(float(alpha))
                    sums = [s + g / steps for s, g in zip(sums, grads)]
                ig = [
                    (g * x).sum(-1)[0].detach().cpu().numpy()
                    for g, x in zip(sums, features)
                ]
                residuals.append(
                    float(sum((x.sum() for x in ig)))
                    - (evaluate(1, False) - evaluate(0, False))
                )
            all_ixg.append(np.stack(ixg))
            all_ig.append(np.stack(ig))
        return dict(
            input_x_gradient=np.mean(all_ixg, 0).tolist(),
            integrated_gradients=np.mean(all_ig, 0).tolist(),
            ig_steps=steps,
            ig_completeness_residual_members=residuals,
            ig_space="cached_last_hidden_state_and_scaled_AV_with_fixed_visibility",
        )
