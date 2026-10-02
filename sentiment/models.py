"""Q2 damaged-input-only models; no feature extractor or clean-view side channel.

EMT/OAGL mechanism adapted from Sun et al., IEEE TAFFC 2024 (online 2023).
Source: https://github.com/sunlicai/EMT-DLFR, commit below.
Retains bidirectional cross/self/GEGLU MPU, global attention pooling, and all
three sharing levels. This is an aligned-feature adaptation, not a published
checkpoint or a claim to reproduce the original benchmark. See provenance().
"""

from __future__ import annotations
import copy
import math
from collections.abc import Mapping
import torch
from torch import Tensor, nn
import torch.nn.functional as F

MODALITIES = ("text", "audio", "vision")
INPUT_DIMS = {"text": 768, "audio": 74, "vision": 35}
EMT_SOURCE = {
    "paper": "https://arxiv.org/abs/2208.07589",
    "repository": "https://github.com/sunlicai/EMT-DLFR",
    "commit": "e4216215565292360e9c3fc43c8cd6e127df19c2",
    "file": "models/subNets/EMT.py",
    "source_sha256": "204b28ad69245d879231b98f48d43db600f87efa6e57cd73df321f525b5f2d28",
    "adaptations": [
        "Fixed precomputed aligned input; linear projections replace raw BERT/LSTM extraction.",
        "128 hidden, 2 fusion layers, 4 heads, 256 gated FFN by default.",
        "Observed-only K/V and global-source pooling; safe empty-context attention.",
        "Masked means initialize globals; pre-fusion modality summaries plus globals feed dual heads.",
        "LayerNorm replaces auxiliary-head BatchNorm to support batch size one.",
        "Optional per-position six-channel structure and observed-only modality gate.",
        "Damaged view only; caller explicitly performs separate original-view forward for training.",
    ],
}


def _configuration(config):
    c = dict(config) if isinstance(config, Mapping) else vars(config).copy()
    merged = dict(c.get("model_initial", {}))
    merged.update(c)
    for canonical, alias, default in (
        ("hidden_dimension", "dim", 128),
        ("layers", "num_layers", 2),
        ("heads", "num_heads", 4),
        ("ffn_dimension", "ffn_dim", 256),
    ):
        merged[canonical] = int(merged.get(canonical, merged.get(alias, default)))
    merged.setdefault("family", "emt")
    merged.setdefault("dropout", 0.1)
    merged.setdefault("max_length", 50)
    merged.setdefault("use_structure", merged["family"] == "structure")
    merged.setdefault("use_gate", merged["family"] == "structure")
    auxiliaries = merged["family"] in ("emt", "structure")
    merged.setdefault("use_reconstruction", auxiliaries)
    merged.setdefault("use_hlfr", auxiliaries)
    merged.setdefault(
        "lambda_reconstruction", 0.02 if merged["use_reconstruction"] else 0.0
    )
    merged.setdefault("lambda_hlfr", 0.1 if merged["use_hlfr"] else 0.0)
    for name in ("mpu_share", "modality_share", "layer_share"):
        merged.setdefault(name, True)
    merged.setdefault("validate_inputs", True)
    merged["emt_source"] = copy.deepcopy(EMT_SOURCE)
    return merged


def masked_mean(x: Tensor, observed: Tensor) -> Tensor:
    clean = torch.where(observed.unsqueeze(-1), x, torch.zeros_like(x))
    return clean.sum(1) / observed.sum(1, keepdim=True).clamp_min(1).to(x.dtype)


def masked_softmax(scores: Tensor, mask: Tensor, dim: int = -1) -> Tensor:
    """Exactly zero for masked entries and all-empty rows, including fp16."""
    scores = scores.float().masked_fill(~mask, -torch.finfo(torch.float32).max)
    weights = torch.softmax(scores, dim=dim)
    weights = torch.where(mask, weights, torch.zeros_like(weights))
    return weights / weights.sum(dim=dim, keepdim=True).clamp_min(1e-12)


class SafeAttention(nn.Module):

    def __init__(self, dim, heads, dropout):
        super().__init__()
        self.heads, self.head_dim = (heads, dim // heads)
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_kv = nn.Linear(dim, 2 * dim, bias=False)
        self.to_out = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, context, observed):
        b, nq, d = query.shape
        nk = context.shape[1]
        context = torch.where(observed[..., None], context, torch.zeros_like(context))
        q = self.to_q(query).reshape(b, nq, self.heads, self.head_dim).transpose(1, 2)
        k, v = self.to_kv(context).chunk(2, -1)
        k = k.reshape(b, nk, self.heads, self.head_dim).transpose(1, 2)
        v = v.reshape(b, nk, self.heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q.float(), k.float().transpose(-1, -2)) / math.sqrt(
            self.head_dim
        )
        a = masked_softmax(scores, observed[:, None, None, :]).to(v.dtype)
        result = torch.matmul(self.dropout(a), v).transpose(1, 2).reshape(b, nq, d)
        result = self.to_out(result)
        return torch.where(
            observed.any(1)[:, None, None], result, torch.zeros_like(result)
        )


class GEGLUFFN(nn.Module):

    def __init__(self, dim, ffn, dropout):
        super().__init__()
        self.in_proj = nn.Linear(dim, 2 * ffn)
        self.out_proj = nn.Linear(ffn, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        value, gate = self.in_proj(x).chunk(2, -1)
        return self.dropout(self.out_proj(value * F.gelu(gate)))


class SelfBlock(nn.Module):

    def __init__(self, dim, heads, ffn, dropout):
        super().__init__()
        self.norm_attn = nn.LayerNorm(dim)
        self.attn = SafeAttention(dim, heads, dropout)
        self.norm_ff = nn.LayerNorm(dim)
        self.ff = GEGLUFFN(dim, ffn, dropout)

    def forward(self, x, observed):
        normalized = self.norm_attn(x)
        x = x + self.attn(normalized, normalized, observed)
        return x + self.ff(self.norm_ff(x))


class HalfMPU(nn.Module):
    """Original ordering: pre-norm cross attention, self attention, GEGLU."""

    def __init__(self, dim, heads, ffn, dropout):
        super().__init__()
        self.norm_query, self.norm_context = (nn.LayerNorm(dim), nn.LayerNorm(dim))
        self.cross = SafeAttention(dim, heads, dropout)
        self.self_block = SelfBlock(dim, heads, ffn, dropout)

    def forward(self, x, context, observed, context_observed):
        x = x + self.cross(
            self.norm_query(x), self.norm_context(context), context_observed
        )
        return self.self_block(x, observed)


def _clones(module, number, share):
    return nn.ModuleList(
        [module if share else copy.deepcopy(module) for _ in range(number)]
    )


class OAGLFusion(nn.Module):

    def __init__(
        self,
        dim,
        layers,
        heads,
        ffn,
        dropout,
        mpu_share=True,
        modality_share=True,
        layer_share=True,
    ):
        super().__init__()
        half = HalfMPU(dim, heads, ffn, dropout)
        bidirectional = _clones(half, 2, mpu_share)
        modalities = _clones(bidirectional, 3, modality_share)
        self.mpus = _clones(modalities, layers, layer_share)
        pool = nn.Sequential(
            nn.Linear(3 * dim, 3 * dim), nn.Tanh(), nn.Linear(3 * dim, 1)
        )
        self.pools = _clones(pool, layers, layer_share)

    def forward(self, global_tokens, local_inputs, masks):
        available = torch.stack([m.any(1) for m in masks], 1)
        b, _, dim = global_tokens.shape
        pooling_weights = available.float() / available.sum(1, keepdim=True).clamp_min(
            1
        )
        for layer, pool in zip(self.mpus, self.pools):
            updated_locals, contexts = ([], [])
            for i, x in enumerate(local_inputs):
                updated_locals.append(
                    layer[i][0](x, global_tokens, masks[i], available)
                )
                contexts.append(layer[i][1](global_tokens, x, available, masks[i]))
            candidates = torch.stack(contexts, 1).reshape(b, 3, 3 * dim)
            pooling_weights = masked_softmax(pool(candidates).squeeze(-1), available)
            global_tokens = (
                (candidates * pooling_weights[..., None]).sum(1).reshape(b, 3, dim)
            )
            global_tokens = torch.where(
                available[..., None], global_tokens, torch.zeros_like(global_tokens)
            )
            local_inputs = updated_locals
        return (global_tokens, local_inputs, pooling_weights)


class Projector(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )

    def forward(self, x):
        return self.net(x)


class Model(nn.Module):
    """forward(batch) never reads labels, original views, teacher, or clean masks.

    `mask[:,0]` MUST exclude text CLS/SEP/PAD. Such special-token identity cannot
    be recovered from an embedding tensor; the data adapter owns that contract.
    Structure channels must be derived from this damaged view, never clean data.
    """

    def __init__(self, config):
        super().__init__()
        self.config = _configuration(config)
        c = self.config
        self.family = c["family"]
        if self.family not in ("text", "late", "emt", "structure"):
            raise ValueError("family must be text, late, emt, or structure")
        d, h, ff = (c["hidden_dimension"], c["heads"], c["ffn_dimension"])
        if d <= 0 or h <= 0 or d % h or (c["layers"] < 1) or (ff < 1):
            raise ValueError("invalid dimensions/heads/layers")
        self.dim = d
        self.active = ("text",) if self.family == "text" else MODALITIES
        self.projections = nn.ModuleDict(
            {m: nn.Linear(INPUT_DIMS[m], d) for m in self.active}
        )
        self.input_dropout = nn.Dropout(float(c["dropout"]))
        position = torch.arange(c["max_length"], dtype=torch.float32)[:, None]
        rates = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
        pe = torch.zeros(c["max_length"], d)
        pe[:, 0::2] = torch.sin(position * rates)
        pe[:, 1::2] = torch.cos(position * rates[: pe[:, 1::2].shape[1]])
        self.register_buffer("position_encoding", pe, persistent=False)
        self.structure_encoder = nn.Linear(6, d) if c["use_structure"] else None
        self.observed_embedding = nn.Embedding(2, d) if c["use_structure"] else None
        if self.family in ("emt", "structure"):
            self.fusion = OAGLFusion(
                d,
                c["layers"],
                h,
                ff,
                c["dropout"],
                c["mpu_share"],
                c["modality_share"],
                c["layer_share"],
            )
            self.local_encoders = None
            fusion_size = 6 * d
        else:
            self.fusion = None
            self.local_encoders = nn.ModuleDict(
                {
                    m: nn.ModuleList(
                        [SelfBlock(d, h, ff, c["dropout"]) for _ in range(c["layers"])]
                    )
                    for m in self.active
                }
            )
            fusion_size = len(self.active) * d
        self.gate_network = (
            nn.Sequential(nn.Linear(d + 6, d), nn.Tanh(), nn.Linear(d, 1))
            if c["use_gate"]
            else None
        )
        self.final = nn.Sequential(
            nn.Linear(fusion_size, d),
            nn.ReLU(),
            nn.Dropout(c["dropout"]),
            nn.LayerNorm(d),
        )
        self.classifier = nn.Linear(d, 3)
        self.regressor = nn.Linear(d, 1)
        self.reconstructors = (
            nn.ModuleDict({m: nn.Linear(d, INPUT_DIMS[m]) for m in self.active})
            if c["use_reconstruction"]
            else None
        )
        aux_sizes = {m: d for m in self.active}
        if self.fusion is not None:
            aux_sizes["global"] = 3 * d
        self.projectors = (
            nn.ModuleDict({k: Projector(v) for k, v in aux_sizes.items()})
            if c["use_hlfr"]
            else None
        )
        self.predictors = (
            nn.ModuleDict(
                {
                    k: nn.Sequential(
                        nn.Linear(v, d), nn.LayerNorm(d), nn.ReLU(), nn.Linear(d, v)
                    )
                    for k, v in aux_sizes.items()
                }
            )
            if c["use_hlfr"]
            else None
        )
        self.register_buffer("prior_probs", torch.full((3,), 1 / 3))
        self.register_buffer("prior_mean", torch.tensor(0.0))

    @torch.no_grad()
    def set_priors(self, probs, mean):
        probs = torch.as_tensor(
            probs, dtype=self.prior_probs.dtype, device=self.prior_probs.device
        )
        mean = torch.as_tensor(
            mean, dtype=self.prior_mean.dtype, device=self.prior_mean.device
        )
        if (
            probs.shape != (3,)
            or not torch.isfinite(probs).all()
            or (probs < 0).any()
            or (probs.sum() <= 0)
        ):
            raise ValueError(
                "priors must be three finite nonnegative masses with positive sum"
            )
        if mean.numel() != 1 or not torch.isfinite(mean).all() or mean.abs() > 3:
            raise ValueError("prior mean must be a finite scalar in [-3,3]")
        self.prior_probs.copy_(probs / probs.sum())
        self.prior_mean.copy_(mean.reshape(()))
        return self

    def provenance(self):
        return {
            "config": copy.deepcopy(self.config),
            "parameters_total": sum((p.numel() for p in self.parameters())),
            "parameters_trainable": sum(
                (p.numel() for p in self.parameters() if p.requires_grad)
            ),
            "adaptation_not_original_sota_reproduction": True,
        }

    def forward(self, batch):
        mask = batch["mask"]
        if mask.ndim != 3 or mask.shape[1] != 3 or mask.dtype != torch.bool:
            raise ValueError("mask must be bool [B,3,L] with text specials excluded")
        b, _, length = mask.shape
        if length > self.config["max_length"] or length < 1:
            raise ValueError("sequence length outside configured range")
        structure = batch.get("structure")
        if self.structure_encoder is not None:
            if structure is None or structure.shape != (b, 3, length, 6):
                raise ValueError("structure must be [B,3,L,6]")
            if self.config["validate_inputs"] and (not torch.isfinite(structure).all()):
                raise ValueError("nonfinite structure")
        inputs, masks, initial, structural_summaries = ([], [], [], [])
        for m in self.active:
            j = MODALITIES.index(m)
            observed, raw = (mask[:, j], batch[m])
            if raw.shape != (b, length, INPUT_DIMS[m]):
                raise ValueError(f"invalid {m} shape")
            raw = torch.where(observed[..., None], raw, torch.zeros_like(raw))
            if self.config["validate_inputs"] and (not torch.isfinite(raw).all()):
                raise ValueError(f"nonfinite observed {m} feature")
            x = self.projections[m](raw) + self.position_encoding[:length].to(raw.dtype)
            if self.structure_encoder is not None:
                x = (
                    x
                    + self.structure_encoder(structure[:, j].to(raw.dtype))
                    + self.observed_embedding(observed.long())
                )
            x = self.input_dropout(x)
            if self.local_encoders is not None:
                for layer in self.local_encoders[m]:
                    x = layer(x, observed)
            inputs.append(x)
            masks.append(observed)
            initial.append(masked_mean(x, observed))
            if self.gate_network is not None:
                structural_summaries.append(
                    structure[:, j].to(raw.dtype).mean(1)
                    if self.structure_encoder is not None
                    else x.new_zeros(b, 6)
                )
        summaries = torch.stack(initial, 1)
        available = torch.stack([m.any(1) for m in masks], 1)
        observed_any = available.any(1)
        global_tokens, global_pool = (None, None)
        if self.fusion is not None:
            global_tokens, inputs, global_pool = self.fusion(summaries, inputs, masks)
        if self.gate_network is not None:
            gate_input = torch.cat(
                [summaries, torch.stack(structural_summaries, 1)], -1
            )
            gates_active = masked_softmax(
                self.gate_network(gate_input).squeeze(-1), available
            )
            factor = gates_active * available.sum(1, keepdim=True).to(
                gates_active.dtype
            )
            task_summaries = summaries * factor[..., None]
        else:
            gates_active = available.float() / available.sum(1, keepdim=True).clamp_min(
                1
            )
            task_summaries = summaries
        joined = [task_summaries.flatten(1)]
        if global_tokens is not None:
            joined.append(global_tokens.flatten(1))
        representation = self.final(torch.cat(joined, -1))
        logits = self.classifier(representation)
        regression = 3 * torch.tanh(self.regressor(representation).squeeze(-1))
        logits = torch.where(
            observed_any[:, None],
            logits,
            self.prior_probs.clamp_min(1e-12).log()[None, :],
        )
        regression = torch.where(observed_any, regression, self.prior_mean)
        representation = torch.where(
            observed_any[:, None], representation, torch.zeros_like(representation)
        )
        gates = logits.new_zeros(b, 3)
        for i, m in enumerate(self.active):
            gates[:, MODALITIES.index(m)] = gates_active[:, i].to(gates.dtype)
        features = dict(zip(self.active, inputs))
        output = {
            "logits": logits,
            "regression": regression,
            "representation": representation,
            "features": features,
            "gates": gates,
            "observed_any": observed_any,
            "mask": mask,
            "global_pool_weights": global_pool,
            "z": {},
            "p": {},
        }
        if self.reconstructors is not None:
            output["recon"] = {
                m: self.reconstructors[m](features[m]) for m in self.active
            }
        if self.projectors is not None:
            aux = dict(zip(self.active, initial))
            if global_tokens is not None:
                aux["global"] = global_tokens.flatten(1)
            output["z"] = {k: self.projectors[k](v) for k, v in aux.items()}
            output["p"] = {k: self.predictors[k](v) for k, v in output["z"].items()}
        return output
