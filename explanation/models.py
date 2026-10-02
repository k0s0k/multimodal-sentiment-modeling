"""Q3 predictor registry and a common differentiable online-BERT wrapper.

The copied sentiment package is a read-only vendor. All six Q3 families use
the same prediction contract; explanations must query their actual outputs.
"""

from __future__ import annotations
import copy
from collections.abc import Mapping
from pathlib import Path
import torch
from torch import nn
from sentiment.models import (
    INPUT_DIMS,
    MODALITIES,
    Model as VendorModel,
    SelfBlock,
    masked_mean,
)


def normalize_model_config(config):
    c = copy.deepcopy(dict(config))
    family = c.get("family", "clean_emt")
    aliases = {
        "P1": "clean_emt",
        "P2": "clean_structure",
        "P3": "structure_aux",
        "P4": "structure_aux",
        "P5": "additive",
        "P6": "additive_pairwise",
    }
    family = aliases.get(family, family)
    c["family"] = family
    c.setdefault("hidden_dimension", 128)
    c.setdefault("layers", 1 if family.startswith("additive") else 2)
    c.setdefault("heads", 4)
    c.setdefault("ffn_dimension", 256)
    c.setdefault("dropout", 0.1)
    c.setdefault("max_length", 50)
    c.setdefault("validate_inputs", True)
    if family in ("clean_emt", "clean_structure", "structure_aux"):
        structured = family != "clean_emt"
        recovery = family == "structure_aux"
        c.setdefault("use_structure", structured)
        c.setdefault("use_gate", structured)
        c.setdefault("use_reconstruction", recovery)
        c.setdefault("use_hlfr", recovery)
    if family.startswith("additive"):
        c.setdefault("pairwise_rank", 16 if family == "additive_pairwise" else 0)
        if c.get("use_reconstruction") or c.get("use_hlfr"):
            raise ValueError("P5/P6 do not define recovery auxiliary heads")
    return c


class AdditiveFusion(nn.Module):
    """Independent temporal branches plus optional explicit low-rank pairs.

    Missing branches and pairs contribute exactly zero to the latent sum.
    Softmax/tanh act after summation: this does not imply output additivity.
    """

    def __init__(self, config):
        super().__init__()
        self.config = normalize_model_config(config)
        self.q3_config = copy.deepcopy(self.config)
        c = self.config
        d, heads = (c["hidden_dimension"], c["heads"])
        if d <= 0 or heads <= 0 or d % heads or (c["layers"] < 1):
            raise ValueError("invalid additive dimensions")
        rank = int(c.get("pairwise_rank", 0))
        if rank < 0:
            raise ValueError("pairwise_rank must be nonnegative")
        self.active = MODALITIES
        self.projections = nn.ModuleDict(
            {m: nn.Linear(INPUT_DIMS[m], d) for m in MODALITIES}
        )
        self.encoders = nn.ModuleDict(
            {
                m: nn.ModuleList(
                    [
                        SelfBlock(d, heads, c["ffn_dimension"], c["dropout"])
                        for _ in range(c["layers"])
                    ]
                )
                for m in MODALITIES
            }
        )
        self.norms = nn.ModuleDict({m: nn.LayerNorm(d) for m in MODALITIES})
        self.branch_heads = nn.ModuleDict({m: nn.Linear(d, 4) for m in MODALITIES})
        self.input_dropout = nn.Dropout(c["dropout"])
        position = torch.arange(c["max_length"], dtype=torch.float32)[:, None]
        rates = torch.exp(
            torch.arange(0, d, 2).float() * (-torch.log(torch.tensor(10000.0)) / d)
        )
        pe = torch.zeros(c["max_length"], d)
        pe[:, 0::2] = torch.sin(position * rates)
        pe[:, 1::2] = torch.cos(position * rates[: pe[:, 1::2].shape[1]])
        self.register_buffer("position_encoding", pe, persistent=False)
        self.output_bias = nn.Parameter(torch.zeros(4))
        self.pair_left, self.pair_right, self.pair_heads = (
            nn.ModuleDict(),
            nn.ModuleDict(),
            nn.ModuleDict(),
        )
        self.pair_indices = ((0, 1), (0, 2), (1, 2)) if rank else ()
        for i, j in self.pair_indices:
            key = MODALITIES[i] + "_" + MODALITIES[j]
            self.pair_left[key] = nn.Linear(d, rank, bias=False)
            self.pair_right[key] = nn.Linear(d, rank, bias=False)
            self.pair_heads[key] = nn.Linear(rank, 4, bias=False)
        self.register_buffer("prior_probs", torch.full((3,), 1 / 3))
        self.register_buffer("prior_mean", torch.tensor(0.0))

    @torch.no_grad()
    def set_priors(self, probs, mean):
        p = torch.as_tensor(
            probs, dtype=self.prior_probs.dtype, device=self.prior_probs.device
        )
        y = torch.as_tensor(
            mean, dtype=self.prior_mean.dtype, device=self.prior_mean.device
        )
        if (
            p.shape != (3,)
            or not torch.isfinite(p).all()
            or (p < 0).any()
            or (p.sum() <= 0)
        ):
            raise ValueError("invalid class prior")
        if y.numel() != 1 or not torch.isfinite(y).all() or y.abs() > 3:
            raise ValueError("invalid regression prior")
        self.prior_probs.copy_(p / p.sum())
        self.prior_mean.copy_(y.reshape(()))
        return self

    def provenance(self):
        return {
            "config": copy.deepcopy(self.config),
            "parameters_total": sum((p.numel() for p in self.parameters())),
            "parameters_trainable": sum(
                (p.numel() for p in self.parameters() if p.requires_grad)
            ),
        }

    def forward(self, batch):
        mask = batch["mask"]
        if mask.dtype != torch.bool or mask.ndim != 3 or mask.shape[1] != 3:
            raise ValueError("mask must be bool [B,3,L]")
        b, _, length = mask.shape
        if not 1 <= length <= self.config["max_length"]:
            raise ValueError("sequence length outside configured range")
        features, summaries, branch_terms = ({}, [], [])
        available = mask.any(-1)
        for j, m in enumerate(MODALITIES):
            raw = batch[m]
            if raw.shape != (b, length, INPUT_DIMS[m]):
                raise ValueError(f"invalid {m} shape")
            raw = torch.where(mask[:, j, :, None], raw, torch.zeros_like(raw))
            if self.config["validate_inputs"] and (not torch.isfinite(raw).all()):
                raise ValueError(f"nonfinite observed {m} input")
            x = self.projections[m](raw) + self.position_encoding[:length].to(raw.dtype)
            x = self.input_dropout(x)
            for block in self.encoders[m]:
                x = block(x, mask[:, j])
            summary = self.norms[m](masked_mean(x, mask[:, j]))
            summary = torch.where(
                available[:, j, None], summary, torch.zeros_like(summary)
            )
            term = self.branch_heads[m](summary)
            term = torch.where(available[:, j, None], term, torch.zeros_like(term))
            features[m] = x
            summaries.append(summary)
            branch_terms.append(term)
        latent = self.output_bias[None, :] + torch.stack(branch_terms, 1).sum(1)
        pair_terms = {}
        for i, j in self.pair_indices:
            key = MODALITIES[i] + "_" + MODALITIES[j]
            product = self.pair_left[key](summaries[i]) * self.pair_right[key](
                summaries[j]
            )
            term = self.pair_heads[key](product)
            term = torch.where(
                (available[:, i] & available[:, j])[:, None],
                term,
                torch.zeros_like(term),
            )
            latent = latent + term
            pair_terms[key] = term
        observed_any = available.any(1)
        logits = torch.where(
            observed_any[:, None],
            latent[:, :3],
            self.prior_probs.clamp_min(1e-12).log()[None],
        )
        regression = torch.where(
            observed_any, 3 * torch.tanh(latent[:, 3]), self.prior_mean
        )
        representation = torch.cat(summaries, -1)
        gates = available.float() / available.sum(1, keepdim=True).clamp_min(1)
        return {
            "logits": logits,
            "regression": regression,
            "representation": representation,
            "features": features,
            "gates": gates,
            "observed_any": observed_any,
            "mask": mask,
            "global_pool_weights": None,
            "z": {},
            "p": {},
            "branch_terms": dict(zip(MODALITIES, branch_terms)),
            "pair_terms": pair_terms,
        }


def make_model(model_config):
    c = normalize_model_config(model_config)
    if c["family"] in ("additive", "additive_pairwise"):
        return AdditiveFusion(c)
    vendor = copy.deepcopy(c)
    vendor["family"] = {
        "clean_emt": "emt",
        "clean_structure": "structure",
        "structure_aux": "structure",
    }.get(c["family"], c["family"])
    if vendor["family"] not in ("emt", "structure", "text", "late"):
        raise ValueError(f"unknown model family {c['family']}")
    model = VendorModel(vendor)
    model.q3_config = c
    return model


class OnlineBertFusion(nn.Module):
    """All-family online encoder; only the last two BERT layers are optimized.

    forward_embeds is deliberately differentiable for optional embedding IG.
    Its input is the WORD embedding only; BERT adds position/type exactly once.
    """

    def __init__(self, fusion, bert, bert_metadata=None):
        super().__init__()
        self.fusion, self.bert = (fusion, bert)
        self.bert_metadata = dict(bert_metadata or {})
        if hasattr(bert, "pooler"):
            self.bert.pooler = None
        if len(self.bert.encoder.layer) != 12:
            raise ValueError("pinned BERT must have 12 layers")
        self.trainable_layer_indices = (10, 11)
        self.bert.requires_grad_(False)
        for i in self.trainable_layer_indices:
            self.bert.encoder.layer[i].requires_grad_(True)
        self.train(self.training)

    @property
    def q3_config(self):
        return self.fusion.q3_config

    def set_priors(self, probs, mean):
        self.fusion.set_priors(probs, mean)
        return self

    def train(self, mode=True):
        super().train(mode)
        self.bert.eval()
        for i in self.trainable_layer_indices:
            self.bert.encoder.layer[i].train(mode)
        self.fusion.train(mode)
        return self

    def encode_text(self, tokens, content, word_embeds=None):
        if (
            tokens.dtype != torch.long
            or tokens.ndim != 3
            or tokens.shape[1:] != (3, 50)
        ):
            raise ValueError("tokens must be int64 [B,3,50]")
        actual = (
            tokens[:, 1].bool()
            & (tokens[:, 0] != 0)
            & (tokens[:, 0] != 101)
            & (tokens[:, 0] != 102)
        )
        if content.dtype != torch.bool or not torch.equal(actual, content):
            raise ValueError("content mask differs from supplied tokens")
        if word_embeds is not None and word_embeds.shape != (*content.shape, 768):
            raise ValueError("word_embeds must be [B,50,768]")
        use = torch.nonzero(content.any(-1), as_tuple=False).flatten()
        result = torch.zeros(
            (*content.shape, 768), device=tokens.device, dtype=torch.float32
        )
        if not len(use):
            return result if word_embeds is None else result + word_embeds.float() * 0
        selected = tokens[use]
        attention = selected[:, 1]
        args = {
            "attention_mask": attention,
            "token_type_ids": torch.where(attention.bool(), selected[:, 2], 0),
            "position_ids": torch.arange(50, device=tokens.device)[None].expand(
                len(use), -1
            ),
            "return_dict": True,
        }
        if word_embeds is None:
            args["input_ids"] = torch.where(attention.bool(), selected[:, 0], 0)
        else:
            args["inputs_embeds"] = word_embeds[use]
        hidden = self.bert(**args).last_hidden_state.float()
        hidden = torch.where(content[use, :, None], hidden, torch.zeros_like(hidden))
        return result.index_copy(0, use, hidden)

    def forward(self, batch):
        text = self.encode_text(batch["tokens"], batch["mask"][:, 0])
        return self.fusion(
            {
                "text": text,
                **{k: batch[k] for k in ("audio", "vision", "mask", "structure")},
            }
        )

    def forward_embeds(self, batch, word_embeds):
        text = self.encode_text(batch["tokens"], batch["mask"][:, 0], word_embeds)
        return self.fusion(
            {
                "text": text,
                **{k: batch[k] for k in ("audio", "vision", "mask", "structure")},
            }
        )

    def delta_state(self):
        prefixes = tuple((f"encoder.layer.{i}." for i in self.trainable_layer_indices))
        return {
            k: v.detach().cpu().clone()
            for k, v in self.bert.state_dict().items()
            if k.startswith(prefixes)
        }

    def load_delta(self, delta):
        if set(delta) != set(self.delta_state()):
            raise ValueError("BERT delta must contain exactly layers 10 and 11")
        result = self.bert.load_state_dict(delta, strict=False)
        if result.unexpected_keys:
            raise ValueError("unexpected BERT delta keys")


def load_checkpoint(path, bert_path=None, device="cpu", trainable=False):
    """Load Q3 or unmodified Q2 frozen fusion states with explicit identity."""
    state = torch.load(Path(path), map_location="cpu", weights_only=False)
    fusion = make_model(state["model_config"])
    fusion.load_state_dict(state["model_state"], strict=True)
    model = fusion
    if state.get("bert_delta") is not None:
        if bert_path is None:
            raise ValueError("fine-tuned checkpoint requires pinned public BERT path")
        from .encoding import FrozenEncoder

        encoder = FrozenEncoder(bert_path, device=device, batch_size=64)
        expected = state.get("bert_metadata", {})
        actual = getattr(encoder, "metadata", {})
        for key in ("repo", "revision", "weights_sha256"):
            if expected.get(key) and actual.get(key) != expected[key]:
                raise ValueError(f"BERT identity mismatch: {key}")
        model = OnlineBertFusion(fusion, encoder.model, actual)
        model.load_delta(state["bert_delta"])
    model.to(device)
    model.train(trainable)
    if not trainable:
        model.requires_grad_(False)
    return (model, state)
