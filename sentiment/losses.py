"""Explicit Q2 training losses. No data loading or implicit clean forward.

Clean/original outputs must be supplied by the training caller. Teacher targets
and SimSiam target branches are detached here. Reliability is a training weight,
not a calibrated probability. Tensor `original_observed` is bool [B,3,L].
"""

from __future__ import annotations
from collections.abc import Mapping
import torch
from torch import Tensor
import torch.nn.functional as F

MODALITIES = ("text", "audio", "vision")


def _options(config):
    c = (
        {}
        if config is None
        else dict(config) if isinstance(config, Mapping) else vars(config).copy()
    )
    result = dict(c.get("loss_initial", {}))
    result.update(c)
    return result


def inverse_sqrt_class_weights(labels=None, *, counts=None, num_classes=3):
    """Train-only counts supplied by caller; absent classes use count floor one."""
    if counts is None:
        if labels is None:
            raise ValueError("provide train labels or train class counts")
        labels = torch.as_tensor(labels, dtype=torch.long)
        counts = torch.bincount(labels, minlength=num_classes)
    counts = torch.as_tensor(counts).float()
    if (
        counts.shape != (num_classes,)
        or not torch.isfinite(counts).all()
        or (counts < 0).any()
        or (counts.sum() <= 0)
    ):
        raise ValueError("invalid train class counts")
    weights = counts.clamp_min(1).rsqrt()
    return weights / weights.mean()


def supervised_loss(
    outputs, labels, targets, *, class_weights=None, lambda_regression=1.0, beta=0.5
):
    labels = labels.long()
    logits, prediction = (outputs["logits"].float(), outputs["regression"].float())
    if logits.shape != (labels.shape[0], 3) or prediction.shape != targets.shape:
        raise ValueError("prediction/target shape mismatch")
    weights = (
        None
        if class_weights is None
        else torch.as_tensor(class_weights, device=logits.device, dtype=torch.float32)
    )
    ce = F.cross_entropy(logits, labels, weight=weights)
    regression = F.smooth_l1_loss(prediction, targets.float(), beta=beta)
    return (ce + lambda_regression * regression, {"ce": ce, "regression": regression})


def reliability_weights(teacher, labels, targets, original_observed, current_observed):
    if original_observed is None or current_observed is None:
        raise ValueError("reliable KD requires original and damaged observation masks")
    if original_observed.dtype != torch.bool or current_observed.dtype != torch.bool:
        raise ValueError("observation masks must be bool")
    if (
        original_observed.shape != current_observed.shape
        or original_observed.ndim != 3
        or original_observed.shape[1] != 3
    ):
        raise ValueError("observation masks must both be [B,3,L]")
    with torch.no_grad():
        count = original_observed.sum(-1)
        retained = (current_observed & original_observed).sum(-1)
        eligible = count > 0
        per_modality = retained.float() / count.clamp_min(1).float()
        retention = (per_modality * eligible).sum(-1) / eligible.sum(-1).clamp_min(1)
        correct_class_prob = (
            teacher["logits"]
            .float()
            .softmax(-1)
            .gather(1, labels.long()[:, None])
            .squeeze(1)
        )
        teacher_error = (targets.float() - teacher["regression"].float()).abs()
        weight = (
            correct_class_prob
            * torch.exp(-teacher_error)
            * (retention / 0.5).clamp(max=1)
        )
        return torch.where(eligible.any(-1), weight, torch.zeros_like(weight)).detach()


def kd_loss(
    student,
    teacher,
    *,
    labels=None,
    targets=None,
    mode="fixed",
    temperature=2.0,
    beta=0.5,
    original_observed=None,
    current_observed=None,
):
    if temperature <= 0:
        raise ValueError("classification KD temperature must be positive")
    s_logits, t_logits = (student["logits"].float(), teacher["logits"].detach().float())
    t_regression = teacher["regression"].detach().float()
    per_cls = (
        F.kl_div(
            F.log_softmax(s_logits / temperature, dim=-1),
            F.softmax(t_logits / temperature, dim=-1),
            reduction="none",
        ).sum(-1)
        * temperature**2
    )
    per_reg = F.smooth_l1_loss(
        student["regression"].float(), t_regression, reduction="none", beta=beta
    )
    if mode == "reliable":
        if labels is None or targets is None:
            raise ValueError("reliable KD requires training labels and targets")
        weight = reliability_weights(
            teacher, labels, targets, original_observed, current_observed
        )
    elif mode == "fixed":
        weight = torch.ones_like(per_cls)
        if original_observed is not None:
            weight = weight * original_observed.flatten(1).any(-1)
    elif mode in ("none", "off"):
        weight = torch.zeros_like(per_cls)
    else:
        raise ValueError("KD mode must be fixed, reliable, or none")
    return (
        ((per_cls + per_reg) * weight).mean(),
        {
            "kd_classification": (per_cls * weight).mean(),
            "kd_regression": (per_reg * weight).mean(),
            "kd_weight_mean": weight.mean(),
            "kd_weights": weight,
        },
    )


def reconstruction_loss(
    predictions,
    targets,
    reconstruction_mask,
    *,
    original_observed,
    current_observed,
    beta=1.0,
    text_mode="smooth_l1",
):
    """Only known observations subsequently hidden by augmentation are targets.

    reconstruction_mask may be [B,3,L] or a dict of [B,L] bool tensors.
    Caller-provided masks are intersected with original & ~current masks. Missing
    target arrays cannot be treated as zero truth. No eligible targets -> zero.
    text_mode='cosine' is an optional adaptation; EMT LLFR uses smooth_l1.
    """
    if not predictions:
        raise ValueError("reconstruction requested without reconstruction outputs")
    first = next(iter(predictions.values()))
    total = first.sum() * 0.0
    parts = {}
    if original_observed is None or current_observed is None:
        raise ValueError("reconstruction requires original and current observed masks")
    if original_observed.dtype != torch.bool or current_observed.dtype != torch.bool:
        raise ValueError("reconstruction observation masks must be bool")
    if original_observed.shape != current_observed.shape:
        raise ValueError("reconstruction observation mask shapes differ")
    for name, pred in predictions.items():
        j = MODALITIES.index(name)
        proposed = (
            reconstruction_mask[name]
            if isinstance(reconstruction_mask, Mapping)
            else reconstruction_mask[:, j]
        )
        if proposed.dtype != torch.bool or proposed.shape != pred.shape[:2]:
            raise ValueError(f"invalid {name} reconstruction mask")
        mask = proposed & original_observed[:, j] & ~current_observed[:, j]
        target = targets[name].detach()
        if target.shape != pred.shape:
            raise ValueError(f"invalid {name} reconstruction target shape")
        if mask.any():
            p, t = (pred[mask].float(), target[mask].float())
            if not torch.isfinite(t).all():
                raise ValueError(f"nonfinite observed reconstruction target: {name}")
            if name == "text" and text_mode == "cosine":
                loss = (1 - F.cosine_similarity(p, t, dim=-1, eps=1e-08)).mean()
            elif text_mode in ("cosine", "smooth_l1"):
                loss = F.smooth_l1_loss(p, t, beta=beta)
            else:
                raise ValueError("text_mode must be cosine or smooth_l1")
        else:
            loss = pred.sum() * 0.0
        total = total + loss
        parts["reconstruction_" + name] = loss
    return (total, parts)


def simsiam_loss(damaged, original, *, original_observed=None):
    """Symmetric original HLFR negative cosine, per-modality + global sum.

    Every target z is stop-gradient; predictions p remain differentiable. Never
    suppress a synthetically missing branch merely because it is now empty: its
    original observed view may provide a legitimate training target. Natural
    all-empty original modalities/samples do not provide such a target.
    """
    names = (
        set(damaged.get("p", {}))
        & set(original.get("z", {}))
        & set(original.get("p", {}))
        & set(damaged.get("z", {}))
    )
    if not names:
        raise ValueError("HLFR requires matching projector/predictor outputs")
    total = damaged["logits"].sum() * 0.0
    parts = {}
    for name in sorted(names):
        loss = -0.5 * (
            F.cosine_similarity(
                damaged["p"][name].float(),
                original["z"][name].detach().float(),
                dim=-1,
                eps=1e-08,
            )
            + F.cosine_similarity(
                original["p"][name].float(),
                damaged["z"][name].detach().float(),
                dim=-1,
                eps=1e-08,
            )
        )
        if original_observed is not None:
            eligible = (
                original_observed.flatten(1).any(-1)
                if name == "global"
                else original_observed[:, MODALITIES.index(name)].any(-1)
            )
            loss = torch.where(eligible, loss, torch.zeros_like(loss))
            value = loss.sum() / eligible.sum().clamp_min(1)
        else:
            value = loss.mean()
        parts["hlfr_" + name] = value
        total = total + value
    return (total, parts)


def compute_loss(
    outputs,
    labels,
    targets,
    config=None,
    *,
    teacher_outputs=None,
    clean_outputs=None,
    original_outputs=None,
    recon_targets=None,
    recon_mask=None,
    original_observed=None,
    current_observed=None,
    class_weights=None,
):
    """Return (scalar differentiable loss, dict of scalar diagnostics).

    Aliases clean_outputs/original_outputs refer to a separate student forward,
    NOT frozen teacher outputs. Do not pass both. All optional terms require an
    explicit positive weight and the corresponding caller-provided data.
    """
    c = _options(config)
    if clean_outputs is not None and original_outputs is not None:
        raise ValueError("supply only one original-view output argument")
    original = original_outputs if original_outputs is not None else clean_outputs
    beta = float(c.get("smooth_l1_beta", 0.5))
    regression_weight = float(c.get("lambda_regression", 1.0))
    total, components = supervised_loss(
        outputs,
        labels,
        targets,
        class_weights=class_weights,
        lambda_regression=regression_weight,
        beta=beta,
    )
    components["supervised"] = total
    if current_observed is None:
        current_observed = outputs.get("mask")
    if original_observed is None and original is not None:
        original_observed = original.get("mask")
    original_weight = float(c.get("lambda_original_view", 0.5))
    if original is not None and original_weight:
        original_loss, _ = supervised_loss(
            original,
            labels,
            targets,
            class_weights=class_weights,
            lambda_regression=regression_weight,
            beta=beta,
        )
        components["original_supervised"] = original_loss
        total = total + original_weight * original_loss
    kd_weight = float(c.get("lambda_kd", 0.1))
    kd_mode = c.get("kd_mode", "fixed")
    if teacher_outputs is not None and kd_weight and (kd_mode not in ("none", "off")):
        loss, parts = kd_loss(
            outputs,
            teacher_outputs,
            labels=labels,
            targets=targets,
            mode=kd_mode,
            temperature=float(c.get("classification_kd_temperature", 2.0)),
            beta=beta,
            original_observed=original_observed,
            current_observed=current_observed,
        )
        components.update({k: v for k, v in parts.items() if k != "kd_weights"})
        components["kd"] = loss
        total = total + kd_weight * loss
    reconstruction_weight = float(
        c.get("lambda_reconstruction", c.get("lambda_reconstruction_default", 0.0))
    )
    if reconstruction_weight:
        if recon_targets is None or recon_mask is None:
            raise ValueError(
                "positive reconstruction weight requires explicit targets and mask"
            )
        loss, parts = reconstruction_loss(
            outputs.get("recon", {}),
            recon_targets,
            recon_mask,
            original_observed=original_observed,
            current_observed=current_observed,
            beta=float(c.get("reconstruction_beta", 1.0)),
            text_mode=c.get("text_reconstruction_mode", "smooth_l1"),
        )
        components.update(parts)
        components["reconstruction"] = loss
        total = total + reconstruction_weight * loss
    hlfr_weight = float(c.get("lambda_hlfr", 0.0))
    if hlfr_weight:
        if original is None:
            raise ValueError(
                "positive HLFR weight requires separate original student forward"
            )
        loss, parts = simsiam_loss(
            outputs, original, original_observed=original_observed
        )
        components.update(parts)
        components["hlfr"] = loss
        total = total + hlfr_weight * loss
    components["total"] = total
    return (total, components)
