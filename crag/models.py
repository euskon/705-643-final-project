"""OpenPoints models with a swappable head: build, load pretrained weights, freeze, save.

Every model is an OpenPoints ``BaseSeg`` = ``encoder`` (the convolutional
backbone: set-abstraction / InvResMLP stages) + ``decoder`` (feature
propagation back to every point) + ``head`` (per-point classifier). The
pretraining head predicts the 4 pseudolabel classes; fine-tuning throws it
away and attaches a fresh head with its own class count. Checkpoints store
the three parts separately (``encoder``/``decoder``/``head``) as well as the
full state, so a backbone can be reused without the head.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn

from crag.paths import PROJECT_DIR, add_import_paths

add_import_paths()
from openpoints.models import build_model_from_cfg  # noqa: E402
from openpoints.transforms import DataTransforms  # noqa: E402

log = logging.getLogger(__name__)

PARTS = ("encoder", "decoder", "head")


def _contiguous_grouping() -> None:
    """OpenPoints' grouping op asserts contiguous inputs; with a single input channel the
    stem convolution can return a non-contiguous tensor. Make them contiguous first."""
    import openpoints.models.layers.group as G

    if getattr(G.grouping_operation, "_crag_wrapped", False):
        return
    orig = G.grouping_operation

    def grouping_operation(features, idx):
        return orig(features.contiguous(), idx.contiguous())

    grouping_operation._crag_wrapped = True
    G.grouping_operation = grouping_operation


_contiguous_grouping()


# Python >= 3.10 removed collections.Iterable, which OpenPoints' PointCloudRotation
# uses; this is the z-only rotation airborne LiDAR needs anyway (gravity stays down).
@DataTransforms.register_module()
class PointCloudRotationZ:
    def __init__(self, **kwargs):
        pass

    def __call__(self, data):
        pos = data["pos"]
        theta = torch.rand(1).item() * 2 * torch.pi
        c, s = torch.cos(torch.tensor(theta)), torch.sin(torch.tensor(theta))
        rot = torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=pos.dtype, device=pos.device)
        data["pos"] = pos @ rot.T
        return data


def build_model(model_cfg) -> nn.Module:
    model = build_model_from_cfg(model_cfg)
    missing = [p for p in PARTS if not hasattr(model, p)]
    if missing:
        raise ValueError(f"expected an OpenPoints BaseSeg with {PARTS}; missing {missing}")
    return model


def _state_dict_from_file(path: Path) -> dict:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if "model" in ckpt or "state_dict" in ckpt:
        sd = ckpt.get("model", ckpt.get("state_dict"))
    elif any(isinstance(ckpt.get(p), dict) for p in PARTS):  # our backbone.pt: {"encoder": {...}, "decoder": {...}}
        sd = {f"{p}.{k}": v for p in PARTS if isinstance(ckpt.get(p), dict) for k, v in ckpt[p].items()}
    else:
        sd = ckpt
    return {k.removeprefix("module."): v for k, v in sd.items() if torch.is_tensor(v)}


def load_pretrained(model: nn.Module, path: str | Path, parts: tuple[str, ...] = ("encoder", "decoder"), strict_shapes: bool = False) -> dict:
    """Copy matching tensors of ``parts`` from a checkpoint (OpenPoints or ours).

    Tensors whose shape differs (e.g. a 13-class S3DIS head vs our 4 classes,
    or a different ``in_channels``) are skipped and reported.
    """
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_DIR / path
    sd = _state_dict_from_file(path)
    own = model.state_dict()
    wanted = {k: v for k, v in sd.items() if k.split(".")[0] in parts}
    loaded = {k: v for k, v in wanted.items() if k in own and own[k].shape == v.shape}
    mismatched = sorted(k for k in wanted if k in own and own[k].shape != wanted[k].shape)
    if strict_shapes and mismatched:
        raise ValueError(f"shape mismatch loading {path}: {mismatched[:5]}")
    model.load_state_dict(loaded, strict=False)
    report = {
        "path": str(path),
        "parts": list(parts),
        "loaded": len(loaded),
        "mismatched": mismatched,
        "not_in_checkpoint": sorted(k for k in own if k.split(".")[0] in parts and k not in sd),
    }
    log.info("pretrained %s: loaded %d tensors (%d shape-mismatched, %d missing)", path.name, len(loaded), len(mismatched), len(report["not_in_checkpoint"]))
    return report


# tensors whose input size depends on the number of feature channels
FIRST_LAYER_PREFIXES = ("encoder.encoder.0.",)  # PointNeXt-S stem; PointNet++ first SA conv (+ MSG branches)
# PointNeXt-S: the stem; PointNet++ (incl. MSG): the first set-abstraction convolution and the
# last decoder step, which concatenates the raw input features back in
FIRST_LAYER_PATTERNS = ("encoder.encoder.0.", "encoder.SA_modules.0.local_aggregations.0.SA_CONFIG_operator.convs.0.",
                        "encoder.SA_modules.0.local_aggregations.1.SA_CONFIG_operator.convs.0.", "decoder.FP_modules.0.convs.0.")


def check_first_layer_only(report: dict | None) -> None:
    """After a channel change only the first layer may fail to load; anything else is an architecture mismatch."""
    if not report:
        return
    bad = [k for k in report["mismatched"] if not k.startswith("head.") and not any(k.startswith(p) for p in FIRST_LAYER_PATTERNS)]
    if bad:
        raise ValueError(f"shape mismatch outside the first layer (architecture mismatch?): {bad[:5]}")
    first = [k for k in report["mismatched"] if not k.startswith("head.")]
    if first:
        log.info("first layer rebuilt (channel count changed): %s", first)


def set_frozen(model: nn.Module, frozen_parts: tuple[str, ...]) -> None:
    """Freeze whole parts (no grads, BatchNorm stats fixed). Call again to unfreeze."""
    model._frozen_parts = tuple(frozen_parts)
    for p in PARTS:
        for prm in getattr(model, p).parameters():
            prm.requires_grad = p not in frozen_parts
    apply_bn_freeze(model)


def apply_bn_freeze(model: nn.Module) -> None:
    """Put frozen parts in eval mode (call after every ``model.train()``)."""
    for p in getattr(model, "_frozen_parts", ()):
        getattr(model, p).eval()


def param_groups(model: nn.Module, lr: float, weight_decay: float, lr_mult: dict | None = None) -> list[dict]:
    """AdamW groups per part, with per-part LR multipliers; no decay on biases / norms."""
    lr_mult = lr_mult or {}
    groups = []
    for part in PARTS:
        decay, no_decay = [], []
        for name, prm in getattr(model, part).named_parameters():
            if not prm.requires_grad:
                continue
            (no_decay if prm.ndim <= 1 or name.endswith(".bias") else decay).append(prm)
        m = float(lr_mult.get(part, 1.0))
        for params, wd in ((decay, weight_decay), (no_decay, 0.0)):
            if params:
                groups.append({"params": params, "lr": lr * m, "base_lr": lr * m, "weight_decay": wd, "part": part})
    return groups


def split_state(model: nn.Module) -> dict:
    return {p: getattr(model, p).state_dict() for p in PARTS}


def count_parameters(model: nn.Module) -> dict:
    return {
        p: (sum(x.numel() for x in getattr(model, p).parameters() if x.requires_grad), sum(x.numel() for x in getattr(model, p).parameters()))
        for p in PARTS
    }
