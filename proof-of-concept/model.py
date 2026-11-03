"""PointNet++ (single-scale grouping) classifier + loading pretrained weights.

The architecture is adapted from yanx27/Pointnet_Pointnet2_pytorch
(MIT License, Copyright (c) 2019 benny): ``models/pointnet2_cls_ssg.py`` and
``models/pointnet2_utils.py``. Layer names are kept identical so its
ModelNet40 checkpoint (``pointnet2_ssg_wo_normals``, xyz-only input, 92.2%
instance accuracy) loads straight into :class:`PointNet2Cls`; only the final
40-way ``fc3`` is replaced by a fresh ``num_classes``-way layer.

Fine-tuning modes (see :func:`set_trainable`):

* ``"scratch"``  - random init, everything trains (the baseline)
* ``"head"``     - pretrained set-abstraction backbone frozen (BatchNorm
                   statistics too), only fc1-fc3 train - a "linear-ish probe"
* ``"finetune"`` - pretrained init, everything trains (backbone at a lower LR)
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import PRETRAINED_DIR

PRETRAINED_URLS = {
    "modelnet40_ssg": (
        "https://github.com/yanx27/Pointnet_Pointnet2_pytorch/raw/master/"
        "log/classification/pointnet2_ssg_wo_normals/checkpoints/best_model.pth"
    ),
}


# ---------------------------------------------------------------------------
# PointNet++ building blocks (yanx27, lightly tidied)
# ---------------------------------------------------------------------------


def square_distance(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    """Pairwise squared distances, [B, N, C] x [B, M, C] -> [B, N, M]."""
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src**2, -1).unsqueeze(-1)
    dist += torch.sum(dst**2, -1).unsqueeze(1)
    return dist


def index_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Gather ``points[b, idx[b, ...]]`` for every batch element."""
    B = points.shape[0]
    view_shape = [B] + [1] * (idx.dim() - 1)
    batch = torch.arange(B, device=points.device).view(view_shape).expand_as(idx)
    return points[batch, idx, :]


def farthest_point_sample(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
    """Indices [B, npoint] of a farthest-point subsample of ``xyz`` [B, N, 3]."""
    B, N, _ = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=xyz.device)
    distance = torch.full((B, N), 1e10, device=xyz.device)
    farthest = torch.randint(0, N, (B,), device=xyz.device)
    batch = torch.arange(B, device=xyz.device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch, farthest, :].view(B, 1, 3)
        distance = torch.minimum(distance, torch.sum((xyz - centroid) ** 2, -1))
        farthest = torch.max(distance, -1)[1]
    return centroids


def query_ball_point(
    radius: float, nsample: int, xyz: torch.Tensor, new_xyz: torch.Tensor
) -> torch.Tensor:
    """Up to ``nsample`` neighbour indices within ``radius`` of each query point."""
    B, N, _ = xyz.shape
    S = new_xyz.shape[1]
    group_idx = (
        torch.arange(N, dtype=torch.long, device=xyz.device)
        .view(1, 1, N)
        .repeat(B, S, 1)
    )
    group_idx[square_distance(new_xyz, xyz) > radius**2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    # pad short neighbourhoods with their first (always present) neighbour
    group_first = group_idx[:, :, :1].expand(-1, -1, nsample)
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx


class PointNetSetAbstraction(nn.Module):
    """Sample centroids (FPS) -> group neighbours (ball query) -> shared MLP -> max-pool."""

    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
        super().__init__()
        self.npoint, self.radius, self.nsample, self.group_all = (
            npoint,
            radius,
            nsample,
            group_all,
        )
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last = in_channel
        for out in mlp:
            self.mlp_convs.append(nn.Conv2d(last, out, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out))
            last = out

    def forward(self, xyz: torch.Tensor, points: torch.Tensor | None):
        """xyz [B, 3, N], points [B, D, N] -> new_xyz [B, 3, S], new_points [B, D', S]."""
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)
        B, N, C = xyz.shape
        if self.group_all:
            new_xyz = torch.zeros(B, 1, C, device=xyz.device)
            grouped = xyz.view(B, 1, N, C)
            if points is not None:
                grouped = torch.cat([grouped, points.view(B, 1, N, -1)], dim=-1)
        else:
            new_xyz = index_points(xyz, farthest_point_sample(xyz, self.npoint))
            idx = query_ball_point(self.radius, self.nsample, xyz, new_xyz)
            grouped = index_points(xyz, idx) - new_xyz.unsqueeze(2)
            if points is not None:
                grouped = torch.cat([grouped, index_points(points, idx)], dim=-1)
        x = grouped.permute(0, 3, 2, 1)  # [B, C+D, nsample, S]
        for conv, bn in zip(self.mlp_convs, self.mlp_bns):
            x = F.relu(bn(conv(x)))
        return new_xyz.permute(0, 2, 1), torch.max(x, 2)[0]


class PointNet2Cls(nn.Module):
    """PointNet++ SSG classifier on xyz-only input [B, 3, N]; returns logits [B, num_classes].

    The set-abstraction radii (0.2, 0.4) are in the units of the input, so
    patches must be normalised the way ModelNet40 was (unit sphere) for the
    pretrained filters to see neighbourhoods of the size they were trained on.
    """

    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.sa1 = PointNetSetAbstraction(
            512, 0.2, 32, 3, [64, 64, 128], group_all=False
        )
        self.sa2 = PointNetSetAbstraction(
            128, 0.4, 64, 128 + 3, [128, 128, 256], group_all=False
        )
        self.sa3 = PointNetSetAbstraction(
            None, None, None, 256 + 3, [256, 512, 1024], group_all=True
        )
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(256, num_classes)
        self.frozen_backbone = False

    @property
    def backbone(self) -> list[nn.Module]:
        return [self.sa1, self.sa2, self.sa3]

    def embed(self, xyz: torch.Tensor) -> torch.Tensor:
        """Global 1024-d feature of each point set."""
        l1_xyz, l1_points = self.sa1(xyz, None)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        _, l3_points = self.sa3(l2_xyz, l2_points)
        return l3_points.view(xyz.shape[0], 1024)

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        x = self.embed(xyz)
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        return self.fc3(x)

    def train(self, mode: bool = True):
        super().train(mode)
        # a frozen backbone must not keep updating its BatchNorm running stats
        if self.frozen_backbone:
            for m in self.backbone:
                m.eval()
        return self


# ---------------------------------------------------------------------------
# pretrained weights
# ---------------------------------------------------------------------------


def download_pretrained(
    name: str = "modelnet40_ssg", dest_dir: Path = PRETRAINED_DIR
) -> Path:
    """Fetch a yanx27 checkpoint once and return its local path."""
    path = Path(dest_dir) / f"{name}.pth"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(PRETRAINED_URLS[name], path)
    return path


def load_pretrained(model: PointNet2Cls, name: str = "modelnet40_ssg") -> dict:
    """Copy every pretrained tensor whose name and shape match; ``fc3`` stays fresh.

    Returns a small report of what was loaded and skipped.
    """
    # the checkpoint stores its accuracies as numpy scalars; allow exactly
    # those types instead of falling back to an unrestricted pickle load
    numpy_scalars = [
        (np._core.multiarray.scalar, "numpy.core.multiarray.scalar"),
        np.dtype,
        type(np.dtype(np.float64)),
    ]
    with torch.serialization.safe_globals(numpy_scalars):
        ckpt = torch.load(
            download_pretrained(name), map_location="cpu", weights_only=True
        )
    pretrained = ckpt.get("model_state_dict", ckpt)
    own = model.state_dict()
    loaded = {
        k: v for k, v in pretrained.items() if k in own and own[k].shape == v.shape
    }
    model.load_state_dict(loaded, strict=False)
    return {
        "checkpoint": name,
        "loaded": len(loaded),
        "skipped": sorted(set(pretrained) - set(loaded)),
        "fresh": sorted(set(own) - set(loaded)),
        "pretrain_epoch": ckpt.get("epoch"),
        "pretrain_instance_acc": float(ckpt.get("instance_acc", float("nan"))),
    }


def set_trainable(model: PointNet2Cls, mode: str) -> None:
    """Apply one of the fine-tuning modes from the module docstring."""
    if mode not in {"scratch", "head", "finetune"}:
        raise ValueError(f"unknown mode {mode!r}")
    model.frozen_backbone = mode == "head"
    for m in model.backbone:
        for p in m.parameters():
            p.requires_grad = not model.frozen_backbone
    model.train(model.training)


def build_model(mode: str, num_classes: int = 2) -> tuple[PointNet2Cls, dict | None]:
    """Model for a fine-tuning mode, plus the pretrained-load report (None for scratch)."""
    model = PointNet2Cls(num_classes)
    report = None if mode == "scratch" else load_pretrained(model)
    set_trainable(model, mode)
    return model, report


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """(trainable, total) parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total
