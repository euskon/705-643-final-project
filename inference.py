"""Sliding-window inference: P(climbable) on a regular grid over the region.

Each grid node gets the same input the model was trained on - the vertical
column of points within ``PatchConfig.radius_m`` - so the output is directly
comparable to the labelled patches.

Example::

    grid = predict_grid(trainer.model, pc, patch_cfg, spacing_m=5.0)
    plot_probability_map(grid, positives)
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from config import PatchConfig
from dataset import normalize_patch, resample
from labeling import PointCloud, extract_patch


@torch.no_grad()
def predict_grid(
    model: torch.nn.Module,
    pc: PointCloud,
    cfg: PatchConfig,
    spacing_m: float = 5.0,
    bounds_xy: tuple[float, float, float, float] | None = None,
    batch_size: int = 64,
) -> pd.DataFrame:
    """One row per grid node with enough points: ``x, y, n_points, prob``."""
    device = next(model.parameters()).device
    minx, miny, maxx, maxy = bounds_xy or pc.bounds_xy
    xs = np.arange(minx + cfg.radius_m, maxx - cfg.radius_m, spacing_m)
    ys = np.arange(miny + cfg.radius_m, maxy - cfg.radius_m, spacing_m)
    nodes = np.array(np.meshgrid(xs, ys)).reshape(2, -1).T
    rng = np.random.default_rng(0)
    model.eval()
    rows, batch, batch_rows = [], [], []

    def flush():
        x = torch.from_numpy(np.stack(batch)).to(device)
        probs = torch.softmax(model(x), 1)[:, 1].cpu().numpy()
        for r, p in zip(batch_rows, probs):
            rows.append({**r, "prob": float(p)})
        batch.clear()
        batch_rows.clear()

    for c in tqdm(nodes, desc="grid"):
        pts = extract_patch(pc, c, cfg)
        if len(pts) < cfg.min_points:
            continue
        pts = normalize_patch(resample(pts, cfg.num_points, rng), cfg)
        batch.append(np.ascontiguousarray(pts.T, dtype=np.float32))
        batch_rows.append({"x": c[0], "y": c[1], "n_points": len(pts)})
        if len(batch) == batch_size:
            flush()
    if batch:
        flush()
    return pd.DataFrame(rows)


def plot_probability_map(
    grid: pd.DataFrame, samples: pd.DataFrame | None = None, ax=None, title: str = ""
):
    """Heat map of P(climbable) with the labelled patches overlaid."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8))
    sc = ax.scatter(
        grid["x"],
        grid["y"],
        c=grid["prob"],
        s=6,
        marker="s",
        cmap="magma",
        vmin=0,
        vmax=1,
    )
    plt.colorbar(sc, ax=ax, label="P(climbable)", shrink=0.7)
    if samples is not None:
        for split, marker in (("train", "o"), ("val", "s"), ("test", "^")):
            pos = samples[
                (samples["label"] == 1) & (samples.get("split", "train") == split)
            ]
            if len(pos):
                ax.scatter(
                    pos["x"],
                    pos["y"],
                    s=14,
                    facecolors="none",
                    edgecolors="cyan",
                    marker=marker,
                    linewidths=0.8,
                    label=f"MP route ({split})",
                )
        ax.legend(loc="lower left", fontsize=8)
    ax.set(
        aspect="equal", xlabel="UTM 15N easting (m)", ylabel="northing (m)", title=title
    )
    return ax
