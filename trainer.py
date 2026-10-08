"""Small training / evaluation harness for the climbable-rock patch classifier.

A trimmed-down version of the assignment harness: same per-run output folder,
log, metrics CSV, history plot and best checkpoint, without W&B or resume
(each run is minutes, not hours).

Example::

    from trainer import TrainConfig, Trainer, compare_modes

    trainer = Trainer(TrainConfig(run_name="finetune", mode="finetune", epochs=40))
    trainer.fit(loaders["train"], loaders["val"])
    trainer.evaluate(loaders["test"])

    # or all three fine-tuning modes on the same split in one go
    results = compare_modes(loaders, TrainConfig(epochs=40))

Every run writes into ``<output_dir>/<run_name>/``: ``train.log``,
``metrics.csv`` (one row per epoch), ``history.png``, ``best.pt`` (weights
with the best validation balanced accuracy) and ``summary.json``.
"""

from __future__ import annotations

import json
import logging
import random
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader

from config import PROJECT_DIR
from model import build_model, count_parameters

MODES = ("scratch", "head", "finetune")


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_device(preference: str = "auto") -> torch.device:
    if preference != "auto":
        return torch.device(preference)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass
class TrainConfig:
    run_name: str = "finetune"
    mode: str = "finetune"  # scratch | head | finetune (see model.set_trainable)
    epochs: int = 40
    lr: float = 1e-3
    # backbone LR = lr * backbone_lr_mult in "finetune" mode, so the pretrained
    # filters move slowly while the new head catches up
    backbone_lr_mult: float = 0.1
    weight_decay: float = 1e-4
    # weight the loss by inverse class frequency (there are several
    # negatives per positive)
    class_weighted: bool = True
    early_stop_patience: int = 12
    seed: int = 42
    device: str = "auto"
    output_dir: str = str(PROJECT_DIR / "runs")


def binary_metrics(
    labels: np.ndarray, probs: np.ndarray, threshold: float = 0.5
) -> dict:
    """Metrics for P(climbable); balanced accuracy and AUROC cope with class imbalance."""
    preds = (probs >= threshold).astype(int)
    out = {
        "acc": float((preds == labels).mean()),
        "balanced_acc": float(balanced_accuracy_score(labels, preds)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "auroc": float(roc_auc_score(labels, probs))
        if len(np.unique(labels)) == 2
        else float("nan"),
    }
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    out.update(tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))
    return out


class Trainer:
    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        seed_everything(cfg.seed)
        self.device = pick_device(cfg.device)
        self.run_dir = Path(cfg.output_dir) / cfg.run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.log = self._make_logger()
        self.model, self.pretrained_report = build_model(cfg.mode)
        self.model.to(self.device)
        trainable, total = count_parameters(self.model)
        self.log.info(
            "run=%s mode=%s device=%s params %d/%d trainable",
            cfg.run_name,
            cfg.mode,
            self.device,
            trainable,
            total,
        )
        if self.pretrained_report:
            self.log.info("pretrained: %s", self.pretrained_report)
        self.optimizer = torch.optim.AdamW(
            self._param_groups(), weight_decay=cfg.weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max(cfg.epochs, 1)
        )
        self.history: list[dict] = []

    def _make_logger(self) -> logging.Logger:
        log = logging.getLogger(f"trainer.{self.cfg.run_name}")
        log.setLevel(logging.INFO)
        log.handlers.clear()
        fmt = logging.Formatter("%(asctime)s %(message)s", "%H:%M:%S")
        for h in (
            logging.FileHandler(self.run_dir / "train.log", mode="w"),
            logging.StreamHandler(),
        ):
            h.setFormatter(fmt)
            log.addHandler(h)
        log.propagate = False
        return log

    def _param_groups(self) -> list[dict]:
        backbone = [
            p for m in self.model.backbone for p in m.parameters() if p.requires_grad
        ]
        backbone_ids = {id(p) for p in backbone}
        head = [
            p
            for p in self.model.parameters()
            if p.requires_grad and id(p) not in backbone_ids
        ]
        groups = [{"params": head, "lr": self.cfg.lr}]
        if backbone:
            mult = self.cfg.backbone_lr_mult if self.cfg.mode == "finetune" else 1.0
            groups.append({"params": backbone, "lr": self.cfg.lr * mult})
        return groups

    def _class_weights(self, loader: DataLoader) -> torch.Tensor | None:
        if not self.cfg.class_weighted:
            return None
        counts = np.bincount(loader.dataset.labels, minlength=2).astype(float)
        weights = counts.sum() / (2 * np.maximum(counts, 1))
        return torch.tensor(weights, dtype=torch.float32, device=self.device)

    @torch.no_grad()
    def predict(self, loader: DataLoader) -> tuple[np.ndarray, np.ndarray, float]:
        """(labels, P(climbable), mean loss) over a loader."""
        self.model.eval()
        labels, probs, losses = [], [], []
        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)
            logits = self.model(x)
            losses.append(F.cross_entropy(logits, y, reduction="sum").item())
            probs.append(torch.softmax(logits, 1)[:, 1].cpu().numpy())
            labels.append(y.cpu().numpy())
        labels, probs = np.concatenate(labels), np.concatenate(probs)
        return labels, probs, sum(losses) / max(len(labels), 1)

    def evaluate(self, loader: DataLoader) -> dict:
        labels, probs, loss = self.predict(loader)
        return {"loss": loss, **binary_metrics(labels, probs)}

    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> pd.DataFrame:
        weights = self._class_weights(train_loader)
        best, since_best = -np.inf, 0
        for epoch in range(1, self.cfg.epochs + 1):
            t0 = time.time()
            self.model.train()
            total, n = 0.0, 0
            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)
                loss = F.cross_entropy(self.model(x), y, weight=weights)
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()
                total += loss.item() * len(y)
                n += len(y)
            self.scheduler.step()
            val = self.evaluate(val_loader)
            row = {
                "epoch": epoch,
                "train_loss": total / max(n, 1),
                **{f"val_{k}": v for k, v in val.items()},
            }
            row["lr"] = self.optimizer.param_groups[0]["lr"]
            row["time_s"] = time.time() - t0
            self.history.append(row)
            score = val["balanced_acc"] + 1e-3 * np.nan_to_num(
                val["auroc"]
            )  # AUROC breaks ties
            if score > best:
                best, since_best = score, 0
                torch.save(
                    {
                        "model": self.model.state_dict(),
                        "config": asdict(self.cfg),
                        "epoch": epoch,
                    },
                    self.run_dir / "best.pt",
                )
            else:
                since_best += 1
            self.log.info(
                "epoch %3d  train_loss %.4f  val_loss %.4f  val_bal_acc %.3f  val_auroc %.3f  val_f1 %.3f  %.1fs%s",
                epoch,
                row["train_loss"],
                val["loss"],
                val["balanced_acc"],
                val["auroc"],
                val["f1"],
                row["time_s"],
                "  *" if since_best == 0 else "",
            )
            self._write_history()
            if since_best >= self.cfg.early_stop_patience:
                self.log.info(
                    "early stop: no val improvement for %d epochs", since_best
                )
                break
        self.load_best()
        return pd.DataFrame(self.history)

    def load_best(self) -> None:
        state = torch.load(
            self.run_dir / "best.pt", map_location=self.device, weights_only=True
        )
        self.model.load_state_dict(state["model"])

    def _write_history(self) -> None:
        df = pd.DataFrame(self.history)
        df.to_csv(self.run_dir / "metrics.csv", index=False)
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
        axes[0].plot(df["epoch"], df["train_loss"], label="train")
        axes[0].plot(df["epoch"], df["val_loss"], label="val")
        axes[0].set(xlabel="epoch", ylabel="cross-entropy", title="loss")
        axes[0].legend()
        for k in ("val_balanced_acc", "val_auroc", "val_f1"):
            axes[1].plot(df["epoch"], df[k], label=k.removeprefix("val_"))
        axes[1].set(xlabel="epoch", ylim=(0, 1.02), title="validation")
        axes[1].legend()
        fig.suptitle(self.cfg.run_name)
        fig.tight_layout()
        fig.savefig(self.run_dir / "history.png", dpi=110)
        plt.close(fig)

    def final_report(self, loaders: dict[str, DataLoader]) -> dict:
        """Best-checkpoint metrics on every split, written to ``summary.json``."""
        summary = {
            "run": self.cfg.run_name,
            "mode": self.cfg.mode,
            "epochs_run": len(self.history),
        }
        for split in ("train", "val", "test"):
            summary[split] = self.evaluate(loaders[split])
        (self.run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        return summary


def compare_modes(
    loaders: dict[str, DataLoader], base: TrainConfig, modes=MODES
) -> tuple[pd.DataFrame, dict]:
    """Train one model per fine-tuning mode on the same split; one row per (mode, split)."""
    rows, trainers = [], {}
    for mode in modes:
        trainer = Trainer(
            replace(
                base,
                run_name=f"{base.run_name}-{mode}" if base.run_name else mode,
                mode=mode,
            )
        )
        trainer.fit(loaders["train"], loaders["val"])
        report = trainer.final_report(loaders)
        trainers[mode] = trainer
        for split in ("val", "test"):
            rows.append({"mode": mode, "split": split, **report[split]})
    return pd.DataFrame(rows), trainers
