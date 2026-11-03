"""One training loop for pretraining and fine-tuning (per-point segmentation).

Features required by the plan:

* **AMP** (``use_amp``) - fp16 autocast + GradScaler; OpenPoints' CUDA ops
  cast their own inputs to fp32 where needed.
* **Gradient accumulation** (``grad_accum_steps``) - small GPU batches (4-8
  blocks of 8192 points fit a 10 GB RTX 3080) with an effective batch of
  ``batch_size * grad_accum_steps``.
* **Checkpointing on validation loss** - ``best.pt`` is written only when
  ``val_loss < best_val_loss - checkpoint.min_delta`` (default 0.001);
  ``last.pt`` every epoch (and on Ctrl+C) for resuming multi-day runs.
  Checkpoints hold the full state plus ``encoder``/``decoder``/``head``
  separately; ``backbone.pt`` holds encoder+decoder only.
* **W&B** - config, per-step train loss/LR, per-epoch val metrics, the best
  checkpoint as an artifact at the end.
* **Resume** - ``resume: auto`` picks up ``<run_dir>/last.pt``.

The data side is pluggable: pass datasets that yield OpenPoints dicts.
"""

from __future__ import annotations

import json
import logging
import math
import signal
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from crag import models, wandb_utils

log = logging.getLogger(__name__)


def seed_everything(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def cfg_get(cfg, key, default=None):
    """Nested ``a.b.c`` lookup on EasyConfig / dict."""
    cur = cfg
    for part in key.split("."):
        if cur is None:
            return default
        cur = cur.get(part) if hasattr(cur, "get") else getattr(cur, part, None)
    return default if cur is None else cur


def to_plain(cfg):
    """EasyConfig -> plain dict (for JSON / W&B)."""
    if hasattr(cfg, "items"):
        return {k: to_plain(v) for k, v in cfg.items()}
    if isinstance(cfg, (list, tuple)):
        return [to_plain(v) for v in cfg]
    return cfg


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------


class ConfusionMeter:
    def __init__(self, num_classes: int, ignore_index: int, device):
        self.n, self.ignore = num_classes, ignore_index
        self.mat = torch.zeros(num_classes, num_classes, dtype=torch.long, device=device)

    @torch.no_grad()
    def update(self, logits: torch.Tensor, target: torch.Tensor) -> None:
        pred = logits.argmax(1).flatten()
        t = target.flatten()
        ok = t != self.ignore
        self.mat += torch.bincount(t[ok] * self.n + pred[ok], minlength=self.n**2).reshape(self.n, self.n)

    def compute(self, class_names) -> dict:
        m = self.mat.double().cpu().numpy()
        tp = np.diag(m)
        gt, pr = m.sum(1), m.sum(0)
        with np.errstate(invalid="ignore", divide="ignore"):
            iou = tp / (gt + pr - tp)
            recall = tp / gt
        present = gt > 0
        out = {
            "oa": float(tp.sum() / max(m.sum(), 1)),
            "miou": float(np.nanmean(iou[present])) if present.any() else float("nan"),
            "macc": float(np.nanmean(recall[present])) if present.any() else float("nan"),
        }
        for i, c in enumerate(class_names):
            out[f"iou_{c}"] = float(iou[i]) if present[i] else float("nan")
            out[f"recall_{c}"] = float(recall[i]) if present[i] else float("nan")
        out["confusion"] = m.astype(int).tolist()
        return out


# ---------------------------------------------------------------------------
# trainer
# ---------------------------------------------------------------------------


@dataclass
class EpochResult:
    epoch: int
    train_loss: float
    val: dict
    improved: bool


class Trainer:
    def __init__(self, cfg, model: torch.nn.Module, train_ds, val_ds, class_names, run_dir: Path, train_sampler=None, class_weights=None, val_hook=None):
        self.cfg = cfg
        self.class_names = list(class_names)
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.ignore_index = int(cfg_get(cfg, "ignore_index", -1))
        self.use_amp = bool(cfg_get(cfg, "use_amp", True)) and self.device.type == "cuda"
        self.accum = int(cfg_get(cfg, "grad_accum_steps", 1))
        self.val_hook = val_hook  # fn(trainer, logits_list...) -> extra metrics (fine-tuning)
        nw = int(cfg_get(cfg, "dataloader.num_workers", 4))
        self.train_loader = DataLoader(
            train_ds,
            batch_size=int(cfg.batch_size),
            sampler=train_sampler,
            shuffle=train_sampler is None,
            drop_last=True,  # BatchNorm needs > 1 sample
            num_workers=nw,
            pin_memory=self.device.type == "cuda",
            persistent_workers=nw > 0,
            worker_init_fn=_worker_init,
        )
        self.val_loader = DataLoader(
            val_ds,
            batch_size=int(cfg_get(cfg, "val_batch_size", cfg.batch_size)),
            shuffle=False,
            num_workers=nw,
            pin_memory=self.device.type == "cuda",
            persistent_workers=nw > 0,
        )
        self.class_weights = None if class_weights is None else torch.as_tensor(class_weights, dtype=torch.float32, device=self.device)
        self.label_smoothing = float(cfg_get(cfg, "criterion_args.label_smoothing", 0.0))
        self.optimizer = torch.optim.AdamW(
            models.param_groups(model, float(cfg_get(cfg, "optimizer.lr", 1e-3)), float(cfg_get(cfg, "optimizer.weight_decay", 1e-4)), cfg_get(cfg, "lr_mult")),
        )
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        self.epochs = int(cfg.epochs)
        self.steps_per_epoch = max(len(self.train_loader) // self.accum, 1)
        self.total_steps = self.epochs * self.steps_per_epoch
        self.warmup_steps = int(cfg_get(cfg, "warmup_epochs", 0) * self.steps_per_epoch)
        self.min_lr_ratio = float(cfg_get(cfg, "min_lr", 1e-5)) / float(cfg_get(cfg, "optimizer.lr", 1e-3))
        self.min_delta = float(cfg_get(cfg, "checkpoint.min_delta", 0.001))
        self.patience = int(cfg_get(cfg, "early_stop_patience", 0))  # 0 = off
        self.max_steps = int(cfg_get(cfg, "max_steps", 0))  # smoke tests: stop after N optimizer steps
        self.limit_val_batches = int(cfg_get(cfg, "limit_val_batches", 0))
        self.log_every = int(cfg_get(cfg, "log_every", 50))
        self.unfreeze_schedule = cfg_get(cfg, "unfreeze_schedule", None)  # {epoch: [parts to unfreeze]}
        self.epoch, self.step, self.best_val, self.since_best = 0, 0, math.inf, 0
        self.history: list[dict] = []
        self.wandb_run = None
        self._stop = False

    # -- schedule -------------------------------------------------------------
    def lr_factor(self, step: int) -> float:
        if step < self.warmup_steps:
            return (step + 1) / max(self.warmup_steps, 1)
        t = (step - self.warmup_steps) / max(self.total_steps - self.warmup_steps, 1)
        return self.min_lr_ratio + (1 - self.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * min(t, 1.0)))

    def _set_lr(self) -> None:
        f = self.lr_factor(self.step)
        for g in self.optimizer.param_groups:
            g["lr"] = g["base_lr"] * f

    # -- checkpoints ------------------------------------------------------------
    def _ckpt(self) -> dict:
        return {
            "model": self.model.state_dict(),
            "parts": models.split_state(self.model),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
            "epoch": self.epoch,
            "step": self.step,
            "best_val_loss": self.best_val,
            "since_best": self.since_best,
            "history": self.history,
            "class_names": self.class_names,
            "cfg": to_plain(self.cfg),
            "frozen_parts": list(getattr(self.model, "_frozen_parts", ())),
        }

    def save(self, name: str) -> Path:
        path = self.run_dir / name
        tmp = path.with_suffix(".tmp")
        torch.save(self._ckpt(), tmp)
        tmp.replace(path)
        return path

    def save_best(self) -> None:
        self.save("best.pt")
        parts = models.split_state(self.model)
        torch.save({"encoder": parts["encoder"], "decoder": parts["decoder"], "cfg": to_plain(self.cfg), "epoch": self.epoch, "val_loss": self.best_val}, self.run_dir / "backbone.pt")

    def resume(self, path: Path) -> None:
        ck = torch.load(path, map_location=self.device, weights_only=False)
        if ck.get("frozen_parts") is not None:
            models.set_frozen(self.model, tuple(ck["frozen_parts"]))
            self.optimizer = torch.optim.AdamW(models.param_groups(self.model, float(cfg_get(self.cfg, "optimizer.lr", 1e-3)), float(cfg_get(self.cfg, "optimizer.weight_decay", 1e-4)), cfg_get(self.cfg, "lr_mult")))
        self.model.load_state_dict(ck["model"])
        try:
            self.optimizer.load_state_dict(ck["optimizer"])
        except ValueError:
            log.warning("optimizer state does not match (param groups changed) - starting it fresh")
        self.scaler.load_state_dict(ck["scaler"])
        self.epoch, self.step = ck["epoch"], ck["step"]
        self.best_val, self.since_best = ck["best_val_loss"], ck.get("since_best", 0)
        self.history = ck.get("history", [])
        log.info("resumed from %s at epoch %d (best val loss %.4f)", path, self.epoch, self.best_val)

    # -- loss / eval --------------------------------------------------------------
    def loss_fn(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits.float(), y, weight=self.class_weights, ignore_index=self.ignore_index, label_smoothing=self.label_smoothing)

    def forward(self, batch: dict) -> torch.Tensor:
        return model_forward(self.model, batch, self.device)

    def evaluate(self, loader: DataLoader | None = None, limit: int = 0) -> dict:
        return run_eval(self.model, loader or self.val_loader, self.class_names, self.ignore_index, self.device, self.use_amp, limit, self.val_hook)

    # -- main loop --------------------------------------------------------------
    def fit(self, wandb_mode: str = "online", wandb_project: str = wandb_utils.DEFAULT_PROJECT, run_name: str | None = None, job_type: str = "pretrain") -> dict:
        resume = cfg_get(self.cfg, "resume", "auto")
        last = self.run_dir / "last.pt"
        if resume == "auto" and last.exists():
            self.resume(last)
        elif resume not in (None, "", "auto", "none"):
            self.resume(Path(resume))
        (self.run_dir / "config.json").write_text(json.dumps(to_plain(self.cfg), indent=1, default=str))
        self.wandb_run = wandb_utils.init(
            wandb_mode,
            wandb_project,
            name=run_name or self.run_dir.name,
            id=self._wandb_id(),
            resume="allow",
            job_type=job_type,
            config=to_plain(self.cfg),
            group=cfg_get(self.cfg, "wandb.group"),
            tags=cfg_get(self.cfg, "wandb.tags"),
        )
        old = signal.signal(signal.SIGINT, self._on_sigint)
        try:
            while self.epoch < self.epochs and not self._stop:
                self._apply_unfreeze()
                res = self._train_one_epoch()
                if res is None:
                    break
                self._log_epoch(res)
                if self.patience and self.since_best >= self.patience:
                    log.info("early stop: no val-loss improvement > %.4f for %d epochs", self.min_delta, self.patience)
                    break
                if self.max_steps and self.step >= self.max_steps:
                    break
        finally:
            self.save("last.pt")
            signal.signal(signal.SIGINT, old)
        summary = {"best_val_loss": self.best_val, "epochs_run": self.epoch, "steps": self.step, "run_dir": str(self.run_dir)}
        (self.run_dir / "summary.json").write_text(json.dumps(summary, indent=1))
        if self.wandb_run is not None:
            if (self.run_dir / "best.pt").exists() and cfg_get(self.cfg, "wandb.log_best_artifact", True):
                import wandb

                art = wandb.Artifact(f"{self.run_dir.name}-best", type="model", metadata=summary)
                art.add_file(str(self.run_dir / "best.pt"))
                art.add_file(str(self.run_dir / "backbone.pt"))
                self.wandb_run.log_artifact(art)
            self.wandb_run.summary.update(summary)
            self.wandb_run.finish()
        return summary

    def _wandb_id(self) -> str:
        p = self.run_dir / "wandb_id.txt"
        if p.exists():
            return p.read_text().strip()
        import secrets

        wid = f"{self.run_dir.name[:40]}-{secrets.token_hex(3)}"
        p.write_text(wid)
        return wid

    def _on_sigint(self, *_):
        log.warning("interrupt: finishing the current step, saving last.pt (Ctrl+C again to abort)")
        self._stop = True
        signal.signal(signal.SIGINT, signal.default_int_handler)

    def _apply_unfreeze(self) -> None:
        if not self.unfreeze_schedule:
            return
        frozen = list(getattr(self.model, "_frozen_parts", ()))
        changed = False
        for ep, parts in dict(self.unfreeze_schedule).items():
            if self.epoch >= int(ep):
                for p in parts:
                    if p in frozen:
                        frozen.remove(p)
                        changed = True
        if changed:
            models.set_frozen(self.model, tuple(frozen))
            self.optimizer = torch.optim.AdamW(models.param_groups(self.model, float(cfg_get(self.cfg, "optimizer.lr", 1e-3)), float(cfg_get(self.cfg, "optimizer.weight_decay", 1e-4)), cfg_get(self.cfg, "lr_mult")))
            log.info("epoch %d: unfroze -> frozen parts now %s", self.epoch, frozen)

    def _train_one_epoch(self) -> EpochResult | None:
        self.model.train()
        models.apply_bn_freeze(self.model)
        t0, total, n = time.time(), 0.0, 0
        self.optimizer.zero_grad(set_to_none=True)
        for k, batch in enumerate(self.train_loader):
            y = batch["y"].to(self.device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=self.use_amp):
                logits = self.forward(batch)
            loss = self.loss_fn(logits, y)
            if not torch.isfinite(loss):
                log.warning("non-finite loss at step %d (all points ignored?) - skipping batch", self.step)
                self.optimizer.zero_grad(set_to_none=True)
                continue
            self.scaler.scale(loss / self.accum).backward()
            total += float(loss) * len(y)
            n += len(y)
            if (k + 1) % self.accum:
                continue
            self._set_lr()
            self.scaler.unscale_(self.optimizer)
            gn = torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(cfg_get(self.cfg, "grad_norm_clip", 10.0)))
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)
            self.step += 1
            if self.step % self.log_every == 0:
                lr = self.optimizer.param_groups[0]["lr"]
                log.info("epoch %d step %d  loss %.4f  lr %.2e  grad_norm %.2f  %.1f samples/s", self.epoch + 1, self.step, total / max(n, 1), lr, float(gn), n / (time.time() - t0))
                if self.wandb_run is not None:
                    self.wandb_run.log({"train/loss": total / max(n, 1), "train/lr": lr, "train/grad_norm": float(gn), "epoch": self.epoch + k / len(self.train_loader)}, step=self.step)
            if self._stop or (self.max_steps and self.step >= self.max_steps):
                break
        if n == 0:
            log.error("no training batches - is the training split empty?")
            return None
        self.epoch += 1
        val = self.evaluate(limit=self.limit_val_batches)
        improved = val["loss"] < self.best_val - self.min_delta
        if improved:
            self.best_val, self.since_best = val["loss"], 0
            self.save_best()
        else:
            self.since_best += 1
        self.save("last.pt")
        return EpochResult(self.epoch, total / n, val, improved)

    def _log_epoch(self, r: EpochResult) -> None:
        row = {"epoch": r.epoch, "step": self.step, "train_loss": r.train_loss, **{f"val_{k}": v for k, v in r.val.items() if k != "confusion"}}
        self.history.append(row)
        with open(self.run_dir / "metrics.jsonl", "a") as fh:
            fh.write(json.dumps(row, default=float) + "\n")
        per_class = "  ".join(f"{c}={r.val.get(f'iou_{c}', float('nan')):.3f}" for c in self.class_names)
        log.info(
            "epoch %d  train_loss %.4f  val_loss %.4f  mIoU %.3f  OA %.3f  [%s]%s",
            r.epoch, r.train_loss, r.val["loss"], r.val["miou"], r.val["oa"], per_class,
            f"  * saved best (Δ>{self.min_delta})" if r.improved else f"  (no improvement for {self.since_best})",
        )
        if self.wandb_run is not None:
            import wandb

            log_d = {f"val/{k}": v for k, v in r.val.items() if k != "confusion" and isinstance(v, (int, float))}
            log_d.update({"train/epoch_loss": r.train_loss, "epoch": r.epoch, "val/best_loss": self.best_val})
            cm = np.array(r.val["confusion"])
            log_d["val/confusion"] = wandb.Table(columns=["true"] + self.class_names, data=[[self.class_names[i]] + cm[i].tolist() for i in range(len(cm))])
            self.wandb_run.log(log_d, step=self.step)


def model_forward(model, batch: dict, device) -> torch.Tensor:
    pos = batch["pos"].to(device, non_blocking=True)
    x = batch["x"].to(device, non_blocking=True).transpose(1, 2).contiguous()
    return model({"pos": pos, "x": x})


@torch.no_grad()
def run_eval(model, loader, class_names, ignore_index: int, device, use_amp: bool = True, limit: int = 0, hook=None) -> dict:
    """Mean per-point CE (unweighted, unsmoothed - comparable across runs), OA, mIoU,
    per-class IoU/recall, confusion matrix; plus whatever ``hook`` adds."""
    model.eval()
    meter = ConfusionMeter(len(class_names), ignore_index, device)
    total, n_valid = 0.0, 0
    state = hook.start() if hook else None
    for k, batch in enumerate(loader):
        if limit and k >= limit:
            break
        y = batch["y"].to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp and device.type == "cuda"):
            logits = model_forward(model, batch, device)
        valid = int((y != ignore_index).sum())
        if valid:
            total += float(F.cross_entropy(logits.float(), y, ignore_index=ignore_index, reduction="sum"))
            n_valid += valid
        meter.update(logits, y)
        if hook:
            hook.update(state, batch, logits)
    out = {"loss": total / max(n_valid, 1), **meter.compute(class_names)}
    if hook:
        out.update(hook.finish(state))
    return out


def _worker_init(worker_id: int) -> None:
    # distinct numpy streams per worker (training randomness only; eval is seeded per item)
    seed = (torch.initial_seed() + worker_id) % 2**32
    np.random.seed(seed)
