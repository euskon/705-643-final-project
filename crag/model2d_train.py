"""2D model: ResNet on top-down LiDAR images, trained and scored like the 3D fine-tuning.

Everything that is not the input copies the 3D pipeline (``crag/finetune.py``
and ``cfgs/finetune/*.yaml``), so a 2D vs 3D difference comes from the input:

* the same samples, labels and splits (``data/raster/<name>`` is made from
  ``data/finetune/<name>`` by ``scripts/model2d_rasterize.py``); samples whose
  3D cylinder holds no labelled point (``n_center == 0``) are dropped
  everywhere, because the 3D model neither trains on nor scores them;
* class-balanced, confidence-weighted sampling with replacement
  (``samples_per_epoch`` per epoch), AdamW, warm-up then cosine decay, early
  stopping on validation loss, the best checkpoint by validation loss;
* the same per-sample metrics (:func:`sample_metrics`, a verbatim copy of the
  3D one; ``tests/test_model2d_train.py`` checks the copy still matches).

Inputs, built on the GPU from the stored layers (``crag.model2d_raster.CHANNELS``)
after augmentation:

* ``gradient``: d(height)/d(east), d(height)/d(north) of the bare-earth
  height (Rafique et al. 2022);
* ``inbetween``: lowest return, highest return, return count.

Normalisation is the paper's "Gaussian" one (training-split mean and standard
deviation), made rotation-consistent: both gradient channels share one
standard deviation and have mean 0, since rotations and flips swap and negate
them.

Position channel (``data.position_channel``): one more input layer holding each
pixel's distance from the image centre (0 at the centre, 1 at the middle of
an edge). A sample's label describes only the spot at the centre, while
ResNet averages its features over the whole image at the end and so loses
*where* a feature was; this channel lets it tell "wall at the centre" from
"wall at the edge". The 3D model doesn't need it: it is scored on the
centre cylinder only.
"""

from __future__ import annotations

import ast
import copy
import json
import logging
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from sklearn.metrics import average_precision_score, balanced_accuracy_score, f1_score, roc_auc_score

from crag.model2d_raster import CHANNELS, RASTER_DIR
from crag.paths import PROJECT_DIR

log = logging.getLogger(__name__)

CFG_DIR = PROJECT_DIR / "cfgs" / "model2d"
RUNS_DIR = PROJECT_DIR / "runs" / "model2d"
INPUTS = {"gradient": 2, "inbetween": 3}  # data channels per input type
HCR = "horseshoe-canyon-ranch"
HAND_SOURCES = ("handgathered", "handgathered_nc")
PRED_COLS = ["sample_id", "slug", "source", "label", "confidence", "split", "score", "lon", "lat"]  # as the 3D predictions CSV

# ---------------------------------------------------------------------------
# metrics: verbatim copy of crag.finetune (that module needs OpenPoints to import)
# ---------------------------------------------------------------------------
HARD_NEGATIVE_SOURCES = ("band", "handgathered_nc")


def sample_metrics(pred: pd.DataFrame, threshold: float = 0.5) -> dict:
    p = pred[pred["score"].notna()]
    y, s = p["label"].to_numpy(), p["score"].to_numpy()
    out = {"sample_n": int(len(p)), "sample_n_pos": int((y == 1).sum()), "sample_n_neg": int((y == 0).sum())}
    if len(np.unique(y)) == 2:
        out.update(
            sample_auroc=float(roc_auc_score(y, s)),
            sample_ap=float(average_precision_score(y, s)),
            sample_bal_acc=float(balanced_accuracy_score(y, s >= threshold)),
            sample_f1=float(f1_score(y, s >= threshold)),
        )
        hard = p["source"].isin(HARD_NEGATIVE_SOURCES) | (p["label"] == 1)
        if (p.loc[hard, "label"] == 0).any():
            out["sample_auroc_hard"] = float(roc_auc_score(p.loc[hard, "label"], p.loc[hard, "score"]))
    for src, g in p.groupby("source"):
        # positives: recall; negatives: specificity (fraction correctly scored below threshold)
        ok = (g["score"] >= threshold) if g["label"].iloc[0] == 1 else (g["score"] < threshold)
        out[f"acc_{src}"] = float(ok.mean())
    return out


def source_of(module_path: Path, name: str) -> str:
    """Source text of a top-level function or assignment in a file, without importing it."""
    text = module_path.read_text(encoding="utf-8")
    for node in ast.parse(text).body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_source_segment(text, node)
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == name for t in node.targets):
            return ast.get_source_segment(text, node)
    raise KeyError(name)


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------
def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_cfg(strategy_file: str | Path, overrides: list[str] = ()) -> dict:
    """``cfgs/model2d/default.yaml`` + a strategy file + ``a.b=value`` overrides (values parsed as YAML)."""
    cfg = yaml.safe_load((CFG_DIR / "default.yaml").read_text())
    p = Path(strategy_file)
    if not p.exists():
        p = CFG_DIR / f"{strategy_file}.yaml"
    cfg = _merge(cfg, yaml.safe_load(p.read_text()) or {})
    cfg["strategy"] = cfg.get("strategy") or p.stem
    for o in overrides:
        key, val = o.split("=", 1)
        node = cfg
        *path, last = key.split(".")
        for k in path:
            node = node.setdefault(k, {})
        node[last] = yaml.safe_load(val)
    if cfg["data"]["input"] not in INPUTS:
        raise ValueError(f"data.input must be one of {list(INPUTS)}")
    return cfg


def run_name(cfg: dict) -> str:
    pos = "" if cfg["data"]["position_channel"] else "-nopos"
    return f"{cfg['data']['input']}-{cfg['strategy']}{pos}-s{cfg['seed']}"


# runs are grouped by what happened before fine-tuning:
#   none        -> runs/model2d/no_pseudolabel/  (ImageNet weights, or random init for scratch)
#   pseudolabel -> runs/model2d/pseudolabel/     (+ pretraining on the 4 LiDAR terrain pseudolabel classes)
PRETRAIN_GROUPS = {"none": "no_pseudolabel", "pseudolabel": "pseudolabel"}
BASELINES_DIR = RUNS_DIR / "baselines"


def pretrain_group(cfg: dict) -> str:
    stage = cfg.get("pretrain", "none")
    if stage not in PRETRAIN_GROUPS:
        raise ValueError(f"pretrain must be one of {list(PRETRAIN_GROUPS)}, got {stage!r}")
    return PRETRAIN_GROUPS[stage]


def run_dir(cfg: dict, name: str | None = None) -> Path:
    """``runs/model2d/<pretrain group>/<run name>``."""
    return RUNS_DIR / pretrain_group(cfg) / (name or run_name(cfg))


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def load_data(cfg: dict) -> tuple[np.ndarray, pd.DataFrame, dict]:
    """(all stored images, the usable samples' index, the raster meta.json)."""
    d = RASTER_DIR / cfg["data"]["name"]
    meta = json.loads((d / "meta.json").read_text())
    if meta["channels"] != list(CHANNELS):
        raise ValueError(f"{d}: channels {meta['channels']} != {list(CHANNELS)}; re-run scripts/model2d_rasterize.py")
    images = np.load(d / "images.npy", mmap_mode="r")
    index = pd.read_csv(d / "index.csv")
    keep = (index["n_center"] > 0) & (index["n_points"] > 0) & (index["confidence"] >= float(cfg["data"].get("min_confidence", 0.0)))
    keep &= ~index["source"].isin(cfg["data"].get("exclude_sources") or [])
    return images, index[keep].reset_index(drop=True), meta


def make_input(stored: torch.Tensor, kind: str, res_m: float) -> torch.Tensor:
    """(B, len(CHANNELS), H, W) stored layers -> (B, C, H, W) model input, before normalisation."""
    if kind == "gradient":
        d_row, d_col = torch.gradient(stored[:, CHANNELS.index("dem")], spacing=res_m, dim=(-2, -1))
        return torch.stack([d_col, -d_row], 1)  # (east, north): rows run north -> south
    if kind == "inbetween":
        return stored[:, [CHANNELS.index("zmin"), CHANNELS.index("zmax"), CHANNELS.index("count")]]
    raise ValueError(kind)


def norm_stats(x: torch.Tensor, kind: str) -> tuple[list[float], list[float]]:
    """Per-channel (mean, std) of training inputs; gradient channels share one std and mean 0."""
    x = x.double()
    if kind == "gradient":
        s = float(x.pow(2).mean().sqrt())
        return [0.0, 0.0], [s, s]
    return x.mean((0, 2, 3)).tolist(), x.std((0, 2, 3)).clamp(min=1e-6).tolist()


def dihedral(x: torch.Tensor, ops: torch.Tensor) -> torch.Tensor:
    """Apply one of the 8 rotations / mirror flips to each image: op k = k%4 quarter turns, then a flip if k >= 4."""
    out = x.clone()
    for k in range(8):
        m = ops == k
        if m.any():
            y = torch.rot90(x[m], k % 4, dims=(-2, -1))
            out[m] = torch.flip(y, dims=(-1,)) if k >= 4 else y
    return out


def position_map(n: int) -> torch.Tensor:
    """(1, 1, n, n) distance of each pixel centre from the image centre, in half-widths."""
    c = (torch.arange(n, dtype=torch.float32) + 0.5 - n / 2) / (n / 2)
    yy, xx = torch.meshgrid(c, c, indexing="ij")
    return torch.sqrt(xx**2 + yy**2)[None, None]


class InputPipeline(nn.Module):
    """Stored layers -> normalised model input (+ position channel), on the GPU."""

    def __init__(self, kind: str, res_m: float, n: int, mean: list[float], std: list[float], position: bool):
        super().__init__()
        self.kind, self.res_m, self.position = kind, res_m, position
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32)[None, :, None, None])
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32)[None, :, None, None])
        self.register_buffer("pos", position_map(n))

    def forward(self, stored: torch.Tensor) -> torch.Tensor:
        x = (make_input(stored, self.kind, self.res_m) - self.mean) / self.std
        if self.position:
            x = torch.cat([x, self.pos.expand(len(x), -1, -1, -1)], 1)
        return x


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------
class Model2D(nn.Module):
    """torchvision ResNet with an adapted first layer and a one-logit head (P(climbable) = sigmoid)."""

    def __init__(self, arch: str, n_data: int, position: bool, pretrained: bool, input_size: int, position_init_std: float = 1e-3):
        super().__init__()
        import torchvision

        weights = "DEFAULT" if pretrained else None
        net = getattr(torchvision.models, arch)(weights=weights)
        old = net.conv1
        n_in = n_data + int(position)
        conv = nn.Conv2d(n_in, old.out_channels, old.kernel_size, old.stride, old.padding, bias=False)
        if pretrained:
            w = old.weight.data  # (64, 3, 7, 7), ImageNet RGB
            # 3 data channels: keep the RGB filters; otherwise every channel gets the RGB mean,
            # scaled so a filter's response keeps its size
            wd = w if n_data == 3 else w.mean(1, keepdim=True).repeat(1, n_data, 1, 1) * (3.0 / n_data)
            with torch.no_grad():
                conv.weight[:, :n_data] = wd
                if position:  # new channel starts small: the pretrained filters work as before until training finds a use for it
                    conv.weight[:, n_data:].normal_(0.0, position_init_std)
        net.conv1 = conv
        net.fc = nn.Linear(net.fc.in_features, 1)
        self.net, self.input_size = net, input_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.input_size and x.shape[-1] != self.input_size:
            x = F.interpolate(x, size=(self.input_size, self.input_size), mode="bilinear", align_corners=False)
        return self.net(x).squeeze(1)

    def parts(self) -> dict[str, list[nn.Parameter]]:
        head = list(self.net.fc.parameters())
        ids = {id(p) for p in head}
        return {"backbone": [p for p in self.parameters() if id(p) not in ids], "head": head}


def build(cfg: dict, n: int, res_m: float, mean, std) -> tuple[InputPipeline, Model2D]:
    m, d = cfg["model"], cfg["data"]
    pipe = InputPipeline(d["input"], res_m, n, mean, std, d["position_channel"])
    model = Model2D(m["arch"], INPUTS[d["input"]], d["position_channel"], m["pretrained"], m["input_size"], m.get("position_init_std", 1e-3))
    return pipe, model


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------
def balanced_weights(index: pd.DataFrame) -> np.ndarray:
    """Equal total weight per class, and within a class proportional to confidence (as the 3D sampler)."""
    lab, conf = index["label"].to_numpy(), index["confidence"].to_numpy(dtype=float)
    w = np.zeros(len(lab))
    for c in (0, 1):
        m = lab == c
        if m.any():
            w[m] = conf[m] / conf[m].sum()
    return w


def lr_at(epoch: int, cfg: dict) -> float:
    base, lo, warm, total = cfg["optimizer"]["lr"], cfg["min_lr"], cfg["warmup_epochs"], cfg["epochs"]
    if epoch < warm:
        return base * (epoch + 1) / warm
    t = (epoch - warm) / max(total - warm, 1)
    return lo + 0.5 * (base - lo) * (1 + math.cos(math.pi * t))


@torch.no_grad()
def predict(pipe: InputPipeline, model: Model2D, images: torch.Tensor, batch_size: int = 256) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(images), batch_size):
        with torch.autocast("cuda", dtype=torch.float16, enabled=images.is_cuda):
            out.append(model(pipe(images[i : i + batch_size])).float())
    return torch.sigmoid(torch.cat(out)).cpu().numpy()


def predictions(index: pd.DataFrame, score: np.ndarray) -> pd.DataFrame:
    return index.assign(score=score)[PRED_COLS]


def bce(score: np.ndarray, label: np.ndarray) -> float:
    s = np.clip(score, 1e-7, 1 - 1e-7)
    return float(-np.mean(label * np.log(s) + (1 - label) * np.log(1 - s)))


def train(cfg: dict, run_dir: Path, device: str = "cuda") -> dict:
    """Train one run; writes ``config.yaml``, ``log.csv`` and ``best.pt`` (lowest val loss) to ``run_dir``."""
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    seed = int(cfg["seed"])
    torch.manual_seed(seed)
    np.random.seed(seed)
    images, index, meta = load_data(cfg)
    res_m, n = meta["config"]["res_m"], meta["shape"][-1]
    split = {s: index[index["split"] == s].reset_index(drop=True) for s in ("train", "val")}
    data = {s: torch.from_numpy(np.ascontiguousarray(images[split[s]["row"].to_numpy()])).to(device) for s in split}
    kind = cfg["data"]["input"]
    mean, std = norm_stats(make_input(data["train"], kind, res_m), kind)
    pipe, model = build(cfg, n, res_m, mean, std)
    pipe, model = pipe.to(device), model.to(device)

    parts = model.parts()
    frozen = set(cfg.get("freeze") or [])
    for name in frozen:
        for p in parts[name]:
            p.requires_grad_(False)
    groups = [{"params": ps, "lr_mult": float(cfg["lr_mult"].get(name, 1.0)), "name": name} for name, ps in parts.items() if name not in frozen]
    opt = torch.optim.AdamW(groups, lr=cfg["optimizer"]["lr"], weight_decay=cfg["optimizer"]["weight_decay"])
    scaler = torch.amp.GradScaler("cuda", enabled=cfg["use_amp"] and device == "cuda")
    gen = torch.Generator().manual_seed(seed)
    weights = torch.as_tensor(balanced_weights(split["train"]), dtype=torch.double)
    y_train = torch.as_tensor(split["train"]["label"].to_numpy(), dtype=torch.float32, device=device)
    y_val = split["val"]["label"].to_numpy()
    bs = int(cfg["batch_size"])
    log.info("%s: %d train / %d val samples, input %s%s, %s, frozen %s", run_dir.name, len(split["train"]), len(split["val"]),
             kind, " + position" if cfg["data"]["position_channel"] else "", cfg["model"]["arch"], sorted(frozen) or "none")

    best, best_epoch, rows = math.inf, -1, []
    for epoch in range(int(cfg["epochs"])):
        t0 = time.time()
        lr = lr_at(epoch, cfg)
        for g in opt.param_groups:
            g["lr"] = lr * g["lr_mult"]
        model.train()
        if "backbone" in frozen:  # frozen BatchNorm keeps its pretrained statistics
            for mod in model.net.modules():
                if isinstance(mod, nn.BatchNorm2d):
                    mod.eval()
        order = torch.multinomial(weights, int(cfg["data"]["samples_per_epoch"]), replacement=True, generator=gen).to(device)
        losses = []
        for i in range(0, len(order), bs):
            idx = order[i : i + bs]
            x = data["train"][idx]
            if cfg["augment"]:
                x = dihedral(x, torch.randint(0, 8, (len(idx),), generator=gen).to(device))
            with torch.autocast("cuda", dtype=torch.float16, enabled=scaler.is_enabled()):
                loss = F.binary_cross_entropy_with_logits(model(pipe(x)).float(), y_train[idx])
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_([p for g in groups for p in g["params"]], cfg["grad_norm_clip"])
            scaler.step(opt)
            scaler.update()
            losses.append(loss.item())
        score = predict(pipe, model, data["val"])
        val_loss = bce(score, y_val)
        m = sample_metrics(predictions(split["val"], score))
        row = {"epoch": epoch, "lr": lr, "train_loss": float(np.mean(losses)), "val_loss": val_loss,
               **{f"val_{k}": m.get(k) for k in ("sample_auroc", "sample_ap", "sample_bal_acc", "sample_auroc_hard")}, "seconds": time.time() - t0}
        improved = val_loss < best - cfg["checkpoint"]["min_delta"]
        if improved:
            best, best_epoch = val_loss, epoch
            torch.save({"model": model.state_dict(), "pipe": pipe.state_dict(), "cfg": cfg, "epoch": epoch, "val_loss": val_loss,
                        "norm": {"mean": mean, "std": std}, "res_m": res_m, "n": n}, run_dir / "best.pt")
        rows.append({**row, "best": improved})
        pd.DataFrame(rows).to_csv(run_dir / "log.csv", index=False)
        log.info("epoch %2d  lr %.1e  train %.4f  val %.4f  val AUROC %.3f  hard %.3f  %4.1fs%s", epoch, lr, row["train_loss"], val_loss,
                 row["val_sample_auroc"] or float("nan"), row["val_sample_auroc_hard"] or float("nan"), row["seconds"], "  *" if improved else "")
        if epoch - best_epoch >= int(cfg["early_stop_patience"]):
            log.info("early stop: no val-loss improvement for %d epochs", cfg["early_stop_patience"])
            break
    return {"best_epoch": best_epoch, "best_val_loss": best, "epochs_run": len(rows)}


def load_run(run_dir: Path, device: str = "cuda") -> tuple[dict, InputPipeline, Model2D]:
    ck = torch.load(run_dir / "best.pt", map_location=device, weights_only=False)
    cfg = ck["cfg"]
    pipe, model = build({**cfg, "model": {**cfg["model"], "pretrained": False}}, ck["n"], ck["res_m"], ck["norm"]["mean"], ck["norm"]["std"])
    pipe.load_state_dict(ck["pipe"])
    model.load_state_dict(ck["model"])
    return {**cfg, "best_epoch": ck["epoch"]}, pipe.to(device), model.to(device)


# ---------------------------------------------------------------------------
# evaluation (shared by trained runs and the no-training baselines)
# ---------------------------------------------------------------------------
def best_cutoff(pred: pd.DataFrame) -> float:
    """Cutoff on P(climbable) with the highest balanced accuracy (chosen on validation predictions)."""
    p = pred[pred["score"].notna()]
    y, s = p["label"].to_numpy(), p["score"].to_numpy()
    cands = np.unique(s)
    accs = [balanced_accuracy_score(y, s >= c) for c in cands]
    return float(cands[int(np.argmax(accs))])


def bootstrap_auroc(y: np.ndarray, s: np.ndarray, n: int = 1000, seed: int = 0) -> tuple[float, float, float]:
    """AUROC and its 95% range over ``n`` resamples of the samples."""
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) == 2:
            vals.append(roc_auc_score(y[i], s[i]))
    return float(roc_auc_score(y, s)), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def evaluate(pred: pd.DataFrame) -> dict:
    """Metrics per split and area, and the Horseshoe Canyon table (as docs/results/v3/horseshoe_confidence_and_cutoffs.txt)."""
    out = {"splits": {}, "per_area": {}}
    for s, g in pred.groupby("split"):
        out["splits"][s] = sample_metrics(g)
        out["per_area"][s] = {a: sample_metrics(ga) for a, ga in g.groupby("slug")}
    val = pred[pred["split"] == "val"]
    cut = best_cutoff(val) if val["label"].nunique() == 2 else 0.5
    test = pred[pred["split"] == "test"]
    out["val_cutoff"] = cut
    out["test_bal_acc_valcut"] = float(balanced_accuracy_score(test["label"], test["score"] >= cut)) if test["label"].nunique() == 2 else None
    h = test[test["slug"] == HCR]
    if h["label"].nunique() == 2:
        hand = h[h["source"].isin(HAND_SOURCES)]
        a, lo, hi = bootstrap_auroc(h["label"].to_numpy(), h["score"].to_numpy())
        ha, hlo, hhi = bootstrap_auroc(hand["label"].to_numpy(), hand["score"].to_numpy())
        out["horseshoe"] = {
            "n": int(len(h)), "auroc": a, "auroc_lo": lo, "auroc_hi": hi,
            "hand_auroc": ha, "hand_auroc_lo": hlo, "hand_auroc_hi": hhi,
            "bal_acc_cut05": float(balanced_accuracy_score(h["label"], h["score"] >= 0.5)),
            "bal_acc_valcut": float(balanced_accuracy_score(h["label"], h["score"] >= cut)),
            "climb_found": float((hand.loc[hand["label"] == 1, "score"] >= cut).mean()),
            "nc_rejected": float((hand.loc[hand["label"] == 0, "score"] < cut).mean()),
        }
    return out


def write_eval(run_dir: Path, pred: pd.DataFrame, extra: dict | None = None) -> dict:
    pred.to_csv(run_dir / "predictions_best.csv", index=False)
    res = {**(extra or {}), **evaluate(pred)}
    (run_dir / "eval_best.json").write_text(json.dumps(res, indent=1, default=float))
    return res
