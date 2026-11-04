"""Trainer behaviour (min_delta checkpointing, resume, freezing) with a tiny stand-in model,
plus the real OpenPoints models on the GPU."""

import json

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from crag import models, wandb_utils
from crag.train import Trainer


class PointMLP(nn.Module):
    """Minimal encoder/decoder/head model with the BaseSeg call signature."""

    def __init__(self, c_in=4, n_cls=3):
        super().__init__()
        self.encoder = nn.Sequential(nn.Conv1d(c_in + 3, 16, 1), nn.BatchNorm1d(16), nn.ReLU())
        self.decoder = nn.Sequential(nn.Conv1d(16, 16, 1), nn.ReLU())
        self.head = nn.Conv1d(16, n_cls, 1)

    def forward(self, data):
        x = torch.cat([data["pos"].transpose(1, 2), data["x"]], 1)
        return self.head(self.decoder(self.encoder(x)))


class ToyDS(Dataset):
    def __init__(self, n=32, seed=0):
        self.n, self.seed = n, seed

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        g = torch.Generator().manual_seed(self.seed * 1000 + i)
        pos = torch.rand(64, 3, generator=g)
        y = (pos[:, 2] * 3).long().clamp(max=2)  # class = height band: learnable
        y[:4] = -1  # some ignored points
        return {"pos": pos, "x": torch.cat([pos, pos[:, 2:3]], 1), "y": y}


def cfg(tmp_path, **kw):
    from openpoints.utils import EasyConfig

    c = EasyConfig()
    c.update({"batch_size": 8, "val_batch_size": 8, "epochs": 4, "grad_accum_steps": 2, "use_amp": False, "dataloader": {"num_workers": 0},
              "optimizer": {"lr": 0.01, "weight_decay": 0.0}, "warmup_epochs": 0, "min_lr": 1e-4, "checkpoint": {"min_delta": 0.001},
              "early_stop_patience": 0, "resume": "auto", "log_every": 1000, "ignore_index": -1, "wandb": {"log_best_artifact": False}})
    c.update(kw)
    return c


def test_min_delta_checkpointing_and_resume(tmp_path):
    torch.manual_seed(0)
    t = Trainer(cfg(tmp_path), PointMLP(), ToyDS(32), ToyDS(16, seed=1), ["a", "b", "c"], tmp_path / "run")
    t.fit(wandb_mode="disabled")
    hist = [json.loads(l) for l in (tmp_path / "run" / "metrics.jsonl").read_text().splitlines()]
    assert len(hist) == 4
    # best.pt corresponds to the last epoch whose val loss beat the running best by > min_delta
    best, best_epoch = np.inf, None
    for h in hist:
        if h["val_loss"] < best - 0.001:
            best, best_epoch = h["val_loss"], h["epoch"]
    ck = torch.load(tmp_path / "run" / "best.pt", weights_only=False)
    assert ck["epoch"] == best_epoch and np.isclose(ck["best_val_loss"], best)
    assert set(ck["parts"]) == {"encoder", "decoder", "head"}
    bb = torch.load(tmp_path / "run" / "backbone.pt", weights_only=False)
    assert set(bb) >= {"encoder", "decoder"} and "head" not in bb
    # resuming with more epochs continues from epoch 4
    t2 = Trainer(cfg(tmp_path, epochs=6), PointMLP(), ToyDS(32), ToyDS(16, seed=1), ["a", "b", "c"], tmp_path / "run")
    t2.fit(wandb_mode="disabled")
    assert t2.epoch == 6 and len(t2.history) == 6


def test_huge_min_delta_saves_only_first_epoch(tmp_path):
    t = Trainer(cfg(tmp_path, checkpoint={"min_delta": 1e9}), PointMLP(), ToyDS(32), ToyDS(16, seed=1), ["a", "b", "c"], tmp_path / "r2")
    t.fit(wandb_mode="disabled")
    # the first epoch always beats +inf; later epochs can never improve by 1e9
    assert torch.load(tmp_path / "r2" / "best.pt", weights_only=False)["epoch"] == 1
    assert torch.load(tmp_path / "r2" / "last.pt", weights_only=False)["epoch"] == 4


def test_freeze_and_unfreeze(tmp_path):
    m = PointMLP()
    models.set_frozen(m, ("encoder", "decoder"))
    m.train()
    models.apply_bn_freeze(m)
    assert not m.encoder.training and m.head.training
    assert all(not p.requires_grad for p in m.encoder.parameters())
    before = [p.clone() for p in m.encoder.parameters()]
    t = Trainer(cfg(tmp_path, epochs=3, unfreeze_schedule={2: ["decoder", "encoder"]}), m, ToyDS(32), ToyDS(16, seed=1), ["a", "b", "c"], tmp_path / "r3")
    t.fit(wandb_mode="disabled")
    assert getattr(t.model, "_frozen_parts", ()) == ()
    assert any(not torch.equal(a, b.cpu()) for a, b in zip(before, t.model.encoder.parameters()))  # trained after unfreezing


def test_param_groups_lr_mult():
    m = PointMLP()
    g = models.param_groups(m, 1e-3, 1e-4, {"encoder": 0.1})
    lrs = {(x["part"], x["weight_decay"] > 0): x["lr"] for x in g}
    assert lrs[("encoder", True)] == pytest.approx(1e-4) and lrs[("head", True)] == pytest.approx(1e-3)
    assert lrs[("encoder", False)] == pytest.approx(1e-4)  # BN / bias: no decay


def test_wandb_key_parsing(tmp_path, monkeypatch):
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    f = tmp_path / ".keys"
    monkeypatch.setattr(wandb_utils, "KEYS_FILE", f)
    f.write_text("# comment\nWANDB_API_KEY='abcdefghijklmnopqrstuvwxyz0123'\n")
    assert wandb_utils.load_key() == "abcdefghijklmnopqrstuvwxyz0123"
    f.write_text("wandb_v1_" + "x" * 40 + "\n")
    assert wandb_utils.load_key() == "wandb_v1_" + "x" * 40


gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="OpenPoints CUDA ops need a GPU")


@gpu
@pytest.mark.parametrize("name", ["pointnext-s", "pointnet2"])
def test_openpoints_models_load_and_fit_memory(name, tmp_path):
    from crag.pretrain import build_model_from, load_cfg
    from crag.pseudolabels import CLASSES

    c = load_cfg(f"cfgs/pretrain/{name}.yaml")
    model, report = build_model_from(c)
    assert report["loaded"] > 50 and not report["mismatched"]
    model = model.cuda().train()
    opt = torch.optim.AdamW(model.parameters(), 1e-3)
    torch.cuda.reset_peak_memory_stats()
    B, N = 8, 8192
    data = {"pos": torch.rand(B, N, 3, device="cuda") * 3.2, "x": torch.rand(B, 4, N, device="cuda")}
    with torch.autocast("cuda", dtype=torch.float16):
        out = model(data)
        loss = nn.functional.cross_entropy(out.float(), torch.randint(0, len(CLASSES), (B, N), device="cuda"))
    loss.backward()
    opt.step()
    assert out.shape == (B, len(CLASSES), N)
    assert torch.cuda.max_memory_allocated() / 1e9 < 8.0  # fits a 10 GB RTX 3080 at batch 8 x 8192
    # backbone round trip into a fresh 2-class model
    torch.save({"encoder": model.encoder.state_dict(), "decoder": model.decoder.state_dict()}, tmp_path / "backbone.pt")
    c.model.cls_args.num_classes = 2
    m2 = models.build_model(c.model)
    rep = models.load_pretrained(m2, tmp_path / "backbone.pt")
    assert rep["loaded"] == len([k for k in m2.state_dict() if k.split(".")[0] in ("encoder", "decoder")])
