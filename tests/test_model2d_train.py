"""2D training pipeline (crag/model2d_train.py): inputs, augmentation, model, metrics. CPU only, no data needed."""

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from crag import model2d_train as m2  # noqa: E402
from crag.model2d_raster import CHANNELS, gradient  # noqa: E402
from crag.paths import PROJECT_DIR  # noqa: E402

RES = 0.5


def _stored(dem: np.ndarray) -> torch.Tensor:
    x = np.zeros((1, len(CHANNELS), *dem.shape), np.float32)
    x[0, CHANNELS.index("dem")] = dem
    return torch.from_numpy(x)


def test_metric_code_is_the_3d_copy():
    """sample_metrics / pair_metrics / _boot_ci / HARD_NEGATIVE_SOURCES must stay identical to crag/finetune.py (read as text, not imported)."""
    for name in ("sample_metrics", "pair_metrics", "_boot_ci", "HARD_NEGATIVE_SOURCES"):
        assert m2.source_of(PROJECT_DIR / "crag" / "finetune.py", name) == m2.source_of(PROJECT_DIR / "crag" / "model2d_train.py", name), name


def test_gradient_input_matches_numpy_version():
    rng = np.random.default_rng(0)
    dem = rng.normal(size=(40, 40)).cumsum(0).astype(np.float32)
    got = m2.make_input(_stored(dem), "gradient", RES)[0].numpy()
    assert np.allclose(got, gradient(dem, RES), atol=1e-5)
    assert m2.make_input(_stored(dem), "inbetween", RES).shape == (1, 3, 40, 40)


def test_augmentation_keeps_gradient_true_to_the_image():
    # rotate a "rises to the north" slope a quarter turn anticlockwise: it now rises to the west
    y = np.arange(40)[::-1, None].repeat(40, 1).astype(np.float32) * RES  # row 0 = north = highest
    x = m2.dihedral(_stored(y), torch.tensor([1]))
    g = m2.make_input(x, "gradient", RES)[0]
    assert torch.allclose(g[0], torch.full_like(g[0], -1.0), atol=1e-5)  # height falls going east
    assert torch.allclose(g[1], torch.zeros_like(g[1]), atol=1e-5)
    # all 8 operations are distinct and invertible on a random image
    r = torch.rand(8, 1, 6, 6).expand(8, 1, 6, 6).clone()
    r[:] = r[0]
    outs = m2.dihedral(r, torch.arange(8))
    assert len({tuple(o.flatten().tolist()) for o in outs}) == 8


def test_position_channel():
    p = m2.position_map(40)[0, 0]
    assert p.shape == (40, 40)
    assert p[19:21, 19:21].max() < 0.04 and abs(float(p[19, 0]) - 1.0) < 0.03 and float(p.max()) == pytest.approx(np.sqrt(2) * 39 / 40, abs=1e-4)
    for k in range(8):  # unchanged by every rotation / flip
        assert torch.equal(m2.dihedral(p[None, None], torch.tensor([k]))[0, 0], p)
    pipe = m2.InputPipeline("gradient", RES, 40, [0.0, 0.0], [1.0, 1.0], position=True)
    assert pipe(_stored(np.zeros((40, 40), np.float32))).shape == (1, 3, 40, 40)


@pytest.mark.parametrize("n_data,position", [(2, True), (3, True), (2, False)])
def test_model_shapes_and_first_layer(n_data, position):
    m = m2.Model2D("resnet18", n_data, position, pretrained=False, input_size=64)
    assert m.net.conv1.in_channels == n_data + int(position)
    out = m(torch.zeros(4, n_data + int(position), 40, 40))
    assert out.shape == (4,)
    parts = m.parts()
    assert len(parts["head"]) == 2 and len(parts["backbone"]) + 2 == len(list(m.parameters()))


def test_balanced_sampler_weights():
    idx = pd.DataFrame({"label": [1, 0, 0, 0, 0], "confidence": [1.0, 1.0, 1.0, 0.5, 0.5]})
    w = m2.balanced_weights(idx)
    assert w[idx["label"] == 1].sum() == pytest.approx(1.0) and w[idx["label"] == 0].sum() == pytest.approx(1.0)
    assert w[1] == pytest.approx(2 * w[3])


def test_lr_schedule():
    cfg = {"optimizer": {"lr": 1e-3}, "min_lr": 1e-6, "warmup_epochs": 2, "epochs": 10}
    lrs = [m2.lr_at(e, cfg) for e in range(10)]
    assert lrs[0] == pytest.approx(5e-4) and lrs[1] == pytest.approx(1e-3) and lrs[2] == pytest.approx(1e-3)
    assert all(a >= b for a, b in zip(lrs[2:], lrs[3:])) and lrs[-1] > 1e-6


def test_evaluate_horseshoe_block_and_cutoff():
    rng = np.random.default_rng(0)
    rows = []
    for split in ("val", "test"):
        for i in range(60):
            lab = i % 2
            src = ("handgathered" if lab else "handgathered_nc") if split == "test" else ("osm" if lab else "band")
            rows.append({"sample_id": f"{split}{i}", "slug": "horseshoe-canyon-ranch", "source": src, "label": lab, "confidence": 1.0,
                         "split": split, "score": 0.6 * lab + 0.2 + rng.normal(0, 0.15), "lon": 0.0, "lat": 0.0})
    res = m2.evaluate(pd.DataFrame(rows))
    h = res["horseshoe"]
    assert h["n"] == 60 and h["auroc_lo"] <= h["auroc"] <= h["auroc_hi"] and h["hand_auroc"] == pytest.approx(h["auroc"])
    assert 0.2 < res["val_cutoff"] < 0.8 and h["bal_acc_valcut"] > 0.8


def test_load_cfg_overrides():
    cfg = m2.load_cfg("full_discriminative", ["data.input=inbetween", "seed=2", "data.position_channel=false"])
    assert cfg["data"]["input"] == "inbetween" and cfg["seed"] == 2 and cfg["lr_mult"]["backbone"] == 0.1
    assert m2.run_name(cfg) == "inbetween-full_discriminative-nopos-s2"
    with pytest.raises(ValueError):
        m2.load_cfg("scratch", ["data.input=rgb"])


def test_run_dir_groups_by_pretrain_stage():
    cfg = m2.load_cfg("scratch", ["seed=1"])
    assert cfg["pretrain"] == "none"
    assert m2.run_dir(cfg) == m2.RUNS_DIR / "no_pseudolabel" / "gradient-scratch-nopos-s1"
    assert m2.run_dir(cfg, "smoke") == m2.RUNS_DIR / "no_pseudolabel" / "smoke"
    pl = m2.load_cfg("scratch", ["seed=1", "pretrain=pseudolabel"])
    assert m2.run_dir(pl) == m2.RUNS_DIR / "pseudolabel" / "gradient-scratch-nopos-s1"
    assert m2.run_name(m2.load_cfg("scratch", ["seed=1", "data.position_channel=true"])) == "gradient-scratch-s1"
    with pytest.raises(ValueError):
        m2.run_dir(m2.load_cfg("scratch", ["pretrain=imagenet"]))
