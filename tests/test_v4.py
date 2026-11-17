"""V4 fixes: wall height and snap (plan 4), one labelling rule (plan 1), scale (plan 2), channels (plan 3)."""

import json

import numpy as np
import pandas as pd
import pytest
import torch
from scipy.spatial import cKDTree

from crag import hcr_points as H
from crag.channels import CHANNELS, point_channels, write_channels_json
from crag.datasets import BlockDataset, BlockSpec, make_item, resample_keep
from crag.finetune import pair_metrics
from crag.finetune_data import (StripConfig, apply_hcr_fold, block_codes, face_strip_mask, pick_points, split_strip_widths, wall_top)
from crag.groundtruth import GroundTruthConfig, snap_one, wall_foot
from crag.labeling import PointCloud
from crag.negatives import NegativeConfig, clean_samples, spacing_report
from crag.pretrain import load_cfg
from crag.train import per_sample_loss
from crag.walls import wall_grid, wall_height_at

RNG = np.random.default_rng


def step_scene(h, behind_slope=0.0, seed=0, face_pts=0, lean_deg=90.0, y=(0, 100)):
    """Floor z=0 for x<0, a wall at x=0 (h tall), ground behind it (optionally rising)."""
    r = RNG(seed)
    floor = np.column_stack([r.uniform(-40, 0, 20000), r.uniform(*y, 20000), r.normal(0, 0.03, 20000)])
    lean = 0.0 if lean_deg >= 90 else h / np.tan(np.radians(lean_deg))
    tx = r.uniform(lean + 0.01, 40, 16000)
    top = np.column_stack([tx, r.uniform(*y, 16000), h + behind_slope * (tx - lean)])
    parts, steep = [floor, top], [np.zeros(20000, bool), np.zeros(16000, bool)]
    if face_pts:
        z = r.uniform(0, h, face_pts)
        parts.append(np.column_stack([z / np.tan(np.radians(lean_deg)) if lean_deg < 90 else r.normal(0, 0.1, face_pts), r.uniform(*y, face_pts), z]))
        steep.append(np.ones(face_pts, bool))
    return np.vstack(parts), np.concatenate(steep)


def pc_of(xyz, steep):
    si = np.flatnonzero(steep)
    return PointCloud(xyz=xyz, normals=np.zeros_like(xyz), steep=steep, tree_xy=cKDTree(xyz[:, :2]), steep_idx=si, steep_tree_xy=cKDTree(xyz[si, :2]))


# ---------------------------------------------------------------- plan 4


def test_wall_height_at_synthetic():
    for h in (12.0, 5.0):
        xyz, _ = step_scene(h)
        g = wall_grid(xyz, np.ones(len(xyz), bool))
        assert wall_height_at(g, [-0.5, 50.0], 0.0)[0] == pytest.approx(h, abs=1)
    flat = RNG(0).uniform(0, 50, (5000, 3)) * [1, 1, 0]
    assert wall_height_at(wall_grid(flat, np.ones(5000, bool)), [25.0, 25.0], 0.0)[0] == 0


def test_wall_height_ignores_hillside():
    xyz, _ = step_scene(10.0, behind_slope=20 / 39)
    g = wall_grid(xyz, np.ones(len(xyz), bool))
    assert wall_height_at(g, [-0.5, 50.0], 0.0)[0] == pytest.approx(10, abs=2)


def two_walls():
    """A 5 m wall at x=0 and a 12 m wall at x=12 (behind it), floor in front."""
    r = RNG(3)
    floor = np.column_stack([r.uniform(-40, 0, 20000), r.uniform(0, 100, 20000), r.normal(0, 0.03, 20000)])
    ledge = np.column_stack([r.uniform(0.01, 12, 8000), r.uniform(0, 100, 8000), np.full(8000, 5.0)])
    top = np.column_stack([r.uniform(12.01, 40, 12000), r.uniform(0, 100, 12000), np.full(12000, 17.0)])
    f1 = np.column_stack([r.normal(0, 0.1, 3000), r.uniform(0, 100, 3000), r.uniform(0, 5, 3000)])
    f2 = np.column_stack([12 + r.normal(0, 0.1, 6000), r.uniform(0, 100, 6000), r.uniform(5, 17, 6000)])
    xyz = np.vstack([floor, ledge, top, f1, f2])
    steep = np.r_[np.zeros(40000, bool), np.ones(9000, bool)]
    return xyz, steep


def test_snap_skips_short_wall_and_too_short():
    xyz, steep = two_walls()
    pc = pc_of(xyz, steep)
    g = GroundTruthConfig(max_dz_m=6.0)
    s = snap_one(pc, np.array([-4.0, 50.0]), g=g)
    assert s["status"] == "ok" and s["snap_x"] > 10 and s["skipped_shorter_wall"]
    s4 = snap_one(pc, np.array([-4.0, 50.0]), g=GroundTruthConfig(min_wall_height_m=4.0))
    assert s4["status"] == "ok" and s4["snap_x"] < 2
    xyz5, st5 = step_scene(5.0, face_pts=3000)
    assert snap_one(pc_of(xyz5, st5), np.array([-3.0, 50.0]))["status"] == "too_short"
    assert snap_one(pc_of(xyz5, st5), np.array([-35.0, 50.0]))["status"] == "off_wall"


def test_toward_wall_keeps_spacing_and_spacing_report():
    xyz, steep = step_scene(15.0, face_pts=6000)
    pc = pc_of(xyz, steep)
    a, b = snap_one(pc, np.array([-8.0, 48.0])), snap_one(pc, np.array([-8.0, 52.0]))
    assert np.hypot(a["snap_x"] - b["snap_x"], a["snap_y"] - b["snap_y"]) == pytest.approx(4, abs=0.6)
    df = pd.DataFrame({"slug": "s", "route_uid": ["a", "b"], "label": [1, 0], "status": "ok", "x": [-8.0, -8.0], "y": [48.0, 53.0],
                       "snap_x": [0.0, 0.0], "snap_y": [50.0, 50.5], "out_dx": -1.0, "out_dy": 0.0})
    assert spacing_report(df)["squashed_pair"].all()


def test_wall_foot_from_any_height():
    xyz, steep = step_scene(20.0, face_pts=8000, lean_deg=70)
    pc = pc_of(xyz, steep)
    lean = 20 / np.tan(np.radians(70))
    feet = [wall_foot(pc, p) for p in ([lean - 0.3, 50.0], [lean / 2, 50.0], [-3.0, 50.0])]
    assert all(f["status"] == "ok" for f in feet)
    xs = [f["foot_x"] for f in feet]
    assert max(xs) - min(xs) < 0.5 and abs(np.mean(xs)) < 0.6


def test_negatives_height_rule():
    from crag.negatives import centre_on_foot

    xyz, steep = step_scene(5.0, face_pts=3000)

    class Cloud:
        def crop(self, cx, cy, half):
            m = (np.abs(xyz[:, 0] - cx) <= half) & (np.abs(xyz[:, 1] - cy) <= half)
            return {"xyz": xyz[m], "steep": steep[m], "normals": np.zeros((m.sum(), 3)), "multi": np.zeros(m.sum(), bool)}

    assert centre_on_foot(Cloud(), pc_of(xyz, steep), np.array([0.0, 50.0]), 6.0)["status"] == "too_short"
    assert centre_on_foot(Cloud(), pc_of(xyz, steep), np.array([0.0, 50.0]), 4.0)["status"] == "ok"


def test_snap_script_keeps_rows_and_protects_judging(tmp_path):
    orig, work = tmp_path / "orig", tmp_path / "work"
    orig.mkdir(), work.mkdir()
    df = pd.DataFrame({"name": ["a", "b", "c"], "area_name": "West Side", "lat": [1.0, 2.0, 3.0], "lon": [1.0, 2.0, 3.0], "comment": ["Climb", "?", "Nc"],
                       "time": ["t1", "t2", "t3"], "mp_url": None})
    df.to_csv(orig / "W.csv", index=False)
    snap = lambda xy: {"status": "ok", "snap_x": xy[0] + 1, "snap_y": xy[1], "snap_dist_m": 1.0, "along_shift_m": 0.0, "wall_height_m": 9.0, "out_dx": -1.0, "out_dy": 0.0}
    ident = lambda a, b: np.column_stack([a, b])
    p = H.snap_waypoint_files(snap, ident, ident, orig, work, ("W.csv",))
    w = pd.read_csv(work / "W.csv")
    assert len(w) == 3 and w["name"].tolist() == ["a", "b", "c"] and w.loc[1, "snap_status"] == "unlabelled" and w.loc[1, "lat"] == 2.0
    assert w.loc[0, "lon"] == 2.0 and w.loc[0, "raw_lon"] == 1.0 and pd.read_csv(orig / "W.csv")["lon"].tolist() == [1.0, 2.0, 3.0]
    w["judge_result"] = w["judge_result"].astype(object)
    w.loc[0, "judge_result"] = "agreed"
    w.to_csv(work / "W.csv", index=False)
    with pytest.raises(SystemExit):
        H.snap_waypoint_files(snap, ident, ident, orig, work, ("W.csv",))
    with pytest.raises(SystemExit):
        H.require_hcr("red-river-gorge")


def _case():
    return {"easy": True, "candidates": [{"id": 1, "roles": ["automatic snap"], "east_m": 5.0, "north_m": 0.0}, {"id": 2, "roles": ["anchor"], "east_m": 5.0, "north_m": 4.0}]}


def test_judge_apply_rules():
    case = _case()
    d = lambda ch, **k: {"gt_id": "x", "choice": ch, "confidence": "high", **k}
    assert H.resolve(case, d(1), d(1), None)["judge_result"] == "agreed"
    r = H.resolve(case, d(1), d(2), d(2))
    assert r["judge_result"] == "majority" and r["snap_status"] == "judged" and np.allclose(r["offset"], [5, 4])
    r = H.resolve(case, d(1), d(2), d("none"))
    assert r["judge_result"] == "unresolved" and np.allclose(r["offset"], [5, 0])
    assert H.resolve(case, d("none"), d("none"), None)["snap_status"] == "judge_none"
    assert H.same_decision(case, d("custom", custom_east_m=5.5, custom_north_m=0.5), d(1))
    ok = lambda c, e, n: None if abs(e - 5) < 2 else "off the wall"
    with pytest.raises(ValueError, match="does not exist"):
        H.validate_decisions([d(7)], {"x": case}, ok)
    with pytest.raises(ValueError, match="custom spot rejected"):
        H.validate_decisions([d("custom", custom_east_m=-20, custom_north_m=0)], {"x": case}, ok)
    rate, changed = H.calibration({"x": case}, {"x": d(2)})
    assert rate == 0 and changed == ["x"]


# ---------------------------------------------------------------- plan 1


def wall_crop(h=20.0, lean_deg=90.0, seed=0):
    xyz, steep = step_scene(h, face_pts=20000, lean_deg=lean_deg, seed=seed, y=(-20, 20))
    m = (np.abs(xyz[:, 0]) <= 15) & (np.abs(xyz[:, 1]) <= 15)
    return {"xyz": xyz[m], "steep": steep[m]}


def test_face_strip_follows_leaning_wall():
    c = wall_crop(20.0, lean_deg=70)
    m, top = face_strip_mask(c["xyz"], c["steep"], (0.0, 0.0), 0.0, (-1.0, 0.0), StripConfig())  # no grid height: gap rule
    assert top >= 19
    assert face_strip_mask(c["xyz"], c["steep"], (0.0, 0.0), 0.0, (-1.0, 0.0), StripConfig(), wall_height_m=10.0)[1] == 11.0 and c["xyz"][m, 2].max() > 18
    tube = np.hypot(c["xyz"][:, 0], c["xyz"][:, 1]) <= 2.5
    assert c["xyz"][tube & c["steep"], 2].max() < 8  # the old upright tube loses the face


def test_wall_top_stops_at_ledge():
    up = np.r_[np.linspace(0, 10, 200), np.linspace(14.5, 25, 200)]
    assert wall_top(up, StripConfig(top_gap_m=2.0)) == pytest.approx(11, abs=1)


def test_point_pick_is_balanced():
    rng = RNG(0)
    up = rng.uniform(0, 20, 500)
    k = pick_points(up, StripConfig(), rng)
    lo, hi = min(up.min(), 0.0), up.max()
    band = np.clip(((up[k] - lo) / (hi - lo) * 4).astype(int), 0, 3)
    assert len(k) == 48 and np.bincount(band, minlength=4).max() == 12
    few = pick_points(np.r_[rng.uniform(0, 1, 40), [20.0]], StripConfig(), rng)
    assert len(few) == 13  # the thin top band is not topped up


def test_same_labels_for_both_classes():
    c = wall_crop(20.0)
    idx = np.arange(len(c["xyz"]))
    out = []
    for lab in (1, 0):
        s = pd.Series({"x": 0.0, "y": 0.0, "base_z": 0.0, "out_dx": -1.0, "out_dy": 0.0, "label": lab, "source": "osm" if lab else "handgathered_nc"})
        codes, info = block_codes(c, idx, s, pd.DataFrame(), None, StripConfig(), rng=RNG(5))
        out.append((codes == lab, info))
    assert out[0][1] == out[1][1] and (out[0][0] == out[1][0]).all()
    tiny = {"xyz": c["xyz"][:30], "steep": np.zeros(30, bool)}
    s = pd.Series({"x": 0.0, "y": 0.0, "base_z": 0.0, "out_dx": -1.0, "out_dy": 0.0, "label": 1, "source": "osm"})
    assert block_codes(tiny, np.arange(30), s, pd.DataFrame(), None)[0] is None  # under 12 points -> dropped


def _pair(dy, conf=(False, False)):
    return pd.DataFrame({"slug": "s", "sample_id": ["c", "n"], "route_uid": ["c", "n"], "label": [1, 0], "source": ["osm", "handgathered_nc"], "x": [0.0, 0.0],
                         "y": [0.0, dy], "out_dx": -1.0, "out_dy": 0.0, "confidence": 1.0, "confirmed": list(conf)})


def test_close_pair_strips_split_and_too_narrow():
    s, drops = split_strip_widths(_pair(3.0))
    assert not drops and s["has_opposite_neighbour"].all()
    # c is at y=0, n at y=3; along = (0, -1) for out=(-1, 0), so n is on c's negative side
    assert sorted([s.loc[0, "strip_left_m"], s.loc[0, "strip_right_m"]]) == pytest.approx([1.25, 2.5])
    assert s.loc[0, ["strip_left_m", "strip_right_m"]].min() == s.loc[1, ["strip_left_m", "strip_right_m"]].min()
    c = wall_crop()
    m = [face_strip_mask(c["xyz"], c["steep"], (0.0, y), 0.0, (-1.0, 0.0), StripConfig(), r.strip_left_m, r.strip_right_m)[0] for y, r in zip((0.0, 3.0), s.itertuples())]
    assert not (m[0] & m[1]).any()
    gap = c["xyz"][~(m[0] | m[1]) & c["steep"] & (np.abs(c["xyz"][:, 1] - 1.5) < 0.2), 1]
    assert len(gap)  # an unlabelled gap between the two strips
    s2, drops2 = split_strip_widths(_pair(1.5))
    assert len(s2) == 0 and {d["reason"] for d in drops2} == {"strip_too_narrow"}
    s3, _ = split_strip_widths(_pair(2.5).assign(label=1))
    assert (s3["strip_left_m"] == 2.5).all() and (s3["strip_right_m"] == 2.5).all()  # same label: not split


def test_pair_under_1m_and_duplicates():
    kept, drops = clean_samples(_pair(0.6), NegativeConfig())
    assert kept.empty and {d["reason"] for d in drops} == {"pair_under_1m"}
    kept, _ = clean_samples(_pair(0.6, (True, True)), NegativeConfig())
    assert len(kept) == 2
    kept, drops = clean_samples(_pair(1.5).assign(label=1), NegativeConfig())
    assert len(kept) == 1 and drops[0]["reason"] == "duplicate"


def test_pair_acc():
    p = pd.DataFrame({"slug": "s", "x": [0, 0, 100, 0], "y": [0, 3, 0, 50], "label": [1, 0, 0, 1], "score": [0.8, 0.3, 0.1, 0.2]})
    m = pair_metrics(p)
    assert m["pair_n"] == 1 and m["pair_acc"] == 1.0
    assert m["auroc_near"] == 1.0 and m["auroc_isolated"] == 1.0


def test_hcr_fold_split():
    s = pd.DataFrame({"slug": ["horseshoe-canyon-ranch"] * 4 + ["rrg"], "x": [0, 1, 100, 101, 0.0], "y": 0.0,
                      "area_name": ["East Side", None, "West Side", None, None], "source": ["handgathered", "band", "handgathered_nc", "far_cliff", "osm"]})
    sp = apply_hcr_fold(s, np.array(["test"] * 4 + ["val"], dtype=object), "hcr-a")
    assert sp.tolist() == ["train", "train", "test", "test", "val"]


def test_resampling_keeps_labelled_points_and_shuffles():
    rng = RNG(0)
    keep = np.zeros(3000, bool)
    keep[rng.choice(3000, 40, replace=False)] = True
    for n_out in (4096, 2048):
        sel = resample_keep(keep, n_out, rng)
        assert len(sel) == n_out and np.isin(np.flatnonzero(keep), sel).all() and (np.bincount(sel[keep[sel]], minlength=3000)[keep] == 1).all()
    assert not (np.diff(resample_keep(keep, 2048, rng)) > 0).all()


def test_per_sample_loss_weight():
    logits = torch.zeros(2, 2, 60)
    y = torch.full((2, 60), -1)
    y[0, :12], y[1, :48] = 1, 1
    logits[0, 0, :12] = 3.0  # sample 0 is wrong on its 12 points
    w = torch.zeros(2, 60)
    w[0, :12], w[1, :48] = 1 / 12, 1 / 48
    l = per_sample_loss(logits, y, w, -1)
    ce0 = torch.nn.functional.cross_entropy(logits[0:1, :, :12], y[0:1, :12])
    ce1 = torch.nn.functional.cross_entropy(logits[1:2, :, :48], y[1:2, :48])
    assert float(l) == pytest.approx(float((ce0 + ce1) / 2), rel=1e-5)


# ---------------------------------------------------------------- plan 2 / 3


def test_configs_share_scale_and_jitter_in_metres():
    pre, ft = load_cfg("cfgs/pretrain/default.yaml"), load_cfg("cfgs/finetune/default.yaml")
    for k in ("coord_scale", "num_points"):
        assert pre.dataset[k] == ft.dataset[k]
    for c in (pre, ft):
        assert c.datatransforms.kwargs.jitter_sigma / c.dataset.coord_scale == pytest.approx(0.03, rel=0.05)
        assert c.datatransforms.kwargs.jitter_clip / c.dataset.coord_scale == pytest.approx(0.12, rel=0.05)
    assert BlockSpec().coord_scale == pre.dataset.coord_scale


def _tiny_blocks(tmp_path, channels=True):
    n = 500  # channel value = point index / n must survive float16
    r = RNG(0)
    pts = np.column_stack([r.uniform(-10, 10, n), r.uniform(-10, 10, n), r.uniform(0, 20, n)]).astype(np.float32)
    labs = np.full(n, 2, np.int8)
    labs[:30] = 1
    d = tmp_path / "s" / "blocks"
    d.mkdir(parents=True)
    np.save(d / "samples.points.npy", pts)
    np.save(d / "samples.labels.npy", labs)
    if channels:
        feat = np.repeat(np.arange(n, dtype=np.float32)[:, None] / n, len(CHANNELS), 1).astype(np.float16)  # channel value = point index
        write_channels_json(tmp_path, CHANNELS)
    else:
        feat = np.ones(n, np.float16)
    np.save(d / "samples.feat.npy", feat)
    index = pd.DataFrame({"slug": ["s"], "tile": ["samples"], "start": [0], "stop": [n], "label": [1], "confidence": [1.0], "n_climbable": [30], "n_not_climbable": [0]})
    return index, pts


def test_channels_follow_points_and_thinning(tmp_path):
    index, pts = _tiny_blocks(tmp_path)
    spec = BlockSpec(num_points=2048, feature_keys="hag,vert")
    ds = BlockDataset(index, spec, None, train=True, root=tmp_path, class_names=("a", "b"), center_codes=(0, 1), thin={"prob": 1.0, "max_frac": 0.4},
                      sample_weights_per_point=True)
    np.random.seed(0)
    it = ds[0]
    assert it["x"].shape == (2048, 2) and int(it["center"].sum()) == 30 and float(it["w"].sum()) == pytest.approx(1.0)
    src = np.round(it["x"][:, 0].numpy() * 500).astype(int)  # recovered point index
    assert np.allclose(it["pos"].numpy() / spec.coord_scale, pts[src], atol=0.05)
    assert (it["center"].numpy() == (src < 30)).all()
    ev = BlockDataset(index, spec, None, train=False, root=tmp_path, center_codes=(0, 1), thin={"prob": 1.0, "max_frac": 0.4})
    assert ev.thin is None and torch.equal(ev[0]["x"], ev[0]["x"])


def test_old_blocks_read_as_intensity(tmp_path):
    index, _ = _tiny_blocks(tmp_path, channels=False)
    assert BlockDataset(index, BlockSpec(num_points=512, feature_keys="intensity"), None, train=False, root=tmp_path)[0]["x"].shape == (512, 1)
    with pytest.raises(ValueError, match="rebuild"):
        BlockDataset(index, BlockSpec(num_points=512, feature_keys="hag"), None, train=False, root=tmp_path)


def test_feature_keys_and_no_position_in_features():
    pos = RNG(0).uniform(0, 1, (100, 3)).astype(np.float32)
    cols = {"hag": np.full(100, 0.1, np.float32), "vert": np.full(100, 0.2, np.float32)}
    x = make_item(pos, np.zeros(100, np.int64), cols, ["vert", "hag"])["x"]
    assert torch.allclose(x[:, 0], torch.tensor(0.2)) and torch.allclose(x[:, 1], torch.tensor(0.1))
    x2 = make_item(pos + 5, np.zeros(100, np.int64), cols, ["vert", "hag"])["x"]
    assert torch.equal(x, x2)


def test_point_channels_ranges():
    c = {"hag": np.array([0.0, 15.0, np.nan]), "normals": np.array([[0, 0, 1.0], [1.0, 0, 0], [0, 0, 0]]), "planarity": np.array([0.9, 0.5, np.nan]),
         "return_number": np.array([1, 1, 1]), "number_of_returns": np.array([1, 2, 3]), "intensity": np.array([100, 200, 400]), "i99": 200.0,
         "wall_dist": np.array([0.0, 30.0, np.nan])}
    stats = {}
    ch = point_channels(c, None, CHANNELS, stats).astype(float)
    col = {k: ch[:, j] for j, k in enumerate(CHANNELS)}
    assert col["hag"].tolist() == pytest.approx([0, 0.75, 0], abs=1e-3)
    assert col["vert"].tolist() == [1, 0, 1] and col["single"].tolist() == [1, 0, 0]
    assert col["ret_pos"].tolist() == pytest.approx([1, 0.5, 1 / 3], abs=1e-3) and col["intensity"].tolist() == pytest.approx([0.5, 1, 1])
    assert col["wall_dist"].tolist() == pytest.approx([0, 2, 2]) and stats["hag_missing"] == 1 and stats["normal_missing"] == 1
    assert (ch >= -0.1).all() and (ch <= 3).all()


gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="OpenPoints CUDA ops need a GPU")


@gpu
@pytest.mark.parametrize("name", ["pointnext-s", "pointnet2"])
def test_in_channels_auto_and_first_layer_only(name, tmp_path):
    from crag import models
    from crag.pretrain import build_model_from

    c = load_cfg(f"cfgs/pretrain/{name}.yaml", ["dataset.feature_keys=hag,planarity"])
    model, report = build_model_from(c)  # S3DIS weights (4 channels) into a 2-channel model
    assert c.model.encoder_args.in_channels == 2
    assert report["mismatched"]
    models.check_first_layer_only(report)
    torch.save({"encoder": model.encoder.state_dict(), "decoder": model.decoder.state_dict(), "cfg": {"dataset": {"coord_scale": 0.16}}}, tmp_path / "backbone.pt")
    from crag import finetune

    ft = load_cfg("cfgs/finetune/default.yaml", [f"init.arch=cfgs/pretrain/{name}.yaml", f"init.weights={tmp_path / 'backbone.pt'}"])
    with pytest.raises(ValueError, match="coord_scale"):
        finetune.build_model(ft)
    ft = load_cfg("cfgs/finetune/default.yaml", [f"init.arch=cfgs/pretrain/{name}.yaml", f"init.weights={tmp_path / 'backbone.pt'}", "dataset.coord_scale=0.16"])
    m2, rep = finetune.build_model(ft)
    assert ft.model.encoder_args.in_channels == 4 and rep["mismatched"]
    models.check_first_layer_only(rep)
