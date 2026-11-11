"""Pseudolabels, block packing, cross-referencing and hard-negative generation on synthetic scenes."""

import numpy as np
import pandas as pd
import pytest
from scipy.spatial import cKDTree

from crag import crossref
from crag.finetune_data import LABEL_MODES, augment_negative, block_codes
from crag.negatives import NegativeConfig, band_candidates, greedy_select
from crag.datasets import BlockDataset, BlockSpec
from crag.labeling import compute_normals, slope_deg, surface_geometry
from crag.pseudolabels import IGNORE, META_COLS, PseudoLabelConfig, base_classes, make_blocks, write_blocks
from crag.tiles import Tile
from crag.groundtruth import GroundTruthConfig, snap_one
from crag.labeling import LabelConfig, PointCloud
from crag.walls import WallConfig, point_wall_distance, wall_grid


def test_base_classes():
    cfg = PseudoLabelConfig(min_planarity=0.5, multi_return_rock="ignore")
    up, wall, slope45 = [0, 0, 1], [1, 0, 0], [0.7071, 0, 0.7071]
    normals = np.array([up, wall, wall, slope45, up, wall], float)
    rn = np.array([1, 1, 2, 1, 1, 1])
    nr = np.array([1, 1, 2, 1, 3, 1])  # point 2: last of 2 returns; point 4: 1st of 3 -> canopy
    planarity = np.array([0.9, 0.9, 0.9, 0.9, 0.9, 0.1])  # point 5: steep but not a flat patch (trunk / branch)
    las = np.ones(6, np.uint8)
    cls, cliff = base_classes(normals, rn, nr, las, cfg, planarity)
    # ground, cliff, multi-return rock -> ignore, steep_terrain (45 deg), vegetation, steep_terrain (not planar)
    assert cls.tolist() == [0, 3, IGNORE, 2, 1, 2]
    assert cliff.tolist() == [False, True, False, False, False, False]
    for mode, code in (("cliff", 3), ("steep_terrain", 2)):
        c = PseudoLabelConfig(min_planarity=0.5, multi_return_rock=mode)
        assert base_classes(normals, rn, nr, las, c, planarity)[0][2] == code
    las[0] = 6
    assert base_classes(normals, rn, nr, las, cfg, planarity)[0][0] == IGNORE
    d = base_classes(normals, rn, nr, las, PseudoLabelConfig(), planarity)[0]  # defaults: no planarity gate, multi-return rock = cliff
    assert d.tolist() == [IGNORE, 3, 3, 2, 1, 3]
    las[1] = 3  # LAS vegetation wins over geometry
    assert base_classes(normals, rn, nr, las, cfg, planarity)[0][1] == 1
    # wall gate: a steep point more than wall.near_m from the bare-earth wall is steep_terrain, not cliff
    las[:] = 1
    wall_dist = np.array([0, 1.0, 0, 0, 0, 9.0])
    assert base_classes(normals, rn, nr, las, PseudoLabelConfig(), planarity, wall_dist)[0].tolist() == [0, 3, 3, 2, 1, 2]
    assert base_classes(normals, rn, nr, las, PseudoLabelConfig(wall_gate=False), planarity, wall_dist)[0][5] == 3


def _cliff_scene(rng, height=15.0):
    """Flat valley floor (z=0, x<0), a vertical face at x in [0, 1], a plateau (z=height, x>1),
    and a 10 m tree trunk standing on the valley floor at (-20, 50)."""
    floor = np.column_stack([rng.uniform(-50, 0, 20000), rng.uniform(0, 100, 20000), rng.normal(0, 0.03, 20000)])
    face = np.column_stack([rng.uniform(0, 1, 6000), rng.uniform(0, 100, 6000), rng.uniform(0, height, 6000)])
    top = np.column_stack([rng.uniform(1, 30, 12000), rng.uniform(0, 100, 12000), height + rng.normal(0, 0.03, 12000)])
    trunk = np.column_stack([-20 + rng.normal(0, 0.1, 400), 50 + rng.normal(0, 0.1, 400), rng.uniform(0, 10, 400)])
    xyz = np.vstack([floor, face, top, trunk])
    kind = np.repeat(["floor", "face", "top", "trunk"], [len(floor), len(face), len(top), len(trunk)])
    return xyz, kind


def test_wall_mask_finds_walls_not_trees():
    xyz, kind = _cliff_scene(np.random.default_rng(0))
    g = wall_grid(xyz, np.ones(len(xyz), bool))
    d = point_wall_distance(xyz, np.ones(len(xyz), bool))
    assert np.median(d[kind == "face"]) <= 1.0
    assert d[kind == "trunk"].min() > 10  # the trunk's lowest point is the DEM; it never makes a wall
    assert g.distance(np.array([[-8.0, 50.0]]))[0] == pytest.approx(8, abs=2)
    low = _cliff_scene(np.random.default_rng(0), height=2.0)[0]  # a 2 m step is not a wall
    assert not wall_grid(low, np.ones(len(low), bool)).wall.any()


def _scene_pc(xyz, steep):
    from scipy.spatial import cKDTree as T
    si = np.flatnonzero(steep)
    return PointCloud(xyz=xyz, normals=np.zeros_like(xyz), steep=steep, tree_xy=T(xyz[:, :2]), steep_idx=si,
                      steep_tree_xy=T(xyz[si, :2]), wall_dist=point_wall_distance(xyz, np.ones(len(xyz), bool)))


def test_wall_check_moves_points_to_the_wall_or_rejects_them():
    xyz, kind = _cliff_scene(np.random.default_rng(1))
    pc = _scene_pc(xyz, kind == "face")
    s = snap_one(pc, np.array([-9.0, 50.0]), LabelConfig(), GroundTruthConfig())  # stood 9 m out on the valley floor
    assert s["status"] == "ok"
    assert s["snap_dist_m"] == pytest.approx(9, abs=1.5) and s["snap_x"] < 1.5
    assert s["wall_dist_m"] == pytest.approx(9, abs=2) and s["wall_height_m"] > 10
    assert snap_one(pc, np.array([-35.0, 50.0]), LabelConfig(), GroundTruthConfig())["status"] == "off_wall"  # too far
    # steep points on a 2 m step (e.g. a boulder edge) are not a wall to climb
    low, lk = _cliff_scene(np.random.default_rng(1), height=2.0)
    assert snap_one(_scene_pc(low, lk == "face"), np.array([-3.0, 50.0]), LabelConfig(), GroundTruthConfig())["status"] == "off_wall"


def test_surface_geometry_ignores_canopy():
    rng = np.random.default_rng(0)
    # a vertical rock face (x = 0) with a blob of foliage hanging 1 m in front of it
    face = np.column_stack([rng.normal(0, 0.02, 3000), rng.uniform(0, 20, 3000), rng.uniform(0, 20, 3000)])
    leaves = rng.normal([1.0, 10.0, 10.0], 0.6, (3000, 3))
    xyz = np.vstack([face, leaves])
    surface = np.r_[np.ones(3000, bool), np.zeros(3000, bool)]
    normals, planarity = surface_geometry(xyz, surface, k=12, voxel_m=0.5)
    near = (np.abs(face[:, 1] - 10) < 1.5) & (np.abs(face[:, 2] - 10) < 1.5)  # rock right behind the leaves
    assert slope_deg(normals[:3000][near]).min() > 80
    assert np.median(planarity[:3000]) > 0.5
    mixed = compute_normals(xyz, 12, 0.5)  # the old all-points fit is pulled off the wall by the leaves
    assert slope_deg(mixed[:3000][near]).min() < slope_deg(normals[:3000][near]).min()
    line = np.column_stack([rng.normal(0, 0.02, 500), rng.normal(0, 0.02, 500), rng.uniform(0, 20, 500)])  # a trunk
    assert np.median(surface_geometry(line, np.ones(500, bool), k=12, voxel_m=0.5)[1]) < 0.3


def test_blocks():
    cfg = PseudoLabelConfig()
    rng = np.random.default_rng(0)
    # a 40 m tall wall along x = 0, plus flat ground
    wall = np.column_stack([np.zeros(4000), rng.uniform(0, 100, 4000), rng.uniform(100, 140, 4000)])
    ground = np.column_stack([rng.uniform(1, 100, 6000), rng.uniform(0, 100, 6000), np.full(6000, 100.0)])
    xyz = np.vstack([wall, ground])
    lab = np.r_[np.full(4000, 3), np.zeros(6000)].astype(np.int8)
    blocks = make_blocks(xyz, lab, np.ones(len(xyz), np.float32), Tile(-100.0, 0.0, 200.0), cfg, seed=1)
    meta = blocks["meta"]
    assert meta.shape[1] == len(META_COLS)
    assert meta[-1, 2] == len(blocks["points"]) == len(blocks["labels"])
    assert (meta[:, 1][1:] == meta[:, 2][:-1]).all()  # contiguous offsets
    assert np.abs(blocks["points"][:, :2]).max() <= cfg.block_size_m / 2 + 1e-4
    assert (blocks["points"][:, 2] >= 0).all()
    assert set(np.unique(blocks["labels"])) <= {0, 3}


def test_window_sampling(tmp_path):
    cfg = PseudoLabelConfig()
    rng = np.random.default_rng(0)
    # ground over a 200 m tile with a wall along x = 10 (the edge between two grid blocks)
    ground = np.column_stack([rng.uniform(0, 200, 60000), rng.uniform(0, 200, 60000), np.full(60000, 100.0)])
    wall = np.column_stack([np.full(8000, 10.0), rng.uniform(0, 200, 8000), rng.uniform(100, 130, 8000)])
    xyz = np.vstack([ground, wall])
    lab = np.r_[np.zeros(60000), np.full(8000, 3)].astype(np.int8)
    tile = Tile(-10.0, 0.0, 200.0)
    blocks = make_blocks(xyz, lab, np.ones(len(xyz), np.float32), tile, cfg, seed=1)
    write_blocks(tmp_path / "a" / "blocks", tile.id, blocks, "x")
    index = pd.DataFrame(blocks["meta"], columns=META_COLS)
    index.insert(0, "tile", tile.id)
    index.insert(0, "slug", "a")
    spec = BlockSpec(num_points=2048)
    plain = BlockDataset(index, spec, None, train=True, root=tmp_path)
    win = BlockDataset(index, spec, None, train=True, root=tmp_path, window={"block_size_m": 20.0, "rock_center_prob": 1.0})
    np.random.seed(0)
    k = int(np.flatnonzero(index["n_cliff"].to_numpy() > 0)[0])
    for ds in (plain, win):
        it = ds[k]
        assert it["pos"].shape == (2048, 3) and it["y"].shape == (2048,)
    local, raw, _ = win._window(k, np.random.default_rng(1))
    assert np.abs(local[:, :2]).max() <= 10.0 + 1e-4 and local[:, 2].min() == 0
    # centred on rock: the wall runs through the middle of the window, not along its edge
    rock_x = local[raw == 3, 0]
    assert len(rock_x) and np.abs(rock_x).max() < 1.0
    assert not win.window is None and plain.window is None
    assert BlockDataset(index, spec, None, train=False, root=tmp_path, window={"block_size_m": 20.0}).window is None  # eval: fixed blocks


def test_crossref_tiers():
    mpr = pd.DataFrame({
        "route": ["The Big Crack", "Sunshine Dihedral", "Moonlight", "Far Away"],
        "route_id": [11, 12, 13, 14], "wall": ["W1", "W1", "W2", "W9"], "location": ["W1 > A"] * 2 + ["W2 > A", "W9 > B"],
        "rating": ["5.9", "5.10b", "5.11a", "5.9"], "length_m": [20.0, 25, 30, 10], "lat": [37.0, 37.0, 37.001, 38.0], "lon": [-83.0, -83.0, -83.001, -84.0],
    })
    osm = pd.DataFrame({
        "name": ["big crack", "Sunshine Dihedrel", "Moonlight", "Far Away", "Unknown"],
        "grade": ["5.9", "5.10c", "5.12d", "5.9", None], "mp_route_id_tag": [np.nan, np.nan, np.nan, np.nan, np.nan],
        "lat": [37.0001] * 5, "lon": [-83.0001] * 5, "osm_type": "node", "osm_id": range(5),
    })
    osm.loc[4, "mp_route_id_tag"] = 13  # links Moonlight directly
    m = crossref.match_routes(osm, mpr, max_pin_dist_m=2000)
    assert m["match_type"].tolist() == ["name", "fuzzy", "none", "none", "mp_url"]
    # "Moonlight" by name is blocked twice: grade 5.12d vs 5.11a, and the MP route is already taken by the URL match
    assert m.loc[4, "mp_route_id"] == 13 and m.loc[0, "mp_route_id"] == 11
    assert crossref.normalize_name("The Café & Bar (aka X)") == "cafe and bar"
    cov = crossref.crag_coverage(m, mpr)
    assert cov.set_index("wall").loc["W1", "coverage"] == 1.0


class FakeCloud:
    """A straight 30 m cliff along x = 0 (steep points) with ground in front."""

    def __init__(self):
        rng = np.random.default_rng(0)
        n = 30000
        wall = np.column_stack([rng.normal(0, 0.3, n), rng.uniform(-200, 200, n), rng.uniform(100, 130, n)])
        ground = np.column_stack([rng.uniform(2, 60, n), rng.uniform(-200, 200, n), np.full(n, 100.0)])
        self.xyz = np.vstack([wall, ground])
        self.steep = np.r_[np.ones(n, bool), np.zeros(n, bool)]
        self.normals = np.tile([1.0, 0, 0], (2 * n, 1)).astype(np.float32)
        self.normals[n:] = [0, 0, 1]

    def crop(self, cx, cy, half):
        m = (np.abs(self.xyz[:, 0] - cx) <= half) & (np.abs(self.xyz[:, 1] - cy) <= half)
        return {"xyz": self.xyz[m], "steep": self.steep[m], "normals": self.normals[m], "intensity": np.ones(m.sum(), np.float32),
                "cls": np.where(self.steep[m], 3, 0).astype(np.int8), "multi": np.zeros(m.sum(), bool)}


def test_band_candidates_and_selection():
    cloud = FakeCloud()
    cfg = NegativeConfig(band_min_m=10, band_max_m=40, doc_min_routes=3, min_coverage=0.5)
    ys = [-60.0, -54.0, -48.0, 48.0, 54.0, 60.0]  # two documented clusters, a 96 m gap between them
    pos = pd.DataFrame({"status": "ok", "snap_x": 0.0, "snap_y": ys, "mp_wall": "W"})
    known = pos[["snap_x", "snap_y"]].to_numpy()
    band = band_candidates(cloud, pos, known, {"W": 1.0}, cfg)
    d = cKDTree(known).query(band[["x", "y"]].to_numpy())[0]
    assert len(band) and (d >= 10 - 1e-6).all() and (d <= 40 + 1e-6).all()
    assert band["bracketed"].any() and (band.loc[band["bracketed"], "y"].abs() < 48).all()  # between the clusters
    assert band_candidates(cloud, pos, known, {"W": 0.2}, cfg).empty  # undocumented wall -> no negatives
    sel = greedy_select(band, 5, 6.0)
    assert len(sel) == 5
    xy = sel[["x", "y"]].to_numpy()
    assert min(np.hypot(*(xy[i] - xy[j])) for i in range(5) for j in range(i)) >= 6.0


def test_block_codes_and_augmentation():
    cloud = FakeCloud()
    crop = cloud.crop(0, 0, 10)
    idx = np.arange(len(crop["xyz"]))
    s = pd.Series({"x": 0.0, "y": 0.0, "label": 1, "source": "osm", "base_z": 100.0, "top_z": 115.0})
    others = pd.DataFrame({"x": [0.0], "y": [8.0], "label": [0]})
    codes = block_codes(crop, idx, s, others, None, 2.5, 7.5, max_center_points=0)
    xyz = crop["xyz"]
    own = (np.hypot(xyz[:, 0], xyz[:, 1]) <= 2.5) & crop["steep"] & (xyz[:, 2] <= 115)
    assert (codes[own] == 1).all()
    assert (codes[(np.hypot(xyz[:, 0], xyz[:, 1] - 8) <= 2.5) & crop["steep"]] == 4).all()
    capped = block_codes(crop, idx, s, others, None, 2.5, 7.5, max_center_points=48)
    assert (capped == 1).sum() == 48
    mapped = np.vectorize(LABEL_MODES["center_only"].get)(codes)
    assert set(np.unique(mapped)) <= {-1, 1}
    local = (xyz - [0, 0, 100]).astype(np.float32)
    sq = augment_negative(local, codes, "squash", np.array([1.0, 0.0]))
    assert sq is not None and (sq[1] == 1).sum() == 0 and sq[0][:, 2].max() < local[:, 2].max()


def test_assign_splits_balances_giant_group():
    import pandas as pd

    from crag.finetune_data import assign_splits

    # one area: a giant group (60% of samples) plus 20 small groups
    rows = [{"slug": "a", "group": "a:big", "label": i % 2, "source": "band" if i % 3 == 0 else "osm"} for i in range(600)]
    for g in range(20):
        rows += [{"slug": "a", "group": f"a:g{g}", "label": i % 2, "source": "band" if i % 3 == 0 else "osm"} for i in range(20)]
    rows += [{"slug": "hold", "group": "hold:0", "label": 1, "source": "handgathered"}]
    s = pd.DataFrame(rows)
    split = pd.Series(assign_splits(s, 0.15, 0.15, test_areas=("hold",)), index=s.index)
    assert (split[s["slug"] == "hold"] == "test").all()
    assert (split[s["group"] == "a:big"] == "train").all()  # the giant group can't swamp val/test
    a = split[s["slug"] == "a"].value_counts(normalize=True)
    assert abs(a["val"] - 0.15) < 0.05 and abs(a["test"] - 0.15) < 0.05
    assert (s.assign(split=split).groupby("group")["split"].nunique() == 1).all()  # groups stay whole
