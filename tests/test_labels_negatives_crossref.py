"""Pseudolabels, block packing, cross-referencing and hard-negative generation on synthetic scenes."""

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from crag import crossref
from crag.finetune_data import LABEL_MODES, augment_negative, block_codes
from crag.negatives import NegativeConfig, band_candidates, greedy_select
from crag.pseudolabels import IGNORE, PseudoLabelConfig, base_classes, final_labels, make_blocks
from crag.tiles import Tile


def test_base_classes():
    cfg = PseudoLabelConfig()
    up, wall, slope45 = [0, 0, 1], [1, 0, 0], [0.7071, 0, 0.7071]
    normals = np.array([up, wall, wall, slope45, up], float)
    rn = np.array([1, 1, 1, 1, 1])
    nr = np.array([1, 1, 2, 1, 3])  # point 2: steep but multi-return (last return of 2); point 4: 1st of 3 -> canopy
    las = np.ones(5, np.uint8)
    cls, steep = base_classes(normals, rn, nr, las, cfg)
    # ground, cliff, steep_terrain (not single-return), steep_terrain (45 deg), vegetation
    assert cls.tolist() == [0, 3, 1, 2, 1]  # non-last returns are canopy, whatever the slope
    assert steep.tolist()[:2] == [False, True]
    las[0] = 6
    assert base_classes(normals, rn, nr, las, cfg)[0][0] == IGNORE


def test_final_labels_and_blocks():
    cfg = PseudoLabelConfig()
    rng = np.random.default_rng(0)
    # a 40 m tall wall along x = 0, plus flat ground
    wall = np.column_stack([np.zeros(4000), rng.uniform(0, 100, 4000), rng.uniform(100, 140, 4000)])
    ground = np.column_stack([rng.uniform(1, 100, 6000), rng.uniform(0, 100, 6000), np.full(6000, 100.0)])
    xyz = np.vstack([wall, ground])
    cls = np.r_[np.full(4000, 3), np.zeros(6000)].astype(np.int8)
    pos = pd.DataFrame({"x": [0.0], "y": [10.0], "base_z": [100.0], "top_z": [115.0]})
    anchors = np.array([[0.0, 10.0], [0.0, 98.0]])  # 40 m ignore buffers leave y in (50, 58) free
    lab = final_labels(xyz, cls, pos, anchors, cfg)
    d20 = np.hypot(xyz[:, 0], xyz[:, 1] - 10)
    in_cyl = (d20 <= 2.5) & (xyz[:, 2] <= 115) & (cls == 3)
    assert (lab[in_cyl] == 4).all() and in_cyl.any()
    assert (lab[(cls == 3) & (d20 > 2.5) & (d20 < 30)] == IGNORE).all()  # near a pin, not in a cylinder
    far = (cls == 3) & (xyz[:, 1] > 51) & (xyz[:, 1] < 57)
    assert (lab[far] == 3).all()  # > 40 m from every anchor
    assert (lab[cls == 0] == 0).all()
    blocks = make_blocks(xyz, lab, np.ones(len(xyz), np.float32), Tile(-100.0, 0.0, 200.0), cfg, seed=1)
    meta = blocks["meta"]
    assert meta[-1, 2] == len(blocks["points"]) == len(blocks["labels"])
    assert (meta[:, 1][1:] == meta[:, 2][:-1]).all()  # contiguous offsets
    assert np.abs(blocks["points"][:, :2]).max() <= cfg.block_size_m / 2 + 1e-4
    assert (blocks["points"][:, 2] >= 0).all()


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
