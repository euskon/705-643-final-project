#!/usr/bin/env python
"""Leak test for a fine-tuning set: can a few summary numbers per sample, with no shape
information, tell climbable from not climbable?

Two groups of numbers (plan 1, section 2):

* **A - how the sample was built** (should say nothing about the rock): number of labelled
  points, their height in the block, their height span, their share of the block height,
  their horizontal distance from the block centre, and (V4) strip width and whether a
  close opposite-label sample split the strip. **Target: AUROC 0.45-0.55.**
* **B - plain wall geometry** (may honestly relate to climbing): block height, wall height
  and wall top (V4), share of rock points in the block. This is the **simple baseline the
  network must beat**.

One logistic regression per group is fitted on the train split and scored on validation,
the Red River Gorge test samples, all Horseshoe samples and Horseshoe Climb vs Nc, with 95%
ranges from 1,000 resamples; plus each single number's AUROC (which rule leaks), a separate
"climbable vs far cliff" group-B score, and per stored channel a label check and an area
check (Horseshoe vs Red River Gorge).

    python scripts/s3_leak_test.py --name v3
    python scripts/s3_leak_test.py --name v4

Writes ``data/finetune/<name>/leak_test.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from crag.channels import read_channels_json
from crag.finetune_data import blocks_root, dataset_dir

HCR, RRG = "horseshoe-canyon-ranch", "red-river-gorge"
GROUP_A = ["n_lab", "lab_z_med", "lab_z_span", "lab_rel_height", "lab_xy_dist", "strip_width_m", "has_opposite_neighbour"]
GROUP_B = ["block_height", "wall_height_m", "wall_top_m", "rock_share"]


def features(name: str) -> pd.DataFrame:
    index = pd.read_csv(dataset_dir(name) / "index.csv", low_memory=False)
    index = index[~index["source"].str.startswith("aug_")].reset_index(drop=True)
    root = blocks_root(name)
    chans = read_channels_json(root) or ("intensity",)
    rows = []
    for slug, g in index.groupby("slug"):
        d = root / slug / "blocks"
        pts, labs, feat = (np.load(d / f"samples.{k}.npy", mmap_mode="r") for k in ("points", "labels", "feat"))
        for r in g.itertuples():
            p, l = np.asarray(pts[r.start:r.stop]), np.asarray(labs[r.start:r.stop])
            f = np.asarray(feat[r.start:r.stop], np.float32).reshape(r.stop - r.start, -1)
            own = np.isin(l, (0, 1))
            h = float(p[:, 2].max()) if len(p) else 0.0
            z = p[own, 2]
            row = {"i": r.Index, "n_lab": int(own.sum()), "block_height": h, "rock_share": float(np.isin(l, (0, 1, 3, 4, -1)).mean())}
            if own.any():
                row.update(lab_z_med=float(np.median(z)), lab_z_span=float(z.max() - z.min()), lab_rel_height=float(np.median(z) / max(h, 1e-6)),
                           lab_xy_dist=float(np.median(np.hypot(p[own, 0], p[own, 1]))))
                row.update({f"ch_{c}": float(np.median(f[own, j])) for j, c in enumerate(chans)})
            rows.append(row)
    F = pd.DataFrame(rows).set_index("i").sort_index()
    out = index.join(F, rsuffix="_f")
    if "strip_left_m" in out:
        out["strip_width_m"] = out["strip_left_m"] + out["strip_right_m"]
    if "has_opposite_neighbour" in out:
        out["has_opposite_neighbour"] = out["has_opposite_neighbour"].astype(float)
    return out[out["n_lab"] > 0].reset_index(drop=True)


def boot_auc(y, s, n=1000, seed=0) -> dict:
    y, s = np.asarray(y), np.asarray(s)
    if len(np.unique(y)) < 2:
        return {"auroc": None, "n": int(len(y))}
    rng, vals = np.random.default_rng(seed), []
    for _ in range(n):
        k = rng.integers(0, len(y), len(y))
        if len(np.unique(y[k])) == 2:
            vals.append(roc_auc_score(y[k], s[k]))
    return {"auroc": round(float(roc_auc_score(y, s)), 3), "lo": round(float(np.quantile(vals, 0.025)), 3), "hi": round(float(np.quantile(vals, 0.975)), 3),
            "n": int(len(y)), "n_pos": int((y == 1).sum())}


def evaluation_sets(df: pd.DataFrame) -> dict[str, pd.Series]:
    hcr = df["slug"] == HCR
    hand = df["source"].isin(["handgathered", "handgathered_nc"])
    return {"val": df["split"] == "val", "rrg_test": (df["split"] == "test") & (df["slug"] == RRG), "test": df["split"] == "test",
            "hcr_all": hcr & (df["split"] == "test"), "hcr_climb_vs_nc": hcr & hand & (df["split"] == "test")}


def fit_score(df: pd.DataFrame, cols: list[str], sets: dict) -> dict:
    cols = [c for c in cols if c in df and df[c].notna().any()]
    if not cols:
        return {}
    X = df[cols].astype(float).fillna(df[cols].astype(float).median())
    tr = df["split"] == "train"
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(X[tr], df.loc[tr, "label"])
    s = model.predict_proba(X)[:, 1]
    return {"columns": cols, **{k: boot_auc(df.loc[m, "label"], s[m.to_numpy()]) for k, m in sets.items() if m.any()}}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True)
    args = ap.parse_args()
    df = features(args.name)
    sets = evaluation_sets(df)
    res = {"name": args.name, "n_samples": int(len(df)), "group_A": fit_score(df, GROUP_A, sets), "group_B": fit_score(df, GROUP_B, sets)}
    single = {}
    for c in GROUP_A + GROUP_B:
        if c in df and df[c].notna().sum() > 10 and df[c].nunique() > 1:
            single[c] = {k: boot_auc(df.loc[m, "label"], df.loc[m, c].fillna(df[c].median()), n=200)["auroc"] for k, m in {"all": df["label"].notna(), **sets}.items() if m.any()}
    res["single"] = single
    fc = df[(df["label"] == 1) | (df["source"] == "far_cliff")]
    if (fc["source"] == "far_cliff").any():
        res["group_B_climb_vs_far_cliff"] = fit_score(fc, GROUP_B, {"val": fc["split"] == "val", "test": fc["split"] == "test", "all": fc["label"].notna()})
    chans = {}
    for c in [c for c in df if c.startswith("ch_")]:
        lab = {k: boot_auc(df.loc[m, "label"], df.loc[m, c], n=200)["auroc"] for k, m in (("val", sets["val"]), ("hcr_climb_vs_nc", sets["hcr_climb_vs_nc"])) if m.any()}
        ar = df["slug"].isin([HCR, RRG])
        area = boot_auc((df.loc[ar, "slug"] == HCR).astype(int), df.loc[ar, c], n=200)["auroc"]
        flag = area is not None and max(area, 1 - area) > 0.8 and all(v is None or 0.45 <= v <= 0.55 for v in lab.values())
        chans[c[3:]] = {"label": lab, "area_hcr_vs_rrg": area, "flag_area_identifier": bool(flag)}
    res["channels"] = chans
    a = res["group_A"]
    res["group_A_pass"] = all(v.get("auroc") is None or 0.45 <= v["auroc"] <= 0.55 or (v["lo"] <= 0.5 <= v["hi"]) for k, v in a.items() if k != "columns")
    (dataset_dir(args.name) / "leak_test.json").write_text(json.dumps(res, indent=1))
    print(f"leak test {args.name}: {len(df)} samples")
    for g in ("group_A", "group_B"):
        print(f"\n{g} ({', '.join(res[g].get('columns', []))})")
        for k, v in res[g].items():
            if k != "columns":
                print(f"  {k:18s} AUROC {v['auroc']}  [{v.get('lo')}, {v.get('hi')}]  n={v['n']}")
    print("\nsingle numbers (AUROC):")
    print(pd.DataFrame(single).T.to_string())
    if "group_B_climb_vs_far_cliff" in res:
        print("\nclimbable vs far cliff, group B:", {k: v["auroc"] for k, v in res["group_B_climb_vs_far_cliff"].items() if k != "columns"})
    print("\nchannels:", json.dumps(chans))
    print("\ngroup A within 0.45-0.55 (or range includes 0.5):", "PASS" if res["group_A_pass"] else "FAIL - a rule still differs by label")


if __name__ == "__main__":
    main()
