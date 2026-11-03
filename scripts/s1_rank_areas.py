#!/usr/bin/env python
"""Step 1a - rank candidate Mountain Project areas by popularity (MP page views).

Mountain Project publishes no popularity ranking, but every area page shows
"Page Views: N total". This script fetches each candidate in
``registry/mp_candidates.csv`` (one cached request per area, 60 s apart),
drops candidates nested inside another candidate (so crags aren't counted
twice), and writes ``registry/mp_ranked.csv`` sorted by total page views.

    python scripts/s1_rank_areas.py              # ~1 min per uncached area
    python scripts/s1_rank_areas.py --refresh    # re-fetch page views

To consider another area, add a row to ``registry/mp_candidates.csv``.
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from crag import mp
from crag.cli import setup_logging
from crag.paths import CANDIDATES_CSV, REGISTRY_DIR

log = logging.getLogger("rank_areas")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--candidates", type=Path, default=CANDIDATES_CSV)
    ap.add_argument("--out", type=Path, default=REGISTRY_DIR / "mp_ranked.csv")
    ap.add_argument("--refresh", action="store_true", help="re-fetch area pages")
    args = ap.parse_args()
    setup_logging()

    cands = pd.read_csv(args.candidates, dtype={"mp_id": "int64"})
    rows = []
    for i, c in enumerate(cands.itertuples(), 1):
        log.info("[%d/%d] %s (%s)", i, len(cands), c.name, c.mp_id)
        info = mp.parse_area_page(mp.fetch_area_page(c.mp_id, refresh=args.refresh))
        rows.append(
            {
                "candidate_name": c.name,
                "mp_id": c.mp_id,
                "name": info["name"],
                "state": info["state"],
                "url": info["url"] or mp.area_url(c.mp_id),
                "lat": info["lat"],
                "lon": info["lon"],
                "page_views_total": info["page_views_total"],
                "page_views_month": info["page_views_month"],
                "ancestors": ">".join(str(a) for a, _ in info["breadcrumb"]),
                "breadcrumb": " > ".join(n for _, n in info["breadcrumb"]),
            }
        )
    df = pd.DataFrame(rows)
    ids = set(df["mp_id"])
    df["nested_in"] = [
        next((int(a) for a in anc.split(">") if a and int(a) in ids), None) for anc in df["ancestors"]
    ]
    df["name_mismatch"] = [
        not (str(n).lower().startswith(str(c).lower()[:8]) or str(c).lower().startswith(str(n).lower()[:8]))
        for n, c in zip(df["name"], df["candidate_name"])
    ]
    df["retrieved"] = dt.date.today().isoformat()
    keep = df[df["nested_in"].isna()].sort_values("page_views_total", ascending=False)
    keep = keep.assign(rank=range(1, len(keep) + 1))
    nested = df[df["nested_in"].notna()].assign(rank=None)
    out = pd.concat([keep, nested], ignore_index=True)
    out.to_csv(args.out, index=False)
    for r in nested.itertuples():
        log.info("dropped nested candidate %s (inside %s)", r.name, r.nested_in)
    for r in df[df["name_mismatch"]].itertuples():
        log.warning("check candidate %r: MP page is titled %r", r.candidate_name, r.name)
    log.info("wrote %s (%d ranked areas)", args.out, len(keep))
    print(keep[["rank", "name", "state", "page_views_total"]].head(60).to_string(index=False))


if __name__ == "__main__":
    main()
