"""Every directory the pipeline reads or writes, in one place.

Large data can live on another disk: set ``CRAG_DATA_DIR`` (defaults to
``<repo>/data``) and ``CRAG_RUNS_DIR`` (defaults to ``<repo>/runs``).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("CRAG_DATA_DIR", PROJECT_DIR / "data"))
RUNS_DIR = Path(os.environ.get("CRAG_RUNS_DIR", PROJECT_DIR / "runs"))

REGISTRY_DIR = PROJECT_DIR / "registry"
AREAS_JSON = Path(os.environ.get("CRAG_REGISTRY", REGISTRY_DIR / "areas.json"))
CANDIDATES_CSV = REGISTRY_DIR / "mp_candidates.csv"
HELDOUT_REGIONS_JSON = REGISTRY_DIR / "heldout_regions.json"

RAW_DIR = DATA_DIR / "raw"  # cached HTTP responses (MP pages, EPT index, Overpass)
MP_ROUTES_DIR = DATA_DIR / "mountain_project"  # merged route-finder CSVs (committed)
LIDAR_DIR = DATA_DIR / "lidar"  # raw EPT tiles, one .npz per tile
PRETRAIN_DIR = DATA_DIR / "pretrain"  # pseudolabelled blocks for step 2
ROUTES_DIR = DATA_DIR / "routes"  # step 3: OSM routes + cross-reference
FINETUNE_DIR = DATA_DIR / "finetune"  # step 3/4: positives + hard negatives + blocks

PRETRAINED_DIR = PROJECT_DIR / "pretrained"
POINTNEXT_DIR = PROJECT_DIR / "third_party" / "PointNeXt"
CFG_DIR = PROJECT_DIR / "cfgs"
KEYS_FILE = PROJECT_DIR / ".keys"


def add_import_paths() -> None:
    """Make ``openpoints`` (the PointNeXt submodule) importable and quiet its warnings."""
    import warnings

    if str(POINTNEXT_DIR) not in sys.path:
        sys.path.insert(0, str(POINTNEXT_DIR))
    # OpenPoints predates torch 2.x: silence its deprecation noise only
    warnings.filterwarnings("ignore", category=FutureWarning, message=r".*torch\.cuda\.amp.*")
    warnings.filterwarnings("ignore", category=UserWarning, module=r"openpoints.*")
    warnings.filterwarnings("ignore", message=r".*torch\.cuda\.\*DtypeTensor.*")
    warnings.filterwarnings("ignore", message=r"To copy construct from a tensor.*")
