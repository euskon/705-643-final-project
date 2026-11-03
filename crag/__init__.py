"""Multi-area climbing-crag pipeline: registry -> LiDAR -> pseudolabels -> pretrain -> fine-tune.

Steps (each has a runnable script in ``scripts/``; see ``docs/pipeline.md``):

1. ``registry``/``mp``/``ept_index``/``tiles``/``labeling``/``pseudolabels`` -
   build the area registry, download USGS LiDAR tiles incrementally and turn
   them into pseudolabelled training blocks.
2. ``models``/``train``/``datasets`` - pretrain an OpenPoints (PointNeXt /
   PointNet++) backbone on the pseudolabels.
3. ``openclimbing``/``crossref``/``groundtruth``/``negatives`` - route-level
   GPS from OpenStreetMap (what openclimbing.org renders), matched to Mountain
   Project, plus the hand-gathered waypoints and verified hard negatives.
4. ``finetune`` - fine-tune the pretrained backbone with a swappable head.

The cliff heuristics (``labeling``), waypoint loading/snapping (``groundtruth``)
and the EPT reader (``tiles.read_ept``) were carried over from the single-area
proof of concept, which lives on unchanged in ``proof-of-concept/``.
"""

import os as _os
import sys as _sys

# PDAL/PROJ find their database through PROJ_DATA, which `conda activate` sets.
# Set it here too so `<env>/bin/python scripts/...` works without activation.
_proj = _os.path.join(_sys.prefix, "share", "proj")
if _os.path.isdir(_proj):
    _os.environ.setdefault("PROJ_DATA", _proj)
    _os.environ.setdefault("PROJ_LIB", _proj)
