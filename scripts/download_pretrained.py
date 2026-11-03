#!/usr/bin/env python
"""Download the official OpenPoints S3DIS (area 5) checkpoints used to initialise pretraining.

    python scripts/download_pretrained.py

* PointNeXt-S  -> pretrained/openpoints/s3dis-pointnext-s/
* PointNet++   -> pretrained/openpoints/s3dis-pointnet2/

Links: third_party/PointNeXt/docs/modelzoo.md ("S3DIS (Area 5) Segmentation").
Skips folders that already contain a ``*_ckpt_best.pth``.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FOLDERS = {
    "s3dis-pointnext-s": "https://drive.google.com/drive/folders/1UG8hh_CrUf-OhrYbcDd0zvDtoInrGP1u",
    "s3dis-pointnet2": "https://drive.google.com/drive/folders/1NCy1Av1-TSs_46ngOk181A3BUhc8hpWV",
}

for name, url in FOLDERS.items():
    out = ROOT / "pretrained" / "openpoints" / name
    if list(out.glob("**/*_ckpt_best.pth")):
        print(f"{name}: present")
        continue
    out.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, "-m", "gdown", "--folder", url, "-O", str(out)], check=True)
    print(f"{name}: downloaded to {out}")
