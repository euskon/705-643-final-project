import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import crag.paths  # noqa: E402

crag.paths.add_import_paths()
