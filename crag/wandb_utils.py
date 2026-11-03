"""Weights & Biases helpers. The API key is read from ``.keys`` (gitignored) - never logged.

``.keys`` may hold the bare key on one line, or ``WANDB_API_KEY=...``.
Every script takes ``--wandb {online,offline,disabled}``; ``offline`` writes
runs to ``./wandb`` for a later ``wandb sync``.
"""

from __future__ import annotations

import logging
import os
import re

from crag.paths import KEYS_FILE, RUNS_DIR

log = logging.getLogger(__name__)

DEFAULT_PROJECT = "crag-detection"
DEFAULT_ENTITY = os.environ.get("WANDB_ENTITY")  # None -> the key's default entity


def load_key() -> str | None:
    if os.environ.get("WANDB_API_KEY"):
        return os.environ["WANDB_API_KEY"]
    if not KEYS_FILE.exists():
        return None
    for line in KEYS_FILE.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = re.match(r"(?:WANDB_API_KEY\s*=\s*)?['\"]?([A-Za-z0-9_\-]{20,})['\"]?$", line)
        if m:
            return m.group(1)
    return None


def init(mode: str = "online", project: str = DEFAULT_PROJECT, **kwargs):
    """``wandb.init`` with the key from ``.keys``; returns the run or None if disabled."""
    if mode == "disabled":
        return None
    import wandb

    if mode == "online":
        key = load_key()
        if not key:
            log.warning("no W&B key in %s or $WANDB_API_KEY - logging offline", KEYS_FILE)
            mode = "offline"
        else:
            os.environ["WANDB_API_KEY"] = key
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    kwargs.setdefault("dir", str(RUNS_DIR))
    kwargs.setdefault("entity", DEFAULT_ENTITY)
    return wandb.init(project=project, mode=mode, **kwargs)


def add_cli(ap, default: str = "online") -> None:
    ap.add_argument("--wandb", choices=["online", "offline", "disabled"], default=default, help="W&B logging mode")
    ap.add_argument("--wandb-project", default=DEFAULT_PROJECT)
