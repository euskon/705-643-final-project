"""Tiny helpers shared by the ``scripts/`` entry points."""

from __future__ import annotations

import logging
import sys


def setup_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname).1s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )
    for noisy in ("urllib3", "matplotlib", "PIL", "fiona", "pyproj"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    # OpenPoints prints model configs through the *root* logger at INFO
    for h in logging.getLogger().handlers:
        h.addFilter(lambda r: not (r.name == "root" and r.levelno < logging.WARNING))


def parse_list(s: str | None) -> list[str] | None:
    """``"a,b , c"`` -> ``["a", "b", "c"]``; None/empty -> None (meaning "all")."""
    if not s:
        return None
    return [x.strip() for x in s.split(",") if x.strip()]
