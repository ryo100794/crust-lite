"""crust-lite research prototype."""

from __future__ import annotations

import os
import sys
from pathlib import Path

__version__ = "0.1.0"


def add_local_deps() -> None:
    """Prefer dependencies installed inside the project work tree."""
    root = Path(__file__).resolve().parents[2]
    deps = Path(os.environ.get("CRUST_LITE_DEPS", root / ".deps"))
    if deps.exists():
        deps_str = str(deps)
        if deps_str not in sys.path:
            sys.path.insert(0, deps_str)


add_local_deps()
