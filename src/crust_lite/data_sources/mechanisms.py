from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Protocol

from crust_lite.config import AppConfig
from crust_lite.io.parquet import write_table
from crust_lite.logging import get_logger
from crust_lite.paths import ProjectPaths, resolve_input

LOGGER = get_logger(__name__)


class MechanismProvider(Protocol):
    def fetch(self, config: AppConfig, paths: ProjectPaths) -> list[dict[str, Any]]:
        """Return mechanism rows."""


REQUIRED_COLUMNS = {
    "mechanism_id",
    "event_id",
    "strike1",
    "dip1",
    "rake1",
    "strike2",
    "dip2",
    "rake2",
    "scalar_moment_nm",
    "source",
}


def read_mechanism_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        rows = [dict(row) for row in csv.DictReader(fh)]
    if not rows:
        return []
    missing = REQUIRED_COLUMNS.difference(rows[0])
    if missing:
        raise ValueError(f"Mechanism CSV missing columns {sorted(missing)}: {path}")
    numeric = [
        "strike1",
        "dip1",
        "rake1",
        "strike2",
        "dip2",
        "rake2",
        "scalar_moment_nm",
    ]
    for row in rows:
        for key in numeric:
            row[key] = float(row[key])
    return rows


def fetch_mechanisms(config: AppConfig, paths: ProjectPaths, sample: bool = False) -> dict[str, Any]:
    paths.ensure()
    fallback = paths.data_raw / "sample" / "sample_mechanisms.csv"
    if sample:
        source_path = fallback
        is_sample = True
        rows = read_mechanism_csv(source_path)
    elif config.data_sources.mechanism_csv:
        source_path = resolve_input(paths.root, config.data_sources.mechanism_csv, fallback)
        is_sample = source_path == fallback
        rows = read_mechanism_csv(source_path)
    else:
        source_path = fallback
        rows = []
        is_sample = False
        LOGGER.info("No mechanism_csv configured; writing empty mechanism table")
    write_table(
        rows,
        paths.data_processed / "mechanism.parquet",
        {"is_sample_data": is_sample, "source_path": str(source_path)},
    )
    return {"is_sample_data": is_sample, "mechanism_count": len(rows), "source_path": str(source_path)}
