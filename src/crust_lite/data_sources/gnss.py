from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from crust_lite.config import AppConfig
from crust_lite.geo import LocalProjector
from crust_lite.io.parquet import write_table
from crust_lite.paths import ProjectPaths, resolve_input

REQUIRED_COLUMNS = {
    "station_id",
    "date",
    "lat",
    "lon",
    "east_m",
    "north_m",
    "up_m",
    "sigma_e",
    "sigma_n",
    "sigma_u",
}


def read_gnss_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        rows = [dict(row) for row in csv.DictReader(fh)]
    if not rows:
        return []
    missing = REQUIRED_COLUMNS.difference(rows[0])
    if missing:
        raise ValueError(f"GNSS CSV missing columns {sorted(missing)}: {path}")
    numeric = ["lat", "lon", "east_m", "north_m", "up_m", "sigma_e", "sigma_n", "sigma_u"]
    for row in rows:
        for key in numeric:
            row[key] = float(row[key])
    return rows


def fetch_gnss(config: AppConfig, paths: ProjectPaths, sample: bool = False) -> dict[str, Any]:
    paths.ensure()
    if not config.data_sources.use_gnss and not sample:
        write_table([], paths.data_processed / "gnss_daily.parquet", {"is_sample_data": False})
        return {"is_sample_data": False, "gnss_rows": 0}
    fallback = paths.data_raw / "sample" / "sample_gnss_daily.csv"
    source_path = resolve_input(paths.root, config.data_sources.gnss_csv, fallback)
    if sample:
        source_path = fallback
    rows = read_gnss_csv(source_path)
    projector = LocalProjector(config.region)
    out_rows: list[dict[str, Any]] = []
    for row in rows:
        x_m, y_m = projector.lonlat_to_xy(float(row["lon"]), float(row["lat"]))
        out_rows.append({**row, "x_m": x_m, "y_m": y_m, "is_sample_data": source_path == fallback})
    write_table(
        out_rows,
        paths.data_processed / "gnss_daily.parquet",
        {"is_sample_data": source_path == fallback, "source_path": str(source_path)},
    )
    return {"is_sample_data": source_path == fallback, "gnss_rows": len(out_rows)}
