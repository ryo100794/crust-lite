from __future__ import annotations

from collections import defaultdict
from datetime import date
from typing import Any

import numpy as np

from crust_lite.geo import clamp01
from crust_lite.io.parquet import read_table, write_table
from crust_lite.logging import get_logger
from crust_lite.paths import ProjectPaths

LOGGER = get_logger(__name__)


def _year_fraction(value: str) -> float:
    d = date.fromisoformat(value)
    start = date(d.year, 1, 1)
    end = date(d.year + 1, 1, 1)
    return d.year + (d - start).days / max(1, (end - start).days)


def _slope(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    coeffs = np.polyfit(np.asarray(xs), np.asarray(ys), deg=1)
    return float(coeffs[0])


def build_gnss_features(paths: ProjectPaths) -> dict[str, Any]:
    daily_path = paths.data_processed / "gnss_daily.parquet"
    if not daily_path.exists():
        write_table([], paths.data_processed / "gnss_features.parquet", {"is_sample_data": False})
        return {"station_count": 0, "is_sample_data": False}
    rows = read_table(daily_path)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["station_id"])].append(row)
    features: list[dict[str, Any]] = []
    speeds: list[float] = []
    for station_id, station_rows in grouped.items():
        station_rows.sort(key=lambda row: str(row["date"]))
        years = [_year_fraction(str(row["date"])) for row in station_rows]
        east = [float(row["east_m"]) for row in station_rows]
        north = [float(row["north_m"]) for row in station_rows]
        up = [float(row["up_m"]) for row in station_rows]
        ve = _slope(years, east)
        vn = _slope(years, north)
        vu = _slope(years, up)
        speed = float((ve * ve + vn * vn) ** 0.5)
        speeds.append(speed)
        last = station_rows[-1]
        features.append(
            {
                "station_id": station_id,
                "lat": float(last["lat"]),
                "lon": float(last["lon"]),
                "x_m": float(last["x_m"]),
                "y_m": float(last["y_m"]),
                "east_velocity_m_per_yr": ve,
                "north_velocity_m_per_yr": vn,
                "up_velocity_m_per_yr": vu,
                "horizontal_speed_m_per_yr": speed,
                "strain_gradient_score": 0.5,
                "is_sample_data": str(last.get("is_sample_data", "")).lower() == "true",
            }
        )
    if speeds:
        lo = min(speeds)
        hi = max(speeds)
        span = max(1e-9, hi - lo)
        for row in features:
            row["strain_gradient_score"] = clamp01((float(row["horizontal_speed_m_per_yr"]) - lo) / span)
    is_sample = any(bool(row["is_sample_data"]) for row in features)
    write_table(
        features,
        paths.data_processed / "gnss_features.parquet",
        {"is_sample_data": is_sample, "station_count": len(features)},
    )
    LOGGER.info("Built GNSS features for %d stations", len(features))
    return {"station_count": len(features), "is_sample_data": is_sample}
