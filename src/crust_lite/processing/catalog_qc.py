from __future__ import annotations

from datetime import datetime
from typing import Any

from crust_lite.config import AppConfig
from crust_lite.geo import LocalProjector, in_bbox
from crust_lite.io.parquet import read_table, write_table
from crust_lite.logging import get_logger
from crust_lite.paths import ProjectPaths

LOGGER = get_logger(__name__)


def _parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def run_catalog_qc(config: AppConfig, paths: ProjectPaths) -> dict[str, Any]:
    rows = read_table(paths.data_processed / "event.parquet")
    mechanisms = read_table(paths.data_processed / "mechanism.parquet") if (
        paths.data_processed / "mechanism.parquet"
    ).exists() else []
    waveform_rows = read_table(paths.data_processed / "waveform_feature.parquet") if (
        paths.data_processed / "waveform_feature.parquet"
    ).exists() else []
    mechanism_ids = {row.get("event_id") for row in mechanisms}
    waveform_ids = {row.get("event_id") for row in waveform_rows}
    projector = LocalProjector(config.region)

    seen: set[tuple[str, str]] = set()
    out: list[dict[str, Any]] = []
    for row in rows:
        try:
            lon = float(row["lon"])
            lat = float(row["lat"])
            depth_km = float(row["depth_km"])
            magnitude = float(row["magnitude"])
            event_time = _parse_time(str(row["time_utc"]))
        except Exception as exc:
            LOGGER.warning("Dropping event with invalid values %s: %s", row.get("event_id"), exc)
            continue
        if not in_bbox(lon, lat, config.region.bbox):
            continue
        if depth_km < 0 or depth_km > config.filters.max_depth_km:
            continue
        if magnitude < config.filters.min_magnitude:
            continue
        if not (config.region.start_date <= event_time.date() <= config.region.end_date):
            continue
        dedupe_key = (event_time.isoformat(timespec="seconds"), f"{lon:.3f}:{lat:.3f}:{depth_km:.1f}")
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        x_m, y_m = projector.lonlat_to_xy(lon, lat)
        completeness = 1.0
        if row.get("magnitude_type") in ("", None):
            completeness -= 0.15
        if depth_km == 0:
            completeness -= 0.15
        out.append(
            {
                **row,
                "time_utc": event_time.isoformat().replace("+00:00", "Z"),
                "lat": lat,
                "lon": lon,
                "depth_km": depth_km,
                "magnitude": magnitude,
                "has_mechanism": row.get("event_id") in mechanism_ids,
                "has_waveform_feature": row.get("event_id") in waveform_ids,
                "x_m": x_m,
                "y_m": y_m,
                "z_m": depth_km * 1000.0,
                "qc_pass": True,
                "quality_score": max(0.0, min(1.0, completeness)),
            }
        )
    out.sort(key=lambda item: str(item["time_utc"]))
    if not out:
        raise ValueError("No events remain after catalog QC")
    is_sample = any(str(row.get("is_sample_data", "")).lower() == "true" for row in out)
    write_table(
        out,
        paths.data_interim / "event_qc.parquet",
        {"is_sample_data": is_sample, "input_count": len(rows), "output_count": len(out)},
    )
    LOGGER.info("Catalog QC retained %d/%d events", len(out), len(rows))
    return {"event_count": len(out), "is_sample_data": is_sample}
