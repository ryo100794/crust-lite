from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from crust_lite.config import AppConfig
from crust_lite.geo import LocalProjector
from crust_lite.io.parquet import write_table
from crust_lite.logging import get_logger
from crust_lite.paths import ProjectPaths, resolve_input

LOGGER = get_logger(__name__)


EVENT_COLUMNS = {
    "event_id",
    "time_utc",
    "lat",
    "lon",
    "depth_km",
    "magnitude",
    "magnitude_type",
    "catalog_source",
}


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        rows = [dict(row) for row in csv.DictReader(fh)]
    if not rows:
        raise ValueError(f"Event CSV is empty: {path}")
    missing = EVENT_COLUMNS.difference(rows[0])
    if missing:
        raise ValueError(f"Event CSV missing columns {sorted(missing)}: {path}")
    return rows


def _fetch_fdsn(config: AppConfig) -> list[dict[str, Any]]:
    try:
        from obspy import UTCDateTime  # type: ignore
        from obspy.clients.fdsn import Client  # type: ignore
    except Exception as exc:
        raise RuntimeError(f"ObsPy FDSN client is unavailable: {exc}") from exc

    min_lon, min_lat, max_lon, max_lat = config.region.bbox
    client = Client(config.data_sources.fdsn_client)
    catalog = client.get_events(
        starttime=UTCDateTime(str(config.region.start_date)),
        endtime=UTCDateTime(str(config.region.end_date)),
        minlongitude=min_lon,
        maxlongitude=max_lon,
        minlatitude=min_lat,
        maxlatitude=max_lat,
        minmagnitude=config.filters.min_magnitude,
        maxdepth=config.filters.max_depth_km,
    )
    rows: list[dict[str, Any]] = []
    for idx, event in enumerate(catalog):
        origin = event.preferred_origin() or event.origins[0]
        magnitude = event.preferred_magnitude() or event.magnitudes[0]
        rows.append(
            {
                "event_id": str(event.resource_id or f"fdsn_{idx:06d}"),
                "time_utc": origin.time.isoformat().replace("+00:00", "Z"),
                "lat": float(origin.latitude),
                "lon": float(origin.longitude),
                "depth_km": float(origin.depth or 0.0) / 1000.0,
                "magnitude": float(magnitude.mag),
                "magnitude_type": str(magnitude.magnitude_type or ""),
                "catalog_source": config.data_sources.catalog_source,
            }
        )
    return rows


def fetch_events(config: AppConfig, paths: ProjectPaths, sample: bool = False) -> dict[str, Any]:
    paths.ensure()
    sample_path = paths.data_raw / "sample" / "sample_events.csv"
    is_sample = sample
    source_note = "sample_requested" if sample else ""
    if sample:
        rows = _read_csv(sample_path)
        is_sample = True
        source_note = source_note or "sample_requested"
    elif config.data_sources.event_csv:
        event_path = resolve_input(paths.root, config.data_sources.event_csv, sample_path)
        rows = _read_csv(event_path)
        is_sample = False
        source_note = f"local_event_csv:{event_path}"
        LOGGER.info("Loaded %d events from local CSV %s", len(rows), event_path)
    elif config.data_sources.use_fdsn:
        try:
            rows = _fetch_fdsn(config)
            LOGGER.info("Fetched %d events from %s", len(rows), config.data_sources.fdsn_client)
        except Exception as exc:
            LOGGER.warning("FDSN fetch failed; falling back to sample events: %s", exc)
            rows = _read_csv(sample_path)
            is_sample = True
            source_note = f"fdsn_failed: {type(exc).__name__}: {exc}"
    else:
        rows = _read_csv(sample_path)
        is_sample = True
        source_note = source_note or "sample_or_fdsn_disabled"

    projector = LocalProjector(config.region)
    out_rows: list[dict[str, Any]] = []
    for row in rows:
        lon = float(row["lon"])
        lat = float(row["lat"])
        depth_km = float(row["depth_km"])
        x_m, y_m = projector.lonlat_to_xy(lon, lat)
        out_rows.append(
            {
                "event_id": row["event_id"],
                "time_utc": row["time_utc"],
                "lat": lat,
                "lon": lon,
                "depth_km": depth_km,
                "magnitude": float(row["magnitude"]),
                "magnitude_type": row.get("magnitude_type", ""),
                "catalog_source": row.get("catalog_source", config.data_sources.catalog_source),
                "has_mechanism": False,
                "has_waveform_feature": False,
                "x_m": x_m,
                "y_m": y_m,
                "z_m": depth_km * 1000.0,
                "is_sample_data": is_sample,
            }
        )
    write_table(
        out_rows,
        paths.data_processed / "event.parquet",
        {
            "is_sample_data": is_sample,
            "catalog_source": config.data_sources.catalog_source,
            "source_note": source_note,
        },
    )
    return {"is_sample_data": is_sample, "event_count": len(out_rows), "source_note": source_note}
