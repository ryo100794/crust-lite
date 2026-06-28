from __future__ import annotations

from datetime import datetime
from typing import Any

from crust_lite.config import AppConfig
from crust_lite.io.database import connect, materialize_file, materialize_known_tables
from crust_lite.io.parquet import read_table, write_table
from crust_lite.logging import get_logger
from crust_lite.paths import ProjectPaths

LOGGER = get_logger(__name__)

DEFAULT_EPOCHS = [
    {
        "epoch_id": "pre_instrumental_to_1922",
        "start_date": "0001-01-01",
        "end_date": "1922-12-31",
        "typical_location_uncertainty_km": 50.0,
        "typical_depth_uncertainty_km": 30.0,
        "typical_magnitude_uncertainty": 0.5,
        "time_uncertainty_s": 86400.0,
        "catalog_completeness_magnitude": 6.5,
        "analysis_weight": 0.15,
        "notes": "Historical and macroseismic era; useful for long-term spatial memory but not dense-rate inference.",
    },
    {
        "epoch_id": "early_instrumental_1923_1994",
        "start_date": "1923-01-01",
        "end_date": "1994-12-31",
        "typical_location_uncertainty_km": 15.0,
        "typical_depth_uncertainty_km": 20.0,
        "typical_magnitude_uncertainty": 0.3,
        "time_uncertainty_s": 3600.0,
        "catalog_completeness_magnitude": 5.0,
        "analysis_weight": 0.40,
        "notes": "Instrumental but sparse/heterogeneous network; suitable for regional long-term rates with uncertainty.",
    },
    {
        "epoch_id": "modern_dense_1995_2010",
        "start_date": "1995-01-01",
        "end_date": "2010-12-31",
        "typical_location_uncertainty_km": 5.0,
        "typical_depth_uncertainty_km": 8.0,
        "typical_magnitude_uncertainty": 0.15,
        "time_uncertainty_s": 60.0,
        "catalog_completeness_magnitude": 3.0,
        "analysis_weight": 0.75,
        "notes": "Dense modern instrumental era; reasonable for seismicity geometry and rate proxies.",
    },
    {
        "epoch_id": "recent_high_density_2011_present",
        "start_date": "2011-01-01",
        "end_date": "9999-12-31",
        "typical_location_uncertainty_km": 3.0,
        "typical_depth_uncertainty_km": 5.0,
        "typical_magnitude_uncertainty": 0.1,
        "time_uncertainty_s": 10.0,
        "catalog_completeness_magnitude": 2.0,
        "analysis_weight": 0.90,
        "notes": "High-density modern catalog; aftershock-rich periods still need declustering for rate studies.",
    },
]


def _parse_time(value: Any) -> datetime:
    return datetime.fromisoformat(str(value).replace("Z", "+00:00")).replace(tzinfo=None)


def _epoch_for_time(value: Any) -> dict[str, Any]:
    dt = _parse_time(value).date().isoformat()
    for epoch in DEFAULT_EPOCHS:
        start = str(epoch["start_date"])
        end = str(epoch["end_date"])
        if start <= dt <= end:
            return epoch
    return DEFAULT_EPOCHS[-1]


def build_historical_quality(config: AppConfig, paths: ProjectPaths) -> dict[str, Any]:
    event_path = paths.data_interim / "event_qc.parquet"
    if not event_path.exists():
        return {"skipped": True, "reason": "event_qc_not_found"}
    events = read_table(event_path)
    rows: list[dict[str, Any]] = []
    for row in events:
        epoch = _epoch_for_time(row["time_utc"])
        mag = float(row.get("magnitude", 0.0) or 0.0)
        completeness = float(epoch["catalog_completeness_magnitude"])
        completeness_weight = min(1.0, max(0.0, (mag - completeness + 1.0)))
        analysis_weight = float(epoch["analysis_weight"]) * completeness_weight
        rows.append(
            {
                "event_id": row.get("event_id"),
                "time_utc": row.get("time_utc"),
                "epoch_id": epoch["epoch_id"],
                "typical_location_uncertainty_km": epoch["typical_location_uncertainty_km"],
                "typical_depth_uncertainty_km": epoch["typical_depth_uncertainty_km"],
                "typical_magnitude_uncertainty": epoch["typical_magnitude_uncertainty"],
                "time_uncertainty_s": epoch["time_uncertainty_s"],
                "catalog_completeness_magnitude": completeness,
                "analysis_weight": analysis_weight,
                "is_above_epoch_completeness": mag >= completeness,
                "quality_notes": epoch["notes"],
                "is_sample_data": str(row.get("is_sample_data", "")).lower() == "true",
            }
        )
    epoch_rows = [{**epoch} for epoch in DEFAULT_EPOCHS]
    write_table(
        epoch_rows,
        paths.data_processed / "data_quality_epoch.parquet",
        {"description": "Default long-term historical data quality epochs for Japan-wide analysis"},
    )
    write_table(
        rows,
        paths.data_processed / "historical_data_profile.parquet",
        {
            "description": "Per-event quality profile used to retain historical data with explicit uncertainty/weighting",
            "region": config.region.name,
        },
    )
    materialize_known_tables(paths)
    materialize_file(paths, "data_quality_epoch", paths.data_processed / "data_quality_epoch.parquet")
    materialize_file(paths, "historical_data_profile", paths.data_processed / "historical_data_profile.parquet")
    con = connect(paths)
    try:
        summary_rows = con.execute(
            """
            SELECT epoch_id, COUNT(*) AS event_count, AVG(analysis_weight) AS mean_analysis_weight,
                   SUM(CASE WHEN is_above_epoch_completeness THEN 1 ELSE 0 END) AS complete_event_count
            FROM historical_data_profile
            GROUP BY epoch_id
            ORDER BY MIN(time_utc)
            """
        ).fetchall()
    finally:
        con.close()
    LOGGER.info("Built historical quality profile for %d events", len(rows))
    return {
        "event_count": len(rows),
        "epoch_count": len(epoch_rows),
        "summary": [tuple(row) for row in summary_rows],
    }
