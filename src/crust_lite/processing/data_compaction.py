from __future__ import annotations

import json
import math
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

from crust_lite.config import AppConfig
from crust_lite.io.database import (
    connect,
    database_engine,
    materialize_file,
    materialize_rows,
)
from crust_lite.io.parquet import read_table, write_sidecar, write_table
from crust_lite.logging import get_logger
from crust_lite.paths import ProjectPaths

LOGGER = get_logger(__name__)


def _sql_literal(path: Path) -> str:
    return "'" + str(path).replace("'", "''") + "'"


def _table_exists(con: Any, table_name: str, engine: str) -> bool:
    if engine == "duckdb":
        row = con.execute(
            "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = ?", (table_name,)
        ).fetchone()
        return bool(row and row[0])
    row = con.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1", (table_name,)
    ).fetchone()
    return row is not None


def _parse_time(value: Any) -> datetime:
    return datetime.fromisoformat(str(value).replace("Z", "+00:00")).replace(tzinfo=None)


def _bin_date(value: datetime, days: int) -> str:
    origin = date(1970, 1, 1)
    delta_days = (value.date() - origin).days
    start = origin + timedelta(days=(delta_days // days) * days)
    return start.isoformat()


def _floor_bin(value: float, width: float) -> float:
    if width <= 0:
        return value
    return math.floor(value / width) * width


def _bool_value(value: Any) -> bool:
    return str(value).lower() == "true" or value is True


def _input_event_path(paths: ProjectPaths) -> Path:
    event_qc = paths.data_interim / "event_qc.parquet"
    return event_qc if event_qc.exists() else paths.data_processed / "event.parquet"


def compact_data(config: AppConfig, paths: ProjectPaths) -> dict[str, Any]:
    """Build analysis-ready compact tables.

    The event-level compact table keeps only columns needed by downstream
    exploration. The binned summary normalizes heterogeneous historical data to
    common time, space, depth, magnitude, and quality-epoch granularity.
    """
    paths.ensure()
    event_path = _input_event_path(paths)
    if not event_path.exists():
        raise FileNotFoundError(f"Cannot compact data because event table is missing: {event_path}")

    time_bin_days = max(1, int(config.visualization_3d.time_bin_days))
    spatial_bin_km = 10.0
    depth_bin_km = 5.0
    magnitude_bin = 0.5
    if hasattr(config, "preprocessing"):
        prep = config.preprocessing
        time_bin_days = max(1, int(prep.time_bin_days))
        spatial_bin_km = float(prep.spatial_bin_km)
        depth_bin_km = float(prep.depth_bin_km)
        magnitude_bin = float(prep.magnitude_bin)

    engine = database_engine(paths)
    if engine == "duckdb":
        result = _compact_with_duckdb(
            paths,
            event_path,
            time_bin_days=time_bin_days,
            spatial_bin_km=spatial_bin_km,
            depth_bin_km=depth_bin_km,
            magnitude_bin=magnitude_bin,
            region_name=config.region.name,
        )
    else:
        result = _compact_with_python(
            paths,
            event_path,
            time_bin_days=time_bin_days,
            spatial_bin_km=spatial_bin_km,
            depth_bin_km=depth_bin_km,
            magnitude_bin=magnitude_bin,
            region_name=config.region.name,
        )

    report_path = paths.outputs_reports / "data_compaction.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(_render_report(result), encoding="utf-8")
    LOGGER.info(
        "Compacted event data: %s events -> %s bins",
        result.get("event_compact_count"),
        result.get("event_bin_count"),
    )
    return result


def _compact_with_duckdb(
    paths: ProjectPaths,
    event_path: Path,
    *,
    time_bin_days: int,
    spatial_bin_km: float,
    depth_bin_km: float,
    magnitude_bin: float,
    region_name: str,
) -> dict[str, Any]:
    materialize_file(paths, "event_compaction_source", event_path)
    historical_path = paths.data_processed / "historical_data_profile.parquet"
    if historical_path.exists():
        materialize_file(paths, "historical_data_profile", historical_path)
    gnss_path = paths.data_processed / "gnss_features.parquet"
    has_gnss_rows = False
    if gnss_path.exists():
        try:
            has_gnss_rows = bool(read_table(gnss_path))
        except Exception:
            has_gnss_rows = False
        if has_gnss_rows:
            materialize_file(paths, "gnss_features", gnss_path)

    spatial_bin_m = spatial_bin_km * 1000.0
    event_out = paths.data_processed / "event_compact.parquet"
    summary_out = paths.data_processed / "event_bin_summary.parquet"
    gnss_out = paths.data_processed / "gnss_compact.parquet"
    con = connect(paths)
    try:
        engine = database_engine(paths)
        source_cols = {str(row[1]) for row in con.execute("PRAGMA table_info('event_compaction_source')").fetchall()}
        quality_expr = (
            "COALESCE(CAST(e.quality_score AS DOUBLE), 1.0)"
            if "quality_score" in source_cols
            else "1.0"
        )
        sample_expr = (
            "LOWER(CAST(e.is_sample_data AS VARCHAR)) IN ('true', '1')"
            if "is_sample_data" in source_cols
            else "false"
        )
        has_mechanism_expr = (
            "LOWER(CAST(e.has_mechanism AS VARCHAR)) IN ('true', '1')"
            if "has_mechanism" in source_cols
            else "false"
        )
        has_waveform_expr = (
            "LOWER(CAST(e.has_waveform_feature AS VARCHAR)) IN ('true', '1')"
            if "has_waveform_feature" in source_cols
            else "false"
        )
        has_history = _table_exists(con, "historical_data_profile", engine)
        history_join = (
            """
            LEFT JOIN historical_data_profile h
              ON CAST(e.event_id AS VARCHAR) = CAST(h.event_id AS VARCHAR)
            """
            if has_history
            else ""
        )
        history_fields = (
            f"""
            COALESCE(CAST(h.epoch_id AS VARCHAR), 'unprofiled') AS epoch_id,
            COALESCE(CAST(h.analysis_weight AS DOUBLE), {quality_expr}, 1.0) AS analysis_weight,
            COALESCE(CAST(h.catalog_completeness_magnitude AS DOUBLE), NULL) AS catalog_completeness_magnitude,
            """
            if has_history
            else f"""
            'unprofiled' AS epoch_id,
            {quality_expr} AS analysis_weight,
            NULL::DOUBLE AS catalog_completeness_magnitude,
            """
        )
        con.execute(
            f"""
            CREATE OR REPLACE TABLE event_compact AS
            WITH base AS (
              SELECT
                CAST(e.event_id AS VARCHAR) AS event_id,
                CAST(e.time_utc AS VARCHAR) AS time_utc,
                CAST(REPLACE(CAST(e.time_utc AS VARCHAR), 'Z', '+00:00') AS TIMESTAMP) AS event_time,
                CAST(e.lat AS DOUBLE) AS lat,
                CAST(e.lon AS DOUBLE) AS lon,
                CAST(e.x_m AS DOUBLE) AS x_m,
                CAST(e.y_m AS DOUBLE) AS y_m,
                CAST(e.z_m AS DOUBLE) AS z_m,
                CAST(e.depth_km AS DOUBLE) AS depth_km,
                CAST(e.magnitude AS DOUBLE) AS magnitude,
                CAST(e.magnitude_type AS VARCHAR) AS magnitude_type,
                CAST(e.catalog_source AS VARCHAR) AS catalog_source,
                {has_mechanism_expr} AS has_mechanism,
                {has_waveform_expr} AS has_waveform_feature,
                {quality_expr} AS quality_score,
                {sample_expr} AS is_sample_data,
                {history_fields}
              FROM event_compaction_source e
              {history_join}
            )
            SELECT
              event_id,
              time_utc,
              CAST(event_time AS DATE)::VARCHAR AS event_date,
              CAST(
                DATE '1970-01-01'
                + CAST(
                    FLOOR(DATE_DIFF('day', DATE '1970-01-01', CAST(event_time AS DATE)) / {time_bin_days})
                    * {time_bin_days}
                  AS INTEGER)
                AS VARCHAR
              ) AS time_bin_start,
              epoch_id,
              analysis_weight,
              catalog_completeness_magnitude,
              lat,
              lon,
              x_m,
              y_m,
              z_m,
              depth_km,
              magnitude,
              magnitude_type,
              catalog_source,
              has_mechanism,
              has_waveform_feature,
              CAST(FLOOR(x_m / {spatial_bin_m}) AS BIGINT) AS spatial_bin_x,
              CAST(FLOOR(y_m / {spatial_bin_m}) AS BIGINT) AS spatial_bin_y,
              ROUND(FLOOR(depth_km / {depth_bin_km}) * {depth_bin_km}, 3) AS depth_bin_km,
              ROUND(FLOOR(magnitude / {magnitude_bin}) * {magnitude_bin}, 3) AS magnitude_bin,
              quality_score,
              is_sample_data
            FROM base
            """
        )
        con.execute(
            """
            CREATE OR REPLACE TABLE event_bin_summary AS
            SELECT
              time_bin_start,
              epoch_id,
              spatial_bin_x,
              spatial_bin_y,
              depth_bin_km,
              magnitude_bin,
              COUNT(*) AS event_count,
              SUM(analysis_weight) AS weighted_event_count,
              AVG(quality_score) AS mean_quality_score,
              AVG(analysis_weight) AS mean_analysis_weight,
              AVG(x_m) AS centroid_x_m,
              AVG(y_m) AS centroid_y_m,
              AVG(z_m) AS centroid_z_m,
              AVG(lat) AS centroid_lat,
              AVG(lon) AS centroid_lon,
              AVG(depth_km) AS mean_depth_km,
              MAX(magnitude) AS max_magnitude,
              MIN(event_date) AS first_event_date,
              MAX(event_date) AS last_event_date,
              BOOL_OR(is_sample_data) AS is_sample_data
            FROM event_compact
            GROUP BY
              time_bin_start,
              epoch_id,
              spatial_bin_x,
              spatial_bin_y,
              depth_bin_km,
              magnitude_bin
            """
        )
        con.execute(f"COPY event_compact TO {_sql_literal(event_out)} (FORMAT PARQUET)")
        con.execute(f"COPY event_bin_summary TO {_sql_literal(summary_out)} (FORMAT PARQUET)")
        event_count = int(con.execute("SELECT COUNT(*) FROM event_compact").fetchone()[0])
        bin_count = int(con.execute("SELECT COUNT(*) FROM event_bin_summary").fetchone()[0])
        sample_count = int(
            con.execute("SELECT COUNT(*) FROM event_compact WHERE is_sample_data").fetchone()[0]
        )
        gnss_count = 0
        if has_gnss_rows and _table_exists(con, "gnss_features", engine):
            con.execute(
                f"""
                CREATE OR REPLACE TABLE gnss_compact AS
                SELECT
                  CAST(station_id AS VARCHAR) AS station_id,
                  CAST(lat AS DOUBLE) AS lat,
                  CAST(lon AS DOUBLE) AS lon,
                  CAST(x_m AS DOUBLE) AS x_m,
                  CAST(y_m AS DOUBLE) AS y_m,
                  CAST(FLOOR(CAST(x_m AS DOUBLE) / {spatial_bin_m}) AS BIGINT) AS spatial_bin_x,
                  CAST(FLOOR(CAST(y_m AS DOUBLE) / {spatial_bin_m}) AS BIGINT) AS spatial_bin_y,
                  CAST(east_velocity_m_per_yr AS DOUBLE) AS east_velocity_m_per_yr,
                  CAST(north_velocity_m_per_yr AS DOUBLE) AS north_velocity_m_per_yr,
                  CAST(up_velocity_m_per_yr AS DOUBLE) AS up_velocity_m_per_yr,
                  CAST(horizontal_speed_m_per_yr AS DOUBLE) AS horizontal_speed_m_per_yr,
                  CAST(strain_gradient_score AS DOUBLE) AS strain_gradient_score,
                  LOWER(CAST(is_sample_data AS VARCHAR)) IN ('true', '1') AS is_sample_data
                FROM gnss_features
                """
            )
        else:
            con.execute(
                """
                CREATE OR REPLACE TABLE gnss_compact (
                  station_id VARCHAR,
                  lat DOUBLE,
                  lon DOUBLE,
                  x_m DOUBLE,
                  y_m DOUBLE,
                  spatial_bin_x BIGINT,
                  spatial_bin_y BIGINT,
                  east_velocity_m_per_yr DOUBLE,
                  north_velocity_m_per_yr DOUBLE,
                  up_velocity_m_per_yr DOUBLE,
                  horizontal_speed_m_per_yr DOUBLE,
                  strain_gradient_score DOUBLE,
                  is_sample_data BOOLEAN
                )
                """
            )
        con.execute(f"COPY gnss_compact TO {_sql_literal(gnss_out)} (FORMAT PARQUET)")
        gnss_count = int(con.execute("SELECT COUNT(*) FROM gnss_compact").fetchone()[0])
    finally:
        con.close()

    metadata = {
        "region": region_name,
        "compaction_method": "duckdb_columnar_binning",
        "time_bin_days": time_bin_days,
        "spatial_bin_km": spatial_bin_km,
        "depth_bin_km": depth_bin_km,
        "magnitude_bin": magnitude_bin,
        "event_source": str(event_path),
        "event_compact_count": event_count,
        "event_bin_count": bin_count,
        "gnss_compact_count": gnss_count,
        "is_sample_data": sample_count > 0,
        "notes": "Raw granularity is retained outside compact tables; compact tables normalize analysis granularity and avoid full raw scans.",
    }
    for path in [event_out, summary_out, gnss_out]:
        write_sidecar(path, metadata)
    (paths.data_processed / "data_compaction_manifest.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )
    materialize_rows(paths, "data_compaction_manifest", [metadata])
    return metadata


def _compact_with_python(
    paths: ProjectPaths,
    event_path: Path,
    *,
    time_bin_days: int,
    spatial_bin_km: float,
    depth_bin_km: float,
    magnitude_bin: float,
    region_name: str,
) -> dict[str, Any]:
    history_rows = {}
    history_path = paths.data_processed / "historical_data_profile.parquet"
    if history_path.exists():
        history_rows = {str(row.get("event_id")): row for row in read_table(history_path)}
    compact: list[dict[str, Any]] = []
    spatial_bin_m = spatial_bin_km * 1000.0
    for row in read_table(event_path):
        event_time = _parse_time(row["time_utc"])
        hist = history_rows.get(str(row.get("event_id")), {})
        x_m = float(row["x_m"])
        y_m = float(row["y_m"])
        depth = float(row["depth_km"])
        mag = float(row["magnitude"])
        compact.append(
            {
                "event_id": row.get("event_id"),
                "time_utc": row.get("time_utc"),
                "event_date": event_time.date().isoformat(),
                "time_bin_start": _bin_date(event_time, time_bin_days),
                "epoch_id": hist.get("epoch_id", "unprofiled"),
                "analysis_weight": float(hist.get("analysis_weight", row.get("quality_score", 1.0))),
                "catalog_completeness_magnitude": hist.get("catalog_completeness_magnitude"),
                "lat": float(row["lat"]),
                "lon": float(row["lon"]),
                "x_m": x_m,
                "y_m": y_m,
                "z_m": float(row["z_m"]),
                "depth_km": depth,
                "magnitude": mag,
                "magnitude_type": row.get("magnitude_type", ""),
                "catalog_source": row.get("catalog_source", ""),
                "has_mechanism": _bool_value(row.get("has_mechanism")),
                "has_waveform_feature": _bool_value(row.get("has_waveform_feature")),
                "spatial_bin_x": math.floor(x_m / spatial_bin_m),
                "spatial_bin_y": math.floor(y_m / spatial_bin_m),
                "depth_bin_km": _floor_bin(depth, depth_bin_km),
                "magnitude_bin": _floor_bin(mag, magnitude_bin),
                "quality_score": float(row.get("quality_score", 1.0)),
                "is_sample_data": _bool_value(row.get("is_sample_data")),
            }
        )

    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in compact:
        key = (
            row["time_bin_start"],
            row["epoch_id"],
            row["spatial_bin_x"],
            row["spatial_bin_y"],
            row["depth_bin_km"],
            row["magnitude_bin"],
        )
        grouped[key].append(row)
    summary: list[dict[str, Any]] = []
    for key, rows in grouped.items():
        dates = [str(row["event_date"]) for row in rows]
        weights = [float(row["analysis_weight"]) for row in rows]
        summary.append(
            {
                "time_bin_start": key[0],
                "epoch_id": key[1],
                "spatial_bin_x": key[2],
                "spatial_bin_y": key[3],
                "depth_bin_km": key[4],
                "magnitude_bin": key[5],
                "event_count": len(rows),
                "weighted_event_count": sum(weights),
                "mean_quality_score": sum(float(row["quality_score"]) for row in rows) / len(rows),
                "mean_analysis_weight": sum(weights) / len(rows),
                "centroid_x_m": sum(float(row["x_m"]) for row in rows) / len(rows),
                "centroid_y_m": sum(float(row["y_m"]) for row in rows) / len(rows),
                "centroid_z_m": sum(float(row["z_m"]) for row in rows) / len(rows),
                "centroid_lat": sum(float(row["lat"]) for row in rows) / len(rows),
                "centroid_lon": sum(float(row["lon"]) for row in rows) / len(rows),
                "mean_depth_km": sum(float(row["depth_km"]) for row in rows) / len(rows),
                "max_magnitude": max(float(row["magnitude"]) for row in rows),
                "first_event_date": min(dates),
                "last_event_date": max(dates),
                "is_sample_data": any(bool(row["is_sample_data"]) for row in rows),
            }
        )
    event_out = paths.data_processed / "event_compact.parquet"
    summary_out = paths.data_processed / "event_bin_summary.parquet"
    gnss_out = paths.data_processed / "gnss_compact.parquet"
    write_table(compact, event_out)
    write_table(summary, summary_out)
    write_table([], gnss_out, {"is_sample_data": False, "reason": "sqlite_python_fallback"})
    metadata = {
        "region": region_name,
        "compaction_method": "python_fallback_binning",
        "time_bin_days": time_bin_days,
        "spatial_bin_km": spatial_bin_km,
        "depth_bin_km": depth_bin_km,
        "magnitude_bin": magnitude_bin,
        "event_source": str(event_path),
        "event_compact_count": len(compact),
        "event_bin_count": len(summary),
        "gnss_compact_count": 0,
        "is_sample_data": any(bool(row["is_sample_data"]) for row in compact),
        "notes": "Python fallback used; keep raw archive external and use compact tables for repeated analysis.",
    }
    for path in [event_out, summary_out, gnss_out]:
        write_sidecar(path, metadata)
    (paths.data_processed / "data_compaction_manifest.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )
    materialize_rows(paths, "data_compaction_manifest", [metadata])
    return metadata


def _render_report(result: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Data Compaction",
            "",
            "This preprocessing step normalizes heterogeneous input data to compact analysis granularity.",
            "It does not discard the raw archive policy; it creates derived tables for repeated exploration.",
            "",
            f"- Region: `{result.get('region')}`",
            f"- Method: `{result.get('compaction_method')}`",
            f"- Time bin: `{result.get('time_bin_days')}` days",
            f"- Spatial bin: `{result.get('spatial_bin_km')}` km",
            f"- Depth bin: `{result.get('depth_bin_km')}` km",
            f"- Magnitude bin: `{result.get('magnitude_bin')}`",
            f"- Event compact rows: `{result.get('event_compact_count')}`",
            f"- Event summary bins: `{result.get('event_bin_count')}`",
            f"- GNSS compact rows: `{result.get('gnss_compact_count')}`",
            f"- Sample data present: `{str(result.get('is_sample_data')).lower()}`",
            "",
            "Generated files:",
            "",
            "- `data/processed/event_compact.parquet`",
            "- `data/processed/event_bin_summary.parquet`",
            "- `data/processed/gnss_compact.parquet`",
            "- `data/processed/data_compaction_manifest.json`",
            "",
            "Interpretation note: compact bins are for relative state exploration and are not earthquake-date predictions.",
            "",
        ]
    )

