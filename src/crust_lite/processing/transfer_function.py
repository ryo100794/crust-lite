"""Relative site transfer-function estimation from waveform spectra.

The implementation keeps amplitude, phase, and group-delay summaries rather
than reducing waveforms to amplitude-only spectra.  Outputs are CPU-side
features for structure-anomaly screening and array projection; this is not a
full waveform inversion.
"""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any

from crust_lite.config import AppConfig
from crust_lite.geo import LocalProjector, clamp01, distance_to_polyline_km
from crust_lite.io.database import (
    connect,
    copy_table_to_parquet,
    database_engine,
    materialize_file,
    materialize_known_tables,
    write_rows_csv_stream,
)
from crust_lite.io.geopackage import read_features
from crust_lite.io.parquet import read_sidecar, write_sidecar, write_table
from crust_lite.logging import get_logger
from crust_lite.paths import ProjectPaths, resolve_input
from crust_lite.resources import choose_execution_plan

LOGGER = get_logger(__name__)

# Minimum columns required from external waveform collectors. Keeping the
# contract small lets public FDSN, Hi-net, and sample data share one pipeline.
SPECTRA_COLUMNS = {
    "event_id",
    "station_id",
    "frequency_hz",
    "amplitude",
    "phase_rad",
    "group_delay_s",
}




def _count_csv_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return max(0, sum(1 for _line in fh) - 1)


def _resolve_spectra_path(config: AppConfig, paths: ProjectPaths, sample: bool) -> tuple[Path, bool]:
    fallback = paths.data_raw / "sample" / "sample_waveform_spectra.csv"
    if sample or not config.data_sources.waveform_spectra_csv:
        return fallback, True
    return resolve_input(paths.root, config.data_sources.waveform_spectra_csv, fallback), False


def _read_spectra_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        rows = [dict(row) for row in csv.DictReader(fh)]
    if not rows:
        raise ValueError(f"Waveform spectra CSV is empty: {path}")
    missing = SPECTRA_COLUMNS.difference(rows[0])
    if missing:
        raise ValueError(f"Waveform spectra CSV missing columns {sorted(missing)}: {path}")
    return rows


def _wrap_phase(value: float) -> float:
    return (value + math.pi) % (2.0 * math.pi) - math.pi


def _phase_mean(values: list[float]) -> float:
    if not values:
        return 0.0
    sin_sum = sum(math.sin(v) for v in values)
    cos_sum = sum(math.cos(v) for v in values)
    return math.atan2(sin_sum, cos_sum)


def _corr(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2 or len(xs) != len(ys):
        return 0.0
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 0.0 or vy <= 0.0:
        return 0.0
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=False)) / math.sqrt(vx * vy)


def _prepare_spectra(config: AppConfig, paths: ProjectPaths, sample: bool = False) -> tuple[list[dict[str, Any]], bool, str]:
    path, is_sample = _resolve_spectra_path(config, paths, sample)
    rows = _read_spectra_csv(path)
    return rows, is_sample, str(path)


def _fault_distance_km(config: AppConfig, paths: ProjectPaths, lon: float, lat: float) -> tuple[float, str]:
    """Measure station proximity to known or inferred fault traces."""
    projector = LocalProjector(config.region)
    point = projector.lonlat_to_xy(lon, lat)
    best = float("inf")
    best_id = ""
    for layer in (paths.data_processed / "fault_segment.gpkg", paths.data_processed / "inferred_faults.gpkg"):
        if not layer.exists():
            continue
        for feature in read_features(layer):
            props = feature.get("properties", {})
            geom = feature.get("geometry", {})
            if geom.get("local_trace_m"):
                line = [(float(x), float(y)) for x, y in geom["local_trace_m"]]
            elif geom.get("type") == "LineString":
                line = projector.line_lonlat_to_xy(geom.get("coordinates", []))
            else:
                continue
            dist = distance_to_polyline_km(point, line)
            if dist < best:
                best = dist
                best_id = str(props.get("segment_id", ""))
    return best, best_id



def write_empty_transfer_outputs(paths: ProjectPaths, reason: str, is_sample_data: bool = False) -> dict[str, Any]:
    """Write schema-compatible empty outputs when waveform input is unavailable."""
    metadata = {
        "is_sample_data": is_sample_data,
        "method": "not_generated",
        "reason": reason,
        "spectrum_rows": 0,
        "transfer_rows": 0,
        "validation_rows": 0,
    }
    empty_specs = {
        "waveform_spectrum.parquet": [
            "event_id", "station_id", "time_utc", "lat", "lon", "x_m", "y_m",
            "frequency_hz", "amplitude", "log_amplitude", "phase_rad", "group_delay_s",
            "p_residual_s", "s_residual_s", "source", "is_sample_data",
        ],
        "site_transfer_function.parquet": [
            "station_id", "frequency_hz", "transfer_amp_log", "transfer_amp_ratio",
            "transfer_phase_rad", "group_delay_s", "n_events", "lat", "lon", "x_m", "y_m",
            "nearest_fault_km", "nearest_fault_id", "is_sample_data",
        ],
        "transfer_validation.parquet": [
            "heldout_event_id", "n_observations", "amplitude_prediction_corr",
            "phase_median_abs_error_rad", "delay_spread_s", "validity_score", "is_sample_data",
        ],
        "structure_anomaly.parquet": [
            "station_id", "lat", "lon", "x_m", "y_m", "amplitude_anomaly_score",
            "phase_anomaly_score", "delay_anomaly_score", "structure_singularity_score",
            "nearest_fault_km", "nearest_fault_id", "interpretation", "is_sample_data",
        ],
    }
    for filename, columns in empty_specs.items():
        write_rows_csv_stream([], paths.data_processed / filename, columns, metadata)
    return {"skipped": True, "reason": reason, **metadata}

def _quote_sql_path(path: Path) -> str:
    return "'" + str(path).replace("'", "''") + "'"


def _write_station_fault_distance(config: AppConfig, paths: ProjectPaths) -> None:
    con = connect(paths)
    try:
        stations = con.execute(
            """
            SELECT station_id, AVG(CAST(lat AS DOUBLE)) AS lat, AVG(CAST(lon AS DOUBLE)) AS lon
            FROM waveform_spectrum
            GROUP BY station_id
            """
        ).fetchall()
        con.execute("DROP TABLE IF EXISTS station_fault_distance")
        con.execute(
            """
            CREATE TABLE station_fault_distance (
              station_id TEXT, lat DOUBLE, lon DOUBLE, x_m DOUBLE, y_m DOUBLE,
              nearest_fault_km DOUBLE, nearest_fault_id TEXT
            )
            """
        )
        rows = []
        projector = LocalProjector(config.region)
        for station_id, lat, lon in stations:
            lat_f = float(lat or 0.0)
            lon_f = float(lon or 0.0)
            x_m, y_m = projector.lonlat_to_xy(lon_f, lat_f) if lat_f and lon_f else (0.0, 0.0)
            nearest_dist, nearest_fault = _fault_distance_km(config, paths, lon_f, lat_f)
            rows.append((station_id, lat_f, lon_f, x_m, y_m, nearest_dist, nearest_fault))
        if rows:
            con.executemany("INSERT INTO station_fault_distance VALUES (?, ?, ?, ?, ?, ?, ?)", rows)
        con.commit()
    finally:
        con.close()


def _estimate_transfer_functions_db(
    config: AppConfig,
    paths: ProjectPaths,
    source_path: Path,
    is_sample: bool,
    source_count: int,
) -> dict[str, Any]:
    # DuckDB keeps the high-row-count path in SQL so spectra do not have to be
    # expanded into a large Python list on memory-constrained machines.
    if database_engine(paths) != "duckdb":
        raise RuntimeError("Large waveform transfer-function estimation requires source-built DuckDB")
    with source_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        missing = SPECTRA_COLUMNS.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Waveform spectra CSV missing columns {sorted(missing)}: {source_path}")
    con = connect(paths)
    literal = _quote_sql_path(source_path)
    try:
        con.execute(
            "CREATE OR REPLACE TABLE waveform_spectrum AS "
            "SELECT CAST(event_id AS VARCHAR) AS event_id, "
            "CAST(station_id AS VARCHAR) AS station_id, "
            "CAST(COALESCE(time_utc, '') AS VARCHAR) AS time_utc, "
            "CAST(COALESCE(lat, 0) AS DOUBLE) AS lat, "
            "CAST(COALESCE(lon, 0) AS DOUBLE) AS lon, "
            "CAST(frequency_hz AS DOUBLE) AS frequency_hz, "
            "GREATEST(CAST(amplitude AS DOUBLE), 1e-30) AS amplitude, "
            "LN(GREATEST(CAST(amplitude AS DOUBLE), 1e-30)) AS log_amplitude, "
            "ATAN2(SIN(CAST(phase_rad AS DOUBLE)), COS(CAST(phase_rad AS DOUBLE))) AS phase_rad, "
            "CAST(group_delay_s AS DOUBLE) AS group_delay_s, "
            "CAST(COALESCE(p_residual_s, 0) AS DOUBLE) AS p_residual_s, "
            "CAST(COALESCE(s_residual_s, 0) AS DOUBLE) AS s_residual_s, "
            f"CAST({'TRUE' if is_sample else 'FALSE'} AS BOOLEAN) AS is_sample_data "
            f"FROM read_csv_auto({literal}, header=true, sample_size=-1)"
        )
    finally:
        con.close()
    _write_station_fault_distance(config, paths)
    con = connect(paths)
    try:
        con.execute(
            """
            CREATE OR REPLACE TABLE site_transfer_function AS
            WITH global_freq AS (
              SELECT frequency_hz,
                     MEDIAN(log_amplitude) AS global_log_amp,
                     ATAN2(AVG(SIN(phase_rad)), AVG(COS(phase_rad))) AS global_phase,
                     MEDIAN(group_delay_s) AS global_delay
              FROM waveform_spectrum
              GROUP BY frequency_hz
            ), station_tf AS (
              SELECT w.station_id, w.frequency_hz,
                     MEDIAN(w.log_amplitude - g.global_log_amp) AS transfer_amp_log,
                     EXP(MEDIAN(w.log_amplitude - g.global_log_amp)) AS transfer_amp_ratio,
                     ATAN2(AVG(SIN(w.phase_rad - g.global_phase)), AVG(COS(w.phase_rad - g.global_phase))) AS transfer_phase_rad,
                     MEDIAN(w.group_delay_s - g.global_delay) AS group_delay_s,
                     COUNT(*) AS n_events
              FROM waveform_spectrum w
              JOIN global_freq g USING (frequency_hz)
              GROUP BY w.station_id, w.frequency_hz
            )
            SELECT s.station_id, s.frequency_hz, s.transfer_amp_log, s.transfer_amp_ratio,
                   s.transfer_phase_rad, s.group_delay_s, s.n_events,
                   d.lat, d.lon, d.x_m, d.y_m, d.nearest_fault_km, d.nearest_fault_id,
                   CAST(? AS BOOLEAN) AS is_sample_data
            FROM station_tf s
            LEFT JOIN station_fault_distance d USING (station_id)
            """,
            (is_sample,),
        )
        con.execute(
            """
            CREATE OR REPLACE TABLE structure_anomaly AS
            WITH maxima AS (
              SELECT GREATEST(MAX(ABS(transfer_amp_log)), 1e-12) AS max_amp,
                     GREATEST(MAX(ABS(transfer_phase_rad)), 1e-12) AS max_phase,
                     GREATEST(MAX(ABS(group_delay_s)), 1e-12) AS max_delay
              FROM site_transfer_function
            ), per_station AS (
              SELECT station_id, AVG(lat) AS lat, AVG(lon) AS lon, AVG(x_m) AS x_m, AVG(y_m) AS y_m,
                     AVG(ABS(transfer_amp_log) / max_amp) AS amplitude_anomaly_score,
                     AVG(ABS(transfer_phase_rad) / max_phase) AS phase_anomaly_score,
                     AVG(ABS(group_delay_s) / max_delay) AS delay_anomaly_score,
                     AVG(nearest_fault_km) AS nearest_fault_km,
                     ANY_VALUE(nearest_fault_id) AS nearest_fault_id
              FROM site_transfer_function, maxima
              GROUP BY station_id
            )
            SELECT *,
                   LEAST(1.0, GREATEST(0.0, 0.4 * amplitude_anomaly_score + 0.3 * phase_anomaly_score + 0.3 * delay_anomaly_score)) AS structure_singularity_score,
                   'relative transfer-function anomaly; not a direct subsurface inversion' AS interpretation,
                   CAST(? AS BOOLEAN) AS is_sample_data
            FROM per_station
            """,
            (is_sample,),
        )
        con.execute(
            """
            CREATE OR REPLACE TABLE transfer_validation AS
            WITH global_freq AS (
              SELECT frequency_hz, MEDIAN(log_amplitude) AS global_log_amp,
                     ATAN2(AVG(SIN(phase_rad)), AVG(COS(phase_rad))) AS global_phase
              FROM waveform_spectrum
              GROUP BY frequency_hz
            ), pred AS (
              SELECT w.event_id, w.log_amplitude AS observed_amp,
                     g.global_log_amp + tf.transfer_amp_log AS predicted_amp,
                     ABS(ATAN2(SIN(w.phase_rad - (g.global_phase + tf.transfer_phase_rad)),
                               COS(w.phase_rad - (g.global_phase + tf.transfer_phase_rad)))) AS phase_error
              FROM waveform_spectrum w
              JOIN global_freq g USING (frequency_hz)
              JOIN site_transfer_function tf USING (station_id, frequency_hz)
            ), stats AS (
              SELECT event_id, COUNT(*) AS n_observations,
                     SUM(observed_amp) AS sx, SUM(predicted_amp) AS sy,
                     SUM(observed_amp * observed_amp) AS sx2, SUM(predicted_amp * predicted_amp) AS sy2,
                     SUM(observed_amp * predicted_amp) AS sxy, AVG(phase_error) AS phase_median_abs_error_rad
              FROM pred
              GROUP BY event_id
            )
            , scored AS (
              SELECT event_id AS heldout_event_id, n_observations,
                     CASE WHEN (n_observations * sx2 - sx * sx) <= 0 OR (n_observations * sy2 - sy * sy) <= 0
                          THEN 0.0
                          ELSE (n_observations * sxy - sx * sy) / SQRT((n_observations * sx2 - sx * sx) * (n_observations * sy2 - sy * sy))
                     END AS amplitude_prediction_corr,
                     phase_median_abs_error_rad,
                     0.0 AS delay_spread_s
              FROM stats
            )
            SELECT *,
                   LEAST(1.0, GREATEST(0.0, 0.55 * GREATEST(0.0, amplitude_prediction_corr) + 0.30 * (1.0 - LEAST(phase_median_abs_error_rad, PI()) / PI()) + 0.15)) AS validity_score,
                   CAST(? AS BOOLEAN) AS is_sample_data
            FROM scored
            """,
            (is_sample,),
        )
    finally:
        con.close()
    exports: list[tuple[str, str, dict[str, Any]]] = [
        ("waveform_spectrum", "waveform_spectrum.parquet", {"is_sample_data": is_sample, "source_path": str(source_path), "representation": "complex_spectrum_with_group_delay", "execution": "duckdb_sql"}),
        ("site_transfer_function", "site_transfer_function.parquet", {"is_sample_data": is_sample, "method": "robust_relative_complex_spectral_ratio_db", "spectrum_rows": source_count}),
        ("transfer_validation", "transfer_validation.parquet", {"is_sample_data": is_sample, "validation": "all_event_correlation_db"}),
        ("structure_anomaly", "structure_anomaly.parquet", {"is_sample_data": is_sample, "uses_phase_and_delay": True}),
    ]
    for table, filename, metadata in exports:
        copy_table_to_parquet(paths, table, paths.data_processed / filename, metadata)
    materialize_known_tables(paths)
    con = connect(paths)
    try:
        transfer_rows = con.execute("SELECT COUNT(*) FROM site_transfer_function").fetchone()[0]
        validation_rows = con.execute("SELECT COUNT(*) FROM transfer_validation").fetchone()[0]
        station_count = con.execute("SELECT COUNT(DISTINCT station_id) FROM waveform_spectrum").fetchone()[0]
    finally:
        con.close()
    summary = {
        "is_sample_data": is_sample,
        "spectrum_rows": source_count,
        "transfer_rows": int(transfer_rows),
        "validation_rows": int(validation_rows),
        "station_count": int(station_count),
        "source_path": str(source_path),
        "method": "database relative complex transfer function using amplitude, phase, group delay, and event correlations",
        "execution_engine": "duckdb",
    }
    write_sidecar(paths.data_processed / "site_transfer_function.parquet", {**summary, "physical_format": "parquet_duckdb"})
    LOGGER.info("Estimated DB transfer functions for %d stations using %d spectra rows", station_count, source_count)
    return summary


def estimate_transfer_functions(config: AppConfig, paths: ProjectPaths, sample: bool = False) -> dict[str, Any]:
    """Estimate transfer functions and validation metrics for one config."""
    spectra_path, is_sample_path = _resolve_spectra_path(config, paths, sample)
    source_count = _count_csv_rows(spectra_path)
    plan = choose_execution_plan(
        config,
        operation="transfer_function",
        engine=database_engine(paths),
        estimated_rows=source_count,
        estimated_row_bytes=256,
    )
    LOGGER.info("Transfer-function execution plan: %s", plan.as_metadata())
    if not plan.use_in_memory and database_engine(paths) == "duckdb":
        return _estimate_transfer_functions_db(config, paths, spectra_path, is_sample_path, source_count)
    rows, is_sample, source_path = _prepare_spectra(config, paths, sample=sample)
    spectra_rows: list[dict[str, Any]] = []
    projector = LocalProjector(config.region)
    for row in rows:
        lat = float(row.get("lat", 0.0) or 0.0)
        lon = float(row.get("lon", 0.0) or 0.0)
        x_m, y_m = projector.lonlat_to_xy(lon, lat) if lat and lon else (0.0, 0.0)
        amp = max(float(row["amplitude"]), 1e-30)
        spectra_rows.append(
            {
                "event_id": row["event_id"],
                "station_id": row["station_id"],
                "time_utc": row.get("time_utc", ""),
                "lat": lat,
                "lon": lon,
                "x_m": x_m,
                "y_m": y_m,
                "frequency_hz": float(row["frequency_hz"]),
                "amplitude": amp,
                "log_amplitude": math.log(amp),
                "phase_rad": _wrap_phase(float(row["phase_rad"])),
                "group_delay_s": float(row.get("group_delay_s", 0.0) or 0.0),
                "p_residual_s": float(row.get("p_residual_s", 0.0) or 0.0),
                "s_residual_s": float(row.get("s_residual_s", 0.0) or 0.0),
                "source": row.get("source", source_path),
                "is_sample_data": is_sample,
            }
        )
    write_table(
        spectra_rows,
        paths.data_processed / "waveform_spectrum.parquet",
        {"is_sample_data": is_sample, "source_path": source_path, "representation": "complex_spectrum_with_group_delay"},
    )
    materialize_file(paths, "waveform_spectrum", paths.data_processed / "waveform_spectrum.parquet")

    by_freq: dict[float, list[dict[str, Any]]] = defaultdict(list)
    for row in spectra_rows:
        by_freq[float(row["frequency_hz"])].append(row)
    global_log_by_freq = {freq: median([float(row["log_amplitude"]) for row in items]) for freq, items in by_freq.items()}
    global_phase_by_freq = {freq: _phase_mean([float(row["phase_rad"]) for row in items]) for freq, items in by_freq.items()}
    global_delay_by_freq = {freq: median([float(row["group_delay_s"]) for row in items]) for freq, items in by_freq.items()}

    station_freq: dict[tuple[str, float], list[dict[str, Any]]] = defaultdict(list)
    for row in spectra_rows:
        station_freq[(str(row["station_id"]), float(row["frequency_hz"]))].append(row)

    tf_rows: list[dict[str, Any]] = []
    station_locs: dict[str, tuple[float, float, float, float]] = {}
    for row in spectra_rows:
        station_locs[str(row["station_id"])] = (float(row["lat"]), float(row["lon"]), float(row["x_m"]), float(row["y_m"]))
    for (station, freq), items in sorted(station_freq.items()):
        log_ratio_values = [float(item["log_amplitude"]) - global_log_by_freq[freq] for item in items]
        phase_values = [_wrap_phase(float(item["phase_rad"]) - global_phase_by_freq[freq]) for item in items]
        delay_values = [float(item["group_delay_s"]) - global_delay_by_freq[freq] for item in items]
        amp_log = median(log_ratio_values)
        phase = _phase_mean(phase_values)
        delay = median(delay_values)
        lat, lon, x_m, y_m = station_locs[station]
        nearest_dist, nearest_fault = _fault_distance_km(config, paths, lon, lat)
        tf_rows.append(
            {
                "station_id": station,
                "frequency_hz": freq,
                "transfer_amp_log": amp_log,
                "transfer_amp_ratio": math.exp(amp_log),
                "transfer_phase_rad": phase,
                "group_delay_s": delay,
                "n_events": len(items),
                "lat": lat,
                "lon": lon,
                "x_m": x_m,
                "y_m": y_m,
                "nearest_fault_km": nearest_dist,
                "nearest_fault_id": nearest_fault,
                "is_sample_data": is_sample,
            }
        )

    amp_abs = [abs(float(row["transfer_amp_log"])) for row in tf_rows]
    phase_abs = [abs(float(row["transfer_phase_rad"])) for row in tf_rows]
    delay_abs = [abs(float(row["group_delay_s"])) for row in tf_rows]
    max_amp = max(amp_abs or [1.0])
    max_phase = max(phase_abs or [1.0])
    max_delay = max(delay_abs or [1.0])
    anomaly_rows: list[dict[str, Any]] = []
    for station in sorted({str(row["station_id"]) for row in tf_rows}):
        items = [row for row in tf_rows if row["station_id"] == station]
        amp_score = sum(abs(float(row["transfer_amp_log"])) / max_amp for row in items) / len(items)
        phase_score = sum(abs(float(row["transfer_phase_rad"])) / max_phase for row in items) / len(items)
        delay_score = sum(abs(float(row["group_delay_s"])) / max_delay for row in items) / len(items)
        lat, lon, x_m, y_m = station_locs[station]
        nearest_dist, nearest_fault = _fault_distance_km(config, paths, lon, lat)
        anomaly_rows.append(
            {
                "station_id": station,
                "lat": lat,
                "lon": lon,
                "x_m": x_m,
                "y_m": y_m,
                "amplitude_anomaly_score": clamp01(amp_score),
                "phase_anomaly_score": clamp01(phase_score),
                "delay_anomaly_score": clamp01(delay_score),
                "structure_singularity_score": clamp01(0.4 * amp_score + 0.3 * phase_score + 0.3 * delay_score),
                "nearest_fault_km": nearest_dist,
                "nearest_fault_id": nearest_fault,
                "interpretation": "relative transfer-function anomaly; not a direct subsurface inversion",
                "is_sample_data": is_sample,
            }
        )

    validation_rows: list[dict[str, Any]] = []
    events = sorted({str(row["event_id"]) for row in spectra_rows})
    for event_id in events:
        train_rows = [row for row in spectra_rows if row["event_id"] != event_id]
        test_rows = [row for row in spectra_rows if row["event_id"] == event_id]
        if not train_rows or not test_rows:
            continue
        train_by_freq: dict[float, list[dict[str, Any]]] = defaultdict(list)
        for row in train_rows:
            train_by_freq[float(row["frequency_hz"])].append(row)
        train_global_log = {freq: median([float(row["log_amplitude"]) for row in items]) for freq, items in train_by_freq.items()}
        train_global_phase = {freq: _phase_mean([float(row["phase_rad"]) for row in items]) for freq, items in train_by_freq.items()}
        train_station_freq: dict[tuple[str, float], list[dict[str, Any]]] = defaultdict(list)
        for row in train_rows:
            train_station_freq[(str(row["station_id"]), float(row["frequency_hz"]))].append(row)
        tf_amp = {
            key: median([float(row["log_amplitude"]) - train_global_log[key[1]] for row in items])
            for key, items in train_station_freq.items()
        }
        tf_phase = {
            key: _phase_mean([_wrap_phase(float(row["phase_rad"]) - train_global_phase[key[1]]) for row in items])
            for key, items in train_station_freq.items()
        }
        observed_amp: list[float] = []
        predicted_amp: list[float] = []
        observed_phase: list[float] = []
        predicted_phase: list[float] = []
        delays: list[float] = []
        for row in test_rows:
            key = (str(row["station_id"]), float(row["frequency_hz"]))
            if key not in tf_amp or key[1] not in train_global_log:
                continue
            observed_amp.append(float(row["log_amplitude"]))
            predicted_amp.append(train_global_log[key[1]] + tf_amp[key])
            observed_phase.append(float(row["phase_rad"]))
            predicted_phase.append(_wrap_phase(train_global_phase[key[1]] + tf_phase[key]))
            delays.append(float(row["group_delay_s"]))
        amp_corr = _corr(observed_amp, predicted_amp)
        phase_error = median([abs(_wrap_phase(o - p)) for o, p in zip(observed_phase, predicted_phase, strict=False)] or [math.pi])
        validation_rows.append(
            {
                "heldout_event_id": event_id,
                "n_observations": len(observed_amp),
                "amplitude_prediction_corr": amp_corr,
                "phase_median_abs_error_rad": phase_error,
                "delay_spread_s": (max(delays) - min(delays)) if delays else 0.0,
                "validity_score": clamp01(0.55 * max(0.0, amp_corr) + 0.30 * (1.0 - min(phase_error, math.pi) / math.pi) + 0.15),
                "is_sample_data": is_sample,
            }
        )

    write_table(tf_rows, paths.data_processed / "site_transfer_function.parquet", {"is_sample_data": is_sample, "method": "robust_relative_complex_spectral_ratio"})
    write_table(validation_rows, paths.data_processed / "transfer_validation.parquet", {"is_sample_data": is_sample, "validation": "leave_one_event_out"})
    write_table(anomaly_rows, paths.data_processed / "structure_anomaly.parquet", {"is_sample_data": is_sample, "uses_phase_and_delay": True})
    materialize_known_tables(paths)
    con = connect(paths)
    try:
        for table, path in {
            "site_transfer_function": paths.data_processed / "site_transfer_function.parquet",
            "transfer_validation": paths.data_processed / "transfer_validation.parquet",
            "structure_anomaly": paths.data_processed / "structure_anomaly.parquet",
        }.items():
            materialize_file(paths, table, path)
    finally:
        con.close()
    summary = {
        "is_sample_data": is_sample,
        "spectrum_rows": len(spectra_rows),
        "transfer_rows": len(tf_rows),
        "validation_rows": len(validation_rows),
        "station_count": len(station_locs),
        "source_path": source_path,
        "method": "relative complex transfer function using amplitude, phase, group delay, and arrival residual proxies",
    }
    tf_path = paths.data_processed / "site_transfer_function.parquet"
    write_sidecar(tf_path, {**read_sidecar(tf_path), **summary})
    LOGGER.info("Estimated transfer functions for %d stations using %d spectra rows", len(station_locs), len(spectra_rows))
    return summary
