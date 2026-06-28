from __future__ import annotations

import csv
import json
import sqlite3
import sys
from collections.abc import Iterable
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from crust_lite.io.parquet import read_sidecar, read_table, write_sidecar
from crust_lite.paths import ProjectPaths


def duckdb_database_path(paths: ProjectPaths) -> Path:
    return paths.data_processed / "crust_lite.duckdb"


def sqlite_database_path(paths: ProjectPaths) -> Path:
    return paths.data_processed / "crust_lite.sqlite"


def database_path(paths: ProjectPaths) -> Path:
    return duckdb_database_path(paths) if duckdb_available(paths) else sqlite_database_path(paths)


def _source_duckdb_dir(paths: ProjectPaths) -> Path:
    return paths.root / ".deps-duckdb-src"


def _activate_source_duckdb(paths: ProjectPaths) -> bool:
    source = _source_duckdb_dir(paths)
    if not (source / "duckdb").exists() or not any(source.glob("_duckdb*.so")):
        return False
    source_str = str(source)
    if sys.path[:1] != [source_str]:
        sys.path = [entry for entry in sys.path if entry != source_str]
        sys.path.insert(0, source_str)
    return True


def duckdb_available(paths: ProjectPaths | None = None) -> bool:
    # Avoid importing the binary wheel in .deps: it aborts this Python runtime.
    # We only try DuckDB when the source-built isolated directory is present.
    if paths is None:
        return any(Path(entry).name == ".deps-duckdb-src" for entry in sys.path)
    if not _activate_source_duckdb(paths):
        return False
    try:
        import duckdb  # type: ignore  # noqa: F401
    except Exception:
        return False

    return True


def database_engine(paths: ProjectPaths) -> str:
    return "duckdb" if duckdb_available(paths) else "sqlite"


def database_available() -> bool:
    return True


def connect(paths: ProjectPaths) -> Any:
    paths.data_processed.mkdir(parents=True, exist_ok=True)
    if duckdb_available(paths):
        import duckdb  # type: ignore

        return duckdb.connect(str(duckdb_database_path(paths)))
    con = sqlite3.connect(sqlite_database_path(paths))
    con.row_factory = sqlite3.Row
    return con


def _ident(name: str) -> str:
    clean = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in name)
    if not clean:
        raise ValueError("empty database identifier")
    return '"' + clean + '"'


def _sql_string(value: Path | str) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _row_value(row: Any, key: str, index: int) -> Any:
    if isinstance(row, sqlite3.Row):
        return row[key]
    return row[index]


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


def _sqlite_type(value: Any) -> str:
    if isinstance(value, bool | int):
        return "INTEGER"
    if isinstance(value, float):
        return "REAL"
    return "TEXT"


def _normalise(value: Any) -> Any:
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, bool):
        return int(value)
    return value


def _create_table_for_rows(con: Any, table_name: str, rows: list[dict[str, Any]]) -> list[str]:
    columns = sorted({key for row in rows for key in row})
    if not columns:
        return []
    types: dict[str, str] = {}
    for column in columns:
        sample = next((row.get(column) for row in rows if row.get(column) not in (None, "")), "")
        types[column] = _sqlite_type(sample)
    con.execute(f"DROP TABLE IF EXISTS {_ident(table_name)}")
    cols_sql = ", ".join(f"{_ident(column)} {types[column]}" for column in columns)
    con.execute(f"CREATE TABLE {_ident(table_name)} ({cols_sql})")
    return columns


def materialize_rows(paths: ProjectPaths, table_name: str, rows: list[dict[str, Any]]) -> bool:
    if not rows:
        return False
    con = connect(paths)
    try:
        columns = _create_table_for_rows(con, table_name, rows)
        placeholders = ", ".join(["?"] * len(columns))
        col_sql = ", ".join(_ident(column) for column in columns)
        con.executemany(
            f"INSERT INTO {_ident(table_name)} ({col_sql}) VALUES ({placeholders})",
            [tuple(_normalise(row.get(column)) for column in columns) for row in rows],
        )
        con.commit()
    finally:
        con.close()
    return True


def materialize_file(paths: ProjectPaths, table_name: str, path: Path) -> bool:
    if not path.exists():
        return False
    engine = database_engine(paths)
    con = connect(paths)
    try:
        if table_name == "stress_state" and _table_exists(con, table_name, engine):
            return True
    finally:
        con.close()
    meta = read_sidecar(path)
    if engine == "duckdb":
        return _materialize_file_duckdb(paths, table_name, path, meta)
    if table_name == "stress_state" and meta.get("physical_format") not in {"csv_fallback", "sqlite_csv_export"}:
        return False
    if meta.get("physical_format") in {"csv_fallback", "sqlite_csv_export"} or path.suffix == ".csv":
        return _materialize_csv_sqlite(paths, table_name, path)
    return materialize_rows(paths, table_name, read_table(path))


def _materialize_file_duckdb(paths: ProjectPaths, table_name: str, path: Path, meta: dict[str, Any]) -> bool:
    con = connect(paths)
    try:
        fmt = str(meta.get("physical_format", ""))
        literal = _sql_string(path)
        if fmt in {"csv_fallback", "sqlite_csv_export"} or path.suffix == ".csv":
            expr = f"read_csv_auto({literal}, header=true)"
        else:
            expr = f"read_parquet({literal})"
        con.execute(f"CREATE OR REPLACE TABLE {_ident(table_name)} AS SELECT * FROM {expr}")
    finally:
        con.close()
    return True


def _materialize_csv_sqlite(paths: ProjectPaths, table_name: str, path: Path, batch_size: int = 50_000) -> bool:
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        columns = list(reader.fieldnames or [])
        if not columns:
            return False
        con = connect(paths)
        try:
            con.execute(f"DROP TABLE IF EXISTS {_ident(table_name)}")
            cols_sql = ", ".join(f"{_ident(column)} TEXT" for column in columns)
            con.execute(f"CREATE TABLE {_ident(table_name)} ({cols_sql})")
            placeholders = ", ".join(["?"] * len(columns))
            col_sql = ", ".join(_ident(column) for column in columns)
            insert_sql = f"INSERT INTO {_ident(table_name)} ({col_sql}) VALUES ({placeholders})"
            batch: list[tuple[Any, ...]] = []
            for row in reader:
                batch.append(tuple(row.get(column) for column in columns))
                if len(batch) >= batch_size:
                    con.executemany(insert_sql, batch)
                    batch.clear()
            if batch:
                con.executemany(insert_sql, batch)
            con.commit()
        finally:
            con.close()
    return True


def initialize_mesh_schema(paths: ProjectPaths) -> None:
    con = connect(paths)
    try:
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS mesh_dataset (
              mesh_id TEXT PRIMARY KEY,
              source TEXT,
              crs TEXT,
              vertical_datum TEXT,
              node_count BIGINT,
              element_count BIGINT,
              storage_uri TEXT,
              storage_format TEXT,
              notes TEXT
            )
            """
        )
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS mesh_node (
              mesh_id TEXT,
              node_id BIGINT,
              x_m DOUBLE,
              y_m DOUBLE,
              z_m DOUBLE,
              lon DOUBLE,
              lat DOUBLE,
              depth_km DOUBLE
            )
            """
        )
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS mesh_element (
              mesh_id TEXT,
              element_id BIGINT,
              element_type TEXT,
              node_ids TEXT,
              region_tag TEXT,
              material_id TEXT
            )
            """
        )
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS mesh_field_index (
              mesh_id TEXT,
              field_name TEXT,
              time_label TEXT,
              storage_uri TEXT,
              storage_format TEXT,
              array_path TEXT,
              units TEXT,
              notes TEXT
            )
            """
        )
        con.commit()
    finally:
        con.close()


def materialize_known_tables(paths: ProjectPaths) -> dict[str, str]:
    mapping = {
        "event": paths.data_processed / "event.parquet",
        "event_qc": paths.data_interim / "event_qc.parquet",
        "mechanism": paths.data_processed / "mechanism.parquet",
        "gnss_daily": paths.data_processed / "gnss_daily.parquet",
        "gnss_features": paths.data_processed / "gnss_features.parquet",
        "jshis_features": paths.data_processed / "jshis_features.parquet",
        "waveform_feature": paths.data_processed / "waveform_feature.parquet",
        "data_quality_epoch": paths.data_processed / "data_quality_epoch.parquet",
        "historical_data_profile": paths.data_processed / "historical_data_profile.parquet",
        "waveform_spectrum": paths.data_processed / "waveform_spectrum.parquet",
        "site_transfer_function": paths.data_processed / "site_transfer_function.parquet",
        "transfer_validation": paths.data_processed / "transfer_validation.parquet",
        "structure_anomaly": paths.data_processed / "structure_anomaly.parquet",
        "domestic_data_source": paths.data_processed / "domestic_data_source.parquet",
        "domestic_ingest_plan": paths.data_processed / "domestic_ingest_plan.parquet",
        "stress_state": paths.data_processed / "stress_state.parquet",
        "failure_scenarios": paths.outputs_tables / "failure_scenarios.parquet",
        "fault_ranking": paths.outputs_tables / "fault_ranking.csv",
    }
    results: dict[str, str] = {}
    engine = database_engine(paths)
    initialize_mesh_schema(paths)
    for name, path in mapping.items():
        if not path.exists():
            continue
        try:
            results[name] = "materialized" if materialize_file(paths, name, path) else "skipped"
        except Exception as exc:
            results[name] = f"failed: {type(exc).__name__}: {exc}"
    metadata_path = paths.data_processed / f"crust_lite.{engine}.metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "engine": engine,
                "database_path": str(database_path(paths)),
                "tables": results,
                "mesh_schema": {
                    "status": "initialized_empty_extension_schema",
                    "tables": ["mesh_dataset", "mesh_node", "mesh_element", "mesh_field_index"],
                    "large_field_storage": "HDF5/Zarr/XDMF or solver-native files indexed by mesh_field_index",
                },
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return results


def create_stress_table(paths: ProjectPaths) -> Any:
    con = connect(paths)
    con.execute("DROP TABLE IF EXISTS stress_state")
    con.execute(
        """
        CREATE TABLE stress_state (
          segment_id TEXT,
          date TEXT,
          event_id TEXT,
          cfs_pa DOUBLE,
          shear_pa DOUBLE,
          normal_pa DOUBLE,
          cfs_score_approx DOUBLE,
          stress_method TEXT,
          effective_friction DOUBLE,
          distance_km DOUBLE,
          mechanism_alignment DOUBLE,
          is_sample_data BOOLEAN
        )
        """
    )
    con.execute("CREATE INDEX IF NOT EXISTS idx_stress_segment ON stress_state(segment_id)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_stress_date_segment ON stress_state(date, segment_id)")
    return con


def export_table_to_csv_path(
    paths: ProjectPaths,
    table_name: str,
    output: Path,
    metadata: dict[str, Any],
    batch_size: int = 50_000,
) -> None:
    engine = database_engine(paths)
    con = connect(paths)
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        cursor = con.execute(f"SELECT * FROM {_ident(table_name)}")
        columns = [desc[0] for desc in cursor.description]
        with output.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(columns)
            while True:
                rows = cursor.fetchmany(batch_size)
                if not rows:
                    break
                writer.writerows([tuple(row) for row in rows])
    finally:
        con.close()
    write_sidecar(output, {**metadata, "physical_format": f"{engine}_csv_export", "database_engine": engine})


def copy_table_to_parquet(paths: ProjectPaths, table_name: str, output: Path, metadata: dict[str, Any]) -> None:
    if database_engine(paths) == "duckdb":
        con = connect(paths)
        output.parent.mkdir(parents=True, exist_ok=True)
        try:
            con.execute(f"COPY {_ident(table_name)} TO {_sql_string(output)} (FORMAT PARQUET)")
        finally:
            con.close()
        write_sidecar(output, {**metadata, "physical_format": "parquet_duckdb", "database_engine": "duckdb"})
        return
    export_table_to_csv_path(paths, table_name, output, metadata)


def write_rows_csv_stream(
    rows: Iterable[dict[str, Any]],
    path: Path,
    fieldnames: list[str],
    metadata: dict[str, Any],
) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
            count += 1
    write_sidecar(path, {**metadata, "physical_format": "csv_fallback"})
    return count


def _stress_value_sql(engine: str, scale: bool) -> str:
    multiplier = " * 1.0e5" if scale else ""
    if engine == "duckdb":
        return f"COALESCE(TRY_CAST(cfs_pa AS DOUBLE), TRY_CAST(cfs_score_approx AS DOUBLE){multiplier}, 0.0)"
    return f"COALESCE(CAST(NULLIF(cfs_pa, '') AS REAL), CAST(NULLIF(cfs_score_approx, '') AS REAL){multiplier}, 0.0)"


def stress_sum_by_segment(paths: ProjectPaths) -> dict[str, float] | None:
    engine = database_engine(paths)
    con = connect(paths)
    try:
        if not _table_exists(con, "stress_state", engine):
            con.close()
            if not materialize_file(paths, "stress_state", paths.data_processed / "stress_state.parquet"):
                return None
            con = connect(paths)
        rows = con.execute(
            f"""
            SELECT CAST(segment_id AS TEXT) AS segment_id,
                   SUM({_stress_value_sql(engine, scale=True)}) AS cfs_sum
            FROM stress_state
            GROUP BY segment_id
            """
        ).fetchall()
    finally:
        con.close()
    return {str(_row_value(row, "segment_id", 0)): float(_row_value(row, "cfs_sum", 1) or 0.0) for row in rows}


def latest_stress_by_segment(paths: ProjectPaths) -> dict[str, float] | None:
    engine = database_engine(paths)
    con = connect(paths)
    try:
        if not _table_exists(con, "stress_state", engine):
            con.close()
            if not materialize_file(paths, "stress_state", paths.data_processed / "stress_state.parquet"):
                return None
            con = connect(paths)
        rows = con.execute(
            f"""
            WITH latest AS (SELECT MAX(date) AS d FROM stress_state)
            SELECT CAST(segment_id AS TEXT) AS segment_id,
                   AVG({_stress_value_sql(engine, scale=False)}) AS value
            FROM stress_state, latest
            WHERE date = latest.d
            GROUP BY segment_id
            """
        ).fetchall()
    finally:
        con.close()
    return {str(_row_value(row, "segment_id", 0)): float(_row_value(row, "value", 1) or 0.0) for row in rows}


def stress_time_bins(paths: ProjectPaths, requested_bin_days: int, max_frames: int) -> tuple[list[str], dict[tuple[str, str], float], str, int]:
    engine = database_engine(paths)
    con = connect(paths)
    try:
        if not _table_exists(con, "stress_state", engine):
            con.close()
            if not materialize_file(paths, "stress_state", paths.data_processed / "stress_state.parquet"):
                return [], {}, "database_unavailable", requested_bin_days
            con = connect(paths)
        meta = con.execute("SELECT MIN(date) AS min_date, MAX(date) AS max_date, MAX(stress_method) AS method FROM stress_state").fetchone()
        min_date_raw = _row_value(meta, "min_date", 0) if meta else None
        max_date_raw = _row_value(meta, "max_date", 1) if meta else None
        if min_date_raw is None or max_date_raw is None:
            return [], {}, "empty", requested_bin_days
        min_date = date.fromisoformat(str(min_date_raw)[:10])
        max_date = date.fromisoformat(str(max_date_raw)[:10])
        days = (max_date - min_date).days + 1
        actual_bin = max(1, int(requested_bin_days))
        if days / actual_bin > max_frames:
            import math

            actual_bin = max(1, int(math.ceil(days / max_frames)))
        if engine == "duckdb":
            rows = con.execute(
                f"""
                SELECT CAST(FLOOR(date_diff('day', CAST(? AS DATE), CAST(date AS DATE)) / ?) AS INTEGER) AS bin_id,
                       CAST(segment_id AS TEXT) AS segment_id,
                       AVG({_stress_value_sql(engine, scale=False)}) AS value
                FROM stress_state
                GROUP BY bin_id, segment_id
                ORDER BY bin_id, segment_id
                """,
                (min_date.isoformat(), actual_bin),
            ).fetchall()
        else:
            rows = con.execute(
                f"""
                SELECT CAST(((julianday(date) - julianday(?)) / ?) AS INTEGER) AS bin_id,
                       CAST(segment_id AS TEXT) AS segment_id,
                       AVG({_stress_value_sql(engine, scale=False)}) AS value
                FROM stress_state
                GROUP BY bin_id, segment_id
                ORDER BY bin_id, segment_id
                """,
                (min_date.isoformat(), actual_bin),
            ).fetchall()
        method = str(_row_value(meta, "method", 2) or "fallback_approximation")
    finally:
        con.close()
    labels_by_bin: dict[int, str] = {}
    values: dict[tuple[str, str], float] = {}
    for row in rows:
        bin_id = int(_row_value(row, "bin_id", 0))
        label = labels_by_bin.setdefault(bin_id, (min_date + timedelta(days=bin_id * actual_bin)).isoformat())
        values[(label, str(_row_value(row, "segment_id", 1)))] = float(_row_value(row, "value", 2) or 0.0)
    return [labels_by_bin[idx] for idx in sorted(labels_by_bin)], values, method, actual_bin


def failure_values_for_years(paths: ProjectPaths, years: list[int], value_column: str) -> dict[tuple[int, str], float] | None:
    if not years:
        return {}
    allowed = {"failure_index_p50", "failure_index_p95", "prob_index_gt_1", "uncertainty_score"}
    if value_column not in allowed:
        value_column = "failure_index_p50"
    engine = database_engine(paths)
    con = connect(paths)
    try:
        if not _table_exists(con, "failure_scenarios", engine):
            con.close()
            if not materialize_file(paths, "failure_scenarios", paths.outputs_tables / "failure_scenarios.parquet"):
                return None
            con = connect(paths)
        placeholders = ", ".join(["?"] * len(years))
        rows = con.execute(
            f"""
            SELECT CAST(year AS INTEGER) AS year, CAST(segment_id AS TEXT) AS segment_id,
                   CAST({_ident(value_column)} AS DOUBLE) AS value
            FROM failure_scenarios
            WHERE CAST(year AS INTEGER) IN ({placeholders})
            """,
            tuple(int(year) for year in years),
        ).fetchall()
    finally:
        con.close()
    return {
        (int(_row_value(row, "year", 0)), str(_row_value(row, "segment_id", 1))): float(_row_value(row, "value", 2) or 0.0)
        for row in rows
    }
