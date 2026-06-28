from __future__ import annotations

import csv
import math
from contextlib import suppress
from datetime import datetime
from typing import Any

from crust_lite.config import AppConfig
from crust_lite.geo import angle_difference_deg, clamp01
from crust_lite.io.database import (
    connect,
    copy_table_to_parquet,
    database_available,
    database_engine,
    materialize_known_tables,
    write_rows_csv_stream,
)
from crust_lite.io.geopackage import read_features
from crust_lite.io.parquet import read_table, write_table
from crust_lite.logging import get_logger
from crust_lite.paths import ProjectPaths
from crust_lite.resources import ExecutionPlan, choose_execution_plan

LOGGER = get_logger(__name__)

SIGN_CONVENTION = "DeltaCFS = DeltaTau + mu_prime * DeltaSigmaN; opening normal stress is positive"
STRESS_COLUMNS = [
    "segment_id",
    "date",
    "event_id",
    "cfs_pa",
    "shear_pa",
    "normal_pa",
    "cfs_score_approx",
    "stress_method",
    "effective_friction",
    "distance_km",
    "mechanism_alignment",
    "is_sample_data",
]


def _cutde_available() -> bool:
    try:
        import cutde  # type: ignore  # noqa: F401

        return True
    except Exception:
        return False


def _feature_center(feature: dict[str, Any]) -> tuple[float, float, float]:
    props = feature.get("properties", {})
    return (
        float(props.get("center_x_m", 0.0)),
        float(props.get("center_y_m", 0.0)),
        float(props.get("center_depth_km", props.get("bottom_depth_km", 10.0) / 2.0)) * 1000.0,
    )


def _event_time(row: dict[str, Any]) -> str:
    value = str(row["time_utc"])
    return datetime.fromisoformat(value.replace("Z", "+00:00")).date().isoformat()


def _fallback_row(
    event: dict[str, Any],
    feature: dict[str, Any],
    effective_friction: float,
) -> dict[str, Any]:
    props = feature.get("properties", {})
    fx, fy, fz = _feature_center(feature)
    ex, ey, ez = float(event["x_m"]), float(event["y_m"]), float(event["z_m"])
    distance_km = max(0.1, math.dist((fx, fy, fz), (ex, ey, ez)) / 1000.0)
    mag = float(event["magnitude"])
    strike = float(props.get("strike", 0.0))
    alignment = 1.0 - min(90.0, angle_difference_deg(strike, strike)) / 90.0
    magnitude_scale = clamp01((mag - 2.0) / 5.0)
    cfs_score = clamp01(math.exp(-distance_km / 35.0) * (0.35 + 0.65 * magnitude_scale) * alignment)
    return {
        "segment_id": props.get("segment_id", ""),
        "date": _event_time(event),
        "event_id": event["event_id"],
        "cfs_pa": None,
        "shear_pa": None,
        "normal_pa": None,
        "cfs_score_approx": cfs_score,
        "stress_method": "fallback_approximation",
        "effective_friction": effective_friction,
        "distance_km": distance_km,
        "mechanism_alignment": alignment,
        "is_sample_data": str(event.get("is_sample_data", "")).lower() == "true"
        or bool(props.get("is_sample_data")),
    }


def _load_features(paths: ProjectPaths) -> list[dict[str, Any]]:
    features: list[dict[str, Any]] = []
    for path in (paths.data_processed / "fault_segment.gpkg", paths.data_processed / "inferred_faults.gpkg"):
        if path.exists():
            features.extend(read_features(path))
    if not features:
        raise ValueError("No known or inferred faults are available for stress calculation")
    return features


def _row_tuple(row: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(row.get(column) for column in STRESS_COLUMNS)


def _metadata(is_sample: bool, plan: ExecutionPlan | None = None) -> dict[str, Any]:
    return {
        "is_sample_data": is_sample,
        "stress_method": "fallback_approximation",
        "sign_convention": SIGN_CONVENTION,
        "warning": "cfs_score_approx is a relative score, not a Pa-valued stress change",
        "storage": "database_csv_stream_bulk_load" if database_available() else "csv_stream_fallback",
        "execution_plan": plan.as_metadata() if plan else {},
    }


def _compute_stress_duckdb(
    events: list[dict[str, Any]],
    features: list[dict[str, Any]],
    paths: ProjectPaths,
    effective_friction: float,
    is_sample: bool,
    plan: ExecutionPlan,
) -> int:
    build_csv = paths.data_interim / "stress_state_build.csv"
    build_csv.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with build_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=STRESS_COLUMNS)
        writer.writeheader()
        for event in events:
            for feature in features:
                writer.writerow(_fallback_row(event, feature, effective_friction))
                count += 1
    engine = database_engine(paths)
    if engine == "duckdb":
        con = connect(paths)
        try:
            literal = "'" + str(build_csv).replace("'", "''") + "'"
            con.execute(
                "CREATE OR REPLACE TABLE stress_state AS "
                f"SELECT * FROM read_csv_auto({literal}, header=true, sample_size=-1)"
            )
        finally:
            con.close()
    else:
        con = connect(paths)
        try:
            con.execute("DROP TABLE IF EXISTS stress_state")
            con.execute(
                """
                CREATE TABLE stress_state (
                  segment_id TEXT,
                  date TEXT,
                  event_id TEXT,
                  cfs_pa REAL,
                  shear_pa REAL,
                  normal_pa REAL,
                  cfs_score_approx REAL,
                  stress_method TEXT,
                  effective_friction REAL,
                  distance_km REAL,
                  mechanism_alignment REAL,
                  is_sample_data INTEGER
                )
                """
            )
            placeholders = ", ".join(["?"] * len(STRESS_COLUMNS))
            insert_sql = f"INSERT INTO stress_state VALUES ({placeholders})"
            with build_csv.open("r", encoding="utf-8", newline="") as fh:
                reader = csv.DictReader(fh)
                batch: list[tuple[Any, ...]] = []
                for row in reader:
                    batch.append(tuple(row.get(column) for column in STRESS_COLUMNS))
                    if len(batch) >= plan.batch_rows:
                        con.executemany(insert_sql, batch)
                        batch.clear()
                if batch:
                    con.executemany(insert_sql, batch)
            con.commit()
        finally:
            con.close()
    copy_table_to_parquet(paths, "stress_state", paths.data_processed / "stress_state.parquet", _metadata(is_sample, plan))
    with suppress(FileNotFoundError):
        build_csv.unlink()
    materialize_known_tables(paths)
    return count


def _compute_stress_memory(
    events: list[dict[str, Any]],
    features: list[dict[str, Any]],
    paths: ProjectPaths,
    effective_friction: float,
    is_sample: bool,
    plan: ExecutionPlan,
) -> int:
    rows = [
        _fallback_row(event, feature, effective_friction)
        for event in events
        for feature in features
    ]
    write_table(rows, paths.data_processed / "stress_state.parquet", _metadata(is_sample, plan))
    materialize_known_tables(paths)
    return len(rows)


def _compute_stress_csv(
    events: list[dict[str, Any]],
    features: list[dict[str, Any]],
    paths: ProjectPaths,
    effective_friction: float,
    is_sample: bool,
) -> int:
    def rows() -> Any:
        for event in events:
            for feature in features:
                yield _fallback_row(event, feature, effective_friction)

    return write_rows_csv_stream(
        rows(),
        paths.data_processed / "stress_state.parquet",
        STRESS_COLUMNS,
        _metadata(is_sample),
    )


def compute_stress(config: AppConfig, paths: ProjectPaths) -> dict[str, Any]:
    events = read_table(paths.data_interim / "event_qc.parquet")
    features = _load_features(paths)
    effective_friction = sum(config.simulation.effective_friction_range) / 2.0
    if _cutde_available():
        # MVP extension point: event magnitude would be converted to a rectangular
        # source using a Wells-Coppersmith-style log-length/log-width scaling,
        # split into triangular elements, and projected to receiver planes. The
        # current implementation keeps the interface but uses the fallback unless
        # this block is expanded in a v1 stress kernel.
        LOGGER.warning("cutde is importable, but the MVP still uses the approximation kernel")
    is_sample = any(str(row.get("is_sample_data", "")).lower() == "true" for row in events) or any(
        bool(feature.get("properties", {}).get("is_sample_data")) for feature in features
    )
    estimated_pairs = len(events) * len(features)
    plan = choose_execution_plan(
        config,
        operation="stress_state",
        engine=database_engine(paths),
        estimated_rows=estimated_pairs,
        estimated_row_bytes=256,
    )
    LOGGER.info("Stress execution plan: %s", plan.as_metadata())
    if plan.use_in_memory:
        count = _compute_stress_memory(events, features, paths, effective_friction, is_sample, plan)
    elif database_available():
        count = _compute_stress_duckdb(events, features, paths, effective_friction, is_sample, plan)
    else:
        count = _compute_stress_csv(events, features, paths, effective_friction, is_sample)
    LOGGER.info("Computed fallback stress scores for %d event-fault pairs", count)
    return {"stress_rows": count, "stress_method": "fallback_approximation", "is_sample_data": is_sample}
