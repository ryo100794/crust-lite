"""CPU-side pre-GPU preparation for dense Gaussian splat fusion.

This module intentionally runs before H200/5090 work.  It converts compact
Gaussian primitives into LOD voxel density tables and shard indexes so GPU
jobs can focus on fusion/optimization instead of raw waveform preprocessing.
The products are research state indicators, not earthquake predictions.
"""

from __future__ import annotations

import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

from crust_lite.config import AppConfig
from crust_lite.io.database import connect, database_engine, materialize_file
from crust_lite.io.parquet import read_sidecar, read_table, write_sidecar, write_table
from crust_lite.logging import get_logger
from crust_lite.paths import ProjectPaths

LOGGER = get_logger(__name__)

VOXEL_COLUMNS = [
    "lod",
    "voxel_size_m",
    "ix",
    "iy",
    "iz",
    "center_x_m",
    "center_y_m",
    "center_z_m",
    "density",
    "weighted_amplitude",
    "weighted_opacity",
    "direct_density",
    "reflected_density",
    "scattered_density",
    "residual_density",
    "expanded_contribution_count",
    "n_events",
    "is_sample_data",
]

SHARD_COLUMNS = [
    "lod",
    "shard_x",
    "shard_y",
    "shard_z",
    "voxel_count",
    "density_sum",
    "event_count",
    "min_ix",
    "max_ix",
    "min_iy",
    "max_iy",
    "min_iz",
    "max_iz",
    "storage_uri",
]


def _sql_literal(path: Path) -> str:
    return "'" + str(path).replace("'", "''") + "'"


def _lod_sizes(sample: bool) -> list[tuple[int, float]]:
    if sample:
        return [(0, 20_000.0), (1, 40_000.0)]
    return [(0, 2_500.0), (1, 5_000.0), (2, 10_000.0), (3, 20_000.0), (4, 40_000.0)]


def _offsets(sample: bool) -> list[tuple[int, int, int]]:
    radius = 1 if sample else 2
    values: list[tuple[int, int, int]] = []
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            for dz in range(-radius, radius + 1):
                if dx * dx + dy * dy + dz * dz <= radius * radius:
                    values.append((dx, dy, dz))
    return values


def _env_int(name: str, default: int, min_value: int, max_value: int) -> int:
    raw = os.environ.get(name)
    if raw:
        try:
            return max(min_value, min(max_value, int(raw)))
        except ValueError:
            LOGGER.warning("Ignoring invalid integer environment value %s=%r", name, raw)
    return default


def _env_float(name: str, default: float, min_value: float, max_value: float) -> float:
    raw = os.environ.get(name)
    if raw:
        try:
            return max(min_value, min(max_value, float(raw)))
        except ValueError:
            LOGGER.warning("Ignoring invalid float environment value %s=%r", name, raw)
    return default


def _image_kernel_offsets(sample: bool) -> list[tuple[int, int]]:
    radius = _env_int("CRUST_LITE_SPLAT_IMAGE_RADIUS", 2 if sample else 4, 0, 24)
    values: list[tuple[int, int]] = []
    for du in range(-radius, radius + 1):
        for dv in range(-radius, radius + 1):
            if du * du + dv * dv <= radius * radius:
                values.append((du, dv))
    return values or [(0, 0)]


def _projection_image_pixel_size_m(sample: bool) -> float:
    return _env_float("CRUST_LITE_SPLAT_IMAGE_PIXEL_M", 10_000.0 if sample else 2_500.0, 250.0, 100_000.0)


def _build_projection_view_images_duckdb(con: Any, paths: ProjectPaths, is_sample: bool) -> dict[str, Any]:
    spectrum_path = paths.data_processed / "waveform_spectrum.parquet"
    projection_path = paths.data_processed / "waveform_array_projection.parquet"
    part_dir = paths.data_processed / "splat_view_image_parts"
    part_index_path = paths.data_processed / "splat_view_image_part_index.parquet"
    if not spectrum_path.exists() or not projection_path.exists():
        return {
            "view_image_status": "skipped_missing_waveform_spectrum_or_projection",
            "view_image_rows": 0,
            "view_image_part_rows": 0,
        }
    materialize_file(paths, "waveform_spectrum", spectrum_path)
    materialize_file(paths, "waveform_array_projection", projection_path)

    pixel_size_m = _projection_image_pixel_size_m(is_sample)
    offsets = _image_kernel_offsets(is_sample)
    radius_label = max(max(abs(du), abs(dv)) for du, dv in offsets)
    shard_count = _env_int("CRUST_LITE_SPLAT_IMAGE_SHARDS", 8 if is_sample else 64, 1, 4096)
    part_dir = paths.data_processed / f"splat_view_image_parts_r{radius_label}_p{int(pixel_size_m)}_s{shard_count}"
    part_index_path = paths.data_processed / f"splat_view_image_part_index_r{radius_label}_p{int(pixel_size_m)}_s{shard_count}.parquet"
    part_dir.mkdir(parents=True, exist_ok=True)
    resume_parts = str(os.environ.get("CRUST_LITE_SPLAT_IMAGE_RESUME", "")).lower() in {"1", "true", "yes", "on"}
    if resume_parts:
        for old in part_dir.glob("part-*.parquet"):
            if old.stat().st_size == 0:
                old.unlink()
    else:
        for old in part_dir.glob("part-*.parquet"):
            old.unlink()
    offset_values = ", ".join(f"({du}, {dv})" for du, dv in offsets)
    kernel_scale = max(1.0, math.sqrt(max(len(offsets), 1)) / 2.0)
    con.execute("CREATE OR REPLACE TEMP TABLE splat_image_offsets AS SELECT * FROM (VALUES " + offset_values + ") AS t(du, dv)")

    part_rows: list[dict[str, Any]] = []
    total_rows = 0
    total_views = 0
    for shard_id in range(shard_count):
        part_path = part_dir / f"part-{shard_id:05d}.parquet"
        if resume_parts and part_path.exists() and part_path.stat().st_size > 0:
            row_count, view_count, intensity_sum = con.execute(
                f"SELECT COUNT(*), COUNT(DISTINCT view_id), SUM(intensity) FROM read_parquet({_sql_literal(part_path)})"
            ).fetchone()
            row_count = int(row_count or 0)
            view_count = int(view_count or 0)
            intensity_sum = float(intensity_sum or 0.0)
            LOGGER.info(
                "Skipping existing splat view-image shard %d/%d rows=%d",
                shard_id + 1,
                shard_count,
                row_count,
            )
            part_rows.append(
                {
                    "part_id": shard_id,
                    "storage_uri": str(part_path),
                    "pixel_rows": row_count,
                    "view_count": view_count,
                    "intensity_sum": intensity_sum,
                    "pixel_size_m": pixel_size_m,
                    "kernel_offsets": len(offsets),
                    "kernel_radius": radius_label,
                    "is_sample_data": is_sample,
                }
            )
            total_rows += row_count
            total_views += view_count
            continue
        if part_path.exists():
            part_path.unlink()
        LOGGER.info("Building splat view-image shard %d/%d", shard_id + 1, shard_count)
        con.execute(
            f"""
            CREATE OR REPLACE TEMP TABLE splat_view_image_pixel_part AS
            WITH spectrum AS (
              SELECT
                CAST(event_id AS VARCHAR) AS event_id,
                CAST(station_id AS VARCHAR) AS station_id,
                COALESCE(TRY_CAST(frequency_hz AS DOUBLE), 0.0) AS frequency_hz,
                6378137.0 * RADIANS(COALESCE(TRY_CAST(lon AS DOUBLE), 0.0)) AS station_x_m,
                6378137.0 * LN(TAN(PI() / 4.0 + RADIANS(GREATEST(LEAST(COALESCE(TRY_CAST(lat AS DOUBLE), 0.0), 85.0), -85.0)) / 2.0)) AS station_y_m,
                COALESCE(TRY_CAST(amplitude AS DOUBLE), 0.0) AS spectrum_amplitude,
                COALESCE(TRY_CAST(phase_rad AS DOUBLE), 0.0) AS station_phase_rad,
                COALESCE(TRY_CAST(group_delay_s AS DOUBLE), 0.0) AS group_delay_s,
                LOWER(CAST(COALESCE(is_sample_data, false) AS VARCHAR)) IN ('true', '1') AS is_sample_data
              FROM waveform_spectrum
              WHERE event_id IS NOT NULL AND station_id IS NOT NULL AND COALESCE(TRY_CAST(frequency_hz AS DOUBLE), 0.0) > 0.0
            ),
            projection AS (
              SELECT
                CAST(event_id AS VARCHAR) AS event_id,
                COALESCE(TRY_CAST(frequency_hz AS DOUBLE), 0.0) AS frequency_hz,
                COALESCE(TRY_CAST(projection_x_m AS DOUBLE), 0.0) AS projection_x_m,
                COALESCE(TRY_CAST(projection_y_m AS DOUBLE), 0.0) AS projection_y_m,
                COALESCE(TRY_CAST(projection_z_m AS DOUBLE), 0.0) AS projection_z_m,
                COALESCE(TRY_CAST(beam_energy AS DOUBLE), 0.0) AS beam_energy,
                COALESCE(TRY_CAST(array_coherence AS DOUBLE), 0.5) AS array_coherence,
                COALESCE(TRY_CAST(phase_resultant_rad AS DOUBLE), 0.0) AS projection_phase_rad,
                CAST(COALESCE(primitive_type, 'residual') AS VARCHAR) AS primitive_type,
                CAST(COALESCE(path_family, 'unknown') AS VARCHAR) AS path_family,
                COALESCE(TRY_CAST(projection_rank AS BIGINT), 0) AS projection_rank,
                LOWER(CAST(COALESCE(is_sample_data, false) AS VARCHAR)) IN ('true', '1') AS is_sample_data
              FROM waveform_array_projection
              WHERE COALESCE(TRY_CAST(frequency_hz AS DOUBLE), 0.0) > 0.0
                AND CAST(HASH(CAST(event_id AS VARCHAR)) % {shard_count} AS BIGINT) = {shard_id}
            ),
            joined AS (
              SELECT
                p.event_id,
                s.station_id,
                p.frequency_hz,
                p.primitive_type,
                p.path_family,
                p.projection_rank,
                p.projection_x_m - s.station_x_m AS rel_x_m,
                p.projection_y_m - s.station_y_m AS rel_y_m,
                SQRT(POWER(p.projection_x_m - s.station_x_m, 2) + POWER(p.projection_y_m - s.station_y_m, 2)) AS range_m,
                p.projection_z_m AS depth_m,
                0.5 + 0.5 * COS(s.station_phase_rad - p.projection_phase_rad) AS phase_alignment,
                s.group_delay_s,
                p.beam_energy,
                p.array_coherence,
                s.spectrum_amplitude,
                p.is_sample_data OR s.is_sample_data AS is_sample_data
              FROM projection p
              JOIN spectrum s
                ON p.event_id = s.event_id
               AND ABS(p.frequency_hz - s.frequency_hz) < 1.0e-9
            ),
            base_pixels AS (
              SELECT *, 'xy' AS image_plane, rel_x_m AS base_u_m, rel_y_m AS base_v_m FROM joined
              UNION ALL
              SELECT *, 'range_depth' AS image_plane, range_m AS base_u_m, depth_m AS base_v_m FROM joined
            ),
            expanded AS (
              SELECT
                event_id,
                station_id,
                frequency_hz,
                image_plane,
                primitive_type,
                path_family,
                CAST(FLOOR((base_u_m + o.du * {pixel_size_m}) / {pixel_size_m}) AS BIGINT) AS pixel_u,
                CAST(FLOOR((base_v_m + o.dv * {pixel_size_m}) / {pixel_size_m}) AS BIGINT) AS pixel_v,
                EXP(-0.5 * (POWER(o.du / {kernel_scale}, 2) + POWER(o.dv / {kernel_scale}, 2)))
                  * beam_energy
                  * array_coherence
                  * spectrum_amplitude
                  * phase_alignment AS intensity,
                phase_alignment,
                group_delay_s,
                projection_rank,
                is_sample_data
              FROM base_pixels
              CROSS JOIN splat_image_offsets o
            )
            SELECT
              event_id || ':' || station_id || ':' || CAST(frequency_hz AS VARCHAR) || ':' || image_plane AS view_id,
              event_id,
              station_id,
              frequency_hz,
              image_plane,
              {pixel_size_m}::DOUBLE AS pixel_size_m,
              pixel_u,
              pixel_v,
              (pixel_u + 0.5) * {pixel_size_m}::DOUBLE AS center_u_m,
              (pixel_v + 0.5) * {pixel_size_m}::DOUBLE AS center_v_m,
              primitive_type,
              path_family,
              SUM(intensity) AS intensity,
              AVG(phase_alignment) AS phase_alignment_mean,
              AVG(group_delay_s) AS group_delay_s_mean,
              MIN(projection_rank) AS best_projection_rank,
              COUNT(*) AS contribution_count,
              BOOL_OR(is_sample_data) AS is_sample_data
            FROM expanded
            GROUP BY event_id, station_id, frequency_hz, image_plane, pixel_u, pixel_v, primitive_type, path_family
            HAVING SUM(intensity) > 1.0e-12
            """
        )
        row_count, view_count, intensity_sum = con.execute(
            "SELECT COUNT(*), COUNT(DISTINCT view_id), SUM(intensity) FROM splat_view_image_pixel_part"
        ).fetchone()
        row_count = int(row_count or 0)
        view_count = int(view_count or 0)
        intensity_sum = float(intensity_sum or 0.0)
        if row_count:
            con.execute(f"COPY splat_view_image_pixel_part TO {_sql_literal(part_path)} (FORMAT PARQUET)")
        part_rows.append(
            {
                "part_id": shard_id,
                "storage_uri": str(part_path),
                "pixel_rows": row_count,
                "view_count": view_count,
                "intensity_sum": intensity_sum,
                "pixel_size_m": pixel_size_m,
                "kernel_offsets": len(offsets),
                "kernel_radius": radius_label,
                "is_sample_data": is_sample,
            }
        )
        total_rows += row_count
        total_views += view_count
    write_table(
        part_rows,
        part_index_path,
        {
            "method": "partitioned_cpu_duckdb_event_station_projection_image_shards",
            "physical_format": "parquet",
            "is_sample_data": is_sample,
            "not_prediction": True,
            "pixel_size_m": pixel_size_m,
            "kernel_offsets": len(offsets),
            "kernel_radius": radius_label,
            "part_count": shard_count,
            "view_image_rows": total_rows,
            "resume_parts": resume_parts,
        },
    )
    return {
        "method_view_images": "partitioned_cpu_duckdb_event_station_projection_image_shards",
        "view_image_status": "partitioned",
        "is_sample_data": is_sample,
        "not_prediction": True,
        "pixel_size_m": pixel_size_m,
        "view_image_kernel_offsets": len(offsets),
        "view_image_kernel_radius": radius_label,
        "view_image_rows": total_rows,
        "view_image_part_rows": len(part_rows),
        "view_count": total_views,
        "view_image_resume_parts": resume_parts,
        "image_planes": ["xy", "range_depth"],
        "station_projection_crs_method": "EPSG:3857 spherical mercator from station lat/lon",
        "image_partition_dir": str(part_dir),
        "image_part_index_path": str(part_index_path),
    }


def build_gpu_prep(config: AppConfig, paths: ProjectPaths, sample: bool = False) -> dict[str, Any]:
    """Build CPU-prepared voxel LOD and shard index for later GPU fusion."""
    paths.ensure()
    splat_path = paths.data_processed / "gaussian_splat_primitive.parquet"
    if not splat_path.exists():
        raise FileNotFoundError(f"Cannot build gpu-prep because Gaussian splat table is missing: {splat_path}")
    meta = read_sidecar(splat_path)
    is_sample = bool(meta.get("is_sample_data", sample))
    if database_engine(paths) == "duckdb":
        result = _build_gpu_prep_duckdb(config, paths, splat_path, is_sample=is_sample)
    else:
        result = _build_gpu_prep_python(config, paths, splat_path, is_sample=is_sample)
    _write_manifest(config, paths, result)
    _write_report(paths, result)
    LOGGER.info(
        "Built CPU pre-GPU voxel LOD: voxels=%s shards=%s splats=%s",
        result.get("voxel_rows"),
        result.get("shard_rows"),
        result.get("input_splats"),
    )
    return result


def _build_gpu_prep_duckdb(config: AppConfig, paths: ProjectPaths, splat_path: Path, *, is_sample: bool) -> dict[str, Any]:
    materialize_file(paths, "gaussian_splat_primitive", splat_path)
    voxel_path = paths.data_processed / "gaussian_splat_voxel_lod.parquet"
    shard_path = paths.data_processed / "gaussian_splat_gpu_shard_index.parquet"
    lod_values = ", ".join(f"({lod}, {size})" for lod, size in _lod_sizes(is_sample))
    offset_values = ", ".join(f"({dx}, {dy}, {dz})" for dx, dy, dz in _offsets(is_sample))
    con = connect(paths)
    try:
        con.execute("CREATE OR REPLACE TEMP TABLE gpu_lod AS SELECT * FROM (VALUES " + lod_values + ") AS t(lod, voxel_size_m)")
        con.execute("CREATE OR REPLACE TEMP TABLE gpu_offsets AS SELECT * FROM (VALUES " + offset_values + ") AS t(dx, dy, dz)")
        con.execute(
            """
            CREATE OR REPLACE TABLE gaussian_splat_voxel_lod AS
            WITH clean AS (
              SELECT
                CAST(event_id AS VARCHAR) AS event_id,
                COALESCE(TRY_CAST(x_m AS DOUBLE), 0.0) AS x_m,
                COALESCE(TRY_CAST(y_m AS DOUBLE), 0.0) AS y_m,
                COALESCE(TRY_CAST(z_m AS DOUBLE), 0.0) AS z_m,
                GREATEST(COALESCE(TRY_CAST(sigma_x_m AS DOUBLE), 5000.0), 1.0) AS sigma_x_m,
                GREATEST(COALESCE(TRY_CAST(sigma_y_m AS DOUBLE), 5000.0), 1.0) AS sigma_y_m,
                GREATEST(COALESCE(TRY_CAST(sigma_z_m AS DOUBLE), 5000.0), 1.0) AS sigma_z_m,
                COALESCE(TRY_CAST(amplitude AS DOUBLE), 0.0) AS amplitude,
                COALESCE(TRY_CAST(opacity AS DOUBLE), 0.5) AS opacity,
                CAST(COALESCE(primitive_type, 'residual') AS VARCHAR) AS primitive_type,
                LOWER(CAST(COALESCE(is_sample_data, false) AS VARCHAR)) IN ('true', '1') AS is_sample_data
              FROM gaussian_splat_primitive
            ),
            expanded AS (
              SELECT
                l.lod,
                l.voxel_size_m,
                CAST(FLOOR((c.x_m + o.dx * l.voxel_size_m) / l.voxel_size_m) AS BIGINT) AS ix,
                CAST(FLOOR((c.y_m + o.dy * l.voxel_size_m) / l.voxel_size_m) AS BIGINT) AS iy,
                CAST(FLOOR((c.z_m + o.dz * l.voxel_size_m) / l.voxel_size_m) AS BIGINT) AS iz,
                c.event_id,
                c.primitive_type,
                c.is_sample_data,
                EXP(-0.5 * (
                  POWER(o.dx * l.voxel_size_m / c.sigma_x_m, 2) +
                  POWER(o.dy * l.voxel_size_m / c.sigma_y_m, 2) +
                  POWER(o.dz * l.voxel_size_m / c.sigma_z_m, 2)
                )) * c.amplitude * c.opacity AS contribution,
                c.amplitude,
                c.opacity
              FROM clean c
              CROSS JOIN gpu_lod l
              CROSS JOIN gpu_offsets o
            )
            SELECT
              lod,
              voxel_size_m,
              ix,
              iy,
              iz,
              (ix + 0.5) * voxel_size_m AS center_x_m,
              (iy + 0.5) * voxel_size_m AS center_y_m,
              (iz + 0.5) * voxel_size_m AS center_z_m,
              SUM(contribution) AS density,
              SUM(contribution * amplitude) / NULLIF(SUM(contribution), 0.0) AS weighted_amplitude,
              SUM(contribution * opacity) / NULLIF(SUM(contribution), 0.0) AS weighted_opacity,
              SUM(CASE WHEN primitive_type = 'direct' THEN contribution ELSE 0.0 END) AS direct_density,
              SUM(CASE WHEN primitive_type = 'reflected' THEN contribution ELSE 0.0 END) AS reflected_density,
              SUM(CASE WHEN primitive_type = 'scattered' THEN contribution ELSE 0.0 END) AS scattered_density,
              SUM(CASE WHEN primitive_type = 'residual' THEN contribution ELSE 0.0 END) AS residual_density,
              COUNT(*) AS expanded_contribution_count,
              COUNT(DISTINCT event_id) AS n_events,
              BOOL_OR(is_sample_data) AS is_sample_data
            FROM expanded
            GROUP BY lod, voxel_size_m, ix, iy, iz
            HAVING SUM(contribution) > 1.0e-8
            """
        )
        con.execute(
            """
            CREATE OR REPLACE TABLE gaussian_splat_gpu_shard_index AS
            SELECT
              lod,
              CAST(FLOOR(ix / 64.0) AS BIGINT) AS shard_x,
              CAST(FLOOR(iy / 64.0) AS BIGINT) AS shard_y,
              CAST(FLOOR(iz / 64.0) AS BIGINT) AS shard_z,
              COUNT(*) AS voxel_count,
              SUM(density) AS density_sum,
              SUM(n_events) AS event_count,
              MIN(ix) AS min_ix,
              MAX(ix) AS max_ix,
              MIN(iy) AS min_iy,
              MAX(iy) AS max_iy,
              MIN(iz) AS min_iz,
              MAX(iz) AS max_iz,
              'data/processed/gaussian_splat_voxel_lod.parquet' AS storage_uri
            FROM gaussian_splat_voxel_lod
            GROUP BY lod, shard_x, shard_y, shard_z
            ORDER BY lod, shard_x, shard_y, shard_z
            """
        )
        con.execute(f"COPY gaussian_splat_voxel_lod TO {_sql_literal(voxel_path)} (FORMAT PARQUET)")
        con.execute(f"COPY gaussian_splat_gpu_shard_index TO {_sql_literal(shard_path)} (FORMAT PARQUET)")
        input_splats = con.execute("SELECT COUNT(*) FROM gaussian_splat_primitive").fetchone()[0]
        voxel_rows = con.execute("SELECT COUNT(*) FROM gaussian_splat_voxel_lod").fetchone()[0]
        shard_rows = con.execute("SELECT COUNT(*) FROM gaussian_splat_gpu_shard_index").fetchone()[0]
        lod_summary = [
            {"lod": int(row[0]), "voxel_size_m": float(row[1]), "voxel_rows": int(row[2]), "density_sum": float(row[3] or 0.0)}
            for row in con.execute(
                """
                SELECT lod, voxel_size_m, COUNT(*) AS voxel_rows, SUM(density) AS density_sum
                FROM gaussian_splat_voxel_lod
                GROUP BY lod, voxel_size_m
                ORDER BY lod
                """
            ).fetchall()
        ]
        view_image_meta = _build_projection_view_images_duckdb(con, paths, is_sample)
    finally:
        con.close()
    metadata = {
        "method": "cpu_duckdb_gaussian_splat_voxel_lod_expansion",
        "database_engine": "duckdb",
        "is_sample_data": is_sample,
        "not_prediction": True,
        "input_splats": int(input_splats),
        "voxel_rows": int(voxel_rows),
        "shard_rows": int(shard_rows),
        "lod_summary": lod_summary,
        "kernel_offsets": len(_offsets(is_sample)),
        **view_image_meta,
    }
    write_sidecar(voxel_path, {**metadata, "table": "gaussian_splat_voxel_lod", "physical_format": "parquet_duckdb"})
    write_sidecar(shard_path, {**metadata, "table": "gaussian_splat_gpu_shard_index", "physical_format": "parquet_duckdb"})
    return {**metadata, "voxel_path": str(voxel_path), "shard_path": str(shard_path)}


def _build_gpu_prep_python(config: AppConfig, paths: ProjectPaths, splat_path: Path, *, is_sample: bool) -> dict[str, Any]:
    rows = read_table(splat_path)
    voxel_acc: dict[tuple[int, int, int, int], dict[str, Any]] = {}
    offsets = _offsets(is_sample)
    for row in rows:
        x = float(row.get("x_m", 0.0) or 0.0)
        y = float(row.get("y_m", 0.0) or 0.0)
        z = float(row.get("z_m", 0.0) or 0.0)
        sx = max(float(row.get("sigma_x_m", 5000.0) or 5000.0), 1.0)
        sy = max(float(row.get("sigma_y_m", 5000.0) or 5000.0), 1.0)
        sz = max(float(row.get("sigma_z_m", 5000.0) or 5000.0), 1.0)
        amp = float(row.get("amplitude", 0.0) or 0.0)
        opacity = float(row.get("opacity", 0.5) or 0.5)
        primitive_type = str(row.get("primitive_type", "residual") or "residual")
        event_id = str(row.get("event_id", ""))
        for lod, size in _lod_sizes(is_sample):
            for dx, dy, dz in offsets:
                ix = math.floor((x + dx * size) / size)
                iy = math.floor((y + dy * size) / size)
                iz = math.floor((z + dz * size) / size)
                contribution = math.exp(
                    -0.5 * (((dx * size) / sx) ** 2 + ((dy * size) / sy) ** 2 + ((dz * size) / sz) ** 2)
                ) * amp * opacity
                if contribution <= 1.0e-8:
                    continue
                key = (lod, ix, iy, iz)
                acc = voxel_acc.setdefault(
                    key,
                    {
                        "lod": lod,
                        "voxel_size_m": size,
                        "ix": ix,
                        "iy": iy,
                        "iz": iz,
                        "density": 0.0,
                        "weighted_amplitude_sum": 0.0,
                        "weighted_opacity_sum": 0.0,
                        "direct_density": 0.0,
                        "reflected_density": 0.0,
                        "scattered_density": 0.0,
                        "residual_density": 0.0,
                        "expanded_contribution_count": 0,
                        "events": set(),
                    },
                )
                acc["density"] += contribution
                acc["weighted_amplitude_sum"] += contribution * amp
                acc["weighted_opacity_sum"] += contribution * opacity
                acc[f"{primitive_type}_density" if primitive_type in {"direct", "reflected", "scattered", "residual"} else "residual_density"] += contribution
                acc["expanded_contribution_count"] += 1
                if event_id:
                    acc["events"].add(event_id)
    voxel_rows = []
    shard_acc: dict[tuple[int, int, int, int], dict[str, Any]] = defaultdict(lambda: {"voxel_count": 0, "density_sum": 0.0, "event_count": 0})
    for acc in voxel_acc.values():
        density = float(acc["density"])
        size = float(acc["voxel_size_m"])
        row = {
            "lod": acc["lod"],
            "voxel_size_m": size,
            "ix": acc["ix"],
            "iy": acc["iy"],
            "iz": acc["iz"],
            "center_x_m": (acc["ix"] + 0.5) * size,
            "center_y_m": (acc["iy"] + 0.5) * size,
            "center_z_m": (acc["iz"] + 0.5) * size,
            "density": density,
            "weighted_amplitude": acc["weighted_amplitude_sum"] / max(density, 1.0e-30),
            "weighted_opacity": acc["weighted_opacity_sum"] / max(density, 1.0e-30),
            "direct_density": acc["direct_density"],
            "reflected_density": acc["reflected_density"],
            "scattered_density": acc["scattered_density"],
            "residual_density": acc["residual_density"],
            "expanded_contribution_count": acc["expanded_contribution_count"],
            "n_events": len(acc["events"]),
            "is_sample_data": is_sample,
        }
        voxel_rows.append(row)
        shard_key = (int(row["lod"]), math.floor(int(row["ix"]) / 64), math.floor(int(row["iy"]) / 64), math.floor(int(row["iz"]) / 64))
        shard_acc[shard_key]["voxel_count"] += 1
        shard_acc[shard_key]["density_sum"] += density
        shard_acc[shard_key]["event_count"] += int(row["n_events"])
    shard_rows = [
        {
            "lod": lod,
            "shard_x": sx,
            "shard_y": sy,
            "shard_z": sz,
            "voxel_count": values["voxel_count"],
            "density_sum": values["density_sum"],
            "event_count": values["event_count"],
            "min_ix": 0,
            "max_ix": 0,
            "min_iy": 0,
            "max_iy": 0,
            "min_iz": 0,
            "max_iz": 0,
            "storage_uri": "data/processed/gaussian_splat_voxel_lod.parquet",
        }
        for (lod, sx, sy, sz), values in shard_acc.items()
    ]
    voxel_path = paths.data_processed / "gaussian_splat_voxel_lod.parquet"
    shard_path = paths.data_processed / "gaussian_splat_gpu_shard_index.parquet"
    write_table(voxel_rows, voxel_path, {"physical_format": "parquet_or_csv", "is_sample_data": is_sample, "not_prediction": True})
    write_table(shard_rows, shard_path, {"physical_format": "parquet_or_csv", "is_sample_data": is_sample, "not_prediction": True})
    return {
        "method": "cpu_python_gaussian_splat_voxel_lod_expansion",
        "database_engine": "sqlite_or_python",
        "is_sample_data": is_sample,
        "not_prediction": True,
        "input_splats": len(rows),
        "voxel_rows": len(voxel_rows),
        "shard_rows": len(shard_rows),
        "voxel_path": str(voxel_path),
        "shard_path": str(shard_path),
        "kernel_offsets": len(offsets),
        "view_image_status": "skipped_python_fallback",
        "view_image_rows": 0,
        "view_image_shard_rows": 0,
    }


def _write_manifest(config: AppConfig, paths: ProjectPaths, result: dict[str, Any]) -> None:
    out_dir = paths.root / "outputs" / "gpu_prep"
    out_dir.mkdir(parents=True, exist_ok=True)

    def _relative(value: object, fallback: str) -> str:
        if not value:
            return fallback
        path = Path(str(value))
        try:
            return str(path.resolve().relative_to(paths.root.resolve()))
        except ValueError:
            return str(path)

    view_part_index = _relative(result.get("image_part_index_path"), "data/processed/splat_view_image_part_index.parquet")
    view_partition_dir = _relative(result.get("image_partition_dir"), "data/processed/splat_view_image_parts")
    gpu_handoff = {
        "voxel_lod": _relative(result.get("voxel_path"), "data/processed/gaussian_splat_voxel_lod.parquet"),
        "shard_index": _relative(result.get("shard_path"), "data/processed/gaussian_splat_gpu_shard_index.parquet"),
        "view_image_part_index": view_part_index,
        "view_image_partitions": f"{view_partition_dir}/part-*.parquet",
        "primitive_source": "data/processed/gaussian_splat_primitive.parquet",
    }
    if result.get("view_image_status") != "partitioned":
        gpu_handoff["view_image_pixels"] = "data/processed/splat_view_image_pixel.parquet"

    manifest = {
        **result,
        "region": config.region.name,
        "crs_local": config.region.crs_local,
        "coordinate_convention": "x/y local CRS meters; z_m positive downward; GPU may convert z display separately",
        "gpu_handoff": gpu_handoff,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")


def _write_report(paths: ProjectPaths, result: dict[str, Any]) -> None:
    report = paths.outputs_reports / "gpu_prep.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        "\n".join(
            [
                "# CPU pre-GPU preparation",
                "",
                "This output prepares compact Gaussian splat primitives for later GPU fusion. It is not an earthquake prediction.",
                "",
                f"- Method: `{result.get('method')}`",
                f"- Input splats: `{result.get('input_splats')}`",
                f"- Voxel rows: `{result.get('voxel_rows')}`",
                f"- Shard rows: `{result.get('shard_rows')}`",
                f"- Kernel offsets: `{result.get('kernel_offsets')}`",
                f"- View image rows: `{result.get('view_image_rows')}`",
                f"- View image partitions: `{result.get('view_image_part_rows')}`",
                f"- View count: `{result.get('view_count')}`",
                f"- View pixel size m: `{result.get('pixel_size_m')}`",
                f"- View kernel radius: `{result.get("view_image_kernel_radius")}`",
                f"- View image part index: `{result.get("image_part_index_path")}`",
                f"- View image partition dir: `{result.get("image_partition_dir")}`",
                f"- Voxel table: `{result.get('voxel_path')}`",
                f"- Shard index: `{result.get('shard_path')}`",
                "",
                "The voxel density is a relative waveform-derived state indicator. It is not a unique subsurface inversion.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
