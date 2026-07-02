from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from crust_lite.config import AppConfig
from crust_lite.geo import clamp01
from crust_lite.io.parquet import read_table, write_table
from crust_lite.logging import get_logger
from crust_lite.paths import ProjectPaths

LOGGER = get_logger(__name__)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return default
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value in (None, ""):
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _strike_from_vector(vector: np.ndarray) -> float:
    return math.degrees(math.atan2(float(vector[0]), float(vector[1]))) % 360.0


def _strike_difference_deg(a: float, b: float) -> float:
    diff = abs((a - b + 180.0) % 360.0 - 180.0)
    return min(diff, abs(diff - 180.0))


def _source_path(paths: ProjectPaths, config: AppConfig) -> tuple[Path | None, str]:
    splat = paths.data_processed / "gaussian_splat_primitive.parquet"
    projection = paths.data_processed / "waveform_array_projection.parquet"
    if config.shallow_lineaments.prefer_waveform_splats and splat.exists():
        return splat, "gaussian_splat_primitive"
    if projection.exists():
        return projection, "waveform_array_projection"
    return None, "synthetic_aperture_source_missing"


def _duckdb_limited_rows(path: Path, source_table: str, config: AppConfig) -> list[dict[str, Any]] | None:
    limit = int(config.shallow_lineaments.max_input_rows)
    max_depth_m = float(config.shallow_lineaments.max_depth_km) * 1000.0
    try:
        import duckdb  # type: ignore

        con = duckdb.connect(database=":memory:")
        if source_table == "gaussian_splat_primitive":
            sql = """
                SELECT * FROM read_parquet(?)
                WHERE TRY_CAST(z_m AS DOUBLE) <= ?
                  AND lower(CAST(COALESCE(splat_role, '') AS VARCHAR)) = 'structure'
                ORDER BY COALESCE(TRY_CAST(structure_amplitude AS DOUBLE), 0.0) DESC,
                         COALESCE(TRY_CAST(array_coherence AS DOUBLE), 0.0) DESC
                LIMIT ?
            """
            return [dict(row) for row in con.execute(sql, [str(path), max_depth_m, limit]).fetch_df().to_dict(orient="records")]
        if source_table == "waveform_array_projection":
            sql = """
                SELECT * FROM read_parquet(?)
                WHERE TRY_CAST(projection_z_m AS DOUBLE) <= ?
                  AND lower(CAST(COALESCE(primitive_type, '') AS VARCHAR)) <> 'direct'
                ORDER BY COALESCE(TRY_CAST(beam_power AS DOUBLE), 0.0) DESC,
                         COALESCE(TRY_CAST(array_coherence AS DOUBLE), 0.0) DESC
                LIMIT ?
            """
            return [dict(row) for row in con.execute(sql, [str(path), max_depth_m, limit]).fetch_df().to_dict(orient="records")]
        return []
    except Exception as exc:
        LOGGER.info("DuckDB shallow lineament prefilter unavailable for %s: %s", path, exc)
        return None


def _limited_rows(path: Path, source_table: str, config: AppConfig) -> list[dict[str, Any]]:
    rows = _duckdb_limited_rows(path, source_table, config)
    if rows is None:
        rows = read_table(path)
    max_depth_km = float(config.shallow_lineaments.max_depth_km)
    limit = int(config.shallow_lineaments.max_input_rows)
    out: list[dict[str, Any]] = []
    for row in rows:
        depth_km = _row_depth_km(row, source_table)
        if depth_km <= max_depth_km:
            out.append(row)
    out.sort(key=lambda row: _row_weight(row, source_table, config), reverse=True)
    return out[:limit]


def _row_depth_km(row: dict[str, Any], source_table: str) -> float:
    if source_table == "waveform_array_projection":
        return _safe_float(row.get("projection_z_m", row.get("z_m", 0.0))) / 1000.0
    return _safe_float(row.get("z_m", row.get("depth_km", 0.0))) / (1000.0 if row.get("z_m") not in (None, "") else 1.0)


def _type_weight(row: dict[str, Any]) -> float:
    primitive = str(row.get("primitive_type", row.get("path_family", "event"))).lower()
    path_family = str(row.get("path_family", primitive)).lower()
    if "reflect" in primitive or "reflect" in path_family:
        return 1.0
    if "scatter" in primitive or "scatter" in path_family:
        return 0.9
    if "residual" in primitive or "residual" in path_family:
        return 0.65
    if "direct" in primitive or "direct" in path_family:
        return 0.15
    return 0.45


def _row_weight(row: dict[str, Any], source_table: str, config: AppConfig) -> float:
    if source_table == "gaussian_splat_primitive":
        amplitude = max(_safe_float(row.get("structure_amplitude")), _safe_float(row.get("raw_amplitude")))
        coherence = max(_safe_float(row.get("array_coherence")), 0.35)
        return clamp01(amplitude) * clamp01(coherence) * _type_weight(row)
    if source_table == "waveform_array_projection":
        amplitude = max(_safe_float(row.get("beam_energy")), _safe_float(row.get("beam_power")) / 10.0)
        coherence = max(_safe_float(row.get("array_coherence")), 0.35)
        structural = max(_safe_float(row.get("structural_weight")), 0.2)
        return clamp01(amplitude) * clamp01(coherence) * clamp01(structural) * _type_weight(row)
    return 0.0


def _support_rows(rows: list[dict[str, Any]], source_table: str, config: AppConfig) -> list[dict[str, Any]]:
    supports: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if source_table == "gaussian_splat_primitive":
            if not _bool_value(row.get("is_structure_candidate", True)):
                continue
            x = _safe_float(row.get("x_m"))
            y = _safe_float(row.get("y_m"))
            z = _safe_float(row.get("z_m"))
            frequency_hz = _safe_float(row.get("source_frequency_hz", row.get("frequency_hz", 0.0)))
            frequency_band = str(row.get("frequency_band", f"{frequency_hz:g} Hz"))
        elif source_table == "waveform_array_projection":
            primitive = str(row.get("primitive_type", "direct")).lower()
            if primitive == "direct":
                continue
            x = _safe_float(row.get("projection_x_m", row.get("x_m")))
            y = _safe_float(row.get("projection_y_m", row.get("y_m")))
            z = _safe_float(row.get("projection_z_m", row.get("z_m")))
            frequency_hz = _safe_float(row.get("frequency_hz", 0.0))
            frequency_band = str(row.get("frequency_band", f"{frequency_hz:g} Hz"))
        else:
            continue
        weight = _row_weight(row, source_table, config)
        if weight <= 0.0:
            continue
        supports.append(
            {
                "support_id": index,
                "x_m": x,
                "y_m": y,
                "z_m": z,
                "depth_km": z / 1000.0,
                "frequency_hz": frequency_hz,
                "frequency_band": frequency_band,
                "weight": weight,
                "scatter_weight": weight * _type_weight(row),
                "primitive_type": str(row.get("primitive_type", "event")),
                "path_family": str(row.get("path_family", row.get("primitive_type", "event"))),
                "event_id": str(row.get("event_id", "")),
                "time_utc": str(row.get("time_utc", "")),
                "is_sample_data": _bool_value(row.get("is_sample_data", False)),
                "source_table": source_table,
            }
        )
    return supports


def _cluster_labels(points: np.ndarray, eps_m: float, min_support: int) -> np.ndarray:
    if len(points) < min_support:
        return np.full(len(points), -1, dtype=int)
    try:
        from sklearn.cluster import DBSCAN  # type: ignore

        return DBSCAN(eps=eps_m, min_samples=min_support).fit_predict(points)
    except Exception:
        origin = np.min(points, axis=0)
        cells: dict[tuple[int, int, int], list[int]] = defaultdict(list)
        for idx, point in enumerate(points):
            cell = tuple(np.floor((point - origin) / max(eps_m, 1.0)).astype(int).tolist())
            cells[(int(cell[0]), int(cell[1]), int(cell[2]))].append(idx)
        labels = np.full(len(points), -1, dtype=int)
        cluster_id = 0
        for indices in cells.values():
            if len(indices) >= min_support:
                labels[np.asarray(indices, dtype=int)] = cluster_id
                cluster_id += 1
        return labels


def _candidate_groups(supports: list[dict[str, Any]], config: AppConfig) -> list[tuple[str, list[int]]]:
    if len(supports) < config.shallow_lineaments.min_support:
        return []
    xyz = np.asarray([[row["x_m"], row["y_m"], row["z_m"]] for row in supports], dtype=float)
    labels = _cluster_labels(
        xyz,
        eps_m=float(config.shallow_lineaments.cluster_eps_km) * 1000.0,
        min_support=int(config.shallow_lineaments.min_support),
    )
    groups: list[tuple[str, list[int]]] = []
    for label in sorted(set(int(x) for x in labels if int(x) >= 0)):
        indices = [idx for idx, value in enumerate(labels) if int(value) == label]
        if len(indices) >= config.shallow_lineaments.min_support:
            groups.append((f"dbscan_{label:04d}", indices))

    tile_m = float(config.shallow_lineaments.tile_km) * 1000.0
    depth_m = float(config.shallow_lineaments.tile_depth_km) * 1000.0
    origin = np.min(xyz, axis=0)
    for offset_idx, offset in enumerate(((0.0, 0.0), (0.5 * tile_m, 0.5 * tile_m), (0.5 * tile_m, 0.0), (0.0, 0.5 * tile_m))):
        bins: dict[tuple[int, int, int], list[int]] = defaultdict(list)
        for idx, point in enumerate(xyz):
            key = (
                int(math.floor((point[0] - origin[0] - offset[0]) / max(tile_m, 1.0))),
                int(math.floor((point[1] - origin[1] - offset[1]) / max(tile_m, 1.0))),
                int(math.floor((point[2] - origin[2]) / max(depth_m, 1.0))),
            )
            bins[key].append(idx)
        for tile_idx, indices in enumerate(bins.values()):
            if len(indices) >= config.shallow_lineaments.min_support:
                groups.append((f"tile{offset_idx}_{tile_idx:05d}", indices))
    return groups


def _fit_lineament(
    supports: list[dict[str, Any]],
    indices: list[int],
    group_id: str,
    frequency_hz: float,
    frequency_band: str,
    source_table: str,
    config: AppConfig,
) -> dict[str, Any] | None:
    subset = [supports[idx] for idx in indices]
    if len(subset) < config.shallow_lineaments.min_support:
        return None
    xy = np.asarray([[row["x_m"], row["y_m"]] for row in subset], dtype=float)
    weights = np.asarray([max(float(row["weight"]), 1.0e-6) for row in subset], dtype=float)
    center = np.average(xy, axis=0, weights=weights)
    centered = xy - center
    cov = (centered * weights[:, None]).T @ centered / max(float(np.sum(weights)), 1.0e-12)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    order = np.argsort(eigenvalues)
    small = max(float(eigenvalues[order[0]]), 0.0)
    large = max(float(eigenvalues[order[-1]]), 1.0e-9)
    linearity = clamp01(1.0 - small / large)
    direction = eigenvectors[:, order[-1]]
    along = centered @ direction
    if len(along) >= 4:
        lo, hi = np.percentile(along, [5.0, 95.0])
    else:
        lo, hi = float(np.min(along)), float(np.max(along))
    length_km = max(0.0, float(hi - lo) / 1000.0)
    if length_km < config.shallow_lineaments.min_length_km or linearity < config.shallow_lineaments.min_linearity:
        return None
    x0, y0 = center + direction * lo
    x1, y1 = center + direction * hi
    strike = _strike_from_vector(direction)
    depths = np.asarray([row["depth_km"] for row in subset], dtype=float)
    support_weight = float(np.sum(weights))
    mean_surface = float(np.average([row["weight"] for row in subset], weights=weights))
    mean_scatter = float(np.average([row["scatter_weight"] for row in subset], weights=weights))
    support_score = clamp01(math.log1p(len(subset)) / math.log1p(max(config.shallow_lineaments.min_support * 8, 8)))
    surface_score = clamp01(0.50 * mean_surface + 0.30 * linearity + 0.20 * support_score)
    scatter_score = clamp01(0.55 * mean_scatter + 0.25 * linearity + 0.20 * support_score)
    confidence = clamp01(0.40 * linearity + 0.30 * surface_score + 0.20 * scatter_score + 0.10 * support_score)
    primitive_counts: dict[str, int] = defaultdict(int)
    path_counts: dict[str, int] = defaultdict(int)
    event_ids = {row["event_id"] for row in subset if row.get("event_id")}
    for row in subset:
        primitive_counts[str(row["primitive_type"])] += 1
        path_counts[str(row["path_family"])] += 1
    return {
        "lineament_id": f"shallow_lineament_{source_table}_{frequency_hz:g}_{group_id}",
        "source_table": source_table,
        "method": "frequency_resolved_synthetic_aperture_splat_pca",
        "group_id": group_id,
        "frequency_hz": frequency_hz,
        "frequency_band": frequency_band,
        "x0_m": float(x0),
        "y0_m": float(y0),
        "x1_m": float(x1),
        "y1_m": float(y1),
        "center_x_m": float(center[0]),
        "center_y_m": float(center[1]),
        "center_depth_km": float(np.average(depths, weights=weights)),
        "depth_p05_km": float(np.percentile(depths, 5.0)),
        "depth_p50_km": float(np.percentile(depths, 50.0)),
        "depth_p95_km": float(np.percentile(depths, 95.0)),
        "strike": strike,
        "length_km": length_km,
        "n_support": len(subset),
        "spectral_support_count": len(subset),
        "band_count": 1,
        "weighted_support": support_weight,
        "event_support_count": len(event_ids),
        "linearity_score": linearity,
        "surface_wave_anomaly_score": surface_score,
        "scattering_lineament_score": scatter_score,
        "confidence": confidence,
        "primitive_type_counts": json.dumps(dict(sorted(primitive_counts.items())), sort_keys=True),
        "path_family_counts": json.dumps(dict(sorted(path_counts.items())), sort_keys=True),
        "is_sample_data": any(bool(row.get("is_sample_data")) for row in subset),
        "notes": "synthetic aperture frequency slice; not proof of a unique active fault",
    }


def _spectral_lineaments(supports: list[dict[str, Any]], config: AppConfig) -> list[dict[str, Any]]:
    grouped: dict[tuple[float, str], list[dict[str, Any]]] = defaultdict(list)
    for row in supports:
        grouped[(float(row["frequency_hz"]), str(row["frequency_band"]))].append(row)
    rows: list[dict[str, Any]] = []
    for (frequency_hz, frequency_band), freq_supports in sorted(grouped.items(), key=lambda item: item[0][0]):
        for group_id, indices in _candidate_groups(freq_supports, config):
            lineament = _fit_lineament(
                freq_supports,
                indices,
                group_id,
                frequency_hz,
                frequency_band,
                str(freq_supports[0].get("source_table", "unknown")),
                config,
            )
            if lineament is not None:
                rows.append(lineament)
    rows.sort(key=lambda row: (float(row["confidence"]), float(row["length_km"]), int(row["n_support"])), reverse=True)
    return _dedupe_lineaments(rows, config, same_frequency=True)[: int(config.shallow_lineaments.max_lineaments)]


def _dedupe_lineaments(rows: list[dict[str, Any]], config: AppConfig, same_frequency: bool) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    distance_km = max(2.0, min(15.0, float(config.shallow_lineaments.cluster_eps_km) * 0.6))
    for row in rows:
        duplicate = False
        for kept_row in kept:
            if same_frequency and abs(float(row["frequency_hz"]) - float(kept_row["frequency_hz"])) > 1.0e-9:
                continue
            d_km = math.hypot(float(row["center_x_m"]) - float(kept_row["center_x_m"]), float(row["center_y_m"]) - float(kept_row["center_y_m"])) / 1000.0
            strike_diff = _strike_difference_deg(float(row["strike"]), float(kept_row["strike"]))
            if d_km <= distance_km and strike_diff <= 20.0:
                duplicate = True
                break
        if not duplicate:
            kept.append(row)
    return kept


def _integrated_lineaments(spectral_rows: list[dict[str, Any]], config: AppConfig) -> list[dict[str, Any]]:
    bases = _dedupe_lineaments(spectral_rows, config, same_frequency=False)
    integrated: list[dict[str, Any]] = []
    radius_km = max(3.0, min(20.0, float(config.shallow_lineaments.cluster_eps_km)))
    for idx, base in enumerate(bases[: int(config.shallow_lineaments.max_lineaments)]):
        related = []
        for row in spectral_rows:
            d_km = math.hypot(float(base["center_x_m"]) - float(row["center_x_m"]), float(base["center_y_m"]) - float(row["center_y_m"])) / 1000.0
            if d_km <= radius_km and _strike_difference_deg(float(base["strike"]), float(row["strike"])) <= 25.0:
                related.append(row)
        if not related:
            related = [base]
        weights = np.asarray([max(float(row["confidence"]), 1.0e-6) for row in related], dtype=float)
        center_x = float(np.average([float(row["center_x_m"]) for row in related], weights=weights))
        center_y = float(np.average([float(row["center_y_m"]) for row in related], weights=weights))
        strike = float(np.average([float(row["strike"]) for row in related], weights=weights))
        length = max(float(row["length_km"]) for row in related)
        strike_rad = math.radians(strike)
        half_m = length * 500.0
        dx = math.sin(strike_rad) * half_m
        dy = math.cos(strike_rad) * half_m
        frequencies = sorted({float(row["frequency_hz"]) for row in related})
        row_out = dict(base)
        row_out.update(
            {
                "lineament_id": f"shallow_lineament_integrated_{idx:04d}",
                "method": "spectral_synthetic_aperture_lineament_integration",
                "frequency_hz": -1.0,
                "frequency_band": "integrated_frequency_preserving_summary",
                "x0_m": center_x - dx,
                "y0_m": center_y - dy,
                "x1_m": center_x + dx,
                "y1_m": center_y + dy,
                "center_x_m": center_x,
                "center_y_m": center_y,
                "center_depth_km": float(np.average([float(row["center_depth_km"]) for row in related], weights=weights)),
                "depth_p05_km": min(float(row["depth_p05_km"]) for row in related),
                "depth_p50_km": float(np.average([float(row["depth_p50_km"]) for row in related], weights=weights)),
                "depth_p95_km": max(float(row["depth_p95_km"]) for row in related),
                "strike": strike,
                "length_km": length,
                "n_support": int(sum(int(row["n_support"]) for row in related)),
                "weighted_support": float(sum(float(row["weighted_support"]) for row in related)),
                "spectral_support_count": len(related),
                "band_count": len({str(row["frequency_band"]) for row in related}),
                "frequency_count": len(frequencies),
                "frequencies_hz": json.dumps(frequencies),
                "surface_wave_anomaly_score": clamp01(float(np.average([float(row["surface_wave_anomaly_score"]) for row in related], weights=weights))),
                "scattering_lineament_score": clamp01(float(np.average([float(row["scattering_lineament_score"]) for row in related], weights=weights))),
                "linearity_score": clamp01(float(np.average([float(row["linearity_score"]) for row in related], weights=weights))),
                "confidence": clamp01(max(float(row["confidence"]) for row in related) + 0.05 * math.log1p(len(frequencies))),
                "notes": "integrated summary; frequency-resolved rows are preserved in shallow_lineament_spectral.parquet",
            }
        )
        integrated.append(row_out)
    integrated.sort(key=lambda row: (float(row["confidence"]), float(row["frequency_count"]), float(row["length_km"])), reverse=True)
    return integrated[: int(config.shallow_lineaments.max_lineaments)]


def build_shallow_lineaments(config: AppConfig, paths: ProjectPaths) -> dict[str, Any]:
    paths.ensure()
    spectral_path = paths.data_processed / "shallow_lineament_spectral.parquet"
    integrated_path = paths.data_processed / "shallow_lineament.parquet"
    if not config.shallow_lineaments.enabled:
        metadata = {"enabled": False, "not_prediction": True, "is_sample_data": False}
        write_table([], spectral_path, metadata)
        write_table([], integrated_path, metadata)
        return {"skipped": True, "reason": "shallow_lineaments.enabled=false"}

    source_path, source_table = _source_path(paths, config)
    if source_path is None:
        metadata = {"source_table": source_table, "requires_synthetic_aperture_source": True, "not_prediction": True, "is_sample_data": False, "reason": "no_synthetic_aperture_source"}
        write_table([], spectral_path, metadata)
        write_table([], integrated_path, metadata)
        return {"skipped": True, "reason": "no_synthetic_aperture_source"}

    rows = _limited_rows(source_path, source_table, config)
    supports = _support_rows(rows, source_table, config)
    spectral = _spectral_lineaments(supports, config)
    integrated = _integrated_lineaments(spectral, config)
    is_sample = any(bool(row.get("is_sample_data")) for row in spectral) or any(bool(row.get("is_sample_data")) for row in supports)
    metadata = {
        "source_path": str(source_path),
        "source_table": source_table,
        "input_rows": len(rows),
        "support_rows": len(supports),
        "spectral_lineaments": len(spectral),
        "integrated_lineaments": len(integrated),
        "frequency_resolved_output": str(spectral_path),
        "integrated_output": str(integrated_path),
        "max_depth_km": config.shallow_lineaments.max_depth_km,
        "min_support": config.shallow_lineaments.min_support,
        "cluster_eps_km": config.shallow_lineaments.cluster_eps_km,
        "is_sample_data": is_sample,
        "requires_synthetic_aperture_source": True,
        "not_prediction": True,
        "interpretation": "Relative shallow lineament candidates derived only from synthetic-aperture projection/splat support. Frequency slices are preserved for downstream checks; rows are not earthquake forecasts.",
    }
    write_table(spectral, spectral_path, {**metadata, "table_role": "frequency_resolved"})
    write_table(integrated, integrated_path, {**metadata, "table_role": "integrated_summary"})
    LOGGER.info(
        "Built %d spectral and %d integrated shallow lineaments from %s",
        len(spectral),
        len(integrated),
        source_table,
    )
    return {
        "source_table": source_table,
        "input_rows": len(rows),
        "support_rows": len(supports),
        "spectral_lineament_count": len(spectral),
        "lineament_count": len(integrated),
        "is_sample_data": is_sample,
    }
