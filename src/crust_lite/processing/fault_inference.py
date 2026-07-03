from __future__ import annotations

import math
import os
from collections import defaultdict, deque
from typing import Any

import numpy as np

from crust_lite.config import AppConfig
from crust_lite.data_sources.active_faults import build_known_fault_reference_layers
from crust_lite.geo import (
    LocalProjector,
    angle_difference_deg,
    clamp01,
    distance_to_polyline_km,
    vector_to_strike_dip,
)
from crust_lite.io.geopackage import read_features, write_features
from crust_lite.io.parquet import read_table
from crust_lite.logging import get_logger
from crust_lite.paths import ProjectPaths
from crust_lite.processing.scoring import confidence_from_score, fault_score
from crust_lite.processing.shallow_lineaments import build_shallow_lineaments

LOGGER = get_logger(__name__)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int, min_value: int, max_value: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return max(min_value, min(max_value, int(raw)))
    except ValueError:
        LOGGER.warning("Ignoring invalid integer environment value %s=%r", name, raw)
        return default


def _env_float(name: str, default: float, min_value: float, max_value: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return max(min_value, min(max_value, float(raw)))
    except ValueError:
        LOGGER.warning("Ignoring invalid float environment value %s=%r", name, raw)
        return default


def _cluster_points(points: np.ndarray, eps_m: float, min_samples: int) -> np.ndarray:
    try:
        from sklearn.cluster import DBSCAN  # type: ignore

        return DBSCAN(eps=eps_m, min_samples=min_samples).fit_predict(points)
    except Exception:
        return _grid_cluster_points(points, eps_m=eps_m, min_samples=min_samples)


def _grid_cluster_points(points: np.ndarray, eps_m: float, min_samples: int) -> np.ndarray:
    """Approximate DBSCAN fallback using occupied 3D grid-cell connectivity."""
    if len(points) < min_samples:
        return np.full(len(points), -1, dtype=int)
    origin = np.min(points, axis=0)
    cell_size = max(float(eps_m), 1.0)
    cell_index = np.floor((points - origin) / cell_size).astype(int)
    cells: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    for idx, cell in enumerate(cell_index):
        cells[(int(cell[0]), int(cell[1]), int(cell[2]))].append(idx)

    labels = np.full(len(points), -1, dtype=int)
    visited: set[tuple[int, int, int]] = set()
    cluster_id = 0
    neighbor_offsets = [
        (dx, dy, dz)
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dz in (-1, 0, 1)
        if not (dx == 0 and dy == 0 and dz == 0)
    ]
    for start_cell in cells:
        if start_cell in visited:
            continue
        queue: deque[tuple[int, int, int]] = deque([start_cell])
        visited.add(start_cell)
        component_indices: list[int] = []
        while queue:
            cell = queue.popleft()
            component_indices.extend(cells[cell])
            for offset in neighbor_offsets:
                neighbor = (cell[0] + offset[0], cell[1] + offset[1], cell[2] + offset[2])
                if neighbor in cells and neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        if len(component_indices) >= min_samples:
            labels[np.array(component_indices, dtype=int)] = cluster_id
            cluster_id += 1
    return labels


def _tile_cluster_points(
    points: np.ndarray,
    tile_m: float,
    depth_bin_m: float,
    min_samples: int,
    offset_xy_m: tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    """Split broad connected seismicity into local search tiles.

    The offset pass intentionally repeats the search on shifted grids so narrow
    event alignments that fall across a tile boundary are still tested by PCA.
    """
    origin = np.min(points, axis=0) - np.array([offset_xy_m[0], offset_xy_m[1], 0.0])
    labels = np.full(len(points), -1, dtype=int)
    bins: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    for idx, point in enumerate(points):
        key = (
            int(math.floor((point[0] - origin[0]) / max(tile_m, 1.0))),
            int(math.floor((point[1] - origin[1]) / max(tile_m, 1.0))),
            int(math.floor((point[2] - origin[2]) / max(depth_bin_m, 1.0))),
        )
        bins[key].append(idx)
    cluster_id = 0
    for indices in bins.values():
        if len(indices) < min_samples:
            continue
        labels[np.array(indices, dtype=int)] = cluster_id
        cluster_id += 1
    return labels


def _labels_to_cluster_sets(
    labels: np.ndarray,
    method: str,
    min_events: int,
) -> list[tuple[str, list[int], str]]:
    clusters: dict[int, list[int]] = defaultdict(list)
    for idx, label in enumerate(labels):
        label_int = int(label)
        if label_int >= 0:
            clusters[label_int].append(idx)
    return [
        (f"{method}_{cluster_id:04d}", indices, method)
        for cluster_id, indices in clusters.items()
        if len(indices) >= min_events
    ]


def _candidate_cluster_sets(points: np.ndarray, eps_m: float) -> list[tuple[str, list[int], str]]:
    """Return global and local candidate event groups for PCA fault fitting.

    A national or whole-region catalog often forms one connected DBSCAN cloud.
    Running PCA on that cloud suppresses local active-fault-scale alignments, so
    the MVP combines a global DBSCAN pass with overlapping local search tiles.
    The candidates are deduplicated after scoring.
    """
    min_events = 4
    candidates: list[tuple[str, list[int], str]] = []
    global_labels = _cluster_points(points, eps_m=eps_m, min_samples=min_events)
    candidates.extend(_labels_to_cluster_sets(global_labels, "global_dbscan", min_events))

    if len(points) < 200:
        return candidates

    tile_specs = [
        (80_000.0, 15_000.0, 5, "local80_depth15"),
        (50_000.0, 10_000.0, 5, "local50_depth10"),
        (30_000.0, 8_000.0, 4, "local30_depth8"),
    ]
    if _env_bool("CRUST_LITE_FAULT_HIGH_RES", True):
        tile_specs.extend(
            [
                (20_000.0, 6_000.0, 4, "local20_depth6"),
                (15_000.0, 5_000.0, 4, "local15_depth5"),
            ]
        )
    for tile_m, depth_bin_m, tile_min_events, name in tile_specs:
        offsets = [
            (0.0, 0.0),
            (tile_m / 2.0, tile_m / 2.0),
            (tile_m / 2.0, 0.0),
            (0.0, tile_m / 2.0),
        ]
        for offset_idx, offset in enumerate(offsets):
            labels = _tile_cluster_points(
                points,
                tile_m=tile_m,
                depth_bin_m=depth_bin_m,
                min_samples=tile_min_events,
                offset_xy_m=offset,
            )
            candidates.extend(
                _labels_to_cluster_sets(labels, f"{name}_offset{offset_idx}", tile_min_events)
            )
    return candidates


def _strike_difference_deg(a: float, b: float) -> float:
    diff = abs((a - b + 180.0) % 360.0 - 180.0)
    return min(diff, abs(diff - 180.0))


def _is_regional_sheet(props: dict[str, Any]) -> bool:
    return float(props.get("length_km", 0.0)) > 800.0 or float(props.get("width_km", 0.0)) > 300.0


def _dedupe_fault_features(
    features: list[dict[str, Any]],
    max_features: int = 300,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Keep the highest-scoring local candidates while removing near duplicates."""
    stats = {
        "raw_candidate_count": len(features),
        "regional_sheet_removed_count": 0,
        "duplicate_removed_count": 0,
    }
    if not features:
        return [], stats

    local_features = []
    regional_features = []
    for feature in features:
        props = feature.get("properties", {})
        if _is_regional_sheet(props):
            regional_features.append(feature)
        else:
            local_features.append(feature)
    if local_features:
        features = local_features
        stats["regional_sheet_removed_count"] = len(regional_features)

    def rank(feature: dict[str, Any]) -> tuple[float, float, float]:
        props = feature.get("properties", {})
        return (
            float(props.get("fault_score", 0.0)),
            float(props.get("seismicity_planarity_score", 0.0)),
            float(props.get("n_events", 0.0)),
        )

    distance_threshold_km = _env_float("CRUST_LITE_FAULT_DEDUPE_DISTANCE_KM", 6.0, 1.0, 50.0)
    depth_threshold_km = _env_float("CRUST_LITE_FAULT_DEDUPE_DEPTH_KM", 5.0, 1.0, 30.0)
    strike_threshold_deg = _env_float("CRUST_LITE_FAULT_DEDUPE_STRIKE_DEG", 20.0, 1.0, 90.0)
    stats.update(
        {
            "dedupe_distance_km": int(round(distance_threshold_km)),
            "dedupe_depth_km": int(round(depth_threshold_km)),
            "dedupe_strike_deg": int(round(strike_threshold_deg)),
        }
    )

    kept: list[dict[str, Any]] = []
    for feature in sorted(features, key=rank, reverse=True):
        props = feature.get("properties", {})
        center = (float(props.get("center_x_m", 0.0)), float(props.get("center_y_m", 0.0)))
        depth = float(props.get("center_depth_km", 0.0))
        strike = float(props.get("strike", 0.0))
        duplicate = False
        for kept_feature in kept:
            kept_props = kept_feature.get("properties", {})
            kept_center = (
                float(kept_props.get("center_x_m", 0.0)),
                float(kept_props.get("center_y_m", 0.0)),
            )
            distance_km = (
                math.hypot(center[0] - kept_center[0], center[1] - kept_center[1])
                / 1000.0
            )
            depth_diff_km = abs(depth - float(kept_props.get("center_depth_km", 0.0)))
            strike_diff = _strike_difference_deg(strike, float(kept_props.get("strike", 0.0)))
            if (
                distance_km <= distance_threshold_km
                and depth_diff_km <= depth_threshold_km
                and strike_diff <= strike_threshold_deg
            ):
                duplicate = True
                break
        if duplicate:
            stats["duplicate_removed_count"] += 1
            continue
        kept.append(feature)
        if len(kept) >= max_features:
            break

    for idx, feature in enumerate(kept):
        props = feature.setdefault("properties", {})
        props["raw_segment_id"] = props.get("segment_id", "")
        props["segment_id"] = f"inferred_fault_{idx:04d}"
        props["cluster_id"] = idx
        props["sensitivity_mode"] = "high_resolution_multiscale_overlapping_tiles"
    return kept, stats


def _mechanism_score(strike: float, mechanisms: list[dict[str, Any]], event_ids: set[str]) -> float:
    relevant = [row for row in mechanisms if row.get("event_id") in event_ids]
    if not relevant:
        return 0.5
    scores = []
    for row in relevant:
        d1 = angle_difference_deg(strike, float(row["strike1"]))
        d2 = angle_difference_deg(strike, float(row["strike2"]))
        scores.append(1.0 - min(d1, d2, 90.0) / 90.0)
    return clamp01(float(np.mean(scores)))


def _gnss_score(center: tuple[float, float], gnss_rows: list[dict[str, Any]]) -> float:
    if not gnss_rows:
        return 0.5
    weighted = []
    for row in gnss_rows:
        dx = center[0] - float(row["x_m"])
        dy = center[1] - float(row["y_m"])
        dist_km = max(1.0, math.hypot(dx, dy) / 1000.0)
        weighted.append(float(row.get("strain_gradient_score", 0.5)) / dist_km)
    return clamp01(float(np.mean(weighted)) * 10.0)


def _known_fault_distance_score(
    center: tuple[float, float],
    known_features: list[dict[str, Any]],
    projector: LocalProjector,
    candidate_strike: float | None = None,
) -> tuple[float, float, str, float | None, str]:
    if not known_features:
        return 0.5, float("inf"), "no_known_faults_loaded", None, "no_known_fault_feedback"
    best_distance = float("inf")
    nearest = ""
    best_strike_diff: float | None = None
    for feature in known_features:
        props = feature.get("properties", {}) if isinstance(feature.get("properties"), dict) else {}
        for line in _feature_lines_xy(feature, projector):
            if len(line) < 2:
                continue
            distance = distance_to_polyline_km(center, line)
            if distance >= best_distance:
                continue
            nearest = str(props.get("segment_id", "known_fault"))
            best_distance = distance
            known_strike = _line_strike_deg(line)
            best_strike_diff = (
                _axial_strike_difference(candidate_strike, known_strike)
                if candidate_strike is not None and known_strike is not None
                else None
            )
    score = _known_fault_feedback_score(best_distance, best_strike_diff)
    feedback_status = _known_fault_feedback_status(best_distance, best_strike_diff)
    return score, best_distance, nearest, best_strike_diff, feedback_status


def _feature_lines_xy(feature: dict[str, Any], projector: LocalProjector) -> list[list[tuple[float, float]]]:
    geom = feature.get("geometry", {}) if isinstance(feature.get("geometry"), dict) else {}
    local_lines = geom.get("local_trace_lines_m")
    if local_lines:
        return [[(float(x), float(y)) for x, y in line] for line in local_lines if len(line) >= 2]
    local = geom.get("local_trace_m")
    if local:
        return [[(float(x), float(y)) for x, y in line_or_points] for line_or_points in _normalize_local_lines(local)]
    geom_type = geom.get("type")
    if geom_type == "LineString":
        return [projector.line_lonlat_to_xy(geom.get("coordinates", []))]
    if geom_type == "MultiLineString":
        return [projector.line_lonlat_to_xy(line) for line in geom.get("coordinates", [])]
    return []


def _normalize_local_lines(local: Any) -> list[list[list[float]]]:
    if not isinstance(local, list) or not local:
        return []
    first = local[0]
    if isinstance(first, list) and len(first) >= 2 and isinstance(first[0], (int, float)):
        return [local]
    return [line for line in local if isinstance(line, list)]


def _line_strike_deg(line: list[tuple[float, float]]) -> float | None:
    if len(line) < 2:
        return None
    x0, y0 = line[0]
    x1, y1 = line[-1]
    return (math.degrees(math.atan2(x1 - x0, y1 - y0)) + 360.0) % 360.0


def _axial_strike_difference(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    diff = abs((float(a) - float(b) + 180.0) % 360.0 - 180.0)
    return min(diff, abs(diff - 180.0))


def _known_fault_feedback_score(distance_km: float, strike_diff_deg: float | None) -> float:
    if not math.isfinite(distance_km):
        return 0.5
    distance_score = clamp01(1.0 - min(distance_km, 30.0) / 30.0)
    if strike_diff_deg is None:
        return distance_score
    strike_score = clamp01(1.0 - min(strike_diff_deg, 90.0) / 90.0)
    return clamp01(0.70 * distance_score + 0.30 * strike_score)


def _known_fault_feedback_status(distance_km: float, strike_diff_deg: float | None) -> str:
    if not math.isfinite(distance_km):
        return "no_known_fault_geometry"
    if distance_km <= 5.0 and strike_diff_deg is not None and strike_diff_deg <= 25.0:
        return "known_trace_aligned"
    if distance_km <= 5.0:
        return "near_known_trace_but_oblique"
    if distance_km <= 20.0 and strike_diff_deg is not None and strike_diff_deg <= 25.0:
        return "parallel_offset_from_known_trace"
    if distance_km <= 30.0:
        return "regional_proximity_without_trace_match"
    return "no_near_known_fault_within_threshold"


def _lineament_support_score(row: dict[str, Any]) -> float:
    n_support = max(0.0, float(row.get("n_support", 0.0)))
    frequency_count = max(1.0, float(row.get("frequency_count", row.get("spectral_support_count", 1.0))))
    return clamp01(0.70 * math.log1p(n_support) / math.log1p(80.0) + 0.30 * math.log1p(frequency_count) / math.log1p(8.0))


def _lineament_fault_score(row: dict[str, Any], known_score: float) -> float:
    linearity = clamp01(float(row.get("linearity_score", 0.0)))
    surface = clamp01(float(row.get("surface_wave_anomaly_score", 0.0)))
    scattering = clamp01(float(row.get("scattering_lineament_score", 0.0)))
    support = _lineament_support_score(row)
    return clamp01(0.26 * linearity + 0.28 * surface + 0.22 * scattering + 0.10 * support + 0.14 * known_score)


def _lineament_width_km(row: dict[str, Any]) -> float:
    depth_width = max(1.0, float(row.get("depth_p95_km", 0.0)) - float(row.get("depth_p05_km", 0.0)))
    return max(2.0, min(30.0, depth_width + 2.0))


def _lineament_fault_features(
    rows: list[dict[str, Any]],
    known_features: list[dict[str, Any]],
    projector: LocalProjector,
    max_features: int,
) -> list[dict[str, Any]]:
    features: list[dict[str, Any]] = []
    ranked = sorted(
        rows,
        key=lambda row: (
            float(row.get("confidence", 0.0)),
            float(row.get("frequency_count", row.get("spectral_support_count", 1.0))),
            float(row.get("length_km", 0.0)),
        ),
        reverse=True,
    )[:max_features]
    for idx, row in enumerate(ranked):
        center = (float(row.get("center_x_m", 0.0)), float(row.get("center_y_m", 0.0)))
        candidate_strike = float(row.get("strike", 0.0))
        known_score, known_distance_km, nearest_known, known_strike_diff, feedback_status = (
            _known_fault_distance_score(
                center,
                known_features,
                projector,
                candidate_strike=candidate_strike,
            )
        )
        score = _lineament_fault_score(row, known_score)
        confidence = clamp01(max(float(row.get("confidence", 0.0)), confidence_from_score(score, int(row.get("n_support", 0) or 0))))
        width_km = _lineament_width_km(row)
        segment_id = f"inferred_fault_sa_{idx:04d}"
        depth_p05 = max(0.0, float(row.get("depth_p05_km", row.get("center_depth_km", 0.0))))
        depth_p95 = max(depth_p05, float(row.get("depth_p95_km", row.get("center_depth_km", depth_p05))))
        props = {
            "segment_id": segment_id,
            "source": "synthetic_aperture_shallow_lineament",
            "source_table": row.get("source_table", "shallow_lineament"),
            "lineament_id": row.get("lineament_id", ""),
            "strike": candidate_strike,
            "dip": 75.0,
            "rake": 0.0,
            "length_km": max(0.5, float(row.get("length_km", 0.5))),
            "width_km": width_km,
            "top_depth_km": depth_p05,
            "bottom_depth_km": depth_p95,
            "center_depth_km": max(0.0, float(row.get("center_depth_km", 0.0))),
            "center_x_m": center[0],
            "center_y_m": center[1],
            "is_inferred": True,
            "cluster_id": row.get("lineament_id", idx),
            "raw_cluster_id": row.get("group_id", ""),
            "inference_method": "synthetic_aperture_frequency_preserving_lineament",
            "n_events": int(row.get("event_support_count", 0) or 0),
            "n_support": int(row.get("n_support", 0) or 0),
            "spectral_support_count": int(row.get("spectral_support_count", 1) or 1),
            "band_count": int(row.get("band_count", row.get("frequency_count", 1)) or 1),
            "frequency_count": int(row.get("frequency_count", 1) or 1),
            "frequency_hz": row.get("frequency_hz", ""),
            "frequency_band": row.get("frequency_band", ""),
            "frequencies_hz": row.get("frequencies_hz", ""),
            "seismicity_planarity_score": clamp01(float(row.get("linearity_score", 0.0))),
            "synthetic_aperture_linearity_score": clamp01(float(row.get("linearity_score", 0.0))),
            "surface_wave_anomaly_score": clamp01(float(row.get("surface_wave_anomaly_score", 0.0))),
            "scattering_lineament_score": clamp01(float(row.get("scattering_lineament_score", 0.0))),
            "mechanism_consistency_score": 0.5,
            "gnss_strain_gradient_score": 0.5,
            "waveform_residual_score": clamp01(max(float(row.get("surface_wave_anomaly_score", 0.0)), float(row.get("scattering_lineament_score", 0.0)))),
            "distance_from_known_fault_score": known_score,
            "distance_to_known_fault_km": known_distance_km,
            "nearest_known_segment_id": nearest_known,
            "strike_difference_to_known_deg": known_strike_diff,
            "known_fault_feedback_status": feedback_status,
            "known_fault_feedback_weight": known_score,
            "fault_score": score,
            "confidence": confidence,
            "notes": (
                f"synthetic_aperture_lineament; nearest_known={nearest_known}; "
                f"known_feedback={feedback_status}; "
                "frequency-resolved source retained in shallow_lineament_spectral.parquet"
            ),
            "is_sample_data": bool(row.get("is_sample_data", False)),
        }
        features.append(
            {
                "type": "Feature",
                "properties": props,
                "geometry": {
                    "type": "LineString",
                    "coordinates": [],
                    "local_trace_m": [
                        [float(row.get("x0_m", center[0])), float(row.get("y0_m", center[1]))],
                        [float(row.get("x1_m", center[0])), float(row.get("y1_m", center[1]))],
                    ],
                },
            }
        )
    return features


def _infer_faults_from_synthetic_aperture_lineaments(config: AppConfig, paths: ProjectPaths) -> dict[str, Any] | None:
    lineament_path = paths.data_processed / "shallow_lineament.parquet"
    known_path = paths.data_processed / "fault_segment.gpkg"
    known_features = read_features(known_path) if known_path.exists() else []
    if not lineament_path.exists():
        build_shallow_lineaments(config, paths)
    rows = read_table(lineament_path) if lineament_path.exists() else []
    require_synthetic = bool(getattr(config.shallow_lineaments, "require_synthetic_aperture_source", False))
    if not rows:
        if require_synthetic:
            raise ValueError(
                "Synthetic-aperture shallow lineaments are required for inferred faults; run array-projection and shallow-lineaments first"
            )
        return None
    projector = LocalProjector(config.region)
    max_features = _env_int("CRUST_LITE_FAULT_MAX_FEATURES", config.shallow_lineaments.max_lineaments, 1, 5000)
    features = _lineament_fault_features(rows, known_features, projector, max_features=max_features)
    if not features:
        if require_synthetic:
            raise ValueError("No inferred faults were produced from synthetic-aperture shallow lineaments")
        return None
    is_sample = any(bool(feature["properties"].get("is_sample_data")) for feature in features)
    write_features(
        features,
        paths.data_processed / "inferred_faults.gpkg",
        {
            "is_sample_data": is_sample,
            "cluster_count": len(features),
            "method": "synthetic_aperture_shallow_lineament_to_fault_candidates",
            "source_table": str(lineament_path),
            "frequency_resolved_source": str(paths.data_processed / "shallow_lineament_spectral.parquet"),
            "requires_synthetic_aperture_source": require_synthetic,
            "not_prediction": True,
            "max_features": max_features,
        },
    )
    known_reference = build_known_fault_reference_layers(config, paths)
    LOGGER.info("Inferred %d synthetic-aperture fault candidates", len(features))
    return {
        "inferred_fault_count": len(features),
        "is_sample_data": is_sample,
        "method": "synthetic_aperture_shallow_lineament_to_fault_candidates",
        "source_table": str(lineament_path),
        "known_fault_reference_layers": known_reference,
    }


def infer_faults(config: AppConfig, paths: ProjectPaths) -> dict[str, Any]:
    if getattr(getattr(config, "shallow_lineaments", None), "enabled", False):
        lineament_result = _infer_faults_from_synthetic_aperture_lineaments(config, paths)
        if lineament_result is not None:
            return lineament_result

    event_qc_path = paths.data_interim / "event_qc.parquet"
    events = read_table(event_qc_path)
    mechanisms = read_table(paths.data_processed / "mechanism.parquet") if (
        paths.data_processed / "mechanism.parquet"
    ).exists() else []
    gnss_rows = read_table(paths.data_processed / "gnss_features.parquet") if (
        paths.data_processed / "gnss_features.parquet"
    ).exists() else []
    known_path = paths.data_processed / "fault_segment.gpkg"
    known_features = read_features(known_path) if known_path.exists() else []
    if len(events) < 4:
        raise ValueError("At least four events are required for fault inference")

    points = np.array(
        [[float(row["x_m"]), float(row["y_m"]), float(row["z_m"])] for row in events],
        dtype=float,
    )
    horizontal_span = max(np.ptp(points[:, 0]), np.ptp(points[:, 1]), 1.0)
    eps_m = min(50_000.0, max(8_000.0, horizontal_span / 20.0))
    candidate_clusters = _candidate_cluster_sets(points, eps_m=eps_m)
    LOGGER.info(
        "Prepared %d raw candidate event groups for multiscale fault inference",
        len(candidate_clusters),
    )
    projector = LocalProjector(config.region)
    features: list[dict[str, Any]] = []
    for raw_cluster_id, indices, inference_method in candidate_clusters:
        if len(indices) < 4:
            continue
        cluster_points = points[indices]
        center = np.mean(cluster_points, axis=0)
        centered = cluster_points - center
        cov = np.cov(centered.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        order = np.argsort(eigenvalues)
        normal = eigenvectors[:, order[0]]
        strike, dip = vector_to_strike_dip(normal)
        along = eigenvectors[:, order[-1]]
        across = eigenvectors[:, order[1]]
        length_km = max(1.0, 4.0 * float(np.std(centered @ along)) / 1000.0)
        width_km = max(1.0, 4.0 * float(np.std(centered @ across)) / 1000.0)
        center_depth_km = max(0.0, float(center[2]) / 1000.0)
        eig_sum = max(1e-9, float(np.sum(eigenvalues)))
        planarity = clamp01(1.0 - float(eigenvalues[order[0]]) / eig_sum)
        event_ids = {str(events[idx]["event_id"]) for idx in indices}
        mech_score = _mechanism_score(strike, mechanisms, event_ids)
        gnss_score = _gnss_score((float(center[0]), float(center[1])), gnss_rows)
        known_score, known_distance_km, nearest_known, known_strike_diff, feedback_status = (
            _known_fault_distance_score(
                (float(center[0]), float(center[1])),
                known_features,
                projector,
                candidate_strike=strike,
            )
        )
        wave_score = 0.5
        score = fault_score(planarity, mech_score, gnss_score, wave_score, known_score)
        confidence = confidence_from_score(score, len(indices))
        segment_id = f"inferred_raw_{len(features):04d}"
        props = {
            "segment_id": segment_id,
            "source": "seismicity_pca_multiscale",
            "strike": strike,
            "dip": dip,
            "rake": -170.0 if mech_score >= 0.5 else 0.0,
            "length_km": length_km,
            "width_km": width_km,
            "top_depth_km": max(
                0.0,
                center_depth_km - width_km * math.sin(math.radians(dip)) / 2.0,
            ),
            "bottom_depth_km": center_depth_km + width_km * math.sin(math.radians(dip)) / 2.0,
            "center_depth_km": center_depth_km,
            "center_x_m": float(center[0]),
            "center_y_m": float(center[1]),
            "is_inferred": True,
            "cluster_id": raw_cluster_id,
            "raw_cluster_id": raw_cluster_id,
            "inference_method": inference_method,
            "n_events": len(indices),
            "seismicity_planarity_score": planarity,
            "mechanism_consistency_score": mech_score,
            "gnss_strain_gradient_score": gnss_score,
            "waveform_residual_score": wave_score,
            "distance_from_known_fault_score": known_score,
            "distance_to_known_fault_km": known_distance_km,
            "nearest_known_segment_id": nearest_known,
            "strike_difference_to_known_deg": known_strike_diff,
            "known_fault_feedback_status": feedback_status,
            "known_fault_feedback_weight": known_score,
            "fault_score": score,
            "confidence": confidence,
            "notes": (
                f"known_fault_extension_candidate; nearest_known={nearest_known}; feedback={feedback_status}"
                if feedback_status == "known_trace_aligned"
                else f"unregistered_or_offset_candidate; nearest_known={nearest_known}; feedback={feedback_status}"
            ),
            "is_sample_data": any(
                str(events[idx].get("is_sample_data", "")).lower() == "true" for idx in indices
            ),
        }
        # Geometry remains a 2D trace in lon/lat-like space for portability of
        # the fallback GeoJSON content. The 3D renderer uses center/strike/dip.
        strike_rad = math.radians(strike)
        half_len_m = length_km * 500.0
        dx = math.sin(strike_rad) * half_len_m
        dy = math.cos(strike_rad) * half_len_m
        lon0, lat0 = float(events[indices[0]]["lon"]), float(events[indices[0]]["lat"])
        lon1, lat1 = float(events[indices[-1]]["lon"]), float(events[indices[-1]]["lat"])
        features.append(
            {
                "type": "Feature",
                "properties": props,
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[lon0, lat0], [lon1, lat1]],
                    "local_trace_m": [
                        [float(center[0] - dx), float(center[1] - dy)],
                        [float(center[0] + dx), float(center[1] + dy)],
                    ],
                },
            }
        )
    if not features:
        raise ValueError("No candidate fault clusters were inferred")
    max_features = _env_int("CRUST_LITE_FAULT_MAX_FEATURES", 1000, 1, 5000)
    features, selection_stats = _dedupe_fault_features(features, max_features=max_features)
    if not features:
        raise ValueError(
            "No local candidate fault clusters remained after regional-sheet filtering"
        )
    is_sample = any(bool(feature["properties"].get("is_sample_data")) for feature in features)
    write_features(
        features,
        paths.data_processed / "inferred_faults.gpkg",
        {
            "is_sample_data": is_sample,
            "cluster_count": len(features),
            "method": "legacy_multiscale_dbscan_tile_pca_disabled_by_default",
            "cluster_eps_m": eps_m,
            "sensitivity_mode": "global_dbscan_plus_80km_50km_30km_20km_15km_overlapping_tiles",
            "max_features": max_features,
            **selection_stats,
        },
    )
    known_reference = build_known_fault_reference_layers(config, paths)
    LOGGER.info(
        "Inferred %d candidate fault segments from %d raw candidates",
        len(features),
        selection_stats["raw_candidate_count"],
    )
    return {
        "inferred_fault_count": len(features),
        "is_sample_data": is_sample,
        "known_fault_reference_layers": known_reference,
        **selection_stats,
    }
