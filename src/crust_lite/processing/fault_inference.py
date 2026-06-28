from __future__ import annotations

import math
from collections import defaultdict, deque
from typing import Any

import numpy as np

from crust_lite.config import AppConfig
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

LOGGER = get_logger(__name__)


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
) -> np.ndarray:
    """Split broad connected seismicity into local search tiles."""
    origin = np.min(points, axis=0)
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
) -> tuple[float, float, str]:
    if not known_features:
        return 0.5, float("inf"), "no_known_faults_loaded"
    distances = []
    nearest = ""
    for feature in known_features:
        geom = feature.get("geometry", {})
        if geom.get("type") != "LineString":
            continue
        line = projector.line_lonlat_to_xy(geom.get("coordinates", []))
        distance = distance_to_polyline_km(center, line)
        distances.append(distance)
        if distance == min(distances):
            nearest = str(feature.get("properties", {}).get("segment_id", "known_fault"))
    min_distance = min(distances) if distances else float("inf")
    score = clamp01(1.0 - min(min_distance, 30.0) / 30.0)
    return score, min_distance, nearest


def infer_faults(config: AppConfig, paths: ProjectPaths) -> dict[str, Any]:
    events = read_table(paths.data_interim / "event_qc.parquet")
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
    labels = _cluster_points(points, eps_m=eps_m, min_samples=4)
    positive_labels = {int(label) for label in labels if int(label) >= 0}
    if len(points) > 1000 and len(positive_labels) <= 1:
        labels = _tile_cluster_points(points, tile_m=120_000.0, depth_bin_m=20_000.0, min_samples=20)
        LOGGER.info(
            "Split broad single cluster into %d local search tiles",
            len({int(label) for label in labels if int(label) >= 0}),
        )
    clusters: dict[int, list[int]] = defaultdict(list)
    for idx, label in enumerate(labels):
        if int(label) >= 0:
            clusters[int(label)].append(idx)
    projector = LocalProjector(config.region)
    features: list[dict[str, Any]] = []
    for cluster_id, indices in clusters.items():
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
        known_score, known_distance_km, nearest_known = _known_fault_distance_score(
            (float(center[0]), float(center[1])), known_features, projector
        )
        wave_score = 0.5
        score = fault_score(planarity, mech_score, gnss_score, wave_score, known_score)
        confidence = confidence_from_score(score, len(indices))
        segment_id = f"inferred_cluster_{cluster_id:03d}"
        props = {
            "segment_id": segment_id,
            "source": "seismicity_pca_dbscan",
            "strike": strike,
            "dip": dip,
            "rake": -170.0 if mech_score >= 0.5 else 0.0,
            "length_km": length_km,
            "width_km": width_km,
            "top_depth_km": max(0.0, center_depth_km - width_km * math.sin(math.radians(dip)) / 2.0),
            "bottom_depth_km": center_depth_km + width_km * math.sin(math.radians(dip)) / 2.0,
            "center_depth_km": center_depth_km,
            "center_x_m": float(center[0]),
            "center_y_m": float(center[1]),
            "is_inferred": True,
            "cluster_id": cluster_id,
            "n_events": len(indices),
            "seismicity_planarity_score": planarity,
            "mechanism_consistency_score": mech_score,
            "gnss_strain_gradient_score": gnss_score,
            "waveform_residual_score": wave_score,
            "distance_from_known_fault_score": known_score,
            "distance_to_known_fault_km": known_distance_km,
            "fault_score": score,
            "confidence": confidence,
            "notes": (
                "known_fault_extension_candidate"
                if known_distance_km <= 5.0
                else f"unregistered_candidate; nearest_known={nearest_known}"
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
    is_sample = any(bool(feature["properties"].get("is_sample_data")) for feature in features)
    write_features(
        features,
        paths.data_processed / "inferred_faults.gpkg",
        {
            "is_sample_data": is_sample,
            "cluster_count": len(features),
            "method": "DBSCAN/PCA with sklearn when available, grid-cell clustering fallback otherwise",
            "cluster_eps_m": eps_m,
            "large_catalog_split": "tile_120km_depth20km_if_single_cluster",
        },
    )
    LOGGER.info("Inferred %d candidate fault segments", len(features))
    return {"inferred_fault_count": len(features), "is_sample_data": is_sample}
