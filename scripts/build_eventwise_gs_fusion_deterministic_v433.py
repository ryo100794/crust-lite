#!/usr/bin/env python3
"""Build eventwise GS with deterministic, halo-stable Gaussian geometry."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

import national_depth_consumer_guard_v1144 as depth_guard


R = 6_378_137.0
PHASES = ("P", "S")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_evaluator(path: Path):
    spec = importlib.util.spec_from_file_location("eventwise_v433_regions", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def bit_count(values: np.ndarray) -> np.ndarray:
    table = np.asarray([int(value).bit_count() for value in range(256)], np.uint8)
    return table[values]


def event_sources(events_path: Path, event_ids: list[str]) -> np.ndarray:
    frame = pd.read_parquet(events_path)
    frame["event_id"] = frame.event_id.astype(str)
    frame = frame.drop_duplicates("event_id").set_index("event_id").loc[event_ids]
    return frame[["x_m", "y_m"]].to_numpy(float)


def p95_scale(values: np.ndarray, active: np.ndarray) -> float:
    selected = values[active]
    return float(np.quantile(selected, 0.95)) if len(selected) else 0.0


def normalize(values: np.ndarray, scale: float) -> np.ndarray:
    if scale <= 0:
        return np.zeros_like(values, dtype=np.float32)
    return np.clip(values / scale, 0.0, 1.0).astype(np.float32)


def local_depth_spacing(depths_km: np.ndarray) -> dict[float, float]:
    unique = np.unique(depths_km)
    result = {}
    for index, value in enumerate(unique):
        neighbors = []
        if index:
            neighbors.append(value - unique[index - 1])
        if index + 1 < len(unique):
            neighbors.append(unique[index + 1] - value)
        result[float(value)] = float(min(neighbors)) * 1000.0
    return result


def fused_geometry(
    xyz_km: np.ndarray,
    selected: np.ndarray,
    intensity: np.ndarray,
    horizontal_spacing_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    physical = np.column_stack((xyz_km[:, 0], xyz_km[:, 1], xyz_km[:, 2] * 1000.0))
    centers = physical[selected].copy()
    sigma_u = 0.85 * horizontal_spacing_m
    axes_u = np.zeros((len(selected), 3), np.float32)
    axes_v = np.zeros((len(selected), 3), np.float32)
    shifts = np.zeros((len(selected), 3), np.float32)
    if not len(selected):
        return centers.astype(np.float32), axes_u, axes_v, shifts
    tree = cKDTree(physical[selected])
    neighbors = min(20, len(selected))
    distance, indices = tree.query(physical[selected], k=neighbors, workers=1)
    if neighbors == 1:
        distance = distance[:, None]
        indices = indices[:, None]
    depth_spacing = local_depth_spacing(xyz_km[:, 2])
    for row, node in enumerate(selected):
        # A regular grid has many nodes tied at the kth distance. cKDTree may
        # return a different tied subset when the same core is queried inside
        # a halo tile. Include the entire kth-distance shell, then break ties
        # by physical x/y/z so the non-tiled and tiled inputs choose identically.
        kth_distance = float(distance[row, -1])
        radius = kth_distance + max(1.0e-6, kth_distance * 1.0e-12)
        candidate_local = np.asarray(tree.query_ball_point(physical[node], r=radius), np.int64)
        candidate_global = selected[candidate_local]
        candidate_delta = physical[candidate_global] - physical[node]
        candidate_distance_squared = np.einsum("ni,ni->n", candidate_delta, candidate_delta)
        candidate_physical = physical[candidate_global]
        deterministic_order = np.lexsort((
            candidate_physical[:, 2],
            candidate_physical[:, 1],
            candidate_physical[:, 0],
            candidate_distance_squared,
        ))
        neighbor_global = candidate_global[deterministic_order[:neighbors]]
        delta = physical[neighbor_global] - physical[node]
        selected_distance = np.sqrt(np.einsum("ni,ni->n", delta, delta))
        weight = (0.05 + intensity[neighbor_global]) ** 2
        weight *= np.exp(-0.5 * (selected_distance / max(2.0 * horizontal_spacing_m, 1.0)) ** 2)
        weight_sum = max(float(weight.sum()), 1.0e-12)
        centroid = np.sum(delta * weight[:, None], axis=0) / weight_sum
        horizontal = float(np.linalg.norm(centroid[:2]))
        horizontal_max = 0.35 * horizontal_spacing_m
        if horizontal > horizontal_max:
            centroid[:2] *= horizontal_max / horizontal
        vertical_max = 0.35 * depth_spacing[float(xyz_km[node, 2])]
        centroid[2] = np.clip(centroid[2], -vertical_max, vertical_max)
        centered = delta - centroid
        covariance = np.einsum("n,ni,nj->ij", weight, centered, centered) / weight_sum
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        major = eigenvectors[:, 2]
        middle = eigenvectors[:, 1]
        ratio = float(np.clip(
            np.sqrt(max(float(eigenvalues[1]), 1.0) / max(float(eigenvalues[2]), 1.0)),
            0.65,
            1.0,
        ))
        centers[row] += centroid
        shifts[row] = centroid
        axes_u[row] = major * sigma_u
        axes_v[row] = middle * (sigma_u * ratio)
    return centers.astype(np.float32), axes_u, axes_v, shifts


def quantiles(values: np.ndarray) -> list[float]:
    if not len(values):
        return [0.0] * 5
    return list(map(float, np.quantile(values, [0.01, 0.10, 0.50, 0.90, 0.99])))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cube", type=Path, required=True)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--evaluator", type=Path, required=True)
    parser.add_argument("--region", required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()

    depth_guard.enforce_consumer(Path(__file__).resolve().parents[1], "dense-gs-reference", args.contract)

    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    if contract.get("known_structure_used") is not False:
        raise RuntimeError("contract does not prohibit known structures")
    evaluator = load_evaluator(args.evaluator)
    lon0, lon1, lat0, lat1 = evaluator.REGIONS[args.region]
    with np.load(args.cube, allow_pickle=False) as cube:
        event_ids = list(map(str, cube["event_ids"].tolist()))
        xyz = np.asarray(cube["xyz"], float)
        arrays = {name: np.asarray(cube[name], np.float32) for phase in PHASES for name in (
            phase,
            f"{phase}_illumination",
            f"{phase}_view_a",
            f"{phase}_view_b",
            f"{phase}_view_a_illumination",
            f"{phase}_view_b_illumination",
        )}
    sources = event_sources(args.events, event_ids)
    lon = np.degrees(xyz[:, 0] / R)
    lat = np.degrees(np.arctan(np.sinh(xyz[:, 1] / R)))
    region_mask = (lon >= lon0) & (lon <= lon1) & (lat >= lat0) & (lat <= lat1)
    region_indices = np.flatnonzero(region_mask)
    region_xyz = xyz[region_indices]
    x_delta = np.diff(np.unique(xyz[:, 0]))
    y_delta = np.diff(np.unique(xyz[:, 1]))
    horizontal_spacing_m = float(min(np.min(x_delta), np.min(y_delta)))

    output = {
        "schema": np.asarray("eventwise-gs-fusion-deterministic-v433"),
        "known_structure_used": np.asarray(False),
        "event_ids": np.asarray(event_ids),
        "region_node_indices": region_indices.astype(np.uint32),
        "region_xyz": region_xyz.astype(np.float64),
        "horizontal_spacing_m": np.asarray(horizontal_spacing_m),
    }
    phase_audits = {}
    for phase in PHASES:
        response = arrays[phase]
        illumination = arrays[f"{phase}_illumination"]
        view_a = arrays[f"{phase}_view_a"]
        view_b = arrays[f"{phase}_view_b"]
        illumination_a = arrays[f"{phase}_view_a_illumination"]
        illumination_b = arrays[f"{phase}_view_b_illumination"]
        offsets = [0]
        node_parts = []
        intensity_parts = []
        reliability_parts = []
        event_scales = []
        fused_sum = np.zeros(len(region_indices), np.float64)
        view_sum = np.zeros(len(region_indices), np.float64)
        exposure_count = np.zeros(len(region_indices), np.uint16)
        detection_count = np.zeros(len(region_indices), np.uint16)
        sectors = np.zeros(len(region_indices), np.uint8)
        for event_index, (source_x, source_y) in enumerate(sources):
            exposed_all = illumination[event_index] > 0.05
            detected_all = (response[event_index] > 1.0e-15) & exposed_all
            scale = p95_scale(response[event_index], detected_all)
            scale_a = p95_scale(
                view_a[event_index],
                (view_a[event_index] > 1.0e-15) & (illumination_a[event_index] > 0.05),
            )
            scale_b = p95_scale(
                view_b[event_index],
                (view_b[event_index] > 1.0e-15) & (illumination_b[event_index] > 0.05),
            )
            event_scales.append([scale, scale_a, scale_b])
            norm = normalize(response[event_index], scale)[region_indices]
            norm_a = normalize(view_a[event_index], scale_a)[region_indices]
            norm_b = normalize(view_b[event_index], scale_b)[region_indices]
            norm_a *= illumination_a[event_index, region_indices] > 0.05
            norm_b *= illumination_b[event_index, region_indices] > 0.05
            reliability = np.sqrt(norm_a * norm_b).astype(np.float32)
            exposed = exposed_all[region_indices]
            detected = detected_all[region_indices]
            local_nodes = np.flatnonzero(detected)
            node_parts.append(local_nodes.astype(np.uint32))
            intensity_parts.append(norm[local_nodes])
            reliability_parts.append(reliability[local_nodes])
            offsets.append(offsets[-1] + len(local_nodes))
            fused_sum += norm * detected
            view_sum += reliability * detected
            exposure_count += exposed
            detection_count += detected
            azimuth = (np.degrees(np.arctan2(source_x - region_xyz[:, 0], source_y - region_xyz[:, 1])) + 360.0) % 360.0
            sector = np.floor(azimuth / 45.0).astype(np.uint8)
            sectors[detected] |= (1 << sector[detected]).astype(np.uint8)
        fused = np.divide(
            fused_sum,
            exposure_count,
            out=np.zeros(len(region_indices), np.float64),
            where=exposure_count > 0,
        )
        view_reliability = np.divide(
            view_sum,
            detection_count,
            out=np.zeros(len(region_indices), np.float64),
            where=detection_count > 0,
        )
        directions = bit_count(sectors)
        eligible = (exposure_count >= 2) & (detection_count >= 2) & (directions >= 2) & (fused > 0)
        selected = np.flatnonzero(eligible)
        centers, axes_u, axes_v, shifts = fused_geometry(
            region_xyz, selected, fused, horizontal_spacing_m
        )
        major = np.linalg.norm(axes_u, axis=1)
        minor = np.linalg.norm(axes_v, axis=1)
        output[f"{phase}_event_offsets"] = np.asarray(offsets, np.uint64)
        output[f"{phase}_event_node_index"] = np.concatenate(node_parts) if node_parts else np.empty(0, np.uint32)
        output[f"{phase}_event_intensity"] = np.concatenate(intensity_parts) if intensity_parts else np.empty(0, np.float32)
        output[f"{phase}_event_view_reliability"] = np.concatenate(reliability_parts) if reliability_parts else np.empty(0, np.float32)
        output[f"{phase}_fused_node_index"] = selected.astype(np.uint32)
        output[f"{phase}_fused_center_xyz_m"] = centers
        output[f"{phase}_fused_axis_u_m"] = axes_u
        output[f"{phase}_fused_axis_v_m"] = axes_v
        output[f"{phase}_fused_intensity"] = fused[selected].astype(np.float32)
        output[f"{phase}_fused_view_reliability"] = view_reliability[selected].astype(np.float32)
        output[f"{phase}_exposure_count"] = exposure_count[selected]
        output[f"{phase}_detection_count"] = detection_count[selected]
        output[f"{phase}_direction_sectors"] = directions[selected]
        shallow = region_xyz[selected, 2] <= 5.0
        horizontal_shift = np.linalg.norm(shifts[:, :2], axis=1)
        phase_audits[phase] = {
            "eventwise_splats": int(offsets[-1]),
            "events": len(event_ids),
            "fused_splats": int(len(selected)),
            "fused_shallow_0_5km": int(shallow.sum()),
            "event_full_view_a_view_b_p95_scale_min_median_max": {
                name: [float(np.min(np.asarray(event_scales)[:, index])), float(np.median(np.asarray(event_scales)[:, index])), float(np.max(np.asarray(event_scales)[:, index]))]
                for index, name in enumerate(("full", "view_a", "view_b"))
            },
            "fused_intensity_q01_q10_q50_q90_q99": quantiles(fused[selected]),
            "fused_view_reliability_q01_q10_q50_q90_q99": quantiles(view_reliability[selected]),
            "event_detection_count_q01_q10_q50_q90_q99": quantiles(detection_count[selected]),
            "source_direction_sectors_q01_q10_q50_q90_q99": quantiles(directions[selected]),
            "sigma_u_over_grid_q01_q10_q50_q90_q99": quantiles(major / horizontal_spacing_m),
            "sigma_v_over_grid_q01_q10_q50_q90_q99": quantiles(minor / horizontal_spacing_m),
            "horizontal_subgrid_shift_over_grid_q01_q10_q50_q90_q99": quantiles(horizontal_shift / horizontal_spacing_m),
            "vertical_subgrid_shift_m_q01_q10_q50_q90_q99": quantiles(np.abs(shifts[:, 2])),
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp.npz")
    np.savez_compressed(temporary, **output)
    temporary.replace(args.output)
    payload = {
        "schema": "eventwise-gs-fusion-deterministic-audit-v433",
        "known_structure_used": False,
        "cube": str(args.cube),
        "cube_sha256": sha256(args.cube),
        "events": str(args.events),
        "events_sha256": sha256(args.events),
        "contract": str(args.contract),
        "contract_sha256": sha256(args.contract),
        "region": args.region,
        "horizontal_spacing_m": horizontal_spacing_m,
        "region_nodes": int(len(region_indices)),
        "phases": phase_audits,
        "output": str(args.output),
        "output_sha256": sha256(args.output),
        "publication_allowed": False,
    }
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.audit.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
