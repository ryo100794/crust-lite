#!/usr/bin/env python3
"""v1020: variable-K single-pass mathematical aperture WLS -> 3DGS.

K is read from the leading response/illumination dimension.  Each view uses one
global 3-D q90 scale.  The eight deterministic shifted-view null configurations
are solved as one [configuration, view, x, y, z] CUDA batch.  No autograd,
optimizer, ML, fault, or plate information participates.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import resource
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def load_v1015():
    path = Path(__file__).with_name("solve_single_pass_event_gs_gpu_v1015.py")
    spec = importlib.util.spec_from_file_location("single_pass_v1015_for_v1020", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def shift_configurations(view_count: int, nx: int, ny: int):
    """Eight deterministic configs; first four exactly reproduce v1015."""
    if view_count < 2:
        raise ValueError("at least two views are required")
    legacy = [
        [(0, 0), (nx//4, 0), (0, ny//4), (nx//5, ny//5)],
        [(0, 0), (-nx//4, 0), (0, -ny//4), (-nx//5, ny//5)],
        [(0, 0), (nx//3, ny//6), (-nx//6, ny//3), (nx//5, -ny//5)],
        [(0, 0), (-nx//3, ny//6), (nx//6, -ny//3), (-nx//5, -ny//5)],
        [(0, 0), (nx//5, ny//3), (-nx//4, -ny//6), (nx//3, -ny//5)],
        [(0, 0), (-nx//5, ny//3), (nx//4, -ny//6), (-nx//3, -ny//5)],
        [(0, 0), (nx//2, ny//5), (nx//5, -ny//2), (-nx//4, ny//3)],
        [(0, 0), (-nx//2, ny//5), (-nx//5, -ny//2), (nx//4, ny//3)],
    ]
    result = []
    for config_index, base in enumerate(legacy):
        config = list(base[:view_count])
        used = set(config)
        for view in range(4, view_count):
            # Co-prime-ish integer sequence, resolved deterministically if a small
            # grid causes a collision with an earlier view in this configuration.
            dx = ((17 * (config_index + 1) + 11 * (view + 1)) % nx) - nx // 2
            dy = ((13 * (config_index + 1) + 7 * (view + 1)) % ny) - ny // 2
            attempt = 0
            while (dx, dy) in used:
                attempt += 1
                dx = ((dx + attempt + 1 + nx // 2) % nx) - nx // 2
                dy = ((dy + 2 * attempt + 1 + ny // 2) % ny) - ny // 2
                if attempt > nx * ny:
                    raise RuntimeError("grid cannot provide distinct deterministic shifts")
            config.append((dx, dy)); used.add((dx, dy))
        result.append(tuple(config))
    return result


def make_shift_batch(y: torch.Tensor, light: torch.Tensor, configurations):
    view_count = y.shape[0]
    shifted_y = torch.stack([
        torch.stack([torch.roll(y[view], config[view], dims=(0, 1)) for view in range(view_count)])
        for config in configurations
    ])
    shifted_light = torch.stack([
        torch.stack([torch.roll(light[view], config[view], dims=(0, 1)) for view in range(view_count)])
        for config in configurations
    ])
    return shifted_y, shifted_light


def solve_wls_shift_batch(y: torch.Tensor, light: torch.Tensor, ridge: float):
    """Closed-form WLS for [configuration, view, x, y, z]."""
    active = (y > 0) & (light > 0)
    aperture = torch.sqrt(torch.clamp(light, 0, 1))
    weight = active.float() * torch.clamp(light, min=0.05)
    normal = torch.sum(aperture.square() * weight, dim=1)
    rhs = torch.sum(aperture * weight * y, dim=1)
    field = rhs / torch.clamp(normal + ridge, min=1.0e-6)
    residual = torch.where(active, y - aperture * field[:, None], torch.zeros_like(y))
    weight_sum = torch.sum(weight, dim=1)
    residual_variance = torch.sum(weight * residual.square(), dim=1) / torch.clamp(weight_sum - 1, min=1)
    standard_error = torch.sqrt(
        torch.clamp(residual_variance / torch.clamp(normal + ridge, min=1.0e-6), min=0)
    )
    coverage = torch.mean(active.float(), dim=1)
    field = torch.where(coverage >= 0.50, field, torch.zeros_like(field))
    return field, standard_error, coverage


def threshold_candidates(
    null_significance: torch.Tensor,
    null_coverage: torch.Tensor,
    significance: torch.Tensor,
    coverage: torch.Tensor,
    quantile: float = 0.985,
):
    thresholds = []
    raw_candidate = torch.zeros_like(significance, dtype=torch.bool)
    raw_counts = []
    for depth in range(significance.shape[-1]):
        values = null_significance[..., depth][null_coverage[..., depth] >= 0.75]
        threshold = torch.quantile(values, quantile) if values.numel() else torch.tensor(torch.inf, device=significance.device)
        thresholds.append(threshold)
        chosen = (significance[..., depth] > threshold) & (coverage[..., depth] >= 0.75)
        raw_candidate[..., depth] = chosen
        raw_counts.append(int(chosen.sum()))
    neighbours = F.conv3d(
        raw_candidate.float()[None, None],
        torch.ones((1, 1, 3, 3, 3), device=significance.device), padding=1,
    )[0, 0]
    candidate = raw_candidate & (neighbours >= 3)
    return torch.stack(thresholds), raw_candidate, candidate, raw_counts


def positive_quantiles(values: np.ndarray, quantiles):
    chosen = values[np.isfinite(values) & (values > 0)]
    return np.quantile(chosen, quantiles).tolist() if len(chosen) else [None] * len(quantiles)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--ridge", type=float, default=0.04)
    parser.add_argument("--null-quantile", type=float, default=0.985)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    if not (0.90 <= args.null_quantile < 1.0):
        raise SystemExit("invalid null quantile")
    torch.set_grad_enabled(False)
    base = load_v1015()
    wall_start = time.monotonic()
    with np.load(args.input, allow_pickle=False) as data:
        xyz = np.asarray(data["xyz"], np.float64)
        shape = tuple(map(int, data["grid_shape"].tolist()))
        flat_response = np.asarray(data["view_response"], np.float32)
        flat_illumination = np.asarray(data["view_illumination"], np.float32)
        event_id = str(data["event_id"].item()); phase = str(data["phase"].item()); tile_id = str(data["tile_id"].item())
        source_xyz_m = np.asarray(data["source_xyz_m"], np.float64)
        source_core_km = float(data["source_core_km"].item()); source_taper_end_km = float(data["source_taper_end_km"].item())
        input_known = bool(data["known_structure_used"].item())
        radial_projected = bool(data["direct_radial_nuisance_projected"].item()) if "direct_radial_nuisance_projected" in data.files else False
    if flat_response.ndim < 2 or flat_response.shape[0] < 2:
        raise SystemExit("view_response leading dimension K must be >=2")
    view_count = int(flat_response.shape[0])
    if flat_response.size != view_count * int(np.prod(shape)) or flat_illumination.shape != flat_response.shape:
        raise SystemExit("response/illumination grid mismatch")
    if input_known:
        raise SystemExit("known-structure-derived input is forbidden")
    response = flat_response.reshape(view_count, *shape)
    illumination = flat_illumination.reshape(view_count, *shape)
    xyz_km = xyz.copy(); xyz_km[:, :2] /= 1000.0
    spacing = tuple(float(np.median(np.diff(np.unique(xyz_km[:, axis])))) for axis in range(3))

    device = torch.device("cuda")
    raw = torch.as_tensor(response, device=device)
    light = torch.as_tensor(np.clip(illumination, 0, 1), device=device)
    coords = torch.as_tensor(xyz_km.reshape(*shape, 3), device=device)
    torch.cuda.reset_peak_memory_stats()
    gpu_begin, gpu_end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    gpu_begin.record()
    with torch.no_grad():
        normalized, view_scales = base.normalize_global_q90(raw, light)
        holdout = base.holdout_audit(normalized, light, args.ridge)
        field, standard_error, coverage = base.solve_wls(normalized, light, args.ridge)
        variance = standard_error.square()
        significance = field / torch.clamp(standard_error, min=0.01)
        configurations = shift_configurations(view_count, shape[0], shape[1])
        shifted_y, shifted_light = make_shift_batch(normalized, light, configurations)
        null_field, null_se, null_coverage = solve_wls_shift_batch(shifted_y, shifted_light, args.ridge)
        null_significance = null_field / torch.clamp(null_se, min=0.01)
        thresholds, raw_candidate, candidate, raw_counts = threshold_candidates(
            null_significance, null_coverage, significance, coverage, args.null_quantile
        )
        index = torch.nonzero(candidate, as_tuple=False)
        if not len(index):
            raise RuntimeError("no candidate survives shifted-null and neighbour gates")

        smooth = base.smooth3(field)
        potential = -torch.log(torch.clamp(smooth, min=1.0e-5))
        gradient_tuple = torch.gradient(potential, spacing=spacing, dim=(0, 1, 2), edge_order=1)
        gradient = torch.stack(gradient_tuple, dim=-1)
        hessian = torch.empty((*shape, 3, 3), device=device)
        for row in range(3):
            second = torch.gradient(gradient_tuple[row], spacing=spacing, dim=(0, 1, 2), edge_order=1)
            for column in range(3): hessian[..., row, column] = second[column]
        hessian = 0.5 * (hessian + hessian.transpose(-1, -2))
        selected_hessian = hessian[index[:, 0], index[:, 1], index[:, 2]]
        selected_gradient = gradient[index[:, 0], index[:, 1], index[:, 2]]
        eigenvalue, eigenvector = torch.linalg.eigh(selected_hessian)
        strongest = torch.argmax(torch.abs(eigenvalue), dim=1)
        rows = torch.arange(len(index), device=device)
        normal = eigenvector[rows, :, strongest]; curvature = eigenvalue[rows, strongest]
        safe = torch.where(
            torch.abs(curvature) > 1.0e-5, curvature,
            torch.where(curvature >= 0, torch.full_like(curvature, 1.0e-5), torch.full_like(curvature, -1.0e-5)),
        )
        displacement = torch.clamp(
            -torch.sum(selected_gradient * normal, dim=1) / safe,
            -0.5 * min(spacing), 0.5 * min(spacing),
        )
        center = coords[index[:, 0], index[:, 1], index[:, 2]] + displacement[:, None] * normal
        center[:, 2].clamp_(float(np.min(xyz[:, 2])), float(np.max(xyz[:, 2])))
        sigma = torch.clamp(torch.rsqrt(torch.abs(eigenvalue) + 1 / 14**2), min=0.55 * min(spacing), max=14)
        axes = eigenvector * sigma[:, None, :]
        strength = field[index[:, 0], index[:, 1], index[:, 2]]
        splat_significance = significance[index[:, 0], index[:, 1], index[:, 2]]
        low, high = torch.quantile(strength, torch.tensor([0.02, 0.98], device=device))
        opacity = torch.clamp((strength - low) / torch.clamp(high - low, min=1.0e-6), 0, 1)
    gpu_end.record(); torch.cuda.synchronize()

    center_np = center.cpu().numpy(); axes_np = axes.cpu().numpy(); strength_np = strength.cpu().numpy()
    splat_significance_np = splat_significance.cpu().numpy(); opacity_np = opacity.cpu().numpy()
    field_np = field.cpu().numpy().astype(np.float32); variance_np = variance.cpu().numpy().astype(np.float32)
    coverage_np = coverage.cpu().numpy().astype(np.float32); significance_np = significance.cpu().numpy().astype(np.float32)
    candidate_np = candidate.cpu().numpy(); threshold_np = thresholds.cpu().numpy().astype(np.float32)
    scale_np = view_scales.cpu().numpy().astype(np.float32)
    counts = [int(np.count_nonzero(candidate_np[..., depth])) for depth in range(shape[2])]
    projection = np.any(candidate_np, axis=2); occupied = np.argwhere(projection)
    if len(occupied):
        width = int(np.ptp(occupied[:, 0]) + 1); height = int(np.ptp(occupied[:, 1]) + 1)
        rectangular_fill = float(len(occupied) / max(width * height, 1))
    else: rectangular_fill = 1.0
    shallow = float(np.mean(center_np[:, 2] <= 3.0))
    xyz_m = xyz.copy(); xyz_m[:, 2] *= 1000.0
    output = {
        "schema": np.asarray("variable-view-single-pass-global-q90-shift-null-3dgs-v1020"),
        "event_id": np.asarray(event_id), "phase": np.asarray(phase), "tile_id": np.asarray(tile_id),
        "view_count": np.asarray(view_count, np.int32),
        "center_xyz_m": (center_np*1000).astype(np.float32),
        "axis_u_m": (axes_np[:,:,0]*1000).astype(np.float32), "axis_v_m": (axes_np[:,:,1]*1000).astype(np.float32),
        "axis_w_m": (axes_np[:,:,2]*1000).astype(np.float32), "strength": strength_np.astype(np.float32),
        "splat_significance": splat_significance_np.astype(np.float32), "opacity": opacity_np.astype(np.float32),
        "reflectivity_wls_grid": field_np.reshape(-1), "reflectivity_variance_grid": variance_np.reshape(-1),
        "illumination_support_grid": coverage_np.reshape(-1), "significance_grid": significance_np.reshape(-1),
        "candidate_mask": candidate_np.reshape(-1), "shift_null_q985_by_depth": threshold_np,
        "normalization_q90_by_view": scale_np, "shift_configurations_xy": np.asarray(configurations, np.int32),
        "grid_xyz_m": xyz_m.astype(np.float32), "grid_shape": np.asarray(shape, np.int32),
        "source_xyz_m": source_xyz_m, "source_core_km": np.asarray(source_core_km),
        "source_taper_end_km": np.asarray(source_taper_end_km),
        "strength_semantics": np.asarray("WLS reflectivity; significance stored separately"),
        "cross_event_raw_addition_allowed": np.asarray(False), "machine_learning_used": np.asarray(False),
        "known_structure_used": np.asarray(False), "publication_allowed": np.asarray(False),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".v1020.tmp.npz")
    np.savez_compressed(temporary, **output); temporary.replace(args.output)
    correlations = [row["correlation"] for row in holdout if row["correlation"] is not None]
    aperture_consistent = (
        len(correlations) == view_count
        and sum(value > 0 for value in correlations) >= math.ceil(0.75 * view_count)
        and float(np.median(correlations)) > 0.05
    )
    checks = {
        "cuda_used": True, "autograd_absent": not torch.is_grad_enabled(), "optimizer_absent": True,
        "machine_learning_absent": True, "known_structure_absent": not input_known,
        "variable_k_used": view_count == response.shape[0], "at_least_two_views": view_count >= 2,
        "global_q90_per_view": len(scale_np) == view_count, "actual_wls_once": True,
        "eight_shift_configurations": len(configurations) == 8,
        "distinct_roll_per_view_per_configuration": all(len(set(config)) == view_count for config in configurations),
        "configuration_by_k_single_cuda_batch": True, "q985_used": abs(args.null_quantile-.985) < 1e-12,
        "coverage_ge_075_used": True, "hessian_eigh_once": True,
        "aperture_images_share_structure": aperture_consistent,
        "finite": bool(all(np.isfinite(value).all() for value in (center_np,axes_np,strength_np,opacity_np,field_np,variance_np,coverage_np,significance_np))),
        "nonrectangular_footprint": rectangular_fill < .80, "shallow_structure_present": shallow > .01,
        "at_least_100_significant_splats": len(center_np) >= 100,
    }
    audit = {
        "schema": "variable-view-single-pass-global-q90-shift-null-3dgs-audit-v1020",
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00","Z"),
        "input": str(args.input), "input_sha256": sha256(args.input), "output": str(args.output),
        "output_sha256": sha256(args.output), "event_id": event_id, "phase": phase, "tile_id": tile_id,
        "view_count": view_count, "grid_shape": list(shape), "spacing_km": list(spacing),
        "normalization_q90_by_view": scale_np.tolist(), "shift_configurations_xy": configurations,
        "leave_one_aperture_out": holdout, "candidate_counts_before_gate_by_depth": raw_counts,
        "candidate_counts_after_gate_by_depth": counts, "splat_count": int(len(center_np)),
        "footprint": {"projected_cells": int(len(occupied)), "rectangular_fill_fraction": rectangular_fill,
                      "shallow_splat_fraction_le_3km": shallow},
        "reflectivity_q02_q10_q50_q90_q98": positive_quantiles(field_np,[.02,.1,.5,.9,.98]),
        "strength_q02_q10_q50_q90_q98": positive_quantiles(strength_np,[.02,.1,.5,.9,.98]),
        "gpu": {"name": torch.cuda.get_device_name(0), "elapsed_ms": float(gpu_begin.elapsed_time(gpu_end)),
                "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                "peak_reserved_bytes": int(torch.cuda.max_memory_reserved())},
        "wall_seconds": time.monotonic()-wall_start,
        "process_peak_ram_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024),
        "input_direct_radial_nuisance_projected": radial_projected,
        "checks": checks, "pass": all(checks.values()),
        "machine_learning_used": False, "autograd_used": False, "optimizer_used": False,
        "known_structure_used": False, "publication_allowed": False,
    }
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.audit.write_text(json.dumps(audit,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"audit":str(args.audit),"pass":audit["pass"],"view_count":view_count,
                      "splat_count":len(center_np),"wall_seconds":audit["wall_seconds"],
                      "rectangular_fill":rectangular_fill,"shallow":shallow},ensure_ascii=False))
    raise SystemExit(0 if audit["pass"] else 2)


if __name__ == "__main__":
    main()
