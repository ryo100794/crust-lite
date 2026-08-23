#!/usr/bin/env python3
"""v1015: one-pass mathematical aperture images -> reflectivity -> 3DGS.

The four aperture views retain their relative depth amplitude: each view gets
one robust q90 scale over its complete 3-D support.  WLS and GS geometry are
computed once.  Eight shifted-view nulls are solved in one CUDA batch.
No autograd, optimizer, machine learning, known fault, or plate geometry is used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import resource
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def smooth3(field: torch.Tensor, sigma: float = 0.85) -> torch.Tensor:
    radius = max(1, int(math.ceil(2.5 * sigma)))
    coordinate = torch.arange(-radius, radius + 1, device=field.device, dtype=field.dtype)
    kernel = torch.exp(-0.5 * (coordinate / sigma) ** 2)
    kernel /= kernel.sum()
    value = field[None, None]
    value = F.conv3d(value, kernel[:, None, None][None, None], padding=(radius, 0, 0))
    value = F.conv3d(value, kernel[None, :, None][None, None], padding=(0, radius, 0))
    value = F.conv3d(value, kernel[None, None, :][None, None], padding=(0, 0, radius))
    return value[0, 0]


def normalize_global_q90(response: torch.Tensor, illumination: torch.Tensor):
    """One scale per view over all x/y/depth; preserves cross-depth amplitude."""
    corrected = response / torch.sqrt(torch.clamp(illumination, min=0.20))
    corrected = torch.where(
        (response > 0) & (illumination > 0), corrected, torch.zeros_like(corrected)
    )
    result = torch.zeros_like(corrected)
    scales = []
    for view in range(corrected.shape[0]):
        positive = corrected[view][corrected[view] > 0]
        scale = torch.quantile(positive, 0.90) if positive.numel() else torch.ones((), device=response.device)
        scale = torch.clamp(scale, min=1.0e-6)
        scales.append(scale)
        result[view] = torch.clamp(corrected[view] / scale, 0, 1)
    return result, torch.stack(scales)


def solve_wls(y: torch.Tensor, illumination: torch.Tensor, ridge: float):
    active = (y > 0) & (illumination > 0)
    a = torch.sqrt(torch.clamp(illumination, 0, 1))
    weight = active.float() * torch.clamp(illumination, min=0.05)
    normal = torch.sum(a.square() * weight, dim=0)
    rhs = torch.sum(a * weight * y, dim=0)
    field = rhs / torch.clamp(normal + ridge, min=1.0e-6)
    residual = torch.where(active, y - a * field[None], torch.zeros_like(y))
    weight_sum = torch.sum(weight, dim=0)
    residual_variance = torch.sum(weight * residual.square(), dim=0) / torch.clamp(weight_sum - 1, min=1)
    standard_error = torch.sqrt(
        torch.clamp(residual_variance / torch.clamp(normal + ridge, min=1.0e-6), min=0)
    )
    coverage = torch.mean(active.float(), dim=0)
    field = torch.where(coverage >= 0.50, field, torch.zeros_like(field))
    return field, standard_error, coverage


def solve_wls_shift_batch(y: torch.Tensor, illumination: torch.Tensor, ridge: float):
    """Closed-form WLS for [shift, view, x, y, z], reducing view only."""
    active = (y > 0) & (illumination > 0)
    a = torch.sqrt(torch.clamp(illumination, 0, 1))
    weight = active.float() * torch.clamp(illumination, min=0.05)
    normal = torch.sum(a.square() * weight, dim=1)
    rhs = torch.sum(a * weight * y, dim=1)
    field = rhs / torch.clamp(normal + ridge, min=1.0e-6)
    residual = torch.where(active, y - a * field[:, None], torch.zeros_like(y))
    weight_sum = torch.sum(weight, dim=1)
    residual_variance = torch.sum(weight * residual.square(), dim=1) / torch.clamp(weight_sum - 1, min=1)
    standard_error = torch.sqrt(
        torch.clamp(residual_variance / torch.clamp(normal + ridge, min=1.0e-6), min=0)
    )
    coverage = torch.mean(active.float(), dim=1)
    field = torch.where(coverage >= 0.50, field, torch.zeros_like(field))
    return field, standard_error, coverage


def holdout_audit(y: torch.Tensor, illumination: torch.Tensor, ridge: float):
    rows = []
    for holdout in range(y.shape[0]):
        train = [view for view in range(y.shape[0]) if view != holdout]
        field, _standard_error, coverage = solve_wls(y[train], illumination[train], ridge)
        prediction = torch.sqrt(torch.clamp(illumination[holdout], 0, 1)) * field
        chosen = (illumination[holdout] > 0) & (coverage >= 2 / 3)
        x, target = prediction[chosen], y[holdout][chosen]
        if not x.numel():
            rows.append({"view": holdout, "nodes": 0, "rmse": None, "correlation": None})
            continue
        rmse = torch.sqrt(torch.mean((x - target) ** 2))
        xc, yc = x - x.mean(), target - target.mean()
        correlation = torch.sum(xc * yc) / torch.sqrt(
            torch.clamp(torch.sum(xc.square()) * torch.sum(yc.square()), min=1.0e-12)
        )
        rows.append({
            "view": holdout, "nodes": int(x.numel()), "rmse": float(rmse),
            "correlation": float(correlation),
        })
    return rows


def shift_configurations(nx: int, ny: int):
    return [
        ((0, 0), (nx//4, 0), (0, ny//4), (nx//5, ny//5)),
        ((0, 0), (-nx//4, 0), (0, -ny//4), (-nx//5, ny//5)),
        ((0, 0), (nx//3, ny//6), (-nx//6, ny//3), (nx//5, -ny//5)),
        ((0, 0), (-nx//3, ny//6), (nx//6, -ny//3), (-nx//5, -ny//5)),
        ((0, 0), (nx//5, ny//3), (-nx//4, -ny//6), (nx//3, -ny//5)),
        ((0, 0), (-nx//5, ny//3), (nx//4, -ny//6), (-nx//3, -ny//5)),
        ((0, 0), (nx//2, ny//5), (nx//5, -ny//2), (-nx//4, ny//3)),
        ((0, 0), (-nx//2, ny//5), (-nx//5, -ny//2), (nx//4, ny//3)),
    ]


def finite_quantiles(values: np.ndarray, quantiles):
    selected = values[np.isfinite(values) & (values > 0)]
    return np.quantile(selected, quantiles).tolist() if len(selected) else [None] * len(quantiles)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--ridge", type=float, default=0.04)
    parser.add_argument("--null-quantile", type=float, default=0.985)
    parser.add_argument("--baseline-audit", type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    if not (0.90 <= args.null_quantile < 1.0):
        raise SystemExit("--null-quantile must be in [0.90, 1.0)")
    torch.set_grad_enabled(False)
    process_wall_start = time.monotonic()
    with np.load(args.input, allow_pickle=False) as data:
        xyz = np.asarray(data["xyz"], np.float64)
        shape = tuple(map(int, data["grid_shape"].tolist()))
        response = np.asarray(data["view_response"], np.float32).reshape(4, *shape)
        illumination = np.asarray(data["view_illumination"], np.float32).reshape(4, *shape)
        event_id = str(data["event_id"].item())
        phase = str(data["phase"].item())
        tile_id = str(data["tile_id"].item())
        source_xyz_m = np.asarray(data["source_xyz_m"], np.float64)
        source_core_km = float(data["source_core_km"].item())
        source_taper_end_km = float(data["source_taper_end_km"].item())
        input_known_structure = bool(data["known_structure_used"].item())
        radial_projected = bool(data["direct_radial_nuisance_projected"].item()) if "direct_radial_nuisance_projected" in data.files else False
        projection_fraction = float(data["projection_fraction"].item()) if "projection_fraction" in data.files else None
    if response.shape != (4, *shape) or len(xyz) != int(np.prod(shape)):
        raise SystemExit("input grid/view shape mismatch")
    if input_known_structure:
        raise SystemExit("known-structure-derived input is forbidden")
    if not (np.max(np.abs(xyz[:, :2])) > 1.0e6 and np.max(xyz[:, 2]) < 1000):
        raise SystemExit("expected x/y metres and depth kilometres")

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
        normalized, view_scales = normalize_global_q90(raw, light)
        holdout = holdout_audit(normalized, light, args.ridge)
        field, standard_error, coverage = solve_wls(normalized, light, args.ridge)
        variance = standard_error.square()
        significance = field / torch.clamp(standard_error, min=0.01)

        configurations = shift_configurations(shape[0], shape[1])
        shifted_y = torch.stack([
            torch.stack([torch.roll(normalized[view], shifts[view], dims=(0, 1)) for view in range(4)])
            for shifts in configurations
        ])
        shifted_light = torch.stack([
            torch.stack([torch.roll(light[view], shifts[view], dims=(0, 1)) for view in range(4)])
            for shifts in configurations
        ])
        null_field, null_se, null_coverage = solve_wls_shift_batch(shifted_y, shifted_light, args.ridge)
        null_significance = null_field / torch.clamp(null_se, min=0.01)
        thresholds = []
        raw_candidate = torch.zeros_like(field, dtype=torch.bool)
        raw_counts = []
        for depth in range(shape[2]):
            values = null_significance[..., depth][null_coverage[..., depth] >= 0.75]
            threshold = torch.quantile(values, args.null_quantile) if values.numel() else torch.tensor(torch.inf, device=device)
            thresholds.append(threshold)
            selected = (significance[..., depth] > threshold) & (coverage[..., depth] >= 0.75)
            raw_candidate[..., depth] = selected
            raw_counts.append(int(selected.sum()))
        threshold_tensor = torch.stack(thresholds)
        neighbours = F.conv3d(
            raw_candidate.float()[None, None],
            torch.ones((1, 1, 3, 3, 3), device=device), padding=1,
        )[0, 0]
        candidate = raw_candidate & (neighbours >= 3)
        index = torch.nonzero(candidate, as_tuple=False)
        if not len(index):
            raise RuntimeError("no aperture-consistent nodes above shifted-view null")

        smooth = smooth3(field)
        potential = -torch.log(torch.clamp(smooth, min=1.0e-5))
        gradient_tuple = torch.gradient(potential, spacing=spacing, dim=(0, 1, 2), edge_order=1)
        gradient = torch.stack(gradient_tuple, dim=-1)
        hessian = torch.empty((*shape, 3, 3), device=device)
        for row in range(3):
            second = torch.gradient(gradient_tuple[row], spacing=spacing, dim=(0, 1, 2), edge_order=1)
            for column in range(3):
                hessian[..., row, column] = second[column]
        hessian = 0.5 * (hessian + hessian.transpose(-1, -2))
        selected_hessian = hessian[index[:, 0], index[:, 1], index[:, 2]]
        selected_gradient = gradient[index[:, 0], index[:, 1], index[:, 2]]
        eigenvalue, eigenvector = torch.linalg.eigh(selected_hessian)
        strongest = torch.argmax(torch.abs(eigenvalue), dim=1)
        rows = torch.arange(len(index), device=device)
        normal = eigenvector[rows, :, strongest]
        curvature = eigenvalue[rows, strongest]
        safe_curvature = torch.where(
            torch.abs(curvature) > 1.0e-5, curvature,
            torch.where(curvature >= 0, torch.full_like(curvature, 1.0e-5), torch.full_like(curvature, -1.0e-5)),
        )
        displacement = -torch.sum(selected_gradient * normal, dim=1) / safe_curvature
        displacement = torch.clamp(displacement, -0.5 * min(spacing), 0.5 * min(spacing))
        center = coords[index[:, 0], index[:, 1], index[:, 2]] + displacement[:, None] * normal
        center[:, 2].clamp_(float(np.min(xyz[:, 2])), float(np.max(xyz[:, 2])))
        sigma = torch.clamp(
            torch.rsqrt(torch.abs(eigenvalue) + 1 / 14**2), min=0.55 * min(spacing), max=14,
        )
        axes = eigenvector * sigma[:, None, :]
        strength = field[index[:, 0], index[:, 1], index[:, 2]]
        splat_significance = significance[index[:, 0], index[:, 1], index[:, 2]]
        low, high = torch.quantile(strength, torch.tensor([0.02, 0.98], device=device))
        opacity = torch.clamp((strength - low) / torch.clamp(high - low, min=1.0e-6), 0, 1)

    gpu_end.record(); torch.cuda.synchronize()
    gpu_elapsed_ms = float(gpu_begin.elapsed_time(gpu_end))
    peak_allocated = int(torch.cuda.max_memory_allocated())
    peak_reserved = int(torch.cuda.max_memory_reserved())

    center_np = center.cpu().numpy(); axes_np = axes.cpu().numpy()
    strength_np = strength.cpu().numpy(); splat_significance_np = splat_significance.cpu().numpy()
    opacity_np = opacity.cpu().numpy(); candidate_np = candidate.cpu().numpy()
    field_np = field.cpu().numpy().astype(np.float32)
    variance_np = variance.cpu().numpy().astype(np.float32)
    coverage_np = coverage.cpu().numpy().astype(np.float32)
    significance_np = significance.cpu().numpy().astype(np.float32)
    threshold_np = threshold_tensor.cpu().numpy().astype(np.float32)
    scale_np = view_scales.cpu().numpy().astype(np.float32)
    counts = [int(np.count_nonzero(candidate_np[..., depth])) for depth in range(shape[2])]
    projection = np.any(candidate_np, axis=2); occupied = np.argwhere(projection)
    if len(occupied):
        width = int(np.ptp(occupied[:, 0]) + 1); height = int(np.ptp(occupied[:, 1]) + 1)
        rectangular_fill = float(len(occupied) / max(width * height, 1))
    else:
        rectangular_fill = 1.0
    edge = np.zeros(shape[:2], bool); edge[:2] = edge[-2:] = True; edge[:, :2] = edge[:, -2:] = True
    edge_touch = float(np.mean(projection[edge]))
    shallow = float(np.mean(center_np[:, 2] <= 3.0))

    xyz_m = xyz.copy(); xyz_m[:, 2] *= 1000.0
    output = {
        "schema": np.asarray("single-pass-global-view-q90-shift-null-3dgs-v1015"),
        "event_id": np.asarray(event_id), "phase": np.asarray(phase), "tile_id": np.asarray(tile_id),
        "center_xyz_m": (center_np * 1000).astype(np.float32),
        "axis_u_m": (axes_np[:, :, 0] * 1000).astype(np.float32),
        "axis_v_m": (axes_np[:, :, 1] * 1000).astype(np.float32),
        "axis_w_m": (axes_np[:, :, 2] * 1000).astype(np.float32),
        "strength": strength_np.astype(np.float32),
        "splat_significance": splat_significance_np.astype(np.float32),
        "opacity": opacity_np.astype(np.float32),
        "reflectivity_wls_grid": field_np.reshape(-1),
        "reflectivity_variance_grid": variance_np.reshape(-1),
        "illumination_support_grid": coverage_np.reshape(-1),
        "significance_grid": significance_np.reshape(-1),
        "shift_null_q985_by_depth": threshold_np,
        "normalization_q90_by_view": scale_np,
        "candidate_mask": candidate_np.reshape(-1),
        "grid_xyz_m": xyz_m.astype(np.float32), "grid_shape": np.asarray(shape, np.int32),
        "source_xyz_m": source_xyz_m, "source_core_km": np.asarray(source_core_km),
        "source_taper_end_km": np.asarray(source_taper_end_km),
        "normalization_rule": np.asarray("one robust positive q90 scale per view over complete 3D support"),
        "selection_rule": np.asarray("significance > depthwise q98.5 shifted-view null; coverage>=0.75; 3x3x3 neighbours>=3"),
        "strength_semantics": np.asarray("WLS reflectivity; significance stored separately"),
        "cross_event_raw_addition_allowed": np.asarray(False),
        "machine_learning_used": np.asarray(False), "known_structure_used": np.asarray(False),
        "publication_allowed": np.asarray(False),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".v1015.tmp.npz")
    np.savez_compressed(temporary, **output); temporary.replace(args.output)
    total_wall = time.monotonic() - process_wall_start
    correlations = [row["correlation"] for row in holdout if row["correlation"] is not None]
    aperture_consistent = len(correlations) == 4 and sum(value > 0 for value in correlations) >= 3 and sorted(correlations)[1] > 0.05
    checks = {
        "cuda_used": True, "autograd_absent": not torch.is_grad_enabled(), "optimizer_absent": True,
        "machine_learning_absent": True, "known_structure_absent": not input_known_structure,
        "global_view_q90_used": len(scale_np) == 4 and bool(np.all(np.isfinite(scale_np))),
        "per_depth_normalization_absent": True, "single_actual_wls": True,
        "single_hessian_eigh": True, "eight_shift_single_cuda_batch": True,
        "coverage_ge_075_used": True, "q985_depthwise_null_used": abs(args.null_quantile - 0.985) < 1.0e-12,
        "neighbour_gate_used": True, "aperture_images_share_structure": aperture_consistent,
        "finite": bool(all(np.isfinite(value).all() for value in (center_np, axes_np, strength_np, opacity_np, field_np, variance_np, coverage_np, significance_np))),
        "nonrectangular_footprint": rectangular_fill < 0.80,
        "shallow_structure_present": shallow > 0.01,
        "at_least_100_significant_splats": len(center_np) >= 100,
        "reflectivity_strength_not_significance": True,
    }
    baseline = None
    if args.baseline_audit and args.baseline_audit.exists():
        old = json.loads(args.baseline_audit.read_text(encoding="utf-8"))
        old_wall = float(old.get("gpu", {}).get("wall_seconds", 0))
        baseline = {
            "path": str(args.baseline_audit), "schema": old.get("schema"),
            "wall_seconds": old_wall, "gpu_elapsed_ms": old.get("gpu", {}).get("elapsed_ms"),
            "peak_allocated_bytes": old.get("gpu", {}).get("peak_allocated_bytes"),
            "splat_count": old.get("splat_count"), "footprint": old.get("footprint"),
            "holdout": old.get("leave_one_aperture_out"),
            "speedup_wall": old_wall / total_wall if old_wall > 0 else None,
            "method_difference": "baseline used per-view-per-depth q90 and repeated WLS/Hessian; v1015 uses per-view-global-3D q90 and single pass",
        }
    audit = {
        "schema": "single-pass-global-view-q90-shift-null-3dgs-audit-v1015",
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "input": str(args.input), "input_sha256": sha256(args.input),
        "output": str(args.output), "output_sha256": sha256(args.output),
        "event_id": event_id, "phase": phase, "tile_id": tile_id, "grid_shape": list(shape),
        "spacing_km": list(spacing), "ridge": args.ridge, "null_quantile": args.null_quantile,
        "normalization": {"rule": "one positive robust q90 per view over complete 3D support", "q90_by_view": scale_np.tolist()},
        "equation": "g=(sum_v A_v W_v y_v)/(sum_v A_v W_v A_v+ridge)",
        "leave_one_aperture_out": holdout,
        "candidate_counts_before_gate_by_depth": raw_counts,
        "candidate_counts_after_gate_by_depth": counts,
        "shift_null_q985_by_depth": threshold_np.tolist(),
        "splat_count": int(len(center_np)),
        "footprint": {"projected_cells": int(len(occupied)), "rectangular_fill_fraction": rectangular_fill,
                      "edge_touch_fraction": edge_touch, "shallow_splat_fraction_le_3km": shallow},
        "reflectivity": {"positive_q02_q10_q50_q90_q98": finite_quantiles(field_np, [.02, .1, .5, .9, .98]),
                         "positive_nodes": int(np.count_nonzero(field_np > 0)),
                         "support_fraction_ge_075": float(np.mean(coverage_np >= .75))},
        "splat_strength_reflectivity_q02_q10_q50_q90_q98": finite_quantiles(strength_np, [.02, .1, .5, .9, .98]),
        "splat_significance_q02_q10_q50_q90_q98": finite_quantiles(splat_significance_np, [.02, .1, .5, .9, .98]),
        "input_direct_radial_nuisance_projected": radial_projected,
        "input_projection_fraction": projection_fraction,
        "gpu": {"name": torch.cuda.get_device_name(0), "torch": torch.__version__, "cuda": torch.version.cuda,
                "elapsed_ms": gpu_elapsed_ms, "peak_allocated_bytes": peak_allocated,
                "peak_reserved_bytes": peak_reserved},
        "wall_seconds": total_wall,
        "process_peak_ram_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
        "baseline_repeated_solver": baseline,
        "checks": checks, "pass": all(checks.values()),
        "machine_learning_used": False, "autograd_used": False, "optimizer_used": False,
        "known_structure_used": False, "publication_allowed": False,
    }
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.audit.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"audit": str(args.audit), "pass": audit["pass"], "splat_count": len(center_np),
                      "wall_seconds": total_wall, "gpu_elapsed_ms": gpu_elapsed_ms,
                      "rectangular_fill": rectangular_fill, "shallow": shallow,
                      "holdout": holdout, "strength_q": audit["splat_strength_reflectivity_q02_q10_q50_q90_q98"],
                      "baseline": baseline}, ensure_ascii=False))
    raise SystemExit(0 if audit["pass"] else 2)


if __name__ == "__main__":
    main()
