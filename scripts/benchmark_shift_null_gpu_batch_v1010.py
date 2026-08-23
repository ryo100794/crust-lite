#!/usr/bin/env python3
"""Numerical CPU/GPU proof for the v1008 shifted-aperture null calculation.

This benchmark is intentionally isolated from publication outputs.  It uses only
closed-form WLS, shifts, quantiles, convolution, derivatives, and symmetric
eigendecomposition.  It has no autograd, optimizer, ML, fault, or plate input.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import resource
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def load_base(script_dir: Path):
    path = script_dir / "solve_aperture_images_to_event_gs_wls_gpu_v992.py"
    spec = importlib.util.spec_from_file_location("wls_base_v992_for_gpu_batch_v1010", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def configurations(nx: int, ny: int):
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


def solve_wls_batched(y: torch.Tensor, light: torch.Tensor, ridge: float):
    """WLS for [shift, view, x, y, z], reducing the view axis only."""
    active = (y > 0) & (light > 0)
    a = torch.sqrt(torch.clamp(light, 0, 1))
    w = active.float() * torch.clamp(light, min=0.05)
    normal = torch.sum(a.square() * w, dim=1)
    rhs = torch.sum(a * w * y, dim=1)
    field = rhs / torch.clamp(normal + ridge, min=1.0e-6)
    residual = torch.where(active, y - a * field[:, None], torch.zeros_like(y))
    weight_sum = torch.sum(w, dim=1)
    variance = torch.sum(w * residual.square(), dim=1) / torch.clamp(weight_sum - 1, min=1)
    se = torch.sqrt(torch.clamp(variance / torch.clamp(normal + ridge, min=1.0e-6), min=0))
    coverage = torch.mean(active.float(), dim=1)
    field = torch.where(coverage >= 0.50, field, torch.zeros_like(field))
    return field, se, coverage


def make_shift_batch(y: torch.Tensor, light: torch.Tensor, shifts):
    shifted_y = torch.stack([
        torch.stack([torch.roll(y[v], config[v], dims=(0, 1)) for v in range(4)])
        for config in shifts
    ])
    shifted_light = torch.stack([
        torch.stack([torch.roll(light[v], config[v], dims=(0, 1)) for v in range(4)])
        for config in shifts
    ])
    return shifted_y, shifted_light


def thresholds_and_mask(null_z, null_coverage, significance, coverage, q=0.99):
    nz = significance.shape[-1]
    thresholds = []
    candidate = torch.zeros_like(significance, dtype=torch.bool)
    for depth in range(nz):
        values = null_z[..., depth][null_coverage[..., depth] >= 0.75]
        threshold = torch.quantile(values, q) if values.numel() else torch.tensor(torch.inf, device=null_z.device)
        thresholds.append(threshold)
        candidate[..., depth] = (significance[..., depth] > threshold) & (coverage[..., depth] >= 0.75)
    threshold_tensor = torch.stack(thresholds)
    neighbours = F.conv3d(
        candidate.float()[None, None],
        torch.ones((1, 1, 3, 3, 3), device=candidate.device),
        padding=1,
    )[0, 0]
    return threshold_tensor, candidate, candidate & (neighbours >= 3)


def cuda_measure(call):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    wall = time.monotonic()
    begin.record()
    result = call()
    end.record()
    torch.cuda.synchronize()
    return result, {
        "wall_seconds": time.monotonic() - wall,
        "cuda_elapsed_ms": float(begin.elapsed_time(end)),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--script-dir", type=Path, required=True)
    parser.add_argument("--ridge", type=float, default=0.04)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")
    torch.set_grad_enabled(False)
    base = load_base(args.script_dir)
    with np.load(args.input, allow_pickle=False) as data:
        shape = tuple(map(int, data["grid_shape"].tolist()))
        response = np.asarray(data["view_response"], np.float32).reshape(4, *shape)
        illumination = np.asarray(data["view_illumination"], np.float32).reshape(4, *shape)
        event_id = str(data["event_id"].item())
        phase = str(data["phase"].item())
        tile_id = str(data["tile_id"].item())
        xyz = np.asarray(data["xyz"], np.float64)
    shifts = configurations(shape[0], shape[1])

    device = torch.device("cuda")
    raw_gpu = torch.as_tensor(response, device=device)
    light_gpu = torch.as_tensor(np.clip(illumination, 0, 1), device=device)
    y_gpu, profile_normalize = cuda_measure(lambda: base.normalize_aperture_images(raw_gpu, light_gpu))
    actual_gpu, profile_actual = cuda_measure(lambda: base.solve_wls(y_gpu, light_gpu, args.ridge))
    field_gpu, se_gpu, coverage_gpu = actual_gpu
    significance_gpu = field_gpu / torch.clamp(se_gpu, min=0.01)

    def sequential_gpu():
        fields, ses, coverages = [], [], []
        for config in shifts:
            sy = torch.stack([torch.roll(y_gpu[v], config[v], dims=(0, 1)) for v in range(4)])
            sl = torch.stack([torch.roll(light_gpu[v], config[v], dims=(0, 1)) for v in range(4)])
            f, se, cov = base.solve_wls(sy, sl, args.ridge)
            fields.append(f); ses.append(se); coverages.append(cov)
        return torch.stack(fields), torch.stack(ses), torch.stack(coverages)

    seq_gpu, profile_sequential = cuda_measure(sequential_gpu)

    def batched_gpu():
        sy, sl = make_shift_batch(y_gpu, light_gpu, shifts)
        return solve_wls_batched(sy, sl, args.ridge)

    bat_gpu, profile_batched = cuda_measure(batched_gpu)
    seq_field, seq_se, seq_coverage = seq_gpu
    bat_field, bat_se, bat_coverage = bat_gpu
    seq_z = seq_field / torch.clamp(seq_se, min=0.01)
    bat_z = bat_field / torch.clamp(bat_se, min=0.01)
    seq_selection, profile_seq_selection = cuda_measure(
        lambda: thresholds_and_mask(seq_z, seq_coverage, significance_gpu, coverage_gpu)
    )
    bat_selection, profile_bat_selection = cuda_measure(
        lambda: thresholds_and_mask(bat_z, bat_coverage, significance_gpu, coverage_gpu)
    )

    # Profile the remaining v1008 mathematical GS geometry on the batched mask.
    def geometry_gpu():
        threshold, raw_candidate, gated = bat_selection
        xyz_km = xyz.copy(); xyz_km[:, :2] /= 1000.0
        spacing = tuple(float(np.median(np.diff(np.unique(xyz_km[:, i])))) for i in range(3))
        smooth = base.smooth3(field_gpu)
        potential = -torch.log(torch.clamp(smooth, min=1e-5))
        gradient_tuple = torch.gradient(potential, spacing=spacing, dim=(0, 1, 2), edge_order=1)
        hessian = torch.empty((*shape, 3, 3), device=device)
        for row in range(3):
            second = torch.gradient(gradient_tuple[row], spacing=spacing, dim=(0, 1, 2), edge_order=1)
            for column in range(3): hessian[..., row, column] = second[column]
        hessian = 0.5 * (hessian + hessian.transpose(-1, -2))
        index = torch.nonzero(gated, as_tuple=False)
        selected_hessian = hessian[index[:, 0], index[:, 1], index[:, 2]]
        eigenvalue, eigenvector = torch.linalg.eigh(selected_hessian)
        return eigenvalue, eigenvector

    _geometry, profile_geometry = cuda_measure(geometry_gpu)

    # Independent CPU reference uses the unchanged scalar-view v992 WLS eight times.
    cpu_wall = time.monotonic()
    raw_cpu = torch.from_numpy(response.copy())
    light_cpu = torch.from_numpy(np.clip(illumination, 0, 1).copy())
    y_cpu = base.normalize_aperture_images(raw_cpu, light_cpu)
    field_cpu, se_cpu, coverage_cpu = base.solve_wls(y_cpu, light_cpu, args.ridge)
    significance_cpu = field_cpu / torch.clamp(se_cpu, min=0.01)
    cpu_fields, cpu_ses, cpu_coverages = [], [], []
    for config in shifts:
        sy = torch.stack([torch.roll(y_cpu[v], config[v], dims=(0, 1)) for v in range(4)])
        sl = torch.stack([torch.roll(light_cpu[v], config[v], dims=(0, 1)) for v in range(4)])
        f, se, cov = base.solve_wls(sy, sl, args.ridge)
        cpu_fields.append(f); cpu_ses.append(se); cpu_coverages.append(cov)
    cpu_field = torch.stack(cpu_fields); cpu_se = torch.stack(cpu_ses); cpu_coverage = torch.stack(cpu_coverages)
    cpu_z = cpu_field / torch.clamp(cpu_se, min=0.01)
    cpu_selection = thresholds_and_mask(cpu_z, cpu_coverage, significance_cpu, coverage_cpu)
    cpu_wall_seconds = time.monotonic() - cpu_wall

    seq_threshold, seq_raw, seq_gated = seq_selection
    bat_threshold, bat_raw, bat_gated = bat_selection
    cpu_threshold, cpu_raw, cpu_gated = cpu_selection
    def max_abs(a, b): return float(torch.max(torch.abs(a-b)).item())
    agreement = {
        "gpu_batch_vs_gpu_sequential": {
            "field_max_abs": max_abs(bat_field, seq_field),
            "standard_error_max_abs": max_abs(bat_se, seq_se),
            "coverage_max_abs": max_abs(bat_coverage, seq_coverage),
            "threshold_max_abs": max_abs(bat_threshold, seq_threshold),
            "raw_mask_mismatch": int(torch.count_nonzero(bat_raw != seq_raw)),
            "gated_mask_mismatch": int(torch.count_nonzero(bat_gated != seq_gated)),
        },
        "gpu_batch_vs_cpu_reference": {
            "field_max_abs": max_abs(bat_field.cpu(), cpu_field),
            "standard_error_max_abs": max_abs(bat_se.cpu(), cpu_se),
            "coverage_max_abs": max_abs(bat_coverage.cpu(), cpu_coverage),
            "threshold_max_abs": max_abs(bat_threshold.cpu(), cpu_threshold),
            "raw_mask_mismatch": int(torch.count_nonzero(bat_raw.cpu() != cpu_raw)),
            "gated_mask_mismatch": int(torch.count_nonzero(bat_gated.cpu() != cpu_gated)),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        schema=np.asarray("shift-null-cuda-batch-proof-v1010"),
        threshold_q99=bat_threshold.cpu().numpy().astype(np.float32),
        candidate_mask=bat_gated.cpu().numpy(),
        machine_learning_used=np.asarray(False),
        known_structure_used=np.asarray(False),
    )
    speedup = profile_sequential["cuda_elapsed_ms"] / max(profile_batched["cuda_elapsed_ms"], 1e-9)
    audit = {
        "schema": "shift-null-cuda-batch-proof-audit-v1010",
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "input": str(args.input), "input_sha256": sha256(args.input),
        "output": str(args.output), "output_sha256": sha256(args.output),
        "event_id": event_id, "phase": phase, "tile_id": tile_id,
        "grid_shape": list(shape), "shift_count": len(shifts), "ridge": args.ridge,
        "method": "closed-form diagonal WLS; eight shifted nulls batched on CUDA",
        "profile": {
            "gpu_normalize": profile_normalize,
            "gpu_actual_wls": profile_actual,
            "gpu_shift_null_sequential": profile_sequential,
            "gpu_shift_null_batched": profile_batched,
            "gpu_selection_sequential": profile_seq_selection,
            "gpu_selection_batched": profile_bat_selection,
            "gpu_smooth_hessian_eigh": profile_geometry,
            "cpu_reference_wall_seconds": cpu_wall_seconds,
            "process_peak_ram_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
            "shift_null_wls_cuda_speedup": speedup,
        },
        "agreement": agreement,
        "candidate_count_before_gate": int(torch.count_nonzero(bat_raw)),
        "candidate_count_after_gate": int(torch.count_nonzero(bat_gated)),
        "checks": {
            "cuda_used": True,
            "autograd_absent": not torch.is_grad_enabled(),
            "optimizer_absent": True,
            "machine_learning_absent": True,
            "known_structure_absent": True,
            "gpu_batch_matches_gpu_sequential": agreement["gpu_batch_vs_gpu_sequential"]["gated_mask_mismatch"] == 0,
            "gpu_batch_matches_cpu_reference": agreement["gpu_batch_vs_cpu_reference"]["gated_mask_mismatch"] == 0,
        },
    }
    audit["pass"] = all(audit["checks"].values())
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.audit.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"audit": str(args.audit), "pass": audit["pass"], "speedup": speedup,
                      "agreement": agreement, "profile": audit["profile"]}, ensure_ascii=False))
    raise SystemExit(0 if audit["pass"] else 2)


if __name__ == "__main__":
    main()
