#!/usr/bin/env python3
"""Weighted least-squares aperture images -> one event-illuminated 3-D GS.

This is a deterministic inverse problem. CUDA accelerates linear algebra and
physical derivatives; autograd, training, learned weights, and ML optimizers are absent.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def smooth3(field: torch.Tensor, sigma: float = 0.85) -> torch.Tensor:
    radius = int(math.ceil(3 * sigma))
    x = torch.arange(-radius, radius + 1, device=field.device, dtype=field.dtype)
    kernel = torch.exp(-0.5 * (x / sigma) ** 2)
    kernel /= kernel.sum()
    value = field[None, None]
    for axis, shape in enumerate(((len(kernel), 1, 1), (1, len(kernel), 1), (1, 1, len(kernel)))):
        padding = [0, 0, 0]
        padding[axis] = radius
        value = F.conv3d(value, kernel.reshape(shape)[None, None], padding=tuple(padding))
    return value[0, 0]


def normalize_aperture_images(response: torch.Tensor, illumination: torch.Tensor) -> torch.Tensor:
    """Put independently synthesized views on one robust per-depth amplitude scale."""
    corrected = response / torch.sqrt(torch.clamp(illumination, min=0.20))
    corrected = torch.where((response > 0) & (illumination > 0), corrected, torch.zeros_like(corrected))
    result = torch.zeros_like(corrected)
    for view in range(corrected.shape[0]):
        for depth in range(corrected.shape[-1]):
            layer = corrected[view, :, :, depth]
            positive = layer[layer > 0]
            scale = torch.quantile(positive, 0.90) if positive.numel() else torch.ones((), device=layer.device)
            result[view, :, :, depth] = torch.clamp(layer / torch.clamp(scale, min=1.0e-6), 0, 1)
    return result


def solve_wls(y: torch.Tensor, illumination: torch.Tensor, ridge: float):
    """Closed-form diagonal weighted LS for y_v=A_v g + e_v."""
    active = (y > 0) & (illumination > 0)
    a = torch.sqrt(torch.clamp(illumination, 0, 1))
    w = active.float() * torch.clamp(illumination, min=0.05)
    normal = torch.sum(a.square() * w, dim=0)
    rhs = torch.sum(a * w * y, dim=0)
    g = rhs / (normal + ridge)
    prediction = a * g[None]
    residual = torch.where(active, y - prediction, torch.zeros_like(y))
    weight_sum = torch.sum(w, dim=0)
    residual_variance = torch.sum(w * residual.square(), dim=0) / torch.clamp(weight_sum - 1, min=1)
    standard_error = torch.sqrt(torch.clamp(residual_variance / torch.clamp(normal + ridge, min=1.0e-6), min=0))
    coverage = active.float().mean(dim=0)
    g = torch.where(coverage >= 0.50, g, torch.zeros_like(g))
    return g, standard_error, coverage


def holdout_audit(y: torch.Tensor, illumination: torch.Tensor, ridge: float) -> list[dict]:
    rows = []
    for holdout in range(y.shape[0]):
        train = [v for v in range(y.shape[0]) if v != holdout]
        field, _se, coverage = solve_wls(y[train], illumination[train], ridge)
        prediction = torch.sqrt(torch.clamp(illumination[holdout], 0, 1)) * field
        chosen = (illumination[holdout] > 0) & (coverage >= 2 / 3)
        x, target = prediction[chosen], y[holdout][chosen]
        if not x.numel():
            rows.append({"view": holdout, "nodes": 0, "rmse": None, "correlation": None})
            continue
        rmse = torch.sqrt(torch.mean((x - target) ** 2))
        xc, yc = x - x.mean(), target - target.mean()
        corr = torch.sum(xc * yc) / torch.sqrt(torch.clamp(torch.sum(xc.square()) * torch.sum(yc.square()), min=1e-12))
        rows.append({"view": holdout, "nodes": int(x.numel()), "rmse": float(rmse), "correlation": float(corr)})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--audit", type=Path, required=True)
    ap.add_argument("--ridge", type=float, default=0.04)
    ap.add_argument("--layer-quantile", type=float, default=0.58)
    ap.add_argument("--max-splats", type=int, default=24000)
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; CPU fallback disabled")
    torch.set_grad_enabled(False)
    device = torch.device("cuda")
    with np.load(args.input, allow_pickle=False) as data:
        if bool(data["known_structure_used"].item()):
            raise RuntimeError("known structure input forbidden")
        xyz = np.asarray(data["xyz"], np.float64)
        shape = tuple(map(int, data["grid_shape"].tolist()))
        response = np.asarray(data["view_response"], np.float32)
        illumination = np.asarray(data["view_illumination"], np.float32)
        source_m = np.asarray(data["source_xyz_m"], np.float64)
        event_id, phase, tile_id = (str(data[k].item()) for k in ("event_id", "phase", "tile_id"))
        source_core_km = float(data["source_core_km"].item())
    if response.shape != (4, len(xyz)) or int(np.prod(shape)) != len(xyz):
        raise RuntimeError("four aperture images on a Cartesian volume are required")
    if not (np.max(np.abs(xyz[:, :2])) > 1e6 and np.max(xyz[:, 2]) < 1000):
        raise RuntimeError("expected x/y metres and depth kilometres")

    xyz_km = xyz.copy()
    xyz_km[:, :2] /= 1000
    source_km = source_m / 1000
    spacing = tuple(float(np.median(np.diff(np.unique(xyz_km[:, i])))) for i in range(3))
    coords = torch.as_tensor(xyz_km.reshape(*shape, 3), device=device)
    raw = torch.as_tensor(response.reshape(4, *shape), device=device)
    light = torch.as_tensor(np.clip(illumination.reshape(4, *shape), 0, 1), device=device)
    torch.cuda.reset_peak_memory_stats()
    start, stop = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    wall = time.monotonic()

    y = normalize_aperture_images(raw, light)
    field, standard_error, coverage = solve_wls(y, light, args.ridge)
    holdout = holdout_audit(y, light, args.ridge)
    field = smooth3(field)
    potential = -torch.log(torch.clamp(field, min=1e-5))
    grad_tuple = torch.gradient(potential, spacing=spacing, dim=(0, 1, 2), edge_order=1)
    gradient = torch.stack(grad_tuple, dim=-1)
    hessian = torch.empty((*shape, 3, 3), device=device)
    for row in range(3):
        second = torch.gradient(grad_tuple[row], spacing=spacing, dim=(0, 1, 2), edge_order=1)
        for column in range(3):
            hessian[..., row, column] = second[column]
    hessian = 0.5 * (hessian + hessian.transpose(-1, -2))

    candidate = torch.zeros_like(field, dtype=torch.bool)
    counts = []
    for depth in range(shape[2]):
        layer = field[:, :, depth]
        eligible = layer[(layer > 0) & (coverage[:, :, depth] >= 0.75)]
        threshold = torch.quantile(eligible, args.layer_quantile) if eligible.numel() else torch.inf
        selected = (layer >= threshold) & (coverage[:, :, depth] >= 0.75)
        candidate[:, :, depth] = selected
        counts.append(int(selected.sum()))
    index = torch.nonzero(candidate, as_tuple=False)
    values = field[candidate]
    if len(index) > args.max_splats:
        keep = torch.topk(values, args.max_splats, sorted=False).indices
        index, values = index[keep], values[keep]
    if not len(index):
        raise RuntimeError("least-squares event field produced no supported splats")

    h = hessian[index[:, 0], index[:, 1], index[:, 2]]
    g = gradient[index[:, 0], index[:, 1], index[:, 2]]
    eigenvalue, eigenvector = torch.linalg.eigh(h)
    strongest = torch.argmax(torch.abs(eigenvalue), dim=1)
    row = torch.arange(len(index), device=device)
    normal = eigenvector[row, :, strongest]
    curvature = eigenvalue[row, strongest]
    displacement = -torch.sum(g * normal, dim=1) / torch.where(
        torch.abs(curvature) > 1e-5, curvature, torch.full_like(curvature, 1e-5)
    )
    displacement = torch.clamp(displacement, -0.5 * min(spacing), 0.5 * min(spacing))
    center = coords[index[:, 0], index[:, 1], index[:, 2]] + displacement[:, None] * normal
    sigma = torch.clamp(torch.rsqrt(torch.abs(eigenvalue) + 1 / 14**2), min=0.55 * min(spacing), max=14)
    axes = eigenvector * sigma[:, None, :]
    local_se = standard_error[index[:, 0], index[:, 1], index[:, 2]]
    local_coverage = coverage[index[:, 0], index[:, 1], index[:, 2]]
    precision_weight = 1 / torch.clamp(local_se, min=0.03)
    strength = values * precision_weight * local_coverage
    lo, hi = torch.quantile(strength, torch.tensor([0.02, 0.98], device=device))
    opacity = torch.clamp((strength - lo) / torch.clamp(hi - lo, min=1e-6), 0, 1)
    stop.record()
    torch.cuda.synchronize()

    center_np, axes_np = center.cpu().numpy(), axes.cpu().numpy()
    strength_np, opacity_np = strength.cpu().numpy(), opacity.cpu().numpy()
    se_np, coverage_np = local_se.cpu().numpy(), local_coverage.cpu().numpy()
    source_distance = np.linalg.norm(center_np - source_km[None], axis=1)
    output = {
        "schema": np.asarray("aperture-image-wls-event-illuminated-3dgs-v992"),
        "event_id": np.asarray(event_id), "phase": np.asarray(phase), "tile_id": np.asarray(tile_id),
        "center_xyz_m": (center_np * 1000).astype(np.float32),
        "axis_u_m": (axes_np[:, :, 0] * 1000).astype(np.float32),
        "axis_v_m": (axes_np[:, :, 1] * 1000).astype(np.float32),
        "axis_w_m": (axes_np[:, :, 2] * 1000).astype(np.float32),
        "strength": strength_np.astype(np.float32), "opacity": opacity_np.astype(np.float32),
        "least_squares_standard_error": se_np.astype(np.float32),
        "cross_view_coverage": coverage_np.astype(np.float32),
        "machine_learning_used": np.asarray(False), "known_structure_used": np.asarray(False),
        "publication_allowed": np.asarray(False),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp = args.output.with_name(args.output.name + ".tmp.npz")
    np.savez_compressed(temp, **output)
    temp.replace(args.output)
    checks = {
        "cuda_used": True, "autograd_absent": not torch.is_grad_enabled(), "optimizer_absent": True,
        "xy_m_depth_km_converted": True,
        "physical_spacing_km": bool(np.allclose(spacing, (1.875, 1.875, 1.5), atol=1e-6, rtol=0)),
        "finite": bool(all(np.isfinite(x).all() for x in (center_np, axes_np, strength_np, opacity_np))),
        "depth_extent": bool(np.ptp(center_np[:, 2]) >= 0.65 * np.ptp(xyz_km[:, 2])),
        "source_core_absent": bool(np.all(source_distance >= source_core_km - min(spacing))),
        "three_or_four_views": bool(np.all(coverage_np >= 0.75)), "known_structure_absent": True,
    }
    audit = {
        "schema": "aperture-image-wls-event-illuminated-3dgs-audit-v992",
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "equation": "g=(sum_v A_v W_v y_v)/(sum_v A_v W_v A_v + lambda)",
        "input": str(args.input), "input_sha256": digest(args.input), "event_id": event_id,
        "phase": phase, "tile_id": tile_id, "ridge": args.ridge,
        "input_units": {"x": "m", "y": "m", "depth": "km"}, "internal_unit": "km",
        "spacing_km": spacing, "splat_count": int(len(center_np)), "counts_by_depth": counts,
        "depth_km_min_max": [float(center_np[:, 2].min()), float(center_np[:, 2].max())],
        "strength_q02_q10_q50_q90_q98": np.quantile(strength_np, [.02, .1, .5, .9, .98]).tolist(),
        "leave_one_aperture_out": holdout,
        "gpu": {"name": torch.cuda.get_device_name(0), "torch": torch.__version__,
                "cuda": torch.version.cuda, "elapsed_ms": float(start.elapsed_time(stop)),
                "wall_seconds": time.monotonic() - wall,
                "peak_allocated_bytes": int(torch.cuda.max_memory_allocated())},
        "output": str(args.output), "output_sha256": digest(args.output), "checks": checks,
        "pass": all(checks.values()), "machine_learning_used": False, "autograd_used": False,
        "known_structure_used": False, "publication_allowed": False,
    }
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.audit.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False))
    raise SystemExit(0 if audit["pass"] else 2)


if __name__ == "__main__":
    main()
