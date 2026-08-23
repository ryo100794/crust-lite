#!/usr/bin/env python3
"""Torch CPU/CUDA parity for component-window physical removal."""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch


def remove(
    sensor_values: torch.Tensor,
    orientations: torch.Tensor,
    rays: torch.Tensor,
    component_tapers: torch.Tensor,
    phase_code: torch.Tensor,
) -> torch.Tensor:
    # values/tapers [K,3,T], orientations [K,3,3], rays [K,3].
    rays = rays / torch.linalg.vector_norm(rays, dim=1, keepdim=True).clamp_min(1.0e-15)
    inverse = torch.linalg.pinv(orientations, rtol=1.0e-7)
    enu = torch.einsum("kij,kjt->kit", inverse, sensor_values)
    longitudinal = torch.einsum("ki,kjt,kj->kit", rays, enu, rays)
    projected_enu = torch.where(phase_code[:, None, None] == 0, longitudinal, enu - longitudinal)
    projected_sensor = torch.einsum("kij,kjt->kit", orientations, projected_enu)
    return sensor_values - component_tapers * projected_sensor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", required=True)
    args = parser.parse_args()
    with np.load(args.fixture, allow_pickle=False) as fixture:
        values = fixture["values"].astype(np.float64)
        orientations = fixture["orientations"].astype(np.float64)
        rays = fixture["rays"].astype(np.float64)
        tapers = fixture["component_tapers"].astype(np.float64)
        phase_code = fixture["phase_code"].astype(np.int64)
        expected = fixture["clean_cpu"].astype(np.float64)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for component-window parity")
    cpu = remove(
        torch.from_numpy(values), torch.from_numpy(orientations), torch.from_numpy(rays),
        torch.from_numpy(tapers), torch.from_numpy(phase_code),
    ).numpy()
    gpu = remove(
        torch.from_numpy(values).cuda(), torch.from_numpy(orientations).cuda(),
        torch.from_numpy(rays).cuda(), torch.from_numpy(tapers).cuda(),
        torch.from_numpy(phase_code).cuda(),
    ).cpu().numpy()
    scale = max(float(np.max(np.abs(expected))), 1.0e-30)
    cpu_abs = float(np.max(np.abs(expected - cpu)))
    gpu_abs = float(np.max(np.abs(expected - gpu)))
    result = {
        "schema": "hinet-direct-wave-component-window-cpu-gpu-parity-v1144",
        "device": torch.cuda.get_device_name(0),
        "dtype": "float64",
        "station_phase_cases": int(values.shape[0]),
        "samples_per_case": int(values.shape[2]),
        "numpy_vs_torch_cpu_max_abs": cpu_abs,
        "numpy_vs_torch_cpu_max_rel": cpu_abs / scale,
        "numpy_vs_cuda_max_abs": gpu_abs,
        "numpy_vs_cuda_max_rel": gpu_abs / scale,
        "torch_cpu_vs_cuda_max_abs": float(np.max(np.abs(cpu - gpu))),
        "pass": bool(gpu_abs <= max(1.0e-9, scale * 1.0e-12)),
    }
    print(json.dumps(result, indent=2))
    if not result["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
