#!/usr/bin/env python3
"""v994: v417 isochron migration using direct-polarization residual wavefields.

The direct arrival remains available only as a source phase/energy calibration.
Before candidate-ray projection, its polarization subspace is removed from the
reconstructed displacement. This prevents direct-wave geometry from becoming GS.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def dominant_direct_polarization(migration, components, orientation, pick_s, direct_ray, phase):
    if phase == "P":
        vector = np.asarray(direct_ray, np.complex128)
        return vector / max(float(np.linalg.norm(vector)), 1.0e-12)
    offsets = np.linspace(-1.0, 3.0, 41)
    rows, indices = [], []
    for index, component in enumerate(components):
        values = migration.interpolate_complex(
            component["times"], component["analytic"], pick_s + offsets
        )
        if np.isfinite(values.real).all() and np.isfinite(values.imag).all():
            indices.append(index)
            rows.append(values)
    if len(indices) < 2 or np.linalg.matrix_rank(orientation[indices], tol=1.0e-6) < 2:
        # A deterministic transverse direction is the safe fallback for S.
        trial = np.cross(np.asarray(direct_ray, float), np.asarray([0.0, 0.0, 1.0]))
        if np.linalg.norm(trial) < 1.0e-6:
            trial = np.asarray([1.0, 0.0, 0.0])
        return trial.astype(np.complex128) / np.linalg.norm(trial)
    displacement = np.linalg.pinv(orientation[indices], rcond=1.0e-6) @ np.asarray(rows)
    u, _singular, _vh = np.linalg.svd(displacement, full_matrices=False)
    vector = u[:, 0]
    return vector / max(float(np.linalg.norm(vector)), 1.0e-12)


SOURCE = Path(__file__).with_name("stream_tile_safe_isochron_v417.py")
source = SOURCE.read_text(encoding="utf-8")
ray_block = '''    ray = np.asarray([
        np.sin(np.radians(incident)) * np.sin(np.radians(backaz)),
        np.sin(np.radians(incident)) * np.cos(np.radians(backaz)),
        np.cos(np.radians(incident)),
    ])
    combined = np.zeros(len(xyz), dtype=np.complex128)'''
ray_replacement = '''    ray = np.asarray([
        np.sin(np.radians(incident)) * np.sin(np.radians(backaz)),
        np.sin(np.radians(incident)) * np.cos(np.radians(backaz)),
        np.cos(np.radians(incident)),
    ])
    direct_azimuth_early = (np.degrees(np.arctan2(sx - station["x_m"], sy - station["y_m"])) + 360.0) % 360.0
    direct_incidence_early = math.degrees(math.atan2(direct_horizontal, max(sz, 0.1)))
    direct_ray_early = np.asarray([
        math.sin(math.radians(direct_incidence_early)) * math.sin(math.radians(direct_azimuth_early)),
        math.sin(math.radians(direct_incidence_early)) * math.cos(math.radians(direct_azimuth_early)),
        math.cos(math.radians(direct_incidence_early)),
    ])
    direct_polarization = dominant_direct_polarization(
        migration, components, orientation, station_pick, direct_ray_early, phase
    )
    residual_projector = np.eye(3, dtype=np.complex128) - np.outer(
        direct_polarization, np.conjugate(direct_polarization)
    )
    combined = np.zeros(len(xyz), dtype=np.complex128)'''
if source.count(ray_block) != 1:
    raise RuntimeError("v417 ray block differs")
source = source.replace(ray_block, ray_replacement)

projection_block = '''        displacement = inverse @ samples[component_indices][:, selected]
        covariance = inverse @ inverse.T
        scalar, noise_gain = base.projected_scalar(displacement, ray[:, selected], covariance, phase)'''
projection_replacement = '''        displacement = inverse @ samples[component_indices][:, selected]
        displacement = residual_projector @ displacement
        covariance = residual_projector @ (inverse @ inverse.T) @ np.conjugate(residual_projector.T)
        covariance = np.asarray(np.real(covariance), float)
        scalar, noise_gain = base.projected_scalar(displacement, ray[:, selected], covariance, phase)'''
if source.count(projection_block) != 1:
    raise RuntimeError("v417 projection block differs")
source = source.replace(projection_block, projection_replacement)

gate_block = '''    excess_hard, excess_soft = ((2.5, 5.0) if phase == "P" else (4.0, 8.0))
    path_gate = migration.smooth_rise(differential_time, excess_hard, excess_soft)
    coda_hard = station_pick + (2.5 if phase == "P" else 4.0)
    coda_soft = coda_hard + (2.0 if phase == "P" else 3.0)'''
gate_replacement = '''    gate_sigma = max(float(np.median([component.get("gate_sigma_s", 1.0) for component in components])), 0.25)
    excess_hard = max(0.75, 1.25 * gate_sigma)
    excess_soft = excess_hard + max(1.5, 2.0 * gate_sigma)
    path_gate = migration.smooth_rise(differential_time, excess_hard, excess_soft)
    coda_hard = station_pick + excess_hard
    coda_soft = station_pick + excess_soft'''
if source.count(gate_block) != 1:
    raise RuntimeError("v417 gate block differs")
source = source.replace(gate_block, gate_replacement)

namespace = {
    "__name__": __name__,
    "__file__": str(SOURCE),
    "dominant_direct_polarization": dominant_direct_polarization,
}
exec(compile(source, str(SOURCE) + "#direct-polarization-residual-v994", "exec"), namespace)
globals().update({key: value for key, value in namespace.items() if not key.startswith("__")})
