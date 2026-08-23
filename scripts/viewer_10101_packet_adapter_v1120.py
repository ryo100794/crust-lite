#!/usr/bin/env python3
"""Canonical packet adapter matching the active v1089 P/S/L byte contract.

No historical dashboard module is imported. The active v1089 data selection,
v1049 normalization/quantization, and v987 Natural Earth land geometry are
expressed directly for the review-only canonical server.
"""
from __future__ import annotations

import json
import math
import threading
from pathlib import Path

import numpy as np


PROJECT = Path("/workspace/equake/crust-lite")
HERE = Path(__file__).resolve().parent
EVENT = "hinet_20260802000524"
MODEL = (
    PROJECT
    / "data/interim/phase_aware_gs_20260811/recovery_v360"
    / "hinet_event_ps_fused_overlap_evidence_dev_v1076"
    / f"{EVENT}_PS_fused_overlap_evidence_v1076.npz"
)
POLYGONS = PROJECT / "outputs/3d/japan_naturalearth_10m_v987.json"
META = HERE / "viewer_10101_canonical_meta_v1120.json"
DTYPE = np.dtype(
    [("center", "<i2", (3,)), ("axes", "i1", (3, 3)), ("strength", "u1")]
)
LOCK = threading.RLock()
CACHE: dict[str, bytes] = {}


def load_meta() -> dict:
    data = json.loads(META.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("canonical meta is not an object")
    return data


def _mercator(lon: float, lat: float) -> tuple[float, float]:
    radius = 6_378_137.0
    return (
        radius * math.radians(lon),
        radius * math.log(math.tan(math.pi / 4.0 + math.radians(lat) / 2.0)),
    )


def _point_in_polygon(
    x: float, y: float, polygon: list[tuple[float, float]]
) -> bool:
    inside = False
    previous = polygon[-1]
    for current in polygon:
        x1, y1 = previous
        x2, y2 = current
        if (y1 > y) != (y2 > y):
            if x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
                inside = not inside
        previous = current
    return inside


def _land_segments() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    payload = json.loads(POLYGONS.read_text(encoding="utf-8"))
    rings = [
        [_mercator(float(lon), float(lat)) for lon, lat in ring]
        for ring in payload["rings_lon_lat"]
    ]
    centers: list[tuple[float, float, float]] = []
    axes_u: list[tuple[float, float, float]] = []
    axes_v: list[tuple[float, float, float]] = []
    spacing = 7000.0
    radius = 5000.0
    for polygon in rings:
        west, east = min(p[0] for p in polygon), max(p[0] for p in polygon)
        south, north = min(p[1] for p in polygon), max(p[1] for p in polygon)
        if east - west >= spacing and north - south >= spacing:
            for x in np.arange(
                math.floor(west / spacing) * spacing, east + spacing, spacing
            ):
                for y in np.arange(
                    math.floor(south / spacing) * spacing, north + spacing, spacing
                ):
                    if _point_in_polygon(float(x), float(y), polygon):
                        centers.append((float(x), float(y), -900.0))
                        axes_u.append((radius, 0.0, 0.0))
                        axes_v.append((0.0, radius, 0.0))
        for a, b in zip(polygon[:-1], polygon[1:]):
            dx, dy = (b[0] - a[0]) / 2.0, (b[1] - a[1]) / 2.0
            length = max(math.hypot(dx, dy), 1.0)
            centers.append(
                ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0, -1800.0)
            )
            axes_u.append((dx, dy, 0.0))
            axes_v.append((-dy / length * 1700.0, dx / length * 1700.0, 0.0))
    return (
        np.asarray(centers, np.float32),
        np.asarray(axes_u, np.float32),
        np.asarray(axes_v, np.float32),
    )


def _land() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    center, axis_u, axis_v = _land_segments()
    axis_w = np.zeros_like(axis_u)
    axis_w[:, 2] = 500.0
    return (
        center,
        np.stack((axis_u, axis_v, axis_w), axis=1),
        np.ones(len(center), np.float32),
    )


def _phase(phase: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with LOCK, np.load(MODEL, allow_pickle=False) as data:
        center = np.asarray(data["center_xyz_m"], np.float32)
        axes = np.stack(
            [
                np.asarray(data["axis_u_m"], np.float32),
                np.asarray(data["axis_v_m"], np.float32),
                np.asarray(data["axis_w_m"], np.float32),
            ],
            axis=1,
        )
        support = np.asarray(data["phase_support"], np.uint8)
        significance = np.asarray(data["splat_significance"], np.float32)
        reliability = np.asarray(data["aperture_reliability"], np.float32)
    selected = support == 3 if phase == "P" else support != 3
    raw = np.log1p(np.maximum(significance, 0.0)) * np.sqrt(
        np.maximum(reliability, 0.0)
    )
    finite = np.isfinite(raw)
    lo, hi = (
        np.quantile(raw[finite], [0.02, 0.98])
        if np.any(finite)
        else (0.0, 1.0)
    )
    display = np.clip((raw - lo) / max(float(hi - lo), 1e-6), 0.0, 1.0) ** 0.72
    if phase == "S":
        display *= 0.70
    return center[selected], axes[selected], display[selected].astype(np.float32)


def _pack(
    center: np.ndarray, axes: np.ndarray, strength: np.ndarray, meta: dict
) -> bytes:
    origin = np.asarray(meta["origin_xyz_m"], np.float32)
    packed = np.zeros(len(center), dtype=DTYPE)
    center_step = np.asarray([1000.0, 1000.0, 500.0], np.float32)
    packed["center"] = np.clip(
        np.rint((center - origin) / center_step), -32768, 32767
    ).astype("<i2")
    packed["axes"] = np.clip(np.rint(axes / 500.0), -127, 127).astype("i1")
    packed["strength"] = np.rint(np.clip(strength, 0, 1) * 255).astype("u1")
    return packed.tobytes()


def packet(phase: str) -> bytes:
    if phase not in {"P", "S", "L"}:
        raise ValueError(f"unsupported phase: {phase}")
    with LOCK:
        if phase not in CACHE:
            arrays = _land() if phase == "L" else _phase(phase)
            CACHE[phase] = _pack(*arrays, load_meta())
        return CACHE[phase]

