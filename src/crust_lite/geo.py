from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np

from crust_lite.config import RegionConfig

EARTH_RADIUS_M = 6_371_000.0


@dataclass(frozen=True)
class LocalProjector:
    region: RegionConfig
    _transformer: object | None = field(init=False, repr=False)
    _lon0: float = field(init=False, repr=False)
    _lat0: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_transformer", self._make_transformer())
        min_lon, min_lat, max_lon, max_lat = self.region.bbox
        object.__setattr__(self, "_lon0", (min_lon + max_lon) / 2.0)
        object.__setattr__(self, "_lat0", (min_lat + max_lat) / 2.0)

    def _make_transformer(self) -> object | None:
        try:
            from pyproj import Transformer  # type: ignore

            return Transformer.from_crs("EPSG:4326", self.region.crs_local, always_xy=True)
        except Exception:
            return None

    def lonlat_to_xy(self, lon: float, lat: float) -> tuple[float, float]:
        if self._transformer is not None:
            x, y = self._transformer.transform(lon, lat)  # type: ignore[attr-defined]
            return float(x), float(y)
        lon0 = math.radians(self._lon0)
        lat0 = math.radians(self._lat0)
        x = EARTH_RADIUS_M * (math.radians(lon) - lon0) * math.cos(lat0)
        y = EARTH_RADIUS_M * (math.radians(lat) - lat0)
        return x, y

    def line_lonlat_to_xy(self, coordinates: Iterable[Iterable[float]]) -> list[tuple[float, float]]:
        return [self.lonlat_to_xy(float(lon), float(lat)) for lon, lat in coordinates]


def in_bbox(lon: float, lat: float, bbox: tuple[float, float, float, float]) -> bool:
    min_lon, min_lat, max_lon, max_lat = bbox
    return min_lon <= lon <= max_lon and min_lat <= lat <= max_lat


def clamp01(value: float) -> float:
    if math.isnan(value) or math.isinf(value):
        return 0.0
    return max(0.0, min(1.0, value))


def vector_to_strike_dip(normal: np.ndarray) -> tuple[float, float]:
    n = normal / (np.linalg.norm(normal) + 1e-12)
    if n[2] < 0:
        n = -n
    dip = math.degrees(math.acos(clamp01(abs(n[2]))))
    strike_vector = np.array([-n[1], n[0], 0.0])
    if np.linalg.norm(strike_vector) < 1e-9:
        strike = 0.0
    else:
        strike = math.degrees(math.atan2(strike_vector[0], strike_vector[1])) % 360.0
    return strike, max(0.0, min(90.0, dip))


def angle_difference_deg(a: float, b: float) -> float:
    diff = abs((a - b + 180.0) % 360.0 - 180.0)
    return min(diff, 360.0 - diff)


def polyline_length_km(points_xy: list[tuple[float, float]]) -> float:
    if len(points_xy) < 2:
        return 0.0
    total = 0.0
    for (x0, y0), (x1, y1) in zip(points_xy[:-1], points_xy[1:], strict=False):
        total += math.hypot(x1 - x0, y1 - y0)
    return total / 1000.0


def distance_point_to_segment_m(
    point: tuple[float, float], start: tuple[float, float], end: tuple[float, float]
) -> float:
    px, py = point
    x0, y0 = start
    x1, y1 = end
    vx, vy = x1 - x0, y1 - y0
    denom = vx * vx + vy * vy
    if denom == 0:
        return math.hypot(px - x0, py - y0)
    t = max(0.0, min(1.0, ((px - x0) * vx + (py - y0) * vy) / denom))
    return math.hypot(px - (x0 + t * vx), py - (y0 + t * vy))


def distance_to_polyline_km(point: tuple[float, float], line: list[tuple[float, float]]) -> float:
    if len(line) < 2:
        return float("inf")
    return min(distance_point_to_segment_m(point, a, b) for a, b in zip(line[:-1], line[1:], strict=False)) / 1000.0


def fault_rectangle_vertices(
    center_x_m: float,
    center_y_m: float,
    center_depth_km: float,
    strike: float,
    dip: float,
    length_km: float,
    width_km: float,
) -> np.ndarray:
    strike_rad = math.radians(strike)
    dip_rad = math.radians(max(1.0, min(89.0, dip)))
    along = np.array([math.sin(strike_rad), math.cos(strike_rad), 0.0])
    down_dip_az = strike_rad + math.pi / 2.0
    down = np.array(
        [
            math.sin(down_dip_az) * math.cos(dip_rad),
            math.cos(down_dip_az) * math.cos(dip_rad),
            math.sin(dip_rad),
        ]
    )
    center = np.array([center_x_m, center_y_m, center_depth_km * 1000.0])
    half_l = length_km * 500.0
    half_w = width_km * 500.0
    return np.array(
        [
            center - along * half_l - down * half_w,
            center + along * half_l - down * half_w,
            center + along * half_l + down * half_w,
            center - along * half_l + down * half_w,
        ]
    )
