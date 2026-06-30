from __future__ import annotations

import math
from typing import Literal, TypedDict


class TectonicLine(TypedDict):
    name: str
    plate: str
    kind: str
    color: tuple[float, float, float, float]
    coordinates: list[tuple[float, float, float]]


class TectonicContext(TypedDict):
    boundaries: list[TectonicLine]
    interfaces: list[TectonicLine]
    source: str
    note: str


BoundaryKind = Literal["trench", "trough", "plate_interface"]


PACIFIC_COLOR = (0.10, 0.70, 1.00, 0.58)
PHILIPPINE_SEA_COLOR = (0.95, 0.82, 0.20, 0.58)
BOUNDARY_COLOR = (0.35, 0.95, 1.00, 0.86)
TROUGH_COLOR = (1.00, 0.82, 0.25, 0.86)


JAPAN_TRENCH = [
    (145.4, 44.4),
    (144.8, 43.2),
    (144.2, 42.0),
    (143.9, 40.5),
    (143.5, 39.0),
    (143.0, 37.5),
    (142.3, 36.0),
    (141.6, 34.7),
]


IZU_BONIN_TRENCH = [
    (141.6, 34.7),
    (142.0, 33.0),
    (142.4, 31.2),
    (142.8, 29.5),
    (143.3, 27.7),
    (143.9, 25.8),
]


NANKAI_TROUGH = [
    (131.2, 31.8),
    (132.6, 32.3),
    (134.1, 32.8),
    (135.6, 33.1),
    (137.1, 33.5),
    (138.8, 34.2),
    (140.0, 34.9),
]


RYUKYU_TRENCH = [
    (122.8, 23.8),
    (124.7, 24.4),
    (126.5, 25.3),
    (128.3, 26.4),
    (130.0, 28.1),
    (131.2, 31.8),
]


def _line_with_depth(coords: list[tuple[float, float]], depth_km: float = 0.0) -> list[tuple[float, float, float]]:
    return [(lon, lat, depth_km) for lon, lat in coords]


def _offset_lonlat(lon: float, lat: float, east_km: float, north_km: float) -> tuple[float, float]:
    cos_lat = max(0.1, math.cos(math.radians(lat)))
    return lon + east_km / (111.32 * cos_lat), lat + north_km / 110.57


def _densify_polyline(coords: list[tuple[float, float]], samples_per_segment: int) -> list[tuple[float, float]]:
    if len(coords) < 2:
        return coords
    dense: list[tuple[float, float]] = []
    for start, end in zip(coords, coords[1:], strict=False):
        lon1, lat1 = start
        lon2, lat2 = end
        for step in range(samples_per_segment):
            f = step / samples_per_segment
            dense.append((lon1 + (lon2 - lon1) * f, lat1 + (lat2 - lat1) * f))
    dense.append(coords[-1])
    return dense


def _surface_wire_from_trench(
    name: str,
    plate: str,
    trench: list[tuple[float, float]],
    color: tuple[float, float, float, float],
    *,
    dip_east_km_per_step: float,
    dip_north_km_per_step: float,
    depth_start_km: float,
    depth_step_km: float,
    down_dip_steps: int,
    along_samples_per_segment: int,
) -> list[TectonicLine]:
    along = _densify_polyline(trench, along_samples_per_segment)
    grid: list[list[tuple[float, float, float]]] = []
    for i in range(down_dip_steps + 1):
        row: list[tuple[float, float, float]] = []
        for lon, lat in along:
            xlon, xlat = _offset_lonlat(lon, lat, dip_east_km_per_step * i, dip_north_km_per_step * i)
            row.append((xlon, xlat, depth_start_km + depth_step_km * i))
        grid.append(row)

    lines: list[TectonicLine] = []
    for i, row in enumerate(grid):
        lines.append(
            {
                "name": f"{name} depth-contour-{i:02d}",
                "plate": plate,
                "kind": "plate_interface",
                "color": color,
                "coordinates": row,
            }
        )
    column_step = max(1, len(along) // 18)
    for j in range(0, len(along), column_step):
        lines.append(
            {
                "name": f"{name} downdip-{j:03d}",
                "plate": plate,
                "kind": "plate_interface",
                "color": color,
                "coordinates": [grid[i][j] for i in range(len(grid))],
            }
        )
    return lines


def japan_tectonic_context() -> TectonicContext:
    """Return a lightweight Japan plate-tectonic context for visualization.

    This is a schematic overlay for orientation. It is not a Slab2/J-SHIS/GSI
    analytical plate model and must not be used as a quantitative interface.
    """
    boundaries: list[TectonicLine] = [
        {
            "name": "Japan Trench",
            "plate": "Pacific plate subduction boundary",
            "kind": "trench",
            "color": BOUNDARY_COLOR,
            "coordinates": _line_with_depth(JAPAN_TRENCH),
        },
        {
            "name": "Izu-Bonin Trench",
            "plate": "Pacific plate subduction boundary",
            "kind": "trench",
            "color": BOUNDARY_COLOR,
            "coordinates": _line_with_depth(IZU_BONIN_TRENCH),
        },
        {
            "name": "Nankai Trough",
            "plate": "Philippine Sea plate subduction boundary",
            "kind": "trough",
            "color": TROUGH_COLOR,
            "coordinates": _line_with_depth(NANKAI_TROUGH),
        },
        {
            "name": "Ryukyu Trench",
            "plate": "Philippine Sea plate subduction boundary",
            "kind": "trench",
            "color": TROUGH_COLOR,
            "coordinates": _line_with_depth(RYUKYU_TRENCH),
        },
    ]
    interfaces = []
    interfaces.extend(
        _surface_wire_from_trench(
            "Pacific slab schematic",
            "Pacific plate",
            [*JAPAN_TRENCH, *IZU_BONIN_TRENCH[1:]],
            PACIFIC_COLOR,
            dip_east_km_per_step=-45.0,
            dip_north_km_per_step=4.0,
            depth_start_km=12.0,
            depth_step_km=18.0,
            down_dip_steps=13,
            along_samples_per_segment=5,
        )
    )
    interfaces.extend(
        _surface_wire_from_trench(
            "Philippine Sea slab schematic",
            "Philippine Sea plate",
            [*RYUKYU_TRENCH, *NANKAI_TROUGH[1:]],
            PHILIPPINE_SEA_COLOR,
            dip_east_km_per_step=-23.0,
            dip_north_km_per_step=18.0,
            depth_start_km=8.0,
            depth_step_km=9.0,
            down_dip_steps=10,
            along_samples_per_segment=5,
        )
    )
    return {
        "boundaries": boundaries,
        "interfaces": interfaces,
        "source": "schematic_japan_plate_context_v0",
        "note": "Schematic plate boundaries and slab-interface wireframes for visual context only; not a quantitative plate model.",
    }
