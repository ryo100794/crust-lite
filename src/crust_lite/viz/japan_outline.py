from __future__ import annotations

import logging
import math
from functools import lru_cache
from typing import Any, NotRequired, TypedDict

LOGGER = logging.getLogger(__name__)
EARTH_RADIUS_KM = 6371.0088


class JapanOutline(TypedDict):
    name: str
    coordinates: list[tuple[float, float]]
    source: NotRequired[str]
    target_segment_km: NotRequired[float]


# Simplified offline outlines for cartographic context only. These coordinates are
# intentionally coarse and must not be used as analytical coastlines.
JAPAN_ARCHIPELAGO_OUTLINES: list[JapanOutline] = [
    {
        "name": "Hokkaido",
        "coordinates": [
            (140.65, 41.42),
            (141.38, 41.35),
            (142.28, 41.63),
            (143.22, 41.92),
            (144.03, 42.39),
            (145.36, 43.02),
            (145.82, 43.47),
            (145.35, 44.10),
            (144.10, 44.42),
            (142.95, 44.54),
            (141.72, 45.02),
            (141.00, 44.36),
            (140.72, 43.55),
            (140.45, 42.75),
            (140.65, 41.42),
        ],
    },
    {
        "name": "Honshu",
        "coordinates": [
            (140.95, 41.52),
            (141.42, 40.82),
            (141.82, 39.72),
            (141.55, 38.75),
            (141.15, 37.78),
            (140.98, 36.95),
            (140.52, 36.20),
            (139.80, 35.55),
            (139.15, 35.08),
            (138.30, 34.78),
            (137.10, 34.62),
            (135.85, 34.45),
            (134.62, 34.30),
            (133.42, 34.10),
            (132.18, 34.06),
            (130.92, 34.32),
            (130.88, 34.76),
            (131.72, 35.02),
            (132.65, 35.25),
            (133.80, 35.52),
            (134.90, 35.54),
            (135.95, 35.72),
            (136.55, 36.25),
            (137.20, 36.75),
            (137.78, 37.42),
            (138.55, 37.88),
            (139.15, 38.55),
            (139.83, 39.25),
            (140.33, 40.18),
            (140.72, 40.95),
            (140.95, 41.52),
        ],
    },
    {
        "name": "Shikoku",
        "coordinates": [
            (132.02, 33.22),
            (132.82, 33.05),
            (133.75, 33.20),
            (134.72, 33.48),
            (134.82, 33.90),
            (134.05, 34.28),
            (132.95, 34.28),
            (132.15, 33.92),
            (132.02, 33.22),
        ],
    },
    {
        "name": "Kyushu",
        "coordinates": [
            (129.20, 31.12),
            (129.78, 31.20),
            (130.55, 31.25),
            (131.02, 31.70),
            (131.15, 32.35),
            (130.98, 32.92),
            (130.52, 33.42),
            (129.95, 33.72),
            (129.56, 33.35),
            (129.72, 32.82),
            (129.92, 32.32),
            (129.55, 31.82),
            (129.20, 31.12),
        ],
    },
    {
        "name": "Sado",
        "coordinates": [
            (138.18, 37.80),
            (138.48, 37.95),
            (138.58, 38.25),
            (138.32, 38.34),
            (138.02, 38.10),
            (138.18, 37.80),
        ],
    },
    {
        "name": "Tsushima",
        "coordinates": [
            (129.15, 34.08),
            (129.35, 34.22),
            (129.45, 34.58),
            (129.25, 34.72),
            (129.05, 34.40),
            (129.15, 34.08),
        ],
    },
    {
        "name": "Amami_Okinawa_chain",
        "coordinates": [
            (127.65, 26.05),
            (127.95, 26.40),
            (128.25, 26.75),
            (128.65, 27.35),
            (129.25, 28.25),
            (129.72, 28.90),
            (130.02, 29.55),
            (130.35, 30.25),
        ],
    },
]


def _haversine_km(start: tuple[float, float], end: tuple[float, float]) -> float:
    lon1, lat1 = start
    lon2, lat2 = end
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    a = math.sin(d_phi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_KM * math.atan2(math.sqrt(a), math.sqrt(max(0.0, 1.0 - a)))


def _densify_lonlat(coords: list[tuple[float, float]], max_segment_km: float) -> list[tuple[float, float]]:
    if len(coords) < 2:
        return coords
    max_segment_km = max(0.1, max_segment_km)
    dense: list[tuple[float, float]] = [coords[0]]
    for start, end in zip(coords, coords[1:], strict=False):
        distance_km = _haversine_km(start, end)
        steps = max(1, int(math.ceil(distance_km / max_segment_km)))
        lon1, lat1 = start
        lon2, lat2 = end
        for step in range(1, steps + 1):
            fraction = step / steps
            dense.append((lon1 + (lon2 - lon1) * fraction, lat1 + (lat2 - lat1) * fraction))
    return dense


def _geometry_polygons(geometry: Any) -> list[Any]:
    geom_type = getattr(geometry, "geom_type", "")
    if geom_type == "Polygon":
        return [geometry]
    if geom_type == "MultiPolygon":
        return list(getattr(geometry, "geoms", []))
    return []


@lru_cache(maxsize=8)
def _natural_earth_japan_outlines(max_segment_km: float) -> tuple[JapanOutline, ...]:
    """Load Japan coast outlines from Natural Earth 10m and densify visually.

    Natural Earth 10m is cartographic data, not an analytical 1 km coastline.
    The densification below makes adjacent rendered vertices roughly 1 km apart
    so the WebGL context no longer looks faceted when zoomed.
    """
    try:
        import cartopy.io.shapereader as shpreader
    except Exception as exc:  # pragma: no cover - optional dependency fallback
        LOGGER.warning("Cannot load Natural Earth coastline because cartopy is unavailable: %s", exc)
        return ()
    try:
        path = shpreader.natural_earth(resolution="10m", category="cultural", name="admin_0_countries")
        reader = shpreader.Reader(path)
        outlines: list[JapanOutline] = []
        for record in reader.records():
            attrs = record.attributes
            if not (
                attrs.get("ADMIN") == "Japan"
                or attrs.get("NAME") == "Japan"
                or attrs.get("SOVEREIGNT") == "Japan"
            ):
                continue
            for index, polygon in enumerate(_geometry_polygons(record.geometry)):
                coords = [(float(lon), float(lat)) for lon, lat in polygon.exterior.coords]
                if len(coords) < 4:
                    continue
                outlines.append(
                    {
                        "name": f"Japan_NE10m_{index:03d}",
                        "coordinates": _densify_lonlat(coords, max_segment_km=max_segment_km),
                        "source": "natural_earth_10m_admin_0_japan",
                        "target_segment_km": max_segment_km,
                    }
                )
        if outlines:
            return tuple(outlines)
    except Exception as exc:  # pragma: no cover - network/cache dependent fallback
        LOGGER.warning("Cannot load Natural Earth Japan coastline; using offline coarse fallback: %s", exc)
    return ()


def _outline_bounds(outline: JapanOutline) -> tuple[float, float, float, float]:
    coords = outline["coordinates"]
    lons = [coord[0] for coord in coords]
    lats = [coord[1] for coord in coords]
    return min(lons), min(lats), max(lons), max(lats)


def outline_intersects_bbox(outline: JapanOutline, bbox: tuple[float, float, float, float], margin_deg: float) -> bool:
    min_lon, min_lat, max_lon, max_lat = bbox
    min_lon -= margin_deg
    min_lat -= margin_deg
    max_lon += margin_deg
    max_lat += margin_deg
    outline_min_lon, outline_min_lat, outline_max_lon, outline_max_lat = _outline_bounds(outline)
    return not (
        outline_max_lon < min_lon
        or outline_min_lon > max_lon
        or outline_max_lat < min_lat
        or outline_min_lat > max_lat
    )


def local_context_outlines(
    bbox: tuple[float, float, float, float],
    margin_deg: float = 2.5,
    target_segment_km: float = 1.0,
    prefer_high_resolution: bool = True,
) -> list[JapanOutline]:
    source_outlines = (
        list(_natural_earth_japan_outlines(round(target_segment_km, 3)))
        if prefer_high_resolution
        else []
    )
    if not source_outlines:
        source_outlines = JAPAN_ARCHIPELAGO_OUTLINES
    selected = [
        outline
        for outline in source_outlines
        if outline_intersects_bbox(outline, bbox, margin_deg=margin_deg)
    ]
    return selected or source_outlines
