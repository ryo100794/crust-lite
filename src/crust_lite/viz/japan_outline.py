from __future__ import annotations

from typing import TypedDict


class JapanOutline(TypedDict):
    name: str
    coordinates: list[tuple[float, float]]


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


def outline_intersects_bbox(outline: JapanOutline, bbox: tuple[float, float, float, float], margin_deg: float) -> bool:
    min_lon, min_lat, max_lon, max_lat = bbox
    min_lon -= margin_deg
    min_lat -= margin_deg
    max_lon += margin_deg
    max_lat += margin_deg
    return any(min_lon <= lon <= max_lon and min_lat <= lat <= max_lat for lon, lat in outline["coordinates"])


def local_context_outlines(bbox: tuple[float, float, float, float], margin_deg: float = 2.5) -> list[JapanOutline]:
    selected = [
        outline
        for outline in JAPAN_ARCHIPELAGO_OUTLINES
        if outline_intersects_bbox(outline, bbox, margin_deg=margin_deg)
    ]
    return selected or JAPAN_ARCHIPELAGO_OUTLINES
