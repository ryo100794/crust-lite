from __future__ import annotations

import json
from pathlib import Path
from typing import Any

Feature = dict[str, Any]


def write_features(
    features: list[Feature], path: Path, metadata: dict[str, Any] | None = None
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    collection = {
        "type": "FeatureCollection",
        "features": features,
        "metadata": {
            "physical_format": "geojson_fallback_content",
            **(metadata or {}),
        },
    }
    path.write_text(json.dumps(collection, indent=2, sort_keys=True, default=str), encoding="utf-8")
    path.with_suffix(path.suffix + ".metadata.json").write_text(
        json.dumps(collection["metadata"], indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )


def read_features(path: Path) -> list[Feature]:
    if not path.exists():
        raise FileNotFoundError(f"Required geospatial layer not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("type") != "FeatureCollection":
        raise ValueError(f"{path} is not a FeatureCollection")
    return list(data.get("features", []))


def read_metadata(path: Path) -> dict[str, Any]:
    sidecar = path.with_suffix(path.suffix + ".metadata.json")
    if sidecar.exists():
        return json.loads(sidecar.read_text(encoding="utf-8"))
    data = json.loads(path.read_text(encoding="utf-8"))
    return dict(data.get("metadata", {}))
