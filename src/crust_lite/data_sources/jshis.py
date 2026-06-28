from __future__ import annotations

import json
from typing import Any

from crust_lite.config import AppConfig
from crust_lite.io.parquet import write_table
from crust_lite.logging import get_logger
from crust_lite.paths import ProjectPaths

LOGGER = get_logger(__name__)


def fetch_jshis(config: AppConfig, paths: ProjectPaths, sample: bool = False) -> dict[str, Any]:
    paths.ensure()
    raw_dir = paths.data_raw / "jshis"
    raw_dir.mkdir(parents=True, exist_ok=True)
    if not config.data_sources.use_jshis and not sample:
        write_table([], paths.data_processed / "jshis_features.parquet", {"is_sample_data": False})
        return {"is_sample_data": False, "jshis_rows": 0}
    min_lon, min_lat, max_lon, max_lat = config.region.bbox
    payload: dict[str, Any] = {
        "source": "synthetic_sample",
        "note": "J-SHIS online API is not required for the MVP sample run.",
        "bbox": [min_lon, min_lat, max_lon, max_lat],
    }
    (raw_dir / "sample_jshis.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    rows = [
        {
            "layer_id": "sample_ground_hazard",
            "center_lon": (min_lon + max_lon) / 2.0,
            "center_lat": (min_lat + max_lat) / 2.0,
            "relative_hazard_score": 0.5,
            "source": "synthetic_sample",
            "is_sample_data": True,
        }
    ]
    write_table(
        rows,
        paths.data_processed / "jshis_features.parquet",
        {"is_sample_data": True, "source_note": "sample_jshis_placeholder"},
    )
    LOGGER.info("Wrote J-SHIS placeholder features for sample-capable run")
    return {"is_sample_data": True, "jshis_rows": len(rows)}
