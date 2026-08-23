#!/usr/bin/env python3
"""Minimal same-input comparison of legacy known leakage and v1107."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from crust_lite.cli import command_build_features, command_fetch
from crust_lite.config import load_config
from crust_lite.io.geopackage import read_features, write_features
from crust_lite.paths import ProjectPaths
from crust_lite.processing.fault_inference import infer_faults as infer_legacy
from crust_lite.processing.fault_inference_resolver_v1107 import (
    FORBIDDEN_ANALYSIS_FIELDS,
    infer_faults as infer_known_free,
)
from tests.helpers import isolated_project


def reference(segment_id: str, coordinates: list[list[float]]) -> list[dict]:
    return [
        {
            "type": "Feature",
            "properties": {"segment_id": segment_id},
            "geometry": {"type": "LineString", "coordinates": coordinates},
        }
    ]


def snapshot(paths: ProjectPaths) -> dict:
    path = paths.data_processed / "inferred_faults.gpkg"
    features = read_features(path)
    scores = [float(x["properties"]["fault_score"]) for x in features]
    return {
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "features": len(features),
        "score_min": min(scores),
        "score_max": max(scores),
        "score_mean": sum(scores) / len(scores),
        "forbidden_field_rows": sum(
            bool(FORBIDDEN_ANALYSIS_FIELDS.intersection(x["properties"])) for x in features
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="nr-known-leak-008-") as raw:
        root = Path(raw)
        config_path = isolated_project(root)
        command_fetch(str(config_path), sample=True)
        command_build_features(str(config_path))
        config = load_config(config_path)
        paths = ProjectPaths.from_config(config)
        known = paths.data_processed / "fault_segment.gpkg"

        near = reference("near", [[130.70, 32.75], [130.90, 32.95]])
        far = reference("far", [[145.0, 44.0], [146.0, 45.0]])

        write_features(near, known, {"classification": "external_geology_reference"})
        legacy_near_result = infer_legacy(config, paths)
        legacy_near = snapshot(paths)
        write_features(far, known, {"classification": "external_geology_reference"})
        legacy_far_result = infer_legacy(config, paths)
        legacy_far = snapshot(paths)

        write_features(near, known, {"classification": "external_geology_reference"})
        clean_near_result = infer_known_free(config, paths)
        clean_near = snapshot(paths)
        write_features(far, known, {"classification": "external_geology_reference"})
        clean_far_result = infer_known_free(config, paths)
        clean_far = snapshot(paths)

    result = {
        "schema": "nr-known-leak-008-minimal-comparison-v1107",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "fixture": "same bundled Kumamoto sample observations; only external known-fault geometry changed",
        "legacy_near": legacy_near,
        "legacy_far": legacy_far,
        "known_free_near": clean_near,
        "known_free_far": clean_far,
        "legacy_result": {"near": legacy_near_result, "far": legacy_far_result},
        "known_free_result": {"near": clean_near_result, "far": clean_far_result},
        "checks": {
            "legacy_output_depends_on_known_geometry": legacy_near["sha256"] != legacy_far["sha256"],
            "known_free_output_independent_of_known_geometry": clean_near["sha256"] == clean_far["sha256"],
            "known_free_forbidden_field_rows_zero": clean_near["forbidden_field_rows"] == 0,
            "legacy_forbidden_fields_present": legacy_near["forbidden_field_rows"] > 0,
        },
    }
    result["pass"] = all(result["checks"].values())
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["pass"] else 1)


if __name__ == "__main__":
    main()
