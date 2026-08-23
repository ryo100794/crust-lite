"""Fail-closed resolver for observation-only fault inference v1107."""

from __future__ import annotations

from typing import Any

from crust_lite.config import AppConfig
from crust_lite.io.geopackage import read_features
from crust_lite.paths import ProjectPaths
from crust_lite.processing.fault_inference_known_free_v1107 import infer_faults as _infer_known_free


POLICY = "known-structure-free-normal-analysis-v1107"
FORBIDDEN_ANALYSIS_FIELDS = frozenset(
    {
        "distance_from_known_fault_score",
        "distance_to_known_fault_km",
        "nearest_known_segment_id",
        "strike_difference_to_known_deg",
        "known_fault_feedback_status",
        "known_fault_feedback_weight",
    }
)


def infer_faults(
    config: AppConfig,
    paths: ProjectPaths,
    *,
    mode: str = "known_free",
) -> dict[str, Any]:
    """Run only the observation-derived implementation and validate its output."""
    if mode != "known_free":
        raise ValueError(f"fail closed: unsupported fault inference mode {mode!r}")
    result = _infer_known_free(config, paths)
    output = paths.data_processed / "inferred_faults.gpkg"
    features = read_features(output)
    leaked = sorted(
        {
            key
            for feature in features
            for key in feature.get("properties", {})
            if key in FORBIDDEN_ANALYSIS_FIELDS
        }
    )
    if leaked:
        raise RuntimeError(f"known-structure fields leaked into normal output: {leaked}")
    if result.get("known_structure_input") is not False:
        raise RuntimeError("known-free inference did not attest known_structure_input=false")
    return {**result, "normal_analysis_policy": POLICY}
