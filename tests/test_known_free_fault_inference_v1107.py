from __future__ import annotations

import hashlib

import pytest

from crust_lite.cli import command_build_features
from crust_lite.config import load_config
from crust_lite.io.geopackage import read_features, write_features
from crust_lite.paths import ProjectPaths
from crust_lite.processing.fault_inference_resolver_v1107 import (
    FORBIDDEN_ANALYSIS_FIELDS,
    infer_faults,
)
from crust_lite.processing.scoring_known_free_v1107 import FAULT_SCORE_WEIGHTS, fault_score
from tests.helpers import (
    DEVELOPMENT_FIXTURE_MARKER,
    isolated_project,
    materialize_observation_only_development_fixture,
)


def test_fault_score_is_observation_only_and_normalized() -> None:
    assert sum(FAULT_SCORE_WEIGHTS.values()) == pytest.approx(1.0)
    assert fault_score(1.0, 0.0, 0.0, 0.0) == pytest.approx(7.0 / 18.0)
    with pytest.raises(TypeError):
        fault_score(1.0, 1.0, 1.0, 1.0, 1.0)  # type: ignore[call-arg]


def test_normal_inference_is_invariant_to_known_fault_geometry(tmp_path) -> None:
    config_path = str(isolated_project(tmp_path))
    fixture = materialize_observation_only_development_fixture(config_path)
    assert fixture["status"] == "DEVELOPMENT_FIXTURE_ONLY"
    assert fixture["network_requests"] == 0
    assert fixture["known_structure_artifacts_written"] == 0
    command_build_features(config_path)
    config = load_config(config_path)
    paths = ProjectPaths.from_config(config)
    known_path = paths.data_processed / "fault_segment.gpkg"

    write_features(
        [
            {
                "type": "Feature",
                "properties": {"segment_id": "deliberately_near"},
                "geometry": {"type": "LineString", "coordinates": [[130.7, 32.8], [130.8, 32.9]]},
            }
        ],
        known_path,
        {"classification": "external_geology_reference"},
    )
    first = infer_faults(config, paths)
    first_bytes = (paths.data_processed / "inferred_faults.gpkg").read_bytes()
    first_features = read_features(paths.data_processed / "inferred_faults.gpkg")

    write_features(
        [
            {
                "type": "Feature",
                "properties": {"segment_id": "deliberately_far"},
                "geometry": {"type": "LineString", "coordinates": [[145.0, 44.0], [146.0, 45.0]]},
            }
        ],
        known_path,
        {"classification": "external_geology_reference"},
    )
    second = infer_faults(config, paths)
    second_bytes = (paths.data_processed / "inferred_faults.gpkg").read_bytes()
    second_features = read_features(paths.data_processed / "inferred_faults.gpkg")

    assert first["normal_analysis_policy"] == "known-structure-free-normal-analysis-v1107"
    assert first == second
    assert hashlib.sha256(first_bytes).hexdigest() == hashlib.sha256(second_bytes).hexdigest()
    assert first_features == second_features
    assert all(FORBIDDEN_ANALYSIS_FIELDS.isdisjoint(f["properties"]) for f in first_features)


def test_development_fixture_requires_explicit_isolated_marker(tmp_path) -> None:
    config_path = isolated_project(tmp_path)
    (tmp_path / DEVELOPMENT_FIXTURE_MARKER).unlink()
    with pytest.raises(RuntimeError, match="explicit isolated-project marker"):
        materialize_observation_only_development_fixture(config_path)


def test_resolver_fails_closed_for_legacy_mode(tmp_path) -> None:
    config = load_config(str(isolated_project(tmp_path)))
    with pytest.raises(ValueError, match="fail closed"):
        infer_faults(config, ProjectPaths.from_config(config), mode="legacy")
