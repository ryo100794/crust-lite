from __future__ import annotations

from crust_lite.cli import command_build_features, command_fetch
from crust_lite.cli_known_free_v1107 import command_infer_faults
from crust_lite.config import load_config
from crust_lite.io.geopackage import read_features
from crust_lite.paths import ProjectPaths
from crust_lite.processing.fault_inference_resolver_v1107 import FORBIDDEN_ANALYSIS_FIELDS
from tests.helpers import isolated_project


def test_versioned_cli_routes_normal_inference_to_known_free_resolver(tmp_path) -> None:
    config_path = str(isolated_project(tmp_path))
    command_fetch(config_path, sample=True)
    command_build_features(config_path)
    result = command_infer_faults(config_path)
    paths = ProjectPaths.from_config(load_config(config_path))
    features = read_features(paths.data_processed / "inferred_faults.gpkg")
    assert result["known_structure_input"] is False
    assert result["normal_analysis_policy"] == "known-structure-free-normal-analysis-v1107"
    assert features
    assert all(FORBIDDEN_ANALYSIS_FIELDS.isdisjoint(f["properties"]) for f in features)
