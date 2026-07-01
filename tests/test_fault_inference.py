from __future__ import annotations

from crust_lite.cli import command_build_features, command_fetch, command_infer_faults
from crust_lite.config import load_config
from crust_lite.io.geopackage import read_features
from crust_lite.paths import ProjectPaths
from tests.helpers import isolated_project


def test_fault_inference_from_sample_points(tmp_path) -> None:
    config_path = str(isolated_project(tmp_path))
    command_fetch(config_path, sample=True)
    command_build_features(config_path)
    result = command_infer_faults(config_path)
    cfg = load_config(config_path)
    paths = ProjectPaths.from_config(cfg)
    features = read_features(paths.data_processed / "inferred_faults.gpkg")
    assert result["inferred_fault_count"] >= 1
    assert features
    assert 0.0 <= float(features[0]["properties"]["fault_score"]) <= 1.0
