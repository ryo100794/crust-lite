from __future__ import annotations

from crust_lite.cli import (
    command_array_projection,
    command_build_features,
    command_fetch,
    command_infer_faults,
    command_shallow_lineaments,
    command_simulate,
    command_stress,
)
from crust_lite.config import load_config
from crust_lite.io.parquet import read_table
from crust_lite.paths import ProjectPaths
from crust_lite.processing.stress import _feature_center
from tests.helpers import isolated_project


def test_simulation_columns(tmp_path) -> None:
    config_path = str(isolated_project(tmp_path))
    command_fetch(config_path, sample=True)
    command_build_features(config_path)
    command_array_projection(config_path, sample=True)
    command_shallow_lineaments(config_path)
    command_infer_faults(config_path)
    command_stress(config_path)
    command_simulate(config_path)
    paths = ProjectPaths.from_config(load_config(config_path))
    rows = read_table(paths.outputs_tables / "failure_scenarios.parquet")
    assert rows
    expected = {
        "segment_id",
        "year",
        "failure_index_p05",
        "failure_index_p50",
        "failure_index_p95",
        "prob_index_gt_1",
        "stress_rate_pa_per_yr",
        "uncertainty_score",
        "simulation_notes",
    }
    assert expected.issubset(rows[0])

def test_stress_feature_center_handles_missing_known_fault_depth() -> None:
    center = _feature_center(
        {
            "properties": {
                "segment_id": "known_without_depth",
                "center_x_m": 1.0,
                "center_y_m": 2.0,
                "center_depth_km": None,
                "bottom_depth_km": None,
            }
        }
    )

    assert center == (1.0, 2.0, 5000.0)
