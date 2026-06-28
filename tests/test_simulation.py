from __future__ import annotations

from crust_lite.cli import (
    command_build_features,
    command_fetch,
    command_infer_faults,
    command_simulate,
    command_stress,
)
from crust_lite.config import load_config
from crust_lite.io.parquet import read_table
from crust_lite.paths import ProjectPaths


def test_simulation_columns() -> None:
    config_path = "configs/kumamoto.yml"
    command_fetch(config_path, sample=True)
    command_build_features(config_path)
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
