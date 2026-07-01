from __future__ import annotations

from crust_lite.cli import command_run_all
from crust_lite.config import load_config
from crust_lite.io.metadata import read_metadata
from crust_lite.paths import ProjectPaths
from crust_lite.viz.visualize_3d import plot_z_m
from tests.helpers import isolated_project


def test_run_all_sample_and_3d_outputs(tmp_path) -> None:
    config_path = str(isolated_project(tmp_path))
    command_run_all(config_path, sample=True)
    paths = ProjectPaths.from_config(load_config(config_path))
    events_html = paths.outputs_3d / "events_faults_timeseries.html"
    stress_html = paths.outputs_3d / "stress_timeseries_3d.html"
    failure_html = paths.outputs_3d / "failure_scenarios_3d.html"
    metadata_json = paths.outputs_3d / "metadata.json"
    japan_html = paths.outputs_3d / "japan_archipelago_context.html"
    for path in [
        events_html,
        stress_html,
        failure_html,
        japan_html,
        metadata_json,
        paths.outputs_3d / "index.html",
    ]:
        assert path.exists()
    text = events_html.read_text(encoding="utf-8")
    assert "webgl2_event_fault_point_sprite" in text
    assert "getContext('webgl2'" in text
    assert "is_sample_data=true" in text
    assert "map overlay" in text
    failure_text = failure_html.read_text(encoding="utf-8")
    assert "地震発生" not in failure_text
    meta = read_metadata(metadata_json)
    assert meta["displayed_event_count"] > 0
    assert meta["actual_time_bin_days"] == 1
    assert meta["playback_frame_interval_ms"] == 80
    assert meta["map_overlay"] == "local_crs_bbox_graticule_surface_with_japan_archipelago_outline"
    assert meta["japan_context_file"] == "japan_archipelago_context.html"
    assert "Japan archipelago" in japan_html.read_text(encoding="utf-8")


def test_vertical_exaggeration_transform() -> None:
    assert plot_z_m(1000.0, 2.0) == -2000.0
