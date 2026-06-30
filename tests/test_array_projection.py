from __future__ import annotations

from pathlib import Path
from shutil import copy2, copytree

from crust_lite.cli import (
    command_array_projection,
    command_build_features,
    command_fetch,
    command_transfer_functions,
)
from crust_lite.config import load_config
from crust_lite.io.parquet import read_sidecar, read_table
from crust_lite.paths import ProjectPaths


def _sample_project(tmp_path: Path) -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    config_dir = tmp_path / "configs"
    sample_dir = tmp_path / "data" / "raw" / "sample"
    config_dir.mkdir(parents=True)
    sample_dir.parent.mkdir(parents=True)
    copy2(repo_root / "configs" / "kumamoto.yml", config_dir / "kumamoto.yml")
    copytree(repo_root / "data" / "raw" / "sample", sample_dir)
    return config_dir / "kumamoto.yml"


def test_sample_waveform_array_projection_outputs(tmp_path: Path) -> None:
    config_path = str(_sample_project(tmp_path))
    command_fetch(config_path, sample=True)
    command_build_features(config_path)
    command_transfer_functions(config_path, sample=True)
    result = command_array_projection(config_path, sample=True)
    paths = ProjectPaths.from_config(load_config(config_path))

    projection = read_table(paths.data_processed / "waveform_array_projection.parquet")
    splats = read_table(paths.data_processed / "gaussian_splat_primitive.parquet")
    assert projection
    assert splats
    assert result["projection_rows"] == len(projection)
    assert result["splat_rows"] == len(splats)
    assert {
        "beam_energy",
        "phase_coherence",
        "delay_fit",
        "array_coherence",
        "beam_power",
        "aperture_km",
        "slowness_x_s_per_km",
        "slowness_y_s_per_km",
        "projection_x_m",
        "projection_y_m",
        "projection_z_m",
        "primitive_type",
        "path_family",
        "late_phase_delay_s",
        "excess_path_km",
    }.issubset(projection[0])
    assert {
        "sigma_x_m",
        "sigma_y_m",
        "sigma_z_m",
        "opacity",
        "phase_rad",
        "array_coherence",
        "aperture_km",
        "dominant_source",
        "primitive_type",
        "path_family",
        "late_phase_delay_s",
        "excess_path_km",
        "source_event_x_m",
        "source_event_y_m",
        "source_event_z_m",
    }.issubset(splats[0])
    assert paths.outputs_3d.joinpath("gaussian_splat_primitives.ply").exists()
    html = paths.outputs_3d.joinpath("array_projection_splats.html")
    assert html.exists()
    html_text = html.read_text(encoding="utf-8")
    assert "webgl2_gaussian_point_sprite" in html_text
    assert "getContext('webgl2'" in html_text
    webgl_meta = paths.outputs_3d.joinpath("array_projection_splats.metadata.json").read_text(encoding="utf-8")
    assert "outline-only Japan context" in webgl_meta
    assert "disabled_to_avoid_hiding_subsurface_splats" in webgl_meta
    meta = read_sidecar(paths.data_processed / "waveform_array_projection.parquet")
    assert meta["synthetic_aperture_enabled"] is True
    assert meta["uses_phase"] is True
    assert meta["uses_group_delay"] is True
    assert meta["not_prediction"] is True
    assert "direct" in meta["projection_type_counts"]
