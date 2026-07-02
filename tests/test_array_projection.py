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
from crust_lite.processing.array_projection import (
    _depth_quadrature_points,
    _resolution_sigma_m,
    _vertical_resolution_sigma_m,
)

DEPTH_UNCERTAINTY_COLUMNS = {
    "depth_p05_km",
    "depth_p50_km",
    "depth_p95_km",
    "depth_velocity_min_km_s",
    "depth_velocity_max_km_s",
    "depth_velocity_samples",
    "depth_uncertainty_method",
}

PROJECTION_REFINEMENT_COLUMNS = {
    "projection_refinement_dx_m",
    "projection_refinement_dy_m",
    "projection_refinement_score_gain",
    "projection_refinement_method",
}

ARRAY_PROJECTION_DERIVED_COLUMNS = DEPTH_UNCERTAINTY_COLUMNS | PROJECTION_REFINEMENT_COLUMNS


def _assert_has_columns(row: dict[str, object], columns: set[str]) -> None:
    missing = columns - set(row)
    assert not missing, f"missing columns: {sorted(missing)}"


def _assert_depth_uncertainty_row(row: dict[str, object]) -> None:
    assert float(row["depth_p05_km"]) <= float(row["depth_p50_km"]) <= float(row["depth_p95_km"])
    assert float(row["depth_velocity_min_km_s"]) <= float(row["depth_velocity_max_km_s"])
    assert int(row["depth_velocity_samples"]) > 0
    assert row["depth_uncertainty_method"]
    assert row["projection_refinement_method"]


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
        "z_m",
        "primitive_type",
        "path_family",
        "late_phase_delay_s",
        "excess_path_km",
    }.issubset(projection[0])
    _assert_has_columns(projection[0], ARRAY_PROJECTION_DERIVED_COLUMNS)
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
        "z_m",
    }.issubset(splats[0])
    _assert_has_columns(splats[0], ARRAY_PROJECTION_DERIVED_COLUMNS)

    for row in projection:
        _assert_depth_uncertainty_row(row)

    for row in splats:
        _assert_depth_uncertainty_row(row)
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


def test_near_surface_adaptive_resolution(tmp_path: Path) -> None:
    cfg = load_config(str(_sample_project(tmp_path)))

    shallow_samples = _depth_quadrature_points(cfg, 1.0, 2.0, 3.0, "reflected", True)
    deep_samples = _depth_quadrature_points(cfg, 25.0, 30.0, 35.0, "reflected", True)

    assert len(shallow_samples) == cfg.waveform_array.near_surface_depth_quadrature_samples
    assert len(deep_samples) == cfg.waveform_array.depth_quadrature_samples
    assert "near_surface" in shallow_samples[0][4]
    assert "near_surface" not in deep_samples[0][4]

    shallow_sigma_z = _vertical_resolution_sigma_m(cfg, 2.0, 10_000.0)
    deep_sigma_z = _vertical_resolution_sigma_m(cfg, 40.0, 10_000.0)
    assert shallow_sigma_z < deep_sigma_z
    assert shallow_sigma_z >= cfg.waveform_array.near_surface_splat_sigma_vertical_m

    shallow_sigma_xy = _resolution_sigma_m(cfg, 2.0, 50.0, 2.0)
    deep_sigma_xy = _resolution_sigma_m(cfg, 2.0, 50.0, 40.0)
    assert shallow_sigma_xy <= deep_sigma_xy
