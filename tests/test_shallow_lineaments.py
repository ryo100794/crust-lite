from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from crust_lite.config import load_config
from crust_lite.io.geopackage import read_features, write_features
from crust_lite.io.parquet import read_sidecar, read_table, write_table
from crust_lite.paths import ProjectPaths
from crust_lite.processing.fault_inference import infer_faults
from crust_lite.processing.shallow_lineaments import build_shallow_lineaments
from crust_lite.viz.webgl_splats import write_webgl_splat_preview
from tests.helpers import isolated_project


def _config_and_paths(tmp_path: Path):
    config_path = isolated_project(tmp_path)
    config = load_config(config_path)
    paths = ProjectPaths.from_config(config)
    paths.ensure()
    return config, paths


def _synthetic_aperture_splats() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for frequency in (1.0, 2.0):
        for idx, x_m in enumerate([-12000.0, -8000.0, -4000.0, 0.0, 4000.0, 8000.0, 12000.0]):
            rows.append(
                {
                    "primitive_id": f"sa_{frequency:g}_{idx}",
                    "event_id": f"e{idx}",
                    "x_m": x_m,
                    "y_m": 0.04 * x_m + ((idx % 2) - 0.5) * 120.0,
                    "z_m": 2500.0 + idx * 20.0,
                    "source_frequency_hz": frequency,
                    "frequency_band": f"{frequency:g} Hz late-phase window",
                    "primitive_type": "scattered" if idx % 2 else "reflected",
                    "path_family": "late_phase_scattering" if idx % 2 else "late_phase_reflection",
                    "structure_amplitude": 0.82,
                    "raw_amplitude": 0.88,
                    "array_coherence": 0.86,
                    "is_structure_candidate": True,
                    "splat_role": "structure",
                    "is_sample_data": True,
                }
            )
    return rows


def test_shallow_lineaments_preserve_frequency_slices(tmp_path: Path) -> None:
    config, paths = _config_and_paths(tmp_path)
    write_table(_synthetic_aperture_splats(), paths.data_processed / "gaussian_splat_primitive.parquet")

    result = build_shallow_lineaments(config, paths)

    spectral = read_table(paths.data_processed / "shallow_lineament_spectral.parquet")
    integrated = read_table(paths.data_processed / "shallow_lineament.parquet")
    assert result["source_table"] == "gaussian_splat_primitive"
    assert result["spectral_lineament_count"] >= 2
    assert result["lineament_count"] >= 1
    assert {float(row["frequency_hz"]) for row in spectral} == {1.0, 2.0}
    assert all(row["source_table"] == "gaussian_splat_primitive" for row in spectral)
    assert all("event_qc" not in str(row.get("source_table", "")) for row in spectral)
    assert all(0.0 <= float(row["surface_wave_anomaly_score"]) <= 1.0 for row in spectral)
    assert int(integrated[0]["frequency_count"]) >= 2
    assert "frequency-resolved rows are preserved" in str(integrated[0]["notes"])
    meta = read_sidecar(paths.data_processed / "shallow_lineament.parquet")
    assert meta["requires_synthetic_aperture_source"] is True
    assert meta["not_prediction"] is True


def test_inferred_faults_use_synthetic_aperture_lineaments_without_event_catalog(tmp_path: Path) -> None:
    config, paths = _config_and_paths(tmp_path)
    write_table(_synthetic_aperture_splats(), paths.data_processed / "gaussian_splat_primitive.parquet")
    build_shallow_lineaments(config, paths)

    result = infer_faults(config, paths)

    features = read_features(paths.data_processed / "inferred_faults.gpkg")
    assert result["method"] == "synthetic_aperture_shallow_lineament_to_fault_candidates"
    assert features
    assert all(feature["properties"]["source"] == "synthetic_aperture_shallow_lineament" for feature in features)
    assert all(feature["properties"].get("derived_from_event_catalog") is None for feature in features)
    assert float(features[0]["properties"]["fault_score"]) > 0.0
    assert "local_trace_m" in features[0]["geometry"]


def test_inferred_faults_require_synthetic_aperture_source(tmp_path: Path) -> None:
    config, paths = _config_and_paths(tmp_path)

    result = build_shallow_lineaments(config, paths)

    assert result["skipped"] is True
    assert result["reason"] == "no_synthetic_aperture_source"
    with pytest.raises(ValueError, match="Synthetic-aperture shallow lineaments are required"):
        infer_faults(config, paths)


def _regional_synthetic_aperture_splats() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    def add_region(region: str, center_x_m: float, center_y_m: float, frequency: float, amplitude: float) -> None:
        for idx, x_offset_m in enumerate([-12000.0, -8000.0, -4000.0, 0.0, 4000.0, 8000.0, 12000.0]):
            rows.append(
                {
                    "primitive_id": f"{region}_{frequency:g}_{idx}",
                    "event_id": f"{region}_e{idx}",
                    "x_m": center_x_m + x_offset_m,
                    "y_m": center_y_m + 0.035 * x_offset_m + ((idx % 2) - 0.5) * 80.0,
                    "z_m": 1800.0 + idx * 15.0,
                    "source_frequency_hz": frequency,
                    "frequency_band": f"{frequency:g} Hz late-phase window",
                    "primitive_type": "reflected" if idx % 2 == 0 else "scattered",
                    "path_family": "late_phase_reflection" if idx % 2 == 0 else "late_phase_scattering",
                    "structure_amplitude": amplitude,
                    "raw_amplitude": amplitude,
                    "array_coherence": 0.88,
                    "is_structure_candidate": True,
                    "splat_role": "structure",
                    "is_sample_data": True,
                }
            )

    add_region("strong", 0.0, 0.0, 1.0, 0.95)
    add_region("strong", 0.0, 0.0, 2.0, 0.94)
    add_region("weak_remote", 300000.0, 0.0, 1.0, 0.55)
    return rows


def test_shallow_lineament_selection_preserves_remote_region(tmp_path: Path) -> None:
    config, paths = _config_and_paths(tmp_path)
    config = replace(
        config,
        shallow_lineaments=replace(
            config.shallow_lineaments,
            max_lineaments=2,
            cluster_eps_km=14.0,
            tile_km=30.0,
            min_support=5,
        ),
    )
    write_table(_regional_synthetic_aperture_splats(), paths.data_processed / "gaussian_splat_primitive.parquet")

    build_shallow_lineaments(config, paths)

    integrated = read_table(paths.data_processed / "shallow_lineament.parquet")
    centers = sorted(float(row["center_x_m"]) for row in integrated)
    assert len(integrated) == 2
    assert centers[0] < 100000.0
    assert centers[1] > 200000.0
    assert {row["selection_method"] for row in integrated} <= {
        "spatial_coverage_balanced_top_score",
        "spatial_coverage_balanced_score_fill",
        "score_ranked_no_decimation",
    }
    meta = read_sidecar(paths.data_processed / "shallow_lineament.parquet")
    assert meta["lineament_selection_method"] == "spatial_coverage_balanced_top_score_then_score_fill"


def test_webgl_splats_show_reference_fault_layer_without_counting_as_known(tmp_path: Path) -> None:
    config, paths = _config_and_paths(tmp_path)
    rows = [
        {
            "x_m": 0.0,
            "y_m": 0.0,
            "z_m": 1200.0,
            "source_event_x_m": -1000.0,
            "source_event_y_m": -1000.0,
            "source_event_z_m": 1000.0,
            "amplitude": 1.0,
            "structure_amplitude": 1.0,
            "sigma_x_m": 1500.0,
            "sigma_y_m": 1500.0,
            "opacity": 0.7,
            "primitive_type": "reflected",
            "path_family": "late_phase_reflection",
            "splat_role": "structure",
            "depth_p05_km": 1.0,
            "depth_p50_km": 1.2,
            "depth_p95_km": 1.4,
            "depth_velocity_min_km_s": 2.8,
            "depth_velocity_max_km_s": 4.6,
            "depth_velocity_samples": 7,
            "depth_uncertainty_method": "test",
            "projection_refinement_method": "test",
            "projection_refinement_dx_m": 0.0,
            "projection_refinement_dy_m": 0.0,
            "projection_refinement_score_gain": 0.0,
            "is_sample_data": True,
        }
    ]
    write_features(
        [],
        paths.data_processed / "fault_segment.gpkg",
        {"does_not_create_known_faults": True},
    )
    write_features(
        [
            {
                "type": "Feature",
                "properties": {
                    "segment_id": "reference_only_trace",
                    "source": "coarse_reference_seed",
                    "known_fault_data_policy": "reference_only_not_known_fault",
                    "is_reference_only": True,
                    "center_x_m": 0.0,
                    "center_y_m": 0.0,
                    "strike": 45.0,
                    "dip": 70.0,
                    "length_km": 8.0,
                    "width_km": 5.0,
                },
                "geometry": {"type": "LineString", "local_trace_m": [[-4000.0, -4000.0], [4000.0, 4000.0]], "coordinates": []},
            }
        ],
        paths.data_processed / "reference_fault_segment.gpkg",
        {"table_role": "reference_only_not_known_fault"},
    )

    write_webgl_splat_preview(config, paths, rows, is_sample=True)

    html = (paths.outputs_3d / "array_projection_splats.html").read_text(encoding="utf-8")
    meta = (paths.outputs_3d / "array_projection_splats.metadata.json").read_text(encoding="utf-8")
    assert "参考断層" in html
    assert "公式既知" in html
    assert '"fault_overlay_known_count": 0' in meta
    assert '"fault_overlay_reference_count": 1' in meta
    assert "reference_only_traces" in meta
