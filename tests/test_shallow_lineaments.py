from __future__ import annotations

from pathlib import Path

import pytest

from crust_lite.config import load_config
from crust_lite.io.geopackage import read_features
from crust_lite.io.parquet import read_sidecar, read_table, write_table
from crust_lite.paths import ProjectPaths
from crust_lite.processing.fault_inference import infer_faults
from crust_lite.processing.shallow_lineaments import build_shallow_lineaments
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
