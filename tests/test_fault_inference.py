from __future__ import annotations

from crust_lite.config import load_config
from crust_lite.io.geopackage import read_features
from crust_lite.io.parquet import write_table
from crust_lite.paths import ProjectPaths
from crust_lite.processing.fault_inference import infer_faults
from crust_lite.processing.shallow_lineaments import build_shallow_lineaments
from tests.helpers import isolated_project


def test_fault_inference_from_synthetic_aperture_lineaments(tmp_path) -> None:
    config_path = str(isolated_project(tmp_path))
    cfg = load_config(config_path)
    paths = ProjectPaths.from_config(cfg)
    paths.ensure()
    rows = []
    for frequency in (1.0, 2.0):
        for idx, x_m in enumerate([-12000.0, -8000.0, -4000.0, 0.0, 4000.0, 8000.0, 12000.0]):
            rows.append(
                {
                    "primitive_id": f"fault_sa_{frequency:g}_{idx}",
                    "event_id": f"e{idx}",
                    "x_m": x_m,
                    "y_m": 0.03 * x_m,
                    "z_m": 2200.0,
                    "source_frequency_hz": frequency,
                    "frequency_band": f"{frequency:g} Hz late-phase window",
                    "primitive_type": "scattered",
                    "path_family": "late_phase_scattering",
                    "structure_amplitude": 0.86,
                    "raw_amplitude": 0.90,
                    "array_coherence": 0.88,
                    "is_structure_candidate": True,
                    "splat_role": "structure",
                    "is_sample_data": True,
                }
            )
    write_table(rows, paths.data_processed / "gaussian_splat_primitive.parquet")
    build_shallow_lineaments(cfg, paths)

    result = infer_faults(cfg, paths)

    features = read_features(paths.data_processed / "inferred_faults.gpkg")
    assert result["inferred_fault_count"] >= 1
    assert features
    assert features[0]["properties"]["source"] == "synthetic_aperture_shallow_lineament"
    assert 0.0 <= float(features[0]["properties"]["fault_score"]) <= 1.0
