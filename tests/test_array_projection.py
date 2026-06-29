from __future__ import annotations

from crust_lite.cli import (
    command_array_projection,
    command_build_features,
    command_fetch,
    command_transfer_functions,
)
from crust_lite.config import load_config
from crust_lite.io.parquet import read_sidecar, read_table
from crust_lite.paths import ProjectPaths


def test_sample_waveform_array_projection_outputs() -> None:
    config_path = "configs/kumamoto.yml"
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
    assert {"beam_energy", "phase_coherence", "delay_fit", "projection_x_m", "projection_y_m"}.issubset(projection[0])
    assert {"sigma_x_m", "sigma_y_m", "sigma_z_m", "opacity", "phase_rad"}.issubset(splats[0])
    assert paths.outputs_3d.joinpath("gaussian_splat_primitives.ply").exists()
    assert paths.outputs_3d.joinpath("array_projection_splats.html").exists()
    meta = read_sidecar(paths.data_processed / "waveform_array_projection.parquet")
    assert meta["uses_phase"] is True
    assert meta["uses_group_delay"] is True
    assert meta["not_prediction"] is True
