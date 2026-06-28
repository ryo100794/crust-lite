from __future__ import annotations

from crust_lite.io.parquet import write_table
from crust_lite.paths import ProjectPaths


def build_waveform_features(paths: ProjectPaths, is_sample_data: bool = False) -> dict[str, object]:
    path = paths.data_processed / "waveform_feature.parquet"
    if path.exists():
        return {"waveform_rows": 0, "is_sample_data": is_sample_data}
    write_table(
        [],
        path,
        {"is_sample_data": is_sample_data, "source_note": "waveform_features_not_available"},
    )
    return {"waveform_rows": 0, "is_sample_data": is_sample_data}
