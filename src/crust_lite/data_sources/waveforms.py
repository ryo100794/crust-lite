from __future__ import annotations

from crust_lite.config import AppConfig
from crust_lite.io.parquet import write_table
from crust_lite.logging import get_logger
from crust_lite.paths import ProjectPaths

LOGGER = get_logger(__name__)


def fetch_waveforms(config: AppConfig, paths: ProjectPaths, sample: bool = False) -> dict[str, object]:
    paths.ensure()
    if not config.data_sources.use_waveforms:
        write_table(
            [],
            paths.data_processed / "waveform_feature.parquet",
            {"is_sample_data": sample, "source_note": "use_waveforms=false"},
        )
        LOGGER.info("Skipping waveform retrieval because use_waveforms=false")
        return {"is_sample_data": sample, "waveform_rows": 0, "skipped": True}
    # The extension point is explicit. A future implementation can add ObsPy station
    # selection and MiniSEED caching here without touching downstream modules.
    write_table(
        [],
        paths.data_processed / "waveform_feature.parquet",
        {"is_sample_data": sample, "source_note": "waveform_fetch_not_implemented_in_mvp"},
    )
    LOGGER.warning("Waveform fetching is optional and not implemented in this MVP")
    return {"is_sample_data": sample, "waveform_rows": 0, "skipped": True}
