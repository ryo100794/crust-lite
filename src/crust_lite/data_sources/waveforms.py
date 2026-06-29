"""Waveform feature ingestion.

MVP waveform handling accepts pre-collected feature CSV files so expensive
download and signal processing can run independently from the main pipeline.
When no waveform data is configured, schema-compatible empty outputs keep later
stages deterministic.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from crust_lite.config import AppConfig
from crust_lite.io.parquet import write_table
from crust_lite.logging import get_logger
from crust_lite.paths import ProjectPaths, resolve_input

LOGGER = get_logger(__name__)

FEATURE_COLUMNS = {
    "event_id",
    "station_id",
    "channel",
    "pga",
    "pgv",
    "psa_0p3",
    "psa_1p0",
    "psa_3p0",
    "p_residual_s",
    "s_residual_s",
    "amp_residual_log",
    "source",
}


def _read_feature_csv(path: Path) -> list[dict[str, Any]]:
    """Read externally generated waveform features with column validation."""
    with path.open("r", encoding="utf-8", newline="") as fh:
        rows = [dict(row) for row in csv.DictReader(fh)]
    if not rows:
        return []
    missing = FEATURE_COLUMNS.difference(rows[0])
    if missing:
        raise ValueError(f"Waveform feature CSV missing columns {sorted(missing)}: {path}")
    numeric = [
        "pga",
        "pgv",
        "psa_0p3",
        "psa_1p0",
        "psa_3p0",
        "p_residual_s",
        "s_residual_s",
        "amp_residual_log",
    ]
    for row in rows:
        for key in numeric:
            row[key] = float(row.get(key, 0.0) or 0.0)
    return rows


def fetch_waveforms(config: AppConfig, paths: ProjectPaths, sample: bool = False) -> dict[str, object]:
    """Populate waveform_feature.parquet from sample or configured CSV input."""
    paths.ensure()
    feature_csv = config.data_sources.waveform_feature_csv
    if feature_csv:
        fallback = paths.data_raw / "sample" / "sample_waveform_feature.csv"
        source_path = resolve_input(paths.root, feature_csv, fallback)
        if not source_path.exists():
            write_table(
                [],
                paths.data_processed / "waveform_feature.parquet",
                {"is_sample_data": sample, "source_note": f"waveform_feature_csv_missing:{source_path}"},
            )
            LOGGER.warning("Configured waveform feature CSV is missing; continuing without waveform features: %s", source_path)
            return {"is_sample_data": sample, "waveform_rows": 0, "skipped": True, "missing_source_path": str(source_path)}
        rows = _read_feature_csv(source_path)
        is_sample = sample or source_path == fallback
        write_table(
            rows,
            paths.data_processed / "waveform_feature.parquet",
            {
                "is_sample_data": is_sample,
                "source_note": f"local_waveform_feature_csv:{source_path}",
                "waveform_rows": len(rows),
            },
        )
        LOGGER.info("Loaded %d waveform feature rows from %s", len(rows), source_path)
        return {"is_sample_data": is_sample, "waveform_rows": len(rows), "source_path": str(source_path)}

    if not config.data_sources.use_waveforms:
        write_table(
            [],
            paths.data_processed / "waveform_feature.parquet",
            {"is_sample_data": sample, "source_note": "use_waveforms=false"},
        )
        LOGGER.info("Skipping waveform retrieval because use_waveforms=false")
        return {"is_sample_data": sample, "waveform_rows": 0, "skipped": True}

    write_table(
        [],
        paths.data_processed / "waveform_feature.parquet",
        {"is_sample_data": sample, "source_note": "waveform_fetch_not_configured; use scripts/collect_fdsn_waveform_spectra.py or staged domestic archives"},
    )
    LOGGER.warning("Waveform auto-fetch is not configured in fetch(); use the FDSN collector script or staged domestic waveform archives")
    return {"is_sample_data": sample, "waveform_rows": 0, "skipped": True}
