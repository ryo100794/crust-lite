from __future__ import annotations

import csv
import hashlib
from pathlib import Path
from shutil import copy2, copytree
from typing import Any

from crust_lite.config import load_config
from crust_lite.geo import LocalProjector
from crust_lite.io.parquet import write_table
from crust_lite.paths import ProjectPaths


DEVELOPMENT_FIXTURE_MARKER = ".crust-lite-development-observation-fixture-v1"
DEVELOPMENT_FIXTURE_MARKER_BODY = "development-only; observation inputs; network=0; known-structure=0\n"


def isolated_project(tmp_path: Path, config_name: str = "kumamoto.yml") -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    config_dir = tmp_path / "configs"
    sample_dir = tmp_path / "data" / "raw" / "sample"
    config_dir.mkdir(parents=True, exist_ok=True)
    sample_dir.parent.mkdir(parents=True, exist_ok=True)
    copy2(repo_root / "configs" / config_name, config_dir / config_name)
    if (repo_root / "data" / "raw" / "sample").exists() and not sample_dir.exists():
        copytree(repo_root / "data" / "raw" / "sample", sample_dir)
    (tmp_path / DEVELOPMENT_FIXTURE_MARKER).write_text(
        DEVELOPMENT_FIXTURE_MARKER_BODY,
        encoding="utf-8",
    )
    return config_dir / config_name


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = [dict(row) for row in csv.DictReader(stream)]
    if not rows:
        raise ValueError(f"development fixture CSV is empty: {path}")
    return rows


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def materialize_observation_only_development_fixture(
    config_path: str | Path,
) -> dict[str, Any]:
    """Build test-only observation tables without invoking any fetch path.

    The marker is created only by :func:`isolated_project`; this helper refuses
    the repository root or any unmarked project. It intentionally has no
    network client and never reads or materializes known-fault/J-SHIS inputs.
    """

    config = load_config(config_path)
    paths = ProjectPaths.from_config(config)
    repo_root = Path(__file__).resolve().parents[1]
    marker = paths.root / DEVELOPMENT_FIXTURE_MARKER
    if paths.root.resolve() == repo_root.resolve():
        raise RuntimeError("development fixture refuses the repository root")
    if not marker.is_file() or marker.read_text(encoding="utf-8") != DEVELOPMENT_FIXTURE_MARKER_BODY:
        raise RuntimeError("development fixture requires the explicit isolated-project marker")

    known_path = paths.data_processed / "fault_segment.gpkg"
    if known_path.exists():
        raise RuntimeError("development observation fixture refuses pre-existing known structure")

    paths.ensure()
    sample = paths.data_raw / "sample"
    event_path = sample / "sample_events.csv"
    mechanism_path = sample / "sample_mechanisms.csv"
    gnss_path = sample / "sample_gnss_daily.csv"
    projector = LocalProjector(config.region)

    events: list[dict[str, Any]] = []
    for row in _csv_rows(event_path):
        lon, lat, depth_km = float(row["lon"]), float(row["lat"]), float(row["depth_km"])
        x_m, y_m = projector.lonlat_to_xy(lon, lat)
        events.append(
            {
                "event_id": row["event_id"],
                "time_utc": row["time_utc"],
                "lat": lat,
                "lon": lon,
                "depth_km": depth_km,
                "magnitude": float(row["magnitude"]),
                "magnitude_type": row.get("magnitude_type", ""),
                "catalog_source": row.get("catalog_source", "development_fixture"),
                "has_mechanism": False,
                "has_waveform_feature": False,
                "x_m": x_m,
                "y_m": y_m,
                "z_m": depth_km * 1000.0,
                "is_sample_data": True,
            }
        )
    write_table(
        events,
        paths.data_processed / "event.parquet",
        {
            "fixture_contract": "development-observation-only-v1",
            "is_sample_data": True,
            "network_requests": 0,
            "known_structure_input_count": 0,
            "source_sha256": _sha256(event_path),
        },
    )

    mechanisms: list[dict[str, Any]] = []
    numeric_mechanism = {
        "strike1", "dip1", "rake1", "strike2", "dip2", "rake2", "scalar_moment_nm"
    }
    for row in _csv_rows(mechanism_path):
        mechanisms.append(
            {key: float(value) if key in numeric_mechanism else value for key, value in row.items()}
        )
    write_table(
        mechanisms,
        paths.data_processed / "mechanism.parquet",
        {
            "fixture_contract": "development-observation-only-v1",
            "is_sample_data": True,
            "source_sha256": _sha256(mechanism_path),
        },
    )

    gnss_rows: list[dict[str, Any]] = []
    numeric_gnss = {"lat", "lon", "east_m", "north_m", "up_m", "sigma_e", "sigma_n", "sigma_u"}
    for row in _csv_rows(gnss_path):
        converted = {key: float(value) if key in numeric_gnss else value for key, value in row.items()}
        x_m, y_m = projector.lonlat_to_xy(float(converted["lon"]), float(converted["lat"]))
        gnss_rows.append({**converted, "x_m": x_m, "y_m": y_m, "is_sample_data": True})
    write_table(
        gnss_rows,
        paths.data_processed / "gnss_daily.parquet",
        {
            "fixture_contract": "development-observation-only-v1",
            "is_sample_data": True,
            "source_sha256": _sha256(gnss_path),
        },
    )
    write_table(
        [],
        paths.data_processed / "waveform_feature.parquet",
        {
            "fixture_contract": "development-observation-only-v1",
            "is_sample_data": True,
            "source_note": "explicit empty development fixture; no waveform fetch",
        },
    )

    if known_path.exists():
        raise RuntimeError("development fixture unexpectedly created known structure")
    return {
        "status": "DEVELOPMENT_FIXTURE_ONLY",
        "fixture_contract": "development-observation-only-v1",
        "event_count": len(events),
        "mechanism_count": len(mechanisms),
        "gnss_row_count": len(gnss_rows),
        "network_requests": 0,
        "known_structure_artifacts_written": 0,
    }
