from __future__ import annotations

import json
import zipfile
from pathlib import Path

from crust_lite.config import load_config
from crust_lite.data_sources.active_faults import (
    build_known_fault_reference_layers,
    fetch_active_faults,
)
from crust_lite.geo import LocalProjector
from crust_lite.io.geopackage import read_features, read_metadata, write_features
from crust_lite.io.parquet import read_sidecar, read_table
from crust_lite.paths import ProjectPaths
from tests.helpers import isolated_project


def _write_fault(path: Path, geometry_accuracy: str | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    props = {
        "segment_id": "fault_from_source",
        "source": "user_supplied_official_extract",
        "fault_type": "active_fault_trace",
    }
    if geometry_accuracy is not None:
        props["geometry_accuracy"] = geometry_accuracy
        props["source"] = "japan_major_active_faults_coarse_seed_v0"
    data = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": props,
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[130.70, 32.60], [130.90, 32.80], [131.10, 33.00]],
                },
            }
        ],
    }
    path.write_text(json.dumps(data), encoding="utf-8")


def _config_with_fault_file(tmp_path: Path, fault_file: str) -> tuple[object, ProjectPaths]:
    config_path = isolated_project(tmp_path)
    text = config_path.read_text()
    text = text.replace(
        "  use_active_faults: true\n  use_waveforms:",
        f"  use_active_faults: true\n  active_fault_file: {fault_file}\n  use_waveforms:",
    )
    config_path.write_text(text)
    cfg = load_config(config_path)
    paths = ProjectPaths.from_config(cfg)
    paths.ensure()
    return cfg, paths


def test_coarse_reference_seed_is_not_known_fault_data(tmp_path: Path) -> None:
    _write_fault(
        tmp_path / "data" / "raw" / "active_faults" / "coarse.geojson",
        geometry_accuracy="coarse_seed_not_official_trace",
    )
    cfg, paths = _config_with_fault_file(tmp_path, "data/raw/active_faults/coarse.geojson")

    result = fetch_active_faults(cfg, paths, sample=False)

    assert result["fault_count"] == 0
    assert result["reference_only_count"] == 1
    assert read_features(paths.data_processed / "fault_segment.gpkg") == []
    reference = read_features(paths.data_processed / "reference_fault_segment.gpkg")
    assert reference[0]["properties"]["known_fault_data_policy"] == "reference_only_not_known_fault"
    meta = read_metadata(paths.data_processed / "fault_segment.gpkg")
    assert meta["does_not_create_known_faults"] is True


def test_user_supplied_fault_data_builds_reference_sampling_layers(tmp_path: Path) -> None:
    _write_fault(tmp_path / "data" / "raw" / "active_faults" / "official_extract.geojson")
    cfg, paths = _config_with_fault_file(tmp_path, "data/raw/active_faults/official_extract.geojson")

    result = fetch_active_faults(cfg, paths, sample=False)

    assert result["fault_count"] == 1
    features = read_features(paths.data_processed / "fault_segment.gpkg")
    props = features[0]["properties"]
    assert props["source_quality"] == "user_supplied_data_source"
    assert props["known_fault_data_policy"] == "accepted_source_feature"
    trace_rows = read_table(paths.data_processed / "known_fault_trace_point.parquet")
    sub_rows = read_table(paths.data_processed / "known_fault_subsegment.parquet")
    assert trace_rows
    assert sub_rows
    assert all(str(row["derivative_role"]) == "trace_sample_from_source_geometry" for row in trace_rows)
    assert all(str(row["is_known_fault_segment"]).lower() in {"false", "0"} for row in trace_rows)
    meta = read_sidecar(paths.data_processed / "known_fault_trace_point.parquet")
    assert meta["derived_from_source_geometry_only"] is True
    assert meta["does_not_create_known_faults"] is True


def _write_aist_style_kmz(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    kml = """<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
  <Document>
    <Placemark>
      <name>001-01 テスト活動セグメント</name>
      <description><![CDATA[https://gbank.gsj.jp/activefault/test/001]]></description>
      <LineString>
        <coordinates>130.70,32.60,0 130.90,32.80,0 131.10,33.00,0</coordinates>
      </LineString>
    </Placemark>
    <Placemark>
      <name>002-01 分岐活動セグメント</name>
      <description><![CDATA[https://gbank.gsj.jp/activefault/test/002]]></description>
      <MultiGeometry>
        <LineString>
          <coordinates>130.20,32.20,0 130.30,32.30,0</coordinates>
        </LineString>
        <LineString>
          <coordinates>130.30,32.30,0 130.45,32.36,0</coordinates>
        </LineString>
      </MultiGeometry>
    </Placemark>
  </Document>
</kml>
"""
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("doc.kml", kml)


def test_aist_gsj_kmz_faults_are_loaded_as_known_source(tmp_path: Path) -> None:
    _write_aist_style_kmz(tmp_path / "data" / "raw" / "active_faults" / "aist.kmz")
    cfg, paths = _config_with_fault_file(tmp_path, "data/raw/active_faults/aist.kmz")

    result = fetch_active_faults(cfg, paths, sample=False)

    assert result["fault_count"] == 2
    features = read_features(paths.data_processed / "fault_segment.gpkg")
    ids = {feature["properties"]["segment_id"] for feature in features}
    assert ids == {"001-01", "002-01"}
    assert {feature["properties"]["source"] for feature in features} == {
        "aist_gsj_active_fault_database"
    }
    assert {feature["properties"]["source_quality"] for feature in features} == {
        "official_aist_gsj_active_fault_database"
    }
    multi = next(feature for feature in features if feature["properties"]["segment_id"] == "002-01")
    assert multi["geometry"]["type"] == "MultiLineString"
    assert len(multi["geometry"].get("local_trace_lines_m", [])) == 2
    trace_rows = read_table(paths.data_processed / "known_fault_trace_point.parquet")
    assert any(row["source_segment_id"] == "002-01" and str(row["line_index"]) == "1" for row in trace_rows)


def test_known_inferred_comparison_records_feedback_reason(tmp_path: Path) -> None:
    _write_aist_style_kmz(tmp_path / "data" / "raw" / "active_faults" / "aist.kmz")
    cfg, paths = _config_with_fault_file(tmp_path, "data/raw/active_faults/aist.kmz")
    fetch_active_faults(cfg, paths, sample=False)
    projector = LocalProjector(cfg.region)
    x0, y0 = projector.lonlat_to_xy(130.72, 32.62)
    x1, y1 = projector.lonlat_to_xy(130.92, 32.82)
    write_features(
        [
            {
                "type": "Feature",
                "properties": {
                    "segment_id": "inferred_test_001",
                    "strike": 45.0,
                    "fault_score": 0.8,
                    "is_inferred": True,
                    "center_x_m": (x0 + x1) / 2.0,
                    "center_y_m": (y0 + y1) / 2.0,
                },
                "geometry": {
                    "type": "LineString",
                    "coordinates": [],
                    "local_trace_m": [[x0, y0], [x1, y1]],
                },
            }
        ],
        paths.data_processed / "inferred_faults.gpkg",
        {"not_prediction": True},
    )

    build_known_fault_reference_layers(cfg, paths)

    rows = read_table(paths.data_processed / "known_fault_inferred_comparison.parquet")
    assert rows
    row = rows[0]
    assert row["known_segment_id"] == "001-01"
    assert row["difference_reason"] == "known_trace_aligned"
    assert row["feedback_action"] == "boost_and_use_known_trace_as_alignment_reference"
    assert 0.0 <= float(row["feedback_weight"]) <= 1.0
