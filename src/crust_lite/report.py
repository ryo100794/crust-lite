from __future__ import annotations

from pathlib import Path
from typing import Any

from crust_lite.config import AppConfig
from crust_lite.io.geopackage import read_features
from crust_lite.io.metadata import read_metadata
from crust_lite.io.parquet import read_sidecar, read_table
from crust_lite.paths import ProjectPaths
from crust_lite.viz.maps import write_static_maps

NO_PREDICTION_TEXT = (
    "このプロトタイプは地震発生日を予測するものではありません。"
    "表示される ranking と failure_index は入力データと仮定に基づく相対指標です。"
    "防災判断には公的機関の情報を使用してください。"
)


def _safe_rows(path: Path) -> list[dict[str, Any]]:
    return read_table(path) if path.exists() else []


def _safe_features(path: Path) -> list[dict[str, Any]]:
    return read_features(path) if path.exists() else []


def _top_inferred(paths: ProjectPaths) -> list[dict[str, Any]]:
    rows = [feature.get("properties", {}) for feature in _safe_features(paths.data_processed / "inferred_faults.gpkg")]
    return sorted(rows, key=lambda row: float(row.get("fault_score", 0.0)), reverse=True)[:10]


def _top_scenarios(paths: ProjectPaths, column: str) -> list[dict[str, Any]]:
    rows = [
        row
        for row in _safe_rows(paths.outputs_tables / "failure_scenarios.parquet")
        if str(row.get("year")) == "100"
    ]
    return sorted(rows, key=lambda row: float(row.get(column, 0.0)), reverse=True)[:10]


def _markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return "_No rows available._"
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = []
    for row in rows:
        body.append("| " + " | ".join(str(row.get(col, "")) for col in columns) + " |")
    return "\n".join([header, sep, *body])


def write_summary(config: AppConfig, paths: ProjectPaths) -> Path:
    paths.outputs_reports.mkdir(parents=True, exist_ok=True)
    events = _safe_rows(paths.data_interim / "event_qc.parquet")
    gnss = _safe_rows(paths.data_processed / "gnss_features.parquet")
    known = _safe_features(paths.data_processed / "fault_segment.gpkg")
    inferred = _safe_features(paths.data_processed / "inferred_faults.gpkg")
    event_meta = read_sidecar(paths.data_interim / "event_qc.parquet")
    stress_meta = read_sidecar(paths.data_processed / "stress_state.parquet")
    meta3d = read_metadata(paths.outputs_3d / "metadata.json")
    db_meta = read_metadata(paths.data_processed / "crust_lite.duckdb.metadata.json")
    tf_meta = read_sidecar(paths.data_processed / "site_transfer_function.parquet")
    domestic_sources = _safe_rows(paths.data_processed / "domestic_ingest_plan.parquet")
    top_faults = _top_inferred(paths)
    top_p50 = _top_scenarios(paths, "failure_index_p50")
    top_p95 = _top_scenarios(paths, "failure_index_p95")
    html_files = [
        paths.outputs_3d / "index.html",
        paths.outputs_3d / "events_faults_timeseries.html",
        paths.outputs_3d / "stress_timeseries_3d.html",
        paths.outputs_3d / "failure_scenarios_3d.html",
        paths.outputs_3d / "japan_archipelago_context.html",
        paths.outputs_3d / "metadata.json",
    ]
    lines = [
        "# crust-lite summary",
        "",
        NO_PREDICTION_TEXT,
        "",
        "## 1. Target region and period",
        "",
        f"- region: `{config.region.name}`",
        f"- bbox: `{config.region.bbox}`",
        f"- local CRS: `{config.region.crs_local}`",
        f"- period: `{config.region.start_date}` to `{config.region.end_date}`",
        "",
        "## 2. Data used",
        "",
        f"- event catalog rows: {len(events)}",
        f"- GNSS stations: {len(gnss)}",
        f"- known fault segments: {len(known)}",
        f"- inferred fault segments: {len(inferred)}",
        f"- is_sample_data: {str(event_meta.get('is_sample_data', False)).lower()}",
        "",
        "## 3. Counts",
        "",
        f"- events: {len(events)}",
        f"- GNSS stations: {len(gnss)}",
        f"- known faults: {len(known)}",
        f"- inferred faults: {len(inferred)}",
        "",
        "## 4. Top inferred fault candidates",
        "",
        _markdown_table(
            top_faults,
            ["segment_id", "n_events", "fault_score", "confidence", "distance_to_known_fault_km"],
        ),
        "",
        "## 5. Top 100-year failure_index_p50",
        "",
        _markdown_table(top_p50, ["segment_id", "year", "failure_index_p50", "uncertainty_score"]),
        "",
        "## 6. Top 100-year failure_index_p95",
        "",
        _markdown_table(top_p95, ["segment_id", "year", "failure_index_p95", "uncertainty_score"]),
        "",
        "## 7. Main uncertainties",
        "",
        "- Stress calculation uses `fallback_approximation` unless a future cutde kernel is implemented.",
        "- Sample data are artificial and cannot be used for scientific interpretation.",
        "- GNSS strain gradient is a local low-cost proxy, not a full strain inversion.",
        "- Candidate planes are inferred from catalog geometry and can be biased by catalog completeness.",
        "- Transfer functions are relative complex spectral ratios, not a unique 3D velocity inversion.",
        "",
        "## 8. Unused data",
        "",
        "- Full waveform inversion is not implemented in the MVP.",
        "- PyLith finite-element execution is an extension point only.",
        "- F-net and GEONET automatic retrieval are v1 items.",
        "",
        "## 9. Not an earthquake prediction",
        "",
        NO_PREDICTION_TEXT,
        "",
        "## 10. Next analyses",
        "",
        "- F-net mechanism automatic retrieval",
        "- GEONET daily coordinate automatic retrieval",
        "- J-SHIS API stabilization",
        "- K-NET/KiK-net waveform features",
        "- cutde triangular dislocation stress calculation",
        "- PyLith local 2D or coarse 3D validation",
        "- OpenQuake or J-SHIS hazard comparison",
        "- Backtesting evaluation",
        "",
        "## 11. Domestic Japan Data Ingest",
        "",
        f"- registered domestic source classes: {len(domestic_sources)}",
        f"- configured/local domestic source classes: {sum(1 for row in domestic_sources if str(row.get('ingest_status', '')).endswith('_configured'))}",
        "- full raw waveform, high-rate GNSS/RINEX, InSAR, and offshore archives are tracked as staged external archives because they are PB-scale and often require agency credentials or bulk agreements.",
        "- historical catalog rows are retained with epoch-dependent quality, completeness, and analysis weight fields rather than being discarded for lower precision.",
        "- registry tables: `domestic_data_source`, `domestic_ingest_plan`",
        "",
        "## 12. Database and mesh storage",
        "",
        f"- database engine: {db_meta.get('engine', 'not_available')}",
        f"- database path: `{db_meta.get('database_path', 'not_available')}`",
        f"- materialized tables: {', '.join(sorted((db_meta.get('tables') or {}).keys())) if isinstance(db_meta.get('tables'), dict) else 'not_available'}",
        "- mesh policy: DB stores mesh metadata, node/element index tables, and field file indexes; large time-varying arrays should remain in HDF5/Zarr/XDMF or solver-native files.",
        "- mesh tables: `mesh_dataset`, `mesh_node`, `mesh_element`, `mesh_field_index`",
        "",
        "## 13. Complex Transfer Function Outputs",
        "",
        f"- transfer method: {tf_meta.get('method', 'not_generated')}",
        f"- spectrum rows: {tf_meta.get('spectrum_rows', 'not_generated')}",
        f"- transfer rows: {tf_meta.get('transfer_rows', 'not_generated')}",
        f"- validation rows: {tf_meta.get('validation_rows', 'not_generated')}",
        "- phase/group delay are retained so time-delay and triangulation information is not discarded by amplitude-only spectra.",
        "- outputs: `waveform_spectrum.parquet`, `site_transfer_function.parquet`, `transfer_validation.parquet`, `structure_anomaly.parquet`",
        "",
        "## 14. 3D visualization outputs",
        "",
        "\n".join(f"- `{path.relative_to(paths.root)}`: {'exists' if path.exists() else 'missing'}" for path in html_files),
        "",
        f"- displayed events: {meta3d.get('displayed_event_count', 'not_generated')}",
        f"- displayed known + inferred faults: {meta3d.get('displayed_fault_count', 'not_generated')}",
        f"- displayed time frames: {meta3d.get('displayed_frame_count', 'not_generated')}",
        f"- decimation method: {meta3d.get('decimation_method', 'not_generated')}",
        f"- vertical exaggeration: {meta3d.get('vertical_exaggeration', config.visualization_3d.vertical_exaggeration)}",
        f"- map overlay: {meta3d.get('map_overlay', 'local_crs_bbox_graticule')}",
        f"- stress display: {stress_meta.get('stress_method', 'unknown')}",
        "- failure_index is dimensionless; values above 1 mean model-threshold exceedance under assumptions, not an occurrence statement.",
        "",
    ]
    out = paths.outputs_reports / "summary.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def export_outputs(config: AppConfig, paths: ProjectPaths) -> dict[str, Any]:
    maps = write_static_maps(config, paths) if config.outputs.write_maps else {"map_files": []}
    report = write_summary(config, paths) if config.outputs.write_report else None
    return {"maps": maps, "report": str(report) if report else None}
