from __future__ import annotations

from typing import Any

from crust_lite.config import AppConfig
from crust_lite.io.database import materialize_rows
from crust_lite.io.parquet import write_table
from crust_lite.logging import get_logger
from crust_lite.paths import ProjectPaths

LOGGER = get_logger(__name__)

DOMESTIC_SOURCES: list[dict[str, Any]] = [
    {
        "source_id": "jma_unified_hypocenter_catalog",
        "agency": "Japan Meteorological Agency",
        "domain": "earthquake_catalog",
        "data_type": "hypocenter_origin_magnitude_intensity",
        "time_coverage": "historical_to_present; precision varies by epoch",
        "spatial_coverage": "Japan and surrounding region",
        "access_mode": "bulk_or_official_service_required",
        "implemented_ingest": "not_yet; local CSV schema supported through event_csv after conversion",
        "expected_storage_class": "feature_db",
        "estimated_curated_size": "5-50 GB",
        "estimated_full_size": "50-300 GB",
        "priority": 1,
        "notes": "Primary domestic long-term catalog. Must preserve epoch-dependent uncertainty and completeness magnitude.",
    },
    {
        "source_id": "jma_seismic_intensity_catalog",
        "agency": "Japan Meteorological Agency",
        "domain": "intensity_catalog",
        "data_type": "observed_intensity_points_and_event_intensity_summary",
        "time_coverage": "historical_to_present; station network and intensity scale details vary by epoch",
        "spatial_coverage": "Japan intensity observation network",
        "access_mode": "official_service_or_bulk_conversion_required",
        "implemented_ingest": "not_yet; planned station/event intensity table",
        "expected_storage_class": "feature_db",
        "estimated_curated_size": "10-200 GB",
        "estimated_full_size": "0.1-2 TB",
        "priority": 1,
        "notes": "Useful for long-term shaking constraints and catalog completeness, but not a waveform substitute.",
    },
    {
        "source_id": "nied_hinet_continuous_waveform",
        "agency": "NIED",
        "domain": "waveform",
        "data_type": "continuous_high_sensitivity_waveform",
        "time_coverage": "modern_dense_era",
        "spatial_coverage": "Japan dense seismic network",
        "access_mode": "account_or_bulk_access_required",
        "implemented_ingest": "not_yet; MiniSEED cache layout reserved",
        "expected_storage_class": "raw_continuous_waveform",
        "estimated_curated_size": "0.5-2 PB",
        "estimated_full_size": "2-10 PB",
        "priority": 1,
        "notes": "Do not load into memory. Store by network/station/year/day; derive complex transfer rows into DuckDB/Parquet.",
    },
    {
        "source_id": "nied_knet_kiknet_strong_motion",
        "agency": "NIED",
        "domain": "waveform",
        "data_type": "strong_motion_event_waveform_and_borehole_surface_pairs",
        "time_coverage": "modern_dense_era",
        "spatial_coverage": "Japan strong-motion network",
        "access_mode": "official_download_or_bulk_access_required",
        "implemented_ingest": "not_yet; waveform_feature and waveform_spectra_csv schemas prepared",
        "expected_storage_class": "event_window_waveform",
        "estimated_curated_size": "10-100 TB",
        "estimated_full_size": "0.1-0.5 PB",
        "priority": 1,
        "notes": "Important for empirical transfer functions because KiK-net has borehole/surface pairs.",
    },
    {
        "source_id": "nied_fnet_mechanism_waveform",
        "agency": "NIED",
        "domain": "mechanism_waveform",
        "data_type": "moment_tensor_focal_mechanism_broadband_waveform",
        "time_coverage": "modern_instrumental",
        "spatial_coverage": "Japan and surrounding region",
        "access_mode": "official_service_or_bulk_download_required",
        "implemented_ingest": "partial; mechanism local CSV supported, auto-fetch not yet",
        "expected_storage_class": "feature_db_plus_waveform",
        "estimated_curated_size": "10-500 GB",
        "estimated_full_size": "1-20 TB",
        "priority": 1,
        "notes": "Needed for mechanism consistency and stress projection.",
    },
    {
        "source_id": "gsi_geonet_daily_highrate_rinex",
        "agency": "GSI",
        "domain": "gnss",
        "data_type": "daily_coordinates_highrate_rinex_station_metadata",
        "time_coverage": "modern_geodetic_era",
        "spatial_coverage": "Japan GEONET stations",
        "access_mode": "official_download_or_bulk_access_required",
        "implemented_ingest": "partial; local daily coordinate CSV supported, RINEX mirror not yet",
        "expected_storage_class": "gnss_features_and_raw_archive",
        "estimated_curated_size": "5-100 GB",
        "estimated_full_size": "0.5-5 PB",
        "priority": 1,
        "notes": "Daily products are small; high-rate/raw RINEX dominates capacity.",
    },
    {
        "source_id": "aist_active_fault_database",
        "agency": "AIST",
        "domain": "active_fault",
        "data_type": "fault_trace_segment_attributes",
        "time_coverage": "geologic_static_with_versions",
        "spatial_coverage": "Japan",
        "access_mode": "official_gis_or_database_download_required",
        "implemented_ingest": "partial; local GeoJSON/GeoPackage supported",
        "expected_storage_class": "geospatial_vector",
        "estimated_curated_size": "1-20 GB",
        "estimated_full_size": "20-200 GB",
        "priority": 1,
        "notes": "Core comparator for inferred/unregistered candidate faults.",
    },
    {
        "source_id": "gsj_seamless_geology_and_ground_layers",
        "agency": "AIST/GSJ and related domestic ground datasets",
        "domain": "geology_ground",
        "data_type": "geologic_units_surface_ground_condition_context_layers",
        "time_coverage": "static_versioned_geology_and_ground_models",
        "spatial_coverage": "Japan",
        "access_mode": "official_gis_or_tile_download_required",
        "implemented_ingest": "not_yet; planned raster/vector layer index",
        "expected_storage_class": "raster_vector_feature_layers",
        "estimated_curated_size": "10-200 GB",
        "estimated_full_size": "0.1-2 TB",
        "priority": 2,
        "notes": "Ground/geology context for transfer-function anomalies and known-fault comparison.",
    },
    {
        "source_id": "jshis_hazard_soil_fault_model",
        "agency": "NIED J-SHIS",
        "domain": "hazard_soil_fault_model",
        "data_type": "hazard_layers_soil_amplification_fault_models",
        "time_coverage": "versioned_static_and_model_updates",
        "spatial_coverage": "Japan",
        "access_mode": "web_api_or_bulk_download_limited",
        "implemented_ingest": "partial; best-effort API hook and local processed schema",
        "expected_storage_class": "raster_vector_feature_layers",
        "estimated_curated_size": "20-300 GB",
        "estimated_full_size": "0.5-3 TB",
        "priority": 2,
        "notes": "Useful as contextual prior, not a deterministic predictor.",
    },
    {
        "source_id": "herp_source_fault_models_and_long_term_evaluations",
        "agency": "Headquarters for Earthquake Research Promotion",
        "domain": "source_fault_model",
        "data_type": "evaluated_source_faults_long_term_assessment_model_metadata",
        "time_coverage": "versioned_static_and_assessment_updates",
        "spatial_coverage": "Japan and surrounding subduction zones",
        "access_mode": "official_documents_gis_or_manual_conversion_required",
        "implemented_ingest": "not_yet; planned model metadata and geometry index",
        "expected_storage_class": "geospatial_vector_plus_document_metadata",
        "estimated_curated_size": "1-50 GB",
        "estimated_full_size": "10-500 GB",
        "priority": 2,
        "notes": "Reference source models for plate-boundary and active-fault comparison; not used as deterministic prediction.",
    },
    {
        "source_id": "gsi_national_geospatial_base_dem_coastline",
        "agency": "GSI",
        "domain": "base_map_topography",
        "data_type": "dem_coastline_administrative_grid_and_reference_layers",
        "time_coverage": "versioned_static_with_updates",
        "spatial_coverage": "Japan",
        "access_mode": "official_tile_or_bulk_download_required",
        "implemented_ingest": "not_yet; planned context layer index for visualization and meshing",
        "expected_storage_class": "raster_vector_context_layers",
        "estimated_curated_size": "50-500 GB",
        "estimated_full_size": "0.5-5 TB",
        "priority": 3,
        "notes": "Used for map overlays, DEM-derived context, and future mesh construction; analytical fields remain separate from event data.",
    },
    {
        "source_id": "sar_insar_alos_sentinel_japan",
        "agency": "JAXA/ESA/related archives",
        "domain": "insar",
        "data_type": "sar_raw_slc_interferograms_velocity_fields",
        "time_coverage": "sensor_era_dependent",
        "spatial_coverage": "Japan frames/tracks",
        "access_mode": "archive_credentials_processing_pipeline_required",
        "implemented_ingest": "not_yet; storage layout and DB index recommended",
        "expected_storage_class": "array_archive_indexed_by_db",
        "estimated_curated_size": "50-500 TB",
        "estimated_full_size": "1-10 PB",
        "priority": 2,
        "notes": "Do not store dense rasters in DuckDB; index HDF5/Zarr/COG products.",
    },
    {
        "source_id": "marine_ocean_bottom_and_offshore_geodesy",
        "agency": "JMA/JAMSTEC/GSI/JCG/NIED as applicable",
        "domain": "offshore_geophysics",
        "data_type": "ocean_bottom_seismic_pressure_gnss_acoustic",
        "time_coverage": "network_dependent",
        "spatial_coverage": "offshore Japan trenches and margins",
        "access_mode": "project_or_agency_specific_access_required",
        "implemented_ingest": "not_yet; manifest only",
        "expected_storage_class": "raw_archive_plus_feature_db",
        "estimated_curated_size": "10-500 TB",
        "estimated_full_size": "0.5-5 PB",
        "priority": 3,
        "notes": "Important for plate mechanics, but access and formats are heterogeneous.",
    },
]


def build_domestic_ingest_plan(config: AppConfig, paths: ProjectPaths) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for source in DOMESTIC_SOURCES:
        status = "manifested"
        blocker = "bulk_download_credentials_storage_or_converter_required"
        catalog_source = config.data_sources.catalog_source.lower()
        is_domestic_catalog = any(token in catalog_source for token in ("jma", "japan meteorological", "domestic", "kisho"))
        if source["source_id"] == "jma_unified_hypocenter_catalog" and config.data_sources.event_csv and is_domestic_catalog:
            status = "local_csv_configured"
            blocker = "none_for_configured_csv_subset"
        if source["source_id"] == "gsi_geonet_daily_highrate_rinex" and config.data_sources.gnss_csv:
            status = "local_daily_csv_configured"
            blocker = "raw_rinex_not_mirrored"
        if source["source_id"] == "aist_active_fault_database" and config.data_sources.active_fault_file:
            status = "local_geospatial_file_configured"
            blocker = "none_for_configured_file_subset"
        if source["source_id"] == "nied_fnet_mechanism_waveform" and config.data_sources.mechanism_csv:
            status = "local_mechanism_csv_configured"
            blocker = "waveform_auto_fetch_not_configured"
        rows.append(
            {
                **source,
                "region": config.region.name,
                "requested_start_date": config.region.start_date.isoformat(),
                "requested_end_date": config.region.end_date.isoformat(),
                "ingest_status": status,
                "current_blocker": blocker,
                "storage_policy": (
                    "duckdb_for_indexes_and_features; parquet_for_tables; hdf5_zarr_xdmf_vtk_exodus_for_dense_arrays"
                ),
                "prediction_disclaimer": "research_state_assessment_only_not_occurrence_prediction",
            }
        )
    write_table(
        rows,
        paths.data_processed / "domestic_data_source.parquet",
        {"description": "Domestic Japan data source ingest registry", "source_count": len(rows)},
    )
    write_table(
        rows,
        paths.data_processed / "domestic_ingest_plan.parquet",
        {"description": "Planned ingest status for domestic Japan data sources", "source_count": len(rows)},
    )
    materialize_rows(paths, "domestic_data_source", rows)
    materialize_rows(paths, "domestic_ingest_plan", rows)
    report = paths.outputs_reports / "domestic_data_ingest_plan.md"
    lines = [
        "# Domestic Japan Data Ingest Plan",
        "",
        "This is the ingest registry for all domestic data classes discussed so far. Full raw ingestion is staged because waveforms, high-rate GNSS, InSAR, and FEM ensembles are PB-scale and often require credentials or bulk agreements.",
        "",
        "| source_id | domain | status | blocker | curated size | full size |",
        "| --- | --- | --- | --- | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {source_id} | {domain} | {ingest_status} | {current_blocker} | {estimated_curated_size} | {estimated_full_size} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## Storage Policy",
            "",
            "- DuckDB: catalogs, indexes, metadata, transfer functions, sparse stress, quality epochs.",
            "- Parquet: durable partitioned tables.",
            "- HDF5/Zarr/XDMF/VTK/Exodus: dense waveform-derived arrays, InSAR stacks, FEM fields, solver exchange.",
            "- All historical data remain usable with explicit epoch quality, completeness magnitude, and analysis weights.",
        ]
    )
    report.write_text("\n".join(lines), encoding="utf-8")
    LOGGER.info("Wrote domestic ingest registry for %d source classes", len(rows))
    return {"source_count": len(rows), "report": str(report)}
