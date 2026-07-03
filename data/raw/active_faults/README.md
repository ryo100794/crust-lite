# Active fault source data

Place authoritative or user-supplied active-fault trace data here and point
`data_sources.active_fault_file` at it. Supported inputs are GeoJSON, GeoPackage,
Shapefile, and CSV trace vertices with `segment_id`, `lon`, and `lat`.

The file `japan_major_active_faults_coarse_seed.geojson` is retained only as a
reference seed for software integration. It is not an official trace dataset and
is rejected from `data/processed/fault_segment.gpkg` by default.
