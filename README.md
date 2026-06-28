# crust-lite

`crust-lite` is a low-resource Python OSS prototype for research workflows that
combine earthquake catalogs, focal mechanisms, GNSS, known active faults, and
J-SHIS-like ground or hazard layers around Japan.

This project does not predict earthquake dates. It outputs observed-data-based
fault candidates, differences from known faults, static stress-change proxies,
and dimensionless relative failure-index scenarios under explicit assumptions.
Do not use the outputs for disaster-response decisions. Use official agency
information for public safety decisions.

## Install

Python 3.12 or newer is expected. To avoid installing dependencies into the
system environment, install them into the project working directory:

```bash
cd crust-lite
bash scripts/bootstrap_deps.sh
```

Then run with:

```bash
PYTHONPATH=.deps-duckdb-src:src:.deps python3 -m crust_lite.cli run-all --config configs/kumamoto.yml --sample
```

## Shared `/workspace` Setup for RunPod/H200

For the current large-node workflow, keep the whole project under one shared
folder:

```text
/workspace/equake/
  crust-lite/
    .venv/
    data/
    outputs/
  .cache/
  tmp/
  logs/
```

Bootstrap or rebuild the project-local environment with:

```bash
cd /workspace/equake/crust-lite
bash scripts/bootstrap_workspace.sh
```

If the same folder is moved or mounted onto another machine such as an H200
node and the existing virtual environment is not ABI-compatible, rebuild only
the project-local environment:

```bash
cd /workspace/equake/crust-lite
bash scripts/bootstrap_workspace.sh --recreate
```

Before running preprocessing, data layout checks, or algorithm validation:

```bash
cd /workspace/equake/crust-lite
source scripts/workspace_env.sh
bash scripts/validate_workspace.sh quick
```

Validation modes:

- `quick`: import checks, path containment checks, Ruff, mypy, and pytest.
- `sample`: `quick` plus a complete artificial-sample `run-all`.
- `full`: `quick` plus domestic data registry materialization and 3D HTML refresh
  from the existing staged outputs.

The scripts set `HOME`, pip cache, temporary files, logs, `.venv`, data, and
outputs under `/workspace/equake`. Nothing is installed into the system Python
environment. When bringing results back to another machine, copy only
`/workspace/equake/crust-lite/outputs/` unless code or staged data also changed.

### CPU Preprocessing for H200 Runs

Use one CPU-side command to finish public catalog collection, local/registry data ingestion, QC, feature extraction, historical quality profiling, granularity normalization, compact Parquet output, and DuckDB materialization before using the H200 node:

```bash
cd /workspace/equake/crust-lite
bash scripts/h200_prepare.sh --config configs/east_japan_usgs.yml
```

By default, missing configured public USGS catalog CSV data are fetched. Use `--refresh-usgs` to refetch the public catalog, or `--no-fetch` to require pre-staged files. Restricted domestic sources such as GEONET, F-net, K-NET/KiK-net, and some J-SHIS products are represented by the domestic ingest registry and local import contracts unless credentials/local archives are configured.

The CPU preparation command writes compact analysis tables and a DuckDB database:

- `data/processed/event_compact.parquet`
- `data/processed/event_bin_summary.parquet`
- `data/processed/gnss_compact.parquet`
- `data/processed/data_compaction_manifest.json`
- `data/processed/crust_lite.duckdb`
- `outputs/reports/data_compaction.md`

Then run the H200 stage without repeating data collection or CPU preprocessing:

```bash
bash scripts/h200_run.sh --config configs/east_japan_usgs.yml
```

That stage runs `infer-faults`, `stress`, `simulate`, `export`, and `viz-3d` against the prepared data products. Use `--full` only when intentionally rerunning the complete pipeline on the H200 node.

The code also has minimal fallbacks so the artificial sample pipeline can run
without the full geospatial stack. In that mode, files with `.parquet` and
`.gpkg` extensions may contain CSV or GeoJSON fallback content and include
sidecar metadata explaining the physical format.

## Commands

```bash
PYTHONPATH=.deps-duckdb-src:src:.deps python3 -m crust_lite.cli fetch --config configs/kumamoto.yml
PYTHONPATH=.deps-duckdb-src:src:.deps python3 -m crust_lite.cli domestic-ingest --config configs/japan_all_domestic.yml
PYTHONPATH=.deps-duckdb-src:src:.deps python3 -m crust_lite.cli build-features --config configs/kumamoto.yml
PYTHONPATH=.deps-duckdb-src:src:.deps python3 -m crust_lite.cli infer-faults --config configs/kumamoto.yml
PYTHONPATH=.deps-duckdb-src:src:.deps python3 -m crust_lite.cli stress --config configs/kumamoto.yml
PYTHONPATH=.deps-duckdb-src:src:.deps python3 -m crust_lite.cli simulate --config configs/kumamoto.yml
PYTHONPATH=.deps-duckdb-src:src:.deps python3 -m crust_lite.cli export --config configs/kumamoto.yml
PYTHONPATH=.deps-duckdb-src:src:.deps python3 -m crust_lite.cli viz-3d --config configs/kumamoto.yml
PYTHONPATH=.deps-duckdb-src:src:.deps python3 -m crust_lite.cli dashboard --config configs/kumamoto.yml
PYTHONPATH=.deps-duckdb-src:src:.deps python3 -m crust_lite.cli run-all --config configs/kumamoto.yml --sample
```

3D options:

```bash
PYTHONPATH=.deps-duckdb-src:src:.deps python3 -m crust_lite.cli viz-3d --config configs/kumamoto.yml --mode cumulative
PYTHONPATH=.deps-duckdb-src:src:.deps python3 -m crust_lite.cli viz-3d --config configs/kumamoto.yml --mode window
PYTHONPATH=.deps-duckdb-src:src:.deps python3 -m crust_lite.cli viz-3d --config configs/kumamoto.yml --time-bin-days 30
PYTHONPATH=.deps-duckdb-src:src:.deps python3 -m crust_lite.cli viz-3d --config configs/kumamoto.yml --max-events 20000
```

## Configurations

Initial region configs are:

- `configs/kumamoto.yml`
- `configs/noto.yml`
- `configs/east_japan_usgs.yml`
- `configs/japan_all_domestic.yml`

Each config validates region bbox, local CRS, date range, filters, simulation
uncertainty ranges, data-source flags, and 3D visualization settings.

## Domestic Japan Data Registry

The command below registers every domestic data class discussed in this prototype into DuckDB-backed tables without attempting to load PB-scale raw archives into memory:

```bash
PYTHONPATH=.deps-duckdb-src:src:.deps python3 -m crust_lite.cli domestic-ingest --config configs/japan_all_domestic.yml
```

It writes:

- `data/processed/domestic_data_source.parquet`
- `data/processed/domestic_ingest_plan.parquet`
- `outputs/reports/domestic_data_ingest_plan.md`

The registry covers JMA long-term catalogs, NIED Hi-net, K-NET/KiK-net, F-net, GSI GEONET, AIST active faults, J-SHIS, SAR/InSAR, and offshore geophysical/geodetic sources. Full raw waveform, high-rate GNSS/RINEX, InSAR, and offshore archives are represented as staged external archives with storage estimates, access blockers, and a DB indexing policy. Configured local CSV/GeoJSON subsets are marked as locally ingestible.

Older historical records are retained with epoch-dependent quality, completeness magnitude, uncertainty, and analysis-weight metadata. They are not discarded simply because their precision is lower than modern instrumental data.

## Input Data Specs

### Event catalog

Output: `data/processed/event.parquet`

Required columns:

- `event_id`
- `time_utc`
- `lat`
- `lon`
- `depth_km`
- `magnitude`
- `magnitude_type`
- `catalog_source`
- `has_mechanism`
- `has_waveform_feature`
- `x_m`
- `y_m`
- `z_m`

`z_m = depth_km * 1000` and is positive downward internally.

### Mechanism CSV

MVP import is local CSV:

```text
mechanism_id,event_id,strike1,dip1,rake1,strike2,dip2,rake2,scalar_moment_nm,source
```

Output: `data/processed/mechanism.parquet`.

### GNSS daily CSV

```text
station_id,date,lat,lon,east_m,north_m,up_m,sigma_e,sigma_n,sigma_u
```

Outputs:

- `data/processed/gnss_daily.parquet`
- `data/processed/gnss_features.parquet`

The MVP computes station horizontal velocities and a simple relative strain
gradient score.

### Active faults

MVP import is local GeoJSON FeatureCollection. Output:
`data/processed/fault_segment.gpkg`. In minimal mode this path contains GeoJSON
fallback content with a metadata sidecar.

### J-SHIS-like layers

The MVP writes a placeholder sample layer if online API retrieval is unavailable.
Raw JSON is stored under `data/raw/jshis/`; processed output is
`data/processed/jshis_features.parquet`.

### Waveforms

Waveforms are optional. With `use_waveforms=false`, the pipeline writes an empty
`data/processed/waveform_feature.parquet` and continues. For transfer-function
analysis, use `crust-lite transfer-functions --config ... --sample` or configure
`data_sources.waveform_spectra_csv`. The spectrum CSV must retain phase and
time-delay information, not just amplitude:

- `event_id`, `station_id`, `time_utc`
- `lat`, `lon`
- `frequency_hz`, `amplitude`, `phase_rad`, `group_delay_s`
- optional `p_residual_s`, `s_residual_s`, `source`

The MVP estimates relative complex site transfer functions, validates them with
leave-one-event-out amplitude/phase prediction, scores structural singularity,
and compares anomaly stations with known/inferred fault distances. This is not a
unique 3D subsurface inversion. Future waveform output
columns are:

```text
event_id,station_id,channel,pga,pgv,psa_0p3,psa_1p0,psa_3p0,p_residual_s,s_residual_s,amp_residual_log,source
```

## Outputs

Main outputs:

- `data/interim/event_qc.parquet`
- `data/processed/inferred_faults.gpkg`
- `data/processed/stress_state.parquet`
- `outputs/tables/failure_scenarios.parquet`
- `outputs/tables/fault_ranking.csv`
- `outputs/maps/event_map.png`
- `outputs/maps/known_faults_map.png`
- `outputs/maps/inferred_faults_map.png`
- `outputs/maps/stress_map_latest.png`
- `outputs/maps/failure_index_100yr_map.png`
- `outputs/reports/summary.md`

3D HTML outputs include a local-CRS map overlay with the configured bbox and latitude/longitude graticule:

- `outputs/3d/index.html`
- `outputs/3d/events_faults_timeseries.html`
- `outputs/3d/stress_timeseries_3d.html`
- `outputs/3d/failure_scenarios_3d.html`
- `outputs/3d/japan_archipelago_context.html`
- `outputs/3d/metadata.json`

Open the 3D directory directly in a browser, or serve it locally:

```bash
python3 -m http.server 8000 -d outputs/3d
```

Then open `http://localhost:8000/index.html`.

## Map Overlay

Static PNG maps and 3D HTML outputs include an offline map overlay built from the configured `region.bbox`. The 3D outputs also include a simplified offline Japan archipelago outline and a separate `japan_archipelago_context.html` overview. The outline is cartographic context only, not analytical coastline data. It does not download external map tiles.

## 3D Depth Convention

Internal depth remains positive downward:

```text
z_m = depth_km * 1000
```

For Plotly display:

```text
plot_z_m = -1.0 * z_m * vertical_exaggeration
```

Tooltips display original `depth_km`, not the exaggerated plotting coordinate.

## Sample Data

`data/raw/sample/` contains artificial data:

- `sample_events.csv`
- `sample_mechanisms.csv`
- `sample_gnss_daily.csv`
- `sample_known_faults.geojson`

The sample is artificial and cannot be used for scientific interpretation.
When sample data are used, outputs include `is_sample_data=true` in metadata and
HTML.

## Stress and Scenario Interpretation

The sign convention is:

```text
DeltaCFS = DeltaTau + mu_prime * DeltaSigmaN
```

Opening normal stress is positive. If `cutde` is unavailable, the MVP writes
`stress_method="fallback_approximation"` and uses `cfs_score_approx`, a
dimensionless relative score. It must not be interpreted as Pa.

`failure_index` is dimensionless:

```text
failure_index_f(t) = max(0, tau_f(t) + CFS_f(t)) / strength_f
```

Values near or above 1 indicate threshold exceedance under the chosen model
assumptions only. They are not event-time predictions.

## Database and Large Mesh Policy

`crust-lite` uses a local analytical database for large tabular products. In this environment DuckDB was built from source into `.deps-duckdb-src` and is preferred over the broken binary wheel; SQLite remains a fallback. Event catalogs, stress rows, scenario rows, and rankings are materialized in `data/processed/crust_lite.duckdb`.

Large finite-element meshes should not be forced entirely into SQL. The recommended v1 layout is:

- DuckDB: mesh dataset metadata, node/element index tables, spatial/time subset queries.
- Parquet: portable node/element tables when they are table-shaped.
- HDF5/Zarr/XDMF/VTK/Exodus: large dense or time-varying field arrays and solver exchange files.

The MVP initializes empty extension tables: `mesh_dataset`, `mesh_node`, `mesh_element`, and `mesh_field_index`.

## Limitations

- No earthquake date, place, or magnitude is predicted.
- No full 3D finite-element model is run in the MVP.
- No full waveform inversion is implemented.
- J-SHIS online acquisition is a placeholder/fallback path in the MVP.
- F-net mechanisms and GEONET daily coordinates require local CSV import in the MVP.
- `cutde` stress calculation is an extension point; fallback stress is approximate.
- Fallback `.parquet` and `.gpkg` files are not strict physical Parquet/GeoPackage
  without the optional dependency stack.
- Rankings are always paired with uncertainty fields and must not be shown as
  deterministic outcomes.

## v1 Candidates

- F-net mechanism automatic retrieval
- GEONET daily coordinate automatic retrieval
- J-SHIS API stabilization
- K-NET/KiK-net waveform features
- Strict triangular dislocation stress calculation with `cutde`
- Local 2D or coarse 3D finite-element validation with PyLith
- OpenQuake or J-SHIS hazard comparison
- Backtesting evaluation
