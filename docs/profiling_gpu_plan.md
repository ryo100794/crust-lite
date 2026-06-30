# crust-lite profiling and GPU execution plan

Date: 2026-06-30 UTC

This plan is for research-state visualization and waveform-derived structure screening. It is not an earthquake-date prediction workflow.


## 2026-06-30 Update

The current CPU handoff is no longer the 6,500-splat baseline. After increasing array projection density and CPU pre-GPU preparation:

| Artifact | Rows / count | Size |
| --- | ---: | ---: |
| waveform spectra | 289,590 rows | 9.1 MiB parquet |
| waveform array projections | 113,504 rows | 9.6 MiB parquet |
| Gaussian splat primitives | 113,504 rows | 12 MiB parquet |
| voxel LOD table | 5,057,513 rows | 260 MiB parquet |
| view-image handoff r12 | 601,854,017 pixel rows / 128 parts | about 10 GiB |
| view-image handoff r20 | 1,264,851,772 pixel rows / 256 parts | about 20 GiB |

The r20 run took `real 4171.731s`, `user 2815.822s`, `sys 209.066s` on a RunPod container restricted to two effective CPUs. This is now a one-hour-class CPU preparation workload and is suitable as the pre-GPU handoff baseline.

Current GPU handoff entry point:

- `outputs/gpu_prep/manifest.json`
- `data/processed/splat_view_image_part_index_r20_p2500_s256.parquet`
- `data/processed/splat_view_image_parts_r20_p2500_s256/part-*.parquet`

Do not load all r20 parts at once on the GPU side. Read the part index and stream mini-batches by part or by groups of parts.

## Current Profiling Baseline

Latest full CPU rebuild:

- Run: `m50_hinet_reflection_splats_20260629T125241Z`
- Config: `configs/_runtime_japan_all_usgs_modern_m2_m50_hinet_sharded.yml`
- Node GPU status: `nvidia-smi not found`, so this is CPU profiling only
- Memory available at preflight: about 539 GiB
- Raw/project data: about 12 GiB
- Outputs: about 79 MiB

Observed row counts and compact handoff sizes:

| Artifact | Rows | Size |
| --- | ---: | ---: |
| events | 40,487 | `event.parquet` 2.1 MiB |
| waveform features | 57,918 | `waveform_feature.parquet` 3.9 MiB |
| waveform spectra | 289,590 | `waveform_spectrum.parquet` 9.1 MiB |
| site transfer functions | 794 stations | `site_transfer_function.parquet` 168 KiB |
| waveform array projections | 6,500 | `waveform_array_projection.parquet` 797 KiB |
| Gaussian splat primitives | 6,500 | `gaussian_splat_primitive.parquet` 816 KiB |
| stress event-fault pairs | 1,659,967 | `stress_state.parquet` 26 MiB |

Approximate CPU timing from the latest log:

| Step | Approx. elapsed |
| --- | ---: |
| local event/waveform fetch from CSV | 10 s |
| catalog QC + GNSS features | 2 s |
| historical quality | 13 s |
| compaction | 3 s |
| DuckDB transfer functions | 48 s |
| waveform array projection and splats | 38 s |
| fault inference | 20 s |
| fallback stress | 23 s |
| simulation | 10 s |
| static maps | 7 s |
| 3D HTML | 11 s |
| run-all after first data log | about 188 s |

Current bottlenecks are transfer-function aggregation, array projection, fault inference, and stress. The current GPU handoff tables are compact enough that GPU loading is not the bottleneck.

## Profiling Plan

1. Keep the current CPU-first full run as the baseline.
2. Add cProfile/per-step wall-clock profiles for:
   - `transfer-functions`
   - `array-projection`
   - `infer-faults`
   - `stress`
   - `viz-3d`
3. Record:
   - wall time
   - max RSS
   - DuckDB database size
   - input/output row counts
   - output file sizes
4. Store reports under:
   - `outputs/reports/profile_summary.md`
   - `outputs/reports/profile_*.prof`
   - `outputs/reports/profile_*.txt`

Suggested commands on the cloud workspace:

```bash
cd /workspace/equake/crust-lite
time .venv/bin/python -m crust_lite.cli transfer-functions --config configs/_runtime_japan_all_usgs_modern_m2_m50_hinet_sharded.yml
time .venv/bin/python -m crust_lite.cli array-projection --config configs/_runtime_japan_all_usgs_modern_m2_m50_hinet_sharded.yml
time .venv/bin/python -m crust_lite.cli infer-faults --config configs/_runtime_japan_all_usgs_modern_m2_m50_hinet_sharded.yml
time .venv/bin/python -m crust_lite.cli stress --config configs/_runtime_japan_all_usgs_modern_m2_m50_hinet_sharded.yml
```

For function-level CPU profiling:

```bash
cd /workspace/equake/crust-lite
.venv/bin/python -m cProfile -o outputs/reports/profile_array_projection.prof -m crust_lite.cli array-projection --config configs/_runtime_japan_all_usgs_modern_m2_m50_hinet_sharded.yml
.venv/bin/python - <<'PY'
import pstats
p = pstats.Stats("outputs/reports/profile_array_projection.prof")
p.strip_dirs().sort_stats("cumtime").print_stats(40)
PY
```

## GPU Execution Plan

### Principle

Do not move raw waveform bulk processing to GPU first. CPU/DuckDB should produce compact, normalized tables. GPU should consume compact tensors and spend time on fusion, differentiable rendering, and validation.

### GPU input contract

Minimum GPU inputs:

- `data/processed/gaussian_splat_primitive.parquet`
- `data/processed/waveform_array_projection.parquet`
- `data/processed/waveform_spectrum.parquet`
- `data/processed/site_transfer_function.parquet`
- `data/processed/structure_anomaly.parquet`
- `data/processed/inferred_faults.gpkg`

Initial tensor layout:

- means: `[N, 3]` local CRS meters, z positive-down internally
- display means: `[N, 3]` with `plot_z_m = -z_m * vertical_exaggeration`
- scales: `[N, 3]` from `sigma_x_m`, `sigma_y_m`, `sigma_z_m`
- colors: `[N, 3]`
- opacity: `[N]`
- primitive type: direct/reflected/scattered/residual as integer labels
- quality weights: `array_coherence`, `beam_power`, `scatter_weight`

### 5090 rehearsal

Goal: validate tensor loading, camera/view rendering, primitive filtering, and multi-type splat display before spending H200 time.

Steps:

1. Create a workspace-local GPU venv, not a global install:
   - `/workspace/equake/crust-lite/.venv-gpu`
2. Install GPU deps there only:
   - PyTorch CUDA build matching the host driver
   - `pyarrow`, `duckdb`, `numpy`
   - optional splat renderer dependency if compatible
3. Load 6,500 current splats and render:
   - all primitive types
   - direct-only
   - reflected/scattered-only
   - top-k by `beam_power`
4. Produce:
   - `outputs/gpu/rehearsal_splat_tensor.pt`
   - `outputs/gpu/rehearsal_render_*.png`
   - `outputs/gpu/rehearsal_profile.json`
5. Confirm:
   - no CPU-side full waveform load
   - GPU memory use is stable
   - primitive type filtering works
   - rendered splats are not just hypocenter overlays

### H200 run

Current 113,504 CPU splats plus the r20 view-image handoff are large enough for a 5090 rehearsal and initial H200 data-loader validation. Full H200 optimization still needs balanced partitions and GPU kernels. Before H200:

1. Increase CPU candidate density:
   - `waveform_array.max_events`: 2,000 -> 5,000 or more
   - `top_projections_per_event`: 4 -> 16 or 32
   - `max_projection_rows`: 100,000 -> 500,000+
   - `max_splats`: 100,000 -> 500,000+
2. Add late-phase window extraction from raw waveform windows, not only point spectra:
   - output `waveform_window_beam.parquet`
   - output richer `splat_candidate.parquet`
3. Run CPU preprocessing until output is compact:
   - target first H200 input: 100k to 500k splats
   - target upper test: 1M splats if storage and browser output are separated from GPU tensor output
4. H200 GPU stage:
   - load compact tensors
   - fuse direct/reflected/scattered candidates
   - optimize opacity/scale/position within bounded priors
   - validate against held-out events/stations
   - export GPU-optimized splats separately from CPU candidates

Expected H200 outputs:

- `outputs/gpu/splat_candidates.arrow` or parquet shards
- `outputs/gpu/splat_model.pt`
- `outputs/gpu/optimized_gaussian_splat_primitive.parquet`
- `outputs/gpu/validation_metrics.json`
- `outputs/gpu/render_preview.html`
- `outputs/reports/gpu_run_summary.md`

## Readiness

Ready now:

- CPU/DuckDB preprocessing
- compact handoff tables
- primitive type labels: direct/reflected/scattered/residual
- 3D Plotly QA output

Not ready yet:

- true GPU 3DGS optimization
- H200 utilization at current 6,500 splat scale
- raw-waveform late-window beam volume
- held-out waveform validation loop for optimized splats

Next implementation step:

Build `crust-lite gpu-preflight` and `crust-lite gpu-rehearsal` commands that run entirely from `data/processed` and write only to `outputs/gpu`.
