#!/usr/bin/env bash
# Prepare the shared workspace for an H200 or other large-node run.

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=workspace_env.sh
source "${script_dir}/workspace_env.sh"

recreate_venv=0
fetch_missing=1
refresh_usgs=0
with_tests=0
config="configs/east_japan_usgs.yml"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --recreate-venv)
      recreate_venv=1
      shift
      ;;
    --fetch-usgs|--fetch-missing)
      fetch_missing=1
      shift
      ;;
    --refresh-usgs)
      fetch_missing=1
      refresh_usgs=1
      shift
      ;;
    --no-fetch)
      fetch_missing=0
      shift
      ;;
    --with-tests)
      with_tests=1
      shift
      ;;
    --skip-tests)
      with_tests=0
      shift
      ;;
    --config)
      config="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 2
      ;;
  esac
done

run_id="${CRUST_LITE_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
run_dir="${CRUST_LITE_RUN_DIR:-${CRUST_LITE_WORK_ROOT}/h200_runs/${run_id}}"
mkdir -p "${run_dir}"
export CRUST_LITE_RUN_ID="${run_id}"
export CRUST_LITE_RUN_DIR="${run_dir}"

cd "${CRUST_LITE_PROJECT_ROOT}"

if [[ "${recreate_venv}" == "1" ]]; then
  bash scripts/bootstrap_workspace.sh --recreate | tee "${run_dir}/bootstrap.log"
elif [[ ! -x "${CRUST_LITE_VENV}/bin/python" ]]; then
  bash scripts/bootstrap_workspace.sh | tee "${run_dir}/bootstrap.log"
fi

bash scripts/h200_preflight.sh | tee "${run_dir}/preflight.console.log"

event_csv="$("${CRUST_LITE_VENV}/bin/python" - <<PY
from pathlib import Path
import yaml

config_path = Path("${config}")
data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
value = data.get("data_sources", {}).get("event_csv")
print((Path.cwd() / value).resolve() if value else "")
PY
)"

if [[ -n "${event_csv}" && ( ! -f "${event_csv}" || "${refresh_usgs}" == "1" ) ]]; then
  if [[ "${fetch_missing}" == "1" ]]; then
    mkdir -p "$(dirname -- "${event_csv}")"
    "${CRUST_LITE_VENV}/bin/python" scripts/collect_usgs_catalog.py \
      --config "${config}" \
      --output "${event_csv}" | tee "${run_dir}/collect_usgs.log"
  else
    echo "Missing staged event CSV: ${event_csv}" >&2
    echo "Re-run without --no-fetch, or copy the staged data into the shared workspace." >&2
    exit 3
  fi
fi

if [[ "${with_tests}" == "1" ]]; then
  bash scripts/validate_workspace.sh quick | tee "${run_dir}/validate_workspace.log"
fi

echo "cpu_threads=${CRUST_LITE_CPU_THREADS}" | tee "${run_dir}/cpu_threads.log"
echo "duckdb_threads=${CRUST_LITE_DUCKDB_THREADS}" | tee -a "${run_dir}/cpu_threads.log"

# CPU-side acquisition and preprocessing. This stage intentionally finishes
# catalog QC, feature generation, historical quality profiling, compact Parquet,
# and DuckDB materialization before the H200/GPU run. Restricted domestic data
# sources are represented by the ingest registry and local import contracts
# unless credentials/local archives are configured.
"${CRUST_LITE_VENV}/bin/python" -m crust_lite.cli fetch \
  --config "${config}" | tee "${run_dir}/fetch.log"

"${CRUST_LITE_VENV}/bin/python" -m crust_lite.cli domestic-ingest \
  --config configs/japan_all_domestic.yml | tee "${run_dir}/domestic_ingest.log"

"${CRUST_LITE_VENV}/bin/python" -m crust_lite.cli build-features \
  --config "${config}" | tee "${run_dir}/build_features_and_compaction.log"

"${CRUST_LITE_VENV}/bin/python" -m crust_lite.cli transfer-functions \
  --config "${config}" | tee "${run_dir}/transfer_functions.log"

"${CRUST_LITE_VENV}/bin/python" -m crust_lite.cli array-projection \
  --config "${config}" | tee "${run_dir}/array_projection.log"

"${CRUST_LITE_VENV}/bin/python" - <<PY | tee "${run_dir}/database_materialize.log"
from crust_lite.config import load_config
from crust_lite.io.database import connect, database_engine, database_path, materialize_known_tables
from crust_lite.paths import ProjectPaths

config = load_config("${config}")
paths = ProjectPaths.from_config(config)
results = materialize_known_tables(paths)
print("database_engine", database_engine(paths))
print("database_path", database_path(paths))
for name, status in sorted(results.items()):
    print(f"{name}: {status}")
con = connect(paths)
try:
    for table in ["event_compact", "event_bin_summary", "gnss_compact", "waveform_array_projection", "gaussian_splat_primitive", "domestic_ingest_plan"]:
        try:
            count = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        except Exception as exc:
            count = f"unavailable: {type(exc).__name__}: {exc}"
        print(f"{table}_rows: {count}")
finally:
    con.close()
PY

cat > "${run_dir}/manifest.txt" <<EOF
run_id=${run_id}
project_root=${CRUST_LITE_PROJECT_ROOT}
work_root=${CRUST_LITE_WORK_ROOT}
config=${config}
event_csv=${event_csv}
fetch_missing=${fetch_missing}
refresh_usgs=${refresh_usgs}
with_tests=${with_tests}
cpu_threads=${CRUST_LITE_CPU_THREADS}
duckdb_threads=${CRUST_LITE_DUCKDB_THREADS}
prepared_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
compact_database=${CRUST_LITE_PROJECT_ROOT}/data/processed/crust_lite.duckdb
next_command=bash scripts/h200_run.sh --config ${config}
EOF

echo "h200_prepare_done=${run_dir}"

