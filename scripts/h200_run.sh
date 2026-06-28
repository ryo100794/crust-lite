#!/usr/bin/env bash
# Execute a standard crust-lite run on the shared H200 workspace.

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=workspace_env.sh
source "${script_dir}/workspace_env.sh"

config="configs/east_japan_usgs.yml"
sample=0
viz_only=0
full=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      config="$2"
      shift 2
      ;;
    --sample)
      sample=1
      shift
      ;;
    --viz-only)
      viz_only=1
      shift
      ;;
    --full)
      full=1
      shift
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

cd "${CRUST_LITE_PROJECT_ROOT}"

bash scripts/h200_preflight.sh | tee "${run_dir}/preflight.console.log"

if [[ "${viz_only}" == "1" ]]; then
  "${CRUST_LITE_VENV}/bin/python" -m crust_lite.cli viz-3d \
    --config "${config}" | tee "${run_dir}/viz_3d.log"
else
  if [[ "${full}" == "1" ]]; then
    if [[ "${sample}" == "1" ]]; then
      "${CRUST_LITE_VENV}/bin/python" -m crust_lite.cli run-all \
        --config "${config}" --sample | tee "${run_dir}/run_all.log"
    else
      "${CRUST_LITE_VENV}/bin/python" -m crust_lite.cli run-all \
        --config "${config}" | tee "${run_dir}/run_all.log"
    fi
  else
    for command in infer-faults stress simulate export viz-3d; do
      "${CRUST_LITE_VENV}/bin/python" -m crust_lite.cli "${command}" \
        --config "${config}" | tee "${run_dir}/${command}.log"
    done
  fi
fi

tar -czf "${run_dir}/outputs.tar.gz" outputs

cat > "${run_dir}/run_manifest.txt" <<EOF
run_id=${run_id}
project_root=${CRUST_LITE_PROJECT_ROOT}
work_root=${CRUST_LITE_WORK_ROOT}
config=${config}
sample=${sample}
viz_only=${viz_only}
full=${full}
completed_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
outputs=${CRUST_LITE_PROJECT_ROOT}/outputs
outputs_archive=${run_dir}/outputs.tar.gz
EOF

echo "h200_run_done=${run_dir}"

