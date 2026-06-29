#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${script_dir}/workspace_env.sh"

config="configs/japan_all_usgs_modern_m2.yml"
wait_fdsn=0
wait_hinet=0
run_stage=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      config="$2"
      shift 2
      ;;
    --wait-fdsn)
      wait_fdsn=1
      shift
      ;;
    --wait-hinet)
      wait_hinet=1
      shift
      ;;
    --merge-only)
      run_stage=0
      shift
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 2
      ;;
  esac
done

cd "${CRUST_LITE_PROJECT_ROOT}"

wait_pid_file() {
  local path="$1"
  local label="$2"
  local pid=""
  if [[ -f "${path}" ]]; then
    pid="$(cat "${path}" 2>/dev/null || true)"
  fi
  if [[ -n "${pid}" ]]; then
    echo "waiting_${label}_pid=${pid}"
    while kill -0 "${pid}" 2>/dev/null; do
      sleep 30
    done
  fi
}

if [[ "${wait_fdsn}" == "1" ]]; then
  wait_pid_file /workspace/equake/logs/fdsn_waveforms.pid fdsn
  wait_pid_file /workspace/equake/logs/fdsn_waveforms_m55.pid fdsn_m55
fi
if [[ "${wait_hinet}" == "1" ]]; then
  wait_pid_file /workspace/equake/logs/hinet_waveforms.pid hinet
fi

bash scripts/merge_waveforms_for_config.sh

if [[ "${run_stage}" == "1" ]]; then
  bash scripts/h200_prepare.sh --config "${config}"
  bash scripts/h200_run.sh --config "${config}"
fi
