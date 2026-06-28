#!/usr/bin/env bash
# Source this file before running crust-lite on shared /workspace storage.
# It keeps project-specific caches, temporary files, logs, dependencies, data,
# and outputs under one movable work folder.

set -euo pipefail

_script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export CRUST_LITE_PROJECT_ROOT="${CRUST_LITE_PROJECT_ROOT:-$(cd -- "${_script_dir}/.." && pwd)}"
export CRUST_LITE_WORK_ROOT="${CRUST_LITE_WORK_ROOT:-$(dirname -- "${CRUST_LITE_PROJECT_ROOT}")}"

export HOME="${CRUST_LITE_HOME:-${CRUST_LITE_WORK_ROOT}}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${CRUST_LITE_WORK_ROOT}/.cache}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${XDG_CACHE_HOME}/pip}"
export TMPDIR="${TMPDIR:-${CRUST_LITE_WORK_ROOT}/tmp}"
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"

export CRUST_LITE_VENV="${CRUST_LITE_VENV:-${CRUST_LITE_PROJECT_ROOT}/.venv}"
export CRUST_LITE_LOG_DIR="${CRUST_LITE_LOG_DIR:-${CRUST_LITE_WORK_ROOT}/logs}"
export CRUST_LITE_DATA_DIR="${CRUST_LITE_DATA_DIR:-${CRUST_LITE_PROJECT_ROOT}/data}"
export CRUST_LITE_OUTPUT_DIR="${CRUST_LITE_OUTPUT_DIR:-${CRUST_LITE_PROJECT_ROOT}/outputs}"
export CRUST_LITE_CPU_THREADS="${CRUST_LITE_CPU_THREADS:-$(nproc 2>/dev/null || echo 1)}"
export CRUST_LITE_DUCKDB_THREADS="${CRUST_LITE_DUCKDB_THREADS:-${CRUST_LITE_CPU_THREADS}}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${CRUST_LITE_CPU_THREADS}}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-${CRUST_LITE_CPU_THREADS}}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-${CRUST_LITE_CPU_THREADS}}"

mkdir -p \
  "${XDG_CACHE_HOME}" \
  "${PIP_CACHE_DIR}" \
  "${TMPDIR}" \
  "${CRUST_LITE_LOG_DIR}" \
  "${CRUST_LITE_DATA_DIR}/raw" \
  "${CRUST_LITE_DATA_DIR}/interim" \
  "${CRUST_LITE_DATA_DIR}/processed" \
  "${CRUST_LITE_OUTPUT_DIR}/maps" \
  "${CRUST_LITE_OUTPUT_DIR}/tables" \
  "${CRUST_LITE_OUTPUT_DIR}/dashboard" \
  "${CRUST_LITE_OUTPUT_DIR}/reports" \
  "${CRUST_LITE_OUTPUT_DIR}/3d"

if [[ -x "${CRUST_LITE_VENV}/bin/python" ]]; then
  export PATH="${CRUST_LITE_VENV}/bin:${PATH}"
fi

export PYTHONPATH="${CRUST_LITE_PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

