#!/usr/bin/env bash
# Print a reproducible H200/large-node preflight report.

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=workspace_env.sh
source "${script_dir}/workspace_env.sh"

run_id="${CRUST_LITE_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
run_dir="${CRUST_LITE_RUN_DIR:-${CRUST_LITE_WORK_ROOT}/h200_runs/${run_id}}"
mkdir -p "${run_dir}"

report="${run_dir}/preflight.txt"

{
  echo "run_id=${run_id}"
  echo "timestamp_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "project_root=${CRUST_LITE_PROJECT_ROOT}"
  echo "work_root=${CRUST_LITE_WORK_ROOT}"
  echo "venv=${CRUST_LITE_VENV}"
  echo "cache=${PIP_CACHE_DIR}"
  echo "tmp=${TMPDIR}"
  echo
  echo "== system =="
  uname -a || true
  nproc || true
  free -h || true
  df -h "${CRUST_LITE_WORK_ROOT}" || true
  echo
  echo "== gpu =="
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi || true
  else
    echo "nvidia-smi not found; GPU availability not confirmed on this node."
  fi
  echo
  echo "== git =="
  cd "${CRUST_LITE_PROJECT_ROOT}"
  git status --short --branch 2>/dev/null || true
  git log --oneline -1 2>/dev/null || true
  git remote -v 2>/dev/null || true
  echo
  echo "== python =="
  if [[ -x "${CRUST_LITE_VENV}/bin/python" ]]; then
    "${CRUST_LITE_VENV}/bin/python" - <<'PY'
from pathlib import Path
import sys

import duckdb
import crust_lite
from crust_lite.config import load_config
from crust_lite.io.database import database_engine, database_path
from crust_lite.paths import ProjectPaths

config = load_config("configs/east_japan_usgs.yml")
paths = ProjectPaths.from_config(config)
print("python", sys.version.split()[0])
print("duckdb", duckdb.__version__)
print("crust_lite", Path(crust_lite.__file__).resolve())
print("database_engine", database_engine(paths))
print("database_path", database_path(paths))
PY
  else
    echo "venv python not found: ${CRUST_LITE_VENV}/bin/python"
  fi
  echo
  echo "== data/output size =="
  du -sh \
    "${CRUST_LITE_PROJECT_ROOT}/data" \
    "${CRUST_LITE_PROJECT_ROOT}/outputs" \
    "${CRUST_LITE_PROJECT_ROOT}/.venv" \
    "${CRUST_LITE_WORK_ROOT}/.cache" \
    "${CRUST_LITE_WORK_ROOT}/tmp" 2>/dev/null || true
} | tee "${report}"

echo "preflight_report=${report}"

