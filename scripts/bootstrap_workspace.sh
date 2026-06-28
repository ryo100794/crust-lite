#!/usr/bin/env bash
# Build a project-local Python environment on shared /workspace storage.
# Nothing is installed into the system Python environment.

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=workspace_env.sh
source "${script_dir}/workspace_env.sh"

recreate=0
if [[ "${1:-}" == "--recreate" ]]; then
  recreate=1
fi

if [[ "${recreate}" == "1" && -d "${CRUST_LITE_VENV}" ]]; then
  rm -rf "${CRUST_LITE_VENV}"
fi

python_bin="${PYTHON_BIN:-}"
if [[ -z "${python_bin}" ]]; then
  if command -v python3.12 >/dev/null 2>&1; then
    python_bin="python3.12"
  elif command -v python3 >/dev/null 2>&1; then
    python_bin="python3"
  else
    echo "Python 3.12 is required, but python3.12/python3 was not found." >&2
    exit 2
  fi
fi

"${python_bin}" - <<'PY'
import sys

if sys.version_info < (3, 12):
    raise SystemExit("Python 3.12 or newer is required.")
PY

if [[ ! -x "${CRUST_LITE_VENV}/bin/python" ]]; then
  "${python_bin}" -m venv "${CRUST_LITE_VENV}"
fi

venv_python="${CRUST_LITE_VENV}/bin/python"
"${venv_python}" -m pip install --upgrade pip setuptools wheel
"${venv_python}" -m pip install -e "${CRUST_LITE_PROJECT_ROOT}[dev,fast-json]"

"${venv_python}" - <<'PY'
from pathlib import Path
import sys

import duckdb
import geopandas
import numpy
import pandas
import plotly
import pyproj
import shapely

import crust_lite

print("python", sys.version.split()[0])
print("duckdb", duckdb.__version__)
print("crust_lite", Path(crust_lite.__file__).resolve())
PY

cat <<EOF

crust-lite workspace is ready.
Project root: ${CRUST_LITE_PROJECT_ROOT}
Work root:    ${CRUST_LITE_WORK_ROOT}
Venv:         ${CRUST_LITE_VENV}
Cache:        ${PIP_CACHE_DIR}
Tmp:          ${TMPDIR}

Use:
  source ${CRUST_LITE_PROJECT_ROOT}/scripts/workspace_env.sh
  python -m crust_lite.cli --help
EOF

