#!/usr/bin/env bash
# Validate the shared workspace environment and core algorithms.

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=workspace_env.sh
source "${script_dir}/workspace_env.sh"

mode="${1:-quick}"
python_bin="${CRUST_LITE_VENV}/bin/python"
if [[ ! -x "${python_bin}" ]]; then
  echo "Project venv is missing: ${python_bin}" >&2
  echo "Run: bash scripts/bootstrap_workspace.sh" >&2
  exit 2
fi

cd "${CRUST_LITE_PROJECT_ROOT}"

"${python_bin}" - <<'PY'
from pathlib import Path
import os

import duckdb
import geopandas
import numpy
import pandas
import plotly
import pyproj
import shapely

from crust_lite.config import load_config
from crust_lite.io.database import database_engine
from crust_lite.paths import ProjectPaths

project = Path(os.environ["CRUST_LITE_PROJECT_ROOT"]).resolve()
work = Path(os.environ["CRUST_LITE_WORK_ROOT"]).resolve()
venv = Path(os.environ["CRUST_LITE_VENV"]).resolve()
cache = Path(os.environ["PIP_CACHE_DIR"]).resolve()
tmp = Path(os.environ["TMPDIR"]).resolve()

for label, path in {
    "project": project,
    "venv": venv,
    "cache": cache,
    "tmp": tmp,
}.items():
    if not path.is_relative_to(work):
        raise SystemExit(f"{label} path is outside work root: {path} not under {work}")

config = load_config(project / "configs" / "east_japan_usgs.yml")
paths = ProjectPaths.from_config(config)
print("workspace_root", work)
print("project_root", project)
print("database_engine", database_engine(paths))
print("duckdb", duckdb.__version__)
print("config_region", config.region.name)
PY

"${python_bin}" -m ruff check src tests
"${python_bin}" -m mypy src
"${python_bin}" -m pytest

if [[ "${mode}" == "sample" || "${mode}" == "full" ]]; then
  "${python_bin}" -m crust_lite.cli run-all --config configs/kumamoto.yml --sample
fi

if [[ "${mode}" == "full" ]]; then
  "${python_bin}" -m crust_lite.cli domestic-ingest --config configs/japan_all_domestic.yml
  "${python_bin}" -m crust_lite.cli viz-3d --config configs/east_japan_usgs.yml
fi

echo "workspace validation passed: ${mode}"

