#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
python3 -m pip install --target .deps -e ".[dev,fast-json]"

cat <<'MSG'
Dependencies were installed into .deps only.
Use:
  PYTHONPATH=.deps:src python3 -m crust_lite.cli run-all --config configs/kumamoto.yml --sample
MSG
