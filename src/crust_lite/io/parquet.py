from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def _normalise(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    return value


def write_table(rows: list[dict[str, Any]], path: Path, metadata: dict[str, Any] | None = None) -> None:
    """Write a table.

    Pandas/pyarrow are used when available. In the minimal sample environment this
    falls back to CSV content at the requested path and records the physical
    format in a sidecar metadata file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = dict(metadata or {})
    try:
        import pandas as pd  # type: ignore

        pd.DataFrame(rows).to_parquet(path, index=False)
        metadata["physical_format"] = "parquet"
    except Exception as exc:
        fieldnames = sorted({key for row in rows for key in row})
        with path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: _normalise(row.get(key, "")) for key in fieldnames})
        metadata["physical_format"] = "csv_fallback"
        metadata["parquet_fallback_reason"] = f"{type(exc).__name__}: {exc}"
    write_sidecar(path, metadata)


def read_table(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Required table not found: {path}")
    try:
        import pandas as pd  # type: ignore

        df = pd.read_parquet(path)
        return [dict(row) for row in df.to_dict(orient="records")]
    except Exception:
        with path.open("r", encoding="utf-8", newline="") as fh:
            return [dict(row) for row in csv.DictReader(fh)]


def write_sidecar(path: Path, metadata: dict[str, Any]) -> None:
    sidecar = path.with_suffix(path.suffix + ".metadata.json")
    sidecar.write_text(json.dumps(metadata, indent=2, sort_keys=True, default=str), encoding="utf-8")


def read_sidecar(path: Path) -> dict[str, Any]:
    sidecar = path.with_suffix(path.suffix + ".metadata.json")
    if not sidecar.exists():
        return {}
    return json.loads(sidecar.read_text(encoding="utf-8"))


def require_columns(rows: list[dict[str, Any]], columns: set[str], table_name: str) -> None:
    if not rows:
        raise ValueError(f"{table_name} is empty")
    missing = columns.difference(rows[0])
    if missing:
        raise ValueError(f"{table_name} missing required columns: {sorted(missing)}")
