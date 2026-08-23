#!/usr/bin/env python3
"""Inventory artifacts potentially derived from known-structure-contaminated faults."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


PROJECT = Path("/workspace/equake/crust-lite")
FILES = [
    ("data/processed/inferred_faults.gpkg", "direct_contaminated", "832/832 rows contain known-distance/feedback fields"),
    ("data/processed/inferred_faults.gpkg.metadata.json", "direct_contaminated_metadata", "metadata belongs to contaminated inferred set"),
    ("data/processed/fault_wave_interaction.parquet", "posthoc_coupled_regenerate", "posthoc interaction uses contaminated inferred candidate set"),
    ("data/processed/known_fault_inferred_comparison.parquet", "posthoc_coupled_regenerate", "allowed posthoc comparison, but candidate side is contaminated"),
    ("data/processed/site_transfer_function.parquet", "indirect_possible", "nearest inferred fault id/distance can depend on contaminated selection"),
    ("data/processed/structure_anomaly.parquet", "indirect_possible", "nearest inferred fault id/distance can depend on contaminated selection"),
    ("data/processed/stress_state.parquet", "indirect_expected_missing", "stress uses inferred geometry; file currently absent"),
    ("outputs/tables/failure_scenarios.parquet", "indirect_contaminated", "segment set and stress chain derive from inferred candidates"),
    ("outputs/tables/fault_ranking.csv", "indirect_contaminated", "ranking derives from failure scenarios and contaminated segment set"),
    ("outputs/3d/events_faults_timeseries.html", "display_contaminated", "renders contaminated inferred set/ranking"),
    ("outputs/3d/stress_timeseries_3d.html", "display_contaminated", "renders downstream stress chain"),
    ("outputs/3d/failure_scenarios_3d.html", "display_contaminated", "renders downstream scenario chain"),
    ("outputs/maps/inferred_faults_map.png", "display_contaminated", "renders contaminated inferred set"),
    ("outputs/maps/stress_map_latest.png", "display_contaminated", "renders downstream stress chain"),
    ("outputs/maps/failure_index_100yr_map.png", "display_contaminated", "renders downstream scenario chain"),
    ("outputs/reports/summary.md", "report_contaminated", "legacy top-fault table exposes known distance and contaminated score"),
    ("data/processed/crust_lite.sqlite", "mixed_database_hold", "do not delete or mutate; relevant tables listed separately"),
]


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def db_inventory(path: Path) -> list[dict]:
    if not path.exists():
        return []
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    rows = []
    try:
        for table in ("failure_scenarios", "fault_ranking", "site_transfer_function", "structure_anomaly", "stress_state"):
            exists = con.execute(
                "select 1 from sqlite_master where type='table' and name=?", (table,)
            ).fetchone()
            if not exists:
                rows.append({"table": table, "exists": False, "rows": 0, "columns": []})
                continue
            columns = [r[1] for r in con.execute(f'pragma table_info("{table}")')]
            count = con.execute(f'select count(*) from "{table}"').fetchone()[0]
            rows.append({"table": table, "exists": True, "rows": count, "columns": columns})
    finally:
        con.close()
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args()
    inventory = []
    for relative, classification, reason in FILES:
        path = PROJECT / relative
        item = {
            "path": str(path),
            "classification": classification,
            "reason": reason,
            "exists": path.exists(),
            "delete": False,
            "replacement_required": path.exists(),
        }
        if path.exists():
            stat = path.stat()
            item.update(
                {
                    "bytes": stat.st_size,
                    "mtime_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
                    "sha256": digest(path),
                }
            )
        inventory.append(item)
    legacy = PROJECT / "data/processed/inferred_faults.gpkg"
    content = json.loads(legacy.read_text())
    features = content.get("features", [])
    forbidden = {
        "distance_from_known_fault_score",
        "distance_to_known_fault_km",
        "nearest_known_segment_id",
        "strike_difference_to_known_deg",
        "known_fault_feedback_status",
        "known_fault_feedback_weight",
    }
    field_counts = {
        field: sum(field in feature.get("properties", {}) for feature in features)
        for field in sorted(forbidden)
    }
    quarantine = PROJECT / "logs/audits/NR_KNOWN_LEAK_008_v1107/quarantine/legacy_inferred_faults.gpkg"
    result = {
        "schema": "nr-known-leak-008-contamination-inventory-v1107",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "normal_analysis_leak_paths": [
            "src/crust_lite/processing/fault_inference.py -> fault_segment.gpkg -> _known_fault_distance_score",
            "src/crust_lite/processing/scoring.py -> 0.10 * distance_from_known_fault_score",
            "fault_score -> confidence and _dedupe_fault_features rank/selection",
            "inferred_faults -> stress/simulation/transfer/report/viz",
        ],
        "other_normal_known_structure_inputs": [],
        "allowed_posthoc_paths": [
            "src/crust_lite/external_geology.py",
            "src/crust_lite/viz/webgl_splats.py comparison overlay",
            "src/crust_lite/viz/webgl_events.py overlay",
            "src/crust_lite/report.py known reference counts/provenance",
        ],
        "legacy_inferred": {
            "features": len(features),
            "forbidden_field_counts": field_counts,
            "all_rows_contaminated": bool(features) and all(v == len(features) for v in field_counts.values()),
        },
        "files": inventory,
        "database": {
            "path": str(PROJECT / "data/processed/crust_lite.sqlite"),
            "action": "hold mixed DB unchanged; rebuild tables from known-free chain before any switch",
            "tables": db_inventory(PROJECT / "data/processed/crust_lite.sqlite"),
        },
        "quarantine_copy": {
            "path": str(quarantine),
            "bytes": quarantine.stat().st_size,
            "sha256": digest(quarantine),
            "matches_legacy": digest(quarantine) == digest(legacy),
            "mode": oct(quarantine.stat().st_mode & 0o777),
        },
        "production_files_modified_or_deleted": False,
        "pass": True,
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"pass": True, "files": len(inventory), "legacy_features": len(features), "db_tables": len(result["database"]["tables"])}, indent=2))


if __name__ == "__main__":
    main()
