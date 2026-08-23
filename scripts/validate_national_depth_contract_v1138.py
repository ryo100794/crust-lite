#!/usr/bin/env python3
"""Validate the inactive national 1.875 km physical-depth contract."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import duckdb
import numpy as np


EXPECTED_DEPTHS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.5, 9.0, 11.0, 13.0, 15.0, 18.0, 21.0, 24.0, 27.0, 30.0]
EXPECTED_INTERVALS = [0.5, 0.5, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 1.5, 1.5, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0, 3.0]


class ContractError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def validate_structure(contract: dict) -> None:
    require(contract.get("schema") == "national-1p875-physical-depth-contract-candidate-v1138", "schema")
    require(contract.get("status") == "INACTIVE_CANDIDATE", "status")
    require(contract.get("activation_allowed") is False, "activation must remain blocked")
    require(contract.get("publication_allowed") is False, "publication must remain blocked")
    grid = contract.get("physical_analysis_grid", {})
    require(grid.get("horizontal_spacing_m") == 1875.0, "horizontal spacing")
    require(grid.get("vertical_coordinate_column") == "z_km", "vertical coordinate column")
    require(grid.get("vertical_coordinate_unit") == "km", "vertical coordinate unit")
    require(grid.get("positive_vertical_direction") == "down", "positive vertical direction")
    require(grid.get("depth_planes_km") == EXPECTED_DEPTHS, "depth planes")
    require(grid.get("adjacent_depth_intervals_km") == EXPECTED_INTERVALS, "depth intervals")
    require(grid.get("depth_plane_count") == 19, "depth plane count")
    require(grid.get("uniform_depth_spacing") is False, "uniform spacing must be false")
    require(grid.get("minimum_adjacent_interval_km") == 0.5, "minimum interval")
    require(grid.get("maximum_adjacent_interval_km") == 3.0, "maximum interval")
    require("depth_resolution_km" not in grid, "ambiguous depth_resolution_km field prohibited")
    display = contract.get("display_transform", {})
    require(display.get("part_of_physical_grid") is False, "display transform must be separate")
    require(display.get("vertical_exaggeration_unit") == "dimensionless", "display exaggeration unit")
    require(display.get("changes_physical_depth_planes") is False, "display must not mutate physical depth")
    policy = contract.get("label_policy", {})
    require(policy.get("uniform_0p5km_claim_allowed") is False, "uniform 0.5 km prose must fail closed")
    require(policy.get("quantization_step_is_physical_resolution") is False, "quantization/resolution conflation")
    require(policy.get("vertical_exaggeration_is_physical_resolution") is False, "exaggeration/resolution conflation")


def validate_project(project: Path, contract: dict) -> dict:
    validate_structure(contract)
    checks: dict[str, bool] = {}
    resolved: dict[str, str] = {}
    for name, item in contract["bound_inputs"].items():
        path = project / item["path"]
        actual = sha256(path) if path.is_file() else None
        checks[f"hash_{name}"] = actual == item["sha256"]
        resolved[name] = actual or "MISSING"
    require(all(checks.values()), "bound input hash mismatch")

    grid_path = project / contract["bound_inputs"]["source_grid"]["path"]
    con = duckdb.connect()
    schema = con.execute("describe select * from read_parquet(?)", [str(grid_path)]).fetchall()
    columns = [(str(row[0]), str(row[1])) for row in schema]
    counts = con.execute(
        "select count(*),count(distinct x_m),count(distinct y_m),count(distinct z_km),min(z_km),max(z_km) from read_parquet(?)",
        [str(grid_path)],
    ).fetchone()
    depths = [float(row[0]) for row in con.execute("select distinct z_km from read_parquet(?) order by z_km", [str(grid_path)]).fetchall()]
    dx = [float(row[0]) for row in con.execute(
        "select distinct round(x_m-lag_x,6) from (select x_m,lag(x_m) over(order by x_m) lag_x from (select distinct x_m from read_parquet(?))) where lag_x is not null order by 1",
        [str(grid_path)],
    ).fetchall()]
    dy = [float(row[0]) for row in con.execute(
        "select distinct round(y_m-lag_y,6) from (select y_m,lag(y_m) over(order by y_m) lag_y from (select distinct y_m from read_parquet(?))) where lag_y is not null order by 1",
        [str(grid_path)],
    ).fetchall()]
    per_depth = con.execute("select z_km,count(*) from read_parquet(?) group by z_km order by z_km", [str(grid_path)]).fetchall()
    con.close()
    checks.update({
        "source_columns_exact": columns == [("x_m", "DOUBLE"), ("y_m", "DOUBLE"), ("z_km", "DOUBLE")],
        "source_counts_exact": list(counts) == [11_766_320, 1180, 1320, 19, 0.5, 30.0],
        "source_depths_exact": depths == EXPECTED_DEPTHS,
        "source_depth_interval_vector_exact": np.diff(depths).tolist() == EXPECTED_INTERVALS,
        "source_horizontal_spacing_exact": dx == [1875.0] and dy == [1875.0],
        "every_depth_has_all_horizontal_cells": all(int(row[1]) == 619_280 for row in per_depth),
    })

    ownership = json.loads((project / contract["bound_inputs"]["tile_ownership_manifest"]["path"]).read_text())
    event_contract = json.loads((project / contract["bound_inputs"]["event_phase_tile_contract"]["path"]).read_text())
    materialized = json.loads((project / contract["bound_inputs"]["tile_materialization_audit"]["path"]).read_text())
    checks.update({
        "ownership_187_tiles_19_planes": ownership.get("tile_count") == 187 and all(int(t["core_nodes"]) == int(t["core_horizontal_cells"]) * 19 for t in ownership["tiles"]),
        "ownership_grid_hash_exact": ownership.get("source_grid_sha256") == resolved["source_grid"],
        "event_contract_depths_exact": event_contract.get("required_depths_km") == EXPECTED_DEPTHS,
        "event_contract_grid_hash_exact": event_contract.get("source_grid_sha256") == resolved["source_grid"],
        "event_contract_187_tiles": event_contract.get("tile_count") == len(event_contract.get("tiles", [])) == 187,
        "materialization_pass": materialized.get("pass") is True and materialized.get("source_nodes") == 11_766_320,
        "materialization_source_hash_exact": materialized.get("source_grid_sha256") == resolved["source_grid"],
        "materialization_187_tiles": len(materialized.get("tiles", [])) == 187,
    })
    materialized_by_id = {Path(item["path"]).name.split("_grid1p875")[0]: item for item in materialized["tiles"]}
    tile_hash_mismatches = []
    for tile in event_contract["tiles"]:
        path = project / tile["grid_path"]
        observed = sha256(path) if path.is_file() else None
        item = materialized_by_id.get(tile["id"], {})
        if observed != tile["grid_sha256"] or observed != item.get("sha256"):
            tile_hash_mismatches.append(tile["id"])
    checks["all_187_tile_hashes_exact"] = not tile_hash_mismatches
    tile_glob = str(project / "data/interim/phase_aware_gs_20260811/recovery_v360/national_grid_tiles_v479/tiles/*.parquet")
    con = duckdb.connect()
    tile_depth_rows = con.execute(
        "select filename,count(distinct z_km),list(distinct z_km order by z_km) from read_parquet(?,filename=true) group by filename order by filename",
        [tile_glob],
    ).fetchall()
    con.close()
    checks["all_187_tiles_have_exact_depths"] = len(tile_depth_rows) == 187 and all(int(row[1]) == 19 and [float(v) for v in row[2]] == EXPECTED_DEPTHS for row in tile_depth_rows)

    ui_results = {}
    for item in contract["label_inventory"]:
        path = project / item["path"]
        text = path.read_text()
        present = item["required_literal"] in text
        hash_match = sha256(path) == item["sha256"]
        ui_results[item["path"]] = {"classification": item["classification"], "literal_present": present, "hash_match": hash_match}
        checks[f"label_literal_{len(ui_results)}"] = present
        checks[f"label_hash_{len(ui_results)}"] = hash_match

    if not all(checks.values()):
        raise ContractError("depth contract failed closed: " + ",".join(k for k, v in checks.items() if not v))
    return {
        "schema": "national-1p875-depth-contract-validation-v1138",
        "pass": True,
        "checks": checks,
        "observed": {
            "columns": columns,
            "rows": int(counts[0]),
            "horizontal_cells_per_plane": 619_280,
            "depth_planes_km": depths,
            "adjacent_depth_intervals_km": np.diff(depths).tolist(),
            "horizontal_spacing_m": {"x": dx, "y": dy},
            "tiles": len(tile_depth_rows),
            "tile_hash_mismatches": tile_hash_mismatches,
        },
        "label_inventory": ui_results,
        "activation_allowed": False,
        "publication_allowed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    contract = json.loads(args.contract.read_text())
    result = validate_project(args.project.resolve(), contract)
    encoded = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
