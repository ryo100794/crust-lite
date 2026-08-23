#!/usr/bin/env python3
"""Active fail-closed gate shared by the five national science consumers.

The bound v1138 contract is still inactive, so direct consumer execution is
deliberately blocked after all hash and geometry checks pass.  A future,
separately audited activation must replace the contract and this hash binding;
environment variables and caller flags cannot bypass the block.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


DEPTH_CONTRACT = Path("configs/national_1p875_depth_contract_candidate_v1138.json")
DEPTH_CONTRACT_SHA256 = "d850e8976f17f063a88b48973128700f84bb6f6b56c5e045dad33c07789a8368"
EXPECTED_DEPTHS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.5, 9.0, 11.0, 13.0, 15.0, 18.0, 21.0, 24.0, 27.0, 30.0]
EXPECTED_INTERVALS = [0.5, 0.5, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 1.5, 1.5, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0, 3.0]
CONSUMERS = {
    "parameter-freeze": ("scripts/freeze_shared_isochron_streaming_v472.py", "d79f3b33945a4aabc281b50de05214afa234c81beecb8b678015f5046f112d89"),
    "response-chunks": ("scripts/build_streamed_event_phase_chunks_v517.py", "c2ef6a7f7d7120ca090556e6509ebf656985568db83537c9843b4e8e7c646a84"),
    "event-phase-orchestrator": ("scripts/run_national_event_phase_v520.py", "e9b77b7f2ed85664484f7fa0f19ebb1abec5a60d9ffd64908527556b8723afb2"),
    "tiled-gs-fusion": ("scripts/build_tiled_eventwise_gs_from_chunks_v493.py", "113c892ae0c305c36aa151d190e901c8c112df77ab364f5c8f6c773b3f6afc09"),
    "dense-gs-reference": ("scripts/build_eventwise_gs_fusion_deterministic_v433.py", "cb2338c6e6b2406b6a46fe076b862d277825e2573c4dbd2c2ee4a97b75ad9566"),
}


class DepthConsumerGuardError(RuntimeError):
    pass


class DepthConsumerActivationBlocked(DepthConsumerGuardError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise DepthConsumerGuardError(message)


def inside(project: Path, value: Path) -> Path:
    path = value.resolve()
    require(path == project or project in path.parents, "path escaped project root")
    return path


def validate_structure(contract: dict) -> None:
    require(contract.get("schema") == "national-1p875-physical-depth-contract-candidate-v1138", "depth contract schema mismatch")
    require(contract.get("status") == "INACTIVE_CANDIDATE", "unexpected depth contract status")
    require(contract.get("activation_allowed") is False, "unexpected activation flag")
    require(contract.get("publication_allowed") is False, "publication must remain false")
    require(contract.get("model_or_grid_mutation_allowed") is False, "model/grid mutation must remain false")
    grid = contract.get("physical_analysis_grid", {})
    require(grid.get("horizontal_spacing_m") == 1875.0, "horizontal spacing mismatch")
    require(grid.get("depth_planes_km") == EXPECTED_DEPTHS, "exact 19 physical depth planes required")
    require(grid.get("adjacent_depth_intervals_km") == EXPECTED_INTERVALS, "exact irregular depth intervals required")
    require(grid.get("depth_plane_count") == 19, "depth plane count mismatch")
    require(grid.get("uniform_depth_spacing") is False, "legacy uniform depth spacing prohibited")
    require("depth_resolution_km" not in grid, "ambiguous uniform depth resolution prohibited")
    display = contract.get("display_transform", {})
    require(display.get("part_of_physical_grid") is False, "viewer transform entered physical grid")
    require(display.get("changes_physical_depth_planes") is False, "viewer transform mutates physical depths")
    require(display.get("vertical_exaggeration_unit") == "dimensionless", "viewer scale must be dimensionless")


def inspect_consumer(project: Path, consumer_name: str, supplied_tile_contract: Path) -> dict:
    project = project.resolve()
    require(consumer_name in CONSUMERS, "unregistered consumer")
    contract_path = inside(project, project / DEPTH_CONTRACT)
    require(contract_path.is_file(), "authoritative depth contract missing")
    require(sha256(contract_path) == DEPTH_CONTRACT_SHA256, "authoritative depth contract hash mismatch")
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    validate_structure(contract)

    consumer_relative, consumer_expected = CONSUMERS[consumer_name]
    consumer_path = inside(project, project / consumer_relative)
    require(consumer_path.is_file(), "registered consumer missing")
    require(sha256(consumer_path) == consumer_expected, "registered consumer hash mismatch")

    tile_item = contract["bound_inputs"]["event_phase_tile_contract"]
    canonical_tile = inside(project, project / tile_item["path"])
    supplied = supplied_tile_contract if supplied_tile_contract.is_absolute() else project / supplied_tile_contract
    require(inside(project, supplied) == canonical_tile, "legacy or alternate tile contract prohibited")
    require(canonical_tile.is_file() and sha256(canonical_tile) == tile_item["sha256"], "tile contract hash mismatch")
    tile = json.loads(canonical_tile.read_text(encoding="utf-8"))
    require(tile.get("required_depths_km") == EXPECTED_DEPTHS, "tile depth vector mismatch")
    require(tile.get("required_horizontal_spacing_m") == 1875.0, "tile horizontal spacing mismatch")
    require(tile.get("source_grid_sha256") == contract["bound_inputs"]["source_grid"]["sha256"], "tile/source hash binding mismatch")
    require(tile.get("known_structure_used") is False, "known structure input prohibited")
    require(tile.get("publication_allowed") is False, "publication prohibited")

    source_item = contract["bound_inputs"]["source_grid"]
    source = inside(project, project / source_item["path"])
    require(source.is_file() and sha256(source) == source_item["sha256"], "physical source grid hash mismatch")
    return {
        "schema": "national-depth-consumer-active-guard-v1144",
        "consumer": consumer_name,
        "consumer_sha256": consumer_expected,
        "depth_contract_sha256": DEPTH_CONTRACT_SHA256,
        "tile_contract_sha256": tile_item["sha256"],
        "source_grid_sha256": source_item["sha256"],
        "depth_planes_km": EXPECTED_DEPTHS,
        "uniform_depth_spacing": False,
        "viewer_transform_is_scientific_input": False,
        "known_structure_used": False,
        "publication_allowed": False,
        "activation_allowed": False,
    }


def enforce_consumer(project: Path, consumer_name: str, supplied_tile_contract: Path) -> None:
    result = inspect_consumer(project, consumer_name, supplied_tile_contract)
    if result["activation_allowed"] is not True:
        raise DepthConsumerActivationBlocked(
            "national depth consumer execution blocked: v1138 contract is an inactive candidate"
        )
