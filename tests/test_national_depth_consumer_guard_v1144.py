from __future__ import annotations

import ast
import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock


HERE = Path(__file__).resolve().parent
GUARD_PATH = HERE / "national_depth_consumer_guard_v1144.py"
if not GUARD_PATH.is_file():
    GUARD_PATH = HERE.parents[0] / "scripts/national_depth_consumer_guard_v1144.py"
SPEC = importlib.util.spec_from_file_location("guard", GUARD_PATH)
GUARD = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(GUARD)


CONSUMER_ROOT = HERE if (HERE / "freeze_shared_isochron_streaming_v472.py").is_file() else HERE.parents[0] / "scripts"

CONSUMER_FILES = {
    "parameter-freeze": "freeze_shared_isochron_streaming_v472.py",
    "response-chunks": "build_streamed_event_phase_chunks_v517.py",
    "event-phase-orchestrator": "run_national_event_phase_v520.py",
    "tiled-gs-fusion": "build_tiled_eventwise_gs_from_chunks_v493.py",
    "dense-gs-reference": "build_eventwise_gs_fusion_deterministic_v433.py",
}


def contract(tile_path: str, tile_sha: str, source_path: str, source_sha: str) -> dict:
    return {
        "schema": "national-1p875-physical-depth-contract-candidate-v1138",
        "status": "INACTIVE_CANDIDATE",
        "activation_allowed": False,
        "publication_allowed": False,
        "model_or_grid_mutation_allowed": False,
        "physical_analysis_grid": {
            "horizontal_spacing_m": 1875.0,
            "depth_planes_km": GUARD.EXPECTED_DEPTHS,
            "adjacent_depth_intervals_km": GUARD.EXPECTED_INTERVALS,
            "depth_plane_count": 19,
            "uniform_depth_spacing": False,
        },
        "display_transform": {
            "part_of_physical_grid": False,
            "changes_physical_depth_planes": False,
            "vertical_exaggeration_unit": "dimensionless",
        },
        "bound_inputs": {
            "event_phase_tile_contract": {"path": tile_path, "sha256": tile_sha},
            "source_grid": {"path": source_path, "sha256": source_sha},
        },
    }


class GuardTest(unittest.TestCase):
    def fixture(self):
        temporary = tempfile.TemporaryDirectory()
        project = Path(temporary.name)
        (project / "configs").mkdir()
        (project / "scripts").mkdir()
        source = project / "source.parquet"
        source.write_bytes(b"exact-irregular-source")
        consumer = project / "scripts/consumer.py"
        consumer.write_text("print('guarded')\n")
        tile = project / "tile.json"
        tile.write_text(json.dumps({
            "required_depths_km": GUARD.EXPECTED_DEPTHS,
            "required_horizontal_spacing_m": 1875.0,
            "source_grid_sha256": GUARD.sha256(source),
            "known_structure_used": False,
            "publication_allowed": False,
        }))
        depth = project / "configs/depth.json"
        depth.write_text(json.dumps(contract("tile.json", GUARD.sha256(tile), "source.parquet", GUARD.sha256(source))))
        patches = (
            mock.patch.object(GUARD, "DEPTH_CONTRACT", Path("configs/depth.json")),
            mock.patch.object(GUARD, "DEPTH_CONTRACT_SHA256", GUARD.sha256(depth)),
            mock.patch.dict(GUARD.CONSUMERS, {"fixture": ("scripts/consumer.py", GUARD.sha256(consumer))}, clear=True),
        )
        return temporary, project, source, consumer, tile, patches

    def test_all_five_real_entrypoints_import_and_call_guard(self):
        for name, filename in CONSUMER_FILES.items():
            tree = ast.parse((CONSUMER_ROOT / filename).read_text())
            imports = [node for node in ast.walk(tree) if isinstance(node, ast.Import)]
            calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
            self.assertTrue(any(any(alias.name == "national_depth_consumer_guard_v1144" for alias in node.names) for node in imports), name)
            self.assertTrue(any(isinstance(node.func, ast.Attribute) and node.func.attr == "enforce_consumer" for node in calls), name)

    def test_exact_fixture_inspects_but_execution_stays_blocked(self):
        temporary, project, _, _, tile, patches = self.fixture()
        with temporary, patches[0], patches[1], patches[2]:
            result = GUARD.inspect_consumer(project, "fixture", tile)
            self.assertFalse(result["activation_allowed"])
            with self.assertRaises(GUARD.DepthConsumerActivationBlocked):
                GUARD.enforce_consumer(project, "fixture", tile)

    def test_legacy_alternate_contract_fails_closed(self):
        temporary, project, _, _, tile, patches = self.fixture()
        alternate = project / "legacy.json"
        alternate.write_bytes(tile.read_bytes())
        with temporary, patches[0], patches[1], patches[2]:
            with self.assertRaisesRegex(GUARD.DepthConsumerGuardError, "legacy or alternate"):
                GUARD.inspect_consumer(project, "fixture", alternate)

    def test_tampered_consumer_fails_closed(self):
        temporary, project, _, consumer, tile, patches = self.fixture()
        with temporary, patches[0], patches[1], patches[2]:
            consumer.write_text("print('tampered')\n")
            with self.assertRaisesRegex(GUARD.DepthConsumerGuardError, "consumer hash mismatch"):
                GUARD.inspect_consumer(project, "fixture", tile)

    def test_uniform_or_viewer_physical_conflation_fails_closed(self):
        value = contract("tile", "a" * 64, "source", "b" * 64)
        value["physical_analysis_grid"]["uniform_depth_spacing"] = True
        with self.assertRaises(GUARD.DepthConsumerGuardError):
            GUARD.validate_structure(value)
        value = contract("tile", "a" * 64, "source", "b" * 64)
        value["display_transform"]["changes_physical_depth_planes"] = True
        with self.assertRaises(GUARD.DepthConsumerGuardError):
            GUARD.validate_structure(value)

    def test_path_escape_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaisesRegex(GUARD.DepthConsumerGuardError, "escaped"):
                GUARD.inside(Path(td).resolve(), Path(td).resolve().parent / "outside")


if __name__ == "__main__":
    unittest.main()
