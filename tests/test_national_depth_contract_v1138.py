from __future__ import annotations

import copy
import importlib.util
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("depth_contract", ROOT / "scripts/validate_national_depth_contract_v1138.py")
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def valid_contract() -> dict:
    return {
        "schema": "national-1p875-physical-depth-contract-candidate-v1138",
        "status": "INACTIVE_CANDIDATE",
        "activation_allowed": False,
        "publication_allowed": False,
        "physical_analysis_grid": {
            "horizontal_spacing_m": 1875.0,
            "vertical_coordinate_column": "z_km",
            "vertical_coordinate_unit": "km",
            "positive_vertical_direction": "down",
            "depth_planes_km": MODULE.EXPECTED_DEPTHS,
            "adjacent_depth_intervals_km": MODULE.EXPECTED_INTERVALS,
            "depth_plane_count": 19,
            "uniform_depth_spacing": False,
            "minimum_adjacent_interval_km": 0.5,
            "maximum_adjacent_interval_km": 3.0,
        },
        "display_transform": {
            "part_of_physical_grid": False,
            "vertical_exaggeration_unit": "dimensionless",
            "changes_physical_depth_planes": False,
        },
        "label_policy": {
            "uniform_0p5km_claim_allowed": False,
            "quantization_step_is_physical_resolution": False,
            "vertical_exaggeration_is_physical_resolution": False,
        },
    }


class DepthContractTest(unittest.TestCase):
    def test_exact_irregular_contract_passes(self):
        MODULE.validate_structure(valid_contract())

    def test_uniform_half_km_claim_fails_closed(self):
        value = valid_contract()
        value["physical_analysis_grid"]["uniform_depth_spacing"] = True
        with self.assertRaises(MODULE.ContractError):
            MODULE.validate_structure(value)

    def test_changed_plane_or_interval_fails_closed(self):
        for key, replacement in (("depth_planes_km", MODULE.EXPECTED_DEPTHS[:-1]), ("adjacent_depth_intervals_km", [0.5] * 18)):
            value = valid_contract()
            value["physical_analysis_grid"][key] = replacement
            with self.assertRaises(MODULE.ContractError):
                MODULE.validate_structure(value)

    def test_ambiguous_resolution_field_fails_closed(self):
        value = valid_contract()
        value["physical_analysis_grid"]["depth_resolution_km"] = 0.5
        with self.assertRaises(MODULE.ContractError):
            MODULE.validate_structure(value)

    def test_display_exaggeration_cannot_mutate_physical_grid(self):
        value = valid_contract()
        value["display_transform"]["changes_physical_depth_planes"] = True
        with self.assertRaises(MODULE.ContractError):
            MODULE.validate_structure(value)

    def test_quantization_cannot_be_called_physical_resolution(self):
        value = valid_contract()
        value["label_policy"]["quantization_step_is_physical_resolution"] = True
        with self.assertRaises(MODULE.ContractError):
            MODULE.validate_structure(value)


if __name__ == "__main__":
    unittest.main()
