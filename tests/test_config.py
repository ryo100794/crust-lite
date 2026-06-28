from __future__ import annotations

from pathlib import Path

import pytest

from crust_lite.config import load_config, parse_config


def test_yaml_config_loads() -> None:
    cfg = load_config(Path("configs/kumamoto.yml"))
    assert cfg.region.name == "kumamoto_2016"
    assert cfg.visualization_3d.enabled is True


def test_bbox_and_crs_validation() -> None:
    bad = {
        "region": {
            "name": "bad",
            "bbox": [1, 2, 0, 3],
            "crs_local": "EPSG:6670",
            "start_date": "2020-01-01",
            "end_date": "2020-01-02",
        },
        "filters": {
            "min_magnitude": 2.0,
            "max_depth_km": 40,
            "mechanism_min_magnitude": 4.0,
            "max_events_for_waveforms": 10,
        },
        "simulation": {
            "dt_days": 30,
            "years": 100,
            "n_ensemble": 10,
            "shear_modulus_pa": 3.0e10,
            "poisson_ratio": 0.25,
            "effective_friction_range": [0.2, 0.6],
            "cohesion_pa_range": [1.0e6, 1.0e7],
            "effective_normal_stress_pa_range": [1.0e7, 1.0e8],
            "random_seed": 1,
        },
        "outputs": {"write_maps": True, "write_dashboard": True, "write_report": True},
        "data_sources": {
            "use_fdsn": False,
            "use_jshis": False,
            "use_gnss": False,
            "use_active_faults": False,
            "use_waveforms": False,
        },
    }
    with pytest.raises(ValueError, match="bbox"):
        parse_config(bad)
    bad["region"]["bbox"] = [0, 0, 1, 1]
    bad["region"]["crs_local"] = "WGS84"
    with pytest.raises(ValueError, match="crs_local"):
        parse_config(bad)
