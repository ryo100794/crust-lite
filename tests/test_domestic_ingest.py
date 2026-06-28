from __future__ import annotations

from crust_lite.cli import command_domestic_ingest
from crust_lite.config import load_config
from crust_lite.io.parquet import read_table
from crust_lite.paths import ProjectPaths


def test_domestic_ingest_registry_outputs() -> None:
    result = command_domestic_ingest("configs/japan_all_domestic.yml")
    paths = ProjectPaths.from_config(load_config("configs/japan_all_domestic.yml"))
    source_path = paths.data_processed / "domestic_data_source.parquet"
    plan_path = paths.data_processed / "domestic_ingest_plan.parquet"
    report_path = paths.outputs_reports / "domestic_data_ingest_plan.md"
    assert result["source_count"] >= 9
    assert source_path.exists()
    assert plan_path.exists()
    assert report_path.exists()
    rows = read_table(plan_path)
    source_ids = {row["source_id"] for row in rows}
    assert "jma_unified_hypocenter_catalog" in source_ids
    assert "nied_knet_kiknet_strong_motion" in source_ids
    assert "gsi_geonet_daily_highrate_rinex" in source_ids
    assert "jshis_hazard_soil_fault_model" in source_ids
    assert all("prediction" in row["prediction_disclaimer"] for row in rows)
