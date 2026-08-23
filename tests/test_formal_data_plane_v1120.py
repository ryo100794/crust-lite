from __future__ import annotations

import hashlib
import inspect
import json
import shutil
from pathlib import Path

import pytest

from crust_lite.cli_data_canonical_v1120 import command_fetch
from crust_lite.data_sources.active_faults import fetch_active_faults
from crust_lite.data_sources.events import fetch_events
from crust_lite.data_sources.jshis import fetch_jshis
from crust_lite.data_sources.waveforms import fetch_waveforms
from crust_lite.io import formal_input_resolver_v1120 as resolver
from crust_lite.io.formal_input_resolver_v1120 import FormalInputError


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, body: bytes) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body)
    return {"path": str(path), "sha256": _sha(path), "rows": 1}


def _fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    project_config = Path(__file__).parents[1] / "configs/formal_hinet_only_v1120.yml"
    config = tmp_path / "configs/formal_hinet_only_v1120.yml"
    config.parent.mkdir(parents=True)
    shutil.copy2(project_config, config)

    feature = _write(tmp_path / "data/features.parquet", b"features")
    projection = _write(tmp_path / "data/projections.parquet", b"projections")
    feature_audit = _write(tmp_path / "logs/features.audit.json", b"feature-audit")
    projection_audit = _write(tmp_path / "logs/projections.audit.json", b"projection-audit")
    event = _write(tmp_path / "data/events.parquet", b"events")
    event_audit = _write(tmp_path / "logs/events.audit.json", b"event-audit")
    for spec in (feature, projection, feature_audit, projection_audit, event, event_audit):
        spec["path"] = str(Path(str(spec["path"])).relative_to(tmp_path))

    downstream = {
        "schema": "hinet-downstream-formal-input-pointer-v1",
        "status": "ACTIVE_PASS",
        "source_policy": "NIED_HINET_ONLY",
        "known_structure_input_count": 0,
        "additional_time_correction_s": 0,
        "timezone_heuristic_allowed": False,
        "eligible_trace_count": 100,
        "quarantined_horizontal_trace_count": 2,
        "formal_database": {"sha256": "d" * 64, "read_only": True},
        "features": {
            **feature,
            "audit_path": feature_audit["path"],
            "audit_sha256": feature_audit["sha256"],
        },
        "projections": {
            **projection,
            "audit_path": projection_audit["path"],
            "audit_sha256": projection_audit["sha256"],
        },
    }
    events = {
        "schema": "formal-hinet-v472-adapter-pointer-v1",
        "status": "ACTIVE_PASS",
        "source_policy": "NIED_HINET_ONLY",
        "known_structure_input_count": 0,
        "machine_learning_used": False,
        "additional_time_correction_s": 0,
        "v493_minimum_24_gate_unchanged": True,
        "events": event,
        "features": feature,
        "independent_audit": event_audit,
    }
    (tmp_path / "configs/hinet_downstream_input.json").write_text(
        json.dumps(downstream), encoding="utf-8"
    )
    (tmp_path / "configs/formal_hinet_events.json").write_text(
        json.dumps(events), encoding="utf-8"
    )
    monkeypatch.setattr(
        resolver,
        "database_status",
        lambda _paths: {
            "verified": True,
            "source_policy": "NIED_HINET_ONLY",
            "engine": "duckdb",
            "role": "formal_hinet_only",
            "read_only": True,
            "database_sha256": "d" * 64,
        },
    )
    return config


def _tree_state(root: Path) -> dict[str, tuple[int, int]]:
    return {
        str(path.relative_to(root)): (path.stat().st_size, path.stat().st_mtime_ns)
        for path in root.rglob("*")
        if path.is_file()
    }


def test_normal_fetch_verifies_only_and_writes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _fixture(tmp_path, monkeypatch)
    before = _tree_state(tmp_path)
    result = command_fetch(str(config))
    after = _tree_state(tmp_path)
    assert before == after
    assert result["status"] == "PASS"
    assert result["source_policy"] == "NIED_HINET_ONLY"
    assert result["network_requests"] == 0
    assert result["artifacts_created"] == 0
    assert result["known_structure_input_count"] == 0


def test_pointer_hash_mismatch_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _fixture(tmp_path, monkeypatch)
    (tmp_path / "data/features.parquet").write_bytes(b"tampered")
    with pytest.raises(FormalInputError, match="SHA-256 mismatch"):
        command_fetch(str(config))


def test_sample_is_never_implicit_formal_fallback(tmp_path: Path) -> None:
    with pytest.raises(FormalInputError, match="sample data is never"):
        command_fetch(str(tmp_path / "missing.yml"), sample=True)


def test_unsafe_legacy_config_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _fixture(tmp_path, monkeypatch)
    text = config.read_text(encoding="utf-8").replace(
        "use_fdsn: false", "use_fdsn: true"
    )
    config.write_text(text, encoding="utf-8")
    with pytest.raises(FormalInputError, match="not pointer-only Hi-net"):
        command_fetch(str(config))


@pytest.mark.parametrize(
    "function,message",
    [
        (fetch_events, "generic event fetch is retired"),
        (fetch_waveforms, "generic/mixed waveform fetch is retired"),
        (fetch_active_faults, "active-fault materialization is forbidden"),
        (fetch_jshis, "J-SHIS placeholder/legacy fetch is forbidden"),
    ],
)
def test_legacy_normal_sources_fail_before_any_path_access(function: object, message: str) -> None:
    with pytest.raises(RuntimeError, match=message):
        function(None, None, sample=True)  # type: ignore[operator]


def test_normal_cli_import_graph_excludes_legacy_and_external_geology() -> None:
    source = inspect.getsource(__import__(
        "crust_lite.cli_data_canonical_v1120", fromlist=["*"]
    ))
    forbidden = (
        "data_sources.events",
        "data_sources.waveforms",
        "data_sources.active_faults",
        "data_sources.jshis",
        "external_geology",
        "ComCat",
        "FDSN",
    )
    assert all(token not in source for token in forbidden)


def test_posthoc_cli_is_physically_separate() -> None:
    from crust_lite import cli_posthoc_data_v1120

    source = inspect.getsource(cli_posthoc_data_v1120)
    assert "resolve_external_geology" in source
    assert "POSTHOC_MODE" in source
    with pytest.raises(ValueError, match="explicit mode"):
        cli_posthoc_data_v1120.inspect_dataset(
            "missing.yml", "dataset", "member", mode="normal"
        )
