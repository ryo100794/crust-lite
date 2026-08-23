from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import tarfile
from pathlib import Path

import pytest

from crust_lite.external_geology import (
    ExternalGeologyError,
    POSTHOC_MODE,
    load_known_fault_features,
    public_provenance,
    resolve_external_geology,
)
from crust_lite.paths import ProjectPaths


def _paths(root: Path) -> ProjectPaths:
    return ProjectPaths(
        root=root,
        data_raw=root / "data/raw",
        data_interim=root / "data/interim",
        data_processed=root / "data/processed",
        outputs_maps=root / "outputs/maps",
        outputs_tables=root / "outputs/tables",
        outputs_dashboard=root / "outputs/dashboard",
        outputs_reports=root / "outputs/reports",
        outputs_3d=root / "outputs/3d",
    )


def _md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes(), usedforsecurity=False).hexdigest()


def _fixture(root: Path, *, unsafe: bool = False) -> tuple[ProjectPaths, Path]:
    member = "data/processed/fault_segment.gpkg"
    source = root / "source.bin"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(b"posthoc-only-test")
    archive = root / "fixture.tar"
    with tarfile.open(archive, "w") as tar:
        tar.add(source, arcname="../escape" if unsafe else member)
    policy = {
        "schema": "external-geology-posthoc-policy-v1",
        "runtime": {"rclone_binary": "rclone", "rclone_config": "/private/config", "fetch_timeout_seconds": 5},
        "datasets": {
            "accepted": {
                "posthoc_eligible": True,
                "provider": "Provider",
                "dataset": "Dataset",
                "version": "1",
                "url": "https://example.invalid/data",
                "license": "test",
                "license_url": "https://example.invalid/license",
                "crs": "EPSG:4326",
                "processing": "none",
                "archive_bytes": archive.stat().st_size,
                "archive_md5": _md5(archive),
                "recovery_path": "gdrive:private/archive.tar",
                "primary_member": member,
            },
            "rejected": {
                "posthoc_eligible": False,
                "provider": "unresolved",
                "dataset": "coarse",
                "version": "unversioned",
                "url": None,
                "license": "unresolved",
                "license_url": None,
                "crs": "unresolved",
                "processing": "quarantine",
                "archive_bytes": 0,
                "archive_md5": "0" * 32,
                "recovery_path": "gdrive:private/rejected.tar",
                "rejection_reason": "unresolved provenance",
            },
        },
    }
    inventory = {
        "datasets": [
            {"dataset_id": "accepted", "files": [{"project_relative_path": member}]},
            {"dataset_id": "rejected", "files": [{"project_relative_path": "data/raw/rejected"}]},
        ]
    }
    (root / "configs").mkdir(exist_ok=True)
    (root / "logs").mkdir(exist_ok=True)
    (root / "configs/external_geology_posthoc_v1.json").write_text(json.dumps(policy), encoding="utf-8")
    (root / "logs/EXTERNAL_GEOLOGY_PROVENANCE_20260823.json").write_text(json.dumps(inventory), encoding="utf-8")
    return _paths(root), archive


def test_normal_mode_has_no_known_structure_and_wrong_mode_fails(tmp_path: Path) -> None:
    paths, _archive = _fixture(tmp_path)
    features, provenance = load_known_fault_features(paths)
    assert features == []
    assert provenance["known_structure_absent"] is True
    with pytest.raises(ExternalGeologyError, match="explicit mode"):
        with resolve_external_geology(paths, "accepted", mode="normal"):
            pass


def test_verified_fetch_is_read_only_and_cache_is_cleaned(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    paths, archive = _fixture(tmp_path)
    cache_parent = tmp_path / "cache"

    def fake_run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        shutil.copy2(archive, Path(command[5]))
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    with resolve_external_geology(
        paths, "accepted", mode=POSTHOC_MODE, cache_parent=cache_parent
    ) as resolved:
        member = resolved.member_path("data/processed/fault_segment.gpkg")
        assert member.read_bytes() == b"posthoc-only-test"
        assert member.stat().st_mode & 0o222 == 0
        private_root = resolved.cache_root
    assert not private_root.exists()
    assert list(cache_parent.iterdir()) == []


def test_rejected_unresolved_dataset_never_fetches(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    paths, _archive = _fixture(tmp_path)
    called = False

    def fake_run(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        nonlocal called
        called = True
        raise AssertionError("fetch must not run")

    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(ExternalGeologyError, match="not posthoc eligible"):
        with resolve_external_geology(paths, "rejected", mode=POSTHOC_MODE):
            pass
    assert called is False


def test_offline_failure_is_fail_closed_and_cache_cleaned(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    paths, _archive = _fixture(tmp_path)
    cache_parent = tmp_path / "cache"

    def fail_run(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        raise subprocess.CalledProcessError(1, "rclone")

    monkeypatch.setattr(subprocess, "run", fail_run)
    with pytest.raises(ExternalGeologyError, match="no local fallback"):
        with resolve_external_geology(
            paths, "accepted", mode=POSTHOC_MODE, cache_parent=cache_parent
        ):
            pass
    assert list(cache_parent.iterdir()) == []


def test_unsafe_tar_member_is_rejected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    paths, archive = _fixture(tmp_path, unsafe=True)

    def fake_run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        shutil.copy2(archive, Path(command[5]))
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(ExternalGeologyError, match="unsafe archive member"):
        with resolve_external_geology(paths, "accepted", mode=POSTHOC_MODE):
            pass


def test_public_provenance_is_redacted(tmp_path: Path) -> None:
    paths, _archive = _fixture(tmp_path)
    payload = public_provenance(paths.root)
    text = json.dumps(payload)
    assert payload["normal_pipeline_known_structure_absent"] is True
    assert "gdrive:" not in text
    assert "archive_md5" not in text
    assert "/private/" not in text
    assert "recovery_path" not in text
