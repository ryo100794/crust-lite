"""Fail-closed resolver for posthoc-only external geology archives.

Normal analysis never calls or receives a local external-geology path.  A
caller must explicitly request ``posthoc-known-structure``.  The selected
canonical tar is fetched into a private temporary directory, verified, safely
extracted read-only, and removed when the context exits.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import subprocess
import tarfile
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterator

from crust_lite.io.geopackage import read_features
from crust_lite.paths import ProjectPaths


POSTHOC_MODE = "posthoc-known-structure"
POLICY_RELATIVE_PATH = Path("configs/external_geology_posthoc_v1.json")
INVENTORY_RELATIVE_PATH = Path("logs/EXTERNAL_GEOLOGY_PROVENANCE_20260823.json")
PUBLIC_FIELDS = (
    "provider",
    "dataset",
    "version",
    "url",
    "license",
    "license_url",
    "crs",
    "processing",
    "posthoc_eligible",
    "rejection_reason",
)


class ExternalGeologyError(RuntimeError):
    """Fail-closed external geology policy or retrieval error."""


@dataclass(frozen=True)
class ResolvedExternalGeology:
    dataset_id: str
    cache_root: Path
    members: tuple[str, ...]
    provenance: dict[str, Any]

    def member_path(self, member: str) -> Path:
        if member not in self.members:
            raise ExternalGeologyError(f"member is not allowlisted for {self.dataset_id}: {member}")
        path = (self.cache_root / member).resolve()
        try:
            path.relative_to(self.cache_root.resolve())
        except ValueError as exc:
            raise ExternalGeologyError("resolved member escaped the private cache") from exc
        if not path.exists():
            raise ExternalGeologyError(f"verified archive member is missing: {member}")
        return path


def _md5(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExternalGeologyError(f"external geology manifest is unavailable: {path.name}") from exc
    if not isinstance(payload, dict):
        raise ExternalGeologyError(f"external geology manifest root is not an object: {path.name}")
    return payload


def _load_policy(project_root: Path) -> tuple[dict[str, Any], dict[str, list[str]]]:
    policy = _load_json(project_root / POLICY_RELATIVE_PATH)
    if policy.get("schema") != "external-geology-posthoc-policy-v1":
        raise ExternalGeologyError("unsupported external geology policy schema")
    inventory = _load_json(project_root / INVENTORY_RELATIVE_PATH)
    inventory_members: dict[str, list[str]] = {}
    for dataset in inventory.get("datasets", []):
        dataset_id = str(dataset.get("dataset_id", ""))
        members = [str(item.get("project_relative_path", "")) for item in dataset.get("files", [])]
        if dataset_id and members and all(members):
            inventory_members[dataset_id] = members
    return policy, inventory_members


def public_provenance(project_root: Path) -> dict[str, Any]:
    """Return viewer-safe provenance with no path, checksum, cache, or secret."""
    policy, _members = _load_policy(project_root.resolve())
    datasets = []
    for dataset_id, spec in policy.get("datasets", {}).items():
        row = {"dataset_id": dataset_id}
        for field in PUBLIC_FIELDS:
            if field in spec:
                row[field] = spec[field]
        row["role"] = "posthoc correlation and viewer overlay only"
        row["forbidden_for_analysis"] = True
        datasets.append(row)
    return {
        "schema": "external-geology-public-provenance-v1",
        "mode_required": POSTHOC_MODE,
        "normal_pipeline_known_structure_absent": True,
        "datasets": datasets,
    }


def _safe_members(archive: tarfile.TarFile, allowlist: set[str]) -> list[tarfile.TarInfo]:
    selected: list[tarfile.TarInfo] = []
    seen: set[str] = set()
    for item in archive.getmembers():
        name = item.name.removeprefix("./")
        pure = PurePosixPath(name)
        if pure.is_absolute() or ".." in pure.parts or item.issym() or item.islnk():
            raise ExternalGeologyError(f"unsafe archive member rejected: {name}")
        if item.isfile():
            if name not in allowlist:
                raise ExternalGeologyError(f"archive contains unmanifested file: {name}")
            item.name = name
            selected.append(item)
            seen.add(name)
        elif not item.isdir():
            raise ExternalGeologyError(f"unsupported archive member type: {name}")
    missing = sorted(allowlist.difference(seen))
    if missing:
        raise ExternalGeologyError(f"archive is missing {len(missing)} manifest members")
    return selected


def _make_read_only(root: Path) -> None:
    for path in sorted(root.rglob("*"), key=lambda value: len(value.parts), reverse=True):
        if path.is_file():
            path.chmod(stat.S_IRUSR)
        elif path.is_dir():
            path.chmod(stat.S_IRUSR | stat.S_IXUSR)
    root.chmod(stat.S_IRUSR | stat.S_IXUSR)


def _cleanup_private(root: Path) -> None:
    if not root.exists():
        return
    root.chmod(stat.S_IRWXU)
    for path in root.rglob("*"):
        try:
            path.chmod(stat.S_IRWXU if path.is_dir() else stat.S_IRUSR | stat.S_IWUSR)
        except FileNotFoundError:
            pass
    shutil.rmtree(root)


@contextmanager
def resolve_external_geology(
    paths: ProjectPaths,
    dataset_id: str,
    *,
    mode: str,
    cache_parent: Path | None = None,
) -> Iterator[ResolvedExternalGeology]:
    """Fetch one eligible canonical archive into a verified private cache."""
    if mode != POSTHOC_MODE:
        raise ExternalGeologyError(
            f"external geology requires explicit mode={POSTHOC_MODE}; normal analysis fails closed"
        )
    project_root = paths.root.resolve()
    policy, inventory_members = _load_policy(project_root)
    spec = (policy.get("datasets") or {}).get(dataset_id)
    if not isinstance(spec, dict):
        raise ExternalGeologyError(f"unknown external geology dataset: {dataset_id}")
    if spec.get("posthoc_eligible") is not True:
        reason = str(spec.get("rejection_reason", "not approved for posthoc use"))
        raise ExternalGeologyError(f"dataset is not posthoc eligible: {dataset_id}: {reason}")
    members = inventory_members.get(dataset_id, [])
    if not members:
        raise ExternalGeologyError(f"dataset has no immutable file inventory: {dataset_id}")

    parent = (cache_parent or Path(tempfile.gettempdir())).resolve()
    parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    cache_root = Path(tempfile.mkdtemp(prefix="crust-lite-posthoc-", dir=parent))
    cache_root.chmod(0o700)
    archive_path = cache_root / "canonical.tar"
    runtime = policy.get("runtime") or {}
    command = [
        str(runtime.get("rclone_binary", "rclone")),
        "--config",
        str(runtime.get("rclone_config", "/root/.config/rclone/rclone.conf")),
        "copyto",
        str(spec.get("recovery_path", "")),
        str(archive_path),
        "--checksum",
    ]
    try:
        try:
            subprocess.run(
                command,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                timeout=int(runtime.get("fetch_timeout_seconds", 300)),
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise ExternalGeologyError(
                f"canonical fetch failed closed for {dataset_id}; no local fallback is permitted"
            ) from exc
        if archive_path.stat().st_size != int(spec.get("archive_bytes", -1)):
            raise ExternalGeologyError(f"canonical archive byte-size mismatch: {dataset_id}")
        if _md5(archive_path) != str(spec.get("archive_md5", "")):
            raise ExternalGeologyError(f"canonical archive MD5 mismatch: {dataset_id}")
        with tarfile.open(archive_path, mode="r:") as archive:
            selected = _safe_members(archive, set(members))
            archive.extractall(cache_root, members=selected, filter="data")
        archive_path.unlink()
        _make_read_only(cache_root)
        yield ResolvedExternalGeology(
            dataset_id=dataset_id,
            cache_root=cache_root,
            members=tuple(members),
            provenance={
                key: spec[key]
                for key in PUBLIC_FIELDS
                if key in spec
            },
        )
    finally:
        _cleanup_private(cache_root)


def load_known_fault_features(
    paths: ProjectPaths,
    *,
    mode: str | None = None,
    cache_parent: Path | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load AIST features into memory for explicit posthoc display/correlation."""
    if mode is None:
        return [], {
            "known_structure_absent": True,
            "status": "not_requested",
            "role": "posthoc-only",
        }
    with resolve_external_geology(
        paths,
        "aist_gsj_active_fault_database",
        mode=mode,
        cache_parent=cache_parent,
    ) as resolved:
        path = resolved.member_path("data/processed/fault_segment.gpkg")
        features = read_features(path)
        provenance = {
            **resolved.provenance,
            "known_structure_absent": False,
            "status": "loaded_verified_posthoc",
            "role": "posthoc correlation and viewer overlay only",
        }
    return features, provenance
