from __future__ import annotations

from pathlib import Path

from crust_lite import resources


def test_available_memory_uses_cgroup_when_lower(monkeypatch) -> None:
    monkeypatch.setattr(resources, "_cgroup_available_memory_bytes", lambda: 16_000_000_000)
    monkeypatch.setattr(resources, "_host_available_memory_bytes", lambda: 700_000_000_000)

    assert resources.available_memory_bytes() == 16_000_000_000


def test_read_int_ignores_unlimited_cgroup_value(tmp_path: Path) -> None:
    path = tmp_path / "memory.max"
    path.write_text("max", encoding="utf-8")

    assert resources._read_int(path) is None
