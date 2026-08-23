"""Tests use an explicitly acknowledged, isolated development database."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def explicit_development_database(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CRUST_LITE_DATABASE_MODE", "development-duckdb")
    monkeypatch.setenv("CRUST_LITE_ENABLE_DEVELOPMENT_DATABASE", "1")
    monkeypatch.delenv("CRUST_LITE_DATABASE_POINTER", raising=False)
    monkeypatch.delenv("CRUST_LITE_DATABASE_OVERRIDE", raising=False)
