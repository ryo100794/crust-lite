from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from crust_lite.config import AppConfig


@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    data_raw: Path
    data_interim: Path
    data_processed: Path
    outputs_maps: Path
    outputs_tables: Path
    outputs_dashboard: Path
    outputs_reports: Path
    outputs_3d: Path

    @classmethod
    def from_config(cls, config: AppConfig) -> ProjectPaths:
        root = config.path.resolve().parents[1] if config.path is not None else Path.cwd()
        return cls(
            root=root,
            data_raw=root / "data" / "raw",
            data_interim=root / "data" / "interim",
            data_processed=root / "data" / "processed",
            outputs_maps=root / "outputs" / "maps",
            outputs_tables=root / "outputs" / "tables",
            outputs_dashboard=root / "outputs" / "dashboard",
            outputs_reports=root / "outputs" / "reports",
            outputs_3d=root / "outputs" / "3d",
        )

    def ensure(self) -> None:
        for path in (
            self.data_raw,
            self.data_interim,
            self.data_processed,
            self.outputs_maps,
            self.outputs_tables,
            self.outputs_dashboard,
            self.outputs_reports,
            self.outputs_3d,
        ):
            path.mkdir(parents=True, exist_ok=True)


def resolve_input(root: Path, value: str | None, fallback: Path) -> Path:
    if value is None:
        return fallback
    path = Path(value)
    if not path.is_absolute():
        path = root / path
    return path
