from __future__ import annotations

from pathlib import Path
from shutil import copy2, copytree


def isolated_project(tmp_path: Path, config_name: str = "kumamoto.yml") -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    config_dir = tmp_path / "configs"
    sample_dir = tmp_path / "data" / "raw" / "sample"
    config_dir.mkdir(parents=True, exist_ok=True)
    sample_dir.parent.mkdir(parents=True, exist_ok=True)
    copy2(repo_root / "configs" / config_name, config_dir / config_name)
    if (repo_root / "data" / "raw" / "sample").exists() and not sample_dir.exists():
        copytree(repo_root / "data" / "raw" / "sample", sample_dir)
    return config_dir / config_name
