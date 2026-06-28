from __future__ import annotations

import argparse
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from crust_lite.config import AppConfig, load_config
from crust_lite.data_sources.active_faults import fetch_active_faults
from crust_lite.data_sources.domestic import build_domestic_ingest_plan
from crust_lite.data_sources.events import fetch_events
from crust_lite.data_sources.gnss import fetch_gnss
from crust_lite.data_sources.jshis import fetch_jshis
from crust_lite.data_sources.mechanisms import fetch_mechanisms
from crust_lite.data_sources.waveforms import fetch_waveforms
from crust_lite.logging import configure_logging, get_logger
from crust_lite.paths import ProjectPaths
from crust_lite.processing.catalog_qc import run_catalog_qc
from crust_lite.processing.fault_inference import infer_faults
from crust_lite.processing.gnss_features import build_gnss_features
from crust_lite.processing.historical_quality import build_historical_quality
from crust_lite.processing.simulation import run_simulation
from crust_lite.processing.stress import compute_stress
from crust_lite.processing.transfer_function import (
    estimate_transfer_functions,
    write_empty_transfer_outputs,
)
from crust_lite.processing.waveform_features import build_waveform_features
from crust_lite.report import export_outputs, write_summary
from crust_lite.viz.dashboard import write_dashboard_stub
from crust_lite.viz.visualize_3d import generate_3d_visualizations

LOGGER = get_logger(__name__)


def _context(config_path: str | Path, verbose: bool = False) -> tuple[AppConfig, ProjectPaths]:
    configure_logging(verbose)
    config = load_config(config_path)
    paths = ProjectPaths.from_config(config)
    paths.ensure()
    return config, paths


def command_domestic_ingest(config_path: str, verbose: bool = False) -> dict[str, Any]:
    config, paths = _context(config_path, verbose)
    return build_domestic_ingest_plan(config, paths)


def command_fetch(config_path: str, sample: bool = False, verbose: bool = False) -> dict[str, Any]:
    config, paths = _context(config_path, verbose)
    results = {
        "events": fetch_events(config, paths, sample=sample),
        "mechanisms": fetch_mechanisms(config, paths, sample=sample),
        "gnss": fetch_gnss(config, paths, sample=sample),
        "active_faults": fetch_active_faults(config, paths, sample=sample),
        "jshis": fetch_jshis(config, paths, sample=sample),
        "waveforms": fetch_waveforms(config, paths, sample=sample),
    }
    LOGGER.info("Fetch complete: %s", results)
    return results


def command_build_features(config_path: str, verbose: bool = False) -> dict[str, Any]:
    config, paths = _context(config_path, verbose)
    qc = run_catalog_qc(config, paths)
    gnss = build_gnss_features(paths)
    wave = build_waveform_features(paths, is_sample_data=bool(qc.get("is_sample_data", False)))
    history = build_historical_quality(config, paths)
    return {"catalog_qc": qc, "gnss_features": gnss, "waveform_features": wave, "historical_quality": history}


def command_infer_faults(config_path: str, verbose: bool = False) -> dict[str, Any]:
    config, paths = _context(config_path, verbose)
    return infer_faults(config, paths)


def command_transfer_functions(config_path: str, sample: bool = False, verbose: bool = False) -> dict[str, Any]:
    config, paths = _context(config_path, verbose)
    if not sample and not config.data_sources.waveform_spectra_csv:
        LOGGER.info("Skipping transfer functions because waveform_spectra_csv is not configured")
        return write_empty_transfer_outputs(paths, "waveform_spectra_csv_not_configured", is_sample_data=False)
    return estimate_transfer_functions(config, paths, sample=sample)


def command_stress(config_path: str, verbose: bool = False) -> dict[str, Any]:
    config, paths = _context(config_path, verbose)
    return compute_stress(config, paths)


def command_simulate(config_path: str, verbose: bool = False) -> dict[str, Any]:
    config, paths = _context(config_path, verbose)
    return run_simulation(config, paths)


def command_viz_3d(
    config_path: str,
    mode: str | None = None,
    time_bin_days: int | None = None,
    max_events: int | None = None,
    verbose: bool = False,
) -> dict[str, Any]:
    config, paths = _context(config_path, verbose)
    result = generate_3d_visualizations(
        config,
        paths,
        mode=mode,
        time_bin_days=time_bin_days,
        max_events=max_events,
    )
    if config.outputs.write_report:
        write_summary(config, paths)
    return result


def command_export(config_path: str, verbose: bool = False) -> dict[str, Any]:
    config, paths = _context(config_path, verbose)
    return export_outputs(config, paths)


def command_dashboard(config_path: str, verbose: bool = False) -> dict[str, Any]:
    config, paths = _context(config_path, verbose)
    try:
        import streamlit  # type: ignore  # noqa: F401
    except Exception:
        out = write_dashboard_stub(paths)
        LOGGER.warning("Streamlit is unavailable; wrote dashboard stub: %s", out)
        return {"dashboard_stub": str(out)}
    script = Path(__file__).resolve().parent / "viz" / "dashboard.py"
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(script), "--", "--config", str(config.path or config_path)],
        check=True,
    )
    return {"streamlit": True}


def command_run_all(config_path: str, sample: bool = False, verbose: bool = False) -> dict[str, Any]:
    results: dict[str, Any] = {}
    results["fetch"] = command_fetch(config_path, sample=sample, verbose=verbose)
    results["domestic_ingest"] = command_domestic_ingest(config_path, verbose=verbose)
    results["build_features"] = command_build_features(config_path, verbose=verbose)
    results["transfer_functions"] = command_transfer_functions(config_path, sample=sample, verbose=verbose)
    results["infer_faults"] = command_infer_faults(config_path, verbose=verbose)
    results["stress"] = command_stress(config_path, verbose=verbose)
    results["simulate"] = command_simulate(config_path, verbose=verbose)
    results["export"] = command_export(config_path, verbose=verbose)
    # Requirement: run-all finishes by running viz-3d. The viz command also
    # refreshes the report so the 3D output section is present.
    results["viz_3d"] = command_viz_3d(config_path, verbose=verbose)
    LOGGER.info("run-all complete")
    return results


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="crust-lite")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(name: str) -> argparse.ArgumentParser:
        p = sub.add_parser(name)
        p.add_argument("--config", required=True)
        p.add_argument("--verbose", action="store_true")
        return p

    for name in ["fetch", "domestic-ingest", "build-features", "infer-faults", "transfer-functions", "stress", "simulate", "export", "dashboard"]:
        p = add_common(name)
        if name in {"fetch", "transfer-functions"}:
            p.add_argument("--sample", action="store_true")
    p = add_common("run-all")
    p.add_argument("--sample", action="store_true")
    p = add_common("viz-3d")
    p.add_argument("--mode", choices=["cumulative", "window"], default=None)
    p.add_argument("--time-bin-days", type=int, default=None)
    p.add_argument("--max-events", type=int, default=None)
    return parser


def _run_argparse(argv: list[str] | None = None) -> Any:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    command_map: dict[str, Callable[..., Any]] = {
        "fetch": command_fetch,
        "domestic-ingest": command_domestic_ingest,
        "build-features": command_build_features,
        "infer-faults": command_infer_faults,
        "transfer-functions": command_transfer_functions,
        "stress": command_stress,
        "simulate": command_simulate,
        "export": command_export,
        "dashboard": command_dashboard,
        "run-all": command_run_all,
        "viz-3d": command_viz_3d,
    }
    kwargs = vars(args)
    name = kwargs.pop("command")
    config_path = kwargs.pop("config")
    return command_map[name](config_path, **kwargs)


def _run_typer() -> bool:
    try:
        import typer  # type: ignore
    except Exception:
        return False

    app = typer.Typer(help="crust-lite CLI")

    @app.command("fetch")
    def fetch_cmd(config: str = typer.Option(...), sample: bool = False, verbose: bool = False) -> None:
        command_fetch(config, sample=sample, verbose=verbose)

    @app.command("domestic-ingest")
    def domestic_ingest_cmd(config: str = typer.Option(...), verbose: bool = False) -> None:
        command_domestic_ingest(config, verbose=verbose)

    @app.command("build-features")
    def build_features_cmd(config: str = typer.Option(...), verbose: bool = False) -> None:
        command_build_features(config, verbose=verbose)

    @app.command("infer-faults")
    def infer_faults_cmd(config: str = typer.Option(...), verbose: bool = False) -> None:
        command_infer_faults(config, verbose=verbose)

    @app.command("transfer-functions")
    def transfer_functions_cmd(
        config: str = typer.Option(...), sample: bool = False, verbose: bool = False
    ) -> None:
        command_transfer_functions(config, sample=sample, verbose=verbose)

    @app.command("stress")
    def stress_cmd(config: str = typer.Option(...), verbose: bool = False) -> None:
        command_stress(config, verbose=verbose)

    @app.command("simulate")
    def simulate_cmd(config: str = typer.Option(...), verbose: bool = False) -> None:
        command_simulate(config, verbose=verbose)

    @app.command("export")
    def export_cmd(config: str = typer.Option(...), verbose: bool = False) -> None:
        command_export(config, verbose=verbose)

    @app.command("dashboard")
    def dashboard_cmd(config: str = typer.Option(...), verbose: bool = False) -> None:
        command_dashboard(config, verbose=verbose)

    @app.command("viz-3d")
    def viz_3d_cmd(
        config: str = typer.Option(...),
        mode: str | None = typer.Option(None),
        time_bin_days: int | None = typer.Option(None),
        max_events: int | None = typer.Option(None),
        verbose: bool = False,
    ) -> None:
        command_viz_3d(config, mode=mode, time_bin_days=time_bin_days, max_events=max_events, verbose=verbose)

    @app.command("run-all")
    def run_all_cmd(config: str = typer.Option(...), sample: bool = False, verbose: bool = False) -> None:
        command_run_all(config, sample=sample, verbose=verbose)

    app()
    return True


def main(argv: list[str] | None = None) -> Any:
    if argv is None and _run_typer():
        return None
    return _run_argparse(argv)


if __name__ == "__main__":
    main()
