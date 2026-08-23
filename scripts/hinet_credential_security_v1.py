#!/usr/bin/env python3
"""Secure Hi-net credential reprovisioning and value-free verification.

Secret values are never printed, logged, hashed, backed up, or accepted as CLI
arguments.  Production installation is fixed to a root-only local path.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import json
import os
import stat
import subprocess
import sys
import tempfile
from datetime import date, datetime, timezone
from pathlib import Path


CANONICAL = Path("/root/.config/equake/credentials/hinet.env")
PROJECT = Path("/workspace/equake/crust-lite")
KEYS = ("HINET_USER", "HINET_PASSWORD")
ALIASES = {"HINET_USERNAME": "HINET_USER", "HINET_PASS": "HINET_PASSWORD"}


class SafeFailure(RuntimeError):
    pass


def _safe_result(status: str, **extra):
    return {
        "status": status,
        "secret_values_exposed": False,
        "credential_content_hash_computed": False,
        "credential_backup_created": False,
        **extra,
    }


def _parse_payload(text: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise SafeFailure("credential input must contain only KEY=VALUE lines")
        key, value = line.split("=", 1)
        key = ALIASES.get(key.strip(), key.strip())
        if key not in KEYS:
            raise SafeFailure("credential input contains an unsupported key")
        if key in values:
            raise SafeFailure("credential input contains a duplicate key")
        if not value or value != value.strip() or any(c in value for c in "\r\n\x00"):
            raise SafeFailure("credential value is empty or uses an unsupported representation")
        if value[0] in "\"'" or value[-1] in "\"'":
            raise SafeFailure("outer quote characters are unsupported; supply the literal value")
        values[key] = value
    if any(not values.get(k) for k in KEYS):
        raise SafeFailure("both required Hi-net credential fields must be supplied")
    return values


def _payload_from_environment() -> dict[str, str]:
    def one(primary, alias):
        a, b = os.environ.get(primary), os.environ.get(alias)
        if a and b and a != b:
            raise SafeFailure("conflicting primary and alias environment variables")
        return a or b
    user = one("HINET_USER", "HINET_USERNAME")
    password = one("HINET_PASSWORD", "HINET_PASS")
    if not user or not password:
        raise SafeFailure("RunPod secret environment is incomplete; no file was created")
    return _parse_payload(f"HINET_USER={user}\nHINET_PASSWORD={password}\n")


def _assert_no_symlink_chain(path: Path) -> None:
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        if current.exists() or current.is_symlink():
            if stat.S_ISLNK(os.lstat(current).st_mode):
                raise SafeFailure("target path contains a symbolic link")


def install_values(values: dict[str, str], target: Path = CANONICAL, replace: bool = False, test_mode: bool = False):
    target = Path(target)
    if target != CANONICAL:
        if not test_mode or not str(target).startswith("/tmp/"):
            raise SafeFailure("production target is fixed; alternate targets are test-only under /tmp")
    _assert_no_symlink_chain(target.parent)
    target.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    if target.parent.is_symlink() or not target.parent.is_dir():
        raise SafeFailure("credential directory is not a non-symlink directory")
    os.chown(target.parent, 0, 0)
    os.chmod(target.parent, 0o700)
    if target.exists() or target.is_symlink():
        if target.is_symlink() or not target.is_file():
            raise SafeFailure("existing target is not a non-symlink regular file")
        if not replace:
            raise SafeFailure("credential already exists; explicit --replace is required")
    lock_path = target.parent / ".hinet.env.lock"
    lock_fd = os.open(lock_path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    temp_path = None
    try:
        os.fchmod(lock_fd, 0o600)
        os.fchown(lock_fd, 0, 0)
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        fd, name = tempfile.mkstemp(prefix=".hinet.env.tmp.", dir=target.parent)
        temp_path = Path(name)
        try:
            os.fchmod(fd, 0o600)
            os.fchown(fd, 0, 0)
            payload = f"HINET_USER={values['HINET_USER']}\nHINET_PASSWORD={values['HINET_PASSWORD']}\n".encode()
            os.write(fd, payload)
            os.fsync(fd)
        finally:
            os.close(fd)
        os.replace(temp_path, target)
        temp_path = None
        os.chown(target, 0, 0)
        os.chmod(target, 0o600)
        dfd = os.open(target.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)
    finally:
        if temp_path is not None:
            with contextlib.suppress(FileNotFoundError):
                temp_path.unlink()
        os.close(lock_fd)
    return _safe_result(
        "PASS",
        operation="atomic_install",
        target=str(target),
        target_mode="0600",
        directory_mode="0700",
        owner="root:root",
        atomic_rename=True,
        hinet_env_file=str(target),
    )


def _metadata(path: Path):
    if not path.exists() or path.is_symlink():
        raise SafeFailure("canonical credential is missing or not a regular non-symlink file")
    st = path.stat()
    parent = path.parent.stat()
    return {
        "path": str(path),
        "file_mode": f"{stat.S_IMODE(st.st_mode):04o}",
        "file_uid": st.st_uid,
        "file_gid": st.st_gid,
        "file_regular": stat.S_ISREG(st.st_mode),
        "directory_mode": f"{stat.S_IMODE(parent.st_mode):04o}",
        "directory_uid": parent.st_uid,
        "directory_gid": parent.st_gid,
    }


def _nobody_can_read(path: Path) -> bool:
    pid = os.fork()
    if pid == 0:
        try:
            os.setgroups([])
            os.setgid(65534)
            os.setuid(65534)
            fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
            os.close(fd)
            os._exit(0)
        except Exception:
            os._exit(1)
    _, status = os.waitpid(pid, 0)
    return os.WIFEXITED(status) and os.WEXITSTATUS(status) == 0


def _read_secure(path: Path) -> dict[str, str]:
    m = _metadata(path)
    if not (m["file_mode"] == "0600" and m["file_uid"] == 0 and m["file_gid"] == 0 and m["file_regular"] and m["directory_mode"] == "0700" and m["directory_uid"] == 0 and m["directory_gid"] == 0):
        raise SafeFailure("canonical credential permissions or ownership are unsafe")
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        data = b""
        while True:
            block = os.read(fd, 65536)
            if not block:
                break
            data += block
            if len(data) > 65536:
                raise SafeFailure("credential file is unexpectedly large")
    finally:
        os.close(fd)
    return _parse_payload(data.decode("utf-8"))


def _loader_regression(path: Path, expected: dict[str, str]):
    """Exercise workspace collectors with fixture values, never the real secret."""
    del path, expected
    fixture_user = "nr_sec_003_fixture_user"
    fixture_password = "nr_sec_003_fixture_password"
    with tempfile.TemporaryDirectory(prefix="nr-sec-003-loader-") as td:
        fixture = Path(td) / "hinet.env"
        fixture.write_text(f"HINET_USER={fixture_user}\nHINET_PASSWORD={fixture_password}\n")
        fixture.chmod(0o600)
        code = (
            "import sys; from pathlib import Path; "
            f"sys.path.insert(0,{str(PROJECT / 'scripts')!r}); "
            "from collect_hinet_catalog import _credentials as c; "
            "from collect_hinet_waveforms import _credentials as w; "
            f"p=Path({str(fixture)!r}); a=c(p); "
            "import os; [os.environ.pop(k,None) for k in ('HINET_USER','HINET_USERNAME','HINET_PASSWORD','HINET_PASS')]; "
            "b=w(p); "
            f"raise SystemExit(0 if a==b==({fixture_user!r},{fixture_password!r}) else 1)"
        )
        env = os.environ.copy()
        for key in ("HINET_USER", "HINET_USERNAME", "HINET_PASSWORD", "HINET_PASS"):
            env.pop(key, None)
        proc = subprocess.run(
            [str(PROJECT / ".venv/bin/python"), "-c", code], env=env,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=60, check=False,
        )
    if proc.returncode != 0:
        raise SafeFailure("collector credential-loader fixture regression failed")
    return {"catalog_loader": "PASS", "waveform_loader": "PASS", "pairs_match_in_memory": True,
            "fixture_only": True, "real_secret_passed_to_workspace_code": False, "values_exposed": False}


def _scan_exact_content(roots, secrets):
    hits = []
    scanned = 0
    skipped = 0
    skip_dirs = {".git", ".venv", ".venv-analysis-v1082", ".deps", "__pycache__", "data", "outputs", "tmp"}
    needles = [v.encode() for v in secrets.values() if len(v.encode()) >= 6]
    for root in roots:
        if not root.exists():
            continue
        for base, dirs, files in os.walk(root, followlinks=False):
            dirs[:] = [x for x in dirs if x not in skip_dirs]
            for name in files:
                p = Path(base) / name
                try:
                    st = p.lstat()
                    if not stat.S_ISREG(st.st_mode) or st.st_size > 16 * 1024 * 1024:
                        skipped += 1
                        continue
                    data = p.read_bytes()
                    scanned += 1
                    if any(n in data for n in needles):
                        hits.append(str(p))
                except (OSError, PermissionError):
                    skipped += 1
    return {"scanned_files": scanned, "skipped_files": skipped, "matching_paths": sorted(set(hits)), "match_count": len(set(hits)), "values_exposed": False}


def _local_residue(secrets):
    exact = [
        Path("/workspace/equake/secrets/hinet.env"),
        Path("/workspace/equake/crust-lite/secrets/hinet.env"),
        Path("/workspace/google-drive/hinet.env"),
        Path("/workspace/google-drive/workspace/equake/secrets/hinet.env"),
    ]
    exact_present = [str(p) for p in exact if p.exists() or p.is_symlink()]
    filename_hits = []
    for root in (Path("/workspace/equake"), Path("/workspace/google-drive")):
        if root.exists():
            for base, dirs, files in os.walk(root, followlinks=False):
                dirs[:] = [x for x in dirs if x not in {".git", ".venv", ".venv-analysis-v1082", ".deps", "__pycache__"}]
                for name in files:
                    if name.lower() == "hinet.env":
                        p = Path(base) / name
                        if p != CANONICAL:
                            filename_hits.append(str(p))
    content = _scan_exact_content([PROJECT / x for x in ("scripts", "src", "docs", "configs", "logs", "secrets")], secrets)
    return {"known_old_paths_present": exact_present, "hinet_env_filename_hits": sorted(set(filename_hits)), "exact_secret_content_scan": content}


def _gdrive_residue(timeout=180):
    cfg = Path("/workspace/google-drive/rclone.conf")
    if not cfg.exists():
        return {"status": "NOT_CONFIGURED", "matching_paths": [], "values_exposed": False}
    cmd = ["rclone", "--config", str(cfg), "lsf", "gdrive:workspace/equake", "--recursive", "--files-only", "--format", "p"]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
    except subprocess.TimeoutExpired:
        return {"status": "TIMEOUT", "matching_paths": [], "values_exposed": False}
    if p.returncode != 0:
        return {"status": "ERROR", "error_class": "rclone_nonzero", "matching_paths": [], "values_exposed": False}
    hits = [x for x in p.stdout.splitlines() if Path(x).name.lower() == "hinet.env"]
    return {"status": "PASS" if not hits else "FAIL", "matching_paths": sorted(hits), "remote_files_examined": len(p.stdout.splitlines()), "values_exposed": False}


def _live_catalog(values, day: date):
    try:
        import ssl
        from requests.adapters import HTTPAdapter
        from urllib3.poolmanager import PoolManager
        sys.path.insert(0, "/root/.local/lib/equake/vendor")
        from HinetPy import Client

        class LegacyTLSAdapter(HTTPAdapter):
            def init_poolmanager(self, connections, maxsize, block=False, **kwargs):
                context = ssl.create_default_context()
                context.set_ciphers("DEFAULT@SECLEVEL=1")
                kwargs["ssl_context"] = context
                self.poolmanager = PoolManager(num_pools=connections, maxsize=maxsize, block=block, **kwargs)

        client = Client(values["HINET_USER"], values["HINET_PASSWORD"], timeout=60, retries=1, sleep_time_in_seconds=2, max_sleep_count=5)
        client.session.mount("https://www.hinet.bosai.go.jp", LegacyTLSAdapter())
        events = client._search_event_by_day(day.year, day.month, day.day, region="00", magmin=9.8, magmax=9.9, include_unknown_mag=False)
        return {"status": "PASS", "date": day.isoformat(), "operation": "authenticated remote read-only event search", "returned_event_count": len(events), "local_catalog_written": False, "values_exposed": False}
    except Exception as exc:
        return {"status": "FAIL", "date": day.isoformat(), "error_class": type(exc).__name__, "error_message_logged": False, "values_exposed": False}


def verify(path=CANONICAL, gdrive=False, live=False, live_date=date(2024, 1, 15)):
    path = Path(os.environ.get("HINET_ENV_FILE", str(path)))
    if path != CANONICAL:
        raise SafeFailure("HINET_ENV_FILE must reference the canonical root-only path")
    values = _read_secure(path)
    metadata = _metadata(path)
    nobody = _nobody_can_read(path)
    loader = _loader_regression(path, values)
    local = _local_residue(values)
    remote = _gdrive_residue() if gdrive else {"status": "NOT_REQUESTED", "matching_paths": [], "values_exposed": False}
    catalog = _live_catalog(values, live_date) if live else {"status": "NOT_REQUESTED", "values_exposed": False}
    refs_cmd = ["grep", "-RIlE", "HINET_ENV_FILE|/root/.config/equake/credentials/hinet.env", str(PROJECT / "scripts")]
    refs = subprocess.run(refs_cmd, capture_output=True, text=True, check=False)
    reference_files = sorted(x for x in refs.stdout.splitlines() if x)
    checks = {
        "canonical_metadata": metadata["file_mode"] == "0600" and metadata["directory_mode"] == "0700" and metadata["file_uid"] == metadata["file_gid"] == metadata["directory_uid"] == metadata["directory_gid"] == 0,
        "uid65534_unreadable": not nobody,
        "loaders": loader["catalog_loader"] == loader["waveform_loader"] == "PASS",
        "old_paths_absent": not local["known_old_paths_present"],
        "workspace_hinet_env_absent": not local["hinet_env_filename_hits"],
        "workspace_exact_secret_absent": local["exact_secret_content_scan"]["match_count"] == 0,
        "gdrive_hinet_env_absent": remote["status"] in ("PASS", "NOT_REQUESTED") and not remote["matching_paths"],
        "live_catalog_read_only": catalog["status"] in ("PASS", "NOT_REQUESTED"),
        "hinet_env_file_referenced": bool(reference_files),
    }
    return _safe_result(
        "PASS" if all(checks.values()) else "FAIL",
        audit_id="NR-SEC-003",
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        canonical=metadata,
        uid65534_readable=nobody,
        loader_regression=loader,
        local_residue_audit=local,
        gdrive_residue_audit=remote,
        live_catalog_regression=catalog,
        hinet_env_file_reference_count=len(reference_files),
        hinet_env_file_reference_files=reference_files,
        checks=checks,
        protected_services_or_data_modified=False,
    )


def self_test():
    fake_a = {"HINET_USER": "fixture_user_A", "HINET_PASSWORD": "fixture_password_A"}
    fake_b = {"HINET_USER": "fixture_user_B", "HINET_PASSWORD": "fixture_password_B"}
    with tempfile.TemporaryDirectory(prefix="nr-sec-003-") as td:
        target = Path(td) / "credentials" / "hinet.env"
        first = install_values(fake_a, target, test_mode=True)
        initial_ok = _read_secure(target) == fake_a
        nobody = _nobody_can_read(target)
        second = install_values(fake_b, target, replace=True, test_mode=True)
        replacement_ok = _read_secure(target) == fake_b
        temp_left = list(target.parent.glob(".hinet.env.tmp.*"))
        try:
            _parse_payload("")
            missing_failed_closed = False
        except SafeFailure:
            missing_failed_closed = True
        try:
            install_values(fake_a, target, replace=False, test_mode=True)
            overwrite_failed_closed = False
        except SafeFailure:
            overwrite_failed_closed = True
    checks = {
        "initial_install": first["status"] == "PASS" and initial_ok,
        "atomic_replace": second["status"] == "PASS" and replacement_ok and not temp_left,
        "uid65534_unreadable": not nobody,
        "missing_input_fail_closed": missing_failed_closed,
        "implicit_overwrite_fail_closed": overwrite_failed_closed,
    }
    return _safe_result("PASS" if all(checks.values()) else "FAIL", fixture_only=True, checks=checks)


def _write_audit(result, json_path, md_path):
    if json_path:
        Path(json_path).write_text(json.dumps(result, indent=2) + "\n")
    if md_path:
        checks = "\n".join(f"- `{k}`: **{'PASS' if v else 'FAIL'}**" for k, v in result.get("checks", {}).items())
        Path(md_path).write_text(
            "# NR-SEC-003 credential reprovisioning audit\n\n"
            f"- Timestamp UTC: `{result.get('timestamp_utc')}`\n"
            f"- Result: **{result['status']}**\n"
            "- Secret values exposed/logged/hashed/backed up: **No**\n"
            "- Canonical path: `/root/.config/equake/credentials/hinet.env`\n"
            "- Protected services, DB, waveform data, and analysis jobs modified: **No**\n\n"
            "## Checks\n\n" + checks + "\n\n"
            "The JSON audit contains only filesystem metadata, boolean results, safe paths and counts. It contains no credential values or credential-derived hashes.\n"
        )


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("provision")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--from-env", action="store_true")
    src.add_argument("--from-stdin", action="store_true")
    p.add_argument("--replace", action="store_true")
    v = sub.add_parser("verify")
    v.add_argument("--gdrive-check", action="store_true")
    v.add_argument("--live-catalog-check", action="store_true")
    v.add_argument("--live-date", default="2024-01-15")
    v.add_argument("--audit-json")
    v.add_argument("--audit-md")
    sub.add_parser("self-test")
    args = ap.parse_args()
    try:
        if args.command == "provision":
            values = _payload_from_environment() if args.from_env else _parse_payload(sys.stdin.read())
            result = install_values(values, replace=args.replace)
        elif args.command == "verify":
            result = verify(gdrive=args.gdrive_check, live=args.live_catalog_check, live_date=date.fromisoformat(args.live_date))
            _write_audit(result, args.audit_json, args.audit_md)
        else:
            result = self_test()
        print(json.dumps(result, indent=2))
        raise SystemExit(0 if result["status"] == "PASS" else 1)
    except SafeFailure as exc:
        print(json.dumps(_safe_result("FAIL_CLOSED", safe_message=str(exc)), indent=2))
        raise SystemExit(2)


if __name__ == "__main__":
    main()
