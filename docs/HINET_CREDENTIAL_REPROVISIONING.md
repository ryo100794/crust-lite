# Hi-net credential reprovisioning after a RunPod replacement

This procedure keeps the NIED Hi-net credential outside `/workspace`, Google
Drive, repository files, command-line arguments, logs, hashes, and backups.

## Trust boundary

`/workspace` is a shared FUSE filesystem and is not the execution trust anchor
for a root credential installer. Supply `hinet_credential_security_v1.py` from
a trusted user-side checkout to the new Pod, then place it at:

```text
/root/.local/libexec/equake/hinet_credential_security_v1.py
```

The directory must be `0700`, and the root-owned tool must be `0700`. Do not
execute a workspace copy as root unless its provenance has been independently
verified from the user side.

## Preferred RunPod-secret flow

Configure `HINET_USER` and `HINET_PASSWORD` as RunPod secret environment
variables. Do not place their values in a startup command, shell history, Pod
template text, or Drive file. On the new Pod, run the root-private tool:

```bash
umask 077
/root/.local/libexec/equake/hinet_credential_security_v1.py provision --from-env
unset HINET_USER HINET_USERNAME HINET_PASSWORD HINET_PASS
export HINET_ENV_FILE=/root/.config/equake/credentials/hinet.env
```

The installer refuses an implicit overwrite. Use `--replace` only for an
intentional credential rotation after independently confirming the target.

## Interactive standard-input flow

If RunPod secrets are unavailable, transmit the two `KEY=VALUE` lines over an
already authenticated SSH standard-input channel. Never pass values as CLI
arguments. The installer accepts only `HINET_USER` and `HINET_PASSWORD`, creates
a same-directory `0600` temporary file, calls `fsync`, atomically renames it,
and leaves no backup.

```bash
/root/.local/libexec/equake/hinet_credential_security_v1.py provision --from-stdin
```

## Required verification

```bash
export HINET_ENV_FILE=/root/.config/equake/credentials/hinet.env
/root/.local/libexec/equake/hinet_credential_security_v1.py self-test
/root/.local/libexec/equake/hinet_credential_security_v1.py verify \
  --gdrive-check --live-catalog-check --live-date 2024-01-15
```

Verification checks root ownership, exact `0700/0600` modes, symlink rejection,
UID 65534 denial, both existing collector loaders, an authenticated read-only
catalog request, known former FUSE paths, exact `hinet.env` filenames under the
workspace, exact-secret matches in bounded text/config/log roots, and Google
Drive filenames. It emits only booleans, counts, safe paths, and error classes.

If no credential is supplied, the installer fails closed and creates nothing.
If any verification check fails, do not start a Hi-net acquisition job. Never
copy the canonical file into `/workspace`, the repository, or Google Drive.

## Scope limits

The Drive audit proves absence of a live `hinet.env` pathname in the scanned
Drive namespace. It cannot attest provider-side block erasure or unknown files
outside that namespace. Bootstrap authenticity after Pod replacement remains a
user-side trust responsibility until a separately approved signed distribution
mechanism exists.
