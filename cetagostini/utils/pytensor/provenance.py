"""Provenance utilities for Gemma 3n oracle runs.

Records implementation identity (git commit, clean state, source hashes),
environment identity (Python, platform, hardware, package versions, module
paths), and normalized command lines.  All APIs are deliberately simple and
stateless so that runner modules and cross-report validators can reuse them
without coupling.

The module never inspects or alters installed package files.  It only reads
repository source files and queries ``importlib.metadata`` for version strings.
"""

from __future__ import annotations

import hashlib
import importlib.util
import importlib.metadata
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


GEMMA3N_ENVIRONMENT_YML = (
    "articles/alchemize_pytensor_mlx_gemma_3n/environment.yml"
)

GEMMA3N_IMPLEMENTATION_SOURCE_FILES = (
    "cetagostini/utils/pytensor/api.py",
    "cetagostini/utils/pytensor/backends.py",
    "cetagostini/utils/pytensor/evidence.py",
    "cetagostini/utils/pytensor/gemma3n_pytensor.py",
    "cetagostini/utils/pytensor/gemma3n_weights.py",
    "cetagostini/utils/pytensor/mlx_compat.py",
    "cetagostini/utils/pytensor/provenance.py",
    "cetagostini/utils/pytensor/reports.py",
    "cetagostini/utils/pytensor/run_gemma3n_pytensor.py",
    "cetagostini/utils/pytensor/run_gemma_mlx.py",
    "cetagostini/utils/pytensor/validate_gemma3n_reports.py",
)

GEMMA3N_PROVENANCE_PACKAGES = (
    "numpy",
    "pytensor",
    "numba",
    "mlx",
    "mlx-lm",
    "transformers",
    "safetensors",
)

GEMMA3N_PROVENANCE_MODULES = (
    "numpy",
    "pytensor",
    "numba",
    "mlx",
    "mlx_lm",
    "transformers",
    "safetensors",
)


# ---------------------------------------------------------------------------
# Git helpers
# ---------------------------------------------------------------------------


def _git_run(args: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    """Run a git command in *cwd* and return the completed process."""
    return subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        check=False,
    )


def git_commit(repo_root: Path) -> str:
    """Return the HEAD commit hash for the repository at *repo_root*.

    Raises
    ------
    RuntimeError
        If git is unavailable or *repo_root* is not a git repository.
    """
    result = _git_run(["rev-parse", "HEAD"], repo_root)
    if result.returncode != 0:
        raise RuntimeError(
            f"git rev-parse HEAD failed in {repo_root}: {result.stderr.strip()}"
        )
    return result.stdout.strip()


def git_is_clean(repo_root: Path) -> bool:
    """Return ``True`` if the working tree at *repo_root* is clean.

    A clean tree has no staged or unstaged changes and no untracked files.
    """
    result = _git_run(["status", "--porcelain"], repo_root)
    if result.returncode != 0:
        raise RuntimeError(
            f"git status failed in {repo_root}: {result.stderr.strip()}"
        )
    return result.stdout.strip() == ""


def find_repo_root(start: Path) -> Path:
    """Return the nearest parent containing Git worktree metadata."""
    current = Path(start).resolve()
    if current.is_file():
        current = current.parent
    while current != current.parent:
        if (current / ".git").exists():
            return current
        current = current.parent
    raise RuntimeError(f"not inside a git repository: {start}")


# ---------------------------------------------------------------------------
# File hashing
# ---------------------------------------------------------------------------


def _sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """Compute SHA-256 hex digest of a file using chunked reads."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _sha256_bytes(data: bytes) -> str:
    """Compute SHA-256 hex digest of raw bytes."""
    return hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------------------
# Source hashing
# ---------------------------------------------------------------------------


def hash_source_files(
    repo_root: Path,
    source_paths: Sequence[str | Path],
) -> list[dict[str, str]]:
    """Compute SHA-256 hashes for a list of source files relative to *repo_root*.

    Parameters
    ----------
    repo_root : Path
        Repository root directory.
    source_paths : sequence of str or Path
        Paths relative to *repo_root*.

    Returns
    -------
    list of dict
        Each dict has ``path`` (relative string) and ``sha256`` keys.

    Raises
    ------
    FileNotFoundError
        If any source file does not exist.
    """
    repo_root = Path(repo_root).resolve()
    result: list[dict[str, str]] = []
    for rel in source_paths:
        rel_path = Path(rel)
        abs_path = repo_root / rel_path
        if not abs_path.exists():
            raise FileNotFoundError(f"source file not found: {rel_path}")
        result.append({
            "path": str(rel_path),
            "sha256": _sha256_file(abs_path),
        })
    return result


def hash_implementation_manifest(
    source_hashes: list[dict[str, str]],
) -> str:
    """Compute a single deterministic hash over a list of source file hashes.

    The hash is computed over the sorted concatenation of
    ``path + ":" + sha256 + "\\n"`` for each entry, ensuring deterministic
    output regardless of input order.

    Parameters
    ----------
    source_hashes : list of dict
        Output of :func:`hash_source_files`.

    Returns
    -------
    str
        SHA-256 hex digest of the sorted manifest.
    """
    lines = sorted(
        f"{entry['path']}:{entry['sha256']}\n" for entry in source_hashes
    )
    payload = "".join(lines).encode("utf-8")
    return _sha256_bytes(payload)


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


def collect_package_versions(
    packages: Sequence[str],
) -> dict[str, str]:
    """Collect installed package versions via ``importlib.metadata``.

    Parameters
    ----------
    packages : sequence of str
        Distribution names to query (e.g. ``["numpy", "pytensor"]``).

    Returns
    -------
    dict
        Mapping of distribution name to version string, or ``"unavailable"``
        if the package is not installed.
    """
    result: dict[str, str] = {
        "python": platform.python_version(),
    }
    for dist_name in packages:
        try:
            result[dist_name] = importlib.metadata.version(dist_name)
        except importlib.metadata.PackageNotFoundError:
            result[dist_name] = "unavailable"
    return result


def collect_module_paths(
    module_names: Sequence[str],
) -> dict[str, str | None]:
    """Collect filesystem paths for imported modules.

    Parameters
    ----------
    module_names : sequence of str
        Fully qualified module names (e.g. ``["numpy", "pytensor"]``).

    Returns
    -------
    dict
        Mapping of module name to its ``__file__`` path, or ``None`` if the
        module is not importable or has no ``__file__`` attribute.
    """
    result: dict[str, str | None] = {}
    for name in module_names:
        try:
            mod = sys.modules.get(name)
            if mod is not None:
                result[name] = getattr(mod, "__file__", None)
                continue
            spec = importlib.util.find_spec(name)
            result[name] = None if spec is None else spec.origin
        except (ImportError, ModuleNotFoundError, ValueError):
            result[name] = None
    return result


def collect_environment_info() -> dict[str, Any]:
    """Collect Python, platform, and hardware information.

    Returns
    -------
    dict
        Keys: ``python_version``, ``python_executable``, ``platform_system``,
        ``platform_machine``, ``platform_release``, ``platform_version``.
    """
    return {
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "platform_system": platform.system(),
        "platform_machine": platform.machine(),
        "platform_release": platform.release(),
        "platform_version": platform.version(),
    }


# ---------------------------------------------------------------------------
# Command normalization
# ---------------------------------------------------------------------------


def normalize_command(argv: Sequence[str]) -> list[str]:
    """Normalize a command line by reducing path arguments to basenames.

    Absolute paths (e.g. ``/tmp/snapshot/model.safetensors``) are replaced
    with their basename (``model.safetensors``).  Relative paths that look
    like file references (contain ``/`` or end with a known extension) are
    also reduced.  Non-path arguments are left unchanged.

    This prevents absolute snapshot or temp directory paths from leaking into
    provenance records.

    Parameters
    ----------
    argv : sequence of str
        Command line arguments (typically ``sys.argv``).

    Returns
    -------
    list of str
        Normalized argument list.
    """
    result: list[str] = []
    for arg in argv:
        if "/" in arg or "\\" in arg:
            # os.path.basename on POSIX does not split on backslashes,
            # so handle both separators explicitly.
            normalized = arg.replace("\\", "/")
            basename = normalized.rsplit("/", 1)[-1]
            result.append(basename if basename else arg)
        else:
            result.append(arg)
    return result


# ---------------------------------------------------------------------------
# Implementation manifest
# ---------------------------------------------------------------------------


def build_implementation_manifest(
    repo_root: Path,
    source_files: Sequence[str | Path],
    *,
    require_clean: bool = True,
    packages: Sequence[str] = ("numpy", "pytensor", "numba", "mlx", "mlx-lm", "transformers"),
    modules: Sequence[str] = ("numpy", "pytensor"),
    environment_yml_path: str | Path | None = "environment.yml",
) -> dict[str, Any]:
    """Build a provenance manifest for the current implementation.

    Records:

    - Git commit hash and clean-state flag.
    - SHA-256 of ``environment.yml`` (if it exists).
    - SHA-256 of each listed source file.
    - A single deterministic hash over all source hashes.
    - UTC timestamp.
    - ``sys.executable``.
    - Python, platform, and hardware information.
    - Installed package versions.
    - Imported module filesystem paths.

    Parameters
    ----------
    repo_root : Path
        Repository root directory.
    source_files : sequence of str or Path
        Source file paths relative to *repo_root*.
    require_clean : bool
        When ``True`` (default), raise if the git working tree is dirty.
    packages : sequence of str
        Distribution names to query for versions.
    modules : sequence of str
        Module names to query for filesystem paths.
    environment_yml_path : str, Path, or None
        Path to ``environment.yml`` relative to *repo_root*.  Set to ``None``
        to skip.

    Returns
    -------
    dict
        Provenance manifest dictionary.

    Raises
    ------
    RuntimeError
        If *require_clean* is ``True`` and the working tree is dirty.
    """
    repo_root = Path(repo_root).resolve()

    commit = git_commit(repo_root)
    is_clean = git_is_clean(repo_root)

    if require_clean and not is_clean:
        raise RuntimeError(
            f"git working tree is not clean in {repo_root}; "
            "commit or stash changes before recording provenance"
        )

    environment_yml_sha256: str | None = None
    if environment_yml_path is not None:
        yml_path = repo_root / Path(environment_yml_path)
        if yml_path.exists():
            environment_yml_sha256 = _sha256_file(yml_path)

    source_hashes = hash_source_files(repo_root, source_files)
    manifest_hash = hash_implementation_manifest(source_hashes)

    timestamp = datetime.now(timezone.utc).isoformat()

    return {
        "git_commit": commit,
        "git_clean": is_clean,
        "environment_yml_sha256": environment_yml_sha256,
        "source_hashes": source_hashes,
        "implementation_manifest_sha256": manifest_hash,
        "timestamp_utc": timestamp,
        "python_executable": sys.executable,
        "environment": collect_environment_info(),
        "package_versions": collect_package_versions(packages),
        "module_paths": collect_module_paths(modules),
    }


# ---------------------------------------------------------------------------
# Provenance report binding
# ---------------------------------------------------------------------------


def build_provenance_report(
    run_id: str,
    schema_version: str,
    implementation_manifest: dict[str, Any],
    *,
    command: Sequence[str] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Bind a provenance manifest to a run identity and schema version.

    This is the top-level report that runner modules and cross-report
    validators consume.  It combines the implementation manifest with
    caller-provided run identity, normalized command line, and any
    additional metadata.

    Parameters
    ----------
    run_id : str
        Caller-provided unique run identifier.
    schema_version : str
        Schema version string (e.g. ``GEMMA3N_ORACLE_SCHEMA_VERSION``).
    implementation_manifest : dict
        Output of :func:`build_implementation_manifest`.
    command : sequence of str or None
        Command line arguments to normalize and record.  When ``None``,
        ``sys.argv`` is used.
    extra : dict or None
        Additional metadata to merge into the report.

    Returns
    -------
    dict
        Complete provenance report dictionary.
    """
    argv = list(sys.argv) if command is None else list(command)
    normalized_command = normalize_command(argv)

    report: dict[str, Any] = {
        "run_id": run_id,
        "schema_version": schema_version,
        "implementation": implementation_manifest,
        "command": normalized_command,
    }

    if extra:
        report["extra"] = dict(extra)

    return report
