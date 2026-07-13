"""Tests for the provenance utilities.

Covers:
- Command path redaction (absolute paths reduced to basenames)
- Clean/dirty git behavior via isolated temporary git repos
- Deterministic implementation manifest hashing
- No absolute path leakage
- Successful roundtrip of provenance reports
- Package version collection
- Module path collection
- Environment info collection
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from cetagostini.utils.pytensor.provenance import (
    build_implementation_manifest,
    build_provenance_report,
    collect_environment_info,
    collect_module_paths,
    collect_package_versions,
    git_commit,
    git_is_clean,
    hash_implementation_manifest,
    hash_source_files,
    normalize_command,
)


# ---------------------------------------------------------------------------
# Helpers: isolated git repos
# ---------------------------------------------------------------------------


def _init_git_repo(root: Path) -> None:
    """Initialize a minimal git repository with a single commit."""
    subprocess.run(
        ["git", "init"], cwd=str(root), capture_output=True, check=True
    )
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=str(root), capture_output=True, check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test User"],
        cwd=str(root), capture_output=True, check=True,
    )
    # Create an initial file and commit
    (root / "README.md").write_text("# Test\n", encoding="utf-8")
    subprocess.run(
        ["git", "add", "."], cwd=str(root), capture_output=True, check=True
    )
    subprocess.run(
        ["git", "commit", "-m", "initial"],
        cwd=str(root), capture_output=True, check=True,
    )


# ---------------------------------------------------------------------------
# Command normalization
# ---------------------------------------------------------------------------


class TestNormalizeCommand:
    def test_reduces_absolute_paths(self):
        argv = [
            "python",
            "-m",
            "cetagostini.utils.pytensor.run_gemma3n_pytensor",
            "run",
            "--snapshot",
            "/Users/me/.cache/huggingface/snapshots/abc123/model.safetensors",
        ]
        result = normalize_command(argv)
        assert result[-1] == "model.safetensors"
        assert "/Users/me" not in " ".join(result)

    def test_reduces_temp_paths(self):
        argv = ["run", "--output", "/tmp/pytest-of-user/test123/result.json"]
        result = normalize_command(argv)
        assert result[-1] == "result.json"
        assert "/tmp" not in " ".join(result)

    def test_preserves_non_path_args(self):
        argv = ["python", "-m", "module", "--flag", "value", "42"]
        result = normalize_command(argv)
        assert result == argv

    def test_reduces_windows_style_paths(self):
        argv = ["run", "--snapshot", "C:\\Users\\me\\snapshots\\model.bin"]
        result = normalize_command(argv)
        assert result[-1] == "model.bin"

    def test_reduces_relative_paths_with_slash(self):
        argv = ["run", "--config", "configs/experiment.yaml"]
        result = normalize_command(argv)
        assert result[-1] == "experiment.yaml"

    def test_empty_argv(self):
        assert normalize_command([]) == []

    def test_no_absolute_path_leakage(self):
        """Ensure no absolute path fragments survive normalization."""
        argv = [
            "python",
            "/home/user/project/run.py",
            "--data",
            "/var/data/training.csv",
            "--output",
            "/tmp/results/out.json",
        ]
        result = normalize_command(argv)
        combined = " ".join(result)
        assert "/home" not in combined
        assert "/var" not in combined
        assert "/tmp" not in combined
        assert "run.py" in result
        assert "training.csv" in result
        assert "out.json" in result

    def test_trailing_separator_is_redacted(self):
        result = normalize_command(["run", "--snapshot", "/tmp/snapshot/"])

        assert result == ["run", "--snapshot", "snapshot"]


# ---------------------------------------------------------------------------
# Git helpers
# ---------------------------------------------------------------------------


class TestGitHelpers:
    def test_clean_repo(self, tmp_path):
        _init_git_repo(tmp_path)
        assert git_is_clean(tmp_path) is True

    def test_dirty_repo_untracked(self, tmp_path):
        _init_git_repo(tmp_path)
        (tmp_path / "untracked.txt").write_text("dirty", encoding="utf-8")
        assert git_is_clean(tmp_path) is False

    def test_dirty_repo_modified(self, tmp_path):
        _init_git_repo(tmp_path)
        (tmp_path / "README.md").write_text("modified\n", encoding="utf-8")
        assert git_is_clean(tmp_path) is False

    def test_dirty_repo_staged(self, tmp_path):
        _init_git_repo(tmp_path)
        (tmp_path / "new_file.txt").write_text("staged\n", encoding="utf-8")
        subprocess.run(
            ["git", "add", "new_file.txt"],
            cwd=str(tmp_path), capture_output=True, check=True,
        )
        assert git_is_clean(tmp_path) is False

    def test_git_commit_returns_hash(self, tmp_path):
        _init_git_repo(tmp_path)
        commit = git_commit(tmp_path)
        assert len(commit) == 40
        assert all(c in "0123456789abcdef" for c in commit)

    def test_git_commit_changes_after_new_commit(self, tmp_path):
        _init_git_repo(tmp_path)
        commit1 = git_commit(tmp_path)

        (tmp_path / "new.txt").write_text("new\n", encoding="utf-8")
        subprocess.run(
            ["git", "add", "."], cwd=str(tmp_path), capture_output=True, check=True,
        )
        subprocess.run(
            ["git", "commit", "-m", "second"],
            cwd=str(tmp_path), capture_output=True, check=True,
        )
        commit2 = git_commit(tmp_path)
        assert commit1 != commit2

    def test_not_a_repo(self, tmp_path):
        with pytest.raises(RuntimeError, match="git"):
            git_commit(tmp_path)

    def test_is_clean_not_a_repo(self, tmp_path):
        with pytest.raises(RuntimeError, match="git"):
            git_is_clean(tmp_path)


# ---------------------------------------------------------------------------
# Source hashing
# ---------------------------------------------------------------------------


class TestHashSourceFiles:
    def test_hashes_files(self, tmp_path):
        (tmp_path / "a.py").write_text("print('a')\n", encoding="utf-8")
        (tmp_path / "b.py").write_text("print('b')\n", encoding="utf-8")

        result = hash_source_files(tmp_path, ["a.py", "b.py"])
        assert len(result) == 2
        assert result[0]["path"] == "a.py"
        assert len(result[0]["sha256"]) == 64
        assert result[1]["path"] == "b.py"
        assert len(result[1]["sha256"]) == 64

    def test_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            hash_source_files(tmp_path, ["nonexistent.py"])

    def test_deterministic(self, tmp_path):
        (tmp_path / "a.py").write_text("content\n", encoding="utf-8")
        r1 = hash_source_files(tmp_path, ["a.py"])
        r2 = hash_source_files(tmp_path, ["a.py"])
        assert r1 == r2


# ---------------------------------------------------------------------------
# Implementation manifest hashing
# ---------------------------------------------------------------------------


class TestHashImplementationManifest:
    def test_deterministic_regardless_of_order(self):
        hashes_a = [
            {"path": "b.py", "sha256": "b" * 64},
            {"path": "a.py", "sha256": "a" * 64},
        ]
        hashes_b = [
            {"path": "a.py", "sha256": "a" * 64},
            {"path": "b.py", "sha256": "b" * 64},
        ]
        assert hash_implementation_manifest(hashes_a) == hash_implementation_manifest(hashes_b)

    def test_changes_when_content_changes(self):
        hashes_a = [{"path": "a.py", "sha256": "a" * 64}]
        hashes_b = [{"path": "a.py", "sha256": "b" * 64}]
        assert hash_implementation_manifest(hashes_a) != hash_implementation_manifest(hashes_b)

    def test_changes_when_path_changes(self):
        hashes_a = [{"path": "a.py", "sha256": "x" * 64}]
        hashes_b = [{"path": "b.py", "sha256": "x" * 64}]
        assert hash_implementation_manifest(hashes_a) != hash_implementation_manifest(hashes_b)

    def test_empty_list(self):
        result = hash_implementation_manifest([])
        assert len(result) == 64

    def test_returns_hex_string(self):
        hashes = [{"path": "a.py", "sha256": "a" * 64}]
        result = hash_implementation_manifest(hashes)
        assert all(c in "0123456789abcdef" for c in result)


# ---------------------------------------------------------------------------
# Package versions
# ---------------------------------------------------------------------------


class TestCollectPackageVersions:
    def test_includes_python(self):
        versions = collect_package_versions([])
        assert "python" in versions
        assert versions["python"] != "unavailable"

    def test_numpy_available(self):
        versions = collect_package_versions(["numpy"])
        assert "numpy" in versions
        assert versions["numpy"] != "unavailable"

    def test_unavailable_package(self):
        versions = collect_package_versions(["definitely-not-a-real-package-xyz"])
        assert versions["definitely-not-a-real-package-xyz"] == "unavailable"


# ---------------------------------------------------------------------------
# Module paths
# ---------------------------------------------------------------------------


class TestCollectModulePaths:
    def test_numpy_path(self):
        paths = collect_module_paths(["numpy"])
        assert "numpy" in paths
        assert paths["numpy"] is not None

    def test_unavailable_module(self):
        paths = collect_module_paths(["definitely_not_a_real_module_xyz"])
        assert paths["definitely_not_a_real_module_xyz"] is None


# ---------------------------------------------------------------------------
# Environment info
# ---------------------------------------------------------------------------


class TestCollectEnvironmentInfo:
    def test_has_required_keys(self):
        info = collect_environment_info()
        required = {
            "python_version",
            "python_executable",
            "platform_system",
            "platform_machine",
            "platform_release",
            "platform_version",
        }
        assert required.issubset(info.keys())

    def test_executable_matches_sys(self):
        import sys
        info = collect_environment_info()
        assert info["python_executable"] == sys.executable


# ---------------------------------------------------------------------------
# Implementation manifest
# ---------------------------------------------------------------------------


class TestBuildImplementationManifest:
    def test_clean_repo_roundtrip(self, tmp_path):
        _init_git_repo(tmp_path)
        (tmp_path / "source.py").write_text("x = 1\n", encoding="utf-8")
        subprocess.run(
            ["git", "add", "."], cwd=str(tmp_path), capture_output=True, check=True,
        )
        subprocess.run(
            ["git", "commit", "-m", "add source"],
            cwd=str(tmp_path), capture_output=True, check=True,
        )

        manifest = build_implementation_manifest(
            repo_root=tmp_path,
            source_files=["source.py"],
            packages=["numpy"],
            modules=["numpy"],
            environment_yml_path=None,
        )

        assert "git_commit" in manifest
        assert len(manifest["git_commit"]) == 40
        assert manifest["git_clean"] is True
        assert manifest["environment_yml_sha256"] is None
        assert len(manifest["source_hashes"]) == 1
        assert manifest["source_hashes"][0]["path"] == "source.py"
        assert len(manifest["implementation_manifest_sha256"]) == 64
        assert "timestamp_utc" in manifest
        assert "python_executable" in manifest
        assert "environment" in manifest
        assert "package_versions" in manifest
        assert "module_paths" in manifest

    def test_dirty_repo_raises_when_required(self, tmp_path):
        _init_git_repo(tmp_path)
        (tmp_path / "source.py").write_text("x = 1\n", encoding="utf-8")
        subprocess.run(
            ["git", "add", "."], cwd=str(tmp_path), capture_output=True, check=True,
        )
        subprocess.run(
            ["git", "commit", "-m", "add source"],
            cwd=str(tmp_path), capture_output=True, check=True,
        )

        # Make dirty
        (tmp_path / "dirty.txt").write_text("dirty\n", encoding="utf-8")

        with pytest.raises(RuntimeError, match="not clean"):
            build_implementation_manifest(
                repo_root=tmp_path,
                source_files=["source.py"],
                require_clean=True,
                environment_yml_path=None,
            )

    def test_dirty_repo_allowed_when_not_required(self, tmp_path):
        _init_git_repo(tmp_path)
        (tmp_path / "source.py").write_text("x = 1\n", encoding="utf-8")
        subprocess.run(
            ["git", "add", "."], cwd=str(tmp_path), capture_output=True, check=True,
        )
        subprocess.run(
            ["git", "commit", "-m", "add source"],
            cwd=str(tmp_path), capture_output=True, check=True,
        )

        (tmp_path / "dirty.txt").write_text("dirty\n", encoding="utf-8")

        manifest = build_implementation_manifest(
            repo_root=tmp_path,
            source_files=["source.py"],
            require_clean=False,
            environment_yml_path=None,
        )
        assert manifest["git_clean"] is False

    def test_environment_yml_hashed(self, tmp_path):
        _init_git_repo(tmp_path)
        (tmp_path / "source.py").write_text("x = 1\n", encoding="utf-8")
        (tmp_path / "environment.yml").write_text(
            "name: test\ndependencies:\n  - numpy\n", encoding="utf-8",
        )
        subprocess.run(
            ["git", "add", "."], cwd=str(tmp_path), capture_output=True, check=True,
        )
        subprocess.run(
            ["git", "commit", "-m", "add files"],
            cwd=str(tmp_path), capture_output=True, check=True,
        )

        manifest = build_implementation_manifest(
            repo_root=tmp_path,
            source_files=["source.py"],
            environment_yml_path="environment.yml",
        )
        assert manifest["environment_yml_sha256"] is not None
        assert len(manifest["environment_yml_sha256"]) == 64

    def test_no_absolute_paths_in_manifest(self, tmp_path):
        _init_git_repo(tmp_path)
        (tmp_path / "source.py").write_text("x = 1\n", encoding="utf-8")
        subprocess.run(
            ["git", "add", "."], cwd=str(tmp_path), capture_output=True, check=True,
        )
        subprocess.run(
            ["git", "commit", "-m", "add source"],
            cwd=str(tmp_path), capture_output=True, check=True,
        )

        manifest = build_implementation_manifest(
            repo_root=tmp_path,
            source_files=["source.py"],
            environment_yml_path=None,
        )

        manifest_str = json.dumps(manifest)
        # The repo root path should not appear in source_hashes paths
        for entry in manifest["source_hashes"]:
            assert not os.path.isabs(entry["path"])

    def test_source_file_not_found(self, tmp_path):
        _init_git_repo(tmp_path)
        with pytest.raises(FileNotFoundError):
            build_implementation_manifest(
                repo_root=tmp_path,
                source_files=["nonexistent.py"],
                environment_yml_path=None,
            )


# ---------------------------------------------------------------------------
# Provenance report binding
# ---------------------------------------------------------------------------


class TestBuildProvenanceReport:
    def test_roundtrip(self, tmp_path):
        _init_git_repo(tmp_path)
        (tmp_path / "source.py").write_text("x = 1\n", encoding="utf-8")
        subprocess.run(
            ["git", "add", "."], cwd=str(tmp_path), capture_output=True, check=True,
        )
        subprocess.run(
            ["git", "commit", "-m", "add source"],
            cwd=str(tmp_path), capture_output=True, check=True,
        )

        impl_manifest = build_implementation_manifest(
            repo_root=tmp_path,
            source_files=["source.py"],
            environment_yml_path=None,
        )

        report = build_provenance_report(
            run_id="test-run-001",
            schema_version="gemma3n-oracle-v1",
            implementation_manifest=impl_manifest,
            command=["python", "-m", "module", "--snapshot", "/tmp/snap/model.bin"],
        )

        assert report["run_id"] == "test-run-001"
        assert report["schema_version"] == "gemma3n-oracle-v1"
        assert report["implementation"] is impl_manifest
        assert report["command"] == [
            "python", "-m", "module", "--snapshot", "model.bin",
        ]

    def test_command_normalization(self):
        report = build_provenance_report(
            run_id="r1",
            schema_version="v1",
            implementation_manifest={"git_commit": "abc"},
            command=["run", "--data", "/home/user/data/train.csv"],
        )
        assert report["command"] == ["run", "--data", "train.csv"]

    def test_extra_metadata(self):
        report = build_provenance_report(
            run_id="r1",
            schema_version="v1",
            implementation_manifest={"git_commit": "abc"},
            command=["run"],
            extra={"backend": "numba", "mode": "next_token"},
        )
        assert report["extra"]["backend"] == "numba"
        assert report["extra"]["mode"] == "next_token"

    def test_no_extra_when_none(self):
        report = build_provenance_report(
            run_id="r1",
            schema_version="v1",
            implementation_manifest={"git_commit": "abc"},
            command=["run"],
        )
        assert "extra" not in report

    def test_json_serializable(self, tmp_path):
        """The full report must be JSON-serializable."""
        _init_git_repo(tmp_path)
        (tmp_path / "source.py").write_text("x = 1\n", encoding="utf-8")
        subprocess.run(
            ["git", "add", "."], cwd=str(tmp_path), capture_output=True, check=True,
        )
        subprocess.run(
            ["git", "commit", "-m", "add source"],
            cwd=str(tmp_path), capture_output=True, check=True,
        )

        impl_manifest = build_implementation_manifest(
            repo_root=tmp_path,
            source_files=["source.py"],
            environment_yml_path=None,
        )

        report = build_provenance_report(
            run_id="test-run-002",
            schema_version="gemma3n-oracle-v1",
            implementation_manifest=impl_manifest,
            command=["python", "run.py"],
        )

        serialized = json.dumps(report)
        assert len(serialized) > 0
        deserialized = json.loads(serialized)
        assert deserialized["run_id"] == "test-run-002"
