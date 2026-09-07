"""Integration coverage for the installable source distribution artifact."""

from __future__ import annotations

import subprocess
import sys
import tomllib
from pathlib import Path


def _run(command: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    """Run a subprocess and return captured text output."""
    return subprocess.run(command, cwd=cwd, check=True, capture_output=True, text=True)


def _venv_python(venv_path: Path) -> Path:
    """Return the platform-specific Python executable in a virtual environment."""
    executable_name = "python.exe" if sys.platform == "win32" else "python"
    executable_dir = "Scripts" if sys.platform == "win32" else "bin"
    return venv_path / executable_dir / executable_name


def _declared_console_scripts(pyproject_path: Path) -> tuple[str, ...]:
    """Read shipped console-script names from the packaging contract."""
    with pyproject_path.open("rb") as pyproject_file:
        metadata = tomllib.load(pyproject_file)
    scripts = metadata["project"]["scripts"]
    if not isinstance(scripts, dict) or not scripts:
        raise AssertionError("pyproject.toml must declare console scripts")
    return tuple(sorted(scripts))


def test_source_distribution_builds_and_installs_without_checkout(tmp_path: Path) -> None:
    """Ensure an sdist can rebuild and expose the same public package surface."""
    repository_root = Path(__file__).resolve().parents[1]
    distribution_dir = tmp_path / "dist"
    virtual_environment = tmp_path / "venv"
    distribution_dir.mkdir()
    console_scripts = _declared_console_scripts(repository_root / "pyproject.toml")

    _run(
        [
            sys.executable,
            "-m",
            "build",
            "--sdist",
            "--outdir",
            str(distribution_dir),
        ],
        cwd=repository_root,
    )
    source_distributions = sorted(distribution_dir.glob("*.tar.gz"))
    assert len(source_distributions) == 1

    _run([sys.executable, "-m", "venv", str(virtual_environment)], cwd=repository_root)
    isolated_python = _venv_python(virtual_environment)
    _run(
        [
            str(isolated_python),
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--disable-pip-version-check",
            str(source_distributions[0]),
        ],
        cwd=tmp_path,
    )

    _run(
        [
            str(isolated_python),
            "-c",
            (
                "import remem, experiments; "
                "assert remem.__name__ == 'remem'; "
                "assert experiments.__name__ == 'experiments'"
            ),
        ],
        cwd=tmp_path,
    )

    for executable in console_scripts:
        executable_path = isolated_python.parent / executable
        if sys.platform == "win32":
            executable_path = executable_path.with_suffix(".exe")
        _run([str(executable_path), "--help"], cwd=tmp_path)
