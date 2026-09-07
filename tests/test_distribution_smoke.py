"""Integration coverage for the installable wheel artifact."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def _run(command: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    """Run a subprocess and return captured text output."""
    return subprocess.run(
        command,
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )


def _venv_python(venv_path: Path) -> Path:
    """Return the platform-specific Python executable in a virtual environment."""
    executable_name = "python.exe" if sys.platform == "win32" else "python"
    executable_dir = "Scripts" if sys.platform == "win32" else "bin"
    return venv_path / executable_dir / executable_name


def test_built_wheel_imports_without_source_checkout(tmp_path: Path) -> None:
    """Ensure the distribution contains both runtime packages and console entry points."""
    repository_root = Path(__file__).resolve().parents[1]
    distribution_dir = tmp_path / "dist"
    virtual_environment = tmp_path / "venv"
    distribution_dir.mkdir()

    _run(
        [
            sys.executable,
            "-m",
            "build",
            "--wheel",
            "--outdir",
            str(distribution_dir),
        ],
        cwd=repository_root,
    )

    wheels = sorted(distribution_dir.glob("*.whl"))
    assert len(wheels) == 1

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
            str(wheels[0]),
        ],
        cwd=repository_root,
    )

    smoke = _run(
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
    assert smoke.returncode == 0

    for executable in ("remem-ablation", "remem-benchmark", "remem-paired-benchmark"):
        executable_path = isolated_python.parent / executable
        if sys.platform == "win32":
            executable_path = executable_path.with_suffix(".exe")
        _run([str(executable_path), "--help"], cwd=tmp_path)
