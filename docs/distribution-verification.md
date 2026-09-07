# Distribution artifact verification

The quality suite includes `tests/test_distribution_smoke.py` to verify the package users actually install, rather than only testing an editable source checkout.

The test builds a wheel from the repository, creates an isolated virtual environment, installs the wheel with `--no-deps`, imports both shipped Python packages, and invokes **every console script declared in `pyproject.toml`** with `--help`.

The test reads the console-script names directly from the `[project.scripts]` table using Python's standard-library `tomllib`. This keeps the verification contract coupled to the packaging metadata instead of maintaining a second hand-written list that can silently become stale when a console script is added or renamed.

This catches packaging failures such as missing packages, incorrect wheel contents, and broken console-script metadata that an editable install can hide. The test intentionally fails if the packaging metadata has no console scripts or contains an invalid script name, making malformed packaging configuration visible during the quality run.

## Scope

This is a packaging integrity gate, not an external benchmark gate. It does not execute ALFWorld, WebShop, GRPO, or verl workloads and it does not establish scientific reproducibility or performance.
