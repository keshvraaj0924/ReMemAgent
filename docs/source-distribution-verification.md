# Source Distribution Verification

The packaging test suite verifies the source distribution separately from the wheel artifact.

## Contract

`tests/test_source_distribution_smoke.py` performs the following checks:

1. builds a real `sdist` with `python -m build`;
2. creates a fresh virtual environment outside the repository checkout;
3. installs the generated source archive with `--no-deps`;
4. imports both shipped runtime packages, `remem` and `experiments`;
5. discovers console-script names directly from `pyproject.toml`;
6. executes every declared console script with `--help`.

This catches source-archive omissions that a wheel-only smoke test cannot detect. In particular, the source archive must contain enough project metadata and source files for the build backend to reconstruct the installable package.

## Evidence boundary

Passing this test demonstrates that the current checkout can produce an installable source distribution with the expected public package and CLI surface. It does **not** establish benchmark correctness, external environment compatibility, statistical significance, or production readiness.

The test intentionally uses a fresh environment and runs commands from a temporary working directory so successful imports cannot silently depend on the repository checkout through the current working directory or editable installation.
