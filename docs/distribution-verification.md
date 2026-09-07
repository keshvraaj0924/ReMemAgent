# Distribution artifact verification

The quality suite includes `tests/test_distribution_smoke.py` to verify the package users actually install, rather than only testing an editable source checkout.

The test builds a wheel from the repository, creates an isolated virtual environment, installs the wheel with `--no-deps`, imports both shipped Python packages, and invokes **every console script declared in `pyproject.toml`** with `--help`.

The console-script coverage currently includes:

- `remem-ablation`
- `remem-benchmark`
- `remem-paired-benchmark`
- `remem-verify-benchmark`

This catches packaging failures such as missing packages, incorrect wheel contents, and broken console-script metadata that an editable install can hide. Keeping the script list centralized in the test also makes additions to the packaging contract explicit: a newly declared console script must be added to the smoke test and exercised successfully.

## Scope

This is a packaging integrity gate, not an external benchmark gate. It does not execute ALFWorld, WebShop, GRPO, or verl workloads and it does not establish scientific reproducibility or performance.
