# Distribution artifact verification

The quality suite includes `tests/test_distribution_smoke.py` to verify the package users actually install, rather than only testing an editable source checkout.

The test builds a wheel from the repository, creates an isolated virtual environment, installs the wheel with `--no-deps`, imports both shipped Python packages, and invokes the primary console entry points with `--help`.

This catches packaging failures such as missing packages, incorrect wheel contents, and broken console-script metadata that an editable install can hide.

## Scope

This is a packaging integrity gate, not an external benchmark gate. It does not execute ALFWorld, WebShop, GRPO, or verl workloads and it does not establish scientific reproducibility or performance.
