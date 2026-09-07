# CI workflow activation

The repository quality workflow is checked into `main` as well as the active feature branch.

This placement is intentional. A `pull_request` workflow is evaluated from the base branch configuration, so keeping the workflow only on a feature branch can leave an otherwise valid pull request without an Actions run until the workflow itself reaches the base branch.

## Quality contract

The workflow runs on pushes to `main` and `feat/core-memory-engine`, pull requests targeting `main`, and manual dispatch. Each Python 3.11 and 3.12 job installs the package with development dependencies and then runs:

- strict pytest configuration and the complete test suite;
- Ruff formatting and lint checks;
- mypy type checking;
- `pip check` dependency validation;
- Python compilation checks;
- wheel and source-distribution builds;
- installed-package import verification;
- distribution artifact publication;
- pytest JUnit report publication.

The workflow is an engineering gate. A successful run establishes that the checked-in software and packaging contracts pass the configured automated checks. It does not establish benchmark improvement, statistical significance, reproducibility of external environments, or production readiness.

## Verification boundary

After this workflow is present on the base branch, the feature branch must receive a subsequent commit so GitHub evaluates the pull request against the newly available workflow configuration. The current project status should report the resulting workflow state from GitHub rather than inferring success from the workflow file alone.
