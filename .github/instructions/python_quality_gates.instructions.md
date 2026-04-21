---
applyTo: "**/*.py"
description: "Python quality gate policy for this challenge repository."
---

# Python Quality Gates

## Purpose and Scope

This file defines the verification policy for Python changes in this
repository.

Applies to:

- `download_data.py`
- `submission_utils.py`
- `tests/**/*.py`
- agents or workflows that modify Python code

Does not apply to:

- docs-only changes
- instruction-only markdown changes
- notebook-only edits unless Python files also changed

## Tooling Rules

- Prefer `rtk` for shell commands when available in the runtime.
- Use `uv` for Python execution and dependency management.

## Quality Gate Skills

### python-linting

- Path: `.github/skills/python-linting/SKILL.md`
- Gates: isort, black, flake8, mypy via pre-commit and `uv`
- Pass criteria: all required hooks exit with code `0`

### python-testing

- Path: `.github/skills/python-testing/SKILL.md`
- Gates: pytest correctness plus coverage reporting
- Pass criteria: test commands exit with code `0`

## Default Full Gate Set

Run from the repository root.

### Linting

```bash
rtk pre-commit run isort --all-files
rtk pre-commit run black --all-files
rtk pre-commit run flake8 --all-files
rtk pre-commit run mypy --all-files
```

### Tests

```bash
rtk uv run pytest
```

### Coverage

```bash
rtk uv run coverage run -m pytest
rtk uv run coverage report
```

## Source of Truth

Use these files as the source of truth:

- `pyproject.toml` for tool configuration
- `.pre-commit-config.yaml` for hook wiring
- `.github/skills/python-linting/SKILL.md`
- `.github/skills/python-testing/SKILL.md`

## Notes

- Report failed or blocked gates explicitly.
- Do not claim verification you did not run.
