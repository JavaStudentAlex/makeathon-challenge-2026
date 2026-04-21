---
name: python-linting
description: Run Python linting and formatting through pre-commit for this challenge repository.
---
# Python Linting Skill

## Scope

- Focus on `download_data.py`, `submission_utils.py`, and `tests/`.
- Use `.pre-commit-config.yaml` and `pyproject.toml` as the source of truth.
- Keep changes minimal and non-behavioral unless a lint/type issue requires a
  tiny code fix.

All commands should go through `rtk` when available in the runtime.

## Hook Stack in This Repo

1. `isort`
2. `black`
3. `flake8`
4. `mypy`

Generic hooks from `pre-commit-hooks`:

- `check-json`
- `pretty-format-json`
- `check-yaml`
- `trailing-whitespace`
- `end-of-file-fixer`
- `debug-statements`

## Command Patterns

Incremental:

```bash
rtk pre-commit run --files path/to/file.py
```

Comprehensive:

```bash
rtk pre-commit run --all-files
```

Individual hooks:

```bash
rtk pre-commit run isort --all-files
rtk pre-commit run black --all-files
rtk pre-commit run flake8 --all-files
rtk pre-commit run mypy --all-files
```

## Guardrails

- Do not change runtime behavior solely to satisfy linting unless explicitly
  required.
- Re-run hooks after fixes to confirm a clean pass.
