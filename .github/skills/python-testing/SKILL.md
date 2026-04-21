---
name: python-testing
description: Run pytest and coverage for the challenge repository using uv.
---
# Python Testing Skill

Use this skill to run Python tests with `pytest`.

## Project Paths

- Test directory: `tests/`
- Main Python files: `download_data.py`, `submission_utils.py`
- Pytest config: `pyproject.toml`

## Running Tests

```bash
rtk uv run pytest
rtk uv run pytest -v
rtk uv run pytest -x
rtk uv run pytest tests/test_submission_utils.py -v
```

## Coverage

```bash
rtk uv run coverage run -m pytest
rtk uv run coverage report
rtk uv run coverage html
```

## Pass Criteria

- The selected test command exits with code `0`.
- All collected tests pass.

## Troubleshooting

If dev dependencies are missing:

```bash
rtk uv sync --group dev
```
