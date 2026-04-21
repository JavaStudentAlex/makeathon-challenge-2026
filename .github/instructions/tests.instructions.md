---
applyTo: "tests/**/*.py"
description: "Test conventions, organization, and patterns for this challenge repository."
---

# Test Instructions

## Scope

These instructions cover the test suite under `tests/`.

## Tooling Rules

- Prefer `rtk` for shell commands when available in the runtime.
- Use `uv` for Python execution.

## Behavioral Overlay

For test changes, also apply
`.github/instructions/code_writing_behavior.instructions.md`.

For data-sensitive tests, also apply
`.github/instructions/data_contracts.instructions.md`.

## Test Organization

- Use `tests/` for synthetic, local verification.
- Keep reusable fixtures close to `tests/conftest.py` if they become shared.
- Add integration-style tests only when they can remain local and deterministic.

## Test Conventions

- Use `pytest` as the test framework.
- Use `tmp_path` for file I/O tests.
- Build tiny synthetic GeoTIFFs and GeoJSON objects in tests instead of relying
  on the downloaded challenge dataset.
- Prefer explicit assertions on CRS-sensitive outputs and submission properties.
- Use `pytest.raises` for expected failures.

## Repository-Specific Priorities

- Cover the submission boundary in `submission_utils.py`.
- Validate data contracts with synthetic rasters rather than external data.
- Avoid tests that hit S3 or require the real challenge download.
- If future decoding helpers are added for RADD / GLAD-L / GLAD-S2 labels, add
  focused unit tests for the raw encoding rules.

## Running Tests

```bash
rtk uv run pytest
rtk uv run pytest -v
rtk uv run coverage run -m pytest
rtk uv run coverage report
```
