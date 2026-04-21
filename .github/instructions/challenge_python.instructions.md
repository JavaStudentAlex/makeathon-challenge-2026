---
applyTo: "**/*.py"
description: "Python and geospatial conventions for the challenge repository."
---

# Challenge Python Instructions

## Scope

These instructions cover the Python code in this repository. Test-specific rules
belong in `.github/instructions/tests.instructions.md`.

Primary Python paths today:

- `download_data.py` for dataset download and local filesystem population
- `submission_utils.py` for raster-to-GeoJSON submission conversion
- `tests/` for local synthetic tests

## Tooling Rules

- Prefer `rtk` for shell commands when available in the runtime.
- Use `uv` exclusively for project dependency management and Python execution.

## Behavioral Overlay

For code-writing and refactor tasks, also apply
`.github/instructions/code_writing_behavior.instructions.md`.

## Core Python Rules

- Keep public functions typed.
- Prefer `X | None` over `Optional[X]`.
- Keep functions small and composable.
- Avoid mutable default arguments.
- Preserve current behavior unless the task explicitly changes it.

## Geospatial and Array Conventions

- Be explicit about CRS and raster transform handling.
- Document expected array or raster shapes when they are not obvious.
- Use `(bands, height, width)` for full raster stacks and `(height, width)` for
  single-band arrays.
- Use `np.uint8` for binary masks unless a different dtype is required.
- Use `np.float32` or `np.float64` deliberately instead of relying on implicit
  casting.
- Reproject explicitly when combining modalities on different grids or CRSs.

## Repository Architecture

- Keep reusable geospatial logic in Python modules, not trapped inside the
  notebook.
- `download_data.py` should stay focused on download orchestration and local
  file materialization.
- `submission_utils.py` should stay focused on the submission boundary:
  binarised raster in, valid GeoJSON out.
- If code grows beyond a couple of root scripts, extract modules deliberately
  instead of adding more top-level scripts with overlapping responsibilities.

## Patterns to Avoid

- Hardcoded assumptions that AlphaEarth embeddings share the same CRS or grid as
  Sentinel rasters.
- Silent loss of raster metadata during read/write cycles.
- Network access in tests.
- Logic that only exists in notebook cells when it needs to be tested or reused.
