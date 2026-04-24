# AGENTS.md
## Purpose: Repository Contract and LLM Guidance for This Challenge Repo

This file is the repository-local instruction surface for agent-based work in
this project. It complements the workspace/runtime contract and focuses on the
parts that are specific to the osapiens Makeathon 2026 challenge repository.

## Project Overview

This repository is a geospatial challenge workspace for detecting
deforestation from multimodal satellite data. The current codebase is small and
centered on:

- `challenge.ipynb` for the challenge walkthrough, data examples, and submission
  example
- `download_data.py` for downloading the challenge dataset from S3
- `submission_utils.py` for converting binary prediction rasters into
  submittable GeoJSON
- `tests/` for synthetic, local tests that do not depend on the downloaded
  challenge dataset

## Tech Stack

| Layer | Technology |
|-------|------------|
| Language | Python 3.10+ |
| Geo / Raster | rasterio, geopandas, shapely |
| Arrays / DataFrames | NumPy, pandas |
| Visualization / Notebook | matplotlib, jupyter, ipykernel |
| Dependency Manager | **uv** |
| Linting | isort, black, flake8, mypy (via pre-commit + uv) |
| Testing | pytest, coverage |
| CI | GitHub Actions |

## CLI Guidance for Agent Sessions

- Prefer routing shell commands through `rtk` when available in the execution
  environment.
- Use `uv` for project dependency management and Python command execution.
- Do not use `pip`, `poetry`, or `conda` for project dependency management.
- Bootstrapping the `uv` executable itself is the only acceptable exception.

## Project Structure and Boundaries

- Keep reusable logic in `.py` files. Do not let `challenge.ipynb` become the
  only place where important behavior exists.
- Treat the notebook as an exploration and explanation surface, not as the sole
  source of reusable pipeline logic.
- Keep download logic in `download_data.py`.
- Keep submission conversion logic in `submission_utils.py`.
- If new reusable modules are introduced later, keep them small, typed, and
  testable instead of pushing more complexity into the notebook.

## Dataset and Submission Contracts

Load `.github/instructions/data_contracts.instructions.md` for any work that
touches raster I/O, labels, reprojection, tiling assumptions, or submission
format.

High-level contracts:

- Dataset root after download: `data/makeathon-challenge/`
- Modalities:
  - Sentinel-2 monthly GeoTIFF time series, 12 bands
  - Sentinel-1 monthly RTC GeoTIFF time series, 1 VV band
  - AlphaEarth Foundations annual GeoTIFF embeddings, 64 bands, `EPSG:4326`
- Labels are training-only and remain raw encodings unless a task explicitly
  asks for binarisation or decoding.
- Submission utility expects a single-band binary raster where `1` means
  deforestation and `0` means background.
- Submission output is GeoJSON in `EPSG:4326`.

## Universal Repository Rules

- Keep assumptions explicit, especially around CRS, raster shape, band count,
  nodata handling, and label encoding.
- Do not invent data contracts that are not present in the notebook, README, or
  code.
- For user-facing random choice, probabilistic selection, or diversity prompts
  where no appropriate external RNG or tool applies, load
  `.github/skills/string-seed-of-thought/SKILL.md`; keep it out of geospatial
  data interpretation, raster processing, submission generation, model
  evaluation, seeded experiments, benchmark analysis, and factual claims.
- When combining modalities, reproject and resample deliberately; do not assume
  shared CRS or grid alignment.
- Preserve geospatial metadata when transforming rasters: CRS, transform, band
  count, dtype, nodata.
- Use small synthetic rasters and vector data in tests. Do not require the real
  challenge dataset for automated verification.
- Keep changes reviewable and reversible.
- Never commit or push from an agent session.

## Instruction Map

- Behavioral overlay: `.github/instructions/code_writing_behavior.instructions.md`
- Python / geospatial coding conventions:
  `.github/instructions/challenge_python.instructions.md`
- Dataset and submission contracts:
  `.github/instructions/data_contracts.instructions.md`
- Quality gates:
  `.github/instructions/python_quality_gates.instructions.md`
- Tests:
  `.github/instructions/tests.instructions.md`
- Delegation policy:
  `.github/instructions/delegation_policy.instructions.md`
- Agent maintenance workflow:
  `.github/instructions/agent_maintenance_workflow.instructions.md`
- Read-only QA overlay:
  `.github/instructions/qa_readonly.instructions.md`

## Repository-Specific Skills

- Python linting skill: `.github/skills/python-linting/SKILL.md`
- Python testing skill: `.github/skills/python-testing/SKILL.md`
- String Seed of Thought skill:
  `.github/skills/string-seed-of-thought/SKILL.md`
- Native String Seed of Thought mirror:
  `.agents/skills/string-seed-of-thought/SKILL.md`, mirrored into
  `.claude/skills/string-seed-of-thought`

## Mandatory Guardrails

1. Treat `AGENTS.md` as the base repository contract for this project.
2. For code-writing, review, and refactor tasks, also load
   `.github/instructions/code_writing_behavior.instructions.md`.
3. For data-loading, raster-processing, or submission work, also load
   `.github/instructions/data_contracts.instructions.md`.
4. Use `uv` for dependency management and Python execution.
5. Double-check that the final report matches the actual edits, verification,
   and remaining risks.
6. Never use String Seed of Thought for factual geospatial claims, data
   contracts, submissions, evaluation results, or benchmark reporting.

@/home/alex/.codex/RTK.md
