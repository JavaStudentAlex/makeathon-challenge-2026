---
description: "Manual overlay for read-only QA, review, and exploratory sessions."
---

# QA Read-Only Instructions

## Purpose

Use this file as a lightweight overlay for read-only sessions, reviews,
architecture discussion, and repository analysis.

## Default Mode

- Default to reading, analysis, and explanation unless edits are explicitly
  requested.
- Prefer targeted file reads over broad context loading.
- For review and architecture discussions, apply
  `.github/instructions/code_writing_behavior.instructions.md` as a read-only
  lens.

## Evidence Expectations

- Ground conclusions in specific files, code paths, notebook cells, or config.
- Keep unknowns explicit.
- Lead with findings, risks, or missing coverage before summaries.
- Use `.github/instructions/data_contracts.instructions.md` when the analysis
  touches data shape, raster encoding, or submission format.

## Tooling Expectations

- Do not run heavy verification unless code changed or the claim requires it.
- Prefer small inspection commands and direct file reads for QA sessions.
