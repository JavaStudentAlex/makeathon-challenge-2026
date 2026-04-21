---
description: "Behavioral overlay for code writing, review, and refactor tasks."
---

# Code Writing Behavior

## Purpose

This file defines the expected engineering behavior for day-to-day work in this
repository.

Use it when a task involves:

- writing or editing code
- reviewing code
- refactoring or cleanup work

## Core Behavior

- Make assumptions explicit when data shape, CRS, or label semantics affect the
  result.
- Prefer the smallest correct change. Do not add speculative abstractions.
- Keep edits surgical and grounded in the current repo layout.
- Match local style and existing patterns unless a scoped instruction says
  otherwise.
- If you notice unrelated issues, call them out separately instead of widening
  the change set.

## Execution Pattern

Before implementing:

- state the working assumptions when they materially affect the solution
- identify the simplest viable approach
- define what will verify success

During implementation:

- touch only the files and code paths needed for the task
- remove imports or helpers made unused by your own changes
- keep notebook edits separate from reusable Python logic whenever possible

Before completion:

- verify the claimed outcome with the lightest sufficient evidence
- report open questions, constraints, or gaps explicitly

## Data-Sensitive Work

- Load `.github/instructions/data_contracts.instructions.md` whenever the task
  touches raster data, labels, CRS alignment, or submission format.
- Do not silently binarise or reinterpret raw labels unless the task asks for
  that transformation.
