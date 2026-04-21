---
description: "Delegation, subagent spawning, and repository overlay policy."
---

# Delegation Policy

## Purpose

This file defines the repository policy for delegation and subagent reuse.

Use it when work involves:

- spawning subagents
- preparing delegation prompts
- deciding whether a task should stay local or be delegated

## Tooling Rules

- Prefer `rtk` for delegated shell commands when the runtime supports it.
- Use `uv` for dependency management and Python execution.

## Core Rules

- Keep `AGENTS.md` as the mandatory base contract for delegated work.
- Add the relevant `.github/instructions/*.instructions.md` files to the
  delegation context based on the task scope.
- Include `.github/instructions/code_writing_behavior.instructions.md` whenever
  the delegated task writes, reviews, or refactors code.
- Include `.github/instructions/data_contracts.instructions.md` whenever the
  delegated task touches raster data, labels, CRS alignment, or submissions.
- Keep delegation prompts grounded in real repository files and contracts.

## Spawn Decision Rule

Prefer delegation when it materially improves execution, such as:

- multi-file implementation across clear boundaries
- parallel read-only research
- sidecar verification that does not block the immediate local step

Keep work local when it is:

- simple
- tightly coupled
- a small direct edit
- blocked on immediate local inspection anyway

## Boundaries

- Do not invent tools, permissions, or data availability.
- Do not use delegation to bypass repository instructions.
- Do not hide uncertainty around data contracts; mark unknowns explicitly.
