---
applyTo: ".github/agents/**/*.agent.md"
description: "Standards and workflow for maintaining repository-local custom agents."
---

# Agent Maintenance Workflow and Standards

## Purpose

This document defines standards for maintaining custom agent overlays in this
repository if `.github/agents/` is introduced or expanded later.

## Canonical Paths in This Repository

Use these locations consistently:

- Repo-wide instructions and guardrails: `AGENTS.md`
- Path-scoped instructions: `.github/instructions/*.instructions.md`
- Custom agents: `.github/agents/*.agent.md`
- Reusable skills: `.github/skills/*/SKILL.md`

## Tooling Rules

- Prefer `rtk` for shell commands when available in the runtime.
- Use `uv` for Python execution and dependency management.

## Review Checklist

- Frontmatter is valid YAML.
- Agent name and description are specific and grounded in this repo.
- Paths in examples exist in this repository.
- No custom agent conflicts with `AGENTS.md`.
- Data-handling agents explicitly reference
  `.github/instructions/data_contracts.instructions.md`.
- Code-writing agents explicitly reference
  `.github/instructions/code_writing_behavior.instructions.md`.

## Scope Rules

- Keep agent overlays concise and role-specific.
- Do not encode repository behavior in agents when it belongs in
  `.github/instructions/*.instructions.md`.
- Do not claim tools, permissions, or data availability that are not grounded in
  repository files or the active runtime.
