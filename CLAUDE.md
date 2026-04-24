# CLAUDE.md

@AGENTS.md

The file above is the repository-local contract for this challenge project.
Load it first.

For code-writing, review, or refactor tasks, also load:

- `.github/instructions/code_writing_behavior.instructions.md`

For raster I/O, dataset interpretation, label decoding, reprojection, or
submission-format work, also load:

- `.github/instructions/data_contracts.instructions.md`

For user-facing random choices, probabilistic selection, or diversity prompts
where no appropriate external RNG or tool applies, load:

- `.github/skills/string-seed-of-thought/SKILL.md`

Do not use that skill for factual answers, raster/data interpretation,
submission generation, evaluation, benchmark reporting, or result claims.

## Tooling Rules

- Prefer `rtk` for shell commands when it is available in the runtime.
- Use `uv` as the sole project dependency manager.
- Run Python commands through `uv run`.

## Completion Rule

Before finishing, re-check that:

- the repo-specific data contracts were respected
- the reported verification actually ran
- the final report names any remaining gaps or unverified paths explicitly
