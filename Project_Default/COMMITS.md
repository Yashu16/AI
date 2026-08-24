# Commit message convention

This project uses [Conventional Commits](https://www.conventionalcommits.org/) going forward (adopted 2026-08-24). Earlier commits predate this and don't follow it consistently - that's fine, no need to rewrite history.

## Format

```
<type>: <short summary, imperative mood, no period>

<optional body - only if the "why" isn't obvious from the summary or from
notes/<notebook>.md / planning.md>
```

## Types

| Type | Use for |
|---|---|
| `feat` | New functionality - a new feature, a new model, a new notebook section |
| `fix` | Bug fixes |
| `docs` | Documentation-only changes (planning.md, notes/, README, data_dictionary.md) |
| `chore` | Maintenance, restructuring, tooling - no behavior change |
| `refactor` | Code changes that don't change behavior (renaming, reorganizing cells) |

## Notes

- Summary line: imperative mood ("Add feature" not "Added"/"Adds"), short enough to read in `git log --oneline`.
- Detailed reasoning, experiment results, and bugs found belong in `notes/<notebook>.md`, not in the commit body - keep commits short and point there if needed.
- One commit per coherent unit of work, not one giant commit per session.

## Examples from this project

```
feat: add 5 deferred features tested incrementally against tuned baseline
docs: extra feature engineering did not improve ROC-AUC
chore: split planning.md into forward-looking plan + per-notebook decision logs
```
