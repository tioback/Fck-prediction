# CLAUDE.md — Development Rules

These rules apply to every session, agent, and PR in this project.
Reference: [ROADMAP.md](ROADMAP.md) for phase definitions and work breakdown.

---

## Commits

- Follow [Conventional Commits](https://www.conventionalcommits.org/) strictly: `type(scope): description`
- Valid types: `feat`, `fix`, `refactor`, `chore`, `docs`, `build`, `test`
- Atomic commits — one logical change per commit
- Never skip hooks (`--no-verify`)

## Code Quality

All three must pass before committing:

```bash
uv run ruff check .
uv run ruff format .
uv run mypy src
```

Once the Setup phase is complete, also run:

```bash
uv run vulture src
```

Tool config is in `pyproject.toml` — do not override it inline.

## Clean Code Rules

- No unused imports, variables, parameters, or functions
- No silent exception handling — no bare `except`, no `except Exception: pass`
- No logic in `__init__.py` files
- Functions do one thing — if a function computes and renders, it must be split into separate functions
- No hardcoded scalar values, paths, or parameters — they belong in `config.py`
- English only — all identifiers, comments, and docstrings

## Phase Discipline

Each phase in the ROADMAP has a single stated purpose. Do not mix concerns:

- **Phase 1** (static analysis): type hints and dead code removal only — no logic changes
- **Phase 2** (logic improvement): algorithm and complexity changes only — no renames, no docs
- **Phase 8** (naming/docs): naming and documentation only — no logic changes

If an issue outside the current phase's scope is spotted, note it — do not fix it in place.

## Testing

- Tests live in `tests/unit/` and `tests/integration/`
- Shared fixtures go in `tests/conftest.py` — load shared data once
- Do not test visualization output — test the data that feeds it
- Run tests: `uv run pytest`
- Run with coverage: `uv run pytest --cov`

## PRs

- One PR per logical unit of work matching a ROADMAP sub-item
- PR title follows the same Conventional Commits format as commit messages
- No PR spans multiple phases
