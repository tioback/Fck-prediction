# ROADMAP — v15.py Refactoring

All commits follow Conventional Commits. No logic changes — pure structural extraction.
Reference: ARCHITECTURE.md (section codes S00–S30).

## Commits

| # | Commit message | Sections | Files |
|---|---|---|---|
| 1 | `chore: scaffold module directories and roadmap` | S00 | ROADMAP.md, all `__init__.py` stubs |
| 2 | `refactor(config): extract constants and setup (S00)` | S00 | `config.py` |
| 3 | `refactor(data): extract data loader (S01)` | S01 | `data/loader.py` |
| 4 | `refactor(models): extract model registry (S02)` | S02 | `models/registry.py` |
| 5 | `refactor(preprocessing): extract cleaners and optimizer (S03, S05)` | S03, S05 | `preprocessing/cleaners.py`, `preprocessing/cleaning_optimizer.py` |
| 6 | `refactor(training): extract trainer with reference partition (S06)` | S06 | `training/trainer.py` |
| 7 | `refactor(evaluation): extract monte carlo runs (S04, S07)` | S04, S07 | `evaluation/monte_carlo.py` |
| 8 | `refactor(evaluation): extract PICP analysis (S08)` | S08 | `evaluation/picp.py` |
| 9 | `refactor(evaluation): extract repeated k-fold CV (S09)` | S09 | `evaluation/cross_validation.py` |
| 10 | `refactor(visualization): extract all visualization modules` | S10, S11, S13, S16, S24, S25 | `visualization/taylor_diagram.py`, `visualization/performance_plots.py`, `visualization/correlation.py`, `visualization/prediction_plots.py`, `visualization/radar_chart.py` |
| 11 | `refactor(evaluation): extract ranking and statistical analysis modules` | S12, S14, S17, S20, S21, S22, S23 | `evaluation/ifi.py`, `evaluation/statistical_tests.py`, `evaluation/model_confidence_set.py`, `evaluation/summary_stats.py` |
| 12 | `refactor(evaluation): extract diagnostic analysis modules` | S18, S26, S29 | `evaluation/residual_diagnostics.py`, `evaluation/learning_curves.py`, `evaluation/normality.py` |
| 13 | `refactor(interpretation): extract explainability modules` | S15, S27, S28 | `interpretation/shap_analysis.py`, `interpretation/permutation_importance.py`, `interpretation/pdp.py` |
| 14 | `refactor(inference): extract new mix predictor (S19)` | S19 | `inference/predictor.py` |
| 15 | `refactor(cli): add pipeline orchestrator (S30)` | S30 | `cli.py` |

## Status

- [x] ROADMAP.md
- [x] Commit 1 — scaffold
- [x] Commit 2 — config
- [x] Commit 3 — data
- [x] Commit 4 — models
- [x] Commit 5 — preprocessing
- [x] Commit 6 — training
- [x] Commit 7 — monte carlo
- [x] Commit 8 — picp
- [x] Commit 9 — cross validation
- [x] Commit 10 — visualization
- [x] Commit 11 — evaluation (ranking/stats)
- [x] Commit 12 — evaluation (diagnostics)
- [x] Commit 13 — interpretation
- [x] Commit 14 — inference
- [x] Commit 15 — cli

---

# ROADMAP — Quality Improvement

Goals: clean coding techniques, industry-standard patterns, logic improvements, tests, readability.
All commits follow Conventional Commits. Shared development rules: [CLAUDE.md](CLAUDE.md).

## Setup — Tooling Prerequisites

Add to `pyproject.toml` dev dependencies before any phase begins.

| Tool | Purpose |
|---|---|
| `vulture` | Dead code detection (unused functions, methods, variables) |
| `pre-commit` | Enforce ruff + mypy on every commit |
| `pytest-cov` | Test coverage reports |
| `pytest-xdist` | Parallel test execution |
| `hypothesis` | Property-based testing for data transformations |
| `rich` | Structured console output (replaces print + emojis) |

Commit: `chore(dev-deps): add quality improvement tooling`

---

## Phase 1 — Static Analysis

**Scope:** All modules. Executed per module group, not all at once.
**Rule:** No logic changes — removal and annotation only.

- Add type hints to all public functions and return types
- Remove unused imports (`ruff F401`)
- Remove unused variables (`ruff F841`)
- Remove unused functions and methods (`vulture`)
- Audit `pyproject.toml` packages against actual imports; remove unused dependencies
- Configure `pre-commit` hooks (ruff + mypy)

Each module group is an independent PR. Groups follow the existing module boundaries since Phase 1 requires no cross-cutting analysis.

---

## Phase 2 — Logic Improvement

**Scope:** All modules with compute logic.
**Rule:** At the start of each group, read the relevant code and map the actual call graph and data flow to determine natural groupings by functional cohesion — not by directory structure. Groups and their boundaries are determined during execution.

For each group:
1. Complexity audit (time + space) scoped to that group only
2. Implement improvements — reduce nested loops, eliminate redundant recomputation, pre-allocate structures, identify parallelization opportunities

**Divide and conquer:** each group is an independent PR once its boundaries are established.

---

## Phase 3 — Compute / Visualization Separation

**Scope:** Functions that mix computation and rendering in the same body, making them untestable.
**Rule:** Identify which functions have this problem during execution. Each affected function is split into a pure compute function (returns data) and a separate render function (takes data, produces output). No logic changes.

This phase is a prerequisite for Phase 5.
**Can run in parallel with Phase 4.**

---

## Phase 4 — Configuration Consolidation

**Scope:** Hardcoded values scattered across modules — hyperparameters, inline data, method parameters buried in closures.
**Rule:** All configurable values move to `config.py` or an appropriate data file. No logic changes.

**Can run in parallel with Phase 3.**

---

## Phase 5 — Tests

**Scope:** Compute layer only. Requires Phase 3 complete.
Render/visualization functions are not tested directly — test the data that feeds them.

- **Unit tests:** pure computation functions
- **Integration tests:** pipeline stage contracts — correct input produces correct output shape and type
- **Property-based tests** (`hypothesis`): data transformation functions where invariants can be expressed (e.g. a cleaning method always returns a strict subset of its input)

Shared fixtures in `tests/conftest.py` — dataset loaded once, shared across all tests.

---

## Phase 6 — Logging + Output Management

- Replace all `print()` with structured `logging` (hierarchy: `fck_prediction.data`, `fck_prediction.training`, etc.)
- Add `rich` for console output: progress bars on long-running stages, summary tables at end
- Standardize output file naming convention
- Generate a run manifest at pipeline completion listing all produced files

---

## Phase 7 — CLI Decomposition

**Scope:** `cli.py` `main()` only.

- Extract each pipeline stage into a named function
- Group into high-level runners by concern
- Add per-stage error handling: fail loudly with stage context, not silently

---

## Phase 8 — Naming, Documentation, Comments

Done last — all structure and logic are settled before touching names or docs.

- Rename cryptic variables, parameters, and functions across all modules
- Standardize language to English throughout (remove mixed Portuguese identifiers and comments)
- Add concise docstrings to all public functions (`Args:` / `Returns:` where non-obvious)
- Add inline comments only for non-obvious logic

---

## Quality Improvement Status

- [ ] Setup — tooling prerequisites
- [ ] Phase 1 — Static analysis
- [ ] Phase 2 — Logic improvement
- [ ] Phase 3 — Compute/visualization separation
- [ ] Phase 4 — Configuration consolidation
- [ ] Phase 5 — Tests
- [ ] Phase 6 — Logging + output management
- [ ] Phase 7 — CLI decomposition
- [ ] Phase 8 — Naming, documentation, comments
