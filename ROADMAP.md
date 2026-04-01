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
