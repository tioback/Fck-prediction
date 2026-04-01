# Determinação de Resistência de Concreto

ML pipeline for predicting concrete compressive strength (fck, MPa) — a clean modular rewrite of a legacy 3000-line script.

Nine regression models are evaluated: Linear, BayesianRidge, DecisionTree, RandomForest, GradientBoosting, SVR (rbf/poly), XGBoost, and ANN (MLP).

## Prerequisites

- [Python 3.14+](https://www.python.org/downloads/)
- [uv](https://docs.astral.sh/uv/getting-started/installation/) — dependency and virtual environment manager
- `libomp` (macOS only, required by XGBoost): `brew install libomp`

## Setup

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
uv sync
```

## Running the pipeline

```bash
uv run python -m fck_prediction.cli
```

All outputs are written to `outputs/` (gitignored). The directory is created automatically on first run.

To redirect outputs to a custom location:

```bash
FCK_OUTPUT_DIR=/path/to/my/outputs uv run python -m fck_prediction.cli
```

### Output structure

```
outputs/
├── results/            # Excel tables (.xlsx)
│   ├── shap/           # SHAP importance per model
│   └── mcs/            # Model Confidence Set
├── figures/            # Plots (300 dpi PNG)
│   ├── monte_carlo/
│   ├── cross_validation/
│   ├── taylor/
│   ├── performance/
│   ├── radar/
│   ├── learning_curves/
│   ├── permutation_importance/
│   ├── pdp/
│   ├── qq_plots/
│   ├── shap/
│   ├── dm_heatmap/
│   ├── ifi/
│   ├── correlation/
│   └── prediction/
└── datasets/           # Optimised DEV datasets per model
```

## Project structure

```
Fck-prediction/
├── src/fck_prediction/
│   ├── cli.py                      # Pipeline entry point
│   ├── config.py                   # Constants, output paths, setup
│   ├── data/
│   │   └── loader.py               # Data loading and minimal cleaning
│   ├── models/
│   │   └── registry.py             # Model instantiation
│   ├── preprocessing/
│   │   ├── cleaners.py             # Outlier removal methods
│   │   └── cleaning_optimizer.py  # Per-model cleaning selection
│   ├── training/
│   │   └── trainer.py              # Model training + reference partition
│   ├── evaluation/
│   │   ├── monte_carlo.py
│   │   ├── picp.py
│   │   ├── cross_validation.py
│   │   ├── ifi.py
│   │   ├── statistical_tests.py
│   │   ├── model_confidence_set.py
│   │   ├── summary_stats.py
│   │   ├── residual_diagnostics.py
│   │   ├── learning_curves.py
│   │   └── normality.py
│   ├── visualization/
│   │   ├── taylor_diagram.py
│   │   ├── performance_plots.py
│   │   ├── correlation.py
│   │   ├── prediction_plots.py
│   │   └── radar_chart.py
│   ├── interpretation/
│   │   ├── shap_analysis.py
│   │   ├── permutation_importance.py
│   │   └── pdp.py
│   └── inference/
│       └── predictor.py            # Predictions for new concrete mixes
├── data/
│   └── Concrete_Data.xls
├── ARCHITECTURE.md                 # Section-by-section pipeline reference
└── pyproject.toml
```

## Development

```bash
# Lint and format
uv run ruff check .
uv run ruff format .

# Type check
uv run mypy src
```
