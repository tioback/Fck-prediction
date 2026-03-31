# Determinação de Resistência de Concreto

Concrete resistance determination tool — a clean rewrite of a legacy structural engineering script.

## Prerequisites

- [Python 3.14+](https://www.python.org/downloads/)
- [uv](https://docs.astral.sh/uv/getting-started/installation/) — dependency and virtual environment manager

## Setup

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
uv sync
```

## Project Structure

```
Fck-prediction/
├── src/
│   └── fck_prediction/        # Main package
│       ├── models/            # Domain data models
│       ├── calculation/       # Core concrete resistance logic
│       ├── data/              # XLS parsing and data ingestion
│       └── utils/             # Shared internal utilities
├── tests/
│   ├── unit/
│   └── integration/
├── scripts/                   # Development tooling
├── data/                      # Sample data files
└── pyproject.toml             # Project metadata and dependencies
```

## Development

```bash
# Run tests
uv run pytest

# Lint and format
uv run ruff check .
uv run ruff format .

# Type check
uv run mypy src
```

## Legacy Analysis

To extract structure from the original legacy script:

```bash
uv run python scripts/analyze_legacy.py <path/to/legacy_file.py>
```

Output is saved to `legacy_summary.txt`.
