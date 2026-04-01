import os
import warnings
from pathlib import Path
import numpy as np

# ── Output paths ──────────────────────────────────────────────────────────────

OUTPUT_DIR   = Path(os.environ.get("FCK_OUTPUT_DIR", "outputs"))
RESULTS_DIR  = OUTPUT_DIR / "results"
FIGURES_DIR  = OUTPUT_DIR / "figures"
DATASETS_DIR = OUTPUT_DIR / "datasets"

RES_SHAP = RESULTS_DIR / "shap"
RES_MCS  = RESULTS_DIR / "mcs"

FIG_MONTE_CARLO     = FIGURES_DIR / "monte_carlo"
FIG_PICP            = FIGURES_DIR / "picp"
FIG_CROSS_VAL       = FIGURES_DIR / "cross_validation"
FIG_TAYLOR          = FIGURES_DIR / "taylor"
FIG_PERFORMANCE     = FIGURES_DIR / "performance"
FIG_CORRELATION     = FIGURES_DIR / "correlation"
FIG_DM_HEATMAP      = FIGURES_DIR / "dm_heatmap"
FIG_SHAP            = FIGURES_DIR / "shap"
FIG_PREDICTION      = FIGURES_DIR / "prediction"
FIG_IFI             = FIGURES_DIR / "ifi"
FIG_RADAR           = FIGURES_DIR / "radar"
FIG_LEARNING_CURVES = FIGURES_DIR / "learning_curves"
FIG_PERM_IMP        = FIGURES_DIR / "permutation_importance"
FIG_PDP             = FIGURES_DIR / "pdp"
FIG_QQ              = FIGURES_DIR / "qq_plots"
FIG_RESIDUAL_DIAG   = FIGURES_DIR / "residual_diagnostics"

DIRECTORIES = [
    OUTPUT_DIR, RESULTS_DIR, FIGURES_DIR, DATASETS_DIR,
    RES_SHAP, RES_MCS,
    FIG_MONTE_CARLO, FIG_PICP, FIG_CROSS_VAL, FIG_TAYLOR,
    FIG_PERFORMANCE, FIG_CORRELATION, FIG_DM_HEATMAP, FIG_SHAP,
    FIG_PREDICTION, FIG_IFI, FIG_RADAR, FIG_LEARNING_CURVES,
    FIG_PERM_IMP, FIG_PDP, FIG_QQ, FIG_RESIDUAL_DIAG,
]

# ── Data constants ────────────────────────────────────────────────────────────

DATA_FILE    = Path("data/Concrete_Data.xls")
COLUMN_NAMES = ['C', 'S', 'FA', 'SF', 'LP', 'W', 'SP', 'Gravel', 'Sand', 'Age', 'fck']
TARGET       = 'fck'
RANDOM_SEED  = 42

# ── Experiment constants ──────────────────────────────────────────────────────

N_MONTE_CARLO  = 30
N_MC_OPT       = 30
N_CLEAN_RUNS   = 3
N_SPLITS_CV    = 10
N_REPEATS_CV   = 10
N_REPEATS_PERM = 10
N_GRID_PDP     = 50
N_GRID_LC      = 10
CV_FOLDS_LC    = 5

# ── Metric display ────────────────────────────────────────────────────────────

METRICS_LIST  = ["RMSE", "R2", "MAE", "MAPE"]
METRIC_LABELS = {
    "RMSE": "RMSE (MPa)",
    "R2":   "R²",
    "MAE":  "MAE (MPa)",
    "MAPE": "MAPE (%)",
}

# ── Plot colours ──────────────────────────────────────────────────────────────

TRAINING_COLOR = '#2E86AB'
TESTING_COLOR  = '#A23B72'

# ── PICP ─────────────────────────────────────────────────────────────────────

CONFIDENCE_LEVELS = [0.68, 0.80, 0.90, 0.95, 0.99]
Z_SCORES_MAP      = {0.68: 1.0, 0.80: 1.28, 0.90: 1.645, 0.95: 1.96, 0.99: 2.576}


def setup_environment():
    print("=" * 80)
    print("SISTEMA v14 CORRIGIDO: OTIMIZAÇÃO DE LIMPEZA + REPEATED K-FOLD CV")
    print("Modelos: Linear, BayesianRidge, DecisionTree, RandomForest,")
    print("GradientBoosting, SVR_rbf, SVR_poly, XGBoost, ANN (MLP)")
    print("=" * 80)

    for d in DIRECTORIES:
        d.mkdir(parents=True, exist_ok=True)
        print(f"📁 Diretório criado/verificado: {d}")

    print(f"\n✅ Total de {len(DIRECTORIES)} diretórios preparados")

    np.random.seed(RANDOM_SEED)
    warnings.filterwarnings('ignore')
