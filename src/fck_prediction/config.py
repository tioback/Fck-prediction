import os
import warnings
import numpy as np

# ── Output directories (S00) ─────────────────────────────────────────────────

DIRECTORIES = [
    "Paper", "Paper/Figures", "Paper/Tables", "Paper/Results", "Paper/Script",
    "Paper/Residual_Diagnostics", "Resultados_Artigo", "Figuras_Artigo",
    "Modelos_Salvos", "Figures", "Figuras_Correlacao",
    "Figuras_Shap", "Paper/Results/SHAP_Results", "Paper/Results/MCS_Results",
    "Figuras_Treino_Teste", "Figuras_MonteCarlo", "Figuras_Violin",
    "Figuras_DM_Heatmap", "Figuras_NestedCV", "Figuras_IFI", "Figuras_PICP",
    "Bancos_Otimizados", "Paper/Figures/Validacao_Externa",
    "Paper/Figures/Residuos", "Paper/Figures/KDE_Density",
    "Figuras_Radar", "Figuras_LearningCurves",
    "Figuras_PermutationImportance", "Figuras_PDP", "Figuras_QQPlot",
]

# ── Data constants ────────────────────────────────────────────────────────────

DATA_FILE    = "data/Concrete_Data.xls"
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

    for directory in DIRECTORIES:
        os.makedirs(directory, exist_ok=True)
        print(f"📁 Diretório criado/verificado: {directory}")

    print(f"\n✅ Total de {len(DIRECTORIES)} diretórios preparados")

    np.random.seed(RANDOM_SEED)
    warnings.filterwarnings('ignore')
