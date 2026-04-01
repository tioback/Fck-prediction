"""Pipeline orchestrator — fck prediction (v15 refactored).

Calls all modules in order, threading return values through as function
arguments. No global state; all data flows via explicit returns.

Usage:
    python -m fck_prediction.cli
"""

from datetime import datetime

from fck_prediction.config import (
    DATA_FILE, N_MONTE_CARLO, N_MC_OPT, N_SPLITS_CV, N_REPEATS_CV, setup_environment,
)
from fck_prediction.data.loader import load_data
from fck_prediction.models.registry import get_models
from fck_prediction.preprocessing.cleaners import get_cleaning_methods
from fck_prediction.preprocessing.cleaning_optimizer import optimize_cleaning
from fck_prediction.training.trainer import train_models
from fck_prediction.evaluation.monte_carlo import (
    run_monte_carlo, run_monte_carlo_optimized,
)
from fck_prediction.evaluation.picp import run_picp
from fck_prediction.evaluation.cross_validation import run_repeated_kfold
from fck_prediction.evaluation.ifi import compute_ifi
from fck_prediction.evaluation.statistical_tests import (
    run_dm_heatmap, run_friedman_nemenyi, run_dm_test, run_plackett_luce,
)
from fck_prediction.evaluation.model_confidence_set import compute_mcs
from fck_prediction.evaluation.summary_stats import compute_summary_stats
from fck_prediction.evaluation.residual_diagnostics import run_residual_diagnostics
from fck_prediction.evaluation.learning_curves import run_learning_curves
from fck_prediction.evaluation.normality import run_normality_analysis
from fck_prediction.visualization.taylor_diagram import (
    plot_taylor_diagram_initial, plot_taylor_diagram,
)
from fck_prediction.visualization.performance_plots import plot_performance_metrics
from fck_prediction.visualization.correlation import plot_model_correlation
from fck_prediction.visualization.prediction_plots import plot_predictions
from fck_prediction.visualization.radar_chart import plot_radar
from fck_prediction.interpretation.shap_analysis import run_shap
from fck_prediction.interpretation.permutation_importance import run_permutation_importance
from fck_prediction.interpretation.pdp import run_pdp
from fck_prediction.inference.predictor import predict_new_mixes


def main():
    # ── S00 – Environment ────────────────────────────────────────────────────
    setup_environment()

    # ── S01 – Data ───────────────────────────────────────────────────────────
    df, X_full, y_full, feature_names = load_data(DATA_FILE)

    # ── S02 – Models ─────────────────────────────────────────────────────────
    models, model_list = get_models()

    # ── S03 – Cleaning methods ───────────────────────────────────────────────
    cleaning_methods = get_cleaning_methods()

    # ── S04 – Monte Carlo (original data) ────────────────────────────────────
    df_results, df_vr = run_monte_carlo(X_full, y_full, models, model_list,
                                        n_runs=N_MONTE_CARLO)

    # ── S05 – Cleaning optimizer ─────────────────────────────────────────────
    best_cleaning_for_model, optimized_datasets, best_opt_summary = optimize_cleaning(
        X_full, y_full, models, model_list, cleaning_methods, feature_names)

    # ── S06 – Training (common reference partition) ──────────────────────────
    training_output = train_models(
        X_full, y_full, models, optimized_datasets,
        best_cleaning_for_model, feature_names)

    models_trained      = training_output['models_trained']
    scalers             = training_output['scalers']
    pred_ext            = training_output['pred_ext']
    pred_ref            = training_output['pred_ref']
    pred_train_dict     = training_output['pred_train_dict']
    results_df          = training_output['results_df']
    train_metrics       = training_output['train_metrics']
    error_variance      = training_output['error_variance']
    y_ref_ext           = training_output['y_ref_ext']
    X_tst_ref_sc        = training_output['X_tst_ref_sc']
    X_dev_ref_sc        = training_output['X_dev_ref_sc']
    sc_ref              = training_output['sc_ref']

    # ── S07 – Monte Carlo (optimized data) ───────────────────────────────────
    df_res_opt, df_vr_opt = run_monte_carlo_optimized(
        optimized_datasets, models, model_list,
        best_cleaning_for_model, feature_names, n_runs=N_MC_OPT)

    # ── S08 – PICP ───────────────────────────────────────────────────────────
    picp_df = run_picp(models_trained, optimized_datasets, feature_names)

    # ── S09 – Repeated K-Fold CV ─────────────────────────────────────────────
    repeated_cv_df, cv_summary_df, cv_metrics_df = run_repeated_kfold(
        optimized_datasets, models, model_list,
        best_cleaning_for_model, feature_names,
        n_splits=N_SPLITS_CV, n_repeats=N_REPEATS_CV)

    total_evals = N_SPLITS_CV * N_REPEATS_CV

    # ── S10 – Taylor Diagram (initial) ───────────────────────────────────────
    plot_taylor_diagram_initial(pred_ref, y_ref_ext)

    # ── S11 – Performance plots ───────────────────────────────────────────────
    plot_performance_metrics(train_metrics, error_variance)

    # ── S12 – IFI ranking ────────────────────────────────────────────────────
    ranking = compute_ifi(results_df)

    # ── S13 – Model correlation heatmap ──────────────────────────────────────
    plot_model_correlation(pred_ref)

    # ── S14 – DM heatmap ─────────────────────────────────────────────────────
    dm_df = run_dm_heatmap(pred_ref, y_ref_ext)

    # ── S15 – SHAP ────────────────────────────────────────────────────────────
    all_shap_importance = run_shap(models_trained, model_list, X_tst_ref_sc, feature_names)

    # ── S16 – Prediction plots ────────────────────────────────────────────────
    plot_predictions(pred_ref, y_ref_ext)

    # ── S17 – Summary stats ───────────────────────────────────────────────────
    compute_summary_stats(results_df)

    # ── S18 – Residual diagnostics ────────────────────────────────────────────
    diag_df = run_residual_diagnostics(pred_ref, y_ref_ext)

    # ── S19 – New mix predictions ─────────────────────────────────────────────
    pred_nd = predict_new_mixes(models_trained, scalers, X_full, feature_names)

    # ── S20 – Friedman + Nemenyi ─────────────────────────────────────────────
    fr_df = run_friedman_nemenyi(df_results)

    # ── S21 – DM test (tabular) ───────────────────────────────────────────────
    dm_out = run_dm_test(pred_ref, y_ref_ext)

    # ── S22 – Radar chart ────────────────────────────────────────────────────
    plot_radar(results_df, ranking)

    # ── S23 – Plackett-Luce ───────────────────────────────────────────────────
    pl_df = run_plackett_luce(pred_ref, y_ref_ext)

    # ── S24 – Taylor Diagram (full, with RMSE arcs) ───────────────────────────
    plot_taylor_diagram(pred_ref, y_ref_ext)

    # ── S25 – IFI sensitivity ─────────────────────────────────────────────────
    # (handled inside compute_ifi / ifi module if applicable)

    # ── S26 – Learning curves ─────────────────────────────────────────────────
    lc_results, diag_lc_df = run_learning_curves(
        models_trained, optimized_datasets, feature_names)

    # ── S27 – Permutation importance ──────────────────────────────────────────
    perm_imp_results, perm_imp_summary, feat_stats_perm_df = run_permutation_importance(
        models_trained, optimized_datasets, feature_names, all_shap_importance)

    # ── S28 – PDP ─────────────────────────────────────────────────────────────
    run_pdp(models_trained, optimized_datasets, feature_names, ranking, feat_stats_perm_df)

    # ── S29 – Q-Q plots + normality ───────────────────────────────────────────
    norm_tbl = run_normality_analysis(pred_ref, y_ref_ext, ranking)

    # ── S30 – Executive summary ───────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("🏆 RESUMO EXECUTIVO – PIPELINE v15 CONCLUÍDO")
    print("=" * 80)

    print(f"\n📊 Dataset original: {df.shape[0]} amostras")
    print(f"🔢 Features: {X_full.shape[1]}")
    print(f"🤖 Modelos: {len(models)}")
    print(f"🔄 Monte Carlo original: {N_MONTE_CARLO} runs")
    print(f"🔄 Monte Carlo otimizado: {N_MC_OPT} runs")
    print(f"🔄 Repeated K-Fold CV: {N_SPLITS_CV}×{N_REPEATS_CV} = {total_evals} avaliações/modelo")

    print("\n📊 MÉTODOS DE LIMPEZA OTIMIZADOS:")
    for _, row in best_opt_summary.iterrows():
        print(f"   {row['Model']:20s}: {row['Best_Method']:25s} "
              f"({row['Samples']:.0f} amostras dev)")

    print("\n🥇 RANKING IFI:")
    for i, row in ranking.reset_index(drop=True).iterrows():
        print(f"   {i+1}. {row['Model']:15s}: IFI={row['IFI']:.4f} | "
              f"R²={row['R2']:.4f} | {row['Cleaning_Method']}")

    print("\n🔧 CORREÇÕES E NOVOS BLOCOS (v13 → v15):")
    print("   [FIX-1] Data Leakage eliminado na seleção do método de limpeza")
    print("   [FIX-2] y_ext unificado: partição de referência comum para análises comparativas")
    print("   [FIX-3] clone() substitui blocos if/elif para recriar modelos")
    print("   [FIX-4] PICP: std estimado no DEV, aplicado no TEST")
    print("   [FIX-5] MAPE padronizado via sklearn em todo o script")
    print("   [FIX-6] plt.show() removido; figuras salvas e fechadas com plt.close()")
    print("   [NEW-1] Taylor Diagram (pred_ref / y_ref_ext)")
    print("   [NEW-2] Radar Chart multi-métrico (3 versões: all, ranges, Top 5)")
    print("   [NEW-3] Learning Curves – diagnóstico over/underfitting")
    print("   [NEW-4] Permutation Importance + comparação SHAP vs Perm")
    print("   [NEW-5] Partial Dependence Plots – 1D (1 modelo, Top3, todos) + 2D")
    print("   [NEW-6] Q-Q Plots + testes de normalidade dos resíduos (Shapiro, JB, AD)")

    print("\n📁 ARQUIVOS GERADOS:")
    print("   Paper/Results/                 – Excel com todos os resultados")
    print("   Paper/Figures/                 – Todas as figuras (300 dpi)")
    print("   Resultados_Artigo/             – Resultados intermediários")
    print("   Bancos_Otimizados/             – Datasets DEV otimizados por modelo")
    print("   Figuras_MonteCarlo/            – Monte Carlo original e otimizado")
    print("   Figuras_NestedCV/              – Repeated CV, Taylor, LC, QQ, Violin")
    print("   Figuras_Radar/                 – Radar Charts (3 versões)")
    print("   Figuras_LearningCurves/        – Learning Curves (all, gap, best/worst)")
    print("   Figuras_PermutationImportance/ – Heatmap, barras, SHAP vs Perm, boxplot")
    print("   Figuras_PDP/                   – PDPs 1D (all features, top3) + 2D")
    print("   Figuras_QQPlot/                – Q-Q plots, histogramas, boxplot, Shapiro")
    print("   Figuras_Shap/                  – SHAP bar, summary, dependence, PDP, Perm")
    print("   Figuras_Violin/                – Violin plots")
    print("   Figuras_DM_Heatmap/            – Diebold-Mariano heatmap")
    print("   Figuras_IFI/                   – IFI sensitivity")

    print("\n" + "=" * 80)
    print(f"✅ PROCESSO v15 CONCLUÍDO EM {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)


if __name__ == "__main__":
    main()
