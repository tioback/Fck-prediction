# ARCHITECTURE.md — v15.py Legacy Pipeline Reference

Source: `src/fck_prediction/v15.py` (2981 lines)  
Domain: ML regression to predict concrete compressive strength (fck, MPa)  
Models: Linear, BayesianRidge, DecisionTree, RandomForest, GradientBoosting, SVR_rbf, SVR_poly, XGBoost, ANN (MLP)

---

## Section Map

### S00 — Imports & Environment Setup
**Lines:** 1–91  
**Responsibility:** All library imports, directory tree creation, global random seed.  
**Key inputs:** None  
**Key outputs:** 29 output directories created on disk; `np.random.seed(42)` set; all sklearn/shap/scipy/matplotlib symbols in scope.  
**Dependencies:** None

---

### S01 — Data Loading & Minimal Cleaning
**Lines:** 94–128  
**Responsibility:** Load `Concrete_Data.xls`, rename 11 columns, coerce to numeric, drop NaN rows.  
**Key inputs:** `Concrete_Data.xls` (file on disk)  
**Key outputs:** `df` (clean DataFrame), `X_full` (feature matrix), `y_full` (target Series), `feature_names` (list[str]), `target = 'fck'`  
**Dependencies:** S00

---

### S02 — Model Registry
**Lines:** 130–155  
**Responsibility:** Instantiate and register all 9 sklearn/XGBoost estimators in a dict.  
**Key inputs:** None (constants only)  
**Key outputs:** `models` (dict[str, estimator]), `model_list` (list[str])  
**Dependencies:** S00

---

### S03 — Outlier Cleaning Method Registry
**Lines:** 157–234  
**Responsibility:** Define 8 outlier-removal functions and register 16 parameterized variants in `cleaning_methods`.  
**Key inputs:** None (pure function definitions)  
**Key outputs:** `cleaning_methods` (dict[str, {'func': callable}])  
  - Functions: `clean_isolationforest`, `clean_iqr`, `clean_zscore`, `clean_percentile`, `clean_dbscan`, `clean_elliptic`, `clean_svm`, `clean_lof`  
**Dependencies:** S00

---

### S04 — Monte Carlo + Variance Ratio (Original Data)
**Lines:** 236–341  
**Responsibility:** 30-run Monte Carlo train/test split benchmark on raw (uncleaned) data; compute variance ratio per model per metric; produce boxplot figure.  
**Key inputs:** `X_full`, `y_full`, `models`, `model_list`  
**Key outputs:** `df_results` (DataFrame, all MC runs × models × datasets), `df_vr` (variance ratio table); files: `Resultados_Artigo/Monte_Carlo_Results.xlsx`, `Resultados_Artigo/Variance_Ratio_Results.xlsx`, `Figuras_MonteCarlo/Benchmark_Boxplot_All_Metrics.png`  
**Dependencies:** S01, S02

---

### S05 — Cleaning Method Optimization (No Data Leakage)
**Lines:** 343–480  
**Responsibility:** For each model, evaluate all 16 cleaning methods via 3-run CV on DEV split only (test never seen during cleaning). Select best method per model. Generate canonical optimized DEV datasets.  
**Key inputs:** `X_full`, `y_full`, `models`, `model_list`, `cleaning_methods`, `feature_names`, `target`  
**Key outputs:**  
  - `best_cleaning_for_model` (dict[str, str])  
  - `optimized_datasets` (dict[str, {'dev_df': DataFrame, 'scaler': StandardScaler, 'X_tst_raw': DataFrame, 'y_tst_raw': Series}])  
  - `opt_summary_full` (DataFrame), `best_opt_summary` (DataFrame)  
  - Files: `Bancos_Otimizados/Dataset_Otimizado_{model}.xlsx`, `Resultados_Artigo/Melhor_Metodo_Limpeza_por_Modelo.xlsx`  
**Dependencies:** S01, S02, S03

---

### S06 — Reference Partition + Model Training
**Lines:** 482–592  
**Responsibility:** Define a single canonical reference partition (random_state=42, 80/20 split) shared by all models for comparable evaluation. Train each model on its optimized DEV dataset. Record train/test metrics and predictions on both own test set and the shared reference test set.  
**Key inputs:** `X_full`, `y_full`, `models`, `optimized_datasets`, `best_cleaning_for_model`, `feature_names`, `target`  
**Key outputs:**  
  - `X_dev_ref`, `X_tst_ref`, `y_dev_ref`, `y_tst_ref` — reference partition (raw)  
  - `X_dev_ref_sc`, `X_tst_ref_sc`, `sc_ref` — reference partition (scaled)  
  - `y_ref_ext` (np.ndarray) — fixed test labels for all comparative analyses  
  - `models_trained` (dict[str, fitted estimator])  
  - `scalers` (dict[str, StandardScaler])  
  - `pred_ext` (dict[str, np.ndarray]) — predictions on own test set  
  - `pred_ref` (dict[str, np.ndarray]) — predictions on shared reference test set  
  - `pred_train_dict` (dict[str, np.ndarray])  
  - `train_metrics`, `error_variance` (lists → used to build DataFrames in S09)  
  - `results_df` (DataFrame: Model, R2, RMSE, MAE, MAPE, Cleaning_Method, Samples)  
  - File: `Resultados_Artigo/Results_Otimizado.xlsx`  
**Dependencies:** S01, S02, S05

---

### S07 — Monte Carlo + Variance Ratio (Optimized Data)
**Lines:** 594–708  
**Responsibility:** 30-run Monte Carlo re-split within each model's optimized DEV (never touching held-out test). Compute variance ratios. Produce boxplot annotated with VR, cleaning method, and sample count.  
**Key inputs:** `optimized_datasets`, `models`, `model_list`, `best_cleaning_for_model`, `feature_names`, `target`  
**Key outputs:** `df_res_opt`, `df_vr_opt`; files: `Resultados_Artigo/Monte_Carlo_Results_Otimizado.xlsx`, `Figuras_MonteCarlo/Benchmark_Boxplot_All_Metrics_Otimizado.png`  
**Dependencies:** S05, S06

---

### S08 — PICP (Prediction Interval Coverage Probability)
**Lines:** 710–859  
**Responsibility:** Estimate residual std/quantiles on DEV; apply intervals to TEST. Compute PICP for 5 confidence levels × 2 methods (Gaussian, Quantile). Produce heatmaps and calibration curves.  
**Key inputs:** `optimized_datasets`, `models_trained`, `feature_names`, `target`  
**Key outputs:** `picp_df`; files: `Resultados_Artigo/PICP_Results_Otimizado.xlsx`, `Paper/Results/PICP_Summary_Table.xlsx`, `Figures/PICP_Heatmap_Comparison.png`, `Figures/PICP_Calibration_Curves.png`  
**Dependencies:** S05, S06

---

### S09 — Repeated K-Fold Cross Validation (10×10)
**Lines:** 861–1071  
**Responsibility:** 10-fold × 10-repeat CV on each model's optimized DEV. Compute R² CI, stability heatmap, violin and bar plots with IC. Second pass computes train vs test metrics per fold.  
**Key inputs:** `optimized_datasets`, `models`, `model_list`, `best_cleaning_for_model`, `feature_names`, `target`  
**Key outputs:** `repeated_cv_df`, `cv_summary_df`, `repeated_cv_scores`, `model_order_cv`, `cv_metrics_df`; files: `Resultados_Artigo/Repeated_CV_*.xlsx`, `Figuras_NestedCV/Repeated_CV_*.png`, `Paper/Figures/Repeated_CV_*.png`  
**Dependencies:** S05, S06

---

### S10 — Taylor Diagram (First Pass)
**Lines:** 1073–1145  
**Responsibility:** Compute std, correlation, RMSE of each model vs `y_ref_ext`. Render Taylor Diagram via `skill_metrics` or manual polar fallback.  
**Key inputs:** `pred_ref`, `y_ref_ext`, `feature_names`  
**Key outputs:** `taylor_stats` (DataFrame); files: `Figuras_NestedCV/Taylor_Diagram.png`, `Paper/Results/Taylor_Diagram_Statistics.xlsx`  
**Dependencies:** S06  
**Note:** Functionally duplicated and extended by S22 (lines 1900–1992). Both produce the same output paths; S22 overwrites.

---

### S11 — Performance Metrics & Distribution Plots
**Lines:** 1147–1258  
**Responsibility:** Aggregate `train_metrics` and `error_variance` lists from S06 into DataFrames. Produce boxplots (2×2), error variance boxplot, and violin plots for R², RMSE, MAE, MAPE — training vs testing.  
**Key inputs:** `train_metrics`, `error_variance` (from S06)  
**Key outputs:** `metrics_df`, `error_var_df`; files: `Figures/Boxplots_All_Metrics_Otimizado.png`, `Figures/Error_Variance_Boxplot_Otimizado.png`, `Figuras_Violin/Violin_Plots_All_Metrics_Otimizado.png`  
**Dependencies:** S06

---

### S12 — IFI Sensitivity Analysis
**Lines:** 1260–1340  
**Responsibility:** Compute entropy-weighted composite IFI score from test metrics; rank models. Run sensitivity analysis across R² weight sweep [0.1–0.8]. Produce line plot, heatmap, and bar chart.  
**Key inputs:** `results_df` (from S06)  
**Key outputs:** `ranking` (DataFrame sorted by IFI, adds IFI column to `results_df`); files: `Paper/Results/model_ranking_otimizado.xlsx`, `Figuras_IFI/*.png`, `Paper/Figures/IFI_*.png`  
**Dependencies:** S06  
**Note:** `ranking` is consumed by S22 (Radar Chart top 5), S25 (PDP best model), S26 (Q-Q top 3).

---

### S13 — Model Prediction Correlation Heatmap
**Lines:** 1342–1360  
**Responsibility:** Compute Pearson correlation matrix of all models' predictions on the common reference test set. Render heatmap.  
**Key inputs:** `pred_ref`, `y_ref_ext`  
**Key outputs:** Files: `Figuras_Correlacao/Correlacao_Performance_Modelos_Otimizado.png`  
**Dependencies:** S06

---

### S14 — Diebold-Mariano Heatmap
**Lines:** 1362–1398  
**Responsibility:** Pairwise Diebold-Mariano test on squared errors from reference predictions. Render p-value heatmap.  
**Key inputs:** `pred_ref`, `y_ref_ext`  
**Key outputs:** `dm_df` (pairwise DM stats); files: `Resultados_Artigo/DM_Heatmap_Results_Otimizado.xlsx`, `Figuras_DM_Heatmap/DM_Heatmap_Otimizado.png`  
**Dependencies:** S06

---

### S15 — SHAP Feature Importance
**Lines:** 1400–1526  
**Responsibility:** For each trained model: select explainer (TreeExplainer → LinearExplainer → KernelExplainer). Compute SHAP values on reference test set. Generate bar, summary, and dependence plots. Aggregate into normalized heatmap.  
**Key inputs:** `models_trained`, `model_list`, `X_tst_ref_sc`, `feature_names`  
**Key outputs:** `all_shap_importance` (list of DataFrames), `cons_shap` (consolidated), `piv_shap`, `piv_shap_norm`; files: `Figuras_Shap/SHAP_*.png`, `Paper/Results/SHAP_Results/*.xlsx`  
**Dependencies:** S06

---

### S16 — Parity Plots, Residual Plots, KDE Plots, MAPE Plots
**Lines:** 1528–1612  
**Responsibility:** For each model: scatter (predicted vs actual), residual vs predicted, KDE 2D density, MAPE vs fck — all on reference partition.  
**Key inputs:** `pred_ref`, `y_ref_ext`  
**Key outputs:** Files: `Paper/Figures/Validacao_Externa/Scatter_*.png`, `Paper/Figures/Residuos/Residuals_*.png`, `Paper/Figures/KDE_Density/KDE_*.png`, `Figures/MAPE_vs_fck_*.png`  
**Dependencies:** S06

---

### S17 — Summary Statistics Table
**Lines:** 1614–1645  
**Responsibility:** Compute mean/std/median/min/max/Q1/Q3/IQR of each metric per model from `results_df`. Export pivot tables.  
**Key inputs:** `results_df`  
**Key outputs:** Files: `Resultados_Artigo/Boxplot_Statistics_Detailed_Otimizado.xlsx`, `Boxplot_Means_Otimizado.xlsx`, `Boxplot_Stds_Otimizado.xlsx`  
**Dependencies:** S06

---

### S18 — Econometric Residual Diagnostics
**Lines:** 1647–1726  
**Responsibility:** For each model: run Anderson-Darling normality (Shapiro-Francia), Jarque-Bera, Breusch-Pagan, White heteroscedasticity, Durbin-Watson, Breusch-Godfrey autocorrelation tests. Plot ACF/PACF. Render heteroscedasticity p-value bar chart.  
**Key inputs:** `pred_ref`, `y_ref_ext`  
**Key outputs:** `diag_df`; files: `Paper/Results/residual_diagnostics_full_otimizado.xlsx`, `Paper/Residual_Diagnostics/ACF_PACF_*.png`, `Paper/Figures/heterocedasticidade_pvalues_otimizado.png`  
**Dependencies:** S06

---

### S19 — New Mix Predictions
**Lines:** 1728–1762  
**Responsibility:** Apply trained models + per-model scalers to 18 new concrete mix recipes (hardcoded). Identify best model per mix.  
**Key inputs:** `models_trained`, `scalers`, `X_full`, `feature_names`  
**Key outputs:** `pred_nd` (DataFrame); file: `Paper/Results/predicoes_novas_dosagens_otimizado.xlsx`  
**Dependencies:** S06

---

### S20 — Friedman Test + Nemenyi CD Diagram
**Lines:** 1764–1810  
**Responsibility:** Pivot `df_results` (Monte Carlo original) to rank matrix; Friedman chi-square test; compute critical distance; render CD bar chart.  
**Key inputs:** `df_results` (from S04)  
**Key outputs:** `fr_df`; files: `Paper/Figures/cd_diagram_nemenyi.png`, `Paper/Results/friedman_nemenyi_ranks.xlsx`  
**Dependencies:** S04

---

### S21 — Diebold-Mariano Statistical Test (Table)
**Lines:** 1812–1830  
**Responsibility:** Pairwise DM test (same logic as S14 but saves to different output path without heatmap figure).  
**Key inputs:** `pred_ref`, `y_ref_ext`  
**Key outputs:** File: `Paper/Results/diebold_mariano_otimizado.xlsx`  
**Dependencies:** S06  
**Note:** Functionally overlaps with S14; S14 produces heatmap figure, S21 produces tabular output.

---

### S22 — Model Confidence Set (MCS)
**Lines:** 1832–1881  
**Responsibility:** Greedy MCS: iteratively remove model with highest mean squared loss until one remains. Bar chart colored by inclusion.  
**Key inputs:** `pred_ref`, `y_ref_ext`  
**Key outputs:** `mcs_res`; files: `Paper/Results/model_confidence_set_otimizado.xlsx`, `Paper/Figures/model_confidence_set_otimizado.png`  
**Dependencies:** S06

---

### S23 — Bayesian Plackett-Luce Ranking
**Lines:** 1883–1898  
**Responsibility:** Compute softmax(-RMSE) as ability score for Plackett-Luce ranking.  
**Key inputs:** `pred_ref`, `y_ref_ext`  
**Key outputs:** `pl_df`; file: `Paper/Results/plackett_luce_ranking_otimizado.xlsx`  
**Dependencies:** S06

---

### S24 — Taylor Diagram [NEW-1]
**Lines:** 1900–1992  
**Responsibility:** Full Taylor Diagram with RMSE arc overlay. Supersedes S10 — saves to same output paths plus additional `Figures/Taylor_Diagram.png`.  
**Key inputs:** `pred_ref`, `y_ref_ext`  
**Key outputs:** `taylor_stats_td`; files: `Figuras_NestedCV/Taylor_Diagram.png`, `Paper/Figures/Taylor_Diagram.png`, `Figures/Taylor_Diagram.png`, `Paper/Results/Taylor_Diagram_Statistics.xlsx`  
**Dependencies:** S06

---

### S25 — Radar Chart [NEW-2]
**Lines:** 1994–2114  
**Responsibility:** Normalize R², RMSE, MAE, MAPE (higher=better direction). Render 3 polar radar variants: all models, annotated ranges, Top 5 by IFI. Compute and export radar score ranking.  
**Key inputs:** `results_df`, `ranking` (from S12)  
**Key outputs:** `radar_norm_df`, `radar_ranking_df`; files: `Figuras_Radar/*.png`, `Figuras_NestedCV/*.png`, `Paper/Figures/*.png`, `Resultados_Artigo/Radar_Chart_*.xlsx`  
**Dependencies:** S06, S12

---

### S26 — Learning Curves [NEW-3]
**Lines:** 2116–2307  
**Responsibility:** `sklearn.learning_curve` (5-fold, 10 sizes) on each model's DEV data. Individual subplots, gap comparison bar chart, best vs worst comparison. Diagnostic table (Underfitting / Overfitting / Optimal / Acceptable).  
**Key inputs:** `models_trained`, `optimized_datasets`, `feature_names`, `target`  
**Key outputs:** `lc_results` (dict), `diag_lc_df`; files: `Figuras_LearningCurves/*.png`, `Paper/Figures/*.png`, `Paper/Results/Learning_Curves_Diagnosis.xlsx`  
**Dependencies:** S05, S06

---

### S27 — Permutation Importance [NEW-4]
**Lines:** 2309–2523  
**Responsibility:** `permutation_importance` (10 repeats, R² scoring) on test set for each model. Heatmap (normalized), top-5 bar plots per model, SHAP vs Permutation comparison (first 4 models), cross-model feature boxplot. Export feature stats.  
**Key inputs:** `models_trained`, `optimized_datasets`, `feature_names`, `target`, `all_shap_importance` (from S15)  
**Key outputs:** `perm_imp_results` (dict), `perm_imp_summary` (list), `feat_stats_perm_df`; files: `Figuras_PermutationImportance/*.png`, `Paper/Results/Permutation_Importance_*.xlsx`  
**Dependencies:** S05, S06, S15  
**Note:** `feat_stats_perm_df` consumed by S28 to select top-3 features for PDP comparison.

---

### S28 — Partial Dependence Plots [NEW-5]
**Lines:** 2525–2691  
**Responsibility:** Refit each model on its DEV data (clone + fit). 1D PDP for best model (6 features), Top-3 models comparison, all-models overlay on top-3 permutation-important features, 2D interaction PDP for best model (feature pairs 0-1 and 2-3).  
**Key inputs:** `models_trained`, `optimized_datasets`, `ranking` (from S12), `feat_stats_perm_df` (from S27), `feature_names`, `target`  
**Key outputs:** `pdp_models` (dict), `pdp_summary_tbl`; files: `Figuras_PDP/*.png`, `Paper/Figures/*.png`, `Paper/Results/PDP_Summary.xlsx`  
**Dependencies:** S05, S06, S12, S27

---

### S29 — Q-Q Plots + Residual Normality [NEW-6]
**Lines:** 2693–2923  
**Responsibility:** On reference residuals: Shapiro-Wilk, Jarque-Bera, Anderson-Darling tests. Q-Q plots (all models, top 3 by IFI), residual histograms with normal curve, residual boxplot, Shapiro p-value bar chart. Classify models by normality status and skewness.  
**Key inputs:** `pred_ref`, `y_ref_ext`, `ranking` (from S12)  
**Key outputs:** `qq_results_dict`, `normality_tests`, `norm_df_qq`; files: `Figuras_QQPlot/*.png`, `Paper/Results/Normality_Tests_Results.xlsx`  
**Dependencies:** S06, S12

---

### S30 — Executive Summary
**Lines:** 2925–2981  
**Responsibility:** Print final summary report: dataset shape, model count, MC runs, CV evaluations, best cleaning method per model, IFI ranking, fix/new-block changelog, output directory inventory.  
**Key inputs:** `df`, `X_full`, `models`, `n_monte_carlo`, `n_mc_opt`, `n_splits`, `n_repeats`, `total_evals`, `best_opt_summary`, `ranking`  
**Key outputs:** Stdout only  
**Dependencies:** All prior sections

---

## Global State Flow (Key Variables)

| Variable | Produced by | Consumed by |
|---|---|---|
| `df`, `X_full`, `y_full`, `feature_names`, `target` | S01 | S04–S30 |
| `models`, `model_list` | S02 | S04, S05, S06, S07, S09 |
| `cleaning_methods` | S03 | S05 |
| `df_results`, `df_vr` | S04 | S20 |
| `best_cleaning_for_model`, `optimized_datasets` | S05 | S06, S07, S08, S09, S26, S27, S28 |
| `y_ref_ext`, `X_tst_ref_sc`, `pred_ref` | S06 | S10, S13, S14, S15, S16, S18, S21, S22, S23, S24, S29 |
| `models_trained`, `scalers` | S06 | S08, S15, S16, S18, S19, S26, S27, S28 |
| `results_df` | S06 | S12, S17, S25 |
| `train_metrics`, `error_variance` | S06 | S11 |
| `ranking` | S12 | S25, S26, S28, S29 |
| `all_shap_importance`, `cons_shap` | S15 | S27 |
| `feat_stats_perm_df` | S27 | S28 |
| `lc_results` | S26 | (self-contained) |
| `perm_imp_results` | S27 | (self-contained after S28) |
| `qq_results_dict` | S29 | (self-contained) |

---

## Proposed Module Map

| Section(s) | New File | Responsibility |
|---|---|---|
| S00 | `src/fck_prediction/config.py` | Constants, directory list, random seed, output path helpers |
| S01 | `src/fck_prediction/data/loader.py` | Load Excel, rename columns, coerce types, drop NaN; return `df, X_full, y_full, feature_names` |
| S02 | `src/fck_prediction/models/registry.py` | Return `models` dict and `model_list`; all estimator instantiation |
| S03 | `src/fck_prediction/preprocessing/cleaners.py` | Cleaning functions + `cleaning_methods` registry |
| S04 | `src/fck_prediction/evaluation/monte_carlo.py` | `run_monte_carlo(X, y, models, n_runs)` → df_results, df_vr; boxplot |
| S05 | `src/fck_prediction/preprocessing/cleaning_optimizer.py` | `optimize_cleaning(X_full, y_full, models, cleaning_methods)` → best_cleaning_for_model, optimized_datasets |
| S06 | `src/fck_prediction/training/trainer.py` | Build reference partition; train all models on optimized DEV; return models_trained, scalers, pred_ref, pred_ext, results_df, y_ref_ext, X_tst_ref_sc |
| S07 | `src/fck_prediction/evaluation/monte_carlo.py` | Add `run_monte_carlo_optimized(optimized_datasets, models)` → df_res_opt, df_vr_opt |
| S08 | `src/fck_prediction/evaluation/picp.py` | `compute_picp(models_trained, optimized_datasets)` → picp_df; heatmap + calibration plots |
| S09 | `src/fck_prediction/evaluation/cross_validation.py` | `run_repeated_kfold(optimized_datasets, models, n_splits, n_repeats)` → repeated_cv_df, cv_summary_df; all CV plots |
| S10, S24 | `src/fck_prediction/visualization/taylor_diagram.py` | `plot_taylor_diagram(pred_ref, y_ref_ext)` → figure + stats table (single implementation, called twice) |
| S11 | `src/fck_prediction/visualization/performance_plots.py` | Boxplots, violin plots, error variance plots from metrics_df |
| S12 | `src/fck_prediction/evaluation/ifi.py` | `compute_ifi(results_df)` → ranking DataFrame; sensitivity plots |
| S13 | `src/fck_prediction/visualization/correlation.py` | Pearson correlation heatmap of model predictions |
| S14, S21 | `src/fck_prediction/evaluation/statistical_tests.py` | Diebold-Mariano pairwise test; DM heatmap figure; tabular output |
| S15 | `src/fck_prediction/interpretation/shap_analysis.py` | SHAP explainer selection + bar/summary/dependence plots; consolidated exports |
| S16 | `src/fck_prediction/visualization/prediction_plots.py` | Parity plots, residual scatter, KDE density, MAPE scatter |
| S17 | `src/fck_prediction/evaluation/summary_stats.py` | Descriptive stats table from results_df |
| S18 | `src/fck_prediction/evaluation/residual_diagnostics.py` | Econometric tests (normality, heteroscedasticity, autocorrelation); ACF/PACF plots |
| S19 | `src/fck_prediction/inference/predictor.py` | `predict_new_mixes(models_trained, scalers, new_data)` → predictions DataFrame |
| S20 | `src/fck_prediction/evaluation/statistical_tests.py` | `run_friedman_nemenyi(df_results)` → CD diagram (add to same file as DM) |
| S22 | `src/fck_prediction/evaluation/model_confidence_set.py` | `compute_mcs(pred_ref, y_ref_ext)` → mcs_res + bar chart |
| S23 | `src/fck_prediction/evaluation/statistical_tests.py` | `plackett_luce_ranking(pred_ref, y_ref_ext)` → pl_df |
| S25 | `src/fck_prediction/visualization/radar_chart.py` | `plot_radar(results_df, ranking)` → 3 polar chart variants + ranking table |
| S26 | `src/fck_prediction/evaluation/learning_curves.py` | `compute_learning_curves(models_trained, optimized_datasets)` → lc_results, diag_lc_df; 3 figure types |
| S27 | `src/fck_prediction/interpretation/permutation_importance.py` | `compute_permutation_importance(models_trained, optimized_datasets, shap_importance)` → perm_imp_results, feat_stats_perm_df; 4 figure types |
| S28 | `src/fck_prediction/interpretation/pdp.py` | `compute_pdp(pdp_models, ranking, feat_stats_perm_df)` → 1D and 2D PDP figures |
| S29 | `src/fck_prediction/evaluation/normality.py` | Q-Q plots, histograms, boxplot, Shapiro-Wilk comparison; normality classification |
| S30 | `src/fck_prediction/cli.py` | Entry point; orchestrates all modules in order; prints executive summary |

---

## Suggested Package Structure

```
src/fck_prediction/
├── config.py                        # S00
├── cli.py                           # S30 + orchestration
├── data/
│   └── loader.py                    # S01
├── models/
│   └── registry.py                  # S02
├── preprocessing/
│   ├── cleaners.py                  # S03
│   └── cleaning_optimizer.py        # S05
├── training/
│   └── trainer.py                   # S06
├── evaluation/
│   ├── monte_carlo.py               # S04, S07
│   ├── picp.py                      # S08
│   ├── cross_validation.py          # S09
│   ├── ifi.py                       # S12
│   ├── statistical_tests.py         # S14, S20, S21, S23
│   ├── model_confidence_set.py      # S22
│   ├── learning_curves.py           # S26
│   ├── normality.py                 # S29
│   ├── residual_diagnostics.py      # S18
│   └── summary_stats.py             # S17
├── interpretation/
│   ├── shap_analysis.py             # S15
│   ├── permutation_importance.py    # S27
│   └── pdp.py                       # S28
├── visualization/
│   ├── performance_plots.py         # S11
│   ├── prediction_plots.py          # S16
│   ├── correlation.py               # S13
│   ├── taylor_diagram.py            # S10, S24
│   └── radar_chart.py               # S25
└── inference/
    └── predictor.py                 # S19
```
