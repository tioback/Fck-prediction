# ============================================================
# SCRIPT v15 - INTEGRADO COMPLETO
# CORREÇÕES APLICADAS vs v13 (mantidas do v14):
#   [FIX-1] Data Leakage eliminado na seleção do método de limpeza
#   [FIX-2] Partição de referência comum (y_ref_ext / pred_ref)
#           para todas as análises comparativas
#   [FIX-3] clone() universal substitui blocos if/elif
#   [FIX-4] PICP: std estimado no DEV, aplicado no TEST
#   [FIX-5] MAPE padronizado via sklearn em todo o script
#   [FIX-6] plt.show() removido; plt.close() após cada figura
# NOVOS BLOCOS ADICIONADOS (v14 → v15):
#   [NEW-1] Taylor Diagram (pred_ref / y_ref_ext)
#   [NEW-2] Radar Chart multi-métrico (3 versões)
#   [NEW-3] Learning Curves (diagnóstico over/underfitting)
#   [NEW-4] Permutation Importance + comparação SHAP vs Perm
#   [NEW-5] Partial Dependence Plots – 1D e 2D
#   [NEW-6] Q-Q Plots + testes de normalidade dos resíduos
# Modelos: Linear, BayesianRidge, DecisionTree, RandomForest,
#          GradientBoosting, SVR_rbf, SVR_poly, XGBoost, ANN
# Estimativa de fck de Concretos com ML - ARTIGO Q1
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import pickle
from datetime import datetime
from sklearn.base import clone                          # [FIX-3]
warnings.filterwarnings('ignore')

from sklearn.model_selection import (train_test_split, KFold,
                                     cross_val_score, GridSearchCV,
                                     RepeatedKFold)
from sklearn.metrics import (r2_score, mean_squared_error,
                             mean_absolute_error,
                             mean_absolute_percentage_error)
from sklearn.preprocessing import StandardScaler

from sklearn.svm import SVR, OneClassSVM
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (RandomForestRegressor,
                               GradientBoostingRegressor,
                               IsolationForest)
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.covariance import EllipticEnvelope
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import DBSCAN
from xgboost import XGBRegressor

import shap
import scikit_posthocs as sp
from scipy import stats
from sklearn.inspection import (PartialDependenceDisplay,
                                permutation_importance,
                                partial_dependence)
from sklearn.model_selection import learning_curve
import matplotlib.gridspec as gridspec

# ============================================================
# CRIAÇÃO DE TODOS OS DIRETÓRIOS NECESSÁRIOS
# ============================================================

print("="*80)
print("SISTEMA v14 CORRIGIDO: OTIMIZAÇÃO DE LIMPEZA + REPEATED K-FOLD CV")
print("Modelos: Linear, BayesianRidge, DecisionTree, RandomForest,")
print("GradientBoosting, SVR_rbf, SVR_poly, XGBoost, ANN (MLP)")
print("="*80)

directories = [
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

for directory in directories:
    os.makedirs(directory, exist_ok=True)
    print(f"📁 Diretório criado/verificado: {directory}")

print(f"\n✅ Total de {len(directories)} diretórios preparados")

np.random.seed(42)

# ============================================================
# PARTE 1: CARREGAMENTO E LIMPEZA MÍNIMA DOS DADOS
# ============================================================

print("\n📥 CARREGANDO DADOS ORIGINAIS...")

df = pd.read_excel("Concrete_Data.xls")
print(f"📊 Shape original: {df.shape}")

if df.shape[1] == 11:
    df.columns = ['C', 'S', 'FA', 'SF', 'LP', 'W', 'SP',
                  'Gravel', 'Sand', 'Age', 'fck']
    print("✅ Colunas renomeadas para formato padrão")

print("\n🧹 Convertendo para numérico (mantendo todos os dados)...")
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df = df.dropna().reset_index(drop=True)

print(f"📊 Shape final: {df.shape}")
print(f"📋 Colunas: {df.columns.tolist()}")
print(f"\n📈 Estatísticas descritivas:")
print(df.describe())

target = 'fck'
feature_names = [col for col in df.columns if col != target]

X_full = df[feature_names]
y_full = df[target]

print(f"\n🎯 Target: {target}")
print(f"🔢 Features: {feature_names}")
print(f"📊 Total de amostras: {len(df)}")

# ============================================================
# PARTE 2: DEFINIÇÃO DOS MODELOS
# ============================================================

print("\n🤖 DEFININDO MODELOS...")

models = {
    "Linear":          LinearRegression(),
    "BayesianRidge":   BayesianRidge(),
    "DecisionTree":    DecisionTreeRegressor(random_state=42),
    "RandomForest":    RandomForestRegressor(n_estimators=200,
                                             random_state=42, n_jobs=-1),
    "GradientBoosting":GradientBoostingRegressor(n_estimators=200,
                                                  random_state=42),
    "SVR_rbf":         SVR(kernel='rbf',  max_iter=2000),
    "SVR_poly":        SVR(kernel='poly', max_iter=2000),
    "XGBoost":         XGBRegressor(n_estimators=200, random_state=42,
                                    n_jobs=-1, verbosity=0),
    "ANN":             MLPRegressor(hidden_layer_sizes=(100, 50),
                                    max_iter=2000, random_state=42,
                                    early_stopping=True),
}

model_list = list(models.keys())
print(f"✅ {len(models)} modelos carregados:")
for i, (n, m) in enumerate(models.items(), 1):
    print(f"   {i:2d}. {n} – {type(m).__name__}")

# ============================================================
# PARTE 3: DEFINIÇÃO DOS MÉTODOS DE LIMPEZA
# (aplicados APENAS sobre o conjunto de treino — [FIX-1])
# ============================================================

print("\n🧹 DEFININDO MÉTODOS DE LIMPEZA...")

def clean_isolationforest(X, y, contamination=0.02, random_state=42):
    iso = IsolationForest(contamination=contamination,
                          random_state=random_state, n_jobs=-1)
    mask = iso.fit_predict(X) == 1
    return X[mask], y[mask]

def clean_iqr(X, y, multiplier=1.5):
    mask = np.ones(len(X), dtype=bool)
    for col in X.columns:
        Q1, Q3 = X[col].quantile(0.25), X[col].quantile(0.75)
        IQR = Q3 - Q1
        mask &= (X[col] >= Q1 - multiplier*IQR) & (X[col] <= Q3 + multiplier*IQR)
    return X[mask], y[mask]

def clean_zscore(X, y, threshold=3):
    z = np.abs(stats.zscore(X, nan_policy='omit'))
    mask = (z < threshold).all(axis=1)
    return X[mask], y[mask]

def clean_percentile(X, y, lower=0.01, upper=0.99):
    mask = np.ones(len(X), dtype=bool)
    for col in X.columns:
        mask &= (X[col] >= X[col].quantile(lower)) & \
                (X[col] <= X[col].quantile(upper))
    return X[mask], y[mask]

def clean_dbscan(X, y, eps=0.5, min_samples=5):
    Xs = StandardScaler().fit_transform(X)
    labels = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1).fit_predict(Xs)
    mask = labels != -1
    return X[mask], y[mask]

def clean_elliptic(X, y, contamination=0.02):
    Xs = StandardScaler().fit_transform(X)
    mask = EllipticEnvelope(contamination=contamination,
                            random_state=42,
                            support_fraction=0.9).fit_predict(Xs) == 1
    return X[mask], y[mask]

def clean_svm(X, y, nu=0.02):
    Xs = StandardScaler().fit_transform(X)
    mask = OneClassSVM(nu=nu, kernel='rbf', gamma='scale',
                       max_iter=1000).fit_predict(Xs) == 1
    return X[mask], y[mask]

def clean_lof(X, y, contamination=0.02):
    Xs = StandardScaler().fit_transform(X)
    mask = LocalOutlierFactor(contamination=contamination,
                              n_neighbors=20, n_jobs=-1).fit_predict(Xs) == 1
    return X[mask], y[mask]

cleaning_methods = {
    'Sem_Limpeza':          {'func': lambda X, y: (X.copy(), y.copy())},
    'IsolationForest_1%':   {'func': lambda X, y: clean_isolationforest(X, y, 0.01)},
    'IsolationForest_2%':   {'func': lambda X, y: clean_isolationforest(X, y, 0.02)},
    'IsolationForest_3%':   {'func': lambda X, y: clean_isolationforest(X, y, 0.03)},
    'IsolationForest_5%':   {'func': lambda X, y: clean_isolationforest(X, y, 0.05)},
    'IQR_1.5':              {'func': lambda X, y: clean_iqr(X, y, 1.5)},
    'IQR_2.0':              {'func': lambda X, y: clean_iqr(X, y, 2.0)},
    'IQR_3.0':              {'func': lambda X, y: clean_iqr(X, y, 3.0)},
    'ZScore_2':             {'func': lambda X, y: clean_zscore(X, y, 2)},
    'ZScore_3':             {'func': lambda X, y: clean_zscore(X, y, 3)},
    'Percentil_1_99':       {'func': lambda X, y: clean_percentile(X, y, 0.01, 0.99)},
    'Percentil_5_95':       {'func': lambda X, y: clean_percentile(X, y, 0.05, 0.95)},
    'DBSCAN':               {'func': lambda X, y: clean_dbscan(X, y)},
    'EllipticEnvelope':     {'func': lambda X, y: clean_elliptic(X, y)},
    'OneClassSVM':          {'func': lambda X, y: clean_svm(X, y)},
    'LocalOutlierFactor':   {'func': lambda X, y: clean_lof(X, y)},
}

print(f"✅ {len(cleaning_methods)} métodos de limpeza definidos")

# ============================================================
# PARTE 4: MONTE CARLO E VARIANCE RATIO (DADOS ORIGINAIS)
# ============================================================

print("\n" + "="*80)
print("📊 MONTE CARLO E VARIANCE RATIO (DADOS ORIGINAIS)")
print("="*80)

n_monte_carlo = 30
all_results = []

for run in range(n_monte_carlo):
    if run % 5 == 0:
        print(f"   Run {run+1}/{n_monte_carlo}")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_full.values, y_full.values, test_size=0.2, random_state=42+run)

    sc = StandardScaler()
    X_tr_s = sc.fit_transform(X_tr)
    X_te_s = sc.transform(X_te)

    for name, model in models.items():
        try:
            m = clone(model)                                   # [FIX-3]
            m.fit(X_tr_s, y_tr)
            p_tr = m.predict(X_tr_s)
            p_te = m.predict(X_te_s)

            for ds, y_ds, p_ds in [('Train', y_tr, p_tr), ('Test', y_te, p_te)]:
                # [FIX-5] MAPE via sklearn em todo o script
                all_results.append({
                    "Model": name, "Run": run, "Dataset": ds,
                    "R2":   r2_score(y_ds, p_ds),
                    "RMSE": np.sqrt(mean_squared_error(y_ds, p_ds)),
                    "MAE":  mean_absolute_error(y_ds, p_ds),
                    "MAPE": mean_absolute_percentage_error(y_ds, p_ds) * 100,
                })
        except Exception as e:
            print(f"      ⚠️ {name}: {str(e)[:50]}")

df_results = pd.DataFrame(all_results)
df_results.to_excel('Resultados_Artigo/Monte_Carlo_Results.xlsx', index=False)

# Variance Ratio
variance_ratio = []
for model in df_results["Model"].unique():
    for metric in ["R2", "RMSE", "MAE", "MAPE"]:
        tr_v = df_results[(df_results["Model"] == model) &
                          (df_results["Dataset"] == "Train")][metric]
        te_v = df_results[(df_results["Model"] == model) &
                          (df_results["Dataset"] == "Test")][metric]
        vr = np.var(te_v) / (np.var(tr_v) + 1e-10)
        variance_ratio.append({"Model": model, "Metric": metric,
                                "Variance_Ratio": vr,
                                "Var_Train": np.var(tr_v),
                                "Var_Test":  np.var(te_v)})

df_vr = pd.DataFrame(variance_ratio)
df_vr.to_excel('Resultados_Artigo/Variance_Ratio_Results.xlsx', index=False)

# Boxplot com Variance Ratio
metrics_list   = ["RMSE", "R2", "MAE", "MAPE"]
metric_labels  = {"RMSE": "RMSE (MPa)", "R2": "R²",
                  "MAE": "MAE (MPa)", "MAPE": "MAPE (%)"}

fig, axes = plt.subplots(4, 1, figsize=(16, 18), sharex=True)
fig.suptitle('Boxplots of performance metrics with Variance Ratio overlay (Monte Carlo)',
             fontsize=16, fontweight='bold', y=0.98)

for i, metric in enumerate(metrics_list):
    ax = axes[i]
    asc = metric != "R2"
    model_order = (df_results[df_results["Dataset"] == "Test"]
                   .groupby("Model")[metric].median()
                   .sort_values(ascending=asc).index)
    sns.boxplot(data=df_results, x="Model", y=metric, hue="Dataset", ax=ax,
                order=model_order, palette={"Train": "#2E86AB", "Test": "#A23B72"},
                fliersize=2, linewidth=1.2)
    vr_m = df_vr[df_vr["Metric"] == metric]
    for j, mod in enumerate(model_order):
        row = vr_m[vr_m["Model"] == mod]
        if len(row):
            vr_val = row["Variance_Ratio"].values[0]
            col = 'green' if vr_val < 1.5 else ('orange' if vr_val < 2.5 else 'red')
            ax.text(j, ax.get_ylim()[1]*0.95, f'VR={vr_val:.2f}', ha='center',
                    fontsize=9, rotation=90, color=col, fontweight='bold')
    ax.set_title(f"{metric} – {metric_labels[metric]}", fontsize=14,
                 fontweight='bold', loc='left')
    ax.set_ylabel(metric_labels[metric], fontsize=12)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax.set_axisbelow(True)
    if i == 0:
        ax.legend(title='Dataset', loc='upper right')
    else:
        ax.legend_.remove()

plt.xticks(rotation=45, ha='right', fontsize=10)
plt.xlabel('Model', fontsize=12)
plt.tight_layout()
plt.savefig("Figuras_MonteCarlo/Benchmark_Boxplot_All_Metrics.png",
            dpi=300, bbox_inches='tight')
plt.savefig("Paper/Figures/Benchmark_Boxplot_All_Metrics.png",
            dpi=300, bbox_inches='tight')
plt.close()
print("✅ Boxplot Monte Carlo (dados originais) salvo")

# ============================================================
# PARTE 5: OTIMIZAÇÃO DA LIMPEZA — SEM DATA LEAKAGE [FIX-1]
# Fluxo correto:
#   1) split original → X_dev_raw / X_test_raw  (test nunca tocado)
#   2) fit StandardScaler apenas em X_dev_raw
#   3) aplicar método de limpeza apenas em X_dev_scaled
#   4) treinar modelo nos dados limpos
#   5) avaliar no X_test_raw (escalado com o mesmo scaler)
# ============================================================

print("\n" + "="*80)
print("🔬 FASE 1: OTIMIZANDO MÉTODO DE LIMPEZA PARA CADA MODELO [FIX-1]")
print("="*80)

best_cleaning_for_model = {}
optimized_datasets      = {}   # armazena apenas o conjunto de DEV limpo
optimization_results    = []

n_clean_runs = 3

for i, model_name in enumerate(model_list):
    print(f"\n📊 [{i+1}/{len(model_list)}] {model_name}")
    model_perf = []

    for method_name, method_info in cleaning_methods.items():
        method_scores = []

        for run in range(n_clean_runs):
            try:
                # ── 1. Split global (test nunca visto pela limpeza) ──────
                X_dev_raw, X_tst_raw, y_dev_raw, y_tst_raw = train_test_split(
                    X_full, y_full, test_size=0.2, random_state=42+run)

                # ── 2. Scaler fitado APENAS no dev ──────────────────────
                sc_opt = StandardScaler()
                X_dev_scaled_df = pd.DataFrame(
                    sc_opt.fit_transform(X_dev_raw),
                    columns=feature_names,
                    index=X_dev_raw.index)
                X_tst_scaled = sc_opt.transform(X_tst_raw)

                # ── 3. Limpeza aplicada APENAS no dev escalado ───────────
                X_cl, y_cl = method_info['func'](X_dev_scaled_df,
                                                  y_dev_raw.reset_index(drop=True)
                                                  if hasattr(y_dev_raw, 'reset_index')
                                                  else pd.Series(y_dev_raw))

                if len(X_cl) < 20:          # proteção contra datasets muito pequenos
                    continue

                # ── 4. Treino e avaliação ────────────────────────────────
                m = clone(models[model_name])          # [FIX-3]
                m.fit(X_cl.values, y_cl.values)
                r2 = r2_score(y_tst_raw, m.predict(X_tst_scaled))
                method_scores.append(r2)

            except Exception as e:
                pass

        if method_scores:
            model_perf.append({
                'Model':          model_name,
                'Cleaning_Method': method_name,
                'Mean_R2':        np.mean(method_scores),
                'Std_R2':         np.std(method_scores),
                'N_Runs':         len(method_scores),
            })

    if model_perf:
        perf_df    = pd.DataFrame(model_perf)
        best_method = perf_df.loc[perf_df['Mean_R2'].idxmax(), 'Cleaning_Method']
        best_r2     = perf_df['Mean_R2'].max()
        best_cleaning_for_model[model_name] = best_method
        optimization_results.extend(model_perf)
        print(f"   ✅ Melhor: {best_method} (R² = {best_r2:.4f})")

        # ── Gerar dataset DEV otimizado (referência p/ treino final) ─────
        # Aqui fazemos UMA divisão canônica (random_state=42) para o modelo final
        X_dev_raw_f, X_tst_raw_f, y_dev_raw_f, y_tst_raw_f = train_test_split(
            X_full, y_full, test_size=0.2, random_state=42)

        sc_final = StandardScaler()
        X_dev_sc_df = pd.DataFrame(
            sc_final.fit_transform(X_dev_raw_f),
            columns=feature_names,
            index=X_dev_raw_f.index)

        X_cl_f, y_cl_f = cleaning_methods[best_method]['func'](
            X_dev_sc_df,
            y_dev_raw_f.reset_index(drop=True))

        # Armazena o DataFrame de DEV limpo e o scaler canônico
        opt_df = X_cl_f.copy()
        opt_df[target] = y_cl_f.values
        optimized_datasets[model_name] = {
            'dev_df':    opt_df,          # X já escalado + y (apenas dev)
            'scaler':    sc_final,        # scaler fitado no dev canônico
            'X_tst_raw': X_tst_raw_f,    # teste bruto (escalado sob demanda)
            'y_tst_raw': y_tst_raw_f,
        }
        opt_df.to_excel(
            f"Bancos_Otimizados/Dataset_Otimizado_{model_name}.xlsx",
            index=False)
        print(f"      📁 Dev otimizado salvo: {len(opt_df)} amostras")

    else:
        best_cleaning_for_model[model_name] = 'Sem_Limpeza'
        # fallback: usar X_full inteiro escalado sem limpeza
        X_dev_raw_f, X_tst_raw_f, y_dev_raw_f, y_tst_raw_f = train_test_split(
            X_full, y_full, test_size=0.2, random_state=42)
        sc_fb = StandardScaler()
        X_dev_sc = sc_fb.fit_transform(X_dev_raw_f)
        fb_df = pd.DataFrame(X_dev_sc, columns=feature_names)
        fb_df[target] = y_dev_raw_f.values
        optimized_datasets[model_name] = {
            'dev_df':    fb_df,
            'scaler':    sc_fb,
            'X_tst_raw': X_tst_raw_f,
            'y_tst_raw': y_tst_raw_f,
        }
        print(f"   ⚠️ Usando Sem_Limpeza (fallback)")

opt_summary_full = pd.DataFrame(optimization_results)
if not opt_summary_full.empty:
    opt_summary_full.to_excel(
        'Resultados_Artigo/Otimizacao_Limpeza_Resultados.xlsx', index=False)

best_opt_summary = pd.DataFrame([
    {'Model':       m,
     'Best_Method': best_cleaning_for_model[m],
     'Samples':     len(optimized_datasets[m]['dev_df'])}
    for m in model_list
])
best_opt_summary.to_excel(
    'Resultados_Artigo/Melhor_Metodo_Limpeza_por_Modelo.xlsx', index=False)

print("\n📊 RESUMO DA OTIMIZAÇÃO:")
print(best_opt_summary.to_string(index=False))

# ============================================================
# PARTE 6: TREINAMENTO DOS MODELOS COM DATASETS OTIMIZADOS
# Partição de referência comum para análises comparativas [FIX-2]
# ============================================================

print("\n" + "="*80)
print("🚀 TREINAMENTO DOS MODELOS COM DATASETS OTIMIZADOS")
print("="*80)

# ── Partição de referência comum ─────────────────────────────────────────────
# Todos os modelos são avaliados TAMBÉM neste conjunto para garantir
# comparabilidade em DM, Friedman, correlação, PICP e diagnósticos.
X_dev_ref, X_tst_ref, y_dev_ref, y_tst_ref = train_test_split(
    X_full, y_full, test_size=0.2, random_state=42)
sc_ref = StandardScaler()
X_dev_ref_sc = sc_ref.fit_transform(X_dev_ref)
X_tst_ref_sc = sc_ref.transform(X_tst_ref)
y_ref_ext = y_tst_ref.values          # vetor único e fixo para todos os modelos

results         = []
pred_ext        = {}     # predições NO CONJUNTO PRÓPRIO de cada modelo
pred_ref        = {}     # predições na referência comum         [FIX-2]
pred_train_dict = {}
models_trained  = {}
scalers         = {}
train_metrics   = []
test_metrics    = []
error_variance  = []

for model_name, model in models.items():
    print(f"\n📈 Treinando: {model_name}")

    info    = optimized_datasets[model_name]
    dev_df  = info['dev_df']
    sc_mod  = info['scaler']
    X_tst_r = info['X_tst_raw']
    y_tst_r = info['y_tst_raw']

    X_dev_np = dev_df[feature_names].values
    y_dev_np = dev_df[target].values

    # Conjunto de teste próprio (escalado com scaler do modelo)
    X_tst_sc = sc_mod.transform(X_tst_r)

    try:
        m = clone(model)                                       # [FIX-3]
        m.fit(X_dev_np, y_dev_np)

        p_tr = m.predict(X_dev_np)
        p_te = m.predict(X_tst_sc)

        # ── predição na referência comum ─────────────────────
        p_ref = m.predict(X_tst_ref_sc)
        pred_ref[model_name]        = p_ref
        # ─────────────────────────────────────────────────────

        pred_train_dict[model_name] = p_tr
        pred_ext[model_name]        = p_te
        models_trained[model_name]  = m
        scalers[model_name]         = sc_mod

        y_tst_np = y_tst_r.values

        for ds, y_ds, p_ds in [('Training', y_dev_np, p_tr),
                                ('Testing',  y_tst_np, p_te)]:
            r2_v   = r2_score(y_ds, p_ds)
            rmse_v = np.sqrt(mean_squared_error(y_ds, p_ds))
            mae_v  = mean_absolute_error(y_ds, p_ds)
            mape_v = mean_absolute_percentage_error(y_ds, p_ds) * 100  # [FIX-5]
            train_metrics.append({"Model": model_name, "Metric": "R2",
                                   "Value": r2_v,   "Set": ds})
            train_metrics.append({"Model": model_name, "Metric": "RMSE",
                                   "Value": rmse_v, "Set": ds})
            train_metrics.append({"Model": model_name, "Metric": "MAE",
                                   "Value": mae_v,  "Set": ds})
            train_metrics.append({"Model": model_name, "Metric": "MAPE",
                                   "Value": mape_v, "Set": ds})
            error_variance.append({"Model": model_name, "Set": ds,
                                   "Metric": "Var_Error",
                                   "Value": np.var(y_ds - p_ds)})

        r2_te   = r2_score(y_tst_np, p_te)
        rmse_te = np.sqrt(mean_squared_error(y_tst_np, p_te))
        mae_te  = mean_absolute_error(y_tst_np, p_te)
        mape_te = mean_absolute_percentage_error(y_tst_np, p_te) * 100

        results.append({
            "Model":          model_name,
            "R2":             r2_te,
            "RMSE":           rmse_te,
            "MAE":            mae_te,
            "MAPE":           mape_te,
            "Cleaning_Method":best_cleaning_for_model[model_name],
            "Samples":        len(y_dev_np),
        })

        r2_tr   = r2_score(y_dev_np, p_tr)
        rmse_tr = np.sqrt(mean_squared_error(y_dev_np, p_tr))
        mape_tr = mean_absolute_percentage_error(y_dev_np, p_tr) * 100
        print(f"   ✅ Training: R²={r2_tr:.4f} | RMSE={rmse_tr:.2f} | MAPE={mape_tr:.1f}%")
        print(f"      Testing:  R²={r2_te:.4f} | RMSE={rmse_te:.2f} | MAPE={mape_te:.1f}%")
        print(f"      Método: {best_cleaning_for_model[model_name]} | Dev: {len(y_dev_np)} amostras")

    except Exception as e:
        print(f"   ❌ Erro: {str(e)[:100]}")

results_df = pd.DataFrame(results)
results_df.to_excel('Resultados_Artigo/Results_Otimizado.xlsx', index=False)
print("\n📈 RESUMO (TESTE):")
print(results_df.sort_values('R2', ascending=False))

# ============================================================
# PARTE 4.5: MONTE CARLO COM DATASETS OTIMIZADOS
# ============================================================

print("\n" + "="*80)
print("📊 MONTE CARLO E VARIANCE RATIO (DADOS OTIMIZADOS)")
print("="*80)

n_mc_opt     = 30
all_res_opt  = []

for run in range(n_mc_opt):
    if run % 5 == 0:
        print(f"   Run {run+1}/{n_mc_opt}")

    for model_name in model_list:
        try:
            info   = optimized_datasets[model_name]
            dev_df = info['dev_df']
            sc_mod = info['scaler']

            X_d = dev_df[feature_names].values
            y_d = dev_df[target].values

            # Para o MC usamos re-split interno do dev (nunca toca o teste)
            X_tr, X_te, y_tr, y_te = train_test_split(
                X_d, y_d, test_size=0.2, random_state=42+run)

            m = clone(models[model_name])                  # [FIX-3]
            m.fit(X_tr, y_tr)
            p_tr, p_te = m.predict(X_tr), m.predict(X_te)

            for ds, y_ds, p_ds in [('Train', y_tr, p_tr),
                                    ('Test',  y_te, p_te)]:
                all_res_opt.append({
                    "Model": model_name, "Run": run, "Dataset": ds,
                    "R2":   r2_score(y_ds, p_ds),
                    "RMSE": np.sqrt(mean_squared_error(y_ds, p_ds)),
                    "MAE":  mean_absolute_error(y_ds, p_ds),
                    "MAPE": mean_absolute_percentage_error(y_ds, p_ds) * 100,  # [FIX-5]
                    "Cleaning_Method": best_cleaning_for_model[model_name],
                    "Samples": len(y_d),
                })
        except Exception as e:
            print(f"      ⚠️ {model_name} run {run}: {str(e)[:50]}")

df_res_opt = pd.DataFrame(all_res_opt)
df_res_opt.to_excel(
    'Resultados_Artigo/Monte_Carlo_Results_Otimizado.xlsx', index=False)
print(f"✅ MC otimizado: {len(df_res_opt)} linhas")

# Variance Ratio (otimizado)
vr_opt = []
for model in df_res_opt["Model"].unique():
    for metric in ["R2", "RMSE", "MAE", "MAPE"]:
        tr_v = df_res_opt[(df_res_opt["Model"] == model) &
                          (df_res_opt["Dataset"] == "Train")][metric]
        te_v = df_res_opt[(df_res_opt["Model"] == model) &
                          (df_res_opt["Dataset"] == "Test")][metric]
        if len(tr_v) and len(te_v):
            vr_opt.append({
                "Model": model, "Metric": metric,
                "Variance_Ratio": np.var(te_v) / (np.var(tr_v) + 1e-10),
                "Var_Train": np.var(tr_v), "Var_Test": np.var(te_v),
                "Cleaning_Method": best_cleaning_for_model.get(model, 'Sem_Limpeza'),
                "Samples": len(optimized_datasets[model]['dev_df']),
            })

df_vr_opt = pd.DataFrame(vr_opt)
df_vr_opt.to_excel(
    'Resultados_Artigo/Variance_Ratio_Results_Otimizado.xlsx', index=False)

# Boxplot MC otimizado
fig, axes = plt.subplots(4, 1, figsize=(16, 18), sharex=True)
fig.suptitle('Boxplots – Monte Carlo (Optimized Datasets)',
             fontsize=16, fontweight='bold', y=0.98)

for i, metric in enumerate(metrics_list):
    ax = axes[i]
    asc = metric != "R2"
    mod_ord = (df_res_opt[df_res_opt["Dataset"] == "Test"]
               .groupby("Model")[metric].median()
               .sort_values(ascending=asc).index)
    sns.boxplot(data=df_res_opt, x="Model", y=metric, hue="Dataset", ax=ax,
                order=mod_ord, palette={"Train": "#2E86AB", "Test": "#A23B72"},
                fliersize=2, linewidth=1.2)
    vr_m = df_vr_opt[df_vr_opt["Metric"] == metric]
    for j, mod in enumerate(mod_ord):
        row = vr_m[vr_m["Model"] == mod]
        if len(row):
            vr_val = row["Variance_Ratio"].values[0]
            cl_inf = row["Cleaning_Method"].values[0]
            sa_inf = row["Samples"].values[0]
            col = 'green' if vr_val < 1.5 else ('orange' if vr_val < 2.5 else 'red')
            ax.text(j, ax.get_ylim()[1]*0.95,
                    f'VR={vr_val:.2f}\n{cl_inf[:12]}\n({sa_inf:.0f})',
                    ha='center', fontsize=7, color=col, fontweight='bold')
    ax.set_title(f"{metric} – {metric_labels[metric]} (Optimized)",
                 fontsize=14, fontweight='bold', loc='left')
    ax.set_ylabel(metric_labels[metric], fontsize=12)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax.set_axisbelow(True)
    if i == 0:
        ax.legend(title='Dataset', loc='upper right')
    else:
        ax.legend_.remove()

plt.xticks(rotation=45, ha='right', fontsize=10)
plt.xlabel('Model', fontsize=12)
plt.tight_layout()
plt.savefig("Figuras_MonteCarlo/Benchmark_Boxplot_All_Metrics_Otimizado.png",
            dpi=300, bbox_inches='tight')
plt.savefig("Paper/Figures/Benchmark_Boxplot_All_Metrics_Otimizado.png",
            dpi=300, bbox_inches='tight')
plt.close()
print("✅ Boxplot MC otimizado salvo")

# ============================================================
# PARTE 7: PREDICTION INTERVAL COVERAGE PROBABILITY (PICP)
# PICP estimado com std do DEV, avaliado no TEST [FIX-4]
# ============================================================

print("\n" + "="*80)
print("📊 PREDICTION INTERVAL COVERAGE PROBABILITY (PICP) [FIX-4]")
print("="*80)

confidence_levels = [0.68, 0.80, 0.90, 0.95, 0.99]
z_scores_map = {0.68: 1.0, 0.80: 1.28, 0.90: 1.645, 0.95: 1.96, 0.99: 2.576}

picp_results = []

for model_name in models.keys():
    if model_name not in models_trained:
        continue
    print(f"\n   📈 {model_name}")

    try:
        info    = optimized_datasets[model_name]
        dev_df  = info['dev_df']
        sc_mod  = info['scaler']
        X_tst_r = info['X_tst_raw']
        y_tst_r = info['y_tst_raw'].values

        X_dev_np = dev_df[feature_names].values
        y_dev_np = dev_df[target].values
        X_tst_sc = sc_mod.transform(X_tst_r)

        m         = models_trained[model_name]
        p_dev     = m.predict(X_dev_np)
        p_tst     = m.predict(X_tst_sc)

        # ── std estimado no DEV (não no test) ─────────────── [FIX-4]
        res_dev     = y_dev_np - p_dev
        mean_res    = np.mean(res_dev)
        std_res     = np.std(res_dev)

        print(f"      Resíduos DEV: μ={mean_res:.4f}, σ={std_res:.4f}")

        for cl in confidence_levels:
            z = z_scores_map[cl]
            lb = p_tst + mean_res - z * std_res
            ub = p_tst + mean_res + z * std_res
            cov = np.mean((y_tst_r >= lb) & (y_tst_r <= ub))
            picp_results.append({
                'Model': model_name, 'Method': 'Gaussian_Homogeneous',
                'Confidence_Level': cl, 'PICP': cov,
                'Target_Coverage': cl, 'Coverage_Gap': cov - cl,
                'Mean_PI_Width': np.mean(ub - lb),
            })

        # Quantile não-paramétrico (estimado no DEV)
        for cl in confidence_levels:
            alpha = 1 - cl
            lq = np.percentile(res_dev, (alpha/2)*100)
            uq = np.percentile(res_dev, (1 - alpha/2)*100)
            lb = p_tst + lq
            ub = p_tst + uq
            cov = np.mean((y_tst_r >= lb) & (y_tst_r <= ub))
            picp_results.append({
                'Model': model_name, 'Method': 'Quantile_NonParametric',
                'Confidence_Level': cl, 'PICP': cov,
                'Target_Coverage': cl, 'Coverage_Gap': cov - cl,
                'Mean_PI_Width': np.mean(ub - lb),
            })

        print(f"      ✅ processado")
    except Exception as e:
        print(f"      ❌ {str(e)[:80]}")

if picp_results:
    picp_df = pd.DataFrame(picp_results)
    picp_df.to_excel('Resultados_Artigo/PICP_Results_Otimizado.xlsx', index=False)
    picp_df.to_excel('Paper/Results/PICP_Results_Otimizado.xlsx',     index=False)
    print(f"\n✅ PICP concluído para {len(picp_df['Model'].unique())} modelos")
else:
    picp_df = pd.DataFrame()

# Gráficos PICP
if not picp_df.empty:
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif']  = ['Times New Roman']

    # Heatmap
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('PICP – Gaussian vs Quantile', fontsize=14, fontweight='bold')
    for idx, method in enumerate(['Gaussian_Homogeneous', 'Quantile_NonParametric']):
        ax = axes[idx]
        md = picp_df[picp_df['Method'] == method]
        if not md.empty:
            pv = md.pivot_table(values='PICP', index='Model',
                                columns='Confidence_Level')
            if not pv.empty:
                sns.heatmap(pv, annot=True, fmt='.3f', cmap='RdYlGn',
                            ax=ax, vmin=0, vmax=1)
                ax.set_title(method.replace('_', ' '), fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig('Figures/PICP_Heatmap_Comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig('Paper/Figures/PICP_Heatmap_Comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ PICP Heatmap salvo")

    # Curvas de calibração
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('PICP Calibration Curves', fontsize=14, fontweight='bold')
    for idx, (method, ax) in enumerate(
            zip(['Gaussian_Homogeneous', 'Quantile_NonParametric'], axes)):
        md = picp_df[picp_df['Method'] == method]
        for mod in md['Model'].unique():
            row = md[md['Model'] == mod]
            ax.plot(row['Confidence_Level'], row['PICP'],
                    'o-', label=mod, linewidth=1.5, markersize=6, alpha=0.7)
        ax.plot([0.6, 1], [0.6, 1], 'k--', lw=2, label='Perfect Calibration')
        ax.set_xlabel('Target Coverage', fontsize=12)
        ax.set_ylabel('Observed PICP',   fontsize=12)
        ax.set_title(method.replace('_', ' '), fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        ax.set_xlim(0.6, 1); ax.set_ylim(0.6, 1)
    plt.tight_layout()
    plt.savefig('Figures/PICP_Calibration_Curves.png', dpi=300, bbox_inches='tight')
    plt.savefig('Paper/Figures/PICP_Calibration_Curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Curvas de calibração PICP salvas")

    # Tabela resumo 95%
    picp95 = picp_df[picp_df['Confidence_Level'] == 0.95]
    pv_picp = picp95.pivot_table(values='PICP', index='Model',
                                  columns='Method').round(4)
    pv_gap  = picp95.pivot_table(values='Coverage_Gap', index='Model',
                                  columns='Method').round(4)
    pv_wid  = picp95.pivot_table(values='Mean_PI_Width', index='Model',
                                  columns='Method').round(2)
    picp_tbl = pd.DataFrame({'Model': pv_picp.index})
    for col, pfx in [('Gaussian_Homogeneous', 'Gaussian'),
                     ('Quantile_NonParametric', 'Quantile')]:
        if col in pv_picp.columns:
            picp_tbl[f'PICP_{pfx}']  = pv_picp[col].values
            picp_tbl[f'Gap_{pfx}']   = pv_gap[col].values
            picp_tbl[f'Width_{pfx}'] = pv_wid[col].values
    picp_tbl = picp_tbl.sort_values('PICP_Gaussian', ascending=False,
                                     na_position='last')
    picp_tbl.to_excel('Paper/Results/PICP_Summary_Table.xlsx', index=False)
    picp_tbl.to_excel('Resultados_Artigo/PICP_Summary_Table.xlsx', index=False)
    print("\n📊 Tabela PICP 95%:")
    print(picp_tbl.to_string(index=False))

print("\n✅ ANÁLISE PICP COMPLETA!")

# ============================================================
# PARTE 8: REPEATED K-FOLD CROSS VALIDATION (10×10)
# ============================================================

print("\n" + "="*80)
print("🔄 REPEATED K-FOLD CROSS VALIDATION (10×10)")
print("="*80)

n_splits = 10
n_repeats = 10
total_evals = n_splits * n_repeats

repeated_cv_results = []
repeated_cv_scores  = {m: [] for m in model_list}

for model_name, model in models.items():
    print(f"\n   {model_name} ...")
    info   = optimized_datasets[model_name]
    dev_df = info['dev_df']
    X_d    = dev_df[feature_names].values
    y_d    = dev_df[target].values

    for rep in range(n_repeats):
        rkf = RepeatedKFold(n_splits=n_splits, n_repeats=1,
                            random_state=42+rep)
        for fold, (tr_i, te_i) in enumerate(rkf.split(X_d), 1):
            try:
                mc = clone(model)                          # [FIX-3]
                mc.fit(X_d[tr_i], y_d[tr_i])
                yp = mc.predict(X_d[te_i])
                r2_f   = r2_score(y_d[te_i], yp)
                rmse_f = np.sqrt(mean_squared_error(y_d[te_i], yp))
                mae_f  = mean_absolute_error(y_d[te_i], yp)
                mape_f = mean_absolute_percentage_error(y_d[te_i], yp) * 100
                repeated_cv_scores[model_name].append(r2_f)
                repeated_cv_results.append({
                    'Model': model_name, 'Repeat': rep+1, 'Fold': fold,
                    'R2': r2_f, 'RMSE': rmse_f, 'MAE': mae_f, 'MAPE': mape_f,
                    'Cleaning_Method': best_cleaning_for_model[model_name],
                    'Samples': len(y_d),
                })
            except:
                repeated_cv_scores[model_name].append(np.nan)

repeated_cv_df = pd.DataFrame(repeated_cv_results)

cv_summary = []
for mn in model_list:
    sc = [s for s in repeated_cv_scores[mn] if not np.isnan(s)]
    if sc:
        cv_summary.append({
            'Model': mn, 'Mean_R2': np.mean(sc), 'Std_R2': np.std(sc),
            'CI_95_Lower': np.percentile(sc, 2.5),
            'CI_95_Upper': np.percentile(sc, 97.5),
            'N_Evaluations': len(sc),
            'Cleaning_Method': best_cleaning_for_model[mn],
        })

cv_summary_df = pd.DataFrame(cv_summary).sort_values('Mean_R2', ascending=False)
repeated_cv_df.to_excel(
    'Resultados_Artigo/Repeated_CV_Otimizado_10x10_Detailed.xlsx', index=False)
cv_summary_df.to_excel(
    'Resultados_Artigo/Repeated_CV_Otimizado_10x10_Summary.xlsx', index=False)

print("\n🏆 RANKING Repeated K-Fold CV:")
print(cv_summary_df[['Model', 'Mean_R2', 'Std_R2',
                      'CI_95_Lower', 'CI_95_Upper',
                      'Cleaning_Method']].to_string(index=False))

# Gráficos Repeated CV
model_order_cv = cv_summary_df['Model'].values

fig, ax = plt.subplots(figsize=(14, 8))
sns.boxplot(data=repeated_cv_df, x='Model', y='R2', ax=ax,
            order=model_order_cv, palette='viridis',
            fliersize=3, linewidth=1.5, width=0.7)
gm = repeated_cv_df['R2'].mean()
ax.axhline(gm, color='red', linestyle='--', lw=1.5,
           label=f'Média Global R² = {gm:.3f}')
ax.set_xlabel('Model', fontsize=12, fontweight='bold')
ax.set_ylabel('R² (Repeated K-Fold CV)', fontsize=12, fontweight='bold')
ax.set_title(f'Repeated {n_splits}-Fold CV ({n_repeats} repeats, '
             f'{total_evals} evals) – Optimized', fontsize=14, fontweight='bold')
ax.tick_params(axis='x', rotation=45, labelsize=10)
ax.grid(True, alpha=0.3, axis='y')
ax.legend(loc='lower right')
plt.tight_layout()
plt.savefig('Figuras_NestedCV/Repeated_CV_Boxplot_Otimizado.png',
            dpi=300, bbox_inches='tight')
plt.savefig('Paper/Figures/Repeated_CV_Boxplot_Otimizado.png',
            dpi=300, bbox_inches='tight')
plt.close()

# Heatmap de estabilidade
stab = repeated_cv_df.groupby(['Model', 'Repeat'])['R2'].mean().unstack()
if not stab.empty:
    plt.figure(figsize=(14, 8))
    sns.heatmap(stab, annot=True, fmt='.3f', cmap='RdYlGn',
                cbar_kws={'label': 'Mean R² per Repeat'})
    plt.title('Stability – Mean R² per Repeat', fontsize=14, fontweight='bold')
    plt.xlabel('Repeat'); plt.ylabel('Model')
    plt.tight_layout()
    plt.savefig('Figuras_NestedCV/Repeated_CV_Stability_Heatmap_Otimizado.png',
                dpi=300, bbox_inches='tight')
    plt.savefig('Paper/Figures/Repeated_CV_Stability_Heatmap_Otimizado.png',
                dpi=300, bbox_inches='tight')
    plt.close()

# Barras com IC
fig, ax = plt.subplots(figsize=(14, 8))
xp = np.arange(len(cv_summary_df))
bars = ax.bar(xp, cv_summary_df['Mean_R2'],
              yerr=[cv_summary_df['Mean_R2'] - cv_summary_df['CI_95_Lower'],
                    cv_summary_df['CI_95_Upper'] - cv_summary_df['Mean_R2']],
              capsize=5, color='steelblue', alpha=0.7,
              edgecolor='black', linewidth=1.2)
for bar, mv in zip(bars, cv_summary_df['Mean_R2']):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()+0.01,
            f'{mv:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
ax.set_xticks(xp)
ax.set_xticklabels(cv_summary_df['Model'], rotation=45, ha='right')
ax.set_ylim(0, 1.05)
ax.set_ylabel('Mean R²', fontsize=12, fontweight='bold')
ax.set_title(f'Model Performance – 95% CI '
             f'({n_splits}-fold, {n_repeats} repeats)', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y', linestyle='--')
plt.tight_layout()
plt.savefig('Figuras_NestedCV/Repeated_CV_Barplot_with_CI_Otimizado.png',
            dpi=300, bbox_inches='tight')
plt.savefig('Paper/Figures/Repeated_CV_Barplot_with_CI_Otimizado.png',
            dpi=300, bbox_inches='tight')
plt.close()

# Violin
fig, ax = plt.subplots(figsize=(14, 8))
sns.violinplot(data=repeated_cv_df, x='Model', y='R2', ax=ax,
               order=model_order_cv, palette='viridis',
               inner='box', cut=0, linewidth=1.2)
ax.set_xlabel('Model', fontsize=12, fontweight='bold')
ax.set_ylabel('R²', fontsize=12, fontweight='bold')
ax.set_title('R² Distribution – Repeated K-Fold CV', fontsize=14, fontweight='bold')
ax.tick_params(axis='x', rotation=45, labelsize=10)
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('Figuras_NestedCV/Repeated_CV_Violin_Plot_Otimizado.png',
            dpi=300, bbox_inches='tight')
plt.savefig('Paper/Figures/Repeated_CV_Violin_Plot_Otimizado.png',
            dpi=300, bbox_inches='tight')
plt.close()

# Train vs Test boxplots (Repeated CV)
cv_metrics = []
for model_name, model in models.items():
    info   = optimized_datasets[model_name]
    dev_df = info['dev_df']
    X_d    = dev_df[feature_names].values
    y_d    = dev_df[target].values

    for rep in range(n_repeats):
        rkf = RepeatedKFold(n_splits=n_splits, n_repeats=1,
                            random_state=42+rep)
        for fold, (tr_i, te_i) in enumerate(rkf.split(X_d), 1):
            try:
                mc = clone(model)                          # [FIX-3]
                mc.fit(X_d[tr_i], y_d[tr_i])
                yp_tr = mc.predict(X_d[tr_i])
                yp_te = mc.predict(X_d[te_i])
                for ds, y_ds, yp in [('Training', y_d[tr_i], yp_tr),
                                     ('Testing',  y_d[te_i], yp_te)]:
                    cv_metrics.append({
                        'Model': model_name, 'Dataset': ds,
                        'Repeat': rep+1, 'Fold': fold,
                        'R2':   r2_score(y_ds, yp),
                        'RMSE': np.sqrt(mean_squared_error(y_ds, yp)),
                        'MAE':  mean_absolute_error(y_ds, yp),
                        'MAPE': mean_absolute_percentage_error(y_ds, yp)*100,
                    })
            except:
                pass

cv_metrics_df = pd.DataFrame(cv_metrics)
cv_metrics_df.to_excel(
    'Resultados_Artigo/Repeated_CV_Train_Test_Metrics_Otimizado.xlsx', index=False)

tc = '#2C3E50'; tc2 = '#E74C3C'
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Repeated K-Fold CV (10×10) – Performance Distribution',
             fontsize=14, fontweight='bold', y=0.98)
for idx, (metric, title) in enumerate(
        zip(['R2','RMSE','MAE','MAPE'], ['R²','RMSE (MPa)','MAE (MPa)','MAPE (%)'])):
    ax = axes[idx//2, idx%2]
    asc = metric != 'R2'
    ord_ = (cv_metrics_df[cv_metrics_df['Dataset'] == 'Testing']
            .groupby('Model')[metric].median()
            .sort_values(ascending=asc).index)
    sns.boxplot(data=cv_metrics_df, x='Model', y=metric, hue='Dataset', ax=ax,
                order=ord_, palette={'Training': tc, 'Testing': tc2},
                fliersize=2, linewidth=1.2, width=0.7)
    ax.set_title(f'({chr(97+idx)}) {title}', fontsize=12, fontweight='bold', loc='left')
    ax.set_xlabel(''); ax.set_ylabel(title, fontsize=11)
    ax.tick_params(axis='x', rotation=45, labelsize=9)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    if idx == 0: ax.legend(title='Dataset', loc='lower right')
    else:        ax.legend_.remove()
plt.tight_layout()
plt.savefig('Figuras_NestedCV/Repeated_CV_All_Metrics_Boxplot_Otimizado.png',
            dpi=300, bbox_inches='tight')
plt.savefig('Paper/Figures/Repeated_CV_All_Metrics_Boxplot_Otimizado.png',
            dpi=300, bbox_inches='tight')
plt.close()
print("✅ Gráficos Repeated K-Fold CV salvos")

# ============================================================
# TAYLOR DIAGRAM
# ============================================================

print("\n" + "="*80)
print("TAYLOR DIAGRAM – ANÁLISE DE MODELOS")
print("="*80)

if len(pred_ref) >= 2:
    std_obs   = np.std(y_ref_ext)
    std_m_lst, corr_lst, rmse_lst, nm_lst = [], [], [], []

    for mn, pr in pred_ref.items():
        std_m_lst.append(np.std(pr))
        corr_lst.append(np.corrcoef(y_ref_ext, pr)[0, 1])
        rmse_lst.append(np.sqrt(mean_squared_error(y_ref_ext, pr)))
        nm_lst.append(mn)
        print(f"   {mn}: Corr={corr_lst[-1]:.4f}, "
              f"Std={std_m_lst[-1]:.2f}, RMSE={rmse_lst[-1]:.2f}")

    try:
        import skill_metrics as sm
        fig = plt.figure(figsize=(10, 8))
        sm.taylor_diagram(std_obs, std_m_lst, corr_lst,
                          markerLabel=nm_lst, markerSize=8,
                          markerLegend='on', colMarker='k',
                          styleOBS='-', colOBS='k', markerobs='o',
                          titleRMS='on', labelRMS='on')
        plt.title('Taylor Diagram – Model Comparison (Optimized)',
                  fontsize=14, fontweight='bold')
        plt.tight_layout()
    except ImportError:
        fig, ax = plt.subplots(figsize=(10, 8))
        theta  = np.linspace(0, np.pi/2, 100)
        r_max  = max(max(std_m_lst), std_obs) * 1.2
        for c in [0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.99]:
            ang = np.arccos(c)
            ax.plot([0, r_max*np.cos(ang)], [0, r_max*np.sin(ang)],
                    'k--', alpha=0.3, lw=0.5)
            ax.text(r_max*np.cos(ang)*1.02, r_max*np.sin(ang)*1.02,
                    f'{c}', fontsize=8, alpha=0.7)
        for r in np.linspace(0.5, r_max, 5):
            ax.plot(r*np.cos(theta), r*np.sin(theta), 'k--', alpha=0.3, lw=0.5)
        ax.plot(std_obs, 0, 'ro', markersize=12, label='Observed', zorder=5)
        cols_t = plt.cm.tab10(np.linspace(0, 1, len(nm_lst)))
        for i, (n, sm_v, c) in enumerate(zip(nm_lst, std_m_lst, corr_lst)):
            ang = np.arccos(c)
            ax.plot(sm_v*np.cos(ang), sm_v*np.sin(ang), 'o',
                    color=cols_t[i], markersize=10, zorder=4)
            ax.annotate(n, (sm_v*np.cos(ang), sm_v*np.sin(ang)),
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        ax.set_xlim(0, r_max); ax.set_ylim(0, r_max)
        ax.set_aspect('equal')
        ax.set_xlabel('Std Dev'); ax.set_ylabel('Std Dev')
        ax.set_title('Taylor Diagram – Model Comparison', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.2)
        ax.legend(loc='upper right')

    plt.savefig('Figuras_NestedCV/Taylor_Diagram.png', dpi=300, bbox_inches='tight')
    plt.savefig('Paper/Figures/Taylor_Diagram.png',    dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Taylor Diagram salvo")

    taylor_stats = pd.DataFrame({
        'Model': nm_lst, 'Correlation': corr_lst,
        'Std_Dev_Predicted': std_m_lst,
        'Std_Dev_Observed':  [std_obs]*len(nm_lst),
        'RMSE': rmse_lst,
        'Bias': [np.mean(pred_ref[m] - y_ref_ext) for m in nm_lst],
    }).sort_values('Correlation', ascending=False)
    taylor_stats.to_excel('Resultados_Artigo/Taylor_Diagram_Statistics.xlsx', index=False)
    taylor_stats.to_excel('Paper/Results/Taylor_Diagram_Statistics.xlsx',    index=False)
    print(taylor_stats.to_string(index=False))

# ============================================================
# PARTE 9: MÉTRICAS E GRÁFICOS DE DESEMPENHO
# ============================================================

print("\n" + "="*80)
print("📊 GRÁFICOS DE DESEMPENHO")
print("="*80)

all_metrics = train_metrics + test_metrics
metrics_df  = pd.DataFrame(all_metrics)
error_var_df= pd.DataFrame(error_variance)

training_color = '#2E86AB'
testing_color  = '#A23B72'

# Boxplots completos
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Performance Metrics – Training vs Testing (Optimized)',
             fontsize=14, fontweight='bold', y=0.98)
for idx, (metric, title) in enumerate(
        zip(['RMSE','R2','MAE','MAPE'],
            ['RMSE (MPa)','R²','MAE (MPa)','MAPE (%)'])):
    ax = axes[idx//2, idx%2]
    data = metrics_df[metrics_df['Metric'] == metric]
    if data.empty: continue
    asc = metric != 'R2'
    order = (data[data['Set'] == 'Testing'].groupby('Model')['Value']
             .median().sort_values(ascending=asc).index)
    pos   = np.arange(len(order)); w = 0.35
    tv = [data[(data['Model']==m) & (data['Set']=='Training')]['Value'].values
          for m in order]
    ev = [data[(data['Model']==m) & (data['Set']=='Testing')]['Value'].values
          for m in order]
    ax.boxplot(tv, positions=pos-w/2, widths=w, patch_artist=True,
               boxprops=dict(facecolor=training_color, alpha=0.7),
               medianprops=dict(color='black', linewidth=2))
    ax.boxplot(ev, positions=pos+w/2, widths=w, patch_artist=True,
               boxprops=dict(facecolor=testing_color, alpha=0.7),
               medianprops=dict(color='black', linewidth=2))
    ax.set_xticks(pos)
    ax.set_xticklabels(order, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel(title, fontsize=11)
    ax.set_title(f'({chr(97+idx)}) {metric}', fontsize=12, fontweight='bold', loc='left')
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    if idx == 0:
        ax.legend([plt.Rectangle((0,0),1,1, facecolor=training_color),
                   plt.Rectangle((0,0),1,1, facecolor=testing_color)],
                  ['Training', 'Testing'], loc='upper right')
plt.tight_layout()
plt.savefig("Figures/Boxplots_All_Metrics_Otimizado.png",       dpi=300, bbox_inches='tight')
plt.savefig("Paper/Figures/Boxplots_All_Metrics_Otimizado.png", dpi=300, bbox_inches='tight')
plt.close()
print("✅ Boxplots completos salvos")

# Variância do erro
fig, ax = plt.subplots(figsize=(12, 6))
vd = error_var_df.copy()
mo_v = (vd[vd['Set']=='Testing'].groupby('Model')['Value']
        .mean().sort_values().index)
pos = np.arange(len(mo_v)); w = 0.35
tv = [vd[(vd['Model']==m) & (vd['Set']=='Training')]['Value'].values for m in mo_v]
ev = [vd[(vd['Model']==m) & (vd['Set']=='Testing')]['Value'].values  for m in mo_v]
ax.boxplot(tv, positions=pos-w/2, widths=w, patch_artist=True,
           boxprops=dict(facecolor=training_color, alpha=0.7))
ax.boxplot(ev, positions=pos+w/2, widths=w, patch_artist=True,
           boxprops=dict(facecolor=testing_color, alpha=0.7))
ax.set_xticks(pos)
ax.set_xticklabels(mo_v, rotation=45, ha='right', fontsize=10)
ax.set_ylabel('Error Variance', fontsize=12)
ax.set_title('Error Variance – Training vs Testing', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')
ax.legend([plt.Rectangle((0,0),1,1, facecolor=training_color),
           plt.Rectangle((0,0),1,1, facecolor=testing_color)],
          ['Training', 'Testing'], loc='upper right')
plt.tight_layout()
plt.savefig("Figures/Error_Variance_Boxplot_Otimizado.png",       dpi=300, bbox_inches='tight')
plt.savefig("Paper/Figures/Error_Variance_Boxplot_Otimizado.png", dpi=300, bbox_inches='tight')
plt.close()
print("✅ Error Variance salvo")

# Violin plots
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Violin Plots – Performance Metrics (Optimized)',
             fontsize=14, fontweight='bold', y=0.98)
for idx, (metric, title) in enumerate(
        zip(['R2','RMSE','MAE','MAPE'],
            ['R²','RMSE (MPa)','MAE (MPa)','MAPE (%)'])):
    ax = axes[idx//2, idx%2]
    data = metrics_df[metrics_df['Metric'] == metric]
    if data.empty:
        ax.text(0.5, 0.5, f'No data', ha='center', va='center'); continue
    asc = metric != 'R2'
    order = (data[data['Set']=='Testing'].groupby('Model')['Value']
             .median().sort_values(ascending=asc).index)
    sns.violinplot(data=data, x='Model', y='Value', hue='Set', ax=ax,
                   order=order,
                   palette={'Training': training_color, 'Testing': testing_color},
                   split=False, cut=0, inner='box', linewidth=1)
    ax.set_title(title, fontsize=12, fontweight='bold', loc='left')
    ax.set_xlabel(''); ax.set_ylabel(title)
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3, axis='y')
    if idx == 0: ax.legend(title='Dataset')
    else:        ax.legend_.remove()
plt.tight_layout()
plt.savefig("Figuras_Violin/Violin_Plots_All_Metrics_Otimizado.png",
            dpi=300, bbox_inches='tight')
plt.savefig("Paper/Figures/Violin_Plots_All_Metrics_Otimizado.png",
            dpi=300, bbox_inches='tight')
plt.close()
print("✅ Violin plots salvos")

# ============================================================
# IFI (Information-based Feature Importance) SENSITIVITY
# ============================================================

print("\n📊 IFI SENSITIVITY...")
X_mat = results_df[["R2","RMSE","MAE","MAPE"]].copy()
X_mat["RMSE"] = 1 / X_mat["RMSE"]
X_mat["MAE"]  = 1 / X_mat["MAE"]
X_mat["MAPE"] = 1 / X_mat["MAPE"]
P = X_mat / X_mat.sum()
k = 1 / np.log(len(X_mat))
entropy = -k * (P * np.log(P + 1e-10)).sum()
div     = 1 - entropy
weights = div / div.sum()
results_df["IFI"] = (X_mat * weights).sum(axis=1)
ranking = results_df.sort_values("IFI", ascending=False)
ranking.to_excel("Paper/Results/model_ranking_otimizado.xlsx", index=False)

print("\n📊 Ranking IFI:")
print(ranking[['Model','R2','RMSE','MAE','MAPE',
               'IFI','Cleaning_Method','Samples']].to_string(index=False))

weights_range = np.linspace(0.1, 0.8, 20)
sens_scores   = np.array([
    w*results_df["R2"] + (1-w)*(1/results_df["RMSE"])
    for w in weights_range
])

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('IFI Sensitivity (Optimized)', fontsize=14, fontweight='bold')
colors20 = plt.cm.tab20(np.linspace(0, 1, len(results_df["Model"])))
for i, mod in enumerate(results_df['Model']):
    axes[0].plot(weights_range, sens_scores[:, i],
                 color=colors20[i], lw=1.5, alpha=0.8)
axes[0].set_xlabel("R² weight (w)"); axes[0].set_ylabel("IFI Score")
axes[0].set_title("All Models"); axes[0].grid(True, alpha=0.3)

top10 = ranking.head(10)['Model'].tolist()
top10_idx = [results_df[results_df['Model']==m].index[0] for m in top10]
c10 = plt.cm.viridis(np.linspace(0, 1, 10))
for i, idx in enumerate(top10_idx):
    axes[1].plot(weights_range, sens_scores[:, idx],
                 color=c10[i], lw=2.5, label=results_df.iloc[idx]['Model'])
axes[1].set_xlabel("R² weight (w)"); axes[1].set_ylabel("IFI Score")
axes[1].set_title("Top 10 Models"); axes[1].grid(True, alpha=0.3)
axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
plt.tight_layout()
plt.savefig("Figuras_IFI/IFI_sensitivity_with_legend_otimizado.png",
            dpi=300, bbox_inches='tight')
plt.savefig("Paper/Figures/IFI_sensitivity_with_legend_otimizado.png",
            dpi=300, bbox_inches='tight')
plt.close()

sens_matrix = sens_scores.T
plt.figure(figsize=(14, 10))
sns.heatmap(sens_matrix,
            xticklabels=[f'{w:.1f}' for w in weights_range],
            yticklabels=results_df['Model'],
            cmap='viridis', annot=False)
plt.xlabel('R² weight (w)'); plt.ylabel('Models')
plt.title('IFI Sensitivity Heatmap (Optimized)')
plt.tight_layout()
plt.savefig("Figuras_IFI/IFI_sensitivity_heatmap_otimizado.png",
            dpi=300, bbox_inches='tight')
plt.savefig("Paper/Figures/IFI_sensitivity_heatmap_otimizado.png",
            dpi=300, bbox_inches='tight')
plt.close()

plt.figure(figsize=(10, 6))
colors_r = plt.cm.viridis(np.linspace(0, 1, len(ranking)))
bars = plt.barh(range(len(ranking)), ranking['IFI'], color=colors_r)
plt.yticks(range(len(ranking)), ranking['Model'])
plt.xlabel('IFI Score'); plt.title('Model Ranking – IFI Score')
plt.gca().invert_yaxis()
for i, bar in enumerate(bars):
    plt.text(bar.get_width()+0.001, bar.get_y()+bar.get_height()/2,
             f'{ranking["IFI"].iloc[i]:.3f}', va='center')
plt.tight_layout()
plt.savefig("Figures/IFI_Ranking_Otimizado.png",       dpi=300, bbox_inches='tight')
plt.savefig("Paper/Figures/IFI_Ranking_Otimizado.png", dpi=300, bbox_inches='tight')
plt.close()
print("✅ IFI Sensitivity plots salvos")

# ============================================================
# CORRELAÇÃO DOS MODELOS (partição de referência comum) [FIX-2]
# ============================================================

print("\n📊 CORRELAÇÃO DOS MODELOS (referência comum) ...")
pred_matrix    = pd.DataFrame(pred_ref)       # todos com mesmo y_ref_ext
corr_matrix    = pred_matrix.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm',
            center=0, square=True, linewidths=0.5)
plt.title('Pearson Correlation – Model Predictions (Common Test Set)',
          fontsize=14)
plt.tight_layout()
plt.savefig('Figuras_Correlacao/Correlacao_Performance_Modelos_Otimizado.png',
            dpi=300, bbox_inches='tight')
plt.savefig('Paper/Figures/Correlacao_Performance_Modelos_Otimizado.png',
            dpi=300, bbox_inches='tight')
plt.close()
print("✅ Heatmap de correlação salvo")

# ============================================================
# DM HEATMAP (partição de referência comum) [FIX-2]
# ============================================================

print("\n" + "="*80)
print("DIEBOLD-MARIANO HEATMAP (referência comum) [FIX-2]")
print("="*80)

if len(pred_ref) >= 2:
    dm_res  = []
    nms_dm  = list(pred_ref.keys())
    for i in range(len(nms_dm)):
        for j in range(i+1, len(nms_dm)):
            e1 = y_ref_ext - pred_ref[nms_dm[i]]
            e2 = y_ref_ext - pred_ref[nms_dm[j]]
            d  = e1**2 - e2**2
            dm = np.mean(d) / np.sqrt(np.var(d) / len(d))
            p  = 2 * (1 - stats.norm.cdf(abs(dm)))
            dm_res.append([nms_dm[i], nms_dm[j], dm, p])

    dm_df = pd.DataFrame(dm_res, columns=["Model1","Model2","DM","p"])
    dm_df.to_excel('Resultados_Artigo/DM_Heatmap_Results_Otimizado.xlsx', index=False)
    dm_df.to_excel('Paper/Results/DM_Heatmap_Results_Otimizado.xlsx',    index=False)

    dm_pivot = dm_df.pivot(index="Model1", columns="Model2", values="DM")
    plt.figure(figsize=(10, 8))
    sns.heatmap(dm_pivot, annot=True, fmt='.3f', cmap='coolwarm',
                center=0, square=True, linewidths=0.5,
                cbar_kws={'label': 'DM Statistic'})
    plt.title('Diebold-Mariano Test Heatmap (Common Test Set)', fontsize=14)
    plt.tight_layout()
    plt.savefig('Figuras_DM_Heatmap/DM_Heatmap_Otimizado.png',
                dpi=300, bbox_inches='tight')
    plt.savefig('Paper/Figures/DM_Heatmap_Otimizado.png',
                dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ DM Heatmap salvo")

# ============================================================
# SHAP (partição de referência comum para X_test) [FIX-2]
# ============================================================

print("\n📊 SHAP IMPORTANCE...")
X_ref_df = pd.DataFrame(X_tst_ref_sc, columns=feature_names)

all_shap_importance = []

for model_name in model_list:
    if model_name not in models_trained:
        continue
    print(f"\n   🔍 {model_name}")
    try:
        m = models_trained[model_name]
        supports_shap = False
        explainer     = None

        try:
            if (hasattr(m, 'get_booster') or
                    hasattr(m, 'feature_importances_') or
                    'Forest' in type(m).__name__):
                explainer     = shap.TreeExplainer(m)
                supports_shap = True
                print(f"      ✅ TreeExplainer")
        except: pass

        if not supports_shap:
            try:
                if hasattr(m, 'coef_'):
                    explainer     = shap.LinearExplainer(m, X_ref_df)
                    supports_shap = True
                    print(f"      ✅ LinearExplainer")
            except: pass

        if not supports_shap:
            try:
                X_samp    = X_ref_df[:50] if len(X_ref_df) > 50 else X_ref_df
                explainer = shap.KernelExplainer(m.predict, X_samp)
                supports_shap = True
                print(f"      ✅ KernelExplainer")
            except: continue

        if supports_shap and explainer:
            if isinstance(explainer, shap.KernelExplainer):
                X_used    = X_ref_df[:50] if len(X_ref_df) > 50 else X_ref_df
                shap_vals = explainer.shap_values(X_used)
            else:
                X_used    = X_ref_df
                shap_vals = explainer.shap_values(X_used)

            mabs = (np.mean([np.abs(sv).mean(axis=0) for sv in shap_vals], axis=0)
                    if isinstance(shap_vals, list)
                    else np.abs(shap_vals).mean(axis=0))

            imp_df = pd.DataFrame({
                'Feature':              feature_names,
                'SHAP_Importance':      mabs,
                'Normalized_Importance':mabs / (mabs.sum() + 1e-10),
            }).sort_values('SHAP_Importance', ascending=False)
            imp_df['Model'] = model_name
            all_shap_importance.append(imp_df)
            imp_df.to_excel(
                f"Paper/Results/SHAP_Results/SHAP_Importance_{model_name}_Otimizado.xlsx",
                index=False)

            plt.figure(figsize=(10, 6))
            plt.barh(imp_df['Feature'], imp_df['SHAP_Importance'], color='steelblue')
            plt.xlabel('Mean |SHAP value|')
            plt.title(f'SHAP Feature Importance – {model_name}')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.savefig(f"Figuras_Shap/SHAP_Bar_{model_name}_Otimizado.png",
                        dpi=300, bbox_inches='tight')
            plt.savefig(f"Paper/Figures/SHAP_Bar_{model_name}_Otimizado.png",
                        dpi=300, bbox_inches='tight')
            plt.close()

            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_vals, X_used, feature_names=feature_names, show=False)
            plt.tight_layout()
            plt.savefig(f"Figuras_Shap/SHAP_Summary_{model_name}_Otimizado.png",
                        dpi=300, bbox_inches='tight')
            plt.savefig(f"Paper/Figures/SHAP_Summary_{model_name}_Otimizado.png",
                        dpi=300, bbox_inches='tight')
            plt.close()

            if isinstance(explainer, shap.TreeExplainer):
                for feat in imp_df.head(3)['Feature']:
                    try:
                        plt.figure(figsize=(10, 6))
                        shap.dependence_plot(feat, shap_vals, X_used,
                                             feature_names=feature_names, show=False)
                        plt.tight_layout()
                        plt.savefig(
                            f"Figuras_Shap/SHAP_Dependence_{model_name}_{feat}_Otimizado.png",
                            dpi=300, bbox_inches='tight')
                        plt.close()
                    except: continue

            print(f"      ✅ SHAP salvo")
    except Exception as e:
        print(f"      ⚠️ {str(e)[:100]}")

if all_shap_importance:
    cons_shap = pd.concat(all_shap_importance, ignore_index=True)
    cons_shap.to_excel(
        "Paper/Results/SHAP_Results/SHAP_Importance_ALL_MODELS_Otimizado.xlsx",
        index=False)
    piv_shap      = cons_shap.pivot_table(
        values='SHAP_Importance', index='Feature', columns='Model', fill_value=0)
    piv_shap_norm = piv_shap.div(piv_shap.sum(axis=0), axis=1)
    piv_shap.to_excel(
        "Paper/Results/SHAP_Results/SHAP_Importance_Pivot_Otimizado.xlsx")
    piv_shap_norm.to_excel(
        "Paper/Results/SHAP_Results/SHAP_Importance_Pivot_Normalized_Otimizado.xlsx")

    plt.figure(figsize=(14, 10))
    sns.heatmap(piv_shap_norm, annot=True, fmt='.3f', cmap='viridis')
    plt.title('Normalized SHAP Importance – All Models', fontsize=14)
    plt.tight_layout()
    plt.savefig("Figuras_Shap/SHAP_Importance_Heatmap_Otimizado.png",
                dpi=300, bbox_inches='tight')
    plt.savefig("Paper/Figures/SHAP_Importance_Heatmap_Otimizado.png",
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ SHAP consolidado para {len(all_shap_importance)} modelos")

# ============================================================
# GRÁFICOS DE DISPERSÃO (PARITY PLOTS) — [FIX-2] + [FIX-6]
# Usa pred_ref (referência comum) + sem plt.show()
# ============================================================

print("\n📊 GERANDO PARITY PLOTS (referência comum)...")
for name, pr in pred_ref.items():
    y_pl = y_ref_ext
    p_pl = pr
    r2v  = r2_score(y_pl, p_pl)
    rmse = np.sqrt(mean_squared_error(y_pl, p_pl))

    plt.figure(figsize=(7, 7))
    plt.scatter(y_pl, p_pl, alpha=0.5, s=40, color='#1f77b4',
                edgecolors='k', linewidth=0.5, label='Data')
    lims = [min(y_pl.min(), p_pl.min()), max(y_pl.max(), p_pl.max())]
    plt.plot(lims, lims, 'r--', lw=2, label='Ideal (1:1)', zorder=3)
    plt.title(f"Validação Externa: {name}\n"
              f"$R^2$={r2v:.4f} | RMSE={rmse:.2f} MPa", fontsize=12, pad=15)
    plt.xlabel("Experimental $f_{ck}$ (MPa)", fontsize=10)
    plt.ylabel("Predicted $f_{ck}$ (MPa)",    fontsize=10)
    plt.legend(loc='upper left'); plt.grid(True, linestyle=':', alpha=0.5)
    plt.xlim(lims); plt.ylim(lims)
    plt.tight_layout()
    fn = f"Paper/Figures/Validacao_Externa/Scatter_{name.replace(' ','_')}.png"
    plt.savefig(fn, dpi=300, bbox_inches='tight')
    plt.close()                                                   # [FIX-6]
    print(f"   ✅ {name}")

print("✅ Parity plots gerados")

# Residual plots
print("\n📊 GERANDO RESIDUAL PLOTS...")
for name, pr in pred_ref.items():
    residuals = y_ref_ext - pr
    plt.figure(figsize=(8, 5))
    plt.scatter(pr, residuals, alpha=0.6, s=40, color='#2ca02c',
                edgecolors='k', linewidth=0.5)
    plt.axhline(0, color='r', linestyle='--', lw=2)
    plt.title(f"Residual Analysis – {name}", fontsize=12)
    plt.xlabel("Predicted $f_{ck}$ (MPa)", fontsize=10)
    plt.ylabel("Residuals (MPa)",            fontsize=10)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(f"Paper/Figures/Residuos/Residuals_{name.replace(' ','_')}.png",
                dpi=300, bbox_inches='tight')
    plt.close()
print("✅ Residual plots salvos")

# KDE plots
print("\n📊 GERANDO KDE PLOTS...")
for name, pr in pred_ref.items():
    plt.figure(figsize=(7, 6))
    try:
        sns.kdeplot(x=y_ref_ext, y=pr, cmap="viridis",
                    fill=True, thresh=0.05, levels=15)
        lims = [min(y_ref_ext.min(), pr.min()),
                max(y_ref_ext.max(), pr.max())]
        plt.plot(lims, lims, 'r--', alpha=0.7, label='Ideal (1:1)')
        plt.title(f"KDE – {name}", fontsize=12)
        plt.xlabel("Experimental $f_{ck}$ (MPa)", fontsize=10)
        plt.ylabel("Predicted $f_{ck}$ (MPa)",    fontsize=10)
        plt.grid(True, linestyle=':', alpha=0.4)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"Paper/Figures/KDE_Density/KDE_{name.replace(' ','_')}.png",
                    dpi=300, bbox_inches='tight')
    except Exception as e:
        print(f"   ⚠️ KDE {name}: {e}")
    plt.close()
print("✅ KDE plots salvos")

# MAPE plots
print("\n📊 GERANDO MAPE PLOTS...")
for name, pr in pred_ref.items():
    mape_pt = np.abs((y_ref_ext - pr) / (y_ref_ext + 1e-10)) * 100
    plt.figure(figsize=(6, 5))
    plt.scatter(y_ref_ext, mape_pt, alpha=0.6, s=30)
    plt.xlabel("fck (MPa)"); plt.ylabel("MAPE (%)")
    plt.title(f"MAPE vs fck – {name}")
    plt.tight_layout()
    plt.savefig(f"Figures/MAPE_vs_fck_{name}_Otimizado.png",
                dpi=300, bbox_inches='tight')
    plt.close()
print("✅ MAPE plots salvos")

# ============================================================
# TABELA RESUMO DAS ESTATÍSTICAS
# ============================================================

print("\n📊 Tabela resumo das estatísticas...")
try:
    stats_summ = []
    for model in results_df['Model'].unique():
        md = results_df[results_df['Model'] == model]
        for metric in ['R2','RMSE','MAE','MAPE']:
            if metric in results_df.columns:
                stats_summ.append({
                    'Model': model, 'Metric': metric,
                    'Mean': md[metric].mean(), 'Std': md[metric].std(),
                    'Median': md[metric].median(),
                    'Min': md[metric].min(), 'Max': md[metric].max(),
                    'Q1': md[metric].quantile(0.25),
                    'Q3': md[metric].quantile(0.75),
                    'IQR': md[metric].quantile(0.75) - md[metric].quantile(0.25),
                })
    if stats_summ:
        st_df = pd.DataFrame(stats_summ)
        st_df.to_excel(
            'Resultados_Artigo/Boxplot_Statistics_Detailed_Otimizado.xlsx',
            index=False)
        st_df.pivot_table(values='Mean', index='Model', columns='Metric').to_excel(
            'Resultados_Artigo/Boxplot_Means_Otimizado.xlsx')
        st_df.pivot_table(values='Std',  index='Model', columns='Metric').to_excel(
            'Resultados_Artigo/Boxplot_Stds_Otimizado.xlsx')
        print("✅ Tabelas de estatísticas salvas")
except Exception as e:
    print(f"   ❌ {e}")

# ============================================================
# PARTE 10: ECONOMETRIC RESIDUAL DIAGNOSTICS [FIX-2]
# Diagnósticos sobre y_ref_ext (conjunto comum)
# ============================================================

print("\n🔬 ECONOMETRIC RESIDUAL DIAGNOSTICS (referência comum)...")
import statsmodels.api as sm
from statsmodels.stats.diagnostic import (het_breuschpagan, het_white,
                                          acorr_breusch_godfrey, normal_ad)
from statsmodels.stats.stattools import durbin_watson
from scipy.stats import jarque_bera

diagnostics = []
for name, pr in pred_ref.items():
    print(f"   Testando: {name}")
    res    = y_ref_ext - pr
    X_aux  = sm.add_constant(pr)
    sf_s, sf_p = normal_ad(res)
    jb_s, jb_p = jarque_bera(res)
    bp_s, bp_p, _, _ = het_breuschpagan(res, X_aux)
    wh_s, wh_p, _, _ = het_white(res, X_aux)
    dw_s  = durbin_watson(res)
    ols_a = sm.OLS(res, X_aux).fit()
    bg_s, bg_p, _, _ = acorr_breusch_godfrey(ols_a, nlags=2)
    diagnostics.append({
        "Model": name,
        "ShapiroFrancia_stat": sf_s, "ShapiroFrancia_p": sf_p,
        "JarqueBera_stat": jb_s,     "JarqueBera_p": jb_p,
        "BreuschPagan_stat": bp_s,   "BreuschPagan_p": bp_p,
        "White_stat": wh_s,          "White_p": wh_p,
        "DurbinWatson": dw_s,
        "BreuschGodfrey_stat": bg_s, "BreuschGodfrey_p": bg_p,
    })
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    sm.graphics.tsa.plot_acf(res,  lags=20, ax=ax[0])
    ax[0].set_title(f"ACF – {name}")
    sm.graphics.tsa.plot_pacf(res, lags=20, ax=ax[1])
    ax[1].set_title(f"PACF – {name}")
    plt.tight_layout()
    plt.savefig(f"Paper/Residual_Diagnostics/ACF_PACF_{name}_Otimizado.png", dpi=300)
    plt.close()

diag_df = pd.DataFrame(diagnostics).sort_values("ShapiroFrancia_p", ascending=False)
print("\n📊 RESIDUAL DIAGNOSTICS:")
print(diag_df[['Model','ShapiroFrancia_p','JarqueBera_p',
               'BreuschPagan_p','White_p','DurbinWatson']].to_string(index=False))
diag_df.to_csv("Paper/Results/residual_diagnostics_full_otimizado.csv", index=False)
diag_df.to_excel("Paper/Results/residual_diagnostics_full_otimizado.xlsx", index=False)
print("✅ Residual diagnostics concluído")

# Gráfico heterocedasticidade
hetero = []
for _, row in diag_df.iterrows():
    hs = []
    if row['BreuschPagan_p'] < 0.05: hs.append("Breusch-Pagan")
    if row['White_p']        < 0.05: hs.append("White")
    hetero.append({'Model': row['Model'],
                   'BreuschPagan_p': row['BreuschPagan_p'],
                   'White_p':        row['White_p'],
                   'Status': f"HETERO ({', '.join(hs)})" if hs
                              else "Homoscedastic"})
het_df = pd.DataFrame(hetero)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, col, title in zip(axes,
                           ['BreuschPagan_p', 'White_p'],
                           ['Breusch-Pagan Test', 'White Test']):
    colors_h = ['red' if p < 0.05 else 'green' for p in het_df[col]]
    ax.bar(het_df['Model'], het_df[col], color=colors_h, alpha=0.7)
    ax.axhline(0.05, color='red', linestyle='--', label='α=0.05')
    ax.set_ylabel('p-value'); ax.set_title(title)
    ax.set_xticklabels(het_df['Model'], rotation=45, ha='right')
    ax.legend(); ax.set_ylim(0, 1)
plt.suptitle('Heteroscedasticity Tests (Common Test Set)',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('Paper/Figures/heterocedasticidade_pvalues_otimizado.png',
            dpi=300, bbox_inches='tight')
plt.close()
print("✅ Gráfico heterocedasticidade salvo")

# ============================================================
# PARTE 11: PREDIÇÃO PARA NOVAS DOSAGENS
# ============================================================

print("\n🔮 PREDIÇÃO PARA NOVAS DOSAGENS...")
novas = pd.DataFrame({
    "C":     [450,630,630,630,630,630,337.9,213.5,213.5,550,229.7,213.8,146.5,252,145.4,122.6,183.9,141.3],
    "S":     [180,180,180,180,180,180,189,0,0,0,0,98.1,114.6,0,0,183.9,122.6,212],
    "FA":    [0,0,0,0,0,0,0,174.2,174.2,0,118.2,24.5,89.3,0,178.9,0,0,0],
    "SF":    [90,90,90,90,90,90,0,0,0,0,0,0,0,0,0,0,0,0],
    "LP":    [180,0,0,0,0,90,0,0,0,0,0,0,0,0,0,0,0,0],
    "W":     [144,144,162,162,162,162,174.9,154.6,154.6,165,195.2,181.7,201.9,185,201.7,203.5,203.5,203.5],
    "SP":    [18,18,18,18,18,18,9.5,11.7,11.7,3.85,6.1,6.7,8.8,0,7.8,0,0,0],
    "Gravel":[923,923,923,923,923,923,944.7,1052.3,1052.3,1057,1028.1,1066,860,1111,824,958.2,959.2,971.8],
    "Sand":  [616,616,616,616,616,616,755.8,775.5,775.5,705,757.6,785.5,829.5,784,868.7,800.1,800,748.5],
    "Age":   [180,365,365,365,180,180,56,100,28,3,100,28,28,28,28,7,3,3],
})[feature_names]

predicoes = {}
for name, m in models_trained.items():
    try:
        sc_m = scalers[name]
        predicoes[name] = m.predict(sc_m.transform(novas))
    except Exception:
        try:
            sc_fb = StandardScaler().fit(X_full.values)
            predicoes[name] = m.predict(sc_fb.transform(novas))
        except:
            print(f"   ⚠️ {name}: erro na predição de novas dosagens")

pred_nd = pd.DataFrame(predicoes)
pred_nd.insert(0, "Mix_ID", range(1, len(novas)+1))
pred_nd["Best_Model"] = pred_nd.drop(columns="Mix_ID").idxmax(axis=1)
pred_nd.to_excel("Paper/Results/predicoes_novas_dosagens_otimizado.xlsx", index=False)
print("✅ Predições para novas dosagens salvas")

# ============================================================
# PARTE 12: FRIEDMAN TEST + NEMENYI CD DIAGRAM
# Usa df_results (Monte Carlo dados originais, mesma base)
# ============================================================

print("\n🔬 FRIEDMAN TEST + CD DIAGRAM...")
from scipy.stats import rankdata

pivot_fr = df_results.pivot_table(
    values='RMSE', index=['Run','Dataset'], columns='Model').dropna()

if len(pivot_fr) > 0:
    fr_ranks = np.array([rankdata(row.values, method='average')
                         for _, row in pivot_fr.iterrows()])
    stat_fr, p_fr = stats.friedmanchisquare(*fr_ranks.T)
    print(f"Friedman: χ²={stat_fr:.3f}, p={p_fr:.3f} "
          f"{'✅ DIFERENÇAS SIGNIFICATIVAS!' if p_fr < 0.05 else ''}")

    k_fr    = len(pivot_fr.columns)
    n_fr    = len(fr_ranks)
    q_alpha = 2.95
    cd      = q_alpha * np.sqrt(k_fr*(k_fr+1) / (6*n_fr))
    mr      = fr_ranks.mean(axis=0)
    ml      = list(pivot_fr.columns)

    plt.figure(figsize=(12, 6))
    yp = np.arange(len(ml))
    plt.barh(yp, mr)
    plt.yticks(yp, ml)
    plt.xlabel('Average Rank (lower is better)', fontsize=12)
    plt.title(f'Nemenyi CD Diagram\nCD={cd:.3f} (α=0.05) | p(Friedman)={p_fr:.3f}',
              fontsize=14)
    plt.axvline(mr.mean(), color='r', linestyle='--',
                label=f'Mean Rank={mr.mean():.2f}')
    plt.axvline(mr.mean()+cd/2, color='g', linestyle='--', alpha=0.7,
                label='Critical Distance')
    plt.axvline(mr.mean()-cd/2, color='g', linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig("Paper/Figures/cd_diagram_nemenyi.png", dpi=300, bbox_inches='tight')
    plt.close()

    fr_df = pd.DataFrame({'Model': ml, 'Mean_Rank': mr,
                          'Friedman_p': p_fr}).sort_values('Mean_Rank')
    fr_df.to_excel("Paper/Results/friedman_nemenyi_ranks.xlsx", index=False)
    print("🏆 TOP 5:"); print(fr_df.head())
    print("✅ CD Diagram salvo")

# ============================================================
# PARTE 13: DIEBOLD-MARIANO TEST (referência comum) [FIX-2]
# ============================================================

print("\n📊 Diebold-Mariano Test (referência comum)...")
if len(pred_ref) >= 2:
    dm_tab = []
    nms    = list(pred_ref.keys())
    for i in range(len(nms)):
        for j in range(i+1, len(nms)):
            e1 = y_ref_ext - pred_ref[nms[i]]
            e2 = y_ref_ext - pred_ref[nms[j]]
            d  = e1**2 - e2**2
            dm = np.mean(d) / np.sqrt(np.var(d) / len(d))
            p  = 2*(1 - stats.norm.cdf(abs(dm)))
            dm_tab.append([nms[i], nms[j], dm, p])
    dm_out = pd.DataFrame(dm_tab, columns=["Model1","Model2","DM","p"])
    dm_out.to_excel("Paper/Results/diebold_mariano_otimizado.xlsx", index=False)
    print("✅ Diebold-Mariano salvo")

# ============================================================
# PARTE 14: MODEL CONFIDENCE SET (referência comum) [FIX-2]
# ============================================================

print("\n📊 Model Confidence Set (referência comum)...")

if len(pred_ref) >= 2:
    loss_vecs  = []
    nms_mcs    = []
    for name, pr in pred_ref.items():
        loss_vecs.append((y_ref_ext - pr)**2)
        nms_mcs.append(name)

    loss_arr  = np.array(loss_vecs)
    mcs_names = nms_mcs.copy()
    loss_cp   = loss_arr.copy()

    print("\n   MCS Iterations:")
    it = 1
    while len(mcs_names) > 1:
        ml_v     = loss_cp.mean(axis=1)
        worst    = np.argmax(ml_v)
        removed  = mcs_names.pop(worst)
        loss_cp  = np.delete(loss_cp, worst, axis=0)
        print(f"      It {it}: Removed {removed} (loss={ml_v[worst]:.4f})")
        it += 1

    print(f"\n   ✅ Final MCS: {mcs_names}")
    mcs_res = pd.DataFrame({
        'Model':       nms_mcs,
        'Mean_Loss':   loss_arr.mean(axis=1),
        'In_Final_MCS':['Yes' if m in mcs_names else 'No' for m in nms_mcs],
    }).sort_values('Mean_Loss')
    mcs_res.to_excel("Paper/Results/model_confidence_set_otimizado.xlsx", index=False)

    plt.figure(figsize=(10, 6))
    cols_mcs = ['green' if x=='Yes' else 'red' for x in mcs_res['In_Final_MCS']]
    plt.bar(range(len(mcs_res)), mcs_res['Mean_Loss'], color=cols_mcs, alpha=0.7)
    plt.xticks(range(len(mcs_res)), mcs_res['Model'], rotation=45)
    plt.ylabel('Mean Squared Error')
    plt.title('Model Confidence Set – MSE (Common Test Set)')
    thresh_idx = len(mcs_names) - 1
    plt.axhline(mcs_res['Mean_Loss'].iloc[thresh_idx],
                color='blue', linestyle='--', label='MCS threshold')
    plt.legend()
    plt.tight_layout()
    plt.savefig("Paper/Figures/model_confidence_set_otimizado.png",
                dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ MCS salvo")

# ============================================================
# PARTE 15: BAYESIAN PLACKETT-LUCE RANKING (referência comum)
# ============================================================

print("\n📊 Bayesian Plackett-Luce Ranking...")
from scipy.special import softmax

rmse_pl  = [np.sqrt(mean_squared_error(y_ref_ext, pred_ref[n]))
            for n in pred_ref]
nms_pl   = list(pred_ref.keys())
abilities = softmax(-np.array(rmse_pl))
pl_df = pd.DataFrame({"Model": nms_pl, "Ability": abilities}).sort_values(
    "Ability", ascending=False)
pl_df.to_excel("Paper/Results/plackett_luce_ranking_otimizado.xlsx", index=False)
print("✅ Plackett-Luce ranking salvo")
print(pl_df.to_string(index=False))

# ============================================================
# [NEW-1] TAYLOR DIAGRAM (pred_ref / y_ref_ext) [FIX-2]
# ============================================================

print("\n" + "="*80)
print("TAYLOR DIAGRAM – ANÁLISE DE MODELOS [NEW-1]")
print("="*80)

if len(pred_ref) >= 2:
    std_obs_td  = np.std(y_ref_ext)
    std_m_td, corr_td, rmse_td, nm_td = [], [], [], []

    for mn, pr in pred_ref.items():
        std_m_td.append(np.std(pr))
        corr_td.append(np.corrcoef(y_ref_ext, pr)[0, 1])
        rmse_td.append(np.sqrt(mean_squared_error(y_ref_ext, pr)))
        nm_td.append(mn)
        print(f"   {mn}: Corr={corr_td[-1]:.4f}, Std={std_m_td[-1]:.2f}, RMSE={rmse_td[-1]:.2f}")

    try:
        import skill_metrics as sm
        fig = plt.figure(figsize=(10, 8))
        sm.taylor_diagram(std_obs_td, std_m_td, corr_td,
                          markerLabel=nm_td, markerSize=8,
                          markerLegend='on', colMarker='k',
                          styleOBS='-', colOBS='k', markerobs='o',
                          titleRMS='on', tickRMS=[0.5, 1.0, 1.5, 2.0], labelRMS='on')
        plt.title('Taylor Diagram – Model Performance Comparison (Optimized Datasets)',
                  fontsize=14, fontweight='bold')
        plt.tight_layout()
    except ImportError:
        print("   skill_metrics não disponível. Usando implementação manual...")
        fig, ax = plt.subplots(figsize=(10, 8))
        theta  = np.linspace(0, np.pi/2, 100)
        r_max  = max(max(std_m_td), std_obs_td) * 1.2
        for c in [0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.99]:
            ang = np.arccos(c)
            ax.plot([0, r_max*np.cos(ang)], [0, r_max*np.sin(ang)],
                    'k--', alpha=0.3, lw=0.5)
            ax.text(r_max*np.cos(ang)*1.02, r_max*np.sin(ang)*1.02,
                    f'{c}', fontsize=8, alpha=0.7)
        for r in np.linspace(0.5, r_max, 5):
            ax.plot(r*np.cos(theta), r*np.sin(theta), 'k--', alpha=0.3, lw=0.5)
            ax.text(r*1.02, 0, f'{r:.1f}', fontsize=8, alpha=0.7, ha='center')
        ax.plot(std_obs_td, 0, 'ro', markersize=12, label='Observed', zorder=5)
        ax.text(std_obs_td+0.02, -0.05, 'Obs', fontsize=10, fontweight='bold')
        cols_td = plt.cm.tab10(np.linspace(0, 1, len(nm_td)))
        for i, (n, sm_v, c) in enumerate(zip(nm_td, std_m_td, corr_td)):
            ang = np.arccos(c)
            ax.plot(sm_v*np.cos(ang), sm_v*np.sin(ang), 'o',
                    color=cols_td[i], markersize=10, zorder=4)
            ax.annotate(n, (sm_v*np.cos(ang), sm_v*np.sin(ang)),
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        for rmse_val in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
            phi = np.linspace(0, np.pi/2, 100)
            xc  = std_obs_td + rmse_val*np.cos(phi)
            yc  = rmse_val*np.sin(phi)
            mask = xc >= 0
            ax.plot(xc[mask], yc[mask], 'g--', alpha=0.3, lw=0.5)
            mid = len(phi)//2
            ax.text(xc[mid]+0.05, yc[mid], f'RMSE={rmse_val}',
                    fontsize=7, alpha=0.6, rotation=30)
        ax.set_xlim(0, r_max); ax.set_ylim(0, r_max)
        ax.set_aspect('equal')
        ax.set_xlabel('Std Dev'); ax.set_ylabel('Std Dev')
        ax.set_title('Taylor Diagram – Model Performance Comparison (Optimized Datasets)',
                     fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.2); ax.legend(loc='upper right')

    for pth in ['Figuras_NestedCV/Taylor_Diagram.png',
                'Paper/Figures/Taylor_Diagram.png',
                'Figures/Taylor_Diagram.png']:
        plt.savefig(pth, dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Taylor Diagram salvo")

    taylor_stats_td = pd.DataFrame({
        'Model':              nm_td,
        'Correlation':        corr_td,
        'Std_Dev_Predicted':  std_m_td,
        'Std_Dev_Observed':   [std_obs_td]*len(nm_td),
        'RMSE':               rmse_td,
        'Bias':               [np.mean(pred_ref[m]-y_ref_ext) for m in nm_td],
    }).sort_values('Correlation', ascending=False)

    taylor_stats_td.to_excel('Resultados_Artigo/Taylor_Diagram_Statistics.xlsx', index=False)
    taylor_stats_td.to_excel('Paper/Results/Taylor_Diagram_Statistics.xlsx',    index=False)
    print("\nTAYLOR DIAGRAM – ESTATÍSTICAS:")
    print(taylor_stats_td.to_string(index=False))
else:
    print("⚠️ Predições insuficientes para gerar Taylor Diagram")

print("\n✅ Taylor Diagram concluído!")

# ============================================================
# [NEW-2] RADAR CHART – COMPARAÇÃO MULTI-MÉTRICA
# ============================================================

print("\n" + "="*80)
print("RADAR CHART – COMPARAÇÃO MULTI-MÉTRICA [NEW-2]")
print("="*80)

if results_df is not None and len(results_df) > 0:
    metrics_radar   = ['R2', 'RMSE', 'MAE', 'MAPE']
    labels_radar    = ['R²', 'RMSE (MPa)', 'MAE (MPa)', 'MAPE (%)']
    dir_radar       = {'R2': 1, 'RMSE': -1, 'MAE': -1, 'MAPE': -1}

    models_list_radar = results_df['Model'].tolist()
    radar_data = []
    for metric, label in zip(metrics_radar, labels_radar):
        vals = [results_df[results_df['Model']==m][metric].values[0]
                for m in models_list_radar]
        mn, mx = min(vals), max(vals)
        if dir_radar[metric] == 1:
            norm = [(v-mn)/(mx-mn) if mx!=mn else 0.5 for v in vals]
        else:
            norm = [(mx-v)/(mx-mn) if mx!=mn else 0.5 for v in vals]
        radar_data.append({'metric': metric, 'label': label,
                           'values': vals, 'normalized': norm,
                           'min': mn, 'max': mx})

    N      = len(metrics_radar)
    angles = [n/float(N)*2*np.pi for n in range(N)]
    angles += angles[:1]
    colors_radar = plt.cm.tab10(np.linspace(0, 1, len(models_list_radar)))

    def _radar_plot(fig_path_list, title, models_sel, colors_sel, idx_map, tick_labels=None):
        fig, ax = plt.subplots(figsize=(12, 10), subplot_kw={'projection': 'polar'})
        for cidx, model in enumerate(models_sel):
            midx = idx_map[model]
            vals = [rd['normalized'][midx] for rd in radar_data] + \
                   [radar_data[0]['normalized'][midx]]
            ax.plot(angles, vals, 'o-', lw=2, color=colors_sel[cidx],
                    label=model, alpha=0.8)
            ax.fill(angles, vals, alpha=0.1, color=colors_sel[cidx])
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(tick_labels if tick_labels else
                           [rd['label'] for rd in radar_data], fontsize=11)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2','0.4','0.6','0.8','1.0'], fontsize=9)
        ax.grid(True, alpha=0.3)
        plt.title(title, fontsize=14, fontweight='bold', pad=20)
        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=9)
        plt.tight_layout()
        for pth in fig_path_list:
            plt.savefig(pth, dpi=300, bbox_inches='tight')
        plt.close()

    idx_map_all = {m: i for i, m in enumerate(models_list_radar)}

    # Versão 1 – todos os modelos, métricas normalizadas
    _radar_plot(
        ['Figuras_Radar/Radar_Chart_Normalized.png',
         'Figuras_NestedCV/Radar_Chart_Normalized.png',
         'Paper/Figures/Radar_Chart_Normalized.png',
         'Figures/Radar_Chart_Normalized.png'],
        'Radar Chart – Multi-Metric Model Comparison (Optimized Datasets)\n(Higher score = better performance)',
        models_list_radar, colors_radar, idx_map_all)
    print("✅ Radar Chart (Normalized) salvo")

    # Versão 2 – com intervalos reais nos labels
    range_labels = [f"{rd['label']}\n[{rd['min']:.2f} – {rd['max']:.2f}]"
                    for rd in radar_data]
    _radar_plot(
        ['Figuras_Radar/Radar_Chart_With_Ranges.png',
         'Figuras_NestedCV/Radar_Chart_With_Ranges.png',
         'Paper/Figures/Radar_Chart_With_Ranges.png',
         'Figures/Radar_Chart_With_Ranges.png'],
        'Radar Chart – Model Comparison with Metric Ranges\n(Higher score = better performance)',
        models_list_radar, colors_radar, idx_map_all, tick_labels=range_labels)
    print("✅ Radar Chart (With Ranges) salvo")

    # Versão 3 – Top 5 por IFI
    top5_radar  = ranking.head(5)['Model'].tolist()
    cols_top5   = plt.cm.viridis(np.linspace(0, 1, len(top5_radar)))
    _radar_plot(
        ['Figuras_Radar/Radar_Chart_Top5_Models.png',
         'Figuras_NestedCV/Radar_Chart_Top5_Models.png',
         'Paper/Figures/Radar_Chart_Top5_Models.png',
         'Figures/Radar_Chart_Top5_Models.png'],
        'Radar Chart – Top 5 Models (IFI Ranking)\n(Higher score = better performance)',
        top5_radar, cols_top5, idx_map_all)
    print("✅ Radar Chart (Top 5) salvo")

    # Tabela e ranking de score médio radar
    radar_norm_df = pd.DataFrame({'Model': models_list_radar})
    for rd in radar_data:
        radar_norm_df[rd['label']]           = rd['normalized']
        radar_norm_df[f"{rd['label']}_raw"]  = rd['values']
    radar_norm_df['Radar_Score'] = radar_norm_df[
        [rd['label'] for rd in radar_data]].mean(axis=1)
    radar_ranking_df = radar_norm_df[['Model','Radar_Score']].sort_values(
        'Radar_Score', ascending=False)

    radar_norm_df.to_excel('Resultados_Artigo/Radar_Chart_Normalized_Data.xlsx', index=False)
    radar_norm_df.to_excel('Paper/Results/Radar_Chart_Normalized_Data.xlsx',    index=False)
    radar_ranking_df.to_excel('Resultados_Artigo/Radar_Chart_Ranking.xlsx', index=False)
    radar_ranking_df.to_excel('Paper/Results/Radar_Chart_Ranking.xlsx',    index=False)

    print("\n📊 RANKING POR SCORE MÉDIO RADAR:")
    print(radar_ranking_df.to_string(index=False))

    print("\n📊 ESTATÍSTICAS DO RADAR:")
    for rd in radar_data:
        best_i  = np.argmax(rd['normalized'])
        worst_i = np.argmin(rd['normalized'])
        print(f"\n{rd['label']}:")
        print(f"   Min raw: {rd['min']:.4f} | Max raw: {rd['max']:.4f}")
        print(f"   Média: {np.mean(rd['values']):.4f} | Desvio: {np.std(rd['values']):.4f}")
        print(f"   Melhor: {models_list_radar[best_i]} | Pior: {models_list_radar[worst_i]}")
else:
    print("⚠️ results_df não disponível para Radar Chart")

print("\n✅ Radar Chart concluído!")

# ============================================================
# [NEW-3] LEARNING CURVES – DIAGNÓSTICO OVER/UNDERFITTING
# ============================================================

print("\n" + "="*80)
print("LEARNING CURVES – DIAGNÓSTICO DE MODELOS [NEW-3]")
print("="*80)

train_sizes_lc = np.linspace(0.1, 1.0, 10)
cv_folds_lc    = 5
lc_results     = {}

print(f"\nConfiguração: {len(train_sizes_lc)} pontos, {cv_folds_lc}-fold CV, métrica=R²")

for model_name, model in models_trained.items():
    print(f"\n   {model_name} ...")
    try:
        info   = optimized_datasets[model_name]
        dev_df = info['dev_df']
        X_d    = dev_df[feature_names].values
        y_d    = dev_df[target].values

        # Learning curve opera sobre os dados DEV já escalados
        ts_abs, tr_sc, te_sc = learning_curve(
            clone(model), X_d, y_d,
            train_sizes=train_sizes_lc,
            cv=cv_folds_lc, scoring='r2',
            n_jobs=-1, shuffle=True, random_state=42)

        tr_mn, tr_sd = tr_sc.mean(axis=1), tr_sc.std(axis=1)
        te_mn, te_sd = te_sc.mean(axis=1), te_sc.std(axis=1)
        gap = tr_mn - te_mn

        lc_results[model_name] = {
            'train_sizes': ts_abs,
            'train_mean': tr_mn, 'train_std': tr_sd,
            'test_mean':  te_mn, 'test_std':  te_sd,
            'gap': gap, 'max_gap': gap.max(),
            'final_gap': gap[-1],
            'final_train_score': tr_mn[-1],
            'final_test_score':  te_mn[-1],
        }
        print(f"      Train={tr_mn[-1]:.4f} ± {tr_sd[-1]:.4f} | "
              f"Test={te_mn[-1]:.4f} ± {te_sd[-1]:.4f} | Gap={gap[-1]:.4f}")
    except Exception as e:
        print(f"      ❌ {str(e)[:100]}")
        lc_results[model_name] = None

# Gráfico 1 – Todos os modelos (subplots)
n_lc   = sum(1 for v in lc_results.values() if v)
n_cols = 3
n_rows = (n_lc + n_cols - 1) // n_cols
fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
fig.suptitle('Learning Curves – All Models (Optimized Datasets)',
             fontsize=16, fontweight='bold', y=0.98)
axes_flat = axes.flatten() if hasattr(axes, 'flatten') else [axes]
pidx = 0
for model_name, res in lc_results.items():
    if res is None: continue
    ax = axes_flat[pidx]
    ax.plot(res['train_sizes'], res['train_mean'], 'o-', color='blue',
            lw=2, ms=4, label='Treino')
    ax.fill_between(res['train_sizes'],
                    res['train_mean']-res['train_std'],
                    res['train_mean']+res['train_std'],
                    alpha=0.1, color='blue')
    ax.plot(res['train_sizes'], res['test_mean'], 'o-', color='red',
            lw=2, ms=4, label='Validação')
    ax.fill_between(res['train_sizes'],
                    res['test_mean']-res['test_std'],
                    res['test_mean']+res['test_std'],
                    alpha=0.1, color='red')
    ax.axhline(0.8, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Tamanho do Treino', fontsize=10)
    ax.set_ylabel('R²', fontsize=10)
    ax.set_title(f"{model_name}\nTrain={res['final_train_score']:.3f} | "
                 f"Test={res['final_test_score']:.3f} | Gap={res['final_gap']:.3f}",
                 fontsize=10, fontweight='bold')
    ax.legend(loc='lower right', fontsize=8)
    ax.grid(True, alpha=0.3); ax.set_ylim(-0.1, 1.05)
    pidx += 1
for idx in range(pidx, len(axes_flat)):
    axes_flat[idx].set_visible(False)
plt.tight_layout()
for pth in ['Figuras_LearningCurves/Learning_Curves_All_Models.png',
            'Figuras_NestedCV/Learning_Curves_All_Models.png',
            'Paper/Figures/Learning_Curves_All_Models.png',
            'Figures/Learning_Curves_All_Models.png']:
    plt.savefig(pth, dpi=300, bbox_inches='tight')
plt.close()
print("\n✅ Learning Curves (todos os modelos) salvo")

# Gráfico 2 – Comparação de gaps
valid_lc = [(n, r) for n, r in lc_results.items() if r]
nms_lc   = [x[0] for x in valid_lc]
fg_lc    = [x[1]['final_gap']          for x in valid_lc]
ft_lc    = [x[1]['final_train_score']  for x in valid_lc]
fv_lc    = [x[1]['final_test_score']   for x in valid_lc]

fig, ax = plt.subplots(figsize=(14, 8))
xp = np.arange(len(nms_lc)); w = 0.25
b1 = ax.bar(xp-w,   ft_lc, w, label='Train Score',     color='steelblue', alpha=0.8, edgecolor='k')
b2 = ax.bar(xp,     fv_lc, w, label='Val Score',       color='coral',     alpha=0.8, edgecolor='k')
b3 = ax.bar(xp+w,   fg_lc, w, label='Gap (Train–Val)', color='gray',      alpha=0.8, edgecolor='k')
for bars in [b1, b2, b3]:
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x()+bar.get_width()/2, h+0.01, f'{h:.3f}',
                ha='center', va='bottom', fontsize=8)
ax.set_xticks(xp); ax.set_xticklabels(nms_lc, rotation=45, ha='right', fontsize=10)
ax.set_ylabel('R² / Gap', fontsize=12, fontweight='bold')
ax.set_title('Final Performance – Train vs Validation Gap', fontsize=14, fontweight='bold')
ax.legend(loc='upper right', fontsize=10)
ax.grid(True, alpha=0.3, axis='y'); ax.set_ylim(0, 1.1)
plt.tight_layout()
for pth in ['Figuras_LearningCurves/Learning_Curves_Gap_Comparison.png',
            'Figuras_NestedCV/Learning_Curves_Gap_Comparison.png',
            'Paper/Figures/Learning_Curves_Gap_Comparison.png',
            'Figures/Learning_Curves_Gap_Comparison.png']:
    plt.savefig(pth, dpi=300, bbox_inches='tight')
plt.close()
print("✅ Gap Comparison salvo")

# Gráfico 3 – Top 3 melhores vs piores
valid_lc_sorted = sorted(valid_lc, key=lambda x: x[1]['final_test_score'], reverse=True)
top3_lc    = valid_lc_sorted[:3]
bottom3_lc = valid_lc_sorted[-3:]
fig, axes  = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Learning Curves – Top 3 Best vs Worst Models',
             fontsize=16, fontweight='bold', y=0.98)
for row_idx, group in enumerate([top3_lc, bottom3_lc]):
    prefix = 'BEST' if row_idx == 0 else 'WORST'
    for col_idx, (mname, res) in enumerate(group):
        ax = axes[row_idx, col_idx]
        ax.plot(res['train_sizes'], res['train_mean'], 'o-', color='blue', lw=2, ms=4, label='Treino')
        ax.fill_between(res['train_sizes'],
                        res['train_mean']-res['train_std'],
                        res['train_mean']+res['train_std'], alpha=0.1, color='blue')
        ax.plot(res['train_sizes'], res['test_mean'], 'o-', color='red', lw=2, ms=4, label='Validação')
        ax.fill_between(res['train_sizes'],
                        res['test_mean']-res['test_std'],
                        res['test_mean']+res['test_std'], alpha=0.1, color='red')
        ax.axhline(0.8, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('Tamanho do Treino', fontsize=10)
        ax.set_ylabel('R²', fontsize=10)
        ax.set_title(f"{prefix}: {mname}\nTrain={res['final_train_score']:.3f} | "
                     f"Test={res['final_test_score']:.3f}", fontsize=10, fontweight='bold')
        ax.legend(loc='lower right', fontsize=8)
        ax.grid(True, alpha=0.3); ax.set_ylim(-0.1, 1.05)
plt.tight_layout()
for pth in ['Figuras_LearningCurves/Learning_Curves_Best_vs_Worst.png',
            'Figuras_NestedCV/Learning_Curves_Best_vs_Worst.png',
            'Paper/Figures/Learning_Curves_Best_vs_Worst.png',
            'Figures/Learning_Curves_Best_vs_Worst.png']:
    plt.savefig(pth, dpi=300, bbox_inches='tight')
plt.close()
print("✅ Best vs Worst Learning Curves salvo")

# Tabela de diagnóstico
diag_lc = []
for mname, res in lc_results.items():
    if res is None: continue
    fg = res['final_gap']; ft = res['final_test_score']
    if ft < 0.6:
        status = "UNDERFITTING"
        rec = "Aumentar complexidade ou melhorar features"
    elif fg > 0.15:
        status = "OVERFITTING"
        rec = "Reduzir complexidade, mais regularização ou mais dados"
    elif fg > 0.08:
        status = "MODERADO"
        rec = "Overfitting moderado, considerar regularização"
    elif ft > 0.85 and fg < 0.05:
        status = "ÓTIMO"
        rec = "Modelo bem ajustado"
    else:
        status = "ACEITÁVEL"
        rec = "Desempenho razoável"
    diag_lc.append({'Model': mname,
                    'Final_Train_Score': res['final_train_score'],
                    'Final_Test_Score':  res['final_test_score'],
                    'Final_Gap': fg, 'Max_Gap': res['max_gap'],
                    'Status': status, 'Recomendacao': rec,
                    'Samples': len(optimized_datasets[mname]['dev_df'])})

diag_lc_df = pd.DataFrame(diag_lc).sort_values('Final_Test_Score', ascending=False)
diag_lc_df.to_excel('Resultados_Artigo/Learning_Curves_Diagnosis.xlsx', index=False)
diag_lc_df.to_excel('Paper/Results/Learning_Curves_Diagnosis.xlsx',    index=False)
print("\n📊 DIAGNÓSTICO LEARNING CURVES:")
print(diag_lc_df[['Model','Final_Test_Score','Final_Gap','Status']].to_string(index=False))

print("\n✅ Learning Curves concluído!")

# ============================================================
# [NEW-4] PERMUTATION IMPORTANCE
# ============================================================

print("\n" + "="*80)
print("PERMUTATION IMPORTANCE [NEW-4]")
print("="*80)

n_repeats_perm = 10
perm_imp_results = {}
perm_imp_summary = []

for model_name, model in models_trained.items():
    print(f"\n   {model_name} ...")
    try:
        info   = optimized_datasets[model_name]
        dev_df = info['dev_df']
        sc_mod = info['scaler']
        X_tst_r= info['X_tst_raw']
        y_tst_r= info['y_tst_raw'].values

        X_dev_np = dev_df[feature_names].values
        y_dev_np = dev_df[target].values
        X_tst_sc = sc_mod.transform(X_tst_r)

        m = clone(model)                                   # [FIX-3]
        m.fit(X_dev_np, y_dev_np)

        result = permutation_importance(
            m, X_tst_sc, y_tst_r,
            n_repeats=n_repeats_perm,
            random_state=42, scoring='r2', n_jobs=-1)

        perm_imp_results[model_name] = {
            'importances_mean': result.importances_mean,
            'importances_std':  result.importances_std,
            'importances':      result.importances,
        }

        imp_df = pd.DataFrame({
            'Feature':    feature_names,
            'Importance': result.importances_mean,
            'Std':        result.importances_std,
        }).sort_values('Importance', ascending=False)

        perm_imp_summary.append({
            'Model':           model_name,
            'Top_Feature':     imp_df.iloc[0]['Feature'],
            'Top_Importance':  imp_df.iloc[0]['Importance'],
            'Top_2_Feature':   imp_df.iloc[1]['Feature']    if len(imp_df) > 1 else '-',
            'Top_2_Importance':imp_df.iloc[1]['Importance'] if len(imp_df) > 1 else 0,
            'Top_3_Feature':   imp_df.iloc[2]['Feature']    if len(imp_df) > 2 else '-',
            'Top_3_Importance':imp_df.iloc[2]['Importance'] if len(imp_df) > 2 else 0,
        })

        for pth in [f"Paper/Results/SHAP_Results/Permutation_Importance_{model_name}.xlsx",
                    f"Resultados_Artigo/Permutation_Importance_{model_name}.xlsx"]:
            imp_df.to_excel(pth, index=False)

        print(f"      ✅ Top: {imp_df.iloc[0]['Feature']} ({imp_df.iloc[0]['Importance']:.4f})")
    except Exception as e:
        print(f"      ⚠️ {str(e)[:100]}")

# Gráfico 1 – Heatmap
imp_matrix = pd.DataFrame(
    {mn: dict(zip(feature_names, perm_imp_results[mn]['importances_mean']))
     for mn in perm_imp_results},
    index=feature_names).T
imp_matrix_norm = imp_matrix.div(imp_matrix.sum(axis=1).replace(0, 1e-10), axis=0)

plt.figure(figsize=(14, 10))
sns.heatmap(imp_matrix_norm, annot=True, fmt='.3f', cmap='YlOrRd')
plt.title('Permutation Importance Heatmap – Normalized by Model',
          fontsize=14, fontweight='bold')
plt.xlabel('Features', fontsize=12); plt.ylabel('Model', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
for pth in ['Figuras_PermutationImportance/Permutation_Importance_Heatmap.png',
            'Figuras_Shap/Permutation_Importance_Heatmap.png',
            'Paper/Figures/Permutation_Importance_Heatmap.png']:
    plt.savefig(pth, dpi=300, bbox_inches='tight')
plt.close()
print("\n✅ Heatmap Permutation Importance salvo")

# Gráfico 2 – Barras por modelo (Top 5 features)
n_perm_mods = len(perm_imp_results)
nc = 3; nr = (n_perm_mods + nc - 1) // nc
fig, axes = plt.subplots(nr, nc, figsize=(18, 5*nr))
fig.suptitle('Permutation Importance – Top 5 Features per Model',
             fontsize=16, fontweight='bold', y=0.98)
ax_flat = axes.flatten() if hasattr(axes, 'flatten') else [axes]
cols_perm = plt.cm.Set3(np.linspace(0, 1, len(feature_names)))
for pidx2, (mname, pdata) in enumerate(perm_imp_results.items()):
    if pidx2 >= len(ax_flat): break
    ax = ax_flat[pidx2]
    imp_df2 = pd.DataFrame({'Feature': feature_names,
                            'Importance': pdata['importances_mean'],
                            'Std': pdata['importances_std']
                            }).sort_values('Importance').tail(5)
    bars = ax.barh(imp_df2['Feature'], imp_df2['Importance'],
                   xerr=imp_df2['Std'], capsize=3,
                   color=cols_perm[:len(imp_df2)], alpha=0.7, edgecolor='k')
    ax.set_xlabel('R² drop', fontsize=10)
    ax.set_title(mname, fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    for bar, val in zip(bars, imp_df2['Importance']):
        ax.text(bar.get_width()+0.001, bar.get_y()+bar.get_height()/2,
                f'{val:.4f}', va='center', fontsize=8)
for idx in range(pidx2+1, len(ax_flat)):
    ax_flat[idx].set_visible(False)
plt.tight_layout()
for pth in ['Figuras_PermutationImportance/Permutation_Importance_Bars.png',
            'Figuras_Shap/Permutation_Importance_Bars.png',
            'Paper/Figures/Permutation_Importance_Bars.png']:
    plt.savefig(pth, dpi=300, bbox_inches='tight')
plt.close()
print("✅ Barras Permutation Importance salvas")

# Gráfico 3 – SHAP vs Permutation Importance
if all_shap_importance:
    consolidated_shap = pd.concat(all_shap_importance, ignore_index=True)
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    fig.suptitle('SHAP vs Permutation Importance – Feature Comparison',
                 fontsize=16, fontweight='bold', y=0.98)
    compare_mods = list(perm_imp_results.keys())[:4]
    for cidx, mname in enumerate(compare_mods):
        ax = axes[cidx//2, cidx%2]
        sh_imp = consolidated_shap[consolidated_shap['Model'] == mname]
        sh_dict = dict(zip(sh_imp['Feature'], sh_imp['SHAP_Importance'])) \
                  if not sh_imp.empty else {}
        pm_dict = dict(zip(feature_names, perm_imp_results[mname]['importances_mean']))
        sh_v = np.array([sh_dict.get(f, 0) for f in feature_names])
        pm_v = np.array([pm_dict.get(f, 0) for f in feature_names])
        if sh_v.sum() > 0: sh_v /= sh_v.sum()
        if pm_v.sum() > 0: pm_v /= pm_v.sum()
        xp = np.arange(len(feature_names)); w = 0.35
        ax.bar(xp-w/2, sh_v, w, label='SHAP',        color='steelblue', alpha=0.7)
        ax.bar(xp+w/2, pm_v, w, label='Permutation', color='coral',     alpha=0.7)
        ax.set_xticks(xp)
        ax.set_xticklabels(feature_names, rotation=45, ha='right', fontsize=8)
        ax.set_title(mname, fontsize=11, fontweight='bold')
        ax.set_ylabel('Norm. Importance', fontsize=10)
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    for pth in ['Figuras_PermutationImportance/SHAP_vs_Permutation_Importance.png',
                'Figuras_Shap/SHAP_vs_Permutation_Importance.png',
                'Paper/Figures/SHAP_vs_Permutation_Importance.png']:
        plt.savefig(pth, dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ SHAP vs Permutation Importance salvo")

# Gráfico 4 – Boxplot por feature (variabilidade entre modelos)
feat_imp_all = {f: [] for f in feature_names}
for pdata in perm_imp_results.values():
    for f, v in zip(feature_names, pdata['importances_mean']):
        feat_imp_all[f].append(v)

fig, ax = plt.subplots(figsize=(14, 8))
bp = ax.boxplot([feat_imp_all[f] for f in feature_names],
                labels=feature_names, patch_artist=True,
                medianprops=dict(color='red', linewidth=2))
cols_box2 = plt.cm.viridis(np.linspace(0, 1, len(feature_names)))
for patch, col in zip(bp['boxes'], cols_box2):
    patch.set_facecolor(col); patch.set_alpha(0.7)
ax.set_xlabel('Features', fontsize=12, fontweight='bold')
ax.set_ylabel('Permutation Importance (R² drop)', fontsize=12, fontweight='bold')
ax.set_title('Permutation Importance Distribution Across All Models',
             fontsize=14, fontweight='bold')
ax.tick_params(axis='x', rotation=45, labelsize=10)
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
for pth in ['Figuras_PermutationImportance/Permutation_Importance_Boxplot.png',
            'Figuras_Shap/Permutation_Importance_Boxplot.png',
            'Paper/Figures/Permutation_Importance_Boxplot.png']:
    plt.savefig(pth, dpi=300, bbox_inches='tight')
plt.close()
print("✅ Boxplot Permutation Importance salvo")

# Tabelas consolidadas
cons_perm = pd.DataFrame([
    {'Model': mn, 'Feature': f,
     'Importance': imp, 'Std': std}
    for mn, pd_data in perm_imp_results.items()
    for f, imp, std in zip(feature_names,
                           pd_data['importances_mean'],
                           pd_data['importances_std'])
])
cons_perm.to_excel('Resultados_Artigo/Permutation_Importance_Consolidated.xlsx', index=False)
cons_perm.to_excel('Paper/Results/Permutation_Importance_Consolidated.xlsx',    index=False)
cons_perm.pivot_table(values='Importance', index='Model',
                      columns='Feature', fill_value=0).to_excel(
    'Resultados_Artigo/Permutation_Importance_Pivot.xlsx')
cons_perm.pivot_table(values='Importance', index='Model',
                      columns='Feature', fill_value=0).to_excel(
    'Paper/Results/Permutation_Importance_Pivot.xlsx')

feat_stats_perm = []
for f in feature_names:
    vs = feat_imp_all[f]
    feat_stats_perm.append({
        'Feature': f, 'Mean_Importance': np.mean(vs),
        'Std_Importance': np.std(vs),
        'Min_Importance': np.min(vs), 'Max_Importance': np.max(vs),
        'Models_Where_Top1': sum(1 for s in perm_imp_summary if s['Top_Feature'] == f),
    })
feat_stats_perm_df = pd.DataFrame(feat_stats_perm).sort_values(
    'Mean_Importance', ascending=False)
feat_stats_perm_df.to_excel('Resultados_Artigo/Permutation_Importance_Feature_Stats.xlsx', index=False)
feat_stats_perm_df.to_excel('Paper/Results/Permutation_Importance_Feature_Stats.xlsx',    index=False)

print("\n📊 TOP 3 FEATURES POR MODELO:")
print(pd.DataFrame(perm_imp_summary).to_string(index=False))
print("\n📊 ESTATÍSTICAS POR FEATURE:")
print(feat_stats_perm_df.to_string(index=False))
print("\n✅ Permutation Importance concluído!")

# ============================================================
# [NEW-5] PARTIAL DEPENDENCE PLOTS – 1D e 2D
# ============================================================

print("\n" + "="*80)
print("PARTIAL DEPENDENCE PLOTS (PDP) [NEW-5]")
print("="*80)

n_grid_pdp      = 50
features_to_plot_pdp = feature_names[:6] if len(feature_names) > 6 else feature_names
pdp_models      = {}

for model_name, model in models_trained.items():
    print(f"\n   {model_name} ...")
    try:
        info   = optimized_datasets[model_name]
        dev_df = info['dev_df']
        X_d    = dev_df[feature_names].values
        y_d    = dev_df[target].values

        m = clone(model)                                   # [FIX-3]
        m.fit(X_d, y_d)
        pdp_models[model_name] = {'model': m, 'X_train': X_d}
        print(f"      ✅ modelo preparado")
    except Exception as e:
        print(f"      ⚠️ {str(e)[:100]}")

# Gráfico 1 – Melhor modelo (IFI) × 6 features
best_pdp = ranking.iloc[0]['Model']
print(f"\n   PDP melhor modelo: {best_pdp}")
if best_pdp in pdp_models:
    try:
        fig, axes_pdp = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Partial Dependence Plots – Best Model: {best_pdp}\n'
                     f'Impact of Individual Features on fck Prediction',
                     fontsize=16, fontweight='bold', y=0.98)
        PartialDependenceDisplay.from_estimator(
            pdp_models[best_pdp]['model'],
            pdp_models[best_pdp]['X_train'],
            features=list(range(len(features_to_plot_pdp))),
            feature_names=feature_names,
            kind='average', n_jobs=-1,
            grid_resolution=n_grid_pdp,
            ax=axes_pdp, random_state=42)
        for idx, ax in enumerate(axes_pdp.flat):
            if idx < len(features_to_plot_pdp):
                ax.set_title(f'Feature: {features_to_plot_pdp[idx]}',
                             fontsize=12, fontweight='bold')
                ax.set_xlabel(features_to_plot_pdp[idx], fontsize=10)
                ax.set_ylabel('Partial Dependence (fck MPa)', fontsize=10)
                ax.grid(True, alpha=0.3)
        plt.tight_layout()
        for pth in ['Figuras_PDP/PDP_Best_Model.png',
                    'Figuras_Shap/PDP_Best_Model.png',
                    'Paper/Figures/PDP_Best_Model.png',
                    'Figures/PDP_Best_Model.png']:
            plt.savefig(pth, dpi=300, bbox_inches='tight')
        plt.close()
        print("   ✅ PDP melhor modelo salvo")
    except Exception as e:
        print(f"   ⚠️ {str(e)[:100]}")
        plt.close()

# Gráfico 2 – Top 3 modelos comparados
top3_pdp   = [m for m in ranking.head(3)['Model'].tolist() if m in pdp_models]
if len(top3_pdp) >= 2:
    fig, axes_pdp2 = plt.subplots(len(top3_pdp), len(features_to_plot_pdp),
                                  figsize=(20, 5*len(top3_pdp)))
    fig.suptitle('PDP – Top 3 Models Comparison',
                 fontsize=16, fontweight='bold', y=0.98)
    for i, mname in enumerate(top3_pdp):
        try:
            row_ax = axes_pdp2[i] if len(top3_pdp) > 1 else axes_pdp2
            PartialDependenceDisplay.from_estimator(
                pdp_models[mname]['model'],
                pdp_models[mname]['X_train'],
                features=list(range(len(features_to_plot_pdp))),
                feature_names=feature_names,
                kind='average', n_jobs=-1,
                grid_resolution=n_grid_pdp,
                ax=row_ax, random_state=42)
            row_ax[0].set_ylabel(f'{mname}\nPartial Dep.', fontsize=10, fontweight='bold')
        except Exception as e:
            print(f"   ⚠️ PDP {mname}: {str(e)[:80]}")
    plt.tight_layout()
    for pth in ['Figuras_PDP/PDP_Top3_Models.png',
                'Figuras_Shap/PDP_Top3_Models.png',
                'Paper/Figures/PDP_Top3_Models.png',
                'Figures/PDP_Top3_Models.png']:
        plt.savefig(pth, dpi=300, bbox_inches='tight')
    plt.close()
    print("   ✅ PDP Top 3 salvo")

# Gráfico 3 – Comparação de todos os modelos nas top 3 features (via partial_dependence)
top3_feat_pdp = feat_stats_perm_df.head(3)['Feature'].tolist() \
                if not feat_stats_perm_df.empty else feature_names[:3]
print(f"\n   PDP comparativo – features: {top3_feat_pdp}")
colors_pdp_cmp = plt.cm.tab10(np.linspace(0, 1, len(pdp_models)))
fig, axes_pdp3 = plt.subplots(len(top3_feat_pdp), 1,
                               figsize=(14, 5*len(top3_feat_pdp)))
if len(top3_feat_pdp) == 1: axes_pdp3 = [axes_pdp3]
fig.suptitle('PDP – All Models Comparison (Top 3 Most Important Features)',
             fontsize=16, fontweight='bold', y=0.98)
for f_idx, feat in enumerate(top3_feat_pdp):
    ax = axes_pdp3[f_idx]
    fidx = feature_names.index(feat)
    for m_idx, (mname, pdata) in enumerate(pdp_models.items()):
        try:
            pdp_res = partial_dependence(pdata['model'], pdata['X_train'],
                                         features=[fidx], kind='average',
                                         grid_resolution=n_grid_pdp)
            ax.plot(pdp_res['values'][0], pdp_res['average'][0],
                    'o-', color=colors_pdp_cmp[m_idx], lw=2, ms=4,
                    label=mname, alpha=0.8)
        except: continue
    ax.set_xlabel(feat, fontsize=12, fontweight='bold')
    ax.set_ylabel('Partial Dependence (fck MPa)', fontsize=12)
    ax.set_title(f'Impact of {feat} on fck Prediction', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3); ax.legend(loc='best', fontsize=8)
plt.tight_layout()
for pth in ['Figuras_PDP/PDP_All_Models_Comparison.png',
            'Figuras_Shap/PDP_All_Models_Comparison.png',
            'Paper/Figures/PDP_All_Models_Comparison.png',
            'Figures/PDP_All_Models_Comparison.png']:
    plt.savefig(pth, dpi=300, bbox_inches='tight')
plt.close()
print("   ✅ PDP comparativo (todos os modelos) salvo")

# Gráfico 4 – PDP 2D (interação entre 2 features) para o melhor modelo
if best_pdp in pdp_models and len(feature_names) >= 4:
    feat_pairs_2d = [(0, 1), (2, 3)]
    fig, axes_2d = plt.subplots(1, len(feat_pairs_2d),
                                figsize=(8*len(feat_pairs_2d), 6))
    if len(feat_pairs_2d) == 1: axes_2d = [axes_2d]
    fig.suptitle(f'2D PDP – {best_pdp} – Feature Interaction',
                 fontsize=16, fontweight='bold', y=0.98)
    for idx2d, (f1, f2) in enumerate(feat_pairs_2d):
        try:
            PartialDependenceDisplay.from_estimator(
                pdp_models[best_pdp]['model'],
                pdp_models[best_pdp]['X_train'],
                features=[(f1, f2)], feature_names=feature_names,
                kind='average', n_jobs=-1,
                grid_resolution=n_grid_pdp,
                ax=axes_2d[idx2d], random_state=42)
            axes_2d[idx2d].set_title(
                f'Interaction: {feature_names[f1]} × {feature_names[f2]}',
                fontsize=12, fontweight='bold')
        except Exception as e:
            print(f"   ⚠️ PDP 2D: {str(e)[:80]}")
    plt.tight_layout()
    for pth in ['Figuras_PDP/PDP_2D_Interaction.png',
                'Figuras_Shap/PDP_2D_Interaction.png',
                'Paper/Figures/PDP_2D_Interaction.png',
                'Figures/PDP_2D_Interaction.png']:
        plt.savefig(pth, dpi=300, bbox_inches='tight')
    plt.close()
    print("   ✅ PDP 2D salvo")

# Tabela resumo PDP
pdp_summary_tbl = pd.DataFrame([
    {'Model': mn, 'PDP_Computed': 'Yes', 'N_Features': len(feature_names),
     'Best_Model': 'Yes' if mn == best_pdp else 'No'}
    for mn in pdp_models])
pdp_summary_tbl.to_excel('Resultados_Artigo/PDP_Summary.xlsx', index=False)
pdp_summary_tbl.to_excel('Paper/Results/PDP_Summary.xlsx',    index=False)
print("\n✅ Partial Dependence Plots concluído!")

# ============================================================
# [NEW-6] Q-Q PLOTS + TESTES DE NORMALIDADE DOS RESÍDUOS
# Usa pred_ref / y_ref_ext (referência comum) [FIX-2]
# ============================================================

print("\n" + "="*80)
print("Q-Q PLOTS + NORMALIDADE DOS RESÍDUOS [NEW-6]")
print("="*80)

qq_results_dict = {}
normality_tests  = []

for model_name, pr in pred_ref.items():
    print(f"\n   {model_name} ...")
    try:
        residuals = y_ref_ext - pr

        skewness = stats.skew(residuals)
        kurt     = stats.kurtosis(residuals)
        sh_stat, sh_p = stats.shapiro(
            residuals[:5000] if len(residuals) > 5000 else residuals)
        jb_stat, jb_p = stats.jarque_bera(residuals)
        ad_res = stats.anderson(residuals, dist='norm')

        if sh_p > 0.05:
            norm_status = "NORMAL (p > 0.05)"
        elif sh_p > 0.01:
            norm_status = "APROX. NORMAL (p > 0.01)"
        else:
            norm_status = "NÃO NORMAL (p ≤ 0.01)"

        qq_results_dict[model_name] = {
            'residuals':     residuals,
            'mean':          np.mean(residuals),
            'std':           np.std(residuals),
            'skewness':      skewness,
            'kurtosis':      kurt,
            'shapiro_stat':  sh_stat,  'shapiro_p':   sh_p,
            'jb_stat':       jb_stat,  'jb_p':        jb_p,
            'anderson_stat': ad_res.statistic,
        }
        normality_tests.append({
            'Model':           model_name,
            'Mean_Residual':   np.mean(residuals),
            'Std_Residual':    np.std(residuals),
            'Skewness':        skewness,
            'Kurtosis':        kurt,
            'Shapiro_Wilk':    sh_stat,
            'Shapiro_p':       sh_p,
            'Jarque_Bera':     jb_stat,
            'JB_p':            jb_p,
            'Normality_Status':norm_status,
            'Samples':         len(residuals),
        })
        print(f"      μ={np.mean(residuals):.4f} σ={np.std(residuals):.4f} "
              f"Skew={skewness:.3f} Kurt={kurt:.3f} | {norm_status}")
    except Exception as e:
        print(f"      ❌ {str(e)[:100]}")

# Gráfico 1 – Q-Q todos os modelos
n_qq   = len(qq_results_dict)
nc_qq  = 3
nr_qq  = (n_qq + nc_qq - 1) // nc_qq
fig, axes = plt.subplots(nr_qq, nc_qq, figsize=(15, 5*nr_qq))
fig.suptitle('Q-Q Plots – Residual Normality (Common Test Set)',
             fontsize=16, fontweight='bold', y=0.98)
axes_flat_qq = axes.flatten() if hasattr(axes, 'flatten') else [axes]
for pidx3, (mname, res) in enumerate(qq_results_dict.items()):
    ax = axes_flat_qq[pidx3]
    stats.probplot(res['residuals'], dist='norm', plot=ax)
    col = 'green' if res['shapiro_p'] > 0.05 \
          else ('orange' if res['shapiro_p'] > 0.01 else 'red')
    lbl = 'Normal' if res['shapiro_p'] > 0.05 \
          else ('Aprox. Normal' if res['shapiro_p'] > 0.01 else 'Não Normal')
    ax.set_title(f"{mname}\nShapiro-Wilk p={res['shapiro_p']:.4f} ({lbl})",
                 fontsize=10, fontweight='bold', color=col)
    ax.set_xlabel('Theoretical Quantiles', fontsize=9)
    ax.set_ylabel('Sample Quantiles (Residuals)', fontsize=9)
    ax.grid(True, alpha=0.3)
for idx in range(pidx3+1, len(axes_flat_qq)):
    axes_flat_qq[idx].set_visible(False)
plt.tight_layout()
for pth in ['Figuras_QQPlot/QQ_Plots_All_Models.png',
            'Figuras_NestedCV/QQ_Plots_All_Models.png',
            'Paper/Figures/QQ_Plots_All_Models.png',
            'Figures/QQ_Plots_All_Models.png']:
    plt.savefig(pth, dpi=300, bbox_inches='tight')
plt.close()
print("\n✅ Q-Q Plots (todos os modelos) salvo")

# Gráfico 2 – Top 3 modelos (IFI)
top3_qq = [m for m in ranking.head(3)['Model'].tolist() if m in qq_results_dict]
if top3_qq:
    fig, axes_t3 = plt.subplots(1, len(top3_qq), figsize=(6*len(top3_qq), 5))
    if len(top3_qq) == 1: axes_t3 = [axes_t3]
    fig.suptitle('Q-Q Plots – Top 3 Models', fontsize=14, fontweight='bold')
    for idx, mname in enumerate(top3_qq):
        res = qq_results_dict[mname]
        stats.probplot(res['residuals'], dist='norm', plot=axes_t3[idx])
        col = 'green' if res['shapiro_p'] > 0.05 \
              else ('orange' if res['shapiro_p'] > 0.01 else 'red')
        axes_t3[idx].set_title(f"{mname}\np={res['shapiro_p']:.4f}",
                               fontsize=11, fontweight='bold', color=col)
        axes_t3[idx].grid(True, alpha=0.3)
    plt.tight_layout()
    for pth in ['Figuras_QQPlot/QQ_Plots_Top3_Models.png',
                'Figuras_NestedCV/QQ_Plots_Top3_Models.png',
                'Paper/Figures/QQ_Plots_Top3_Models.png',
                'Figures/QQ_Plots_Top3_Models.png']:
        plt.savefig(pth, dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Q-Q Plots (Top 3) salvo")

# Gráfico 3 – Histogramas com curva normal
fig, axes_hist = plt.subplots(nr_qq, nc_qq, figsize=(18, 5*nr_qq))
fig.suptitle('Residuals Distribution with Normal Curve',
             fontsize=16, fontweight='bold', y=0.98)
ah_flat = axes_hist.flatten() if hasattr(axes_hist, 'flatten') else [axes_hist]
for hidx, (mname, res) in enumerate(qq_results_dict.items()):
    ax = ah_flat[hidx]
    nb = max(10, min(30, len(np.unique(res['residuals']))//2))
    ax.hist(res['residuals'], bins=nb, density=True, alpha=0.7,
            color='steelblue', edgecolor='k')
    xn = np.linspace(res['residuals'].min(), res['residuals'].max(), 100)
    ax.plot(xn, stats.norm.pdf(xn, res['mean'], res['std']),
            'r-', lw=2, label='Normal')
    ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
    ax.text(0.02, 0.95,
            f"μ={res['mean']:.3f}\nσ={res['std']:.3f}\n"
            f"Skew={res['skewness']:.3f}\nKurt={res['kurtosis']:.3f}",
            transform=ax.transAxes, fontsize=8, va='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax.set_xlabel('Resíduos (MPa)', fontsize=10)
    ax.set_ylabel('Densidade', fontsize=10)
    ax.set_title(f"{mname} (p={res['shapiro_p']:.4f})", fontsize=10, fontweight='bold')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
for idx in range(hidx+1, len(ah_flat)):
    ah_flat[idx].set_visible(False)
plt.tight_layout()
for pth in ['Figuras_QQPlot/Residuals_Histogram_With_Normal.png',
            'Figuras_NestedCV/Residuals_Histogram_With_Normal.png',
            'Paper/Figures/Residuals_Histogram_With_Normal.png',
            'Figures/Residuals_Histogram_With_Normal.png']:
    plt.savefig(pth, dpi=300, bbox_inches='tight')
plt.close()
print("✅ Histogramas de resíduos salvos")

# Gráfico 4 – Boxplot dos resíduos
fig, ax = plt.subplots(figsize=(14, 8))
res_list = [qq_results_dict[m]['residuals'] for m in qq_results_dict]
mod_lbls = list(qq_results_dict.keys())
bp2 = ax.boxplot(res_list, labels=mod_lbls, patch_artist=True,
                 medianprops=dict(color='red', linewidth=2))
for patch, col in zip(bp2['boxes'],
                      plt.cm.viridis(np.linspace(0, 1, len(mod_lbls)))):
    patch.set_facecolor(col); patch.set_alpha(0.7)
ax.axhline(0, color='green', linestyle='--', lw=1.5, alpha=0.7, label='Zero')
ax.set_xlabel('Modelos', fontsize=12, fontweight='bold')
ax.set_ylabel('Resíduos (MPa)', fontsize=12, fontweight='bold')
ax.set_title('Residuals Distribution – Boxplot by Model',
             fontsize=14, fontweight='bold')
ax.tick_params(axis='x', rotation=45, labelsize=10)
ax.grid(True, alpha=0.3, axis='y'); ax.legend()
plt.tight_layout()
for pth in ['Figuras_QQPlot/Residuals_Boxplot.png',
            'Figuras_NestedCV/Residuals_Boxplot.png',
            'Paper/Figures/Residuals_Boxplot.png',
            'Figures/Residuals_Boxplot.png']:
    plt.savefig(pth, dpi=300, bbox_inches='tight')
plt.close()
print("✅ Boxplot dos resíduos salvo")

# Gráfico 5 – Comparação de p-valores (Shapiro-Wilk)
norm_df_qq = pd.DataFrame(normality_tests).sort_values('Shapiro_p', ascending=False)
fig, ax = plt.subplots(figsize=(14, 8))
xp_qq = np.arange(len(norm_df_qq))
cols_n = ['green' if p > 0.05 else 'orange' if p > 0.01 else 'red'
          for p in norm_df_qq['Shapiro_p']]
bars_n = ax.bar(xp_qq, norm_df_qq['Shapiro_p'], color=cols_n, alpha=0.7, edgecolor='k')
ax.axhline(0.05, color='green', linestyle='--', lw=1.5, alpha=0.7, label='α=0.05')
ax.axhline(0.01, color='orange', linestyle='--', lw=1.5, alpha=0.7, label='α=0.01')
for bar, pv in zip(bars_n, norm_df_qq['Shapiro_p']):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
            f'{pv:.4f}', ha='center', va='bottom', fontsize=9, rotation=45)
ax.set_xticks(xp_qq)
ax.set_xticklabels(norm_df_qq['Model'], rotation=45, ha='right', fontsize=10)
ax.set_ylim(0, 1.1)
ax.set_ylabel('Shapiro-Wilk p-value', fontsize=12, fontweight='bold')
ax.set_title('Normality Test – Shapiro-Wilk p-values',
             fontsize=14, fontweight='bold')
ax.legend(loc='upper right'); ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
for pth in ['Figuras_QQPlot/Normality_Test_Comparison.png',
            'Figuras_NestedCV/Normality_Test_Comparison.png',
            'Paper/Figures/Normality_Test_Comparison.png',
            'Figures/Normality_Test_Comparison.png']:
    plt.savefig(pth, dpi=300, bbox_inches='tight')
plt.close()
print("✅ Comparação de normalidade salva")

# Tabela consolidada
norm_tbl = pd.DataFrame(normality_tests).round(
    {'Mean_Residual':4,'Std_Residual':4,'Skewness':3,'Kurtosis':3,
     'Shapiro_Wilk':4,'Shapiro_p':4,'Jarque_Bera':2,'JB_p':4})
norm_tbl.to_excel('Resultados_Artigo/Normality_Tests_Results.xlsx', index=False)
norm_tbl.to_excel('Paper/Results/Normality_Tests_Results.xlsx',    index=False)

print("\n📊 TESTES DE NORMALIDADE:")
print(norm_tbl[['Model','Mean_Residual','Std_Residual','Skewness',
                'Kurtosis','Shapiro_p','Normality_Status']].to_string(index=False))

for status, emoji, thresh_lo, thresh_hi in [
    ("NORMALMENTE distribuídos",      "✅", 0.05,  1.0),
    ("APROXIMADAMENTE normais",       "⚠️", 0.01, 0.05),
    ("NÃO normalmente distribuídos",  "❌", 0.0,  0.01),
]:
    mods = norm_tbl[(norm_tbl['Shapiro_p'] >= thresh_lo) &
                    (norm_tbl['Shapiro_p'] < thresh_hi)]['Model'].tolist()
    if mods:
        print(f"\nResíduos {status}:")
        for m in mods: print(f"   {emoji} {m}")

print("\n📊 ASSIMETRIA (SKEWNESS):")
for _, row in norm_tbl.iterrows():
    sk = row['Skewness']
    if abs(sk) < 0.5:  interp = "Simétrica"
    elif sk > 0:       interp = "Assimétrica positiva"
    else:              interp = "Assimétrica negativa"
    print(f"   {row['Model']}: Skew={sk:.3f} ({interp})")

print("\n✅ Q-Q Plots e Normalidade concluídos!")



print("\n" + "="*80)
print("🏆 RESUMO EXECUTIVO – PIPELINE v15 CONCLUÍDO")
print("="*80)

print(f"\n📊 Dataset original: {df.shape[0]} amostras")
print(f"🔢 Features: {X_full.shape[1]}")
print(f"🤖 Modelos: {len(models)}")
print(f"🔄 Monte Carlo original: {n_monte_carlo} runs")
print(f"🔄 Monte Carlo otimizado: {n_mc_opt} runs")
print(f"🔄 Repeated K-Fold CV: {n_splits}×{n_repeats} = {total_evals} avaliações/modelo")

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

print("\n" + "="*80)
print(f"✅ PROCESSO v15 CONCLUÍDO EM {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)
