import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (r2_score, mean_squared_error,
                             mean_absolute_error,
                             mean_absolute_percentage_error)

from fck_prediction.config import (N_MONTE_CARLO, N_MC_OPT,
                                    METRICS_LIST, METRIC_LABELS, TARGET,
                                    FIG_MONTE_CARLO, RESULTS_DIR)


def run_monte_carlo(X_full, y_full, models, model_list, n_runs=N_MONTE_CARLO):
    """30-run Monte Carlo benchmark on raw (uncleaned) data with variance ratio (S04).

    Returns
    -------
    df_results : DataFrame — all MC runs × models × datasets
    df_vr      : DataFrame — variance ratio table
    """
    print("\n" + "=" * 80)
    print("📊 MONTE CARLO E VARIANCE RATIO (DADOS ORIGINAIS)")
    print("=" * 80)

    all_results = []

    for run in range(n_runs):
        if run % 5 == 0:
            print(f"   Run {run+1}/{n_runs}")

        X_tr, X_te, y_tr, y_te = train_test_split(
            X_full.values, y_full.values, test_size=0.2, random_state=42 + run)

        sc = StandardScaler()
        X_tr_s = sc.fit_transform(X_tr)
        X_te_s = sc.transform(X_te)

        for name, model in models.items():
            try:
                m = clone(model)
                m.fit(X_tr_s, y_tr)
                p_tr = m.predict(X_tr_s)
                p_te = m.predict(X_te_s)

                for ds, y_ds, p_ds in [('Train', y_tr, p_tr), ('Test', y_te, p_te)]:
                    all_results.append({
                        "Model":   name, "Run": run, "Dataset": ds,
                        "R2":   r2_score(y_ds, p_ds),
                        "RMSE": np.sqrt(mean_squared_error(y_ds, p_ds)),
                        "MAE":  mean_absolute_error(y_ds, p_ds),
                        "MAPE": mean_absolute_percentage_error(y_ds, p_ds) * 100,
                    })
            except Exception as e:
                print(f"      ⚠️ {name}: {str(e)[:50]}")

    df_results = pd.DataFrame(all_results)
    df_results.to_excel(RESULTS_DIR / "Monte_Carlo_Results.xlsx", index=False)

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
    df_vr.to_excel(RESULTS_DIR / "Variance_Ratio_Results.xlsx", index=False)

    # Boxplot
    fig, axes = plt.subplots(4, 1, figsize=(16, 18), sharex=True)
    fig.suptitle('Boxplots of performance metrics with Variance Ratio overlay (Monte Carlo)',
                 fontsize=16, fontweight='bold', y=0.98)

    for i, metric in enumerate(METRICS_LIST):
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
                ax.text(j, ax.get_ylim()[1] * 0.95, f'VR={vr_val:.2f}', ha='center',
                        fontsize=9, rotation=90, color=col, fontweight='bold')
        ax.set_title(f"{metric} – {METRIC_LABELS[metric]}", fontsize=14,
                     fontweight='bold', loc='left')
        ax.set_ylabel(METRIC_LABELS[metric], fontsize=12)
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')
        ax.set_axisbelow(True)
        if i == 0:
            ax.legend(title='Dataset', loc='upper right')
        else:
            ax.legend_.remove()

    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.xlabel('Model', fontsize=12)
    plt.tight_layout()
    plt.savefig(FIG_MONTE_CARLO / "Benchmark_Boxplot_All_Metrics.png",
                dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Boxplot Monte Carlo (dados originais) salvo")

    return df_results, df_vr


def run_monte_carlo_optimized(optimized_datasets, models, model_list,
                               best_cleaning_for_model, feature_names,
                               n_runs=N_MC_OPT):
    """30-run Monte Carlo re-split within each model's optimised DEV (S07).

    Returns
    -------
    df_res_opt : DataFrame
    df_vr_opt  : DataFrame
    """
    print("\n" + "=" * 80)
    print("📊 MONTE CARLO E VARIANCE RATIO (DADOS OTIMIZADOS)")
    print("=" * 80)

    all_res_opt = []

    for run in range(n_runs):
        if run % 5 == 0:
            print(f"   Run {run+1}/{n_runs}")

        for model_name in model_list:
            try:
                info   = optimized_datasets[model_name]
                dev_df = info['dev_df']

                X_d = dev_df[feature_names].values
                y_d = dev_df[TARGET].values

                X_tr, X_te, y_tr, y_te = train_test_split(
                    X_d, y_d, test_size=0.2, random_state=42 + run)

                m = clone(models[model_name])
                m.fit(X_tr, y_tr)
                p_tr, p_te = m.predict(X_tr), m.predict(X_te)

                for ds, y_ds, p_ds in [('Train', y_tr, p_tr),
                                        ('Test',  y_te, p_te)]:
                    all_res_opt.append({
                        "Model": model_name, "Run": run, "Dataset": ds,
                        "R2":   r2_score(y_ds, p_ds),
                        "RMSE": np.sqrt(mean_squared_error(y_ds, p_ds)),
                        "MAE":  mean_absolute_error(y_ds, p_ds),
                        "MAPE": mean_absolute_percentage_error(y_ds, p_ds) * 100,
                        "Cleaning_Method": best_cleaning_for_model[model_name],
                        "Samples": len(y_d),
                    })
            except Exception as e:
                print(f"      ⚠️ {model_name} run {run}: {str(e)[:50]}")

    df_res_opt = pd.DataFrame(all_res_opt)
    df_res_opt.to_excel(
        RESULTS_DIR / "Monte_Carlo_Results_Otimizado.xlsx", index=False)
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
        RESULTS_DIR / "Variance_Ratio_Results_Otimizado.xlsx", index=False)

    # Boxplot MC otimizado
    fig, axes = plt.subplots(4, 1, figsize=(16, 18), sharex=True)
    fig.suptitle('Boxplots – Monte Carlo (Optimized Datasets)',
                 fontsize=16, fontweight='bold', y=0.98)

    for i, metric in enumerate(METRICS_LIST):
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
                vr_val  = row["Variance_Ratio"].values[0]
                cl_inf  = row["Cleaning_Method"].values[0]
                sa_inf  = row["Samples"].values[0]
                col = 'green' if vr_val < 1.5 else ('orange' if vr_val < 2.5 else 'red')
                ax.text(j, ax.get_ylim()[1] * 0.95,
                        f'VR={vr_val:.2f}\n{cl_inf[:12]}\n({sa_inf:.0f})',
                        ha='center', fontsize=7, color=col, fontweight='bold')
        ax.set_title(f"{metric} – {METRIC_LABELS[metric]} (Optimized)",
                     fontsize=14, fontweight='bold', loc='left')
        ax.set_ylabel(METRIC_LABELS[metric], fontsize=12)
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')
        ax.set_axisbelow(True)
        if i == 0:
            ax.legend(title='Dataset', loc='upper right')
        else:
            ax.legend_.remove()

    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.xlabel('Model', fontsize=12)
    plt.tight_layout()
    plt.savefig(FIG_MONTE_CARLO / "Benchmark_Boxplot_All_Metrics_Otimizado.png",
                dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Boxplot MC otimizado salvo")

    return df_res_opt, df_vr_opt
