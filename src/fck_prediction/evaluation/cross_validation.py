import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import clone
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import (r2_score, mean_squared_error,
                             mean_absolute_error,
                             mean_absolute_percentage_error)

from fck_prediction.config import (N_SPLITS_CV, N_REPEATS_CV, TARGET,
                                    TRAINING_COLOR, TESTING_COLOR,
                                    FIG_CROSS_VAL, RESULTS_DIR)


def run_repeated_kfold(optimized_datasets, models, model_list,
                       best_cleaning_for_model, feature_names,
                       n_splits=N_SPLITS_CV, n_repeats=N_REPEATS_CV):
    """10×10 Repeated K-Fold CV on each model's optimised DEV data (S09).

    Returns
    -------
    repeated_cv_df  : DataFrame — per-fold detail
    cv_summary_df   : DataFrame — per-model summary with 95% CI
    cv_metrics_df   : DataFrame — train + test metrics per fold
    """
    print("\n" + "=" * 80)
    print("🔄 REPEATED K-FOLD CROSS VALIDATION (10×10)")
    print("=" * 80)

    total_evals = n_splits * n_repeats

    repeated_cv_results = []
    repeated_cv_scores  = {m: [] for m in model_list}

    for model_name, model in models.items():
        print(f"\n   {model_name} ...")
        info   = optimized_datasets[model_name]
        dev_df = info['dev_df']
        X_d    = dev_df[feature_names].values
        y_d    = dev_df[TARGET].values

        for rep in range(n_repeats):
            rkf = RepeatedKFold(n_splits=n_splits, n_repeats=1,
                                random_state=42 + rep)
            for fold, (tr_i, te_i) in enumerate(rkf.split(X_d), 1):
                try:
                    mc = clone(model)
                    mc.fit(X_d[tr_i], y_d[tr_i])
                    yp = mc.predict(X_d[te_i])
                    r2_f   = r2_score(y_d[te_i], yp)
                    rmse_f = np.sqrt(mean_squared_error(y_d[te_i], yp))
                    mae_f  = mean_absolute_error(y_d[te_i], yp)
                    mape_f = mean_absolute_percentage_error(y_d[te_i], yp) * 100
                    repeated_cv_scores[model_name].append(r2_f)
                    repeated_cv_results.append({
                        'Model': model_name, 'Repeat': rep + 1, 'Fold': fold,
                        'R2': r2_f, 'RMSE': rmse_f, 'MAE': mae_f, 'MAPE': mape_f,
                        'Cleaning_Method': best_cleaning_for_model[model_name],
                        'Samples': len(y_d),
                    })
                except Exception:
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
        RESULTS_DIR / "Repeated_CV_Otimizado_10x10_Detailed.xlsx", index=False)
    cv_summary_df.to_excel(
        RESULTS_DIR / "Repeated_CV_Otimizado_10x10_Summary.xlsx", index=False)

    print("\n🏆 RANKING Repeated K-Fold CV:")
    print(cv_summary_df[['Model', 'Mean_R2', 'Std_R2',
                          'CI_95_Lower', 'CI_95_Upper',
                          'Cleaning_Method']].to_string(index=False))

    # ── Plots ─────────────────────────────────────────────────────────────────
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
    plt.savefig(FIG_CROSS_VAL / "Repeated_CV_Boxplot_Otimizado.png",
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
        plt.savefig(FIG_CROSS_VAL / "Repeated_CV_Stability_Heatmap_Otimizado.png",
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
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f'{mv:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.set_xticks(xp)
    ax.set_xticklabels(cv_summary_df['Model'], rotation=45, ha='right')
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('Mean R²', fontsize=12, fontweight='bold')
    ax.set_title(f'Model Performance – 95% CI '
                 f'({n_splits}-fold, {n_repeats} repeats)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    plt.tight_layout()
    plt.savefig(FIG_CROSS_VAL / "Repeated_CV_Barplot_with_CI_Otimizado.png",
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
    plt.savefig(FIG_CROSS_VAL / "Repeated_CV_Violin_Plot_Otimizado.png",
                dpi=300, bbox_inches='tight')
    plt.close()

    # ── Train vs Test boxplots ─────────────────────────────────────────────────
    cv_metrics = []
    for model_name, model in models.items():
        info   = optimized_datasets[model_name]
        dev_df = info['dev_df']
        X_d    = dev_df[feature_names].values
        y_d    = dev_df[TARGET].values

        for rep in range(n_repeats):
            rkf = RepeatedKFold(n_splits=n_splits, n_repeats=1,
                                random_state=42 + rep)
            for fold, (tr_i, te_i) in enumerate(rkf.split(X_d), 1):
                try:
                    mc = clone(model)
                    mc.fit(X_d[tr_i], y_d[tr_i])
                    yp_tr = mc.predict(X_d[tr_i])
                    yp_te = mc.predict(X_d[te_i])
                    for ds, y_ds, yp in [('Training', y_d[tr_i], yp_tr),
                                         ('Testing',  y_d[te_i], yp_te)]:
                        cv_metrics.append({
                            'Model': model_name, 'Dataset': ds,
                            'Repeat': rep + 1, 'Fold': fold,
                            'R2':   r2_score(y_ds, yp),
                            'RMSE': np.sqrt(mean_squared_error(y_ds, yp)),
                            'MAE':  mean_absolute_error(y_ds, yp),
                            'MAPE': mean_absolute_percentage_error(y_ds, yp) * 100,
                        })
                except Exception:
                    pass

    cv_metrics_df = pd.DataFrame(cv_metrics)
    cv_metrics_df.to_excel(
        RESULTS_DIR / "Repeated_CV_Train_Test_Metrics_Otimizado.xlsx", index=False)

    tc = '#2C3E50'; tc2 = '#E74C3C'
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Repeated K-Fold CV (10×10) – Performance Distribution',
                 fontsize=14, fontweight='bold', y=0.98)
    for idx, (metric, title) in enumerate(
            zip(['R2', 'RMSE', 'MAE', 'MAPE'],
                ['R²', 'RMSE (MPa)', 'MAE (MPa)', 'MAPE (%)'])):
        ax = axes[idx // 2, idx % 2]
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
    plt.savefig(FIG_CROSS_VAL / "Repeated_CV_All_Metrics_Boxplot_Otimizado.png",
                dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Gráficos Repeated K-Fold CV salvos")

    return repeated_cv_df, cv_summary_df, cv_metrics_df
