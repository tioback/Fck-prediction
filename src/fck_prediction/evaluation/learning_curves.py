import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.model_selection import learning_curve

from fck_prediction.config import (N_GRID_LC, CV_FOLDS_LC, TARGET)


def run_learning_curves(models_trained, optimized_datasets, feature_names,
                        n_points=N_GRID_LC, cv_folds=CV_FOLDS_LC):
    """Learning curves diagnostic: over/underfitting per model [NEW-3] (S26).

    Returns
    -------
    lc_results  : dict[str, dict | None]
    diag_lc_df  : DataFrame with Status and Recomendacao per model
    """
    print("\n" + "=" * 80)
    print("LEARNING CURVES – DIAGNÓSTICO DE MODELOS [NEW-3]")
    print("=" * 80)

    train_sizes_lc = np.linspace(0.1, 1.0, n_points)
    lc_results     = {}

    print(f"\nConfiguração: {len(train_sizes_lc)} pontos, {cv_folds}-fold CV, métrica=R²")

    for model_name, model in models_trained.items():
        print(f"\n   {model_name} ...")
        try:
            info   = optimized_datasets[model_name]
            dev_df = info['dev_df']
            X_d    = dev_df[feature_names].values
            y_d    = dev_df[TARGET].values

            ts_abs, tr_sc, te_sc = learning_curve(
                clone(model), X_d, y_d,
                train_sizes=train_sizes_lc,
                cv=cv_folds, scoring='r2',
                n_jobs=-1, shuffle=True, random_state=42)

            tr_mn, tr_sd = tr_sc.mean(axis=1), tr_sc.std(axis=1)
            te_mn, te_sd = te_sc.mean(axis=1), te_sc.std(axis=1)
            gap = tr_mn - te_mn

            lc_results[model_name] = {
                'train_sizes':        ts_abs,
                'train_mean':         tr_mn, 'train_std': tr_sd,
                'test_mean':          te_mn, 'test_std':  te_sd,
                'gap':                gap, 'max_gap': gap.max(),
                'final_gap':          gap[-1],
                'final_train_score':  tr_mn[-1],
                'final_test_score':   te_mn[-1],
            }
            print(f"      Train={tr_mn[-1]:.4f} ± {tr_sd[-1]:.4f} | "
                  f"Test={te_mn[-1]:.4f} ± {te_sd[-1]:.4f} | Gap={gap[-1]:.4f}")
        except Exception as e:
            print(f"      ❌ {str(e)[:100]}")
            lc_results[model_name] = None

    # ── Gráfico 1 – Todos os modelos (subplots) ───────────────────────────────
    n_lc   = sum(1 for v in lc_results.values() if v)
    n_cols = 3
    n_rows = (n_lc + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    fig.suptitle('Learning Curves – All Models (Optimized Datasets)',
                 fontsize=16, fontweight='bold', y=0.98)
    axes_flat = axes.flatten() if hasattr(axes, 'flatten') else [axes]
    pidx = 0
    for model_name, res in lc_results.items():
        if res is None:
            continue
        ax = axes_flat[pidx]
        ax.plot(res['train_sizes'], res['train_mean'], 'o-', color='blue',
                lw=2, ms=4, label='Treino')
        ax.fill_between(res['train_sizes'],
                        res['train_mean'] - res['train_std'],
                        res['train_mean'] + res['train_std'],
                        alpha=0.1, color='blue')
        ax.plot(res['train_sizes'], res['test_mean'], 'o-', color='red',
                lw=2, ms=4, label='Validação')
        ax.fill_between(res['train_sizes'],
                        res['test_mean'] - res['test_std'],
                        res['test_mean'] + res['test_std'],
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

    # ── Gráfico 2 – Comparação de gaps ───────────────────────────────────────
    valid_lc = [(n, r) for n, r in lc_results.items() if r]
    nms_lc   = [x[0] for x in valid_lc]
    fg_lc    = [x[1]['final_gap']         for x in valid_lc]
    ft_lc    = [x[1]['final_train_score'] for x in valid_lc]
    fv_lc    = [x[1]['final_test_score']  for x in valid_lc]

    fig, ax = plt.subplots(figsize=(14, 8))
    xp = np.arange(len(nms_lc)); w = 0.25
    b1 = ax.bar(xp - w,   ft_lc, w, label='Train Score',     color='steelblue', alpha=0.8, edgecolor='k')
    b2 = ax.bar(xp,       fv_lc, w, label='Val Score',       color='coral',     alpha=0.8, edgecolor='k')
    b3 = ax.bar(xp + w,   fg_lc, w, label='Gap (Train–Val)', color='gray',      alpha=0.8, edgecolor='k')
    for bars in [b1, b2, b3]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01, f'{h:.3f}',
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

    # ── Gráfico 3 – Top 3 melhores vs piores ─────────────────────────────────
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
            ax.plot(res['train_sizes'], res['train_mean'], 'o-', color='blue',
                    lw=2, ms=4, label='Treino')
            ax.fill_between(res['train_sizes'],
                            res['train_mean'] - res['train_std'],
                            res['train_mean'] + res['train_std'],
                            alpha=0.1, color='blue')
            ax.plot(res['train_sizes'], res['test_mean'], 'o-', color='red',
                    lw=2, ms=4, label='Validação')
            ax.fill_between(res['train_sizes'],
                            res['test_mean'] - res['test_std'],
                            res['test_mean'] + res['test_std'],
                            alpha=0.1, color='red')
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

    # ── Tabela de diagnóstico ─────────────────────────────────────────────────
    diag_lc = []
    for mname, res in lc_results.items():
        if res is None:
            continue
        fg = res['final_gap']; ft = res['final_test_score']
        if ft < 0.6:
            status = "UNDERFITTING"
            rec    = "Aumentar complexidade ou melhorar features"
        elif fg > 0.15:
            status = "OVERFITTING"
            rec    = "Reduzir complexidade, mais regularização ou mais dados"
        elif fg > 0.08:
            status = "MODERADO"
            rec    = "Overfitting moderado, considerar regularização"
        elif ft > 0.85 and fg < 0.05:
            status = "ÓTIMO"
            rec    = "Modelo bem ajustado"
        else:
            status = "ACEITÁVEL"
            rec    = "Desempenho razoável"
        diag_lc.append({'Model':             mname,
                        'Final_Train_Score': res['final_train_score'],
                        'Final_Test_Score':  res['final_test_score'],
                        'Final_Gap':         fg, 'Max_Gap': res['max_gap'],
                        'Status':            status, 'Recomendacao': rec,
                        'Samples':           len(optimized_datasets[mname]['dev_df'])})

    diag_lc_df = pd.DataFrame(diag_lc).sort_values('Final_Test_Score', ascending=False)
    diag_lc_df.to_excel('Resultados_Artigo/Learning_Curves_Diagnosis.xlsx', index=False)
    diag_lc_df.to_excel('Paper/Results/Learning_Curves_Diagnosis.xlsx',    index=False)
    print("\n📊 DIAGNÓSTICO LEARNING CURVES:")
    print(diag_lc_df[['Model', 'Final_Test_Score', 'Final_Gap', 'Status']].to_string(index=False))

    print("\n✅ Learning Curves concluído!")
    return lc_results, diag_lc_df
