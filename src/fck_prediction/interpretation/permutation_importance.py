import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import clone
from sklearn.inspection import permutation_importance

from fck_prediction.config import (N_REPEATS_PERM, TARGET)


def run_permutation_importance(models_trained, optimized_datasets, feature_names,
                                all_shap_importance, n_repeats=N_REPEATS_PERM):
    """Permutation importance (R² drop) per model on test set [NEW-4] (S27).

    Also produces SHAP vs Permutation comparison for the first 4 models.

    Returns
    -------
    perm_imp_results    : dict[str, {'importances_mean', 'importances_std', 'importances'}]
    perm_imp_summary    : list[dict]  — top-3 features per model
    feat_stats_perm_df  : DataFrame   — stats per feature across all models
    """
    print("\n" + "=" * 80)
    print("PERMUTATION IMPORTANCE [NEW-4]")
    print("=" * 80)

    perm_imp_results = {}
    perm_imp_summary = []

    for model_name, model in models_trained.items():
        print(f"\n   {model_name} ...")
        try:
            info    = optimized_datasets[model_name]
            dev_df  = info['dev_df']
            sc_mod  = info['scaler']
            X_tst_r = info['X_tst_raw']
            y_tst_r = info['y_tst_raw'].values

            X_dev_np = dev_df[feature_names].values
            y_dev_np = dev_df[TARGET].values
            X_tst_sc = sc_mod.transform(X_tst_r)

            m = clone(model)
            m.fit(X_dev_np, y_dev_np)

            result = permutation_importance(
                m, X_tst_sc, y_tst_r,
                n_repeats=n_repeats,
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

    # ── Heatmap ───────────────────────────────────────────────────────────────
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

    # ── Barras por modelo (Top 5 features) ───────────────────────────────────
    n_perm_mods = len(perm_imp_results)
    nc = 3; nr = (n_perm_mods + nc - 1) // nc
    fig, axes = plt.subplots(nr, nc, figsize=(18, 5 * nr))
    fig.suptitle('Permutation Importance – Top 5 Features per Model',
                 fontsize=16, fontweight='bold', y=0.98)
    ax_flat = axes.flatten() if hasattr(axes, 'flatten') else [axes]
    cols_perm = plt.cm.Set3(np.linspace(0, 1, len(feature_names)))
    for pidx2, (mname, pdata) in enumerate(perm_imp_results.items()):
        if pidx2 >= len(ax_flat):
            break
        ax = ax_flat[pidx2]
        imp_df2 = pd.DataFrame({'Feature':    feature_names,
                                'Importance': pdata['importances_mean'],
                                'Std':        pdata['importances_std']
                                }).sort_values('Importance').tail(5)
        bars = ax.barh(imp_df2['Feature'], imp_df2['Importance'],
                       xerr=imp_df2['Std'], capsize=3,
                       color=cols_perm[:len(imp_df2)], alpha=0.7, edgecolor='k')
        ax.set_xlabel('R² drop', fontsize=10)
        ax.set_title(mname, fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        for bar, val in zip(bars, imp_df2['Importance']):
            ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                    f'{val:.4f}', va='center', fontsize=8)
    for idx in range(pidx2 + 1, len(ax_flat)):
        ax_flat[idx].set_visible(False)
    plt.tight_layout()
    for pth in ['Figuras_PermutationImportance/Permutation_Importance_Bars.png',
                'Figuras_Shap/Permutation_Importance_Bars.png',
                'Paper/Figures/Permutation_Importance_Bars.png']:
        plt.savefig(pth, dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Barras Permutation Importance salvas")

    # ── SHAP vs Permutation Importance ───────────────────────────────────────
    if all_shap_importance:
        consolidated_shap = pd.concat(all_shap_importance, ignore_index=True)
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))
        fig.suptitle('SHAP vs Permutation Importance – Feature Comparison',
                     fontsize=16, fontweight='bold', y=0.98)
        compare_mods = list(perm_imp_results.keys())[:4]
        for cidx, mname in enumerate(compare_mods):
            ax = axes[cidx // 2, cidx % 2]
            sh_imp = consolidated_shap[consolidated_shap['Model'] == mname]
            sh_dict = dict(zip(sh_imp['Feature'], sh_imp['SHAP_Importance'])) \
                      if not sh_imp.empty else {}
            pm_dict = dict(zip(feature_names, perm_imp_results[mname]['importances_mean']))
            sh_v = np.array([sh_dict.get(f, 0) for f in feature_names])
            pm_v = np.array([pm_dict.get(f, 0) for f in feature_names])
            if sh_v.sum() > 0: sh_v /= sh_v.sum()
            if pm_v.sum() > 0: pm_v /= pm_v.sum()
            xp = np.arange(len(feature_names)); w = 0.35
            ax.bar(xp - w / 2, sh_v, w, label='SHAP',        color='steelblue', alpha=0.7)
            ax.bar(xp + w / 2, pm_v, w, label='Permutation', color='coral',     alpha=0.7)
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

    # ── Boxplot por feature (variabilidade entre modelos) ────────────────────
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

    # ── Tabelas consolidadas ──────────────────────────────────────────────────
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
            'Feature':          f,
            'Mean_Importance':  np.mean(vs),
            'Std_Importance':   np.std(vs),
            'Min_Importance':   np.min(vs),
            'Max_Importance':   np.max(vs),
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

    return perm_imp_results, perm_imp_summary, feat_stats_perm_df
