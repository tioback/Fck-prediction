import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.inspection import PartialDependenceDisplay, partial_dependence

from fck_prediction.config import (N_GRID_PDP, TARGET)


def run_pdp(models_trained, optimized_datasets, feature_names, ranking,
            feat_stats_perm_df, n_grid=N_GRID_PDP):
    """Partial Dependence Plots – 1D and 2D [NEW-5] (S28).

    Plots:
    1. Best model (IFI) × top 6 features
    2. Top 3 models comparison
    3. All models overlay on top 3 most important features
    4. 2D interaction for best model (first 2 feature pairs)

    Returns
    -------
    None  (all output saved to disk)
    """
    print("\n" + "=" * 80)
    print("PARTIAL DEPENDENCE PLOTS (PDP) [NEW-5]")
    print("=" * 80)

    features_to_plot_pdp = feature_names[:6] if len(feature_names) > 6 else feature_names
    pdp_models = {}

    for model_name, model in models_trained.items():
        print(f"\n   {model_name} ...")
        try:
            info   = optimized_datasets[model_name]
            dev_df = info['dev_df']
            X_d    = dev_df[feature_names].values
            y_d    = dev_df[TARGET].values

            m = clone(model)
            m.fit(X_d, y_d)
            pdp_models[model_name] = {'model': m, 'X_train': X_d}
            print(f"      ✅ modelo preparado")
        except Exception as e:
            print(f"      ⚠️ {str(e)[:100]}")

    # ── Gráfico 1 – Melhor modelo (IFI) × 6 features ─────────────────────────
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
                grid_resolution=n_grid,
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

    # ── Gráfico 2 – Top 3 modelos comparados ────────────────────────────────
    top3_pdp = [m for m in ranking.head(3)['Model'].tolist() if m in pdp_models]
    if len(top3_pdp) >= 2:
        fig, axes_pdp2 = plt.subplots(len(top3_pdp), len(features_to_plot_pdp),
                                      figsize=(20, 5 * len(top3_pdp)))
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
                    grid_resolution=n_grid,
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

    # ── Gráfico 3 – Todos os modelos, top 3 features ─────────────────────────
    top3_feat_pdp = (feat_stats_perm_df.head(3)['Feature'].tolist()
                     if not feat_stats_perm_df.empty else feature_names[:3])
    print(f"\n   PDP comparativo – features: {top3_feat_pdp}")
    colors_pdp_cmp = plt.cm.tab10(np.linspace(0, 1, len(pdp_models)))
    fig, axes_pdp3 = plt.subplots(len(top3_feat_pdp), 1,
                                   figsize=(14, 5 * len(top3_feat_pdp)))
    if len(top3_feat_pdp) == 1:
        axes_pdp3 = [axes_pdp3]
    fig.suptitle('PDP – All Models Comparison (Top 3 Most Important Features)',
                 fontsize=16, fontweight='bold', y=0.98)
    for f_idx, feat in enumerate(top3_feat_pdp):
        ax = axes_pdp3[f_idx]
        fidx = feature_names.index(feat)
        for m_idx, (mname, pdata) in enumerate(pdp_models.items()):
            try:
                pdp_res = partial_dependence(pdata['model'], pdata['X_train'],
                                             features=[fidx], kind='average',
                                             grid_resolution=n_grid)
                ax.plot(pdp_res['values'][0], pdp_res['average'][0],
                        'o-', color=colors_pdp_cmp[m_idx], lw=2, ms=4,
                        label=mname, alpha=0.8)
            except Exception:
                continue
        ax.set_xlabel(feat, fontsize=12, fontweight='bold')
        ax.set_ylabel('Partial Dependence (fck MPa)', fontsize=12)
        ax.set_title(f'Impact of {feat} on fck Prediction', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=8)
    plt.tight_layout()
    for pth in ['Figuras_PDP/PDP_All_Models_Comparison.png',
                'Figuras_Shap/PDP_All_Models_Comparison.png',
                'Paper/Figures/PDP_All_Models_Comparison.png',
                'Figures/PDP_All_Models_Comparison.png']:
        plt.savefig(pth, dpi=300, bbox_inches='tight')
    plt.close()
    print("   ✅ PDP comparativo (todos os modelos) salvo")

    # ── Gráfico 4 – PDP 2D (interação) para o melhor modelo ─────────────────
    if best_pdp in pdp_models and len(feature_names) >= 4:
        feat_pairs_2d = [(0, 1), (2, 3)]
        fig, axes_2d = plt.subplots(1, len(feat_pairs_2d),
                                    figsize=(8 * len(feat_pairs_2d), 6))
        if len(feat_pairs_2d) == 1:
            axes_2d = [axes_2d]
        fig.suptitle(f'2D PDP – {best_pdp} – Feature Interaction',
                     fontsize=16, fontweight='bold', y=0.98)
        for idx2d, (f1, f2) in enumerate(feat_pairs_2d):
            try:
                PartialDependenceDisplay.from_estimator(
                    pdp_models[best_pdp]['model'],
                    pdp_models[best_pdp]['X_train'],
                    features=[(f1, f2)], feature_names=feature_names,
                    kind='average', n_jobs=-1,
                    grid_resolution=n_grid,
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

    # ── Tabela resumo ─────────────────────────────────────────────────────────
    pdp_summary_tbl = pd.DataFrame([
        {'Model': mn, 'PDP_Computed': 'Yes', 'N_Features': len(feature_names),
         'Best_Model': 'Yes' if mn == best_pdp else 'No'}
        for mn in pdp_models])
    pdp_summary_tbl.to_excel('Resultados_Artigo/PDP_Summary.xlsx', index=False)
    pdp_summary_tbl.to_excel('Paper/Results/PDP_Summary.xlsx',    index=False)
    print("\n✅ Partial Dependence Plots concluído!")
