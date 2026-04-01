import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

from fck_prediction.config import FIG_SHAP, RES_SHAP


def run_shap(models_trained, model_list, X_tst_ref_sc, feature_names):
    """SHAP feature importance for all trained models (S15).

    Uses TreeExplainer → LinearExplainer → KernelExplainer, in priority order.
    All computed on the common reference test set [FIX-2].

    Returns
    -------
    all_shap_importance : list[DataFrame]  — one per model (may be empty)
    """
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
            except Exception:
                pass

            if not supports_shap:
                try:
                    if hasattr(m, 'coef_'):
                        explainer     = shap.LinearExplainer(m, X_ref_df)
                        supports_shap = True
                        print(f"      ✅ LinearExplainer")
                except Exception:
                    pass

            if not supports_shap:
                try:
                    X_samp    = X_ref_df[:50] if len(X_ref_df) > 50 else X_ref_df
                    explainer = shap.KernelExplainer(m.predict, X_samp)
                    supports_shap = True
                    print(f"      ✅ KernelExplainer")
                except Exception:
                    continue

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
                    'Feature':               feature_names,
                    'SHAP_Importance':       mabs,
                    'Normalized_Importance': mabs / (mabs.sum() + 1e-10),
                }).sort_values('SHAP_Importance', ascending=False)
                imp_df['Model'] = model_name
                all_shap_importance.append(imp_df)
                imp_df.to_excel(
                    RES_SHAP / f"SHAP_Importance_{model_name}_Otimizado.xlsx",
                    index=False)

                plt.figure(figsize=(10, 6))
                plt.barh(imp_df['Feature'], imp_df['SHAP_Importance'], color='steelblue')
                plt.xlabel('Mean |SHAP value|')
                plt.title(f'SHAP Feature Importance – {model_name}')
                plt.gca().invert_yaxis()
                plt.tight_layout()
                plt.savefig(FIG_SHAP / f"SHAP_Bar_{model_name}_Otimizado.png",
                            dpi=300, bbox_inches='tight')
                plt.close()

                plt.figure(figsize=(10, 6))
                shap.summary_plot(shap_vals, X_used, feature_names=feature_names, show=False)
                plt.tight_layout()
                plt.savefig(FIG_SHAP / f"SHAP_Summary_{model_name}_Otimizado.png",
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
                                FIG_SHAP / f"SHAP_Dependence_{model_name}_{feat}_Otimizado.png",
                                dpi=300, bbox_inches='tight')
                            plt.close()
                        except Exception:
                            continue

                print(f"      ✅ SHAP salvo")
        except Exception as e:
            print(f"      ⚠️ {str(e)[:100]}")

    if all_shap_importance:
        cons_shap = pd.concat(all_shap_importance, ignore_index=True)
        cons_shap.to_excel(
            RES_SHAP / "SHAP_Importance_ALL_MODELS_Otimizado.xlsx",
            index=False)
        piv_shap      = cons_shap.pivot_table(
            values='SHAP_Importance', index='Feature', columns='Model', fill_value=0)
        piv_shap_norm = piv_shap.div(piv_shap.sum(axis=0), axis=1)
        piv_shap.to_excel(RES_SHAP / "SHAP_Importance_Pivot_Otimizado.xlsx")
        piv_shap_norm.to_excel(RES_SHAP / "SHAP_Importance_Pivot_Normalized_Otimizado.xlsx")

        plt.figure(figsize=(14, 10))
        import seaborn as sns
        sns.heatmap(piv_shap_norm, annot=True, fmt='.3f', cmap='viridis')
        plt.title('Normalized SHAP Importance – All Models', fontsize=14)
        plt.tight_layout()
        plt.savefig(FIG_SHAP / "SHAP_Importance_Heatmap_Otimizado.png",
                    dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ SHAP consolidado para {len(all_shap_importance)} modelos")

    return all_shap_importance
