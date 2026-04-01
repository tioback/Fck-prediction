import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from fck_prediction.config import (TARGET, CONFIDENCE_LEVELS, Z_SCORES_MAP)


def run_picp(models_trained, optimized_datasets, feature_names):
    """Prediction Interval Coverage Probability — Gaussian and Quantile methods (S08).

    std estimated on DEV, applied on TEST [FIX-4].

    Returns
    -------
    picp_df : DataFrame (empty DataFrame if no results)
    """
    print("\n" + "=" * 80)
    print("📊 PREDICTION INTERVAL COVERAGE PROBABILITY (PICP) [FIX-4]")
    print("=" * 80)

    picp_results = []

    for model_name in models_trained.keys():
        print(f"\n   📈 {model_name}")

        try:
            info    = optimized_datasets[model_name]
            dev_df  = info['dev_df']
            sc_mod  = info['scaler']
            X_tst_r = info['X_tst_raw']
            y_tst_r = info['y_tst_raw'].values

            X_dev_np = dev_df[feature_names].values
            y_dev_np = dev_df[TARGET].values
            X_tst_sc = sc_mod.transform(X_tst_r)

            m     = models_trained[model_name]
            p_dev = m.predict(X_dev_np)
            p_tst = m.predict(X_tst_sc)

            # ── std estimado no DEV (não no test) ─────────────── [FIX-4]
            res_dev   = y_dev_np - p_dev
            mean_res  = np.mean(res_dev)
            std_res   = np.std(res_dev)

            print(f"      Resíduos DEV: μ={mean_res:.4f}, σ={std_res:.4f}")

            for cl in CONFIDENCE_LEVELS:
                z  = Z_SCORES_MAP[cl]
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
            for cl in CONFIDENCE_LEVELS:
                alpha = 1 - cl
                lq = np.percentile(res_dev, (alpha / 2) * 100)
                uq = np.percentile(res_dev, (1 - alpha / 2) * 100)
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

    # ── Gráficos PICP ─────────────────────────────────────────────────────────
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
    return picp_df
