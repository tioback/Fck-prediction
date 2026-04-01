import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from fck_prediction.config import FIG_RESIDUAL_DIAG, RESULTS_DIR


def run_residual_diagnostics(pred_ref, y_ref_ext):
    """Econometric residual diagnostics on the common reference partition (S18).

    Tests: Shapiro-Francia (Anderson-Darling), Jarque-Bera, Breusch-Pagan,
           White, Durbin-Watson, Breusch-Godfrey. ACF/PACF plots per model.

    Returns
    -------
    diag_df : DataFrame with all test statistics
    """
    print("\n🔬 ECONOMETRIC RESIDUAL DIAGNOSTICS (referência comum)...")

    import statsmodels.api as sm
    from statsmodels.stats.diagnostic import (het_breuschpagan, het_white,
                                              acorr_breusch_godfrey, normal_ad)
    from statsmodels.stats.stattools import durbin_watson
    from scipy.stats import jarque_bera

    diagnostics = []
    for name, pr in pred_ref.items():
        print(f"   Testando: {name}")
        res   = y_ref_ext - pr
        X_aux = sm.add_constant(pr)
        sf_s, sf_p       = normal_ad(res)
        jb_s, jb_p       = jarque_bera(res)
        bp_s, bp_p, _, _ = het_breuschpagan(res, X_aux)
        wh_s, wh_p, _, _ = het_white(res, X_aux)
        dw_s             = durbin_watson(res)
        ols_a            = sm.OLS(res, X_aux).fit()
        bg_s, bg_p, _, _ = acorr_breusch_godfrey(ols_a, nlags=2)
        diagnostics.append({
            "Model":               name,
            "ShapiroFrancia_stat": sf_s, "ShapiroFrancia_p": sf_p,
            "JarqueBera_stat":     jb_s, "JarqueBera_p":     jb_p,
            "BreuschPagan_stat":   bp_s, "BreuschPagan_p":   bp_p,
            "White_stat":          wh_s, "White_p":          wh_p,
            "DurbinWatson":        dw_s,
            "BreuschGodfrey_stat": bg_s, "BreuschGodfrey_p": bg_p,
        })
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        sm.graphics.tsa.plot_acf(res,  lags=20, ax=ax[0])
        ax[0].set_title(f"ACF – {name}")
        sm.graphics.tsa.plot_pacf(res, lags=20, ax=ax[1])
        ax[1].set_title(f"PACF – {name}")
        plt.tight_layout()
        plt.savefig(FIG_RESIDUAL_DIAG / f"ACF_PACF_{name}_Otimizado.png", dpi=300)
        plt.close()

    diag_df = pd.DataFrame(diagnostics).sort_values("ShapiroFrancia_p", ascending=False)
    print("\n📊 RESIDUAL DIAGNOSTICS:")
    print(diag_df[['Model', 'ShapiroFrancia_p', 'JarqueBera_p',
                   'BreuschPagan_p', 'White_p', 'DurbinWatson']].to_string(index=False))
    diag_df.to_csv(RESULTS_DIR / "residual_diagnostics_full_otimizado.csv", index=False)
    diag_df.to_excel(RESULTS_DIR / "residual_diagnostics_full_otimizado.xlsx", index=False)
    print("✅ Residual diagnostics concluído")

    # ── Heteroscedasticity chart ──────────────────────────────────────────────
    hetero = []
    for _, row in diag_df.iterrows():
        hs = []
        if row['BreuschPagan_p'] < 0.05: hs.append("Breusch-Pagan")
        if row['White_p']        < 0.05: hs.append("White")
        hetero.append({'Model':          row['Model'],
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
    plt.savefig(FIG_RESIDUAL_DIAG / 'heterocedasticidade_pvalues_otimizado.png',
                dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Gráfico heterocedasticidade salvo")

    return diag_df
