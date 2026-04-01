import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from fck_prediction.config import FIG_QQ, RESULTS_DIR


def run_normality_analysis(pred_ref, y_ref_ext, ranking):
    """Q-Q plots + residual normality tests on common reference partition [NEW-6] (S29).

    Parameters
    ----------
    pred_ref   : dict[str, np.ndarray]
    y_ref_ext  : np.ndarray
    ranking    : DataFrame with 'Model' column (IFI order)

    Returns
    -------
    norm_tbl : DataFrame with normality test results per model
    """
    print("\n" + "=" * 80)
    print("Q-Q PLOTS + NORMALIDADE DOS RESÍDUOS [NEW-6]")
    print("=" * 80)

    qq_results_dict = {}
    normality_tests  = []

    for model_name, pr in pred_ref.items():
        print(f"\n   {model_name} ...")
        try:
            residuals = y_ref_ext - pr
            skewness  = stats.skew(residuals)
            kurt      = stats.kurtosis(residuals)
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
                'shapiro_stat':  sh_stat, 'shapiro_p':   sh_p,
                'jb_stat':       jb_stat, 'jb_p':        jb_p,
                'anderson_stat': ad_res.statistic,
            }
            normality_tests.append({
                'Model':            model_name,
                'Mean_Residual':    np.mean(residuals),
                'Std_Residual':     np.std(residuals),
                'Skewness':         skewness,
                'Kurtosis':         kurt,
                'Shapiro_Wilk':     sh_stat,
                'Shapiro_p':        sh_p,
                'Jarque_Bera':      jb_stat,
                'JB_p':             jb_p,
                'Normality_Status': norm_status,
                'Samples':          len(residuals),
            })
            print(f"      μ={np.mean(residuals):.4f} σ={np.std(residuals):.4f} "
                  f"Skew={skewness:.3f} Kurt={kurt:.3f} | {norm_status}")
        except Exception as e:
            print(f"      ❌ {str(e)[:100]}")

    # ── Q-Q todos os modelos ──────────────────────────────────────────────────
    n_qq  = len(qq_results_dict)
    nc_qq = 3
    nr_qq = (n_qq + nc_qq - 1) // nc_qq
    fig, axes = plt.subplots(nr_qq, nc_qq, figsize=(15, 5 * nr_qq))
    fig.suptitle('Q-Q Plots – Residual Normality (Common Test Set)',
                 fontsize=16, fontweight='bold', y=0.98)
    axes_flat_qq = axes.flatten() if hasattr(axes, 'flatten') else [axes]
    pidx3 = 0
    for model_name_qq, res in qq_results_dict.items():
        ax = axes_flat_qq[pidx3]
        stats.probplot(res['residuals'], dist='norm', plot=ax)
        col = ('green' if res['shapiro_p'] > 0.05
               else ('orange' if res['shapiro_p'] > 0.01 else 'red'))
        lbl = ('Normal' if res['shapiro_p'] > 0.05
               else ('Aprox. Normal' if res['shapiro_p'] > 0.01 else 'Não Normal'))
        ax.set_title(f"{model_name_qq}\nShapiro-Wilk p={res['shapiro_p']:.4f} ({lbl})",
                     fontsize=10, fontweight='bold', color=col)
        ax.set_xlabel('Theoretical Quantiles', fontsize=9)
        ax.set_ylabel('Sample Quantiles (Residuals)', fontsize=9)
        ax.grid(True, alpha=0.3)
        pidx3 += 1
    for idx in range(pidx3, len(axes_flat_qq)):
        axes_flat_qq[idx].set_visible(False)
    plt.tight_layout()
    plt.savefig(FIG_QQ / 'QQ_Plots_All_Models.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("\n✅ Q-Q Plots (todos os modelos) salvo")

    # ── Top 3 modelos (IFI) ───────────────────────────────────────────────────
    top3_qq = [m for m in ranking.head(3)['Model'].tolist() if m in qq_results_dict]
    if top3_qq:
        fig, axes_t3 = plt.subplots(1, len(top3_qq), figsize=(6 * len(top3_qq), 5))
        if len(top3_qq) == 1: axes_t3 = [axes_t3]
        fig.suptitle('Q-Q Plots – Top 3 Models', fontsize=14, fontweight='bold')
        for idx, mname in enumerate(top3_qq):
            res = qq_results_dict[mname]
            stats.probplot(res['residuals'], dist='norm', plot=axes_t3[idx])
            col = ('green' if res['shapiro_p'] > 0.05
                   else ('orange' if res['shapiro_p'] > 0.01 else 'red'))
            axes_t3[idx].set_title(f"{mname}\np={res['shapiro_p']:.4f}",
                                   fontsize=11, fontweight='bold', color=col)
            axes_t3[idx].grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(FIG_QQ / 'QQ_Plots_Top3_Models.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ Q-Q Plots (Top 3) salvo")

    # ── Histogramas com curva normal ──────────────────────────────────────────
    fig, axes_hist = plt.subplots(nr_qq, nc_qq, figsize=(18, 5 * nr_qq))
    fig.suptitle('Residuals Distribution with Normal Curve',
                 fontsize=16, fontweight='bold', y=0.98)
    ah_flat = axes_hist.flatten() if hasattr(axes_hist, 'flatten') else [axes_hist]
    for hidx, (mname, res) in enumerate(qq_results_dict.items()):
        ax = ah_flat[hidx]
        nb = max(10, min(30, len(np.unique(res['residuals'])) // 2))
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
    for idx in range(hidx + 1, len(ah_flat)):
        ah_flat[idx].set_visible(False)
    plt.tight_layout()
    plt.savefig(FIG_QQ / 'Residuals_Histogram_With_Normal.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Histogramas de resíduos salvos")

    # ── Boxplot dos resíduos ──────────────────────────────────────────────────
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
    plt.savefig(FIG_QQ / 'Residuals_Boxplot.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Boxplot dos resíduos salvo")

    # ── Comparação de p-valores (Shapiro-Wilk) ────────────────────────────────
    norm_df_qq = pd.DataFrame(normality_tests).sort_values('Shapiro_p', ascending=False)
    fig, ax = plt.subplots(figsize=(14, 8))
    xp_qq = np.arange(len(norm_df_qq))
    cols_n = ['green' if p > 0.05 else 'orange' if p > 0.01 else 'red'
              for p in norm_df_qq['Shapiro_p']]
    bars_n = ax.bar(xp_qq, norm_df_qq['Shapiro_p'], color=cols_n, alpha=0.7, edgecolor='k')
    ax.axhline(0.05, color='green', linestyle='--', lw=1.5, alpha=0.7, label='α=0.05')
    ax.axhline(0.01, color='orange', linestyle='--', lw=1.5, alpha=0.7, label='α=0.01')
    for bar, pv in zip(bars_n, norm_df_qq['Shapiro_p']):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f'{pv:.4f}', ha='center', va='bottom', fontsize=9, rotation=45)
    ax.set_xticks(xp_qq)
    ax.set_xticklabels(norm_df_qq['Model'], rotation=45, ha='right', fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel('Shapiro-Wilk p-value', fontsize=12, fontweight='bold')
    ax.set_title('Normality Test – Shapiro-Wilk p-values',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='upper right'); ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(FIG_QQ / 'Normality_Test_Comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Comparação de normalidade salva")

    # ── Tabela consolidada ────────────────────────────────────────────────────
    norm_tbl = pd.DataFrame(normality_tests).round(
        {'Mean_Residual': 4, 'Std_Residual': 4, 'Skewness': 3, 'Kurtosis': 3,
         'Shapiro_Wilk': 4, 'Shapiro_p': 4, 'Jarque_Bera': 2, 'JB_p': 4})
    norm_tbl.to_excel(RESULTS_DIR / 'Normality_Tests_Results.xlsx', index=False)

    print("\n📊 TESTES DE NORMALIDADE:")
    print(norm_tbl[['Model', 'Mean_Residual', 'Std_Residual', 'Skewness',
                    'Kurtosis', 'Shapiro_p', 'Normality_Status']].to_string(index=False))

    for status, emoji, thresh_lo, thresh_hi in [
        ("NORMALMENTE distribuídos",     "✅", 0.05, 1.0),
        ("APROXIMADAMENTE normais",      "⚠️", 0.01, 0.05),
        ("NÃO normalmente distribuídos", "❌", 0.0,  0.01),
    ]:
        mods = norm_tbl[(norm_tbl['Shapiro_p'] >= thresh_lo) &
                        (norm_tbl['Shapiro_p'] < thresh_hi)]['Model'].tolist()
        if mods:
            print(f"\nResíduos {status}:")
            for m in mods:
                print(f"   {emoji} {m}")

    print("\n📊 ASSIMETRIA (SKEWNESS):")
    for _, row in norm_tbl.iterrows():
        sk = row['Skewness']
        if abs(sk) < 0.5:  interp = "Simétrica"
        elif sk > 0:       interp = "Assimétrica positiva"
        else:              interp = "Assimétrica negativa"
        print(f"   {row['Model']}: Skew={sk:.3f} ({interp})")

    print("\n✅ Q-Q Plots e Normalidade concluídos!")
    return norm_tbl
