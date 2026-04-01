import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.special import softmax
from scipy.stats import rankdata


def run_dm_heatmap(pred_ref, y_ref_ext):
    """Pairwise Diebold-Mariano heatmap on squared errors (S14).

    Returns
    -------
    dm_df : DataFrame with columns Model1, Model2, DM, p
    """
    print("\n" + "=" * 80)
    print("DIEBOLD-MARIANO HEATMAP (referência comum) [FIX-2]")
    print("=" * 80)

    if len(pred_ref) < 2:
        return pd.DataFrame()

    dm_res = []
    nms_dm = list(pred_ref.keys())
    for i in range(len(nms_dm)):
        for j in range(i + 1, len(nms_dm)):
            e1 = y_ref_ext - pred_ref[nms_dm[i]]
            e2 = y_ref_ext - pred_ref[nms_dm[j]]
            d  = e1 ** 2 - e2 ** 2
            dm = np.mean(d) / np.sqrt(np.var(d) / len(d))
            p  = 2 * (1 - stats.norm.cdf(abs(dm)))
            dm_res.append([nms_dm[i], nms_dm[j], dm, p])

    dm_df = pd.DataFrame(dm_res, columns=["Model1", "Model2", "DM", "p"])
    dm_df.to_excel('Resultados_Artigo/DM_Heatmap_Results_Otimizado.xlsx', index=False)
    dm_df.to_excel('Paper/Results/DM_Heatmap_Results_Otimizado.xlsx',    index=False)

    dm_pivot = dm_df.pivot(index="Model1", columns="Model2", values="DM")
    plt.figure(figsize=(10, 8))
    import seaborn as sns
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

    return dm_df


def run_friedman_nemenyi(df_results):
    """Friedman test + Nemenyi CD diagram on original MC results (S20).

    Parameters
    ----------
    df_results : DataFrame from run_monte_carlo() (S04)

    Returns
    -------
    fr_df : DataFrame with Model, Mean_Rank, Friedman_p
    """
    print("\n🔬 FRIEDMAN TEST + CD DIAGRAM...")

    pivot_fr = df_results.pivot_table(
        values='RMSE', index=['Run', 'Dataset'], columns='Model').dropna()

    if len(pivot_fr) == 0:
        return pd.DataFrame()

    fr_ranks = np.array([rankdata(row.values, method='average')
                         for _, row in pivot_fr.iterrows()])
    stat_fr, p_fr = stats.friedmanchisquare(*fr_ranks.T)
    print(f"Friedman: χ²={stat_fr:.3f}, p={p_fr:.3f} "
          f"{'✅ DIFERENÇAS SIGNIFICATIVAS!' if p_fr < 0.05 else ''}")

    k_fr    = len(pivot_fr.columns)
    n_fr    = len(fr_ranks)
    q_alpha = 2.95
    cd      = q_alpha * np.sqrt(k_fr * (k_fr + 1) / (6 * n_fr))
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
    plt.axvline(mr.mean() + cd / 2, color='g', linestyle='--', alpha=0.7,
                label='Critical Distance')
    plt.axvline(mr.mean() - cd / 2, color='g', linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig("Paper/Figures/cd_diagram_nemenyi.png", dpi=300, bbox_inches='tight')
    plt.close()

    fr_df = pd.DataFrame({'Model': ml, 'Mean_Rank': mr,
                          'Friedman_p': p_fr}).sort_values('Mean_Rank')
    fr_df.to_excel("Paper/Results/friedman_nemenyi_ranks.xlsx", index=False)
    print("🏆 TOP 5:"); print(fr_df.head())
    print("✅ CD Diagram salvo")

    return fr_df


def run_dm_test(pred_ref, y_ref_ext):
    """Pairwise Diebold-Mariano tabular output (S21).

    Returns
    -------
    dm_out : DataFrame with columns Model1, Model2, DM, p
    """
    print("\n📊 Diebold-Mariano Test (referência comum)...")

    if len(pred_ref) < 2:
        return pd.DataFrame()

    dm_tab = []
    nms    = list(pred_ref.keys())
    for i in range(len(nms)):
        for j in range(i + 1, len(nms)):
            e1 = y_ref_ext - pred_ref[nms[i]]
            e2 = y_ref_ext - pred_ref[nms[j]]
            d  = e1 ** 2 - e2 ** 2
            dm = np.mean(d) / np.sqrt(np.var(d) / len(d))
            p  = 2 * (1 - stats.norm.cdf(abs(dm)))
            dm_tab.append([nms[i], nms[j], dm, p])

    dm_out = pd.DataFrame(dm_tab, columns=["Model1", "Model2", "DM", "p"])
    dm_out.to_excel("Paper/Results/diebold_mariano_otimizado.xlsx", index=False)
    print("✅ Diebold-Mariano salvo")

    return dm_out


def run_plackett_luce(pred_ref, y_ref_ext):
    """Bayesian Plackett-Luce ability ranking via softmax(-RMSE) (S23).

    Returns
    -------
    pl_df : DataFrame with Model, Ability — sorted descending
    """
    print("\n📊 Bayesian Plackett-Luce Ranking...")
    from sklearn.metrics import mean_squared_error

    rmse_pl  = [np.sqrt(mean_squared_error(y_ref_ext, pred_ref[n]))
                for n in pred_ref]
    nms_pl   = list(pred_ref.keys())
    abilities = softmax(-np.array(rmse_pl))
    pl_df = pd.DataFrame({"Model": nms_pl, "Ability": abilities}).sort_values(
        "Ability", ascending=False)
    pl_df.to_excel("Paper/Results/plackett_luce_ranking_otimizado.xlsx", index=False)
    print("✅ Plackett-Luce ranking salvo")
    print(pl_df.to_string(index=False))

    return pl_df
