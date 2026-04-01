import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from fck_prediction.config import FIG_IFI, RESULTS_DIR


def compute_ifi(results_df):
    """Entropy-weighted IFI composite score — rank all models (S12).

    Mutates results_df by adding an 'IFI' column.

    Returns
    -------
    ranking : DataFrame sorted by IFI descending
    """
    print("\n📊 IFI SENSITIVITY...")

    X_mat = results_df[["R2", "RMSE", "MAE", "MAPE"]].copy()
    X_mat["RMSE"] = 1 / X_mat["RMSE"]
    X_mat["MAE"]  = 1 / X_mat["MAE"]
    X_mat["MAPE"] = 1 / X_mat["MAPE"]
    P       = X_mat / X_mat.sum()
    k       = 1 / np.log(len(X_mat))
    entropy = -k * (P * np.log(P + 1e-10)).sum()
    div     = 1 - entropy
    weights = div / div.sum()
    results_df["IFI"] = (X_mat * weights).sum(axis=1)
    ranking = results_df.sort_values("IFI", ascending=False)
    ranking.to_excel(RESULTS_DIR / "model_ranking_otimizado.xlsx", index=False)

    print("\n📊 Ranking IFI:")
    print(ranking[['Model', 'R2', 'RMSE', 'MAE', 'MAPE',
                   'IFI', 'Cleaning_Method', 'Samples']].to_string(index=False))

    weights_range = np.linspace(0.1, 0.8, 20)
    sens_scores   = np.array([
        w * results_df["R2"] + (1 - w) * (1 / results_df["RMSE"])
        for w in weights_range
    ])

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('IFI Sensitivity (Optimized)', fontsize=14, fontweight='bold')
    colors20 = plt.cm.tab20(np.linspace(0, 1, len(results_df["Model"])))
    for i, mod in enumerate(results_df['Model']):
        axes[0].plot(weights_range, sens_scores[:, i],
                     color=colors20[i], lw=1.5, alpha=0.8)
    axes[0].set_xlabel("R² weight (w)"); axes[0].set_ylabel("IFI Score")
    axes[0].set_title("All Models"); axes[0].grid(True, alpha=0.3)

    top10     = ranking.head(10)['Model'].tolist()
    top10_idx = [results_df[results_df['Model'] == m].index[0] for m in top10]
    c10 = plt.cm.viridis(np.linspace(0, 1, 10))
    for i, idx in enumerate(top10_idx):
        axes[1].plot(weights_range, sens_scores[:, idx],
                     color=c10[i], lw=2.5, label=results_df.iloc[idx]['Model'])
    axes[1].set_xlabel("R² weight (w)"); axes[1].set_ylabel("IFI Score")
    axes[1].set_title("Top 10 Models"); axes[1].grid(True, alpha=0.3)
    axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    plt.tight_layout()
    plt.savefig(FIG_IFI / "IFI_sensitivity_with_legend_otimizado.png",
                dpi=300, bbox_inches='tight')
    plt.close()

    sens_matrix = sens_scores.T
    plt.figure(figsize=(14, 10))
    sns.heatmap(sens_matrix,
                xticklabels=[f'{w:.1f}' for w in weights_range],
                yticklabels=results_df['Model'],
                cmap='viridis', annot=False)
    plt.xlabel('R² weight (w)'); plt.ylabel('Models')
    plt.title('IFI Sensitivity Heatmap (Optimized)')
    plt.tight_layout()
    plt.savefig(FIG_IFI / "IFI_sensitivity_heatmap_otimizado.png",
                dpi=300, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(10, 6))
    colors_r = plt.cm.viridis(np.linspace(0, 1, len(ranking)))
    bars = plt.barh(range(len(ranking)), ranking['IFI'], color=colors_r)
    plt.yticks(range(len(ranking)), ranking['Model'])
    plt.xlabel('IFI Score'); plt.title('Model Ranking – IFI Score')
    plt.gca().invert_yaxis()
    for i, bar in enumerate(bars):
        plt.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                 f'{ranking["IFI"].iloc[i]:.3f}', va='center')
    plt.tight_layout()
    plt.savefig(FIG_IFI / "IFI_Ranking_Otimizado.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ IFI Sensitivity plots salvos")

    return ranking
