import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from fck_prediction.config import TRAINING_COLOR, TESTING_COLOR


def plot_performance_metrics(train_metrics, error_variance):
    """Boxplots, error variance, and violin plots for train/test metrics (S11).

    Parameters
    ----------
    train_metrics  : list of dicts from training/trainer.py
    error_variance : list of dicts from training/trainer.py
    """
    print("\n" + "=" * 80)
    print("📊 GRÁFICOS DE DESEMPENHO")
    print("=" * 80)

    metrics_df   = pd.DataFrame(train_metrics)
    error_var_df = pd.DataFrame(error_variance)

    # ── Boxplots completos ────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Performance Metrics – Training vs Testing (Optimized)',
                 fontsize=14, fontweight='bold', y=0.98)
    for idx, (metric, title) in enumerate(
            zip(['RMSE', 'R2', 'MAE', 'MAPE'],
                ['RMSE (MPa)', 'R²', 'MAE (MPa)', 'MAPE (%)'])):
        ax = axes[idx // 2, idx % 2]
        data = metrics_df[metrics_df['Metric'] == metric]
        if data.empty:
            continue
        asc = metric != 'R2'
        order = (data[data['Set'] == 'Testing'].groupby('Model')['Value']
                 .median().sort_values(ascending=asc).index)
        pos = np.arange(len(order)); w = 0.35
        tv = [data[(data['Model'] == m) & (data['Set'] == 'Training')]['Value'].values
              for m in order]
        ev = [data[(data['Model'] == m) & (data['Set'] == 'Testing')]['Value'].values
              for m in order]
        ax.boxplot(tv, positions=pos - w / 2, widths=w, patch_artist=True,
                   boxprops=dict(facecolor=TRAINING_COLOR, alpha=0.7),
                   medianprops=dict(color='black', linewidth=2))
        ax.boxplot(ev, positions=pos + w / 2, widths=w, patch_artist=True,
                   boxprops=dict(facecolor=TESTING_COLOR, alpha=0.7),
                   medianprops=dict(color='black', linewidth=2))
        ax.set_xticks(pos)
        ax.set_xticklabels(order, rotation=45, ha='right', fontsize=9)
        ax.set_ylabel(title, fontsize=11)
        ax.set_title(f'({chr(97+idx)}) {metric}', fontsize=12, fontweight='bold', loc='left')
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')
        if idx == 0:
            ax.legend([plt.Rectangle((0, 0), 1, 1, facecolor=TRAINING_COLOR),
                       plt.Rectangle((0, 0), 1, 1, facecolor=TESTING_COLOR)],
                      ['Training', 'Testing'], loc='upper right')
    plt.tight_layout()
    plt.savefig("Figures/Boxplots_All_Metrics_Otimizado.png",       dpi=300, bbox_inches='tight')
    plt.savefig("Paper/Figures/Boxplots_All_Metrics_Otimizado.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Boxplots completos salvos")

    # ── Variância do erro ─────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 6))
    vd = error_var_df.copy()
    mo_v = (vd[vd['Set'] == 'Testing'].groupby('Model')['Value']
            .mean().sort_values().index)
    pos = np.arange(len(mo_v)); w = 0.35
    tv = [vd[(vd['Model'] == m) & (vd['Set'] == 'Training')]['Value'].values for m in mo_v]
    ev = [vd[(vd['Model'] == m) & (vd['Set'] == 'Testing')]['Value'].values  for m in mo_v]
    ax.boxplot(tv, positions=pos - w / 2, widths=w, patch_artist=True,
               boxprops=dict(facecolor=TRAINING_COLOR, alpha=0.7))
    ax.boxplot(ev, positions=pos + w / 2, widths=w, patch_artist=True,
               boxprops=dict(facecolor=TESTING_COLOR, alpha=0.7))
    ax.set_xticks(pos)
    ax.set_xticklabels(mo_v, rotation=45, ha='right', fontsize=10)
    ax.set_ylabel('Error Variance', fontsize=12)
    ax.set_title('Error Variance – Training vs Testing', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend([plt.Rectangle((0, 0), 1, 1, facecolor=TRAINING_COLOR),
               plt.Rectangle((0, 0), 1, 1, facecolor=TESTING_COLOR)],
              ['Training', 'Testing'], loc='upper right')
    plt.tight_layout()
    plt.savefig("Figures/Error_Variance_Boxplot_Otimizado.png",       dpi=300, bbox_inches='tight')
    plt.savefig("Paper/Figures/Error_Variance_Boxplot_Otimizado.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Error Variance salvo")

    # ── Violin plots ──────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Violin Plots – Performance Metrics (Optimized)',
                 fontsize=14, fontweight='bold', y=0.98)
    for idx, (metric, title) in enumerate(
            zip(['R2', 'RMSE', 'MAE', 'MAPE'],
                ['R²', 'RMSE (MPa)', 'MAE (MPa)', 'MAPE (%)'])):
        ax = axes[idx // 2, idx % 2]
        data = metrics_df[metrics_df['Metric'] == metric]
        if data.empty:
            ax.text(0.5, 0.5, f'No data', ha='center', va='center')
            continue
        asc = metric != 'R2'
        order = (data[data['Set'] == 'Testing'].groupby('Model')['Value']
                 .median().sort_values(ascending=asc).index)
        sns.violinplot(data=data, x='Model', y='Value', hue='Set', ax=ax,
                       order=order,
                       palette={'Training': TRAINING_COLOR, 'Testing': TESTING_COLOR},
                       split=False, cut=0, inner='box', linewidth=1)
        ax.set_title(title, fontsize=12, fontweight='bold', loc='left')
        ax.set_xlabel(''); ax.set_ylabel(title)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3, axis='y')
        if idx == 0: ax.legend(title='Dataset')
        else:        ax.legend_.remove()
    plt.tight_layout()
    plt.savefig("Figuras_Violin/Violin_Plots_All_Metrics_Otimizado.png",
                dpi=300, bbox_inches='tight')
    plt.savefig("Paper/Figures/Violin_Plots_All_Metrics_Otimizado.png",
                dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Violin plots salvos")
