import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_radar(results_df, ranking):
    """Multi-metric radar chart — 3 versions [NEW-2] (S25).

    Parameters
    ----------
    results_df : DataFrame with columns R2, RMSE, MAE, MAPE, Model
    ranking    : DataFrame sorted by IFI (must have 'Model' column)
    """
    print("\n" + "=" * 80)
    print("RADAR CHART – COMPARAÇÃO MULTI-MÉTRICA [NEW-2]")
    print("=" * 80)

    if results_df is None or len(results_df) == 0:
        print("⚠️ results_df não disponível para Radar Chart")
        return

    metrics_radar = ['R2', 'RMSE', 'MAE', 'MAPE']
    labels_radar  = ['R²', 'RMSE (MPa)', 'MAE (MPa)', 'MAPE (%)']
    dir_radar     = {'R2': 1, 'RMSE': -1, 'MAE': -1, 'MAPE': -1}

    models_list_radar = results_df['Model'].tolist()
    radar_data = []
    for metric, label in zip(metrics_radar, labels_radar):
        vals = [results_df[results_df['Model'] == m][metric].values[0]
                for m in models_list_radar]
        mn, mx = min(vals), max(vals)
        if dir_radar[metric] == 1:
            norm = [(v - mn) / (mx - mn) if mx != mn else 0.5 for v in vals]
        else:
            norm = [(mx - v) / (mx - mn) if mx != mn else 0.5 for v in vals]
        radar_data.append({'metric': metric, 'label': label,
                           'values': vals, 'normalized': norm,
                           'min': mn, 'max': mx})

    N      = len(metrics_radar)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    colors_radar = plt.cm.tab10(np.linspace(0, 1, len(models_list_radar)))

    def _radar_plot(fig_path_list, title, models_sel, colors_sel, idx_map, tick_labels=None):
        fig, ax = plt.subplots(figsize=(12, 10), subplot_kw={'projection': 'polar'})
        for cidx, model in enumerate(models_sel):
            midx = idx_map[model]
            vals = [rd['normalized'][midx] for rd in radar_data] + \
                   [radar_data[0]['normalized'][midx]]
            ax.plot(angles, vals, 'o-', lw=2, color=colors_sel[cidx],
                    label=model, alpha=0.8)
            ax.fill(angles, vals, alpha=0.1, color=colors_sel[cidx])
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(tick_labels if tick_labels else
                           [rd['label'] for rd in radar_data], fontsize=11)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=9)
        ax.grid(True, alpha=0.3)
        plt.title(title, fontsize=14, fontweight='bold', pad=20)
        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=9)
        plt.tight_layout()
        for pth in fig_path_list:
            plt.savefig(pth, dpi=300, bbox_inches='tight')
        plt.close()

    idx_map_all = {m: i for i, m in enumerate(models_list_radar)}

    # Versão 1 – todos os modelos, métricas normalizadas
    _radar_plot(
        ['Figuras_Radar/Radar_Chart_Normalized.png',
         'Figuras_NestedCV/Radar_Chart_Normalized.png',
         'Paper/Figures/Radar_Chart_Normalized.png',
         'Figures/Radar_Chart_Normalized.png'],
        'Radar Chart – Multi-Metric Model Comparison (Optimized Datasets)\n'
        '(Higher score = better performance)',
        models_list_radar, colors_radar, idx_map_all)
    print("✅ Radar Chart (Normalized) salvo")

    # Versão 2 – com intervalos reais nos labels
    range_labels = [f"{rd['label']}\n[{rd['min']:.2f} – {rd['max']:.2f}]"
                    for rd in radar_data]
    _radar_plot(
        ['Figuras_Radar/Radar_Chart_With_Ranges.png',
         'Figuras_NestedCV/Radar_Chart_With_Ranges.png',
         'Paper/Figures/Radar_Chart_With_Ranges.png',
         'Figures/Radar_Chart_With_Ranges.png'],
        'Radar Chart – Model Comparison with Metric Ranges\n'
        '(Higher score = better performance)',
        models_list_radar, colors_radar, idx_map_all, tick_labels=range_labels)
    print("✅ Radar Chart (With Ranges) salvo")

    # Versão 3 – Top 5 por IFI
    top5_radar = ranking.head(5)['Model'].tolist()
    cols_top5  = plt.cm.viridis(np.linspace(0, 1, len(top5_radar)))
    _radar_plot(
        ['Figuras_Radar/Radar_Chart_Top5_Models.png',
         'Figuras_NestedCV/Radar_Chart_Top5_Models.png',
         'Paper/Figures/Radar_Chart_Top5_Models.png',
         'Figures/Radar_Chart_Top5_Models.png'],
        'Radar Chart – Top 5 Models (IFI Ranking)\n(Higher score = better performance)',
        top5_radar, cols_top5, idx_map_all)
    print("✅ Radar Chart (Top 5) salvo")

    # ── Tabela e ranking de score médio radar ─────────────────────────────────
    radar_norm_df = pd.DataFrame({'Model': models_list_radar})
    for rd in radar_data:
        radar_norm_df[rd['label']]           = rd['normalized']
        radar_norm_df[f"{rd['label']}_raw"]  = rd['values']
    radar_norm_df['Radar_Score'] = radar_norm_df[
        [rd['label'] for rd in radar_data]].mean(axis=1)
    radar_ranking_df = radar_norm_df[['Model', 'Radar_Score']].sort_values(
        'Radar_Score', ascending=False)

    radar_norm_df.to_excel('Resultados_Artigo/Radar_Chart_Normalized_Data.xlsx', index=False)
    radar_norm_df.to_excel('Paper/Results/Radar_Chart_Normalized_Data.xlsx',    index=False)
    radar_ranking_df.to_excel('Resultados_Artigo/Radar_Chart_Ranking.xlsx', index=False)
    radar_ranking_df.to_excel('Paper/Results/Radar_Chart_Ranking.xlsx',    index=False)

    print("\n📊 RANKING POR SCORE MÉDIO RADAR:")
    print(radar_ranking_df.to_string(index=False))

    print("\n📊 ESTATÍSTICAS DO RADAR:")
    for rd in radar_data:
        best_i  = np.argmax(rd['normalized'])
        worst_i = np.argmin(rd['normalized'])
        print(f"\n{rd['label']}:")
        print(f"   Min raw: {rd['min']:.4f} | Max raw: {rd['max']:.4f}")
        print(f"   Média: {np.mean(rd['values']):.4f} | Desvio: {np.std(rd['values']):.4f}")
        print(f"   Melhor: {models_list_radar[best_i]} | Pior: {models_list_radar[worst_i]}")

    print("\n✅ Radar Chart concluído!")
