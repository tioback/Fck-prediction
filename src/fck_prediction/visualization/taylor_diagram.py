import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error


def plot_taylor_diagram_initial(pred_ref, y_ref_ext):
    """First-pass Taylor Diagram — basic polar (S10).

    Saved to Figuras_NestedCV/ and Paper/Figures/.
    This is superseded by plot_taylor_diagram() (S24/NEW-1) which
    overwrites the same paths with RMSE arcs added.
    """
    if len(pred_ref) < 2:
        return None

    std_obs   = np.std(y_ref_ext)
    std_m_lst, corr_lst, rmse_lst, nm_lst = [], [], [], []

    for mn, pr in pred_ref.items():
        std_m_lst.append(np.std(pr))
        corr_lst.append(np.corrcoef(y_ref_ext, pr)[0, 1])
        rmse_lst.append(np.sqrt(mean_squared_error(y_ref_ext, pr)))
        nm_lst.append(mn)
        print(f"   {mn}: Corr={corr_lst[-1]:.4f}, "
              f"Std={std_m_lst[-1]:.2f}, RMSE={rmse_lst[-1]:.2f}")

    try:
        import skill_metrics as sm
        fig = plt.figure(figsize=(10, 8))
        sm.taylor_diagram(std_obs, std_m_lst, corr_lst,
                          markerLabel=nm_lst, markerSize=8,
                          markerLegend='on', colMarker='k',
                          styleOBS='-', colOBS='k', markerobs='o',
                          titleRMS='on', labelRMS='on')
        plt.title('Taylor Diagram – Model Comparison (Optimized)',
                  fontsize=14, fontweight='bold')
        plt.tight_layout()
    except ImportError:
        fig, ax = plt.subplots(figsize=(10, 8))
        theta  = np.linspace(0, np.pi / 2, 100)
        r_max  = max(max(std_m_lst), std_obs) * 1.2
        for c in [0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.99]:
            ang = np.arccos(c)
            ax.plot([0, r_max * np.cos(ang)], [0, r_max * np.sin(ang)],
                    'k--', alpha=0.3, lw=0.5)
            ax.text(r_max * np.cos(ang) * 1.02, r_max * np.sin(ang) * 1.02,
                    f'{c}', fontsize=8, alpha=0.7)
        for r in np.linspace(0.5, r_max, 5):
            ax.plot(r * np.cos(theta), r * np.sin(theta), 'k--', alpha=0.3, lw=0.5)
        ax.plot(std_obs, 0, 'ro', markersize=12, label='Observed', zorder=5)
        cols_t = plt.cm.tab10(np.linspace(0, 1, len(nm_lst)))
        for i, (n, sm_v, c) in enumerate(zip(nm_lst, std_m_lst, corr_lst)):
            ang = np.arccos(c)
            ax.plot(sm_v * np.cos(ang), sm_v * np.sin(ang), 'o',
                    color=cols_t[i], markersize=10, zorder=4)
            ax.annotate(n, (sm_v * np.cos(ang), sm_v * np.sin(ang)),
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        ax.set_xlim(0, r_max); ax.set_ylim(0, r_max)
        ax.set_aspect('equal')
        ax.set_xlabel('Std Dev'); ax.set_ylabel('Std Dev')
        ax.set_title('Taylor Diagram – Model Comparison', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.2)
        ax.legend(loc='upper right')

    plt.savefig('Figuras_NestedCV/Taylor_Diagram.png', dpi=300, bbox_inches='tight')
    plt.savefig('Paper/Figures/Taylor_Diagram.png',    dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Taylor Diagram salvo")

    taylor_stats = pd.DataFrame({
        'Model': nm_lst, 'Correlation': corr_lst,
        'Std_Dev_Predicted': std_m_lst,
        'Std_Dev_Observed':  [std_obs] * len(nm_lst),
        'RMSE': rmse_lst,
        'Bias': [np.mean(pred_ref[m] - y_ref_ext) for m in nm_lst],
    }).sort_values('Correlation', ascending=False)
    taylor_stats.to_excel('Resultados_Artigo/Taylor_Diagram_Statistics.xlsx', index=False)
    taylor_stats.to_excel('Paper/Results/Taylor_Diagram_Statistics.xlsx',    index=False)
    print(taylor_stats.to_string(index=False))

    return taylor_stats


def plot_taylor_diagram(pred_ref, y_ref_ext):
    """Extended Taylor Diagram with RMSE arcs [NEW-1] (S24).

    Overwrites paths from plot_taylor_diagram_initial() and adds
    Figures/Taylor_Diagram.png.
    """
    print("\n" + "=" * 80)
    print("TAYLOR DIAGRAM – ANÁLISE DE MODELOS [NEW-1]")
    print("=" * 80)

    if len(pred_ref) < 2:
        print("⚠️ Predições insuficientes para gerar Taylor Diagram")
        return None

    std_obs_td  = np.std(y_ref_ext)
    std_m_td, corr_td, rmse_td, nm_td = [], [], [], []

    for mn, pr in pred_ref.items():
        std_m_td.append(np.std(pr))
        corr_td.append(np.corrcoef(y_ref_ext, pr)[0, 1])
        rmse_td.append(np.sqrt(mean_squared_error(y_ref_ext, pr)))
        nm_td.append(mn)
        print(f"   {mn}: Corr={corr_td[-1]:.4f}, Std={std_m_td[-1]:.2f}, RMSE={rmse_td[-1]:.2f}")

    try:
        import skill_metrics as sm
        fig = plt.figure(figsize=(10, 8))
        sm.taylor_diagram(std_obs_td, std_m_td, corr_td,
                          markerLabel=nm_td, markerSize=8,
                          markerLegend='on', colMarker='k',
                          styleOBS='-', colOBS='k', markerobs='o',
                          titleRMS='on', tickRMS=[0.5, 1.0, 1.5, 2.0], labelRMS='on')
        plt.title('Taylor Diagram – Model Performance Comparison (Optimized Datasets)',
                  fontsize=14, fontweight='bold')
        plt.tight_layout()
    except ImportError:
        print("   skill_metrics não disponível. Usando implementação manual...")
        fig, ax = plt.subplots(figsize=(10, 8))
        theta  = np.linspace(0, np.pi / 2, 100)
        r_max  = max(max(std_m_td), std_obs_td) * 1.2
        for c in [0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.99]:
            ang = np.arccos(c)
            ax.plot([0, r_max * np.cos(ang)], [0, r_max * np.sin(ang)],
                    'k--', alpha=0.3, lw=0.5)
            ax.text(r_max * np.cos(ang) * 1.02, r_max * np.sin(ang) * 1.02,
                    f'{c}', fontsize=8, alpha=0.7)
        for r in np.linspace(0.5, r_max, 5):
            ax.plot(r * np.cos(theta), r * np.sin(theta), 'k--', alpha=0.3, lw=0.5)
            ax.text(r * 1.02, 0, f'{r:.1f}', fontsize=8, alpha=0.7, ha='center')
        ax.plot(std_obs_td, 0, 'ro', markersize=12, label='Observed', zorder=5)
        ax.text(std_obs_td + 0.02, -0.05, 'Obs', fontsize=10, fontweight='bold')
        cols_td = plt.cm.tab10(np.linspace(0, 1, len(nm_td)))
        for i, (n, sm_v, c) in enumerate(zip(nm_td, std_m_td, corr_td)):
            ang = np.arccos(c)
            ax.plot(sm_v * np.cos(ang), sm_v * np.sin(ang), 'o',
                    color=cols_td[i], markersize=10, zorder=4)
            ax.annotate(n, (sm_v * np.cos(ang), sm_v * np.sin(ang)),
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        for rmse_val in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
            phi = np.linspace(0, np.pi / 2, 100)
            xc  = std_obs_td + rmse_val * np.cos(phi)
            yc  = rmse_val * np.sin(phi)
            mask = xc >= 0
            ax.plot(xc[mask], yc[mask], 'g--', alpha=0.3, lw=0.5)
            mid = len(phi) // 2
            ax.text(xc[mid] + 0.05, yc[mid], f'RMSE={rmse_val}',
                    fontsize=7, alpha=0.6, rotation=30)
        ax.set_xlim(0, r_max); ax.set_ylim(0, r_max)
        ax.set_aspect('equal')
        ax.set_xlabel('Std Dev'); ax.set_ylabel('Std Dev')
        ax.set_title('Taylor Diagram – Model Performance Comparison (Optimized Datasets)',
                     fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.2); ax.legend(loc='upper right')

    for pth in ['Figuras_NestedCV/Taylor_Diagram.png',
                'Paper/Figures/Taylor_Diagram.png',
                'Figures/Taylor_Diagram.png']:
        plt.savefig(pth, dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Taylor Diagram salvo")

    taylor_stats_td = pd.DataFrame({
        'Model':              nm_td,
        'Correlation':        corr_td,
        'Std_Dev_Predicted':  std_m_td,
        'Std_Dev_Observed':   [std_obs_td] * len(nm_td),
        'RMSE':               rmse_td,
        'Bias':               [np.mean(pred_ref[m] - y_ref_ext) for m in nm_td],
    }).sort_values('Correlation', ascending=False)

    taylor_stats_td.to_excel('Resultados_Artigo/Taylor_Diagram_Statistics.xlsx', index=False)
    taylor_stats_td.to_excel('Paper/Results/Taylor_Diagram_Statistics.xlsx',    index=False)
    print("\nTAYLOR DIAGRAM – ESTATÍSTICAS:")
    print(taylor_stats_td.to_string(index=False))

    print("\n✅ Taylor Diagram concluído!")
    return taylor_stats_td
