import pandas as pd


def compute_summary_stats(results_df):
    """Descriptive stats table per model per metric (S17).

    Saves mean, std, median, min, max, Q1, Q3, IQR pivot tables.
    """
    print("\n📊 Tabela resumo das estatísticas...")
    try:
        stats_summ = []
        for model in results_df['Model'].unique():
            md = results_df[results_df['Model'] == model]
            for metric in ['R2', 'RMSE', 'MAE', 'MAPE']:
                if metric in results_df.columns:
                    stats_summ.append({
                        'Model': model, 'Metric': metric,
                        'Mean':   md[metric].mean(),
                        'Std':    md[metric].std(),
                        'Median': md[metric].median(),
                        'Min':    md[metric].min(),
                        'Max':    md[metric].max(),
                        'Q1':     md[metric].quantile(0.25),
                        'Q3':     md[metric].quantile(0.75),
                        'IQR':    md[metric].quantile(0.75) - md[metric].quantile(0.25),
                    })
        if stats_summ:
            st_df = pd.DataFrame(stats_summ)
            st_df.to_excel(
                'Resultados_Artigo/Boxplot_Statistics_Detailed_Otimizado.xlsx',
                index=False)
            st_df.pivot_table(values='Mean', index='Model', columns='Metric').to_excel(
                'Resultados_Artigo/Boxplot_Means_Otimizado.xlsx')
            st_df.pivot_table(values='Std',  index='Model', columns='Metric').to_excel(
                'Resultados_Artigo/Boxplot_Stds_Otimizado.xlsx')
            print("✅ Tabelas de estatísticas salvas")
    except Exception as e:
        print(f"   ❌ {e}")
