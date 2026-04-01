import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_model_correlation(pred_ref):
    """Pearson correlation heatmap of model predictions on the common test set (S13)."""
    print("\n📊 CORRELAÇÃO DOS MODELOS (referência comum) ...")

    pred_matrix = pd.DataFrame(pred_ref)
    corr_matrix = pred_matrix.corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm',
                center=0, square=True, linewidths=0.5)
    plt.title('Pearson Correlation – Model Predictions (Common Test Set)',
              fontsize=14)
    plt.tight_layout()
    plt.savefig('Figuras_Correlacao/Correlacao_Performance_Modelos_Otimizado.png',
                dpi=300, bbox_inches='tight')
    plt.savefig('Paper/Figures/Correlacao_Performance_Modelos_Otimizado.png',
                dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Heatmap de correlação salvo")
