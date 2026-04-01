import pandas as pd
from sklearn.preprocessing import StandardScaler


def predict_new_mixes(models_trained, scalers, X_full, feature_names):
    """Apply trained models to 18 hardcoded new concrete mix recipes (S19).

    Falls back to a StandardScaler fitted on X_full if the model's scaler fails.

    Returns
    -------
    pred_nd : DataFrame  — predictions per mix per model + Best_Model column
              Saved to Paper/Results/predicoes_novas_dosagens_otimizado.xlsx
    """
    print("\n🔮 PREDIÇÃO PARA NOVAS DOSAGENS...")

    novas = pd.DataFrame({
        "C":     [450, 630, 630, 630, 630, 630, 337.9, 213.5, 213.5, 550, 229.7, 213.8, 146.5, 252, 145.4, 122.6, 183.9, 141.3],
        "S":     [180, 180, 180, 180, 180, 180, 189, 0, 0, 0, 0, 98.1, 114.6, 0, 0, 183.9, 122.6, 212],
        "FA":    [0, 0, 0, 0, 0, 0, 0, 174.2, 174.2, 0, 118.2, 24.5, 89.3, 0, 178.9, 0, 0, 0],
        "SF":    [90, 90, 90, 90, 90, 90, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "LP":    [180, 0, 0, 0, 0, 90, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "W":     [144, 144, 162, 162, 162, 162, 174.9, 154.6, 154.6, 165, 195.2, 181.7, 201.9, 185, 201.7, 203.5, 203.5, 203.5],
        "SP":    [18, 18, 18, 18, 18, 18, 9.5, 11.7, 11.7, 3.85, 6.1, 6.7, 8.8, 0, 7.8, 0, 0, 0],
        "Gravel": [923, 923, 923, 923, 923, 923, 944.7, 1052.3, 1052.3, 1057, 1028.1, 1066, 860, 1111, 824, 958.2, 959.2, 971.8],
        "Sand":  [616, 616, 616, 616, 616, 616, 755.8, 775.5, 775.5, 705, 757.6, 785.5, 829.5, 784, 868.7, 800.1, 800, 748.5],
        "Age":   [180, 365, 365, 365, 180, 180, 56, 100, 28, 3, 100, 28, 28, 28, 28, 7, 3, 3],
    })[feature_names]

    predicoes = {}
    for name, m in models_trained.items():
        try:
            sc_m = scalers[name]
            predicoes[name] = m.predict(sc_m.transform(novas))
        except Exception:
            try:
                sc_fb = StandardScaler().fit(X_full.values)
                predicoes[name] = m.predict(sc_fb.transform(novas))
            except Exception:
                print(f"   ⚠️ {name}: erro na predição de novas dosagens")

    pred_nd = pd.DataFrame(predicoes)
    pred_nd.insert(0, "Mix_ID", range(1, len(novas) + 1))
    pred_nd["Best_Model"] = pred_nd.drop(columns="Mix_ID").idxmax(axis=1)
    pred_nd.to_excel("Paper/Results/predicoes_novas_dosagens_otimizado.xlsx", index=False)
    print("✅ Predições para novas dosagens salvas")

    return pred_nd
