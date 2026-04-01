import pandas as pd

from fck_prediction.config import DATA_FILE, COLUMN_NAMES, TARGET


def load_data(filepath=DATA_FILE):
    """Load and minimally clean the concrete dataset (S01).

    Returns
    -------
    df            : cleaned DataFrame
    X_full        : feature DataFrame
    y_full        : target Series
    feature_names : list[str]
    """
    print("\n📥 CARREGANDO DADOS ORIGINAIS...")

    df = pd.read_excel(filepath)
    print(f"📊 Shape original: {df.shape}")

    if df.shape[1] == 11:
        df.columns = COLUMN_NAMES
        print("✅ Colunas renomeadas para formato padrão")

    print("\n🧹 Convertendo para numérico (mantendo todos os dados)...")
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.dropna().reset_index(drop=True)

    print(f"📊 Shape final: {df.shape}")
    print(f"📋 Colunas: {df.columns.tolist()}")
    print(f"\n📈 Estatísticas descritivas:")
    print(df.describe())

    feature_names = [col for col in df.columns if col != TARGET]

    X_full = df[feature_names]
    y_full = df[TARGET]

    print(f"\n🎯 Target: {TARGET}")
    print(f"🔢 Features: {feature_names}")
    print(f"📊 Total de amostras: {len(df)}")

    return df, X_full, y_full, feature_names
