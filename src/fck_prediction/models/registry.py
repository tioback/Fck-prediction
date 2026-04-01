from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor


def get_models():
    """Instantiate and return all model definitions (S02).

    Returns
    -------
    models     : dict[str, estimator]
    model_list : list[str]
    """
    print("\n🤖 DEFININDO MODELOS...")

    models = {
        "Linear":          LinearRegression(),
        "BayesianRidge":   BayesianRidge(),
        "DecisionTree":    DecisionTreeRegressor(random_state=42),
        "RandomForest":    RandomForestRegressor(n_estimators=200,
                                                 random_state=42, n_jobs=-1),
        "GradientBoosting":GradientBoostingRegressor(n_estimators=200,
                                                      random_state=42),
        "SVR_rbf":         SVR(kernel='rbf',  max_iter=2000),
        "SVR_poly":        SVR(kernel='poly', max_iter=2000),
        "XGBoost":         XGBRegressor(n_estimators=200, random_state=42,
                                        n_jobs=-1, verbosity=0),
        "ANN":             MLPRegressor(hidden_layer_sizes=(100, 50),
                                        max_iter=2000, random_state=42,
                                        early_stopping=True),
    }

    model_list = list(models.keys())
    print(f"✅ {len(models)} modelos carregados:")
    for i, (n, m) in enumerate(models.items(), 1):
        print(f"   {i:2d}. {n} – {type(m).__name__}")

    return models, model_list
