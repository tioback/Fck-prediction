import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import DBSCAN
from sklearn.svm import OneClassSVM


# ── Individual cleaning functions (S03) ──────────────────────────────────────

def clean_isolationforest(X, y, contamination=0.02, random_state=42):
    iso = IsolationForest(contamination=contamination,
                          random_state=random_state, n_jobs=-1)
    mask = iso.fit_predict(X) == 1
    return X[mask], y[mask]


def clean_iqr(X, y, multiplier=1.5):
    mask = np.ones(len(X), dtype=bool)
    for col in X.columns:
        Q1, Q3 = X[col].quantile(0.25), X[col].quantile(0.75)
        IQR = Q3 - Q1
        mask &= (X[col] >= Q1 - multiplier * IQR) & (X[col] <= Q3 + multiplier * IQR)
    return X[mask], y[mask]


def clean_zscore(X, y, threshold=3):
    z = np.abs(stats.zscore(X, nan_policy='omit'))
    mask = (z < threshold).all(axis=1)
    return X[mask], y[mask]


def clean_percentile(X, y, lower=0.01, upper=0.99):
    mask = np.ones(len(X), dtype=bool)
    for col in X.columns:
        mask &= (X[col] >= X[col].quantile(lower)) & \
                (X[col] <= X[col].quantile(upper))
    return X[mask], y[mask]


def clean_dbscan(X, y, eps=0.5, min_samples=5):
    Xs = StandardScaler().fit_transform(X)
    labels = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1).fit_predict(Xs)
    mask = labels != -1
    return X[mask], y[mask]


def clean_elliptic(X, y, contamination=0.02):
    Xs = StandardScaler().fit_transform(X)
    mask = EllipticEnvelope(contamination=contamination,
                            random_state=42,
                            support_fraction=0.9).fit_predict(Xs) == 1
    return X[mask], y[mask]


def clean_svm(X, y, nu=0.02):
    Xs = StandardScaler().fit_transform(X)
    mask = OneClassSVM(nu=nu, kernel='rbf', gamma='scale',
                       max_iter=1000).fit_predict(Xs) == 1
    return X[mask], y[mask]


def clean_lof(X, y, contamination=0.02):
    Xs = StandardScaler().fit_transform(X)
    mask = LocalOutlierFactor(contamination=contamination,
                              n_neighbors=20, n_jobs=-1).fit_predict(Xs) == 1
    return X[mask], y[mask]


# ── Registry ──────────────────────────────────────────────────────────────────

def get_cleaning_methods():
    """Return the dict of all 16 cleaning method variants (S03).

    Returns
    -------
    cleaning_methods : dict[str, {'func': callable}]
    """
    print("\n🧹 DEFININDO MÉTODOS DE LIMPEZA...")

    cleaning_methods = {
        'Sem_Limpeza':          {'func': lambda X, y: (X.copy(), y.copy())},
        'IsolationForest_1%':   {'func': lambda X, y: clean_isolationforest(X, y, 0.01)},
        'IsolationForest_2%':   {'func': lambda X, y: clean_isolationforest(X, y, 0.02)},
        'IsolationForest_3%':   {'func': lambda X, y: clean_isolationforest(X, y, 0.03)},
        'IsolationForest_5%':   {'func': lambda X, y: clean_isolationforest(X, y, 0.05)},
        'IQR_1.5':              {'func': lambda X, y: clean_iqr(X, y, 1.5)},
        'IQR_2.0':              {'func': lambda X, y: clean_iqr(X, y, 2.0)},
        'IQR_3.0':              {'func': lambda X, y: clean_iqr(X, y, 3.0)},
        'ZScore_2':             {'func': lambda X, y: clean_zscore(X, y, 2)},
        'ZScore_3':             {'func': lambda X, y: clean_zscore(X, y, 3)},
        'Percentil_1_99':       {'func': lambda X, y: clean_percentile(X, y, 0.01, 0.99)},
        'Percentil_5_95':       {'func': lambda X, y: clean_percentile(X, y, 0.05, 0.95)},
        'DBSCAN':               {'func': lambda X, y: clean_dbscan(X, y)},
        'EllipticEnvelope':     {'func': lambda X, y: clean_elliptic(X, y)},
        'OneClassSVM':          {'func': lambda X, y: clean_svm(X, y)},
        'LocalOutlierFactor':   {'func': lambda X, y: clean_lof(X, y)},
    }

    print(f"✅ {len(cleaning_methods)} métodos de limpeza definidos")
    return cleaning_methods
