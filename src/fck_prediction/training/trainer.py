import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (r2_score, mean_squared_error,
                             mean_absolute_error,
                             mean_absolute_percentage_error)

from fck_prediction.config import TARGET


def train_models(X_full, y_full, models, optimized_datasets,
                 best_cleaning_for_model, feature_names):
    """Train all models on their optimised DEV datasets and record metrics (S06).

    Also builds the common reference partition (random_state=42, 80/20) that is
    shared by all comparative analyses [FIX-2].

    Returns
    -------
    dict with keys:
        models_trained, scalers, pred_ext, pred_ref, pred_train_dict,
        results_df, train_metrics, error_variance,
        y_ref_ext, X_tst_ref_sc, X_dev_ref_sc, sc_ref
    """
    print("\n" + "=" * 80)
    print("🚀 TREINAMENTO DOS MODELOS COM DATASETS OTIMIZADOS")
    print("=" * 80)

    # ── Partição de referência comum ─────────────────────────────────────────
    X_dev_ref, X_tst_ref, y_dev_ref, y_tst_ref = train_test_split(
        X_full, y_full, test_size=0.2, random_state=42)
    sc_ref = StandardScaler()
    X_dev_ref_sc = sc_ref.fit_transform(X_dev_ref)
    X_tst_ref_sc = sc_ref.transform(X_tst_ref)
    y_ref_ext = y_tst_ref.values

    results         = []
    pred_ext        = {}
    pred_ref        = {}
    pred_train_dict = {}
    models_trained  = {}
    scalers         = {}
    train_metrics   = []
    error_variance  = []

    for model_name, model in models.items():
        print(f"\n📈 Treinando: {model_name}")

        info    = optimized_datasets[model_name]
        dev_df  = info['dev_df']
        sc_mod  = info['scaler']
        X_tst_r = info['X_tst_raw']
        y_tst_r = info['y_tst_raw']

        X_dev_np = dev_df[feature_names].values
        y_dev_np = dev_df[TARGET].values

        X_tst_sc = sc_mod.transform(X_tst_r)

        try:
            m = clone(model)
            m.fit(X_dev_np, y_dev_np)

            p_tr = m.predict(X_dev_np)
            p_te = m.predict(X_tst_sc)

            p_ref = m.predict(X_tst_ref_sc)
            pred_ref[model_name]        = p_ref

            pred_train_dict[model_name] = p_tr
            pred_ext[model_name]        = p_te
            models_trained[model_name]  = m
            scalers[model_name]         = sc_mod

            y_tst_np = y_tst_r.values

            for ds, y_ds, p_ds in [('Training', y_dev_np, p_tr),
                                    ('Testing',  y_tst_np, p_te)]:
                r2_v   = r2_score(y_ds, p_ds)
                rmse_v = np.sqrt(mean_squared_error(y_ds, p_ds))
                mae_v  = mean_absolute_error(y_ds, p_ds)
                mape_v = mean_absolute_percentage_error(y_ds, p_ds) * 100
                train_metrics.append({"Model": model_name, "Metric": "R2",
                                       "Value": r2_v,   "Set": ds})
                train_metrics.append({"Model": model_name, "Metric": "RMSE",
                                       "Value": rmse_v, "Set": ds})
                train_metrics.append({"Model": model_name, "Metric": "MAE",
                                       "Value": mae_v,  "Set": ds})
                train_metrics.append({"Model": model_name, "Metric": "MAPE",
                                       "Value": mape_v, "Set": ds})
                error_variance.append({"Model": model_name, "Set": ds,
                                       "Metric": "Var_Error",
                                       "Value": np.var(y_ds - p_ds)})

            r2_te   = r2_score(y_tst_np, p_te)
            rmse_te = np.sqrt(mean_squared_error(y_tst_np, p_te))
            mae_te  = mean_absolute_error(y_tst_np, p_te)
            mape_te = mean_absolute_percentage_error(y_tst_np, p_te) * 100

            results.append({
                "Model":          model_name,
                "R2":             r2_te,
                "RMSE":           rmse_te,
                "MAE":            mae_te,
                "MAPE":           mape_te,
                "Cleaning_Method":best_cleaning_for_model[model_name],
                "Samples":        len(y_dev_np),
            })

            r2_tr   = r2_score(y_dev_np, p_tr)
            rmse_tr = np.sqrt(mean_squared_error(y_dev_np, p_tr))
            mape_tr = mean_absolute_percentage_error(y_dev_np, p_tr) * 100
            print(f"   ✅ Training: R²={r2_tr:.4f} | RMSE={rmse_tr:.2f} | MAPE={mape_tr:.1f}%")
            print(f"      Testing:  R²={r2_te:.4f} | RMSE={rmse_te:.2f} | MAPE={mape_te:.1f}%")
            print(f"      Método: {best_cleaning_for_model[model_name]} | Dev: {len(y_dev_np)} amostras")

        except Exception as e:
            print(f"   ❌ Erro: {str(e)[:100]}")

    results_df = pd.DataFrame(results)
    results_df.to_excel('Resultados_Artigo/Results_Otimizado.xlsx', index=False)
    print("\n📈 RESUMO (TESTE):")
    print(results_df.sort_values('R2', ascending=False))

    return {
        'models_trained':  models_trained,
        'scalers':         scalers,
        'pred_ext':        pred_ext,
        'pred_ref':        pred_ref,
        'pred_train_dict': pred_train_dict,
        'results_df':      results_df,
        'train_metrics':   train_metrics,
        'error_variance':  error_variance,
        'y_ref_ext':       y_ref_ext,
        'X_tst_ref_sc':    X_tst_ref_sc,
        'X_dev_ref_sc':    X_dev_ref_sc,
        'sc_ref':          sc_ref,
    }
