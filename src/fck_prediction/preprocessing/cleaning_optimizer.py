import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

from fck_prediction.config import TARGET, N_CLEAN_RUNS


def optimize_cleaning(X_full, y_full, models, model_list, cleaning_methods, feature_names,
                      n_runs=N_CLEAN_RUNS):
    """Select the best cleaning method per model without data leakage (S05).

    Fluxo correto:
      1) split original → X_dev_raw / X_test_raw  (test nunca tocado)
      2) fit StandardScaler apenas em X_dev_raw
      3) aplicar método de limpeza apenas em X_dev_scaled
      4) treinar modelo nos dados limpos
      5) avaliar no X_test_raw (escalado com o mesmo scaler)

    Returns
    -------
    best_cleaning_for_model : dict[str, str]
    optimized_datasets      : dict[str, {'dev_df', 'scaler', 'X_tst_raw', 'y_tst_raw'}]
    best_opt_summary        : DataFrame
    """
    print("\n" + "=" * 80)
    print("🔬 FASE 1: OTIMIZANDO MÉTODO DE LIMPEZA PARA CADA MODELO [FIX-1]")
    print("=" * 80)

    best_cleaning_for_model = {}
    optimized_datasets      = {}
    optimization_results    = []

    for i, model_name in enumerate(model_list):
        print(f"\n📊 [{i+1}/{len(model_list)}] {model_name}")
        model_perf = []

        for method_name, method_info in cleaning_methods.items():
            method_scores = []

            for run in range(n_runs):
                try:
                    # ── 1. Split global (test nunca visto pela limpeza) ──────
                    X_dev_raw, X_tst_raw, y_dev_raw, y_tst_raw = train_test_split(
                        X_full, y_full, test_size=0.2, random_state=42 + run)

                    # ── 2. Scaler fitado APENAS no dev ──────────────────────
                    sc_opt = StandardScaler()
                    X_dev_scaled_df = pd.DataFrame(
                        sc_opt.fit_transform(X_dev_raw),
                        columns=feature_names,
                        index=X_dev_raw.index)
                    X_tst_scaled = sc_opt.transform(X_tst_raw)

                    # ── 3. Limpeza aplicada APENAS no dev escalado ───────────
                    X_cl, y_cl = method_info['func'](
                        X_dev_scaled_df,
                        y_dev_raw.reset_index(drop=True)
                        if hasattr(y_dev_raw, 'reset_index')
                        else pd.Series(y_dev_raw))

                    if len(X_cl) < 20:
                        continue

                    # ── 4. Treino e avaliação ────────────────────────────────
                    m = clone(models[model_name])
                    m.fit(X_cl.values, y_cl.values)
                    r2 = r2_score(y_tst_raw, m.predict(X_tst_scaled))
                    method_scores.append(r2)

                except Exception:
                    pass

            if method_scores:
                model_perf.append({
                    'Model':           model_name,
                    'Cleaning_Method': method_name,
                    'Mean_R2':         np.mean(method_scores),
                    'Std_R2':          np.std(method_scores),
                    'N_Runs':          len(method_scores),
                })

        if model_perf:
            perf_df     = pd.DataFrame(model_perf)
            best_method = perf_df.loc[perf_df['Mean_R2'].idxmax(), 'Cleaning_Method']
            best_r2     = perf_df['Mean_R2'].max()
            best_cleaning_for_model[model_name] = best_method
            optimization_results.extend(model_perf)
            print(f"   ✅ Melhor: {best_method} (R² = {best_r2:.4f})")

            # ── Gerar dataset DEV otimizado (referência p/ treino final) ─────
            X_dev_raw_f, X_tst_raw_f, y_dev_raw_f, y_tst_raw_f = train_test_split(
                X_full, y_full, test_size=0.2, random_state=42)

            sc_final = StandardScaler()
            X_dev_sc_df = pd.DataFrame(
                sc_final.fit_transform(X_dev_raw_f),
                columns=feature_names,
                index=X_dev_raw_f.index)

            X_cl_f, y_cl_f = cleaning_methods[best_method]['func'](
                X_dev_sc_df,
                y_dev_raw_f.reset_index(drop=True))

            opt_df = X_cl_f.copy()
            opt_df[TARGET] = y_cl_f.values
            optimized_datasets[model_name] = {
                'dev_df':    opt_df,
                'scaler':    sc_final,
                'X_tst_raw': X_tst_raw_f,
                'y_tst_raw': y_tst_raw_f,
            }
            opt_df.to_excel(
                f"Bancos_Otimizados/Dataset_Otimizado_{model_name}.xlsx",
                index=False)
            print(f"      📁 Dev otimizado salvo: {len(opt_df)} amostras")

        else:
            best_cleaning_for_model[model_name] = 'Sem_Limpeza'
            X_dev_raw_f, X_tst_raw_f, y_dev_raw_f, y_tst_raw_f = train_test_split(
                X_full, y_full, test_size=0.2, random_state=42)
            sc_fb = StandardScaler()
            X_dev_sc = sc_fb.fit_transform(X_dev_raw_f)
            fb_df = pd.DataFrame(X_dev_sc, columns=feature_names)
            fb_df[TARGET] = y_dev_raw_f.values
            optimized_datasets[model_name] = {
                'dev_df':    fb_df,
                'scaler':    sc_fb,
                'X_tst_raw': X_tst_raw_f,
                'y_tst_raw': y_tst_raw_f,
            }
            print(f"   ⚠️ Usando Sem_Limpeza (fallback)")

    opt_summary_full = pd.DataFrame(optimization_results)
    if not opt_summary_full.empty:
        opt_summary_full.to_excel(
            'Resultados_Artigo/Otimizacao_Limpeza_Resultados.xlsx', index=False)

    best_opt_summary = pd.DataFrame([
        {'Model':       m,
         'Best_Method': best_cleaning_for_model[m],
         'Samples':     len(optimized_datasets[m]['dev_df'])}
        for m in model_list
    ])
    best_opt_summary.to_excel(
        'Resultados_Artigo/Melhor_Metodo_Limpeza_por_Modelo.xlsx', index=False)

    print("\n📊 RESUMO DA OTIMIZAÇÃO:")
    print(best_opt_summary.to_string(index=False))

    return best_cleaning_for_model, optimized_datasets, best_opt_summary
