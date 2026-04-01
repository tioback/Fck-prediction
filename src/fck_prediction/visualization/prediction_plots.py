import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error


def plot_predictions(pred_ref, y_ref_ext):
    """Parity plots, residual plots, KDE density and MAPE scatter (S16).

    All plots are generated on the common reference partition [FIX-2].
    """
    # ── Parity plots ──────────────────────────────────────────────────────────
    print("\n📊 GERANDO PARITY PLOTS (referência comum)...")
    for name, pr in pred_ref.items():
        y_pl = y_ref_ext
        p_pl = pr
        r2v  = r2_score(y_pl, p_pl)
        rmse = np.sqrt(mean_squared_error(y_pl, p_pl))

        plt.figure(figsize=(7, 7))
        plt.scatter(y_pl, p_pl, alpha=0.5, s=40, color='#1f77b4',
                    edgecolors='k', linewidth=0.5, label='Data')
        lims = [min(y_pl.min(), p_pl.min()), max(y_pl.max(), p_pl.max())]
        plt.plot(lims, lims, 'r--', lw=2, label='Ideal (1:1)', zorder=3)
        plt.title(f"Validação Externa: {name}\n"
                  f"$R^2$={r2v:.4f} | RMSE={rmse:.2f} MPa", fontsize=12, pad=15)
        plt.xlabel("Experimental $f_{ck}$ (MPa)", fontsize=10)
        plt.ylabel("Predicted $f_{ck}$ (MPa)",    fontsize=10)
        plt.legend(loc='upper left'); plt.grid(True, linestyle=':', alpha=0.5)
        plt.xlim(lims); plt.ylim(lims)
        plt.tight_layout()
        fn = f"Paper/Figures/Validacao_Externa/Scatter_{name.replace(' ', '_')}.png"
        plt.savefig(fn, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✅ {name}")
    print("✅ Parity plots gerados")

    # ── Residual plots ────────────────────────────────────────────────────────
    print("\n📊 GERANDO RESIDUAL PLOTS...")
    for name, pr in pred_ref.items():
        residuals = y_ref_ext - pr
        plt.figure(figsize=(8, 5))
        plt.scatter(pr, residuals, alpha=0.6, s=40, color='#2ca02c',
                    edgecolors='k', linewidth=0.5)
        plt.axhline(0, color='r', linestyle='--', lw=2)
        plt.title(f"Residual Analysis – {name}", fontsize=12)
        plt.xlabel("Predicted $f_{ck}$ (MPa)", fontsize=10)
        plt.ylabel("Residuals (MPa)",            fontsize=10)
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.tight_layout()
        plt.savefig(f"Paper/Figures/Residuos/Residuals_{name.replace(' ', '_')}.png",
                    dpi=300, bbox_inches='tight')
        plt.close()
    print("✅ Residual plots salvos")

    # ── KDE plots ─────────────────────────────────────────────────────────────
    print("\n📊 GERANDO KDE PLOTS...")
    for name, pr in pred_ref.items():
        plt.figure(figsize=(7, 6))
        try:
            sns.kdeplot(x=y_ref_ext, y=pr, cmap="viridis",
                        fill=True, thresh=0.05, levels=15)
            lims = [min(y_ref_ext.min(), pr.min()),
                    max(y_ref_ext.max(), pr.max())]
            plt.plot(lims, lims, 'r--', alpha=0.7, label='Ideal (1:1)')
            plt.title(f"KDE – {name}", fontsize=12)
            plt.xlabel("Experimental $f_{ck}$ (MPa)", fontsize=10)
            plt.ylabel("Predicted $f_{ck}$ (MPa)",    fontsize=10)
            plt.grid(True, linestyle=':', alpha=0.4)
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"Paper/Figures/KDE_Density/KDE_{name.replace(' ', '_')}.png",
                        dpi=300, bbox_inches='tight')
        except Exception as e:
            print(f"   ⚠️ KDE {name}: {e}")
        plt.close()
    print("✅ KDE plots salvos")

    # ── MAPE plots ────────────────────────────────────────────────────────────
    print("\n📊 GERANDO MAPE PLOTS...")
    for name, pr in pred_ref.items():
        mape_pt = np.abs((y_ref_ext - pr) / (y_ref_ext + 1e-10)) * 100
        plt.figure(figsize=(6, 5))
        plt.scatter(y_ref_ext, mape_pt, alpha=0.6, s=30)
        plt.xlabel("fck (MPa)"); plt.ylabel("MAPE (%)")
        plt.title(f"MAPE vs fck – {name}")
        plt.tight_layout()
        plt.savefig(f"Figures/MAPE_vs_fck_{name}_Otimizado.png",
                    dpi=300, bbox_inches='tight')
        plt.close()
    print("✅ MAPE plots salvos")
