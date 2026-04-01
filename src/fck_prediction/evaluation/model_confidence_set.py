import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def compute_mcs(pred_ref, y_ref_ext):
    """Greedy Model Confidence Set: iteratively eliminate worst model (S22).

    Returns
    -------
    mcs_res : DataFrame with Model, Mean_Loss, In_Final_MCS
    """
    print("\n📊 Model Confidence Set (referência comum)...")

    if len(pred_ref) < 2:
        return pd.DataFrame()

    loss_vecs = []
    nms_mcs   = []
    for name, pr in pred_ref.items():
        loss_vecs.append((y_ref_ext - pr) ** 2)
        nms_mcs.append(name)

    loss_arr  = np.array(loss_vecs)
    mcs_names = nms_mcs.copy()
    loss_cp   = loss_arr.copy()

    print("\n   MCS Iterations:")
    it = 1
    while len(mcs_names) > 1:
        ml_v    = loss_cp.mean(axis=1)
        worst   = np.argmax(ml_v)
        removed = mcs_names.pop(worst)
        loss_cp = np.delete(loss_cp, worst, axis=0)
        print(f"      It {it}: Removed {removed} (loss={ml_v[worst]:.4f})")
        it += 1

    print(f"\n   ✅ Final MCS: {mcs_names}")
    mcs_res = pd.DataFrame({
        'Model':       nms_mcs,
        'Mean_Loss':   loss_arr.mean(axis=1),
        'In_Final_MCS': ['Yes' if m in mcs_names else 'No' for m in nms_mcs],
    }).sort_values('Mean_Loss')
    mcs_res.to_excel("Paper/Results/model_confidence_set_otimizado.xlsx", index=False)

    plt.figure(figsize=(10, 6))
    cols_mcs = ['green' if x == 'Yes' else 'red' for x in mcs_res['In_Final_MCS']]
    plt.bar(range(len(mcs_res)), mcs_res['Mean_Loss'], color=cols_mcs, alpha=0.7)
    plt.xticks(range(len(mcs_res)), mcs_res['Model'], rotation=45)
    plt.ylabel('Mean Squared Error')
    plt.title('Model Confidence Set – MSE (Common Test Set)')
    thresh_idx = len(mcs_names) - 1
    plt.axhline(mcs_res['Mean_Loss'].iloc[thresh_idx],
                color='blue', linestyle='--', label='MCS threshold')
    plt.legend()
    plt.tight_layout()
    plt.savefig("Paper/Figures/model_confidence_set_otimizado.png",
                dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ MCS salvo")

    return mcs_res
