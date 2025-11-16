import matplotlib.pyplot as plt

def plot_roc(fpr, tpr, auc, out_path="roc.png"):
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC={auc:.4f}")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("False Positive Rate (FAR)")
    plt.ylabel("True Positive Rate (1 - FRR)")
    plt.title("ROC Curve (Siamese Network)")
    plt.legend()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
