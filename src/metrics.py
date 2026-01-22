import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score

def roc_stats_from_dist(distances, labels):
    # For ROC, we need similarity score (higher = more likely same)
    scores = -distances
    fpr, tpr, thr = roc_curve(labels, scores)
    auc = roc_auc_score(labels, scores)

    # Find threshold that maximizes accuracy
    best_acc, best_idx = 0.0, 0
    best_thr, best_dthr = None, None
    for i, sthr in enumerate(thr):
        dthr = -sthr  # convert score thr → distance thr
        preds = (distances <= dthr).astype(int)
        tp = ((preds == 1) & (labels == 1)).sum()
        tn = ((preds == 0) & (labels == 0)).sum()
        fp = ((preds == 1) & (labels == 0)).sum()
        fn = ((preds == 0) & (labels == 1)).sum()
        acc = (tp + tn) / max(1, (tp + tn + fp + fn))
        if acc > best_acc:
            best_acc, best_idx, best_thr, best_dthr = acc, i, sthr, dthr

    # Recompute confusion at best threshold
    preds = (distances <= best_dthr).astype(int)
    tp = ((preds == 1) & (labels == 1)).sum()
    tn = ((preds == 0) & (labels == 0)).sum()
    fp = ((preds == 1) & (labels == 0)).sum()
    fn = ((preds == 0) & (labels == 1)).sum()
    FRR = fn / max(1, (fn + tp))  # False Rejection Rate
    FAR = fp / max(1, (fp + tn))  # False Acceptance Rate
    ACC = (tp + tn) / max(1, (tp + tn + fp + fn))

    return {
        "fpr": fpr,
        "tpr": tpr,
        "thr": thr,
        "auc": auc,
        "best_score_thr": float(best_thr),
        "best_dist_thr": float(best_dthr),
        "FRR": float(FRR),
        "FAR": float(FAR),
        "ACC": float(ACC),
        "TP": int(tp),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
    }
