import os
import csv
import argparse
import importlib
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import font_manager

# Option 1: if installed (common on Windows)
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC"]
plt.rcParams["axes.unicode_minus"] = False

from src.transforms import get_eval_transform
from src.dataset_pairs import PairDataset
from src.metrics import roc_stats_from_dist  # your existing function
from src.viz import plot_roc                 # your existing function


# -----------------------------
# CSV utilities
# -----------------------------
def read_pairs_csv(path):
    pairs = []
    with open(path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            pairs.append((row["path1"], row["path2"], int(row["label"])))
    return pairs


def infer_class_from_path(p):
    # expects .../<class_name>/<file>
    return os.path.basename(os.path.dirname(p))


# -----------------------------
# Model loading (dynamic)
# -----------------------------
def load_model(model_module: str, embedding_dim: int, pretrained: bool, device: str):
    """
    Triplet module example:
      "src.models.mobilenetv3_plus_exact_triplet"
      "src.models.vit_triplet"
    The module must expose:
      - TripletNet (preferred) or SiameseNet
      - pairwise_l2
    """
    m = importlib.import_module(model_module)

    # Prefer TripletNet if present, else SiameseNet
    if hasattr(m, "TripletNet"):
        Net = getattr(m, "TripletNet")
    elif hasattr(m, "SiameseNet"):
        Net = getattr(m, "SiameseNet")
    else:
        raise RuntimeError(f"{model_module} must define TripletNet or SiameseNet")

    if not hasattr(m, "pairwise_l2"):
        raise RuntimeError(f"{model_module} must define pairwise_l2(z1,z2)")

    pairwise_l2 = getattr(m, "pairwise_l2")
    model = Net(embedding_dim=embedding_dim, pretrained=pretrained).to(device)
    return model, pairwise_l2


def forward_pair_embeddings(model, x1, x2):
    """
    Triplet-friendly pair forward:
      - model.forward_pair(x1,x2) if exists (BEST)
      - else uses forward_once(x) if exists
      - else tries model(x1,x2) returning (z1,z2)
    """
    if hasattr(model, "forward_pair"):
        return model.forward_pair(x1, x2)

    if hasattr(model, "forward_once"):
        return model.forward_once(x1), model.forward_once(x2)

    out = model(x1, x2)
    if isinstance(out, (tuple, list)) and len(out) == 2:
        return out[0], out[1]

    raise RuntimeError(
        "Triplet model does not support pair evaluation. "
        "Add forward_pair(x1,x2) or forward_once(x)."
    )


# -----------------------------
# Plot helpers
# -----------------------------
def _annotate_heatmap(ax, mat, fontsize=7):
    n = mat.shape[0]
    for i in range(n):
        for j in range(n):
            v = int(mat[i, j])
            if v != 0:
                ax.text(j, i, str(v), ha="center", va="center", fontsize=fontsize)


def save_confusion_heatmap(mat, classes, title, out_path, xlabel, ylabel, figsize=(10, 10)):
    plt.figure(figsize=figsize)
    ax = plt.gca()
    im = ax.imshow(mat, aspect="auto")
    plt.colorbar(im, fraction=0.046, pad=0.04)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=60, ha="right")
    ax.set_yticklabels(classes)

    _annotate_heatmap(ax, mat)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_bar(values_by_class, title, out_path, ylabel, figsize=(12, 4)):
    classes = list(values_by_class.keys())
    vals = [values_by_class[c] for c in classes]

    plt.figure(figsize=figsize)
    plt.bar(classes, vals)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=60, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# -----------------------------
# Main evaluation
# -----------------------------
def main(args):
    device = "cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu"
    print("Using device:", device)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "roc_per_class").mkdir(exist_ok=True)
    (out_dir / "plots").mkdir(exist_ok=True)

    # 1) Load pairs + dataset
    fixed_pairs = read_pairs_csv(args.pairs_csv)
    transform = get_eval_transform(img_size=args.img_size)

    ds = PairDataset(
        root=args.data_root,
        transform=transform,
        for_eval=True,
        fixed_pairs=fixed_pairs,
    )
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # 2) Load model + checkpoint
    model, pairwise_l2 = load_model(args.model_module, args.embedding_dim, args.pretrained, device)

    ckpt_path = args.ckpt
    if ckpt_path is None:
        ckpts = sorted([p for p in os.listdir(args.checkpoints_dir) if p.endswith(".pt")])
        assert ckpts, f"No checkpoints found in {args.checkpoints_dir}"
        ckpt_path = os.path.join(args.checkpoints_dir, ckpts[-1])

    print("Evaluating checkpoint:", ckpt_path)

    # Safer torch.load (silences FutureWarning when supported)
    if "weights_only" in torch.load.__code__.co_varnames:
        state = torch.load(ckpt_path, map_location=device, weights_only=True)
    else:
        state = torch.load(ckpt_path, map_location=device)

    model.load_state_dict(state)
    model.eval()

    # 3) Run forward on all pairs
    all_dist = []
    all_label = []
    all_p1 = []
    all_p2 = []

    with torch.no_grad():
        for x1, x2, y, p1, p2 in tqdm(dl, desc="Eval"):
            x1, x2 = x1.to(device), x2.to(device)
            z1, z2 = forward_pair_embeddings(model, x1, x2)
            dist = pairwise_l2(z1, z2).detach().cpu().numpy()

            all_dist.append(dist)
            all_label.append(np.array(y))
            all_p1.extend(list(p1))
            all_p2.extend(list(p2))

    distances = np.concatenate(all_dist).astype(np.float32)
    labels = np.concatenate(all_label).astype(np.int32)

    # 4) Overall metrics + ROC
    stats = roc_stats_from_dist(distances, labels)
    thr = float(stats["best_dist_thr"])

    print("\n--- Overall Evaluation Results (Triplet model on pairs) ---")
    print(f"AUC   : {stats['auc']:.4f}")
    print(f"ACC   : {stats['ACC']:.4f} (at best threshold)")
    print(f"FAR   : {stats['FAR']:.4f}")
    print(f"FRR   : {stats['FRR']:.4f}")
    print(f"Best distance threshold: {thr:.6f}\n")

    print("--- Overall Confusion Counts ---")
    print(f"TP: {stats['TP']} | TN: {stats['TN']} | FP: {stats['FP']} | FN: {stats['FN']}")

    roc_path = out_dir / "roc_overall.png"
    plot_roc(stats["fpr"], stats["tpr"], stats["auc"], out_path=str(roc_path))
    print("Saved overall ROC:", roc_path)

    # 5) Build per-class pair-conditioned indices
    per_class_indices = defaultdict(list)
    for i, (p1, p2, y) in enumerate(zip(all_p1, all_p2, labels)):
        c1 = infer_class_from_path(p1)
        c2 = infer_class_from_path(p2)
        if y == 1:
            per_class_indices[c1].append(i)
        else:
            # count negative pair toward BOTH classes
            per_class_indices[c1].append(i)
            per_class_indices[c2].append(i)

    # 6) Per-class metrics + per-class ROC images + CSV
    per_class_rows = []
    per_class_stats = {}

    print("\n--- Per-class Results (pair-conditioned) ---")
    for cls, idxs in sorted(per_class_indices.items()):
        if len(idxs) < args.min_pairs_per_class:
            continue

        d_cls = distances[idxs]
        y_cls = labels[idxs]
        st = roc_stats_from_dist(d_cls, y_cls)

        per_class_stats[cls] = st

        print(f"\nClass: {cls} (pairs={len(idxs)})")
        print(f"  AUC : {st['auc']:.4f}")
        print(f"  ACC : {st['ACC']:.4f}")
        print(f"  FAR : {st['FAR']:.4f}")
        print(f"  FRR : {st['FRR']:.4f}")
        print("  Confusion Matrix:")
        print(f"    TP: {st['TP']}  FN: {st['FN']}")
        print(f"    FP: {st['FP']}  TN: {st['TN']}")

        roc_cls_path = out_dir / "roc_per_class" / f"roc_{cls.replace(' ', '_')}.png"
        plot_roc(st["fpr"], st["tpr"], st["auc"], out_path=str(roc_cls_path))

        per_class_rows.append({
            "class": cls,
            "pairs": len(idxs),
            "auc": st["auc"],
            "acc": st["ACC"],
            "far": st["FAR"],
            "frr": st["FRR"],
            "tp": st["TP"],
            "tn": st["TN"],
            "fp": st["FP"],
            "fn": st["FN"],
        })

    csv_out = out_dir / "per_class_metrics.csv"
    with open(csv_out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=list(per_class_rows[0].keys()) if per_class_rows else
            ["class", "pairs", "auc", "acc", "far", "frr", "tp", "tn", "fp", "fn"]
        )
        w.writeheader()
        for row in per_class_rows:
            w.writerow(row)
    print("\nSaved per-class CSV:", csv_out)

    # 7) Two confusion matrices (global threshold thr)
    class_names = sorted({infer_class_from_path(p) for p in all_p1 + all_p2})
    c2i = {c: i for i, c in enumerate(class_names)}
    n = len(class_names)

    false_accept_cm = np.zeros((n, n), dtype=np.int32)  # FP on negative pairs
    false_reject_cm = np.zeros((n, n), dtype=np.int32)  # FN on positive pairs (diagonal)

    for p1, p2, y, d in zip(all_p1, all_p2, labels, distances):
        c1 = infer_class_from_path(p1)
        c2 = infer_class_from_path(p2)
        pred_same = (d <= thr)

        if y == 0 and pred_same:
            i = c2i[c1]
            j = c2i[c2]
            false_accept_cm[i, j] += 1

        if y == 1 and (not pred_same):
            i = c2i[c1]
            false_reject_cm[i, i] += 1

    fa_path = out_dir / "plots" / "cm_false_accepts.png"
    fr_path = out_dir / "plots" / "cm_false_rejects.png"

    save_confusion_heatmap(
        false_accept_cm, class_names,
        title="False-Accept Confusion Matrix (Triplet model, Pred SAME on Different Writers)",
        out_path=str(fa_path),
        xlabel="Impostor class (path2)",
        ylabel="Anchor class (path1)",
        figsize=(10, 10),
    )
    save_confusion_heatmap(
        false_reject_cm, class_names,
        title="False-Reject Matrix (Triplet model) — Diagonal shows FN count",
        out_path=str(fr_path),
        xlabel="Class",
        ylabel="Class",
        figsize=(10, 10),
    )

    print("\nSaved confusion matrices:")
    print(" ", fa_path)
    print(" ", fr_path)

    # 8) Bar plots per class
    if per_class_stats:
        auc_by_c = {c: per_class_stats[c]["auc"] for c in class_names if c in per_class_stats}
        acc_by_c = {c: per_class_stats[c]["ACC"] for c in class_names if c in per_class_stats}
        far_by_c = {c: per_class_stats[c]["FAR"] for c in class_names if c in per_class_stats}
        frr_by_c = {c: per_class_stats[c]["FRR"] for c in class_names if c in per_class_stats}

        save_bar(auc_by_c, "Per-class AUC (pair-conditioned)", str(out_dir / "plots" / "bar_auc_per_class.png"), "AUC")
        save_bar(acc_by_c, "Per-class ACC (pair-conditioned)", str(out_dir / "plots" / "bar_acc_per_class.png"), "ACC")
        save_bar(far_by_c, "Per-class FAR (pair-conditioned)", str(out_dir / "plots" / "bar_far_per_class.png"), "FAR")
        save_bar(frr_by_c, "Per-class FRR (pair-conditioned)", str(out_dir / "plots" / "bar_frr_per_class.png"), "FRR")

        print("\nSaved bar plots in:", out_dir / "plots")

    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Data
    parser.add_argument("--data_root", type=str, default="./data/test", help="Root folder of test set (class subfolders).")
    parser.add_argument("--pairs_csv", type=str, default="./data/test_pairs.csv", help="CSV with path1,path2,label.")
    parser.add_argument("--img_size", type=int, default=224)

    # Loader
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=2)

    # Model
    parser.add_argument("--model_module", type=str, required=True,
                        help="Triplet module, e.g. src.models.mobilenetv3_plus_exact_triplet or src.models.vit_triplet")
    parser.add_argument("--embedding_dim", type=int, default=10)
    parser.add_argument("--pretrained", action="store_true", help="Build model with ImageNet weights (only affects init).")

    # Checkpoint
    parser.add_argument("--checkpoints_dir", type=str, default="./checkpoints")
    parser.add_argument("--ckpt", type=str, default=None, help="If not set, picks latest .pt in checkpoints_dir")

    # Eval controls
    parser.add_argument("--min_pairs_per_class", type=int, default=200)
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])

    # Output
    parser.add_argument("--out_dir", type=str, default="./eval_outputs_triplet")

    args = parser.parse_args()
    main(args)
