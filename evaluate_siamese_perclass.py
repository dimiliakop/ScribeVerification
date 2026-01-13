import os, csv, torch
import numpy as np
from collections import defaultdict
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.transforms import get_eval_transform
from src.dataset_pairs import PairDataset
from src.models.vit_siamese import SiameseNet, pairwise_l2
from src.metrics import roc_stats_from_dist
from src.viz import plot_roc


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


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # 1. Load test pairs
    fixed_pairs = read_pairs_csv("data/test_pairs.csv")
    transform = get_eval_transform(img_size=224)

    ds = PairDataset(root="./data/test", transform=transform,
                     for_eval=True, fixed_pairs=fixed_pairs)
    dl = DataLoader(ds, batch_size=128, shuffle=False, num_workers=2)

    # 2. Load latest checkpoint
    ckpts = sorted([p for p in os.listdir("checkpoints") if p.endswith(".pt")])
    assert ckpts, "No checkpoints found!"
    ckpt_path = os.path.join("checkpoints", ckpts[-1])
    print("Evaluating checkpoint:", ckpt_path)

    model = SiameseNet(embedding_dim=10, pretrained=False).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    # 3. Compute distances + keep paths
    all_dist = []
    all_label = []
    all_p1 = []
    all_p2 = []

    with torch.no_grad():
        for x1, x2, y, p1, p2 in tqdm(dl, desc="Eval"):
            x1, x2 = x1.to(device), x2.to(device)
            z1, z2 = model(x1, x2)
            dist = pairwise_l2(z1, z2).detach().cpu().numpy()

            all_dist.append(dist)
            all_label.append(np.array(y))
            all_p1.extend(list(p1))
            all_p2.extend(list(p2))

    distances = np.concatenate(all_dist).astype(np.float32)
    labels = np.concatenate(all_label).astype(np.int32)

    # ---- Overall metrics
    stats = roc_stats_from_dist(distances, labels)
    print("\n--- Overall Evaluation Results ---")
    print(f"AUC   : {stats['auc']:.4f}")
    print(f"ACC   : {stats['ACC']:.4f} (at best threshold)")
    print(f"FAR   : {stats['FAR']:.4f}")
    print(f"FRR   : {stats['FRR']:.4f}")
    print(f"Best distance threshold: {stats['best_dist_thr']:.4f}\n")

    # ---- Per-class metrics
    # define the "class" of a pair:
    # - for positive pairs: class = class(path1) (same as path2)
    # - for negative pairs: we will count the pair toward BOTH involved classes
    per_class_indices = defaultdict(list)
    for i, (p1, p2, y) in enumerate(zip(all_p1, all_p2, labels)):
        c1 = infer_class_from_path(p1)
        c2 = infer_class_from_path(p2)
        if y == 1:
            per_class_indices[c1].append(i)
        else:
            per_class_indices[c1].append(i)
            per_class_indices[c2].append(i)

    os.makedirs("roc_per_class", exist_ok=True)

    print("--- Per-class Results (pair-conditioned) ---")
    # optional: require a minimum number of samples per class
    MIN_SAMPLES = 200

    for cls, idxs in sorted(per_class_indices.items()):
        if len(idxs) < MIN_SAMPLES:
            continue

        d_cls = distances[idxs]
        y_cls = labels[idxs]
        st = roc_stats_from_dist(d_cls, y_cls)

        print(f"\nClass: {cls}  (pairs={len(idxs)})")
        print(f"  AUC: {st['auc']:.4f} | ACC: {st['ACC']:.4f} | FAR: {st['FAR']:.4f} | FRR: {st['FRR']:.4f}")

        out_path = os.path.join("roc_per_class", f"roc_{cls.replace(' ', '_')}.png")
        plot_roc(st["fpr"], st["tpr"], st["auc"], out_path=out_path)

    print("\nSaved per-class ROC curves to: roc_per_class/")


if __name__ == "__main__":
    main()
