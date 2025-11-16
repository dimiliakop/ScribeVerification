import os, csv, torch
import numpy as np
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

    # 3. Compute distances
    dists, labels = [], []
    with torch.no_grad():
        for x1, x2, y, _, _ in tqdm(dl, desc="Eval"):
            x1, x2 = x1.to(device), x2.to(device)
            z1, z2 = model(x1, x2)
            dist = pairwise_l2(z1, z2).cpu().numpy()
            dists.append(dist)
            labels.append(y.numpy())

    distances = np.concatenate(dists)
    labels = np.concatenate(labels)

    # 4. Metrics (includes TP/TN/FP/FN)
    stats = roc_stats_from_dist(distances, labels)

    print("\n--- Evaluation Results ---")
    print(f"AUC   : {stats['auc']:.4f}")
    print(f"ACC   : {stats['ACC']:.4f} (at best threshold)")
    print(f"FAR   : {stats['FAR']:.4f}")
    print(f"FRR   : {stats['FRR']:.4f}")
    print(f"Best distance threshold: {stats['best_dist_thr']:.4f}\n")

    # Print confusion matrix counts
    print("--- Confusion Matrix at Best Threshold ---")
    print(f"TP (same, correctly accepted)   : {stats['TP']}")
    print(f"TN (different, correctly rejected): {stats['TN']}")
    print(f"FP (different, wrongly accepted): {stats['FP']}")
    print(f"FN (same, wrongly rejected)     : {stats['FN']}")

    # 5. Save ROC curve
    plot_roc(stats["fpr"], stats["tpr"], stats["auc"], out_path="roc.png")
    print("\nROC curve saved to roc.png")

if __name__ == "__main__":
    main()
