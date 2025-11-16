import random
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch

from src.transforms import get_train_transform
from src.dataset_pairs import PairDataset

def main():
    transform = get_train_transform(img_size=224)
    ds = PairDataset(root="./data/train", transform=transform, pos_neg_ratio=1.0)

    dl = DataLoader(ds, batch_size=4, shuffle=True)

    # take one batch
    x1, x2, y, p1, p2 = next(iter(dl))
    print("Batch shapes:", x1.shape, x2.shape)
    print("Labels:", y.tolist())

    # show examples
    for i in range(4):
        fig, axs = plt.subplots(1,2)
        axs[0].imshow(x1[i].permute(1,2,0).numpy()*0.229+0.485) # undo normalize approx
        axs[0].set_title(f"img1 (label={y[i].item()})")
        axs[1].imshow(x2[i].permute(1,2,0).numpy()*0.229+0.485)
        axs[1].set_title("img2")
        for ax in axs: ax.axis("off")
        plt.show()

if __name__ == "__main__":
    main()
