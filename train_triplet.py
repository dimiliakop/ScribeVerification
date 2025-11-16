import os, torch, argparse
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

from src.transforms import get_train_transform
from src.dataset_triplets import TripletDataset         # 🔹 new dataset
from src.models.vit_triplet import TripletNet  # 🔹 new model
from src.losses import triplet_loss                      # 🔹 new loss

def load_config(path="configs/default.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def main(config_path):
    # Load config
    cfg = load_config(config_path)

    # Reproducibility
    torch.manual_seed(cfg["seed"])

    # Device
    device = "cuda" if torch.cuda.is_available() and cfg["eval"]["device"] == "cuda" else "cpu"
    print("Using device:", device)

    # Data
    transform = get_train_transform(img_size=cfg["data"]["img_size"])
    train_dataset = TripletDataset(
        root=os.path.join(cfg["data"]["root"], "train"),
        transform=transform,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=cfg["train"]["num_workers"],
    )

    # Model
    model = TripletNet(
        embedding_dim=cfg["model"]["embedding_dim"],
        pretrained=cfg["model"]["pretrained"],
    ).to(device)

    # Optimizer
    if cfg["train"]["optimizer"].lower() == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=cfg["train"]["lr"],
            weight_decay=cfg["train"]["weight_decay"],
        )
    else:
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=cfg["train"]["lr"],
            momentum=0.9,
            weight_decay=cfg["train"]["weight_decay"],
        )

    os.makedirs(cfg["train"]["save_dir"], exist_ok=True)

    # Training loop
    for epoch in range(1, cfg["train"]["epochs"] + 1):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg['train']['epochs']}")

        for anchor, positive, negative in pbar:
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)

            z_a, z_p, z_n = model(anchor, positive, negative)
            loss = triplet_loss(z_a, z_p, z_n, margin=cfg["loss"]["margin"])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch} average loss: {avg_loss:.4f}")

        # Save checkpoint
        ckpt = os.path.join(cfg["train"]["save_dir"], f"vit_triplet_e{epoch:02d}.pt")
        torch.save(model.state_dict(), ckpt)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                        help="Path to YAML config file")
    args = parser.parse_args()
    main(args.config)