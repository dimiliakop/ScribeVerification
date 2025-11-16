from torch.utils.data import Dataset
from PIL import Image, ImageOps, UnidentifiedImageError
from pathlib import Path
import os, random, glob, numpy as np, torch

class TripletDataset(Dataset):
    """
    Triplet dataset for training with Triplet Loss:
      - Samples (anchor, positive, negative)
      - Reuses same safe-loading and gray-flip augmentation ideas as PairDataset
      - Automatically skips or resamples corrupted images
    """
    def __init__(self, root, transform, grayflip_for_minority=True, minority_classes=None, num_samples=20000):
        self.root = Path(root)
        self.transform = transform
        self.grayflip_for_minority = grayflip_for_minority
        self.num_samples = num_samples

        # Load all class folders
        self.class_to_imgs = {}
        for cls in sorted(os.listdir(root)):
            cls_dir = self.root / cls
            if not cls_dir.is_dir():
                continue
            imgs = sorted(sum([glob.glob(str(cls_dir / f"*.{ext}"))
                               for ext in ["png","jpg","jpeg","bmp","tif","tiff"]], []))
            if imgs:
                self.class_to_imgs[cls] = imgs

        self.classes = sorted(self.class_to_imgs.keys())
        self.class_sizes = {c: len(self.class_to_imgs[c]) for c in self.classes}

        # Minority detection (same logic as in PairDataset)
        if not minority_classes:
            counts = sorted(self.class_sizes.values())
            median = counts[len(counts)//2]
            self.minority_classes = {c for c,n in self.class_sizes.items() if n < median}
        else:
            self.minority_classes = set(minority_classes)

    def __len__(self):
        return self.num_samples

    def _safe_load(self, path, apply_grayflip=False, max_tries=5):
        cls = Path(path).parent.name
        tries = 0
        while tries < max_tries:
            try:
                img = Image.open(path).convert("RGB")
                if apply_grayflip:
                    img = ImageOps.invert(img.convert("L")).convert("RGB")
                return self.transform(img)
            except (UnidentifiedImageError, OSError):
                path = random.choice(self.class_to_imgs[cls])
                tries += 1

        arr = np.zeros((224, 224, 3), dtype=np.uint8)
        img = Image.fromarray(arr)
        return self.transform(img)

    def __getitem__(self, idx):
        # choose anchor scribe
        c_anchor = random.choice(self.classes)
        imgs_anchor = self.class_to_imgs[c_anchor]
        if len(imgs_anchor) < 2:
            return self.__getitem__(idx)

        anchor_path, pos_path = random.sample(imgs_anchor, 2)

        # choose negative scribe
        c_neg = random.choice(self.classes)
        while c_neg == c_anchor or len(self.class_to_imgs[c_neg]) == 0:
            c_neg = random.choice(self.classes)
        neg_path = random.choice(self.class_to_imgs[c_neg])

        # load images
        anchor = self._safe_load(anchor_path, apply_grayflip=(self.grayflip_for_minority and c_anchor in self.minority_classes and random.random()<0.5))
        positive = self._safe_load(pos_path, apply_grayflip=(self.grayflip_for_minority and c_anchor in self.minority_classes and random.random()<0.5))
        negative = self._safe_load(neg_path, apply_grayflip=(self.grayflip_for_minority and c_neg in self.minority_classes and random.random()<0.5))

        return anchor, positive, negative
