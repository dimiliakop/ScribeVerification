from torch.utils.data import Dataset
from PIL import Image, ImageOps, UnidentifiedImageError
from pathlib import Path
import os, random, glob

class PairDataset(Dataset):
    """
    Siamese pair dataset:
      - Creates positive (same scribe) and negative (different scribe) pairs
      - Adds gray-flip augmentation for minority classes (as in paper)
      - Skips/resamples corrupted images automatically
    """
    def __init__(self, root, transform,
                 pos_neg_ratio=1.0,
                 for_eval=False, fixed_pairs=None,
                 grayflip_for_minority=True,
                 minority_classes=None):
        self.root = Path(root)
        self.transform = transform
        self.pos_neg_ratio = pos_neg_ratio
        self.for_eval = for_eval
        self.fixed_pairs = fixed_pairs
        self.grayflip_for_minority = grayflip_for_minority

        # Load images per scribe
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

        # Minority = below median size if not provided
        if not minority_classes:
            counts = sorted(self.class_sizes.values())
            median = counts[len(counts)//2]
            self.minority_classes = {c for c,n in self.class_sizes.items() if n < median}
        else:
            self.minority_classes = set(minority_classes)

        # Dataset length
        if self.for_eval and self.fixed_pairs is not None:
            self.length = len(self.fixed_pairs)
        else:
            self.length = sum(len(v) for v in self.class_to_imgs.values())

    def __len__(self):
        return self.length

    def _safe_load(self, path, apply_grayflip=False, max_tries=5):
        """
        Try to load an image safely.
        If corrupted, resample another image from the same class up to `max_tries`.
        """
        cls = Path(path).parent.name
        tries = 0
        while tries < max_tries:
            try:
                img = Image.open(path).convert("RGB")
                if apply_grayflip:
                    img = ImageOps.invert(img.convert("L")).convert("RGB")
                return self.transform(img)
            except (UnidentifiedImageError, OSError):
                # pick another file from the same class
                path = random.choice(self.class_to_imgs[cls])
                tries += 1

        # If everything fails, return a blank image (safe fallback)
        import numpy as np
        import torch
        arr = np.zeros((224, 224, 3), dtype=np.uint8)
        img = Image.fromarray(arr)
        return self.transform(img)

    def _sample_positive(self):
        c = random.choice(self.classes)
        imgs = self.class_to_imgs[c]
        if len(imgs) < 2:
            return self._sample_positive()
        p1, p2 = random.sample(imgs, 2)
        return p1, p2, 1, c

    def _sample_negative(self):
        c1, c2 = random.sample(self.classes, 2)
        p1 = random.choice(self.class_to_imgs[c1])
        p2 = random.choice(self.class_to_imgs[c2])
        return p1, p2, 0, (c1, c2)

    def __getitem__(self, idx):
        if self.for_eval and self.fixed_pairs is not None:
            p1, p2, y = self.fixed_pairs[idx]
            x1 = self._safe_load(p1, False)
            x2 = self._safe_load(p2, False)
            return x1, x2, y, p1, p2

        do_pos = random.random() < (self.pos_neg_ratio / (1.0 + self.pos_neg_ratio))
        if do_pos:
            p1, p2, y, c = self._sample_positive()
            x1 = self._safe_load(p1, apply_grayflip=(self.grayflip_for_minority and c in self.minority_classes and random.random()<0.5))
            x2 = self._safe_load(p2, apply_grayflip=(self.grayflip_for_minority and c in self.minority_classes and random.random()<0.5))
            return x1, x2, y, p1, p2
        else:
            p1, p2, y, (c1, c2) = self._sample_negative()
            x1 = self._safe_load(p1, apply_grayflip=(self.grayflip_for_minority and c1 in self.minority_classes and random.random()<0.5))
            x2 = self._safe_load(p2, apply_grayflip=(self.grayflip_for_minority and c2 in self.minority_classes and random.random()<0.5))
            return x1, x2, y, p1, p2
